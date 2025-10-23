"""
Exponential backoff error handling with jitter for AWS Bedrock API calls.
"""

import asyncio
import random
import logging
from typing import Callable, Any, Optional, Type, Union
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


class BackoffHandler:
    """
    Implements exponential backoff with jitter for error recovery.
    
    Provides configurable retry logic with specific handling for different
    error types including throttling and server errors.
    """
    
    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize the backoff handler.
        
        Args:
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Multiplier for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if the exception is retryable, False otherwise
        """
        if isinstance(exception, ClientError):
            error_code = exception.response.get('Error', {}).get('Code', '')
            
            # Retry on throttling errors
            if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                return True
            
            # Retry on server errors (5xx)
            http_status = exception.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 0)
            if 500 <= http_status < 600:
                return True
            
            # Retry on specific service errors
            if error_code in [
                'ServiceUnavailableException',
                'InternalServerException',
                'ModelTimeoutException'
            ]:
                return True
            
            # Don't retry on client errors (4xx) except throttling
            if 400 <= http_status < 500:
                return False
        
        # Retry on connection and timeout errors
        if isinstance(exception, (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError
        )):
            return True
        
        # Don't retry on other exceptions by default
        return False
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay for a given retry attempt.
        
        Args:
            attempt: The current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential delay
        delay = self.base_delay * (self.backoff_factor ** attempt)
        
        # Cap at maximum delay
        delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            # Add random jitter up to 25% of the delay
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        return delay
    
    async def execute_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            The last exception if all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - return the result
                if attempt > 0:
                    logger.info(f"Function succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self.should_retry(e):
                    logger.error(f"Non-retryable error: {e}")
                    raise e
                
                # Check if we've exhausted retries
                if attempt >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) exceeded. Last error: {e}")
                    break
                
                # Calculate delay and wait
                delay = self.calculate_delay(attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed with {type(e).__name__}: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                await asyncio.sleep(delay)
        
        # All retries exhausted - raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Unexpected error: no exception recorded")
    
    def get_error_category(self, exception: Exception) -> str:
        """
        Categorize an exception for logging and monitoring purposes.
        
        Args:
            exception: The exception to categorize
            
        Returns:
            String category name
        """
        if isinstance(exception, ClientError):
            error_code = exception.response.get('Error', {}).get('Code', '')
            http_status = exception.response.get('ResponseMetadata', {}).get('HTTPStatusCode', 0)
            
            if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                return 'throttling'
            elif 500 <= http_status < 600:
                return 'server_error'
            elif 400 <= http_status < 500:
                return 'client_error'
            else:
                return 'aws_error'
        
        elif isinstance(exception, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
            return 'network_error'
        
        else:
            return 'unknown_error'


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    Stops making requests after a threshold of consecutive failures
    and allows recovery after a timeout period.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type to monitor
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = 'closed'  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit breaker state."""
        if self.state == 'closed':
            return True
        
        if self.state == 'open':
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                asyncio.get_event_loop().time() - self.last_failure_time >= self.recovery_timeout):
                self.state = 'half_open'
                return True
            return False
        
        if self.state == 'half_open':
            return True
        
        return False
    
    def record_success(self):
        """Record a successful execution."""
        self.failure_count = 0
        self.state = 'closed'
    
    def record_failure(self):
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
        elif self.state == 'half_open':
            self.state = 'open'
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: The function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RuntimeError: If circuit breaker is open
            Exception: Any exception from the function
        """
        if not self.can_execute():
            raise RuntimeError("Circuit breaker is open")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self.record_success()
            return result
            
        except self.expected_exception as e:
            self.record_failure()
            raise e