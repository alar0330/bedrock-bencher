"""
Structured logging and error reporting for the Bedrock Benchmark Toolkit.

Provides centralized logging configuration with:
- Structured logging using structlog
- Progress reporting for long-running benchmarks
- Error context capture and reporting
- Multiple output formats (JSON, human-readable)
"""

import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

import structlog
from structlog.stdlib import LoggerFactory

from .config import BenchmarkConfig


class BenchmarkLogger:
    """
    Centralized logging configuration for the Bedrock Benchmark Toolkit.
    
    Provides structured logging with context preservation, error reporting,
    and progress tracking capabilities.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize logging configuration.
        
        Args:
            config: BenchmarkConfig instance with logging settings
        """
        self.config = config
        self._configured = False
        self._progress_logger = None
    
    def configure_logging(self):
        """
        Configure structured logging based on configuration settings.
        
        Sets up:
        - Log level and format
        - File and console handlers
        - Structured logging processors
        - Error context capture
        """
        if self._configured:
            return
        
        # Configure standard library logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(message)s',  # structlog will handle formatting
            handlers=[]  # We'll add handlers manually
        )
        
        # Configure structlog processors
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        # Add appropriate final processor based on format
        if self.config.log_format == "structured":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Set up handlers
        self._setup_handlers()
        
        self._configured = True
        
        # Log configuration completion
        logger = structlog.get_logger(__name__)
        logger.info(
            "Logging configured",
            log_level=self.config.log_level,
            log_format=self.config.log_format,
            log_file=self.config.log_file
        )
    
    def _setup_handlers(self):
        """Set up logging handlers for console and file output."""
        root_logger = logging.getLogger()
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level))
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            log_file = Path(self.config.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)  # File gets all logs
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> structlog.BoundLogger:
        """
        Get a structured logger instance.
        
        Args:
            name: Logger name (typically __name__)
        
        Returns:
            Configured structlog BoundLogger instance
        """
        if not self._configured:
            self.configure_logging()
        
        return structlog.get_logger(name)
    
    def get_progress_logger(self) -> 'ProgressLogger':
        """
        Get a progress logger for tracking long-running operations.
        
        Returns:
            ProgressLogger instance
        """
        if not self._progress_logger:
            self._progress_logger = ProgressLogger(self.get_logger("progress"))
        
        return self._progress_logger


class ProgressLogger:
    """
    Specialized logger for tracking progress of long-running benchmarks.
    
    Provides structured progress reporting with timing, completion rates,
    and error tracking.
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        """
        Initialize progress logger.
        
        Args:
            logger: Structured logger instance
        """
        self.logger = logger
        self._active_operations: Dict[str, Dict[str, Any]] = {}
    
    def start_operation(
        self,
        operation_id: str,
        operation_type: str,
        total_items: int,
        **context
    ):
        """
        Start tracking a long-running operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation (e.g., "benchmark_run", "data_export")
            total_items: Total number of items to process
            **context: Additional context information
        """
        start_time = datetime.now()
        
        operation_data = {
            "operation_type": operation_type,
            "total_items": total_items,
            "completed_items": 0,
            "failed_items": 0,
            "start_time": start_time,
            "last_update": start_time,
            **context
        }
        
        self._active_operations[operation_id] = operation_data
        
        self.logger.info(
            "Operation started",
            operation_id=operation_id,
            operation_type=operation_type,
            total_items=total_items,
            **context
        )
    
    def update_progress(
        self,
        operation_id: str,
        completed_delta: int = 0,
        failed_delta: int = 0,
        **context
    ):
        """
        Update progress for an active operation.
        
        Args:
            operation_id: Operation identifier
            completed_delta: Number of newly completed items
            failed_delta: Number of newly failed items
            **context: Additional context information
        """
        if operation_id not in self._active_operations:
            self.logger.warning(
                "Progress update for unknown operation",
                operation_id=operation_id
            )
            return
        
        operation = self._active_operations[operation_id]
        operation["completed_items"] += completed_delta
        operation["failed_items"] += failed_delta
        operation["last_update"] = datetime.now()
        
        # Calculate metrics
        processed_items = operation["completed_items"] + operation["failed_items"]
        completion_rate = (processed_items / operation["total_items"]) * 100
        success_rate = (operation["completed_items"] / processed_items * 100) if processed_items > 0 else 0
        
        elapsed_time = (operation["last_update"] - operation["start_time"]).total_seconds()
        
        # Estimate remaining time
        remaining_items = operation["total_items"] - processed_items
        estimated_remaining = None
        if processed_items > 0 and remaining_items > 0:
            rate = processed_items / elapsed_time
            estimated_remaining = remaining_items / rate if rate > 0 else None
        
        self.logger.info(
            "Progress update",
            operation_id=operation_id,
            operation_type=operation["operation_type"],
            completed_items=operation["completed_items"],
            failed_items=operation["failed_items"],
            total_items=operation["total_items"],
            completion_rate=round(completion_rate, 1),
            success_rate=round(success_rate, 1),
            elapsed_time=round(elapsed_time, 1),
            estimated_remaining_time=round(estimated_remaining, 1) if estimated_remaining else None,
            **context
        )
    
    def complete_operation(
        self,
        operation_id: str,
        success: bool = True,
        **context
    ):
        """
        Mark an operation as completed.
        
        Args:
            operation_id: Operation identifier
            success: Whether the operation completed successfully
            **context: Additional context information
        """
        if operation_id not in self._active_operations:
            self.logger.warning(
                "Completion for unknown operation",
                operation_id=operation_id
            )
            return
        
        operation = self._active_operations[operation_id]
        end_time = datetime.now()
        
        processed_items = operation["completed_items"] + operation["failed_items"]
        success_rate = (operation["completed_items"] / processed_items * 100) if processed_items > 0 else 0
        total_time = (end_time - operation["start_time"]).total_seconds()
        
        self.logger.info(
            "Operation completed",
            operation_id=operation_id,
            operation_type=operation["operation_type"],
            success=success,
            completed_items=operation["completed_items"],
            failed_items=operation["failed_items"],
            total_items=operation["total_items"],
            success_rate=round(success_rate, 1),
            total_time=round(total_time, 1),
            **context
        )
        
        # Remove from active operations
        del self._active_operations[operation_id]
    
    def log_error(
        self,
        operation_id: str,
        error: Exception,
        item_id: Optional[str] = None,
        **context
    ):
        """
        Log an error that occurred during an operation.
        
        Args:
            operation_id: Operation identifier
            error: Exception that occurred
            item_id: Optional identifier for the specific item that failed
            **context: Additional context information
        """
        error_context = {
            "operation_id": operation_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
        
        if item_id:
            error_context["item_id"] = item_id
        
        self.logger.error(
            "Operation error",
            **error_context,
            exc_info=error
        )


class ErrorReporter:
    """
    Centralized error reporting and context capture.
    
    Provides structured error logging with context preservation
    for debugging and monitoring purposes.
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        """
        Initialize error reporter.
        
        Args:
            logger: Structured logger instance
        """
        self.logger = logger
    
    def report_api_error(
        self,
        error: Exception,
        operation: str,
        model_id: Optional[str] = None,
        item_id: Optional[str] = None,
        request_params: Optional[Dict[str, Any]] = None,
        **context
    ):
        """
        Report an API-related error with full context.
        
        Args:
            error: Exception that occurred
            operation: Operation being performed
            model_id: Bedrock model ID
            item_id: Item being processed
            request_params: API request parameters
            **context: Additional context
        """
        error_context = {
            "error_category": "api_error",
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
        
        if model_id:
            error_context["model_id"] = model_id
        if item_id:
            error_context["item_id"] = item_id
        if request_params:
            # Sanitize request params (remove sensitive data)
            sanitized_params = self._sanitize_params(request_params)
            error_context["request_params"] = sanitized_params
        
        self.logger.error(
            "API error occurred",
            **error_context,
            exc_info=error
        )
    
    def report_storage_error(
        self,
        error: Exception,
        operation: str,
        file_path: Optional[Union[str, Path]] = None,
        **context
    ):
        """
        Report a storage-related error.
        
        Args:
            error: Exception that occurred
            operation: Storage operation being performed
            file_path: File path involved in the operation
            **context: Additional context
        """
        error_context = {
            "error_category": "storage_error",
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
        
        if file_path:
            error_context["file_path"] = str(file_path)
        
        self.logger.error(
            "Storage error occurred",
            **error_context,
            exc_info=error
        )
    
    def report_validation_error(
        self,
        error: Exception,
        data_type: str,
        validation_context: Optional[Dict[str, Any]] = None,
        **context
    ):
        """
        Report a data validation error.
        
        Args:
            error: Exception that occurred
            data_type: Type of data being validated
            validation_context: Context about the validation
            **context: Additional context
        """
        error_context = {
            "error_category": "validation_error",
            "data_type": data_type,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        }
        
        if validation_context:
            error_context["validation_context"] = validation_context
        
        self.logger.error(
            "Validation error occurred",
            **error_context,
            exc_info=error
        )
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize request parameters by removing or masking sensitive data.
        
        Args:
            params: Original parameters
        
        Returns:
            Sanitized parameters
        """
        sanitized = {}
        sensitive_keys = {"password", "token", "key", "secret", "credential"}
        
        for key, value in params.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_params(value)
            else:
                sanitized[key] = value
        
        return sanitized


def setup_logging(config: BenchmarkConfig) -> BenchmarkLogger:
    """
    Set up logging configuration for the application.
    
    Args:
        config: BenchmarkConfig instance
    
    Returns:
        Configured BenchmarkLogger instance
    """
    benchmark_logger = BenchmarkLogger(config)
    benchmark_logger.configure_logging()
    return benchmark_logger


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance (convenience function).
    
    Args:
        name: Logger name
    
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)