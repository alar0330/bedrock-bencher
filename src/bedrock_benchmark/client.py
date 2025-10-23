"""
Amazon Bedrock client implementation using the Converse API.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import aioboto3
from botocore.exceptions import ClientError

from .models import BedrockResponse
from .backoff import BackoffHandler
from .logging import get_logger


logger = get_logger(__name__)


class BedrockClient:
    """
    Async client wrapper for Amazon Bedrock using the Converse API.
    
    Provides methods for single and batch model invocations with
    response metadata capture including latency, tokens, and timestamps.
    """
    
    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        aws_profile: Optional[str] = None,
        session: Optional[aioboto3.Session] = None,
        backoff_handler: Optional[BackoffHandler] = None
    ):
        """
        Initialize the Bedrock client.
        
        Args:
            model_id: The Bedrock model identifier to use
            region: AWS region for Bedrock service
            aws_profile: AWS profile name (optional)
            session: Existing aioboto3 session (optional)
            backoff_handler: Custom backoff handler (optional)
        """
        self.model_id = model_id
        self.region = region
        self.aws_profile = aws_profile
        
        if session:
            self.session = session
        else:
            self.session = aioboto3.Session(profile_name=aws_profile)
        
        # Initialize backoff handler
        if backoff_handler:
            self.backoff_handler = backoff_handler
        else:
            self.backoff_handler = BackoffHandler()
        
        self._client = None
        self._client_context = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client_context:
            await self._client_context.__aexit__(exc_type, exc_val, exc_tb)
    
    async def _ensure_client(self):
        """Ensure the Bedrock client is initialized."""
        if not self._client:
            client_context = self.session.client(
                'bedrock-runtime',
                region_name=self.region
            )
            self._client = await client_context.__aenter__()
            # Store the context for cleanup
            self._client_context = client_context
    
    async def invoke_model(
        self,
        prompt: str,
        item_id: str,
        system_prompt: Optional[str] = None,
        **model_params
    ) -> BedrockResponse:
        """
        Invoke a single model request using the Converse API.
        
        Args:
            prompt: The input prompt text
            item_id: Unique identifier for this benchmark item
            system_prompt: Optional system prompt
            **model_params: Additional model parameters (temperature, max_tokens, etc.)
        
        Returns:
            BedrockResponse with response text and metadata
        
        Raises:
            ClientError: For AWS API errors
            Exception: For other errors during invocation
        """
        await self._ensure_client()
        
        # Prepare the conversation messages
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
        
        # Prepare the request
        request_params = {
            "modelId": self.model_id,
            "messages": messages
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]
        
        # Add inference configuration if model parameters are provided
        if model_params:
            inference_config = {}
            if "temperature" in model_params:
                inference_config["temperature"] = model_params["temperature"]
            if "max_tokens" in model_params:
                inference_config["maxTokens"] = model_params["max_tokens"]
            if "top_p" in model_params:
                inference_config["topP"] = model_params["top_p"]
            if "stop_sequences" in model_params:
                inference_config["stopSequences"] = model_params["stop_sequences"]
            
            if inference_config:
                request_params["inferenceConfig"] = inference_config
        
        # Record start time for latency calculation
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Make the API call with backoff handling
            async def make_api_call():
                return await self._client.converse(**request_params)
            
            response = await self.backoff_handler.execute_with_backoff(make_api_call)
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract response data
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])
            
            # Get the text response
            response_text = ""
            if content and len(content) > 0:
                response_text = content[0].get("text", "")
            
            # Extract metadata
            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            
            finish_reason = output.get("message", {}).get("stopReason", "unknown")
            
            return BedrockResponse(
                item_id=item_id,
                response_text=response_text,
                model_id=self.model_id,
                timestamp=timestamp,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                raw_response=response
            )
            
        except ClientError as e:
            logger.error(
                "Bedrock API error",
                item_id=item_id,
                model_id=self.model_id,
                error_code=e.response.get('Error', {}).get('Code'),
                error_message=str(e),
                exc_info=e
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during model invocation",
                item_id=item_id,
                model_id=self.model_id,
                error=str(e),
                exc_info=e
            )
            raise
    
    async def invoke_batch(
        self,
        prompts: List[str],
        item_ids: List[str],
        system_prompt: Optional[str] = None,
        max_concurrent: int = 10,
        **model_params
    ) -> List[BedrockResponse]:
        """
        Invoke multiple model requests concurrently using the Converse API.
        
        Args:
            prompts: List of input prompt texts
            item_ids: List of unique identifiers for benchmark items
            system_prompt: Optional system prompt for all requests
            max_concurrent: Maximum number of concurrent requests
            **model_params: Additional model parameters
        
        Returns:
            List of BedrockResponse objects in the same order as input prompts
        
        Raises:
            ValueError: If prompts and item_ids lists have different lengths
        """
        if len(prompts) != len(item_ids):
            raise ValueError("prompts and item_ids must have the same length")
        
        await self._ensure_client()
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def invoke_with_semaphore(prompt: str, item_id: str) -> BedrockResponse:
            async with semaphore:
                return await self.invoke_model(
                    prompt=prompt,
                    item_id=item_id,
                    system_prompt=system_prompt,
                    **model_params
                )
        
        # Create tasks for all requests
        tasks = [
            invoke_with_semaphore(prompt, item_id)
            for prompt, item_id in zip(prompts, item_ids)
        ]
        
        # Execute all tasks concurrently and return results
        return await asyncio.gather(*tasks)