"""
Amazon Bedrock embeddings client implementation using the InvokeModel API.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

import aioboto3
from botocore.exceptions import ClientError

from .backoff import BackoffHandler
from .image_utils import load_and_encode_image
from .logging import get_logger
from .models import EmbeddingResponse


logger = get_logger(__name__)


class EmbeddingsClient:
    """
    Async client for Amazon Bedrock embedding models using InvokeModel API.
    
    Supports:
    - Titan Multimodal Embeddings (amazon.titan-embed-image-v1)
    - Nova Multimodal Embeddings (amazon.nova-2-multimodal-embeddings-v1:0)
    - Cohere Embeddings v4 (cohere.embed-v4) - with multimodal and interleaved content support
    """
    
    # Model family registry for request/response formatting
    MODEL_FAMILIES = {
        "titan": ["amazon.titan-embed-image-v1"],  # Multi-modal only
        "nova": ["amazon.nova-2-multimodal-embeddings-v1:0"],
        "cohere": ["cohere.embed-v4"]  # v4 only - supports interleaved text+image
    }
    
    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        aws_profile: Optional[str] = None,
        session: Optional[aioboto3.Session] = None,
        backoff_handler: Optional[BackoffHandler] = None
    ):
        """
        Initialize the embeddings client.
        
        Args:
            model_id: The Bedrock embedding model identifier to use
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
    
    def _get_model_family(self, model_id: str) -> str:
        """
        Determine model family from model ID.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Model family name (titan, nova, or cohere)
            
        Raises:
            ValueError: If model is not supported
        """
        # Check registry first
        for family, model_ids in self.MODEL_FAMILIES.items():
            if any(model_id.startswith(mid) for mid in model_ids):
                return family
        
        # Fallback: try prefix matching for new models
        if "titan" in model_id.lower():
            return "titan"
        elif "nova" in model_id.lower():
            return "nova"
        elif "cohere" in model_id.lower():
            return "cohere"
        else:
            raise ValueError(
                f"Unsupported embedding model: {model_id}. "
                f"Supported families: {list(self.MODEL_FAMILIES.keys())}"
            )
    
    def _format_titan_request(
        self,
        text: Optional[str],
        image_base64: Optional[str],
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format request body for Titan Multimodal Embeddings (amazon.titan-embed-image-v1).
        
        Uses embeddingConfig structure:
        {
            "inputText": "...",  // optional
            "inputImage": "...", // optional (base64)
            "embeddingConfig": {
                "outputEmbeddingLength": 256 | 384 | 1024
            }
        }
        
        Args:
            text: Text input (optional)
            image_base64: Base64-encoded image data (optional)
            model_params: User-provided model parameters
            
        Returns:
            Request body dictionary formatted for Titan Multimodal model
        """
        request_body = {}
        
        # Add text input if provided
        if text:
            request_body["inputText"] = text
        
        # Add image input if provided
        if image_base64:
            request_body["inputImage"] = image_base64
        
        # Build embeddingConfig
        embedding_config = {}
        
        # Check for dimensions in various formats
        if "dimensions" in model_params:
            embedding_config["outputEmbeddingLength"] = model_params["dimensions"]
        elif "outputEmbeddingLength" in model_params:
            embedding_config["outputEmbeddingLength"] = model_params["outputEmbeddingLength"]
        elif "embeddingConfig" in model_params:
            embedding_config = model_params["embeddingConfig"]
        
        if embedding_config:
            request_body["embeddingConfig"] = embedding_config
        
        return request_body
    
    def _format_nova_request(
        self,
        text: Optional[str],
        image_base64: Optional[str],
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format request body for Nova Multi-Modal Embeddings models.
        
        Nova uses a nested structure with schemaVersion, taskType, and singleEmbeddingParams.
        
        Args:
            text: Text input (optional)
            image_base64: Base64-encoded image data (optional)
            model_params: User-provided model parameters
            
        Returns:
            Request body dictionary formatted for Nova models
        """
        request_body = {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING"
        }
        
        single_embedding_params = {}
        
        # Set default embedding purpose if not provided
        single_embedding_params["embeddingPurpose"] = model_params.get(
            "embeddingPurpose", "GENERIC_INDEX"
        )
        
        # Set embedding dimension if provided
        if "embeddingDimension" in model_params:
            single_embedding_params["embeddingDimension"] = model_params["embeddingDimension"]
        
        # Add text input if provided
        if text:
            single_embedding_params["text"] = {
                "value": text,
                "truncationMode": model_params.get("truncationMode", "END")
            }
        
        # Add image input if provided
        if image_base64:
            single_embedding_params["image"] = {
                "source": {"bytes": image_base64},
                "format": model_params.get("imageFormat", "png"),
                "detailLevel": model_params.get("detailLevel", "STANDARD_IMAGE")
            }
        
        request_body["singleEmbeddingParams"] = single_embedding_params
        
        return request_body
    
    def _format_cohere_request(
        self,
        text: Optional[str],
        image_base64: Optional[str],
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format request body for Cohere Embeddings models (v3 and v4).
        
        Supports three modes:
        - Text-only: uses 'texts' array (v3 & v4)
        - Image-only: uses 'images' array with data URI (v4)
        - Interleaved: uses 'inputs' with content array (v4)
        
        Args:
            text: Text input (optional)
            image_base64: Base64-encoded image data (optional)
            model_params: User-provided model parameters
            
        Returns:
            Request body dictionary formatted for Cohere models
        """
        request_body = {}
        
        # Handle text-only input
        if text and not image_base64:
            request_body["texts"] = [text]
        
        # Handle image-only input (v4 feature)
        elif image_base64 and not text:
            image_format = model_params.get("imageFormat", "png")
            request_body["images"] = [f"data:image/{image_format};base64,{image_base64}"]
        
        # Handle interleaved text+image (v4 feature)
        elif text and image_base64:
            image_format = model_params.get("imageFormat", "png")
            request_body["inputs"] = [{
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": f"data:image/{image_format};base64,{image_base64}"}
                ]
            }]
        
        # Required parameter
        if "input_type" in model_params:
            request_body["input_type"] = model_params["input_type"]
        
        # Optional v3/v4 parameters
        if "truncate" in model_params:
            request_body["truncate"] = model_params["truncate"]
        
        # v4-specific optional parameters
        if "embedding_types" in model_params:
            request_body["embedding_types"] = model_params["embedding_types"]
        if "output_dimension" in model_params:
            request_body["output_dimension"] = model_params["output_dimension"]
        if "max_tokens" in model_params:
            request_body["max_tokens"] = model_params["max_tokens"]
        
        return request_body
    
    def _parse_titan_response(self, response_body: Dict[str, Any]) -> tuple[list[float], Optional[int]]:
        """
        Parse response from Titan Multi-Modal Embeddings models.
        
        Args:
            response_body: The response body from InvokeModel API
            
        Returns:
            Tuple of (embedding vector, input token count)
        """
        embedding = response_body.get("embedding", [])
        token_count = response_body.get("inputTextTokenCount")
        return embedding, token_count
    
    def _parse_nova_response(self, response_body: Dict[str, Any]) -> tuple[list[float], Optional[int]]:
        """
        Parse response from Nova Multi-Modal Embeddings models.
        
        Args:
            response_body: The response body from InvokeModel API
            
        Returns:
            Tuple of (embedding vector, input token count)
        """
        embedding = response_body.get("embedding", [])
        token_count = response_body.get("inputTextTokenCount")
        return embedding, token_count
    
    def _parse_cohere_response(self, response_body: Dict[str, Any]) -> tuple[list[float], Optional[int]]:
        """
        Parse response from Cohere Embeddings models.
        
        Cohere returns embeddings in an array, so we extract the first one.
        Cohere does not provide token count in the response.
        
        Args:
            response_body: The response body from InvokeModel API
            
        Returns:
            Tuple of (embedding vector, None) - Cohere doesn't provide token count
        """
        embeddings = response_body.get("embeddings", [[]])
        
        # Cohere v4 can return embeddings as a dict (when multiple types requested)
        # or as a list (when single type requested)
        if isinstance(embeddings, dict):
            # Multiple embedding types - get 'float' type by default
            embedding = embeddings.get("float", [[]])[0] if embeddings.get("float") else []
        elif isinstance(embeddings, list):
            # Single embedding type - extract first embedding
            embedding = embeddings[0] if embeddings else []
        else:
            embedding = []
        
        return embedding, None
    
    async def invoke_model(
        self,
        item_id: str,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        **model_params
    ) -> EmbeddingResponse:
        """
        Invoke embedding model with text or image input.
        
        Args:
            item_id: Unique identifier for tracking
            text: Text input (optional)
            image_path: Path to image file (optional)
            **model_params: Model-specific parameters
            
        Returns:
            EmbeddingResponse with embedding vector and metadata
            
        Raises:
            ValueError: If neither text nor image_path is provided
            FileNotFoundError: If image file doesn't exist
        """
        # Subtask 6.1: Add input validation
        if not text and not image_path:
            raise ValueError(
                "At least one of 'text' or 'image_path' must be provided"
            )
        
        # Subtask 6.2: Implement image loading logic
        image_base64 = None
        if image_path:
            try:
                image_base64 = load_and_encode_image(image_path)
            except (FileNotFoundError, ValueError) as e:
                # Log the error and return error response
                logger.error(
                    "Image processing failed",
                    item_id=item_id,
                    model_id=self.model_id,
                    image_path=image_path,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=e
                )
                return EmbeddingResponse(
                    item_id=item_id,
                    embedding=[],
                    model_id=self.model_id,
                    timestamp=datetime.utcnow(),
                    latency_ms=0,
                    is_error=True,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    raw_response={}
                )
        
        # Subtask 6.3: Implement request formatting dispatch
        try:
            model_family = self._get_model_family(self.model_id)
            
            # Call appropriate formatter based on model family
            if model_family == "titan":
                request_body = self._format_titan_request(text, image_base64, model_params)
            elif model_family == "nova":
                request_body = self._format_nova_request(text, image_base64, model_params)
            elif model_family == "cohere":
                request_body = self._format_cohere_request(text, image_base64, model_params)
            else:
                raise ValueError(f"Unsupported model family: {model_family}")
            
            # Subtask 6.4: Implement InvokeModel API call with timing
            start_time = time.perf_counter()
            
            # Define the API call function for backoff handler
            async def invoke_api():
                await self._ensure_client()
                response = await self._client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body)
                )
                return response
            
            # Execute with backoff retry logic
            response = await self.backoff_handler.execute_with_backoff(invoke_api)
            
            # Calculate latency
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Subtask 6.5: Implement response parsing dispatch
            response_body = json.loads(await response['body'].read())
            
            # Call appropriate parser based on model family
            if model_family == "titan":
                embedding, token_count = self._parse_titan_response(response_body)
            elif model_family == "nova":
                embedding, token_count = self._parse_nova_response(response_body)
            elif model_family == "cohere":
                embedding, token_count = self._parse_cohere_response(response_body)
            else:
                raise ValueError(f"Unsupported model family: {model_family}")
            
            # Subtask 6.6: Create and return EmbeddingResponse
            return EmbeddingResponse(
                item_id=item_id,
                embedding=embedding,
                model_id=self.model_id,
                timestamp=datetime.utcnow(),
                latency_ms=latency_ms,
                input_tokens=token_count,
                raw_response=response_body,
                is_error=False
            )
        
        # Subtask 6.7: Implement error handling and logging
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            logger.error(
                "Bedrock API call failed",
                item_id=item_id,
                model_id=self.model_id,
                error_code=error_code,
                error_message=error_message,
                has_text=text is not None,
                has_image=image_path is not None,
                exc_info=e
            )
            
            return EmbeddingResponse(
                item_id=item_id,
                embedding=[],
                model_id=self.model_id,
                timestamp=datetime.utcnow(),
                latency_ms=0,
                is_error=True,
                error_type=error_code,
                error_message=error_message,
                raw_response={}
            )
        
        except Exception as e:
            logger.error(
                "Unexpected error during embedding invocation",
                item_id=item_id,
                model_id=self.model_id,
                error_type=type(e).__name__,
                error_message=str(e),
                has_text=text is not None,
                has_image=image_path is not None,
                exc_info=e
            )
            
            return EmbeddingResponse(
                item_id=item_id,
                embedding=[],
                model_id=self.model_id,
                timestamp=datetime.utcnow(),
                latency_ms=0,
                is_error=True,
                error_type=type(e).__name__,
                error_message=str(e),
                raw_response={}
            )
