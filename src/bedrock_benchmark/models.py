"""
Data models for the Bedrock Benchmark Toolkit.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class BenchmarkItem:
    """Represents a single benchmark item with prompt and expected response."""
    id: str
    prompt: str
    expected_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BedrockResponse:
    """Represents a response from Amazon Bedrock with timing and token metadata."""
    item_id: str
    response_text: str
    model_id: str
    timestamp: datetime
    latency_ms: int
    input_tokens: int
    output_tokens: int
    finish_reason: str
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    model_id: str
    system_prompt: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    max_concurrent: int = 10
    dataset_path: str = ""


@dataclass
class ExperimentMetadata:
    """Metadata for a benchmark experiment."""
    id: str
    name: str
    description: str
    created_at: datetime