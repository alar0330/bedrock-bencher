"""
Storage management for experiments and runs.
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from .models import ExperimentMetadata, RunConfig, BedrockResponse, BenchmarkItem


def make_path_safe(name: str, max_length: int = 30) -> str:
    """Convert name to filesystem-safe string."""
    if not name:
        return "unnamed"
    
    # Convert to lowercase and replace spaces/special chars with hyphens
    safe_name = re.sub(r'[^\w\s-]', '', name.lower())
    safe_name = re.sub(r'[\s_]+', '-', safe_name)
    safe_name = safe_name.strip('-')
    
    # Truncate if too long
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length].rstrip('-')
    
    return safe_name or "unnamed"


def generate_timestamp() -> str:
    """Generate compact timestamp: YYYYMMDD-HHMMSS."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def generate_short_uuid(length: int = 4) -> str:
    """Generate short UUID suffix."""
    return str(uuid.uuid4()).replace('-', '')[:length]


def create_folder_name(human_name: str) -> str:
    """Create folder name: name_timestamp_uuid."""
    safe_name = make_path_safe(human_name)
    timestamp = generate_timestamp()
    short_uuid = generate_short_uuid()
    return f"{safe_name}_{timestamp}_{short_uuid}"


def create_experiment_folder_name(human_name: str) -> str:
    """Create experiment folder name: just the path-safe name (no timestamp/uuid)."""
    return make_path_safe(human_name)


class StorageManager:
    """Manages hierarchical storage of experiments and runs."""
    
    def __init__(self, storage_path: str = "./experiments"):
        """Initialize storage manager with base storage path."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    

    
    def create_experiment(self, name: str, description: str = "") -> str:
        """Create a new experiment or reuse existing one. Return folder name as ID."""
        experiment_id = create_experiment_folder_name(name)
        experiment_path = self.storage_path / experiment_id
        
        # Check if experiment already exists
        if experiment_path.exists():
            # Experiment exists, return existing ID for reuse
            return experiment_id
        
        # Create new experiment
        experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Create runs directory
        runs_path = experiment_path / "runs"
        runs_path.mkdir(exist_ok=True)
        
        # Create experiment metadata
        metadata = ExperimentMetadata(
            id=experiment_id,
            name=name,
            description=description,
            created_at=datetime.now(),
            runs=[]
        )
        
        # Save metadata
        metadata_path = experiment_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'id': metadata.id,
                'name': metadata.name,
                'description': metadata.description,
                'created_at': metadata.created_at.isoformat(),
                'runs': metadata.runs
            }, f, indent=2)
        
        return experiment_id
    
    def create_run(self, experiment_id: str, config: RunConfig, run_name: str = None) -> str:
        """Create a new run within an experiment and return its folder name as ID."""
        experiment_path = self.storage_path / experiment_id
        
        if not experiment_path.exists():
            raise ValueError(f"Experiment {experiment_id} does not exist")
        
        # Generate run folder name
        if not run_name:
            # Default to model name if no run name provided
            run_name = config.model_id.split('.')[-1] if config.model_id else "run"
        
        run_id = create_folder_name(run_name)
        
        # Create run directory
        run_path = experiment_path / "runs" / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        
        # Save run configuration
        config_path = run_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_id': config.model_id,
                'system_prompt': config.system_prompt,
                'model_params': config.model_params,
                'max_concurrent': config.max_concurrent,
                'dataset_path': config.dataset_path,
                'created_at': datetime.now().isoformat()
            }, f, indent=2)
        
        # Create empty responses file
        responses_path = run_path / "responses.jsonl"
        responses_path.touch()
        
        # Update experiment metadata to include this run
        self._add_run_to_experiment(experiment_id, run_id)
        
        return run_id
    
    def save_response(self, run_id: str, response: BedrockResponse) -> None:
        """Save a response to the appropriate run."""
        # Find the run path by searching through experiments
        run_path = self._find_run_path(run_id)
        if not run_path:
            raise ValueError(f"Run {run_id} not found")
        
        responses_path = run_path / "responses.jsonl"
        
        # Append response to JSONL file
        response_data = {
            'item_id': response.item_id,
            'response_text': response.response_text,
            'model_id': response.model_id,
            'timestamp': response.timestamp.isoformat(),
            'latency_ms': response.latency_ms,
            'input_tokens': response.input_tokens,
            'output_tokens': response.output_tokens,
            'finish_reason': response.finish_reason,
            'raw_response': response.raw_response
        }
        
        with open(responses_path, 'a') as f:
            f.write(json.dumps(response_data) + '\n')
    
    def load_responses(self, run_id: str) -> List[BedrockResponse]:
        """Load all responses for a given run."""
        run_path = self._find_run_path(run_id)
        if not run_path:
            raise ValueError(f"Run {run_id} not found")
        
        responses_path = run_path / "responses.jsonl"
        if not responses_path.exists():
            return []
        
        responses = []
        with open(responses_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    response = BedrockResponse(
                        item_id=data['item_id'],
                        response_text=data['response_text'],
                        model_id=data['model_id'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        latency_ms=data['latency_ms'],
                        input_tokens=data['input_tokens'],
                        output_tokens=data['output_tokens'],
                        finish_reason=data['finish_reason'],
                        raw_response=data.get('raw_response', {})
                    )
                    responses.append(response)
        
        return responses
    
    def get_experiment_metadata(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Get experiment metadata by ID."""
        experiment_path = self.storage_path / experiment_id
        metadata_path = experiment_path / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            return ExperimentMetadata(
                id=data['id'],
                name=data['name'],
                description=data['description'],
                created_at=datetime.fromisoformat(data['created_at']),
                runs=data['runs']
            )
    
    def get_run_config(self, run_id: str) -> Optional[RunConfig]:
        """Get run configuration by ID."""
        run_path = self._find_run_path(run_id)
        if not run_path:
            return None
        
        config_path = run_path / "config.json"
        if not config_path.exists():
            return None
        
        with open(config_path, 'r') as f:
            data = json.load(f)
            return RunConfig(
                model_id=data['model_id'],
                system_prompt=data.get('system_prompt'),
                model_params=data.get('model_params', {}),
                max_concurrent=data.get('max_concurrent', 10),
                dataset_path=data.get('dataset_path', "")
            )
    
    def list_experiments(self) -> List[ExperimentMetadata]:
        """List all experiments."""
        experiments = []
        for experiment_dir in self.storage_path.iterdir():
            if experiment_dir.is_dir():
                metadata = self.get_experiment_metadata(experiment_dir.name)
                if metadata:
                    experiments.append(metadata)
        return experiments
    
    def list_runs(self, experiment_id: str) -> List[str]:
        """List all run IDs for an experiment."""
        experiment_path = self.storage_path / experiment_id
        runs_path = experiment_path / "runs"
        
        if not runs_path.exists():
            return []
        
        return [run_dir.name for run_dir in runs_path.iterdir() if run_dir.is_dir()]
    
    def _add_run_to_experiment(self, experiment_id: str, run_id: str) -> None:
        """Add a run ID to the experiment's metadata."""
        metadata = self.get_experiment_metadata(experiment_id)
        if metadata:
            metadata.runs.append(run_id)
            
            # Save updated metadata
            experiment_path = self.storage_path / experiment_id
            metadata_path = experiment_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'id': metadata.id,
                    'name': metadata.name,
                    'description': metadata.description,
                    'created_at': metadata.created_at.isoformat(),
                    'runs': metadata.runs
                }, f, indent=2)
    
    def export_run_to_dataframe(self, run_id: str, dataset_items: Optional[List[BenchmarkItem]] = None) -> pd.DataFrame:
        """Export run results to a pandas DataFrame with proper column mapping."""
        responses = self.load_responses(run_id)
        run_config = self.get_run_config(run_id)
        
        if not responses:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'run_id', 'item_id', 'prompt', 'expected_response', 'actual_response',
                'model_id', 'system_prompt', 'timestamp', 'latency_ms', 
                'input_tokens', 'output_tokens', 'finish_reason'
            ])
        
        # Create dataset lookup for prompts and expected responses
        dataset_lookup = {}
        if dataset_items:
            dataset_lookup = {item.id: item for item in dataset_items}
        
        # Build DataFrame data
        data = []
        for response in responses:
            item = dataset_lookup.get(response.item_id)
            row = {
                'run_id': run_id,
                'item_id': response.item_id,
                'prompt': item.prompt if item else '',
                'expected_response': item.expected_response if item else '',
                'actual_response': response.response_text,
                'model_id': response.model_id,
                'system_prompt': run_config.system_prompt if run_config else '',
                'timestamp': response.timestamp,
                'latency_ms': response.latency_ms,
                'input_tokens': response.input_tokens,
                'output_tokens': response.output_tokens,
                'finish_reason': response.finish_reason
            }
            
            # Add metadata columns if available
            if item and item.metadata:
                for key, value in item.metadata.items():
                    row[f'metadata_{key}'] = value
            
            # Add model parameters if available
            if run_config and run_config.model_params:
                for key, value in run_config.model_params.items():
                    row[f'model_param_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def export_multiple_runs_to_dataframe(self, run_ids: List[str], dataset_items: Optional[List[BenchmarkItem]] = None) -> pd.DataFrame:
        """Export multiple runs to a single DataFrame for comparison."""
        all_dataframes = []
        
        for run_id in run_ids:
            try:
                df = self.export_run_to_dataframe(run_id, dataset_items)
                if not df.empty:
                    all_dataframes.append(df)
            except ValueError as e:
                # Log warning but continue with other runs
                print(f"Warning: Could not export run {run_id}: {e}")
                continue
        
        if not all_dataframes:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'run_id', 'item_id', 'prompt', 'expected_response', 'actual_response',
                'model_id', 'system_prompt', 'timestamp', 'latency_ms', 
                'input_tokens', 'output_tokens', 'finish_reason'
            ])
        
        # Concatenate all DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Sort by run_id and item_id for consistent ordering
        combined_df = combined_df.sort_values(['run_id', 'item_id']).reset_index(drop=True)
        
        return combined_df
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary statistics for a run."""
        responses = self.load_responses(run_id)
        run_config = self.get_run_config(run_id)
        
        if not responses:
            return {
                'run_id': run_id,
                'total_responses': 0,
                'model_id': run_config.model_id if run_config else '',
                'avg_latency_ms': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'success_rate': 0.0
            }
        
        # Calculate summary statistics
        total_responses = len(responses)
        avg_latency = sum(r.latency_ms for r in responses) / total_responses
        total_input_tokens = sum(r.input_tokens for r in responses)
        total_output_tokens = sum(r.output_tokens for r in responses)
        # Count successful responses (both 'stop' and 'end_turn' are successful completions)
        successful_responses = sum(1 for r in responses if r.finish_reason in ['stop', 'end_turn'])
        success_rate = successful_responses / total_responses if total_responses > 0 else 0.0
        
        return {
            'run_id': run_id,
            'total_responses': total_responses,
            'model_id': run_config.model_id if run_config else '',
            'system_prompt': run_config.system_prompt if run_config else '',
            'avg_latency_ms': round(avg_latency, 2),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'success_rate': round(success_rate, 3),
            'created_at': responses[0].timestamp if responses else None
        }
    
    def _find_run_path(self, run_id: str) -> Optional[Path]:
        """Find the path to a run by searching through all experiments."""
        for experiment_dir in self.storage_path.iterdir():
            if experiment_dir.is_dir():
                runs_dir = experiment_dir / "runs"
                if runs_dir.exists():
                    run_path = runs_dir / run_id
                    if run_path.exists():
                        return run_path
        return None