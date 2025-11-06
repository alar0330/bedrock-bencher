"""
Core benchmarking orchestration.
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from .models import BenchmarkItem, BedrockResponse, RunConfig
from .dataset import DatasetLoader
from .client import BedrockClient
from .storage import StorageManager
from .backoff import BackoffHandler
from .logging import get_logger


logger = get_logger(__name__)


class BenchmarkProgress:
    """Tracks progress of a benchmark run."""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time
    
    def update(self, completed: int = 0, failed: int = 0):
        """Update progress counters."""
        self.completed_items += completed
        self.failed_items += failed
        self.last_update = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        processed = self.completed_items + self.failed_items
        if processed == 0:
            return 0.0
        return (self.completed_items / processed) * 100
    
    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as a percentage."""
        processed = self.completed_items + self.failed_items
        return (processed / self.total_items) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (self.last_update - self.start_time).total_seconds()
    
    @property
    def estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        processed = self.completed_items + self.failed_items
        if processed == 0 or processed >= self.total_items:
            return None
        
        rate = processed / self.elapsed_time
        remaining_items = self.total_items - processed
        return remaining_items / rate if rate > 0 else None


class BenchmarkCore:
    """
    Orchestrates the benchmarking process by coordinating dataset loading,
    Bedrock client interactions, and storage management with concurrent processing.
    """
    
    def __init__(
        self,
        storage_manager: Optional[StorageManager] = None,
        progress_callback: Optional[Callable[[BenchmarkProgress], None]] = None
    ):
        """
        Initialize the benchmark coordinator.
        
        Args:
            storage_manager: Storage manager instance (creates default if None)
            progress_callback: Optional callback function for progress updates
        """
        self.storage_manager = storage_manager or StorageManager()
        self.progress_callback = progress_callback
        self._shutdown_requested = False
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, requesting graceful shutdown...")
            self._shutdown_requested = True
        
        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run_benchmark(
        self,
        experiment_id: str,
        run_config: RunConfig,
        dataset_path: Optional[str] = None,
        resume_from_item: Optional[str] = None,
        run_name: Optional[str] = None,
        region: Optional[str] = None
    ) -> str:
        """
        Run a complete benchmark with the given configuration.
        
        Args:
            experiment_id: ID of the experiment to run under
            run_config: Configuration for this benchmark run
            dataset_path: Path to dataset file (overrides run_config.dataset_path if provided)
            resume_from_item: Item ID to resume from (for interrupted runs)
            run_name: Human-readable name for this run
            region: AWS region for Bedrock service (defaults to us-east-1)
        
        Returns:
            Run ID of the completed benchmark
        
        Raises:
            ValueError: If experiment doesn't exist or configuration is invalid
            Exception: For other errors during benchmarking
        """
        # Validate experiment exists
        experiment_metadata = self.storage_manager.get_experiment_metadata(experiment_id)
        if not experiment_metadata:
            raise ValueError(f"Experiment {experiment_id} does not exist")
        
        # Use provided dataset path or fall back to config
        dataset_file = dataset_path or run_config.dataset_path
        if not dataset_file:
            raise ValueError("Dataset path must be provided either in run_config or as parameter")
        
        # Update run config with actual dataset path
        run_config.dataset_path = dataset_file
        
        logger.info(
            "Starting benchmark run",
            experiment_id=experiment_id,
            model_id=run_config.model_id,
            dataset_path=dataset_file,
            max_concurrent=run_config.max_concurrent
        )
        
        # Load dataset
        logger.info("Loading dataset", dataset_path=dataset_file)
        dataset_loader = DatasetLoader()
        try:
            dataset_items = dataset_loader.load_dataset(dataset_file)
        except Exception as e:
            logger.error("Failed to load dataset", dataset_path=dataset_file, error=str(e), exc_info=e)
            raise
        
        logger.info("Dataset loaded successfully", dataset_path=dataset_file, item_count=len(dataset_items))
        
        # Create run
        run_id = self.storage_manager.create_run(experiment_id, run_config, run_name)
        logger.info("Created benchmark run", run_id=run_id, experiment_id=experiment_id)
        
        # Filter items if resuming
        if resume_from_item:
            # Find the index of the resume item
            resume_index = None
            for i, item in enumerate(dataset_items):
                if item.id == resume_from_item:
                    resume_index = i
                    break
            
            if resume_index is not None:
                dataset_items = dataset_items[resume_index:]
                logger.info(
                    "Resuming benchmark from specific item",
                    resume_item_id=resume_from_item,
                    remaining_items=len(dataset_items)
                )
            else:
                logger.warning(
                    "Resume item not found, processing all items",
                    resume_item_id=resume_from_item
                )
        
        # Initialize progress tracking
        progress = BenchmarkProgress(len(dataset_items))
        
        # Create Bedrock client
        backoff_handler = BackoffHandler()
        
        try:
            async with BedrockClient(
                model_id=run_config.model_id,
                region=region or "us-east-1",
                backoff_handler=backoff_handler
            ) as client:
                
                # Process items with concurrent execution
                await self._process_items_concurrently(
                    client=client,
                    dataset_items=dataset_items,
                    run_id=run_id,
                    run_config=run_config,
                    progress=progress
                )
                
        except Exception as e:
            logger.error("Error during benchmark execution", error=str(e), exc_info=e)
            raise
        
        # Final progress report
        if self.progress_callback:
            self.progress_callback(progress)
        
        logger.info(
            "Benchmark run completed",
            run_id=run_id,
            processed_items=progress.completed_items + progress.failed_items,
            total_items=progress.total_items,
            success_rate=round(progress.success_rate, 1),
            total_time=round(progress.elapsed_time, 1)
        )
        
        return run_id
    
    async def _process_items_concurrently(
        self,
        client: BedrockClient,
        dataset_items: List[BenchmarkItem],
        run_id: str,
        run_config: RunConfig,
        progress: BenchmarkProgress
    ):
        """
        Process dataset items concurrently with progress tracking and graceful shutdown.
        
        Args:
            client: Bedrock client instance
            dataset_items: List of items to process
            run_id: Run ID for storing results
            run_config: Run configuration
            progress: Progress tracker
        """
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(run_config.max_concurrent)
        
        async def process_single_item(item: BenchmarkItem) -> Optional[BedrockResponse]:
            """Process a single benchmark item with error handling."""
            if self._shutdown_requested:
                return None
            
            async with semaphore:
                try:
                    # Make the API call
                    response = await client.invoke_model(
                        prompt=item.prompt,
                        item_id=item.id,
                        system_prompt=run_config.system_prompt,
                        **run_config.model_params
                    )
                    
                    # Save response immediately
                    self.storage_manager.save_response(run_id, response)
                    
                    # Update progress based on response quality
                    if response.finish_reason in ['stop', 'end_turn']:
                        progress.update(completed=1)
                        logger.debug(f"Completed item {item.id} successfully")
                    else:
                        # API succeeded but response didn't complete successfully
                        progress.update(failed=1)
                        logger.debug(f"Item {item.id} completed with finish_reason: {response.finish_reason}")
                    
                    # Report progress if callback is provided
                    if self.progress_callback:
                        self.progress_callback(progress)
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Failed to process item {item.id}: {e}")
                    progress.update(failed=1)
                    
                    # Report progress even for failures
                    if self.progress_callback:
                        self.progress_callback(progress)
                    
                    return None
        
        # Create tasks for all items
        tasks = [process_single_item(item) for item in dataset_items]
        
        # Process tasks in batches to avoid overwhelming the system
        batch_size = min(100, len(tasks))  # Process in batches of 100 or fewer
        
        for i in range(0, len(tasks), batch_size):
            if self._shutdown_requested:
                logger.info("Shutdown requested, stopping processing")
                break
            
            batch_tasks = tasks[i:i + batch_size]
            
            try:
                # Wait for batch completion
                await asyncio.gather(*batch_tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                # Continue with next batch
                continue
        
        # Handle shutdown case
        if self._shutdown_requested:
            logger.info("Benchmark run interrupted by shutdown request")
            logger.info(f"Processed {progress.completed_items + progress.failed_items}/{progress.total_items} items before shutdown")
    
    def create_experiment(self, name: str, description: str = "") -> str:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Optional experiment description
        
        Returns:
            Experiment ID
        """
        return self.storage_manager.create_experiment(name, description)
    
    def list_experiments(self):
        """List all experiments."""
        return self.storage_manager.list_experiments()
    
    def get_experiment_metadata(self, experiment_id: str):
        """Get experiment metadata."""
        return self.storage_manager.get_experiment_metadata(experiment_id)
    
    def list_runs(self, experiment_id: str):
        """List runs for an experiment."""
        return self.storage_manager.list_runs(experiment_id)
    
    def get_run_summary(self, run_id: str):
        """Get run summary statistics."""
        return self.storage_manager.get_run_summary(run_id)
    
    def export_run_to_dataframe(self, run_id: str, dataset_path: Optional[str] = None):
        """
        Export run results to pandas DataFrame.
        
        Args:
            run_id: Run ID to export
            dataset_path: Optional path to dataset file for including prompts/expected responses
        
        Returns:
            pandas DataFrame with run results
        """
        dataset_items = None
        if dataset_path:
            try:
                dataset_loader = DatasetLoader()
                dataset_items = dataset_loader.load_dataset(dataset_path)
                print(dataset_items[:5])
            except Exception as e:
                logger.warning(f"Could not load dataset for export: {e}")
        
        return self.storage_manager.export_run_to_dataframe(run_id, dataset_items)
    
    def export_multiple_runs_to_dataframe(self, run_ids: List[str], dataset_path: Optional[str] = None):
        """
        Export multiple runs to a single DataFrame for comparison.
        
        Args:
            run_ids: List of run IDs to export
            dataset_path: Optional path to dataset file for including prompts/expected responses
        
        Returns:
            pandas DataFrame with combined run results
        """
        dataset_items = None
        if dataset_path:
            try:
                dataset_loader = DatasetLoader()
                dataset_items = dataset_loader.load_dataset(dataset_path)
            except Exception as e:
                logger.warning(f"Could not load dataset for export: {e}")
        
        return self.storage_manager.export_multiple_runs_to_dataframe(run_ids, dataset_items)