"""
Command-line interface for the Bedrock Benchmark Toolkit.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import click
import pandas as pd

from .core import BenchmarkCore, BenchmarkProgress
from .models import RunConfig
from .storage import StorageManager
from .config import load_config_with_auto_discovery, BenchmarkConfig
from .logging import setup_logging, get_logger


# Global logger - will be configured after config is loaded
logger = None


def setup_progress_callback(verbose: bool = False, progress_logger=None, log_structured: bool = False):
    """Create a progress callback function for CLI output."""
    last_reported_percentage = -1
    
    def progress_callback(progress: BenchmarkProgress):
        nonlocal last_reported_percentage
        
        completion_rate = progress.completion_rate
        
        # Report progress every 5% or if verbose mode is enabled
        if verbose or int(completion_rate) >= last_reported_percentage + 5:
            processed = progress.completed_items + progress.failed_items
            
            # Log structured progress only if explicitly requested (e.g., for file logging)
            if progress_logger and log_structured:
                progress_logger.logger.info(
                    "Benchmark progress",
                    completed_items=progress.completed_items,
                    failed_items=progress.failed_items,
                    total_items=progress.total_items,
                    completion_rate=round(completion_rate, 1),
                    success_rate=round(progress.success_rate, 1),
                    elapsed_time=round(progress.elapsed_time, 1),
                    estimated_remaining_time=round(progress.estimated_remaining_time, 1) if progress.estimated_remaining_time else None
                )
            
            click.echo(f"Progress: {processed}/{progress.total_items} "
                      f"({completion_rate:.1f}%) - "
                      f"Success: {progress.success_rate:.1f}% - "
                      f"Elapsed: {progress.elapsed_time:.1f}s")
            
            # Estimate remaining time
            remaining_time = progress.estimated_remaining_time
            if remaining_time:
                click.echo(f"Estimated remaining time: {remaining_time:.1f}s")
            
            last_reported_percentage = int(completion_rate)
    
    return progress_callback


@click.group()
@click.version_option()
@click.option('--config', '-c', 
              help='Path to configuration file')
@click.option('--storage-path', 
              help='Path to store experiment data (overrides config)', 
              envvar='BEDROCK_BENCHMARK_STORAGE_PATH')
@click.option('--log-level', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
              help='Logging level (overrides config)')
@click.option('--log-file', 
              help='Log file path (overrides config)')
@click.pass_context
def cli(ctx, config: Optional[str], storage_path: Optional[str], 
        log_level: Optional[str], log_file: Optional[str]):
    """Bedrock Benchmark Toolkit - A tool for benchmarking LLMs on Amazon Bedrock."""
    global logger
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Build configuration overrides from CLI options
    config_overrides = {}
    if storage_path:
        config_overrides['storage_path'] = storage_path
    if log_level:
        config_overrides['log_level'] = log_level.upper()
    if log_file:
        config_overrides['log_file'] = log_file
    
    try:
        # Load configuration
        benchmark_config = load_config_with_auto_discovery(
            config_file=config,
            config_overrides=config_overrides
        )
        
        # Set up logging
        benchmark_logger = setup_logging(benchmark_config)
        logger = benchmark_logger.get_logger(__name__)
        
        # Store configuration and components in context
        ctx.obj['config'] = benchmark_config
        ctx.obj['benchmark_logger'] = benchmark_logger
        
        # Initialize storage manager with config
        storage_manager = StorageManager(benchmark_config.storage_path)
        ctx.obj['storage_manager'] = storage_manager
        
        logger.info("CLI initialized", storage_path=benchmark_config.storage_path)
        
    except Exception as e:
        click.echo(f"Error initializing configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--dataset', '-d', required=True, 
              help='Path to JSONL dataset file')
@click.option('--model', '-m', required=True, 
              help='Bedrock model ID to benchmark')
@click.option('--experiment', '-e', 
              help='Experiment ID (if not provided, will create new experiment)')
@click.option('--experiment-name', 
              help='Name for new experiment (used if --experiment not provided)')
@click.option('--experiment-description', 
              help='Description for new experiment')
@click.option('--region', 
              help='AWS region (overrides config)', 
              envvar='AWS_DEFAULT_REGION')
@click.option('--max-concurrent', type=int,
              help='Maximum concurrent requests (overrides config)')
@click.option('--system-prompt', 
              help='System prompt for the model')
@click.option('--temperature', type=float, 
              help='Model temperature parameter')
@click.option('--max-tokens', type=int, 
              help='Maximum tokens to generate')
@click.option('--top-p', type=float, 
              help='Top-p sampling parameter')
@click.option('--run-name', 
              help='Human-readable name for this run (e.g., "baseline", "high-temp")')
@click.option('--resume-from', 
              help='Item ID to resume from (for interrupted runs)')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose progress reporting')
@click.option('--log-progress', is_flag=True, 
              help='Enable structured JSON progress logs (disabled by default to reduce console noise)')
@click.pass_context
def run_benchmark(
    ctx,
    dataset: str,
    model: str,
    experiment: Optional[str],
    experiment_name: Optional[str],
    experiment_description: Optional[str],
    region: Optional[str],
    max_concurrent: Optional[int],
    system_prompt: Optional[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    run_name: Optional[str],
    resume_from: Optional[str],
    verbose: bool,
    log_progress: bool
):
    """Run a benchmark against a Bedrock model."""
    config = ctx.obj['config']
    storage_manager = ctx.obj['storage_manager']
    benchmark_logger = ctx.obj['benchmark_logger']
    
    # Use configuration values with CLI overrides
    aws_region = region or config.aws_region
    concurrent_limit = max_concurrent or config.max_concurrent
    
    # Validate dataset file exists
    if not Path(dataset).exists():
        logger.error("Dataset file not found", dataset_path=dataset)
        click.echo(f"Error: Dataset file not found: {dataset}", err=True)
        sys.exit(1)
    
    # Create or validate experiment
    if experiment:
        # Use existing experiment
        experiment_metadata = storage_manager.get_experiment_metadata(experiment)
        if not experiment_metadata:
            click.echo(f"Error: Experiment {experiment} not found", err=True)
            sys.exit(1)
        experiment_id = experiment
        click.echo(f"Using existing experiment: {experiment_metadata.name} ({experiment_id})")
    else:
        # Create or reuse experiment
        exp_name = experiment_name or f"Benchmark {model}"
        exp_desc = experiment_description or f"Benchmarking {model} on {Path(dataset).name}"
        
        benchmark_core = BenchmarkCore(storage_manager)
        experiment_id = benchmark_core.create_experiment(exp_name, exp_desc)
        
        # Check if experiment already existed
        experiment_metadata = storage_manager.get_experiment_metadata(experiment_id)
        if experiment_metadata and len(experiment_metadata.runs) > 0:
            click.echo(f"Using existing experiment: {exp_name} ({experiment_id}) - {len(experiment_metadata.runs)} existing runs")
        else:
            click.echo(f"Created new experiment: {exp_name} ({experiment_id})")
    
    # Build model parameters
    model_params = {}
    if temperature is not None:
        model_params['temperature'] = temperature
    if max_tokens is not None:
        model_params['max_tokens'] = max_tokens
    if top_p is not None:
        model_params['top_p'] = top_p
    
    # Create run configuration
    run_config = RunConfig(
        model_id=model,
        system_prompt=system_prompt,
        model_params=model_params,
        max_concurrent=concurrent_limit,
        dataset_path=dataset
    )
    
    # Log benchmark start
    logger.info(
        "Starting benchmark run",
        model_id=model,
        dataset_path=dataset,
        aws_region=aws_region,
        max_concurrent=concurrent_limit,
        system_prompt=system_prompt[:100] + "..." if system_prompt and len(system_prompt) > 100 else system_prompt,
        model_params=model_params
    )
    
    click.echo(f"Starting benchmark run...")
    click.echo(f"Model: {model}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Region: {aws_region}")
    click.echo(f"Max concurrent: {concurrent_limit}")
    if system_prompt:
        click.echo(f"System prompt: {system_prompt[:100]}{'...' if len(system_prompt) > 100 else ''}")
    if model_params:
        click.echo(f"Model parameters: {model_params}")
    
    # Set up progress reporting with logging integration
    progress_logger = benchmark_logger.get_progress_logger()
    progress_callback = setup_progress_callback(verbose, progress_logger, log_progress)
    
    # Initialize benchmark core
    benchmark_core = BenchmarkCore(storage_manager, progress_callback)
    
    # Run the benchmark
    try:
        run_id = asyncio.run(benchmark_core.run_benchmark(
            experiment_id=experiment_id,
            run_config=run_config,
            dataset_path=dataset,
            resume_from_item=resume_from,
            run_name=run_name,
            region=aws_region
        ))
        
        logger.info("Benchmark completed successfully", run_id=run_id)
        
        click.echo(f"\nBenchmark completed successfully!")
        click.echo(f"Run ID: {run_id}")
        
        # Show run summary
        summary = benchmark_core.get_run_summary(run_id)
        
        logger.info(
            "Benchmark run summary",
            run_id=run_id,
            total_responses=summary['total_responses'],
            success_rate=summary['success_rate'],
            avg_latency_ms=summary['avg_latency_ms'],
            avg_rps=summary.get('avg_rps', 0),
            total_input_tokens=summary['total_input_tokens'],
            total_output_tokens=summary['total_output_tokens'],
            finish_reason_counts=summary.get('finish_reason_counts', {})
        )
        
        click.echo(f"\nRun Summary:")
        click.echo(f"  Total responses: {summary['total_responses']}")
        
        # Display finish_reason statistics as sub-items
        if 'finish_reason_counts' in summary and summary['finish_reason_counts']:
            finish_reasons = summary['finish_reason_counts']
            total = summary['total_responses']
            
            # Sort by count (descending) for better readability
            sorted_reasons = sorted(finish_reasons.items(), key=lambda x: x[1], reverse=True)
            
            for reason, count in sorted_reasons:
                percentage = (count / total) * 100
                click.echo(f"   └  {reason}: {count} ({percentage:.1f}%)")
        
        click.echo(f"  Success rate: {summary['success_rate']:.1%}")
        click.echo(f"  Average latency: {summary['avg_latency_ms']:.1f}ms")
        click.echo(f"  Average RPS: {summary.get('avg_rps', 0):.2f} requests/sec")
        click.echo(f"  Total input tokens: {summary['total_input_tokens']}")
        click.echo(f"  Total output tokens: {summary['total_output_tokens']}")
        
        click.echo(f"\nTo export results: bedrock-benchmark export-run {run_id}")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        click.echo("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Benchmark failed", error=str(e), exc_info=e)
        click.echo(f"Error running benchmark: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('name')
@click.option('--description', '-d', help='Experiment description')
@click.pass_context
def create_experiment(ctx, name: str, description: Optional[str]):
    """Create a new experiment."""
    storage_manager = ctx.obj['storage_manager']
    
    benchmark_core = BenchmarkCore(storage_manager)
    experiment_id = benchmark_core.create_experiment(name, description or "")
    
    click.echo(f"Created experiment: {name}")
    click.echo(f"Experiment ID: {experiment_id}")


@cli.command()
@click.pass_context
def list_experiments(ctx):
    """List all experiments."""
    storage_manager = ctx.obj['storage_manager']
    
    benchmark_core = BenchmarkCore(storage_manager)
    experiments = benchmark_core.list_experiments()
    
    if not experiments:
        click.echo("No experiments found.")
        return
    
    click.echo("Experiments:")
    for exp in experiments:
        click.echo(f"  {exp.id}: {exp.name}")
        if exp.description:
            click.echo(f"    Description: {exp.description}")
        click.echo(f"    Created: {exp.created_at}")
        click.echo(f"    Runs: {len(exp.runs)}")
        click.echo()


@cli.command()
@click.argument('experiment_id')
@click.pass_context
def list_runs(ctx, experiment_id: str):
    """List runs for an experiment."""
    storage_manager = ctx.obj['storage_manager']
    
    benchmark_core = BenchmarkCore(storage_manager)
    
    # Get experiment metadata
    experiment = benchmark_core.get_experiment_metadata(experiment_id)
    if not experiment:
        click.echo(f"Error: Experiment {experiment_id} not found", err=True)
        sys.exit(1)
    
    click.echo(f"Experiment: {experiment.name} ({experiment_id})")
    
    runs = benchmark_core.list_runs(experiment_id)
    if not runs:
        click.echo("No runs found for this experiment.")
        return
    
    click.echo(f"\nRuns ({len(runs)}):")
    for run_id in runs:
        summary = benchmark_core.get_run_summary(run_id)
        click.echo(f"  {run_id}:")
        click.echo(f"    Model: {summary['model_id']}")
        click.echo(f"    Responses: {summary['total_responses']}")
        click.echo(f"    Success rate: {summary['success_rate']:.1%}")
        click.echo(f"    Avg latency: {summary['avg_latency_ms']:.1f}ms")
        if summary['created_at']:
            click.echo(f"    Created: {summary['created_at']}")
        click.echo()


@cli.command()
@click.argument('run_id')
@click.option('--output', '-o', help='Output file path for CSV export')
@click.option('--dataset', '-d', help='Dataset file path to include prompts/expected responses')
@click.option('--format', type=click.Choice(['csv', 'json', 'parquet']), 
              default='csv', help='Output format')
@click.pass_context
def export_run(ctx, run_id: str, output: Optional[str], dataset: Optional[str], format: str):
    """Export run results to a file."""
    storage_manager = ctx.obj['storage_manager']
    
    benchmark_core = BenchmarkCore(storage_manager)
    
    try:
        # Export to DataFrame
        df = benchmark_core.export_run_to_dataframe(run_id, dataset)
        
        if df.empty:
            click.echo(f"No data found for run {run_id}")
            return
        
        # Determine output file
        if not output:
            output = f"run_{run_id}.{format}"
        
        # Export based on format
        if format == 'csv':
            df.to_csv(output, index=False)
        elif format == 'json':
            df.to_json(output, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output, index=False)
        
        click.echo(f"Exported {len(df)} records to {output}")
        click.echo(f"Columns: {', '.join(df.columns)}")
        
    except Exception as e:
        click.echo(f"Error exporting run: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('run_ids', nargs=-1, required=True)
@click.option('--output', '-o', help='Output file path for CSV export')
@click.option('--dataset', '-d', help='Dataset file path to include prompts/expected responses')
@click.option('--format', type=click.Choice(['csv', 'json', 'parquet']), 
              default='csv', help='Output format')
@click.pass_context
def compare_runs(ctx, run_ids: tuple, output: Optional[str], dataset: Optional[str], format: str):
    """Compare multiple runs in a single file."""
    storage_manager = ctx.obj['storage_manager']
    
    benchmark_core = BenchmarkCore(storage_manager)
    
    try:
        # Export to DataFrame
        df = benchmark_core.export_multiple_runs_to_dataframe(list(run_ids), dataset)
        
        if df.empty:
            click.echo(f"No data found for runs: {', '.join(run_ids)}")
            return
        
        # Determine output file
        if not output:
            output = f"comparison_{len(run_ids)}_runs.{format}"
        
        # Export based on format
        if format == 'csv':
            df.to_csv(output, index=False)
        elif format == 'json':
            df.to_json(output, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output, index=False)
        
        click.echo(f"Exported comparison of {len(run_ids)} runs to {output}")
        click.echo(f"Total records: {len(df)}")
        click.echo(f"Columns: {', '.join(df.columns)}")
        
        # Show summary by run
        run_summary = df.groupby('run_id').agg({
            'actual_response': 'count',
            'latency_ms': 'mean',
            'input_tokens': 'sum',
            'output_tokens': 'sum'
        }).round(2)
        
        click.echo(f"\nSummary by run:")
        click.echo(run_summary.to_string())
        
    except Exception as e:
        click.echo(f"Error comparing runs: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('run_id')
@click.pass_context
def show_run(ctx, run_id: str):
    """Show detailed information about a run."""
    storage_manager = ctx.obj['storage_manager']
    
    benchmark_core = BenchmarkCore(storage_manager)
    
    try:
        summary = benchmark_core.get_run_summary(run_id)
        
        click.echo(f"Run ID: {run_id}")
        click.echo(f"Model: {summary['model_id']}")
        if summary['system_prompt']:
            click.echo(f"System prompt: {summary['system_prompt']}")
        click.echo(f"Total responses: {summary['total_responses']}")
        click.echo(f"Success rate: {summary['success_rate']:.1%}")
        click.echo(f"Average latency: {summary['avg_latency_ms']:.1f}ms")
        click.echo(f"Total input tokens: {summary['total_input_tokens']}")
        click.echo(f"Total output tokens: {summary['total_output_tokens']}")
        if summary['created_at']:
            click.echo(f"Created: {summary['created_at']}")
        
        # Get run configuration
        run_config = storage_manager.get_run_config(run_id)
        if run_config and run_config.model_params:
            click.echo(f"Model parameters: {run_config.model_params}")
        
    except Exception as e:
        click.echo(f"Error showing run: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()