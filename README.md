# Bedrock Benchmark Toolkit

A minimalistic and lightweight toolkit for benchmarking different Large Language Models (LLMs) on Amazon Bedrock. This tool enables concurrent evaluation of LLM performance using JSONL datasets containing prompts and ground truth responses, with built-in error handling, rate limiting, and hierarchical experiment organization.

## Features

- **JSONL Dataset Support**: Load datasets with prompts and ground truth responses
- **Concurrent Processing**: Evaluate multiple prompts simultaneously for faster benchmarking
- **Amazon Bedrock Integration**: Native support for all Bedrock models via Converse API
- **Robust Error Handling**: Exponential backoff retry logic for API throttling and errors
- **Human-Readable Organization**: Experiments and runs use descriptive folder names with timestamps
- **Custom Run Names**: Name your benchmark runs for easy identification and comparison
- **Data Export**: Export results to pandas DataFrames, CSV, JSON, or Parquet formats
- **Progress Tracking**: Real-time progress reporting with detailed logging
- **Resumable Runs**: Resume interrupted benchmarks from any point
- **Analysis Tools**: Built-in Jupyter notebook for data exploration and visualization
- **Flexible Configuration**: Support for configuration files and environment variables

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/alar0330/bedrock-bencher.git
cd bedrock-bencher

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Requirements

- **Python**: 3.9 or higher
- **AWS Credentials**: Configured via AWS CLI, environment variables, or IAM roles
- **Amazon Bedrock Access**: Appropriate permissions to invoke Bedrock models

### AWS Setup

Ensure your AWS credentials are configured and you have access to Amazon Bedrock:

```bash
# Configure AWS credentials (if not already done)
aws configure

# Verify Bedrock access
aws bedrock list-foundation-models --region us-east-1
```

## Quick Start

### 1. Create Sample Dataset

Create a JSONL file with your prompts and expected responses:

```bash
# Use one of the provided examples
cp examples/simple_qa.jsonl my_dataset.jsonl

# Or create your own
echo '{"prompt": "What is the capital of France?", "expected_response": "Paris"}' > my_dataset.jsonl
echo '{"prompt": "What is 2 + 2?", "expected_response": "4"}' >> my_dataset.jsonl
```

### 2. Run Your First Benchmark

```bash
# Run a simple benchmark with custom names
bedrock-bencher run-benchmark \
    --dataset my_dataset.jsonl \
    --model anthropic.claude-3-sonnet-20240229-v1:0 \
    --experiment-name "My First Benchmark" \
    --run-name "Baseline Test"

# Creates organized folders:
# experiments/my-first-benchmark/
#   └── runs/baseline-test_20241024-143055_x9y8/
```

### 3. Export Results

```bash
# Export results to CSV (use the run folder name as ID)
bedrock-bencher export-run baseline-test_20241024-143055_x9y8 --output results.csv

# View the results
head results.csv
```

### 4. Analyze Results

```bash
# Open the analysis notebook
jupyter notebook benchmark_analysis.ipynb

# Or use the CLI for quick summaries
bedrock-bencher show-run baseline-test_20241024-143055_x9y8
```

## Usage Examples

### Basic Benchmarking Workflows

#### Single Model Evaluation

```bash
# Run benchmark with specific model parameters and custom run name
bedrock-bencher run-benchmark \
    --dataset examples/coding_tasks.jsonl \
    --model anthropic.claude-3-sonnet-20240229-v1:0 \
    --temperature 0.7 \
    --max-tokens 1000 \
    --system-prompt "You are a helpful coding assistant." \
    --experiment-name "Coding Tasks Evaluation" \
    --run-name "High Temperature"

# Creates: experiments/coding-tasks-evaluation/
#          └── runs/high-temperature_20241024-143055_x9y8/
```

#### Multi-Model Comparison

```bash
# Create an experiment for comparison (or reuse existing one)
bedrock-bencher create-experiment "Model Comparison" \
    --description "Comparing Claude models on reasoning tasks"

# Run multiple models on the same dataset with descriptive names
# These will be added to the same "model-comparison" experiment folder
bedrock-bencher run-benchmark \
    --dataset examples/reasoning_tasks.jsonl \
    --model anthropic.claude-3-sonnet-20240229-v1:0 \
    --experiment "Model Comparison" \
    --run-name "Claude Sonnet"

bedrock-bencher run-benchmark \
    --dataset examples/reasoning_tasks.jsonl \
    --model anthropic.claude-3-haiku-20240307-v1:0 \
    --experiment "Model Comparison" \
    --run-name "Claude Haiku"

# Compare results using the readable run names
bedrock-bencher compare-runs claude-sonnet_20241024-143055_x9y8 claude-haiku_20241024-143155_m5n4 --output comparison.csv
```

#### Advanced Configuration

```bash
# Use configuration file for complex setups
cat > config.yaml << EOF
aws_region: us-west-2
max_concurrent: 5
storage_path: ./my_experiments
log_level: DEBUG
log_file: benchmark.log
max_retries: 3
base_delay: 2.0
EOF

# Run with configuration
bedrock-bencher --config config.yaml run-benchmark \
    --dataset examples/comprehensive_benchmark.jsonl \
    --model anthropic.claude-3-sonnet-20240229-v1:0
```

### Experiment Management

```bash
# List all experiments (shows human-readable folder names)
bedrock-bencher list-experiments

# List runs in an experiment (use experiment folder name)
bedrock-bencher list-runs qa-baseline

# Show detailed run information (use run folder name)
bedrock-bencher show-run claude-sonnet_20241024-143055_x9y8

# Resume an interrupted run
bedrock-bencher run-benchmark \
    --dataset examples/large_dataset.jsonl \
    --model anthropic.claude-3-sonnet-20240229-v1:0 \
    --run-name "Large Dataset Test" \
    --resume-from "item_123"
```

### Data Export and Analysis

```bash
# Export single run to different formats (use readable run names)
bedrock-bencher export-run claude-sonnet_20241024-143055_x9y8 --format csv --output results.csv
bedrock-bencher export-run claude-sonnet_20241024-143055_x9y8 --format json --output results.json
bedrock-bencher export-run claude-sonnet_20241024-143055_x9y8 --format parquet --output results.parquet

# Export with original dataset for complete analysis
bedrock-bencher export-run claude-sonnet_20241024-143055_x9y8 \
    --dataset examples/coding_tasks.jsonl \
    --output complete_results.csv

# Compare multiple runs using readable names
bedrock-bencher compare-runs \
    claude-sonnet_20241024-143055_x9y8 \
    claude-haiku_20241024-144312_m5n4 \
    gpt4_20241024-145000_p3q2 \
    --output multi_model_comparison.csv
```

## Folder Structure and Organization

The toolkit uses a hierarchical folder structure with human-readable experiment names and timestamped runs for easy navigation and organization:

### Folder Naming Convention

```
experiments/
├── qa-baseline/                                    # experiment-name (path-safe, reusable)
│   ├── metadata.json                              # Experiment details
│   └── runs/
│       ├── claude-sonnet_20241024-143055_x9y8/    # run-name_timestamp_uuid
│       │   ├── config.json                        # Run configuration
│       │   └── responses.jsonl                    # Benchmark results
│       └── claude-haiku_20241024-144312_m5n4/
└── creative-writing/
    └── runs/
        └── baseline-run_20241024-150005_p3q2/
```

### Benefits

- **Experiment Reuse**: Experiments use simple names and can be reused for multiple runs
- **Chronological Run Sorting**: Run folders automatically sort by creation time
- **Human Readable**: Easy to identify experiments and runs at a glance  
- **Unique Run IDs**: Timestamp + UUID prevents run naming conflicts
- **File Explorer Friendly**: Navigate experiments without opening the tool

### Using Folder Names as IDs

The folder names serve as the experiment and run IDs throughout the toolkit:

```bash
# List experiments (shows folder names)
bedrock-bencher list-experiments

# Export using run folder name as ID
bedrock-bencher export-run claude-sonnet_20241024-143055_x9y8 --output results.csv

# Show run details using run folder name
bedrock-bencher show-run claude-sonnet_20241024-143055_x9y8

# List runs in an experiment using experiment folder name
bedrock-bencher list-runs model-comparison
```

### Experiment Reuse

When you specify an experiment name that already exists, the toolkit will reuse the existing experiment folder and add new runs to it. This allows you to group related benchmark runs under the same experiment without creating duplicate experiment folders.

## Analysis and Visualization

### Built-in Analysis Notebook

The toolkit includes a Jupyter notebook for interactive data analysis:

```bash
# Start Jupyter and open the analysis notebook
jupyter notebook benchmark_analysis.ipynb
```

The notebook provides:
- **Experiment and run listing** with human-readable names
- **Single run analysis** with latency and token usage statistics
- **Multi-run comparison** with side-by-side performance metrics
- **Visualizations** including histograms, scatter plots, and box plots
- **Data export** to CSV for further analysis

### Quick CLI Analysis

```bash
# Show run summary
bedrock-bencher show-run claude-sonnet_20241024-143055_x9y8

# List all runs in an experiment  
bedrock-bencher list-runs qa-baseline

# Export and compare multiple runs
bedrock-bencher compare-runs run1_timestamp_uuid run2_timestamp_uuid --output comparison.csv
```

## DataFrame Export Format

When you export run results, you get a pandas DataFrame with the following structure:

### Standard Columns

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | string | Unique identifier for the benchmark run |
| `item_id` | string | Unique identifier for each prompt/response pair |
| `model_id` | string | Bedrock model identifier used |
| `prompt` | string | Original input prompt (if dataset provided) |
| `expected_response` | string | Ground truth response (if dataset provided) |
| `actual_response` | string | LLM-generated response |
| `timestamp` | datetime | When the response was generated |
| `latency_ms` | float | Response latency in milliseconds |
| `input_tokens` | int | Number of input tokens |
| `output_tokens` | int | Number of output tokens |
| `finish_reason` | string | Reason the generation stopped |

### Metadata Columns

If your JSONL dataset includes metadata, additional columns will be created:

```python
# Example: Loading results for analysis
import pandas as pd

# Load exported results
df = pd.read_csv('results.csv')

# Basic analysis
print(f"Average latency: {df['latency_ms'].mean():.2f}ms")
print(f"Total tokens used: {df['input_tokens'].sum() + df['output_tokens'].sum()}")
print(f"Success rate: {(df['finish_reason'] == 'stop').mean():.2%}")

# Group by metadata (if available)
if 'category' in df.columns:
    category_stats = df.groupby('category').agg({
        'latency_ms': 'mean',
        'output_tokens': 'mean',
        'actual_response': 'count'
    })
    print(category_stats)
```

### Multi-Run Comparison

When comparing multiple runs, you get additional columns:

| Column | Type | Description |
|--------|------|-------------|
| `run_name` | string | Human-readable run identifier |
| `experiment_id` | string | Parent experiment identifier |

```python
# Example: Multi-run analysis
comparison_df = pd.read_csv('comparison.csv')

# Compare models by average latency
model_performance = comparison_df.groupby(['run_id', 'model_id']).agg({
    'latency_ms': 'mean',
    'output_tokens': 'mean',
    'input_tokens': 'mean'
}).reset_index()

print(model_performance)

# Statistical comparison
from scipy import stats
model_a_latency = comparison_df[comparison_df['model_id'] == 'model_a']['latency_ms']
model_b_latency = comparison_df[comparison_df['model_id'] == 'model_b']['latency_ms']
t_stat, p_value = stats.ttest_ind(model_a_latency, model_b_latency)
print(f"T-test p-value: {p_value:.4f}")
```

## Configuration

### Configuration File

Create a `config.yaml` file for persistent settings:

```yaml
# AWS Configuration
aws_region: us-east-1
aws_profile: default  # Optional: specific AWS profile

# Concurrency Settings
max_concurrent: 10

# Retry Configuration
max_retries: 5
base_delay: 1.0
max_delay: 60.0

# Storage
storage_path: ./experiments

# Logging
log_level: INFO
log_file: benchmark.log
```

### Environment Variables

Override settings with environment variables:

```bash
export AWS_DEFAULT_REGION=us-west-2
export BEDROCK_BENCHMARK_STORAGE_PATH=/path/to/experiments
export BEDROCK_BENCHMARK_LOG_LEVEL=DEBUG
```

### Command Line Options

All settings can be overridden via command line:

```bash
bedrock-bencher --storage-path ./custom_experiments \
                  --log-level DEBUG \
                  run-benchmark \
                  --region us-west-2 \
                  --max-concurrent 5 \
                  --dataset examples/simple_qa.jsonl \
                  --model anthropic.claude-3-sonnet-20240229-v1:0 \
                  --experiment-name "Custom Test" \
                  --run-name "Debug Run"
```

### Key CLI Options

- `--experiment-name`: Human-readable name for the experiment (creates reusable folder with path-safe name)
- `--run-name`: Human-readable name for the run (creates folder: `name_timestamp_uuid`)  
- `--resume-from`: Resume interrupted runs from a specific item ID
- `--system-prompt`: Custom system prompt for the model
- `--temperature`, `--max-tokens`, `--top-p`: Model parameters

## Sample Datasets

The `examples/` directory contains sample JSONL datasets for testing:

- **`simple_qa.jsonl`**: Basic Q&A pairs without metadata
- **`coding_tasks.jsonl`**: Programming questions with difficulty and language metadata
- **`creative_writing.jsonl`**: Creative prompts with genre and style metadata
- **`reasoning_tasks.jsonl`**: Logic puzzles with cognitive metadata
- **`multilingual.jsonl`**: Translation tasks with language metadata
- **`comprehensive_benchmark.jsonl`**: Mixed dataset with rich metadata

See `examples/README.md` for detailed descriptions.

## Supported Bedrock Models

The toolkit supports all Amazon Bedrock models via the Converse API:

### Anthropic Claude Models
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-opus-20240229-v1:0`
- `anthropic.claude-instant-v1`
- `anthropic.claude-v2:1`

### Amazon Titan Models
- `amazon.titan-text-express-v1`
- `amazon.titan-text-lite-v1`

### AI21 Labs Models
- `ai21.j2-ultra-v1`
- `ai21.j2-mid-v1`

### Cohere Models
- `cohere.command-text-v14`
- `cohere.command-light-text-v14`

### Meta Llama Models
- `meta.llama2-13b-chat-v1`
- `meta.llama2-70b-chat-v1`

## Troubleshooting

### Common Issues

#### AWS Credentials Not Found
```bash
# Error: Unable to locate credentials
aws configure
# or
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

#### Bedrock Access Denied
```bash
# Error: User is not authorized to perform: bedrock:InvokeModel
# Ensure your AWS user/role has the necessary Bedrock permissions
```

#### Rate Limiting
```bash
# Error: Too many requests
# The tool automatically handles rate limiting with exponential backoff
# You can adjust retry settings in configuration:
bedrock-bencher run-benchmark \
    --max-concurrent 2 \  # Reduce concurrency
    --dataset your_dataset.jsonl \
    --model your_model
```

#### Large Dataset Memory Issues
```bash
# For very large datasets, the tool streams data to minimize memory usage
# If you still encounter issues, process in smaller batches:
split -l 1000 large_dataset.jsonl batch_
# Then process each batch separately
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
bedrock-bencher --log-level DEBUG \
                  --log-file debug.log \
                  run-benchmark \
                  --verbose \
                  --dataset examples/simple_qa.jsonl \
                  --model anthropic.claude-3-sonnet-20240229-v1:0
```

## Development

### Setup Development Environment

```bash
# Clone and install in development mode
git clone https://github.com/alar0330/bedrock-bencher.git
cd bedrock-bencher
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bedrock_benchmark

# Run specific test file
pytest tests/test_storage.py

# Run async tests
pytest -v tests/test_client.py
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
bedrock-bencher/
├── src/bedrock_benchmark/          # Main package
│   ├── __init__.py
│   ├── cli.py                      # Command-line interface
│   ├── core.py                     # Benchmark orchestration
│   ├── client.py                   # Bedrock API client
│   ├── dataset.py                  # JSONL dataset handling
│   ├── storage.py                  # Experiment/run storage
│   ├── backoff.py                  # Retry logic
│   ├── config.py                   # Configuration management
│   ├── models.py                   # Data models
│   └── logging.py                  # Structured logging
├── examples/                       # Sample datasets
├── tests/                          # Test suite
├── pyproject.toml                  # Project configuration
└── README.md                       # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Format code (`black src/ tests/`)
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/alar0330/bedrock-bencher/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alar0330/bedrock-bencher/discussions)
- **Documentation**: [Wiki](https://github.com/alar0330/bedrock-bencher/wiki)