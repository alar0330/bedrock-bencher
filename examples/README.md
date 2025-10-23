# Sample JSONL Datasets

This directory contains sample JSONL datasets for testing the Bedrock Benchmark Toolkit. Each dataset demonstrates different prompt types and metadata structures.

## Dataset Descriptions

### simple_qa.jsonl
Basic question-and-answer pairs without metadata. Good for initial testing and simple benchmarks.
- **Size**: 5 items
- **Metadata**: None
- **Use case**: Basic functionality testing

### coding_tasks.jsonl
Programming-related prompts with rich metadata including difficulty, language, and category.
- **Size**: 4 items
- **Metadata**: category, difficulty, language, topic
- **Use case**: Technical knowledge evaluation

### creative_writing.jsonl
Creative writing prompts with metadata about genre, word count, and literary elements.
- **Size**: 3 items
- **Metadata**: genre, word_count, theme, perspective, literary_device, form
- **Use case**: Creative and artistic capability testing

### reasoning_tasks.jsonl
Logic puzzles and reasoning problems with metadata about cognitive aspects.
- **Size**: 3 items
- **Metadata**: type, difficulty, domain, cognitive_bias, strategy
- **Use case**: Logical reasoning and problem-solving evaluation

### multilingual.jsonl
Translation and cross-cultural tasks with language-specific metadata.
- **Size**: 3 items
- **Metadata**: task, source_language, target_language, cultural_significance
- **Use case**: Multilingual and cultural knowledge testing

### comprehensive_benchmark.jsonl
Mixed dataset with various prompt types and comprehensive metadata structures.
- **Size**: 5 items
- **Metadata**: Various rich metadata fields
- **Use case**: Full-featured benchmarking with diverse evaluation criteria

## JSONL Format

Each line in the JSONL files contains a JSON object with the following required fields:
- `prompt`: The input text for the LLM
- `expected_response`: The ground truth or expected output

Optional fields:
- `metadata`: Dictionary containing additional information about the prompt/response pair

## Usage Examples

```bash
# Run benchmark with simple Q&A dataset
bedrock-benchmark run --dataset examples/simple_qa.jsonl --model anthropic.claude-3-sonnet-20240229-v1:0

# Run with metadata-rich dataset
bedrock-benchmark run --dataset examples/comprehensive_benchmark.jsonl --model anthropic.claude-3-sonnet-20240229-v1:0

# Create experiment with multiple datasets
bedrock-benchmark create-experiment "Creative Writing Evaluation"
bedrock-benchmark run --experiment-id exp_123 --dataset examples/creative_writing.jsonl --model anthropic.claude-3-sonnet-20240229-v1:0
bedrock-benchmark run --experiment-id exp_123 --dataset examples/creative_writing.jsonl --model anthropic.claude-3-haiku-20240307-v1:0
```