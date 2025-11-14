# Examples Directory

This directory contains sample JSONL datasets and example scripts for the Bedrock Benchmark Toolkit.

## Contents

- **JSONL Datasets**: Sample datasets for text generation benchmarking
- **Embeddings Examples**: Python scripts demonstrating multi-modal embeddings (see [EMBEDDINGS_README.md](EMBEDDINGS_README.md))

## Sample JSONL Datasets

Sample JSONL datasets for testing text generation with the Bedrock Benchmark Toolkit. Each dataset demonstrates different prompt types and metadata structures.

## Dataset Descriptions

### Text Generation Datasets

#### text_qa_task_100.jsonl
Basic question-and-answer pairs for benchmarking. Good for quick testing and performance evaluation.
- **Size**: 100 items
- **Format**: Simple Q&A pairs with id, prompt, and expected_response
- **Use case**: Quick benchmarking, performance testing, cost estimation

#### text_qa_task_500.jsonl
Extended question-and-answer dataset for comprehensive benchmarking.
- **Size**: 500 items
- **Format**: Simple Q&A pairs with id, prompt, and expected_response
- **Use case**: Comprehensive benchmarking, throughput testing, large-scale evaluation

#### coding_tasks.jsonl
Programming-related prompts with rich metadata including difficulty, language, and category.
- **Size**: 4 items
- **Metadata**: category, difficulty, language, topic
- **Use case**: Technical knowledge evaluation

#### creative_writing.jsonl
Creative writing prompts with metadata about genre, word count, and literary elements.
- **Size**: 3 items
- **Metadata**: genre, word_count, theme, perspective, literary_device, form
- **Use case**: Creative and artistic capability testing

#### multilingual.jsonl
Translation and cross-cultural tasks with language-specific metadata.
- **Size**: 3 items
- **Metadata**: task, source_language, target_language, cultural_significance
- **Use case**: Multilingual and cultural knowledge testing

## JSONL Format

Each line in the JSONL files contains a JSON object with the following required fields:
- `prompt`: The input text for the LLM
- `expected_response`: The ground truth or expected output

Optional fields:
- `metadata`: Dictionary containing additional information about the prompt/response pair

## Usage Examples

**Quick Examples:**

```bash
# Quick benchmark with 100-item dataset
bedrock-bencher run-benchmark --dataset examples/text_qa_task_100.jsonl --model anthropic.claude-3-sonnet-20240229-v1:0

# Comprehensive benchmark with 500-item dataset
bedrock-bencher run-benchmark --dataset examples/text_qa_task_500.jsonl --model anthropic.claude-3-sonnet-20240229-v1:0

# Run with metadata-rich dataset
bedrock-bencher run-benchmark --dataset examples/coding_tasks.jsonl --model anthropic.claude-3-sonnet-20240229-v1:0

# Create experiment with multiple datasets
bedrock-bencher create-experiment "Creative Writing Evaluation"
bedrock-bencher run-benchmark --experiment-id exp_123 --dataset examples/creative_writing.jsonl --model anthropic.claude-3-sonnet-20240229-v1:0
bedrock-bencher run-benchmark --experiment-id exp_123 --dataset examples/creative_writing.jsonl --model anthropic.claude-3-haiku-20240307-v1:0
```

**Run All Text Generation Examples:**
```bash
# Execute the example script (requires AWS credentials and model access)
./examples/text_examples.sh
```

## Embeddings Datasets

### embeddings_sample_texts.jsonl
Text samples across different categories for embedding generation.
- **Size**: 10 items
- **Categories**: technology, science, business, health, education
- **Format**: `{"id": "...", "text": "...", "category": "..."}`
- **Use case**: Text embeddings, semantic search, similarity testing

### embeddings_sample_images.jsonl
Product images for visual embeddings and similarity search.
- **Size**: 10 items
- **Categories**: apparel, accessories, footwear, kitchen, watches, fitness
- **Format**: `{"id": "...", "image_path": "...", "category": "...", "description": "..."}`
- **Use case**: Image embeddings, visual search, product similarity

## Embeddings Examples

Generate embeddings using the `bedrock-bencher embed` command with the sample datasets.

**Quick Examples:**

```bash
# Text embeddings with Titan Multimodal (256 dimensions)
bedrock-bencher embed \
  --dataset examples/embeddings_sample_texts.jsonl \
  --model amazon.titan-embed-image-v1 \
  --output-dimension 256

# Image embeddings with Titan Multimodal
bedrock-bencher embed \
  --dataset examples/embeddings_sample_images.jsonl \
  --model amazon.titan-embed-image-v1 \
  --output-dimension 1024

# Image embeddings with Nova
bedrock-bencher embed \
  --dataset examples/embeddings_sample_images.jsonl \
  --model amazon.nova-2-multimodal-embeddings-v1:0 \
  --embedding-purpose GENERIC_INDEX

# Text embeddings with Cohere v4
bedrock-bencher embed \
  --dataset examples/embeddings_sample_texts.jsonl \
  --model us.cohere.embed-v4:0 \
  --input-type search_document
```

**Run All Examples:**
```bash
# Execute the example script (requires AWS credentials and model access)
./examples/embeddings_examples.sh
```

**Supported Models:**
- Amazon Titan Multimodal Embeddings (`amazon.titan-embed-image-v1`)
- Amazon Nova Multimodal Embeddings (`amazon.nova-2-multimodal-embeddings-v1:0`)
- Cohere Embeddings v4 (`us.cohere.embed-v4:0`)

**See Also:**
- [Embeddings CLI Guide](../docs/CLI_EMBEDDINGS.md) - Complete CLI documentation
- [CLI Quick Reference](../docs/EMBED_COMMAND_REFERENCE.md) - Quick reference card