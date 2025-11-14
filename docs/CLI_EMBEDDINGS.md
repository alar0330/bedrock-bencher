# Embeddings CLI Guide

This guide covers the `embed` command for generating embeddings with Amazon Bedrock embedding models.

## Overview

The `embed` command allows you to generate embeddings for text and images using Titan, Nova, and Cohere embedding models. It follows the same experiment and run structure as the text generation benchmark commands.

## Basic Usage

```bash
bedrock-bencher embed --dataset <dataset.jsonl> --model <model-id> [OPTIONS]
```

## Quick Examples

### Text Embeddings with Titan Multimodal

```bash
bedrock-bencher embed \
  --dataset examples/embeddings_sample_texts.jsonl \
  --model amazon.titan-embed-image-v1 \
  --output-dimension 256 \
  --experiment-name "Text Embeddings Test"
```

### Image Embeddings with Nova

```bash
bedrock-bencher embed \
  --dataset images.jsonl \
  --model amazon.nova-2-multimodal-embeddings-v1:0 \
  --embedding-purpose GENERIC_INDEX \
  --experiment-name "Image Search Index"
```

### Multi-Modal with Cohere v4

```bash
bedrock-bencher embed \
  --dataset multimodal.jsonl \
  --model us.cohere.embed-v4:0 \
  --input-type search_document \
  --experiment-name "Multi-Modal Search"
```

## Dataset Format

The dataset must be a JSONL file where each line is a JSON object with an `id` field and either `text`, `image_path`, or both.

### Text-Only Dataset

```jsonl
{"id": "text_001", "text": "Machine learning is transforming technology"}
{"id": "text_002", "text": "Natural language processing enables AI communication"}
{"id": "text_003", "text": "Deep learning uses neural networks"}
```

### Image-Only Dataset

```jsonl
{"id": "img_001", "image_path": "images/photo1.jpg"}
{"id": "img_002", "image_path": "images/photo2.png"}
{"id": "img_003", "image_path": "images/photo3.jpg"}
```

### Multi-Modal Dataset (Cohere v4 only)

```jsonl
{"id": "mm_001", "text": "A sunset over the ocean", "image_path": "images/sunset.jpg"}
{"id": "mm_002", "text": "Mountain landscape", "image_path": "images/mountain.png"}
```

## Command Options

### Required Options

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Bedrock embedding model ID (required) |

### Dataset Options

| Option | Description |
|--------|-------------|
| `--dataset`, `-d` | Path to JSONL dataset file with text/image inputs |

### Experiment Options

| Option | Description |
|--------|-------------|
| `--experiment`, `-e` | Existing experiment ID to add run to |
| `--experiment-name` | Name for new experiment (if not using existing) |
| `--experiment-description` | Description for new experiment |
| `--run-name` | Human-readable name for this run |

### AWS Configuration

| Option | Description |
|--------|-------------|
| `--region` | AWS region (overrides config file) |
| `--max-concurrent` | Maximum concurrent requests (default: 10) |

### Model-Specific Parameters

#### Titan Models

| Option | Description | Values |
|--------|-------------|--------|
| `--output-dimension` | Embedding vector size | 256, 384, or 1024 |

**Example:**
```bash
bedrock-bencher embed \
  -d texts.jsonl \
  -m amazon.titan-embed-image-v1 \
  --output-dimension 256
```

#### Nova Models

| Option | Description | Values |
|--------|-------------|--------|
| `--output-dimension` | Embedding vector size | 1024 |
| `--embedding-purpose` | Purpose of embeddings | GENERIC_INDEX, SEARCH_QUERY, CLASSIFICATION |
| `--truncate` | Text truncation mode | END, START, NONE |

**Example:**
```bash
bedrock-bencher embed \
  -d texts.jsonl \
  -m amazon.nova-2-multimodal-embeddings-v1:0 \
  --embedding-purpose GENERIC_INDEX \
  --truncate END
```

#### Cohere Models

| Option | Description | Values |
|--------|-------------|--------|
| `--input-type` | Type of input | search_document, search_query, classification |
| `--truncate` | Text truncation mode | END, START, NONE |

**Example:**
```bash
bedrock-bencher embed \
  -d texts.jsonl \
  -m us.cohere.embed-v4:0 \
  --input-type search_document \
  --truncate END
```

### Progress and Logging

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Show detailed progress for each item |
| `--log-progress` | Enable structured JSON progress logs |

## Supported Models

### Titan Models

- `amazon.titan-embed-image-v1` - Multimodal embeddings (text and/or image)

**Dimensions:** 256, 384, or 1024 (configurable)

### Nova Models

- `amazon.nova-2-multimodal-embeddings-v1:0` - Text and image embeddings

**Dimensions:** 1024 (fixed)

### Cohere Models

- `us.cohere.embed-v4:0` - Multimodal embeddings (text, image, or interleaved text+image)

**Dimensions:** 256, 512, 1024, or 1536 (configurable, default 1536)

## Complete Workflow Example

### 1. Create Dataset

```bash
cat > my_texts.jsonl << EOF
{"id": "doc_001", "text": "Artificial intelligence is transforming industries"}
{"id": "doc_002", "text": "Machine learning enables predictive analytics"}
{"id": "doc_003", "text": "Natural language processing powers chatbots"}
EOF
```

### 2. Generate Embeddings

```bash
bedrock-bencher embed \
  --dataset my_texts.jsonl \
  --model amazon.titan-embed-image-v1 \
  --output-dimension 256 \
  --experiment-name "Document Embeddings" \
  --run-name "baseline" \
  --verbose
```

**Output:**
```
Created experiment: Document Embeddings (document-embeddings)

Starting embedding generation...
Model: amazon.titan-embed-image-v1
Dataset: my_texts.jsonl
Region: us-east-1
Max concurrent: 10
Model parameters: {'embeddingConfig': {'outputEmbeddingLength': 256}}
Run ID: baseline_20251107-120000_a1b2

Loaded 3 items from dataset

[1/3] Processing: doc_001
  ✓ Embedding dim: 256, Latency: 245ms
[2/3] Processing: doc_002
  ✓ Embedding dim: 256, Latency: 238ms
[3/3] Processing: doc_003
  ✓ Embedding dim: 256, Latency: 251ms

============================================================
Embedding generation completed!
============================================================
Run ID: baseline_20251107-120000_a1b2
Total items: 3
Successful: 3
Failed: 0
Success rate: 100.0%
Average latency: 244.7ms

To export results: bedrock-bencher export-run baseline_20251107-120000_a1b2
```

### 3. Export Results

```bash
bedrock-bencher export-run baseline_20251107-120000_a1b2 --format csv
```

### 4. Compare Multiple Runs

```bash
# Generate embeddings with different dimensions
bedrock-bencher embed -d my_texts.jsonl -m amazon.titan-embed-image-v1 \
  --output-dimension 256 --run-name "dim-256"

bedrock-bencher embed -d my_texts.jsonl -m amazon.titan-embed-image-v1 \
  --output-dimension 1024 --run-name "dim-1024"

# Compare the runs
bedrock-bencher compare-runs dim-256_* dim-1024_*
```

## Working with Experiments

### List All Experiments

```bash
bedrock-bencher list-experiments
```

### List Runs in an Experiment

```bash
bedrock-bencher list-runs document-embeddings
```

### Show Run Details

```bash
bedrock-bencher show-run baseline_20251107-120000_a1b2
```

## Advanced Usage

### Batch Processing with Multiple Models

```bash
#!/bin/bash
# Compare embeddings across different models

DATASET="my_texts.jsonl"
EXPERIMENT="model-comparison"

# Titan Multimodal
bedrock-bencher embed -d $DATASET \
  -m amazon.titan-embed-image-v1 \
  --output-dimension 256 \
  --experiment-name $EXPERIMENT \
  --run-name "titan-256"

# Nova
bedrock-bencher embed -d $DATASET \
  -m amazon.nova-2-multimodal-embeddings-v1:0 \
  --embedding-purpose GENERIC_INDEX \
  --experiment-name $EXPERIMENT \
  --run-name "nova-1024"

# Cohere v4
bedrock-bencher embed -d $DATASET \
  -m us.cohere.embed-v4:0 \
  --input-type search_document \
  --experiment-name $EXPERIMENT \
  --run-name "cohere-v4"
```

### High-Throughput Processing

```bash
# Process large dataset with high concurrency
bedrock-bencher embed \
  --dataset large_dataset.jsonl \
  --model amazon.titan-embed-image-v1 \
  --max-concurrent 50 \
  --output-dimension 256 \
  --experiment-name "Large Scale Indexing"
```

### Using Configuration File

Create a config file `embeddings-config.yaml`:

```yaml
aws_region: us-east-1
max_concurrent: 20
storage_path: ./my_embeddings
log_level: INFO
log_file: embeddings.log
```

Run with config:

```bash
bedrock-bencher --config embeddings-config.yaml embed \
  --dataset texts.jsonl \
  --model amazon.titan-embed-image-v1
```

## Error Handling

### Common Errors

**Dataset not found:**
```
Error: Dataset file not found: texts.jsonl
```
→ Check the file path is correct

**No valid items in dataset:**
```
Error: No valid items found in dataset
```
→ Ensure JSONL format is correct and items have 'id' field

**Model access denied:**
```
Error generating embeddings: AccessDeniedException
```
→ Request model access in AWS Bedrock console

**Image file not found:**
```
Warning: Line 5 - Image file not found: images/missing.jpg
```
→ Check image paths in dataset are correct

### Verbose Mode for Debugging

Use `--verbose` to see detailed progress:

```bash
bedrock-bencher embed \
  --dataset texts.jsonl \
  --model amazon.titan-embed-image-v1 \
  --verbose
```

## Integration with Analysis Tools

### Export to CSV for Analysis

```bash
bedrock-bencher export-run <run-id> --format csv --output embeddings.csv
```

### Load in Python

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
df = pd.read_csv('embeddings.csv')

# Extract embedding vectors
embeddings = np.array([eval(e) for e in df['embedding']])

# Calculate similarity
similarity_matrix = cosine_similarity(embeddings)
print(similarity_matrix)
```

### Export to Parquet for Big Data

```bash
bedrock-bencher export-run <run-id> --format parquet --output embeddings.parquet
```

## Performance Tips

1. **Concurrency**: Adjust `--max-concurrent` based on your throughput needs
   - Default: 10
   - High throughput: 50-100
   - Rate limit sensitive: 5-10

2. **Batch Size**: Process large datasets in chunks if needed

3. **Region Selection**: Use `--region` to select the closest AWS region

4. **Dimension Selection**: Smaller dimensions (256) are faster than larger (1024)

## See Also

- [Embeddings CLI Examples](../examples/embeddings_examples.sh) - Shell script with CLI examples
- [CLI Quick Reference](EMBED_COMMAND_REFERENCE.md) - Quick reference card
- [Main README](../README.md) - Project overview

## Support

For issues or questions:
1. Run the example script: `./examples/embeddings_examples.sh`
2. Review AWS Bedrock documentation for model-specific details
3. Use `--verbose` flag for detailed debugging output
4. Check the [CLI Quick Reference](EMBED_COMMAND_REFERENCE.md) for common patterns
