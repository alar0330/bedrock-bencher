# `embed` Command Quick Reference

## Syntax

```bash
bedrock-bencher embed --dataset <file.jsonl> --model <model-id> [OPTIONS]
```

## Common Use Cases

### Text Embeddings (Titan Multimodal)
```bash
bedrock-bencher embed \
  -d texts.jsonl \
  -m amazon.titan-embed-image-v1 \
  --output-dimension 256
```

### Image Embeddings (Nova)
```bash
bedrock-bencher embed \
  -d images.jsonl \
  -m amazon.nova-2-multimodal-embeddings-v1:0 \
  --embedding-purpose GENERIC_INDEX
```

### Multi-Modal (Cohere v4)
```bash
bedrock-bencher embed \
  -d multimodal.jsonl \
  -m us.cohere.embed-v4:0 \
  --input-type search_document
```

**Note:** Cohere v4 supports text-only, image-only, or interleaved text+image inputs.

## Dataset Formats

| Type | Format |
|------|--------|
| Text | `{"id": "001", "text": "..."}` |
| Image | `{"id": "001", "image_path": "path/to/img.jpg"}` |
| Multi-modal | `{"id": "001", "text": "...", "image_path": "..."}` |

## Model Parameters

### Titan Models
| Parameter | Values | Description |
|-----------|--------|-------------|
| `--output-dimension` | 256, 384, 1024 | Embedding size |

### Nova Models
| Parameter | Values | Description |
|-----------|--------|-------------|
| `--embedding-purpose` | GENERIC_INDEX, SEARCH_QUERY, CLASSIFICATION | Use case |
| `--output-dimension` | 1024 | Embedding size |
| `--truncate` | END, START, NONE | Text truncation |

### Cohere Models
| Parameter | Values | Description |
|-----------|--------|-------------|
| `--input-type` | search_document, search_query, classification | Use case |
| `--truncate` | END, START, NONE | Text truncation |

## Common Options

| Option | Short | Description |
|--------|-------|-------------|
| `--dataset` | `-d` | JSONL dataset file |
| `--model` | `-m` | Model ID (required) |
| `--experiment` | `-e` | Existing experiment ID |
| `--experiment-name` | | New experiment name |
| `--run-name` | | Run identifier |
| `--region` | | AWS region |
| `--max-concurrent` | | Concurrency limit |
| `--verbose` | `-v` | Detailed progress |

## Workflow

```bash
# 1. Create dataset
cat > data.jsonl << EOF
{"id": "001", "text": "Sample text"}
EOF

# 2. Generate embeddings
bedrock-bencher embed -d data.jsonl -m amazon.titan-embed-image-v1

# 3. Export results
bedrock-bencher export-run <run-id> --format csv

# 4. Analyze
python -c "import pandas as pd; df = pd.read_csv('run_<id>.csv'); print(df)"
```

## Model IDs

| Model | ID |
|-------|-----|
| Titan Multimodal | `amazon.titan-embed-image-v1` |
| Nova Multimodal | `amazon.nova-2-multimodal-embeddings-v1:0` |
| Cohere v4 | `us.cohere.embed-v4:0` |

## Tips

- Use `--verbose` for debugging
- Adjust `--max-concurrent` for throughput
- Smaller dimensions (256) are faster
- Use `--experiment` to group related runs
- Export to CSV for analysis with pandas

## See Also

- Full guide: [CLI_EMBEDDINGS.md](CLI_EMBEDDINGS.md)
- CLI examples: [../examples/embeddings_examples.sh](../examples/embeddings_examples.sh)
- Main README: [../README.md](../README.md)
