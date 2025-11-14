#!/bin/bash
#
# Embeddings Examples
# 
# This script demonstrates how to use the 'bedrock-bencher embed' command
# with the sample datasets in the examples directory.
#
# Prerequisites:
# - AWS credentials configured
# - Access to Bedrock embedding models
# - bedrock-bencher installed
#

set -e  # Exit on error

echo "=========================================="
echo "Bedrock Embeddings Embedding Examples"
echo "=========================================="
echo ""

# Example 1: Basic text embeddings with Titan Multimodal
echo "Example 1: Text embeddings with Titan Multimodal (256 dimensions)"
echo "------------------------------------------------------------------"
bedrock-bencher embed \
  --dataset examples/embeddings_sample_texts.jsonl \
  --model amazon.titan-embed-image-v1 \
  --output-dimension 256 \
  --experiment-name "Embedding Examples" \
  --run-name "titan-multimodal-256" \
  --verbose

echo ""
echo "✓ Example 1 complete!"
echo ""

# Example 2: Text embeddings with higher dimensions
echo "Example 2: Text embeddings with Titan Multimodal (1024 dimensions)"
echo "-------------------------------------------------------------------"
bedrock-bencher embed \
  --dataset examples/embeddings_sample_texts.jsonl \
  --model amazon.titan-embed-image-v1 \
  --output-dimension 1024 \
  --experiment-name "Embedding Examples" \
  --run-name "titan-multimodal-1024" \
  --verbose

echo ""
echo "✓ Example 2 complete!"
echo ""

# Example 3: Text embeddings with Nova
echo "Example 3: Text embeddings with Nova"
echo "-------------------------------------"
bedrock-bencher embed \
  --dataset examples/embeddings_sample_texts.jsonl \
  --model amazon.nova-2-multimodal-embeddings-v1:0 \
  --embedding-purpose GENERIC_INDEX \
  --experiment-name "Embedding Examples" \
  --run-name "nova-text"

echo ""
echo "✓ Example 3 complete!"
echo ""

# Example 4: Text embeddings with Cohere v4
echo "Example 4: Text embeddings with Cohere v4"
echo "------------------------------------------"
bedrock-bencher embed \
  --dataset examples/embeddings_sample_texts.jsonl \
  --model us.cohere.embed-v4:0 \
  --input-type search_document \
  --experiment-name "Embedding Examples" \
  --run-name "cohere-v4"

echo ""
echo "✓ Example 4 complete!"
echo ""

# Example 5: High-throughput processing
echo "Example 5: High-throughput processing (max concurrency)"
echo "--------------------------------------------------------"
bedrock-bencher embed \
  --dataset examples/embeddings_sample_texts.jsonl \
  --model amazon.titan-embed-image-v1 \
  --output-dimension 256 \
  --max-concurrent 20 \
  --experiment-name "Embedding Examples" \
  --run-name "high-throughput"

echo ""
echo "✓ Example 5 complete!"
echo ""

# Example 6: Image embeddings with Titan Multimodal
echo "Example 6: Image embeddings with Titan Multimodal"
echo "--------------------------------------------------"
bedrock-bencher embed \
  --dataset examples/embeddings_sample_images.jsonl \
  --model amazon.titan-embed-image-v1 \
  --output-dimension 1024 \
  --experiment-name "Embedding Examples" \
  --run-name "titan-images"

echo ""
echo "✓ Example 6 complete!"
echo ""

# Example 7: Image embeddings with Cohere v4
echo "Example 7: Image embeddings with Cohere v4"
echo "-------------------------------------------"
bedrock-bencher embed \
  --dataset examples/embeddings_sample_images.jsonl \
  --model us.cohere.embed-v4:0 \
  --input-type search_document \
  --experiment-name "Embedding Examples" \
  --run-name "cohere-v4-images"

echo ""
echo "✓ Example 7 complete!"
echo ""

# List all runs in the experiment
echo "=========================================="
echo "Listing all runs in 'Embedding Examples' experiment"
echo "=========================================="
bedrock-bencher list-runs embedding-examples

echo ""
echo "=========================================="
echo "All examples completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Export a run: bedrock-bencher export-run <run-id> --format csv"
echo "  2. Compare runs: bedrock-bencher compare-runs <run-id-1> <run-id-2>"
echo "  3. Show run details: bedrock-bencher show-run <run-id>"
echo ""
