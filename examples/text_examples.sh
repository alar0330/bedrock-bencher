#!/bin/bash
#
# Text Generation Benchmark Examples
# 
# This script demonstrates how to use the 'bedrock-bencher run-benchmark' command
# with the sample datasets in the examples directory.
#
# Prerequisites:
# - AWS credentials configured
# - Access to Bedrock text generation models
# - bedrock-bencher installed
#

set -e  # Exit on error

echo "=========================================="
echo "Bedrock Text Generation Benchmark Examples"
echo "=========================================="
echo ""

# Example 1: Quick benchmark with 100 Q&A items
echo "Example 1: Quick benchmark with 100 Q&A items (Claude Sonnet)"
echo "--------------------------------------------------------------"
bedrock-bencher run-benchmark \
  --dataset examples/text_qa_task_100.jsonl \
  --model anthropic.claude-3-sonnet-20240229-v1:0 \
  --experiment-name "Benchmark Examples" \
  --run-name "qa-100-sonnet" \
  --verbose

echo ""
echo "✓ Example 1 complete!"
echo ""

# Example 2: Same dataset with different model
echo "Example 2: Quick benchmark with 100 Q&A items (Claude Haiku)"
echo "-------------------------------------------------------------"
bedrock-bencher run-benchmark \
  --dataset examples/text_qa_task_100.jsonl \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --experiment-name "Benchmark Examples" \
  --run-name "qa-100-haiku"

echo ""
echo "✓ Example 2 complete!"
echo ""

# Example 3: Coding tasks
echo "Example 3: Coding tasks benchmark"
echo "----------------------------------"
bedrock-bencher run-benchmark \
  --dataset examples/coding_tasks.jsonl \
  --model anthropic.claude-3-sonnet-20240229-v1:0 \
  --experiment-name "Benchmark Examples" \
  --run-name "coding-sonnet"

echo ""
echo "✓ Example 3 complete!"
echo ""

# Example 4: Creative writing
echo "Example 4: Creative writing benchmark"
echo "--------------------------------------"
bedrock-bencher run-benchmark \
  --dataset examples/creative_writing.jsonl \
  --model anthropic.claude-3-sonnet-20240229-v1:0 \
  --experiment-name "Benchmark Examples" \
  --run-name "creative-sonnet"

echo ""
echo "✓ Example 4 complete!"
echo ""

# Example 5: Multilingual tasks
echo "Example 5: Multilingual tasks benchmark"
echo "----------------------------------------"
bedrock-bencher run-benchmark \
  --dataset examples/multilingual.jsonl \
  --model anthropic.claude-3-sonnet-20240229-v1:0 \
  --experiment-name "Benchmark Examples" \
  --run-name "multilingual-sonnet"

echo ""
echo "✓ Example 5 complete!"
echo ""

# Example 6: High-throughput processing
echo "Example 6: High-throughput processing (max concurrency)"
echo "--------------------------------------------------------"
bedrock-bencher run-benchmark \
  --dataset examples/text_qa_task_100.jsonl \
  --model anthropic.claude-3-haiku-20240307-v1:0 \
  --max-concurrent 20 \
  --experiment-name "Benchmark Examples" \
  --run-name "qa-100-high-throughput"

echo ""
echo "✓ Example 6 complete!"
echo ""

# List all runs in the experiment
echo "=========================================="
echo "Listing all runs in 'Benchmark Examples' experiment"
echo "=========================================="
bedrock-bencher list-runs benchmark-examples

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
