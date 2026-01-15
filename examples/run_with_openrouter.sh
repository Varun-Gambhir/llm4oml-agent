# ============================================================================
# File: examples/run_with_openrouter.sh
# ============================================================================
#!/bin/bash
# Example: Using OpenRouter instead of NVIDIA

export OPENROUTER_API_KEY="your_openrouter_key_here"

# Single algorithm with Claude via OpenRouter
python scripts/run_single.py \
  --provider openrouter \
  --model "anthropic/claude-3.5-sonnet" \
  --algorithm "Adam optimizer" \
  --assumptions "Non-convex, bounded gradients" \
  --output output/openrouter_test \
  --max-iter 3 \
  --timeout 600 \
  --max-retries 3

# Batch processing with multiple OpenRouter models
python scripts/run_batch.py \
  --provider openrouter \
  --input data/algorithms.csv \
  --output results/openrouter_batch \
  --model "anthropic/claude-3.5-sonnet" \
  --multi-judge \
  --judges "anthropic/claude-3.5-sonnet" "openai/gpt-4" "google/gemini-pro-1.5" \
  --max-iter 3 \
  --timeout 600 \
  --max-retries 3