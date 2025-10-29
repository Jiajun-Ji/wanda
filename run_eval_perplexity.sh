#!/bin/bash

# ========================================
# Evaluate Perplexity on WikiText-2
# ========================================
# This script evaluates the perplexity of a fine-tuned model on WikiText-2 test set.
#
# Usage:
#   bash run_eval_perplexity.sh <model_path> [gpu_id]
#
# Examples:
#   bash run_eval_perplexity.sh out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model
#   bash run_eval_perplexity.sh out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model 0
# ========================================

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [gpu_id]"
    echo ""
    echo "Examples:"
    echo "  $0 out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model"
    echo "  $0 out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model 0"
    exit 1
fi

MODEL_PATH=$1
GPU_ID=${2:-0}  # Default to GPU 0

# Check if model exists
if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "Perplexity Evaluation"
echo "========================================="
echo "Model: ${MODEL_PATH}"
echo "GPU: ${GPU_ID}"
echo "========================================="
echo ""

# Run evaluation
cd dense_ft
python ../eval_perplexity.py \
    --model_path ../${MODEL_PATH} \
    --device cuda:0 \
    --seqlen 2048

echo ""
echo "✅ Evaluation complete!"

