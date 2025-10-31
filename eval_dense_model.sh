#!/bin/bash

# Evaluate dense model (before task-specific finetuning)
# on BoolQ and GSM8K using instruction format

set -e

MODEL_PATH="/home/jjji/Research/Hybird-Kernel/wanda/out/progressive_three_tier/iter5/dense_finetuned_model"
CUDA_DEVICE=6,7

echo "=========================================="
echo "Evaluating Dense Model (Before Task-Specific Finetuning)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "=========================================="

# ============ BoolQ Evaluation ============
echo ""
echo "=========================================="
echo "Evaluating on BoolQ..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python eval_boolq_instruction.py \
    --model ${MODEL_PATH} \
    --output_dir eval_results_dense_boolq

echo "BoolQ evaluation completed!"

# ============ GSM8K Evaluation ============
echo ""
echo "=========================================="
echo "Evaluating on GSM8K..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python eval_gsm8k_instruction.py \
    --model ${MODEL_PATH} \
    --max_samples 200 \
    --output_dir eval_results_dense_gsm8k

echo "GSM8K evaluation completed!"

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo "BoolQ results: eval_results_dense_boolq/"
echo "GSM8K results: eval_results_dense_gsm8k/"
echo "=========================================="

