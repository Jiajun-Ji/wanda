#!/bin/bash

# Additional Pruning Script - LOWEST Strategy
# This script performs additional 0.5% pruning on the already 50% sparse Llama-2-7b model
# using Wanda method (pruning weights with LOWEST Wanda scores - least important)

# Set variables
PRUNED_MODEL_PATH="out/llama2_7b/unstructured/wanda/pruned_model"
ADDITIONAL_SPARSITY=0.01  # 0.5% of non-zero weights
OUTPUT_DIR="out/llama2_7b/unstructured/wanda_additional_lowest_0.5"
MODEL_SAVE_DIR="${OUTPUT_DIR}/pruned_model"
NSAMPLES=128
SEED=0
PRUNE_STRATEGY="lowest"

# Optional: Set CUDA device (default: 0)
export CUDA_VISIBLE_DEVICES=0

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Additional Pruning Configuration - LOWEST"
echo "=========================================="
echo "Pre-pruned model: ${PRUNED_MODEL_PATH}"
echo "Additional sparsity: ${ADDITIONAL_SPARSITY} (0.5% of non-zero weights)"
echo "Pruning strategy: ${PRUNE_STRATEGY} (prune LEAST important weights)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model save directory: ${MODEL_SAVE_DIR}"
echo "Calibration samples: ${NSAMPLES}"
echo "Random seed: ${SEED}"
echo "=========================================="
echo ""

# Run additional pruning
python prune_additional.py \
    --model ${PRUNED_MODEL_PATH} \
    --additional_sparsity_ratio ${ADDITIONAL_SPARSITY} \
    --prune_strategy ${PRUNE_STRATEGY} \
    --nsamples ${NSAMPLES} \
    --seed ${SEED} \
    --save ${OUTPUT_DIR} \
    --save_model ${MODEL_SAVE_DIR}

echo ""
echo "=========================================="
echo "Additional Pruning Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Model saved to: ${MODEL_SAVE_DIR}"
echo "=========================================="

