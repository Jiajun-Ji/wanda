#!/bin/bash

# Wanda Block Pruning with Top-K Preservation
# This script prunes blocks but keeps top-k weights within each pruned block

# Configuration
MODEL_PATH="/mnt/sdb/llm_models/Llama-2-7b-hf"
SPARSITY=0.5
BLOCK_SIZE=16
TOPK_PER_BLOCK=10
OUTPUT_DIR="out/llama2_7b/block_${BLOCK_SIZE}x${BLOCK_SIZE}_topk${TOPK_PER_BLOCK}/wanda"

# CUDA device
export CUDA_VISIBLE_DEVICES=2

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Wanda Block Pruning with Top-K"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Sparsity: ${SPARSITY}"
echo "Block size: ${BLOCK_SIZE}x${BLOCK_SIZE}"
echo "Top-K per pruned block: ${TOPK_PER_BLOCK}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Run pruning
python main_block_topk.py \
    --model ${MODEL_PATH} \
    --prune_method wanda \
    --sparsity_ratio ${SPARSITY} \
    --sparsity_type unstructured \
    --block_size ${BLOCK_SIZE} \
    --topk_per_block ${TOPK_PER_BLOCK} \
    --save ${OUTPUT_DIR} \
    --save_model ${OUTPUT_DIR}/pruned_model

echo ""
echo "=========================================="
echo "Pruning Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Model saved to: ${OUTPUT_DIR}/pruned_model"
echo "=========================================="

