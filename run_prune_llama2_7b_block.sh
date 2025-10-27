#!/bin/bash

# Block-wise Wanda Pruning Script for Llama-2-7b
# This script performs 16x16 block-structured pruning using Wanda scoring method

# Configuration
MODEL_PATH="/mnt/sdb/llm_models/Llama-2-7b-hf"
SPARSITY=0.2
BLOCK_SIZE=16
OUTPUT_DIR="out/llama2_7b/block_${BLOCK_SIZE}x${BLOCK_SIZE}_20sparsity/wanda"
MODEL_SAVE_DIR="${OUTPUT_DIR}/pruned_model"
NSAMPLES=128
SEED=0

# CUDA device
export CUDA_VISIBLE_DEVICES=2

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Block-wise Wanda Pruning Configuration"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Sparsity: ${SPARSITY} (50%)"
echo "Block size: ${BLOCK_SIZE}x${BLOCK_SIZE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Model save directory: ${MODEL_SAVE_DIR}"
echo "Calibration samples: ${NSAMPLES}"
echo "Random seed: ${SEED}"
echo "=========================================="
echo ""

# Run block pruning
python main_block.py \
    --model ${MODEL_PATH} \
    --sparsity_ratio ${SPARSITY} \
    --block_size ${BLOCK_SIZE} \
    --nsamples ${NSAMPLES} \
    --seed ${SEED} \
    --save ${OUTPUT_DIR} \
    --save_model ${MODEL_SAVE_DIR}

echo ""
echo "=========================================="
echo "Block Pruning Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Model saved to: ${MODEL_SAVE_DIR}"
echo "=========================================="

