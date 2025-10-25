#!/bin/bash

# Llama-2-7b Pruning Script with Wanda
# This script prunes Llama-2-7b to 50% sparsity using WikiText2 dataset

# Set common variables
MODEL_PATH="/mnt/sdb/llm_models/Llama-2-7b-hf"
SPARSITY_RATIO=0.5
PRUNE_METHOD="wanda"
SPARSITY_TYPE="unstructured"
OUTPUT_DIR="out/llama2_7b/unstructured/wanda"
MODEL_SAVE_DIR="out/llama2_7b/unstructured/wanda/pruned_model"

# Optional: Set CUDA device (default: 0)
export CUDA_VISIBLE_DEVICES=0

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Pruning Llama-2-7b with Wanda"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Pruning Method: ${PRUNE_METHOD}"
echo "Sparsity Ratio: ${SPARSITY_RATIO}"
echo "Sparsity Type: ${SPARSITY_TYPE}"
echo "Calibration Dataset: WikiText2"
echo "Evaluation Dataset: WikiText2"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Model Save Directory: ${MODEL_SAVE_DIR}"
echo "=========================================="

# Run pruning
python main.py \
    --model ${MODEL_PATH} \
    --prune_method ${PRUNE_METHOD} \
    --sparsity_ratio ${SPARSITY_RATIO} \
    --sparsity_type ${SPARSITY_TYPE} \
    --save ${OUTPUT_DIR} \
    --save_model ${MODEL_SAVE_DIR} \
    --nsamples 128 \
    --seed 0

echo "=========================================="
echo "Pruning completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Pruned model saved to: ${MODEL_SAVE_DIR}"
echo "=========================================="

