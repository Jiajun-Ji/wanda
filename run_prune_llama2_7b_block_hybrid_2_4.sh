#!/bin/bash

# Hybrid Block Pruning with 2:4 Sparsity
# Three types of blocks:
# 1. Fully dense blocks (most important)
# 2. 2:4 sparse blocks (moderately important, hardware-friendly)
# 3. Top-K sparse blocks (least important, scattered values)

# Configuration
MODEL_PATH="/mnt/sdb/llm_models/Llama-2-7b-hf"
SPARSITY=0.5
BLOCK_SIZE=16
TOPK_PER_BLOCK=10
TOP_BLOCKS_RATIO=0.1  # Top 60% blocks
SCORE_THRESHOLD=0.75   # 80% score retention for 2:4
OUTPUT_DIR="out/llama2_7b/block_${BLOCK_SIZE}x${BLOCK_SIZE}_hybrid_2_4/wanda_${TOP_BLOCKS_RATIO}"

# GPU configuration
export CUDA_VISIBLE_DEVICES=1

echo "=========================================="
echo "Wanda Hybrid Block Pruning with 2:4"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Sparsity: ${SPARSITY}"
echo "Block size: ${BLOCK_SIZE}x${BLOCK_SIZE}"
echo "Top blocks ratio: ${TOP_BLOCKS_RATIO} (top ${TOP_BLOCKS_RATIO}% blocks)"
echo "Score threshold for 2:4: ${SCORE_THRESHOLD} (${SCORE_THRESHOLD}% score retention)"
echo "Top-K per low-score block: ${TOPK_PER_BLOCK}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

python main_block_hybrid_2_4.py \
    --model ${MODEL_PATH} \
    --prune_method wanda \
    --sparsity_ratio ${SPARSITY} \
    --sparsity_type hybrid_2_4 \
    --block_size ${BLOCK_SIZE} \
    --topk_per_block ${TOPK_PER_BLOCK} \
    --top_blocks_ratio ${TOP_BLOCKS_RATIO} \
    --score_threshold ${SCORE_THRESHOLD} \
    --save ${OUTPUT_DIR}/ \
    --save_model ${OUTPUT_DIR}/pruned_model

echo ""
echo "=========================================="
echo "Hybrid Block Pruning Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Model saved to: ${OUTPUT_DIR}/pruned_model"
echo ""
echo "Block Types:"
echo "  1. Fully dense blocks (most important)"
echo "  2. 2:4 sparse blocks (moderately important)"
echo "  3. Top-K blocks (least important)"
echo "=========================================="

