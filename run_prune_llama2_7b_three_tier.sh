#!/bin/bash

# Three-Tier Fixed Ratio Block Pruning
# Three types of blocks with fixed ratios:
# 1. Top X% blocks: Fully dense (most important)
# 2. Middle Y% blocks: 2:4 sparse (moderately important, hardware-friendly)
# 3. Bottom Z% blocks: Top-K sparse (least important, scattered values)

# Configuration
MODEL_PATH="/mnt/sdb/llm_models/Llama-2-7b-hf"
BLOCK_SIZE=16
TOPK_PER_BLOCK=15

# Three-tier ratios (must sum to 1.0)
TOP_DENSE_RATIO=0.35   # Top 40% blocks → fully dense
MID_2_4_RATIO=0.45     # Middle 40% blocks → 2:4 sparse
BOTTOM_TOPK_RATIO=0.2 # Bottom 20% blocks → top-k sparse

OUTPUT_DIR="out/llama2_7b/block_${BLOCK_SIZE}x${BLOCK_SIZE}_three_tier_${TOP_DENSE_RATIO}_${MID_2_4_RATIO}_${BOTTOM_TOPK_RATIO}/wanda"

# GPU configuration
export CUDA_VISIBLE_DEVICES=6

echo "=========================================="
echo "Wanda Three-Tier Fixed Ratio Block Pruning"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Block size: ${BLOCK_SIZE}x${BLOCK_SIZE}"
echo ""
echo "Three-Tier Configuration:"
echo "  Tier 1 (Fully Dense): Top ${TOP_DENSE_RATIO} (${TOP_DENSE_RATIO}% blocks)"
echo "  Tier 2 (2:4 Sparse):  Middle ${MID_2_4_RATIO} (${MID_2_4_RATIO}% blocks)"
echo "  Tier 3 (Top-K):       Bottom ${BOTTOM_TOPK_RATIO} (${BOTTOM_TOPK_RATIO}% blocks, k=${TOPK_PER_BLOCK})"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

python main_block_three_tier.py \
    --model ${MODEL_PATH} \
    --prune_method wanda \
    --sparsity_type three_tier \
    --block_size ${BLOCK_SIZE} \
    --top_dense_ratio ${TOP_DENSE_RATIO} \
    --mid_2_4_ratio ${MID_2_4_RATIO} \
    --bottom_topk_ratio ${BOTTOM_TOPK_RATIO} \
    --topk_per_block ${TOPK_PER_BLOCK} \
    --save ${OUTPUT_DIR}/ \
    --save_model ${OUTPUT_DIR}/pruned_model

echo ""
echo "=========================================="
echo "Three-Tier Block Pruning Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Model saved to: ${OUTPUT_DIR}/pruned_model"
echo ""
echo "Block Types:"
echo "  1. Fully dense blocks (top ${TOP_DENSE_RATIO}%)"
echo "  2. 2:4 sparse blocks (middle ${MID_2_4_RATIO}%)"
echo "  3. Top-K blocks (bottom ${BOTTOM_TOPK_RATIO}%, k=${TOPK_PER_BLOCK})"
echo "=========================================="

