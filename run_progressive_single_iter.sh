#!/bin/bash

# Run a single iteration of progressive three-tier pruning
# Usage: ./run_progressive_single_iter.sh <iteration_number>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <iteration_number> [previous_model] [previous_tier_maps]"
    echo ""
    echo "Examples:"
    echo "  # Iteration 1 (from base model)"
    echo "  $0 1"
    echo ""
    echo "  # Iteration 2 (from iter1 finetuned model)"
    echo "  $0 2 out/progressive_three_tier/iter1/finetuned_model out/progressive_three_tier/iter1/tier_maps_iter1.pt"
    exit 1
fi

ITER=$1
PREV_MODEL=${2:-"/mnt/sdb/llm_models/Llama-2-7b-hf"}
PREV_TIER_MAPS=${3:-""}

# Configuration
CONFIG_FILE="progressive_config.csv"
BLOCK_SIZE=16
TOPK_PER_BLOCK=15
OUTPUT_BASE="out/progressive_three_tier"

# Environment configuration
PRUNE_ENV="prune_llm"      # Environment for pruning (transformers 4.36.0)

# GPU configuration
export CUDA_VISIBLE_DEVICES=1,6

echo "=========================================="
echo "Progressive Three-Tier Pruning"
echo "Iteration ${ITER}"
echo "=========================================="
echo "Previous model: ${PREV_MODEL}"
if [ ! -z "${PREV_TIER_MAPS}" ]; then
    echo "Previous tier maps: ${PREV_TIER_MAPS}"
else
    echo "Previous tier maps: None (starting from scratch)"
fi
echo "=========================================="
echo ""

# Paths for this iteration
ITER_DIR="${OUTPUT_BASE}/iter${ITER}"
PRUNED_MODEL="${ITER_DIR}/pruned_model"
TIER_MAPS="${ITER_DIR}/tier_maps_iter${ITER}.pt"

mkdir -p ${ITER_DIR}

# Pruning (use prune_llm environment)
echo "Running pruning (using ${PRUNE_ENV} environment)..."
echo "----------------------------------------"

PRUNE_CMD="mamba run -n ${PRUNE_ENV} python main_progressive_three_tier.py \
    --model ${PREV_MODEL} \
    --iteration ${ITER} \
    --config ${CONFIG_FILE} \
    --block_size ${BLOCK_SIZE} \
    --topk_per_block ${TOPK_PER_BLOCK} \
    --save ${ITER_DIR}/ \
    --save_model ${PRUNED_MODEL}"

if [ ! -z "${PREV_TIER_MAPS}" ]; then
    PRUNE_CMD="${PRUNE_CMD} --previous_tier_maps ${PREV_TIER_MAPS}"
fi

echo "Command: ${PRUNE_CMD}"
echo ""

eval ${PRUNE_CMD}

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Pruning Complete!"
    echo "=========================================="
    echo "Pruned model saved to: ${PRUNED_MODEL}"
    echo "Tier maps saved to: ${TIER_MAPS}"
    echo ""
    echo "Next steps:"
    echo "  1. Finetune the model using dense_ft/finetune_sparse_model.py"
    echo "  2. Run next iteration with:"
    echo "     $0 $((ITER+1)) ${ITER_DIR}/finetuned_model ${TIER_MAPS}"
    echo "=========================================="
else
    echo ""
    echo "Error: Pruning failed!"
    exit 1
fi

