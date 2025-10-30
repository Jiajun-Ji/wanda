#!/bin/bash

# Example: Pruning with GSM8K calibration data
# 使用 GSM8K 数据集作为校准数据进行剪枝

MODEL="/mnt/sdb/llm_models/Llama-2-7b-hf"
SPARSITY_RATIOS="0.35 0.45 0.2"  # Dense, 2:4, Top-K
NSAMPLES=128
SAVE_DIR="out/llama2_7b/gsm8k_calibrated"

echo "=========================================="
echo "Pruning with GSM8K Calibration Data"
echo "=========================================="
echo "Model: ${MODEL}"
echo "Sparsity ratios: ${SPARSITY_RATIOS}"
echo "Calibration samples: ${NSAMPLES}"
echo "Save directory: ${SAVE_DIR}"
echo "=========================================="

# Run pruning with GSM8K data
python main_block_three_tier.py \
    --model ${MODEL} \
    --sparsity_ratios ${SPARSITY_RATIOS} \
    --nsamples ${NSAMPLES} \
    --calib_dataset gsm8k \
    --save ${SAVE_DIR}

echo ""
echo "=========================================="
echo "Pruning completed!"
echo "Model saved to: ${SAVE_DIR}"
echo "=========================================="

