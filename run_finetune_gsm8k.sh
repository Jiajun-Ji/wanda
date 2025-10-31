#!/bin/bash

# GSM8K Fine-tuning Script
# Fine-tune pruned model on GSM8K dataset

set -e

# ============ Configuration ============
MODEL_PATH="/home/jjji/Research/Hybird-Kernel/wanda/out/progressive_three_tier/iter5/dense_finetuned_model"
OUTPUT_DIR="out/progressive_three_tier/iter5/gsm8k_finetuned"

# Training hyperparameters
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4  # Effective batch size = 4 * 4 = 16
LEARNING_RATE=2e-5
NUM_EPOCHS=1
MAX_LENGTH=512

# Hardware
CUDA_DEVICE=6,7

# ============ 快速测试配置 ============
# 限制训练样本数量（加快训练速度）
MAX_TRAIN_SAMPLES=1000    # 只用 1000 个样本训练（原本 7473 个）
MAX_EVAL_SAMPLES=200      # 只用 200 个样本评估（原本 1319 个）

# 如果想用全部数据，注释掉上面两行即可

# ============ Run Training ============
echo "=========================================="
echo "GSM8K Fine-tuning"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} = $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo "Learning rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python dense_ft/finetune_gsm8k.py \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --max_length ${MAX_LENGTH} \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --do_train \
    --do_eval \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --save_strategy "epoch" \
    --eval_strategy "epoch" \
    --save_total_limit 1 \
    --bf16 \
    --dataloader_num_workers 4 \
    --preprocessing_num_workers 4 \
    --seed 42 \
    --report_to none

echo "=========================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=========================================="

# ============ Instruction Format 评估 ============
echo ""
echo "=========================================="
echo "Evaluating with Instruction Format..."
echo "=========================================="

# 评估微调后的模型
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python eval_gsm8k_instruction.py \
    --model ${OUTPUT_DIR} \
    --output_dir eval_results_gsm8k_instruction

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: eval_results_gsm8k_instruction"
echo "=========================================="

