#!/bin/bash

# ========================================
# Continue Training from Checkpoint
# ========================================
# This script continues training from the last checkpoint.
#
# Usage:
#   bash run_continue_training.sh [--num_gpus N] [--gpu_ids "X,Y"] [--additional_epochs N]
#
# Examples:
#   bash run_continue_training.sh --additional_epochs 10
#   bash run_continue_training.sh --num_gpus 2 --gpu_ids "2,3" --additional_epochs 10
# ========================================

# Default values
NUM_GPUS=1
GPU_IDS="2,3"
ADDITIONAL_EPOCHS=15
USE_FLASH_ATTN="false"
USE_SDPA="true"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --additional_epochs)
            ADDITIONAL_EPOCHS="$2"
            shift 2
            ;;
        --flash_attn)
            USE_FLASH_ATTN="true"
            shift 1
            ;;
        --sdpa)
            USE_SDPA="true"
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_gpus N              Number of GPUs to use (default: 1)"
            echo "  --gpu_ids \"X,Y\"           Comma-separated GPU IDs (default: \"2\")"
            echo "  --additional_epochs N     Number of additional epochs to train (default: 10)"
            echo "  --flash_attn              Enable Flash Attention 3"
            echo "  --sdpa                    Enable SDPA"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --additional_epochs 10"
            echo "  $0 --num_gpus 2 --gpu_ids \"2,3\" --additional_epochs 10"
            exit 0
            ;;
        *)
            echo "❌ Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ========================================
# Configuration
# ========================================

# Model paths
OUTPUT_DIR="out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model"
CONFIG_NAME="/mnt/sdb/llm_models/Llama-2-7b-hf"
DATASET="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"

# Check if output directory exists
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo "❌ Error: Output directory not found at ${OUTPUT_DIR}"
    echo "Please run the initial training first."
    exit 1
fi

# Find the latest checkpoint (use absolute path)
LATEST_CHECKPOINT=$(ls -d $(pwd)/${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ Error: No checkpoint found in ${OUTPUT_DIR}"
    echo "Please run the initial training first."
    exit 1
fi

echo "========================================="
echo "Continue Training from Checkpoint"
echo "========================================="
echo "Latest checkpoint: ${LATEST_CHECKPOINT}"
echo "Additional epochs: ${ADDITIONAL_EPOCHS}"
echo "GPUs: ${NUM_GPUS} (${GPU_IDS})"
echo "========================================="

# Read current epoch from trainer_state.json
if [ -f "${LATEST_CHECKPOINT}/trainer_state.json" ]; then
    CURRENT_EPOCH=$(python3 -c "import json; print(json.load(open('${LATEST_CHECKPOINT}/trainer_state.json'))['epoch'])")
    TOTAL_EPOCHS=$(python3 -c "print(int(${CURRENT_EPOCH} + ${ADDITIONAL_EPOCHS}))")
    echo "Current epoch: ${CURRENT_EPOCH}"
    echo "Target total epochs: ${TOTAL_EPOCHS}"
else
    echo "⚠️  Warning: Cannot read current epoch, using default"
    TOTAL_EPOCHS=$((3 + ADDITIONAL_EPOCHS))
fi

# Training hyperparameters
LEARNING_RATE=5e-5
BATCH_SIZE=1
EVAL_BATCH_SIZE=4
BLOCK_SIZE=1024
MAX_TRAIN_SAMPLES=30000
MAX_EVAL_SAMPLES=128

# Gradient accumulation
if [ $NUM_GPUS -eq 1 ]; then
    GRADIENT_ACCUMULATION_STEPS=16
elif [ $NUM_GPUS -eq 2 ]; then
    GRADIENT_ACCUMULATION_STEPS=8
elif [ $NUM_GPUS -eq 4 ]; then
    GRADIENT_ACCUMULATION_STEPS=4
else
    GRADIENT_ACCUMULATION_STEPS=$((16 / NUM_GPUS))
    if [ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]; then
        GRADIENT_ACCUMULATION_STEPS=1
    fi
fi

# GPU configuration
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Build attention arguments
ATTN_ARGS=""
if [ "$USE_FLASH_ATTN" = "true" ] && [ "$USE_SDPA" = "true" ]; then
    echo "❌ Error: Cannot use both --flash_attn and --sdpa"
    exit 1
fi

if [ "$USE_FLASH_ATTN" = "true" ]; then
    ATTN_ARGS="--use_flash_attention"
elif [ "$USE_SDPA" = "true" ]; then
    ATTN_ARGS="--use_sdpa"
fi

# ========================================
# Run Training
# ========================================

cd dense_ft

echo ""
echo "🚀 Continuing training..."
echo ""

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    python finetune_sparse_model.py \
        --model_name_or_path ${LATEST_CHECKPOINT} \
        --config_name ${CONFIG_NAME} \
        --dataset_name ${DATASET} \
        --dataset_config_name ${DATASET_CONFIG} \
        --num_train_epochs ${TOTAL_EPOCHS} \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --do_train \
        --do_eval \
        --validation_split_percentage 10 \
        --max_train_samples ${MAX_TRAIN_SAMPLES} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --learning_rate ${LEARNING_RATE} \
        --overwrite_output_dir \
        --output_dir ../${OUTPUT_DIR} \
        --logging_steps 10 \
        --eval_steps 100 \
        --save_steps 100 \
        --logging_first_step \
        --eval_strategy steps \
        --save_strategy steps \
        --save_total_limit 3 \
        --bf16 \
        --gradient_checkpointing \
        --optim adamw_torch \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --max_grad_norm 1.0 \
        --warmup_steps 100 \
        --weight_decay 0.01 \
        --lr_scheduler_type cosine \
        --seed 42 \
        ${ATTN_ARGS}
else
    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=${NUM_GPUS} \
        finetune_sparse_model.py \
        --model_name_or_path ${LATEST_CHECKPOINT} \
        --config_name ${CONFIG_NAME} \
        --dataset_name ${DATASET} \
        --dataset_config_name ${DATASET_CONFIG} \
        --num_train_epochs ${TOTAL_EPOCHS} \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --do_train \
        --do_eval \
        --validation_split_percentage 10 \
        --max_train_samples ${MAX_TRAIN_SAMPLES} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --learning_rate ${LEARNING_RATE} \
        --overwrite_output_dir \
        --output_dir ../${OUTPUT_DIR} \
        --logging_steps 10 \
        --eval_steps 100 \
        --save_steps 100 \
        --logging_first_step \
        --eval_strategy steps \
        --save_strategy steps \
        --save_total_limit 3 \
        --bf16 \
        --gradient_checkpointing \
        --optim adamw_torch \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --max_grad_norm 1.0 \
        --warmup_steps 100 \
        --weight_decay 0.01 \
        --lr_scheduler_type cosine \
        --seed 42 \
        ${ATTN_ARGS}
fi

echo ""
echo "✅ Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"

