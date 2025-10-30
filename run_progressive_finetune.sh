#!/bin/bash

# Progressive Pruning - Finetuning Script
# Finetune a pruned model from progressive pruning

# ========================================
# Parse Arguments
# ========================================

# Default values
PRUNED_MODEL=""
OUTPUT_DIR=""
NUM_EPOCHS=1
NUM_GPUS=1
GPU_IDS="1,6"
USE_FLASH_ATTN="false"
USE_SDPA="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; then
    case $1 in
        --model)
            PRUNED_MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
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
            echo "Usage: $0 --model <pruned_model> --output <output_dir> [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --model PATH        Path to pruned model"
            echo "  --output PATH       Output directory for finetuned model"
            echo ""
            echo "Optional:"
            echo "  --epochs N          Number of epochs (default: 1)"
            echo "  --num_gpus N        Number of GPUs to use (default: 1)"
            echo "  --gpu_ids \"X,Y\"     Comma-separated GPU IDs (default: \"1,6\")"
            echo "  --flash_attn        Enable Flash Attention (requires flash-attn library)"
            echo "  --sdpa              Enable SDPA (PyTorch 2.0+ native)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic usage"
            echo "  $0 --model out/progressive_three_tier/iter1/pruned_model \\"
            echo "     --output out/progressive_three_tier/iter1/finetuned_model"
            echo ""
            echo "  # With Flash Attention"
            echo "  $0 --model out/progressive_three_tier/iter1/pruned_model \\"
            echo "     --output out/progressive_three_tier/iter1/finetuned_model \\"
            echo "     --flash_attn"
            echo ""
            echo "  # Multi-GPU with SDPA"
            echo "  $0 --model out/progressive_three_tier/iter1/pruned_model \\"
            echo "     --output out/progressive_three_tier/iter1/finetuned_model \\"
            echo "     --num_gpus 2 --sdpa"
            exit 0
            ;;
        *)
            echo "❌ Unknown argument: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$PRUNED_MODEL" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "❌ Error: --model and --output are required"
    echo "Run '$0 --help' for usage information."
    exit 1
fi

# Check if model exists
if [ ! -d "$PRUNED_MODEL" ]; then
    echo "❌ Error: Pruned model not found at ${PRUNED_MODEL}"
    exit 1
fi

# ========================================
# Configuration
# ========================================

BASE_MODEL="/mnt/sdb/llm_models/Llama-2-7b-hf"
DATASET="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"

# Training hyperparameters
LEARNING_RATE=2e-5
BATCH_SIZE=4
EVAL_BATCH_SIZE=4
BLOCK_SIZE=1024

# Gradient accumulation
if [ $NUM_GPUS -eq 1 ]; then
    GRADIENT_ACCUMULATION_STEPS=4
elif [ $NUM_GPUS -eq 2 ]; then
    GRADIENT_ACCUMULATION_STEPS=2
else
    GRADIENT_ACCUMULATION_STEPS=$((4 / NUM_GPUS))
    if [ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]; then
        GRADIENT_ACCUMULATION_STEPS=1
    fi
fi

# GPU configuration
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Environment
FINETUNE_ENV="wanda_lora"

# ========================================
# Display configuration
# ========================================

echo "=========================================="
echo "Progressive Pruning - Finetuning"
echo "=========================================="
echo "Pruned model: ${PRUNED_MODEL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET} (${DATASET_CONFIG})"
echo ""
echo "GPU Configuration:"
echo "  Number of GPUs: ${NUM_GPUS}"
echo "  GPU IDs: ${GPU_IDS}"
echo "  Flash Attention: ${USE_FLASH_ATTN}"
echo "  SDPA: ${USE_SDPA}"
echo ""
echo "Training Parameters:"
echo "  Number of epochs: ${NUM_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Per-device batch size: ${BATCH_SIZE}"
echo "  Gradient accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "  Block size: ${BLOCK_SIZE}"
echo "=========================================="
echo ""

# ========================================
# Build attention arguments
# ========================================

ATTN_ARGS=""
if [ "$USE_FLASH_ATTN" = "true" ] && [ "$USE_SDPA" = "true" ]; then
    echo "❌ Error: Cannot use both --flash_attn and --sdpa. Please choose one."
    exit 1
fi

if [ "$USE_FLASH_ATTN" = "true" ]; then
    ATTN_ARGS="--use_flash_attention"
    echo "✅ Using Flash Attention"
elif [ "$USE_SDPA" = "true" ]; then
    ATTN_ARGS="--use_sdpa"
    echo "✅ Using SDPA (PyTorch native)"
fi

echo ""

# ========================================
# Start training
# ========================================

echo "🚀 Starting finetuning (using ${FINETUNE_ENV} environment)..."
echo "=========================================="

cd dense_ft

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    echo "Using single GPU training..."
    mamba run -n ${FINETUNE_ENV} python finetune_sparse_model.py \
        --model_name_or_path ../${PRUNED_MODEL} \
        --config_name ${BASE_MODEL} \
        --dataset_name ${DATASET} \
        --dataset_config_name ${DATASET_CONFIG} \
        --num_train_epochs ${NUM_EPOCHS} \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --do_train \
        --do_eval \
        --validation_split_percentage 10 \
        --learning_rate ${LEARNING_RATE} \
        --overwrite_output_dir \
        --output_dir ../${OUTPUT_DIR} \
        --logging_steps 50 \
        --eval_steps 100 \
        --save_steps 100 \
        --logging_first_step \
        --eval_strategy steps \
        --save_strategy steps \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --greater_is_better False \
        --bf16 \
        --gradient_checkpointing \
        --optim adamw_torch \
        --max_grad_norm 1.0 \
        --warmup_steps 100 \
        --weight_decay 0.01 \
        --lr_scheduler_type cosine \
        --seed 42 \
        ${ATTN_ARGS}
else
    # Multi-GPU training with torchrun
    echo "Using multi-GPU training with torchrun (${NUM_GPUS} GPUs)..."
    mamba run -n ${FINETUNE_ENV} torchrun --nproc_per_node=${NUM_GPUS} finetune_sparse_model.py \
        --model_name_or_path ../${PRUNED_MODEL} \
        --config_name ${BASE_MODEL} \
        --dataset_name ${DATASET} \
        --dataset_config_name ${DATASET_CONFIG} \
        --num_train_epochs ${NUM_EPOCHS} \
        --block_size ${BLOCK_SIZE} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
        --do_train \
        --do_eval \
        --validation_split_percentage 10 \
        --learning_rate ${LEARNING_RATE} \
        --overwrite_output_dir \
        --output_dir ../${OUTPUT_DIR} \
        --logging_steps 50 \
        --eval_steps 100 \
        --save_steps 100 \
        --logging_first_step \
        --eval_strategy steps \
        --save_strategy steps \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --greater_is_better False \
        --bf16 \
        --gradient_checkpointing \
        --optim adamw_torch \
        --max_grad_norm 1.0 \
        --warmup_steps 100 \
        --weight_decay 0.01 \
        --lr_scheduler_type cosine \
        --seed 42 \
        ${ATTN_ARGS}
fi

cd ..

# ========================================
# Training complete
# ========================================

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Finetuning Complete!"
    echo "=========================================="
    echo "Finetuned model saved to: ${OUTPUT_DIR}"

    # Evaluate finetuned model using wanda's eval_ppl
    echo ""
    echo "Evaluating finetuned model on WikiText-2..."
    echo "----------------------------------------"

    PRUNE_ENV="prune_llm"

    EVAL_PPL=$(mamba run -n ${PRUNE_ENV} python wanda/eval_model_ppl.py ${OUTPUT_DIR} ${BASE_MODEL} 2>&1 | grep "Perplexity:" | awk '{print $2}')

    echo "✅ Finetuned model perplexity: ${EVAL_PPL}"

    # Save evaluation result
    OUTPUT_DIR_BASENAME=$(basename ${OUTPUT_DIR})
    OUTPUT_DIR_PARENT=$(dirname ${OUTPUT_DIR})
    echo "${EVAL_PPL}" > ${OUTPUT_DIR}/finetuned_ppl.txt

    echo "=========================================="
else
    echo ""
    echo "❌ Error: Finetuning failed!"
    exit 1
fi

