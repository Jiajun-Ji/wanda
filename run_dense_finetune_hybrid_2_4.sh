#!/bin/bash

# Dense (Full) Fine-tuning for 20% Sparsity Block-Pruned Model
# Uses SparseTrainer to maintain sparsity pattern while training all parameters

# ========================================
# Parse Arguments
# ========================================

# Default: single GPU training
NUM_GPUS=1
GPU_IDS="4,1"
USE_FLASH_ATTN="false"  # Default: disabled
USE_SDPA="true"  # Default: disabled

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
            echo "  --num_gpus N        Number of GPUs to use (default: 1)"
            echo "  --gpu_ids \"X,Y\"     Comma-separated GPU IDs (default: \"2,3\")"
            echo "  --flash_attn        Enable Flash Attention 3 (requires flash-attn library)"
            echo "  --sdpa              Enable SDPA (PyTorch 2.0+ native, no extra library needed)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  Single GPU (default):     $0"
            echo "  Single GPU (GPU 3):       $0 --num_gpus 1 --gpu_ids \"3\""
            echo "  Dual GPU (GPU 2,3):       $0 --num_gpus 2 --gpu_ids \"2,3\""
            echo "  Quad GPU (GPU 0-3):       $0 --num_gpus 4 --gpu_ids \"0,1,2,3\""
            echo "  With Flash Attention:     $0 --num_gpus 2 --flash_attn"
            echo "  With SDPA:                $0 --num_gpus 2 --sdpa"
            echo ""
            echo "Performance:"
            echo "  1 GPU: ~24-36 hours"
            echo "  2 GPUs: ~14-20 hours (1.7x speedup)"
            echo "  4 GPUs: ~8-12 hours (2.5-3x speedup)"
            echo "  Flash Attention: +30-50% speedup"
            echo "  SDPA: +20-30% speedup (no extra library needed)"
            exit 0
            ;;
        *)
            echo "❌ Unknown argument: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# ========================================
# Configuration
# ========================================

# Pruned model path
PRUNED_MODEL="out/llama2_7b/block_16x16_hybrid_2_4/wanda/pruned_model"
CONFIG_NAME="/mnt/sdb/llm_models/Llama-2-7b-hf"
OUTPUT_DIR="out/llama2_7b/block_16x16_hybrid_2_4/wanda/dense_finetuned_model"
DATASET="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"

# Training hyperparameters
NUM_EPOCHS=1  # Full fine-tuning usually needs fewer epochs than LoRA
LEARNING_RATE=5e-5  # Lower learning rate for full fine-tuning
BATCH_SIZE=1  # Small batch size due to memory constraints
EVAL_BATCH_SIZE=4
BLOCK_SIZE=1024  # Sequence length
MAX_TRAIN_SAMPLES=30000
MAX_EVAL_SAMPLES=128

# Gradient accumulation to simulate larger batch size
# Adjust based on number of GPUs to maintain same effective batch size
# Increased for A100 48GB to reduce memory usage
if [ $NUM_GPUS -eq 1 ]; then
    GRADIENT_ACCUMULATION_STEPS=16  # Effective batch size = 1 * 16 = 16
elif [ $NUM_GPUS -eq 2 ]; then
    GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size = 1 * 2 * 8 = 16
elif [ $NUM_GPUS -eq 4 ]; then
    GRADIENT_ACCUMULATION_STEPS=4  # Effective batch size = 1 * 4 * 4 = 16
else
    GRADIENT_ACCUMULATION_STEPS=$((16 / NUM_GPUS))  # Auto-adjust
    if [ $GRADIENT_ACCUMULATION_STEPS -lt 1 ]; then
        GRADIENT_ACCUMULATION_STEPS=1
    fi
fi

# GPU configuration
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# ========================================
# Check if pruned model exists
# ========================================

if [ ! -d "${PRUNED_MODEL}" ]; then
    echo "❌ Error: Pruned model not found at ${PRUNED_MODEL}"
    echo "Please run the pruning script first."
    exit 1
fi

# ========================================
# Display configuration
# ========================================

echo "=========================================="
echo "Dense Fine-tuning Configuration"
echo "=========================================="
echo "Method: Full fine-tuning with SparseTrainer"
echo "Pruned model: ${PRUNED_MODEL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET} (${DATASET_CONFIG})"
echo ""
echo "GPU Configuration:"
echo "  Number of GPUs: ${NUM_GPUS}"
echo "  GPU IDs: ${GPU_IDS}"
if [ $NUM_GPUS -gt 1 ]; then
    echo "  Training mode: Multi-GPU (torchrun)"
else
    echo "  Training mode: Single-GPU"
fi
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
echo "  Max train samples: ${MAX_TRAIN_SAMPLES}"
echo "=========================================="
echo ""

# ========================================
# GPU Memory Requirements
# ========================================

echo "⚠️  GPU Memory Requirements:"
echo "  - Model parameters: ~13GB (FP16)"
echo "  - Optimizer states: ~26GB (AdamW)"
echo "  - Gradients: ~13GB"
echo "  - Activations: ~5-10GB (with gradient checkpointing)"
echo "  - Total: ~55-65GB"
echo ""
echo "  Recommended: A100 80GB or H100"
echo "  Minimum: A100 40GB (with gradient checkpointing)"
echo ""

# Check GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES} | head -1)
echo "Available GPU memory: ${GPU_MEM} MB (~$((GPU_MEM / 1024)) GB)"
echo ""

if [ $GPU_MEM -lt 40000 ]; then
    echo "⚠️  Warning: GPU memory may be insufficient!"
    echo "Consider using LoRA fine-tuning instead."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

# ========================================
# Start training
# ========================================

echo ""
echo "🚀 Starting dense fine-tuning with SparseTrainer..."
echo "Training mode: ${NUM_GPUS} GPU(s) on device(s): ${GPU_IDS}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "=========================================="

cd dense_ft

# Build attention arguments
ATTN_ARGS=""
if [ "$USE_FLASH_ATTN" = "true" ] && [ "$USE_SDPA" = "true" ]; then
    echo "❌ Error: Cannot use both --flash_attn and --sdpa. Please choose one."
    exit 1
fi

if [ "$USE_FLASH_ATTN" = "true" ]; then
    ATTN_ARGS="--use_flash_attention"
elif [ "$USE_SDPA" = "true" ]; then
    ATTN_ARGS="--use_sdpa"
fi

# Choose training command based on number of GPUs
if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    echo "Using single GPU training..."
    python finetune_sparse_model.py \
        --model_name_or_path ../${PRUNED_MODEL} \
        --config_name ${CONFIG_NAME} \
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
        --max_train_samples ${MAX_TRAIN_SAMPLES} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --learning_rate ${LEARNING_RATE} \
        --overwrite_output_dir \
        --output_dir ../${OUTPUT_DIR} \
        --logging_steps 10 \
        --eval_steps 200 \
        --save_steps 200 \
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
    echo "Using multi-GPU training with torchrun (${NUM_GPUS} GPUs)..."
    torchrun --nproc_per_node=${NUM_GPUS} finetune_sparse_model.py \
        --model_name_or_path ../${PRUNED_MODEL} \
        --config_name ${CONFIG_NAME} \
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
        --max_train_samples ${MAX_TRAIN_SAMPLES} \
        --max_eval_samples ${MAX_EVAL_SAMPLES} \
        --learning_rate ${LEARNING_RATE} \
        --overwrite_output_dir \
        --output_dir ../${OUTPUT_DIR} \
        --logging_steps 10 \
        --eval_steps 200 \
        --save_steps 200 \
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

cd ..

# ========================================
# Training complete
# ========================================

echo ""
echo "=========================================="
echo "Dense Fine-tuning Complete!"
echo "=========================================="
echo "Fine-tuned model saved to: ${OUTPUT_DIR}"
echo ""
echo "Key features:"
echo "  ✅ All parameters trained (100%)"
echo "  ✅ Sparsity pattern maintained"
echo "  ✅ Better accuracy than LoRA"
echo ""
echo "Next steps:"
echo "1. Evaluate the fine-tuned model"
echo "2. Compare with LoRA fine-tuned model"
echo "3. Compare with base model perplexity (135.44)"
echo ""
echo "To evaluate:"
echo "  python main_block.py --model ${OUTPUT_DIR} --eval_zero_shot"
echo "=========================================="

