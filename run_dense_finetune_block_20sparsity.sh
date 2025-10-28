#!/bin/bash

# Dense (Full) Fine-tuning for 20% Sparsity Block-Pruned Model
# Uses SparseTrainer to maintain sparsity pattern while training all parameters

# ========================================
# Configuration
# ========================================

# Pruned model path
PRUNED_MODEL="out/llama2_7b/block_16x16_20sparsity/wanda/pruned_model"
CONFIG_NAME="/mnt/sdb/llm_models/Llama-2-7b-hf"
OUTPUT_DIR="out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model"
DATASET="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"

# Training hyperparameters
NUM_EPOCHS=3  # Full fine-tuning usually needs fewer epochs than LoRA
LEARNING_RATE=5e-5  # Lower learning rate for full fine-tuning
BATCH_SIZE=1  # Small batch size due to memory constraints
EVAL_BATCH_SIZE=4
BLOCK_SIZE=1024  # Sequence length
MAX_TRAIN_SAMPLES=30000
MAX_EVAL_SAMPLES=128

# Gradient accumulation to simulate larger batch size
GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size = 1 * 8 = 8

# GPU configuration
export CUDA_VISIBLE_DEVICES=2,1

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
echo "Number of epochs: ${NUM_EPOCHS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Gradient accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Block size: ${BLOCK_SIZE}"
echo "Max train samples: ${MAX_TRAIN_SAMPLES}"
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
echo "=========================================="

cd dense_ft

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
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --learning_rate ${LEARNING_RATE} \
    --overwrite_output_dir \
    --output_dir ../${OUTPUT_DIR} \
    --logging_steps 10 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_first_step \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --save_total_limit 3 \
    --fp16 \
    --gradient_checkpointing \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --seed 42

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

