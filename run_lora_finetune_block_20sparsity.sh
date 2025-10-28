#!/bin/bash

# LoRA Fine-tuning for 20% Sparsity Block-Pruned Model
# This script fine-tunes the 20% sparsity block-pruned model using LoRA

# Configuration
PRUNED_MODEL="out/llama2_7b/block_16x16_20sparsity/wanda/pruned_model"  # Base pruned model
CONFIG_NAME="/mnt/sdb/llm_models/Llama-2-7b-hf"
OUTPUT_DIR="out/llama2_7b/block_16x16_20sparsity/wanda/lora_weights_wikitext_epoch5"
DATASET="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
NUM_EPOCHS=10  # Total 5 epochs
RESUME_CHECKPOINT="out/llama2_7b/block_16x16_20sparsity/wanda/lora_weights_wikitext"  # Resume from epoch 1

# Training hyperparameters
LEARNING_RATE=5e-4
BATCH_SIZE=1
EVAL_BATCH_SIZE=8
BLOCK_SIZE=1024
MAX_TRAIN_SAMPLES=30000
MAX_EVAL_SAMPLES=128

# GPU configuration
export CUDA_VISIBLE_DEVICES=2

echo "=========================================="
echo "LoRA Fine-tuning Configuration"
echo "=========================================="
echo "Pruned model: ${PRUNED_MODEL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET}"
echo "Training samples: ${MAX_TRAIN_SAMPLES}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Block size: ${BLOCK_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================="
echo ""

# Check if pruned model exists
if [ ! -d "${PRUNED_MODEL}" ]; then
    echo "❌ Error: Pruned model not found at ${PRUNED_MODEL}"
    echo "Please run the block pruning script first."
    exit 1
fi

echo "✅ Pruned model found"
echo ""

# Run LoRA fine-tuning
cd lora_ft

echo "🚀 Starting LoRA fine-tuning..."
echo "Expected training time: ~1-2 hours"
echo ""

# Check if resuming from checkpoint
RESUME_ARG=""
if [ -n "${RESUME_CHECKPOINT}" ] && [ -d "../${RESUME_CHECKPOINT}" ]; then
    echo "📂 Resuming from checkpoint: ${RESUME_CHECKPOINT}"
    RESUME_ARG="--resume_from_checkpoint ../${RESUME_CHECKPOINT}"
fi

python finetune_lm.py \
    --model_name_or_path ../${PRUNED_MODEL} \
    --config_name ${CONFIG_NAME} \
    --dataset_name ${DATASET} \
    --dataset_config_name ${DATASET_CONFIG} \
    --num_train_epochs ${NUM_EPOCHS} \
    --block_size ${BLOCK_SIZE} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
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
    ${RESUME_ARG}

cd ..

echo ""
echo "=========================================="
echo "LoRA Fine-tuning Complete!"
echo "=========================================="
echo "LoRA weights saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "1. Evaluate the fine-tuned model"
echo "2. Compare with base model perplexity (135.44)"
echo "=========================================="

