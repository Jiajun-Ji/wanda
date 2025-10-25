#!/bin/bash

# LoRA Fine-tuning Script for Pruned Llama-2-7b
# This script performs LoRA fine-tuning on the pruned model to recover accuracy

# ========================================
# Configuration
# ========================================

# Pruned model path
PRUNED_MODEL="/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model"

# LoRA weights output directory
LORA_OUTPUT="/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/lora_weights"

# Model config name
CONFIG_NAME="meta-llama/Llama-2-7b-hf"

# Training dataset
DATASET="c4"  # or "wikitext"

# Training parameters
NUM_EPOCHS=1
BLOCK_SIZE=1024  # Use 2048 if you have 80GB GPU
TRAIN_BATCH_SIZE=1
EVAL_BATCH_SIZE=8
MAX_TRAIN_SAMPLES=30000  # ~12 hours of training
MAX_EVAL_SAMPLES=128
LEARNING_RATE=1e-4

# LoRA parameters
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# CUDA device
export CUDA_VISIBLE_DEVICES=0

# ========================================
# Check if pruned model exists
# ========================================

if [ ! -d "${PRUNED_MODEL}" ]; then
    echo "Error: Pruned model not found at ${PRUNED_MODEL}"
    echo "Please run the pruning script first."
    exit 1
fi

echo "=========================================="
echo "LoRA Fine-tuning for Pruned Llama-2-7b"
echo "=========================================="
echo "Pruned Model: ${PRUNED_MODEL}"
echo "LoRA Output: ${LORA_OUTPUT}"
echo "Dataset: ${DATASET}"
echo "Training Samples: ${MAX_TRAIN_SAMPLES}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "LoRA Rank: ${LORA_R}"
echo "LoRA Alpha: ${LORA_ALPHA}"
echo "=========================================="

# Create output directory
mkdir -p ${LORA_OUTPUT}

# Change to lora_ft directory
cd /home/jjji/Research/Hybird-Kernel/wanda/lora_ft

# ========================================
# Run LoRA Fine-tuning
# ========================================

echo ""
echo "🚀 Starting LoRA fine-tuning..."
echo "=========================================="

python finetune_lm.py \
    --model_name_or_path ${PRUNED_MODEL} \
    --config_name ${CONFIG_NAME} \
    --dataset_name ${DATASET} \
    --num_train_epochs ${NUM_EPOCHS} \
    --block_size ${BLOCK_SIZE} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --do_train \
    --do_eval \
    --max_train_samples ${MAX_TRAIN_SAMPLES} \
    --max_eval_samples ${MAX_EVAL_SAMPLES} \
    --learning_rate ${LEARNING_RATE} \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --overwrite_output_dir \
    --output_dir ${LORA_OUTPUT}

# ========================================
# Evaluate LoRA Fine-tuned Model
# ========================================

echo ""
echo "=========================================="
echo "📊 Evaluating LoRA fine-tuned model..."
echo "=========================================="

python evaluate_ppl.py \
    --model ${PRUNED_MODEL} \
    --lora_weights ${LORA_OUTPUT}

# ========================================
# Summary
# ========================================

echo ""
echo "=========================================="
echo "✅ LoRA fine-tuning completed!"
echo "=========================================="
echo "LoRA weights saved to: ${LORA_OUTPUT}"
echo ""
echo "To use the fine-tuned model:"
echo "  1. Load the pruned model: ${PRUNED_MODEL}"
echo "  2. Load the LoRA weights: ${LORA_OUTPUT}"
echo ""
echo "Example Python code:"
echo "  from transformers import AutoModelForCausalLM"
echo "  from peft import PeftModel"
echo "  model = AutoModelForCausalLM.from_pretrained('${PRUNED_MODEL}')"
echo "  model = PeftModel.from_pretrained(model, '${LORA_OUTPUT}')"
echo "=========================================="

cd -

