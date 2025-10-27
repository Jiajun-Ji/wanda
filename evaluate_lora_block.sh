#!/bin/bash

# Evaluate LoRA fine-tuned block-pruned model

# Configuration
PRUNED_MODEL="out/llama2_7b/block_16x16/wanda/pruned_model"
LORA_WEIGHTS="out/llama2_7b/block_16x16/wanda/lora_weights_wikitext"

# CUDA device
export CUDA_VISIBLE_DEVICES=2

echo "=========================================="
echo "Evaluating LoRA Fine-tuned Model"
echo "=========================================="
echo "Pruned model: ${PRUNED_MODEL}"
echo "LoRA weights: ${LORA_WEIGHTS}"
echo "=========================================="
echo ""

# Check if files exist
if [ ! -d "${PRUNED_MODEL}" ]; then
    echo "❌ Error: Pruned model not found at ${PRUNED_MODEL}"
    exit 1
fi

if [ ! -d "${LORA_WEIGHTS}" ]; then
    echo "❌ Error: LoRA weights not found at ${LORA_WEIGHTS}"
    echo "Please run LoRA fine-tuning first!"
    exit 1
fi

echo "✅ All files found"
echo ""

# Evaluate
cd lora_ft

echo "🚀 Evaluating on WikiText2..."
echo ""

python evaluate_ppl.py \
    --model ../${PRUNED_MODEL} \
    --lora_weights ../${LORA_WEIGHTS}

cd ..

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="

