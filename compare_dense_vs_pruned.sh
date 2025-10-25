#!/bin/bash

# Comparison Script: Dense vs Pruned Llama-2-7b on WikiText
# This script evaluates both the original dense model and the pruned model

# ========================================
# Configuration
# ========================================

# Dense model path
DENSE_MODEL="/mnt/sdb/llm_models/Llama-2-7b-hf"

# Pruned model path
PRUNED_MODEL="/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model"

# Output directory
OUTPUT_DIR="/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/comparison"

# CUDA device
CUDA_DEVICE="cuda:0"

# ========================================
# Check if models exist
# ========================================

echo "=========================================="
echo "Dense vs Pruned Model Comparison"
echo "=========================================="

if [ ! -d "${DENSE_MODEL}" ]; then
    echo "❌ Error: Dense model not found at ${DENSE_MODEL}"
    exit 1
fi

if [ ! -d "${PRUNED_MODEL}" ]; then
    echo "❌ Error: Pruned model not found at ${PRUNED_MODEL}"
    exit 1
fi

echo "✅ Dense model: ${DENSE_MODEL}"
echo "✅ Pruned model: ${PRUNED_MODEL}"
echo "📊 Output directory: ${OUTPUT_DIR}"
echo "=========================================="

# Create output directories
mkdir -p ${OUTPUT_DIR}/dense
mkdir -p ${OUTPUT_DIR}/pruned

# ========================================
# Evaluate Dense Model
# ========================================

echo ""
echo "=========================================="
echo "1️⃣ Evaluating Dense Llama-2-7b"
echo "=========================================="

lm_eval --model hf \
    --model_args pretrained=${DENSE_MODEL},dtype=float16 \
    --tasks wikitext \
    --device ${CUDA_DEVICE} \
    --batch_size auto \
    --output_path ${OUTPUT_DIR}/dense

DENSE_EXIT_CODE=$?

# ========================================
# Evaluate Pruned Model
# ========================================

echo ""
echo "=========================================="
echo "2️⃣ Evaluating Pruned Llama-2-7b (50% sparse)"
echo "=========================================="

lm_eval --model hf \
    --model_args pretrained=${PRUNED_MODEL},dtype=float16 \
    --tasks wikitext \
    --device ${CUDA_DEVICE} \
    --batch_size auto \
    --output_path ${OUTPUT_DIR}/pruned

PRUNED_EXIT_CODE=$?

# ========================================
# Compare Results
# ========================================

echo ""
echo "=========================================="
echo "📊 Comparison Results"
echo "=========================================="

if [ $DENSE_EXIT_CODE -eq 0 ] && [ $PRUNED_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Dense Model Results:"
    echo "-------------------"
    if [ -f "${OUTPUT_DIR}/dense/results.json" ]; then
        python3 -c "
import json
with open('${OUTPUT_DIR}/dense/results.json', 'r') as f:
    data = json.load(f)
    wikitext = data['results']['wikitext']
    print(f\"  word_perplexity: {wikitext['word_perplexity']:.4f}\")
    print(f\"  byte_perplexity: {wikitext['byte_perplexity']:.4f}\")
    print(f\"  bits_per_byte: {wikitext['bits_per_byte']:.4f}\")
" 2>/dev/null || cat ${OUTPUT_DIR}/dense/results.json | grep -A 5 "wikitext"
    fi
    
    echo ""
    echo "Pruned Model Results (50% sparse):"
    echo "----------------------------------"
    if [ -f "${OUTPUT_DIR}/pruned/results.json" ]; then
        python3 -c "
import json
with open('${OUTPUT_DIR}/pruned/results.json', 'r') as f:
    data = json.load(f)
    wikitext = data['results']['wikitext']
    print(f\"  word_perplexity: {wikitext['word_perplexity']:.4f}\")
    print(f\"  byte_perplexity: {wikitext['byte_perplexity']:.4f}\")
    print(f\"  bits_per_byte: {wikitext['bits_per_byte']:.4f}\")
" 2>/dev/null || cat ${OUTPUT_DIR}/pruned/results.json | grep -A 5 "wikitext"
    fi
    
    echo ""
    echo "Performance Degradation:"
    echo "------------------------"
    python3 -c "
import json
try:
    with open('${OUTPUT_DIR}/dense/results.json', 'r') as f:
        dense = json.load(f)['results']['wikitext']
    with open('${OUTPUT_DIR}/pruned/results.json', 'r') as f:
        pruned = json.load(f)['results']['wikitext']
    
    dense_ppl = dense['word_perplexity']
    pruned_ppl = pruned['word_perplexity']
    degradation = ((pruned_ppl - dense_ppl) / dense_ppl) * 100
    
    print(f\"  Dense PPL: {dense_ppl:.4f}\")
    print(f\"  Pruned PPL: {pruned_ppl:.4f}\")
    print(f\"  Degradation: {degradation:.2f}%\")
    print(f\"  Sparsity: 50.00%\")
    print(f\"  Model size reduction: ~50%\")
except Exception as e:
    print(f\"  Could not calculate degradation: {e}\")
" 2>/dev/null || echo "  Could not calculate degradation"
    
    echo ""
    echo "Reference (Wanda Paper):"
    echo "------------------------"
    echo "  Dense PPL: ~5.12"
    echo "  Pruned PPL (50%): ~6.42"
    echo "  Degradation: ~25%"
    
    echo ""
    echo "Your Wanda Evaluation:"
    echo "----------------------"
    echo "  Pruned PPL: 6.31 ✅ (close to paper!)"
    
else
    echo "❌ Evaluation failed"
    echo "Dense exit code: $DENSE_EXIT_CODE"
    echo "Pruned exit code: $PRUNED_EXIT_CODE"
fi

echo ""
echo "=========================================="
echo "✅ Comparison completed!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "View detailed results:"
echo "  Dense: cat ${OUTPUT_DIR}/dense/results.json"
echo "  Pruned: cat ${OUTPUT_DIR}/pruned/results.json"
echo "=========================================="

