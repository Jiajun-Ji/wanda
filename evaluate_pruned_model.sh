#!/bin/bash

# Evaluation script for pruned Llama-2-7b model using lm-evaluation-harness
# This script evaluates the pruned model on WikiText and other benchmarks

# ========================================
# Configuration
# ========================================

# Pruned model path (relative to wanda directory)
PRUNED_MODEL_PATH="out/llama2_7b/unstructured/wanda/pruned_model"

# Full path to pruned model
FULL_MODEL_PATH="/home/jjji/Research/Hybird-Kernel/wanda/${PRUNED_MODEL_PATH}"

# lm-evaluation-harness directory
EVAL_HARNESS_DIR="/home/jjji/Research/Hybird-Kernel/lm-evaluation-harness"

# Output directory for evaluation results
OUTPUT_DIR="out/llama2_7b/unstructured/wanda/eval_results"

# CUDA device
export CUDA_VISIBLE_DEVICES=0

# ========================================
# Check if model exists
# ========================================

if [ ! -d "${FULL_MODEL_PATH}" ]; then
    echo "Error: Pruned model not found at ${FULL_MODEL_PATH}"
    echo "Please run the pruning script first."
    exit 1
fi

echo "=========================================="
echo "Evaluating Pruned Llama-2-7b Model"
echo "=========================================="
echo "Model Path: ${FULL_MODEL_PATH}"
echo "Evaluation Harness: ${EVAL_HARNESS_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}

# ========================================
# Evaluation 1: WikiText (Perplexity)
# ========================================

echo ""
echo "📊 Evaluation 1: WikiText Perplexity"
echo "=========================================="

cd ${EVAL_HARNESS_DIR}

lm_eval --model hf \
    --model_args pretrained=${FULL_MODEL_PATH},dtype=float16 \
    --tasks wikitext \
    --device cuda:0 \
    --batch_size auto \
    --output_path ${FULL_MODEL_PATH}/../../eval_results/wikitext

echo "✅ WikiText evaluation completed!"

# ========================================
# Evaluation 2: Common Benchmarks
# ========================================

echo ""
echo "📊 Evaluation 2: Common Benchmarks (HellaSwag, PIQA, WinoGrande, ARC)"
echo "=========================================="

lm_eval --model hf \
    --model_args pretrained=${FULL_MODEL_PATH},dtype=float16 \
    --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge \
    --device cuda:0 \
    --batch_size auto \
    --output_path ${FULL_MODEL_PATH}/../../eval_results/benchmarks

echo "✅ Benchmark evaluation completed!"

# ========================================
# Evaluation 3: Zero-shot Tasks (Optional)
# ========================================

echo ""
echo "📊 Evaluation 3: Zero-shot Tasks (BoolQ, RTE, OpenBookQA)"
echo "=========================================="

lm_eval --model hf \
    --model_args pretrained=${FULL_MODEL_PATH},dtype=float16 \
    --tasks boolq,rte,openbookqa \
    --device cuda:0 \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path ${FULL_MODEL_PATH}/../../eval_results/zeroshot

echo "✅ Zero-shot evaluation completed!"

# ========================================
# Summary
# ========================================

echo ""
echo "=========================================="
echo "✅ All evaluations completed!"
echo "=========================================="
echo "Results saved to: ${FULL_MODEL_PATH}/../../eval_results/"
echo ""
echo "To view results:"
echo "  - WikiText: ${FULL_MODEL_PATH}/../../eval_results/wikitext/"
echo "  - Benchmarks: ${FULL_MODEL_PATH}/../../eval_results/benchmarks/"
echo "  - Zero-shot: ${FULL_MODEL_PATH}/../../eval_results/zeroshot/"
echo "=========================================="

cd -

