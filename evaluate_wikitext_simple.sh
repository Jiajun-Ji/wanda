#!/bin/bash

# Simple WikiText evaluation script for pruned Llama-2-7b model

# Model path
PRUNED_MODEL="/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model"

# Output directory
OUTPUT_DIR="/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/eval_results"

# CUDA device
export CUDA_VISIBLE_DEVICES=7

echo "=========================================="
echo "Evaluating Pruned Llama-2-7b on WikiText"
echo "=========================================="
echo "Model: ${PRUNED_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Change to lm-evaluation-harness directory
cd /home/jjji/Research/Hybird-Kernel/lm-evaluation-harness

# Check Python environment
echo ""
echo "🔍 Checking Python environment..."
python --version
if python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
    echo "✅ PyTorch is available"
else
    echo "❌ PyTorch not found. Please install: pip install torch"
    exit 1
fi

echo ""
echo "🚀 Starting WikiText evaluation..."

# Run evaluation using python -m to ensure correct interpreter
python -m lm_eval --model hf \
    --model_args pretrained=${PRUNED_MODEL},dtype=float16 \
    --tasks wikitext \
    --device cuda:7 \
    --batch_size auto \
    --output_path ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "✅ Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

