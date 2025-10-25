#!/bin/bash

# Evaluation script for additionally pruned Llama-2-7b model (50% -> 50.25%)

# Model path - the additionally pruned model
PRUNED_MODEL="out/llama2_7b/unstructured/wanda_additional_0.5/pruned_model"

# Output directory
OUTPUT_DIR="out/llama2_7b/unstructured/wanda_additional_0.5/eval_results"

# CUDA device
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Evaluating Additionally Pruned Model"
echo "=========================================="
echo "Model: ${PRUNED_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "Sparsity: ~50.25% (50% base + 0.5% additional)"
echo "=========================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if lm-evaluation-harness exists
if [ -d "/home/jjji/Research/Hybird-Kernel/lm-evaluation-harness" ]; then
    echo ""
    echo "🔍 Using lm-evaluation-harness for evaluation..."
    
    # Change to lm-evaluation-harness directory
    cd /home/jjji/Research/Hybird-Kernel/lm-evaluation-harness
    
    # Run evaluation
    python -m lm_eval --model hf \
        --model_args pretrained=/home/jjji/Research/Hybird-Kernel/wanda/${PRUNED_MODEL},dtype=float16 \
        --tasks wikitext \
        --device cuda:0 \
        --batch_size auto \
        --output_path /home/jjji/Research/Hybird-Kernel/wanda/${OUTPUT_DIR}
    
    echo ""
    echo "=========================================="
    echo "✅ Evaluation completed!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "=========================================="
else
    echo ""
    echo "⚠️  lm-evaluation-harness not found."
    echo "Using built-in WikiText evaluation..."
    echo ""
    
    # Use the built-in evaluation from main.py
    cd /home/jjji/Research/Hybird-Kernel/wanda
    
    python main.py \
        --model ${PRUNED_MODEL} \
        --prune_method wanda \
        --sparsity_ratio 0 \
        --save ${OUTPUT_DIR}
    
    echo ""
    echo "=========================================="
    echo "✅ Evaluation completed!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "=========================================="
fi

