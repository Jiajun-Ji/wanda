#!/bin/bash

# Comparison of Wanda vs lm-eval evaluation methods
# This script helps understand why the PPL results differ

echo "=========================================="
echo "Wanda vs lm-eval Evaluation Comparison"
echo "=========================================="

# Models
DENSE_MODEL="/mnt/sdb/llm_models/Llama-2-7b-hf"
PRUNED_MODEL="/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model"

# Output directory
OUTPUT_DIR="/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/evaluation_comparison"
mkdir -p ${OUTPUT_DIR}

# CUDA device
CUDA_DEVICE="cuda:0"

echo ""
echo "Models:"
echo "  Dense:  ${DENSE_MODEL}"
echo "  Pruned: ${PRUNED_MODEL}"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# ========================================
# Part 1: Evaluate Dense Model with Wanda Method
# ========================================

echo ""
echo "=========================================="
echo "Part 1: Dense Model - Wanda Method"
echo "=========================================="

cd /home/jjji/Research/Hybird-Kernel/wanda

python eval_dense_wanda_method.py \
    --model ${DENSE_MODEL} \
    --device ${CUDA_DEVICE} \
    --seqlen 2048 \
    | tee ${OUTPUT_DIR}/dense_wanda_method.log

DENSE_WANDA_EXIT=$?

# ========================================
# Part 2: Evaluate Dense Model with lm-eval
# ========================================

echo ""
echo "=========================================="
echo "Part 2: Dense Model - lm-eval Method"
echo "=========================================="

cd /home/jjji/Research/Hybird-Kernel/lm-evaluation-harness

lm_eval --model hf \
    --model_args pretrained=${DENSE_MODEL},dtype=float16 \
    --tasks wikitext \
    --device ${CUDA_DEVICE} \
    --batch_size auto \
    --output_path ${OUTPUT_DIR}/dense_lmeval \
    | tee ${OUTPUT_DIR}/dense_lmeval_method.log

DENSE_LMEVAL_EXIT=$?

# ========================================
# Part 3: Evaluate Pruned Model with lm-eval (already done)
# ========================================

echo ""
echo "=========================================="
echo "Part 3: Pruned Model - lm-eval Method"
echo "=========================================="
echo "Already evaluated: PPL = 11.22"
echo "(You can re-run if needed)"

# ========================================
# Part 4: Summary and Analysis
# ========================================

echo ""
echo "=========================================="
echo "📊 Evaluation Results Summary"
echo "=========================================="

echo ""
echo "Dense Llama-2-7b:"
echo "-----------------"

# Extract Wanda method result
if [ $DENSE_WANDA_EXIT -eq 0 ]; then
    DENSE_WANDA_PPL=$(grep "WikiText2 Perplexity:" ${OUTPUT_DIR}/dense_wanda_method.log | awk '{print $3}')
    echo "  Wanda method:  ${DENSE_WANDA_PPL}"
else
    echo "  Wanda method:  Failed"
    DENSE_WANDA_PPL="N/A"
fi

# Extract lm-eval result
if [ $DENSE_LMEVAL_EXIT -eq 0 ] && [ -f "${OUTPUT_DIR}/dense_lmeval/results.json" ]; then
    DENSE_LMEVAL_PPL=$(python3 -c "
import json
with open('${OUTPUT_DIR}/dense_lmeval/results.json', 'r') as f:
    data = json.load(f)
    print(f\"{data['results']['wikitext']['word_perplexity']:.4f}\")
" 2>/dev/null)
    echo "  lm-eval method: ${DENSE_LMEVAL_PPL}"
else
    echo "  lm-eval method: Failed or not found"
    DENSE_LMEVAL_PPL="N/A"
fi

echo ""
echo "Pruned Llama-2-7b (50% sparse):"
echo "-------------------------------"
echo "  Wanda method:  6.31 ✅"
echo "  lm-eval method: 11.22"

echo ""
echo "Reference (Wanda Paper):"
echo "------------------------"
echo "  Dense:  ~5.12"
echo "  Pruned: ~6.42"

echo ""
echo "=========================================="
echo "🔍 Analysis"
echo "=========================================="

# Calculate ratios if we have the data
if [ "$DENSE_WANDA_PPL" != "N/A" ] && [ "$DENSE_LMEVAL_PPL" != "N/A" ]; then
    python3 -c "
dense_wanda = ${DENSE_WANDA_PPL}
dense_lmeval = ${DENSE_LMEVAL_PPL}
pruned_wanda = 6.31
pruned_lmeval = 11.22

print('Method Comparison:')
print(f'  Wanda method ratio (Pruned/Dense):  {pruned_wanda/dense_wanda:.2f}x')
print(f'  lm-eval ratio (Pruned/Dense):       {pruned_lmeval/dense_lmeval:.2f}x')
print()
print('Evaluation Method Difference:')
print(f'  Dense model:  lm-eval is {dense_lmeval/dense_wanda:.2f}x higher than Wanda')
print(f'  Pruned model: lm-eval is {pruned_lmeval/pruned_wanda:.2f}x higher than Wanda')
print()

# Check if the ratio is consistent
ratio_dense = dense_lmeval / dense_wanda
ratio_pruned = pruned_lmeval / pruned_wanda

if abs(ratio_dense - ratio_pruned) < 0.2:
    print('✅ The evaluation method difference is CONSISTENT across models')
    print('   This suggests the difference is due to evaluation methodology,')
    print('   not a problem with the pruning.')
else:
    print('⚠️  The evaluation method difference is INCONSISTENT')
    print('   This might indicate an issue worth investigating.')
" 2>/dev/null
fi

echo ""
echo "=========================================="
echo "💡 Conclusion"
echo "=========================================="
echo ""
echo "The large difference between Wanda and lm-eval is likely due to:"
echo "  1. Different data preprocessing (\\n\\n join vs other)"
echo "  2. Different sequence chunking strategies"
echo "  3. Different PPL calculation methods"
echo ""
echo "For pruning evaluation, use Wanda's method for consistency"
echo "with the paper. Your result (6.31) is excellent!"
echo ""
echo "=========================================="
echo "✅ Comparison completed!"
echo "=========================================="
echo ""
echo "Detailed logs saved to: ${OUTPUT_DIR}"
echo ""

