#!/bin/bash

# SQuAD & GSM8K Evaluation Script
# 对比原始模型和剪枝微调模型在 SQuAD 和 GSM8K 任务上的性能

# ============================================
# 配置
# ============================================

# 模型路径
ORIGINAL_MODEL="/mnt/sdb/llm_models/Llama-2-7b-hf"
PRUNED_MODEL="out/llama2_7b/block_16x16_three_tier_0.35_0.45_0.2/wanda/dense_finetuned_model"

# 评估任务
TASKS="gsm8k"

# 输出目录
OUTPUT_DIR="eval_results_squad_gsm8k"

# ============================================
# 环境检查
# ============================================

echo "=========================================="
echo "SQuAD & GSM8K Evaluation"
echo "=========================================="
echo ""
echo "Original model: ${ORIGINAL_MODEL}"
echo "Pruned model: ${PRUNED_MODEL}"
echo "Tasks: ${TASKS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "⚠️  WARNING: This evaluation will take a long time!"
echo "   - SQuAD v2: ~11,873 samples (generation task)"
echo "   - GSM8K: ~1,319 samples (math reasoning)"
echo "   - Estimated time: 2-4 hours per model"
echo "   - Total time: 4-8 hours for both models"
echo ""
echo "=========================================="
echo ""

# 询问用户是否继续
read -p "Do you want to continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Evaluation cancelled."
    exit 1
fi

# ============================================
# 运行评估
# ============================================

echo ""
echo "Starting evaluation..."
echo "=========================================="
echo ""

# 切换到 wanda 目录
cd /home/jjji/Research/Hybird-Kernel/wanda

# 运行评估脚本
CUDA_VISIBLE_DEVICES=7 python eval_squad_gsm8k_compare.py \
    --original_model ${ORIGINAL_MODEL} \
    --pruned_model ${PRUNED_MODEL} \
    --tasks ${TASKS} \
    --output_dir ${OUTPUT_DIR}

# ============================================
# 检查结果
# ============================================

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "=========================================="
    echo ""
    
    # 显示结果文件
    echo "Generated files:"
    ls -lh ${OUTPUT_DIR}/
    echo ""
    
    # 如果有 Markdown 报告，显示摘要
    if [ -f "${OUTPUT_DIR}/comparison_report.md" ]; then
        echo "Comparison Report:"
        echo "=========================================="
        head -n 50 ${OUTPUT_DIR}/comparison_report.md
        echo ""
        echo "Full report: ${OUTPUT_DIR}/comparison_report.md"
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Evaluation failed!"
    echo "=========================================="
    exit 1
fi

