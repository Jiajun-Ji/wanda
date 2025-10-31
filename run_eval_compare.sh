#!/bin/bash

# 对比原始模型和剪枝微调模型的Zero-Shot性能
# Usage: bash run_eval_compare.sh

# 设置路径
ORIGINAL_MODEL="/home/jjji/Research/Hybird-Kernel/wanda/out/progressive_three_tier/iter5/finetuned_model"
PRUNED_MODEL="/home/jjji/Research/Hybird-Kernel/wanda/out/progressive_three_tier/iter5/dense_finetuned_model"

# 评估任务列表
TASKS="boolq rte hellaswag winogrande arc_easy arc_challenge openbookqa"

# 输出目录
OUTPUT_DIR="eval_results_three_tier_finetuned_vs_dense_finetuned"

# 其他参数
NSAMPLES=128
SEED=0

echo "=========================================="
echo "Zero-Shot Evaluation Comparison"
echo "=========================================="
echo ""
echo "Original Model: $ORIGINAL_MODEL"
echo "Pruned Model:   $PRUNED_MODEL"
echo "Tasks:          $TASKS"
echo "Output Dir:     $OUTPUT_DIR"
echo ""
echo "=========================================="
echo ""

# 运行评估
CUDA_VISIBLE_DEVICES=1 python eval_zero_shot_compare.py \
    --original_model "$ORIGINAL_MODEL" \
    --pruned_model "$PRUNED_MODEL" \
    --tasks $TASKS \
    --output_dir "$OUTPUT_DIR" \
    --nsamples $NSAMPLES \
    --seed $SEED

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

