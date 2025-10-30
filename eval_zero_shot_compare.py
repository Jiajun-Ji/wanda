#!/usr/bin/env python3
"""
Zero-Shot Evaluation Script for Comparing Models
对比原始模型和剪枝微调模型在多个下游任务上的性能

Usage:
    python eval_zero_shot_compare.py \
        --original_model /mnt/sdb/llm_models/Llama-2-7b-hf \
        --pruned_model wanda/out/llama2_7b/block_16x16_three_tier_0.35_0.45_0.2/wanda/dense_finetuned_model \
        --tasks boolq rte hellaswag winogrande arc_easy arc_challenge openbookqa \
        --output_dir eval_results
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

# 添加 lm-evaluation-harness 到 Python 路径
LM_EVAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lm-evaluation-harness')
if os.path.exists(LM_EVAL_PATH) and LM_EVAL_PATH not in sys.path:
    sys.path.insert(0, LM_EVAL_PATH)
    print(f"Added lm-evaluation-harness to path: {LM_EVAL_PATH}")

from lib.eval import eval_zero_shot, eval_ppl
from lib.prune import check_sparsity

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_path, cache_dir=None):
    """加载LLM模型"""
    print(f"\n{'='*80}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*80}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    model.seqlen = model.config.max_position_embeddings
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model type: {model.config.model_type}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Num layers: {model.config.num_hidden_layers}")
    print(f"Sequence length: {model.seqlen}")
    
    return model


def evaluate_model(model_path, model_name, tokenizer, tasks, args):
    """评估单个模型"""
    print(f"\n{'#'*80}")
    print(f"# Evaluating: {model_name}")
    print(f"{'#'*80}\n")
    
    # 加载模型
    model = get_llm(model_path, args.cache_dir)
    
    # 检查稀疏度
    sparsity = check_sparsity(model)
    print(f"\n{'*'*30}")
    print(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"{'*'*30}\n")
    
    # 评估WikiText2困惑度
    device = torch.device("cuda:0")
    if "30b" in model_path or "65b" in model_path or "70b" in model_path:
        device = model.hf_device_map.get("lm_head", device)
    
    print(f"Evaluating WikiText2 perplexity...")
    ppl = eval_ppl(args, model, tokenizer, device)
    print(f"WikiText2 PPL: {ppl:.4f}\n")
    
    # 评估Zero-Shot任务
    print(f"Evaluating Zero-Shot tasks: {tasks}")
    print(f"{'='*80}\n")
    
    use_accelerate = False
    if "30b" in model_path or "65b" in model_path or "70b" in model_path:
        use_accelerate = True
    
    start_time = time.time()
    results = eval_zero_shot(
        model_path, 
        model, 
        tokenizer, 
        task_list=tasks,
        num_fewshot=0,
        use_accelerate=use_accelerate
    )
    eval_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    print(f"{'='*80}\n")
    
    # 整理结果
    result_summary = {
        'model_name': model_name,
        'model_path': model_path,
        'sparsity': float(sparsity),
        'wikitext_ppl': float(ppl),
        'eval_time_seconds': eval_time,
        'tasks': {}
    }
    
    # 提取每个任务的准确率
    for task in tasks:
        if task in results['results']:
            task_result = results['results'][task]
            # 尝试获取准确率指标
            acc = None
            if 'acc' in task_result:
                acc = task_result['acc']
            elif 'acc_norm' in task_result:
                acc = task_result['acc_norm']
            elif 'accuracy' in task_result:
                acc = task_result['accuracy']
            
            result_summary['tasks'][task] = {
                'accuracy': acc,
                'full_results': task_result
            }
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    
    return result_summary, results


def print_comparison_table(original_results, pruned_results, tasks):
    """打印对比表格"""
    print(f"\n{'='*100}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*100}\n")
    
    # 基本信息
    print(f"Original Model: {original_results['model_name']}")
    print(f"  - Sparsity: {original_results['sparsity']*100:.2f}%")
    print(f"  - WikiText2 PPL: {original_results['wikitext_ppl']:.4f}")
    print()
    print(f"Pruned Model: {pruned_results['model_name']}")
    print(f"  - Sparsity: {pruned_results['sparsity']*100:.2f}%")
    print(f"  - WikiText2 PPL: {pruned_results['wikitext_ppl']:.4f}")
    print(f"  - PPL Degradation: {pruned_results['wikitext_ppl'] - original_results['wikitext_ppl']:.4f}")
    print()
    
    # 任务对比表格
    print(f"{'Task':<20} {'Original':<15} {'Pruned':<15} {'Difference':<15} {'Relative':<15}")
    print(f"{'-'*100}")
    
    for task in tasks:
        orig_acc = original_results['tasks'].get(task, {}).get('accuracy')
        prun_acc = pruned_results['tasks'].get(task, {}).get('accuracy')
        
        if orig_acc is not None and prun_acc is not None:
            diff = prun_acc - orig_acc
            relative = (diff / orig_acc * 100) if orig_acc != 0 else 0
            
            print(f"{task:<20} {orig_acc:<15.4f} {prun_acc:<15.4f} {diff:<15.4f} {relative:>+14.2f}%")
        else:
            print(f"{task:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print(f"{'='*100}\n")
    
    # 计算平均性能
    orig_accs = [r['accuracy'] for r in original_results['tasks'].values() if r['accuracy'] is not None]
    prun_accs = [r['accuracy'] for r in pruned_results['tasks'].values() if r['accuracy'] is not None]
    
    if orig_accs and prun_accs:
        avg_orig = sum(orig_accs) / len(orig_accs)
        avg_prun = sum(prun_accs) / len(prun_accs)
        avg_diff = avg_prun - avg_orig
        avg_relative = (avg_diff / avg_orig * 100) if avg_orig != 0 else 0
        
        print(f"{'AVERAGE':<20} {avg_orig:<15.4f} {avg_prun:<15.4f} {avg_diff:<15.4f} {avg_relative:>+14.2f}%")
        print(f"{'='*100}\n")


def save_results(original_results, pruned_results, output_dir):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON格式
    json_path = os.path.join(output_dir, f"comparison_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'original': original_results,
            'pruned': pruned_results,
            'timestamp': timestamp
        }, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # 保存Markdown格式
    md_path = os.path.join(output_dir, f"comparison_{timestamp}.md")
    with open(md_path, 'w') as f:
        f.write(f"# Model Comparison Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Models\n\n")
        f.write(f"- **Original**: `{original_results['model_path']}`\n")
        f.write(f"- **Pruned**: `{pruned_results['model_path']}`\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Original | Pruned | Difference |\n")
        f.write(f"|--------|----------|--------|------------|\n")
        f.write(f"| Sparsity | {original_results['sparsity']*100:.2f}% | {pruned_results['sparsity']*100:.2f}% | {(pruned_results['sparsity']-original_results['sparsity'])*100:.2f}% |\n")
        f.write(f"| WikiText2 PPL | {original_results['wikitext_ppl']:.4f} | {pruned_results['wikitext_ppl']:.4f} | {pruned_results['wikitext_ppl']-original_results['wikitext_ppl']:.4f} |\n\n")
        
        f.write(f"## Task Results\n\n")
        f.write(f"| Task | Original | Pruned | Difference | Relative |\n")
        f.write(f"|------|----------|--------|------------|----------|\n")
        
        for task in original_results['tasks'].keys():
            orig_acc = original_results['tasks'][task].get('accuracy')
            prun_acc = pruned_results['tasks'][task].get('accuracy')
            
            if orig_acc is not None and prun_acc is not None:
                diff = prun_acc - orig_acc
                relative = (diff / orig_acc * 100) if orig_acc != 0 else 0
                f.write(f"| {task} | {orig_acc:.4f} | {prun_acc:.4f} | {diff:+.4f} | {relative:+.2f}% |\n")
        
    print(f"Markdown report saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare original and pruned models on zero-shot tasks')
    
    # 模型路径
    parser.add_argument('--original_model', type=str, 
                        default='/mnt/sdb/llm_models/Llama-2-7b-hf',
                        help='Path to original model')
    parser.add_argument('--pruned_model', type=str,
                        default='out/llama2_7b/block_16x16_three_tier_0.35_0.45_0.2/wanda/dense_finetuned_model',
                        help='Path to pruned/finetuned model')
    
    # 评估任务
    # parser.add_argument('--tasks', nargs='+', 
    #                     default=['boolq', 'rte', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa'],
    #                     help='List of tasks to evaluate')

    parser.add_argument('--tasks', nargs='+', 
                        default=['boolq'],
                        help='List of tasks to evaluate')
    
    # 其他参数
    parser.add_argument('--cache_dir', type=str, default='llm_weights',
                        help='Cache directory for models')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of samples for PPL evaluation')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 加载tokenizer（两个模型共用）
    print(f"\nLoading tokenizer from: {args.original_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.original_model, use_fast=False)
    print(f"Tokenizer loaded successfully!\n")
    
    # 评估原始模型
    original_results, _ = evaluate_model(
        args.original_model,
        "Original Model",
        tokenizer,
        args.tasks,
        args
    )
    
    # 评估剪枝模型
    pruned_results, _ = evaluate_model(
        args.pruned_model,
        "Pruned & Finetuned Model",
        tokenizer,
        args.tasks,
        args
    )
    
    # 打印对比表格
    print_comparison_table(original_results, pruned_results, args.tasks)
    
    # 保存结果
    save_results(original_results, pruned_results, args.output_dir)
    
    print(f"\n{'='*100}")
    print(f"Evaluation completed successfully!")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()

