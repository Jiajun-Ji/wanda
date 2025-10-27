#!/usr/bin/env python3
"""
Compare Three Pruning Strategies:
1. Base 50% pruned model
2. Additional 0.5% pruning with HIGHEST scores (most important weights)
3. Additional 0.5% pruning with LOWEST scores (least important weights)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.eval import eval_ppl
from lib.prune import check_sparsity
import argparse
import sys

def evaluate_model(model_path, model_name, args, device):
    """Evaluate a single model and return results"""
    print("\n" + "="*80)
    print(f"Evaluating: {model_name}")
    print("="*80)
    print(f"Model path: {model_path}")
    
    # Load model
    print("\n📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    model.seqlen = args.seqlen
    model.eval()
    print("✅ Model loaded")
    
    # Check sparsity
    print("\n🔍 Checking sparsity...")
    sparsity = check_sparsity(model)
    print(f"Sparsity: {sparsity*100:.4f}%")
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print("✅ Tokenizer loaded")
    
    # Evaluate
    print("\n🚀 Evaluating on WikiText2...")
    ppl = eval_ppl(args, model, tokenizer, device)
    print(f"✅ Perplexity: {ppl:.4f}")
    
    return {
        'name': model_name,
        'path': model_path,
        'sparsity': sparsity,
        'perplexity': ppl
    }

def main():
    parser = argparse.ArgumentParser(description='Compare pruning strategies')
    parser.add_argument('--base_model', type=str,
                        default='out/llama2_7b/unstructured/wanda/pruned_model',
                        help='Path to base 50%% pruned model')
    parser.add_argument('--highest_model', type=str,
                        default='out/llama2_7b/unstructured/wanda_additional_highest_0.5/pruned_model',
                        help='Path to model with highest 0.5%% pruned')
    parser.add_argument('--lowest_model', type=str,
                        default='out/llama2_7b/unstructured/wanda_additional_lowest_0.5/pruned_model',
                        help='Path to model with lowest 0.5%% pruned')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length for evaluation')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("="*80)
    print("Pruning Strategy Comparison Evaluation")
    print("="*80)
    print("This script compares three models:")
    print("1. Base: 50% pruned with Wanda")
    print("2. Highest: Base + 0.5% highest-score weights pruned (most important)")
    print("3. Lowest: Base + 0.5% lowest-score weights pruned (least important)")
    print("="*80)
    
    # Evaluate all three models
    results = []
    
    # Base model
    results.append(evaluate_model(
        args.base_model,
        "Base (50% Wanda)",
        args,
        device
    ))
    
    # Highest model
    results.append(evaluate_model(
        args.highest_model,
        "Highest 0.5% Pruned",
        args,
        device
    ))
    
    # Lowest model
    results.append(evaluate_model(
        args.lowest_model,
        "Lowest 0.5% Pruned",
        args,
        device
    ))
    
    # Print comparison table
    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<30} {'Sparsity':<15} {'Perplexity':<15} {'PPL Change':<15}")
    print("-"*80)
    
    base_ppl = results[0]['perplexity']
    
    for result in results:
        ppl_change = result['perplexity'] - base_ppl
        ppl_change_str = f"+{ppl_change:.4f}" if ppl_change > 0 else f"{ppl_change:.4f}"
        
        print(f"{result['name']:<30} "
              f"{result['sparsity']*100:>6.4f}%      "
              f"{result['perplexity']:>8.4f}      "
              f"{ppl_change_str:>8}")
    
    print("="*80)
    
    # Analysis
    print("\n💡 ANALYSIS")
    print("="*80)
    
    highest_ppl = results[1]['perplexity']
    lowest_ppl = results[2]['perplexity']
    
    highest_degradation = ((highest_ppl / base_ppl) - 1) * 100
    lowest_degradation = ((lowest_ppl / base_ppl) - 1) * 100
    
    print(f"\n1. Pruning HIGHEST 0.5% (most important weights):")
    print(f"   - Perplexity change: {highest_ppl - base_ppl:+.4f}")
    print(f"   - Relative degradation: {highest_degradation:+.2f}%")
    
    print(f"\n2. Pruning LOWEST 0.5% (least important weights):")
    print(f"   - Perplexity change: {lowest_ppl - base_ppl:+.4f}")
    print(f"   - Relative degradation: {lowest_degradation:+.2f}%")
    
    print(f"\n3. Comparison:")
    print(f"   - Highest vs Lowest PPL difference: {highest_ppl - lowest_ppl:.4f}")
    print(f"   - Impact ratio: {abs(highest_degradation / lowest_degradation):.2f}x")
    
    # Interpretation
    print("\n📚 INTERPRETATION:")
    print("-"*80)
    
    if abs(highest_degradation) > abs(lowest_degradation) * 2:
        print("✅ Clear distinction: Important weights have much stronger impact")
        print("   → Wanda scoring effectively identifies critical weights")
    elif abs(highest_degradation) > abs(lowest_degradation):
        print("⚠️  Moderate distinction: Important weights have stronger impact")
        print("   → Wanda scoring shows some effectiveness")
    else:
        print("❌ Weak distinction: Similar impact from both strategies")
        print("   → May need to investigate Wanda scoring or model characteristics")
    
    print("-"*80)
    
    # Reference values
    print("\n📖 REFERENCE VALUES:")
    print("-"*80)
    print("Dense Llama-2-7b (Wanda paper):     ~5.12")
    print("50% Pruned (Wanda paper):           ~6.42")
    print(f"Your Base 50% Pruned:               {base_ppl:.4f}")
    print(f"Your Highest 0.5% Pruned:           {highest_ppl:.4f}")
    print(f"Your Lowest 0.5% Pruned:            {lowest_ppl:.4f}")
    print("-"*80)
    
    print("\n" + "="*80)
    print("✅ Comparison completed!")
    print("="*80)

if __name__ == "__main__":
    main()

