#!/usr/bin/env python3
"""
Compare Unstructured vs Block-wise Pruning
This script compares:
1. Unstructured Wanda pruning (element-wise)
2. Block-wise Wanda pruning (16x16 blocks)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.eval import eval_ppl
from lib.prune import check_sparsity
import argparse
import sys
import os

def evaluate_model(model_path, model_name, args, device):
    """Evaluate a single model and return results"""
    print("\n" + "="*80)
    print(f"Evaluating: {model_name}")
    print("="*80)
    print(f"Model path: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None
    
    # Load model
    print("\n📥 Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True
        )
        model.seqlen = args.seqlen
        model.eval()
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
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
    parser = argparse.ArgumentParser(description='Compare pruning types')
    parser.add_argument('--unstructured_model', type=str,
                        default='out/llama2_7b/unstructured/wanda/pruned_model',
                        help='Path to unstructured pruned model')
    parser.add_argument('--block_model', type=str,
                        default='out/llama2_7b/block_16x16/wanda/pruned_model',
                        help='Path to block pruned model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length for evaluation')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("="*80)
    print("Pruning Type Comparison Evaluation")
    print("="*80)
    print("This script compares:")
    print("1. Unstructured Wanda pruning (element-wise, 50%)")
    print("2. Block-wise Wanda pruning (16x16 blocks, 50%)")
    print("="*80)
    
    # Evaluate models
    results = []
    
    # Unstructured model
    print("\n" + "="*80)
    print("PART 1: Unstructured Pruning")
    print("="*80)
    result = evaluate_model(
        args.unstructured_model,
        "Unstructured Wanda (50%)",
        args,
        device
    )
    if result:
        results.append(result)
    
    # Block model
    print("\n" + "="*80)
    print("PART 2: Block-wise Pruning")
    print("="*80)
    result = evaluate_model(
        args.block_model,
        "Block 16x16 Wanda (50%)",
        args,
        device
    )
    if result:
        results.append(result)
    
    # Print comparison
    if len(results) < 2:
        print("\n❌ Not enough models to compare. Please ensure both models exist.")
        return
    
    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    print(f"{'Pruning Type':<30} {'Sparsity':<15} {'Perplexity':<15} {'PPL Diff':<15}")
    print("-"*80)
    
    unstructured_ppl = results[0]['perplexity']
    
    for i, result in enumerate(results):
        ppl_diff = result['perplexity'] - unstructured_ppl
        ppl_diff_str = f"+{ppl_diff:.4f}" if ppl_diff > 0 else f"{ppl_diff:.4f}"
        
        print(f"{result['name']:<30} "
              f"{result['sparsity']*100:>6.4f}%      "
              f"{result['perplexity']:>8.4f}      "
              f"{ppl_diff_str:>8}")
    
    print("="*80)
    
    # Analysis
    print("\n💡 ANALYSIS")
    print("="*80)
    
    block_ppl = results[1]['perplexity']
    ppl_increase = block_ppl - unstructured_ppl
    relative_increase = (ppl_increase / unstructured_ppl) * 100
    
    print(f"\n1. Perplexity Comparison:")
    print(f"   - Unstructured: {unstructured_ppl:.4f}")
    print(f"   - Block 16x16: {block_ppl:.4f}")
    print(f"   - Difference: {ppl_increase:+.4f} ({relative_increase:+.2f}%)")
    
    print(f"\n2. Sparsity Comparison:")
    print(f"   - Unstructured: {results[0]['sparsity']*100:.4f}%")
    print(f"   - Block 16x16: {results[1]['sparsity']*100:.4f}%")
    
    print(f"\n3. Trade-off Analysis:")
    if abs(relative_increase) < 5:
        print("   ✅ Excellent: Block pruning achieves similar performance")
        print("   → Block-structured sparsity is viable for deployment")
    elif abs(relative_increase) < 10:
        print("   ⚠️  Acceptable: Moderate performance gap")
        print("   → Consider block pruning for hardware acceleration needs")
    else:
        print("   ❌ Significant gap: Block pruning loses considerable accuracy")
        print("   → May need to adjust block size or pruning strategy")
    
    print("\n4. Deployment Considerations:")
    print("   - Unstructured: Better accuracy, harder to accelerate")
    print("   - Block 16x16: Slightly lower accuracy, easier to accelerate")
    print("   - Expected speedup with block sparsity: 1.5-2.0x")
    
    print("-"*80)
    
    # Reference values
    print("\n📖 REFERENCE VALUES:")
    print("-"*80)
    print("Dense Llama-2-7b (Wanda paper):           ~5.12")
    print("50% Unstructured (Wanda paper):           ~6.42")
    print(f"Your 50% Unstructured:                    {unstructured_ppl:.4f}")
    print(f"Your 50% Block 16x16:                     {block_ppl:.4f}")
    print("-"*80)
    
    # Save results
    output_file = "comparison_results.txt"
    with open(output_file, "w") as f:
        f.write("Pruning Type Comparison Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Unstructured Wanda (50%): PPL = {unstructured_ppl:.4f}, Sparsity = {results[0]['sparsity']*100:.4f}%\n")
        f.write(f"Block 16x16 Wanda (50%): PPL = {block_ppl:.4f}, Sparsity = {results[1]['sparsity']*100:.4f}%\n")
        f.write(f"\nPerplexity Difference: {ppl_increase:+.4f} ({relative_increase:+.2f}%)\n")
    
    print(f"\n📄 Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("✅ Comparison completed!")
    print("="*80)

if __name__ == "__main__":
    main()

