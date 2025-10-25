#!/usr/bin/env python3
"""
Evaluate Additionally Pruned Llama-2-7b (50% -> 50.25%)
This script evaluates the model after additional 0.5% pruning of highest-score weights
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.eval import eval_ppl
from lib.prune import check_sparsity
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Evaluate additionally pruned model')
    parser.add_argument('--model', type=str, 
                        default='out/llama2_7b/unstructured/wanda_additional_0.5/pruned_model',
                        help='Path to additionally pruned model')
    parser.add_argument('--base_model', type=str,
                        default='out/llama2_7b/unstructured/wanda/pruned_model',
                        help='Path to base 50% pruned model (for comparison)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length for evaluation')
    parser.add_argument('--compare', action='store_true',
                        help='Also evaluate base model for comparison')
    args = parser.parse_args()
    
    print("="*80)
    print("Evaluating Additionally Pruned Model (50% -> 50.25%)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Sequence Length: {args.seqlen}")
    print("="*80)
    
    # Load additionally pruned model
    print("\n📥 Loading additionally pruned model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
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
    print(f"Model sparsity: {sparsity*100:.4f}%")
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    print("✅ Tokenizer loaded")
    
    # Evaluate additionally pruned model
    print("\n🚀 Starting evaluation on WikiText2...")
    print("This may take a few minutes...")
    
    device = torch.device(args.device)
    ppl_additional = eval_ppl(args, model, tokenizer, device)
    
    print("\n" + "="*80)
    print("📊 Evaluation Results - Additionally Pruned Model")
    print("="*80)
    print(f"Sparsity: {sparsity*100:.4f}%")
    print(f"WikiText2 Perplexity: {ppl_additional:.4f}")
    print("="*80)
    
    # Compare with base model if requested
    if args.compare:
        print("\n" + "="*80)
        print("Evaluating Base 50% Pruned Model for Comparison")
        print("="*80)
        
        # Load base model
        print("\n📥 Loading base 50% pruned model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True
        )
        base_model.seqlen = args.seqlen
        base_model.eval()
        print("✅ Base model loaded")
        
        # Check base sparsity
        print("\n🔍 Checking base sparsity...")
        base_sparsity = check_sparsity(base_model)
        print(f"Base model sparsity: {base_sparsity*100:.4f}%")
        
        # Evaluate base model
        print("\n🚀 Evaluating base model...")
        ppl_base = eval_ppl(args, base_model, tokenizer, device)
        
        print("\n" + "="*80)
        print("📊 Comparison Results")
        print("="*80)
        print(f"Base Model (50% sparse):")
        print(f"  - Sparsity: {base_sparsity*100:.4f}%")
        print(f"  - Perplexity: {ppl_base:.4f}")
        print(f"\nAdditionally Pruned Model (50.25% sparse):")
        print(f"  - Sparsity: {sparsity*100:.4f}%")
        print(f"  - Perplexity: {ppl_additional:.4f}")
        print(f"\nPerformance Change:")
        print(f"  - Sparsity increase: +{(sparsity-base_sparsity)*100:.4f}%")
        print(f"  - Perplexity increase: +{ppl_additional-ppl_base:.4f}")
        print(f"  - Relative degradation: {((ppl_additional/ppl_base)-1)*100:.2f}%")
        print("="*80)
    
    # Reference values
    print("\n📚 Reference Values:")
    print("-"*80)
    print("Dense Llama-2-7b (Wanda paper):           ~5.12")
    print("50% Pruned (Wanda paper):                 ~6.42")
    print(f"Your Base 50% Pruned:                     {ppl_base if args.compare else 'N/A (use --compare)'}")
    print(f"Your Additionally Pruned (50.25%):       {ppl_additional:.4f}")
    print("-"*80)
    
    # Analysis
    print("\n💡 Analysis:")
    print("-"*80)
    if args.compare and ppl_base > 0:
        degradation = ((ppl_additional / ppl_base) - 1) * 100
        print(f"Pruning the top 0.5% highest-score weights caused:")
        print(f"  - {degradation:.2f}% relative performance degradation")
        print(f"  - Absolute PPL increase: {ppl_additional - ppl_base:.4f}")
        
        if degradation < 5:
            print("\n✅ Small degradation: Model is robust to removing important weights")
        elif degradation < 15:
            print("\n⚠️  Moderate degradation: Important weights have noticeable impact")
        else:
            print("\n❌ Large degradation: Important weights are critical for performance")
    else:
        print("Use --compare flag to see detailed comparison with base model")
    print("-"*80)
    
    print("\n" + "="*80)
    print("✅ Evaluation completed!")
    print("="*80)

if __name__ == "__main__":
    main()

