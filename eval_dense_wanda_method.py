#!/usr/bin/env python3
"""
Evaluate Dense Llama-2-7b using Wanda's evaluation method
This allows fair comparison with the pruned model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.eval import eval_ppl
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Evaluate dense model using Wanda method')
    parser.add_argument('--model', type=str, default='/mnt/sdb/llm_models/Llama-2-7b-hf',
                        help='Path to dense model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length for evaluation')
    args = parser.parse_args()
    
    print("="*60)
    print("Evaluating Dense Model with Wanda's Method")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Sequence Length: {args.seqlen}")
    print("="*60)
    
    # Load model
    print("\n📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True
    )
    model.seqlen = args.seqlen
    model.eval()
    print("✅ Model loaded")
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    print("✅ Tokenizer loaded")
    
    # Evaluate
    print("\n🚀 Starting evaluation on WikiText2...")
    print("This may take a few minutes...")
    
    device = torch.device(args.device)
    ppl = eval_ppl(args, model, tokenizer, device)
    
    print("\n" + "="*60)
    print("📊 Evaluation Results")
    print("="*60)
    print(f"WikiText2 Perplexity: {ppl:.4f}")
    print("="*60)
    
    # Compare with reference values
    print("\n📚 Reference Values:")
    print("-"*60)
    print("Wanda Paper (Dense Llama-2-7b):  ~5.12")
    print("Wanda Paper (Pruned 50%):        ~6.42")
    print(f"Your Pruned Model (Wanda eval):   6.31")
    print(f"Your Dense Model (Wanda eval):    {ppl:.4f}")
    print("-"*60)
    
    # Calculate expected pruned performance
    if ppl > 0:
        expected_pruned = ppl * (6.42 / 5.12)  # Scale based on paper ratio
        print(f"\nExpected Pruned PPL (based on paper ratio): {expected_pruned:.4f}")
        print(f"Actual Pruned PPL: 6.31")
        
        if abs(expected_pruned - 6.31) < 1.0:
            print("✅ Your pruning results are consistent with the paper!")
        else:
            print("⚠️  Some deviation from expected values")
    
    print("\n" + "="*60)
    print("✅ Evaluation completed!")
    print("="*60)

if __name__ == "__main__":
    main()

