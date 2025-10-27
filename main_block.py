"""
Block-wise Pruning Script for Wanda
This script implements 16x16 block-structured pruning using Wanda scoring method.
"""

import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda_block, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser(description='Block-wise Wanda Pruning')
    parser.add_argument('--model', type=str, required=True, help='LLaMA model path')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Target sparsity level')
    parser.add_argument('--block_size', type=int, default=16, help='Block size for structured pruning')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--eval_zero_shot", action="store_true", help='Evaluate zero-shot performance')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"Loading LLM model: {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]
    print(f"Using device: {device}")

    # Perform block pruning
    if args.sparsity_ratio > 0:
        print("\n" + "="*80)
        print("Starting Block-wise Wanda Pruning")
        print("="*80)
        print(f"Target sparsity: {args.sparsity_ratio*100:.2f}%")
        print(f"Block size: {args.block_size}x{args.block_size}")
        print("="*80 + "\n")
        
        prune_wanda_block(args, model, tokenizer, device, block_size=args.block_size)

    # Check final sparsity
    print("\n" + "*"*80)
    sparsity_ratio = check_sparsity(model)
    print(f"Final sparsity: {sparsity_ratio*100:.4f}%")
    print("*"*80 + "\n")

    # Evaluate perplexity
    print("\n" + "="*80)
    print("Evaluating Perplexity on WikiText2")
    print("="*80)
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"\nWikiText2 Perplexity: {ppl_test:.4f}")
    print("="*80 + "\n")

    # Save results
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_filepath = os.path.join(args.save, f"log_{model_name}.txt")
        with open(save_filepath, "w") as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Pruning method: Wanda Block (block_size={args.block_size})\n")
            f.write(f"Target sparsity: {args.sparsity_ratio*100:.2f}%\n")
            f.write(f"Actual sparsity: {sparsity_ratio*100:.4f}%\n")
            f.write(f"WikiText2 Perplexity: {ppl_test:.4f}\n")
        print(f"Results saved to: {save_filepath}")

    # Evaluate zero-shot performance
    if args.eval_zero_shot:
        print("\n" + "="*80)
        print("Evaluating Zero-shot Performance")
        print("="*80)
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        
        print("\nZero-shot Results:")
        print("-"*80)
        for task, acc in results.items():
            print(f"{task}: {acc:.4f}")
        print("-"*80)
        
        if args.save:
            with open(save_filepath, "a") as f:
                f.write("\nZero-shot Results:\n")
                for task, acc in results.items():
                    f.write(f"{task}: {acc:.4f}\n")

    # Save model
    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"\nPruned model saved to: {args.save_model}")

    print("\n" + "="*80)
    print("Block Pruning Complete!")
    print("="*80)

if __name__ == '__main__':
    main()

