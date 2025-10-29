#!/usr/bin/env python3
"""
Evaluate perplexity on WikiText-2 for a fine-tuned model.

Usage:
    python eval_perplexity.py --model_path <path_to_model>

Example:
    python eval_perplexity.py --model_path out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.data import get_loaders
import time


def eval_ppl_wikitext(model, testloader, device):
    """Evaluate perplexity on WikiText-2 test set."""
    model.eval()
    
    nlls = []
    n_samples = len(testloader)
    
    print(f"Evaluating on {n_samples} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            if i % 50 == 0:
                print(f"Processing sample {i}/{n_samples}")
            
            # Move batch to device
            input_ids = batch[0].to(device)
            
            # Forward pass
            outputs = model(input_ids, labels=input_ids)
            neg_log_likelihood = outputs.loss
            
            nlls.append(neg_log_likelihood)
    
    # Calculate perplexity
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity on WikiText-2")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)"
    )
    
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    # Add seqlen attribute for compatibility with get_loaders
    model.seqlen = args.seqlen
    
    # Get WikiText-2 test loader
    print("\nLoading WikiText-2 test set...")
    _, testloader = get_loaders(
        "wikitext2",
        seed=0,
        seqlen=args.seqlen,
        tokenizer=tokenizer
    )
    
    # Evaluate perplexity
    print("\n" + "="*50)
    print("Evaluating perplexity on WikiText-2...")
    print("="*50)
    
    start_time = time.time()
    ppl = eval_ppl_wikitext(model, testloader, device)
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Dataset: WikiText-2 (test set)")
    print(f"Perplexity: {ppl:.4f}")
    print(f"Evaluation time: {elapsed_time:.2f} seconds")
    print("="*50)
    
    # Save results to file
    output_file = f"{args.model_path}/wikitext2_perplexity.txt"
    with open(output_file, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: WikiText-2 (test set)\n")
        f.write(f"Perplexity: {ppl:.4f}\n")
        f.write(f"Evaluation time: {elapsed_time:.2f} seconds\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

