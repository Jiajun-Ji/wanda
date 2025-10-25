"""
Additional Pruning Script for Pre-pruned Sparse Models
This script performs additional pruning on an already pruned sparse model using Wanda method.
It finds the top 0.5% highest-score weights among remaining non-zero weights and prunes them.

Author: Jiajun Ji
Date: 2025-10-25
"""

import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.prune import find_layers, check_sparsity, prepare_calibration_input
from lib.layerwrapper import WrappedGPT
from lib.data import get_loaders
from lib.eval import eval_ppl


def prune_additional_wanda(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Perform additional pruning on a pre-pruned sparse model using Wanda method.
    
    Args:
        args: Command line arguments
        model: Pre-pruned sparse model
        tokenizer: Tokenizer for the model
        device: Device to run on
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("=" * 80)
    print("Loading calibration data (WikiText2)...")
    print("=" * 80)
    import time
    start_time = time.time()
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer
    )
    load_time = time.time() - start_time
    print(f"Dataset loading complete (took {load_time:.2f}s)\n")
    
    # Prepare calibration inputs
    print("Preparing calibration inputs...")
    start_time = time.time()
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device
        )
    prep_time = time.time() - start_time
    print(f"Calibration inputs prepared (took {prep_time:.2f}s)\n")

    layers = model.model.layers
    
    # Statistics for tracking
    total_params = 0
    total_nonzero_before = 0
    total_pruned_additional = 0
    
    print("=" * 80)
    print(f"Starting Additional Pruning Process - Strategy: {args.prune_strategy.upper()}")
    print("=" * 80)
    if args.prune_strategy == 'highest':
        print("Strategy: Pruning weights with HIGHEST Wanda scores (most important)")
    else:
        print("Strategy: Pruning weights with LOWEST Wanda scores (least important)")
    print("=" * 80)
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # Handle multi-GPU case
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attention_mask = attention_mask.to(dev)
            position_ids = position_ids.to(dev)

        # Wrap layers to collect activation statistics
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        # Register hooks to collect activations
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # Forward pass to collect activation statistics
        print(f"Layer {i}: Collecting activation statistics ({args.nsamples} samples)...")
        for j in range(args.nsamples):
            if j % 32 == 0:
                print(f"  Processing sample {j}/{args.nsamples}...", end='\r')
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]
        print(f"  Processed all {args.nsamples} samples" + " " * 20)
        
        # Remove hooks
        for h in handles:
            h.remove()

        # Perform additional pruning on each sublayer
        print(f"\n{'='*80}")
        print(f"Processing Layer {i}")
        print(f"{'='*80}")
        
        for name in subset:
            W = subset[name].weight.data
            
            # Count current non-zero weights
            nonzero_mask = (W != 0)
            num_nonzero = nonzero_mask.sum().item()
            num_total = W.numel()
            
            total_params += num_total
            total_nonzero_before += num_nonzero
            
            if num_nonzero == 0:
                print(f"  {name}: All weights are zero, skipping...")
                continue
            
            # Calculate Wanda score: |W| * sqrt(activation_norm)
            W_metric = torch.abs(W) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )
            
            # Only consider non-zero weights
            W_metric_nonzero = W_metric[nonzero_mask]
            
            # Calculate number of weights to prune (0.5% of non-zero weights)
            num_to_prune = int(num_nonzero * args.additional_sparsity_ratio)
            
            if num_to_prune == 0:
                print(f"  {name}: Too few non-zero weights to prune, skipping...")
                continue
            
            # Find the threshold based on pruning strategy
            if args.prune_strategy == 'highest':
                # Prune weights with HIGHEST scores (most important)
                threshold_value = torch.topk(
                    W_metric_nonzero,
                    num_to_prune,
                    largest=True
                )[0][-1]
                # Prune weights that: (1) are non-zero AND (2) have score >= threshold
                additional_prune_mask = (W_metric >= threshold_value) & nonzero_mask
            else:  # lowest
                # Prune weights with LOWEST scores (least important)
                threshold_value = torch.topk(
                    W_metric_nonzero,
                    num_to_prune,
                    largest=False
                )[0][-1]
                # Prune weights that: (1) are non-zero AND (2) have score <= threshold
                additional_prune_mask = (W_metric <= threshold_value) & nonzero_mask
            
            # Apply pruning
            W[additional_prune_mask] = 0
            
            # Count newly pruned weights
            num_pruned = additional_prune_mask.sum().item()
            total_pruned_additional += num_pruned
            
            # Calculate new sparsity for this layer
            new_nonzero = (W != 0).sum().item()
            layer_sparsity = 1.0 - (new_nonzero / num_total)
            
            print(f"  {name}:")
            print(f"    - Total params: {num_total:,}")
            print(f"    - Non-zero before: {num_nonzero:,} ({100*(num_nonzero/num_total):.2f}%)")
            print(f"    - Additional pruned: {num_pruned:,} ({100*(num_pruned/num_nonzero):.2f}% of non-zero)")
            print(f"    - Non-zero after: {new_nonzero:,} ({100*(new_nonzero/num_total):.2f}%)")
            print(f"    - Layer sparsity: {layer_sparsity*100:.4f}%")

        # Update inputs for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0), 
                    attention_mask=attention_mask, 
                    position_ids=position_ids
                )[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Additional Pruning Summary")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Non-zero before additional pruning: {total_nonzero_before:,} ({100*(total_nonzero_before/total_params):.4f}%)")
    print(f"Additional pruned: {total_pruned_additional:,} ({100*(total_pruned_additional/total_nonzero_before):.4f}% of non-zero)")
    print(f"Non-zero after: {total_nonzero_before - total_pruned_additional:,}")
    print(f"Final sparsity: {100*(1 - (total_nonzero_before - total_pruned_additional)/total_params):.4f}%")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Additional pruning on pre-pruned sparse models using Wanda method"
    )
    
    # Model arguments
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to the pre-pruned model directory'
    )
    parser.add_argument(
        '--cache_dir', 
        default="llm_weights", 
        type=str,
        help='Directory for caching model weights'
    )
    
    # Pruning arguments
    parser.add_argument(
        '--additional_sparsity_ratio',
        type=float,
        default=0.005,
        help='Additional sparsity ratio (default: 0.005 = 0.5%% of non-zero weights)'
    )
    parser.add_argument(
        '--prune_strategy',
        type=str,
        default='highest',
        choices=['highest', 'lowest'],
        help='Pruning strategy: "highest" prunes weights with highest Wanda scores (most important), '
             '"lowest" prunes weights with lowest Wanda scores (least important)'
    )
    
    # Calibration arguments
    parser.add_argument(
        '--seed', 
        type=int, 
        default=0, 
        help='Random seed for sampling calibration data'
    )
    parser.add_argument(
        '--nsamples', 
        type=int, 
        default=128, 
        help='Number of calibration samples'
    )
    
    # Output arguments
    parser.add_argument(
        '--save', 
        type=str, 
        default=None,
        help='Directory to save results (logs)'
    )
    parser.add_argument(
        '--save_model', 
        type=str, 
        default=None,
        help='Directory to save the additionally pruned model'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--eval_zero_shot', 
        action='store_true',
        help='Evaluate zero-shot performance after pruning'
    )
    
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 80)
    print("Additional Pruning Configuration")
    print("=" * 80)
    print(f"Model path: {args.model}")
    print(f"Additional sparsity ratio: {args.additional_sparsity_ratio*100}% of non-zero weights")
    print(f"Calibration samples: {args.nsamples}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {device}")
    print("=" * 80 + "\n")

    # Load pre-pruned model
    print("Loading pre-pruned model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.seqlen = 2048  # Set sequence length
    print("Model loaded successfully\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    print("Tokenizer loaded successfully\n")

    # Check initial sparsity
    print("Checking initial sparsity...")
    initial_sparsity = check_sparsity(model)
    print(f"\nInitial sparsity: {initial_sparsity*100:.4f}%\n")

    # Perform additional pruning
    prune_additional_wanda(args, model, tokenizer, device)

    # Check final sparsity
    print("\nChecking final sparsity...")
    final_sparsity = check_sparsity(model)
    print(f"\nFinal sparsity: {final_sparsity*100:.4f}%\n")

    # Evaluate perplexity
    print("Evaluating WikiText perplexity...")
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"\nWikiText Perplexity: {ppl_test:.4f}\n")

    # Save results
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        save_filepath = os.path.join(args.save, "log_additional_prune.txt")
        with open(save_filepath, "w") as f:
            print("Configuration", file=f)
            print(f"Model: {args.model}", file=f)
            print(f"Additional sparsity ratio: {args.additional_sparsity_ratio}", file=f)
            print(f"Calibration samples: {args.nsamples}", file=f)
            print(f"Random seed: {args.seed}", file=f)
            print("", file=f)
            print("Results", file=f)
            print(f"Initial sparsity: {initial_sparsity:.6f}", file=f)
            print(f"Final sparsity: {final_sparsity:.6f}", file=f)
            print(f"WikiText PPL: {ppl_test:.4f}", file=f)
        print(f"Results saved to {save_filepath}")

    # Save model
    if args.save_model:
        print(f"\nSaving additionally pruned model to {args.save_model}...")
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print("Model saved successfully!")


if __name__ == '__main__':
    main()

