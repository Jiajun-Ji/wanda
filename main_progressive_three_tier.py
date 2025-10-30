import argparse
import os
import csv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import (
    prune_wanda_progressive_three_tier, 
    check_sparsity, 
    save_tier_map,
    load_tier_map
)
from lib.eval import eval_ppl

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


def load_progressive_config(config_path):
    """
    Load progressive pruning configuration from CSV file.
    
    Returns:
        List of iteration configs
    """
    iterations = []
    with open(config_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append({
                'iteration': int(row['iteration']),
                'dense': float(row['dense']),
                'mid_2_4': float(row['mid_2_4']),
                'topk': float(row['topk']),
                'dense_to_2_4': float(row['dense_to_2_4']),
                'mid_2_4_to_topk': float(row['mid_2_4_to_topk']),
                'epochs': int(row['epochs'])
            })
    return iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model path')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    
    # Progressive pruning specific arguments
    parser.add_argument('--config', type=str, default='progressive_config.csv', help='Path to progressive config CSV')
    parser.add_argument('--iteration', type=int, required=True, help='Current iteration number (1-5)')
    parser.add_argument('--previous_tier_maps', type=str, default=None, help='Path to previous iteration tier maps')
    parser.add_argument('--block_size', type=int, default=16, help='Block size for block pruning')
    parser.add_argument('--topk_per_block', type=int, default=10, help='Number of weights to keep in topk blocks')

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Load progressive configuration
    iterations = load_progressive_config(args.config)
    
    # Find current iteration config
    current_config = None
    for config in iterations:
        if config['iteration'] == args.iteration:
            current_config = config
            break
    
    if current_config is None:
        raise ValueError(f"Iteration {args.iteration} not found in config file {args.config}")
    
    print("="*80)
    print(f"Progressive Three-Tier Pruning - Iteration {args.iteration}")
    print("="*80)
    print(f"Configuration:")
    print(f"  Dense:  {current_config['dense']*100:.0f}%")
    print(f"  2:4:    {current_config['mid_2_4']*100:.0f}%")
    print(f"  TopK:   {current_config['topk']*100:.0f}%")
    print(f"  Epochs: {current_config['epochs']}")
    print("="*80)

    # Load model
    model_name = args.model.split("/")[-1]
    print(f"\nLoading model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]
    print(f"Using device: {device}\n")

    # Load previous tier maps if continuing from previous iteration
    previous_tier_maps = None
    if args.previous_tier_maps is not None and os.path.exists(args.previous_tier_maps):
        print(f"Loading previous tier maps from {args.previous_tier_maps}")
        previous_tier_maps = torch.load(args.previous_tier_maps)
        print(f"Loaded tier maps for {len(previous_tier_maps)} layers\n")

    # Setup log file
    log_file = os.path.join(args.save, f"pruning_iter{args.iteration}.log")
    os.makedirs(args.save, exist_ok=True)

    # Perform progressive pruning
    tier_maps, stats = prune_wanda_progressive_three_tier(
        args, model, tokenizer, device,
        iteration_config=current_config,
        previous_tier_maps=previous_tier_maps,
        block_size=args.block_size,
        topk_per_block=args.topk_per_block,
        log_file=log_file
    )

    # Check sparsity
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"Sparsity sanity check: {sparsity_ratio:.4f}")
    print("*"*30)
    
    # Evaluate perplexity
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"WikiText perplexity: {ppl_test:.4f}")

    # Save results
    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        
        # Save log
        save_filepath = os.path.join(args.save, f"log_iter{args.iteration}.txt")
        with open(save_filepath, "w") as f:
            print(f"iteration\tdense\tmid_2_4\ttopk\tsparsity\tppl", file=f, flush=True)
            print(f"{args.iteration}\t{current_config['dense']:.2f}\t{current_config['mid_2_4']:.2f}\t{current_config['topk']:.2f}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
        
        # Save tier maps
        tier_maps_path = os.path.join(args.save, f"tier_maps_iter{args.iteration}.pt")
        torch.save(tier_maps, tier_maps_path)
        print(f"\nTier maps saved to: {tier_maps_path}")

    # Save model
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"Model saved to: {args.save_model}")
    
    print("\n" + "="*80)
    print(f"Iteration {args.iteration} Complete!")
    print("="*80)
    print(f"Next steps:")
    print(f"  1. Finetune the model for {current_config['epochs']} epochs")
    print(f"  2. Run iteration {args.iteration + 1} with --previous_tier_maps {tier_maps_path}")
    print("="*80)


if __name__ == '__main__':
    main()

