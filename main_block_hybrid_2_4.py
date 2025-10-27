import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda_hybrid_2_4, check_sparsity, find_layers
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level (for reference)')
    parser.add_argument("--sparsity_type", type=str, default="hybrid_2_4")
    parser.add_argument("--prune_method", type=str, default="wanda")
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    
    # Hybrid 2:4 specific arguments
    parser.add_argument('--block_size', type=int, default=16, help='Block size for block pruning')
    parser.add_argument('--topk_per_block', type=int, default=10, help='Number of weights to keep in low-score blocks')
    parser.add_argument('--top_blocks_ratio', type=float, default=0.6, help='Ratio of top blocks to consider (default 0.6 = 60%)')
    parser.add_argument('--score_threshold', type=float, default=0.8, help='Score threshold for accepting 2:4 sparsity (default 0.8 = 80%)')

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured" and args.sparsity_type != "hybrid_2_4":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("="*80)
        print(f"Pruning model with Wanda Hybrid 2:4 method")
        print(f"Block size: {args.block_size}x{args.block_size}")
        print(f"Top blocks ratio: {args.top_blocks_ratio} (top {args.top_blocks_ratio*100:.0f}% blocks)")
        print(f"Score threshold for 2:4: {args.score_threshold} ({args.score_threshold*100:.0f}% of original score)")
        print(f"Top-K per low-score block: {args.topk_per_block}")
        print("="*80)
        
        prune_wanda_hybrid_2_4(
            args, model, tokenizer, device, 
            args.block_size, args.topk_per_block, 
            args.top_blocks_ratio, args.score_threshold
        )

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{model_name}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()

