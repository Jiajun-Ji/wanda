#!/usr/bin/env python
"""
Evaluate model perplexity on WikiText-2 using wanda's eval_ppl function.
Usage: python eval_model_ppl.py <model_path> [base_model_path]
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.eval import eval_ppl

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_model_ppl.py <model_path> [base_model_path]")
        sys.exit(1)

    model_path = sys.argv[1]
    base_model = sys.argv[2] if len(sys.argv) > 2 else "/mnt/sdb/llm_models/Llama-2-7b-hf"

    # Convert to absolute path
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)

    print(f"Loading model from: {model_path}")
    print(f"Using tokenizer from: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        local_files_only=True
    )
    model.seqlen = 2048

    class Args:
        pass
    args = Args()

    print("Evaluating on WikiText-2...")
    ppl = eval_ppl(args, model, tokenizer)
    print(f"Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()

