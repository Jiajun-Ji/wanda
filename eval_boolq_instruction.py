#!/usr/bin/env python
# coding=utf-8
"""
Evaluate BoolQ with instruction format (matching training format).
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_boolq_instruction(model, tokenizer, max_samples=None):
    """Evaluate on BoolQ using instruction format."""
    # Load dataset
    dataset = load_dataset("super_glue", "boolq", split="validation")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc="Evaluating"):
        passage = example['passage']
        question = example['question']
        label = example['label']  # 1 = True, 0 = False
        
        # Format instruction (same as training)
        instruction = f"你会看到一个问题和一段文本。回答\"True\"或\"False\"。\n问题: {question}\n文本: {passage}\n答案:"
        prompt = f"<s>[INST] {instruction} [/INST]"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        generated_text = generated_text.strip()
        
        # Check answer
        predicted_label = None
        if "True" in generated_text or "true" in generated_text:
            predicted_label = 1
        elif "False" in generated_text or "false" in generated_text:
            predicted_label = 0
        
        if predicted_label == label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="eval_results_boolq_instruction")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Evaluate
    start_time = time.time()
    results = evaluate_boolq_instruction(model, tokenizer, args.max_samples)
    eval_time = time.time() - start_time
    
    results["eval_time_seconds"] = eval_time
    results["model_path"] = args.model
    
    # Print results
    print("\n" + "="*50)
    print("BoolQ Evaluation Results (Instruction Format)")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Eval time: {eval_time:.2f}s")
    print("="*50)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

