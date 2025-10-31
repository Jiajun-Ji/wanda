#!/usr/bin/env python
# coding=utf-8
"""
Evaluate GSM8K with instruction format (matching training format).
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text):
    """
    Extract numerical answer from GSM8K response.
    Looks for #### followed by a number.
    """
    # Try to find #### pattern
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        # Remove commas and convert to number
        answer_str = match.group(1).replace(',', '')
        try:
            # Try as integer first
            return int(answer_str)
        except ValueError:
            # Try as float
            try:
                return float(answer_str)
            except ValueError:
                return None
    
    # Fallback: try to find any number at the end
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        last_number = numbers[-1].replace(',', '')
        try:
            return int(last_number)
        except ValueError:
            try:
                return float(last_number)
            except ValueError:
                return None
    
    return None


def evaluate_gsm8k_instruction(model, tokenizer, max_samples=None):
    """Evaluate on GSM8K using instruction format."""
    # Load dataset
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc="Evaluating"):
        question = example['question']
        answer_text = example['answer']
        
        # Extract ground truth answer
        gt_answer = extract_answer(answer_text)
        if gt_answer is None:
            continue
        
        # Format instruction (same as training)
        instruction = f"Solve this math problem step by step.\nQuestion: {question}\nAnswer:"
        prompt = f"<s>[INST] {instruction} [/INST]"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract predicted answer
        pred_answer = extract_answer(generated_text)
        
        # Check if correct
        if pred_answer is not None and abs(pred_answer - gt_answer) < 1e-6:
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
    parser.add_argument("--output_dir", type=str, default="eval_results_gsm8k_instruction")
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
    results = evaluate_gsm8k_instruction(model, tokenizer, args.max_samples)
    eval_time = time.time() - start_time
    
    results["eval_time_seconds"] = eval_time
    results["model_path"] = args.model
    
    # Print results
    print("\n" + "="*50)
    print("GSM8K Evaluation Results (Instruction Format)")
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

