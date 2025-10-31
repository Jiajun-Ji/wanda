#!/usr/bin/env python
# coding=utf-8
"""Test BoolQ tokenization and label masking."""

from transformers import AutoTokenizer

# Load tokenizer
model_path = "/home/jjji/Research/Hybird-Kernel/wanda/out/progressive_three_tier/iter5/dense_finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test example
question = "Is the sun a star?"
passage = "The Sun is the star at the center of the Solar System."
answer = "True"

# Format
instruction = f"你会看到一个问题和一段文本。回答\"True\"或\"False\"。\n问题: {question}\n文本: {passage}\n答案:"
full_text = f"<s>[INST] {instruction} [/INST] {answer}</s>"

print("="*80)
print("Full text:")
print(full_text)
print("="*80)

# Tokenize
tokenized = tokenizer(
    full_text,
    max_length=512,
    truncation=True,
    padding=False,
    add_special_tokens=False,
)

input_ids = tokenized["input_ids"]
print(f"\nTotal tokens: {len(input_ids)}")
print(f"\nTokenized IDs: {input_ids[:50]}...")  # First 50 tokens

# Decode to see what it looks like
decoded = tokenizer.decode(input_ids)
print(f"\nDecoded text:\n{decoded}")

# Test finding [/INST]
print("\n" + "="*80)
print("Testing [/INST] detection:")
print("="*80)

# Method 1: Tokenize [/INST] separately
inst_end_tokens = tokenizer("[/INST]", add_special_tokens=False)["input_ids"]
print(f"\n[/INST] tokens (separate): {inst_end_tokens}")
print(f"[/INST] decoded: '{tokenizer.decode(inst_end_tokens)}'")

# Method 2: Search in full sequence
inst_end_pos = None
for j in range(len(input_ids) - len(inst_end_tokens) + 1):
    if input_ids[j:j+len(inst_end_tokens)] == inst_end_tokens:
        inst_end_pos = j + len(inst_end_tokens)
        print(f"\nFound [/INST] at position {j}, ends at {inst_end_pos}")
        print(f"Before [/INST]: {tokenizer.decode(input_ids[:j])}")
        print(f"After [/INST]: {tokenizer.decode(input_ids[inst_end_pos:])}")
        break

if inst_end_pos is None:
    print("\n❌ [/INST] NOT FOUND!")
    print("\nSearching for [/INST] manually in tokens:")
    for i in range(len(input_ids)):
        decoded_token = tokenizer.decode([input_ids[i]])
        if '[' in decoded_token or ']' in decoded_token or 'INST' in decoded_token:
            print(f"  Position {i}: {input_ids[i]} -> '{decoded_token}'")

# Test label creation
print("\n" + "="*80)
print("Testing label creation:")
print("="*80)

labels = [-100] * len(input_ids)
if inst_end_pos is not None:
    labels[inst_end_pos:] = input_ids[inst_end_pos:]
    
    non_masked_count = sum(1 for l in labels if l != -100)
    print(f"\nTotal labels: {len(labels)}")
    print(f"Masked labels (-100): {sum(1 for l in labels if l == -100)}")
    print(f"Non-masked labels: {non_masked_count}")
    print(f"\nNon-masked tokens (answer): {tokenizer.decode([l for l in labels if l != -100])}")
else:
    print("\n❌ All labels are -100 (no learning signal!)")

print("\n" + "="*80)

