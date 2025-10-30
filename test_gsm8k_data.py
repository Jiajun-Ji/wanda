#!/usr/bin/env python3
"""
Test GSM8K data loading for pruning
"""

import sys
import torch
from transformers import AutoTokenizer

# Add lib to path
sys.path.insert(0, 'lib')
from data import get_gsm8k

def test_gsm8k_loading():
    """Test if GSM8K data can be loaded correctly"""
    
    print("=" * 60)
    print("Testing GSM8K Data Loading")
    print("=" * 60)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/sdb/llm_models/Llama-2-7b-hf",
        use_fast=False
    )
    print("   ✓ Tokenizer loaded")
    
    # Load GSM8K data
    print("\n2. Loading GSM8K data...")
    nsamples = 5
    seqlen = 2048
    seed = 0
    
    trainloader, testenc = get_gsm8k(nsamples, seed, seqlen, tokenizer)
    print(f"   ✓ Loaded {len(trainloader)} samples")
    
    # Check data format
    print("\n3. Checking data format...")
    for i, (inp, tar) in enumerate(trainloader[:3]):
        print(f"\n   Sample {i+1}:")
        print(f"   - Input shape: {inp.shape}")
        print(f"   - Target shape: {tar.shape}")
        
        # Decode first 100 tokens to see the text
        text = tokenizer.decode(inp[0][:100], skip_special_tokens=True)
        print(f"   - Text preview: {text[:150]}...")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_gsm8k_loading()

