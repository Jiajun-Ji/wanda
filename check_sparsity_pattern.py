#!/usr/bin/env python
"""
Check if the sparsity pattern is maintained during training.
Compares the pruned model with a checkpoint to verify that zero weights remain zero.
"""

import torch
import argparse
import os
from transformers import AutoModelForCausalLM
import numpy as np


def find_layers(module, layers=[torch.nn.Linear], name=''):
    """Recursively find all linear layers in the model."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """Calculate overall sparsity of the model."""
    layers = model.model.layers
    count = 0 
    total_params = 0
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

    return float(count) / total_params


def compare_sparsity_patterns(pruned_model_path, checkpoint_path):
    """
    Compare sparsity patterns between pruned model and checkpoint.
    
    Returns:
        - total_zero_weights: Total number of zero weights in pruned model
        - changed_weights: Number of zero weights that became non-zero
        - percentage: Percentage of zero weights that changed
    """
    print("=" * 80)
    print("Checking Sparsity Pattern Preservation")
    print("=" * 80)
    
    # Load pruned model
    print(f"\n1. Loading pruned model from: {pruned_model_path}")
    pruned_model = AutoModelForCausalLM.from_pretrained(
        pruned_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load to CPU to save GPU memory
        low_cpu_mem_usage=True,
    )
    
    # Load checkpoint
    print(f"2. Loading checkpoint from: {checkpoint_path}")
    checkpoint_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    
    # Calculate sparsity
    print("\n3. Calculating sparsity...")
    pruned_sparsity = check_sparsity(pruned_model)
    checkpoint_sparsity = check_sparsity(checkpoint_model)
    
    print(f"   Pruned model sparsity: {pruned_sparsity:.4f} ({pruned_sparsity*100:.2f}%)")
    print(f"   Checkpoint sparsity:   {checkpoint_sparsity:.4f} ({checkpoint_sparsity*100:.2f}%)")
    
    # Compare layer by layer
    print("\n4. Comparing sparsity patterns layer by layer...")
    
    pruned_layers = pruned_model.model.layers
    checkpoint_layers = checkpoint_model.model.layers
    
    total_zero_weights = 0
    total_changed_weights = 0
    total_params = 0
    
    violations = []
    
    for i in range(len(pruned_layers)):
        pruned_subset = find_layers(pruned_layers[i])
        checkpoint_subset = find_layers(checkpoint_layers[i])
        
        layer_violations = 0
        layer_zeros = 0
        
        for name in pruned_subset:
            pruned_W = pruned_subset[name].weight.data
            checkpoint_W = checkpoint_subset[name].weight.data
            
            # Find zero weights in pruned model
            zero_mask = (pruned_W == 0)
            layer_zeros += zero_mask.sum().item()
            total_zero_weights += zero_mask.sum().item()
            total_params += pruned_W.numel()
            
            # Check if any zero weights became non-zero
            changed_mask = zero_mask & (checkpoint_W != 0)
            num_changed = changed_mask.sum().item()
            layer_violations += num_changed
            total_changed_weights += num_changed
            
            if num_changed > 0:
                violations.append({
                    'layer': i,
                    'name': name,
                    'changed': num_changed,
                    'total_zeros': zero_mask.sum().item(),
                    'percentage': num_changed / zero_mask.sum().item() * 100
                })
        
        if layer_violations > 0:
            print(f"   ⚠️  Layer {i}: {layer_violations} zero weights changed (out of {layer_zeros})")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Total zero weights in pruned model: {total_zero_weights:,} ({total_zero_weights/total_params*100:.2f}%)")
    print(f"Zero weights that became non-zero: {total_changed_weights:,}")
    
    if total_changed_weights == 0:
        print("\n✅ SUCCESS: Sparsity pattern is perfectly preserved!")
        print("   All zero weights remained zero during training.")
    else:
        violation_percentage = total_changed_weights / total_zero_weights * 100
        print(f"\n❌ VIOLATION: {violation_percentage:.4f}% of zero weights became non-zero!")
        print(f"   This indicates the mask_grad() function is not working correctly.")
        
        print("\nTop 10 layers with most violations:")
        violations.sort(key=lambda x: x['changed'], reverse=True)
        for v in violations[:10]:
            print(f"   Layer {v['layer']}, {v['name']}: {v['changed']} changed ({v['percentage']:.2f}%)")
    
    print("=" * 80)
    
    return total_zero_weights, total_changed_weights, violation_percentage if total_changed_weights > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Check sparsity pattern preservation")
    parser.add_argument(
        "--pruned_model",
        type=str,
        default="out/llama2_7b/block_16x16_20sparsity/wanda/pruned_model",
        help="Path to pruned model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to check"
    )
    
    args = parser.parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.pruned_model):
        print(f"❌ Error: Pruned model not found at {args.pruned_model}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return
    
    # Compare
    compare_sparsity_patterns(args.pruned_model, args.checkpoint)


if __name__ == "__main__":
    main()

