#!/usr/bin/env python3
"""
Test progressive pruning functions
"""

import torch
from lib.prune import (
    compute_all_block_scores_unfold,
    compute_2_4_block_scores_batch,
    apply_2_4_sparsity_batch,
    apply_topk_sparsity_batch,
    progressive_three_tier_iteration,
    initialize_tier_map_from_ratios,
    apply_tier_map_to_weights,
    TIER_DENSE,
    TIER_2_4,
    TIER_TOPK
)

def test_compute_all_block_scores():
    """Test vectorized block score computation"""
    print("Testing compute_all_block_scores_unfold...")
    
    # Create test matrix
    W_metric = torch.randn(256, 256).cuda()
    block_size = 16
    
    # Compute scores
    block_scores = compute_all_block_scores_unfold(W_metric, block_size)
    
    # Check shape
    expected_rows = 256 // 16
    expected_cols = 256 // 16
    assert block_scores.shape == (expected_rows, expected_cols), f"Expected shape ({expected_rows}, {expected_cols}), got {block_scores.shape}"
    
    print(f"  ✅ Block scores shape: {block_scores.shape}")
    print(f"  ✅ Score range: [{block_scores.min():.4f}, {block_scores.max():.4f}]")
    print()


def test_progressive_iteration():
    """Test one progressive iteration"""
    print("Testing progressive_three_tier_iteration...")
    
    # Create test weight matrix
    W = torch.randn(256, 256).cuda()
    W_metric = torch.abs(W) * torch.randn(256).cuda().abs()
    
    block_size = 16
    num_blocks_row = 256 // 16
    num_blocks_col = 256 // 16
    
    # Initialize all blocks as dense
    current_tier_map = torch.full((num_blocks_row, num_blocks_col), TIER_DENSE, dtype=torch.long).cuda()
    
    # Target: 80% dense, 10% 2:4, 10% topk
    target_dense = 0.8
    target_2_4 = 0.1
    target_topk = 0.1
    
    # Run iteration
    updated_tier_map, stats = progressive_three_tier_iteration(
        W, W_metric,
        current_tier_map,
        target_dense,
        target_2_4,
        target_topk,
        block_size=16,
        topk_per_block=10
    )
    
    # Check results
    total_blocks = num_blocks_row * num_blocks_col
    final_dense = (updated_tier_map == TIER_DENSE).sum().item()
    final_2_4 = (updated_tier_map == TIER_2_4).sum().item()
    final_topk = (updated_tier_map == TIER_TOPK).sum().item()
    
    print(f"  Total blocks: {total_blocks}")
    print(f"  Dense blocks: {final_dense} ({final_dense/total_blocks*100:.1f}%)")
    print(f"  2:4 blocks: {final_2_4} ({final_2_4/total_blocks*100:.1f}%)")
    print(f"  TopK blocks: {final_topk} ({final_topk/total_blocks*100:.1f}%)")
    print(f"  Dense degraded: {stats['dense_degraded']}")
    print(f"  2:4 degraded: {stats['mid_2_4_degraded']}")
    
    # Verify ratios are close to target
    assert abs(final_dense/total_blocks - target_dense) < 0.05, "Dense ratio mismatch"
    assert abs(final_2_4/total_blocks - target_2_4) < 0.05, "2:4 ratio mismatch"
    assert abs(final_topk/total_blocks - target_topk) < 0.05, "TopK ratio mismatch"
    
    print("  ✅ Ratios match targets")
    print()


def test_two_stage_degradation():
    """Test two-stage degradation: Dense→2:4, then re-evaluate 2:4→TopK"""
    print("Testing two-stage degradation...")
    
    W = torch.randn(256, 256).cuda()
    W_metric = torch.abs(W) * torch.randn(256).cuda().abs()
    
    block_size = 16
    num_blocks_row = 256 // 16
    num_blocks_col = 256 // 16
    total_blocks = num_blocks_row * num_blocks_col
    
    # Start with 90% dense, 10% 2:4, 0% topk
    tier_map = initialize_tier_map_from_ratios(W_metric, 0.9, 0.1, 0.0, block_size)
    
    print(f"  Initial state:")
    print(f"    Dense: {(tier_map == TIER_DENSE).sum().item()} ({(tier_map == TIER_DENSE).sum().item()/total_blocks*100:.0f}%)")
    print(f"    2:4: {(tier_map == TIER_2_4).sum().item()} ({(tier_map == TIER_2_4).sum().item()/total_blocks*100:.0f}%)")
    print(f"    TopK: {(tier_map == TIER_TOPK).sum().item()} ({(tier_map == TIER_TOPK).sum().item()/total_blocks*100:.0f}%)")
    
    # Apply initial sparsity
    apply_tier_map_to_weights(W, tier_map, block_size, topk_per_block=10)
    
    # Iteration 2: Target 80% dense, 10% 2:4, 10% topk
    tier_map, stats = progressive_three_tier_iteration(
        W, W_metric,
        tier_map,
        0.8, 0.1, 0.1,
        block_size=16,
        topk_per_block=10
    )
    
    print(f"  After iteration:")
    print(f"    Dense: {(tier_map == TIER_DENSE).sum().item()} ({(tier_map == TIER_DENSE).sum().item()/total_blocks*100:.0f}%)")
    print(f"    2:4: {(tier_map == TIER_2_4).sum().item()} ({(tier_map == TIER_2_4).sum().item()/total_blocks*100:.0f}%)")
    print(f"    TopK: {(tier_map == TIER_TOPK).sum().item()} ({(tier_map == TIER_TOPK).sum().item()/total_blocks*100:.0f}%)")
    print(f"    Dense→2:4: {stats['dense_degraded']} blocks")
    print(f"    2:4→TopK: {stats['mid_2_4_degraded']} blocks")
    
    # Verify: should have degraded 10% dense→2:4, then 10% 2:4→topk
    expected_dense_degraded = int(total_blocks * 0.1)
    expected_2_4_degraded = int(total_blocks * 0.1)
    
    assert abs(stats['dense_degraded'] - expected_dense_degraded) <= 1, "Dense degradation count mismatch"
    assert abs(stats['mid_2_4_degraded'] - expected_2_4_degraded) <= 1, "2:4 degradation count mismatch"
    
    print("  ✅ Two-stage degradation works correctly")
    print()


def test_sparsity_application():
    """Test that sparsity is actually applied to weights"""
    print("Testing sparsity application...")
    
    W = torch.randn(256, 256).cuda()
    W_original = W.clone()
    
    block_size = 16
    num_blocks_col = 256 // 16
    
    # Apply 2:4 to first block
    block_indices = torch.tensor([0]).cuda()
    apply_2_4_sparsity_batch(W, block_indices, num_blocks_col, block_size)
    
    # Check that first block has ~50% sparsity
    first_block = W[:16, :16]
    sparsity = (first_block == 0).sum().item() / first_block.numel()
    print(f"  First block sparsity after 2:4: {sparsity*100:.1f}%")
    assert 0.45 < sparsity < 0.55, "2:4 sparsity should be ~50%"
    
    # Apply TopK to second block
    W = W_original.clone()
    block_indices = torch.tensor([1]).cuda()
    apply_topk_sparsity_batch(W, block_indices, num_blocks_col, block_size, k=10)
    
    # Check that second block has high sparsity
    second_block = W[:16, 16:32]
    non_zero = (second_block != 0).sum().item()
    print(f"  Second block non-zero weights after TopK(10): {non_zero}")
    assert non_zero <= 10, f"TopK should keep at most 10 weights, got {non_zero}"
    
    print("  ✅ Sparsity patterns applied correctly")
    print()


def main():
    print("="*60)
    print("Progressive Pruning Function Tests")
    print("="*60)
    print()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping tests")
        return
    
    try:
        test_compute_all_block_scores()
        test_sparsity_application()
        test_progressive_iteration()
        test_two_stage_degradation()
        
        print("="*60)
        print("✅ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

