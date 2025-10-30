# Progressive Three-Tier Block Pruning

## Overview

This implements a **progressive/iterative pruning strategy** with three-tier block sparsity:
- **Dense blocks**: 0% sparsity (all weights kept)
- **2:4 blocks**: 50% sparsity (hardware-friendly structured sparsity)
- **TopK blocks**: ~96% sparsity (only top-k weights kept per block)

## Environment Requirements

**Two mamba/conda environments are required** (see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for details):

1. **prune_llm**: For pruning (transformers 4.36.0)
2. **wanda_lora**: For finetuning (transformers 4.57.1)

The scripts automatically switch between environments using `mamba run -n <env_name>`.

### Key Features

1. **Progressive degradation**: Gradually increase sparsity over multiple iterations
2. **Two-stage degradation per iteration**:
   - Stage 1: Dense → 2:4 (evaluate all dense blocks, degrade lowest)
   - Stage 2: Re-evaluate all 2:4 blocks (including newly degraded), degrade lowest → TopK
3. **Importance-based**: Uses latest fine-tuned weights for evaluation
4. **Hierarchical**: One-way degradation (Dense → 2:4 → TopK, irreversible)
5. **Optimized**: Fully vectorized GPU operations for fast computation

## Default Configuration

The default configuration (`progressive_config.csv`) runs 5 iterations:

| Iteration | Dense | 2:4 | TopK | Dense→2:4 | 2:4→TopK | Epochs |
|-----------|-------|-----|------|-----------|----------|--------|
| 1 | 90% | 10% | 0% | 10% | - | 2 |
| 2 | 80% | 10% | 10% | 10% | 10% | 2 |
| 3 | 65% | 20% | 15% | 15% | 5% | 2 |
| 4 | 50% | 30% | 20% | 15% | 5% | 2 |
| 5 | 35% | 45% | 20% | 15% | - | 3 |

**Final target**: 35% Dense, 45% 2:4, 20% TopK (~42% overall sparsity)

## Usage

### Option 1: Automatic Pipeline (Recommended for full run)

Run all 5 iterations automatically:

```bash
./run_progressive_three_tier.sh
```

This will:
1. Prune iteration 1
2. Finetune for 2 epochs
3. Prune iteration 2 (using finetuned weights from iter1)
4. Finetune for 2 epochs
5. ... continue until iteration 5

**Total time**: ~10-15 hours on A100 48GB

### Option 2: Manual Iteration Control (Recommended for testing)

Run one iteration at a time:

```bash
# Iteration 1 (from base model)
./run_progressive_single_iter.sh 1

# Manually finetune (see below)

# Iteration 2 (from iter1 finetuned model)
./run_progressive_single_iter.sh 2 \
    out/progressive_three_tier/iter1/finetuned_model \
    out/progressive_three_tier/iter1/tier_maps_iter1.pt

# ... and so on
```

### Manual Finetuning Between Iterations

After each pruning iteration, finetune the model:

```bash
cd dense_ft

python finetune_sparse_model.py \
    --model_name_or_path ../out/progressive_three_tier/iter1/pruned_model \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --bf16 \
    --output_dir ../out/progressive_three_tier/iter1/finetuned_model \
    --logging_steps 50 \
    --eval_steps 100 \
    --save_steps 100 \
    --eval_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --save_total_limit 2 \
    --overwrite_output_dir

cd ..
```

## Customization

### Modify Iteration Schedule

Edit `progressive_config.csv`:

```csv
iteration,dense,mid_2_4,topk,dense_to_2_4,mid_2_4_to_topk,epochs
1,0.95,0.05,0.00,0.05,0.00,1
2,0.90,0.05,0.05,0.05,0.05,1
...
```

**Rules**:
- `dense + mid_2_4 + topk` must equal 1.0
- Ratios should be monotonic: dense decreases, topk increases
- `dense_to_2_4` and `mid_2_4_to_topk` are for reference only (calculated automatically)

### Modify Block Size or TopK

Edit the scripts:

```bash
BLOCK_SIZE=32  # Change from 16 to 32
TOPK_PER_BLOCK=20  # Change from 10 to 20
```

### Modify GPU

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use multiple GPUs
```

## Output Structure

```
out/progressive_three_tier/
├── iter1/
│   ├── pruned_model/          # Model after pruning
│   ├── finetuned_model/       # Model after finetuning
│   ├── tier_maps_iter1.pt     # Tier assignments for each block
│   └── log_iter1.txt          # Sparsity and perplexity log
├── iter2/
│   ├── ...
├── ...
└── iter5/
    └── finetuned_model/       # Final model (35% Dense, 45% 2:4, 20% TopK)
```

## Algorithm Details

### Two-Stage Degradation

**Example: Iteration 2 (90%→80% Dense, 10%→10% 2:4, 0%→10% TopK)**

```
Input state: (90% Dense, 10% 2:4, 0% TopK)

Stage 1: Dense Degradation
  1. Evaluate all 90% Dense blocks using latest weights
  2. Compute importance scores: score = sum(|W| × activation_norm)
  3. Sort by score, select lowest 10%
  4. Apply 2:4 sparsity to selected blocks
  5. Update tier map: DENSE → 2:4
  → Intermediate state: (80% Dense, 20% 2:4, 0% TopK)

Stage 2: 2:4 Degradation
  1. Re-evaluate ALL 20% 2:4 blocks (including newly degraded)
  2. For each 2:4 block, compute score using only non-zero weights
  3. Sort by score, select lowest 10%
  4. Apply TopK sparsity (keep top-10 weights per block)
  5. Update tier map: 2:4 → TOPK
  → Final state: (80% Dense, 10% 2:4, 10% TopK)

Finetune: 2 epochs
```

### Importance Score Calculation

**For Dense blocks**:
```python
score = sum(|W| × sqrt(||activation||²))
```

**For 2:4 blocks** (only non-zero weights):
```python
nonzero_mask = (W != 0)
score = sum(|W[nonzero_mask]| × sqrt(||activation[nonzero_mask]||²))
```

### Optimization

- **Vectorized block score computation**: Uses `unfold` for 100-1000x speedup
- **GPU-accelerated sorting**: All operations on GPU
- **Batch processing**: Applies sparsity to multiple blocks efficiently

## Comparison with Other Methods

| Method | Sparsity | Strategy | Iterations |
|--------|----------|----------|------------|
| **Wanda (one-shot)** | 50% | Unstructured | 1 |
| **Hybrid 2:4** | ~30-40% | Mixed (Dense/2:4/TopK) | 1 |
| **Three-tier fixed** | ~30-40% | Fixed ratios | 1 |
| **Progressive (this)** | ~42% | Gradual degradation | 5 |

**Advantages of progressive pruning**:
- Better accuracy preservation (gradual adaptation)
- Uses latest weights for importance evaluation
- Allows model to recover between iterations
- More stable training

## Troubleshooting

### Out of Memory

Reduce batch size or use gradient accumulation:
```bash
BATCH_SIZE=2
GRADIENT_ACCUMULATION=8
```

### Slow Pruning

Check GPU utilization:
```bash
nvidia-smi
```

Ensure vectorized operations are being used (should be very fast, <5 min per iteration).

### Perplexity Not Improving

- Increase finetuning epochs
- Adjust learning rate
- Check if sparsity is too aggressive

## References

- **Wanda**: [Sun et al., 2023](https://arxiv.org/abs/2306.11695)
- **2:4 Structured Sparsity**: [NVIDIA Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
- **Progressive Pruning**: Inspired by iterative magnitude pruning

## Contact

For questions or issues, please check the main project README or open an issue.

