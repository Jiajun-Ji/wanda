# Block-wise Wanda Pruning Guide

## 概述

本指南介绍如何使用16x16块结构化剪枝方法对Llama-2-7b模型进行剪枝。

## 什么是块剪枝？

### 非结构化剪枝 vs 块剪枝

| 特性 | 非结构化剪枝 | 块剪枝 (16x16) |
|------|-------------|---------------|
| **剪枝粒度** | 单个权重 | 16x16权重块 |
| **稀疏模式** | 随机分散 | 块状结构化 |
| **硬件加速** | 困难 | 容易 |
| **精度** | 更高 | 略低 |
| **实用性** | 研究 | 部署 |

### 算法流程

```
1. 计算Wanda分数矩阵: Score = |W| × √(activation)
2. 将权重矩阵划分为16x16的块
3. 计算每个块的平均分数
4. 根据稀疏度，保留分数最高的块
5. 将分数最低的块全部置零
```

## 使用方法

### 方法1：使用运行脚本（推荐）

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda
./run_prune_llama2_7b_block.sh
```

### 方法2：直接使用Python脚本

```bash
python main_block.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --sparsity_ratio 0.5 \
    --block_size 16 \
    --nsamples 128 \
    --seed 0 \
    --save out/llama2_7b/block_16x16/wanda \
    --save_model out/llama2_7b/block_16x16/wanda/pruned_model
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型路径 | 必需 |
| `--sparsity_ratio` | 目标稀疏度 | 0.5 (50%) |
| `--block_size` | 块大小 | 16 |
| `--nsamples` | 校准样本数 | 128 |
| `--seed` | 随机种子 | 0 |
| `--save` | 结果保存路径 | None |
| `--save_model` | 模型保存路径 | None |

## 评估方法

### 评估单个模型

```bash
python eval_additional_pruned.py \
    --model out/llama2_7b/block_16x16/wanda/pruned_model
```

### 对比非结构化 vs 块剪枝

```bash
python eval_compare_pruning_types.py \
    --unstructured_model out/llama2_7b/unstructured/wanda/pruned_model \
    --block_model out/llama2_7b/block_16x16/wanda/pruned_model
```

## 输出示例

### 剪枝过程输出

```
================================================================================
Wanda Block Pruning (Block Size: 16x16)
================================================================================
Target sparsity: 50.00%
Block size: 16x16
================================================================================

Loading calibration data (WikiText2)...
Dataset loading complete

================================================================================
Pruning Layer 0
================================================================================
  self_attn.q_proj:
    - Shape: [4096, 4096]
    - Total params: 16,777,216
    - Pruned params: 8,388,608
    - Actual sparsity: 50.0000%
    - Target sparsity: 50.00%
  ...
```

### 评估输出

```
📊 COMPARISON RESULTS
================================================================================
Pruning Type                   Sparsity        Perplexity      PPL Diff
--------------------------------------------------------------------------------
Unstructured Wanda (50%)       50.0000%        6.3100          0.0000
Block 16x16 Wanda (50%)        50.0000%        6.5200          +0.2100
================================================================================

💡 ANALYSIS
================================================================================
1. Perplexity Comparison:
   - Unstructured: 6.3100
   - Block 16x16: 6.5200
   - Difference: +0.2100 (+3.33%)

2. Trade-off Analysis:
   ✅ Excellent: Block pruning achieves similar performance
   → Block-structured sparsity is viable for deployment
```

## 实验建议

### 1. 不同块大小对比

```bash
# 8x8 块
python main_block.py --block_size 8 --save out/llama2_7b/block_8x8/wanda

# 16x16 块（推荐）
python main_block.py --block_size 16 --save out/llama2_7b/block_16x16/wanda

# 32x32 块
python main_block.py --block_size 32 --save out/llama2_7b/block_32x32/wanda
```

### 2. 不同稀疏度对比

```bash
# 30% 稀疏度
python main_block.py --sparsity_ratio 0.3 --save out/llama2_7b/block_16x16/wanda_30

# 50% 稀疏度
python main_block.py --sparsity_ratio 0.5 --save out/llama2_7b/block_16x16/wanda_50

# 70% 稀疏度
python main_block.py --sparsity_ratio 0.7 --save out/llama2_7b/block_16x16/wanda_70
```

## 预期结果

### 性能预期

| 模型 | 稀疏度 | 困惑度 | 相对变化 |
|------|--------|--------|----------|
| Dense | 0% | 5.12 | - |
| Unstructured 50% | 50% | 6.31 | +23% |
| Block 16x16 50% | 50% | 6.5-7.0 | +27-37% |

### 加速预期

- **非结构化剪枝**：理论加速2x，实际加速1.0-1.2x（硬件支持差）
- **块剪枝**：理论加速2x，实际加速1.5-2.0x（硬件友好）

## 技术细节

### 块分数计算

```python
# 对每个16x16块计算平均分数
block_score = block.mean()  # 使用平均值，不受块大小影响
```

### 边界处理

- 如果权重矩阵维度不是16的倍数，边界块会小于16x16
- 使用平均值而非总和，确保公平比较

### 稀疏度定义

- 按块数量计算稀疏度：剪掉50%的块
- 实际权重稀疏度接近50%（取决于边界块）

## 常见问题

### Q1: 为什么选择16x16？

A: 16x16是GPU硬件加速的常见块大小，平衡了精度和加速效果。

### Q2: 块剪枝比非结构化剪枝差多少？

A: 通常困惑度增加3-10%，但推理速度可提升50-100%。

### Q3: 可以用于其他模型吗？

A: 可以，支持所有Llama系列模型（7B, 13B, 30B, 65B）。

### Q4: 如何选择块大小？

A: 
- 8x8: 更精细，精度更高，加速效果略差
- 16x16: 推荐，平衡精度和速度
- 32x32: 更粗糙，精度较低，加速效果更好

## 文件清单

| 文件 | 说明 |
|------|------|
| `lib/prune.py` | 核心剪枝函数（已添加块剪枝） |
| `main_block.py` | 主执行脚本 |
| `run_prune_llama2_7b_block.sh` | 运行脚本 |
| `eval_compare_pruning_types.py` | 对比评估脚本 |
| `BLOCK_PRUNING_GUIDE.md` | 本文档 |

## 参考资料

- Wanda论文: https://arxiv.org/abs/2306.11695
- Block Sparse论文: https://arxiv.org/abs/2104.08378
- NVIDIA Block Sparse: https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/

