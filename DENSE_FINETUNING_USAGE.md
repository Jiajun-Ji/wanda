# 全量微调使用指南

## 📋 概述

本指南介绍如何使用 `run_dense_finetune_block_20sparsity.sh` 脚本对剪枝后的模型进行全量微调。

脚本支持**单卡**和**多卡**训练,可以根据你的GPU资源灵活选择。

---

## 🚀 快速开始

### 单卡训练 (默认)

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda
bash run_dense_finetune_block_20sparsity.sh
```

**默认配置**:
- GPU: 单卡 (GPU 2)
- 训练时间: ~24-36小时
- 显存需求: ~55-65GB

---

### 双卡训练 (推荐)

```bash
bash run_dense_finetune_block_20sparsity.sh --num_gpus 2 --gpu_ids "2,3"
```

**配置**:
- GPU: 双卡 (GPU 2,3)
- 训练时间: ~14-20小时 (1.7倍加速)
- 显存需求: ~55-65GB (每卡)

---

### 四卡训练

```bash
bash run_dense_finetune_block_20sparsity.sh --num_gpus 4 --gpu_ids "0,1,2,3"
```

**配置**:
- GPU: 四卡 (GPU 0-3)
- 训练时间: ~8-12小时 (2.5-3倍加速)
- 显存需求: ~55-65GB (每卡)

---

## 📖 参数说明

### 命令行参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--num_gpus` | GPU数量 | 1 | `--num_gpus 2` |
| `--gpu_ids` | GPU编号 (逗号分隔) | "2" | `--gpu_ids "2,3"` |
| `-h, --help` | 显示帮助信息 | - | `--help` |

### 训练参数 (脚本内配置)

| 参数 | 值 | 说明 |
|------|-----|------|
| `NUM_EPOCHS` | 3 | 训练轮数 |
| `LEARNING_RATE` | 5e-5 | 学习率 |
| `BATCH_SIZE` | 1 | 每卡batch size |
| `GRADIENT_ACCUMULATION_STEPS` | 自动调整 | 梯度累积步数 |
| `BLOCK_SIZE` | 1024 | 序列长度 |
| `MAX_TRAIN_SAMPLES` | 30000 | 最大训练样本数 |

---

## 🔧 梯度累积自动调整

脚本会根据GPU数量自动调整梯度累积步数,以保持**相同的有效batch size**:

| GPU数量 | 梯度累积步数 | 有效Batch Size |
|---------|-------------|---------------|
| 1 | 8 | 1 × 1 × 8 = 8 |
| 2 | 4 | 1 × 2 × 4 = 8 |
| 4 | 2 | 1 × 4 × 2 = 8 |

**公式**: `有效Batch Size = per_device_batch_size × num_gpus × gradient_accumulation_steps`

---

## 📊 性能对比

### 训练时间估算

| GPU配置 | 训练时间 | 加速比 | 效率 |
|---------|---------|--------|------|
| 1×A100 80GB | 24-36h | 1.0× | 100% |
| 2×A100 80GB | 14-20h | 1.7× | 85% |
| 4×A100 80GB | 8-12h | 2.5-3× | 65-75% |

### 显存需求

每张GPU需要:
- 模型参数 (FP16): ~13GB
- 优化器状态 (AdamW): ~26GB
- 梯度: ~13GB
- 激活值 (gradient checkpointing): ~5-10GB
- **总计**: ~55-65GB

**推荐配置**:
- ✅ A100 80GB (最佳)
- ✅ A100 40GB (可用,需gradient checkpointing)
- ⚠️ V100 32GB (不够)

---

## 💡 使用示例

### 示例1: 单卡训练 (GPU 3)

```bash
bash run_dense_finetune_block_20sparsity.sh --num_gpus 1 --gpu_ids "3"
```

### 示例2: 双卡训练 (GPU 0,1)

```bash
bash run_dense_finetune_block_20sparsity.sh --num_gpus 2 --gpu_ids "0,1"
```

### 示例3: 查看帮助

```bash
bash run_dense_finetune_block_20sparsity.sh --help
```

输出:
```
Usage: run_dense_finetune_block_20sparsity.sh [OPTIONS]

Options:
  --num_gpus N        Number of GPUs to use (default: 1)
  --gpu_ids "X,Y"     Comma-separated GPU IDs (default: "2")
  -h, --help          Show this help message

Examples:
  Single GPU (default):     ./run_dense_finetune_block_20sparsity.sh
  Single GPU (GPU 3):       ./run_dense_finetune_block_20sparsity.sh --num_gpus 1 --gpu_ids "3"
  Dual GPU (GPU 2,3):       ./run_dense_finetune_block_20sparsity.sh --num_gpus 2 --gpu_ids "2,3"
  Quad GPU (GPU 0-3):       ./run_dense_finetune_block_20sparsity.sh --num_gpus 4 --gpu_ids "0,1,2,3"

Performance:
  1 GPU: ~24-36 hours
  2 GPUs: ~14-20 hours (1.7x speedup)
  4 GPUs: ~8-12 hours (2.5-3x speedup)
```

---

## 🔍 训练过程监控

### 训练开始时的输出

```
==========================================
Dense Fine-tuning Configuration
==========================================
Method: Full fine-tuning with SparseTrainer
Pruned model: out/llama2_7b/block_16x16_20sparsity/wanda/pruned_model
Output directory: out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model
Dataset: wikitext (wikitext-2-raw-v1)

GPU Configuration:
  Number of GPUs: 2
  GPU IDs: 2,3
  Training mode: Multi-GPU (torchrun)

Training Parameters:
  Number of epochs: 3
  Learning rate: 5e-5
  Per-device batch size: 1
  Gradient accumulation: 4
  Effective batch size: 8
  Block size: 1024
  Max train samples: 30000
==========================================

🚀 Starting dense fine-tuning with SparseTrainer...
Training mode: 2 GPU(s) on device(s): 2,3
Effective batch size: 8
==========================================
Using multi-GPU training with torchrun (2 GPUs)...
```

### 训练过程中的输出

```
*** Starting sparse full fine-tuning ***
Initial model sparsity: 0.2000
Trainable params: 6,738,415,616 || All params: 6,738,415,616 || Trainable%: 100.00

Step 10: loss=2.5432
Step 20: loss=2.3456
Step 30: loss=2.1234
...
```

---

## ⚠️ 注意事项

### 1. GPU内存检查

脚本会自动检查GPU内存:

```bash
Available GPU memory: 81920 MB (~80 GB)
```

如果内存不足 (<40GB),会提示:

```
⚠️  Warning: GPU memory may be insufficient!
Consider using LoRA fine-tuning instead.
Continue anyway? (y/n)
```

### 2. 多卡训练要求

- 所有GPU必须有足够的显存 (≥40GB)
- GPU之间需要高速互联 (NVLink推荐)
- 需要安装 `torch.distributed`

### 3. 学习率调整

如果修改了有效batch size,可能需要调整学习率:

```bash
# 线性缩放规则
# 如果有效batch size从8增大到16
# 学习率从5e-5增大到1e-4
```

---

## 📝 输出文件

训练完成后,模型保存在:

```
out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model/
├── config.json
├── generation_config.json
├── pytorch_model.bin  (~13GB)
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── trainer_state.json
```

---

## 🎯 下一步

训练完成后:

1. **评估模型**:
   ```bash
   python main_block.py --model out/llama2_7b/block_16x16_20sparsity/wanda/dense_finetuned_model --eval_zero_shot
   ```

2. **对比性能**:
   - 剪枝后: 困惑度 135.44
   - LoRA微调: 困惑度 56.82
   - 全量微调: 困惑度 40-50 (预期)

3. **验证稀疏性**:
   ```python
   from dense_ft.sparse_trainer import check_sparsity
   model = AutoModelForCausalLM.from_pretrained("out/.../dense_finetuned_model")
   sparsity = check_sparsity(model)
   print(f"Sparsity: {sparsity:.4f}")  # 应该仍然是 ~0.20
   ```

---

## 🆚 LoRA vs 全量微调对比

| 特性 | LoRA微调 | 全量微调 |
|------|---------|---------|
| 可训练参数 | <1% (~4M) | 100% (~7B) |
| GPU内存 | ~20GB | ~55-65GB |
| 训练时间 (单卡) | ~12h | ~24-36h |
| 训练时间 (双卡) | ~7h | ~14-20h |
| 效果 | 好 (56.82) | 更好 (40-50预期) |
| 保存大小 | ~16MB | ~13GB |
| 推荐场景 | 快速实验 | 追求最佳性能 |

---

## 🐛 常见问题

### Q1: 如何查看可用的GPU?

```bash
nvidia-smi
```

### Q2: 如何停止训练?

按 `Ctrl+C`,模型会保存最后一个checkpoint。

### Q3: 如何从checkpoint恢复训练?

脚本会自动检测并从最后的checkpoint恢复。

### Q4: 多卡训练比单卡慢?

检查:
- GPU之间是否有NVLink连接
- 是否有其他进程占用GPU
- 网络通信是否正常

---

## 📚 相关文档

- [LoRA微调指南](FINETUNING_GUIDE_CN.md)
- [剪枝指南](md/prune/QUICKSTART.md)
- [项目概览](md/prune/PROJECT_OVERVIEW.md)

