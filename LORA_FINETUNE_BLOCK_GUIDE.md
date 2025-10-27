# LoRA微调块剪枝模型指南

## 📋 概述

本指南介绍如何使用LoRA微调来恢复16x16块剪枝模型的精度。

## 🎯 为什么需要LoRA微调？

### 当前问题
- **块剪枝后困惑度**: 8207.5186 ❌ 异常高
- **预期困惑度**: 6.5-7.0
- **原因**: 块剪枝过于激进，需要微调恢复精度

### LoRA微调优势

| 特性 | 说明 |
|------|------|
| **训练参数** | <1% (只训练LoRA层) |
| **显存需求** | ~20GB |
| **训练时间** | ~12小时 (30000样本) |
| **性能恢复** | 预期恢复80-90% |
| **保持稀疏** | ✅ 自动保持块稀疏结构 |

## 🚀 快速开始

### 步骤1: 安装依赖

```bash
# 激活conda环境
conda activate prune_llm

# 安装PEFT库（包含LoRA）
pip install peft

# 验证安装
python -c "from peft import LoraConfig; print('✅ PEFT installed')"
```

### 步骤2: 运行LoRA微调

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda

# 运行LoRA微调（预计12小时）
./run_lora_finetune_block.sh
```

### 步骤3: 评估微调后的模型

```bash
# 评估LoRA微调后的模型
./evaluate_lora_block.sh
```

## 📝 详细配置

### LoRA微调参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model_name_or_path` | `out/llama2_7b/block_16x16/wanda/pruned_model` | 块剪枝模型路径 |
| `--config_name` | `meta-llama/Llama-2-7b-hf` | 模型配置 |
| `--dataset_name` | `c4` | 训练数据集 |
| `--num_train_epochs` | `1` | 训练轮数 |
| `--block_size` | `1024` | 上下文长度 |
| `--max_train_samples` | `30000` | 训练样本数 |
| `--learning_rate` | `1e-4` | 学习率 |
| `lora_r` | `8` | LoRA秩 |
| `lora_alpha` | `16` | LoRA缩放因子 |
| `target_modules` | `["q_proj", "v_proj"]` | 目标模块 |

### 自定义参数

如果需要调整参数，可以直接修改 `run_lora_finetune_block.sh`：

```bash
# 快速测试（1000样本，约30分钟）
MAX_TRAIN_SAMPLES=1000

# 标准训练（30000样本，约12小时）
MAX_TRAIN_SAMPLES=30000

# 完整训练（更多样本，更长时间）
MAX_TRAIN_SAMPLES=100000

# 如果有80GB GPU，可以增加上下文长度
BLOCK_SIZE=2048
```

## 🔍 LoRA原理

### 工作机制

```
原始稀疏权重 (冻结，不更新)
    ↓
    + LoRA低秩矩阵 (可训练)
    ↓
输出 = W_sparse × X + (B × A) × X
```

其中：
- `W_sparse`: 块剪枝后的稀疏权重（冻结）
- `B × A`: LoRA低秩分解矩阵（可训练）
- `B`: [d, r], `A`: [r, k]，其中 r << min(d, k)

### 关键代码

<augment_code_snippet path="wanda/lora_ft/finetune_lm.py" mode="EXCERPT">
```python
# LoRA配置
config = LoraConfig(
    r=8,                              # LoRA秩
    lora_alpha=16,                    # 缩放因子
    target_modules=["q_proj","v_proj"], # 目标模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 应用LoRA
model = get_peft_model(model, config)
```
</augment_code_snippet>

## 📊 预期结果

### 性能恢复预期

| 阶段 | 困惑度 | 说明 |
|------|--------|------|
| 块剪枝后 | 8207.52 | ❌ 当前状态 |
| LoRA微调后 | 7.0-8.0 | ✅ 预期恢复 |
| 理想目标 | 6.5-7.0 | 🎯 最佳情况 |

### 对比参考

| 模型 | 稀疏度 | 困惑度 |
|------|--------|--------|
| Dense基线 | 0% | 5.12 |
| 非结构化50% | 50% | 6.31 |
| 非结构化50% + LoRA | 50% | ~5.8 |
| 块16x16 50% | 50% | 8207.52 |
| 块16x16 50% + LoRA | 50% | ? (待测试) |

## 🔧 故障排除

### 问题1: PEFT未安装

```bash
# 错误信息
ModuleNotFoundError: No module named 'peft'

# 解决方案
pip install peft
```

### 问题2: 显存不足

```bash
# 错误信息
CUDA out of memory

# 解决方案1: 减小batch size
per_device_train_batch_size=1  # 已经是最小

# 解决方案2: 减小上下文长度
BLOCK_SIZE=512  # 从1024减到512

# 解决方案3: 使用梯度累积
gradient_accumulation_steps=4
```

### 问题3: 数据集下载失败

```bash
# 错误信息
ConnectionError: Couldn't reach the Hugging Face Hub

# 解决方案: 使用本地数据集或代理
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题4: 训练时间过长

```bash
# 快速测试方案
MAX_TRAIN_SAMPLES=1000  # 减少到1000样本
NUM_EPOCHS=1            # 保持1轮

# 预期时间: ~30分钟
```

## 📈 监控训练

### 训练日志

训练过程中会输出：
```
***** Running training *****
  Num examples = 30000
  Num Epochs = 1
  Total optimization steps = 30000
  
Step 100/30000 | Loss: 2.345 | LR: 1e-4
Step 200/30000 | Loss: 2.123 | LR: 1e-4
...
```

### 评估指标

```
***** Running evaluation *****
  Num examples = 128
  
Eval Loss: 2.045
Eval Perplexity: 7.732
```

## 🎯 下一步

### 如果LoRA效果好
```bash
# 1. 保存最终模型
# LoRA权重已保存在: out/llama2_7b/block_16x16/wanda/lora_weights

# 2. 可选: 合并LoRA权重到基础模型
# (需要额外脚本)

# 3. 部署使用
# 加载基础模型 + LoRA权重
```

### 如果LoRA效果不佳
```bash
# 选项1: 增加训练样本
MAX_TRAIN_SAMPLES=100000

# 选项2: 调整LoRA参数
lora_r=16  # 增加秩
lora_alpha=32

# 选项3: 尝试Dense微调
# (需要更多显存和时间)

# 选项4: 调整块大小重新剪枝
python main_block.py --block_size 8  # 更小的块
```

## 💡 关键要点

1. **LoRA不修改原始权重**: 块稀疏结构完全保持
2. **参数高效**: 只训练<1%的参数
3. **易于部署**: LoRA权重可独立保存和加载
4. **多任务适配**: 可为不同任务训练不同LoRA

## 📚 参考资料

- **LoRA论文**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Wanda论文**: [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)
- **PEFT库**: [Hugging Face PEFT](https://github.com/huggingface/peft)

## 🔗 相关脚本

| 脚本 | 说明 |
|------|------|
| `run_lora_finetune_block.sh` | 运行LoRA微调 |
| `evaluate_lora_block.sh` | 评估LoRA微调后的模型 |
| `lora_ft/finetune_lm.py` | LoRA微调主脚本 |
| `lora_ft/evaluate_ppl.py` | 评估脚本 |

