# Wanda 支持的模型列表

本文档列出了 Wanda 剪枝框架支持的所有 LLM 模型。

---

## 📋 支持的模型架构

Wanda 使用 Hugging Face 的 `AutoModelForCausalLM`，理论上支持所有 Causal LM 架构。以下是经过测试和验证的模型：

### ✅ 官方支持（已测试）

| 模型系列 | 模型大小 | Hugging Face 模型 ID | 脚本 |
|---------|---------|---------------------|------|
| **LLaMA-1** | 7B | `decapoda-research/llama-7b-hf` | `scripts/llama_7b.sh` |
| **LLaMA-1** | 13B | `decapoda-research/llama-13b-hf` | `scripts/llama_13b.sh` |
| **LLaMA-1** | 30B | `decapoda-research/llama-30b-hf` | `scripts/llama_30b.sh` |
| **LLaMA-1** | 65B | `decapoda-research/llama-65b-hf` | `scripts/llama_65b.sh` |
| **LLaMA-2** | 7B | `meta-llama/Llama-2-7b-hf` | ✅ |
| **LLaMA-2** | 13B | `meta-llama/Llama-2-13b-hf` | ✅ |
| **LLaMA-2** | 70B | `meta-llama/Llama-2-70b-hf` | ✅ |
| **OPT** | 125M - 66B | `facebook/opt-*` | `main_opt.py` |

### 🔧 理论支持（未官方测试，但应该可用）

基于 `AutoModelForCausalLM` 的任何模型都应该可以工作，包括：

| 模型系列 | 示例模型 ID | 说明 |
|---------|------------|------|
| **LLaMA-3** | `meta-llama/Meta-Llama-3-8B` | 最新的 LLaMA 系列 |
| **Mistral** | `mistralai/Mistral-7B-v0.1` | Mistral AI 的开源模型 |
| **Mixtral** | `mistralai/Mixtral-8x7B-v0.1` | MoE 架构 |
| **Qwen** | `Qwen/Qwen-7B` | 阿里巴巴的通义千问 |
| **Baichuan** | `baichuan-inc/Baichuan2-7B-Base` | 百川智能 |
| **Yi** | `01-ai/Yi-6B` | 零一万物 |
| **DeepSeek** | `deepseek-ai/deepseek-llm-7b-base` | DeepSeek |
| **Phi** | `microsoft/phi-2` | Microsoft 小模型 |
| **Gemma** | `google/gemma-7b` | Google |
| **BLOOM** | `bigscience/bloom-*` | BigScience |
| **GPT-Neo/J** | `EleutherAI/gpt-neo-*` | EleutherAI |
| **Falcon** | `tiiuae/falcon-*` | TII |

---

## 🚀 使用方法

### 1️⃣ LLaMA 系列

#### LLaMA-1
```bash
python main.py \
    --model decapoda-research/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/
```

#### LLaMA-2
```bash
python main.py \
    --model meta-llama/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/
```

#### LLaMA-3（理论支持）
```bash
python main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama3_8b/unstructured/wanda/
```

### 2️⃣ OPT 系列

使用专门的 `main_opt.py` 脚本：

```bash
python main_opt.py \
    --model facebook/opt-6.7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/opt_6.7b/unstructured/wanda/
```

**可用的 OPT 模型**：
- `facebook/opt-125m`
- `facebook/opt-350m`
- `facebook/opt-1.3b`
- `facebook/opt-2.7b`
- `facebook/opt-6.7b`
- `facebook/opt-13b`
- `facebook/opt-30b`
- `facebook/opt-66b`

### 3️⃣ 其他模型（通用方法）

对于其他基于 Transformer 的 Causal LM 模型，直接使用 `main.py`：

```bash
python main.py \
    --model <huggingface_model_id> \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/<model_name>/unstructured/wanda/
```

---

## 🔍 模型架构要求

Wanda 对模型架构有以下要求：

### ✅ 必须满足

1. **Causal Language Model**: 模型必须是自回归的因果语言模型
2. **Transformer 架构**: 基于 Transformer 的架构
3. **线性层**: 包含标准的 `nn.Linear` 层（用于剪枝）
4. **Hugging Face 兼容**: 可以通过 `AutoModelForCausalLM.from_pretrained()` 加载

### ⚠️ 架构差异处理

不同模型架构的层命名可能不同：

| 模型 | 层访问路径 | 说明 |
|------|-----------|------|
| **LLaMA** | `model.layers` | 标准 Transformer 层 |
| **OPT** | `model.decoder.layers` | Decoder-only 架构 |
| **BLOOM** | `transformer.h` | GPT 风格命名 |
| **GPT-Neo** | `transformer.h` | GPT 风格命名 |

**解决方案**：
- LLaMA 系列使用 `main.py` 和 `lib/prune.py`
- OPT 系列使用 `main_opt.py` 和 `lib/prune_opt.py`
- 其他模型可能需要修改层访问路径

---

## 🧪 测试新模型

如果你想在新模型上使用 Wanda，按以下步骤测试：

### 步骤 1: 检查模型兼容性

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "your-model-id"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 检查模型结构
print(model)

# 查找 Transformer 层
# LLaMA: model.layers
# OPT: model.decoder.layers
# BLOOM: transformer.h
```

### 步骤 2: 小规模测试

```bash
python main.py \
    --model your-model-id \
    --prune_method wanda \
    --sparsity_ratio 0.1 \
    --sparsity_type unstructured \
    --nsamples 16 \
    --save out/test/
```

### 步骤 3: 检查剪枝结果

```python
from lib.prune import check_sparsity

# 加载剪枝后的模型
pruned_model = AutoModelForCausalLM.from_pretrained("out/test/")

# 检查稀疏度
sparsity = check_sparsity(pruned_model)
print(f"Sparsity: {sparsity}")
```

---

## 📊 性能基准

### LLaMA-2 性能（WikiText2 PPL）

| 模型 | Dense | Magnitude 50% | Wanda 50% | SparseGPT 50% |
|------|-------|---------------|-----------|---------------|
| LLaMA-2-7B | 5.12 | 14.89 | **6.29** | 6.15 |
| LLaMA-2-13B | 4.57 | 6.37 | **5.01** | 4.95 |
| LLaMA-2-70B | 3.12 | 4.98 | **3.56** | 3.49 |

### LLaMA-1 性能（WikiText2 PPL）

| 模型 | Dense | Magnitude 50% | Wanda 50% | SparseGPT 50% |
|------|-------|---------------|-----------|---------------|
| LLaMA-7B | 5.68 | 15.87 | **6.96** | 6.61 |
| LLaMA-13B | 5.09 | 7.75 | **5.59** | 5.50 |
| LLaMA-30B | 4.10 | 21.18 | **4.60** | 4.48 |
| LLaMA-65B | 3.53 | 4.48 | **3.80** | 3.73 |

---

## ⚙️ 混合三层剪枝支持

你的项目扩展了 Wanda，支持混合三层剪枝模式：

### 支持的模型

所有支持标准 Wanda 的模型都支持混合三层剪枝：

```bash
python main_block_three_tier.py \
    --model meta-llama/Llama-2-7b-hf \
    --sparsity_ratios 0.35 0.45 0.2 \
    --save out/llama2_7b/block_16x16_three_tier/
```

**三层模式**：
1. **Tier 1 (Dense)**: 35% 的块保持密集
2. **Tier 2 (2:4 Sparse)**: 45% 的块使用 2:4 稀疏
3. **Tier 3 (Top-K)**: 20% 的块使用极度稀疏（Top-K）

---

## 🔧 添加新模型支持

如果你的模型不在支持列表中，可以按以下步骤添加：

### 1. 确定层访问路径

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-model")

# 打印模型结构
print(model)

# 常见路径：
# - model.layers (LLaMA, Mistral, Qwen)
# - model.decoder.layers (OPT)
# - transformer.h (BLOOM, GPT-Neo)
# - model.model.layers (某些模型)
```

### 2. 修改 `lib/prune.py`

如果层路径不是 `model.layers`，需要修改：

```python
# 在 prune_wanda() 函数中
# 原代码：
layers = model.model.layers

# 修改为：
layers = model.model.decoder.layers  # 对于 OPT
# 或
layers = model.transformer.h  # 对于 BLOOM/GPT-Neo
```

### 3. 测试并提交

测试成功后，欢迎提交 PR 或 Issue！

---

## 📚 参考资料

- [Wanda 论文](https://arxiv.org/abs/2306.11695)
- [Wanda GitHub](https://github.com/locuslab/wanda)
- [Hugging Face Models](https://huggingface.co/models)
- [Transformers 文档](https://huggingface.co/docs/transformers)

---

## 💡 常见问题

### Q1: 我的模型不在列表中，能用 Wanda 吗？

**A**: 如果你的模型是基于 Transformer 的 Causal LM，很可能可以使用。先用小规模测试（`--sparsity_ratio 0.1 --nsamples 16`）验证。

### Q2: OPT 和 LLaMA 有什么区别？

**A**: 主要是层访问路径不同：
- LLaMA: `model.layers`
- OPT: `model.decoder.layers`

因此 OPT 需要使用专门的 `main_opt.py`。

### Q3: 支持 Encoder-Decoder 模型（如 T5）吗？

**A**: 不支持。Wanda 专门为 Decoder-only 的 Causal LM 设计。

### Q4: 支持量化模型吗？

**A**: 理论上支持，但需要确保模型可以加载为 `float16`。量化模型（如 GPTQ、AWQ）可能需要额外处理。

### Q5: 多 GPU 支持吗？

**A**: 支持！使用 `device_map="auto"` 自动分配。对于 30B+ 模型，代码会自动使用多 GPU。

---

**最后更新**: 2025-01-XX  
**维护者**: Jiajun Ji  
**项目**: Wanda Hybrid Pruning

