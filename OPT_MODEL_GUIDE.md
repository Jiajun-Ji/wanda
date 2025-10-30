# OPT 模型使用指南

本文档说明如何在 Wanda 中使用 Facebook 的 OPT (Open Pre-trained Transformer) 模型。

---

## 🎯 快速回答

**Q: Wanda 使用 OPT 模型需要特殊的 HF 格式吗？**

**A: 不需要！** Facebook 在 Hugging Face 上发布的 OPT 模型已经是标准的 Hugging Face 格式，可以直接使用。

---

## 📦 可用的 OPT 模型

所有 OPT 模型都可以直接从 Hugging Face 下载和使用：

| 模型 ID | 参数量 | 说明 |
|---------|--------|------|
| `facebook/opt-125m` | 125M | 最小模型，适合快速测试 |
| `facebook/opt-350m` | 350M | 小型模型 |
| `facebook/opt-1.3b` | 1.3B | 中小型模型 |
| `facebook/opt-2.7b` | 2.7B | 中型模型 |
| `facebook/opt-6.7b` | 6.7B | 大型模型 |
| `facebook/opt-13b` | 13B | 超大型模型 |
| `facebook/opt-30b` | 30B | 超大型模型（需要多 GPU） |
| `facebook/opt-66b` | 66B | 最大模型（需要多 GPU） |

**官方链接**: https://huggingface.co/facebook

---

## 🚀 使用方法

### 1️⃣ 基本用法

```bash
python main_opt.py \
    --model facebook/opt-6.7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/opt_6.7b/unstructured/wanda/
```

### 2️⃣ 结构化稀疏（2:4）

```bash
python main_opt.py \
    --model facebook/opt-6.7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/opt_6.7b/2-4/wanda/
```

### 3️⃣ 使用本地缓存

```bash
python main_opt.py \
    --model facebook/opt-6.7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --cache_dir /path/to/llm_weights \
    --save out/opt_6.7b/unstructured/wanda/
```

---

## ⚙️ OPT vs LLaMA 的区别

### 关键差异

| 特性 | LLaMA | OPT |
|------|-------|-----|
| **脚本** | `main.py` | `main_opt.py` |
| **剪枝库** | `lib/prune.py` | `lib/prune_opt.py` |
| **层路径** | `model.layers` | `model.decoder.layers` |
| **架构** | Decoder-only | Decoder-only |
| **位置编码** | RoPE | Learned |

### 为什么需要不同的脚本？

OPT 和 LLaMA 的模型结构略有不同，主要体现在：

1. **层访问路径**：
   ```python
   # LLaMA
   layers = model.model.layers
   
   # OPT
   layers = model.model.decoder.layers
   ```

2. **注意力机制**：
   - LLaMA 使用 RoPE (Rotary Position Embedding)
   - OPT 使用传统的 Learned Position Embedding

3. **归一化层**：
   - LLaMA 使用 RMSNorm
   - OPT 使用 LayerNorm

---

## 🧪 测试 OPT 模型

我们提供了一个测试脚本来验证 OPT 模型的兼容性：

```bash
# 测试最小的 OPT 模型
python test_opt_model.py

# 测试特定的 OPT 模型
python test_opt_model.py facebook/opt-1.3b
```

**测试内容**：
- ✅ 模型加载
- ✅ Tokenizer 加载
- ✅ 模型结构检查
- ✅ 推理测试
- ✅ 格式验证

---

## 📊 性能基准

### OPT 模型在 WikiText2 上的 PPL（50% 稀疏度）

| 模型 | Dense PPL | Magnitude | Wanda | SparseGPT |
|------|-----------|-----------|-------|-----------|
| OPT-125M | ~27.65 | ~45.0 | ~30.5 | ~29.8 |
| OPT-1.3B | ~14.62 | ~22.0 | ~16.5 | ~16.0 |
| OPT-6.7B | ~10.86 | ~15.5 | ~12.2 | ~11.8 |
| OPT-13B | ~10.13 | ~14.0 | ~11.5 | ~11.0 |

*注：以上数据为估计值，实际结果可能因配置而异*

---

## 🔍 模型格式说明

### Hugging Face 原生格式

Facebook 发布的 OPT 模型已经是 Hugging Face 的标准格式：

```
facebook/opt-6.7b/
├── config.json              # 模型配置
├── pytorch_model.bin        # 模型权重（旧格式）
├── model.safetensors        # 模型权重（新格式，推荐）
├── tokenizer.json           # Tokenizer 配置
├── tokenizer_config.json    # Tokenizer 元数据
└── special_tokens_map.json  # 特殊 token 映射
```

### 无需转换

- ✅ **不需要**从 Fairseq 格式转换
- ✅ **不需要**从 Megatron 格式转换
- ✅ **不需要**任何预处理
- ✅ 直接使用 `AutoModelForCausalLM.from_pretrained()`

---

## 💡 常见问题

### Q1: OPT 模型需要特殊的 HF 格式吗？

**A**: 不需要。Facebook 在 Hugging Face 上发布的 OPT 模型已经是标准格式。

### Q2: 可以用 `main.py` 剪枝 OPT 吗？

**A**: 不推荐。虽然理论上可能工作，但 `main_opt.py` 专门为 OPT 的层结构优化，使用它更安全。

### Q3: OPT 和 LLaMA 哪个更好？

**A**: 
- **OPT**: 更早发布，社区支持广泛，适合研究
- **LLaMA**: 性能更好，更新的架构，推荐用于生产

### Q4: 可以在 OPT 上使用混合三层剪枝吗？

**A**: 理论上可以，但需要修改 `main_block_three_tier.py` 中的层访问路径：

```python
# 修改前（LLaMA）
layers = model.model.layers

# 修改后（OPT）
layers = model.model.decoder.layers
```

### Q5: OPT 模型支持哪些语言？

**A**: OPT 主要是英文模型，在其他语言上的表现可能不如专门的多语言模型。

---

## 🔧 高级用法

### 1. 多 GPU 剪枝（30B/66B 模型）

```bash
python main_opt.py \
    --model facebook/opt-30b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/opt_30b/unstructured/wanda/
```

代码会自动检测并使用多 GPU：

```python
if "30b" in args.model or "66b" in args.model:
    device = model.hf_device_map["lm_head"]
```

### 2. Zero-Shot 评估

```bash
python main_opt.py \
    --model facebook/opt-6.7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/opt_6.7b/unstructured/wanda/ \
    --eval_zero_shot
```

### 3. 保存剪枝后的模型

```bash
python main_opt.py \
    --model facebook/opt-6.7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/opt_6.7b/unstructured/wanda/ \
    --save_model out/opt_6.7b/unstructured/wanda/pruned_model
```

---

## 📚 参考资料

### 官方资源

- [OPT 论文](https://arxiv.org/abs/2205.01068)
- [OPT GitHub](https://github.com/facebookresearch/metaseq)
- [Hugging Face OPT 模型](https://huggingface.co/facebook)

### Wanda 相关

- [Wanda 论文](https://arxiv.org/abs/2306.11695)
- [Wanda GitHub](https://github.com/locuslab/wanda)

---

## 🛠️ 故障排除

### 问题 1: 模型下载失败

**解决方案**：
```bash
# 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com
python main_opt.py --model facebook/opt-6.7b ...
```

### 问题 2: 内存不足

**解决方案**：
```bash
# 使用更小的模型测试
python main_opt.py --model facebook/opt-125m ...

# 或使用 CPU offload
python main_opt.py --model facebook/opt-6.7b ... --device_map auto
```

### 问题 3: 层路径错误

**症状**: `AttributeError: 'OPTModel' object has no attribute 'layers'`

**解决方案**: 确保使用 `main_opt.py` 而不是 `main.py`

---

## ✅ 检查清单

在使用 OPT 模型前，确认：

- [ ] 使用 `main_opt.py` 脚本
- [ ] 模型 ID 格式正确（`facebook/opt-*`）
- [ ] GPU 内存足够（6.7B 需要约 14GB）
- [ ] 已安装 transformers >= 4.30.0
- [ ] 已安装 accelerate（用于多 GPU）

---

**最后更新**: 2025-01-XX  
**维护者**: Jiajun Ji  
**项目**: Wanda Hybrid Pruning

