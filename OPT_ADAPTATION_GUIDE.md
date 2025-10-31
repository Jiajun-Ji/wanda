# OPT 模型适配指南

## 📋 概述

本指南说明如何将 Progressive Three-Tier Pruning 适配到 OPT 模型，并支持不同的校准/微调数据集（WikiText2 和 C4）。

---

## 🔍 关键架构差异

### Llama vs OPT

| 组件 | Llama | OPT |
|------|-------|-----|
| **层访问路径** | `model.model.layers` | `model.model.decoder.layers` |
| **层数量获取** | `len(model.model.layers)` | `len(model.model.decoder.layers)` |
| **Embedding** | `model.embed_tokens` | `model.embed_tokens` |
| **Position IDs** | ✅ 需要（RoPE） | ❌ 不需要（Learned PE） |
| **Device Map Key** | `model.layers.{i}` | `model.decoder.layers.{i}` |

---

## 📁 需要修改/创建的文件

### 1. ✅ 已创建的文件

#### `run_progressive_three_tier_universal.sh`
- **功能**：统一的启动脚本，支持 Llama 和 OPT
- **配置项**：
  ```bash
  MODEL_TYPE="opt"  # 或 "llama"
  DATASET="wikitext2"  # 或 "c4"
  BASE_MODEL="/path/to/opt-model"
  ```

#### `main_progressive_three_tier_opt.py`
- **功能**：OPT 版本的 progressive pruning 主脚本
- **导入**：使用 `lib.prune_opt` 而不是 `lib.prune`

---

### 2. ⚠️ 需要修改的文件

#### `lib/prune_opt.py`
**需要添加的函数**：

```python
def prune_wanda_progressive_three_tier_opt(
    args, model, tokenizer, device,
    block_size,
    target_dense_ratio,
    target_2_4_ratio,
    target_topk_ratio,
    dense_to_2_4_ratio,
    mid_2_4_to_topk_ratio,
    topk_per_block,
    previous_tier_maps=None
):
    """
    Progressive three-tier pruning for OPT models.
    
    Key differences from Llama version:
    1. Use model.model.decoder.layers instead of model.model.layers
    2. No position_ids in forward pass
    3. Different device map keys
    """
    # Implementation similar to prune_wanda_progressive_three_tier in lib/prune.py
    # but adapted for OPT architecture
```

**需要添加的辅助函数**：

```python
def save_tier_map(tier_maps, filepath):
    """Save tier maps to file."""
    torch.save({
        'tier_map': tier_maps,
    }, filepath)

def load_tier_map(filepath):
    """Load tier maps from file."""
    data = torch.load(filepath)
    return data['tier_map']
```

---

## 🔧 具体修改步骤

### 步骤 1：在 `lib/prune_opt.py` 中添加函数

需要从 `lib/prune.py` 复制以下函数并修改：

1. **`prune_wanda_progressive_three_tier_opt()`**
   - 复制 `prune_wanda_progressive_three_tier()` 的实现
   - 修改所有 `model.model.layers` → `model.model.decoder.layers`
   - 移除所有 `position_ids` 相关代码
   - 修改 device map 检查：`f"model.layers.{i}"` → `f"model.decoder.layers.{i}"`

2. **`save_tier_map()` 和 `load_tier_map()`**
   - 直接复制即可，无需修改

### 步骤 2：修改关键代码段

#### 原始代码（Llama）：
```python
# lib/prune.py
layers = model.model.layers

# Forward pass
outs[j] = layer(inps[j].unsqueeze(0), 
                attention_mask=attention_mask, 
                position_ids=position_ids)[0]

# Device map check
if f"model.layers.{i}" in model.hf_device_map:
    dev = model.hf_device_map[f"model.layers.{i}"]
```

#### 修改后代码（OPT）：
```python
# lib/prune_opt.py
layers = model.model.decoder.layers

# Forward pass (no position_ids)
outs[j] = layer(inps[j].unsqueeze(0), 
                attention_mask=attention_mask)[0]

# Device map check
if f"model.decoder.layers.{i}" in model.hf_device_map:
    dev = model.hf_device_map[f"model.decoder.layers.{i}"]
```

---

## 🚀 使用方法

### 方案 A：使用统一脚本（推荐）

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda

# 1. 编辑 run_progressive_three_tier_universal.sh
# 修改以下配置：
MODEL_TYPE="opt"
BASE_MODEL="/mnt/sdb/llm_models/opt-1.3b"
DATASET="wikitext2"  # 第一次运行

# 2. 运行
chmod +x run_progressive_three_tier_universal.sh
./run_progressive_three_tier_universal.sh

# 3. 第二次运行（使用 C4）
# 修改配置：
DATASET="c4"
OUTPUT_BASE="out/progressive_three_tier_opt_c4"

# 4. 再次运行
./run_progressive_three_tier_universal.sh
```

### 方案 B：手动运行每个步骤

```bash
# Iteration 1
CUDA_VISIBLE_DEVICES=0 python main_progressive_three_tier_opt.py \
    --model /mnt/sdb/llm_models/opt-1.3b \
    --iteration 1 \
    --config progressive_config.csv \
    --block_size 16 \
    --topk_per_block 15 \
    --save out/progressive_three_tier_opt_wikitext2/iter1/ \
    --save_model out/progressive_three_tier_opt_wikitext2/iter1/pruned_model

# Finetune
cd dense_ft
CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 finetune_sparse_model.py \
    --model_name_or_path ../out/progressive_three_tier_opt_wikitext2/iter1/pruned_model \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --bf16 \
    --output_dir ../out/progressive_three_tier_opt_wikitext2/iter1/finetuned_model

# ... 重复 iteration 2-5
```

---

## 📊 数据集配置

### WikiText2
```bash
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
```

### C4
```bash
DATASET_NAME="allenai/c4"
DATASET_CONFIG="en"
```

**注意**：C4 数据集较大，首次加载可能需要较长时间。

---

## ✅ 验证清单

完成修改后，请检查：

- [ ] `lib/prune_opt.py` 中添加了 `prune_wanda_progressive_three_tier_opt()`
- [ ] `lib/prune_opt.py` 中添加了 `save_tier_map()` 和 `load_tier_map()`
- [ ] 所有 `model.model.layers` 改为 `model.model.decoder.layers`
- [ ] 移除了所有 `position_ids` 参数
- [ ] Device map 检查使用 `model.decoder.layers.{i}`
- [ ] `run_progressive_three_tier_universal.sh` 可执行权限已设置
- [ ] 模型路径和数据集配置正确

---

## 🐛 常见问题

### Q1: 报错 `AttributeError: 'OPTForCausalLM' object has no attribute 'layers'`
**原因**：使用了 Llama 的层访问路径  
**解决**：改为 `model.model.decoder.layers`

### Q2: 报错 `TypeError: forward() got an unexpected keyword argument 'position_ids'`
**原因**：OPT 不使用 position_ids  
**解决**：移除 forward 调用中的 `position_ids=position_ids`

### Q3: C4 数据集加载很慢
**原因**：C4 数据集较大（~300GB）  
**解决**：
- 首次加载会下载并缓存
- 可以使用 `--max_train_samples` 限制样本数量
- 或者先用 WikiText2 测试流程

---

## 📝 总结

### 需要做的修改

1. **创建文件**（已完成）：
   - ✅ `run_progressive_three_tier_universal.sh`
   - ✅ `main_progressive_three_tier_opt.py`

2. **修改文件**（待完成）：
   - ⚠️ `lib/prune_opt.py`：添加 progressive three-tier 函数

3. **配置修改**：
   - 修改 `run_progressive_three_tier_universal.sh` 中的：
     - `MODEL_TYPE`
     - `BASE_MODEL`
     - `DATASET`

### 运行两次的配置

#### 第一次：OPT + WikiText2
```bash
MODEL_TYPE="opt"
BASE_MODEL="/mnt/sdb/llm_models/opt-1.3b"
DATASET="wikitext2"
OUTPUT_BASE="out/progressive_three_tier_opt_wikitext2"
```

#### 第二次：OPT + C4
```bash
MODEL_TYPE="opt"
BASE_MODEL="/mnt/sdb/llm_models/opt-1.3b"
DATASET="c4"
OUTPUT_BASE="out/progressive_three_tier_opt_c4"
```

---

## 🔗 相关文件

- `lib/prune.py`：Llama 版本的参考实现
- `lib/prune_opt.py`：OPT 版本（需要添加函数）
- `main_progressive_three_tier.py`：Llama 版本的主脚本
- `main_progressive_three_tier_opt.py`：OPT 版本的主脚本
- `progressive_config.csv`：迭代配置（通用）

