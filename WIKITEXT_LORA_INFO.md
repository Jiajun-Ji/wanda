# WikiText数据集LoRA微调说明

## 📊 WikiText vs C4 数据集对比

| 特性 | WikiText-2 | C4 |
|------|-----------|-----|
| **训练样本数** | ~36K tokens (~2-3K sequences) | 数百万sequences |
| **数据质量** | 高质量维基百科文章 | 网页爬取数据 |
| **训练时间** | ~1-2小时 | ~12小时 |
| **适用场景** | 快速验证、学术研究 | 生产部署 |

## ⚠️ 重要提示

### WikiText数据集较小

WikiText-2训练集只有约**2-3K个序列**（取决于分词方式），远小于C4数据集。

**影响**：
- ✅ **训练快**: 1-2小时即可完成
- ⚠️ **样本少**: 可能无法充分微调
- ⚠️ **过拟合风险**: 容易过拟合到WikiText

### 建议的训练样本数

```bash
# WikiText-2 实际可用样本数
# 训练集: ~2-3K sequences (block_size=1024)
# 验证集: ~200 sequences

# 推荐设置
MAX_TRAIN_SAMPLES=2000   # 使用全部训练数据
MAX_EVAL_SAMPLES=128     # 验证样本
```

## 🚀 使用方法

### 方法1: 使用提供的脚本（已配置WikiText）

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda

# 直接运行（已配置为WikiText）
./run_lora_finetune_block.sh
```

### 方法2: 手动运行

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda/lora_ft

python finetune_lm.py \
    --model_name_or_path ../out/llama2_7b/block_16x16/wanda/pruned_model \
    --config_name meta-llama/Llama-2-7b-hf \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --num_train_epochs 1 \
    --block_size 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --max_train_samples 2000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir ../out/llama2_7b/block_16x16/wanda/lora_weights_wikitext
```

## 📝 配置说明

### 当前配置

```bash
DATASET="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
MAX_TRAIN_SAMPLES=30000  # 实际会被限制为~2000
OUTPUT_DIR="out/llama2_7b/block_16x16/wanda/lora_weights_wikitext"
```

### 推荐调整

由于WikiText数据集较小，建议：

```bash
# 选项1: 多轮训练
NUM_EPOCHS=3  # 增加到3轮

# 选项2: 调整学习率
LEARNING_RATE=5e-5  # 降低学习率避免过拟合

# 选项3: 增加LoRA秩
# 修改 finetune_lm.py 中的 lora_r=16
```

## 🎯 预期效果

### WikiText微调预期

| 阶段 | 困惑度 | 说明 |
|------|--------|------|
| 块剪枝后 | 8207.52 | ❌ 当前状态 |
| WikiText LoRA (1 epoch) | 50-100 | ⚠️ 可能不够 |
| WikiText LoRA (3 epochs) | 20-50 | ✅ 预期改善 |
| 理想目标 | 6.5-7.0 | 🎯 需要更多数据 |

**注意**: WikiText数据量较小，可能无法完全恢复到理想困惑度。

## 💡 建议

### 如果WikiText效果不佳

#### 选项1: 使用C4数据集（推荐）

```bash
# 修改 run_lora_finetune_block.sh
DATASET="c4"
# 删除 DATASET_CONFIG 行
MAX_TRAIN_SAMPLES=30000
```

#### 选项2: 增加训练轮数

```bash
NUM_EPOCHS=5  # WikiText上训练5轮
```

#### 选项3: 混合数据集

```python
# 需要修改 finetune_lm.py
# 同时使用 WikiText + C4
```

## 📊 数据集详细信息

### WikiText-2-raw-v1

```
训练集:
- 原始tokens: ~2,088,628
- Sequences (block_size=1024): ~2,000
- Sequences (block_size=2048): ~1,000

验证集:
- 原始tokens: ~217,646
- Sequences (block_size=1024): ~200

测试集:
- 原始tokens: ~245,569
- Sequences (block_size=1024): ~240
```

### C4 (对比)

```
训练集:
- 数百万sequences
- 可以设置任意 max_train_samples

验证集:
- 数千sequences
```

## 🔧 故障排除

### 问题1: 训练样本不足警告

```
Warning: max_train_samples (30000) is larger than dataset size (2000)
Using all available samples: 2000
```

**解决**: 这是正常的，会自动使用全部可用样本。

### 问题2: 过拟合

```
Train Loss: 0.5
Eval Loss: 2.5  # 远大于训练loss
```

**解决**:
```bash
# 降低学习率
LEARNING_RATE=5e-5

# 增加dropout
# 修改 finetune_lm.py 中的 lora_dropout=0.1
```

### 问题3: 效果不佳

**解决**: 切换到C4数据集
```bash
# 修改 run_lora_finetune_block.sh
DATASET="c4"
MAX_TRAIN_SAMPLES=30000
```

## 📈 监控建议

### 训练过程

```bash
# 观察训练loss是否下降
Step 100/2000 | Train Loss: 3.5
Step 200/2000 | Train Loss: 2.8
Step 500/2000 | Train Loss: 2.1
...

# 观察评估loss
Eval Loss: 2.3
Eval Perplexity: 9.97
```

### 判断标准

- ✅ **训练loss持续下降**: 模型在学习
- ⚠️ **训练loss下降，eval loss上升**: 过拟合
- ❌ **训练loss不下降**: 学习率过高或数据问题

## 🎓 总结

### WikiText优势
- ✅ 训练快（1-2小时）
- ✅ 数据质量高
- ✅ 适合快速验证

### WikiText劣势
- ❌ 数据量小
- ❌ 可能无法充分微调
- ❌ 容易过拟合

### 推荐策略

1. **快速验证**: 先用WikiText测试（1-2小时）
2. **查看效果**: 如果困惑度降到50以下，说明有效
3. **完整训练**: 切换到C4进行完整训练（12小时）

## 🔗 相关文件

| 文件 | 说明 |
|------|------|
| `run_lora_finetune_block.sh` | 已配置WikiText |
| `evaluate_lora_block.sh` | 评估脚本 |
| `LORA_FINETUNE_BLOCK_GUIDE.md` | 完整指南 |

