# GSM8K 剪枝使用指南

使用 GSM8K 数据集作为校准数据进行剪枝。

---

## 快速开始

### 1. 测试数据加载

```bash
cd wanda
python test_gsm8k_data.py
```

### 2. 使用 GSM8K 剪枝

```bash
# 方式 1: 使用脚本
./run_prune_with_gsm8k.sh

# 方式 2: 直接命令
python main_block_three_tier.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --calib_dataset gsm8k \
    --nsamples 128 \
    --save out/llama2_7b/gsm8k_calibrated
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--calib_dataset` | `wikitext2` | 校准数据集：`wikitext2`, `c4`, `gsm8k` |
| `--nsamples` | `128` | 校准样本数 |

---

## 对比实验

```bash
# WikiText2 剪枝（通用）
python main_block_three_tier.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --calib_dataset wikitext2 \
    --save out/llama2_7b/wikitext2_calibrated

# GSM8K 剪枝（数学任务优化）
python main_block_three_tier.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --calib_dataset gsm8k \
    --save out/llama2_7b/gsm8k_calibrated
```

---

## 注意事项

- ✅ 不影响已有功能（默认仍使用 wikitext2）
- ✅ GSM8K 剪枝适合数学推理任务
- ⚠️ 可能在其他任务上性能下降
- 💡 推荐：wikitext2 剪枝 + gsm8k 微调

---

## 实现细节

### 数据格式

```python
# GSM8K 样本格式
text = f"Question: {question}\nAnswer: {answer}"
```

### 修改的文件

1. `lib/data.py` - 添加 `get_gsm8k()` 函数
2. `main_block_three_tier.py` - 添加 `--calib_dataset` 参数
3. `lib/prune.py` - 支持自定义数据集

---

**维护者**: Jiajun Ji  
**项目**: Wanda Hybrid Pruning

