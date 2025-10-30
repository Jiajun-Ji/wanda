# SQuAD & GSM8K 评估指南

本文档说明如何在 SQuAD 和 GSM8K 任务上评估原始模型和剪枝微调模型。

---

## 📋 任务说明

### SQuAD v2.0 (Stanford Question Answering Dataset)

- **任务类型**: 阅读理解问答
- **数据集大小**: ~11,873 个测试样本
- **评估方式**: 生成式（需要生成答案文本）
- **指标**: 
  - **Exact Match (EM)**: 精确匹配率
  - **F1 Score**: F1 分数
- **特点**: 包含无法回答的问题（SQuAD 2.0 的特色）

**示例**：
```
Context: "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy..."
Question: "In what country is Normandy located?"
Answer: "France"
```

### GSM8K (Grade School Math 8K)

- **任务类型**: 小学数学推理
- **数据集大小**: 1,319 个测试样本
- **评估方式**: 生成式（需要生成推理过程和答案）
- **指标**: 
  - **Accuracy**: 答案准确率
- **特点**: 需要多步推理的数学问题

**示例**：
```
Question: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
Answer: "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. #### 72"
```

---

## 🚀 快速开始

### 方法 1: 使用 Shell 脚本（推荐）

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda

# 运行评估
./run_eval_squad_gsm8k.sh
```

脚本会：
1. 显示配置信息
2. 询问是否继续（因为评估时间很长）
3. 依次评估原始模型和剪枝模型
4. 生成对比报告

### 方法 2: 直接使用 Python 脚本

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda

python eval_squad_gsm8k_compare.py \
    --original_model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --pruned_model out/llama2_7b/block_16x16_three_tier_0.35_0.45_0.2/wanda/dense_finetuned_model \
    --tasks squadv2 gsm8k \
    --output_dir eval_results_squad_gsm8k
```

---

## ⚙️ 配置选项

### 模型路径

```bash
# 原始模型
--original_model /mnt/sdb/llm_models/Llama-2-7b-hf

# 剪枝微调模型
--pruned_model out/llama2_7b/block_16x16_three_tier_0.35_0.45_0.2/wanda/dense_finetuned_model
```

### 评估任务

```bash
# 默认：SQuAD v2 和 GSM8K
--tasks squadv2 gsm8k

# 只评估 SQuAD
--tasks squadv2

# 只评估 GSM8K
--tasks gsm8k

# 添加其他 GSM8K 变体
--tasks squadv2 gsm8k gsm8k_cot
```

### 其他选项

```bash
# 输出目录
--output_dir eval_results_squad_gsm8k

# 模型缓存目录
--cache_dir llm_weights

# PPL 评估样本数
--nsamples 128

# 随机种子
--seed 0
```

---

## ⏱️ 评估时间估计

### SQuAD v2.0

| 模型大小 | 样本数 | 预计时间 |
|---------|--------|---------|
| 7B | 11,873 | 1.5-2.5 小时 |
| 13B | 11,873 | 2.5-3.5 小时 |
| 70B | 11,873 | 6-8 小时 |

### GSM8K

| 模型大小 | 样本数 | 预计时间 |
|---------|--------|---------|
| 7B | 1,319 | 0.5-1 小时 |
| 13B | 1,319 | 1-1.5 小时 |
| 70B | 1,319 | 2-3 小时 |

### 总时间

- **7B 模型**: 4-7 小时（两个模型）
- **13B 模型**: 7-10 小时（两个模型）
- **70B 模型**: 16-22 小时（两个模型）

**建议**: 使用 `tmux` 或 `screen` 在后台运行，避免 SSH 断开导致评估中断。

---

## 📊 输出结果

### 文件结构

```
eval_results_squad_gsm8k/
├── original_model_results.json      # 原始模型结果（JSON）
├── pruned_model_results.json        # 剪枝模型结果（JSON）
├── comparison_report.md             # 对比报告（Markdown）
└── comparison_results.json          # 对比结果（JSON）
```

### 结果示例

#### JSON 格式

```json
{
  "model_path": "/mnt/sdb/llm_models/Llama-2-7b-hf",
  "sparsity": 0.0,
  "wikitext_ppl": 5.12,
  "tasks": {
    "squadv2": {
      "exact_match": 0.6234,
      "f1": 0.7123
    },
    "gsm8k": {
      "accuracy": 0.1523
    }
  }
}
```

#### Markdown 报告

```markdown
# Model Comparison Report

## Model Information
- Original Model: /mnt/sdb/llm_models/Llama-2-7b-hf
- Pruned Model: out/llama2_7b/.../dense_finetuned_model

## Summary
| Metric | Original | Pruned | Difference |
|--------|----------|--------|------------|
| Sparsity | 0.00% | 65.00% | +65.00% |
| WikiText2 PPL | 5.1200 | 6.8900 | +1.7700 |

## Task Results
| Task | Original | Pruned | Difference | Relative |
|------|----------|--------|------------|----------|
| squadv2 (EM) | 0.6234 | 0.5123 | -0.1111 | -17.82% |
| squadv2 (F1) | 0.7123 | 0.6234 | -0.0889 | -12.48% |
| gsm8k | 0.1523 | 0.0987 | -0.0536 | -35.19% |
```

---

## 🔍 与 BoolQ 等任务的区别

### 任务类型对比

| 特性 | BoolQ/RTE/HellaSwag | SQuAD/GSM8K |
|------|---------------------|-------------|
| **任务类型** | 分类（选择题） | 生成（开放式） |
| **评估方式** | Loglikelihood | Generation |
| **速度** | 快（~1-2 小时） | 慢（~4-8 小时） |
| **指标** | Accuracy | EM/F1/Accuracy |
| **难度** | 相对简单 | 更具挑战性 |

### 为什么 SQuAD/GSM8K 更慢？

1. **生成任务**: 需要生成完整的答案文本，而不是简单的分类
2. **多次前向传播**: 每个 token 都需要一次前向传播
3. **更长的输出**: SQuAD 答案可能很长，GSM8K 需要生成推理过程
4. **更大的样本量**: SQuAD v2 有 11,873 个样本

---

## 💡 使用建议

### 1. 先测试小样本

在正式评估前，先用少量样本测试：

```bash
# 修改 eval_squad_gsm8k_compare.py 中的 limit 参数
# 在 evaluate_model 函数中添加：
results = eval_zero_shot(
    model_name=model_name,
    model=model,
    tokenizer=tokenizer,
    task_list=args.tasks,
    num_fewshot=0,
    limit=100  # 只评估 100 个样本
)
```

### 2. 使用后台运行

```bash
# 使用 tmux
tmux new -s eval_squad_gsm8k
./run_eval_squad_gsm8k.sh
# Ctrl+B, D 分离会话

# 重新连接
tmux attach -t eval_squad_gsm8k
```

### 3. 监控进度

```bash
# 查看输出日志
tail -f eval_results_squad_gsm8k/evaluation.log

# 查看 GPU 使用情况
watch -n 1 nvidia-smi
```

### 4. 分批评估

如果时间有限，可以分批评估：

```bash
# 第一天：只评估 SQuAD
python eval_squad_gsm8k_compare.py --tasks squadv2

# 第二天：只评估 GSM8K
python eval_squad_gsm8k_compare.py --tasks gsm8k
```

---

## 🐛 故障排除

### 问题 1: 内存不足

**症状**: `CUDA out of memory`

**解决方案**:
```bash
# 减少 batch size（修改 lib/eval.py）
# 或使用 CPU offload
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 问题 2: 评估中断

**症状**: SSH 断开导致评估停止

**解决方案**:
```bash
# 使用 nohup
nohup ./run_eval_squad_gsm8k.sh > eval.log 2>&1 &

# 或使用 tmux/screen
```

### 问题 3: 任务未找到

**症状**: `Task 'squadv2' not found`

**解决方案**:
```bash
# 检查 lm-evaluation-harness 是否正确安装
python -c "
import sys
sys.path.insert(0, '/home/jjji/Research/Hybird-Kernel/lm-evaluation-harness')
from lm_eval.tasks import TaskManager
tm = TaskManager()
print('squadv2' in tm.all_tasks)
print('gsm8k' in tm.all_tasks)
"
```

---

## 📚 参考资料

### SQuAD

- [SQuAD 2.0 论文](https://arxiv.org/abs/1806.03822)
- [SQuAD 官网](https://rajpurkar.github.io/SQuAD-explorer/)
- [Hugging Face SQuAD](https://huggingface.co/datasets/squad_v2)

### GSM8K

- [GSM8K 论文](https://arxiv.org/abs/2110.14168)
- [GSM8K GitHub](https://github.com/openai/grade-school-math)
- [Hugging Face GSM8K](https://huggingface.co/datasets/gsm8k)

### LM Evaluation Harness

- [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [文档](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)

---

## ✅ 检查清单

评估前确认：

- [ ] 模型路径正确
- [ ] 有足够的磁盘空间（至少 50GB）
- [ ] 有足够的 GPU 内存（7B 模型需要 ~16GB）
- [ ] 已安装 lm-evaluation-harness
- [ ] 使用 tmux/screen 或 nohup
- [ ] 预留足够的时间（4-8 小时）

---

**最后更新**: 2025-01-XX  
**维护者**: Jiajun Ji  
**项目**: Wanda Hybrid Pruning

