# Zero-Shot 评估对比脚本使用说明

## 📋 功能说明

这个脚本用于对比**原始模型**和**剪枝微调模型**在多个下游任务上的Zero-Shot性能。

### 支持的评估任务

| 任务 | 全称 | 类型 | 说明 |
|------|------|------|------|
| **boolq** | BoolQ | 布尔问答 | 判断问题答案是True/False |
| **rte** | RTE | 文本蕴含 | 判断两个句子是否有蕴含关系 |
| **hellaswag** | HellaSwag | 常识推理 | 选择最合理的句子续写 |
| **winogrande** | WinoGrande | 代词消歧 | 判断代词指代对象 |
| **arc_easy** | ARC-Easy | 科学问答(简单) | 小学科学选择题 |
| **arc_challenge** | ARC-Challenge | 科学问答(困难) | 中学科学选择题 |
| **openbookqa** | OpenBookQA | 开放书籍问答 | 基于科学知识的问答 |

### 评估指标

- **WikiText2 PPL**: 困惑度（越低越好）
- **Accuracy**: 各任务的准确率（越高越好）
- **Sparsity**: 模型稀疏度

---

## 🚀 快速开始

### 方法1: 使用Shell脚本（推荐）

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda
bash run_eval_compare.sh
```

### 方法2: 直接运行Python脚本

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda

python eval_zero_shot_compare.py \
    --original_model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --pruned_model out/llama2_7b/block_16x16_three_tier_0.35_0.45_0.2/wanda/dense_finetuned_model \
    --tasks boolq rte hellaswag winogrande arc_easy arc_challenge openbookqa \
    --output_dir eval_results
```

---

## ⚙️ 参数说明

### 必需参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--original_model` | 原始模型路径 | `/mnt/sdb/llm_models/Llama-2-7b-hf` |
| `--pruned_model` | 剪枝微调模型路径 | `out/llama2_7b/.../dense_finetuned_model` |

### 可选参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tasks` | 评估任务列表 | 所有7个任务 |
| `--output_dir` | 结果保存目录 | `eval_results` |
| `--nsamples` | PPL评估样本数 | 128 |
| `--seed` | 随机种子 | 0 |
| `--cache_dir` | 模型缓存目录 | `llm_weights` |

---

## 📊 输出结果

### 1. 终端输出

脚本会在终端打印详细的对比表格：

```
==================================================================================================
COMPARISON RESULTS
==================================================================================================

Original Model: Original Model
  - Sparsity: 0.00%
  - WikiText2 PPL: 5.4723

Pruned Model: Pruned & Finetuned Model
  - Sparsity: 50.23%
  - WikiText2 PPL: 6.8945
  - PPL Degradation: 1.4222

Task                 Original        Pruned          Difference      Relative       
----------------------------------------------------------------------------------------------------
boolq                0.6234          0.5987          -0.0247         -3.96%
rte                  0.5812          0.5523          -0.0289         -4.97%
hellaswag            0.5789          0.5456          -0.0333         -5.75%
winogrande           0.6934          0.6712          -0.0222         -3.20%
arc_easy             0.7456          0.7234          -0.0222         -2.98%
arc_challenge        0.4523          0.4312          -0.0211         -4.66%
openbookqa           0.3456          0.3289          -0.0167         -4.83%
==================================================================================================

AVERAGE              0.5743          0.5502          -0.0241         -4.19%
==================================================================================================
```

### 2. JSON文件

保存在 `eval_results/comparison_YYYYMMDD_HHMMSS.json`

```json
{
  "original": {
    "model_name": "Original Model",
    "model_path": "/mnt/sdb/llm_models/Llama-2-7b-hf",
    "sparsity": 0.0,
    "wikitext_ppl": 5.4723,
    "tasks": {
      "boolq": {
        "accuracy": 0.6234,
        "full_results": {...}
      },
      ...
    }
  },
  "pruned": {...}
}
```

### 3. Markdown报告

保存在 `eval_results/comparison_YYYYMMDD_HHMMSS.md`

包含完整的对比表格和分析，方便分享和查看。

---

## 🔧 自定义评估

### 只评估部分任务

```bash
python eval_zero_shot_compare.py \
    --original_model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --pruned_model out/.../dense_finetuned_model \
    --tasks boolq hellaswag winogrande
```

### 评估不同的剪枝模型

```bash
python eval_zero_shot_compare.py \
    --original_model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --pruned_model out/llama2_7b/another_pruned_model \
    --output_dir eval_results_v2
```

### 修改Shell脚本中的路径

编辑 `run_eval_compare.sh`：

```bash
# 修改这些变量
ORIGINAL_MODEL="/path/to/your/original/model"
PRUNED_MODEL="path/to/your/pruned/model"
TASKS="boolq rte hellaswag"  # 只评估这3个任务
```

---

## ⏱️ 预计运行时间

| 模型大小 | 任务数量 | 预计时间 |
|---------|---------|---------|
| 7B | 7个任务 | ~30-60分钟 |
| 7B | 3个任务 | ~15-30分钟 |
| 13B | 7个任务 | ~60-120分钟 |

**注意**：
- 首次运行会下载数据集，时间会更长
- 使用多GPU可以加速评估
- HellaSwag数据集较大，评估时间较长

---

## 📝 注意事项

### 1. 环境要求

确保你在正确的conda环境中：

```bash
# 使用prune_llm环境（transformers 4.36.0）
conda activate prune_llm
```

### 2. 显存要求

- **7B模型**: 至少需要1张24GB显存的GPU（如RTX 3090/4090, A5000）
- **13B模型**: 至少需要1张40GB显存的GPU（如A100）
- 如果显存不足，脚本会自动使用`device_map="auto"`分配到多GPU

### 3. 数据集下载

首次运行会自动下载评估数据集到 `~/.cache/huggingface/datasets/`

如果下载失败，可以手动设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 4. 结果解读

- **PPL Degradation**: 困惑度增加，表示语言建模能力下降
- **Accuracy Difference**: 负值表示性能下降，正值表示性能提升
- **Relative**: 相对变化百分比，通常剪枝后会有3-5%的性能下降

---

## 🐛 常见问题

### Q1: 提示找不到模型

**A**: 检查模型路径是否正确：

```bash
ls -lh /mnt/sdb/llm_models/Llama-2-7b-hf
ls -lh out/llama2_7b/block_16x16_three_tier_0.35_0.45_0.2/wanda/dense_finetuned_model
```

### Q2: CUDA out of memory

**A**: 减少评估任务数量或使用更大显存的GPU：

```bash
# 只评估3个任务
python eval_zero_shot_compare.py --tasks boolq rte hellaswag
```

### Q3: 评估速度太慢

**A**: 
1. 使用更少的任务
2. 减少PPL评估样本数：`--nsamples 64`
3. 使用多GPU加速

### Q4: 数据集下载失败

**A**: 使用国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
```

---

## 📚 参考资料

- [Wanda论文](https://arxiv.org/abs/2306.11695)
- [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

## 📧 联系方式

如有问题，请查看项目README或提交Issue。

