# 剪枝模型评估指南

## 📍 剪枝模型位置

你的剪枝后的Llama-2-7b模型已保存在:

```
/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model/
```

### 模型文件结构

```
pruned_model/
├── config.json                          # 模型配置
├── generation_config.json               # 生成配置
├── pytorch_model-00001-of-00002.bin    # 模型权重(第1部分)
├── pytorch_model-00002-of-00002.bin    # 模型权重(第2部分)
├── pytorch_model.bin.index.json        # 权重索引
├── special_tokens_map.json             # 特殊token映射
├── tokenizer.json                       # tokenizer配置
├── tokenizer.model                      # tokenizer模型
└── tokenizer_config.json               # tokenizer配置
```

## 📊 剪枝结果

根据你的输出:

- **稀疏度**: 50.00% (每层都是精确的50%稀疏度)
- **WikiText困惑度**: **6.306** 
- **论文参考值**: 6.42 (Llama-2-7b, 50%稀疏度)

**结论**: 你的剪枝效果**非常好**,甚至略优于论文报告的结果! 🎉

## 🚀 使用lm-evaluation-harness评估

### 方法1: 使用提供的脚本(推荐)

#### 简单评估(仅WikiText)

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda
chmod +x evaluate_wikitext_simple.sh
./evaluate_wikitext_simple.sh
```

#### 完整评估(WikiText + 多个基准测试)

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda
chmod +x evaluate_pruned_model.sh
./evaluate_pruned_model.sh
```

### 方法2: 手动运行命令

#### 评估WikiText

```bash
cd /home/jjji/Research/Hybird-Kernel/lm-evaluation-harness

lm_eval --model hf \
    --model_args pretrained=/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model,dtype=float16 \
    --tasks wikitext \
    --device cuda:0 \
    --batch_size auto \
    --output_path /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/eval_results
```

#### 评估多个基准测试

```bash
lm_eval --model hf \
    --model_args pretrained=/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model,dtype=float16 \
    --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,boolq,rte,openbookqa \
    --device cuda:0 \
    --batch_size auto \
    --output_path /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/eval_results
```

## 📋 支持的评估任务

### 常用基准测试

| 任务 | 说明 | 评估指标 |
|------|------|---------|
| `wikitext` | WikiText语言建模 | Perplexity (困惑度) |
| `hellaswag` | 常识推理 | Accuracy |
| `piqa` | 物理常识问答 | Accuracy |
| `winogrande` | 代词消歧 | Accuracy |
| `arc_easy` | ARC简单版 | Accuracy |
| `arc_challenge` | ARC挑战版 | Accuracy |
| `boolq` | 布尔问答 | Accuracy |
| `rte` | 文本蕴含 | Accuracy |
| `openbookqa` | 开放书籍问答 | Accuracy |

### 查看所有可用任务

```bash
cd /home/jjji/Research/Hybird-Kernel/lm-evaluation-harness
lm_eval --tasks list
```

## 🔧 命令参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型类型 | `hf` (HuggingFace) |
| `--model_args` | 模型参数 | `pretrained=<path>,dtype=float16` |
| `--tasks` | 评估任务 | `wikitext` 或 `hellaswag,piqa` |
| `--device` | 计算设备 | `cuda:0` |
| `--batch_size` | 批次大小 | `auto` (自动), `8`, `16` 等 |
| `--output_path` | 结果保存路径 | `/path/to/output` |
| `--num_fewshot` | Few-shot样本数 | `0` (zero-shot), `5` 等 |

## 📈 预期评估时间

基于Llama-2-7b模型:

- **WikiText**: ~5-10分钟
- **单个基准测试**: ~10-20分钟
- **完整评估(8个任务)**: ~1-2小时

## 🔍 查看评估结果

评估完成后,结果会保存在指定的输出目录:

```bash
# 查看结果目录
ls -lh /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/eval_results/

# 查看JSON结果
cat /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/eval_results/results.json
```

结果文件通常包括:
- `results.json`: 详细的评估结果
- `samples_*.jsonl`: 每个样本的预测结果
- 日志文件

## 📊 与原始模型对比

### 创建对比评估

如果你想对比剪枝前后的性能:

```bash
# 评估原始模型
lm_eval --model hf \
    --model_args pretrained=/mnt/sdb/llm_models/Llama-2-7b-hf,dtype=float16 \
    --tasks wikitext,hellaswag,piqa \
    --device cuda:0 \
    --batch_size auto \
    --output_path /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/dense_baseline

# 评估剪枝模型
lm_eval --model hf \
    --model_args pretrained=/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model,dtype=float16 \
    --tasks wikitext,hellaswag,piqa \
    --device cuda:0 \
    --batch_size auto \
    --output_path /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/pruned_50
```

## 🐍 使用Python API评估

如果你想在Python代码中使用:

```python
import lm_eval
from lm_eval.models.huggingface import HFLM

# 加载剪枝模型
model_path = "/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model"
model = HFLM(pretrained=model_path, dtype="float16")

# 运行评估
results = lm_eval.simple_evaluate(
    model=model,
    tasks=["wikitext", "hellaswag"],
    num_fewshot=0,
    batch_size="auto"
)

# 打印结果
print(results["results"])
```

## 🛠️ 故障排除

### 问题1: 找不到模型

**错误**: `OSError: /path/to/model does not appear to be a valid model`

**解决方案**: 检查模型路径是否正确
```bash
ls -lh /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model/
```

### 问题2: CUDA内存不足

**错误**: `CUDA out of memory`

**解决方案**: 减小batch size
```bash
lm_eval ... --batch_size 4  # 或更小的值
```

### 问题3: lm_eval命令找不到

**错误**: `command not found: lm_eval`

**解决方案**: 安装lm-evaluation-harness
```bash
cd /home/jjji/Research/Hybird-Kernel/lm-evaluation-harness
pip install -e .
```

## 📚 参考资料

- **lm-evaluation-harness文档**: [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- **Wanda论文**: [arXiv:2306.11695](https://arxiv.org/abs/2306.11695)
- **Llama-2论文**: [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)

## ✅ 快速检查清单

- [x] 剪枝完成 (稀疏度: 50%, PPL: 6.306)
- [x] 模型已保存
- [ ] 安装lm-evaluation-harness
- [ ] 运行WikiText评估
- [ ] 运行基准测试评估
- [ ] 对比原始模型性能
- [ ] 保存评估结果

## 💡 下一步建议

1. **基础评估**: 先运行WikiText评估,验证模型加载正确
2. **扩展评估**: 运行常用基准测试(HellaSwag, PIQA等)
3. **性能对比**: 与原始dense模型对比性能下降
4. **应用测试**: 在你的具体应用场景中测试模型

## 🎯 预期性能参考

根据Wanda论文,Llama-2-7b在50%稀疏度下的预期性能:

| 任务 | Dense | Wanda 50% | 你的结果 |
|------|-------|-----------|---------|
| WikiText PPL | 5.12 | 6.42 | **6.31** ✅ |
| HellaSwag | - | - | 待评估 |
| PIQA | - | - | 待评估 |
| WinoGrande | - | - | 待评估 |

你的WikiText结果已经**优于论文报告值**! 🎉

