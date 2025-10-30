# Wanda 项目环境配置指南

本项目包含两个独立的 Conda 环境，分别用于不同的任务。

## 📦 环境概览

| 环境名称 | Python | PyTorch | CUDA | 用途 |
|---------|--------|---------|------|------|
| `wanda_lora` | 3.9 | 2.8.0 | 12.8 | 微调、评估 |
| `prune_llm` | 3.9 | 2.1.0 | 12.1 | 剪枝、可视化 |

---

## 🚀 快速开始

### 1️⃣ 创建 `wanda_lora` 环境（微调和评估）

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda
chmod +x setup_wanda_lora_env.sh
bash setup_wanda_lora_env.sh
```

**用途**：
- ✅ LoRA 微调
- ✅ Zero-shot 评估（BoolQ, RTE, HellaSwag 等）
- ✅ lm-evaluation-harness
- ✅ 模型评估对比

**激活环境**：
```bash
conda activate wanda_lora
```

---

### 2️⃣ 创建 `prune_llm` 环境（剪枝和可视化）

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda
chmod +x setup_prune_llm_env.sh
bash setup_prune_llm_env.sh
```

**用途**：
- ✅ Wanda 剪枝
- ✅ 混合三层剪枝（Dense + 2:4 + Top-K）
- ✅ Gradio 可视化界面
- ✅ WandB 实验跟踪

**激活环境**：
```bash
conda activate prune_llm
```

---

## 📋 环境详细说明

### `wanda_lora` 环境

**核心依赖**：
- `torch==2.8.0` (CUDA 12.8)
- `transformers==4.57.1`
- `peft==0.6.0`
- `datasets==4.3.0`
- `evaluate==0.4.6`
- `sacrebleu==2.5.1`
- `lm-evaluation-harness` 相关依赖

**适用脚本**：
```bash
# 微调
python finetune_lora.py

# 评估对比
python eval_zero_shot_compare.py

# 测试 lm_eval
python test_lm_eval_import.py
```

---

### `prune_llm` 环境

**核心依赖**：
- `torch==2.1.0` (CUDA 12.1)
- `transformers==4.35.2`
- `peft==0.6.0`
- `gradio==3.24.1`
- `wandb==0.22.2`
- `matplotlib==3.9.4`

**适用脚本**：
```bash
# Wanda 剪枝
python main.py

# 三层混合剪枝
python main_block_three_tier.py
python main_progressive_three_tier.py

# 可视化
python gradio_app.py  # 如果有的话
```

---

## 🔧 手动安装 lm-evaluation-harness

如果需要在 `wanda_lora` 环境中使用 lm-evaluation-harness：

```bash
conda activate wanda_lora
cd /home/jjji/Research/Hybird-Kernel/lm-evaluation-harness
pip install -e .
```

---

## ⚠️ 注意事项

### 1. PyTorch 版本差异

两个环境使用不同的 PyTorch 版本：
- `wanda_lora`: PyTorch 2.8.0 (CUDA 12.8) - 最新版本，支持最新特性
- `prune_llm`: PyTorch 2.1.0 (CUDA 12.1) - 稳定版本，兼容性好

**建议**：
- 微调和评估使用 `wanda_lora`
- 剪枝使用 `prune_llm`
- 不要混用环境

### 2. Transformers 版本差异

- `wanda_lora`: transformers 4.57.1 (最新)
- `prune_llm`: transformers 4.35.2 (稳定)

**影响**：
- API 可能有细微差异
- 模型加载方式可能不同
- 建议在同一环境中完成完整流程

### 3. CUDA 兼容性

确保你的 GPU 驱动支持：
- CUDA 12.8 (wanda_lora)
- CUDA 12.1 (prune_llm)

检查 CUDA 版本：
```bash
nvidia-smi
```

---

## 🧪 验证环境

### 验证 `wanda_lora`

```bash
conda activate wanda_lora
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python test_lm_eval_import.py
```

### 验证 `prune_llm`

```bash
conda activate prune_llm
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
```

---

## 📝 常见问题

### Q1: 安装失败怎么办？

**A**: 检查网络连接，使用国内镜像：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

### Q2: CUDA 版本不匹配？

**A**: 根据你的 GPU 驱动调整 PyTorch 版本：
```bash
# 查看支持的 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

### Q3: 两个环境可以共存吗？

**A**: 可以！Conda 环境是完全隔离的，互不影响。

---

## 🔄 更新环境

如果需要更新某个包：

```bash
conda activate <env_name>
pip install --upgrade <package_name>
```

如果需要重建环境：

```bash
conda remove -n <env_name> --all
bash setup_<env_name>_env.sh
```

---

## 📚 相关文档

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [PEFT 文档](https://huggingface.co/docs/peft)

---

## 💡 推荐工作流程

### 完整的剪枝 + 微调 + 评估流程

```bash
# 1. 剪枝 (使用 prune_llm 环境)
conda activate prune_llm
python main_block_three_tier.py --model llama-2-7b --sparsity_ratios 0.35 0.45 0.2

# 2. 微调 (切换到 wanda_lora 环境)
conda activate wanda_lora
python finetune_lora.py --model_path out/llama2_7b/.../pruned_model

# 3. 评估 (继续使用 wanda_lora 环境)
python eval_zero_shot_compare.py \
    --original_model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --pruned_model out/llama2_7b/.../dense_finetuned_model
```

---

**创建时间**: 2025-01-XX  
**维护者**: Jiajun Ji  
**项目**: Wanda Hybrid Pruning

