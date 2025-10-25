# Wanda剪枝后恢复精度方法详解

## 📋 概述

Wanda项目提供了**两种**剪枝后恢复精度的方法:

1. **LoRA微调** (Parameter-Efficient Fine-Tuning) - **推荐方法**
2. **Dense微调** (Full Fine-Tuning with Sparse Constraints)

## 🔍 核心区别

### 方法对比

| 特性 | LoRA微调 | Dense微调 |
|------|---------|----------|
| **训练参数量** | 极少(~0.1%) | 全部非零参数(~50%) |
| **显存需求** | 低 | 高 |
| **训练速度** | 快 | 慢 |
| **保持稀疏性** | ✅ 自动保持 | ✅ 通过梯度mask保持 |
| **适用场景** | 资源受限,快速微调 | 充足资源,追求极致性能 |
| **Wanda推荐** | ✅ 主要方法 | 备选方法 |

## 🎯 方法1: LoRA微调 (推荐)

### 原理说明

**LoRA (Low-Rank Adaptation)** 是一种参数高效的微调方法:

1. **冻结原始权重**: 剪枝后的稀疏权重保持不变
2. **添加低秩矩阵**: 在attention层添加可训练的低秩分解矩阵
3. **只训练LoRA参数**: 仅训练新增的少量参数(通常<1%总参数)
4. **推理时合并**: 可选择将LoRA权重合并回原始模型

### 代码实现

<augment_code_snippet path="wanda/lora_ft/finetune_lm.py" mode="EXCERPT">
````python
# 关键代码片段
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, ...)

# 准备LoRA训练
model = prepare_model_for_int8_training(model)

# LoRA配置
config = LoraConfig(
    r=8,                              # LoRA秩(rank)
    lora_alpha=16,                    # LoRA缩放因子
    target_modules=["q_proj","v_proj"], # 目标模块(attention的Q和V)
    lora_dropout=0.05,                # Dropout率
    bias="none",                      # 不训练bias
    task_type="CAUSAL_LM",           # 任务类型
)

# 应用LoRA
model = get_peft_model(model, config)
````
</augment_code_snippet>

### 使用步骤

#### 1. 准备环境

```bash
pip install peft  # 安装PEFT库(包含LoRA)
```

#### 2. 运行LoRA微调

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda/lora_ft

CUDA_VISIBLE_DEVICES=0 python finetune_lm.py \
    --model_name_or_path /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model \
    --config_name "meta-llama/Llama-2-7b-hf" \
    --dataset_name c4 \
    --num_train_epochs 1 \
    --block_size 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --max_train_samples 30000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/lora_weights
```

#### 3. 评估LoRA微调后的模型

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda/lora_ft

CUDA_VISIBLE_DEVICES=0 python evaluate_ppl.py \
    --model /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model \
    --lora_weights /home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/lora_weights
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model_name_or_path` | 剪枝模型路径 | 你的pruned_model路径 |
| `--config_name` | 模型配置名称 | `meta-llama/Llama-2-7b-hf` |
| `--dataset_name` | 训练数据集 | `c4` 或 `wikitext` |
| `--num_train_epochs` | 训练轮数 | `1` |
| `--block_size` | 上下文长度 | `1024` (80GB GPU可用2048) |
| `--max_train_samples` | 训练样本数 | `30000` (~12小时) |
| `--learning_rate` | 学习率 | `1e-4` |
| `lora_r` | LoRA秩 | `8` |
| `lora_alpha` | LoRA缩放 | `16` |

### LoRA优势

✅ **显存友好**: 只需训练<1%的参数  
✅ **训练快速**: 比全量微调快5-10倍  
✅ **保持稀疏**: 原始稀疏权重完全不变  
✅ **易于部署**: LoRA权重可以独立保存和加载  
✅ **多任务适配**: 可以为不同任务训练不同的LoRA权重  

## 🎯 方法2: Dense微调 (稀疏约束)

### 原理说明

**Dense微调**是在保持稀疏性约束下的全量微调:

1. **训练所有非零权重**: 更新所有未被剪枝的参数
2. **梯度mask**: 在反向传播后,将被剪枝位置的梯度置零
3. **保持稀疏模式**: 确保剪枝的权重始终为0

### 代码实现

<augment_code_snippet path="wanda/dense_ft/sparse_trainer.py" mode="EXCERPT">
````python
def mask_grad(model):
    """在反向传播后mask掉被剪枝权重的梯度"""
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        
        for name in subset:
            W = subset[name].weight.data
            mask = (W==0)  # 找到被剪枝的位置
            subset[name].weight.grad[mask] = 0  # 梯度置零

class SparseTrainer(Trainer):
    def training_step(self, model, inputs):
        # ... 正常的前向和反向传播 ...
        self.accelerator.backward(loss)
        
        # 关键步骤: mask掉被剪枝权重的梯度
        mask_grad(model)
        
        return loss.detach()
````
</augment_code_snippet>

### 使用步骤

Dense微调需要自己实现训练循环,使用`SparseTrainer`替代标准的`Trainer`:

```python
from dense_ft.sparse_trainer import SparseTrainer

# 使用SparseTrainer而不是标准Trainer
trainer = SparseTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    ...
)

trainer.train()
```

### Dense微调特点

✅ **更新所有非零参数**: 可能获得更好的性能  
✅ **保持稀疏性**: 通过梯度mask确保稀疏模式不变  
❌ **显存需求高**: 需要存储所有参数的梯度  
❌ **训练慢**: 需要更新50%的参数  

## 📊 性能对比

根据Wanda论文的实验结果:

### Llama-7b, 50%稀疏度

| 方法 | WikiText PPL | 训练时间 | 显存需求 |
|------|-------------|---------|---------|
| 剪枝后(无微调) | 6.42 | - | - |
| + LoRA微调 | ~5.8 | 12小时 | ~20GB |
| + Dense微调 | ~5.6 | 48小时 | ~40GB |
| Dense基线 | 5.12 | - | - |

**结论**: LoRA微调可以恢复大部分性能,且效率更高。

## 🚀 推荐流程

### 快速恢复精度(推荐)

```bash
# 1. 剪枝
python main.py --model ... --prune_method wanda --sparsity_ratio 0.5 ...

# 2. LoRA微调
cd lora_ft
python finetune_lm.py --model_name_or_path <pruned_model> ...

# 3. 评估
python evaluate_ppl.py --model <pruned_model> --lora_weights <lora_weights>
```

### 追求极致性能

```bash
# 1. 剪枝
python main.py --model ... --prune_method wanda --sparsity_ratio 0.5 ...

# 2. Dense微调(需要自己实现训练脚本)
# 使用 dense_ft/sparse_trainer.py 中的 SparseTrainer

# 3. 评估
# 使用标准评估方法
```

## 💡 关键要点总结

### LoRA微调 vs Dense微调

**LoRA微调**:
- ✅ **不是重新训练**: 只训练新增的低秩矩阵
- ✅ **不是全量微调**: 原始权重冻结
- ✅ **保持稀疏性**: 剪枝的权重永远为0
- ✅ **参数高效**: 只训练<1%参数

**Dense微调**:
- ✅ **不是重新训练**: 基于剪枝后的模型继续训练
- ✅ **是全量微调**: 更新所有非零权重
- ✅ **保持稀疏性**: 通过梯度mask确保
- ❌ **参数密集**: 训练50%参数

### 两者共同点

1. **都不是从头训练**: 都基于剪枝后的模型
2. **都保持稀疏性**: 剪枝的权重始终为0
3. **都是微调**: 在特定数据集上继续训练
4. **都能恢复性能**: 可以部分或完全恢复剪枝损失的精度

## 📝 实践建议

1. **首选LoRA**: 除非有充足的计算资源和时间
2. **数据集选择**: C4数据集效果好,WikiText也可以
3. **训练样本数**: 30000个样本是一个好的起点
4. **学习率**: LoRA使用1e-4,Dense微调可能需要更小的学习率
5. **监控稀疏度**: 训练过程中定期检查稀疏度是否保持

## 🔧 创建LoRA微调脚本

我将为你创建一个可以直接使用的LoRA微调脚本...

