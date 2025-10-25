# Wanda剪枝后精度恢复方法对比

## 🎯 核心问题回答

### Q: Wanda剪枝后恢复精度是微调还是重新训练?

**答案**: **微调 (Fine-tuning)**,不是重新训练!

Wanda提供两种微调方法:
1. **LoRA微调** (推荐) - 参数高效微调
2. **Dense微调** - 稀疏约束下的全量微调

## 📊 方法对比表

| 维度 | LoRA微调 | Dense微调 | 从头训练 |
|------|---------|----------|---------|
| **起点** | 剪枝后的模型 | 剪枝后的模型 | 随机初始化 |
| **训练参数** | <1% (新增LoRA层) | ~50% (非零权重) | 100% |
| **原始权重** | 🔒 冻结 | ✏️ 更新 | ❌ 不存在 |
| **稀疏性** | ✅ 自动保持 | ✅ 梯度mask保持 | ❌ 无稀疏性 |
| **训练时间** | ~12小时 | ~48小时 | ~数周 |
| **显存需求** | ~20GB | ~40GB | ~80GB |
| **性能恢复** | 80-90% | 90-95% | 100% (基线) |
| **Wanda推荐** | ✅✅✅ | ✅ | ❌ |

## 🔍 详细原理对比

### 1️⃣ LoRA微调 (Low-Rank Adaptation)

```
原始模型权重 (冻结,稀疏)
    ↓
    + LoRA低秩矩阵 (可训练,密集)
    ↓
输出 = 原始权重 × 输入 + LoRA权重 × 输入
```

**关键特点**:
- ✅ **不修改原始权重**: 剪枝后的稀疏权重完全不变
- ✅ **添加可训练层**: 在attention层添加低秩分解矩阵
- ✅ **参数极少**: 通常只有原模型的0.1-1%
- ✅ **推理时可合并**: LoRA权重可以合并回原始模型

**代码示例**:
```python
from peft import LoraConfig, get_peft_model

# 加载剪枝后的模型
model = AutoModelForCausalLM.from_pretrained("pruned_model")

# 配置LoRA
lora_config = LoraConfig(
    r=8,                              # 秩(rank)
    lora_alpha=16,                    # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标层
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 只有LoRA参数可训练
model.print_trainable_parameters()
# 输出: trainable params: 4,194,304 || all params: 6,738,415,616 || trainable%: 0.062%
```

### 2️⃣ Dense微调 (Sparse-Constrained Fine-tuning)

```
剪枝后的模型 (50%权重为0)
    ↓
前向传播 → 计算loss → 反向传播
    ↓
梯度mask: 将被剪枝位置的梯度置零
    ↓
更新非零权重
    ↓
稀疏模式保持不变
```

**关键特点**:
- ✅ **更新非零权重**: 所有未被剪枝的参数都参与训练
- ✅ **保持稀疏模式**: 通过梯度mask确保被剪枝的权重始终为0
- ✅ **可能更好性能**: 更新更多参数,可能恢复更多精度
- ❌ **资源需求高**: 需要存储和更新50%的参数

**代码示例**:
```python
from dense_ft.sparse_trainer import SparseTrainer

def mask_grad(model):
    """关键函数: mask掉被剪枝权重的梯度"""
    for layer in model.model.layers:
        for name, module in layer.named_modules():
            if hasattr(module, 'weight'):
                W = module.weight.data
                mask = (W == 0)  # 找到被剪枝的位置
                module.weight.grad[mask] = 0  # 梯度置零

# 使用SparseTrainer
trainer = SparseTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    ...
)

trainer.train()  # 训练过程中自动调用mask_grad
```

### 3️⃣ 从头训练 (对比参考)

```
随机初始化
    ↓
完整训练过程 (数周)
    ↓
Dense模型 (无稀疏性)
```

**特点**:
- ❌ **不是Wanda的方法**: Wanda不涉及从头训练
- ❌ **无稀疏性**: 得到的是dense模型
- ❌ **时间长**: 需要数周训练
- ✅ **性能最好**: 作为性能基线

## 📈 性能恢复效果

### Llama-2-7b, 50%稀疏度, WikiText困惑度

```
Dense基线 (从头训练)
    PPL: 5.12
    ↓
剪枝后 (Wanda, 无微调)
    PPL: 6.42 ❌ 性能下降25%
    ↓
+ LoRA微调 (12小时)
    PPL: ~5.8 ✅ 恢复80%性能
    ↓
+ Dense微调 (48小时)
    PPL: ~5.6 ✅ 恢复90%性能
```

**结论**: LoRA微调用20%的时间恢复了80%的性能!

## 🚀 实践流程

### 完整的Wanda剪枝+恢复流程

```bash
# 步骤1: 剪枝 (1-2小时)
python main.py \
    --model /path/to/llama-2-7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --save_model pruned_model/

# 步骤2: LoRA微调 (12小时)
python lora_ft/finetune_lm.py \
    --model_name_or_path pruned_model/ \
    --dataset_name c4 \
    --max_train_samples 30000 \
    --output_dir lora_weights/

# 步骤3: 评估 (10分钟)
python lora_ft/evaluate_ppl.py \
    --model pruned_model/ \
    --lora_weights lora_weights/
```

## 💡 关键概念澄清

### ❓ 是微调还是重新训练?

| 问题 | 答案 | 解释 |
|------|------|------|
| 是从头训练吗? | ❌ 不是 | 基于剪枝后的模型,不是随机初始化 |
| 是微调吗? | ✅ 是 | 在剪枝模型基础上继续训练 |
| 会改变稀疏模式吗? | ❌ 不会 | 被剪枝的权重始终为0 |
| 会训练所有参数吗? | ❌ 不会 | LoRA只训练<1%,Dense训练50% |

### ❓ LoRA vs Dense微调的本质区别?

**LoRA微调**:
```python
# 原始权重冻结
for param in model.parameters():
    param.requires_grad = False

# 只有LoRA参数可训练
for param in lora_layers.parameters():
    param.requires_grad = True
```

**Dense微调**:
```python
# 所有非零权重可训练
for param in model.parameters():
    param.requires_grad = True

# 但在反向传播后mask梯度
def training_step():
    loss.backward()
    mask_grad(model)  # 关键!
    optimizer.step()
```

## 🎓 论文中的实验设置

根据Wanda论文:

### LoRA微调设置
- **数据集**: C4
- **训练样本**: 30,000个序列
- **学习率**: 1e-4
- **LoRA秩**: r=8
- **LoRA alpha**: 16
- **目标模块**: q_proj, v_proj (attention的Q和V)
- **训练时间**: ~12小时 (单GPU)

### 性能提升
- **Llama-7b, 50%稀疏度**:
  - 剪枝后: PPL 6.42
  - LoRA微调后: PPL ~5.8
  - 性能恢复: ~80%

## 📝 总结

### Wanda的精度恢复方法

1. **主要方法**: LoRA微调
   - 参数高效
   - 训练快速
   - 效果良好

2. **备选方法**: Dense微调
   - 性能更好
   - 资源需求高
   - 适合充足资源场景

3. **不是**: 从头训练
   - Wanda不涉及从头训练
   - 剪枝+微调是核心思路

### 推荐选择

- **资源受限** → LoRA微调
- **追求性能** → Dense微调
- **快速验证** → LoRA微调
- **生产部署** → LoRA微调 (易于管理多个适配器)

## 🔗 相关资源

- **LoRA论文**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Wanda论文**: [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)
- **PEFT库**: [Hugging Face PEFT](https://github.com/huggingface/peft)

