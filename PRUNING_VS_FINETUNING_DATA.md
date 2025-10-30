# 剪枝 vs 微调：数据使用的区别

本文档详细解释剪枝和微调时数据使用的区别，以及为什么问答数据集可以用于剪枝但需要特殊处理。

---

## 🔍 核心区别

### 剪枝时的数据使用

**目的**: 计算激活值，确定权重重要性

**过程**:
```python
# 1. 前向传播（不计算梯度）
for j in range(nsamples):
    with torch.no_grad():  # 不计算梯度
        outs[j] = layer(inps[j])  # 只是前向传播

# 2. 收集激活值
wrapped_layers[name].add_batch(inp[0].data, out.data)

# 3. 计算 Wanda score
W_metric = |W| * sqrt(activation_norm)

# 4. 根据 score 剪枝
prune_weights_with_lowest_scores()
```

**关键点**:
- ✅ **只需要输入文本**（不需要标签）
- ✅ **只做前向传播**（不更新权重）
- ✅ **只收集激活值**（用于计算重要性）
- ✅ **不需要理解任务**（只需要文本分布）

---

### 微调时的数据使用

**目的**: 更新权重，恢复性能

**过程**:
```python
# 1. 前向传播（计算梯度）
outputs = model(input_ids, labels=labels)  # 需要标签

# 2. 计算损失
loss = outputs.loss

# 3. 反向传播
loss.backward()

# 4. 更新权重
optimizer.step()
```

**关键点**:
- ⚠️ **需要输入-输出对**（需要标签）
- ⚠️ **需要计算损失**（需要知道正确答案）
- ⚠️ **需要更新权重**（需要梯度）
- ⚠️ **需要理解任务**（需要特殊格式）

---

## 📊 对比表格

| 特性 | 剪枝（Pruning） | 微调（Fine-tuning） |
|------|----------------|-------------------|
| **数据需求** | 只需要输入文本 | 需要输入-输出对 |
| **梯度计算** | ❌ 不需要 | ✅ 需要 |
| **权重更新** | ❌ 不更新 | ✅ 更新 |
| **任务理解** | ❌ 不需要 | ✅ 需要 |
| **Prompt 格式** | ❌ 不需要 | ✅ 需要 |
| **标签** | ❌ 不需要 | ✅ 需要 |
| **目的** | 确定权重重要性 | 恢复/提升性能 |

---

## 🤔 问答数据集用于剪枝：可行吗？

### 答案：完全可行！

**原因**:
1. ✅ 剪枝只需要文本输入，不需要标签
2. ✅ 问答数据集有丰富的文本（context + question）
3. ✅ 不需要特殊的 prompt 格式
4. ✅ 只需要前向传播，不需要理解任务

### 示例

#### WikiText2（当前使用）

```python
# 纯文本
text = "The Normans were the people who in the 10th and 11th centuries..."

# 直接 tokenize
input_ids = tokenizer(text, return_tensors='pt')

# 前向传播
outputs = model(input_ids)  # 收集激活值
```

#### SQuAD（问答数据集）

```python
# 问答数据
context = "The Normans were the people who in the 10th and 11th centuries..."
question = "In what country is Normandy located?"

# 方式 1: 只使用 context（类似 WikiText2）
text = context
input_ids = tokenizer(text, return_tensors='pt')
outputs = model(input_ids)  # 收集激活值

# 方式 2: 使用 context + question（更贴近实际使用）
text = f"{context} {question}"
input_ids = tokenizer(text, return_tensors='pt')
outputs = model(input_ids)  # 收集激活值

# 方式 3: 使用简单的格式（推荐）
text = f"Context: {context}\nQuestion: {question}"
input_ids = tokenizer(text, return_tensors='pt')
outputs = model(input_ids)  # 收集激活值
```

**关键点**:
- ✅ 不需要答案（answer）
- ✅ 不需要特殊的 prompt
- ✅ 只需要把文本拼接起来
- ✅ 格式可以很简单

---

## 🎯 "特殊的 Prompt" 是什么意思？

### 剪枝时：不需要特殊 Prompt

```python
# 简单拼接即可
text = f"{context} {question}"

# 或者稍微格式化
text = f"Context: {context}\nQuestion: {question}"

# 甚至只用 context
text = context
```

**原因**: 剪枝只需要文本输入，不需要理解任务格式。

---

### 微调时：需要特殊 Prompt

```python
# ❌ 错误：简单拼接
text = f"{context} {question} {answer}"
# 问题：模型不知道这是一个问答任务

# ✅ 正确：使用 Prompt 模板
text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the question based on the context.

### Input:
Context: {context}
Question: {question}

### Response:
{answer}"""

# 或者使用更简单的格式
text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
```

**原因**: 微调需要告诉模型：
1. 这是一个什么任务（问答）
2. 输入是什么（context + question）
3. 输出是什么（answer）
4. 如何格式化（Instruction, Input, Response）

---

## 📋 实际例子对比

### 例子：SQuAD 数据

```json
{
  "context": "The Normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France.",
  "question": "In what country is Normandy located?",
  "answer": "France"
}
```

### 剪枝时使用（简单）

```python
# lib/data.py
def get_squad(nsamples, seed, seqlen, tokenizer):
    dataset = load_dataset('squad_v2', split='train')
    
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(dataset) - 1)
        sample = dataset[i]
        
        # 简单拼接即可
        text = f"{sample['context']} {sample['question']}"
        
        # Tokenize
        inp = tokenizer(text, return_tensors='pt', max_length=seqlen, truncation=True)
        
        trainloader.append((inp.input_ids, inp.input_ids.clone()))
    
    return trainloader, testenc
```

**关键点**:
- ✅ 不需要 answer
- ✅ 不需要特殊格式
- ✅ 简单拼接即可

---

### 微调时使用（复杂）

```python
# dense_ft/finetune_squad.py
def preprocess_squad(examples, tokenizer):
    inputs = []
    for context, question, answers in zip(...):
        answer_text = answers['text'][0]
        
        # 需要特殊的 Prompt 格式
        text = f"""Answer the question based on the context.

Context: {context}
Question: {question}
Answer: {answer_text}"""
        
        inputs.append(text)
    
    # Tokenize
    model_inputs = tokenizer(inputs, ...)
    
    # 设置 labels（用于计算损失）
    model_inputs['labels'] = model_inputs['input_ids'].clone()
    
    return model_inputs
```

**关键点**:
- ⚠️ 需要 answer（用于计算损失）
- ⚠️ 需要特殊格式（告诉模型这是问答任务）
- ⚠️ 需要设置 labels（用于训练）

---

## 💡 为什么问答数据集用于剪枝"不太好"？

### 不是"不太好"，而是"需要权衡"

#### ✅ 优点

1. **更贴近实际使用**
   - 如果你的目标是问答任务
   - 使用问答数据剪枝会保留更多相关权重

2. **文本质量高**
   - SQuAD 的 context 都是维基百科文章
   - 质量和 WikiText2 类似

3. **文本多样性**
   - 包含各种主题的文章
   - 问题形式多样

#### ⚠️ 缺点

1. **失去通用性**
   - 剪枝后的模型可能在其他任务上性能下降
   - 例如：用 SQuAD 剪枝，在 GSM8K 上可能表现不好

2. **文本分布偏差**
   - SQuAD 主要是事实性问答
   - 可能不适合其他类型的任务

3. **数据量限制**
   - SQuAD 训练集 ~87k 样本
   - WikiText2 是连续文本，可以无限采样

---

## 🎯 推荐策略

### 策略 1: 通用剪枝（推荐）

```bash
# 使用 WikiText2 剪枝
python main_block_three_tier.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --calibration_dataset wikitext2 \
    --nsamples 128
```

**优点**:
- ✅ 保持通用性
- ✅ 在多个任务上表现稳定

**适用场景**:
- 需要在多个任务上使用
- 不确定最终用途

---

### 策略 2: 任务特定剪枝

```bash
# 使用 SQuAD 剪枝
python main_block_three_tier.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --calibration_dataset squad \
    --nsamples 128
```

**优点**:
- ✅ 在目标任务上性能更好
- ✅ 激活值更贴近实际使用

**缺点**:
- ⚠️ 在其他任务上可能性能下降

**适用场景**:
- 只关心特定任务
- 追求极致性能

---

### 策略 3: 混合剪枝

```bash
# 使用 WikiText2 + SQuAD 混合剪枝
# 需要修改代码支持混合数据集
```

**优点**:
- ✅ 平衡通用性和任务性能

---

## 📊 实验建议

### 对比实验

| 实验 | 剪枝数据 | 微调数据 | 评估任务 |
|------|---------|---------|---------|
| **Exp 1** | WikiText2 | WikiText2 | SQuAD, GSM8K, BoolQ |
| **Exp 2** | SQuAD | SQuAD | SQuAD, GSM8K, BoolQ |
| **Exp 3** | WikiText2 | SQuAD | SQuAD, GSM8K, BoolQ |

**预期结果**:
- Exp 1: 通用性能最好
- Exp 2: SQuAD 性能最好，但其他任务下降
- Exp 3: 平衡方案，SQuAD 性能好，通用性保持

---

## ✅ 总结

### Q1: 问答数据集用于剪枝可行吗？

**A: 完全可行！**

- ✅ 剪枝只需要文本输入，不需要标签
- ✅ 不需要特殊的 prompt 格式
- ✅ 简单拼接 context + question 即可

### Q2: 为什么说"不太好"？

**A: 不是"不太好"，而是"需要权衡"**

- ✅ 如果只关心特定任务，用任务数据剪枝更好
- ⚠️ 如果需要通用性，用 WikiText2 剪枝更好
- 💡 推荐：WikiText2 剪枝 + 任务数据微调

### Q3: "特殊的 Prompt" 是什么意思？

**A: 只在微调时需要，剪枝时不需要**

- **剪枝**: 简单拼接文本即可
- **微调**: 需要 Prompt 模板告诉模型任务格式

---

## 🔧 实现建议

### 剪枝时

```python
# 简单即可
text = f"{context} {question}"
```

### 微调时

```python
# 需要格式化
text = f"""Context: {context}
Question: {question}
Answer: {answer}"""
```

---

**最后更新**: 2025-01-XX  
**维护者**: Jiajun Ji  
**项目**: Wanda Hybrid Pruning

