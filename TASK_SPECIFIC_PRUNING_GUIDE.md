# 任务特定剪枝与微调指南

本文档说明如何使用特定任务数据集（如 SQuAD、GSM8K）进行剪枝和微调。

---

## 📋 核心问题

### Q1: 剪枝时能用任务数据集作为校准数据吗？

**A: 可以，但需要权衡。**

#### ✅ 优点
- 剪枝后的模型在目标任务上性能更好
- 激活值更贴近实际使用场景
- 可能保留更多任务相关的权重

#### ⚠️ 缺点
- 失去模型的通用能力
- 可能在其他任务上性能下降
- 需要修改代码添加数据集支持

#### 📊 实验对比

| 校准数据 | 目标任务性能 | 通用性能 | 适用场景 |
|---------|------------|---------|---------|
| **WikiText2** | 中等 | 高 | 通用模型 |
| **任务数据集** | 高 | 低 | 特定任务 |
| **混合数据** | 较高 | 中等 | 折中方案 |

---

### Q2: 微调时能用任务数据集吗？

**A: 可以，而且推荐！**

#### ✅ 优点
- 针对性优化，性能提升明显
- 可以使用任务特定的训练策略
- 更符合实际应用需求

#### ⚠️ 注意事项
- 需要特殊的 prompt 格式
- 需要修改训练脚本
- 可能需要更多的训练数据

---

## 🎯 推荐策略

### 策略 1: 通用剪枝 + 任务微调（推荐）

```
WikiText2 剪枝 → 任务数据微调 → 任务评估
```

**优点**：
- ✅ 保持模型通用性
- ✅ 针对任务优化
- ✅ 平衡性能和通用性

**适用场景**：
- 需要在多个任务上使用
- 希望保持一定的通用能力
- 有足够的任务数据进行微调

### 策略 2: 任务剪枝 + 任务微调

```
任务数据剪枝 → 任务数据微调 → 任务评估
```

**优点**：
- ✅ 最大化任务性能
- ✅ 端到端优化

**缺点**：
- ⚠️ 失去通用性
- ⚠️ 可能过拟合

**适用场景**：
- 只关心特定任务
- 有大量任务数据
- 追求极致性能

### 策略 3: 混合剪枝 + 任务微调

```
(WikiText2 + 任务数据) 剪枝 → 任务数据微调 → 任务评估
```

**优点**：
- ✅ 平衡通用性和任务性能
- ✅ 更稳健

**适用场景**：
- 需要在多个相关任务上使用
- 希望保持一定的通用能力

---

## 🔧 实现方案

### 方案 1: 添加任务数据集支持到剪枝

#### 步骤 1: 修改 `lib/data.py`

添加 SQuAD 和 GSM8K 数据加载函数：

```python
# 添加到 lib/data.py

def get_squad(nsamples, seed, seqlen, tokenizer):
    """
    加载 SQuAD 数据集用于剪枝校准
    使用 context + question 作为输入
    """
    from datasets import load_dataset
    
    # 加载 SQuAD v2
    dataset = load_dataset('squad_v2', split='train')
    
    # 生成样本
    random.seed(seed)
    trainloader = []
    
    for _ in range(nsamples):
        # 随机选择一个样本
        i = random.randint(0, len(dataset) - 1)
        sample = dataset[i]
        
        # 构造输入：context + question
        text = f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
        
        # Tokenize
        enc = tokenizer(text, return_tensors='pt', max_length=seqlen, truncation=True)
        inp = enc.input_ids
        
        # 创建 target（剪枝时不使用，但保持格式一致）
        tar = inp.clone()
        tar[:, :-1] = -100
        
        trainloader.append((inp, tar))
    
    # 返回训练数据和测试数据（使用 WikiText2 作为测试）
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', verification_mode='no_checks')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    return trainloader, testenc


def get_gsm8k(nsamples, seed, seqlen, tokenizer):
    """
    加载 GSM8K 数据集用于剪枝校准
    使用 question (+ answer) 作为输入
    """
    from datasets import load_dataset
    
    # 加载 GSM8K
    dataset = load_dataset('gsm8k', 'main', split='train')
    
    # 生成样本
    random.seed(seed)
    trainloader = []
    
    for _ in range(nsamples):
        # 随机选择一个样本
        i = random.randint(0, len(dataset) - 1)
        sample = dataset[i]
        
        # 构造输入：question + answer（用于剪枝时计算激活值）
        text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        
        # Tokenize
        enc = tokenizer(text, return_tensors='pt', max_length=seqlen, truncation=True)
        inp = enc.input_ids
        
        # 创建 target
        tar = inp.clone()
        tar[:, :-1] = -100
        
        trainloader.append((inp, tar))
    
    # 返回训练数据和测试数据
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', verification_mode='no_checks')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    
    return trainloader, testenc


# 修改 get_loaders 函数
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "squad" in name:
        return get_squad(nsamples, seed, seqlen, tokenizer)
    if "gsm8k" in name:
        return get_gsm8k(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unknown dataset: {name}")
```

#### 步骤 2: 使用任务数据集剪枝

```bash
# 使用 SQuAD 数据集剪枝
python main_block_three_tier.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --sparsity_ratios 0.35 0.45 0.2 \
    --nsamples 128 \
    --save out/llama2_7b/squad_pruned/ \
    --calibration_dataset squad  # 新增参数

# 使用 GSM8K 数据集剪枝
python main_block_three_tier.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --sparsity_ratios 0.35 0.45 0.2 \
    --nsamples 128 \
    --save out/llama2_7b/gsm8k_pruned/ \
    --calibration_dataset gsm8k  # 新增参数
```

**注意**：需要在 `main_block_three_tier.py` 中添加 `--calibration_dataset` 参数。

---

### 方案 2: 使用任务数据集微调

#### 步骤 1: 创建任务特定的微调脚本

创建 `dense_ft/finetune_squad.py`：

```python
#!/usr/bin/env python3
"""
使用 SQuAD 数据集微调剪枝后的模型
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def preprocess_squad(examples, tokenizer, max_length=512):
    """
    预处理 SQuAD 数据
    格式：Context: ... Question: ... Answer: ...
    """
    inputs = []
    for context, question, answers in zip(
        examples['context'],
        examples['question'],
        examples['answers']
    ):
        # 提取答案文本
        answer_text = answers['text'][0] if answers['text'] else "No answer"
        
        # 构造输入
        text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer_text}"
        inputs.append(text)
    
    # Tokenize
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # 设置 labels（用于计算损失）
    model_inputs['labels'] = model_inputs['input_ids'].clone()
    
    return model_inputs

# 主函数
def main():
    # 加载模型和 tokenizer
    model_path = "out/llama2_7b/pruned_model"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained("/mnt/sdb/llm_models/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载 SQuAD 数据集
    dataset = load_dataset('squad_v2')
    
    # 预处理数据
    train_dataset = dataset['train'].map(
        lambda x: preprocess_squad(x, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="out/llama2_7b/squad_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()

if __name__ == "__main__":
    main()
```

#### 步骤 2: 运行任务微调

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda/dense_ft

# 微调 SQuAD
python finetune_squad.py

# 微调 GSM8K
python finetune_gsm8k.py
```

---

## 📊 性能对比实验

### 实验设计

| 实验 | 剪枝数据 | 微调数据 | 评估任务 |
|------|---------|---------|---------|
| **Baseline** | WikiText2 | WikiText2 | SQuAD, GSM8K, BoolQ |
| **Exp 1** | WikiText2 | SQuAD | SQuAD, GSM8K, BoolQ |
| **Exp 2** | SQuAD | SQuAD | SQuAD, GSM8K, BoolQ |
| **Exp 3** | WikiText2 | GSM8K | SQuAD, GSM8K, BoolQ |
| **Exp 4** | GSM8K | GSM8K | SQuAD, GSM8K, BoolQ |

### 预期结果

- **Exp 1**: SQuAD ↑, GSM8K ≈, BoolQ ↓
- **Exp 2**: SQuAD ↑↑, GSM8K ↓, BoolQ ↓↓
- **Exp 3**: GSM8K ↑, SQuAD ≈, BoolQ ↓
- **Exp 4**: GSM8K ↑↑, SQuAD ↓, BoolQ ↓↓

---

## 💡 最佳实践

### 1. 选择合适的策略

- **通用模型**: 使用 WikiText2 剪枝 + WikiText2 微调
- **单任务优化**: 使用任务数据剪枝 + 任务数据微调
- **多任务平衡**: 使用 WikiText2 剪枝 + 任务数据微调

### 2. 数据量建议

| 阶段 | 推荐样本数 | 说明 |
|------|-----------|------|
| **剪枝校准** | 128-256 | 足够计算激活值 |
| **微调训练** | 1000+ | 越多越好 |
| **评估测试** | 全量 | 使用完整测试集 |

### 3. 超参数调整

```python
# 剪枝
--nsamples 128          # 校准样本数
--seqlen 2048           # 序列长度

# 微调
--num_train_epochs 3    # 训练轮数
--learning_rate 2e-5    # 学习率
--batch_size 4          # 批次大小
```

---

## ⚠️ 注意事项

### 1. 数据格式

不同任务的数据格式不同，需要正确处理：

- **SQuAD**: `{"context": "...", "question": "...", "answers": {...}}`
- **GSM8K**: `{"question": "...", "answer": "..."}`
- **WikiText2**: 纯文本

### 2. Prompt 设计

任务特定微调需要设计合适的 prompt：

```python
# SQuAD
prompt = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"

# GSM8K
prompt = f"Question: {question}\nLet's solve this step by step:\n{answer}"
```

### 3. 评估指标

不同任务使用不同的评估指标：

- **SQuAD**: Exact Match, F1 Score
- **GSM8K**: Accuracy
- **WikiText2**: Perplexity

---

## 📚 参考资料

- [Wanda 论文](https://arxiv.org/abs/2306.11695)
- [SQuAD 数据集](https://rajpurkar.github.io/SQuAD-explorer/)
- [GSM8K 数据集](https://github.com/openai/grade-school-math)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)

---

**最后更新**: 2025-01-XX  
**维护者**: Jiajun Ji  
**项目**: Wanda Hybrid Pruning

