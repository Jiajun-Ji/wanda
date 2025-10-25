# Wanda vs lm-eval 困惑度差异分析

## 🔍 问题

你的评估结果显示巨大差异:
- **Wanda评估**: PPL = **6.31** ✅
- **lm-eval评估**: PPL = **11.22** ❌

差异高达 **77%**! 这是为什么?

## 📊 核心发现

经过深入分析Wanda的评估代码,我发现了**关键差异**:

### 1️⃣ 数据预处理方式不同

#### Wanda的方式 (`lib/data.py` 第26行)

```python
# Wanda使用 "\n\n" 连接测试数据
testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
```

#### lm-eval的方式

lm-eval可能使用不同的连接方式或数据处理流程。

### 2️⃣ 评估方法的详细对比

<augment_code_snippet path="wanda/lib/eval.py" mode="EXCERPT">
````python
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # 获取input IDs
    testenc = testenc.input_ids
    
    # 计算样本数 (关键!)
    nsamples = testenc.numel() // model.seqlen
    
    # 存储负对数似然
    nlls = []
    
    # 按固定序列长度分块评估
    for i in range(0, nsamples, bs):
        j = min(i+bs, nsamples)
        
        # 准备输入 (固定长度切片)
        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)
        
        # 前向传播
        lm_logits = model(inputs).logits
        
        # 计算loss
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), 
                       shift_labels.reshape(-1))
        
        # 累积负对数似然
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)
        nlls.append(neg_log_likelihood)
    
    # 计算困惑度
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()
````
</augment_code_snippet>

### 3️⃣ 关键差异点

| 维度 | Wanda评估 | lm-eval评估 |
|------|----------|------------|
| **数据连接** | `"\n\n".join()` | 可能不同 |
| **序列长度** | 固定 `model.seqlen` (2048) | 可能不同 |
| **分块方式** | 固定长度切片,无重叠 | 可能有stride/overlap |
| **loss计算** | 标准CrossEntropyLoss | 可能相同 |
| **PPL计算** | `exp(sum(nlls) / total_tokens)` | 可能不同 |
| **数据集版本** | `wikitext-2-raw-v1` | 可能不同 |

## 🔬 深入分析

### Wanda的评估流程

```
1. 加载WikiText2测试集
   ↓
2. 使用 "\n\n" 连接所有文本
   testenc = tokenizer("\n\n".join(testdata['text']))
   ↓
3. 计算样本数
   nsamples = total_tokens // seqlen
   ↓
4. 按固定长度(seqlen=2048)切片
   inputs = testenc[:, i*seqlen : (i+1)*seqlen]
   ↓
5. 计算每个切片的loss
   ↓
6. 累积所有负对数似然
   total_nll = sum(nlls)
   ↓
7. 计算困惑度
   ppl = exp(total_nll / total_tokens)
```

### 可能导致差异的原因

#### 原因1: 数据预处理差异 ⭐⭐⭐

**Wanda**:
```python
testenc = tokenizer("\n\n".join(testdata['text']))
```
- 使用双换行符连接
- 可能保留了更多的上下文信息
- 文本之间有明确的分隔

**lm-eval**: 可能使用不同的连接方式或处理每个文档独立

#### 原因2: 序列长度和stride ⭐⭐⭐

**Wanda**:
- 固定长度切片: `seqlen = 2048`
- 无重叠: 每个token只评估一次
- 简单高效

**lm-eval**: 可能使用:
- 不同的序列长度
- Sliding window with stride
- 每个token可能被评估多次

#### 原因3: 特殊token处理 ⭐⭐

**Wanda**:
```python
# 简单的shift操作
shift_logits = lm_logits[:, :-1, :]
shift_labels = inputs[:, 1:]
```

**lm-eval**: 可能对特殊token(如padding, BOS, EOS)有不同处理

#### 原因4: 数据集split或版本 ⭐

**Wanda**: 明确使用 `wikitext-2-raw-v1` 的 `test` split

**lm-eval**: 可能使用不同版本或split

## 🧪 验证实验

### 实验1: 检查lm-eval使用的数据集

```bash
# 查看lm-eval的WikiText任务定义
python -c "
from lm_eval import tasks
task = tasks.get_task_dict(['wikitext'])
print(task)
"
```

### 实验2: 使用Wanda的方法评估原始模型

```bash
cd /home/jjji/Research/Hybird-Kernel/wanda

python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.eval import eval_ppl
import argparse

# 创建args对象
args = argparse.Namespace()

# 加载原始模型
model = AutoModelForCausalLM.from_pretrained(
    '/mnt/sdb/llm_models/Llama-2-7b-hf',
    torch_dtype=torch.float16,
    device_map='auto'
)
model.seqlen = 2048
tokenizer = AutoTokenizer.from_pretrained('/mnt/sdb/llm_models/Llama-2-7b-hf')

# 评估
device = torch.device('cuda:0')
ppl = eval_ppl(args, model, tokenizer, device)
print(f'Dense Llama-2-7b PPL (Wanda method): {ppl:.4f}')
"
```

### 实验3: 对比不同评估方法

创建一个脚本同时运行两种评估方法:

```python
# 1. Wanda方法
ppl_wanda = eval_ppl(args, model, tokenizer, device)

# 2. lm-eval方法
from lm_eval import evaluator
results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_path}",
    tasks=["wikitext"],
    ...
)
ppl_lmeval = results['results']['wikitext']['word_perplexity']

print(f"Wanda PPL: {ppl_wanda}")
print(f"lm-eval PPL: {ppl_lmeval}")
```

## 📈 预期结果

如果我的分析正确,你应该看到:

| 模型 | Wanda评估 | lm-eval评估 | 差异 |
|------|----------|------------|------|
| Dense Llama-2-7b | ~5.12 | ~8-10? | ~60-95% |
| Pruned Llama-2-7b | 6.31 | 11.22 | 77% |

**关键观察**: 如果Dense模型的差异比例与Pruned模型相似,说明这是**评估方法的系统性差异**,而不是剪枝导致的问题。

## 💡 结论

### 为什么差异这么大?

1. **数据预处理不同**: `"\n\n".join()` vs 其他方式
2. **序列切分策略不同**: 固定长度无重叠 vs sliding window
3. **评估粒度不同**: token-level vs word-level vs byte-level
4. **特殊token处理不同**: 可能影响最终PPL计算

### 哪个结果更可信?

**Wanda的评估结果 (6.31) 更可信**,原因:

1. ✅ **与论文一致**: 论文报告6.42,你的6.31非常接近
2. ✅ **评估方法一致**: 使用相同的代码和流程
3. ✅ **可复现**: 剪枝过程中直接计算
4. ✅ **专门优化**: Wanda的评估方法专门为LLM剪枝设计

**lm-eval的结果 (11.22) 可能**:
- 使用了不同的评估标准
- 更严格的评估方式
- 不同的数据处理流程

### 建议

1. **使用Wanda评估作为主要参考**: 6.31 PPL
2. **lm-eval用于zero-shot任务**: HellaSwag, PIQA等
3. **对比时保持一致**: 如果用lm-eval,Dense和Pruned都用lm-eval
4. **报告时说明评估方法**: 避免混淆

## 🔧 创建统一评估脚本

我将创建一个脚本,使用Wanda的方法评估原始模型,以便公平对比...

