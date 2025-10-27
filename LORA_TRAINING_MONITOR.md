# LoRA训练实时监控指南

## 📊 实时输出说明

LoRA训练过程中会有**详细的实时输出**，帮助你监控训练进度和效果。

## 🚀 完整输出示例

### **阶段1: 初始化（0-2分钟）**

```bash
========================================
LoRA Fine-tuning Configuration
========================================
Pruned model: out/llama2_7b/block_16x16/wanda/pruned_model
Config: meta-llama/Llama-2-7b-hf
Output directory: out/llama2_7b/block_16x16/wanda/lora_weights_wikitext
Dataset: wikitext
Training samples: 30000
Learning rate: 1e-4
Block size: 1024
Epochs: 1
========================================

✅ Pruned model found

🚀 Starting LoRA fine-tuning...
Expected training time: ~1-2 hours

# 加载模型
Loading checkpoint shards: 100%|████████| 2/2 [00:15<00:00]
✅ Model loaded successfully

# 应用LoRA
Applying LoRA configuration...
trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
✅ LoRA applied: only 0.062% parameters are trainable

# 加载数据集
Loading dataset 'wikitext' (wikitext-2-raw-v1)...
Found cached dataset wikitext
✅ Dataset loaded
  - Train samples: 2,000 sequences
  - Eval samples: 128 sequences
```

### **阶段2: 训练过程（1-2小时）**

```bash
***** Running training *****
  Num examples = 2000
  Num Epochs = 1
  Instantaneous batch size per device = 1
  Total train batch size (w. parallel, distributed & accumulation) = 1
  Gradient Accumulation steps = 1
  Total optimization steps = 2000
  Number of trainable parameters = 4,194,304

# 每10步输出一次训练loss
{'loss': 3.4521, 'learning_rate': 1e-04, 'epoch': 0.005}
Step 10/2000   [>                                ] 0.5%  | Loss: 3.4521 | Time: 0:00:30

{'loss': 3.2134, 'learning_rate': 1e-04, 'epoch': 0.01}
Step 20/2000   [>                                ] 1.0%  | Loss: 3.2134 | Time: 0:01:00

{'loss': 2.9876, 'learning_rate': 1e-04, 'epoch': 0.025}
Step 50/2000   [=>                               ] 2.5%  | Loss: 2.9876 | Time: 0:02:30

{'loss': 2.6543, 'learning_rate': 1e-04, 'epoch': 0.05}
Step 100/2000  [===>                             ] 5.0%  | Loss: 2.6543 | Time: 0:05:00

# 每200步进行一次评估
***** Running Evaluation *****
  Num examples = 128
  Batch size = 8

Evaluating: 100%|████████████████████| 16/16 [00:30<00:00]

{'eval_loss': 2.3456, 'eval_runtime': 30.12, 'eval_samples_per_second': 4.25, 
 'eval_steps_per_second': 0.53, 'epoch': 0.1}

Eval at step 200:
  - Eval Loss: 2.3456
  - Perplexity: 10.44
  - Best so far: Yes ✅

{'loss': 2.2345, 'learning_rate': 1e-04, 'epoch': 0.1}
Step 200/2000  [======>                          ] 10.0% | Loss: 2.2345 | Time: 0:10:00

# 继续训练...
Step 300/2000  [=========>                       ] 15.0% | Loss: 2.0123 | Time: 0:15:00

# 第二次评估
***** Running Evaluation *****
Evaluating: 100%|████████████████████| 16/16 [00:30<00:00]

{'eval_loss': 2.1234, 'eval_runtime': 30.45, 'eval_samples_per_second': 4.20,
 'eval_steps_per_second': 0.52, 'epoch': 0.2}

Eval at step 400:
  - Eval Loss: 2.1234
  - Perplexity: 8.36
  - Best so far: Yes ✅ (improved from 10.44)

Step 400/2000  [============>                    ] 20.0% | Loss: 1.8765 | Time: 0:20:00

# 每500步保存checkpoint
Saving model checkpoint to out/.../lora_weights_wikitext/checkpoint-500
Configuration saved
Model weights saved
✅ Checkpoint saved at step 500

Step 500/2000  [===============>                 ] 25.0% | Loss: 1.7654 | Time: 0:25:00

# 继续训练...
Step 1000/2000 [==============================>  ] 50.0% | Loss: 1.5432 | Time: 0:50:00
Step 1500/2000 [============================================>] 75.0% | Loss: 1.4321 | Time: 1:15:00
Step 2000/2000 [=================================================] 100% | Loss: 1.3876 | Time: 1:40:00

Training completed in 1:40:23
```

### **阶段3: 最终评估（1-2分钟）**

```bash
***** Running Final Evaluation *****
  Num examples = 128
  Batch size = 8

Evaluating: 100%|████████████████████| 16/16 [00:30<00:00]

***** Final eval results *****
  eval_loss = 1.9876
  eval_runtime = 30.67
  eval_samples_per_second = 4.17
  eval_steps_per_second = 0.52
  perplexity = 7.30
  epoch = 1.0

✅ Final Perplexity: 7.30
   (Improved from 8207.52 → 7.30, reduction: 99.91%)
```

### **阶段4: 保存模型（10-30秒）**

```bash
Saving model checkpoint to out/llama2_7b/block_16x16/wanda/lora_weights_wikitext
Configuration saved in .../adapter_config.json
Model weights saved in .../adapter_model.bin
Tokenizer saved in .../tokenizer_config.json

***** Training summary *****
  Total steps: 2000
  Total time: 1:40:23
  Final train loss: 1.3876
  Final eval loss: 1.9876
  Final perplexity: 7.30
  Best checkpoint: checkpoint-1800

========================================
LoRA Fine-tuning Complete!
========================================
LoRA weights saved to: out/llama2_7b/block_16x16/wanda/lora_weights_wikitext

Next steps:
1. Evaluate the fine-tuned model:
   ./evaluate_lora_block.sh
========================================
```

---

## 📈 关键指标解读

### **1. Training Loss（训练损失）**

```bash
Step 10   | Loss: 3.4521  # 初期较高
Step 100  | Loss: 2.6543  # 快速下降
Step 500  | Loss: 1.7654  # 继续下降
Step 1000 | Loss: 1.5432  # 逐渐收敛
Step 2000 | Loss: 1.3876  # 最终收敛
```

**判断标准**：
- ✅ **持续下降**: 训练正常，模型在学习
- ⚠️ **剧烈波动**: 学习率可能过高
- ⚠️ **下降缓慢**: 学习率可能过低
- ❌ **不下降**: 数据或配置有问题

### **2. Eval Loss（评估损失）**

```bash
Step 200  | Eval Loss: 2.3456 | PPL: 10.44
Step 400  | Eval Loss: 2.1234 | PPL: 8.36  ✅ 改善
Step 600  | Eval Loss: 2.0123 | PPL: 7.48  ✅ 继续改善
```

**判断标准**：
- ✅ **持续下降**: 泛化能力提升
- ⚠️ **上升**: 可能过拟合
- ⚠️ **远高于train loss**: 过拟合

### **3. Perplexity（困惑度）**

```bash
Initial (block pruned): 8207.52  ❌
After LoRA (step 200):  10.44    ✅ 大幅改善
After LoRA (step 400):  8.36     ✅ 继续改善
Final (step 2000):      7.30     ✅ 接近目标
```

**目标**：
- 🎯 **理想**: 6.5-7.0
- ✅ **良好**: 7.0-10.0
- ⚠️ **一般**: 10.0-50.0
- ❌ **较差**: >50.0

---

## 🔍 实时监控技巧

### **1. 使用 `tee` 保存日志**

```bash
./run_lora_finetune_block.sh 2>&1 | tee lora_training.log
```

这样可以：
- 实时查看输出
- 同时保存到文件
- 事后分析训练过程

### **2. 使用 `watch` 监控GPU**

在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

监控：
- GPU利用率（应该接近100%）
- 显存使用（应该稳定）
- 温度（不应过高）

### **3. 使用 `tail` 实时查看日志**

如果在后台运行：
```bash
tail -f lora_training.log
```

### **4. 绘制训练曲线**

训练完成后，可以从日志中提取loss：
```bash
grep "{'loss':" lora_training.log > losses.txt
```

---

## ⚠️ 异常情况处理

### **情况1: Loss不下降**

```bash
Step 100 | Loss: 3.4521
Step 200 | Loss: 3.4523
Step 300 | Loss: 3.4519
```

**可能原因**：
- 学习率过低
- 数据加载有问题
- 模型冻结设置错误

**解决方案**：
```bash
# 增加学习率
LEARNING_RATE=5e-4  # 从1e-4增加到5e-4
```

### **情况2: Loss剧烈波动**

```bash
Step 100 | Loss: 2.5
Step 110 | Loss: 1.8
Step 120 | Loss: 3.2
Step 130 | Loss: 2.1
```

**可能原因**：
- 学习率过高
- Batch size太小

**解决方案**：
```bash
# 降低学习率
LEARNING_RATE=5e-5  # 从1e-4降低到5e-5

# 或增加梯度累积
gradient_accumulation_steps=4
```

### **情况3: Eval Loss上升**

```bash
Step 200 | Train: 2.0 | Eval: 2.3
Step 400 | Train: 1.5 | Eval: 2.5  ⚠️ 上升
Step 600 | Train: 1.2 | Eval: 2.8  ❌ 继续上升
```

**可能原因**：
- 过拟合

**解决方案**：
```bash
# 增加dropout
lora_dropout=0.1  # 从0.05增加到0.1

# 或减少训练轮数
NUM_EPOCHS=1  # 不要训练太多轮
```

### **情况4: OOM（显存不足）**

```bash
RuntimeError: CUDA out of memory
```

**解决方案**：
```bash
# 减小上下文长度
BLOCK_SIZE=512  # 从1024减到512

# 或使用梯度检查点
--gradient_checkpointing
```

---

## 📊 输出参数说明

### **已添加的监控参数**

```bash
--logging_steps 10              # 每10步输出一次训练loss
--eval_steps 200                # 每200步评估一次
--save_steps 500                # 每500步保存checkpoint
--logging_first_step            # 输出第一步的loss
--evaluation_strategy steps     # 按步数评估
--save_strategy steps           # 按步数保存
--load_best_model_at_end        # 加载最佳模型
--metric_for_best_model eval_loss  # 使用eval_loss选择最佳模型
```

### **自定义监控频率**

如果想要更频繁的输出：
```bash
--logging_steps 5    # 每5步输出
--eval_steps 100     # 每100步评估
```

如果想要减少输出：
```bash
--logging_steps 50   # 每50步输出
--eval_steps 500     # 每500步评估
```

---

## 💡 最佳实践

1. **使用 `tee` 保存日志**: 方便事后分析
2. **监控GPU使用**: 确保资源充分利用
3. **观察loss趋势**: 判断训练是否正常
4. **定期查看eval**: 避免过拟合
5. **保存checkpoint**: 防止训练中断

---

## 🎯 预期时间线（WikiText）

```
0:00:00 - 0:02:00   初始化（加载模型、数据）
0:02:00 - 0:05:00   前100步（loss快速下降）
0:05:00 - 0:20:00   100-400步（第一次评估）
0:20:00 - 0:50:00   400-1000步（中期训练）
0:50:00 - 1:20:00   1000-1800步（后期收敛）
1:20:00 - 1:40:00   1800-2000步（最终收敛）
1:40:00 - 1:42:00   最终评估和保存

总计: ~1小时40分钟
```

---

## 📝 总结

LoRA训练过程中会有**非常详细的实时输出**，包括：

✅ **训练进度**: 每10步输出loss  
✅ **评估结果**: 每200步评估perplexity  
✅ **Checkpoint**: 每500步保存模型  
✅ **最终结果**: 训练结束后的完整评估  

你可以通过这些输出实时监控训练效果，及时发现和解决问题！

