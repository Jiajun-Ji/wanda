# Llama-2-7b 剪枝指南 (使用Wanda)

## 📋 概述

本指南说明如何使用Wanda方法对Llama-2-7b模型进行50%稀疏度的非结构化剪枝。

## 🔧 代码修改说明

为了解决C4数据集加载问题并使用WikiText2数据集,我们对原始代码进行了以下修改:

### 1. 修改 `lib/data.py`
- **位置**: 第43-44行
- **原因**: 修复C4数据集配置名称错误
- **修改**: 将 `'allenai--c4'` 改为 `'en'`
- **状态**: ✅ 已完成(备用方案)

### 2. 修改 `lib/prune.py`
- **位置**: 第132行 (prune_wanda函数)
- **位置**: 第218行 (prune_sparsegpt函数)
- **位置**: 第310行 (prune_ablate函数)
- **修改**: 将校准数据集从 `"c4"` 改为 `"wikitext2"`
- **原因**: 使用WikiText2作为校准数据集
- **状态**: ✅ 已完成

## 🚀 使用方法

### 方法1: 使用提供的脚本(推荐)

```bash
cd wanda
chmod +x run_prune_llama2_7b.sh
./run_prune_llama2_7b.sh
```

### 方法2: 手动运行命令

```bash
cd wanda
python main.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama2_7b/unstructured/wanda/ \
    --save_model out/llama2_7b/unstructured/wanda/pruned_model \
    --nsamples 128 \
    --seed 0
```

## 📊 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--model` | `/mnt/sdb/llm_models/Llama-2-7b-hf` | 模型路径 |
| `--prune_method` | `wanda` | 剪枝方法 |
| `--sparsity_ratio` | `0.5` | 稀疏度(50%) |
| `--sparsity_type` | `unstructured` | 非结构化剪枝 |
| `--save` | `out/llama2_7b/...` | 结果保存路径 |
| `--save_model` | `out/llama2_7b/.../pruned_model` | 剪枝模型保存路径 |
| `--nsamples` | `128` | 校准样本数量 |
| `--seed` | `0` | 随机种子 |

## 📈 预期输出

剪枝完成后,你将看到:

1. **稀疏度检查**: 确认实际稀疏度约为50%
2. **WikiText困惑度(PPL)**: 评估剪枝后模型性能
3. **日志文件**: `out/llama2_7b/unstructured/wanda/log_wanda.txt`
4. **剪枝模型**: `out/llama2_7b/unstructured/wanda/pruned_model/`

### 预期性能参考

根据Wanda论文,Llama-2-7b在50%非结构化稀疏度下的预期困惑度:

- **Dense模型**: ~5.12
- **Wanda剪枝(50%)**: ~6.42

## 🔍 数据集说明

### 校准数据集(Calibration Dataset)
- **当前使用**: WikiText2
- **作用**: 用于计算激活值,指导剪枝过程
- **样本数**: 128个序列

### 评估数据集(Evaluation Dataset)
- **使用**: WikiText2
- **作用**: 评估剪枝后模型的困惑度(PPL)
- **自动执行**: 无需额外配置

## 🛠️ 故障排除

### 问题1: C4数据集加载失败
**错误**: `ValueError: BuilderConfig 'allenai--c4' not found`

**解决方案**: 
- 已修改为使用WikiText2数据集
- 或者修改 `lib/data.py` 第43-44行,将 `'allenai--c4'` 改为 `'en'`

### 问题2: 内存不足
**解决方案**:
- 减少 `--nsamples` 参数(默认128)
- 使用更小的模型或更少的GPU

### 问题3: CUDA设备选择
**解决方案**:
```bash
export CUDA_VISIBLE_DEVICES=0  # 使用GPU 0
# 或
export CUDA_VISIBLE_DEVICES=1  # 使用GPU 1
```

## 📝 其他剪枝选项

### 结构化剪枝 (2:4)
```bash
python main.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 2:4 \
    --save out/llama2_7b/2-4/wanda/
```

### 结构化剪枝 (4:8)
```bash
python main.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type 4:8 \
    --save out/llama2_7b/4-8/wanda/
```

## 🔄 恢复到C4数据集

如果你想使用C4数据集而不是WikiText2:

1. 修改 `lib/prune.py` 中的三处:
   - 第132行: 改回 `"c4"`
   - 第218行: 改回 `"c4"`
   - 第310行: 改回 `"c4"`

2. 确保 `lib/data.py` 第43-44行使用 `'en'` 配置

## 📚 参考资料

- **Wanda论文**: [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)
- **原始仓库**: [locuslab/wanda](https://github.com/locuslab/wanda)
- **模型**: Llama-2-7b-hf

## ✅ 检查清单

- [x] 修改数据集加载代码
- [x] 创建执行脚本
- [x] 验证模型路径
- [ ] 运行剪枝
- [ ] 检查输出结果
- [ ] 验证剪枝模型性能

## 💡 提示

1. **首次运行**: WikiText2数据集会自动下载,可能需要几分钟
2. **模型加载**: Llama-2-7b加载需要约13GB显存
3. **剪枝时间**: 整个剪枝过程可能需要30-60分钟
4. **保存模型**: 剪枝后的模型大小约为原模型的50%(因为权重被置零)

## 📧 联系方式

如有问题,请参考原始Wanda仓库的Issues或联系作者。

