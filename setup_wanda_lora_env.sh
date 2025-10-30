#!/bin/bash
# 复现 wanda_lora 微调环境
# 用途：用于模型微调和评估

set -e

ENV_NAME="wanda_lora"

echo "========================================="
echo "创建 ${ENV_NAME} 环境"
echo "========================================="

# 创建 conda 环境
conda create -n ${ENV_NAME} python=3.9 -y

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "安装核心依赖..."

# PyTorch (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Transformers 生态
pip install transformers==4.57.1
pip install tokenizers==0.22.1
pip install accelerate==1.10.1
pip install peft==0.6.0
pip install safetensors==0.6.2

# Hugging Face
pip install huggingface-hub==0.36.0
pip install datasets==4.3.0
pip install evaluate==0.4.6

# 评估工具
pip install sacrebleu==2.5.1
pip install rouge_score==0.1.2
pip install nltk==3.9.2

# 数据处理
pip install pandas==2.3.3
pip install numpy==1.26.3
pip install scipy==1.13.1
pip install scikit-learn==1.6.1
pip install pyarrow==21.0.0

# 工具库
pip install tqdm==4.67.1
pip install tqdm-multiprocess==0.0.11
pip install requests==2.32.5
pip install PyYAML==6.0.3
pip install regex==2025.10.23
pip install sentencepiece==0.2.1
pip install protobuf==6.33.0

# lm-evaluation-harness 依赖
pip install jsonlines==4.0.0
pip install pytablewriter==1.2.1
pip install word2number==1.1
pip install more-itertools==10.8.0
pip install numexpr==2.10.2
pip install sqlitedict==2.1.7
pip install zstandard==0.25.0
pip install dill==0.4.0

echo "========================================="
echo "✅ ${ENV_NAME} 环境创建完成！"
echo "========================================="
echo ""
echo "激活环境："
echo "  conda activate ${ENV_NAME}"
echo ""
echo "用途："
echo "  - 模型微调 (LoRA)"
echo "  - Zero-shot 评估"
echo "  - lm-evaluation-harness"

