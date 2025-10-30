#!/bin/bash
# 复现 prune_llm 剪枝环境
# 用途：用于模型剪枝和可视化

set -e

ENV_NAME="prune_llm"

echo "========================================="
echo "创建 ${ENV_NAME} 环境"
echo "========================================="

# 创建 conda 环境
conda create -n ${ENV_NAME} python=3.9 -y

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "安装核心依赖..."

# PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Transformers 生态
pip install transformers==4.35.2
pip install tokenizers==0.15.2
pip install accelerate==1.10.1
pip install peft==0.6.0
pip install safetensors==0.6.2

# Hugging Face
pip install huggingface-hub==0.36.0
pip install datasets==2.14.7
pip install evaluate==0.4.5

# 数据处理
pip install pandas==2.3.1
pip install numpy==1.26.3
pip install scipy==1.13.1
pip install scikit-learn==1.6.1
pip install pyarrow==21.0.0

# 可视化
pip install matplotlib==3.9.4
pip install gradio==3.24.1

# Web 框架 (Gradio 依赖)
pip install fastapi==0.116.1
pip install uvicorn==0.35.0
pip install starlette==0.47.2

# 工具库
pip install tqdm==4.67.1
pip install requests==2.32.5
pip install PyYAML==6.0.2
pip install regex==2025.10.23
pip install sentencepiece==0.2.1
pip install protobuf==6.33.0
pip install sacremoses==0.0.53

# 实验跟踪
pip install wandb==0.22.2

# 其他工具
pip install GitPython==3.1.45
pip install cookiecutter==2.6.0
pip install rich==14.1.0

echo "========================================="
echo "✅ ${ENV_NAME} 环境创建完成！"
echo "========================================="
echo ""
echo "激活环境："
echo "  conda activate ${ENV_NAME}"
echo ""
echo "用途："
echo "  - Wanda 剪枝"
echo "  - 混合剪枝 (三层稀疏)"
echo "  - Gradio 可视化"
echo "  - WandB 实验跟踪"

