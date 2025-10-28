#!/bin/bash

# ========================================
# Environment Setup Script for Wanda Project
# ========================================
# This script sets up the mamba/conda environment with all required dependencies
# for the Wanda sparse pruning and fine-tuning project.
#
# Usage:
#   bash setup_environment.sh
#
# Requirements:
#   - Mamba/Conda/Miniconda installed
#   - CUDA 11.8 or compatible version
#   - At least 10GB free disk space
# ========================================

set -e  # Exit on error

echo "========================================="
echo "Wanda Environment Setup"
echo "========================================="

# ========================================
# Step 1: Create mamba/conda environment
# ========================================

ENV_NAME="wanda_lora"
PYTHON_VERSION="3.9"

# Detect whether to use mamba or conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✅ Using mamba (faster)"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "✅ Using conda"
else
    echo "❌ Error: Neither mamba nor conda found. Please install Miniconda or Mambaforge."
    exit 1
fi

echo ""
echo "Step 1: Creating environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."

if ${CONDA_CMD} env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        ${CONDA_CMD} env remove -n ${ENV_NAME} -y
    else
        echo "Skipping environment creation. Using existing environment."
        SKIP_ENV_CREATE=true
    fi
fi

if [ -z "$SKIP_ENV_CREATE" ]; then
    ${CONDA_CMD} create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    echo "✅ Environment created"
fi

# Activate environment
echo "Activating environment..."
if command -v mamba &> /dev/null; then
    source $(mamba info --base)/etc/profile.d/mamba.sh
    mamba activate ${ENV_NAME}
else
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${ENV_NAME}
fi

# ========================================
# Step 2: Install PyTorch with CUDA support
# ========================================

echo ""
echo "Step 2: Installing PyTorch 2.5.1 with CUDA 11.8..."

# Use Aliyun mirror for faster download in China
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    || pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://mirrors.aliyun.com/pypi/simple/

echo "✅ PyTorch installed"

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# ========================================
# Step 3: Install Transformers and related libraries
# ========================================

echo ""
echo "Step 3: Installing Transformers 4.57.1 and related libraries..."

pip install transformers==4.57.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install accelerate==1.10.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install datasets==3.2.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install tokenizers==0.21.0 -i https://mirrors.aliyun.com/pypi/simple/

echo "✅ Transformers and related libraries installed"

# ========================================
# Step 4: Install evaluation libraries
# ========================================

echo ""
echo "Step 4: Installing evaluation libraries..."

pip install lm-eval==0.4.5 -i https://mirrors.aliyun.com/pypi/simple/
pip install evaluate==0.4.3 -i https://mirrors.aliyun.com/pypi/simple/

echo "✅ Evaluation libraries installed"

# ========================================
# Step 5: Install LoRA/PEFT libraries
# ========================================

echo ""
echo "Step 5: Installing PEFT (LoRA) libraries..."

pip install peft==0.14.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install bitsandbytes==0.45.0 -i https://mirrors.aliyun.com/pypi/simple/

echo "✅ PEFT libraries installed"

# ========================================
# Step 6: Install other dependencies
# ========================================

echo ""
echo "Step 6: Installing other dependencies..."

pip install sentencepiece==0.2.0 -i https://mirrors.aliyun.com/pypi/simple/
pip install protobuf==5.29.2 -i https://mirrors.aliyun.com/pypi/simple/
pip install scipy==1.13.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install scikit-learn==1.6.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install pandas==2.2.3 -i https://mirrors.aliyun.com/pypi/simple/
pip install tqdm==4.67.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install wandb==0.19.1 -i https://mirrors.aliyun.com/pypi/simple/
pip install tensorboard==2.18.0 -i https://mirrors.aliyun.com/pypi/simple/

echo "✅ Other dependencies installed"

# ========================================
# Step 7: Optional - Install Flash Attention 3
# ========================================

echo ""
echo "Step 7: Installing Flash Attention (optional)..."
read -p "Do you want to install Flash Attention 3? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Flash Attention 3..."
    pip install flash-attn --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ \
        || echo "⚠️  Flash Attention installation failed. You can skip this if you plan to use SDPA instead."
    echo "✅ Flash Attention installed (or skipped)"
else
    echo "⏭️  Skipping Flash Attention installation"
fi

# ========================================
# Step 8: Verify installation
# ========================================

echo ""
echo "========================================="
echo "Step 8: Verifying installation..."
echo "========================================="

python -c "
import torch
import transformers
import accelerate
import datasets
import peft

print('✅ All core libraries imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Accelerate: {accelerate.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ========================================
# Step 9: Save environment info
# ========================================

echo ""
echo "Step 9: Saving environment information..."

pip list --format=freeze > requirements_installed.txt
${CONDA_CMD} env export > environment_installed.yml

echo "✅ Environment info saved to:"
echo "   - requirements_installed.txt"
echo "   - environment_installed.yml"

# ========================================
# Completion
# ========================================

echo ""
echo "========================================="
echo "✅ Environment setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
if command -v mamba &> /dev/null; then
    echo "  mamba activate ${ENV_NAME}"
else
    echo "  conda activate ${ENV_NAME}"
fi
echo ""
echo "To verify the installation, run:"
echo "  python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo ""
echo "Next steps:"
echo "  1. Run pruning: bash run_prune_llama2_7b_block_hybrid_2_4.sh"
echo "  2. Run fine-tuning: bash run_dense_finetune_hybrid_2_4.sh"
echo ""

