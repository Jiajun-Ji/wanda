#!/bin/bash

# Create a new conda environment specifically for LoRA fine-tuning
# This environment will have the latest versions of all libraries

ENV_NAME="wanda_lora"

echo "=========================================="
echo "Creating LoRA Fine-tuning Environment"
echo "=========================================="
echo ""

echo "Environment name: ${ENV_NAME}"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists!"
    echo ""
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Aborted."
        exit 1
    fi
fi

echo "Creating new conda environment..."
conda create -n ${ENV_NAME} python=3.9 -y

echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo ""
echo "Installing PyTorch..."
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing Transformers and related libraries..."
pip install transformers datasets accelerate evaluate

echo ""
echo "Installing PEFT (LoRA)..."
pip install peft

echo ""
echo "Installing additional dependencies..."
pip install sentencepiece protobuf

echo ""
echo "=========================================="
echo "Environment Creation Complete!"
echo "=========================================="
echo ""

echo "Installed versions:"
pip list | grep -E "torch|transformers|accelerate|peft|datasets"
echo ""

echo "✅ Environment '${ENV_NAME}' created successfully!"
echo ""
echo "To use this environment:"
echo "  1. Activate: conda activate ${ENV_NAME}"
echo "  2. Run LoRA: cd /home/jjji/Research/Hybird-Kernel/wanda && ./run_lora_finetune_block.sh"
echo ""
echo "Note: Make sure to activate '${ENV_NAME}' before running LoRA fine-tuning!"

