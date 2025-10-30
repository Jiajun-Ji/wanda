#!/bin/bash

# Finetune iteration 1 pruned model

# Configuration
BASE_MODEL="/mnt/sdb/llm_models/Llama-2-7b-hf"
PRUNED_MODEL="out/progressive_three_tier/iter1/pruned_model"
FINETUNED_MODEL="out/progressive_three_tier/iter1/finetuned_model"
ITER_DIR="out/progressive_three_tier/iter1"

# Environment
FINETUNE_ENV="wanda_lora"
PRUNE_ENV="prune_llm"

# GPU configuration
FINETUNE_GPUS="0,1"
NUM_FINETUNE_GPUS=2
PRUNE_GPU="0"

# Training configuration
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=2e-5
EPOCHS=1

# Attention configuration
USE_FLASH_ATTN="false"
USE_SDPA="false"

echo "=========================================="
echo "Finetuning Iteration 1"
echo "=========================================="
echo "Pruned model: ${PRUNED_MODEL}"
echo "Output: ${FINETUNED_MODEL}"
echo "Epochs: ${EPOCHS}"
echo "GPUs: ${FINETUNE_GPUS} (${NUM_FINETUNE_GPUS} GPUs)"
echo "=========================================="

# Check if pruned model exists
if [ ! -d "${PRUNED_MODEL}" ]; then
    echo "❌ Error: Pruned model not found: ${PRUNED_MODEL}"
    exit 1
fi

echo "✅ Found pruned model"
echo ""

# Step 1: Finetuning
echo "Step 1: Finetuning (switching to ${FINETUNE_ENV} environment)..."
echo "----------------------------------------"

cd /home/jjji/Research/Hybird-Kernel/wanda/dense_ft

# Build attention arguments
ATTN_ARGS=""
if [ "$USE_FLASH_ATTN" = "true" ] && [ "$USE_SDPA" = "true" ]; then
    echo "❌ Error: Cannot use both Flash Attention and SDPA. Please choose one."
    exit 1
fi

if [ "$USE_FLASH_ATTN" = "true" ]; then
    ATTN_ARGS="--use_flash_attention"
    echo "Using Flash Attention"
elif [ "$USE_SDPA" = "true" ]; then
    ATTN_ARGS="--use_sdpa"
    echo "Using SDPA (PyTorch native)"
fi

# Choose training command based on number of GPUs
if [ $NUM_FINETUNE_GPUS -eq 1 ]; then
    # Single GPU training
    CUDA_VISIBLE_DEVICES=${FINETUNE_GPUS} mamba run -n ${FINETUNE_ENV} python finetune_sparse_model.py \
        --model_name_or_path ../${PRUNED_MODEL} \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
        --num_train_epochs ${EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --bf16 \
        --output_dir ../${FINETUNED_MODEL} \
        --logging_steps 10 \
        --eval_steps 30 \
        --save_steps 30 \
        --eval_strategy steps \
        --save_strategy steps \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --greater_is_better False \
        --overwrite_output_dir \
        ${ATTN_ARGS}
else
    # Multi-GPU training with torchrun
    CUDA_VISIBLE_DEVICES=${FINETUNE_GPUS} mamba run -n ${FINETUNE_ENV} torchrun \
        --nproc_per_node=${NUM_FINETUNE_GPUS} \
        --master_port=29500 \
        finetune_sparse_model.py \
        --model_name_or_path ../${PRUNED_MODEL} \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
        --num_train_epochs ${EPOCHS} \
        --learning_rate ${LEARNING_RATE} \
        --bf16 \
        --output_dir ../${FINETUNED_MODEL} \
        --logging_steps 10 \
        --eval_steps 30 \
        --save_steps 30 \
        --eval_strategy steps \
        --save_strategy steps \
        --load_best_model_at_end \
        --metric_for_best_model eval_loss \
        --greater_is_better False \
        --overwrite_output_dir \
        ${ATTN_ARGS}
fi

if [ $? -ne 0 ]; then
    echo "❌ Error: Finetuning failed"
    exit 1
fi

echo ""
echo "✅ Finetuning complete!"

# Step 2: Evaluate finetuned model
echo ""
echo "Step 2: Evaluating finetuned model on WikiText-2..."
echo "Using GPU: ${PRUNE_GPU}"
echo "Using environment: ${FINETUNE_ENV} (to avoid pyarrow issues)"
echo "----------------------------------------"

cd /home/jjji/Research/Hybird-Kernel

EVAL_PPL=$(CUDA_VISIBLE_DEVICES=${PRUNE_GPU} mamba run -n ${FINETUNE_ENV} python wanda/eval_model_ppl.py ${FINETUNED_MODEL} ${BASE_MODEL} 2>&1 | grep "Perplexity:" | awk '{print $2}')

echo "✅ Finetuned model perplexity: ${EVAL_PPL}"

# Save evaluation result
echo "${EVAL_PPL}" > ${ITER_DIR}/finetuned_ppl.txt

echo ""
echo "=========================================="
echo "Iteration 1 Complete!"
echo "=========================================="
echo "Finetuned model: ${FINETUNED_MODEL}"
echo "Perplexity: ${EVAL_PPL}"
echo ""
echo "Next step: Run iteration 2"
echo "  ./run_progressive_resume.sh 2"
echo "=========================================="

