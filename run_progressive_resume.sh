#!/bin/bash

# Resume Progressive Three-Tier Block Pruning from a specific iteration
# Usage: ./run_progressive_resume.sh <start_iteration>

# Get absolute path to wanda directory
WANDA_DIR="/home/jjji/Research/Hybird-Kernel/wanda"

# Configuration
BASE_MODEL="/mnt/sdb/llm_models/Llama-2-7b-hf"
CONFIG_FILE="${WANDA_DIR}/progressive_config.csv"
BLOCK_SIZE=16
TOPK_PER_BLOCK=15
OUTPUT_BASE="out/progressive_three_tier"

# Environment configuration
PRUNE_ENV="prune_llm"      # Environment for pruning (transformers 4.36.0)
FINETUNE_ENV="wanda_lora"  # Environment for finetuning (transformers 4.57.1)

# GPU configuration
PRUNE_GPU="0"              # GPU for pruning (single GPU)
FINETUNE_GPUS="0,3"        # GPUs for finetuning (can be multiple)
NUM_FINETUNE_GPUS=2        # Number of GPUs for finetuning

# Training configuration
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=2e-5

# Attention configuration (默认不启用,需要手动修改)
USE_FLASH_ATTN="false"  # Set to "true" to enable Flash Attention (requires flash-attn library)
USE_SDPA="false"        # Set to "true" to enable SDPA (PyTorch native, no extra library needed)

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <start_iteration>"
    echo "Example: $0 2  # Resume from iteration 2"
    exit 1
fi

START_ITER=$1

echo "=========================================="
echo "Resume Progressive Three-Tier Pruning"
echo "=========================================="
echo "Base model: ${BASE_MODEL}"
echo "Config file: ${CONFIG_FILE}"
echo "Output directory: ${OUTPUT_BASE}"
echo "Starting from iteration: ${START_ITER}"
echo "=========================================="

# Check if previous iteration exists
if [ ${START_ITER} -gt 1 ]; then
    PREV_ITER=$((START_ITER - 1))
    PREV_MODEL="${OUTPUT_BASE}/iter${PREV_ITER}/finetuned_model"
    PREV_TIER_MAPS="${OUTPUT_BASE}/iter${PREV_ITER}/tier_maps_iter${PREV_ITER}.pt"
    
    if [ ! -d "${PREV_MODEL}" ]; then
        echo "❌ Error: Previous iteration model not found: ${PREV_MODEL}"
        echo "Please make sure iteration ${PREV_ITER} is complete."
        exit 1
    fi
    
    if [ ! -f "${PREV_TIER_MAPS}" ]; then
        echo "❌ Error: Previous tier maps not found: ${PREV_TIER_MAPS}"
        exit 1
    fi
    
    echo "✅ Found previous iteration ${PREV_ITER}"
    echo "   Model: ${PREV_MODEL}"
    echo "   Tier maps: ${PREV_TIER_MAPS}"
else
    PREV_MODEL="${BASE_MODEL}"
    PREV_TIER_MAPS=""
    echo "Starting from base model: ${BASE_MODEL}"
fi

echo ""

# Function to run one iteration
run_iteration() {
    local ITER=$1
    local PREV_MODEL=$2
    local PREV_TIER_MAPS=$3
    
    echo "=========================================="
    echo "Iteration ${ITER}"
    echo "=========================================="
    
    # Read configuration for this iteration
    CONFIG_LINE=$(awk -F',' -v iter=${ITER} '$1 == iter {print $0}' ${CONFIG_FILE})
    if [ -z "${CONFIG_LINE}" ]; then
        echo "Error: No configuration found for iteration ${ITER}"
        return 1
    fi
    
    DENSE_RATIO=$(echo ${CONFIG_LINE} | cut -d',' -f2)
    MID_RATIO=$(echo ${CONFIG_LINE} | cut -d',' -f3)
    TOPK_RATIO=$(echo ${CONFIG_LINE} | cut -d',' -f4)
    EPOCHS=$(echo ${CONFIG_LINE} | cut -d',' -f7)
    
    ITER_DIR="${OUTPUT_BASE}/iter${ITER}"
    PRUNED_MODEL="${ITER_DIR}/pruned_model"
    FINETUNED_MODEL="${ITER_DIR}/finetuned_model"
    TIER_MAPS="${ITER_DIR}/tier_maps_iter${ITER}.pt"
    
    mkdir -p ${ITER_DIR}
    
    # Step 1: Pruning (use prune_llm environment on single GPU)
    echo ""
    echo "Step 1: Pruning (switching to ${PRUNE_ENV} environment)..."
    echo "Using GPU: ${PRUNE_GPU}"
    echo "----------------------------------------"
    
    # Make sure we're in the wanda directory
    cd /home/jjji/Research/Hybird-Kernel/wanda

    PRUNE_CMD="CUDA_VISIBLE_DEVICES=${PRUNE_GPU} mamba run -n ${PRUNE_ENV} python main_progressive_three_tier.py \
        --model ${PREV_MODEL} \
        --iteration ${ITER} \
        --config ${CONFIG_FILE} \
        --block_size ${BLOCK_SIZE} \
        --topk_per_block ${TOPK_PER_BLOCK} \
        --save ${ITER_DIR}/ \
        --save_model ${PRUNED_MODEL}"

    if [ ! -z "${PREV_TIER_MAPS}" ]; then
        PRUNE_CMD="${PRUNE_CMD} --previous_tier_maps ${PREV_TIER_MAPS}"
    fi

    eval ${PRUNE_CMD}

    if [ $? -ne 0 ]; then
        echo "Error: Pruning failed for iteration ${ITER}"
        exit 1
    fi

    # Evaluate pruned model
    echo ""
    echo "Evaluating pruned model on WikiText-2..."
    echo "Using GPU: ${PRUNE_GPU}"
    echo "Using environment: ${FINETUNE_ENV} (to avoid pyarrow issues)"
    echo "----------------------------------------"

    cd /home/jjji/Research/Hybird-Kernel

    PRUNED_PPL=$(CUDA_VISIBLE_DEVICES=${PRUNE_GPU} mamba run -n ${FINETUNE_ENV} python wanda/eval_model_ppl.py ${PRUNED_MODEL} ${BASE_MODEL} 2>&1 | grep "Perplexity:" | awk '{print $2}')
    
    echo "✅ Pruned model perplexity: ${PRUNED_PPL}"
    
    # Save evaluation result
    echo "${PRUNED_PPL}" > ${ITER_DIR}/pruned_ppl.txt

    # Step 2: Finetuning (use wanda_lora environment on multiple GPUs)
    echo ""
    echo "Step 2: Finetuning for ${EPOCHS} epochs (switching to ${FINETUNE_ENV} environment)..."
    echo "Using GPUs: ${FINETUNE_GPUS} (${NUM_FINETUNE_GPUS} GPUs)"
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
        echo "Error: Finetuning failed for iteration ${ITER}"
        exit 1
    fi

    # Step 3: Evaluate finetuned model using wanda's eval_ppl
    echo ""
    echo "Step 3: Evaluating finetuned model on WikiText-2..."
    echo "Using GPU: ${PRUNE_GPU}"
    echo "Using environment: ${FINETUNE_ENV} (to avoid pyarrow issues)"
    echo "----------------------------------------"

    cd /home/jjji/Research/Hybird-Kernel

    EVAL_PPL=$(CUDA_VISIBLE_DEVICES=${PRUNE_GPU} mamba run -n ${FINETUNE_ENV} python wanda/eval_model_ppl.py ${FINETUNED_MODEL} ${BASE_MODEL} 2>&1 | grep "Perplexity:" | awk '{print $2}')

    echo "✅ Finetuned model perplexity: ${EVAL_PPL}"

    # Save evaluation result
    echo "${EVAL_PPL}" > ${ITER_DIR}/finetuned_ppl.txt

    echo ""
    echo "Iteration ${ITER} complete!"
    echo "  Pruned model: ${PRUNED_MODEL}"
    echo "  Finetuned model: ${FINETUNED_MODEL}"
    echo "  Tier maps: ${TIER_MAPS}"
    echo "  Pruned PPL: ${PRUNED_PPL}"
    echo "  Finetuned PPL: ${EVAL_PPL}"
    echo "=========================================="
}

# Run iterations from START_ITER to 5
for ITER in $(seq ${START_ITER} 5); do
    if [ ${ITER} -eq ${START_ITER} ]; then
        run_iteration ${ITER} "${PREV_MODEL}" "${PREV_TIER_MAPS}"
    else
        PREV_ITER=$((ITER - 1))
        PREV_MODEL="${OUTPUT_BASE}/iter${PREV_ITER}/finetuned_model"
        PREV_TIER_MAPS="${OUTPUT_BASE}/iter${PREV_ITER}/tier_maps_iter${PREV_ITER}.pt"
        run_iteration ${ITER} "${PREV_MODEL}" "${PREV_TIER_MAPS}"
    fi
done

echo ""
echo "=========================================="
echo "Progressive Pruning Complete!"
echo "=========================================="
echo "Final model: ${OUTPUT_BASE}/iter5/finetuned_model"
echo "Final ratios: Dense=35%, 2:4=45%, TopK=20%"
echo ""
echo "Perplexity Summary:"
echo "----------------------------------------"
for i in $(seq ${START_ITER} 5); do
    if [ -f "${OUTPUT_BASE}/iter${i}/pruned_ppl.txt" ]; then
        PRUNED_PPL=$(cat ${OUTPUT_BASE}/iter${i}/pruned_ppl.txt)
        FINETUNED_PPL=$(cat ${OUTPUT_BASE}/iter${i}/finetuned_ppl.txt)
        echo "Iteration ${i}: Pruned=${PRUNED_PPL}, Finetuned=${FINETUNED_PPL}"
    fi
done
echo "=========================================="

