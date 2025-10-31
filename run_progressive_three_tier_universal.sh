#!/bin/bash

# Universal Progressive Three-Tier Block Pruning
# Supports: Llama, OPT models
# Supports: WikiText2, C4 datasets

# Get absolute path to wanda directory
WANDA_DIR="/home/jjji/Research/Hybird-Kernel/wanda"

# ============ Model Configuration ============
# Choose model type: "llama" or "opt"
MODEL_TYPE="opt"  # 修改这里

# Model path
if [ "$MODEL_TYPE" = "llama" ]; then
    BASE_MODEL="/mnt/sdb/llm_models/Llama-2-7b-hf"
elif [ "$MODEL_TYPE" = "opt" ]; then
    BASE_MODEL="/mnt/sdb/llm_models/opt-30b"  # 修改为你的 OPT 模型路径
fi

# ============ Dataset Configuration ============
# Choose dataset: "wikitext2" or "c4"
DATASET="wikitext2"  # 修改这里：wikitext2 或 c4

# ============ Output Configuration ============
OUTPUT_BASE="out/progressive_three_tier_${MODEL_TYPE}_${DATASET}"

# ============ Other Configuration ============
CONFIG_FILE="${WANDA_DIR}/progressive_config.csv"
BLOCK_SIZE=16
TOPK_PER_BLOCK=15

# Environment configuration
PRUNE_ENV="prune_llm"      # Environment for pruning (transformers 4.36.0)
FINETUNE_ENV="wanda_lora"  # Environment for finetuning (transformers 4.57.1)

# GPU configuration
PRUNE_GPU="4,5,6,7"              # GPU for pruning (single GPU)
FINETUNE_GPUS="4,5,6,7"        # GPUs for finetuning (can be multiple)
NUM_FINETUNE_GPUS=4        # Number of GPUs for finetuning

# Training configuration
BATCH_SIZE=1                    # Per-device batch size
GRADIENT_ACCUMULATION=8         # Gradient accumulation steps
LEARNING_RATE=5e-5              # Learning rate
BLOCK_SIZE=1024                 # Sequence length
MAX_TRAIN_SAMPLES=1000         # Maximum training samples
MAX_EVAL_SAMPLES=128            # Maximum evaluation samples

# Attention configuration
USE_FLASH_ATTN="false"
USE_SDPA="false"

echo "=========================================="
echo "Progressive Three-Tier Pruning Pipeline"
echo "=========================================="
echo "Model type: ${MODEL_TYPE}"
echo "Base model: ${BASE_MODEL}"
echo "Dataset: ${DATASET}"
echo "Config file: ${CONFIG_FILE}"
echo "Output directory: ${OUTPUT_BASE}"
echo "=========================================="
echo ""

# Function to run one iteration
run_iteration() {
    local ITER=$1
    local PREV_MODEL=$2
    local PREV_TIER_MAPS=$3
    local EPOCHS=$4
    
    echo ""
    echo "=========================================="
    echo "Iteration ${ITER}"
    echo "=========================================="
    
    # Paths for this iteration
    ITER_DIR="${OUTPUT_BASE}/iter${ITER}"
    PRUNED_MODEL="${ITER_DIR}/pruned_model"
    FINETUNED_MODEL="${ITER_DIR}/finetuned_model"
    TIER_MAPS="${ITER_DIR}/tier_maps_iter${ITER}.pt"
    
    mkdir -p ${ITER_DIR}
    
    # Step 1: Pruning
    echo ""
    echo "Step 1: Pruning (switching to ${PRUNE_ENV} environment)..."
    echo "Using GPU: ${PRUNE_GPU}"
    echo "----------------------------------------"

    cd /home/jjji/Research/Hybird-Kernel/wanda

    # Choose pruning script based on model type
    if [ "$MODEL_TYPE" = "llama" ]; then
        PRUNE_SCRIPT="main_progressive_three_tier.py"
    elif [ "$MODEL_TYPE" = "opt" ]; then
        PRUNE_SCRIPT="main_progressive_three_tier_opt.py"
    fi

    PRUNE_CMD="CUDA_VISIBLE_DEVICES=${PRUNE_GPU} mamba run -n ${PRUNE_ENV} python ${PRUNE_SCRIPT} \
        --model ${PREV_MODEL} \
        --iteration ${ITER} \
        --config ${CONFIG_FILE} \
        --block_size ${BLOCK_SIZE} \
        --topk_per_block ${TOPK_PER_BLOCK} \
        --calib_dataset ${DATASET} \
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
    echo "Using environment: ${FINETUNE_ENV}"
    echo "----------------------------------------"

    cd /home/jjji/Research/Hybird-Kernel

    PRUNED_PPL=$(CUDA_VISIBLE_DEVICES=${PRUNE_GPU} mamba run -n ${FINETUNE_ENV} python wanda/eval_model_ppl.py ${PRUNED_MODEL} ${BASE_MODEL} 2>&1 | grep "Perplexity:" | awk '{print $2}')

    echo "✅ Pruned model perplexity: ${PRUNED_PPL}"
    echo "${PRUNED_PPL}" > ${ITER_DIR}/pruned_ppl.txt

    # Step 2: Finetuning
    echo ""
    echo "Step 2: Finetuning for ${EPOCHS} epochs (switching to ${FINETUNE_ENV} environment)..."
    echo "Using GPUs: ${FINETUNE_GPUS} (${NUM_FINETUNE_GPUS} GPUs)"
    echo "Dataset: ${DATASET}"
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

    # Dataset configuration
    if [ "$DATASET" = "wikitext2" ]; then
        DATASET_NAME="wikitext"
        DATASET_CONFIG="wikitext-2-raw-v1"
    elif [ "$DATASET" = "c4" ]; then
        DATASET_NAME="allenai/c4"
        DATASET_CONFIG="en"
    fi

    # Choose training command based on number of GPUs
    if [ $NUM_FINETUNE_GPUS -eq 1 ]; then
        # Single GPU training
        CUDA_VISIBLE_DEVICES=${FINETUNE_GPUS} mamba run -n ${FINETUNE_ENV} python finetune_sparse_model.py \
            --model_name_or_path ../${PRUNED_MODEL} \
            --dataset_name ${DATASET_NAME} \
            --dataset_config_name ${DATASET_CONFIG} \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --per_device_eval_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
            --num_train_epochs ${EPOCHS} \
            --block_size ${BLOCK_SIZE} \
            --max_train_samples ${MAX_TRAIN_SAMPLES} \
            --max_eval_samples ${MAX_EVAL_SAMPLES} \
            --do_train \
            --do_eval \
            --validation_split_percentage 10 \
            --learning_rate ${LEARNING_RATE} \
            --bf16 \
            --gradient_checkpointing \
            --warmup_steps 100 \
            --weight_decay 0.01 \
            --lr_scheduler_type cosine \
            --output_dir ../${FINETUNED_MODEL} \
            --logging_steps 10 \
            --eval_steps 30 \
            --save_steps 30 \
            --logging_first_step \
            --eval_strategy steps \
            --save_strategy steps \
            --save_total_limit 5 \
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
            --dataset_name ${DATASET_NAME} \
            --dataset_config_name ${DATASET_CONFIG} \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --per_device_eval_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
            --num_train_epochs ${EPOCHS} \
            --block_size ${BLOCK_SIZE} \
            --max_train_samples ${MAX_TRAIN_SAMPLES} \
            --max_eval_samples ${MAX_EVAL_SAMPLES} \
            --do_train \
            --do_eval \
            --validation_split_percentage 10 \
            --learning_rate ${LEARNING_RATE} \
            --bf16 \
            --gradient_checkpointing \
            --warmup_steps 100 \
            --weight_decay 0.01 \
            --lr_scheduler_type cosine \
            --output_dir ../${FINETUNED_MODEL} \
            --logging_steps 10 \
            --eval_steps 30 \
            --save_steps 30 \
            --logging_first_step \
            --eval_strategy steps \
            --save_strategy steps \
            --save_total_limit 5 \
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

    # Step 3: Evaluate finetuned model
    echo ""
    echo "Step 3: Evaluating finetuned model on WikiText-2..."
    echo "Using GPU: ${PRUNE_GPU}"
    echo "Using environment: ${FINETUNE_ENV}"
    echo "----------------------------------------"

    cd /home/jjji/Research/Hybird-Kernel

    EVAL_PPL=$(CUDA_VISIBLE_DEVICES=${PRUNE_GPU} mamba run -n ${FINETUNE_ENV} python wanda/eval_model_ppl.py ${FINETUNED_MODEL} ${BASE_MODEL} 2>&1 | grep "Perplexity:" | awk '{print $2}')

    echo "✅ Finetuned model perplexity: ${EVAL_PPL}"
    echo "${EVAL_PPL}" > ${ITER_DIR}/finetuned_ppl.txt

    echo ""
    echo "Iteration ${ITER} complete!"
    echo "  Pruned model: ${PRUNED_MODEL}"
    echo "  Finetuned model: ${FINETUNED_MODEL}"
    echo "  Tier maps: ${TIER_MAPS}"
    echo "  Finetuned PPL: ${EVAL_PPL}"
    echo "=========================================="
}

# Read config and run iterations
echo "Reading configuration from ${CONFIG_FILE}..."
echo ""

# Iteration 1: (90%, 10%, 0%)
run_iteration 1 "${BASE_MODEL}" "" 2

# Iteration 2: (80%, 10%, 10%)
run_iteration 2 "${OUTPUT_BASE}/iter1/finetuned_model" "${OUTPUT_BASE}/iter1/tier_maps_iter1.pt" 2

# Iteration 3: (65%, 20%, 15%)
run_iteration 3 "${OUTPUT_BASE}/iter2/finetuned_model" "${OUTPUT_BASE}/iter2/tier_maps_iter2.pt" 2

# Iteration 4: (50%, 30%, 20%)
run_iteration 4 "${OUTPUT_BASE}/iter3/finetuned_model" "${OUTPUT_BASE}/iter3/tier_maps_iter3.pt" 2

# Iteration 5: (35%, 45%, 20%)
run_iteration 5 "${OUTPUT_BASE}/iter4/finetuned_model" "${OUTPUT_BASE}/iter4/tier_maps_iter4.pt" 3

echo ""
echo "=========================================="
echo "Progressive Pruning Complete!"
echo "=========================================="
echo "Model type: ${MODEL_TYPE}"
echo "Dataset: ${DATASET}"
echo "Final model: ${OUTPUT_BASE}/iter5/finetuned_model"
echo "Final ratios: Dense=35%, 2:4=45%, TopK=20%"
echo ""
echo "Perplexity Summary:"
echo "----------------------------------------"
for i in 1 2 3 4 5; do
    if [ -f "${OUTPUT_BASE}/iter${i}/pruned_ppl.txt" ]; then
        PRUNED_PPL=$(cat ${OUTPUT_BASE}/iter${i}/pruned_ppl.txt)
        FINETUNED_PPL=$(cat ${OUTPUT_BASE}/iter${i}/finetuned_ppl.txt)
        echo "Iteration ${i}: Pruned=${PRUNED_PPL}, Finetuned=${FINETUNED_PPL}"
    fi
done
echo "=========================================="

