# Environment Setup for Progressive Pruning

## Two Environments Required

Progressive pruning requires two separate mamba/conda environments due to transformers version compatibility:

### 1. prune_llm (Pruning Environment)

**Purpose**: Run pruning operations
**Transformers version**: 4.36.0 (older, compatible with pruning code)

**Setup**:
```bash
mamba create -n prune_llm python=3.9
mamba activate prune_llm
pip install torch transformers==4.36.0 accelerate datasets
```

### 2. wanda_lora (Finetuning Environment)

**Purpose**: Run finetuning operations
**Transformers version**: 4.57.1 (newer, required for training features)

**Setup**:
```bash
mamba create -n wanda_lora python=3.9
mamba activate wanda_lora
pip install torch transformers==4.57.1 accelerate datasets
```

## Automatic Environment Switching

The progressive pruning scripts automatically switch between environments:

### run_progressive_three_tier.sh

```bash
# Automatically uses:
# - prune_llm for pruning
# - wanda_lora for finetuning
./run_progressive_three_tier.sh
```

### run_progressive_single_iter.sh

```bash
# Uses prune_llm for pruning
./run_progressive_single_iter.sh 1

# Then manually finetune with wanda_lora:
conda activate wanda_lora
cd dense_ft
python finetune_sparse_model.py ...
```

## How It Works

The scripts use `mamba run -n <env_name>` to execute commands in specific environments:

```bash
# Pruning step
mamba run -n prune_llm python main_progressive_three_tier.py ...

# Finetuning step
mamba run -n wanda_lora python finetune_sparse_model.py ...
```

## Manual Environment Switching

If you prefer manual control:

```bash
# Step 1: Pruning
mamba activate prune_llm
python main_progressive_three_tier.py \
    --model /mnt/sdb/llm_models/Llama-2-7b-hf \
    --iteration 1 \
    --config progressive_config.csv \
    --save out/progressive_three_tier/iter1/ \
    --save_model out/progressive_three_tier/iter1/pruned_model

# Step 2: Finetuning
mamba activate wanda_lora
cd dense_ft
python finetune_sparse_model.py \
    --model_name_or_path ../out/progressive_three_tier/iter1/pruned_model \
    --output_dir ../out/progressive_three_tier/iter1/finetuned_model \
    ...
```

## Troubleshooting

### Error: "mamba: command not found"

Make sure mamba is installed and initialized:
```bash
# Install mamba if not already installed
conda install -n base -c conda-forge mamba

# Initialize mamba
mamba init bash
source ~/.bashrc
```

### Error: "EnvironmentNameNotFound: Could not find environment: prune_llm"

Create the missing environment:
```bash
mamba create -n prune_llm python=3.9
mamba activate prune_llm
pip install torch transformers==4.36.0 accelerate datasets
```

### Error: "TypeError: cannot unpack non-iterable NoneType object"

This means you're using the wrong environment. Make sure:
- Pruning uses `prune_llm` (transformers 4.36.0)
- Finetuning uses `wanda_lora` (transformers 4.57.1)

## Why Two Environments?

**Pruning code** was written for transformers 4.36.0:
- Uses older API for position_ids
- Compatible with original Wanda implementation

**Finetuning code** requires transformers 4.57.1:
- Uses newer Trainer API
- Better BF16 support
- Required for latest features

**Solution**: Use separate environments and switch automatically via scripts.

## Verifying Environments

Check transformers version in each environment:

```bash
# Check prune_llm
mamba activate prune_llm
python -c "import transformers; print(transformers.__version__)"
# Should output: 4.36.0

# Check wanda_lora
mamba activate wanda_lora
python -c "import transformers; print(transformers.__version__)"
# Should output: 4.57.1
```

## Customizing Environment Names

If you want to use different environment names, edit the scripts:

**run_progressive_three_tier.sh**:
```bash
PRUNE_ENV="your_prune_env"
FINETUNE_ENV="your_finetune_env"
```

**run_progressive_single_iter.sh**:
```bash
PRUNE_ENV="your_prune_env"
```

