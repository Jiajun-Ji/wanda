#!/bin/bash

# Environment Check Script
# This script checks if all required packages are installed correctly

echo "=========================================="
echo "Environment Diagnostic Check"
echo "=========================================="

# Check Python version
echo ""
echo "1️⃣ Python Version:"
python --version
which python

# Check if torch is installed
echo ""
echo "2️⃣ PyTorch Installation:"
if python -c "import torch; print(f'✅ PyTorch {torch.__version__} is installed'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null; then
    :
else
    echo "❌ PyTorch is NOT installed in current environment"
    echo "   Install with: pip install torch"
fi

# Check if transformers is installed
echo ""
echo "3️⃣ Transformers Installation:"
if python -c "import transformers; print(f'✅ Transformers {transformers.__version__} is installed')" 2>/dev/null; then
    :
else
    echo "❌ Transformers is NOT installed"
    echo "   Install with: pip install transformers"
fi

# Check if lm_eval is installed
echo ""
echo "4️⃣ lm-evaluation-harness Installation:"
if python -c "import lm_eval; print(f'✅ lm_eval is installed')" 2>/dev/null; then
    :
else
    echo "❌ lm_eval is NOT installed"
    echo "   Install with: pip install lm_eval"
fi

# Check lm_eval command
echo ""
echo "5️⃣ lm_eval Command:"
if which lm_eval > /dev/null 2>&1; then
    echo "✅ lm_eval command found at: $(which lm_eval)"
    echo "   Testing lm_eval import..."
    if lm_eval --help > /dev/null 2>&1; then
        echo "   ✅ lm_eval command works"
    else
        echo "   ❌ lm_eval command fails (likely environment issue)"
        echo "   Use 'python -m lm_eval' instead"
    fi
else
    echo "❌ lm_eval command not found"
    echo "   Use 'python -m lm_eval' instead"
fi

# Check CUDA devices
echo ""
echo "6️⃣ CUDA Devices:"
if python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null; then
    :
else
    echo "❌ Cannot check CUDA devices"
fi

# Check if pruned model exists
echo ""
echo "7️⃣ Pruned Model:"
PRUNED_MODEL="/home/jjji/Research/Hybird-Kernel/wanda/out/llama2_7b/unstructured/wanda/pruned_model"
if [ -d "${PRUNED_MODEL}" ]; then
    echo "✅ Pruned model found at: ${PRUNED_MODEL}"
    echo "   Files:"
    ls -lh ${PRUNED_MODEL}/*.bin ${PRUNED_MODEL}/config.json 2>/dev/null | head -5
else
    echo "❌ Pruned model NOT found at: ${PRUNED_MODEL}"
fi

# Summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="

# Check if all critical packages are available
ALL_OK=true

if ! python -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch is missing"
    ALL_OK=false
fi

if ! python -c "import transformers" 2>/dev/null; then
    echo "❌ Transformers is missing"
    ALL_OK=false
fi

if ! python -c "import lm_eval" 2>/dev/null; then
    echo "❌ lm_eval is missing"
    ALL_OK=false
fi

if [ "$ALL_OK" = true ]; then
    echo "✅ All required packages are installed!"
    echo ""
    echo "Recommended command to run evaluation:"
    echo "  python -m lm_eval --model hf \\"
    echo "      --model_args pretrained=${PRUNED_MODEL},dtype=float16 \\"
    echo "      --tasks wikitext \\"
    echo "      --device cuda:7 \\"
    echo "      --batch_size auto"
else
    echo ""
    echo "⚠️  Some packages are missing. Install them with:"
    echo "  pip install torch transformers lm_eval"
fi

echo "=========================================="

