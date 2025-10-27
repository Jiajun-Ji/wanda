#!/bin/bash

# Fix LoRA dependencies - downgrade PEFT to be compatible with transformers 4.35.0

echo "=========================================="
echo "Fixing LoRA Dependencies"
echo "=========================================="
echo ""

echo "Current versions:"
pip list | grep -E "peft|accelerate|transformers"
echo ""

echo "Problem:"
echo "  - PEFT 0.17.1 requires transformers >= 4.46.0"
echo "  - Current transformers 4.35.0 is too old"
echo ""

echo "Solution:"
echo "  - Downgrade PEFT to 0.6.0 (compatible with transformers 4.35.0)"
echo ""

echo "Downgrading PEFT..."
pip install peft==0.6.0

echo ""
echo "=========================================="
echo "Dependency Fix Complete"
echo "=========================================="
echo ""

echo "New versions:"
pip list | grep -E "peft|accelerate|transformers"
echo ""

echo "✅ Dependencies fixed!"
echo ""
echo "Compatible versions:"
echo "  - transformers 4.35.0 ✅"
echo "  - accelerate 0.36.0 ✅"
echo "  - peft 0.6.0 ✅"
echo ""
echo "You can now run: ./run_lora_finetune_block.sh"

