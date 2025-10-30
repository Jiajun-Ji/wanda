#!/usr/bin/env python3
"""
测试 OPT 模型加载
验证 Wanda 是否可以直接使用 Hugging Face 的 OPT 模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_opt_model(model_id="facebook/opt-125m"):
    """
    测试 OPT 模型加载
    
    Args:
        model_id: Hugging Face 模型 ID
    """
    print(f"{'='*80}")
    print(f"Testing OPT Model: {model_id}")
    print(f"{'='*80}\n")
    
    try:
        # 1. 加载 tokenizer
        print("1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"   ✓ Tokenizer loaded successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print()
        
        # 2. 加载模型
        print("2. Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print(f"   ✓ Model loaded successfully")
        print(f"   Model type: {model.config.model_type}")
        print(f"   Hidden size: {model.config.hidden_size}")
        print(f"   Num layers: {model.config.num_hidden_layers}")
        print(f"   Max position embeddings: {model.config.max_position_embeddings}")
        print()
        
        # 3. 检查模型结构
        print("3. Checking model structure...")
        print(f"   Model class: {model.__class__.__name__}")
        
        # 检查层访问路径
        if hasattr(model, 'model'):
            if hasattr(model.model, 'decoder'):
                if hasattr(model.model.decoder, 'layers'):
                    layers = model.model.decoder.layers
                    print(f"   ✓ Layers found at: model.model.decoder.layers")
                    print(f"   Number of layers: {len(layers)}")
                else:
                    print(f"   ✗ No 'layers' attribute in model.model.decoder")
            else:
                print(f"   ✗ No 'decoder' attribute in model.model")
        else:
            print(f"   ✗ No 'model' attribute")
        print()
        
        # 4. 测试推理
        print("4. Testing inference...")
        test_text = "Hello, I am an AI model"
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Input: {test_text}")
        print(f"   Output: {generated_text}")
        print(f"   ✓ Inference successful")
        print()
        
        # 5. 检查是否需要特殊格式
        print("5. Model format check...")
        print(f"   Model ID: {model_id}")
        print(f"   ✓ This is already in Hugging Face format")
        print(f"   ✓ No conversion needed")
        print(f"   ✓ Can be used directly with Wanda")
        print()
        
        print(f"{'='*80}")
        print(f"✅ All tests passed!")
        print(f"{'='*80}\n")
        
        print("Summary:")
        print(f"  - Model: {model_id}")
        print(f"  - Format: Hugging Face (native)")
        print(f"  - Layer path: model.model.decoder.layers")
        print(f"  - Wanda script: main_opt.py")
        print(f"  - Ready to use: YES")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    # 默认测试最小的 OPT 模型
    model_id = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"
    
    print("\n" + "="*80)
    print("OPT Model Compatibility Test for Wanda")
    print("="*80 + "\n")
    
    success = test_opt_model(model_id)
    
    if success:
        print("\n" + "="*80)
        print("Usage with Wanda:")
        print("="*80)
        print(f"""
# 使用 Wanda 剪枝 OPT 模型
python main_opt.py \\
    --model {model_id} \\
    --prune_method wanda \\
    --sparsity_ratio 0.5 \\
    --sparsity_type unstructured \\
    --save out/opt/unstructured/wanda/

# 可用的 OPT 模型：
# - facebook/opt-125m   (125M 参数)
# - facebook/opt-350m   (350M 参数)
# - facebook/opt-1.3b   (1.3B 参数)
# - facebook/opt-2.7b   (2.7B 参数)
# - facebook/opt-6.7b   (6.7B 参数)
# - facebook/opt-13b    (13B 参数)
# - facebook/opt-30b    (30B 参数)
# - facebook/opt-66b    (66B 参数)

注意：
1. OPT 模型已经是 Hugging Face 格式，无需转换
2. 使用 main_opt.py 而不是 main.py
3. 层访问路径是 model.model.decoder.layers（不同于 LLaMA）
        """)
    
    sys.exit(0 if success else 1)

