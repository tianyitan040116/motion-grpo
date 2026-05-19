# -*- coding: utf-8 -*-
"""
简单测试：验证本地模型加载是否成功
"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

print("="*60)
print("测试本地模型加载")
print("="*60)

model_path = os.environ.get(
    "MOTION_AGENT_LLM_BACKBONE",
    r"C:\Users\tianyi\Downloads\gemma-2-2b-it"
    if os.path.exists(r"C:\Users\tianyi\Downloads\gemma-2-2b-it")
    else "/root/autodl-tmp/gemma-2-2b-it"
)

print(f"\n[1/3] 检查模型路径...")
print(f"路径: {model_path}")

if os.path.exists(model_path):
    print("[OK] 路径存在")
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        print("[OK] config.json 存在")
    else:
        print("[ERROR] config.json 不存在!")
        sys.exit(1)
else:
    print("[ERROR] 路径不存在!")
    sys.exit(1)

print(f"\n[2/3] 加载 Tokenizer (local_files_only=True)...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    print(f"[OK] Tokenizer 加载成功")
    print(f"    词汇表大小: {len(tokenizer)}")
except Exception as e:
    print(f"[ERROR] Tokenizer 加载失败: {e}")
    sys.exit(1)

print(f"\n[3/3] 加载模型 (local_files_only=True)...")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"    使用设备: {device}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16 if device == 'cuda:0' else torch.float32,
        device_map=device if device == 'cuda:0' else None
    )
    print(f"[OK] 模型加载成功!")
    print(f"    参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    print(f"\n[测试] 简单推理测试...")
    test_text = "Hello"
    inputs = tokenizer(test_text, return_tensors="pt")
    if device == 'cuda:0':
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=10)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"    输入: '{test_text}'")
    print(f"    输出: '{generated_text}'")
    print(f"[OK] 推理成功!")

except Exception as e:
    print(f"[ERROR] 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("[SUCCESS] 本地模型加载测试通过!")
print("="*60)
print("\n现在可以运行实际的训练/测试脚本了")
