"""Test LLM + LoRA loading"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
import os

print("="*60)
print("Testing LLM + LoRA Loading")
print("="*60)

llm_path = os.environ.get(
    "MOTION_AGENT_LLM_BACKBONE",
    r"C:\Users\tianyi\Downloads\gemma-2-2b-it"
    if os.path.exists(r"C:\Users\tianyi\Downloads\gemma-2-2b-it")
    else "/root/autodl-tmp/gemma-2-2b-it"
)
device = 'cuda:0'

print("\n[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
print(f"[OK] Tokenizer loaded, vocab size: {len(tokenizer)}")

print("\n[2/4] Loading LLM...")
llm = AutoModelForCausalLM.from_pretrained(llm_path, local_files_only=True)
print(f"[OK] LLM loaded")

print("\n[3/4] Moving LLM to device...")
llm = llm.to(device)
print(f"[OK] LLM moved to {device}")

print("\n[4/4] Adding LoRA adapters...")
lora_config_t2m = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
llm = get_peft_model(llm, lora_config_t2m, adapter_name='t2m')
print("[OK] First adapter added")

lora_config_m2t = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['o_proj', 'q_proj', 'up_proj', 'v_proj', 'k_proj', 'down_proj', 'gate_proj'],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
llm.add_adapter('m2t', lora_config_m2t)
print("[OK] Second adapter added")

llm.eval()
print("\n" + "="*60)
print("[SUCCESS] All steps completed!")
print("="*60)
