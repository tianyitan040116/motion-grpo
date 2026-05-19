"""
简单的动作生成脚本
用法: python generate_motion_simple.py --checkpoint experiments_grpo/grpo_kinematic/grpo_model.pth --text "a person walks forward"
"""
import torch
import argparse
import os
import numpy as np
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from models.mllm import MotionLLM
from options.get_eval_option import get_opt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--text', type=str, required=True, help='Motion description')
    parser.add_argument('--output', type=str, default='generated_motion.npy', help='Output .npy file')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--llm-backbone', type=str, default='C:/Users/tianyi/Downloads/gemma-2-2b-it')
    parser.add_argument('--lora-r-t2m', type=int, default=64)
    parser.add_argument('--lora-alpha-t2m', type=int, default=64)
    parser.add_argument('--lora-r-m2t', type=int, default=32)
    parser.add_argument('--lora-alpha-m2t', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--dataname', type=str, default='t2m')
    parser.add_argument('--code-dim', type=int, default=512)
    parser.add_argument('--nb-code', type=int, default=512)
    parser.add_argument('--mu', type=float, default=0.99)
    parser.add_argument('--down-t', type=int, default=2)
    parser.add_argument('--stride-t', type=int, default=2)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dilation-growth-rate', type=int, default=3)
    parser.add_argument('--output-emb-width', type=int, default=512)
    parser.add_argument('--vq-act', type=str, default='relu')
    parser.add_argument('--vq-norm', type=str, default=None)
    parser.add_argument('--quantizer', type=str, default='ema_reset')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--vq-path', type=str, default='ckpt/vqvae.pth')
    return parser.parse_args()

def main():
    args = get_args()

    print(f"Loading model from: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Text: {args.text}")

    # Initialize model
    model = MotionLLM(args)

    # Load checkpoint using the correct method
    model.load_model(args.checkpoint)
    model.eval()

    print("\nGenerating motion...")

    # Generate motion
    with torch.no_grad():
        motion_tokens = model.generate_one_motion_sampling(
            args.text,
            temperature=1.0,
            top_p=0.9,
            max_length=200
        )

        # Decode to motion
        motion = model.net.forward_decoder(motion_tokens.unsqueeze(0))

        # IMPORTANT: Denormalize before saving
        motion_np = model.denormalize(motion.cpu().numpy())[0]  # [T, 263]

    # Save
    np.save(args.output, motion_np)
    print(f"\nMotion saved to: {args.output}")
    print(f"  Shape: {motion_np.shape}")
    print(f"  Duration: {motion_np.shape[0] / 20:.2f} seconds (assuming 20 FPS)")

if __name__ == '__main__':
    main()
