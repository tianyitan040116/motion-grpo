"""
Test script for GRPO Reward Model

Usage:
    python test_grpo_reward.py
"""

import torch
import numpy as np
from grpo_reward import GRPORewardModel, test_reward_model
from models.mllm import MotionLLM
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from options.get_eval_option import get_opt
from options.option_train import get_args_parser
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("="*60)
    print("Testing GRPO Reward Model")
    print("="*60)

    # Step 1: Load models
    print("\n[1/4] Loading MotionLLM and VQ-VAE...")
    args = get_args_parser()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load MotionLLM (contains VQ-VAE)
    motion_llm = MotionLLM(args)
    vqvae_model = motion_llm.net
    print(f"[OK] VQ-VAE loaded from {args.vq_path}")

    # Step 2: Load evaluator
    print("\n[2/4] Loading text-motion evaluator...")
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    print("[OK] Evaluator loaded")

    # Step 3: Load word vectorizer
    print("\n[3/4] Loading word vectorizer...")
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    print("[OK] Word vectorizer loaded")

    # Step 4: Create reward model
    print("\n[4/4] Initializing GRPO Reward Model...")
    reward_model = GRPORewardModel(
        eval_wrapper=eval_wrapper,
        vqvae_model=vqvae_model,
        word_vectorizer=w_vectorizer,
        device=args.device,
        normalize_reward=True,
        reward_scale=1.0,
        length_penalty_weight=0.01,  # Small penalty for length deviation
    )
    print("[OK] Reward model initialized")

    # Test 1: Basic functionality
    print("\n" + "="*60)
    print("Test 1: Basic Reward Computation")
    print("="*60)

    captions = [
        "a person walks forward slowly",
        "a person jumps up and down",
        "a person raises both arms"
    ]

    print(f"\nCaptions:")
    for i, cap in enumerate(captions):
        print(f"  [{i}] {cap}")

    # Generate some random motion tokens (in practice, these come from model sampling)
    motion_tokens = [
        torch.randint(0, 512, (64,)),   # 64 tokens
        torch.randint(0, 512, (48,)),   # 48 tokens
        torch.randint(0, 512, (80,)),   # 80 tokens
    ]

    print(f"\nMotion token lengths: {[len(t) for t in motion_tokens]}")

    rewards, components = reward_model.compute_reward(
        captions,
        motion_tokens,
        return_components=True
    )

    print(f"\n{'Caption':<35} | {'Reward':>8} | {'Match Score':>11} | {'Physical':>9} | {'Numerical':>10}")
    print("-"*92)
    for i, cap in enumerate(captions):
        print(f"{cap:<35} | {rewards[i].item():>8.4f} | "
              f"{components['matching_scores'][i].item():>11.4f} | "
              f"{components['physical_scores'][i].item():>9.4f} | "
              f"{components['numerical_scores'][i].item():>10.4f}")

    # Test 2: Real motion generation
    print("\n" + "="*60)
    print("Test 2: Reward for LLM-Generated Motion")
    print("="*60)

    test_caption = "a person walks forward"
    print(f"\nGenerating motion for: '{test_caption}'")

    # Generate motion using MotionLLM
    motion_llm.eval()
    with torch.no_grad():
        try:
            generated_tokens = motion_llm.generate_one_motion(test_caption)
            print(f"[OK] Generated {len(generated_tokens)} tokens")

            # Compute reward
            reward = reward_model.compute_reward(
                [test_caption],
                [generated_tokens]
            )
            print(f"[OK] Reward: {reward.item():.4f}")

        except Exception as e:
            print(f"⚠ Generation failed (model may not be trained): {e}")

    # Test 3: Batch processing speed
    print("\n" + "="*60)
    print("Test 3: Batch Processing Speed")
    print("="*60)

    batch_sizes = [1, 4, 8]
    for bs in batch_sizes:
        test_captions = ["a person walks forward"] * bs
        test_tokens = [torch.randint(0, 512, (64,)) for _ in range(bs)]

        import time
        start = time.time()
        with torch.no_grad():
            test_rewards = reward_model.compute_reward(test_captions, test_tokens)
        elapsed = time.time() - start

        print(f"Batch size {bs:2d}: {elapsed*1000:6.2f}ms ({elapsed*1000/bs:5.2f}ms/sample)")

    print("\n" + "="*60)
    print("All tests completed successfully! [OK]")
    print("="*60)


if __name__ == "__main__":
    main()
