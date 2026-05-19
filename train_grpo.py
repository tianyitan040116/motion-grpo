"""
GRPO (Group Relative Policy Optimization) Training for MotionLLM

This script implements GRPO training for text-to-motion generation.
Key differences from supervised fine-tuning:
1. No ground-truth motion needed - trains purely from reward signal
2. Samples multiple motions per caption (group sampling)
3. Uses advantage estimation from reward model
4. Optimizes policy using ratio-clipping and KL penalty

Reference: https://arxiv.org/abs/2402.03300

=== CHANGELOG (for group meeting reference) ===

[v4] Frozen SFT reference for KL anchoring (2026-04-10)
  PROBLEM: v3's KL penalty used per-batch old_log_probs as reference. This
  resets to ~0 each batch, providing NO long-term drift constraint. LogProb
  declined monotonically from -6.5 to -8.2 over 41 batches.

  ROOT CAUSE: DeepSeekMath GRPO uses TWO references:
  - Ratio (clipping): vs per-batch snapshot -> we did this correctly
  - KL penalty (anchoring): vs FROZEN SFT model -> we used per-batch snapshot (WRONG)

  SOLUTION: Added frozen SFT reference model via weight-swap mechanism.
  - save_reference_state(): saves SFT LoRA + embeddings + lm_head at init
  - _swap_to_ref() / _swap_to_policy(): zero-copy LoRA pointer swap
  - KL penalty now computed against frozen SFT (grows over training, anchors policy)
  - Ratio still computed against per-batch snapshot (for clipping, as before)
  - grpo_kl_target: 0.5 -> 5.0 (measures drift from SFT, not within-batch)

  NEW METRIC: sft_kl (KL divergence vs frozen SFT reference)

[v3] Proper GRPO with K inner optimization steps (2026-04-10)
  PROBLEM: v2's single-step REINFORCE had unconstrained gradient variance.
  With ratio ≡ 1.0, clipping had no effect, causing LogProb collapse
  (from -2 to -15 in ~50 batches). No mechanism to constrain policy drift.

  SOLUTION (DeepSeekMath-style GRPO):
  - Added K=3 inner optimization steps per batch
  - Snapshot old_log_probs from current policy (on-policy reference, NOT base Gemma)
  - Compute importance ratio = exp(new_lp - old_lp), starts at 1.0, drifts with steps
  - Clipped surrogate: min(ratio*adv, clip(ratio, 1-eps, 1+eps)*adv)
  - Per-token KL penalty: exp(old-new) - (old-new) - 1
  - Early exit if KL > target (default 0.5)
  - Fixed LR schedule: replaced hardcoded 1000 steps with epochs * batches_per_epoch

  HYPERPARAMETER CHANGES:
  - learning_rate: 1e-5 -> 2e-6 (lower for RL stability with inner loop)
  - grpo_beta: 0.001 -> 0.04 (meaningful now with on-policy reference)
  - grpo_kl_target: 1000.0 -> 0.5 (meaningful now)
  - Added: --inner-steps 3

  NEW METRICS: kl, ratio, clip_fraction, inner_steps_used

[v2] REINFORCE with group-relative advantages (removed KL penalty)
  PROBLEM: Reference model was base Gemma (disable_adapter), but policy had
  SFT LoRA weights. KL ~ 500 constant, drowning out policy gradient.
  SOLUTION: Removed ref model forward pass, simplified to REINFORCE.
  RESULT: Training ran but LogProb collapsed after ~50 batches.

[v1] Initial GRPO implementation
  PROBLEM: CUDA OOM with original batch/chunk sizes.
  SOLUTION: Reduced chunk_size, sub_batch_size, batch_size. Added empty_cache().

=== END CHANGELOG ===
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import rnn as rnn_utils
from dataset import dataset_TM_eval
from utils.evaluation import evaluation_test
import os
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from models.mllm import MotionLLM
from options.option_train import get_args_parser
from grpo_reward import GRPORewardModel
import logging
import json
import sys
import argparse
from typing import List, Tuple
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_DEFAULT_LOCAL_LLM = r"C:\Users\tianyi\Downloads\gemma-2-2b-it"
_DEFAULT_REMOTE_LLM = "/root/autodl-tmp/gemma-2-2b-it"
_DEFAULT_LLM_BACKBONE = os.environ.get(
    "MOTION_AGENT_LLM_BACKBONE",
    _DEFAULT_LOCAL_LLM if os.path.exists(_DEFAULT_LOCAL_LLM) else _DEFAULT_REMOTE_LLM,
)


def maybe_empty_cache(device: str):
    """Release cached CUDA blocks when training on a CUDA device."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_logger(out_dir):
    logger = logging.getLogger('GRPO')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run_grpo.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def get_grpo_args():
    """Extended argument parser for GRPO-specific hyperparameters"""
    parser = argparse.ArgumentParser(description='GRPO Training for MotionLLM',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Device
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    # GRPO-specific hyperparameters
    parser.add_argument('--num-samples-per-prompt', type=int, default=4,
                        help='Number of motion samples to generate per caption (G)')
    parser.add_argument('--grpo-beta', type=float, default=0.04,
                        help='KL penalty coefficient against frozen SFT reference')
    parser.add_argument('--grpo-kl-target', type=float, default=5.0,
                        help='KL divergence vs frozen SFT; early exit inner loop if exceeded')
    parser.add_argument('--inner-steps', type=int, default=3,
                        help='Number of inner optimization steps per batch (K in GRPO)')
    parser.add_argument('--grpo-clip-epsilon', type=float, default=0.2,
                        help='PPO-style clipping epsilon (ratio clipped to [1-eps, 1+eps])')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='Scaling factor for rewards')
    parser.add_argument('--reward-length-penalty', type=float, default=0.01,
                        help='Penalty per generated length fraction; discourages max-length run-ons')
    parser.add_argument('--reward-tau', type=float, default=0.1,
                        help='Temperature for InfoNCE reward (default 0.1)')
    parser.add_argument('--physical-weight', type=float, default=0.5,
                        help='Weight for physical plausibility reward (includes stillness penalty)')
    parser.add_argument('--numerical-weight', type=float, default=0.8,
                        help='Weight for numerical accuracy + kinematic reward')
    parser.add_argument('--constraint-parser-mode', type=str, default='hybrid',
                        choices=['regex', 'llm', 'hybrid'],
                        help='Caption constraint parser: regex only, LLM only, or LLM with regex fallback/backfill')
    parser.add_argument('--constraint-parser-max-new-tokens', type=int, default=192,
                        help='Max new tokens for LLM constraint JSON parsing')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature for generation')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Reserved for future micro-batch accumulation; currently must be 1')

    # Training parameters
    parser.add_argument('--learning-rate', type=float, default=2e-6, help='learning rate (lower for RL stability)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size (number of captions)')
    parser.add_argument('--epochs', type=int, default=10, help='number of GRPO epochs')
    parser.add_argument('--warmup-steps', type=int, default=20, help='warmup steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--max-train-batches', type=int, default=0,
                        help='If >0, stop each epoch after this many processed batches (useful for dry runs)')

    # Model checkpoints
    parser.add_argument('--sft-checkpoint', type=str, default=None,
                        help='Path to SFT checkpoint to load (warmstart GRPO)')
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help='Path to GRPO checkpoint to resume training')

    # Validation
    parser.add_argument('--epochs-start-val', type=int, default=2, help='start validation after N epochs')
    parser.add_argument('--epochs-val-interval', type=int, default=2, help='validation interval')

    # LLM parameters (from original train script)
    parser.add_argument('--llm-backbone', type=str, default=_DEFAULT_LLM_BACKBONE)
    parser.add_argument('--lora-r-t2m', type=int, default=64)
    parser.add_argument('--lora-alpha-t2m', type=int, default=64)
    parser.add_argument('--lora-r-m2t', type=int, default=32)
    parser.add_argument('--lora-alpha-m2t', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)

    # VQ-VAE parameters (from original train script)
    parser.add_argument('--dataname', type=str, default='t2m')
    parser.add_argument("--code-dim", type=int, default=512)
    parser.add_argument("--nb-code", type=int, default=512)
    parser.add_argument("--mu", type=float, default=0.99)
    parser.add_argument("--down-t", type=int, default=2)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--output-emb-width", type=int, default=512)
    parser.add_argument('--vq-act', type=str, default='relu')
    parser.add_argument('--vq-norm', type=str, default=None)
    parser.add_argument("--quantizer", type=str, default='ema_reset')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--vq-path', type=str, default="ckpt/vqvae.pth")

    # Output
    parser.add_argument('--out-dir', type=str, default='experiments_grpo')
    parser.add_argument('--exp-name', type=str, default='grpo_test')

    return parser.parse_args()


class GRPOTrainer:
    """
    GRPO Trainer for MotionLLM

    Implements Group Relative Policy Optimization with:
    - Frozen SFT reference weights for KL anchoring
    - Group sampling (G samples per prompt)
    - Advantage estimation from reward model
    - PPO-style clipped inner-loop updates
    """

    def __init__(self, args, model: MotionLLM, reward_model: GRPORewardModel, logger):
        self.args = args
        self.model = model
        self.reward_model = reward_model
        self.logger = logger
        self.device = args.device
        if getattr(args, 'gradient_accumulation_steps', 1) != 1:
            raise ValueError(
                "--gradient-accumulation-steps is not implemented for GRPO yet; "
                "use 1 to avoid stale on-policy references."
            )

        # Set model to training mode
        self.model.train()
        self.model.training_task = 't2m'
        self.model.llm.set_adapter('t2m')

        # Enable gradient checkpointing to reduce activation memory
        self.model.llm.enable_input_require_grads()
        self.model.llm.gradient_checkpointing_enable()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler (cosine with warmup)
        self.global_step = 0
        self.warmup_steps = args.warmup_steps
        self.total_train_steps = 1000  # overridden from main() after dataloader is created

        # Statistics
        self.running_stats = {
            'loss': [],
            'reward': [],
            'mean_logprob': [],
            'kl': [],
            'sft_kl': [],
            'ratio': [],
            'clip_fraction': [],
            'inner_steps_used': [],
        }

        # Frozen SFT reference state (populated by save_reference_state() after SFT load)
        self.ref_lora = None
        self.ref_embeddings = None
        self.ref_lm_head = None

    def get_lr(self):
        """Cosine learning rate schedule with warmup"""
        if self.global_step < self.warmup_steps:
            return self.args.learning_rate * (self.global_step / self.warmup_steps)
        else:
            progress = (self.global_step - self.warmup_steps) / max(1, self.total_train_steps - self.warmup_steps)
            return self.args.learning_rate * 0.5 * (1.0 + np.cos(np.pi * progress))

    def update_lr(self):
        """Update learning rate"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def save_reference_state(self):
        """Save current model weights as frozen SFT reference for KL anchoring.
        Must be called AFTER loading SFT checkpoint, BEFORE any GRPO updates."""
        self.ref_lora = {}
        for name, param in self.model.llm.named_parameters():
            if 'lora' in name:
                self.ref_lora[name] = param.data.clone()
        nb = self.model.nb_text_tokens
        self.ref_embeddings = self.model.llm.get_input_embeddings().weight.data[nb:].clone()
        self.ref_lm_head = self.model.llm.lm_head.weight.data[nb:].clone()
        self.logger.info(f"[OK] Frozen SFT reference saved ({len(self.ref_lora)} LoRA tensors)")

    def _swap_to_ref(self):
        """Temporarily point trainable policy weights at the frozen SFT snapshot."""
        if self.ref_lora is None or self.ref_embeddings is None or self.ref_lm_head is None:
            raise RuntimeError("Frozen SFT reference is missing; call save_reference_state() before training.")
        self._policy_lora = {}
        for name, param in self.model.llm.named_parameters():
            if name in self.ref_lora:
                self._policy_lora[name] = param.data  # save pointer (zero-copy)
                param.data = self.ref_lora[name]       # point to ref (zero-copy)
        nb = self.model.nb_text_tokens
        emb = self.model.llm.get_input_embeddings().weight
        lmh = self.model.llm.lm_head.weight
        self._policy_emb = emb.data[nb:].clone()
        self._policy_lmh = lmh.data[nb:].clone()
        emb.data[nb:] = self.ref_embeddings
        lmh.data[nb:] = self.ref_lm_head

    def _swap_to_policy(self):
        """Restore the current policy weights after a reference forward pass."""
        for name, param in self.model.llm.named_parameters():
            if name in self._policy_lora:
                param.data = self._policy_lora[name]
        nb = self.model.nb_text_tokens
        self.model.llm.get_input_embeddings().weight.data[nb:] = self._policy_emb
        self.model.llm.lm_head.weight.data[nb:] = self._policy_lmh

    def group_sample(self, captions: List[str], num_samples: int) -> List[List[torch.Tensor]]:
        """
        Sample G motion sequences for each caption using batch generation.

        Args:
            captions: List of B captions
            num_samples: G (number of samples per caption)

        Returns:
            sampled_motions: List[B][G] of motion token tensors
        """
        batch_size = len(captions)

        # Expand: repeat each caption G times -> B*G captions
        expanded_captions = []
        for caption in captions:
            expanded_captions.extend([caption] * num_samples)

        self.model.eval()
        with torch.no_grad():
            flat_motions = self.model.generate_batch_motion_sampling(
                expanded_captions,
                temperature=self.args.temperature,
                top_p=0.9,
                max_length=200,
                sub_batch_size=128,
            )
        maybe_empty_cache(self.device)

        # Reshape flat list back to [B][G]
        all_sampled_motions = []
        for i in range(batch_size):
            caption_samples = flat_motions[i * num_samples : (i + 1) * num_samples]
            if len(caption_samples) != num_samples:
                self.logger.warning(
                    "Generation returned %d/%d samples for caption index %d",
                    len(caption_samples), num_samples, i,
                )
            all_sampled_motions.append(caption_samples)

        self.model.train()
        return all_sampled_motions

    def compute_batch_log_probs(
        self,
        captions: List[str],
        motion_tokens_list: List[torch.Tensor],
        use_ref_model: bool = False,
        chunk_size: int = 4
    ) -> List[torch.Tensor]:
        """
        Compute log probabilities for a batch of caption-motion pairs,
        using mini-batch chunking to avoid OOM on large batches.

        Args:
            captions: List of N text prompts
            motion_tokens_list: List of N motion token tensors
            use_ref_model: If True, disable LoRA adapters. Frozen SFT reference
                log-probs are computed by swapping weights before calling this.
            chunk_size: Max sequences per forward pass (controls VRAM usage)

        Returns:
            List of N log-prob tensors, one per sample
        """
        prompt_prefix = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\nGenerate a motion matching the following input "
            "human motion description\n\n### Input:\n"
        )
        prompt_suffix = "\n\nResponse: <Motion>"
        eom_eos_ids = self.model.tokenizer.encode(
            '</Motion><eos>', return_tensors="pt", add_special_tokens=False
        ).squeeze(0)

        batch_input_ids = []
        prompt_lens = []
        motion_lens = []
        motion_vocab_ids_list = []

        for caption, motion_tokens in zip(captions, motion_tokens_list):
            full_input = prompt_prefix + caption + prompt_suffix
            prompt_ids = self.model.tokenizer.encode(
                full_input, return_tensors="pt", add_special_tokens=True
            ).squeeze(0)

            motion_vocab_ids = motion_tokens + (len(self.model.tokenizer) - self.args.nb_code)
            input_ids = torch.cat([
                prompt_ids,
                motion_vocab_ids.to(prompt_ids.device),
                eom_eos_ids.to(prompt_ids.device),
            ])

            batch_input_ids.append(input_ids)
            prompt_lens.append(len(prompt_ids))
            motion_lens.append(len(motion_tokens))
            motion_vocab_ids_list.append(motion_vocab_ids)

        pad_value = (
            self.model.tokenizer.pad_token_id
            if self.model.tokenizer.pad_token_id is not None
            else self.model.tokenizer.eos_token_id
        )

        # Chunked forward passes to fit in VRAM
        all_log_probs = []
        N = len(captions)
        label = "ref_logprobs" if use_ref_model else "policy_logprobs"

        for chunk_start in tqdm(range(0, N, chunk_size), desc=label,
                                total=(N + chunk_size - 1) // chunk_size, leave=False):
            chunk_end = min(chunk_start + chunk_size, N)
            chunk_ids = batch_input_ids[chunk_start:chunk_end]

            padded_ids = rnn_utils.pad_sequence(
                chunk_ids, batch_first=True, padding_value=pad_value
            ).to(self.device)
            attention_mask = padded_ids.ne(pad_value).long().to(self.device)

            if use_ref_model:
                with self.model.llm.disable_adapter():
                    outputs = self.model.llm(
                        input_ids=padded_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
            else:
                self.model.llm.set_adapter('t2m')
                outputs = self.model.llm(
                    input_ids=padded_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

            logits = outputs.logits

            for i in range(len(chunk_ids)):
                idx = chunk_start + i
                p_len = prompt_lens[idx]
                m_len = motion_lens[idx]
                sample_logits = logits[i, p_len - 1 : p_len - 1 + m_len, :]
                log_probs_all = F.log_softmax(sample_logits, dim=-1)
                mvids = motion_vocab_ids_list[idx].to(self.device)
                log_probs = log_probs_all[torch.arange(m_len, device=self.device), mvids]
                all_log_probs.append(log_probs)

            del outputs, logits, padded_ids, attention_mask
            maybe_empty_cache(self.device)

        return all_log_probs

    def compute_grpo_loss(
        self,
        flat_captions: List[str],
        flat_motions: List[torch.Tensor],
        flat_advantages: torch.Tensor,
        old_log_probs_list: List[torch.Tensor],
        ref_log_probs_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute GRPO clipped surrogate loss + KL penalty for one inner step.

        Uses TWO separate references (DeepSeekMath-style):
        - old_log_probs: per-batch policy snapshot -> for importance ratio & clipping
        - ref_log_probs: frozen SFT model -> for KL anchoring (prevents long-term drift)

        Args:
            flat_captions: List of N captions (already flattened from B*G)
            flat_motions: List of N motion token tensors
            flat_advantages: Tensor of N advantage values
            old_log_probs_list: List of N detached log-prob tensors (per-batch snapshot, for ratio)
            ref_log_probs_list: List of N detached log-prob tensors (frozen SFT, for KL)

        Returns:
            loss: scalar tensor with gradient
            stats: dict with kl, sft_kl, ratio, clip_fraction, mean_logprob
        """
        if not flat_captions:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                'mean_logprob': 0.0, 'kl': 0.0, 'sft_kl': 0.0, 'ratio': 1.0, 'clip_fraction': 0.0
            }

        N = len(flat_captions)
        eps = self.args.grpo_clip_epsilon
        beta = self.args.grpo_beta

        # Forward pass with gradients -> new log probs
        new_log_probs_list = self.compute_batch_log_probs(
            flat_captions, flat_motions, use_ref_model=False
        )

        total_loss = torch.tensor(0.0, device=self.device)
        total_kl = 0.0
        total_sft_kl = 0.0
        total_ratio = 0.0
        total_clip_count = 0
        total_token_count = 0
        total_mean_logprob = 0.0

        for i in range(N):
            old_lp = old_log_probs_list[i]  # per-batch snapshot, for ratio
            ref_lp = ref_log_probs_list[i]  # frozen SFT, for KL anchoring
            new_lp = new_log_probs_list[i]  # current policy, has grad

            # Per-token importance ratio (vs per-batch snapshot)
            ratio = torch.exp(new_lp - old_lp)

            # Per-token KL divergence vs FROZEN SFT reference (for anchoring)
            sft_kl = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0

            adv = flat_advantages[i]

            # Clipped surrogate objective (per-token, then mean over tokens)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
            clipped_surrogate = torch.min(surr1, surr2).mean()

            # KL penalty vs frozen SFT (mean over tokens)
            kl_penalty = sft_kl.mean()

            # Loss for this sample: -surrogate + beta * KL_sft
            sample_loss = -clipped_surrogate + beta * kl_penalty
            total_loss = total_loss + sample_loss / N

            # Stats (detached)
            with torch.no_grad():
                # KL vs per-batch snapshot (for monitoring inner-step drift)
                per_batch_kl = torch.exp(old_lp - new_lp) - (old_lp - new_lp) - 1.0
                total_kl += per_batch_kl.mean().item()
                total_sft_kl += sft_kl.mean().item()
                total_ratio += ratio.mean().item()
                clipped = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float()
                total_clip_count += clipped.sum().item()
                total_token_count += len(ratio)
                total_mean_logprob += new_lp.mean().item()

        stats = {
            'mean_logprob': total_mean_logprob / N,
            'kl': total_kl / N,
            'sft_kl': total_sft_kl / N,
            'ratio': total_ratio / N,
            'clip_fraction': total_clip_count / max(total_token_count, 1),
        }
        return total_loss, stats

    def train_step(self, batch) -> dict:
        """
        Single GRPO training step with K inner optimization steps.

        Steps:
        1. Group sample G motions per caption (on-policy generation)
        2. Compute rewards and group-relative advantages
        3. Snapshot old_log_probs from current policy (no_grad)
        4. K inner steps: clipped surrogate + KL penalty, early exit if KL > target
        5. Update learning rate and return metrics
        """
        word_embeddings, pos_one_hots, captions, sent_len, motion, m_length, token, name = batch
        batch_size = len(captions)
        G = self.args.num_samples_per_prompt

        # Step 1: Group sampling
        self.logger.debug(f"Sampling {G} motions per caption...")
        sampled_motions_all = self.group_sample(captions, G)

        # Disable dropout for ALL log-prob computations (old, ref, and inner-loop new).
        # model.eval() disables dropout but does NOT prevent gradient computation.
        # This is critical: with trained LoRA weights, lora_dropout=0.1 in train()
        # mode causes different dropout masks between old/ref/new forward passes,
        # injecting noise that dominates the KL and ratio signals.
        self.model.eval()

        # Step 2: Compute rewards and advantages (batched)
        flat_captions = []
        flat_motions = []
        valid_indices = []  # track (b, g) for valid samples
        for b in range(batch_size):
            for g in range(G):
                if len(sampled_motions_all[b][g]) == 0:
                    continue
                flat_captions.append(captions[b])
                flat_motions.append(sampled_motions_all[b][g])
                valid_indices.append((b, g))

        if not flat_captions:
            return {
                'loss': 0.0, 'reward': 0.0, 'mean_logprob': 0.0,
                'kl': 0.0, 'ratio': 1.0, 'clip_fraction': 0.0,
                'inner_steps_used': 0, 'lr': self.update_lr(),
                'pos_sim': 0.0, 'neg_sim': 0.0,
                'physical': 0.0, 'numerical': 0.0, 'num_frac': 0.0,
            }

        # Compute rewards for all valid samples
        all_rewards = self.reward_model.compute_reward(flat_captions, flat_motions)
        reward_stats = getattr(self.reward_model, '_reward_stats', {})

        # Group rewards by caption and compute group-relative advantages
        # Build per-caption reward groups
        caption_rewards = {}  # b -> list of (flat_idx, reward)
        for flat_idx, (b, g) in enumerate(valid_indices):
            if b not in caption_rewards:
                caption_rewards[b] = []
            caption_rewards[b].append((flat_idx, all_rewards[flat_idx]))

        flat_advantages = torch.zeros(len(flat_captions), device=self.device)
        for b, items in caption_rewards.items():
            rewards = torch.stack([r for _, r in items])
            mean_r = rewards.mean()
            std_r = rewards.std(unbiased=False)
            if not torch.isfinite(std_r) or std_r < 1e-8:
                std_r = torch.ones((), device=self.device, dtype=rewards.dtype)
            for (flat_idx, _), adv in zip(items, (rewards - mean_r) / std_r):
                flat_advantages[flat_idx] = adv

        total_reward = all_rewards.mean().item()

        # Free sampling/reward memory before forward passes
        maybe_empty_cache(self.device)

        # Step 3: Snapshot old_log_probs from CURRENT POLICY (on-policy reference for ratio)
        with torch.no_grad():
            old_log_probs_list = self.compute_batch_log_probs(
                flat_captions, flat_motions, use_ref_model=False
            )
            old_log_probs_list = [lp.detach() for lp in old_log_probs_list]

        maybe_empty_cache(self.device)

        # Step 3b: Compute ref_log_probs from FROZEN SFT model (for KL anchoring)
        self._swap_to_ref()
        try:
            with torch.no_grad():
                ref_log_probs_list = self.compute_batch_log_probs(
                    flat_captions, flat_motions, use_ref_model=False
                )
                ref_log_probs_list = [lp.detach() for lp in ref_log_probs_list]
        finally:
            self._swap_to_policy()

        maybe_empty_cache(self.device)

        # Step 4: K inner optimization steps
        K = self.args.inner_steps
        last_stats = {}
        inner_steps_used = 0

        for k in range(K):
            self.optimizer.zero_grad()

            loss, stats = self.compute_grpo_loss(
                flat_captions, flat_motions, flat_advantages, old_log_probs_list, ref_log_probs_list
            )

            loss_value = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            del loss
            maybe_empty_cache(self.device)

            last_stats = stats
            last_stats['loss'] = loss_value
            inner_steps_used = k + 1

            # Early exit if KL vs frozen SFT exceeds target
            if stats['sft_kl'] > self.args.grpo_kl_target:
                self.logger.debug(f"  Inner step {k+1}/{K}: SFT_KL={stats['sft_kl']:.4f} > target={self.args.grpo_kl_target}, early exit")
                break

        # Restore train mode for next batch's group_sample
        self.model.train()

        # Step 5: Update learning rate
        current_lr = self.update_lr()
        self.global_step += 1

        metrics = {
            'loss': last_stats.get('loss', 0.0),
            'reward': total_reward,
            'mean_logprob': last_stats.get('mean_logprob', 0.0),
            'kl': last_stats.get('kl', 0.0),
            'sft_kl': last_stats.get('sft_kl', 0.0),
            'ratio': last_stats.get('ratio', 1.0),
            'clip_fraction': last_stats.get('clip_fraction', 0.0),
            'inner_steps_used': inner_steps_used,
            'lr': current_lr,
            'pos_sim': reward_stats.get('pos_sim_mean', 0.0),
            'neg_sim': reward_stats.get('neg_sim_mean', 0.0),
            'physical': reward_stats.get('physical_mean', 0.0),
            'numerical': reward_stats.get('numerical_mean', 0.0),
            'num_frac': reward_stats.get('numerical_frac', 0.0),
            'kinematic': reward_stats.get('kinematic_mean', 0.0),
            'kin_frac': reward_stats.get('kinematic_frac', 0.0),
            'executor': reward_stats.get('executor_mean', 0.0),
            'exec_frac': reward_stats.get('executor_frac', 0.0),
        }

        return metrics

    def train_epoch(self, train_loader, epoch: int, start_batch_idx: int = 0, best_reward: float = -float('inf')):
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            'loss': [],
            'reward': [],
            'mean_logprob': [],
            'kl': [],
            'sft_kl': [],
            'ratio': [],
            'clip_fraction': [],
            'inner_steps_used': [],
            'pos_sim': [],
            'neg_sim': [],
            'physical': [],
            'numerical': [],
            'kinematic': [],
            'executor': [],
        }
        last_batch_idx = start_batch_idx - 1
        processed_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            if batch_idx < start_batch_idx:
                continue
            metrics = self.train_step(batch)
            last_batch_idx = batch_idx
            processed_batches += 1

            # Log metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])

            # Update progress bar with live metrics
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                reward=f"{metrics['reward']:.4f}",
                sft_kl=f"{metrics['sft_kl']:.4f}",
                ratio=f"{metrics['ratio']:.3f}",
                clip=f"{metrics['clip_fraction']:.3f}",
                K=f"{metrics['inner_steps_used']}",
                lr=f"{metrics['lr']:.1e}",
            )

            # Detailed log every batch
            self.logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {metrics['loss']:.4f}, Reward: {metrics['reward']:.4f}, "
                f"LogProb: {metrics['mean_logprob']:.4f}, "
                f"KL: {metrics['kl']:.4f}, SFT_KL: {metrics['sft_kl']:.4f}, "
                f"Ratio: {metrics['ratio']:.3f}, "
                f"ClipFrac: {metrics['clip_fraction']:.3f}, "
                f"InnerK: {metrics['inner_steps_used']}, "
                f"LR: {metrics['lr']:.6f}, "
                f"PosSim: {metrics['pos_sim']:.4f}, NegSim: {metrics['neg_sim']:.4f}, "
                f"Phys: {metrics.get('physical', 0):.4f}, Num: {metrics.get('numerical', 0):.4f}, "
                f"Kin: {metrics.get('kinematic', 0):.4f}, Exec: {metrics.get('executor', 0):.4f}"
            )

            # Save checkpoint every 20 batches
            if (batch_idx + 1) % 20 == 0:
                state_path = os.path.join(self.args.out_dir, 'grpo_state.pth')
                self.save_state(state_path, epoch, batch_idx + 1, best_reward=best_reward)
                self.logger.info(f"[CHECKPOINT] Saved at Epoch {epoch} Batch {batch_idx+1}")

            if self.args.max_train_batches > 0 and processed_batches >= self.args.max_train_batches:
                self.logger.info(
                    f"[DRY RUN] Reached max_train_batches={self.args.max_train_batches}, "
                    f"stopping epoch {epoch + 1} early after batch {batch_idx + 1}"
                )
                break

        # Aggregate epoch metrics
        avg_metrics = {
            key: (float(np.mean(values)) if len(values) > 0 else 0.0)
            for key, values in epoch_metrics.items()
        }
        avg_metrics['last_batch_idx'] = last_batch_idx
        avg_metrics['processed_batches'] = processed_batches
        return avg_metrics

    def save_state(self, path, epoch, next_batch_idx, best_reward=-float('inf')):
        """Save full training state.

        `epoch` and `batch_idx` describe the next batch to run, not the batch
        that just finished. This avoids replaying an already-optimized batch
        after resume.
        """
        state = {
            'state_schema_version': 2,
            'epoch': epoch,
            'batch_idx': next_batch_idx,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': best_reward,
            # Save frozen SFT reference for KL anchoring
            'ref_lora': self.ref_lora,
            'ref_embeddings': self.ref_embeddings,
            'ref_lm_head': self.ref_lm_head,
        }
        torch.save(state, path)
        self.model.save_model(path.replace('_state.pth', '_model.pth'))

    def load_state(self, path):
        """Load training state to resume"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = state['global_step']
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        # Restore frozen SFT reference
        if 'ref_lora' in state and state['ref_lora'] is not None:
            self.ref_lora = state['ref_lora']
            self.ref_embeddings = state['ref_embeddings']
            self.ref_lm_head = state['ref_lm_head']
            self.logger.info(f"[RESUME] Restored frozen SFT reference ({len(self.ref_lora)} LoRA tensors)")
        best_reward = state.get('best_reward', -float('inf'))
        if state.get('state_schema_version', 1) < 2:
            next_epoch = state['epoch']
            next_batch = state['batch_idx'] + 1
            self.logger.info(
                "[RESUME] Converted legacy checkpoint cursor to Epoch %d, Batch %d",
                next_epoch, next_batch + 1,
            )
            return next_epoch, next_batch, best_reward
        return state['epoch'], state['batch_idx'], best_reward


def main():
    args = get_grpo_args()

    # Setup output directory
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)
    logger = get_logger(args.out_dir)
    logger.info("="*60)
    logger.info("GRPO Training for MotionLLM")
    logger.info("="*60)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # Step 1: Load MotionLLM
    logger.info("\n[1/5] Loading MotionLLM...")
    model = MotionLLM(args)
    model.train()

    # Load SFT checkpoint if provided (warm-start)
    if args.sft_checkpoint is not None and os.path.exists(args.sft_checkpoint):
        logger.info(f"Loading SFT checkpoint from {args.sft_checkpoint}")
        model.load_model(args.sft_checkpoint)

    logger.info(f"[OK] Model loaded on {args.device}")

    # Step 2: Load evaluation models for reward computation
    logger.info("\n[2/5] Loading reward model components...")
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    logger.info("[OK] Evaluator loaded")

    # Step 3: Create reward model
    logger.info("\n[3/5] Initializing GRPO reward model...")
    reward_model = GRPORewardModel(
        eval_wrapper=eval_wrapper,
        vqvae_model=model.net,
        word_vectorizer=w_vectorizer,
        device=args.device,
        normalize_reward=False,  # Keep reward scale unsquashed for better discrimination
        reward_scale=args.reward_scale,
        length_penalty_weight=args.reward_length_penalty,
        tau=args.reward_tau,
        physical_weight=args.physical_weight,
        numerical_weight=args.numerical_weight,
        llm=model.llm,
        tokenizer=model.tokenizer,
        constraint_parser_mode=args.constraint_parser_mode,
        constraint_parser_max_new_tokens=args.constraint_parser_max_new_tokens,
    )
    logger.info(
        f"[OK] Reward model initialized (tau={args.reward_tau}, physical_w={args.physical_weight}, "
        f"numerical_w={args.numerical_weight}, parser={args.constraint_parser_mode})"
    )

    # Step 4: Load datasets
    logger.info("\n[4/5] Loading datasets...")
    train_loader = dataset_TM_eval.DATALoader(
        args.dataname,
        "train",
        args.batch_size,
        w_vectorizer,
        unit_length=2**args.down_t
    )
    val_loader = dataset_TM_eval.DATALoader(
        args.dataname,
        "val",
        32,  # Fixed batch size for validation
        w_vectorizer,
        unit_length=2**args.down_t
    )
    logger.info(f"[OK] Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Resolve resume paths before freezing the SFT reference. A resume checkpoint
    # should restore its original reference instead of replacing it with the
    # partially trained policy.
    resume_state_path = os.path.join(args.out_dir, 'grpo_state.pth')
    resume_model_path = os.path.join(args.out_dir, 'grpo_model.pth')
    if args.resume_checkpoint is not None:
        resume_state_path = args.resume_checkpoint
        resume_model_path = args.resume_checkpoint.replace('_state.pth', '_model.pth')
    has_resume = os.path.exists(resume_state_path) and os.path.exists(resume_model_path)

    # Step 5: Create GRPO trainer
    logger.info("\n[5/5] Initializing GRPO trainer...")
    trainer = GRPOTrainer(args, model, reward_model, logger)
    trainer.total_train_steps = args.epochs * len(train_loader)
    trainer.update_lr()  # Apply warmup lr=0 before first batch (override constructor default)
    if not has_resume:
        trainer.save_reference_state()  # Freeze SFT weights as KL anchor
    logger.info("[OK] Trainer initialized")
    logger.info(f"[OK] Hyperparameters: G={args.num_samples_per_prompt}, K={args.inner_steps}, beta={args.grpo_beta}, "
                f"clip_eps={args.grpo_clip_epsilon}, kl_target={args.grpo_kl_target}, lr={args.learning_rate}")

    # Resume from checkpoint if available
    start_epoch = 0
    start_batch = 0
    best_reward = -float('inf')
    if has_resume:
        model.load_model(resume_model_path)
        start_epoch, start_batch, best_reward = trainer.load_state(resume_state_path)
        if start_batch >= len(train_loader):
            start_epoch += 1
            start_batch = 0
        # If ref state was not in checkpoint, re-freeze current weights as fallback
        if trainer.ref_lora is None:
            if args.sft_checkpoint is not None and os.path.exists(args.sft_checkpoint):
                logger.info("[RESUME] No reference in checkpoint; rebuilding it from SFT checkpoint")
                model.load_model(args.sft_checkpoint)
                trainer.save_reference_state()
                model.load_model(resume_model_path)
            else:
                raise RuntimeError(
                    "Resume checkpoint does not contain a frozen SFT reference. "
                    "Provide --sft-checkpoint so it can be rebuilt safely."
                )
        logger.info(f"[RESUME] Loaded checkpoint cursor Epoch {start_epoch}, Batch {start_batch+1}, "
                     f"global_step {trainer.global_step}, best_reward {best_reward:.4f}")
    else:
        start_batch = 0

    # Training loop
    logger.info("\n" + "="*60)
    logger.info("Starting GRPO Training")
    logger.info("="*60)

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*60}")

        # Train for one epoch (resume from start_batch for first epoch only)
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        epoch_metrics = trainer.train_epoch(train_loader, epoch, start_batch_idx=epoch_start_batch, best_reward=best_reward)

        # Log epoch summary
        logger.info(
            f"\nEpoch {epoch+1} Summary: "
            f"Loss: {epoch_metrics['loss']:.4f}, "
            f"Reward: {epoch_metrics['reward']:.4f}, "
            f"LogProb: {epoch_metrics['mean_logprob']:.4f}, "
            f"KL: {epoch_metrics['kl']:.4f}, "
            f"SFT_KL: {epoch_metrics['sft_kl']:.4f}, "
            f"Ratio: {epoch_metrics['ratio']:.3f}, "
            f"ClipFrac: {epoch_metrics['clip_fraction']:.3f}, "
            f"InnerK: {epoch_metrics['inner_steps_used']:.1f}, "
            f"ProcessedBatches: {int(epoch_metrics.get('processed_batches', 0))}"
        )

        # Save latest checkpoint (both model and full state for resume)
        checkpoint_path = os.path.join(args.out_dir, 'motionllm_grpo_latest.pth')
        model.save_model(checkpoint_path)
        state_path = os.path.join(args.out_dir, 'grpo_state.pth')
        trainer.save_state(
            state_path,
            epoch + 1,
            0,
            best_reward=best_reward,
        )
        logger.info(f"[OK] Epoch checkpoint saved to {checkpoint_path}")

        # Validation
        if epoch >= args.epochs_start_val and (epoch + 1) % args.epochs_val_interval == 0:
            logger.info("\nRunning validation...")
            model.eval()
            fid, div, top1, top2, top3, matching, multi = evaluation_test(
                args.out_dir,
                val_loader,
                model,
                eval_wrapper=eval_wrapper,
                draw=False,
                savenpy=True
            )
            model.train()

            logger.info(
                f"Validation Results: FID: {fid:.4f}, Div: {div:.4f}, "
                f"Top1: {top1:.4f}, Top2: {top2:.4f}, Top3: {top3:.4f}, "
                f"Matching: {matching:.4f}, Multi: {multi:.4f}"
            )

            # Save best model based on reward (or matching score)
            if epoch_metrics['reward'] > best_reward:
                best_reward = epoch_metrics['reward']
                best_path = os.path.join(args.out_dir, 'motionllm_grpo_best.pth')
                model.save_model(best_path)
                logger.info(f"[OK] Best model saved! Reward: {best_reward:.4f}")

    logger.info("\n" + "="*60)
    logger.info("GRPO Training Completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
