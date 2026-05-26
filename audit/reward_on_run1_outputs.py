"""Audit-style cross-check: run the NEW reward (post-collapse-fix) on the
SFT-vs-collapsed-GRPO .npy pairs we generated in this morning's comparison,
and confirm SFT > GRPO_best under the same caption.

Pipeline per file:
    motion_raw [T,263]  (denormalized)
    -> normalize back   -> VQ encode -> tokens
    -> compute_reward(caption, [tokens]) -> per-component scores

Outputs:
    audit/reward_on_run1_outputs.json
    console table

Usage on remote:
    /root/miniconda3/bin/python audit/reward_on_run1_outputs.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from grpo_reward import GRPORewardModel
from models.vqvae import HumanVQVAE
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt


PROMPTS = [
    ('num_walk3m',     'a person walks forward three meters'),
    ('num_spin_cw360', 'a man spins clockwise three hundred and sixty degrees'),
    ('num_jump3',      'a person jumps three times in place'),
    ('dir_back',       'a person walks backward'),
    ('dir_turn_left',  'the person turns left and walks forward'),
    ('dir_turn_right', 'the person walks forward then turns right'),
    ('pure_stretch',   'a person stretches their arms above their head'),
    ('pure_kick',      'a man kicks with his right leg'),
    ('pure_sit_stand', 'a person sits down and then stands up'),
    ('fwd3_back1',     'walk forward three steps then walk backward one step'),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--vq-path', default='ckpt/vqvae.pth')
    p.add_argument('--comparison-dir', default='comparison_outputs_run1')
    p.add_argument('--out', default='audit/reward_on_run1_outputs.json')
    p.add_argument('--reward-tau', type=float, default=1.0)
    p.add_argument('--physical-weight', type=float, default=0.5)
    p.add_argument('--numerical-weight', type=float, default=1.0)
    # VQ-VAE arch args
    p.add_argument('--nb-code', type=int, default=512)
    p.add_argument('--code-dim', type=int, default=512)
    p.add_argument('--output-emb-width', type=int, default=512)
    p.add_argument('--down-t', type=int, default=2)
    p.add_argument('--stride-t', type=int, default=2)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--dilation-growth-rate', type=int, default=3)
    p.add_argument('--vq-act', default='relu')
    p.add_argument('--vq-norm', default=None)
    p.add_argument('--quantizer', default='ema_reset')
    p.add_argument('--mu', type=float, default=0.99)
    p.add_argument('--dataname', default='t2m')
    p.add_argument('--nb-joints', type=int, default=22)
    p.add_argument('--beta', type=float, default=1.0)
    return p.parse_args()


def load_motion_pad_to_4(path):
    arr = np.load(path).astype(np.float32)
    T = (arr.shape[0] // 4) * 4
    if T == 0:
        return None
    return arr[:T]


def encode_motion(vq, motion_denorm: np.ndarray, mean: np.ndarray, std: np.ndarray, device: str):
    norm = (motion_denorm - mean) / std
    t = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        try:
            tokens = vq.encode(t)
        except Exception:
            tokens = vq.vqvae.encode(t)
    if tokens.dim() == 2:
        tokens = tokens.squeeze(0)
    return tokens.long().detach().to(device)


def main():
    args = parse_args()
    print(f'device: {args.device}')

    mean = np.load(REPO / 'dataset' / 'Mean.npy')
    std = np.load(REPO / 'dataset' / 'Std.npy')

    print('[1/3] loading VQ-VAE ...')
    vq = HumanVQVAE(args, args.nb_code, args.code_dim, args.output_emb_width,
                    args.down_t, args.stride_t, args.width, args.depth,
                    args.dilation_growth_rate, args.vq_act, args.vq_norm).to(args.device)
    ckpt = torch.load(args.vq_path, map_location=args.device)
    vq.load_state_dict(ckpt['net'], strict=True)
    vq.eval()

    print('[2/3] loading evaluator ...')
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    wrapper_opt = get_opt('checkpoints/t2m/Comp_v6_KLD005/opt.txt', args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    print('[3/3] building reward model ...')
    reward_model = GRPORewardModel(
        eval_wrapper=eval_wrapper,
        vqvae_model=vq,
        word_vectorizer=w_vectorizer,
        device=args.device,
        normalize_reward=False,
        reward_scale=1.0,
        length_penalty_weight=0.0,
        tau=args.reward_tau,
        physical_weight=args.physical_weight,
        numerical_weight=args.numerical_weight,
        constraint_parser_mode='regex',
    )

    cdir = REPO / args.comparison_dir
    captions = []
    tokens_list = []
    keys = []  # (prompt_key, tag, method)

    for key, prompt in PROMPTS:
        for tag in ('sft', 'grpo_best'):
            for method in ('beam', 'sample_s42', 'sample_s1337'):
                p = cdir / f'{key}__{tag}__{method}.npy'
                if not p.exists():
                    continue
                m = load_motion_pad_to_4(p)
                if m is None or m.shape[0] < 20:
                    continue
                toks = encode_motion(vq, m, mean, std, args.device)
                tokens_list.append(toks)
                captions.append(prompt)
                keys.append((key, tag, method))

    print(f'\nscoring {len(tokens_list)} samples ...')
    t0 = time.time()
    rewards, comp = reward_model.compute_reward(captions, tokens_list, return_components=True)
    print(f'  done in {time.time()-t0:.1f}s')

    rewards_np = rewards.detach().cpu().numpy()

    def to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    cnp = {k: to_np(v) for k, v in comp.items() if isinstance(v, torch.Tensor)}

    # Print per-prompt comparison
    print()
    print(f"{'prompt':<18} {'method':<14} {'SFT_rew':>8} {'GRPO_rew':>9} {'diff':>7} "
          f"{'SFT_gate':>9} {'GRPO_gate':>10} {'SFT_phys':>9} {'GRPO_phys':>10}")
    print('-' * 110)
    rows = []
    for key, prompt in PROMPTS:
        for method in ('beam', 'sample_s42', 'sample_s1337'):
            try:
                i_sft = keys.index((key, 'sft', method))
                i_grpo = keys.index((key, 'grpo_best', method))
            except ValueError:
                continue
            r_sft = rewards_np[i_sft]
            r_grpo = rewards_np[i_grpo]
            g_sft = cnp['energy_gates'][i_sft]
            g_grpo = cnp['energy_gates'][i_grpo]
            p_sft = cnp['physical_gated'][i_sft]
            p_grpo = cnp['physical_gated'][i_grpo]
            print(f"{key:<18} {method:<14} {r_sft:>8.3f} {r_grpo:>9.3f} "
                  f"{r_sft - r_grpo:>+7.3f} {g_sft:>9.2f} {g_grpo:>10.2f} "
                  f"{p_sft:>+9.2f} {p_grpo:>+10.2f}")
            rows.append({
                'prompt_key': key, 'method': method, 'prompt': prompt,
                'sft_reward': float(r_sft), 'grpo_reward': float(r_grpo),
                'diff': float(r_sft - r_grpo),
                'sft_energy_gate': float(g_sft), 'grpo_energy_gate': float(g_grpo),
                'sft_physical_gated': float(p_sft), 'grpo_physical_gated': float(p_grpo),
                'sft_caption_sat_gated': float(cnp['caption_sat_gated'][i_sft]),
                'grpo_caption_sat_gated': float(cnp['caption_sat_gated'][i_grpo]),
                'sft_matching': float(cnp['matching_scores'][i_sft]),
                'grpo_matching': float(cnp['matching_scores'][i_grpo]),
            })
        print()

    # Aggregate
    sft_rewards = [r['sft_reward'] for r in rows]
    grpo_rewards = [r['grpo_reward'] for r in rows]
    sft_wins = sum(1 for r in rows if r['diff'] > 0)
    print(f"SFT mean reward:  {np.mean(sft_rewards):.3f}")
    print(f"GRPO mean reward: {np.mean(grpo_rewards):.3f}")
    print(f"SFT wins {sft_wins}/{len(rows)} ({100*sft_wins/max(1,len(rows)):.0f}%)")

    # Save
    (REPO / args.out).write_text(json.dumps({
        'rows': rows,
        'sft_mean': float(np.mean(sft_rewards)),
        'grpo_mean': float(np.mean(grpo_rewards)),
        'sft_wins': sft_wins,
        'n': len(rows),
    }, indent=2))
    print(f'\n=> report: {args.out}')


if __name__ == '__main__':
    main()
