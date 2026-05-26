"""Focused directional audit -- can the reward distinguish forward from
backward, or left from right, on prompts that explicitly call out a
direction?

This is the most pointed test of whether the reward branch reads the
caption's direction. For each motion whose caption contains a directional
word, we run three conditions:

    identity                 : the real motion, the real caption.
    reverse_time             : motion flipped in time + root velocity sign.
                               If caption is "walks forward", reversed
                               motion walks backward; reward should DROP.
    inverted_caption         : swap forward/backward (or left/right) in
                               caption, keep identity motion. Reward
                               should DROP (motion still goes the original
                               way, caption now wants opposite).

If identity wins both, the reward is directionally aware.

Outputs:
    audit/reward_directional_report.json
    console PASS / FAIL summary

Usage:
    /root/miniconda3/bin/python audit/reward_directional_eval.py --n 60
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from grpo_reward import GRPORewardModel
from dataset.prompt_mix import classify_caption
from models.vqvae import HumanVQVAE
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt


DATASET = REPO / 'dataset'
TEXTS = DATASET / 'texts'
MOTIONS = DATASET / 'new_joint_vecs'
MEAN = np.load(DATASET / 'Mean.npy')
STD = np.load(DATASET / 'Std.npy')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=60)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--vq-path', default='ckpt/vqvae.pth')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='audit/reward_directional_report.json')
    p.add_argument('--reward-tau', type=float, default=0.1)
    p.add_argument('--physical-weight', type=float, default=0.5)
    p.add_argument('--numerical-weight', type=float, default=0.8)
    # VQ-VAE arch
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
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--dataname', default='t2m')
    p.add_argument('--nb-joints', type=int, default=22)
    return p.parse_args()


_DIR_PAIRS = [
    ('forward',  'backward'),
    ('backward', 'forward'),
    ('left',     'right'),
    ('right',    'left'),
]


def invert_caption(cap: str) -> Optional[str]:
    """Swap directional words in caption. Return None if no direction word
    is present."""
    lc = cap.lower()
    for a, b in _DIR_PAIRS:
        if re.search(rf'\b{a}\b', lc):
            # Replace ALL occurrences of `a` with a placeholder, then b with a,
            # then placeholder with b -- so a<->b swap is symmetric.
            tmp = re.sub(rf'\b{a}\b', '__TMP__', cap, flags=re.IGNORECASE)
            tmp = re.sub(rf'\b{b}\b', a, tmp, flags=re.IGNORECASE)
            tmp = tmp.replace('__TMP__', b)
            return tmp
    return None


def has_direction(cap: str) -> bool:
    lc = cap.lower()
    return any(re.search(rf'\b{w}\b', lc) for w, _ in _DIR_PAIRS)


def load_directional_pairs(n: int, seed: int):
    rng = random.Random(seed)
    train_ids = (DATASET / 'train.txt').read_text().split()
    rng.shuffle(train_ids)
    out = []
    for mid in train_ids:
        if not (MOTIONS / f'{mid}.npy').exists():
            continue
        tpath = TEXTS / f'{mid}.txt'
        if not tpath.exists():
            continue
        for line in tpath.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split('#')
            if len(parts) < 4:
                continue
            cap = parts[0].strip()
            if not has_direction(cap):
                continue
            inv = invert_caption(cap)
            if inv is None or inv.lower() == cap.lower():
                continue
            arr = np.load(MOTIONS / f'{mid}.npy').astype(np.float32)
            if arr.shape[0] < 60 or arr.shape[0] >= 200:
                continue
            T = (arr.shape[0] // 4) * 4
            arr = arr[:T]
            out.append({'mid': mid, 'caption': cap, 'caption_inverted': inv,
                        'raw': arr})
            break
        if len(out) >= n:
            break
    return out[:n]


def variant_reverse_time(raw: np.ndarray) -> np.ndarray:
    rev = raw[::-1].copy()
    rev[:, 0:3] = -rev[:, 0:3]
    return rev


def encode(vq, raw: np.ndarray, device: str) -> torch.Tensor:
    norm = (raw - MEAN) / STD
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

    pairs = load_directional_pairs(args.n, args.seed)
    print(f'loaded {len(pairs)} directional pairs')
    if not pairs:
        print('NO directional captions found in dataset -- skipping')
        return

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
        eval_wrapper=eval_wrapper, vqvae_model=vq, word_vectorizer=w_vectorizer,
        device=args.device, normalize_reward=False, reward_scale=1.0,
        length_penalty_weight=0.0, tau=args.reward_tau,
        physical_weight=args.physical_weight, numerical_weight=args.numerical_weight,
        constraint_parser_mode='regex',
    )

    # Three conditions per pair:
    #   id_real        identity motion, identity caption  -- baseline
    #   id_reversed    reverse-time motion, identity caption
    #   inv_real       identity motion, inverted caption
    all_captions = []
    all_tokens = []
    index = []  # list of (pair_idx, condition_name)
    for i, p in enumerate(pairs):
        tok_id = encode(vq, p['raw'], args.device)
        tok_rev = encode(vq, variant_reverse_time(p['raw']), args.device)
        if tok_id.numel() == 0 or tok_rev.numel() == 0:
            continue
        all_captions.append(p['caption']);   all_tokens.append(tok_id);  index.append((i, 'id_real'))
        all_captions.append(p['caption']);   all_tokens.append(tok_rev); index.append((i, 'id_reversed'))
        all_captions.append(p['caption_inverted']); all_tokens.append(tok_id); index.append((i, 'inv_real'))

    print(f'\nscoring {len(all_tokens)} samples ...')
    t0 = time.time()
    rewards, comp = reward_model.compute_reward(all_captions, all_tokens, return_components=True)
    print(f'  done in {time.time()-t0:.1f}s')

    rewards_np = rewards.detach().cpu().numpy()

    # Aggregate
    by_cond: Dict[str, List[float]] = defaultdict(list)
    for j, (pi, cn) in enumerate(index):
        by_cond[cn].append(float(rewards_np[j]))

    print('\n' + '='*70)
    print(f"{'condition':<14} {'n':>4} {'mean':>8} {'median':>8} {'p10':>8} {'p90':>8}")
    print('-'*70)
    for cn in ('id_real', 'id_reversed', 'inv_real'):
        arr = np.asarray(by_cond[cn])
        if len(arr) == 0:
            continue
        print(f'{cn:<14} {len(arr):>4} {arr.mean():>8.3f} {np.median(arr):>8.3f} '
              f'{np.percentile(arr,10):>8.3f} {np.percentile(arr,90):>8.3f}')

    # Pairwise per-seed
    print('\n' + '='*70)
    print('PAIRWISE per-seed')
    print('='*70)
    rows = []
    for i in range(len(pairs)):
        ri = {cn: float(rewards_np[j]) for j, (pi, cn) in enumerate(index) if pi == i}
        if {'id_real', 'id_reversed', 'inv_real'} <= set(ri):
            rows.append({'i': i, 'caption': pairs[i]['caption'][:60],
                         'caption_inverted': pairs[i]['caption_inverted'][:60],
                         **ri,
                         'diff_vs_rev': ri['id_real'] - ri['id_reversed'],
                         'diff_vs_inv': ri['id_real'] - ri['inv_real']})
    wins_rev = sum(1 for r in rows if r['diff_vs_rev'] > 0.05)
    losses_rev = sum(1 for r in rows if r['diff_vs_rev'] < -0.05)
    ties_rev = len(rows) - wins_rev - losses_rev
    wins_inv = sum(1 for r in rows if r['diff_vs_inv'] > 0.05)
    losses_inv = sum(1 for r in rows if r['diff_vs_inv'] < -0.05)
    ties_inv = len(rows) - wins_inv - losses_inv
    print(f'id_real vs id_reversed (motion direction flipped, same caption):')
    print(f'  wins={wins_rev}/{len(rows)}, losses={losses_rev}, ties={ties_rev}, '
          f'mean Δ={np.mean([r["diff_vs_rev"] for r in rows]):+.3f}')
    print(f'id_real vs inv_real (same motion, caption direction flipped):')
    print(f'  wins={wins_inv}/{len(rows)}, losses={losses_inv}, ties={ties_inv}, '
          f'mean Δ={np.mean([r["diff_vs_inv"] for r in rows]):+.3f}')

    # Sanity
    print('\n' + '='*70)
    print('SANITY CHECKS')
    print('='*70)
    fails = []
    if not by_cond['id_real'] or not by_cond['id_reversed']:
        fails.append('missing conditions')
    else:
        d_rev = float(np.mean(by_cond['id_real']) - np.mean(by_cond['id_reversed']))
        d_inv = float(np.mean(by_cond['id_real']) - np.mean(by_cond['inv_real']))
        if d_rev <= 0:
            fails.append(f'id_real NOT > id_reversed: Δ={d_rev:+.3f}')
        else:
            print(f'  PASS: id_real > id_reversed by mean Δ={d_rev:+.3f}')
        if d_inv <= 0:
            fails.append(f'id_real NOT > inv_real: Δ={d_inv:+.3f}')
        else:
            print(f'  PASS: id_real > inv_real by mean Δ={d_inv:+.3f}')
    for f in fails:
        print(f'  FAIL: {f}')

    Path(REPO / args.out).write_text(json.dumps({
        'n_pairs': len(rows),
        'by_condition_mean': {cn: float(np.mean(by_cond[cn])) for cn in by_cond},
        'pairwise': rows,
    }, indent=2))
    print(f'\n=> report: {args.out}')


if __name__ == '__main__':
    main()
