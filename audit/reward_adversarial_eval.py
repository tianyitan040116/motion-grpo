"""Adversarial reward audit -- the strongest stress test we can run on the
new energy gate + executor changes without firing up the policy.

For each real (caption, motion) pair we synthesize four degraded variants
that exactly match the failure modes the policy could exploit, then verify
the reward correctly assigns the highest score to the identity sample.

Variants:
    identity        - the real motion, unchanged. Reward should be ~baseline.
    frozen          - every frame replaced by frame[0]. Zero motion of any
                      kind. Energy gate should kill it; physical -> -1.
    drift_only      - root translates linearly along the GT direction but
                      body articulation is removed (frame[t][joints] =
                      frame[0][joints], frame[t][root] is interpolated).
                      Mimics the "stand and slide" failure mode.
    shuffled        - frame order randomly permuted. Same per-frame content
                      but no temporal coherence. Step/phase detectors should
                      fail.
    half_length     - truncated to first 50% of frames. Partial execution.

Pass criteria:
    identity > frozen, identity > drift_only, identity > shuffled
    (half_length can be close to identity; we just log it.)

Outputs:
    audit/reward_adversarial_report.json
    console pass/fail summary

Usage on remote:
    /root/miniconda3/bin/python audit/reward_adversarial_eval.py --n 60
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=60,
                   help='number of real (caption, motion) seeds to degrade')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--vq-path', default='ckpt/vqvae.pth')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='audit/reward_adversarial_report.json')
    # Reward model weights -- same as training defaults
    p.add_argument('--reward-tau', type=float, default=0.1)
    p.add_argument('--physical-weight', type=float, default=0.5)
    p.add_argument('--numerical-weight', type=float, default=0.8)
    # VQ-VAE arch (released 512-codebook checkpoint)
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


DATASET = REPO / 'dataset'
TEXTS = DATASET / 'texts'
MOTIONS = DATASET / 'new_joint_vecs'
MEAN = np.load(DATASET / 'Mean.npy')
STD = np.load(DATASET / 'Std.npy')


def load_pairs(n: int, seed: int):
    """Pick n balanced (caption, motion_id, motion_raw, bucket)."""
    rng = random.Random(seed)
    train_ids = (DATASET / 'train.txt').read_text().split()
    by_bucket: Dict[str, list] = {'numeric': [], 'direction_only': [], 'pure': []}
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
            bucket = classify_caption(cap)
            by_bucket[bucket].append((mid, cap))
            break
    for k in by_bucket:
        rng.shuffle(by_bucket[k])
    per_bucket = n // 3
    picks = []
    for b in ('numeric', 'direction_only', 'pure'):
        picks.extend([(b,) + t for t in by_bucket[b][:per_bucket]])
    rng.shuffle(picks)
    out = []
    for bucket, mid, cap in picks:
        arr = np.load(MOTIONS / f'{mid}.npy').astype(np.float32)
        if arr.shape[0] < 60 or arr.shape[0] >= 200:
            continue
        T = (arr.shape[0] // 4) * 4
        if T == 0:
            continue
        arr = arr[:T]
        out.append({'mid': mid, 'caption': cap, 'bucket': bucket, 'raw': arr})
    return out[:n]


# ---------------------------------------------------------------------------
# Adversarial variants -- operate on the raw 263-dim feature in original space
# ---------------------------------------------------------------------------

def variant_frozen(raw: np.ndarray) -> np.ndarray:
    """Replace every frame with frame[0]. Total motion: zero."""
    f0 = raw[0:1]
    return np.repeat(f0, raw.shape[0], axis=0)


def variant_drift_only(raw: np.ndarray) -> np.ndarray:
    """Body articulation frozen but root translates linearly from raw[0]
    toward raw[-1]'s implied root position. Replicates the run1
    "stand and slide" / mean-velocity failure mode."""
    T = raw.shape[0]
    out = np.repeat(raw[0:1], T, axis=0).copy()
    # Channel 0 = root angular vel (yaw); 1-2 = root xz vel; 3 = root y
    # Keep root velocity from GT (so root moves), but freeze everything else.
    out[:, 0:4] = raw[:, 0:4]
    # The rest (joint local positions, foot_contact, etc.) stays frozen at frame[0].
    return out


def variant_shuffled(raw: np.ndarray, rng: random.Random) -> np.ndarray:
    """Random permutation of frames. Same per-frame content; no temporal
    structure. Phase analyzer should fail, executor counts get garbage."""
    T = raw.shape[0]
    order = list(range(T))
    rng.shuffle(order)
    return raw[order]


def variant_half_length(raw: np.ndarray) -> np.ndarray:
    """First 50% of frames (rounded down to multiple of 4)."""
    T = raw.shape[0] // 2
    T = (T // 4) * 4
    if T == 0:
        T = 4
    return raw[:T]


VARIANTS = ['identity', 'frozen', 'drift_only', 'shuffled', 'half_length']


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

def build_reward_model(args):
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
    return vq, reward_model


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
    rng = random.Random(args.seed)

    pairs = load_pairs(args.n, args.seed)
    print(f'loaded {len(pairs)} seed pairs; buckets={dict(Counter(p["bucket"] for p in pairs))}')

    vq, reward_model = build_reward_model(args)

    # Build flat caption+token lists indexed by (seed_idx, variant).
    all_captions = []
    all_tokens = []
    index = []   # list of (seed_idx, variant_name)
    for i, p in enumerate(pairs):
        for v in VARIANTS:
            if v == 'identity':
                arr = p['raw']
            elif v == 'frozen':
                arr = variant_frozen(p['raw'])
            elif v == 'drift_only':
                arr = variant_drift_only(p['raw'])
            elif v == 'shuffled':
                arr = variant_shuffled(p['raw'], random.Random(args.seed + i))
            elif v == 'half_length':
                arr = variant_half_length(p['raw'])
            tok = encode(vq, arr, args.device)
            if tok.numel() == 0:
                continue
            all_captions.append(p['caption'])
            all_tokens.append(tok)
            index.append((i, v))

    print(f'\nscoring {len(all_tokens)} samples ...')
    t0 = time.time()
    rewards, comp = reward_model.compute_reward(all_captions, all_tokens, return_components=True)
    print(f'  done in {time.time()-t0:.1f}s')

    rewards_np = rewards.detach().cpu().numpy()
    def to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)
    cnp = {k: to_np(v) for k, v in comp.items() if isinstance(v, torch.Tensor)}

    # Aggregate by variant
    by_variant: Dict[str, dict] = {}
    for v in VARIANTS:
        idxs = [j for j, (si, vn) in enumerate(index) if vn == v]
        if not idxs:
            continue
        by_variant[v] = {
            'n': len(idxs),
            'reward_mean': float(rewards_np[idxs].mean()),
            'reward_median': float(np.median(rewards_np[idxs])),
            'reward_p10': float(np.percentile(rewards_np[idxs], 10)),
            'reward_p90': float(np.percentile(rewards_np[idxs], 90)),
            'energy_gate_mean': float(cnp['energy_gates'][idxs].mean()),
            'physical_gated_mean': float(cnp['physical_gated'][idxs].mean()),
            'caption_sat_gated_mean': float(cnp['caption_sat_gated'][idxs].mean()),
            'matching_mean': float(cnp['matching_scores'][idxs].mean()),
        }

    print('\n' + '='*86)
    print(f'{"variant":<14} {"n":>4} {"rew_mean":>9} {"rew_med":>8} {"gate":>7} {"phys_g":>8} {"sat_g":>8} {"match":>7}')
    print('-'*86)
    for v in VARIANTS:
        if v not in by_variant:
            continue
        r = by_variant[v]
        print(f'{v:<14} {r["n"]:>4} {r["reward_mean"]:>9.3f} {r["reward_median"]:>8.3f} '
              f'{r["energy_gate_mean"]:>7.2f} {r["physical_gated_mean"]:>+8.2f} '
              f'{r["caption_sat_gated_mean"]:>8.2f} {r["matching_mean"]:>7.2f}')

    # Per-seed pairwise check: identity vs each variant
    print('\n' + '='*86)
    print('PAIRWISE per-seed (identity vs each variant)')
    print('='*86)
    win_counts: Dict[str, dict] = defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0, 'mean_diff': 0.0})
    pairwise_rows = []
    for seed_idx in range(len(pairs)):
        rew_by_v = {}
        for j, (si, vn) in enumerate(index):
            if si == seed_idx:
                rew_by_v[vn] = float(rewards_np[j])
        if 'identity' not in rew_by_v:
            continue
        rid = rew_by_v['identity']
        for v in VARIANTS:
            if v == 'identity' or v not in rew_by_v:
                continue
            rv = rew_by_v[v]
            diff = rid - rv
            win_counts[v]['mean_diff'] += diff
            if diff > 0.05:
                win_counts[v]['wins'] += 1
            elif diff < -0.05:
                win_counts[v]['losses'] += 1
            else:
                win_counts[v]['ties'] += 1
        pairwise_rows.append({'seed_idx': seed_idx, **rew_by_v,
                              'caption': pairs[seed_idx]['caption'][:60],
                              'bucket': pairs[seed_idx]['bucket']})
    for v, c in win_counts.items():
        n_compared = c['wins'] + c['losses'] + c['ties']
        if n_compared == 0:
            continue
        c['mean_diff'] /= n_compared
        print(f'  identity vs {v:<14} wins={c["wins"]:>3}/{n_compared}  '
              f'losses={c["losses"]:>3}  ties={c["ties"]:>3}  '
              f'mean_diff={c["mean_diff"]:+.3f}')

    # Sanity assertions
    print('\n' + '='*86)
    print('SANITY CHECKS')
    print('='*86)
    fails = []
    expected_lose_vs_identity = ['frozen', 'drift_only', 'shuffled']
    for v in expected_lose_vs_identity:
        if v not in by_variant:
            continue
        diff = by_variant['identity']['reward_mean'] - by_variant[v]['reward_mean']
        if diff <= 0.0:
            fails.append(f'identity mean reward NOT > {v}: '
                         f'identity={by_variant["identity"]["reward_mean"]:.3f} '
                         f'vs {v}={by_variant[v]["reward_mean"]:.3f}')
        else:
            print(f'  PASS: identity > {v} by mean Δ={diff:+.3f}')
    if fails:
        print()
        for f in fails:
            print(f'  FAIL: {f}')
    else:
        print(f'\n  ALL PASS: identity dominates frozen/drift_only/shuffled.')

    Path(REPO / args.out).write_text(json.dumps({
        'n_seeds': len(pairs),
        'by_variant': by_variant,
        'pairwise': pairwise_rows,
        'win_counts': dict(win_counts),
    }, indent=2))
    print(f'\n=> report: {args.out}')


if __name__ == '__main__':
    main()
