"""Real-data sanity check on GRPORewardModel.

Plumbs the *full* reward pipeline (matching + physical + numerical +
direction + executor + kinematic) the trainer uses, on real HumanML3D
GT motions encoded through VQ-VAE and back. This is the strongest
sanity check we can run without actually sampling from the LLM:

  GT motion (raw)  --VQVAE.encode-->  tokens
  tokens  --VQVAE.forward_decoder-->  motion_recon (~lossy, but close to GT)
  GRPORewardModel.compute_reward(caption, [tokens])  ->  reward + components

Two batches are scored:

  matched   - each (caption, motion) pair is from the same clip (correct
              alignment) -- the reward SHOULD be high.
  mismatched- captions and motions are randomly shuffled within the batch
              so they no longer correspond -- the reward SHOULD drop, and
              matching score in particular should drop sharply. If it
              doesn't, the matching branch is broken.

For each batch we print per-component mean / median / min / max, plus a
breakdown by caption bucket (numeric / direction / pure).

Outputs:
  audit/reward_realdata_report.json   structured per-sample log
  console                              summary tables

Usage (on remote, after assets are in place):
    cd /root/autodl-tmp/motion-agent
    /root/miniconda3/bin/python audit/reward_real_eval.py --n 100
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

from grpo_reward import (
    GRPORewardModel,
    Direction,
    parse_numerical_constraints,
    parse_direction_sequence,
)
from dataset.prompt_mix import classify_caption
from models.vqvae import HumanVQVAE
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt


# ---------------------------------------------------------------------------
# Args / config
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100,
                   help="number of caption-motion pairs to score per arm")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vq-path", type=str, default="ckpt/vqvae.pth")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=REPO / "audit" / "reward_realdata_report.json")
    # Mirror the GRPORewardModel reward-component weights from the trainer.
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--reward-length-penalty", type=float, default=0.0)
    p.add_argument("--reward-tau", type=float, default=1.0)
    p.add_argument("--physical-weight", type=float, default=0.5)
    p.add_argument("--numerical-weight", type=float, default=1.0)
    # VQ-VAE arch args (match the published checkpoint: 512 codebook entries).
    # Note this differs from option_train.py's nb-code=1024 default; the
    # released ckpt was trained with 512.
    p.add_argument("--nb-code", type=int, default=512)
    p.add_argument("--code-dim", type=int, default=512)
    p.add_argument("--output-emb-width", type=int, default=512)
    p.add_argument("--down-t", type=int, default=2)
    p.add_argument("--stride-t", type=int, default=2)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dilation-growth-rate", type=int, default=3)
    p.add_argument("--vq-act", type=str, default="relu")
    p.add_argument("--vq-norm", type=str, default=None)
    p.add_argument("--dataname", type=str, default="t2m")
    p.add_argument("--nb-joints", type=int, default=22)
    p.add_argument("--quantizer", type=str, default="ema_reset")
    p.add_argument("--mu", type=float, default=0.99)
    # The reward model itself reads a few more constructor args via kwargs;
    # leave any others to defaults inside GRPORewardModel.
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATASET = REPO / "dataset"
TEXTS = DATASET / "texts"
MOTIONS = DATASET / "new_joint_vecs"
MEAN = np.load(DATASET / "Mean.npy")
STD = np.load(DATASET / "Std.npy")


def load_pairs(n: int, seed: int):
    """Pick n (caption, motion_id, gt_raw, bucket) where motion file exists
    and bucket is balanced across numeric / direction / pure."""
    rng = random.Random(seed)
    train_ids = (DATASET / "train.txt").read_text().split()

    by_bucket: Dict[str, list] = {"numeric": [], "direction_only": [], "pure": []}
    for mid in train_ids:
        if not (MOTIONS / f"{mid}.npy").exists():
            continue
        tpath = TEXTS / f"{mid}.txt"
        if not tpath.exists():
            continue
        for line in tpath.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if len(parts) < 4:
                continue
            cap = parts[0].strip()
            bucket = classify_caption(cap)
            by_bucket[bucket].append((mid, cap))
            break  # one caption per motion is enough for this audit

    for k in by_bucket:
        rng.shuffle(by_bucket[k])
    # Balanced sample
    per_bucket = n // 3
    picks = []
    for b in ("numeric", "direction_only", "pure"):
        picks.extend([(b,) + t for t in by_bucket[b][:per_bucket]])
    rng.shuffle(picks)

    out = []
    for bucket, mid, cap in picks:
        arr = np.load(MOTIONS / f"{mid}.npy").astype(np.float32)
        # train_grpo uses min 40 frames; reward decoder expects multiples of 4
        if arr.shape[0] < 40 or arr.shape[0] >= 200:
            continue
        # truncate to multiple of 4 (down_t=2, stride_t=2 -> ds=4)
        T = (arr.shape[0] // 4) * 4
        if T == 0:
            continue
        arr = arr[:T]
        out.append({"mid": mid, "caption": cap, "bucket": bucket,
                    "gt_raw": arr})
    return out[:n]


# ---------------------------------------------------------------------------
# Reward model construction
# ---------------------------------------------------------------------------

def build_reward_model(args):
    print("[1/3] loading VQ-VAE ...")
    vq = HumanVQVAE(args, args.nb_code, args.code_dim, args.output_emb_width,
                    args.down_t, args.stride_t, args.width, args.depth,
                    args.dilation_growth_rate, args.vq_act, args.vq_norm).to(args.device)
    ckpt = torch.load(args.vq_path, map_location=args.device)
    vq.load_state_dict(ckpt["net"], strict=True)
    vq.eval()

    print("[2/3] loading evaluator ...")
    w_vectorizer = WordVectorizer("./glove", "our_vab")
    wrapper_opt = get_opt("checkpoints/t2m/Comp_v6_KLD005/opt.txt", args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    print("[3/3] building GRPORewardModel ...")
    reward_model = GRPORewardModel(
        eval_wrapper=eval_wrapper,
        vqvae_model=vq,
        word_vectorizer=w_vectorizer,
        device=args.device,
        normalize_reward=False,
        reward_scale=args.reward_scale,
        length_penalty_weight=args.reward_length_penalty,
        tau=args.reward_tau,
        physical_weight=args.physical_weight,
        numerical_weight=args.numerical_weight,
    )
    return vq, reward_model


# ---------------------------------------------------------------------------
# Encode GT motion to tokens
# ---------------------------------------------------------------------------

def encode_gt_to_tokens(vq, gt_raw_np: np.ndarray, device: str) -> torch.Tensor:
    """raw 263-dim motion -> normalized -> VQ-VAE.encode -> token indices.

    Returns 1-D LongTensor of token ids that compute_reward expects.
    """
    norm = (gt_raw_np - MEAN) / STD
    t = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0).to(device)  # [1, T, 263]
    with torch.no_grad():
        # HumanVQVAE.encode returns code indices; signatures vary, try both
        try:
            tokens = vq.encode(t)
        except Exception:
            tokens = vq.vqvae.encode(t)
    # Normalize shape: want 1-D LongTensor [T_tok]
    if tokens.dim() == 2:
        tokens = tokens.squeeze(0)
    return tokens.long().detach().to(device)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def score_batch(reward_model, captions, motion_tokens_list, label: str):
    """Call reward_model.compute_reward and extract per-component scores."""
    t0 = time.time()
    out = reward_model.compute_reward(captions, motion_tokens_list, return_components=True)
    dt = time.time() - t0
    print(f"  [{label}] compute_reward({len(captions)} samples) -> {dt:.1f}s")
    return out


def summarize(name: str, arr: np.ndarray):
    if len(arr) == 0:
        return f"  {name:25} n=0"
    return (f"  {name:25} n={len(arr):>4d}  "
            f"mean={arr.mean():+.3f}  median={np.median(arr):+.3f}  "
            f"p10={np.percentile(arr,10):+.3f}  p90={np.percentile(arr,90):+.3f}")


def main():
    args = parse_args()
    print(f"device: {args.device}")

    pairs = load_pairs(args.n, args.seed)
    print(f"loaded {len(pairs)} (caption, motion) pairs")
    bucket_counts = Counter(p["bucket"] for p in pairs)
    print(f"  buckets: {dict(bucket_counts)}")

    vq, reward_model = build_reward_model(args)

    # Encode all GT motions to tokens (this is the "perfect-token" surrogate
    # for what the trained policy would emit).
    print("\nencoding GT motions to tokens ...")
    tokens_list = []
    for p in pairs:
        toks = encode_gt_to_tokens(vq, p["gt_raw"], args.device)
        tokens_list.append(toks)
        p["tokens_len"] = int(toks.shape[0])

    captions_matched = [p["caption"] for p in pairs]
    # Mismatched arm: shuffle the captions but keep token order. Use a fixed
    # derangement so no element stays put (sanity ensures TRUE mismatch).
    rng = random.Random(args.seed + 1)
    perm = list(range(len(pairs)))
    while True:
        rng.shuffle(perm)
        if all(i != j for i, j in enumerate(perm)):
            break
    captions_mismatched = [pairs[perm[i]]["caption"] for i in range(len(pairs))]

    # Run both arms. compute_reward returns either:
    #   torch.Tensor of shape [B] when return_components=False
    #   tuple(rewards, components_dict) when True
    print("\nscoring MATCHED batch ...")
    matched_out = score_batch(reward_model, captions_matched, tokens_list, "matched")
    print("scoring MISMATCHED batch ...")
    mismatched_out = score_batch(reward_model, captions_mismatched, tokens_list, "mismatched")

    def unpack(out):
        if isinstance(out, tuple):
            rewards, components = out
        else:
            rewards, components = out, {}
        rewards_np = rewards.detach().cpu().numpy()
        comp_np = {}
        for k, v in components.items():
            # Skip non-tensor entries like reward_stats (a dict).
            if isinstance(v, torch.Tensor):
                comp_np[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                comp_np[k] = v
            elif hasattr(v, "__iter__") and not isinstance(v, dict):
                try:
                    comp_np[k] = np.asarray(v)
                except Exception:
                    pass
        return rewards_np, comp_np

    m_r, m_c = unpack(matched_out)
    x_r, x_c = unpack(mismatched_out)

    print("\n" + "="*70)
    print("OVERALL")
    print("="*70)
    print(summarize("matched total reward", m_r))
    print(summarize("mismatched total reward", x_r))
    print()
    print("per-component (matched):")
    for k in sorted(m_c):
        print(summarize(f"  matched.{k}", m_c[k]))
    print()
    print("per-component (mismatched):")
    for k in sorted(x_c):
        print(summarize(f"  mismatched.{k}", x_c[k]))

    # Per-bucket breakdown of matched arm
    print("\n" + "="*70)
    print("MATCHED arm by caption bucket")
    print("="*70)
    bucket_idx = defaultdict(list)
    for i, p in enumerate(pairs):
        bucket_idx[p["bucket"]].append(i)
    for b, idxs in bucket_idx.items():
        if not idxs:
            continue
        print(f"\n[{b}] n={len(idxs)}")
        print(summarize("  total reward", m_r[idxs]))
        for k in sorted(m_c):
            print(summarize(f"  {k}", m_c[k][idxs]))

    # Save full per-sample log
    rows = []
    for i, p in enumerate(pairs):
        rows.append({
            "mid": p["mid"], "bucket": p["bucket"], "caption": p["caption"],
            "T": int(p["gt_raw"].shape[0]),
            "tokens_len": p["tokens_len"],
            "matched_reward": float(m_r[i]),
            "matched": {k: float(v[i]) for k, v in m_c.items()},
            "mismatched_caption": captions_mismatched[i],
            "mismatched_reward": float(x_r[i]),
            "mismatched": {k: float(v[i]) for k, v in x_c.items()},
        })
    args.out.write_text(json.dumps({
        "n": len(pairs),
        "device": args.device,
        "bucket_counts": dict(bucket_counts),
        "summary_matched": {
            "total": {"mean": float(m_r.mean()), "median": float(np.median(m_r))},
            **{k: {"mean": float(v.mean()), "median": float(np.median(v))}
               for k, v in m_c.items()},
        },
        "summary_mismatched": {
            "total": {"mean": float(x_r.mean()), "median": float(np.median(x_r))},
            **{k: {"mean": float(v.mean()), "median": float(np.median(v))}
               for k, v in x_c.items()},
        },
        "samples": rows,
    }, indent=2, default=str))
    print(f"\n=> full report: {args.out}")

    # Final sanity assertions for terminal-visible PASS/FAIL
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)
    fails = []
    if m_r.mean() <= x_r.mean():
        fails.append(f"MATCHED reward ({m_r.mean():.3f}) is NOT > MISMATCHED ({x_r.mean():.3f})")
    if "matching_scores" in m_c and "matching_scores" in x_c:
        if m_c["matching_scores"].mean() <= x_c["matching_scores"].mean():
            fails.append(
                f"matching_score didn't distinguish: matched mean "
                f"{m_c['matching_scores'].mean():.3f} <= mismatched "
                f"{x_c['matching_scores'].mean():.3f}")
    if not fails:
        print("  PASS: matched > mismatched on overall and (if present) matching score")
    else:
        for f in fails:
            print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()
