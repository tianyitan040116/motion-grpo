"""Group-internal reward audit (the real signal GRPO sees).

For each caption we score K candidate motions:
  - one matched (GT motion for that caption)
  - K-1 mismatched (random other motions)

GRPO computes advantage *within* a group of G samples per prompt. The
question is: can the reward consistently rank the matched motion in the
top-K of the group? If matched is rank-1 just 25% of the time at K=4,
the advantage signal is random and GRPO won't learn.

For reference, three baselines bracket the score:
  - random chance:    1/K rank-1 rate
  - matching-only:    use just cos_sim_01 (the only consistently good
                      signal in the cross-caption audit)
  - full reward:      everything (what train_grpo actually uses)

Usage on remote:
    cd /root/autodl-tmp/motion-agent
    /root/miniconda3/bin/python audit/reward_group_eval.py --n-prompts 50 --K 4
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
)
from dataset.prompt_mix import classify_caption
from models.vqvae import HumanVQVAE
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt


DATASET = REPO / "dataset"
TEXTS = DATASET / "texts"
MOTIONS = DATASET / "new_joint_vecs"
MEAN = np.load(DATASET / "Mean.npy")
STD = np.load(DATASET / "Std.npy")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-prompts", type=int, default=50,
                   help="number of distinct captions to evaluate")
    p.add_argument("--K", type=int, default=4,
                   help="group size (matched + K-1 random)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vq-path", type=str, default="ckpt/vqvae.pth")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path,
                   default=REPO / "audit" / "reward_group_report.json")
    # VQ-VAE arch (matches the published ckpt)
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
    return p.parse_args()


def load_pool(n_prompts: int, K: int, seed: int):
    """Return (prompts, motion_pool):
      prompts:    [{mid, caption, bucket}] of size n_prompts
      motion_pool: [{mid, gt_raw}] of size n_prompts * K or so (the random
                   negatives are drawn from here at scoring time)
    """
    rng = random.Random(seed)
    train_ids = (DATASET / "train.txt").read_text().split()

    by_bucket = {"numeric": [], "direction_only": [], "pure": []}
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
            by_bucket[classify_caption(cap)].append((mid, cap))
            break

    for k in by_bucket:
        rng.shuffle(by_bucket[k])

    per_bucket = n_prompts // 3
    prompts = []
    for b in ("numeric", "direction_only", "pure"):
        for mid, cap in by_bucket[b][:per_bucket]:
            arr = np.load(MOTIONS / f"{mid}.npy").astype(np.float32)
            if arr.shape[0] < 40 or arr.shape[0] >= 200:
                continue
            T = (arr.shape[0] // 4) * 4
            prompts.append({"mid": mid, "caption": cap, "bucket": b,
                            "gt_raw": arr[:T]})
            if sum(1 for p in prompts if p["bucket"] == b) >= per_bucket:
                break

    # Motion pool: prompts themselves are the pool (so negatives are real GT
    # motions from other prompts in the same bucket-mix). This is closer to
    # what GRPO sees from the policy than synthetic noise.
    pool = [{"mid": p["mid"], "gt_raw": p["gt_raw"]} for p in prompts]
    rng.shuffle(pool)
    return prompts, pool


def encode(vq, raw_np, device):
    norm = (raw_np - MEAN) / STD
    t = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = vq.encode(t)
    return tokens.squeeze(0).long().detach().to(device)


def main():
    args = parse_args()
    rng = random.Random(args.seed + 100)
    print(f"device: {args.device}  K={args.K}")

    prompts, pool = load_pool(args.n_prompts, args.K, args.seed)
    print(f"loaded {len(prompts)} prompts; pool size = {len(pool)}")
    bucket_counts = Counter(p["bucket"] for p in prompts)
    print(f"buckets: {dict(bucket_counts)}")

    # Build reward stack
    print("[1/3] VQ-VAE ...")
    vq = HumanVQVAE(args, args.nb_code, args.code_dim, args.output_emb_width,
                    args.down_t, args.stride_t, args.width, args.depth,
                    args.dilation_growth_rate, args.vq_act, args.vq_norm).to(args.device)
    ckpt = torch.load(args.vq_path, map_location=args.device, weights_only=False)
    vq.load_state_dict(ckpt["net"], strict=True)
    vq.eval()

    print("[2/3] evaluator ...")
    w_vectorizer = WordVectorizer("./glove", "our_vab")
    wrapper_opt = get_opt("checkpoints/t2m/Comp_v6_KLD005/opt.txt", args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    print("[3/3] reward model ...")
    reward_model = GRPORewardModel(
        eval_wrapper=eval_wrapper, vqvae_model=vq, word_vectorizer=w_vectorizer,
        device=args.device, normalize_reward=False,
        reward_scale=1.0, length_penalty_weight=0.0, tau=1.0,
        physical_weight=0.5, numerical_weight=1.0,
    )

    # Encode all pool motions to tokens once (cache)
    print(f"\nencoding {len(pool)} pool motions ...")
    t0 = time.time()
    for p in pool:
        p["tokens"] = encode(vq, p["gt_raw"], args.device)
    print(f"  done in {time.time()-t0:.1f}s")

    # For each prompt, build a group of K and score it
    results = []
    print(f"\nscoring {len(prompts)} groups of K={args.K} ...")
    t0 = time.time()
    for pi, p in enumerate(prompts):
        # Find matched tokens (this prompt's own GT)
        matched_tokens = next(q["tokens"] for q in pool if q["mid"] == p["mid"])
        # Sample K-1 random negatives from pool, excluding matched
        negs = [q for q in pool if q["mid"] != p["mid"]]
        rng.shuffle(negs)
        neg_tokens = [q["tokens"] for q in negs[:args.K - 1]]
        neg_mids = [q["mid"] for q in negs[:args.K - 1]]

        # Group: [matched, neg1, neg2, ...]
        tokens_list = [matched_tokens] + neg_tokens
        captions = [p["caption"]] * args.K  # same caption for all K

        rewards, comps = reward_model.compute_reward(
            captions, tokens_list, return_components=True,
        )
        rewards_np = rewards.detach().cpu().numpy()
        matching = comps["matching_scores"].detach().cpu().numpy()

        # Rank of matched (lower = better; 0 = top)
        order = np.argsort(-rewards_np)  # descending
        matched_rank = int(np.where(order == 0)[0][0])  # idx 0 is matched

        results.append({
            "mid": p["mid"], "bucket": p["bucket"], "caption": p["caption"],
            "K": args.K,
            "neg_mids": neg_mids,
            "rewards": [float(x) for x in rewards_np],
            "matching": [float(x) for x in matching],
            "matched_idx": 0,
            "matched_rank": matched_rank,
            "matched_reward": float(rewards_np[0]),
            "best_neg_reward": float(rewards_np[1:].max()),
        })
    print(f"  done in {time.time()-t0:.1f}s")

    # ---- summarize ----
    print("\n" + "="*70)
    print(f"GROUP-INTERNAL RANK @ K={args.K}")
    print("="*70)

    by_b = defaultdict(list)
    for r in results:
        by_b[r["bucket"]].append(r)
    by_b["__all__"] = results

    for b, rs in by_b.items():
        if not rs:
            continue
        ranks = np.array([r["matched_rank"] for r in rs])
        top1 = (ranks == 0).mean()
        top2 = (ranks < 2).mean()
        matched_r = np.array([r["matched_reward"] for r in rs])
        best_neg = np.array([r["best_neg_reward"] for r in rs])
        margin = matched_r - best_neg  # positive when matched is best
        print(f"\n[{b}] n={len(rs)}")
        print(f"  rank-1 rate (matched is top): {top1*100:.1f}%   "
              f"(chance @ K={args.K}: {1/args.K*100:.0f}%)")
        print(f"  rank<2 rate:                  {top2*100:.1f}%")
        print(f"  margin (matched - best_neg):  mean={margin.mean():+.3f}  "
              f"median={np.median(margin):+.3f}")
        print(f"  margin > 0 (matched WINS):    {(margin>0).mean()*100:.1f}%")

    # Per-bucket: only the matching-score alone, as a baseline
    print("\n" + "="*70)
    print(f"BASELINE: ranking by matching_scores ONLY")
    print("="*70)
    for b, rs in by_b.items():
        if not rs:
            continue
        wins = 0
        for r in rs:
            m = np.asarray(r["matching"])
            if m.argmax() == 0:
                wins += 1
        print(f"  {b:15s} matching-only rank-1: {wins}/{len(rs)} ({wins/len(rs)*100:.1f}%)")

    # Save
    args.out.write_text(json.dumps({
        "n_prompts": len(prompts), "K": args.K,
        "device": args.device,
        "bucket_counts": dict(bucket_counts),
        "per_prompt": results,
    }, indent=2, default=str))
    print(f"\n=> full report: {args.out}")

    # Pass/fail
    all_top1 = np.mean([r["matched_rank"] == 0 for r in results])
    chance = 1.0 / args.K
    print("\n" + "="*70)
    print(f"OVERALL: rank-1 {all_top1*100:.1f}% vs chance {chance*100:.1f}%")
    if all_top1 > chance + 0.15:
        print(f"  PASS: meaningfully better than chance")
    elif all_top1 > chance:
        print(f"  WEAK: only marginally better than chance")
    else:
        print(f"  FAIL: not better than chance")


if __name__ == "__main__":
    main()
