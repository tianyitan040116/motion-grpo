"""Mac-side dry-run: verify reward signal differentiates 'good' from 'bad' motion.

Without SFT/VQ-VAE/evaluator ckpts we can't run train_grpo.py end-to-end, but
we can verify the part of the reward that purely consumes motion tensors:

  GT motion ~~ what a perfect policy would generate
  shuffled  ~~ correct frames, wrong order (broken dynamics)
  noisy     ~~ random Gaussian (no structure)
  static    ~~ all-zero (no motion)

For each caption we compute the numerical+executor reward path on each
variant. A healthy reward should rank:

  GT >> shuffled >= noisy >= static

and produce no NaN/Inf. If any variant beats GT, the reward is gameable.

Usage:
    /Users/tan/miniforge3/envs/mgpt/bin/python audit/smoke_reward.py
    /Users/tan/miniforge3/envs/mgpt/bin/python audit/smoke_reward.py --n-captions 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from grpo_reward import (
    Direction,
    parse_numerical_constraints,
    parse_direction_sequence,
    analyze_motion_phases,
    score_constraints_against_phases,
    score_direction_sequence,
    _measure_rotation_signed,
    _count_steps_in_range,
    _count_repetitions,
    _stillness_score,
    denormalize_motion,
)
from motion_constraint_executor import MotionConstraintExecutor
from grpo_reward import (
    parse_constraints_regex,
    constraints_to_executor_specs,
    aggregate_executor_score,
)
from utils.motion_utils import recover_from_ric


DATASET = REPO / "dataset"
TEXTS = DATASET / "texts"
MOTIONS = DATASET / "new_joint_vecs"
TRAIN_IDS = (DATASET / "train.txt").read_text().split()
MEAN = np.load(DATASET / "Mean.npy")
STD = np.load(DATASET / "Std.npy")


# ---------------------------------------------------------------------------
# pick captions that exercise multiple reward branches
# ---------------------------------------------------------------------------

def pick_eval_captions(n: int, seed: int = 0) -> List[Tuple[str, str, float, float]]:
    """Sample captions that have at least one numerical constraint AND a
    locally-available GT motion file. Prefer captions with direction to
    stress the most reward paths."""
    rng = random.Random(seed)
    candidates_strong = []  # numeric + direction
    candidates_any = []     # any numeric
    for mid in TRAIN_IDS:
        # require motion file
        if not (MOTIONS / f"{mid}.npy").exists():
            continue
        p = TEXTS / f"{mid}.txt"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if len(parts) < 4:
                continue
            cap = parts[0].strip()
            try:
                t0 = float(parts[2]); t1 = float(parts[3])
            except ValueError:
                t0, t1 = 0.0, 0.0
            cs = parse_numerical_constraints(cap)
            if not cs:
                continue
            has_dir = any(c.direction != Direction.ANY for c in cs)
            if has_dir:
                candidates_strong.append((mid, cap, t0, t1))
            else:
                candidates_any.append((mid, cap, t0, t1))
    rng.shuffle(candidates_strong)
    rng.shuffle(candidates_any)
    picks = candidates_strong[: n // 2] + candidates_any[: n - n // 2]
    return picks[:n]


# ---------------------------------------------------------------------------
# motion variants
# ---------------------------------------------------------------------------

def load_gt_motion(motion_id: str, t0: float, t1: float) -> Optional[np.ndarray]:
    p = MOTIONS / f"{motion_id}.npy"
    if not p.exists():
        return None
    arr = np.load(p).astype(np.float32)
    if t1 > 0:
        f0 = int(round(t0 * 20)); f1 = int(round(t1 * 20))
        if f1 > f0:
            arr = arr[f0:f1]
    if arr.shape[0] < 8:
        return None
    return arr


def variant_gt(gt_raw: np.ndarray) -> np.ndarray:
    """Pass-through. `gt_raw` is HumanML3D raw 263-dim, already in the
    space the reward path expects."""
    return gt_raw.copy()


def variant_shuffle(gt_raw: np.ndarray, seed: int) -> np.ndarray:
    """Same frames, permuted in time. Tests whether the reward can be
    gamed by reordering valid frames."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(gt_raw.shape[0])
    return gt_raw[idx].copy()


def variant_noise(gt_raw: np.ndarray, seed: int) -> np.ndarray:
    """Gaussian noise drawn from the dataset's per-column statistics.

    Real raw motion already lives near (mean=MEAN, std=STD). Drawing noise
    from N(MEAN, STD) gives a stress test in the same range as plausible
    data while violating temporal coherence and physical structure.
    Foot-contact channels are explicitly binarised after sampling so the
    detector sees plausible foot patterns.
    """
    rng = np.random.default_rng(seed)
    out = rng.normal(MEAN[None, :], STD[None, :], size=gt_raw.shape).astype(np.float32)
    out[:, 259:263] = (rng.uniform(0, 1, size=(gt_raw.shape[0], 4)) > 0.5).astype(np.float32)
    return out


def variant_static(gt_raw: np.ndarray) -> np.ndarray:
    """Constant raw mean per frame -- a body holding the dataset's average
    pose with zero variation. Different from N(MEAN, STD) -- this one is
    a frozen pose, no MEAN-driven decoder drift either."""
    return np.broadcast_to(MEAN[None, :], gt_raw.shape).astype(np.float32).copy()


# ---------------------------------------------------------------------------
# isolated reward (numerical + executor + direction; no eval_wrapper)
# ---------------------------------------------------------------------------

EXECUTOR = MotionConstraintExecutor()


def isolated_reward(caption: str, motion_raw_np: np.ndarray) -> dict:
    """Run the reward pipeline minus matching/text branches.

    `motion_raw_np` is HumanML3D's raw 263-dim feature (cols 259:263 are
    binary foot contact channels). HumanML3D files are already in raw
    space, so no denormalization is needed for direct scoring.
    """
    motion_raw = torch.from_numpy(motion_raw_np).float()

    foot_contact = (motion_raw[:, 259:263] > 0.5).float()
    joint_pos = recover_from_ric(motion_raw.unsqueeze(0), joints_num=22).squeeze(0)
    stillness = _stillness_score(joint_pos)
    motion_alive = stillness >= 0.5

    parsed = parse_constraints_regex(caption)
    constraints = parsed.numerical_constraints
    dir_seq = parsed.direction_sequence

    phases = analyze_motion_phases(
        motion_raw, foot_contact, joints=joint_pos,
        min_phase_frames=8, direction_change_threshold=0.6,
    )

    numerical_score = 0.0
    if constraints and motion_alive:
        total_steps = _count_steps_in_range(foot_contact, joints=joint_pos)
        total_rotation = _measure_rotation_signed(motion_raw[:, 0])
        total_reps = _count_repetitions(motion_raw[:, 3])
        numerical_score = score_constraints_against_phases(
            constraints, phases,
            total_steps=total_steps,
            total_rotation_deg=total_rotation,
            total_repetitions=total_reps,
        )

    direction_score = 0.0
    if dir_seq and not constraints and motion_alive:
        direction_score = score_direction_sequence(dir_seq, phases)

    executor_score = 0.0
    specs = constraints_to_executor_specs(parsed, caption)
    if specs and motion_alive:
        try:
            results = EXECUTOR.evaluate(
                motion_raw=motion_raw,
                foot_contact=foot_contact,
                constraints=specs,
                joints=joint_pos,
            )
            executor_score = float(aggregate_executor_score(results))
        except Exception as e:
            executor_score = float("nan")

    return {
        "stillness": float(stillness),
        "motion_alive": bool(motion_alive),
        "numerical": float(numerical_score),
        "direction": float(direction_score),
        "executor": float(executor_score),
        "n_constraints": len(constraints),
        "n_phases": len(phases),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-captions", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "audit" / "smoke_report.json")
    args = ap.parse_args()

    picks = pick_eval_captions(args.n_captions, args.seed)
    print(f"Picked {len(picks)} captions with numerical constraints")

    rows = []
    variants_aggregate = {"gt": [], "shuffle": [], "noise": [], "static": []}
    n_nan = 0

    for k, (mid, cap, t0, t1) in enumerate(picks):
        gt = load_gt_motion(mid, t0, t1)
        if gt is None:
            print(f"  [{k}] {mid}: no motion, skip")
            continue
        T = gt.shape[0]

        variants = {
            "gt":      variant_gt(gt),
            "shuffle": variant_shuffle(gt, seed=args.seed + k),
            "noise":   variant_noise(gt, seed=args.seed + k),
            "static":  variant_static(gt),
        }
        row = {"id": mid, "caption": cap, "T": T, "variants": {}}
        for name, m in variants.items():
            r = isolated_reward(cap, m)
            if any(math.isnan(r[k]) or math.isinf(r[k]) for k in ("numerical", "direction", "executor")):
                n_nan += 1
            row["variants"][name] = r
            total = r["numerical"] + r["direction"] + r["executor"]
            variants_aggregate[name].append(total)
        rows.append(row)

        # Console summary per caption
        line = f"  [{k}] {cap[:55]!r:60}"
        for name in ("gt", "shuffle", "noise", "static"):
            v = row["variants"][name]
            total = v["numerical"] + v["direction"] + v["executor"]
            line += f"  {name}={total:+.2f}"
        print(line)

    print()
    print("=== aggregate (total reward = numerical + direction + executor) ===")
    for name, vals in variants_aggregate.items():
        if vals:
            a = np.array(vals)
            print(f"  {name:8} n={len(a):3}  mean={a.mean():+.3f}  median={np.median(a):+.3f}  "
                  f"min={a.min():+.3f}  max={a.max():+.3f}")
    print()
    print(f"NaN/Inf occurrences across all variants: {n_nan}")

    # Rank check: how often is GT strictly best?
    rank_wins = {"gt": 0, "shuffle": 0, "noise": 0, "static": 0}
    n_rows = 0
    for row in rows:
        totals = {n: row["variants"][n]["numerical"] + row["variants"][n]["direction"]
                  + row["variants"][n]["executor"] for n in variants_aggregate}
        winner = max(totals, key=totals.get)
        rank_wins[winner] += 1
        n_rows += 1
    print("=== winner counts ===")
    for n, c in rank_wins.items():
        print(f"  {n:8} won  {c}/{n_rows} = {c/max(n_rows,1)*100:.1f}%")

    args.out.write_text(json.dumps({
        "n_captions_evaluated": len(rows),
        "aggregates": {n: {"mean": float(np.mean(v)) if v else 0,
                           "median": float(np.median(v)) if v else 0,
                           "n": len(v)} for n, v in variants_aggregate.items()},
        "winner_counts": rank_wins,
        "n_nan": n_nan,
        "rows_first_5": rows[:5],
    }, indent=2, default=str))
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
