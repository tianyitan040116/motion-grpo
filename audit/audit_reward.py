"""Offline audit of the GRPO reward pipeline on HumanML3D train captions.

Three checks, all read-only, no training, no generation:

  A. Parser hit rate + noise sweep
     Run parse_numerical_constraints on every train caption. Report hit-rate
     histogram, sample 30 hits for eyeballing, and surface suspected
     false-positive contexts (e.g. "2x speed", "for 5 minutes", "count to 3").

  B. Detector accuracy on GT motion
     For each caption with a numeric constraint, load the matching GT motion
     and run the same detectors the reward uses. Compare detected
     (steps / degrees / direction) against the parsed target.

  C. Neutral check on captions with no numeric constraints
     For captions that produce zero constraints, verify the reward's
     numerical/direction components don't accidentally fire.

Usage:
    /Users/tan/miniforge3/envs/mgpt/bin/python audit/audit_reward.py
    /Users/tan/miniforge3/envs/mgpt/bin/python audit/audit_reward.py --check A
    /Users/tan/miniforge3/envs/mgpt/bin/python audit/audit_reward.py --max-captions 500
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from grpo_reward import (
    ConstraintPhase,
    Direction,
    parse_numerical_constraints,
    parse_direction_sequence,
    _measure_rotation_signed,
    analyze_motion_phases,
    score_constraints_against_phases,
    _count_steps_in_range,
    denormalize_motion,
)
from motion_step_detector import detect_steps


DATASET = REPO / "dataset"
TEXTS = DATASET / "texts"
MOTIONS = DATASET / "new_joint_vecs"
TRAIN_IDS = (DATASET / "train.txt").read_text().split()


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def iter_train_captions():
    """Yield (motion_id, caption, start_sec, end_sec) for every train caption."""
    for mid in TRAIN_IDS:
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
                t0 = float(parts[2])
                t1 = float(parts[3])
            except ValueError:
                t0, t1 = 0.0, 0.0
            yield mid, cap, t0, t1


def load_motion(motion_id: str, t0: float, t1: float) -> Optional[torch.Tensor]:
    """Load GT motion [T, 263] in RAW 263-dim space.

    HumanML3D's new_joint_vecs/*.npy files are *already* in raw feature
    space -- they have not been normalized by Mean.npy / Std.npy. The
    Text2MotionDataset.__getitem__ path normalises at read time for the
    VQ-VAE; callers that want to score motion directly should use this
    file as-is (no extra denormalize).
    HumanML3D fps = 20. If t0/t1 are 0, use the whole sequence.
    """
    p = MOTIONS / f"{motion_id}.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    if t1 > 0:
        f0 = int(round(t0 * 20))
        f1 = int(round(t1 * 20))
        if f1 > f0:
            arr = arr[f0:f1]
    return torch.from_numpy(arr.astype(np.float32))


# ---------------------------------------------------------------------------
# Check A: parser hit rate + noise sweep
# ---------------------------------------------------------------------------

# Contexts that strongly suggest the matched number is NOT a movement quantity.
# These are patterns the parser should ideally skip, but currently might match.
SUSPECT_CONTEXTS = [
    (re.compile(r"\d+\s*x\s+speed", re.I), "Nx speed"),
    (re.compile(r"\d+\s*x\s*$", re.I), "trailing Nx"),
    (re.compile(r"for\s+\d+\s+(?:second|minute|hour)", re.I), "duration"),
    (re.compile(r"count(?:ing|s|ed)?\s+to\s+\d+", re.I), "count to N"),
    (re.compile(r"\d+\s*(?:second|minute|hour)s?", re.I), "time unit"),
    (re.compile(r"number\s+\d+", re.I), "number N (label)"),
    (re.compile(r"\d+\s*-\s*\d+", re.I), "range N-M"),
    (re.compile(r"figure\s+\d+", re.I), "figure N"),
    (re.compile(r"\bage\s+\d+", re.I), "age"),
    (re.compile(r"\d+\s*years?\s+old", re.I), "age"),
    (re.compile(r"\d+\s*(?:kg|lb|cm|m\b|ft|inch)", re.I), "unit of mass/length"),
    (re.compile(r"at\s+\d+\s*(?:degree|°)", re.I), "angle as posture (at N degrees)"),
]


def check_a_parser(max_captions: Optional[int] = None, seed: int = 0) -> dict:
    hit_hist = Counter()
    type_hist = Counter()
    direction_hist = Counter()
    suspect_hits: List[Tuple[str, str, str]] = []  # (caption, suspect_label, parsed_raw)
    hit_examples: List[Tuple[str, list]] = []
    total = 0
    hits = 0

    for i, (mid, cap, t0, t1) in enumerate(iter_train_captions()):
        if max_captions and i >= max_captions:
            break
        total += 1
        constraints = parse_numerical_constraints(cap)
        hit_hist[len(constraints)] += 1
        if constraints:
            hits += 1
            for c in constraints:
                type_hist[c.type] += 1
                direction_hist[c.direction.value] += 1
            if len(hit_examples) < 200:
                hit_examples.append((cap, [
                    (c.type, c.value, c.direction.value, c.raw) for c in constraints
                ]))

            for pat, label in SUSPECT_CONTEXTS:
                if pat.search(cap):
                    for c in constraints:
                        suspect_hits.append((cap, label, c.raw))
                    break

    rng = random.Random(seed)
    sample = rng.sample(hit_examples, min(30, len(hit_examples)))

    return {
        "total_captions": total,
        "hits": hits,
        "hit_rate": hits / max(total, 1),
        "hit_hist": dict(hit_hist),
        "constraint_type_hist": dict(type_hist),
        "direction_hist": dict(direction_hist),
        "suspect_hits": suspect_hits[:50],
        "suspect_total": len(suspect_hits),
        "eyeball_sample": sample,
    }


# ---------------------------------------------------------------------------
# Check B: detector accuracy on GT motion
# ---------------------------------------------------------------------------

DIRECTION_TO_ANGLE = {
    Direction.FORWARD: 0.0,           # +Z
    Direction.BACKWARD: 180.0,
    Direction.LEFT: 90.0,             # +X in HumanML3D convention
    Direction.RIGHT: -90.0,
    Direction.LEFT_FORWARD: 45.0,
    Direction.RIGHT_FORWARD: -45.0,
    Direction.LEFT_BACKWARD: 135.0,
    Direction.RIGHT_BACKWARD: -135.0,
    Direction.ANY: None,
}


def detect_on_motion(motion_raw: torch.Tensor) -> dict:
    """Run the same detectors reward uses. `motion_raw` is HumanML3D's raw
    263-dim feature (cols 259:263 are the binary foot-contact channels).
    """
    foot_contact = (motion_raw[:, 259:263] > 0.5).float()
    steps = detect_steps(None, foot_contact, detector="move_state").count
    root_rot_vel = motion_raw[:, 0]
    # _measure_rotation_signed already returns degrees (it bakes in 2x for the
    # HumanML3D half-angle quaternion convention and converts to deg).
    rot_deg = _measure_rotation_signed(root_rot_vel)
    return {"steps": steps, "rotation_deg": float(rot_deg)}


def check_b_detector(max_samples: int = 300, seed: int = 0) -> dict:
    rng = random.Random(seed)
    candidates = []  # (mid, cap, t0, t1, constraints)
    for mid, cap, t0, t1 in iter_train_captions():
        cs = parse_numerical_constraints(cap)
        if cs:
            candidates.append((mid, cap, t0, t1, cs))

    sampled = rng.sample(candidates, min(max_samples, len(candidates)))

    # Buckets
    step_errors: List[int] = []
    rot_errors: List[float] = []  # only when GT actually rotated AND sign agrees
    per_sample_log = []
    failures = {"steps": [], "degrees": []}
    deg_buckets = {
        "total": 0,
        "ok_in_range": 0,
        "sign_flip_gt_data_noise": 0,
        "under_motion_gt_data_noise": 0,
        "magnitude_off": 0,
    }
    # End-to-end reward signal on GT motion (this is what GRPO actually sees).
    # If reward is well-aligned, GT motion should score HIGH on its own caption.
    reward_scores_on_gt: List[float] = []
    low_reward_failures = []

    mean = np.load(DATASET / "Mean.npy")
    std = np.load(DATASET / "Std.npy")

    for mid, cap, t0, t1, cs in sampled:
        motion = load_motion(mid, t0, t1)
        if motion is None or motion.shape[0] < 4:
            continue
        foot_contact = (motion[:, 259:263] > 0.5).float()
        detected_steps = detect_steps(None, foot_contact, detector="move_state").count
        # motion from load_motion is already raw; no denormalize needed.
        motion_raw = motion.numpy()
        # _measure_rotation_signed returns degrees (already multiplies the
        # HumanML3D half-angle convention by 2 and converts rad->deg).
        detected_deg = float(_measure_rotation_signed(torch.from_numpy(motion_raw[:, 0])))

        sample_entry = {
            "id": mid, "caption": cap,
            "T": int(motion.shape[0]),
            "detected_steps": detected_steps,
            "detected_deg": round(detected_deg, 1),
            "targets": [],
        }

        for c in cs:
            tgt = {"type": c.type, "value": c.value, "dir": c.direction.value,
                   "range": [c.value_min, c.value_max]}
            if c.type == "steps":
                # Range-aware error: 0 if inside [min,max], else distance to nearest edge.
                if c.value_min is not None and c.value_max is not None:
                    if c.value_min <= detected_steps <= c.value_max:
                        err = 0
                    elif detected_steps < c.value_min:
                        err = detected_steps - c.value_min
                    else:
                        err = detected_steps - c.value_max
                else:
                    err = detected_steps - int(c.value)
                step_errors.append(err)
                tgt["err"] = err
                if abs(err) > 1:
                    failures["steps"].append({
                        "id": mid, "caption": cap,
                        "target": c.value, "range": [c.value_min, c.value_max],
                        "detected": detected_steps,
                    })
            elif c.type == "degrees":
                deg_buckets["total"] += 1
                # Empirical sign convention: detected_deg > 0 = CW = right turn.
                if c.direction in (Direction.LEFT, Direction.RIGHT):
                    measured = -detected_deg if c.direction == Direction.LEFT else detected_deg
                else:
                    measured = detected_deg  # signed, vs signed target

                # Categorize before scoring.
                gt_motion_too_small = abs(detected_deg) < 30
                target_sign = 1 if c.value > 0 else -1
                measured_sign = 1 if measured > 0 else -1
                sign_disagree = (
                    not gt_motion_too_small
                    and abs(c.value) > 30
                    and measured_sign != target_sign
                )

                in_range = False
                if c.value_min is not None and c.value_max is not None:
                    lo, hi = c.value_min, c.value_max
                    in_range = lo <= measured <= hi
                    if in_range:
                        err = 0.0
                    elif measured < lo:
                        err = measured - lo
                    else:
                        err = measured - hi
                else:
                    err = measured - c.value
                    in_range = abs(err) <= 30

                if in_range:
                    deg_buckets["ok_in_range"] += 1
                elif gt_motion_too_small:
                    deg_buckets["under_motion_gt_data_noise"] += 1
                elif sign_disagree:
                    deg_buckets["sign_flip_gt_data_noise"] += 1
                else:
                    deg_buckets["magnitude_off"] += 1
                    # only count "magnitude_off" as a real detector error
                    rot_errors.append(err)

                tgt["err"] = round(err, 1)
                tgt["measured"] = round(measured, 1)
                if not in_range and not gt_motion_too_small and not sign_disagree:
                    failures["degrees"].append({
                        "id": mid, "caption": cap,
                        "target": c.value, "range": [c.value_min, c.value_max],
                        "tgt_dir": c.direction.value,
                        "measured": round(measured, 1),
                        "detected_deg": round(detected_deg, 1),
                    })
            sample_entry["targets"].append(tgt)

        # End-to-end reward score: feed phases + totals into the actual
        # reward function the trainer uses.
        try:
            motion_raw_t = torch.from_numpy(motion_raw.astype(np.float32))
            phases = analyze_motion_phases(motion_raw_t, foot_contact)
            total_rot = float(_measure_rotation_signed(
                torch.from_numpy(motion_raw[:, 0])))
            total_steps_phase = sum(p.step_count for p in phases) or detected_steps
            reward_score = score_constraints_against_phases(
                cs, phases, total_steps_phase, total_rot, total_repetitions=0,
            )
            reward_scores_on_gt.append(reward_score)
            sample_entry["reward_score_on_gt"] = round(reward_score, 3)
            if reward_score < 0.3:
                low_reward_failures.append({
                    "id": mid, "caption": cap, "score": round(reward_score, 3),
                    "n_constraints": len(cs),
                })
        except Exception as e:
            sample_entry["reward_error"] = str(e)

        per_sample_log.append(sample_entry)

    def summarize(errs, tol):
        if not errs:
            return {"n": 0}
        a = np.array(errs, dtype=float)
        within = float((np.abs(a) <= tol).mean())
        return {
            "n": int(len(a)),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "abs_mean": float(np.abs(a).mean()),
            "p90_abs": float(np.percentile(np.abs(a), 90)),
            f"acc_within_{tol}": within,
        }

    return {
        "candidates": len(candidates),
        "sampled": len(sampled),
        "step_summary": summarize(step_errors, 1),
        "rotation_summary_excluding_gt_noise": summarize(rot_errors, 30),
        "degrees_buckets": deg_buckets,
        "reward_on_gt_summary": {
            "n": len(reward_scores_on_gt),
            "mean": float(np.mean(reward_scores_on_gt)) if reward_scores_on_gt else 0.0,
            "median": float(np.median(reward_scores_on_gt)) if reward_scores_on_gt else 0.0,
            "p25": float(np.percentile(reward_scores_on_gt, 25)) if reward_scores_on_gt else 0.0,
            "p75": float(np.percentile(reward_scores_on_gt, 75)) if reward_scores_on_gt else 0.0,
            "frac_ge_0.5": float(np.mean([s >= 0.5 for s in reward_scores_on_gt])) if reward_scores_on_gt else 0.0,
            "frac_ge_0.7": float(np.mean([s >= 0.7 for s in reward_scores_on_gt])) if reward_scores_on_gt else 0.0,
        },
        "low_reward_failures_first_20": low_reward_failures[:20],
        "failures_first_20": {
            "steps": failures["steps"][:20],
            "degrees": failures["degrees"][:20],
        },
        "per_sample_first_15": per_sample_log[:15],
    }


# ---------------------------------------------------------------------------
# Check C: neutrality on captions without numeric constraints
# ---------------------------------------------------------------------------

def check_c_neutrality(n: int = 200, seed: int = 0) -> dict:
    """For captions with 0 numeric constraints, the reward path should not
    contribute numerical/direction scores. This check confirms parser does
    not silently fire on them, and that direction parser also degrades
    gracefully (no spurious 'forward' on captions like 'a person sits').
    """
    rng = random.Random(seed)
    no_num = []
    for mid, cap, t0, t1 in iter_train_captions():
        if not parse_numerical_constraints(cap):
            no_num.append((mid, cap, t0, t1))
    sampled = rng.sample(no_num, min(n, len(no_num)))

    spurious_direction = []
    for mid, cap, t0, t1 in sampled:
        dirs = parse_direction_sequence(cap)
        non_any = [d for d in dirs if d != Direction.ANY]
        if non_any:
            # capture only those where direction is plausibly spurious
            # (i.e. caption is not motion-y at all)
            non_motion = not re.search(
                r"\b(walk|step|run|march|jog|hop|jump|leap|skip|turn|rotate|"
                r"spin|pivot|circle|move|head|face|push|pull|reach)\b",
                cap, re.I,
            )
            if non_motion:
                spurious_direction.append({
                    "id": mid, "caption": cap,
                    "parsed_dirs": [d.value for d in non_any],
                })

    return {
        "sampled_no_numeric": len(sampled),
        "spurious_direction_count": len(spurious_direction),
        "spurious_direction_examples": spurious_direction[:30],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--max-captions", type=int, default=None,
                    help="Cap Check A scan size (None = full train set)")
    ap.add_argument("--max-detector-samples", type=int, default=300)
    ap.add_argument("--neutral-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "audit" / "audit_report.json")
    args = ap.parse_args()

    report = {}
    if args.check in ("A", "all"):
        print("[A] scanning parser on train captions ...")
        report["A_parser"] = check_a_parser(args.max_captions, args.seed)
        a = report["A_parser"]
        print(f"   total={a['total_captions']}  hits={a['hits']}  "
              f"rate={a['hit_rate']*100:.1f}%  suspect={a['suspect_total']}")
    if args.check in ("B", "all"):
        print("[B] running detector on GT motion ...")
        report["B_detector"] = check_b_detector(args.max_detector_samples, args.seed)
        b = report["B_detector"]
        ss = b["step_summary"]
        rs = b["rotation_summary_excluding_gt_noise"]
        db = b["degrees_buckets"]
        rg = b["reward_on_gt_summary"]
        if ss.get("n"):
            print(f"   steps: n={ss['n']} abs_mean={ss['abs_mean']:.2f} "
                  f"acc_within_1={ss['acc_within_1']*100:.1f}%")
        if db["total"]:
            ok = db["ok_in_range"]; flip = db["sign_flip_gt_data_noise"]
            under = db["under_motion_gt_data_noise"]; mag = db["magnitude_off"]
            print(f"   degrees: total={db['total']}  ok_in_range={ok} "
                  f"({ok/db['total']*100:.1f}%)")
            print(f"            gt_data_noise: sign_flip={flip}  under_motion={under}  "
                  f"=> {(flip+under)/db['total']*100:.1f}% of failures")
            print(f"            true magnitude errors={mag} "
                  f"({mag/db['total']*100:.1f}%)")
            if rs.get("n"):
                print(f"            among magnitude-off only: abs_mean={rs['abs_mean']:.1f}")
        if rg["n"]:
            print(f"   reward-on-GT (end-to-end): n={rg['n']} "
                  f"mean={rg['mean']:.3f} median={rg['median']:.3f}")
            print(f"            frac>=0.5: {rg['frac_ge_0.5']*100:.1f}%  "
                  f"frac>=0.7: {rg['frac_ge_0.7']*100:.1f}%")
    if args.check in ("C", "all"):
        print("[C] neutrality on non-numeric captions ...")
        report["C_neutral"] = check_c_neutrality(args.neutral_sample, args.seed)
        c = report["C_neutral"]
        print(f"   sampled={c['sampled_no_numeric']} "
              f"spurious_direction={c['spurious_direction_count']}")

    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
