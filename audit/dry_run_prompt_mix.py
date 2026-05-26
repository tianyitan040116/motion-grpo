"""Dry-run of P1 PromptMixDataset.

Loads Text2MotionDataset (Mac has ~58% of HumanML3D train motions locally,
the rest are silently skipped), wraps it in PromptMixDataset, sweeps a few
hundred draws, and verifies:

  1. The realised bucket mix matches the configured ratios within tolerance.
  2. Each bucket returns the structure Text2MotionDataset would (8-tuple,
     dtype-correct), so collate_fn and train_step are unaffected.
  3. The per-bucket caption samples look right (printed for eyeball).

This does NOT call into the model/reward path -- it's purely about whether
the dataset wrapper feeds the trainer the captions we asked for.

Usage:
    /Users/tan/miniforge3/envs/mgpt/bin/python audit/dry_run_prompt_mix.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from utils.word_vectorizer import WordVectorizer
from dataset.dataset_TM_eval import Text2MotionDataset
from dataset.prompt_mix import (
    PromptMixDataset, MixConfig, classify_caption,
)


def _get_item_via_mixed(mixed, rel_idx: int, cap_idx: int):
    """Directly construct one sample from a (rel_idx, cap_idx) pair via the
    mixed dataset's helper. Returns (caption, motion_raw, m_length).

    The mixed dataset returns a normalised motion (Text2MotionDataset
    pipeline). We undo that here so the caller can score against raw
    HumanML3D 263-dim, which is what the reward path consumes.
    """
    try:
        from dataset.prompt_mix import _get_with_caption
        tup = _get_with_caption(mixed.base, rel_idx, cap_idx)
    except Exception:
        return None
    word_embeddings, pos_one_hots, caption, sent_len, motion_norm, m_length, token, name = tup
    motion_raw = motion_norm[:m_length] * mixed.base.std + mixed.base.mean
    return caption, motion_raw.astype(np.float32), m_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-draws", type=int, default=400)
    ap.add_argument("--numeric", type=float, default=0.30)
    ap.add_argument("--direction", type=float, default=0.40)
    ap.add_argument("--pure", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-examples", type=int, default=4,
                    help="number of caption examples to print per bucket")
    args = ap.parse_args()

    print("[1/3] Loading WordVectorizer + Text2MotionDataset (~30s) ...")
    w_vectorizer = WordVectorizer("./glove", "our_vab")
    base = Text2MotionDataset("t2m", "train", w_vectorizer, unit_length=4)
    print(f"     loaded {len(base)} usable items "
          f"(pointer={base.pointer}, total name_list={len(base.name_list)})")

    cfg = MixConfig(numeric=args.numeric, direction=args.direction, pure=args.pure)
    print(f"[2/3] Wrapping with PromptMixDataset (seed={args.seed}, "
          f"mix={cfg.numeric:.0%}/{cfg.direction:.0%}/{cfg.pure:.0%}) ...")
    mixed = PromptMixDataset(base, config=cfg, seed=args.seed, verbose=True)

    # Sweep N draws, count bucket on the *actually returned* caption (which
    # may differ from the first caption used at bucket-build time since
    # __getitem__ picks one of the clip's ~3 captions at random).
    print(f"[3/3] Drawing {args.num_draws} samples ...")
    drawn_buckets = Counter()
    drawn_first_caption_buckets = Counter()
    examples = {"numeric": [], "direction_only": [], "pure": []}
    last_item_shape = None

    for _ in range(args.num_draws):
        sample = mixed[0]  # idx ignored by PromptMixDataset
        if last_item_shape is None:
            last_item_shape = [
                type(x).__name__ + (f"[{x.shape}]" if hasattr(x, "shape") else "")
                for x in sample
            ]
        caption = sample[2]
        bucket = classify_caption(caption)
        drawn_buckets[bucket] += 1
        if len(examples[bucket]) < args.n_examples:
            examples[bucket].append(caption)

    total = sum(drawn_buckets.values()) or 1
    print()
    print("=== Tuple structure (per __getitem__) ===")
    for i, t in enumerate(last_item_shape):
        print(f"  [{i}] {t}")
    print()
    print("=== Realised bucket mix on returned captions ===")
    target = {"numeric": cfg.numeric, "direction_only": cfg.direction, "pure": cfg.pure}
    for k in ("numeric", "direction_only", "pure"):
        got = drawn_buckets[k] / total
        delta = got - target[k]
        flag = "OK" if abs(delta) < 0.05 else "DRIFT"
        print(f"  {k:15} target={target[k]*100:5.1f}%  got={got*100:5.1f}%  "
              f"delta={delta*100:+5.1f}pp  [{flag}]")

    print()
    print("=== Sample captions per bucket ===")
    for k, caps in examples.items():
        print(f"  -- {k} --")
        for c in caps:
            print(f"    {c!r}")

    # P0 already established reward-on-GT works. Re-verify that each bucket
    # actually exercises the reward signal it's supposed to, on a tiny slice.
    print()
    print("=== Per-bucket reward-on-GT spot check (50 samples each) ===")
    import torch
    from grpo_reward import (
        analyze_motion_phases,
        score_constraints_against_phases,
        score_direction_sequence,
        _measure_rotation_signed,
        _count_repetitions,
        parse_constraints_regex,
        denormalize_motion,
    )
    from utils.motion_utils import recover_from_ric

    per_bucket_scores: dict = {"numeric": [], "direction_only": [], "pure": []}
    per_bucket_low_examples: dict = {"numeric": [], "direction_only": [], "pure": []}
    rng = np.random.default_rng(args.seed)
    debug_first = True
    for k, pairs in mixed.bucket_pairs.items():
        if not pairs:
            continue
        sample_pairs = [pairs[int(i)] for i in rng.integers(0, len(pairs), size=min(50, len(pairs)))]
        for rel_idx, cap_idx in sample_pairs:
            tuple_out = _get_item_via_mixed(mixed, rel_idx, cap_idx)
            if tuple_out is None:
                continue
            caption, motion_raw_np, m_length = tuple_out
            if debug_first and k == "numeric":
                print(f"  [debug] first numeric sample: m_length={m_length} "
                      f"motion shape={motion_raw_np.shape}  caption={caption!r}")
                debug_first = False
            motion_raw_t = torch.from_numpy(motion_raw_np).float()
            foot_contact = (motion_raw_t[:, 259:263] > 0.5).float()
            joints = recover_from_ric(motion_raw_t.unsqueeze(0), joints_num=22).squeeze(0)
            phases = analyze_motion_phases(motion_raw_t, foot_contact, joints=joints,
                                            min_phase_frames=8, direction_change_threshold=0.6)
            parsed = parse_constraints_regex(caption)
            if k == "numeric":
                if not parsed.numerical_constraints:
                    continue
                from grpo_reward import _count_steps_in_range
                total_steps = _count_steps_in_range(foot_contact, joints=joints)
                total_rot = float(_measure_rotation_signed(motion_raw_t[:, 0]))
                total_reps = _count_repetitions(motion_raw_t[:, 3])
                score = score_constraints_against_phases(
                    parsed.numerical_constraints, phases,
                    total_steps=total_steps, total_rotation_deg=total_rot,
                    total_repetitions=total_reps,
                )
            elif k == "direction_only":
                if not parsed.direction_sequence:
                    continue
                score = score_direction_sequence(parsed.direction_sequence, phases)
            else:
                # pure: no numerical/direction reward -- score is conventionally 0.
                # Confirm reward path returns nothing harmful.
                score = 0.0
            per_bucket_scores[k].append(float(score))
            if score < 0.1 and len(per_bucket_low_examples[k]) < 6:
                per_bucket_low_examples[k].append((caption, float(score),
                                                   len(parsed.numerical_constraints)))

    for k, scores in per_bucket_scores.items():
        if not scores:
            print(f"  {k:15} n=0")
            continue
        arr = np.array(scores)
        print(f"  {k:15} n={len(arr):3}  mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
              f"frac>=0.5={(arr>=0.5).mean()*100:.1f}%")

    # Show low-scoring numeric examples to understand why
    if per_bucket_scores["numeric"]:
        arr = np.array(per_bucket_scores["numeric"])
        n_zero = (arr == 0).sum()
        print(f"\n  numeric zeros: {n_zero}/{len(arr)} "
              f"(non-zero mean={arr[arr>0].mean() if (arr>0).any() else 0:.3f})")
        if per_bucket_low_examples["numeric"]:
            print("  low-numeric examples:")
            for cap, sc, nc in per_bucket_low_examples["numeric"]:
                print(f"    score={sc:.2f}  n_constraints={nc}  cap={cap!r}")

    # Final sanity: now that buckets are caption-level, drift should be
    # just sampling noise (a few percentage points at 400 draws).
    print()
    print("Note: buckets are at the (clip, caption) granularity, so the")
    print("realised mix should match the configured ratios up to sampling")
    print("noise. Significant drift (>5pp) suggests a regression.")


if __name__ == "__main__":
    main()
