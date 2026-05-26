"""Prompt-mix dataset for GRPO training.

Wraps the existing Text2MotionDataset and lets the trainer sample captions
from three semantic buckets in a configurable ratio:

  numeric         -- parse_numerical_constraints hits at least one constraint
  direction_only  -- numeric is empty but parse_direction_sequence finds a
                     non-ANY direction
  pure            -- neither (descriptive captions: "person sits down")

Why mix instead of just filtering: P0 audit showed only ~11% of HumanML3D
train captions carry parseable numeric constraints, so a pure-numeric diet
under-samples the broader caption distribution and the policy can over-fit
to step/circle phrasings. Adding direction-only (~30-50% extra) gives the
direction-sequence reward path real signal, and a small pure slice keeps
the SFT-KL anchor's reference distribution honest.

Default mix is 30% numeric / 40% direction / 30% pure (tunable via CLI).
The wrapper exposes the standard 8-tuple Text2MotionDataset returns so the
existing collate_fn and train_step code work unchanged.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate

# Allow `from dataset.prompt_mix import ...` and standalone script use.
import sys as _sys
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from grpo_reward import (
    Direction,
    parse_numerical_constraints,
    parse_direction_sequence,
)
from dataset.dataset_TM_eval import Text2MotionDataset, collate_fn


@dataclass(frozen=True)
class MixConfig:
    """Sampling ratios for the three buckets. Must sum to ~1.0."""
    numeric: float = 0.30
    direction: float = 0.40
    pure: float = 0.30

    def __post_init__(self):
        total = self.numeric + self.direction + self.pure
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"MixConfig fractions must sum to 1.0, got {total:.3f}")


def classify_caption(caption: str) -> str:
    """Return 'numeric', 'direction_only', or 'pure'."""
    if parse_numerical_constraints(caption):
        return "numeric"
    dirs = parse_direction_sequence(caption)
    if any(d != Direction.ANY for d in dirs):
        return "direction_only"
    return "pure"


def bucket_caption_pairs(dataset: Text2MotionDataset) -> Dict[str, List[Tuple[int, int]]]:
    """Bucket every (clip, caption-within-clip) pair, not just (clip).

    A HumanML3D clip has ~3 paraphrases; classifying the clip by its first
    caption only and then re-selecting at __getitem__ time misclassifies
    most draws. Instead we enumerate every caption as its own indexable
    item: bucket key -> list of (clip_idx_post_pointer, caption_idx).

    `clip_idx_post_pointer` is what `dataset[i]` expects (idx = pointer + i,
    we return `i`).
    """
    buckets: Dict[str, List[Tuple[int, int]]] = {
        "numeric": [], "direction_only": [], "pure": [],
    }
    pointer = dataset.pointer
    name_list = dataset.name_list
    for abs_idx, name in enumerate(name_list):
        if abs_idx < pointer:
            continue
        rel_idx = abs_idx - pointer
        texts = dataset.data_dict[name]["text"]
        for cap_idx, text_dict in enumerate(texts):
            caption = text_dict.get("caption", "")
            if not caption:
                continue
            buckets[classify_caption(caption)].append((rel_idx, cap_idx))
    return buckets


def bucket_indices(dataset: Text2MotionDataset) -> Dict[str, List[int]]:
    """Backward-compat alias: clip-level bucketing using the first caption.

    Prefer `bucket_caption_pairs` -- HumanML3D clips have ~3 paraphrases per
    clip, and the bucket assigned by this function only describes the first
    one. Kept around for callers that just want a coarse breakdown.
    """
    out: Dict[str, List[int]] = {"numeric": [], "direction_only": [], "pure": []}
    name_list = dataset.name_list
    for idx, name in enumerate(name_list):
        texts = dataset.data_dict[name]["text"]
        first_caption = texts[0]["caption"] if texts else ""
        out[classify_caption(first_caption)].append(idx)
    return out


class PromptMixDataset(data.Dataset):
    """Wraps a Text2MotionDataset and overrides __getitem__ to sample by
    bucket (numeric / direction_only / pure) at the caption granularity.

    The base dataset stores ~3 paraphrase captions per clip. This wrapper
    enumerates every (clip, caption) pair separately and picks one each
    draw, so the returned caption is guaranteed to come from the bucket we
    intended -- unlike a clip-level bucket where __getitem__ would later
    re-pick a different paraphrase at random.

    `__len__` is set to len(base) so the trainer's per-epoch step count is
    unchanged; each epoch is a fresh stochastic mix.
    """

    def __init__(
        self,
        base: Text2MotionDataset,
        config: MixConfig = MixConfig(),
        seed: int = 0,
        verbose: bool = True,
    ):
        self.base = base
        self.config = config
        self.rng = random.Random(seed)
        self.bucket_pairs: Dict[str, List[Tuple[int, int]]] = bucket_caption_pairs(base)
        sizes = {k: len(v) for k, v in self.bucket_pairs.items()}
        present = {k: v for k, v in sizes.items() if v > 0}
        if not present:
            raise RuntimeError("No usable buckets after pointer filter.")
        self._bucket_keys = list(present.keys())
        weights_all = {
            "numeric": config.numeric,
            "direction_only": config.direction,
            "pure": config.pure,
        }
        active_weights = [weights_all[k] for k in self._bucket_keys]
        total = sum(active_weights)
        self._bucket_probs = [w / total for w in active_weights]
        if verbose:
            print(f"[PromptMix] caption-level bucket sizes: {sizes}")
            print(f"[PromptMix] sampling mix: " + ", ".join(
                f"{k}={p:.0%}" for k, p in zip(self._bucket_keys, self._bucket_probs)
            ))

    def __len__(self) -> int:
        return len(self.base)

    def _sample_bucket_key(self) -> str:
        return self.rng.choices(self._bucket_keys, weights=self._bucket_probs, k=1)[0]

    def __getitem__(self, item: int):
        # DataLoader sweeps `item`; we draw freely from the configured mix.
        bucket = self._sample_bucket_key()
        rel_idx, cap_idx = self.rng.choice(self.bucket_pairs[bucket])
        return _get_with_caption(self.base, rel_idx, cap_idx)


def _get_with_caption(base: Text2MotionDataset, rel_idx: int, cap_idx: int):
    """Same as base[rel_idx] but uses caption `cap_idx` instead of a random
    paraphrase. Faithful copy of Text2MotionDataset.__getitem__ except for
    the caption pick.
    """
    abs_idx = base.pointer + rel_idx
    name = base.name_list[abs_idx]
    rec = base.data_dict[name]
    motion = rec["motion"]
    m_length = rec["length"]
    text_list = rec["text"]
    # Bounded pick instead of random.choice.
    text_data = text_list[cap_idx if cap_idx < len(text_list) else 0]
    caption = text_data["caption"]
    tokens = text_data["tokens"]

    max_text_len = base.max_text_len
    if len(tokens) < max_text_len:
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
    else:
        tokens = tokens[:max_text_len]
        tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        sent_len = len(tokens)
    pos_one_hots = []
    word_embeddings = []
    for token in tokens:
        word_emb, pos_oh = base.w_vectorizer[token]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])
    pos_one_hots = np.concatenate(pos_one_hots, axis=0)
    word_embeddings = np.concatenate(word_embeddings, axis=0)

    unit_length = base.unit_length
    if unit_length < 10:
        coin2 = np.random.choice(["single", "single", "double"])
    else:
        coin2 = "single"

    if coin2 == "double":
        m_length = (m_length // unit_length - 1) * unit_length
    else:
        m_length = (m_length // unit_length) * unit_length
    idx = random.randint(0, len(motion) - m_length)
    motion = motion[idx: idx + m_length]
    motion = (motion - base.mean) / base.std

    max_motion_length = base.max_motion_length
    if m_length < max_motion_length:
        motion = np.concatenate([
            motion,
            np.zeros((max_motion_length - m_length, motion.shape[1])),
        ], axis=0)

    return (
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length,
        "_".join(tokens), name,
    )


def build_mixed_loader(
    dataset_name: str,
    split: str,
    batch_size: int,
    w_vectorizer,
    config: MixConfig = MixConfig(),
    num_workers: int = 8,
    unit_length: int = 4,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """Drop-in replacement for dataset_TM_eval.DATALoader that mixes by bucket."""
    base = Text2MotionDataset(
        dataset_name, split, w_vectorizer, unit_length=unit_length,
    )
    mixed = PromptMixDataset(base, config=config, seed=seed)
    return torch.utils.data.DataLoader(
        mixed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Offline bucket export (no torch deps beyond grpo_reward)
# ---------------------------------------------------------------------------

def dump_buckets_jsonl(
    split_file: Path,
    text_dir: Path,
    out_path: Path,
    motion_dir: Optional[Path] = None,
    require_motion_local: bool = False,
) -> Dict[str, int]:
    """Walk a split file, classify each motion's first caption, and write a
    jsonl of `{id, caption, bucket}` per line. Useful for inspecting bucket
    composition without instantiating the full Text2MotionDataset (which
    loads every motion into memory).

    If `require_motion_local` is True, motions whose .npy is missing on
    disk are skipped -- handy for the Mac side where only ~58% of train
    motions are cached locally.
    """
    counts = {"numeric": 0, "direction_only": 0, "pure": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for mid in split_file.read_text().split():
            if require_motion_local and motion_dir is not None:
                if not (motion_dir / f"{mid}.npy").exists():
                    continue
            tpath = text_dir / f"{mid}.txt"
            if not tpath.exists():
                continue
            first = None
            for line in tpath.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                first = line.split("#")[0].strip()
                break
            if not first:
                continue
            bucket = classify_caption(first)
            counts[bucket] += 1
            fout.write(json.dumps({"id": mid, "caption": first, "bucket": bucket}) + "\n")
    return counts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--require-local-motion", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("dataset/buckets_train.jsonl"))
    args = ap.parse_args()

    counts = dump_buckets_jsonl(
        split_file=Path("dataset") / f"{args.split}.txt",
        text_dir=Path("dataset/texts"),
        motion_dir=Path("dataset/new_joint_vecs"),
        out_path=args.out,
        require_motion_local=args.require_local_motion,
    )
    total = sum(counts.values()) or 1
    print(f"wrote {args.out}")
    for k, c in counts.items():
        print(f"  {k:15} {c:6d}  ({c/total*100:.1f}%)")
