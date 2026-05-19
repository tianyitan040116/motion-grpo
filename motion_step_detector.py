"""Shared hybrid step detector for reward and executor code.

The generated motion contains foot-contact channels, but those channels can be
noisy during RL.  This detector treats contact as a candidate event and checks
it against foot height and foot speed.  If contact labels are missing, it can
still recover a step from a clear landing event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


LEFT_FOOT_JOINT = 10
RIGHT_FOOT_JOINT = 11


@dataclass
class StepEvent:
    """One validated footfall in absolute frame coordinates."""

    foot: str
    start: int
    end: int
    key_frame: int
    source: str
    confidence: float
    valid_ratio: float
    min_height: float
    mean_speed: float


@dataclass
class StepDetectionResult:
    events: List[StepEvent]
    consistency_penalty: float
    contact_events: int
    landing_events: int

    @property
    def count(self) -> int:
        return len(self.events)


def _bool_runs(mask: torch.Tensor) -> List[Tuple[int, int]]:
    values = mask.detach().bool().cpu().tolist()
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, active in enumerate(values):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _fill_short_gaps(mask: torch.Tensor, max_gap: int) -> torch.Tensor:
    if max_gap <= 0 or mask.numel() == 0:
        return mask.bool()

    filled = mask.bool().clone()
    inactive_runs = _bool_runs(~filled)
    for start, end in inactive_runs:
        if start > 0 and end < filled.numel() and end - start <= max_gap:
            filled[start:end] = True
    return filled


def _foot_names(foot: str) -> Iterable[str]:
    foot = str(foot or "any").lower()
    if foot in {"any", "both"}:
        return ("left", "right")
    if foot in {"left", "l_foot", "left_foot"}:
        return ("left",)
    if foot in {"right", "r_foot", "right_foot"}:
        return ("right",)
    return ()


def _event_near(events: List[StepEvent], foot: str, frame: int, min_sep: int) -> bool:
    return any(ev.foot == foot and abs(ev.key_frame - frame) < min_sep for ev in events)


def _dedupe_events(events: List[StepEvent], min_separation: int) -> List[StepEvent]:
    """Merge duplicate contact/landing detections for the same foot."""
    kept: List[StepEvent] = []
    for event in sorted(events, key=lambda ev: (ev.foot, ev.key_frame, -ev.confidence)):
        replace_idx: Optional[int] = None
        for idx, old in enumerate(kept):
            if old.foot == event.foot and abs(old.key_frame - event.key_frame) < min_separation:
                if event.confidence > old.confidence:
                    replace_idx = idx
                else:
                    replace_idx = -1
                break
        if replace_idx is None:
            kept.append(event)
        elif replace_idx >= 0:
            kept[replace_idx] = event
    return sorted(kept, key=lambda ev: ev.key_frame)


def detect_steps(
    joints: torch.Tensor,
    foot_contact: torch.Tensor,
    start: int = 0,
    end: int = -1,
    foot: str = "any",
    *,
    contact_threshold: float = 0.5,
    max_contact_height: float = 0.055,
    max_contact_speed: float = 0.018,
    min_contact_frames: int = 1,
    min_valid_ratio: float = 0.5,
    max_contact_gap: int = 1,
    min_step_separation: int = 5,
    landing_height: float = 0.06,
    swing_height: float = 0.09,
    landing_min_travel: float = 0.035,
    landing_lookback: int = 8,
    landing_confidence: float = 0.8,
) -> StepDetectionResult:
    """Detect physical steps from contact labels plus foot kinematics.

    Contact-based events are accepted only when most frames in the contact run
    are both near the ground and slow in XZ.  Landing events fill gaps when the
    contact labels are absent but the foot visibly swings and settles.
    """
    if end == -1:
        end = int(min(joints.shape[0], foot_contact.shape[0]))
    start = max(0, int(start))
    end = min(int(end), int(joints.shape[0]), int(foot_contact.shape[0]))
    if end - start < 2:
        return StepDetectionResult([], 0.0, 0, 0)

    j = joints[start:end].detach().float()
    fc = foot_contact[start:end].detach().float()
    n_frames = int(j.shape[0])

    ground_source = torch.cat([
        j[:, LEFT_FOOT_JOINT, 1],
        j[:, RIGHT_FOOT_JOINT, 1],
    ])
    ground_y = torch.quantile(ground_source, 0.05).item()

    foot_defs: Dict[str, Tuple[int, Tuple[int, int]]] = {
        "left": (LEFT_FOOT_JOINT, (0, 1)),
        "right": (RIGHT_FOOT_JOINT, (2, 3)),
    }

    events: List[StepEvent] = []
    total_contact = 0.0
    inconsistent_contact = 0.0
    contact_events = 0
    landing_events = 0

    for foot_name in _foot_names(foot):
        if foot_name not in foot_defs:
            continue

        joint_idx, contact_cols = foot_defs[foot_name]
        pos = j[:, joint_idx]
        height = pos[:, 1] - float(ground_y)
        xz = pos[:, [0, 2]]
        xz_speed = torch.cat([
            torch.zeros(1, device=xz.device, dtype=xz.dtype),
            torch.norm(xz[1:] - xz[:-1], dim=-1),
        ])

        contact = (fc[:, contact_cols[0]] + fc[:, contact_cols[1]]) > contact_threshold
        contact = _fill_short_gaps(contact, max_contact_gap)
        near_ground = height <= max_contact_height
        landing_ground = height <= landing_height
        slow = xz_speed <= max_contact_speed
        valid_contact = near_ground & slow

        total_contact += float(contact.float().sum().item())
        inconsistent_contact += float((contact & ~valid_contact).float().sum().item())

        for run_start, run_end in _bool_runs(contact):
            if run_start == 0 or run_end - run_start < min_contact_frames:
                continue
            run_valid = valid_contact[run_start:run_end]
            valid_ratio = float(run_valid.float().mean().item()) if run_valid.numel() else 0.0
            if valid_ratio < min_valid_ratio:
                continue

            run_height = height[run_start:run_end]
            run_speed = xz_speed[run_start:run_end]
            confidence = min(1.0, 0.5 + 0.5 * valid_ratio)
            events.append(StepEvent(
                foot=foot_name,
                start=start + run_start,
                end=start + run_end,
                key_frame=start + run_start,
                source="contact",
                confidence=confidence,
                valid_ratio=valid_ratio,
                min_height=float(run_height.min().item()),
                mean_speed=float(run_speed.mean().item()),
            ))
            contact_events += 1

        # Contact labels can be missing after decoding.  Recover a step when a
        # foot was clearly airborne recently and then settles near the ground.
        for frame in range(1, n_frames):
            if not landing_ground[frame].item() or not slow[frame].item():
                continue
            if landing_ground[frame - 1].item():
                continue
            look_start = max(0, frame - landing_lookback)
            had_swing = bool((height[look_start:frame] >= swing_height).any().item())
            if not had_swing:
                continue
            swing_path = torch.norm(xz[look_start:frame + 1] - xz[frame], dim=-1).max().item()
            if swing_path < landing_min_travel:
                continue
            absolute_frame = start + frame
            if _event_near(events, foot_name, absolute_frame, min_step_separation):
                continue
            events.append(StepEvent(
                foot=foot_name,
                start=max(start, absolute_frame - 1),
                end=min(end, absolute_frame + 1),
                key_frame=absolute_frame,
                source="landing",
                confidence=landing_confidence,
                valid_ratio=1.0,
                min_height=float(height[frame].item()),
                mean_speed=float(xz_speed[frame].item()),
            ))
            landing_events += 1

    consistency_penalty = (
        inconsistent_contact / total_contact if total_contact > 0.0 else 0.0
    )
    return StepDetectionResult(
        events=_dedupe_events(events, min_step_separation),
        consistency_penalty=float(consistency_penalty),
        contact_events=contact_events,
        landing_events=landing_events,
    )


def detect_step_events(
    joints: torch.Tensor,
    foot_contact: torch.Tensor,
    start: int = 0,
    end: int = -1,
    foot: str = "any",
    **kwargs,
) -> List[StepEvent]:
    return detect_steps(joints, foot_contact, start, end, foot, **kwargs).events


def count_steps(
    joints: torch.Tensor,
    foot_contact: torch.Tensor,
    start: int = 0,
    end: int = -1,
    foot: str = "any",
    **kwargs,
) -> int:
    return detect_steps(joints, foot_contact, start, end, foot, **kwargs).count
