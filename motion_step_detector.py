"""Step detector aligned with the New Reward specification.

The canonical algorithm (see New Reward PDF, page 6) is:

    l_move_state = (1 - l_foot_contact_in_263_feat) > 0.5
    r_move_state = (1 - r_foot_contact_in_263_feat) > 0.5
    -> count "rest -> move" rising edges per foot

The legacy hybrid contact + height + speed detector is kept as an opt-in
fallback for callers that explicitly request it via `detector="hybrid"`,
but is no longer the default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


LEFT_FOOT_JOINT = 10
RIGHT_FOOT_JOINT = 11


@dataclass
class StepEvent:
    """One detected step in absolute frame coordinates."""

    foot: str
    start: int
    end: int
    key_frame: int
    source: str
    confidence: float
    # Hybrid-detector diagnostics; default to neutral values when the
    # move-state path does not populate them.
    valid_ratio: float = 1.0
    min_height: float = 0.0
    mean_speed: float = 0.0


@dataclass
class StepDetectionResult:
    events: List[StepEvent]
    consistency_penalty: float
    contact_events: int
    landing_events: int

    @property
    def count(self) -> int:
        return len(self.events)


def _foot_names(foot: str) -> Iterable[str]:
    foot = str(foot or "any").lower()
    if foot in {"any", "both"}:
        return ("left", "right")
    if foot in {"left", "l_foot", "left_foot"}:
        return ("left",)
    if foot in {"right", "r_foot", "right_foot"}:
        return ("right",)
    return ()


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


def detect_move_state_steps(
    foot_contact: torch.Tensor,
    start: int = 0,
    end: int = -1,
    foot: str = "any",
    *,
    contact_threshold: float = 0.5,
    min_step_separation: int = 3,
    min_move_frames: int = 1,
) -> StepDetectionResult:
    """Count steps using the New Reward PDF formula.

    For each foot:
        move_state = (1 - foot_contact_channels) > contact_threshold
    A step is one rising edge (rest -> move). Adjacent edges within
    `min_step_separation` frames are merged so a single transition is not
    counted twice when the contact label flickers.
    """
    if end == -1:
        end = int(foot_contact.shape[0])
    start = max(0, int(start))
    end = min(int(end), int(foot_contact.shape[0]))
    if end - start < 2:
        return StepDetectionResult([], 0.0, 0, 0)

    fc = foot_contact[start:end].detach().float()

    foot_defs: Dict[str, Tuple[int, int]] = {
        "left": (0, 1),
        "right": (2, 3),
    }

    events: List[StepEvent] = []
    for foot_name in _foot_names(foot):
        if foot_name not in foot_defs:
            continue
        i, j = foot_defs[foot_name]
        per_foot_contact = (fc[:, i] + fc[:, j]) * 0.5
        move_state = (1.0 - per_foot_contact) > float(contact_threshold)

        runs = [
            (rs, re) for rs, re in _bool_runs(move_state)
            if re - rs >= int(min_move_frames)
        ]
        # Only count runs that begin with a verifiable rest->move transition.
        # A run that starts at frame 0 lacks an observable preceding rest state
        # so it is not a confirmed rising edge -- the PDF example explicitly
        # assumes the motion starts in a rest (contact) state.
        last_kept_frame: Optional[int] = None
        for run_start, run_end in runs:
            if run_start == 0:
                continue
            abs_frame = start + run_start
            if last_kept_frame is not None and abs_frame - last_kept_frame < int(min_step_separation):
                continue
            last_kept_frame = abs_frame
            events.append(StepEvent(
                foot=foot_name,
                start=abs_frame,
                # A step is a near-instantaneous rest->move transition; we
                # mark a 1-frame window so downstream temporal relations
                # like "before" / "after" treat the step as a point event
                # rather than a long run spanning the rest of the motion.
                end=abs_frame + 1,
                key_frame=abs_frame,
                source="move_state",
                confidence=1.0,
            ))

    events.sort(key=lambda ev: (ev.key_frame, ev.foot))
    return StepDetectionResult(
        events=events,
        consistency_penalty=0.0,
        contact_events=len(events),
        landing_events=0,
    )


def detect_steps(
    joints: torch.Tensor,
    foot_contact: torch.Tensor,
    start: int = 0,
    end: int = -1,
    foot: str = "any",
    *,
    detector: str = "move_state",
    **kwargs,
) -> StepDetectionResult:
    """Top-level step detector.

    Defaults to the PDF-compliant move_state algorithm. Set
    `detector="hybrid"` to opt into the legacy contact+height+speed+landing
    detector kept for backward compatibility.
    """
    detector = str(detector or "move_state").lower()
    if detector == "move_state":
        accepted = {"contact_threshold", "min_step_separation", "min_move_frames"}
        ms_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return detect_move_state_steps(
            foot_contact, start=start, end=end, foot=foot, **ms_kwargs,
        )
    if detector == "hybrid":
        return _detect_steps_hybrid(
            joints, foot_contact, start=start, end=end, foot=foot, **kwargs,
        )
    raise ValueError(f"Unknown step detector: {detector!r}")


def detect_step_events(
    joints: Optional[torch.Tensor],
    foot_contact: torch.Tensor,
    start: int = 0,
    end: int = -1,
    foot: str = "any",
    *,
    detector: str = "move_state",
    **kwargs,
) -> List[StepEvent]:
    if detector == "move_state" or joints is None:
        return detect_move_state_steps(
            foot_contact,
            start=start,
            end=end,
            foot=foot,
            **{k: v for k, v in kwargs.items()
               if k in {"contact_threshold", "min_step_separation", "min_move_frames"}},
        ).events
    return _detect_steps_hybrid(
        joints, foot_contact, start=start, end=end, foot=foot, **kwargs,
    ).events


def count_steps(
    joints: Optional[torch.Tensor],
    foot_contact: torch.Tensor,
    start: int = 0,
    end: int = -1,
    foot: str = "any",
    *,
    detector: str = "move_state",
    **kwargs,
) -> int:
    return len(detect_step_events(
        joints, foot_contact, start=start, end=end, foot=foot,
        detector=detector, **kwargs,
    ))


# ---------------------------------------------------------------------------
# Legacy hybrid detector (opt-in fallback)
# ---------------------------------------------------------------------------


def _fill_short_gaps(mask: torch.Tensor, max_gap: int) -> torch.Tensor:
    if max_gap <= 0 or mask.numel() == 0:
        return mask.bool()
    filled = mask.bool().clone()
    inactive_runs = _bool_runs(~filled)
    for start, end in inactive_runs:
        if start > 0 and end < filled.numel() and end - start <= max_gap:
            filled[start:end] = True
    return filled


def _event_near(events: List[StepEvent], foot: str, frame: int, min_sep: int) -> bool:
    return any(ev.foot == foot and abs(ev.key_frame - frame) < min_sep for ev in events)


def _dedupe_events(events: List[StepEvent], min_separation: int) -> List[StepEvent]:
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


def _detect_steps_hybrid(
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
    """Legacy hybrid step detector.

    Treats foot contact as a candidate event and validates it against foot
    height and XZ speed; recovers steps from clean landing events when
    contact labels are missing. Opt-in via `detector="hybrid"`.
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
