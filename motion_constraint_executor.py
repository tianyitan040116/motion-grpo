"""Composable motion constraint executor.

This module turns motion features into reusable signals, states, templates,
and constraint scores.  It is intentionally independent from GRPO so the same
detectors can be used for reward, debugging, or offline evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from motion_step_detector import detect_step_events


JOINT_INDEX = {
    "pelvis": 0,
    "root": 0,
    "l_hip": 1,
    "r_hip": 2,
    "l_ankle": 7,
    "r_ankle": 8,
    "l_foot": 10,
    "r_foot": 11,
    "head": 15,
    "l_shoulder": 16,
    "r_shoulder": 17,
    "l_elbow": 18,
    "r_elbow": 19,
    "l_wrist": 20,
    "r_wrist": 21,
    "l_hand": 20,
    "r_hand": 21,
    "left_hand": 20,
    "right_hand": 21,
}

_DIRECTION_COMPONENTS = {
    "forward": (0.0, 1.0),
    "backward": (0.0, -1.0),
    "left": (1.0, 0.0),
    "right": (-1.0, 0.0),
    "left_forward": (1.0, 1.0),
    "forward_left": (1.0, 1.0),
    "right_forward": (-1.0, 1.0),
    "forward_right": (-1.0, 1.0),
    "left_backward": (1.0, -1.0),
    "backward_left": (1.0, -1.0),
    "right_backward": (-1.0, -1.0),
    "backward_right": (-1.0, -1.0),
}

_DIRECTION_ALIASES = {
    "forwards": "forward",
    "ahead": "forward",
    "front": "forward",
    "backwards": "backward",
    "back": "backward",
    "front_left": "left_forward",
    "front-left": "left_forward",
    "left-front": "left_forward",
    "left front": "left_forward",
    "forward-left": "left_forward",
    "forward left": "left_forward",
    "left-forward": "left_forward",
    "left forward": "left_forward",
    "front_right": "right_forward",
    "front-right": "right_forward",
    "right-front": "right_forward",
    "right front": "right_forward",
    "forward-right": "right_forward",
    "forward right": "right_forward",
    "right-forward": "right_forward",
    "right forward": "right_forward",
    "back_left": "left_backward",
    "back-left": "left_backward",
    "left-back": "left_backward",
    "backward-left": "left_backward",
    "left-backward": "left_backward",
    "back_right": "right_backward",
    "back-right": "right_backward",
    "right-back": "right_backward",
    "backward-right": "right_backward",
    "right-backward": "right_backward",
}


def _normalize_direction_name(direction: str) -> str:
    raw = str(direction or "forward").strip().lower().replace("-", "_").replace(" ", "_")
    return _DIRECTION_ALIASES.get(raw, raw)


@dataclass
class MotionCache:
    motion_raw: torch.Tensor
    joints: torch.Tensor
    foot_contact: torch.Tensor
    fps: float = 20.0
    body_left: Optional[torch.Tensor] = None
    body_forward: Optional[torch.Tensor] = None
    body_up: Optional[torch.Tensor] = None


@dataclass
class ExecutionResult:
    constraint_id: str
    constraint_type: str
    measured_value: float
    target: Optional[float] = None
    op: str = ""
    reward: float = 0.0
    violation: float = 0.0
    score: float = 0.0
    source_repr: str = ""
    matched_segments: List[Dict[str, Any]] = field(default_factory=list)
    matched_events: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    # Set IDs from the top-level {"constraint_sets": [...]} wrapper, when the
    # caller passes that form. Useful for tracing which prompt a reward came
    # from in logs. Both default to None when a flat list is passed in.
    constraint_set_id: Optional[str] = None
    prompt_id: Optional[str] = None


def _to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().float().cpu().numpy()


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def _bool_segments(mask: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, active in enumerate(mask.tolist()):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask)))
    return segments


def compare_violation(
    measured: float,
    op: str,
    target: float,
    tolerance: float = 0.0,
) -> float:
    normalized_op = str(op).lower()
    if normalized_op == "lt":
        return max(measured - target, 0.0)
    if normalized_op == "le":
        return max(measured - target, 0.0)
    if normalized_op == "gt":
        return max(target - measured, 0.0)
    if normalized_op == "ge":
        return max(target - measured, 0.0)
    if normalized_op == "eq":
        return max(abs(measured - target) - float(tolerance), 0.0)
    raise ValueError(f"Unsupported comparison op: {op}")


def _reward_from_violation(violation: float, weight: float) -> Tuple[float, float]:
    reward = -float(weight) * float(violation)
    score = math.exp(-float(violation))
    return reward, score


def _event_count_reward(measured: float, target: float, tolerance: float, weight: float) -> Tuple[float, float, float]:
    # Schema defines `reward = -weight * |counts - threshold|`. We divide the
    # violation by max(|target|, 1) on purpose: GRPO computes group-relative
    # advantages across prompts with very different counts (e.g. "walk 2
    # steps" vs "walk 10 steps"), and the unnormalized form makes large-target
    # prompts dominate the advantage signal. Keeping the per-target scale
    # uniform here trades strict schema fidelity for cross-prompt stability.
    violation = compare_violation(measured, "eq", target, tolerance)
    normalized = violation / max(abs(float(target)), 1.0)
    score = math.exp(-normalized)
    reward = -float(weight) * normalized
    return violation, reward, score


class MotionConstraintExecutor:
    """Execute signal/count/temporal constraints against one motion sample."""

    def __init__(self, fps: float = 20.0):
        self.fps = fps

    def make_cache(
        self,
        motion_raw: torch.Tensor,
        foot_contact: torch.Tensor,
        joints: torch.Tensor,
    ) -> MotionCache:
        cache = MotionCache(
            motion_raw=motion_raw,
            joints=joints,
            foot_contact=foot_contact,
            fps=self.fps,
        )
        self.estimate_body_frame(cache)
        return cache

    def _resolve_scope(
        self,
        cache: MotionCache,
        constraint: Dict[str, Any],
    ) -> Tuple[MotionCache, Optional[Tuple[int, int]], List[str]]:
        """Apply `constraint["scope"]` and return a (possibly sliced) cache.

        Schema (per New-Reward doc):
          - default / `{"type": "whole_sequence"}` -> full motion (no-op)
          - `{"type": "frame_interval", "start": int, "end": int}` -> [start, end)

        Returns the cache to evaluate against, the resolved (start, end) range
        (or None for whole_sequence), and any limitations that should be
        attached to the result (e.g. unrecognized scope types).
        """
        scope = constraint.get("scope")
        if scope is None:
            return cache, None, []
        if not isinstance(scope, dict):
            return cache, None, [f"scope ignored: expected dict, got {type(scope).__name__}"]

        scope_type = str(scope.get("type", "whole_sequence")).lower()
        if scope_type == "whole_sequence":
            return cache, None, []
        if scope_type != "frame_interval":
            return cache, None, [f"scope ignored: unsupported type {scope_type!r}"]

        T = cache.joints.shape[0]
        try:
            start = int(scope.get("start", 0))
            end = int(scope.get("end", T))
        except (TypeError, ValueError):
            return cache, None, [f"scope ignored: non-integer start/end in {scope!r}"]
        start = max(0, min(start, T))
        end = max(start, min(end, T))
        if end - start < 2:
            return cache, None, [f"scope ignored: range [{start},{end}) is too short to evaluate"]

        # motion_raw and foot_contact share the time axis with joints (dim 0).
        sliced = MotionCache(
            motion_raw=cache.motion_raw[start:end],
            joints=cache.joints[start:end],
            foot_contact=cache.foot_contact[start:end],
            fps=cache.fps,
            body_left=cache.body_left[start:end] if cache.body_left is not None else None,
            body_forward=cache.body_forward[start:end] if cache.body_forward is not None else None,
            body_up=cache.body_up[start:end] if cache.body_up is not None else None,
        )
        return sliced, (start, end), []

    def estimate_body_frame(self, cache: MotionCache) -> MotionCache:
        joints = cache.joints
        T = joints.shape[0]
        device = joints.device
        dtype = joints.dtype

        up = torch.zeros(T, 3, device=device, dtype=dtype)
        up[:, 1] = 1.0

        left_raw = 0.5 * (
            (joints[:, JOINT_INDEX["l_shoulder"]] - joints[:, JOINT_INDEX["r_shoulder"]])
            + (joints[:, JOINT_INDEX["l_hip"]] - joints[:, JOINT_INDEX["r_hip"]])
        )
        left_ground = left_raw.clone()
        left_ground[:, 1] = 0.0

        default_left = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        left = torch.zeros_like(left_ground)
        prev = default_left
        for idx in range(T):
            cur = left_ground[idx]
            if cur.norm() < 1e-8:
                left[idx] = prev
            else:
                left[idx] = cur / cur.norm().clamp(min=1e-8)
                prev = left[idx]

        forward = torch.cross(left, up, dim=-1)
        forward = _normalize(forward)

        cache.body_left = left
        cache.body_forward = forward
        cache.body_up = up
        return cache

    def resolve_entity_positions(self, cache: MotionCache, name: str) -> torch.Tensor:
        key = str(name).lower()
        if key not in JOINT_INDEX:
            raise ValueError(f"Unknown entity: {name}")
        return cache.joints[:, JOINT_INDEX[key]]

    def _axis_vectors(self, cache: MotionCache, direction: str, frame: str = "body") -> torch.Tensor:
        direction = _normalize_direction_name(direction)
        frame = str(frame or "body").lower()
        T = cache.joints.shape[0]
        device = cache.joints.device
        dtype = cache.joints.dtype

        if frame == "body":
            assert cache.body_left is not None and cache.body_forward is not None and cache.body_up is not None
            if direction in {"up", "down"}:
                return cache.body_up if direction == "up" else -cache.body_up
            if direction not in _DIRECTION_COMPONENTS:
                raise ValueError(f"Unsupported body direction: {direction}")
            left_scale, forward_scale = _DIRECTION_COMPONENTS[direction]
            axis = left_scale * cache.body_left + forward_scale * cache.body_forward
            return _normalize(axis)

        world = {
            "forward": torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
            "backward": torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype),
            "left": torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
            "right": torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype),
            "up": torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
            "down": torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype),
        }
        if direction in world:
            return world[direction].view(1, 3).expand(T, 3)
        if direction not in _DIRECTION_COMPONENTS:
            raise ValueError(f"Unsupported world direction: {direction}")
        left_scale, forward_scale = _DIRECTION_COMPONENTS[direction]
        axis = torch.tensor([left_scale, 0.0, forward_scale], device=device, dtype=dtype)
        axis = axis / axis.norm().clamp(min=1e-8)
        return axis.view(1, 3).expand(T, 3)

    def signal(self, cache: MotionCache, name: str, args: Dict[str, Any]) -> np.ndarray:
        name = str(name).lower()
        args = args or {}

        if name == "dist":
            pos_a = self.resolve_entity_positions(cache, args["a"])
            pos_b = self.resolve_entity_positions(cache, args["b"])
            return _to_numpy(torch.norm(pos_a - pos_b, dim=-1))

        if name == "height":
            pos = self.resolve_entity_positions(cache, args["entity"])
            return _to_numpy(pos[:, 1])

        if name == "foot_height":
            # PDF spec (page 5): pos(foot).y, optionally offset by the minimum
            # observed foot height so the result is foot height above the
            # ground. Defaults to the right foot, matching the PDF example.
            foot = str(args.get("foot", "r_foot")).lower()
            joint_key = "l_foot" if foot in {"left", "l_foot"} else "r_foot"
            pos = self.resolve_entity_positions(cache, joint_key)
            height = pos[:, 1]
            if bool(args.get("relative_to_ground", True)):
                ground = torch.minimum(
                    cache.joints[:, JOINT_INDEX["l_foot"], 1].min(),
                    cache.joints[:, JOINT_INDEX["r_foot"], 1].min(),
                )
                height = height - ground
            return _to_numpy(height)

        if name == "relative_height":
            pos = self.resolve_entity_positions(cache, args["entity"])
            base = self.resolve_entity_positions(cache, args.get("base", "pelvis"))
            return _to_numpy(pos[:, 1] - base[:, 1])

        if name == "speed":
            pos = self.resolve_entity_positions(cache, args.get("entity", "pelvis"))
            delta = pos[1:] - pos[:-1]
            speed = torch.norm(delta, dim=-1) * float(args.get("fps", self.fps))
            speed = torch.cat([torch.zeros(1, device=pos.device, dtype=pos.dtype), speed])
            return _to_numpy(speed)

        if name == "directional_displacement":
            # PDF spec (page 4):
            #   root_vel = compute_world_vel(pelvis)
            #   body_vel_dir = root_vel . body_axes[dir]
            #   pos_dis = cumsum(positive(body_vel_dir) * fps)
            #   reward = pos_dis[last] > displace_thresh    (binary at the caller)
            # Default `multiply_fps=False` keeps the historical units (raw
            # cumulative displacement in body-scale units). Set
            # `multiply_fps=True` to follow the PDF formula literally so a
            # threshold can be specified in (body lengths / second).
            entity = args.get("entity", "pelvis")
            direction = args.get("direction", "forward")
            frame = args.get("frame", "body")
            multiply_fps = bool(args.get("multiply_fps", False))
            pos = self.resolve_entity_positions(cache, entity)
            axis = self._axis_vectors(cache, direction, frame)
            delta = pos[1:] - pos[:-1]
            projected = (delta * axis[:-1]).sum(dim=-1)
            if multiply_fps:
                projected = projected * float(cache.fps)
            positive = torch.clamp(projected, min=0.0)
            cumulative = torch.cat([
                torch.zeros(1, device=pos.device, dtype=pos.dtype),
                torch.cumsum(positive, dim=0),
            ])
            return _to_numpy(cumulative)

        if name == "direction_score":
            entity = args.get("entity", "pelvis")
            direction = args.get("direction", "forward")
            frame = args.get("frame", "body")
            pos = self.resolve_entity_positions(cache, entity)
            axis = self._axis_vectors(cache, direction, frame)
            delta = pos[1:] - pos[:-1]
            projected = (delta * axis[:-1]).sum(dim=-1)
            positive = torch.cumsum(torch.clamp(projected, min=0.0), dim=0)
            negative = torch.cumsum(torch.clamp(-projected, min=0.0), dim=0)
            ratio = positive / (positive + negative).clamp(min=1e-8)
            ratio = torch.cat([torch.zeros(1, device=pos.device, dtype=pos.dtype), ratio])
            return _to_numpy(ratio)

        if name == "yaw_rotation":
            direction = str(args.get("direction", "left")).lower()
            signed_deg = cache.motion_raw[:, 0] * 2.0 * (180.0 / math.pi)
            if direction == "right":
                signed_deg = -signed_deg
            elif direction != "left":
                signed_deg = signed_deg.abs()
            return _to_numpy(torch.cumsum(torch.clamp(signed_deg, min=0.0), dim=0))

        if name == "pose_angle":
            a = self.resolve_entity_positions(cache, args["a"])
            b = self.resolve_entity_positions(cache, args["b"])
            c = self.resolve_entity_positions(cache, args["c"])
            ba = _normalize(a - b)
            bc = _normalize(c - b)
            cos_angle = (ba * bc).sum(dim=-1).clamp(-1.0, 1.0)
            return _to_numpy(torch.rad2deg(torch.acos(cos_angle)))

        raise ValueError(f"Unknown signal: {name}")

    def reduce_signal(self, values: np.ndarray, reduce: str) -> float:
        reduce = str(reduce or "last").lower()
        if values.size == 0:
            return 0.0
        if reduce == "min":
            return float(np.min(values))
        if reduce == "max":
            return float(np.max(values))
        if reduce == "mean":
            return float(np.mean(values))
        if reduce == "last":
            return float(values[-1])
        if reduce == "sum":
            return float(np.sum(values))
        raise ValueError(f"Unsupported signal reducer: {reduce}")

    def state(self, cache: MotionCache, name: str, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        name = str(name).lower()
        args = args or {}
        if name != "hands_up":
            raise ValueError(f"Unknown state: {name}")

        head_y = self.resolve_entity_positions(cache, "head")[:, 1]
        margin = float(args.get("margin", -0.03))
        mode = str(args.get("mode", "both")).lower()
        left_up = self.resolve_entity_positions(cache, "l_hand")[:, 1] > head_y + margin
        right_up = self.resolve_entity_positions(cache, "r_hand")[:, 1] > head_y + margin
        if mode == "left":
            mask = left_up
        elif mode == "right":
            mask = right_up
        else:
            mask = left_up & right_up

        segments = []
        for start, end in _bool_segments(_to_numpy(mask), min_len=int(args.get("min_frames", 3))):
            segments.append({
                "start": start,
                "end": end,
                "key_frame": start,
                "score": 1.0,
                "label": "hands_up",
                "meta": {"state": "hands_up", "mode": mode},
            })
        return segments

    def template(self, cache: MotionCache, name: str, args: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        name = str(name).lower()
        args = args or {}
        if name == "step":
            return self._template_step(cache, args), []
        if name in {"direction_phase", "movement_phase", "move_phase"}:
            return self._template_direction_phase(cache, args), []
        if name == "clap":
            return self._template_clap(cache, args)
        if name == "squat_cycle":
            return self._template_squat(cache, args), []
        if name in {"touch_head", "hand_to_head"}:
            return self._template_touch(cache, args, target="head")
        if name in {"hands_close", "bring_hands_together"}:
            return self._template_hands_close(cache, args)
        if name in {"raise_foot", "foot_up"}:
            return self._template_raise_foot(cache, args), []
        if name in {"turn_left", "turn_right"}:
            direction = "left" if name == "turn_left" else "right"
            return self._template_turn(cache, args, direction), []
        raise ValueError(f"Unknown template: {name}")

    def _template_step(self, cache: MotionCache, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        # PDF spec (page 6): l/r_move_state = (1 - foot_contact_263) > 0.5,
        # count rest->move rising edges per foot. The legacy hybrid detector
        # (contact + height + speed + landing) is still reachable via
        # `args["detector"] = "hybrid"` for backwards compatibility.
        foot = str(args.get("foot", "any")).lower()
        detector_kind = str(args.get("detector", "move_state")).lower()
        if detector_kind == "hybrid":
            accepted = (
                "contact_threshold", "max_contact_height", "max_contact_speed",
                "min_contact_frames", "min_valid_ratio", "max_contact_gap",
                "min_step_separation", "landing_height", "swing_height",
                "landing_min_travel", "landing_lookback", "landing_confidence",
            )
        else:
            accepted = ("contact_threshold", "min_step_separation", "min_move_frames")
        detector_options = {k: args[k] for k in accepted if k in args}
        events = detect_step_events(
            cache.joints,
            cache.foot_contact,
            foot=foot,
            detector=detector_kind,
            **detector_options,
        )
        segments: List[Dict[str, Any]] = []
        for event in events:
            label = "l_foot" if event.foot == "left" else "r_foot"
            segments.append({
                "start": int(event.start),
                "end": int(event.end),
                "key_frame": int(event.key_frame),
                "score": float(event.confidence),
                "label": label,
                "meta": {
                    "template": "step",
                    "foot": label,
                    "detector": detector_kind,
                    "source": event.source,
                    "valid_ratio": event.valid_ratio,
                    "min_height": event.min_height,
                    "mean_speed": event.mean_speed,
                },
            })
        return sorted(segments, key=lambda item: item["start"])

    def _template_direction_phase(self, cache: MotionCache, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        direction = _normalize_direction_name(args.get("direction", "forward"))
        frame = str(args.get("frame", "body")).lower()
        entity = args.get("entity", "pelvis")
        min_displacement = float(args.get("min_displacement", 0.12))
        min_frames = int(args.get("min_frames", max(4, round(0.25 * cache.fps))))
        purity_threshold = float(args.get("purity_threshold", 0.55))
        gap_tolerance = int(args.get("gap_tolerance", max(1, round(0.1 * cache.fps))))

        pos = self.resolve_entity_positions(cache, entity)
        axis = self._axis_vectors(cache, direction, frame)
        delta = pos[1:] - pos[:-1]
        projected = (delta * axis[:-1]).sum(dim=-1)
        lateral = torch.norm(delta[:, [0, 2]], dim=-1).clamp(min=1e-8)
        purity = projected / lateral
        moving = lateral > float(args.get("min_speed_per_frame", 0.0025))
        active = _to_numpy((projected > 0.0) & moving & (purity >= purity_threshold)).astype(bool)

        if gap_tolerance > 0 and active.size > 0:
            filled = active.copy()
            false_runs = _bool_segments(~active, min_len=1)
            for start, end in false_runs:
                if start > 0 and end < len(active) and end - start <= gap_tolerance:
                    filled[start:end] = True
            active = filled

        # Frame mask has length T-1 because it describes inter-frame motion.
        segments: List[Dict[str, Any]] = []
        for start, end in _bool_segments(active, min_len=max(1, min_frames - 1)):
            frame_start = int(start)
            frame_end = int(min(end + 1, pos.shape[0]))
            local_projected = projected[start:end]
            local_delta = delta[start:end]
            displacement = float(torch.clamp(local_projected, min=0.0).sum().item())
            path = float(torch.norm(local_delta[:, [0, 2]], dim=-1).sum().item())
            if displacement < min_displacement:
                continue
            local_purity = displacement / max(path, 1e-8)
            segments.append({
                "start": frame_start,
                "end": frame_end,
                "key_frame": frame_start,
                "score": local_purity,
                "label": f"{direction}_phase",
                "meta": {
                    "template": "direction_phase",
                    "direction": direction,
                    "frame": frame,
                    "entity": entity,
                    "displacement": displacement,
                    "path": path,
                    "purity": local_purity,
                    "duration": (frame_end - frame_start) / cache.fps,
                },
            })
        return segments

    def _template_clap(self, cache: MotionCache, args: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        # PDF spec (page 7): basin detection with hysteresis.
        #   enter_threshold = threshold (default 0.077)
        #   exit_threshold  = threshold * 1.6
        # State machine on dist between hands: idle -> inside (dist <= enter)
        # -> idle again only when dist > exit. Each completed (enter, exit)
        # pair counts as one clap.
        threshold = float(args.get("threshold", 0.077))
        enter_threshold = float(args.get("enter_threshold", threshold))
        exit_threshold = float(args.get("exit_threshold", threshold * 1.6))
        min_frames = int(args.get("min_frames", 1))
        dist = self.signal(cache, "dist", {"a": "l_hand", "b": "r_hand"})

        segments: List[Dict[str, Any]] = []
        inside = False
        run_start: Optional[int] = None
        for idx, value in enumerate(dist):
            if not inside and value <= enter_threshold:
                inside = True
                run_start = idx
            elif inside and value > exit_threshold:
                run_end = idx
                if run_start is not None and run_end - run_start >= min_frames:
                    local = dist[run_start:run_end]
                    key = run_start + int(np.argmin(local)) if local.size else run_start
                    segments.append({
                        "start": run_start,
                        "end": run_end,
                        "key_frame": key,
                        "score": 1.0,
                        "label": "clap",
                        "meta": {
                            "template": "clap",
                            "left_hand": "l_hand",
                            "right_hand": "r_hand",
                            "enter_threshold": enter_threshold,
                            "exit_threshold": exit_threshold,
                            "min_dist": float(np.min(local)) if local.size else 0.0,
                        },
                    })
                inside = False
                run_start = None
        if inside and run_start is not None and len(dist) - run_start >= min_frames:
            local = dist[run_start:]
            key = run_start + int(np.argmin(local)) if local.size else run_start
            segments.append({
                "start": run_start,
                "end": len(dist),
                "key_frame": key,
                "score": 1.0,
                "label": "clap",
                "meta": {
                    "template": "clap",
                    "left_hand": "l_hand",
                    "right_hand": "r_hand",
                    "enter_threshold": enter_threshold,
                    "exit_threshold": exit_threshold,
                    "min_dist": float(np.min(local)) if local.size else 0.0,
                    "open_basin": True,
                },
            })

        limitations = [
            "clap uses enter/exit hysteresis on inter-hand distance; "
            "approach and separation kinematics are not explicitly verified."
        ]
        return segments, limitations

    def _template_hands_close(self, cache: MotionCache, args: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        threshold = float(args.get("threshold", 0.12))
        min_frames = int(args.get("min_frames", 2))
        dist = self.signal(cache, "dist", {"a": "l_hand", "b": "r_hand"})
        segments = []
        for start, end in _bool_segments(dist < threshold, min_len=min_frames):
            local = dist[start:end]
            key = start + int(np.argmin(local)) if local.size else start
            segments.append({
                "start": start,
                "end": end,
                "key_frame": key,
                "score": 1.0,
                "label": "hands_close",
                "meta": {
                    "template": "hands_close",
                    "min_dist": float(np.min(local)) if local.size else 0.0,
                },
            })
        return segments, []

    def _template_touch(self, cache: MotionCache, args: Dict[str, Any], target: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        hand = str(args.get("hand", "any")).lower()
        threshold = float(args.get("threshold", 0.18))
        min_frames = int(args.get("min_frames", 1))
        target_name = args.get("target", target)
        hands = []
        if hand in {"any", "left", "l_hand"}:
            hands.append("l_hand")
        if hand in {"any", "right", "r_hand"}:
            hands.append("r_hand")

        segments: List[Dict[str, Any]] = []
        for hand_name in hands:
            dist = self.signal(cache, "dist", {"a": hand_name, "b": target_name})
            for start, end in _bool_segments(dist < threshold, min_len=min_frames):
                local = dist[start:end]
                key = start + int(np.argmin(local)) if local.size else start
                segments.append({
                    "start": start,
                    "end": end,
                    "key_frame": key,
                    "score": 1.0,
                    "label": f"{hand_name}_touch_{target_name}",
                    "meta": {
                        "template": "touch_head",
                        "hand": hand_name,
                        "target": target_name,
                        "min_dist": float(np.min(local)) if local.size else 0.0,
                    },
                })
        return sorted(segments, key=lambda item: item["start"]), []

    def _template_raise_foot(self, cache: MotionCache, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        # PDF spec (page 5):
        #   r_foot_height = pos(r_foot).y
        #   value = max(r_foot_height)
        #   reward = value > threshold     (threshold = 0.08, binary)
        # That collapses to a single segment iff the max foot height exceeds
        # the threshold. mode="binary_max" implements PDF exactly; the legacy
        # mode="segments" (default for backward compatibility) instead emits
        # one segment per sustained period above the threshold so callers can
        # use count semantics (e.g. "raise the right foot twice"). To opt into
        # the PDF formulation, also expose `signal:foot_height` with
        # reduce="max".
        foot = str(args.get("foot", "any")).lower()
        threshold = float(args.get("threshold", 0.08))
        min_frames = int(args.get("min_frames", 2))
        mode = str(args.get("mode", "segments")).lower()
        ground_y = torch.minimum(
            cache.joints[:, JOINT_INDEX["l_foot"], 1].min(),
            cache.joints[:, JOINT_INDEX["r_foot"], 1].min(),
        ).item()
        candidates = []
        if foot in {"any", "left", "l_foot"}:
            candidates.append("l_foot")
        if foot in {"any", "right", "r_foot"}:
            candidates.append("r_foot")

        segments: List[Dict[str, Any]] = []
        for foot_name in candidates:
            height = _to_numpy(cache.joints[:, JOINT_INDEX[foot_name], 1] - ground_y)
            if mode == "binary_max":
                if height.size == 0:
                    continue
                max_h = float(np.max(height))
                if max_h <= threshold:
                    continue
                key = int(np.argmax(height))
                segments.append({
                    "start": key,
                    "end": key + 1,
                    "key_frame": key,
                    "score": float(max_h / max(threshold, 1e-8)),
                    "label": f"raise_{foot_name}",
                    "meta": {
                        "template": "raise_foot",
                        "foot": foot_name,
                        "max_height": max_h,
                        "mode": "binary_max",
                    },
                })
                continue
            for start, end in _bool_segments(height > threshold, min_len=min_frames):
                local = height[start:end]
                key = start + int(np.argmax(local)) if local.size else start
                segments.append({
                    "start": start,
                    "end": end,
                    "key_frame": key,
                    "score": float(np.max(local) / max(threshold, 1e-8)) if local.size else 1.0,
                    "label": f"raise_{foot_name}",
                    "meta": {
                        "template": "raise_foot",
                        "foot": foot_name,
                        "max_height": float(np.max(local)) if local.size else 0.0,
                        "mode": "segments",
                    },
                })
        return sorted(segments, key=lambda item: item["start"])

    def _template_squat(self, cache: MotionCache, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        # PDF spec (page 7-8):
        #   values = pos(pelvis).y
        #   local_min = local_find(values)
        #   left_peak = walk_local_peak_left(local_min)
        #   right_peak = walk_local_peak_right(local_min)
        #   drop = min(left_peak, right_peak) - local_min
        #   event = drop >= threshold     (threshold = 0.15)
        # `walk_local_peak_{left,right}` walks outward from each local minimum
        # until the height stops increasing -- i.e. it finds the nearest local
        # maximum on each side rather than capping at a fixed window.
        pelvis_y = self.signal(cache, "height", {"entity": "pelvis"})
        threshold = float(args.get("threshold", 0.15))
        segments: List[Dict[str, Any]] = []
        n = len(pelvis_y)
        if n < 3:
            return segments

        for idx in range(1, n - 1):
            if not (pelvis_y[idx] <= pelvis_y[idx - 1] and pelvis_y[idx] < pelvis_y[idx + 1]):
                continue
            left_peak_idx = idx
            while left_peak_idx - 1 >= 0 and pelvis_y[left_peak_idx - 1] >= pelvis_y[left_peak_idx]:
                left_peak_idx -= 1
            right_peak_idx = idx
            while right_peak_idx + 1 < n and pelvis_y[right_peak_idx + 1] >= pelvis_y[right_peak_idx]:
                right_peak_idx += 1
            drop = min(pelvis_y[left_peak_idx], pelvis_y[right_peak_idx]) - pelvis_y[idx]
            if drop >= threshold:
                segments.append({
                    "start": int(left_peak_idx),
                    "end": int(max(right_peak_idx + 1, idx + 1)),
                    "key_frame": int(idx),
                    "score": float(drop / max(threshold, 1e-8)),
                    "label": "squat_cycle",
                    "meta": {
                        "template": "squat_cycle",
                        "drop": float(drop),
                        "left_peak_frame": int(left_peak_idx),
                        "right_peak_frame": int(right_peak_idx),
                    },
                })
        return segments

    def _template_turn(self, cache: MotionCache, args: Dict[str, Any], direction: str) -> List[Dict[str, Any]]:
        # PDF spec (page 5):
        #   yaw_vel = root_rot_vel_in_263_feat (motion_raw[:, 0]); 2x for half-angle convention
        #   angle = sum(yaw_vel) in degrees
        #   tl_state = (yaw_vel/threshold).clip(0, None) > threshold  (per-frame "actively turning")
        #   angle_flag = angle in (min_angle, max_angle)
        #   reward = time(tl_state) > time_threshold if angle_flag else 0
        # Per-frame active mask is preserved as before; we additionally enforce
        # the max_angle gate and an explicit time_threshold so segments that do
        # not actually meet the PDF criteria are filtered out (yielding zero
        # reward downstream).
        min_angle = float(args.get("min_angle_deg", 20.0))
        max_angle = float(args.get("max_angle_deg", float("inf")))
        min_frames = int(args.get("min_frames", max(2, round(0.2 * cache.fps))))
        time_threshold_frames = int(args.get("time_threshold_frames", min_frames))
        signed = cache.motion_raw[:, 0] * 2.0 * (180.0 / math.pi)
        if direction == "right":
            signed = -signed
        active = _to_numpy(signed > float(args.get("per_frame_deg", 0.5)))

        segments: List[Dict[str, Any]] = []
        for start, end in _bool_segments(active, min_len=min_frames):
            duration_frames = end - start
            if duration_frames < time_threshold_frames:
                continue
            angle = float(torch.clamp(signed[start:end], min=0.0).sum().item())
            if angle < min_angle or angle > max_angle:
                continue
            segments.append({
                "start": start,
                "end": end,
                "key_frame": start,
                "score": min(1.0, angle / max(min_angle, 1e-8)),
                "label": f"turn_{direction}",
                "meta": {
                    "template": f"turn_{direction}",
                    "angle_deg": angle,
                    "duration": duration_frames / cache.fps,
                    "duration_frames": duration_frames,
                    "min_angle_deg": min_angle,
                    "max_angle_deg": max_angle if max_angle != float("inf") else None,
                    "time_threshold_frames": time_threshold_frames,
                },
            })
        return segments

    def resolve_atoms(self, cache: MotionCache, ref: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        ref_type = str(ref.get("type", "")).lower()
        name = ref.get("name", "")
        args = ref.get("args", {})
        if ref_type == "template":
            return self.template(cache, name, args)
        if ref_type == "state":
            return self.state(cache, name, args), []
        if ref_type == "event":
            return self.template(cache, name, args)
        raise ValueError(f"Temporal atoms must be state/event/template, got: {ref_type}")

    def _segment_measure(self, segment: Dict[str, Any], measure: str, cache: MotionCache) -> float:
        measure = str(measure or "count").lower()
        if measure == "duration":
            return (segment["end"] - segment["start"]) / cache.fps
        if measure in segment.get("meta", {}):
            return float(segment["meta"][measure])
        return float(segment.get("score", 1.0))

    def _filter_by_evidence(
        self,
        cache: MotionCache,
        ref: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        atoms, limitations = self.resolve_atoms(cache, ref)
        if not evidence:
            return atoms, limitations
        measure = evidence.get("measure", "score")
        op = evidence.get("op", "ge")
        value = float(evidence.get("value", 0.0))
        tolerance = float(evidence.get("tolerance", 0.0))
        kept = [
            atom for atom in atoms
            if compare_violation(self._segment_measure(atom, measure, cache), op, value, tolerance) == 0.0
        ]
        return kept, limitations

    def evaluate_signal(self, cache: MotionCache, constraint: Dict[str, Any]) -> ExecutionResult:
        ref = constraint["ref"]
        values = self.signal(cache, ref["name"], ref.get("args", {}))
        measured = self.reduce_signal(values, constraint.get("reduce", "last"))
        target = float(constraint.get("value", 0.0))
        op = constraint.get("op", "ge")
        weight = float(constraint.get("weight", 1.0))
        violation = compare_violation(measured, op, target, float(constraint.get("tolerance", 0.0)))
        reward, score = _reward_from_violation(violation, weight)
        return ExecutionResult(
            constraint_id=constraint.get("id", ref.get("name", "signal")),
            constraint_type="signal",
            measured_value=measured,
            target=target,
            op=op,
            reward=reward,
            violation=violation,
            score=score,
            source_repr=f"signal:{ref.get('name')}({ref.get('args', {})})",
        )

    def evaluate_count(self, cache: MotionCache, constraint: Dict[str, Any]) -> ExecutionResult:
        atoms, limitations = self.resolve_atoms(cache, constraint["ref"])
        measured = float(len(atoms))
        target = float(constraint.get("value", 0.0))
        op = constraint.get("op", "eq")
        weight = float(constraint.get("weight", 1.0))
        tolerance = float(constraint.get("tolerance", 0.0))
        if op == "eq":
            violation, reward, score = _event_count_reward(measured, target, tolerance, weight)
        else:
            violation = compare_violation(measured, op, target, tolerance)
            reward, score = _reward_from_violation(violation, weight)
        return ExecutionResult(
            constraint_id=constraint.get("id", constraint["ref"].get("name", "count")),
            constraint_type="count",
            measured_value=measured,
            target=target,
            op=op,
            reward=reward,
            violation=violation,
            score=score,
            source_repr=f"{constraint['ref'].get('type')}:{constraint['ref'].get('name')}({constraint['ref'].get('args', {})})",
            matched_segments=atoms,
            limitations=limitations,
        )

    def evaluate_phase_count(self, cache: MotionCache, constraint: Dict[str, Any]) -> ExecutionResult:
        phase_ref = constraint["phase_ref"]
        count_ref = constraint["count_ref"]
        phase_atoms, phase_limitations = self.resolve_atoms(cache, phase_ref)
        count_atoms, count_limitations = self.resolve_atoms(cache, count_ref)

        target = float(constraint.get("value", 0.0))
        op = constraint.get("op", "eq")
        tolerance = float(constraint.get("tolerance", 0.0))
        weight = float(constraint.get("weight", 1.0))
        require_order = bool(constraint.get("require_order", False))
        order_index = int(constraint.get("order", 0))

        candidates = phase_atoms
        if require_order:
            candidates = sorted(phase_atoms, key=lambda item: item["start"])
            if 0 <= order_index < len(candidates):
                candidates = [candidates[order_index]]
            else:
                candidates = []

        best_violation = float("inf")
        best_score = 0.0
        best_measured = 0.0
        best_phase: Optional[Dict[str, Any]] = None
        best_counts: List[Dict[str, Any]] = []

        for phase in candidates:
            inside = [
                atom for atom in count_atoms
                if atom["key_frame"] >= phase["start"] and atom["key_frame"] < phase["end"]
            ]
            measured = float(len(inside))
            if op == "eq":
                violation, _, score = _event_count_reward(measured, target, tolerance, weight)
            else:
                violation = compare_violation(measured, op, target, tolerance)
                _, score = _reward_from_violation(violation, weight)
            if violation < best_violation:
                best_violation = violation
                best_score = score
                best_measured = measured
                best_phase = phase
                best_counts = inside

        if best_violation == float("inf"):
            best_violation = compare_violation(0.0, op, target, tolerance)
            best_score = math.exp(-best_violation / max(abs(target), 1.0))

        if op == "eq":
            normalized = best_violation / max(abs(target), 1.0)
            reward = -weight * normalized
        else:
            reward = -weight * best_violation

        return ExecutionResult(
            constraint_id=constraint.get("id", "phase_count"),
            constraint_type="phase_count",
            measured_value=best_measured,
            target=target,
            op=op,
            reward=reward,
            violation=best_violation,
            score=best_score,
            source_repr=(
                f"phase_count:{phase_ref.get('name')}({phase_ref.get('args', {})})/"
                f"{count_ref.get('name')}({count_ref.get('args', {})})"
            ),
            matched_segments=([best_phase] if best_phase is not None else []) + best_counts,
            limitations=phase_limitations + count_limitations,
        )

    def evaluate_phase_signal(self, cache: MotionCache, constraint: Dict[str, Any]) -> ExecutionResult:
        phase_ref = constraint["phase_ref"]
        measure = str(constraint.get("measure", "displacement")).lower()
        phase_atoms, limitations = self.resolve_atoms(cache, phase_ref)
        target = float(constraint.get("value", 0.0))
        op = constraint.get("op", "ge")
        tolerance = float(constraint.get("tolerance", 0.0))
        weight = float(constraint.get("weight", 1.0))

        best_value = 0.0
        best_phase: Optional[Dict[str, Any]] = None
        best_violation = float("inf")
        for phase in phase_atoms:
            value = float(phase.get("meta", {}).get(measure, phase.get("score", 0.0)))
            violation = compare_violation(value, op, target, tolerance)
            if violation < best_violation:
                best_violation = violation
                best_value = value
                best_phase = phase

        if best_violation == float("inf"):
            best_violation = compare_violation(0.0, op, target, tolerance)

        reward, score = _reward_from_violation(best_violation, weight)
        return ExecutionResult(
            constraint_id=constraint.get("id", "phase_signal"),
            constraint_type="phase_signal",
            measured_value=best_value,
            target=target,
            op=op,
            reward=reward,
            violation=best_violation,
            score=score,
            source_repr=f"phase_signal:{phase_ref.get('name')}({phase_ref.get('args', {})})",
            matched_segments=[best_phase] if best_phase is not None else [],
            limitations=limitations,
        )

    def evaluate_absence(self, cache: MotionCache, constraint: Dict[str, Any]) -> ExecutionResult:
        """Reward absence of unwanted events or low unwanted signal magnitude."""
        ref = constraint["ref"]
        ref_type = str(ref.get("type", "")).lower()
        weight = float(constraint.get("weight", 1.0))

        if ref_type == "signal":
            values = self.signal(cache, ref["name"], ref.get("args", {}))
            measured = self.reduce_signal(values, constraint.get("reduce", "max"))
            target = float(constraint.get("value", 0.0))
            op = constraint.get("op", "le")
            violation = compare_violation(measured, op, target, float(constraint.get("tolerance", 0.0)))
            reward, score = _reward_from_violation(violation, weight)
            return ExecutionResult(
                constraint_id=constraint.get("id", ref.get("name", "absence")),
                constraint_type="absence",
                measured_value=measured,
                target=target,
                op=op,
                reward=reward,
                violation=violation,
                score=score,
                source_repr=f"absence:signal:{ref.get('name')}",
            )

        atoms, limitations = self.resolve_atoms(cache, ref)
        measured = float(len(atoms))
        target = float(constraint.get("value", 0.0))
        tolerance = float(constraint.get("tolerance", 0.0))
        violation, reward, score = _event_count_reward(measured, target, tolerance, weight)
        return ExecutionResult(
            constraint_id=constraint.get("id", ref.get("name", "absence")),
            constraint_type="absence",
            measured_value=measured,
            target=target,
            op="eq",
            reward=reward,
            violation=violation,
            score=score,
            source_repr=f"absence:{ref.get('type')}:{ref.get('name')}",
            matched_segments=atoms,
            limitations=limitations,
        )

    def evaluate_temporal(self, cache: MotionCache, constraint: Dict[str, Any]) -> ExecutionResult:
        lhs, lhs_limitations = self.resolve_atoms(cache, constraint["lhs"])
        rhs, rhs_limitations = self.resolve_atoms(cache, constraint["rhs"])
        relation = str(constraint.get("relation", "before")).lower()
        matched: List[Dict[str, Any]] = []

        for left in lhs:
            for right in rhs:
                ok = False
                if relation == "before":
                    ok = left["end"] <= right["start"]
                elif relation == "after":
                    ok = left["start"] >= right["end"]
                elif relation == "overlap":
                    ok = max(left["start"], right["start"]) < min(left["end"], right["end"])
                elif relation == "meet":
                    ok = abs(left["end"] - right["start"]) <= 1
                elif relation == "during":
                    ok = left["start"] >= right["start"] and left["end"] <= right["end"]
                else:
                    raise ValueError(f"Unsupported temporal relation: {relation}")
                if ok:
                    matched.append({"lhs": left, "rhs": right, "relation": relation})

        measured = 1.0 if matched else 0.0
        violation = compare_violation(measured, "eq", 1.0)
        weight = float(constraint.get("weight", 1.0))
        reward, score = _reward_from_violation(violation, weight)
        return ExecutionResult(
            constraint_id=constraint.get("id", "temporal"),
            constraint_type="temporal",
            measured_value=measured,
            target=1.0,
            op="eq",
            reward=reward,
            violation=violation,
            score=score,
            source_repr=f"temporal:{relation}",
            matched_events=matched,
            limitations=lhs_limitations + rhs_limitations,
        )

    def evaluate_temporal_composite(self, cache: MotionCache, constraint: Dict[str, Any]) -> ExecutionResult:
        lhs_ref = constraint["lhs"]["ref"]
        lhs_evidence = constraint["lhs"].get("evidence", {})
        rhs_ref = constraint["rhs"]["ref"]
        rhs_evidence = constraint["rhs"].get("evidence", {})
        relation = constraint.get("relation", {})
        relation_name = relation.get("name", "evidence_before")
        weight = float(constraint.get("weight", 1.0))

        if lhs_ref.get("type") != "signal":
            lhs_atoms, lhs_limitations = self._filter_by_evidence(cache, lhs_ref, lhs_evidence)
            lhs_ok = bool(lhs_atoms)
            lhs_values = None
        else:
            lhs_values = self.signal(cache, lhs_ref["name"], lhs_ref.get("args", {}))
            lhs_measure = lhs_evidence.get("measure", "last")
            lhs_measured = self.reduce_signal(lhs_values, "last" if lhs_measure == "displacement" else lhs_measure)
            lhs_ok = compare_violation(
                lhs_measured,
                lhs_evidence.get("op", "ge"),
                float(lhs_evidence.get("value", 0.0)),
                float(lhs_evidence.get("tolerance", 0.0)),
            ) == 0.0
            lhs_limitations = []

        rhs_atoms, rhs_limitations = self._filter_by_evidence(cache, rhs_ref, rhs_evidence)
        matched_pairs: List[Dict[str, Any]] = []
        measured = 0.0

        if lhs_ok and rhs_atoms and relation_name == "evidence_before":
            value = float(relation.get("value", 0.0))
            op = relation.get("op", "ge")
            for rhs in rhs_atoms:
                anchor = int(rhs["start"] if relation.get("rhs_anchor", "start") == "start" else rhs["end"])
                if lhs_values is not None:
                    anchor = max(0, min(anchor, len(lhs_values) - 1))
                    lhs_pre_value = float(lhs_values[anchor])
                else:
                    lhs_pre_value = 1.0 if any(atom["end"] <= anchor for atom in lhs_atoms) else 0.0
                cur_violation = compare_violation(lhs_pre_value, op, value)
                if cur_violation == 0.0:
                    matched_pairs.append({"lhs_value": lhs_pre_value, "rhs": rhs})
                measured = max(measured, lhs_pre_value)

        gate_ok = lhs_ok and bool(rhs_atoms)
        violation = compare_violation(1.0 if (gate_ok and matched_pairs) else 0.0, "eq", 1.0)
        reward, score = _reward_from_violation(violation, weight)
        return ExecutionResult(
            constraint_id=constraint.get("id", "temporal_composite"),
            constraint_type="temporal_composite",
            measured_value=measured,
            target=float(relation.get("value", 1.0)),
            op=relation.get("op", "ge"),
            reward=reward,
            violation=violation,
            score=score,
            source_repr=f"temporal_composite:{relation_name}",
            matched_events=matched_pairs,
            limitations=lhs_limitations + rhs_limitations,
        )

    def evaluate(
        self,
        motion_raw: torch.Tensor,
        foot_contact: torch.Tensor,
        constraints: Any,
        joints: torch.Tensor,
    ) -> List[ExecutionResult]:
        """Evaluate constraints against a single motion sample.

        `constraints` accepts either:
          - a flat ``list[dict]`` of constraint specs (legacy / direct path), or
          - a wrapper ``dict`` with ``{"constraint_sets": [{"constraint_set_id",
            "prompt_id", "prompt", "constraints": [...]}, ...]}`` per the
            New-Reward schema. The wrapper is flattened internally and each
            result is tagged with its originating ``constraint_set_id`` and
            ``prompt_id`` so callers can trace rewards back to the prompt.

        `scope` on each constraint is honored here (whole_sequence is the
        default; frame_interval slices the shared time axis of motion_raw,
        joints, foot_contact, and the cached body frame).
        """
        flat: List[Tuple[Dict[str, Any], Optional[str], Optional[str]]] = []
        if isinstance(constraints, dict) and "constraint_sets" in constraints:
            for cset in constraints.get("constraint_sets", []) or []:
                set_id = cset.get("constraint_set_id")
                prompt_id = cset.get("prompt_id")
                for constraint in cset.get("constraints", []) or []:
                    flat.append((constraint, set_id, prompt_id))
        else:
            for constraint in constraints or []:
                flat.append((constraint, None, None))

        full_cache = self.make_cache(motion_raw, foot_contact, joints)
        results: List[ExecutionResult] = []
        for constraint, set_id, prompt_id in flat:
            scoped_cache, scope_range, scope_limitations = self._resolve_scope(full_cache, constraint)
            kind = str(constraint.get("kind", "")).lower()
            if kind == "signal":
                result = self.evaluate_signal(scoped_cache, constraint)
            elif kind == "count":
                result = self.evaluate_count(scoped_cache, constraint)
            elif kind == "phase_count":
                result = self.evaluate_phase_count(scoped_cache, constraint)
            elif kind == "phase_signal":
                result = self.evaluate_phase_signal(scoped_cache, constraint)
            elif kind == "absence":
                result = self.evaluate_absence(scoped_cache, constraint)
            elif kind == "temporal":
                result = self.evaluate_temporal(scoped_cache, constraint)
            elif kind == "temporal_composite":
                result = self.evaluate_temporal_composite(scoped_cache, constraint)
            else:
                raise ValueError(f"Unsupported constraint kind: {kind}")

            if scope_limitations:
                result.limitations = list(result.limitations) + scope_limitations
            if scope_range is not None:
                # Shift segment/event frame indices back to the original timeline
                # so callers see coordinates in the un-sliced motion.
                _shift_result_frames(result, scope_range[0])
            result.constraint_set_id = set_id
            result.prompt_id = prompt_id
            results.append(result)
        return results


def _shift_result_frames(result: ExecutionResult, offset: int) -> None:
    """Add `offset` to start/end/key_frame inside matched segments and events."""
    if offset == 0:
        return
    for seg in result.matched_segments:
        for key in ("start", "end", "key_frame"):
            if key in seg and isinstance(seg[key], (int, float)):
                seg[key] = int(seg[key]) + offset
    for event in result.matched_events:
        # temporal_composite emits {"lhs_value": float, "rhs": segment}; shift the segment.
        rhs = event.get("rhs") if isinstance(event, dict) else None
        if isinstance(rhs, dict):
            for key in ("start", "end", "key_frame"):
                if key in rhs and isinstance(rhs[key], (int, float)):
                    rhs[key] = int(rhs[key]) + offset
        # evaluate_temporal emits {"lhs": seg, "rhs": seg, "relation": ...}; shift both.
        lhs = event.get("lhs") if isinstance(event, dict) else None
        if isinstance(lhs, dict):
            for key in ("start", "end", "key_frame"):
                if key in lhs and isinstance(lhs[key], (int, float)):
                    lhs[key] = int(lhs[key]) + offset


def aggregate_executor_score(results: List[ExecutionResult]) -> float:
    if not results:
        return 0.0
    return float(np.mean([result.score for result in results]))
