"""
SpatiotemporalKinematicReward
Direct 3D kinematic reward for GRPO — no motion-to-text conversion.

Input pipeline:
  motion_raw [T, 263]  (denormalized HumanML3D)
      → recover_root_rot_pos  → r_rot_quat [T,4], r_pos [T,3]
      → recover_from_ric      → joints [T, 22, 3]  (world-space)

Sub-goal format (produced by an upstream LLM parser):
  [
    {'direction': 'left',  'target_steps': 6},
    {'direction': 'right', 'target_steps': 3},
    {'direction': 'forward', 'target_steps': None,
     'joint_activation': {'joint_a': 14, 'joint_b': 15, 'min_angle_deg': 20}},
  ]
  Supported directions: 'forward' | 'backward' | 'left' | 'right' | 'any'
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from utils.motion_utils import recover_root_rot_pos, recover_from_ric
from motion_step_detector import detect_steps

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HumanML3D joint indices (22-joint skeleton)
JOINT_NAMES = {
    0: 'pelvis',    1: 'l_hip',      2: 'r_hip',      3: 'spine1',
    4: 'l_knee',    5: 'r_knee',     6: 'spine2',      7: 'l_ankle',
    8: 'r_ankle',   9: 'spine3',    10: 'l_foot',     11: 'r_foot',
   12: 'neck',     13: 'l_collar',  14: 'r_collar',   15: 'head',
   16: 'l_shoulder',17: 'r_shoulder',18: 'l_elbow',   19: 'r_elbow',
   20: 'l_wrist',  21: 'r_wrist',
}

# Foot joint indices for step counting
_FOOT_JOINTS = [7, 8, 10, 11]   # l_ankle, r_ankle, l_foot, r_foot

_DIR2IDX = {
    'forward': 1,
    'backward': 2,
    'left': 3,
    'right': 4,
    'left_forward': 5,
    'right_forward': 6,
    'left_backward': 7,
    'right_backward': 8,
    'any': 0,
}
_IDX2DIR = {v: k for k, v in _DIR2IDX.items()}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubGoal:
    direction: str          # 'forward' | 'backward' | 'left' | 'right' | 'any'
    target_steps: Optional[float] = None
    target_rotation_deg: Optional[float] = None   # signed: + = left/CCW
    joint_activation: Optional[Dict[str, Any]] = None  # see module docstring
    span: str = 'phase'     # 'phase' = sequential matching, 'global' = whole motion

@dataclass
class MotionPhase:
    start: int              # inclusive frame index
    end: int                # exclusive frame index
    direction: str          # dominant direction label
    step_count: int = 0
    rotation_deg: float = 0.0
    consistency: float = 0.0    # 0 = perfect physical consistency, 1 = all fake

# ---------------------------------------------------------------------------
# Step 1 — Sub-goal normalisation
# ---------------------------------------------------------------------------

def parse_subgoals(raw: List[Dict[str, Any]]) -> List[SubGoal]:
    """Validate and convert raw dicts to SubGoal objects."""
    goals = []
    for d in raw:
        direction = d.get('direction', 'any').lower()
        assert direction in _DIR2IDX, f"Unknown direction: {direction}"
        goals.append(SubGoal(
            direction=direction,
            target_steps=d.get('target_steps'),
            target_rotation_deg=d.get('target_rotation_deg'),
            joint_activation=d.get('joint_activation'),
            span=d.get('span', 'phase'),
        ))
    return goals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_smooth(x: torch.Tensor, window: int) -> torch.Tensor:
    """1-D box smoothing along dim-0. x: [T, C] or [T]."""
    if window <= 1:
        return x
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(-1)
    C = x.shape[-1]
    # Use conv1d: [1, C, T]
    kernel = torch.ones(C, 1, window, device=x.device, dtype=x.dtype) / window
    pad = window // 2
    out = F.conv1d(x.T.unsqueeze(0), kernel, padding=pad, groups=C)
    out = out.squeeze(0).T  # [T, C]
    # Trim to original length
    out = out[:x.shape[0]]
    return out.squeeze(-1) if squeeze else out


def _facing_xz(r_rot_quat: torch.Tensor) -> torch.Tensor:
    """
    Character forward direction in world XZ from Y-axis rotation quaternion.

    HumanML3D half-angle convention:
        q = [cos(θ), 0, sin(θ), 0]  →  actual Y-rotation = 2θ
        forward_world = [sin(2θ), cos(2θ)]  in (X, Z)

    Args:
        r_rot_quat: [T, 4]
    Returns:
        facing: [T, 2]  unit vectors (X, Z)
    """
    w = r_rot_quat[:, 0]   # cos(θ)
    y = r_rot_quat[:, 2]   # sin(θ)
    fx = 2.0 * w * y                  # sin(2θ)
    fz = w * w - y * y                # cos(2θ)
    facing = torch.stack([fx, fz], dim=-1)
    # Normalise (should already be unit, but guard against fp drift)
    norm = facing.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return facing / norm


def _direction_axis_xz(direction: str, facing: torch.Tensor) -> torch.Tensor:
    """Return a body-relative XZ axis for cardinal or diagonal directions."""
    direction = str(direction or 'any').lower()
    left = torch.stack([-facing[:, 1], facing[:, 0]], dim=-1)
    components = {
        'forward': (0.0, 1.0),
        'backward': (0.0, -1.0),
        'left': (1.0, 0.0),
        'right': (-1.0, 0.0),
        'left_forward': (1.0, 1.0),
        'right_forward': (-1.0, 1.0),
        'left_backward': (1.0, -1.0),
        'right_backward': (-1.0, -1.0),
    }
    lat, fwd = components.get(direction, (0.0, 0.0))
    if lat == 0.0 and fwd == 0.0:
        return torch.zeros_like(facing)
    axis = lat * left + fwd * facing
    return axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _dominant_direction_label(v_forward: torch.Tensor, v_lateral: torch.Tensor, is_moving: torch.Tensor) -> torch.Tensor:
    """Classify per-frame velocity into 8-way body-relative directions."""
    dev = v_forward.device
    labels = torch.zeros(v_forward.shape[0], dtype=torch.long, device=dev)
    angle = torch.atan2(v_lateral, v_forward)  # left-positive, forward at 0.
    deg = angle * (180.0 / math.pi)

    labels[is_moving & (deg >= -22.5) & (deg <= 22.5)] = _DIR2IDX['forward']
    labels[is_moving & (deg > 22.5) & (deg <= 67.5)] = _DIR2IDX['left_forward']
    labels[is_moving & (deg > 67.5) & (deg <= 112.5)] = _DIR2IDX['left']
    labels[is_moving & (deg > 112.5) & (deg <= 157.5)] = _DIR2IDX['left_backward']
    labels[is_moving & (deg < -22.5) & (deg >= -67.5)] = _DIR2IDX['right_forward']
    labels[is_moving & (deg < -67.5) & (deg >= -112.5)] = _DIR2IDX['right']
    labels[is_moving & (deg < -112.5) & (deg >= -157.5)] = _DIR2IDX['right_backward']
    labels[is_moving & ((deg > 157.5) | (deg < -157.5))] = _DIR2IDX['backward']
    return labels


# ---------------------------------------------------------------------------
# Constants for kinematic reward
# ---------------------------------------------------------------------------

# Parent joint index for each joint (derived from t2m_kinematic_chain)
_PARENT_OF: Dict[int, int] = {
    1: 0, 4: 1, 7: 4, 10: 7,              # left leg
    2: 0, 5: 2, 8: 5, 11: 8,              # right leg
    3: 0, 6: 3, 9: 6, 12: 9, 15: 12,      # spine + head
    13: 9, 16: 13, 18: 16, 20: 18,         # left arm
    14: 9, 17: 14, 19: 17, 21: 19,         # right arm
}

# Empirical typical ROM (degrees) per (parent, child) joint pair
_TYPICAL_ROM_DEG: Dict[Tuple[int, int], float] = {
    (16, 18): 150.0,   # l_shoulder → l_elbow
    (17, 19): 150.0,   # r_shoulder → r_elbow
    (18, 20): 130.0,   # l_elbow → l_wrist
    (19, 21): 130.0,   # r_elbow → r_wrist
    (1, 4):  130.0,    # l_hip → l_knee
    (2, 5):  130.0,    # r_hip → r_knee
    (4, 7):  130.0,    # l_knee → l_ankle
    (5, 8):  130.0,    # r_knee → r_ankle
    (12, 15): 80.0,    # neck → head
    (13, 16): 180.0,   # l_collar → l_shoulder
    (14, 17): 180.0,   # r_collar → r_shoulder
    (3, 6):   60.0,    # spine1 → spine2
    (6, 9):   60.0,    # spine2 → spine3
}
_DEFAULT_ROM_DEG = 120.0

# Hybrid step detection thresholds. The shared detector uses foot contact as a
# candidate and verifies it with foot height, foot XZ speed, and landing cues.
_FOOT_SLIDE_VEL = 0.018      # m/frame, roughly 0.36 m/s at 20 fps
_FOOT_FLOAT_HEIGHT = 0.055   # meters above estimated ground
_VALID_CONTACT_RATIO = 0.5   # valid fraction required inside a contact run

# Phase segmentation thresholds
_MOVE_SPEED = 0.005           # m/frame minimum XZ speed to count as moving


# ---------------------------------------------------------------------------
# Step 2 — Hybrid physical step detection
# ---------------------------------------------------------------------------

def _detect_physical_steps_legacy_contact(
    joints: torch.Tensor,
    foot_contact: torch.Tensor,
    start: int,
    end: int,
) -> Tuple[int, float]:
    """Count steps with physical validation of foot contact channels.

    For each contact period (rising edge in the contact signal), checks
    whether the foot is actually stationary and on the ground.  Periods
    where >50 % of frames are physically inconsistent are rejected.

    Args:
        joints:       [T, 22, 3] world-space joint positions.
        foot_contact: [T, 4]     binary contact (l_heel, l_toe, r_heel, r_toe).
        start:        inclusive frame index.
        end:          exclusive frame index.

    Returns:
        (validated_step_count, consistency_penalty)
        consistency_penalty ∈ [0, 1]: fraction of contact frames that are
        physically inconsistent (sliding or floating).
    """
    fc = foot_contact[start:end]
    j = joints[start:end]
    N = fc.shape[0]
    if N < 2:
        return 0, 0.0

    # --- per-foot contact masks ---
    left_c = (fc[:, 0] + fc[:, 1]) > 0.5     # [N] bool
    right_c = (fc[:, 2] + fc[:, 3]) > 0.5    # [N] bool

    # --- foot XZ velocity (pad to length N) ---
    left_pos = j[:, 10, [0, 2]]              # l_foot XZ
    right_pos = j[:, 11, [0, 2]]             # r_foot XZ
    left_vel = F.pad(
        torch.norm(left_pos[1:] - left_pos[:-1], dim=-1), (1, 0)
    )
    right_vel = F.pad(
        torch.norm(right_pos[1:] - right_pos[:-1], dim=-1), (1, 0)
    )

    # --- foot Y height above estimated ground ---
    ground_y = min(j[:, 10, 1].min().item(), j[:, 11, 1].min().item())
    left_h = j[:, 10, 1] - ground_y
    right_h = j[:, 11, 1] - ground_y

    # --- physical validity: low velocity AND low height ---
    left_valid = (left_vel < _FOOT_SLIDE_VEL) & (left_h < _FOOT_FLOAT_HEIGHT)
    right_valid = (right_vel < _FOOT_SLIDE_VEL) & (right_h < _FOOT_FLOAT_HEIGHT)

    # --- consistency penalty ---
    total_contact = left_c.float().sum() + right_c.float().sum()
    inconsistent = (left_c & ~left_valid).float().sum() + \
                   (right_c & ~right_valid).float().sum()
    consistency_penalty = (
        (inconsistent / total_contact).item() if total_contact > 0 else 0.0
    )

    # --- validated step counting per foot ---
    steps = 0
    for contact_mask, valid_mask in [(left_c, left_valid), (right_c, right_valid)]:
        c = contact_mask.cpu()
        v = valid_mask.cpu()
        # Rising edges: transition from 0 → 1
        edges = (~c[:-1]) & c[1:]
        edge_indices = edges.nonzero(as_tuple=True)[0] + 1  # frame of first contact
        for ei in edge_indices.tolist():
            # Find end of this contact period
            period_end = ei
            while period_end < N and c[period_end]:
                period_end += 1
            period_len = period_end - ei
            if period_len < 1:
                continue
            valid_frac = v[ei:period_end].float().sum().item() / period_len
            if valid_frac >= _VALID_CONTACT_RATIO:
                steps += 1

    return steps, float(consistency_penalty)


def _detect_physical_steps(
    joints: torch.Tensor,
    foot_contact: torch.Tensor,
    start: int,
    end: int,
) -> Tuple[int, float]:
    """Count steps through the shared hybrid detector.

    The shared detector first validates contact runs with foot height and XZ
    speed, then adds landing events when contact labels are missing.  Keeping
    this wrapper preserves the old spatiotemporal reward interface.
    """
    result = detect_steps(
        joints,
        foot_contact,
        start=start,
        end=end,
        max_contact_height=_FOOT_FLOAT_HEIGHT,
        max_contact_speed=_FOOT_SLIDE_VEL,
        min_valid_ratio=_VALID_CONTACT_RATIO,
    )
    return result.count, result.consistency_penalty


# ---------------------------------------------------------------------------
# Step 3 — Joint activation via cumulative angular displacement
# ---------------------------------------------------------------------------

def _calculate_joint_accumulated_rotation(
    joints: torch.Tensor,
    joint_idx: int,
    parent_idx: int,
    start: int,
    end: int,
) -> float:
    """Cumulative angular displacement of a bone vector over a frame range.

    Measures how much the vector (parent → child) rotates frame-to-frame,
    summing absolute angle changes.  Works in world space so global
    translation is factored out automatically.

    Args:
        joints:     [T, 22, 3] world-space joint positions.
        joint_idx:  child joint index  (e.g. 18 = l_elbow).
        parent_idx: parent joint index (e.g. 16 = l_shoulder).
        start:      inclusive frame.
        end:        exclusive frame.

    Returns:
        Cumulative rotation in degrees.
    """
    bone = joints[start:end, joint_idx] - joints[start:end, parent_idx]  # [L, 3]
    L = bone.shape[0]
    if L < 2:
        return 0.0

    bone_len = bone.norm(dim=-1)                       # [L]
    valid = bone_len > 1e-6                            # mask degenerate frames
    bone_n = bone / bone_len.clamp(min=1e-8).unsqueeze(-1)  # [L, 3]

    cos_ang = (bone_n[1:] * bone_n[:-1]).sum(dim=-1)  # [L-1]
    cos_ang = cos_ang.clamp(-1.0, 1.0)
    angles = torch.acos(cos_ang)                       # [L-1] radians

    # Only count frames where both endpoints are valid
    pair_valid = valid[1:] & valid[:-1]
    cumulative_rad = (angles * pair_valid.float()).sum().item()
    return cumulative_rad * (180.0 / math.pi)


# ---------------------------------------------------------------------------
# Step 4 — Vectorized phase segmentation
# ---------------------------------------------------------------------------

def segment_phases(
    motion_raw: torch.Tensor,
    joints: torch.Tensor,
    foot_contact: torch.Tensor,
    min_phase_frames: int = 10,
) -> List[MotionPhase]:
    """Segment a motion into directional phases using vectorized ops.

    Direction is determined relative to the character's facing direction
    (body-relative), not world axes.  Single-frame jitter is removed by
    smoothing a one-hot direction encoding with :func:`_box_smooth`.

    Args:
        motion_raw:   [T, 263] denormalized HumanML3D motion.
        joints:       [T, 22, 3] world-space joint positions.
        foot_contact: [T, 4] binary foot contact labels.
        min_phase_frames: minimum frames per phase (default 10 = 0.5 s).

    Returns:
        List of :class:`MotionPhase` covering the full sequence.
    """
    T = motion_raw.shape[0]
    dev = motion_raw.device

    if T < max(4, min_phase_frames):
        sc, con = _detect_physical_steps(joints, foot_contact, 0, T)
        return [MotionPhase(start=0, end=T, direction='any',
                            step_count=sc, rotation_deg=0.0,
                            consistency=con)]

    # --- root trajectory & facing ---
    r_rot_quat, r_pos = recover_root_rot_pos(motion_raw.unsqueeze(0))
    r_rot_quat = r_rot_quat.squeeze(0)          # [T, 4]
    r_pos = r_pos.squeeze(0)                     # [T, 3]
    root_xz = r_pos[:, [0, 2]]                  # [T, 2]

    raw_vel = root_xz[1:] - root_xz[:-1]        # [T-1, 2]
    smooth_w = min(5, max(2, T // 6))
    vel_xz = _box_smooth(raw_vel, smooth_w)      # [T-1, 2]

    facing = _facing_xz(r_rot_quat)[:-1]         # [T-1, 2]

    # --- project onto facing / perpendicular ---
    v_forward = (vel_xz * facing).sum(dim=-1)                        # [T-1]
    facing_perp = torch.stack([-facing[:, 1], facing[:, 0]], dim=-1) # [T-1,2]
    v_lateral = (vel_xz * facing_perp).sum(dim=-1)                  # [T-1] +left

    speed = vel_xz.norm(dim=-1)                  # [T-1]
    is_moving = speed > _MOVE_SPEED

    # --- per-frame direction label (vectorized) ---
    # 0=stationary, 1..8=cardinal/diagonal body-relative directions.
    num_dirs = 9
    dir_label = _dominant_direction_label(v_forward, v_lateral, is_moving)

    # --- smooth labels via one-hot + box_smooth + argmax ---
    one_hot = torch.zeros(T - 1, num_dirs, device=dev)
    one_hot.scatter_(1, dir_label.unsqueeze(1), 1.0)
    sm = _box_smooth(one_hot, window=min(5, T - 1))  # [T-1, num_dirs]
    dir_smooth = sm.argmax(dim=-1)                    # [T-1]

    # --- boundaries from direction changes ---
    changes = torch.diff(dir_smooth)                  # [T-2]
    raw_bounds = (changes != 0).nonzero(as_tuple=True)[0] + 1  # +1 for diff shift

    # enforce min_phase_frames gap
    boundaries = [0]
    for b in raw_bounds.tolist():
        if b - boundaries[-1] >= min_phase_frames:
            boundaries.append(b)
    boundaries.append(T)

    # --- build MotionPhase per segment ---
    _label2dir = _IDX2DIR
    root_rot_vel = motion_raw[:, 0]
    phases: List[MotionPhase] = []

    for i in range(len(boundaries) - 1):
        sf = boundaries[i]
        ef = boundaries[i + 1]
        if ef - sf < 2:
            continue

        # dominant direction: mode of smoothed labels in this segment
        seg = dir_smooth[sf:min(ef, T - 1)]
        if seg.numel() == 0:
            dom_dir = 'any'
        else:
            dom_dir = _label2dir[seg.mode().values.item()]

        sc, con = _detect_physical_steps(joints, foot_contact, sf, ef)

        # rotation: half-angle convention → multiply by 2
        rot_rad = root_rot_vel[sf:ef].sum().item()
        rot_deg = rot_rad * 2.0 * (180.0 / math.pi)

        phases.append(MotionPhase(
            start=sf, end=ef, direction=dom_dir,
            step_count=sc, rotation_deg=rot_deg, consistency=con,
        ))

    # merge tiny trailing phase into predecessor
    if len(phases) > 1 and (phases[-1].end - phases[-1].start) < min_phase_frames:
        last = phases.pop()
        prev = phases[-1]
        phases[-1] = MotionPhase(
            start=prev.start, end=last.end, direction=prev.direction,
            step_count=prev.step_count + last.step_count,
            rotation_deg=prev.rotation_deg + last.rotation_deg,
            consistency=max(prev.consistency, last.consistency),
        )

    if not phases:
        sc, con = _detect_physical_steps(joints, foot_contact, 0, T)
        phases = [MotionPhase(start=0, end=T, direction='any',
                              step_count=sc, consistency=con)]
    return phases


# ---------------------------------------------------------------------------
# Step 4b — Score a single SubGoal against a frame range
# ---------------------------------------------------------------------------

def _score_subgoal_against_range(
    sg: SubGoal,
    joints: torch.Tensor,
    motion_raw: torch.Tensor,
    start: int,
    end: int,
    phase: MotionPhase,
    sigma_step: float = 1.5,
) -> float:
    """Score one SubGoal against a MotionPhase / frame range.

    Computes a weighted mean of applicable component scores:
    step accuracy, rotation accuracy, joint activation, and consistency.

    Returns:
        Score in [0, 1].
    """
    comp_scores: List[float] = []
    comp_weights: List[float] = []

    # (a) step count
    if sg.target_steps is not None:
        target = sg.target_steps
        sigma = max(target * 0.3, sigma_step)
        s = math.exp(-0.5 * ((phase.step_count - target) / sigma) ** 2)
        comp_scores.append(s)
        comp_weights.append(1.0)

    if sg.direction != 'any':
        r_rot_quat, r_pos = recover_root_rot_pos(motion_raw.unsqueeze(0))
        facing = _facing_xz(r_rot_quat.squeeze(0))[start:max(end, start + 1)]
        root_xz = r_pos.squeeze(0)[start:max(end, start + 1), [0, 2]]
        if root_xz.shape[0] >= 2:
            axis = _direction_axis_xz(sg.direction, facing[:-1])
            delta = root_xz[1:] - root_xz[:-1]
            path = delta.norm(dim=-1).sum().item()
            aligned = torch.clamp((delta * axis).sum(dim=-1), min=0.0).sum().item()
            dir_score = aligned / max(path, 1e-8)
        else:
            dir_score = 0.0
        comp_scores.append(float(np.clip(dir_score, 0.0, 1.0)))
        comp_weights.append(0.8)

    # (b) rotation
    if sg.target_rotation_deg is not None:
        target_rot = sg.target_rotation_deg
        sigma_rot = max(abs(target_rot) * 0.2, 15.0)
        s = math.exp(-0.5 * ((phase.rotation_deg - target_rot) / sigma_rot) ** 2)
        comp_scores.append(s)
        comp_weights.append(1.0)

    # (c) joint activation
    if sg.joint_activation is not None:
        ja = sg.joint_activation
        child = ja['joint_b']
        parent = ja.get('joint_a', _PARENT_OF.get(child, 0))
        min_angle = ja.get('min_angle_deg', 20.0)

        accumulated = _calculate_joint_accumulated_rotation(
            joints, child, parent, start, end,
        )
        rom = _TYPICAL_ROM_DEG.get((parent, child), _DEFAULT_ROM_DEG)
        rom_factor = rom / _DEFAULT_ROM_DEG
        ja_score = min(1.0, accumulated / max(min_angle * rom_factor, 1.0))
        comp_scores.append(ja_score)
        comp_weights.append(0.8)

    # (d) consistency bonus
    comp_scores.append(1.0 - phase.consistency)
    comp_weights.append(0.3)

    if comp_weights:
        w_total = sum(comp_weights)
        return sum(s * w for s, w in zip(comp_scores, comp_weights)) / w_total
    return 1.0


# ---------------------------------------------------------------------------
# Step 5 — Compositional evaluation (top-level reward)
# ---------------------------------------------------------------------------

def evaluate_compositional(
    joints: torch.Tensor,
    motion_raw: torch.Tensor,
    foot_contact: torch.Tensor,
    subgoals: List[SubGoal],
    sigma_step: float = 1.5,
    missing_phase_penalty: float = -0.4,
) -> float:
    """Score a motion against an ordered list of sub-goals.

    Segments the motion into directional phases, then performs strict
    sequential matching for ``span='phase'`` sub-goals and whole-motion
    evaluation for ``span='global'`` sub-goals (concurrent actions like
    "swing your arm while walking").

    An efficiency penalty is applied when a large fraction of frames fall
    outside any matched phase (excessive transition / idle time).

    Args:
        joints:        [T, 22, 3] world-space positions.
        motion_raw:    [T, 263] denormalized HumanML3D.
        foot_contact:  [T, 4] binary contact labels.
        subgoals:      ordered list of :class:`SubGoal`.
        sigma_step:    minimum σ for step-count Gaussian.
        missing_phase_penalty: score assigned to unmatched sub-goals.

    Returns:
        Reward in [0, 1].
    """
    if not subgoals:
        return 1.0

    T = motion_raw.shape[0]
    phases = segment_phases(motion_raw, joints, foot_contact)
    if not phases:
        return 0.0

    # --- split subgoals by span type ---
    phase_goals = [sg for sg in subgoals if sg.span == 'phase']
    global_goals = [sg for sg in subgoals if sg.span == 'global']

    # --- filter trivial (stationary) phases ---
    sig_phases = [
        p for p in phases
        if p.direction != 'any' or p.step_count > 0 or abs(p.rotation_deg) > 10
    ]
    if not sig_phases:
        sig_phases = phases

    # ===================================================================
    # A) Sequential matching for phase-scoped sub-goals
    # ===================================================================
    phase_scores: List[float] = []
    matched_phase_indices: List[int] = []

    if phase_goals:
        matched: List[Tuple[int, Optional[int]]] = []
        last_ph = -1
        for sg_i, sg in enumerate(phase_goals):
            found = False
            for ph_i in range(last_ph + 1, len(sig_phases)):
                if sg.direction == 'any' or sig_phases[ph_i].direction == sg.direction:
                    matched.append((sg_i, ph_i))
                    last_ph = ph_i
                    matched_phase_indices.append(ph_i)
                    found = True
                    break
            if not found:
                matched.append((sg_i, None))

        # first-phase penalty
        first_penalty = 1.0
        if (phase_goals[0].direction != 'any' and sig_phases
                and sig_phases[0].direction != phase_goals[0].direction):
            first_penalty = 0.3

        for sg_i, ph_i in matched:
            sg = phase_goals[sg_i]
            if ph_i is None:
                phase_scores.append(missing_phase_penalty)
                continue
            s = _score_subgoal_against_range(
                sg, joints, motion_raw, sig_phases[ph_i].start, sig_phases[ph_i].end,
                sig_phases[ph_i], sigma_step,
            )
            phase_scores.append(s)

        # apply first-phase penalty to the mean of phase scores
        if phase_scores:
            phase_avg = sum(phase_scores) / len(phase_scores) * first_penalty
        else:
            phase_avg = 0.0
    else:
        phase_avg = 0.0

    # ===================================================================
    # B) Global sub-goals — evaluated over the entire motion
    # ===================================================================
    global_scores: List[float] = []
    if global_goals:
        # Build a "whole motion" pseudo-phase for step/rotation scoring
        total_sc, total_con = _detect_physical_steps(joints, foot_contact, 0, T)
        rot_rad = motion_raw[:, 0].sum().item()
        whole_phase = MotionPhase(
            start=0, end=T, direction='any',
            step_count=total_sc,
            rotation_deg=rot_rad * 2.0 * (180.0 / math.pi),
            consistency=total_con,
        )
        for sg in global_goals:
            s = _score_subgoal_against_range(
                sg, joints, motion_raw, 0, T, whole_phase, sigma_step,
            )
            global_scores.append(s)

    # ===================================================================
    # C) Efficiency penalty — penalise excessive transition frames
    # ===================================================================
    if matched_phase_indices and T > 0:
        covered = sum(
            sig_phases[i].end - sig_phases[i].start
            for i in matched_phase_indices
        )
        coverage = covered / T
        efficiency = 0.8 + 0.2 * min(1.0, coverage)  # [0.8, 1.0]
    else:
        efficiency = 1.0

    # ===================================================================
    # D) Combine
    # ===================================================================
    all_scores = phase_scores + global_scores
    if not all_scores:
        return 0.0

    raw = sum(all_scores) / len(all_scores)
    if phase_scores:
        # re-weight: phase_avg already has first_penalty baked in
        n_ph = len(phase_scores)
        n_gl = len(global_scores)
        raw = (phase_avg * n_ph + sum(global_scores)) / (n_ph + n_gl) if (n_ph + n_gl) > 0 else 0.0
    raw *= efficiency
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Step 6 — Smart Adapter: ConstraintPhase → SubGoal
# ---------------------------------------------------------------------------

# Concurrent-action patterns: if a clause matches, its constraints become
# global (span='global') rather than phase-scoped.
import re as _re

_CONCURRENT_PATTERNS = _re.compile(
    r'\b(?:while|whilst|as\s+(?:he|she|they|it)\s+'
    r'(?:is|are|was|were)?|during|at\s+the\s+same\s+time'
    r'|simultaneously|with\s+(?:his|her|their))\b',
    _re.IGNORECASE,
)

# Body-part → (parent_joint, child_joint) for joint activation detection
_BODYPART_JOINTS: Dict[str, Tuple[int, int]] = {
    'arm':       (16, 20),   # shoulder → wrist  (left default)
    'left arm':  (16, 20),
    'right arm': (17, 21),
    'hand':      (18, 20),
    'left hand': (18, 20),
    'right hand':(19, 21),
    'wrist':     (18, 20),
    'left wrist':(18, 20),
    'right wrist':(19, 21),
    'elbow':     (16, 18),
    'left elbow':(16, 18),
    'right elbow':(17, 19),
    'head':      (12, 15),
    'neck':      (9, 12),
    'leg':       (1, 7),
    'left leg':  (1, 7),
    'right leg': (2, 8),
    'knee':      (1, 4),
    'left knee': (1, 4),
    'right knee':(2, 5),
    'shoulder':  (13, 16),
    'left shoulder': (13, 16),
    'right shoulder':(14, 17),
}

_BODYPART_RE = _re.compile(
    r'\b(?:right\s+|left\s+)?(?:arm|hand|wrist|elbow|head|neck|'
    r'leg|knee|shoulder)s?\b',
    _re.IGNORECASE,
)

# Action verbs that imply joint activation rather than locomotion
_ACTIVATION_VERBS = _re.compile(
    r'\b(?:swing|swinging|wave|waving|rais(?:e|ing)|lift|lifting|'
    r'rotat(?:e|ing)|twist|twisting|bend|bending|stretch|stretching|'
    r'extend|extending|flex|flexing|curl|curling|shak(?:e|ing)|'
    r'nod|nodding|turn|turning)\b',
    _re.IGNORECASE,
)


def constraints_to_subgoals(
    caption: str,
    constraints: list,
) -> List[SubGoal]:
    """Convert ``ConstraintPhase`` objects into ``SubGoal`` objects.

    This adapter handles three cases that raw regex parsing cannot:

    1. **Concurrent actions** — clauses containing "while", "as he is",
       "during", etc. produce ``span='global'`` SubGoals that are
       evaluated over the entire motion rather than a single phase.

    2. **Joint activation from body-part mentions** — phrases like
       "swing your arm" are converted into ``joint_activation`` SubGoals
       with the appropriate (parent, child) joint pair.

    3. **Direction passthrough** — the ``Direction`` enum from
       ``grpo_reward`` is mapped to plain strings for ``SubGoal``.

    Args:
        caption:     the original text caption (needed for concurrent /
                     body-part detection).
        constraints: list of ``ConstraintPhase`` from
                     :func:`grpo_reward.parse_numerical_constraints`.

    Returns:
        List of :class:`SubGoal` ready for :func:`evaluate_compositional`.
    """
    text = caption.lower()

    # --- detect which temporal clauses are concurrent ---
    # Re-split the caption the same way grpo_reward does
    from grpo_reward import _TEMPORAL_SPLIT
    clause_spans: List[Tuple[int, int]] = []
    prev = 0
    for m in _TEMPORAL_SPLIT.finditer(text):
        if m.start() > prev:
            clause_spans.append((prev, m.start()))
        prev = m.end()
    if prev < len(text):
        clause_spans.append((prev, len(text)))
    if not clause_spans:
        clause_spans = [(0, len(text))]

    concurrent_orders: set = set()
    for order, (cs, ce) in enumerate(clause_spans):
        clause = text[cs:ce]
        if _CONCURRENT_PATTERNS.search(clause):
            concurrent_orders.add(order)

    # --- detect body-part activation mentions per clause ---
    clause_activations: Dict[int, Dict[str, Any]] = {}
    for order, (cs, ce) in enumerate(clause_spans):
        clause = text[cs:ce]
        if not _ACTIVATION_VERBS.search(clause):
            continue
        bp_match = _BODYPART_RE.search(clause)
        if bp_match:
            bp = bp_match.group(0).lower().rstrip('s')
            if bp in _BODYPART_JOINTS:
                parent, child = _BODYPART_JOINTS[bp]
                clause_activations[order] = {
                    'joint_a': parent,
                    'joint_b': child,
                    'min_angle_deg': 25.0,
                }

    # --- convert each ConstraintPhase → SubGoal ---
    goals: List[SubGoal] = []
    seen_orders: set = set()

    for c in constraints:
        direction = c.direction.value if hasattr(c.direction, 'value') else str(c.direction)
        span = 'global' if c.order in concurrent_orders else 'phase'

        if c.type == 'steps':
            goals.append(SubGoal(
                direction=direction,
                target_steps=c.value,
                span=span,
            ))
        elif c.type == 'degrees':
            signed_deg = c.value
            if direction == 'right':
                signed_deg = -c.value
            goals.append(SubGoal(
                direction=direction,
                target_rotation_deg=signed_deg,
                span=span,
            ))
        elif c.type == 'repetitions':
            goals.append(SubGoal(
                direction=direction,
                target_steps=c.value,
                span=span,
            ))

        seen_orders.add(c.order)

    # --- add joint-activation SubGoals for clauses with body-part verbs
    #     that didn't already produce a numeric constraint ---
    for order, ja in clause_activations.items():
        span = 'global' if order in concurrent_orders else 'phase'
        # Determine direction from the clause
        cs, ce = clause_spans[order] if order < len(clause_spans) else (0, len(text))
        clause = text[cs:ce]
        direction = 'any'
        for pat_str, d in [
            ('left_forward', 'left_forward'),
            ('left-forward', 'left_forward'),
            ('left forward', 'left_forward'),
            ('right_forward', 'right_forward'),
            ('right-forward', 'right_forward'),
            ('right forward', 'right_forward'),
            ('left_backward', 'left_backward'),
            ('left-backward', 'left_backward'),
            ('left backward', 'left_backward'),
            ('right_backward', 'right_backward'),
            ('right-backward', 'right_backward'),
            ('right backward', 'right_backward'),
            ('right', 'right'),
            ('left', 'left'),
            ('forward', 'forward'),
            ('backward', 'backward'),
        ]:
            if pat_str in clause:
                direction = d
                break

        goals.append(SubGoal(
            direction=direction,
            joint_activation=ja,
            span=span,
        ))

    return goals
