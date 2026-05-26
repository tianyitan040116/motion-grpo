"""
GRPO Reward Module for Motion Generation

Reward Components:
1. Text-Motion Matching Score: Cosine similarity between text and motion embeddings
2. Physical Plausibility: Foot skating penalty + motion smoothness
3. Numerical Accuracy: Direction-aware step counting, signed rotation,
   temporal phase segmentation, and ordered constraint matching
"""

import json
import re
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
from utils.motion_utils import recover_from_ric, recover_root_rot_pos
from spatiotemporal_reward import (
    constraints_to_subgoals,
    evaluate_compositional,
)
from motion_constraint_executor import (
    MotionConstraintExecutor,
    aggregate_executor_score,
)
from motion_step_detector import detect_steps


# ---------------------------------------------------------------------------
# Data structures for phase-aware analysis
# ---------------------------------------------------------------------------

class Direction(Enum):
    FORWARD = 'forward'
    BACKWARD = 'backward'
    LEFT = 'left'
    RIGHT = 'right'
    LEFT_FORWARD = 'left_forward'
    RIGHT_FORWARD = 'right_forward'
    LEFT_BACKWARD = 'left_backward'
    RIGHT_BACKWARD = 'right_backward'
    ANY = 'any'  # no direction specified in caption

@dataclass
class ConstraintPhase:
    """A parsed constraint from the caption with direction and temporal order.

    `value` is the representative (center) target. When the caption is vague
    ("a few steps", "a circle"), `value_min` / `value_max` bracket the
    plausible range and the scoring functions give full credit anywhere
    inside it. When the caption is precise ("3 steps", "180 degrees"), both
    bounds are None and the legacy point-target Gaussian is used.
    For `degrees`, a negative `value` means clockwise; positive = CCW.
    """
    type: str           # 'steps', 'degrees', 'repetitions'
    value: float        # numeric target (signed for degrees)
    direction: Direction
    order: int          # temporal position (0-based)
    raw: str            # original matched text
    value_min: Optional[float] = None
    value_max: Optional[float] = None

@dataclass
class MotionPhase:
    """A detected phase of motion from trajectory analysis."""
    start_frame: int
    end_frame: int
    direction: Direction
    step_count: int
    displacement: float   # meters (XZ plane)
    rotation_deg: float   # signed degrees (positive = left/CCW)
    purity: float = 1.0   # direction purity in [0,1]: cos(actual, ideal_dir)


@dataclass
class ParsedCaptionPhase:
    """Structured phase extracted from caption text."""
    order: int
    action: str = 'move'
    direction: Direction = Direction.ANY
    steps: Optional[float] = None
    degrees: Optional[float] = None
    repetitions: Optional[float] = None
    stop: bool = False
    raw_direction: Optional[str] = None


@dataclass
class ParsedCaptionConstraints:
    """Unified parser output used by reward computation."""
    phases: List[ParsedCaptionPhase]
    numerical_constraints: List[ConstraintPhase]
    direction_sequence: List[Direction]
    source: str
    raw_response: str = ''


# ---------------------------------------------------------------------------
# Numerical constraint parser
# ---------------------------------------------------------------------------

# Maps English words to numbers
_WORD2NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'a couple': 2, 'a few': 3, 'several': 4, 'half': 0.5,
}

_WORD_NUM = r'one|two|three|four|five|six|seven|eight|nine|ten'

_NUM_PATTERNS = [
    # sidesteps (must come before generic steps)
    (r'(\d+)\s+(?:side\s*steps?|sidesteps?)', 'steps'),
    (rf'({_WORD_NUM})\s+(?:side\s*steps?|sidesteps?)', 'steps'),
    # "N steps" / "N step" (digits and words)
    (r'(\d+)\s+steps?', 'steps'),
    (rf'({_WORD_NUM})\s+steps?', 'steps'),
    # "N meters/metres" - treat as steps (adult stride ~ 0.7-0.8m)
    (r'(\d+)\s+met(?:er|re)s?', 'steps'),
    (rf'({_WORD_NUM})\s+met(?:er|re)s?', 'steps'),
    # "a couple/few steps"
    (r'a\s+couple(?:\s+of)?\s+steps?', 'steps_couple'),
    (r'a\s+few\s+steps?', 'steps_few'),
    (r'several\s+steps?', 'steps_several'),
    # "a step" / "a small step"
    (r'\ba\s+(?:small\s+|large\s+|big\s+)?steps?\b', 'steps_one'),
    # "N times"
    (r'(\d+)\s+times?', 'repetitions'),
    (r'(twice)', 'repetitions'),
    (rf'({_WORD_NUM})\s+times?', 'repetitions'),
    # "N degrees" -- but NOT "N degree(s) angle" (a posture, not a body rotation)
    (r'(\d+)\s*degrees?(?!\s+angle)', 'degrees'),
    # "quarter circle" -- must come before "circle"
    (r'(?:a\s+|the\s+)?quarter[-\s]+(?:of\s+a\s+)?circle', 'degrees_quarter_circle'),
    # "half circle" / "half the circle" -- must come before "circle"
    (r'half\s+(?:the\s+|a\s+)?circle', 'degrees_half_circle'),
    # generic "circle" (not preceded by half/quarter)
    (r'(?<!half\s)(?<!quarter\s)(?<!quarter-)(?:the\s+|a\s+)?circle', 'degrees_full_circle'),
]

# Temporal clause delimiters
_TEMPORAL_SPLIT = re.compile(
    r'\b(?:then|before|after|finally|next|afterwards|and\s+then)\b|[;.]'
)

# Direction patterns - searched within each clause near a numeric match
_DIRECTION_PATTERNS = [
    (re.compile(r'\b(?:left[-\s]?forward|forward[-\s]?left|front[-\s]?left|left[-\s]?front)\b'), Direction.LEFT_FORWARD),
    (re.compile(r'\b(?:right[-\s]?forward|forward[-\s]?right|front[-\s]?right|right[-\s]?front)\b'), Direction.RIGHT_FORWARD),
    (re.compile(r'\b(?:left[-\s]?backward|backward[-\s]?left|back[-\s]?left|left[-\s]?back)\b'), Direction.LEFT_BACKWARD),
    (re.compile(r'\b(?:right[-\s]?backward|backward[-\s]?right|back[-\s]?right|right[-\s]?back)\b'), Direction.RIGHT_BACKWARD),
    (re.compile(r'(?:to\s+(?:the\s+)?)?(?:his|her|their|the)\s+right|\bright\b'), Direction.RIGHT),
    (re.compile(r'(?:to\s+(?:the\s+)?)?(?:his|her|their|the)\s+left|\bleft\b'), Direction.LEFT),
    (re.compile(r'\b(?:forward|forwards|ahead)\b'), Direction.FORWARD),
    (re.compile(r'\b(?:backward|backwards)\b'), Direction.BACKWARD),
    (re.compile(r'\bback\b(?!\s+to)'), Direction.BACKWARD),  # "back" but not "back to"
]


# Spin direction patterns (orthogonal to translation direction).
# Sign convention determined empirically from HumanML3D root_rot_vel
# integration: positive cumulative yaw = clockwise (CW) / right turn.
# (The legacy docstring on _measure_rotation_signed claimed the opposite
# but disagreed with the data; spot-checking captioned CW/CCW circles
# confirmed CW measures positive.)
_SPIN_CW = re.compile(r'\b(?:clockwise|cw)\b')
_SPIN_CCW = re.compile(r'\b(?:counter[-\s]*clockwise|counterclockwise|anti[-\s]*clockwise|ccw)\b')


def _extract_spin_sign(text: str) -> Optional[int]:
    """Return +1 for CW (positive yaw), -1 for CCW, None if absent.

    CCW pattern checked first because 'counter clockwise' contains 'clockwise'.
    """
    if _SPIN_CCW.search(text):
        return -1
    if _SPIN_CW.search(text):
        return +1
    return None


def _extract_direction(text: str, match_start: int = -1, match_end: int = -1) -> Direction:
    """Extract movement direction from text, preferring context near the match.

    Window strategy (tightest first, to avoid bleed from adjacent clauses):
      1. text immediately following the match (next ~15 chars)
      2. text immediately preceding the match (prev ~15 chars)
      3. fall back to the whole text passed in (caller already restricts to clause)
    """
    if match_start >= 0 and match_end >= 0:
        # 1. forward window: "2 steps to the right"
        fwd_end = min(len(text), match_end + 15)
        fwd = text[match_end:fwd_end]
        for pat, direction in _DIRECTION_PATTERNS:
            if pat.search(fwd):
                return direction
        # 2. backward window: "right 2 steps"
        bwd_start = max(0, match_start - 15)
        bwd = text[bwd_start:match_start]
        for pat, direction in _DIRECTION_PATTERNS:
            if pat.search(bwd):
                return direction

    # 3. full text (already restricted to clause by caller)
    for pat, direction in _DIRECTION_PATTERNS:
        if pat.search(text):
            return direction
    return Direction.ANY


def parse_numerical_constraints(caption: str) -> List[ConstraintPhase]:
    """Extract numerical constraints with direction and temporal ordering.

    Splits caption into temporal clauses, then extracts numeric patterns
    with associated direction from each clause. Within a clause, duplicate
    (type, value, direction) triples are merged so "three times ... three
    times" in a non-temporal context produces one constraint, not two.
    Across clauses (separated by then/before/after/etc) duplicates are kept
    because they reflect distinct phases ("3 steps right, then 3 steps left").

    For `degrees`, value is signed: positive = CCW / left turn, negative =
    CW / right turn (matches HumanML3D root_rot_vel integration convention).

    Vague quantifiers ("a step", "a few", "a circle", "quarter circle")
    populate value_min / value_max on ConstraintPhase so downstream scoring
    can give full credit anywhere inside the range.
    """
    text = caption.lower()

    # Split into temporal clauses
    clause_spans = []
    prev_end = 0
    for m in _TEMPORAL_SPLIT.finditer(text):
        if m.start() > prev_end:
            clause_spans.append((prev_end, m.start()))
        prev_end = m.end()
    if prev_end < len(text):
        clause_spans.append((prev_end, len(text)))

    # If no delimiters found, treat entire caption as one clause
    if not clause_spans:
        clause_spans = [(0, len(text))]

    constraints = []
    covered = set()  # character positions already matched (global)

    for order, (c_start, c_end) in enumerate(clause_spans):
        clause = text[c_start:c_end]
        spin_sign = _extract_spin_sign(clause)
        # de-dup within a single clause: a "3 steps" mentioned in one breath
        # twice is one constraint, not two.
        clause_seen: set = set()

        for pattern, ctype in _NUM_PATTERNS:
            for m in re.finditer(pattern, clause):
                # Convert to global positions for overlap check
                g_start = c_start + m.start()
                g_end = c_start + m.end()
                span_range = set(range(g_start, g_end))
                if span_range & covered:
                    continue
                covered |= span_range

                # Extract direction from local context around this match
                match_dir = _extract_direction(clause, m.start(), m.end())

                # Resolve value / range
                value_min: Optional[float] = None
                value_max: Optional[float] = None
                if ctype == 'steps_one':
                    value = 1.0
                    value_min, value_max = 1.0, 3.0
                    ctype = 'steps'
                elif ctype == 'steps_couple':
                    value = 2.0
                    value_min, value_max = 2.0, 4.0
                    ctype = 'steps'
                elif ctype == 'steps_few':
                    value = 3.0
                    value_min, value_max = 2.0, 5.0
                    ctype = 'steps'
                elif ctype == 'steps_several':
                    value = 4.0
                    value_min, value_max = 3.0, 6.0
                    ctype = 'steps'
                elif ctype == 'degrees_quarter_circle':
                    value = 90.0
                    value_min, value_max = 45.0, 135.0
                    ctype = 'degrees'
                elif ctype == 'degrees_half_circle':
                    value = 180.0
                    value_min, value_max = 120.0, 240.0
                    ctype = 'degrees'
                elif ctype == 'degrees_full_circle':
                    value = 360.0
                    # Human-walked "circles" in HumanML3D rarely close fully;
                    # empirically GT circles measure 180-540 deg (median ~250).
                    value_min, value_max = 180.0, 540.0
                    ctype = 'degrees'
                else:
                    raw = m.group(1) if m.lastindex else m.group(0)
                    if raw == 'twice':
                        value = 2.0
                    elif raw in _WORD2NUM:
                        value = float(_WORD2NUM[raw])
                    else:
                        try:
                            value = float(raw)
                        except ValueError:
                            continue

                # Apply spin sign to degrees if caption named clockwise/CCW.
                # Otherwise leave unsigned (value > 0); downstream code that
                # cares about sign will use direction (LEFT/RIGHT) if present.
                if ctype == 'degrees' and spin_sign is not None:
                    value = abs(value) * spin_sign
                    if value_min is not None and value_max is not None:
                        lo, hi = abs(value_min), abs(value_max)
                        if spin_sign < 0:
                            value_min, value_max = -hi, -lo
                        else:
                            value_min, value_max = lo, hi

                # Within-clause dedupe key: same target + same direction =
                # same constraint, even if the literal raw text differs.
                # Note: spin sign already baked into value for degrees.
                key = (ctype, value, match_dir)
                if key in clause_seen:
                    continue
                clause_seen.add(key)

                constraints.append(ConstraintPhase(
                    type=ctype,
                    value=value,
                    direction=match_dir,
                    order=order,
                    raw=m.group(0),
                    value_min=value_min,
                    value_max=value_max,
                ))

    return constraints


# ---------------------------------------------------------------------------
# Motion feature extraction (operates on denormalized motion)
# ---------------------------------------------------------------------------


def denormalize_motion(motion, mean, std):
    """Denormalize HumanML3D-style 263-dim motion. Thin wrapper over
    `motion * std + mean` so the reward path and audit/smoke scripts go
    through one definition.

    Note: HumanML3D's normalized "zero motion" decodes to an "average walking
    person" because MEAN encodes a non-zero forward velocity. That is data
    semantics, not a bug -- a truly stationary motion in normalized space is
    `(-MEAN/STD)`, not 0. Tests that need a real static pose should build
    it explicitly.
    """
    if isinstance(motion, torch.Tensor):
        if not isinstance(mean, torch.Tensor):
            mean_t = torch.as_tensor(mean, dtype=motion.dtype, device=motion.device)
            std_t = torch.as_tensor(std, dtype=motion.dtype, device=motion.device)
        else:
            mean_t, std_t = mean, std
        return motion * std_t + mean_t
    return motion * std + mean


def derive_foot_contact_from_joints(
    joints: torch.Tensor,
    height_threshold: float = 0.10,
    speed_threshold: float = 0.030,
    locomotion_path_threshold: float = 0.15,
) -> torch.Tensor:
    """Recover a [T, 4] binary foot-contact mask from recovered joint xyz.

    HumanML3D's foot_contact channels (cols 259:263 of the 263-dim feature)
    are binary {0,1} in raw space, but VQ-VAE decode does NOT preserve
    that binary semantics -- decoded values for those channels saturate
    above 1.0 and are useless for step detection. Joint positions, in
    contrast, are the primary VQ-VAE target and reconstruct cleanly.

    A foot is "in contact" when its ankle joint is close to the ground AND
    moving slowly. Thresholds were tuned against VQ-VAE-reconstructed real
    HumanML3D walks (loose enough to recover most steps after the decoder
    smooths the trajectory, tight enough to avoid false steps from in-place
    actions like clapping or waving).

    To avoid false-positive steps on non-locomotion clips, we also require
    the clip's total root XZ displacement to exceed `locomotion_path_threshold`.
    Below that, the body isn't actually moving and any "steps" picked up by
    the rising-edge detector would be hallucinated from joint jitter.

    Output is laid out [left_heel, left_toe, right_heel, right_toe] like
    the original channels, so downstream code (move_state / hybrid step
    detectors, foot-skating score, phase analyzer) works unchanged.
    """
    T = int(joints.shape[0])
    if T < 2:
        return torch.zeros(T, 4, device=joints.device, dtype=joints.dtype)

    # Locomotion gate: if root barely moves, return all-contact (no steps).
    root_xz = joints[:, 0, [0, 2]]
    root_path = torch.norm(root_xz[1:] - root_xz[:-1], dim=-1).sum().item()
    if root_path < locomotion_path_threshold:
        return torch.ones(T, 4, device=joints.device, dtype=joints.dtype)

    # ankle indices: 7=l_ankle, 8=r_ankle (more stable than foot joints 10/11
    # across VQ-VAE reconstructions).
    l_ankle = joints[:, 7]
    r_ankle = joints[:, 8]

    # Ground reference: 5th percentile of either ankle's y over the clip.
    y_pool = torch.cat([l_ankle[:, 1], r_ankle[:, 1]])
    ground_y = torch.quantile(y_pool, 0.05)

    def _contact_mask(joint_pos):
        height = joint_pos[:, 1] - ground_y
        delta_xz = joint_pos[1:, [0, 2]] - joint_pos[:-1, [0, 2]]
        speed = torch.norm(delta_xz, dim=-1)
        speed = torch.cat([speed.new_zeros(1), speed])
        return ((height < height_threshold) & (speed < speed_threshold)).float()

    l_contact = _contact_mask(l_ankle)
    r_contact = _contact_mask(r_ankle)
    # Heel + toe channels duplicate the per-foot mask so downstream code
    # expecting [lh, lt, rh, rt] sees consistent binary signals.
    return torch.stack([l_contact, l_contact, r_contact, r_contact], dim=-1)


def _count_steps_in_range(
    foot_contact: torch.Tensor,
    start: int = 0,
    end: int = -1,
    joints: Optional[torch.Tensor] = None,
    *,
    detector: str = "move_state",
    **detector_kwargs,
) -> int:
    """Count steps within a frame range.

    Defaults to the move_state formulation: per foot,
    `move_state = (1 - foot_contact_channels) > 0.5`, then count rest->move
    rising edges. The legacy hybrid contact + height + speed + landing
    detector (which requires `joints`) is still available via
    `detector="hybrid"`.

    Args:
        foot_contact: [T, 4] foot contact channels in order
            (left_heel, left_toe, right_heel, right_toe).
        start: start frame (inclusive).
        end: end frame (exclusive), -1 means T.
        joints: optional [T, 22, 3] world-space joint positions. Required
            when `detector="hybrid"`; ignored for the default move_state path.
        detector: "move_state" (default) or "hybrid".
        **detector_kwargs: forwarded to the underlying detector.

    Returns:
        Number of detected steps.
    """
    if end == -1:
        end = foot_contact.shape[0]
    if detector == "hybrid":
        if joints is None:
            raise ValueError("detector='hybrid' requires `joints`.")
        return detect_steps(
            joints, foot_contact, start=start, end=end, detector="hybrid",
            **detector_kwargs,
        ).count
    return detect_steps(
        joints, foot_contact, start=start, end=end, detector="move_state",
        **detector_kwargs,
    ).count


# Keep old interface for backward compatibility
def _count_steps(foot_contact: torch.Tensor) -> int:
    return _count_steps_in_range(foot_contact)


def _measure_rotation_signed(
    root_rot_vel: torch.Tensor,
    start: int = 0,
    end: int = -1,
) -> float:
    """Measure signed rotation in degrees. Positive = left/CCW, negative = right/CW.

    Note: HumanML3D uses half-angle quaternion convention, so actual rotation
    is 2x the cumulative rot_vel.

    Args:
        root_rot_vel: [T] root Y-axis rotation velocity (denormalized)
        start: start frame (inclusive)
        end: end frame (exclusive), -1 means T
    """
    if end == -1:
        end = root_rot_vel.shape[0]
    total_rad = root_rot_vel[start:end].sum().item()
    return total_rad * 2.0 * (180.0 / np.pi)  # 2x for half-angle convention


# Keep old interface for backward compatibility
def _measure_rotation(root_rot_vel: torch.Tensor) -> float:
    return abs(_measure_rotation_signed(root_rot_vel))


def _count_repetitions(root_y: torch.Tensor, threshold: float = 0.03) -> int:
    """Count repetitive vertical events (jumps, squats, etc.)

    Detects peaks in root Y position that rise above a threshold
    relative to a running baseline.

    Args:
        root_y: [T] root Y position (denormalized)
        threshold: minimum height delta to count as event

    Returns:
        Number of detected repetitions.
    """
    y = root_y.cpu().numpy()
    baseline = np.median(y)

    # Find peaks above baseline
    above = y > (baseline + threshold)
    count = 0
    in_peak = False
    for v in above:
        if v and not in_peak:
            count += 1
            in_peak = True
        elif not v:
            in_peak = False

    return count


def _foot_skating_score(
    joint_positions: torch.Tensor,
    foot_contact: torch.Tensor,
    fps: float = 20.0,
) -> float:
    """Compute foot skating penalty.

    When a foot is in ground contact, its velocity should be near zero.
    Returns a score in [0, 1] where 1 = no skating.

    Args:
        joint_positions: [T, J, 3] absolute joint positions
        foot_contact: [T, 4] binary contact labels
        fps: frames per second of the motion data

    Returns:
        Score in [0, 1], higher is better.
    """
    T = joint_positions.shape[0]
    if T < 2:
        return 1.0

    # Joint indices: 10 = left foot, 11 = right foot (t2m skeleton)
    left_foot_pos = joint_positions[:, 10, [0, 2]]   # [T, 2] XZ
    right_foot_pos = joint_positions[:, 11, [0, 2]]   # [T, 2] XZ

    # Velocities (m/frame)
    left_vel = torch.norm(left_foot_pos[1:] - left_foot_pos[:-1], dim=-1)   # [T-1]
    right_vel = torch.norm(right_foot_pos[1:] - right_foot_pos[:-1], dim=-1)  # [T-1]

    # Contact masks (use t-1 to align with velocity)
    left_contact = ((foot_contact[:-1, 0] + foot_contact[:-1, 1]) > 0.5).float()
    right_contact = ((foot_contact[:-1, 2] + foot_contact[:-1, 3]) > 0.5).float()

    # Skating = velocity during contact
    left_skating = (left_vel * left_contact).sum()
    right_skating = (right_vel * right_contact).sum()
    contact_frames = left_contact.sum() + right_contact.sum()

    if contact_frames < 1:
        return 1.0

    avg_skating = (left_skating + right_skating) / contact_frames
    # Convert to score: skating of 0 -> score 1, skating of 0.05+ -> score ~0
    score = torch.exp(-avg_skating * fps * 5.0).item()
    return float(np.clip(score, 0.0, 1.0))


def _smoothness_score(motion: torch.Tensor) -> float:
    """Compute motion smoothness score based on jerk (derivative of acceleration).

    Lower jerk = smoother motion = higher score.

    Args:
        motion: [T, 263] normalized motion

    Returns:
        Score in [0, 1], higher is better.
    """
    T = motion.shape[0]
    if T < 4:
        return 1.0

    # Use global velocity [256:259] for smoothness measurement
    vel = motion[:, 256:259]  # [T, 3]

    # Acceleration
    acc = vel[1:] - vel[:-1]  # [T-1, 3]

    # Jerk
    jerk = acc[1:] - acc[:-1]  # [T-2, 3]

    # Mean jerk magnitude
    jerk_mag = torch.norm(jerk, dim=-1).mean().item()

    # Convert to score: jerk of 0 -> 1, high jerk -> 0
    # Empirical scale: typical jerk in normalized space ~0.01-0.1
    score = np.exp(-jerk_mag * 20.0)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Stillness detection (anti-exploit for GRPO)
# ---------------------------------------------------------------------------


def _stillness_score(joint_pos: torch.Tensor, min_displacement: float = 0.3) -> float:
    """Return 0.0 if motion is broken (frozen / drift / implausibly fast),
    1.0 if it is plausible animated motion.

    Rejects two failure modes:
      (a) Frozen / drift: body never articulates, root either doesn't move
          or only slides at a single constant velocity (typical of MEAN-
          driven decode of zeros or repeated VQ tokens).
      (b) Random noise: per-frame body-relative joint speeds far exceed any
          plausible human value (real walks peak ~0.03 m/frame; Gaussian
          noise easily hits 0.3+). Without this clamp, cumulative directional
          displacement signals leak high reward to gibberish motion.

    The score is `(in-bounds activity) * (physical plausibility)` so a
    motion that articulates believably AND moves the root coherently
    scores ~1.0; frozen, drift, or jittery garbage scores ~0.

    Args:
        joint_pos: [T, 22, 3] joint positions, world frame
        min_displacement: root XZ path threshold for locomotion path
    """
    T = joint_pos.shape[0]
    if T < 2:
        return 0.0

    # --- Root-relative joint velocity (in-place articulation) ---
    rel_pos = joint_pos[:, 1:] - joint_pos[:, 0:1]
    rel_vel = torch.norm(rel_pos[1:] - rel_pos[:-1], dim=-1)  # [T-1, 21]
    mean_rel_speed = rel_vel.mean().item()
    # HumanML3D real walks have mean_rel_speed ~0.001-0.005; frozen ~1e-5.
    # Saturation point raised from 0.0005 -> 0.002 after run1 collapse: at
    # 0.0005 any tiny in-place wiggle scored 1.0, letting the model drift
    # into "stand and twitch" while collecting full physical reward.
    rel_score = float(np.clip(mean_rel_speed / 0.002, 0.0, 1.0))

    # --- Joint velocity variance (anti constant-speed drift) ---
    joint_vel = torch.norm(joint_pos[1:] - joint_pos[:-1], dim=-1)
    vel_std = joint_vel.std().item()
    var_score = float(np.clip(vel_std / 0.001, 0.0, 1.0))

    # --- Root displacement (locomotion) ---
    root_xz = joint_pos[:, 0, [0, 2]]
    root_vel = torch.norm(root_xz[1:] - root_xz[:-1], dim=-1)
    total_path = root_vel.sum().item()
    root_vel_std = root_vel.std().item()
    # A real walk has stride-induced root-speed variation (std ~3e-4 to 1e-3
    # for HumanML3D walks). A constant-velocity drift from MEAN-decoded
    # zeros has std ~2e-5. Require both meaningful path AND speed variation.
    root_path_score = float(np.clip(total_path / min_displacement, 0.0, 1.0))
    root_var_score = float(np.clip(root_vel_std / 0.0002, 0.0, 1.0))
    root_score = root_path_score * root_var_score

    # --- Physical plausibility upper bound ---
    # Real human body-relative joint speed peaks ~0.03 m/frame (sprint).
    # Gaussian-noise decode produces 0.3+; clearly impossible.
    max_rel_speed = rel_vel.max().item()
    if max_rel_speed < 0.10:
        plausibility = 1.0
    else:
        plausibility = max(0.0, 1.0 - (max_rel_speed - 0.10) / 0.20)

    body_score = max(rel_score, var_score)
    base = max(body_score, root_score)
    return float(base * plausibility)


# ---------------------------------------------------------------------------
# Motion-energy gate (P0.C, post-run1 collapse fix)
#
# Background: run1 collapsed because for direction-only and pure-caption
# prompts (70% of the prompt mix) the reward composition fell back to
# matching-only. Combined with a too-easy stillness saturation, the model
# learned a "stand still + wiggle" attractor that scored full physical +
# good matching without doing the requested action.
#
# Fix: require that the generated motion expend at least a small minimum
# amount of energy relevant to the prompt verb -- root path for locomotion,
# yaw rotation for spin/turn, root-y range for jump/sit/stand, body-relative
# speed for everything else. If the gate fails, the final reward is scaled
# down so the gradient pushes back toward action-executing samples even
# when matching and stillness happen to score well.
# ---------------------------------------------------------------------------

_RE_LOCOMOTION = re.compile(
    r'\b(walk|run|jog|step|stride|march|hike|crawl|sprint|skip|sidestep|backward|forward)',
    re.IGNORECASE,
)
_RE_ROTATION = re.compile(
    r'\b(turn|spin|rotate|pivot|twist|swivel|whirl|circle)',
    re.IGNORECASE,
)
_RE_VERTICAL = re.compile(
    r'\b(jump|hop|leap|sit|stand|kneel|squat|crouch|stoop|duck|bend|bow)',
    re.IGNORECASE,
)
_RE_GENERIC_MOTION = re.compile(
    r'\b(kick|punch|throw|wave|stretch|reach|swing|clap|dance|shake|nod|grab|raise|lift|lower|drop|swim)',
    re.IGNORECASE,
)


def _compute_motion_energy(
    joint_pos: torch.Tensor,
    motion_raw: torch.Tensor,
) -> Dict[str, float]:
    """Measure actual motion energy on four axes.

    Returns dict {path_m, yaw_rad, y_range_m, rel_speed} -- the same keys
    used by `_required_minimum_energy` so the gate can compare per-axis.
    """
    T = joint_pos.shape[0]
    if T < 2:
        return {'path_m': 0.0, 'yaw_rad': 0.0, 'y_range_m': 0.0, 'rel_speed': 0.0}

    root_xz = joint_pos[:, 0, [0, 2]]
    path_m = float(torch.norm(root_xz[1:] - root_xz[:-1], dim=-1).sum().item())

    # motion_raw[:, 0] is root angular velocity around Y; sum -> net yaw (rad)
    yaw_rad = float(abs(motion_raw[:, 0].sum().item()))

    root_y = joint_pos[:, 0, 1]
    y_range_m = float((root_y.max() - root_y.min()).item())

    rel_pos = joint_pos[:, 1:] - joint_pos[:, 0:1]
    rel_vel = torch.norm(rel_pos[1:] - rel_pos[:-1], dim=-1)
    rel_speed = float(rel_vel.mean().item())

    return {
        'path_m': path_m,
        'yaw_rad': yaw_rad,
        'y_range_m': y_range_m,
        'rel_speed': rel_speed,
    }


def _required_minimum_energy(
    caption: str,
    parsed: Optional['ParsedCaptionConstraints'],
) -> Dict[str, float]:
    """Translate caption + parsed constraints into minimum energy expected
    on each axis. Thresholds are deliberately conservative -- they catch
    "did nothing" without double-penalizing models that mostly got it
    right (numerical_score already grades closeness).
    """
    req = {'path_m': 0.0, 'yaw_rad': 0.0, 'y_range_m': 0.0, 'rel_speed': 0.0}

    if parsed is not None:
        for c in parsed.numerical_constraints:
            if c.type == 'steps':
                # Adult step ~0.7m, but VQ-VAE round-trip degrades root path
                # by 40-60% (audit on real GT: 3-step prompts gate at 0.66
                # with 0.30 m/step floor). Drop to 0.20 m/step so a real
                # walking GT clears the floor while a fully-stationary
                # collapse (run1 walk3m at 0.32 m) still gets gate ~0.40.
                req['path_m'] = max(req['path_m'], 0.20 * float(c.value))
            elif c.type == 'degrees':
                # Require 40% of the requested rotation (in radians).
                req['yaw_rad'] = max(req['yaw_rad'],
                                     0.40 * abs(float(c.value)) * np.pi / 180.0)
            elif c.type == 'repetitions':
                # Jumps / cycles: require visible vertical movement.
                req['y_range_m'] = max(req['y_range_m'], 0.06)

    # Verb-fallback floors (active even with empty parsed constraints).
    if _RE_LOCOMOTION.search(caption):
        req['path_m'] = max(req['path_m'], 0.20)
    if _RE_ROTATION.search(caption):
        req['yaw_rad'] = max(req['yaw_rad'], 0.50)
    if _RE_VERTICAL.search(caption):
        req['y_range_m'] = max(req['y_range_m'], 0.10)
    if _RE_GENERIC_MOTION.search(caption):
        req['rel_speed'] = max(req['rel_speed'], 0.0015)

    return req


def motion_energy_gate(
    actual: Dict[str, float],
    required: Dict[str, float],
    floor: float = 0.25,
) -> float:
    """Return value in [floor, 1.0] -- multiplicative factor on the reward.

    For each required axis (>0), compute satisfaction ratio in [0, 1].
    Take the MIN over axes (one missing axis is enough to fail). Lerp from
    `floor` (full miss) to 1.0 (full satisfy) so the gradient stays alive
    instead of cliffing to zero.

    Floor 0.25 means a fully-frozen sample sees its reward weighted at 25%,
    while a perfectly-executed sample sees 100%. The training signal in
    between is monotonic.
    """
    active = [k for k, v in required.items() if v > 0]
    if not active:
        return 1.0
    min_ratio = 1.0
    for k in active:
        ratio = actual.get(k, 0.0) / max(required[k], 1e-6)
        ratio = min(1.0, max(0.0, ratio))
        if ratio < min_ratio:
            min_ratio = ratio
    return floor + (1.0 - floor) * min_ratio


# ---------------------------------------------------------------------------
# Phase-aware motion analysis
# ---------------------------------------------------------------------------

def _normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _classify_direction(move_angle_rad: float, initial_facing_rad: float) -> Direction:
    """Classify movement direction relative to character's initial facing.

    In HumanML3D after recover_root_rot_pos: +Z is forward, +X is right.
    atan2(dz, dx) gives the movement angle in the XZ plane.
    """
    relative = _normalize_angle(move_angle_rad - initial_facing_rad)
    deg = np.degrees(relative)
    if -22.5 <= deg <= 22.5:
        return Direction.FORWARD
    elif 22.5 < deg <= 67.5:
        return Direction.LEFT_FORWARD
    elif 67.5 < deg <= 112.5:
        return Direction.LEFT
    elif 112.5 < deg <= 157.5:
        return Direction.LEFT_BACKWARD
    elif -67.5 <= deg < -22.5:
        return Direction.RIGHT_FORWARD
    elif -112.5 <= deg < -67.5:
        return Direction.RIGHT
    elif -157.5 <= deg < -112.5:
        return Direction.RIGHT_BACKWARD
    else:
        return Direction.BACKWARD


# Ideal direction angles (relative to initial facing), in radians
# In the (dx, dz) convention where atan2(dz, dx):
#   FORWARD  = facing        (relative angle 0)
#   LEFT     = facing + 90 deg
#   RIGHT    = facing - 90 deg
#   BACKWARD = facing + 180 deg
_IDEAL_RELATIVE_ANGLE = {
    Direction.FORWARD: 0.0,
    Direction.LEFT: np.pi / 2,
    Direction.RIGHT: -np.pi / 2,
    Direction.BACKWARD: np.pi,
    Direction.LEFT_FORWARD: np.pi / 4,
    Direction.RIGHT_FORWARD: -np.pi / 4,
    Direction.LEFT_BACKWARD: 3 * np.pi / 4,
    Direction.RIGHT_BACKWARD: -3 * np.pi / 4,
}


def _direction_purity(move_angle_rad: float, direction: Direction,
                      initial_facing_rad: float) -> float:
    """Cosine similarity between actual movement and ideal direction axis.

    Returns value in [0, 1]:
      - 1.0 = perfectly aligned
      - 0.71 = 45 deg off
      - 0.5 = 60 deg off
      - 0.0 = 90 deg or more off (or Direction.ANY)
    """
    if direction == Direction.ANY or direction not in _IDEAL_RELATIVE_ANGLE:
        return 1.0
    ideal = initial_facing_rad + _IDEAL_RELATIVE_ANGLE[direction]
    cos_sim = np.cos(_normalize_angle(move_angle_rad - ideal))
    return float(max(0.0, cos_sim))


def _purity_factor(purity: float, threshold: float = 0.6,
                   floor: float = 0.7) -> float:
    """Gentle scoring factor based on direction purity.

    Purity >= threshold (0.6, ~53 deg off): no penalty, factor = 1.0
    Purity = 0.0 (90 deg+ off or opposite): factor = floor (0.7)
    Linear interpolation in between.
    """
    if purity >= threshold:
        return 1.0
    return floor + (1.0 - floor) * (purity / threshold)


def analyze_motion_phases(
    motion_raw: torch.Tensor,
    foot_contact: torch.Tensor,
    joints: Optional[torch.Tensor] = None,
    min_phase_frames: int = 8,
    direction_change_threshold: float = 0.6,
) -> List[MotionPhase]:
    """Segment motion into directional phases.

    Args:
        motion_raw: [T, 263] denormalized motion
        foot_contact: [T, 4] binary foot contact
        joints: optional [T, 22, 3] positions for hybrid step detection
        min_phase_frames: minimum frames per phase (0.4s at 20fps)
        direction_change_threshold: radians (~35 degrees) to trigger new phase

    Returns:
        List of MotionPhase objects in temporal order.
    """
    T = motion_raw.shape[0]
    if T < min_phase_frames:
        return [MotionPhase(
            start_frame=0, end_frame=T, direction=Direction.ANY,
            step_count=_count_steps_in_range(foot_contact, 0, T, joints=joints),
            displacement=0.0, rotation_deg=0.0,
        )]

    # Recover global root trajectory
    r_rot_quat, r_pos = recover_root_rot_pos(motion_raw.unsqueeze(0))
    root_traj = r_pos.squeeze(0)  # [T, 3]
    global_xz = root_traj[:, [0, 2]]  # [T, 2] - X (right), Z (forward)

    # Initial facing angle (from first frame's rotation quaternion)
    # r_rot_quat: [1, T, 4], quaternion encodes Y-axis rotation
    # At frame 0, rotation angle is 0 -> facing +Z -> atan2(1, 0) = pi/2
    initial_facing = np.pi / 2  # +Z direction in atan2(z, x) convention

    # Smoothed movement direction
    window = min(5, T // 4)
    if window < 2:
        window = 2
    disp = global_xz[window:] - global_xz[:-window]  # [T-window, 2]
    speed = torch.norm(disp, dim=-1)  # [T-window]
    speed_threshold = 0.005 * window

    # Movement angles: atan2(dz, dx) to match XZ convention
    angles = torch.atan2(disp[:, 1], disp[:, 0])  # [T-window]
    moving = speed > speed_threshold

    # Detect phase boundaries using cumulative angle change over a short window
    boundaries = []
    last_boundary = 0
    lookback = max(3, min_phase_frames // 2)  # compare direction over a few frames
    for t in range(lookback, len(angles)):
        if t - last_boundary < min_phase_frames:
            continue
        if not moving[t].item():
            # Stationary -> potential boundary
            boundaries.append(t + window // 2)
            last_boundary = t
            continue
        # Find the last moving frame at least `lookback` frames ago
        ref = max(last_boundary, t - lookback)
        if not moving[ref].item():
            continue
        angle_diff = abs(_normalize_angle(
            (angles[t] - angles[ref]).item()
        ))
        if angle_diff > direction_change_threshold:
            # Place boundary at the midpoint of the transition
            boundaries.append((ref + t) // 2 + window // 2)
            last_boundary = t

    # Build phase list from boundaries
    boundary_frames = [0] + [min(b, T) for b in boundaries] + [T]
    # Remove duplicates and sort
    boundary_frames = sorted(set(boundary_frames))

    phases = []
    root_rot_vel = motion_raw[:, 0]

    for i in range(len(boundary_frames) - 1):
        sf = boundary_frames[i]
        ef = boundary_frames[i + 1]
        if ef - sf < 2:
            continue

        # Phase displacement
        phase_disp = global_xz[min(ef - 1, T - 1)] - global_xz[sf]
        disp_mag = torch.norm(phase_disp).item()

        # Phase direction + purity
        if disp_mag > 0.05:
            move_angle = np.arctan2(phase_disp[1].item(), phase_disp[0].item())
            phase_dir = _classify_direction(move_angle, initial_facing)
            purity = _direction_purity(move_angle, phase_dir, initial_facing)
        else:
            phase_dir = Direction.ANY
            purity = 1.0

        phases.append(MotionPhase(
            start_frame=sf,
            end_frame=ef,
            direction=phase_dir,
            step_count=_count_steps_in_range(foot_contact, sf, ef, joints=joints),
            displacement=disp_mag,
            rotation_deg=_measure_rotation_signed(root_rot_vel, sf, ef),
            purity=purity,
        ))

    # Merge tiny phases into neighbors
    merged = []
    for p in phases:
        if merged and (p.end_frame - p.start_frame) < min_phase_frames:
            # Merge into previous phase - weight purity by displacement
            prev = merged[-1]
            total_disp = prev.displacement + p.displacement
            if total_disp > 1e-6:
                merged_purity = (prev.purity * prev.displacement
                                 + p.purity * p.displacement) / total_disp
            else:
                merged_purity = prev.purity
            merged[-1] = MotionPhase(
                start_frame=prev.start_frame,
                end_frame=p.end_frame,
                direction=prev.direction,  # keep dominant phase's direction
                step_count=prev.step_count + p.step_count,
                displacement=total_disp,
                rotation_deg=prev.rotation_deg + p.rotation_deg,
                purity=merged_purity,
            )
        else:
            merged.append(p)

    return merged if merged else [MotionPhase(
        start_frame=0, end_frame=T, direction=Direction.ANY,
        step_count=_count_steps_in_range(foot_contact, 0, T, joints=joints),
        displacement=0.0, rotation_deg=0.0,
    )]


# ---------------------------------------------------------------------------
# Direction sequence matching (no numeric constraints needed)
# ---------------------------------------------------------------------------

# Direction patterns for direction-sequence matching (stricter than _DIRECTION_PATTERNS).
# Require a locomotion verb nearby to avoid matching body-part references
# like "right hand" or "left arm".
_MOTION_VERBS = r'(?:walk|walks|walking|run|runs|running|jog|jogs|jogging|step|steps|stepping|move|moves|moving|go|goes|going|turn|turns|turning|stumble|stumbles|stumbling|shuffle|shuffles|shuffling|slide|slides|sliding|sidestep|sidesteps|sidestepping|march|marches|marching|stride|strides|striding|lunge|lunges|lunging|hop|hops|hopping|jump|jumps|jumping|skip|skips|skipping|crawl|crawls|crawling)'

_DIR_SEQ_PATTERNS = [
    # "walks/steps/moves forward", "walks to the right", etc.
    # Allow a few words between verb and direction (e.g. "walks sideways to the left")
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:left[-\s]?forward|forward[-\s]?left|front[-\s]?left|left[-\s]?front)'), Direction.LEFT_FORWARD),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:right[-\s]?forward|forward[-\s]?right|front[-\s]?right|right[-\s]?front)'), Direction.RIGHT_FORWARD),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:left[-\s]?backward|backward[-\s]?left|back[-\s]?left|left[-\s]?back)'), Direction.LEFT_BACKWARD),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:right[-\s]?backward|backward[-\s]?right|back[-\s]?right|right[-\s]?back)'), Direction.RIGHT_BACKWARD),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:to\s+(?:the\s+)?)?(?:his|her|their|the\s+)?right'), Direction.RIGHT),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:to\s+(?:the\s+)?)?(?:his|her|their|the\s+)?left'), Direction.LEFT),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:forward|forwards|ahead)'), Direction.FORWARD),
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}(?:backward|backwards)'), Direction.BACKWARD),
    # "verb + back" but NOT "back to" (which means "return to")
    (re.compile(rf'{_MOTION_VERBS}\s+(?:\w+\s+){{0,3}}back(?!\s+to)\b'), Direction.BACKWARD),
    # standalone direction at clause start (after temporal split)
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)?(?:forward|forwards|ahead)\b'), Direction.FORWARD),
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)?(?:backward|backwards)\b'), Direction.BACKWARD),
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)?(?:left[-\s]?forward|forward[-\s]?left|front[-\s]?left|left[-\s]?front)\b'), Direction.LEFT_FORWARD),
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)?(?:right[-\s]?forward|forward[-\s]?right|front[-\s]?right|right[-\s]?front)\b'), Direction.RIGHT_FORWARD),
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)(?:right)\b'), Direction.RIGHT),
    (re.compile(r'(?:^|,\s*)\s*(?:to\s+(?:the\s+)?)(?:left)\b'), Direction.LEFT),
]


def _extract_direction_strict(clause: str) -> Direction:
    """Extract movement direction from a clause using strict verb+direction patterns.

    Unlike _extract_direction, this avoids matching body-part references
    like 'right hand' or 'left arm'.
    """
    for pat, direction in _DIR_SEQ_PATTERNS:
        if pat.search(clause):
            return direction
    return Direction.ANY


def parse_direction_sequence(caption: str) -> List[Direction]:
    """Extract ordered direction sequence from caption.

    Works on captions WITHOUT numeric constraints, e.g.:
      "a person walks forward then walks backward" -> [FORWARD, BACKWARD]
      "a person jogs to the left and then turns right" -> [LEFT, RIGHT]

    Uses strict verb+direction patterns to avoid false positives from
    body-part references ("raises right hand" should NOT match).

    Returns empty list if no directions found.
    """
    text = caption.lower()

    # Split into temporal clauses
    clause_spans = []
    prev_end = 0
    for m in _TEMPORAL_SPLIT.finditer(text):
        if m.start() > prev_end:
            clause_spans.append((prev_end, m.start()))
        prev_end = m.end()
    if prev_end < len(text):
        clause_spans.append((prev_end, len(text)))
    if not clause_spans:
        clause_spans = [(0, len(text))]

    directions = []
    for c_start, c_end in clause_spans:
        clause = text[c_start:c_end]
        d = _extract_direction_strict(clause)
        if d != Direction.ANY:
            directions.append(d)

    return directions


_DIRECTION_ALIASES: Dict[str, Direction] = {
    'forward': Direction.FORWARD,
    'forwards': Direction.FORWARD,
    'front': Direction.FORWARD,
    'ahead': Direction.FORWARD,
    'backward': Direction.BACKWARD,
    'backwards': Direction.BACKWARD,
    'back': Direction.BACKWARD,
    'rear': Direction.BACKWARD,
    'left': Direction.LEFT,
    'right': Direction.RIGHT,
    'left_forward': Direction.LEFT_FORWARD,
    'left-forward': Direction.LEFT_FORWARD,
    'left forward': Direction.LEFT_FORWARD,
    'forward_left': Direction.LEFT_FORWARD,
    'forward-left': Direction.LEFT_FORWARD,
    'forward left': Direction.LEFT_FORWARD,
    'front_left': Direction.LEFT_FORWARD,
    'front-left': Direction.LEFT_FORWARD,
    'left_front': Direction.LEFT_FORWARD,
    'left-front': Direction.LEFT_FORWARD,
    'right_forward': Direction.RIGHT_FORWARD,
    'right-forward': Direction.RIGHT_FORWARD,
    'right forward': Direction.RIGHT_FORWARD,
    'forward_right': Direction.RIGHT_FORWARD,
    'forward-right': Direction.RIGHT_FORWARD,
    'forward right': Direction.RIGHT_FORWARD,
    'front_right': Direction.RIGHT_FORWARD,
    'front-right': Direction.RIGHT_FORWARD,
    'right_front': Direction.RIGHT_FORWARD,
    'right-front': Direction.RIGHT_FORWARD,
    'left_backward': Direction.LEFT_BACKWARD,
    'left-backward': Direction.LEFT_BACKWARD,
    'left backward': Direction.LEFT_BACKWARD,
    'backward_left': Direction.LEFT_BACKWARD,
    'backward-left': Direction.LEFT_BACKWARD,
    'backward left': Direction.LEFT_BACKWARD,
    'back_left': Direction.LEFT_BACKWARD,
    'back-left': Direction.LEFT_BACKWARD,
    'right_backward': Direction.RIGHT_BACKWARD,
    'right-backward': Direction.RIGHT_BACKWARD,
    'right backward': Direction.RIGHT_BACKWARD,
    'backward_right': Direction.RIGHT_BACKWARD,
    'backward-right': Direction.RIGHT_BACKWARD,
    'backward right': Direction.RIGHT_BACKWARD,
    'back_right': Direction.RIGHT_BACKWARD,
    'back-right': Direction.RIGHT_BACKWARD,
    'any': Direction.ANY,
    'none': Direction.ANY,
    'stationary': Direction.ANY,
    'stop': Direction.ANY,
}

_STOP_WORDS = ('stop', 'pause', 'stand still', 'remain still', 'freeze')

_LLM_CONSTRAINT_PROMPT = """You are a strict motion-instruction parser.
Convert the sentence into JSON only.

Return exactly one JSON object with this schema:
{
  "phases": [
    {
      "order": 0,
      "action": "walk",
      "direction": "forward",
      "steps": 3,
      "degrees": null,
      "repetitions": null,
      "stop": false
    }
  ],
  "direction_sequence": ["forward"]
}

Rules:
- Split the motion into ordered phases using temporal cues such as then, next, finally, afterwards, before, after.
- Allowed directions are only: forward, backward, left, right, left_forward, right_forward, left_backward, right_backward, any.
- Preserve diagonal directions such as left-forward and right-forward using snake_case labels.
- If a number is not explicitly stated, use null instead of guessing.
- If a phase means stop or pause, set action to "stop" and stop to true.
- direction_sequence should contain only the ordered movement directions, excluding "any".
- Output JSON only. No markdown. No explanation.

Sentence: "{caption}"
JSON:"""


def _normalize_direction_label(direction: Optional[str]) -> Tuple[Direction, Optional[str]]:
    """Normalise free-form direction text into the current 4-way direction enum."""
    if direction is None:
        return Direction.ANY, None

    raw = str(direction).strip().lower()
    if not raw:
        return Direction.ANY, None

    if raw in _DIRECTION_ALIASES:
        return _DIRECTION_ALIASES[raw], raw

    compact = raw.replace('_', '-').replace(' ', '-')
    if compact in _DIRECTION_ALIASES:
        return _DIRECTION_ALIASES[compact], raw

    return Direction.ANY, raw


def _coerce_numeric_value(value: Any) -> Optional[float]:
    """Best-effort conversion for LLM JSON numeric fields."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    raw = str(value).strip().lower()
    if not raw or raw == 'null':
        return None
    if raw == 'twice':
        return 2.0
    if raw in _WORD2NUM:
        return float(_WORD2NUM[raw])
    try:
        return float(raw)
    except ValueError:
        return None


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from LLM output."""
    start = text.find('{')
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return None


def _compile_phases_from_constraints(
    caption: str,
    constraints: List[ConstraintPhase],
    direction_sequence: List[Direction],
    source: str,
) -> List[ParsedCaptionPhase]:
    """Build phase-level constraints from existing regex outputs."""
    order_to_phase: Dict[int, ParsedCaptionPhase] = {}
    for constraint in constraints:
        phase = order_to_phase.setdefault(
            constraint.order,
            ParsedCaptionPhase(order=constraint.order, direction=constraint.direction),
        )
        if phase.direction == Direction.ANY and constraint.direction != Direction.ANY:
            phase.direction = constraint.direction
        if constraint.type == 'steps' and phase.steps is None:
            phase.steps = constraint.value
        elif constraint.type == 'degrees' and phase.degrees is None:
            phase.degrees = constraint.value
        elif constraint.type == 'repetitions' and phase.repetitions is None:
            phase.repetitions = constraint.value

    phases = [order_to_phase[key] for key in sorted(order_to_phase)]
    if not phases and direction_sequence:
        phases = [
            ParsedCaptionPhase(order=idx, direction=direction)
            for idx, direction in enumerate(direction_sequence)
        ]

    lower_caption = caption.lower()
    if phases and any(word in lower_caption for word in _STOP_WORDS):
        last_phase = phases[-1]
        if not last_phase.stop:
            phases.append(ParsedCaptionPhase(
                order=last_phase.order + 1,
                action='stop',
                stop=True,
            ))

    return phases


def _constraints_from_parsed_phases(phases: List[ParsedCaptionPhase]) -> List[ConstraintPhase]:
    """Compile phase-level JSON constraints back into the existing reward schema."""
    constraints: List[ConstraintPhase] = []
    for phase in phases:
        if phase.stop:
            continue
        if phase.steps is not None:
            constraints.append(ConstraintPhase(
                type='steps',
                value=phase.steps,
                direction=phase.direction,
                order=phase.order,
                raw=phase.action,
            ))
        if phase.degrees is not None:
            constraints.append(ConstraintPhase(
                type='degrees',
                value=phase.degrees,
                direction=phase.direction,
                order=phase.order,
                raw=phase.action,
            ))
        if phase.repetitions is not None:
            constraints.append(ConstraintPhase(
                type='repetitions',
                value=phase.repetitions,
                direction=phase.direction,
                order=phase.order,
                raw=phase.action,
            ))
    return constraints


def _direction_sequence_from_phases(phases: List[ParsedCaptionPhase]) -> List[Direction]:
    """Derive ordered directions from structured phases."""
    directions: List[Direction] = []
    for phase in sorted(phases, key=lambda item: item.order):
        if phase.stop or phase.direction == Direction.ANY:
            continue
        directions.append(phase.direction)
    return directions


def parse_constraints_regex(caption: str) -> ParsedCaptionConstraints:
    """Wrap the legacy regex parser in the new structured output format."""
    numerical_constraints = parse_numerical_constraints(caption)
    direction_sequence = parse_direction_sequence(caption)
    phases = _compile_phases_from_constraints(
        caption=caption,
        constraints=numerical_constraints,
        direction_sequence=direction_sequence,
        source='regex',
    )
    return ParsedCaptionConstraints(
        phases=phases,
        numerical_constraints=numerical_constraints,
        direction_sequence=direction_sequence,
        source='regex',
    )


def _caption_number_hint(caption: str, default: float = 1.0) -> float:
    text = caption.lower()
    for word, value in _WORD2NUM.items():
        if re.search(rf'\b{re.escape(word)}\b', text):
            return float(value)
    match = re.search(r'\b(\d+)\b', text)
    if match:
        return float(match.group(1))
    if 'twice' in text:
        return 2.0
    return default


def _add_presence_rule(
    specs: List[Dict[str, Any]],
    rule_id: str,
    template_name: str,
    args: Optional[Dict[str, Any]] = None,
    weight: float = 1.0,
):
    specs.append({
        "id": rule_id,
        "kind": "count",
        "ref": {
            "type": "template",
            "name": template_name,
            "args": args or {},
        },
        "op": "ge",
        "value": 1,
        "weight": weight,
    })


def caption_to_executor_specs(caption: str) -> List[Dict[str, Any]]:
    """Build detector-style constraints from common action phrases.

    This is deliberately rule-based and transparent.  The LLM parser can later
    emit the same schema directly, but these rules already let reward use the
    executor for frequent commands.
    """
    text = caption.lower()
    specs: List[Dict[str, Any]] = []

    if any(phrase in text for phrase in ['bring both hands together', 'hands together', 'bring hands together']):
        _add_presence_rule(
            specs, 'hands_close_present', 'hands_close',
            {"threshold": 0.14, "min_frames": 2}, weight=1.0,
        )

    if 'clap' in text:
        target = _caption_number_hint(text, default=1.0)
        # Clap detection uses a two-threshold hysteresis state machine on
        # inter-hand distance to avoid double-counting boundary jitter.
        specs.append({
            "id": "clap_count",
            "kind": "count",
            "ref": {
                "type": "template",
                "name": "clap",
                "args": {
                    "threshold": 0.077,
                    "enter_threshold": 0.077,
                    "exit_threshold": 0.077 * 1.6,
                    "min_frames": 1,
                },
            },
            "op": "eq",
            "value": target,
            "tolerance": 0.0,
            "weight": 1.0,
        })

    if 'squat' in text:
        target = _caption_number_hint(text, default=1.0)
        # A squat cycle requires the pelvis to dip by at least this much.
        specs.append({
            "id": "squat_count",
            "kind": "count",
            "ref": {
                "type": "template",
                "name": "squat_cycle",
                "args": {"threshold": 0.15},
            },
            "op": "eq",
            "value": target,
            "tolerance": 0.0,
            "weight": 1.0,
        })

    if 'touch' in text and 'head' in text:
        hand = 'any'
        if 'left hand' in text or 'left arm' in text:
            hand = 'left'
        elif 'right hand' in text or 'right arm' in text:
            hand = 'right'
        _add_presence_rule(
            specs, 'touch_head_present', 'touch_head',
            {"hand": hand, "threshold": 0.18, "min_frames": 1}, weight=1.0,
        )

    if 'raise' in text and ('foot' in text or 'leg' in text):
        foot = 'any'
        if 'left' in text:
            foot = 'left'
        elif 'right' in text:
            foot = 'right'
        # Use the binary_max mode: a single event iff the peak foot height
        # clears the threshold at some point, rather than counting every
        # sustained period above it.
        _add_presence_rule(
            specs, 'raise_foot_present', 'raise_foot',
            {"foot": foot, "threshold": 0.08, "mode": "binary_max"}, weight=0.8,
        )

    if 'turn left' in text:
        # Require the turn to span enough frames in addition to clearing the
        # cumulative-angle check, otherwise momentary spikes count as turns.
        _add_presence_rule(
            specs, 'turn_left_present', 'turn_left',
            {"min_angle_deg": 20.0, "time_threshold_frames": 4}, weight=0.8,
        )
    if 'turn right' in text:
        _add_presence_rule(
            specs, 'turn_right_present', 'turn_right',
            {"min_angle_deg": 20.0, "time_threshold_frames": 4}, weight=0.8,
        )

    # "move forward / backward / left / right" emits a compound reward:
    # cumulative positive displacement along the requested body axis must
    # exceed a displacement threshold, AND the per-frame motion purity
    # (pos_dis / (pos_dis + neg_dis)) must exceed a direction threshold.
    # The two scalars combine additively at the executor level.
    direction_terms = [
        ('move forward', 'forward'),
        ('move backward', 'backward'),
        ('move to the left', 'left'),
        ('move to the right', 'right'),
        ('walk forward', 'forward'),
        ('walk backward', 'backward'),
        ('go forward', 'forward'),
        ('go backward', 'backward'),
    ]
    for phrase, axis in direction_terms:
        if phrase in text:
            specs.append({
                "id": f"{axis}_displacement",
                "kind": "signal",
                "ref": {
                    "type": "signal",
                    "name": "directional_displacement",
                    "args": {
                        "entity": "pelvis", "direction": axis, "frame": "body",
                    },
                },
                "reduce": "last",
                "op": "ge",
                "value": 0.25,
                "weight": 1.0,
            })
            specs.append({
                "id": f"{axis}_direction_score",
                "kind": "signal",
                "ref": {
                    "type": "signal",
                    "name": "direction_score",
                    "args": {
                        "entity": "pelvis", "direction": axis, "frame": "body",
                    },
                },
                "reduce": "last",
                "op": "ge",
                "value": 0.6,
                "weight": 0.5,
            })
            break

    # Anti-exploit regularizers for narrow upper-body actions: if the prompt is
    # mostly hands/feet, discourage large root drift from satisfying the reward
    # through unrelated locomotion.
    local_action_terms = ['clap', 'hands together', 'touch', 'raise']
    locomotion_terms = ['walk', 'run', 'move forward', 'step forward', 'turn']
    if any(term in text for term in local_action_terms) and not any(term in text for term in locomotion_terms):
        specs.append({
            "id": "local_action_limit_root_speed",
            "kind": "absence",
            "ref": {
                "type": "signal",
                "name": "speed",
                "args": {"entity": "pelvis"},
            },
            "reduce": "mean",
            "op": "le",
            "value": 0.35,
            "weight": 0.4,
        })

    return specs


def constraints_to_executor_specs(
    parsed: ParsedCaptionConstraints,
    caption: str = '',
) -> List[Dict[str, Any]]:
    """Translate parsed caption constraints into reusable executor specs.

    The executor schema is detector-oriented: rules call signals/templates
    instead of embedding reward math directly in caption-specific code.
    """
    specs: List[Dict[str, Any]] = caption_to_executor_specs(caption)
    text = caption.lower()
    has_temporal_steps = (
        len([c for c in parsed.numerical_constraints if c.type == 'steps']) > 1
        or any(c.direction != Direction.ANY for c in parsed.numerical_constraints if c.type == 'steps')
    )

    for idx, constraint in enumerate(parsed.numerical_constraints):
        # `constraint.value` may be signed for degrees (CW = negative). Any
        # geometric quantity derived below uses magnitude only; sign is
        # carried through `constraint.direction` and the degree scorer.
        target_mag = abs(float(constraint.value))
        if constraint.type == 'steps':
            if constraint.direction != Direction.ANY:
                # Displacement thresholds scale with the requested step count.
                # Roughly 0.5m per step in HumanML3D walks; require ~70% of
                # that as a lower bound so "stepped slightly forward" doesn't
                # satisfy "walks 5 steps forward". P1 (audit/reward_real_eval
                # showed matched vs mismatched on executor was ~50/50 with
                # the looser thresholds).
                step_disp_m = max(0.30, 0.35 * target_mag)
                phase_ref = {
                    "type": "template",
                    "name": "direction_phase",
                    "args": {
                        "entity": "pelvis",
                        "direction": constraint.direction.value,
                        "frame": "body",
                        "min_displacement": step_disp_m,
                        "min_frames": 6,
                        # Tightened from 0.45: the previous threshold let
                        # a sample with mostly-forward + slight-lateral
                        # motion satisfy any of LEFT/RIGHT/FORWARD constraints.
                        "purity_threshold": 0.60,
                    },
                }
                specs.append({
                    "id": f"{constraint.direction.value}_phase_steps_{idx}",
                    "kind": "phase_count",
                    "phase_ref": phase_ref,
                    "count_ref": {
                        "type": "template",
                        "name": "step",
                        "args": {"foot": "any"},
                    },
                    "op": "eq",
                    "value": target_mag,
                    "tolerance": 0.5,
                    "weight": 1.4,
                })
                specs.append({
                    "id": f"{constraint.direction.value}_phase_displacement_{idx}",
                    "kind": "phase_signal",
                    "phase_ref": phase_ref,
                    "measure": "displacement",
                    "op": "ge",
                    "value": step_disp_m,
                    "weight": 0.8,
                })
                specs.append({
                    "id": f"move_{constraint.direction.value}_{idx}",
                    "kind": "signal",
                    "ref": {
                        "type": "signal",
                        "name": "directional_displacement",
                        "args": {
                            "entity": "pelvis",
                            "direction": constraint.direction.value,
                            "frame": "body",
                        },
                    },
                    "reduce": "last",
                    "op": "ge",
                    "value": step_disp_m,
                    "weight": 0.5,
                })
                # Purity gate: directional purity must clear 0.6 (i.e. the
                # body actually went in the requested direction, not just
                # in roughly that quadrant). Without this, noisy /
                # shuffled motion can pile up cumulative positive
                # displacement and game executor_scores.
                specs.append({
                    "id": f"{constraint.direction.value}_purity_{idx}",
                    "kind": "signal",
                    "ref": {
                        "type": "signal",
                        "name": "direction_score",
                        "args": {
                            "entity": "pelvis",
                            "direction": constraint.direction.value,
                            "frame": "body",
                        },
                    },
                    "reduce": "last",
                    "op": "ge",
                    "value": 0.6,
                    "weight": 0.7,
                })
            elif not has_temporal_steps:
                specs.append({
                    "id": f"steps_{idx}",
                    "kind": "count",
                    "ref": {
                        "type": "template",
                        "name": "step",
                        "args": {"foot": "any"},
                    },
                    "op": "eq",
                    "value": target_mag,
                    "tolerance": 1.0,
                    "weight": 1.0,
                })
        elif constraint.type == 'degrees':
            direction = "left"
            if constraint.direction == Direction.RIGHT:
                direction = "right"
            specs.append({
                "id": f"turn_{idx}",
                "kind": "signal",
                "ref": {
                    "type": "signal",
                    "name": "yaw_rotation",
                    "args": {"direction": direction},
                },
                "reduce": "last",
                "op": "ge",
                "value": target_mag,
                "weight": 1.0,
            })
        elif constraint.type == 'repetitions':
            if not any(spec.get("id") == "squat_count" for spec in specs):
                specs.append({
                    "id": f"squat_or_repeat_{idx}",
                    "kind": "count",
                    "ref": {
                        "type": "template",
                        "name": "squat_cycle",
                        "args": {},
                    },
                    "op": "eq",
                    "value": target_mag,
                    "tolerance": 1.0,
                    "weight": 0.6,
                })

    # Direction-only captions such as "walk forward then backward" still get
    # signal evidence even when no numeric target exists.
    if not parsed.numerical_constraints:
        for idx, direction in enumerate(parsed.direction_sequence):
            if direction == Direction.ANY:
                continue
            # Tightened relative to v0: 0.15m displacement was permissive
            # enough that almost any random walk satisfied it, which let
            # mismatched captions tie matched on executor_scores.
            phase_ref = {
                "type": "template",
                "name": "direction_phase",
                "args": {
                    "entity": "pelvis",
                    "direction": direction.value,
                    "frame": "body",
                    "min_displacement": 0.30,
                    "min_frames": 6,
                    "purity_threshold": 0.60,
                },
            }
            specs.append({
                "id": f"dir_phase_{direction.value}_{idx}",
                "kind": "phase_signal",
                "phase_ref": phase_ref,
                "measure": "displacement",
                "op": "ge",
                "value": 0.30,
                "weight": 0.9,
            })
            specs.append({
                "id": f"dir_{direction.value}_{idx}",
                "kind": "signal",
                "ref": {
                    "type": "signal",
                    "name": "directional_displacement",
                    "args": {
                        "entity": "pelvis",
                        "direction": direction.value,
                        "frame": "body",
                    },
                },
                "reduce": "last",
                "op": "ge",
                "value": 0.40,
                "weight": 0.6,
            })
            # Purity gate to prevent random jitter from accumulating
            # positive displacement and satisfying the rule.
            specs.append({
                "id": f"dir_purity_{direction.value}_{idx}",
                "kind": "signal",
                "ref": {
                    "type": "signal",
                    "name": "direction_score",
                    "args": {
                        "entity": "pelvis",
                        "direction": direction.value,
                        "frame": "body",
                    },
                },
                "reduce": "last",
                "op": "ge",
                "value": 0.6,
                "weight": 0.5,
            })

    # Add temporal-composite evidence only for explicit turns.  Plain left/right
    # locomotion must not be rewritten as turn_left/turn_right.
    if len(parsed.direction_sequence) >= 2 and 'turn' in text:
        first, second = parsed.direction_sequence[0], parsed.direction_sequence[1]
        if first != Direction.ANY and second in {Direction.LEFT, Direction.RIGHT}:
            specs.append({
                "id": f"{first.value}_before_turn_{second.value}",
                "kind": "temporal_composite",
                "lhs": {
                    "ref": {
                        "type": "signal",
                        "name": "directional_displacement",
                        "args": {
                            "entity": "pelvis",
                            "direction": first.value,
                            "frame": "body",
                        },
                    },
                    "evidence": {"measure": "displacement", "op": "ge", "value": 0.2},
                },
                "rhs": {
                    "ref": {
                        "type": "template",
                        "name": f"turn_{second.value}",
                        "args": {},
                    },
                    "evidence": {"measure": "duration", "op": "ge", "value": 0.2},
                },
                "relation": {
                    "name": "evidence_before",
                    "rhs_anchor": "start",
                    "measure": "lhs_pre_anchor_displacement",
                    "op": "ge",
                    "value": 0.15,
                },
                "weight": 0.8,
            })

    # Penalize unnecessary opposite turns when the caption only asks for one
    # turn direction. This helps avoid spinning artifacts.
    if 'turn left' in text and 'turn right' not in text:
        specs.append({
            "id": "avoid_extra_right_turn",
            "kind": "absence",
            "ref": {
                "type": "template",
                "name": "turn_right",
                "args": {"min_angle_deg": 20.0},
            },
            "value": 0,
            "weight": 0.4,
        })
    if 'turn right' in text and 'turn left' not in text:
        specs.append({
            "id": "avoid_extra_left_turn",
            "kind": "absence",
            "ref": {
                "type": "template",
                "name": "turn_left",
                "args": {"min_angle_deg": 20.0},
            },
            "value": 0,
            "weight": 0.4,
        })

    return specs


def _direction_continuous_fallback(
    directions: List[Direction],
    joints: Optional[torch.Tensor],
) -> float:
    """Continuous direction-match score from raw root XZ trajectory.

    Used when analyze_motion_phases couldn't isolate clean phases (typical
    after VQ-VAE reconstruction smooths direction changes below the 35-deg
    phase split threshold). Returns a dense value in [0, 1].

    Per requested direction d:
      net_proj   = abs(dot(end_minus_start, d_unit))     # signed
      directional_path = sum_t |delta_xz_t along d_unit| # path that
                                                          contributed to the
                                                          movement on d
      score_d    = max(0, net_proj_along_d / total_path)

    Using net projection (not cumulative positive) means a random walk that
    zig-zags forward and back gets a low score, while a real forward walk
    that mostly went forward gets a high score. Without this, the previous
    cumulative-positive version inflated every "had any forward motion"
    sample to ~1.0 and direction bucket lost rank-1 ground.

    Mean over requested directions (multi-direction caption needs all
    components satisfied).
    """
    if joints is None or joints.shape[0] < 2 or not directions:
        return 0.0

    root_xz = joints[:, 0, [0, 2]]  # [T, 2]
    delta = root_xz[1:] - root_xz[:-1]
    path = torch.norm(delta, dim=-1).sum().item()
    if path < 0.05:
        return 0.0
    net_xz = (root_xz[-1] - root_xz[0])  # [2]

    initial_facing = float(np.pi / 2)
    per_dir = []
    for d in directions:
        if d == Direction.ANY or d not in _IDEAL_RELATIVE_ANGLE:
            continue
        ideal_angle = initial_facing + _IDEAL_RELATIVE_ANGLE[d]
        ux = float(np.cos(ideal_angle))
        uz = float(np.sin(ideal_angle))
        unit = net_xz.new_tensor([ux, uz])
        net_proj = float((net_xz * unit).sum().item())
        # Only positive net projection along the requested direction counts.
        # Normalize by total path to penalize fidgeting / detours.
        score_d = max(0.0, net_proj) / max(path, 1e-6)
        per_dir.append(min(1.0, score_d))

    if not per_dir:
        return 0.0
    return float(min(1.0, sum(per_dir) / len(per_dir)))


def score_direction_sequence(
    directions: List[Direction],
    phases: List[MotionPhase],
    joints: Optional[torch.Tensor] = None,
) -> float:
    """Score how well motion phases match the expected direction sequence.

    Two-mode scoring:
      1. PHASE MODE -- when analyze_motion_phases returned at least one
         significant phase with a non-ANY direction: do greedy sequential
         matching against `phases`, with first-direction bonus and
         redundancy penalty. Same as before.
      2. CONTINUOUS FALLBACK -- when phases are empty (VQ-VAE decoding can
         smooth direction changes below the 35-degree threshold the phase
         analyzer uses), fall back to a dense ratio computed directly from
         root XZ trajectory: average over requested directions of
         (positive projection onto that direction) / (total path length).
         Returns a value in [0, 1] that varies smoothly with motion quality
         instead of collapsing to 0 like the old phase-only path.

    The continuous fallback was added after audit/reward_group_eval.py
    showed direction-only bucket at chance (25% rank-1) because most
    GT motion phases didn't survive VQ-VAE reconstruction.
    """
    if not directions:
        return 0.0  # no directions to match

    sig_phases = [p for p in phases if p.direction != Direction.ANY and p.displacement > 0.05]
    if not sig_phases:
        return _direction_continuous_fallback(directions, joints)

    matched = 0
    last_ph = -1
    purity_sum = 0.0
    for d in directions:
        for ph_i in range(last_ph + 1, len(sig_phases)):
            if sig_phases[ph_i].direction == d:
                matched += 1
                purity_sum += _purity_factor(sig_phases[ph_i].purity)
                last_ph = ph_i
                break

    # Base: fraction matched, each match weighted by its purity factor
    base_score = purity_sum / len(directions)

    # Bonus: if first expected direction matches first significant phase,
    # scaled by that phase's purity (so diagonal first move gets less bonus)
    if sig_phases[0].direction == directions[0]:
        base_score = min(1.0,
                         base_score + 0.15 * _purity_factor(sig_phases[0].purity))

    # Redundancy penalty: extra phases beyond what was requested are
    # penalized to discourage fidgeting / extra body movements.
    # Each extra phase costs 0.1, capped at 0.4 total.
    n_extra = max(0, len(sig_phases) - len(directions))
    redundancy_penalty = min(0.4, 0.1 * n_extra)
    base_score = max(0.0, base_score - redundancy_penalty)

    return base_score


# ---------------------------------------------------------------------------
# Phase-aware constraint scoring
# ---------------------------------------------------------------------------

def _step_accuracy(generated: float, target: float) -> float:
    """Asymmetric step-count accuracy.

    Undershoot is penalized harder than overshoot - the model was observed to
    hesitate and produce fewer steps than requested. Overshoot of 1 is
    tolerated; undershoot of 1 costs ~0.4 in score.

    sigma_under = max(target*0.15, 0.5) -> tight penalty for under
    sigma_over  = max(target*0.35, 1.5) -> loose tolerance for over
    """
    diff = generated - target
    if diff < 0:
        sigma = max(target * 0.15, 0.5)
    else:
        sigma = max(target * 0.35, 1.5)
    return float(np.exp(-0.5 * (diff / sigma) ** 2))


def _acc_in_range(generated: float, c: 'ConstraintPhase',
                  sigma_floor: float, sigma_scale: float = 0.2) -> float:
    """Score `generated` against a constraint that may carry a range.

    Inside [value_min, value_max]: acc = 1.0 (full credit).
    Outside: Gaussian decay from the nearest range edge.
    Falls back to a single-point Gaussian around `value` when the range
    is not populated (precise caption like "3 steps").
    """
    if c.value_min is not None and c.value_max is not None:
        lo, hi = c.value_min, c.value_max
        if lo <= generated <= hi:
            return 1.0
        d = (lo - generated) if generated < lo else (generated - hi)
        sigma = max(abs(c.value) * sigma_scale, sigma_floor)
        return float(np.exp(-0.5 * (d / sigma) ** 2))
    sigma = max(abs(c.value) * sigma_scale, sigma_floor)
    return float(np.exp(-0.5 * ((generated - c.value) / sigma) ** 2))


def _step_accuracy_with_range(generated: float, c: 'ConstraintPhase') -> float:
    """Range-aware step accuracy. Inside [min,max] = 1.0, outside uses the
    same asymmetric Gaussian as `_step_accuracy` from the nearest edge."""
    if c.value_min is not None and c.value_max is not None:
        if c.value_min <= generated <= c.value_max:
            return 1.0
        if generated < c.value_min:
            edge = c.value_min
        else:
            edge = c.value_max
        return _step_accuracy(generated, edge)
    return _step_accuracy(generated, c.value)


def score_constraints_against_phases(
    constraints: List[ConstraintPhase],
    phases: List[MotionPhase],
    total_steps: int,
    total_rotation_deg: float,
    total_repetitions: int,
) -> float:
    """Score parsed constraints against detected motion phases.

    Handles two modes:
    - No temporal ordering: score against global totals with direction bonus
    - With temporal ordering: align constraint groups to phase groups, score per-group
    """
    if not constraints:
        return 0.0

    # Check if there's temporal ordering
    orders = set(c.order for c in constraints)
    has_temporal = len(orders) > 1

    if not has_temporal:
        return _score_global(constraints, phases, total_steps,
                             total_rotation_deg, total_repetitions)
    else:
        return _score_temporal(constraints, phases, total_steps,
                               total_rotation_deg, total_repetitions)


def _score_global(
    constraints: List[ConstraintPhase],
    phases: List[MotionPhase],
    total_steps: int,
    total_rotation_deg: float,
    total_repetitions: int,
) -> float:
    """Score constraints without temporal ordering."""
    scores = []
    for c in constraints:
        if c.type == 'steps':
            if c.direction != Direction.ANY and phases:
                # Sum steps from phases matching this direction
                dir_steps = sum(p.step_count for p in phases
                                if p.direction == c.direction)
                # Also consider total if no phases match direction
                generated = dir_steps if dir_steps > 0 else total_steps
            else:
                generated = total_steps
            acc = _step_accuracy_with_range(generated, c)
            # Direction match is a multiplicative gate, not an additive bonus.
            # Wrong direction caps accuracy at 0.7; correct direction preserves it.
            if c.direction != Direction.ANY:
                matching = [p for p in phases if p.direction == c.direction]
                if matching:
                    # Weight purity by phase displacement
                    total_disp = sum(p.displacement for p in matching)
                    if total_disp > 1e-6:
                        avg_purity = sum(p.purity * p.displacement
                                         for p in matching) / total_disp
                    else:
                        avg_purity = 1.0
                    acc = acc * _purity_factor(avg_purity)
                else:
                    acc = acc * 0.7  # no phase matches direction
            scores.append(acc)

        elif c.type == 'degrees':
            # Sign convention (empirically verified on HumanML3D):
            # total_rotation_deg > 0 -> clockwise / right turn.
            # total_rotation_deg < 0 -> counter-clockwise / left turn.
            if c.direction == Direction.LEFT:
                generated = -total_rotation_deg  # flip so left turns score positive
            elif c.direction == Direction.RIGHT:
                generated = total_rotation_deg   # CW already positive
            elif c.value < 0:
                # Caption named a spin direction; c.value already signed.
                generated = total_rotation_deg
            else:
                generated = abs(total_rotation_deg)
            acc = _acc_in_range(generated, c, sigma_floor=15.0, sigma_scale=0.2)
            scores.append(acc)

        elif c.type == 'repetitions':
            acc = _acc_in_range(total_repetitions, c, sigma_floor=1.0, sigma_scale=0.3)
            scores.append(acc)

    return min(1.0, float(np.mean(scores))) if scores else 0.0


def _score_temporal(
    constraints: List[ConstraintPhase],
    phases: List[MotionPhase],
    total_steps: int,
    total_rotation_deg: float,
    total_repetitions: int,
) -> float:
    """Score constraints with temporal ordering against phase sequence."""
    # Group constraints by temporal order
    from collections import defaultdict
    order_groups = defaultdict(list)
    for c in constraints:
        order_groups[c.order].append(c)
    sorted_orders = sorted(order_groups.keys())
    constraint_groups = [order_groups[o] for o in sorted_orders]
    n_groups = len(constraint_groups)

    if not phases or n_groups == 0:
        return _score_global(constraints, phases, total_steps,
                             total_rotation_deg, total_repetitions)

    # Align phases to constraint groups proportionally by frame count
    total_frames = sum(p.end_frame - p.start_frame for p in phases)
    frames_per_group = total_frames / n_groups if n_groups > 0 else total_frames

    phase_groups: List[List[MotionPhase]] = [[] for _ in range(n_groups)]
    cumulative = 0
    group_idx = 0
    for p in phases:
        phase_groups[group_idx].append(p)
        cumulative += (p.end_frame - p.start_frame)
        if cumulative >= frames_per_group * (group_idx + 1) and group_idx < n_groups - 1:
            group_idx += 1

    # Score each constraint group against its aligned phase group
    group_scores = []
    temporal_dir_matches = 0
    temporal_dir_total = 0

    for cg, pg in zip(constraint_groups, phase_groups):
        pg_steps = sum(p.step_count for p in pg)
        pg_rotation = sum(p.rotation_deg for p in pg)
        pg_reps = total_repetitions  # repetitions are hard to segment

        # Dominant direction of phase group
        if pg:
            dir_counts: Dict[Direction, float] = {}
            for p in pg:
                d = p.direction
                dir_counts[d] = dir_counts.get(d, 0) + p.displacement
            pg_dir = max(dir_counts, key=dir_counts.get) if dir_counts else Direction.ANY
        else:
            pg_dir = Direction.ANY

        for c in cg:
            if c.type == 'steps':
                if c.direction != Direction.ANY:
                    dir_steps = sum(p.step_count for p in pg
                                    if p.direction == c.direction)
                    generated = dir_steps if dir_steps > 0 else pg_steps
                else:
                    generated = pg_steps
                acc = _step_accuracy_with_range(generated, c)
                if c.direction != Direction.ANY:
                    temporal_dir_total += 1
                    matching = [p for p in pg if p.direction == c.direction]
                    if matching and pg_dir == c.direction:
                        temporal_dir_matches += 1
                    if matching:
                        total_disp = sum(p.displacement for p in matching)
                        if total_disp > 1e-6:
                            avg_purity = sum(p.purity * p.displacement
                                             for p in matching) / total_disp
                        else:
                            avg_purity = 1.0
                        acc = acc * _purity_factor(avg_purity)
                    else:
                        acc = acc * 0.7  # direction mismatch
                group_scores.append(acc)

            elif c.type == 'degrees':
                # Same empirical sign convention as _score_global.
                if c.direction == Direction.LEFT:
                    generated = -pg_rotation
                elif c.direction == Direction.RIGHT:
                    generated = pg_rotation
                elif c.value < 0:
                    generated = pg_rotation
                else:
                    generated = abs(pg_rotation)
                acc = _acc_in_range(generated, c, sigma_floor=15.0, sigma_scale=0.2)
                group_scores.append(acc)

            elif c.type == 'repetitions':
                acc = _acc_in_range(pg_reps, c, sigma_floor=1.0, sigma_scale=0.3)
                group_scores.append(acc)

    base_score = float(np.mean(group_scores)) if group_scores else 0.0

    # Temporal order bonus: fraction of directional constraints that matched
    temporal_bonus = 0.0
    if temporal_dir_total > 0:
        temporal_bonus = 0.1 * (temporal_dir_matches / temporal_dir_total)

    # Redundancy penalty: significant phases beyond expected groups indicate
    # fidgeting / extra body moves. Each extra costs 0.08, capped at 0.3.
    sig_phases = [p for p in phases if p.direction != Direction.ANY and p.displacement > 0.05]
    n_extra = max(0, len(sig_phases) - n_groups)
    redundancy_penalty = min(0.3, 0.08 * n_extra)

    return max(0.0, min(1.0, base_score + temporal_bonus - redundancy_penalty))

class GRPORewardModel:
    """
    Reward model for GRPO training.

    Combines:
    1. Text-motion matching (cosine similarity from pretrained evaluator)
    2. Physical plausibility (foot skating + smoothness)
    3. Numerical accuracy (step count, rotation, repetitions)
    4. Optional length regularization to discourage run-on generations
    """

    def __init__(
        self,
        eval_wrapper: EvaluatorModelWrapper,
        vqvae_model,
        word_vectorizer: WordVectorizer,
        device: str = 'cuda:0',
        normalize_reward: bool = True,
        reward_scale: float = 1.0,
        length_penalty_weight: float = 0.0,
        tau: float = 0.1,
        # New reward weights
        physical_weight: float = 0.3,
        numerical_weight: float = 0.5,
        # LLM for caption classification (stillness penalty)
        llm=None,
        tokenizer=None,
        constraint_parser_mode: str = 'hybrid',
        constraint_parser_max_new_tokens: int = 192,
    ):
        self.eval_wrapper = eval_wrapper
        self.vqvae = vqvae_model
        self.w_vectorizer = word_vectorizer
        self.device = device
        self.normalize_reward = normalize_reward
        self.reward_scale = reward_scale
        self.length_penalty_weight = length_penalty_weight
        self.tau = tau
        self.physical_weight = physical_weight
        self.numerical_weight = numerical_weight

        # LLM-based caption classification
        self.llm = llm
        self.tokenizer = tokenizer
        self._motion_caption_cache: Dict[str, bool] = {}
        parser_mode = constraint_parser_mode.lower()
        if parser_mode not in {'regex', 'llm', 'hybrid'}:
            parser_mode = 'hybrid'
        self.constraint_parser_mode = parser_mode
        self.constraint_parser_max_new_tokens = constraint_parser_max_new_tokens
        self._constraint_parse_cache: Dict[str, ParsedCaptionConstraints] = {}
        self.constraint_executor = MotionConstraintExecutor()

        self.vqvae.eval()

        # Load denormalization statistics
        meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        self._mean = torch.from_numpy(
            np.load(f'{meta_dir}/mean.npy')
        ).float().to(device)
        self._std = torch.from_numpy(
            np.load(f'{meta_dir}/std.npy')
        ).float().to(device)

        self._reward_stats = {}

    def _denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        """Denormalize motion from VQ-VAE output space to original space.

        See `denormalize_motion` for why we strip the root-velocity MEAN bias.
        """
        return denormalize_motion(motion, self._mean, self._std)

    def _is_motion_caption(self, caption: str) -> bool:
        """Use Gemma-2 to judge whether caption describes physical movement.

        Results are cached so each unique caption is only classified once.
        Falls back to True (assume motion) if LLM is unavailable.
        """
        if caption in self._motion_caption_cache:
            return self._motion_caption_cache[caption]

        if self.llm is None or self.tokenizer is None:
            # No LLM available - conservatively assume motion
            self._motion_caption_cache[caption] = True
            return True

        prompt = (
            f'Does the following sentence describe a person physically moving '
            f'their body (e.g. walking, running, jumping, kicking)?\n'
            f'Sentence: "{caption}"\n'
            f'Answer only "yes" or "no":'
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        try:
            with torch.no_grad():
                self.llm.disable_adapter_layers()
                out = self.llm.generate(
                    input_ids, max_new_tokens=3, do_sample=False,
                )
        except Exception:
            # LLM failed - assume motion to be safe
            self._motion_caption_cache[caption] = True
            return True
        finally:
            self.llm.enable_adapter_layers()
        answer = self.tokenizer.decode(out[0, len(input_ids[0]):], skip_special_tokens=True)
        is_motion = 'yes' in answer.lower()
        self._motion_caption_cache[caption] = is_motion
        return is_motion

    def _generate_base_llm_text(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> Optional[str]:
        """Run the frozen base Gemma for lightweight parsing/classification."""
        if self.llm is None or self.tokenizer is None:
            return None

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        try:
            with torch.no_grad():
                self.llm.disable_adapter_layers()
                outputs = self.llm.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
        except Exception:
            return None
        finally:
            self.llm.enable_adapter_layers()

        return self.tokenizer.decode(
            outputs[0, len(input_ids[0]):],
            skip_special_tokens=True,
        ).strip()

    def _parse_constraints_with_llm(self, caption: str) -> Optional[ParsedCaptionConstraints]:
        """Parse caption into ordered motion phases using base Gemma JSON output."""
        prompt = _LLM_CONSTRAINT_PROMPT.replace('{caption}', caption)
        response = self._generate_base_llm_text(
            prompt=prompt,
            max_new_tokens=self.constraint_parser_max_new_tokens,
        )
        if not response:
            return None

        json_text = _extract_first_json_object(response)
        if json_text is None:
            return None

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            return None

        raw_phases = payload.get('phases', [])
        if not isinstance(raw_phases, list):
            raw_phases = []

        phases: List[ParsedCaptionPhase] = []
        for index, item in enumerate(raw_phases):
            if not isinstance(item, dict):
                continue

            try:
                order = int(item.get('order', index))
            except (TypeError, ValueError):
                order = index

            action = str(item.get('action', 'move')).strip().lower() or 'move'
            direction, raw_direction = _normalize_direction_label(item.get('direction'))
            stop = bool(item.get('stop', False)) or action in {'stop', 'pause'}
            steps = _coerce_numeric_value(
                item.get('steps', item.get('step_count', item.get('target_steps')))
            )
            degrees = _coerce_numeric_value(
                item.get('degrees', item.get('rotation_degrees', item.get('target_rotation_deg')))
            )
            repetitions = _coerce_numeric_value(
                item.get('repetitions', item.get('repeat_count', item.get('times')))
            )

            if stop:
                direction = Direction.ANY
                steps = None
                degrees = None
                repetitions = None

            phases.append(ParsedCaptionPhase(
                order=order,
                action=action,
                direction=direction,
                steps=steps,
                degrees=degrees,
                repetitions=repetitions,
                stop=stop,
                raw_direction=raw_direction,
            ))

        if not phases:
            raw_constraints = payload.get('numerical_constraints', [])
            if isinstance(raw_constraints, list):
                compiled_constraints: List[ConstraintPhase] = []
                for index, item in enumerate(raw_constraints):
                    if not isinstance(item, dict):
                        continue
                    ctype = str(item.get('type', '')).strip().lower()
                    if ctype not in {'steps', 'degrees', 'repetitions'}:
                        continue
                    value = _coerce_numeric_value(item.get('value'))
                    if value is None:
                        continue
                    try:
                        order = int(item.get('order', index))
                    except (TypeError, ValueError):
                        order = index
                    direction, _ = _normalize_direction_label(item.get('direction'))
                    compiled_constraints.append(ConstraintPhase(
                        type=ctype,
                        value=value,
                        direction=direction,
                        order=order,
                        raw=str(item),
                    ))
                if compiled_constraints:
                    phases = _compile_phases_from_constraints(
                        caption=caption,
                        constraints=compiled_constraints,
                        direction_sequence=[],
                        source='llm',
                    )

        if phases:
            phases = sorted(phases, key=lambda phase: (phase.order, phase.stop))
            for new_order, phase in enumerate(phases):
                phase.order = new_order

        numerical_constraints = _constraints_from_parsed_phases(phases)

        direction_sequence: List[Direction] = []
        raw_direction_sequence = payload.get('direction_sequence', [])
        if isinstance(raw_direction_sequence, list):
            for item in raw_direction_sequence:
                direction, _ = _normalize_direction_label(item)
                if direction != Direction.ANY:
                    direction_sequence.append(direction)

        if not direction_sequence:
            direction_sequence = _direction_sequence_from_phases(phases)

        if not phases and not numerical_constraints and not direction_sequence:
            return None

        return ParsedCaptionConstraints(
            phases=phases,
            numerical_constraints=numerical_constraints,
            direction_sequence=direction_sequence,
            source='llm',
            raw_response=response,
        )

    def _parse_caption_constraints(self, caption: str) -> ParsedCaptionConstraints:
        """Parse constraints with regex/LLM hybrid fallback and cache the result."""
        cached = self._constraint_parse_cache.get(caption)
        if cached is not None:
            return cached

        regex_result = parse_constraints_regex(caption)
        if self.constraint_parser_mode == 'regex':
            parsed = regex_result
        else:
            llm_result = self._parse_constraints_with_llm(caption)
            if llm_result is None:
                source = 'regex_fallback' if self.constraint_parser_mode == 'llm' else 'regex'
                parsed = ParsedCaptionConstraints(
                    phases=regex_result.phases,
                    numerical_constraints=regex_result.numerical_constraints,
                    direction_sequence=regex_result.direction_sequence,
                    source=source,
                )
            elif self.constraint_parser_mode == 'llm':
                parsed = llm_result
            else:
                needs_backfill = False
                phases = llm_result.phases or regex_result.phases
                if not llm_result.phases and regex_result.phases:
                    needs_backfill = True

                numerical_constraints = llm_result.numerical_constraints
                if not numerical_constraints and regex_result.numerical_constraints:
                    numerical_constraints = regex_result.numerical_constraints
                    needs_backfill = True

                direction_sequence = llm_result.direction_sequence
                if not direction_sequence and regex_result.direction_sequence:
                    direction_sequence = regex_result.direction_sequence
                    needs_backfill = True

                parsed = ParsedCaptionConstraints(
                    phases=phases,
                    numerical_constraints=numerical_constraints,
                    direction_sequence=direction_sequence,
                    source='hybrid' if needs_backfill else 'llm',
                    raw_response=llm_result.raw_response,
                )

        self._constraint_parse_cache[caption] = parsed
        return parsed

    @torch.no_grad()
    def compute_reward(
        self,
        captions: List[str],
        motion_tokens_list: List[torch.Tensor],
        return_components: bool = False,
    ) -> torch.Tensor:
        batch_size = len(captions)
        assert len(motion_tokens_list) == batch_size

        # --- Decode motion tokens ---
        motions, motion_lengths = self._decode_motion_tokens(motion_tokens_list)

        # --- Text-motion matching reward (existing) ---
        word_embeddings, pos_one_hots, sent_lens = self._encode_text(captions)

        sent_lens_np = sent_lens.cpu().numpy()
        sorted_indices = np.argsort(-sent_lens_np)
        unsort_indices = np.argsort(sorted_indices)

        word_embeddings_s = word_embeddings[sorted_indices]
        pos_one_hots_s = pos_one_hots[sorted_indices]
        sent_lens_s = sent_lens[sorted_indices]
        motions_s = motions[sorted_indices]
        motion_lengths_s = motion_lengths[sorted_indices]

        text_emb, motion_emb = self.eval_wrapper.get_co_embeddings(
            word_embeddings_s, pos_one_hots_s, sent_lens_s,
            motions_s, motion_lengths_s,
        )

        text_emb = text_emb[unsort_indices]
        motion_emb = motion_emb[unsort_indices]

        matching_scores = self._compute_matching_score(text_emb, motion_emb)

        # --- Physical plausibility reward ---
        physical_scores = torch.zeros(batch_size, device=self.device)

        # --- Numerical accuracy reward ---
        numerical_scores = torch.zeros(batch_size, device=self.device)
        has_numerical = torch.zeros(batch_size, device=self.device)

        # --- Direction sequence reward (covers captions without numbers) ---
        direction_scores = torch.zeros(batch_size, device=self.device)
        has_direction = torch.zeros(batch_size, device=self.device)

        # --- Kinematic reward (spatiotemporal) ---
        kinematic_scores = torch.zeros(batch_size, device=self.device)
        has_kinematic = torch.zeros(batch_size, device=self.device)
        executor_scores = torch.zeros(batch_size, device=self.device)
        has_executor = torch.zeros(batch_size, device=self.device)
        parser_source_counts: Dict[str, int] = {}

        # --- Motion-energy gate (run1 collapse fix) ---
        # Multiplier on the action-specific reward branches (physical + cap_sat)
        # that goes to ~0.25 when the generated motion fails to expend the
        # minimum energy implied by the caption's verb. Defaults to 1.0 (no
        # penalty) for non-motion captions.
        energy_gates = torch.ones(batch_size, device=self.device)

        for i in range(batch_size):
            length = int(motion_lengths[i].item())
            motion_norm = motions[i, :length]  # [T, 263]
            motion_raw = self._denormalize(motion_norm)

            # Recover 3D joint positions first; foot_contact is derived from
            # joints (the foot_contact channels 259:263 are unreliable when
            # the motion comes from VQ-VAE decode -- the decoder doesn't
            # preserve their binary {0,1} semantics).
            joint_pos = recover_from_ric(
                motion_raw.unsqueeze(0), joints_num=22
            ).squeeze(0)  # [T, 22, 3]
            foot_contact = derive_foot_contact_from_joints(joint_pos)

            # Stillness penalty: if caption describes motion but body is still,
            # apply a smooth penalty. Linear mapping: stillness 0->-1, 1->+1.
            # Avoids the discontinuous jump of the old threshold-based approach.
            stillness = _stillness_score(joint_pos)
            if self._is_motion_caption(captions[i]):
                physical_scores[i] = 2.0 * stillness - 1.0  # [-1.0, 1.0]
            else:
                physical_scores[i] = 1.0  # non-motion captions: no penalty

            parsed_constraints = self._parse_caption_constraints(captions[i])
            parser_source_counts[parsed_constraints.source] = (
                parser_source_counts.get(parsed_constraints.source, 0) + 1
            )
            constraints = parsed_constraints.numerical_constraints
            dir_seq = parsed_constraints.direction_sequence
            executor_specs = constraints_to_executor_specs(parsed_constraints, captions[i])

            # Analyze motion phases (shared by numerical + direction scoring)
            phases = analyze_motion_phases(
                motion_raw, foot_contact, joints=joint_pos,
                min_phase_frames=8,
                direction_change_threshold=0.6,
            )

            # Skip precision rewards for frozen motion - phases are unreliable
            # when the body isn't actually moving (root drift fakes displacement).
            motion_is_alive = stillness >= 0.5

            if executor_specs and motion_is_alive:
                try:
                    executor_results = self.constraint_executor.evaluate(
                        motion_raw=motion_raw,
                        foot_contact=foot_contact,
                        constraints=executor_specs,
                        joints=joint_pos,
                    )
                    executor_scores[i] = aggregate_executor_score(executor_results)
                    has_executor[i] = 1.0
                except Exception:
                    executor_scores[i] = 0.0

            if constraints and motion_is_alive:
                has_numerical[i] = 1.0

                # Global fallback values
                total_steps = _count_steps_in_range(foot_contact, joints=joint_pos)
                total_rotation = _measure_rotation_signed(motion_raw[:, 0])
                total_reps = _count_repetitions(motion_raw[:, 3])

                # Phase-aware scoring
                numerical_scores[i] = score_constraints_against_phases(
                    constraints, phases,
                    total_steps=total_steps,
                    total_rotation_deg=total_rotation,
                    total_repetitions=total_reps,
                )

            # -- Direction sequence matching --
            # Works even without numeric constraints: "walks forward then backward"
            if dir_seq and not constraints and motion_is_alive:
                # Only use direction reward when numerical is absent,
                # to avoid double-counting (numerical already checks direction)
                has_direction[i] = 1.0
                # Pass joint_pos so score_direction_sequence can fall back to
                # the dense projection-ratio score when phase analyzer yields
                # no clean phases (common after VQ-VAE decode smoothing).
                direction_scores[i] = score_direction_sequence(
                    dir_seq, phases, joints=joint_pos,
                )

            # -- Kinematic reward (spatiotemporal) --
            # Convert constraints -> SubGoals via smart adapter, then
            # evaluate against 3D joint kinematics directly.
            subgoals = constraints_to_subgoals(captions[i], constraints) if motion_is_alive else []
            if subgoals:
                has_kinematic[i] = 1.0
                try:
                    kinematic_scores[i] = evaluate_compositional(
                        joint_pos, motion_raw, foot_contact, subgoals,
                    )
                except Exception:
                    kinematic_scores[i] = 0.0

            # -- Motion-energy gate (P0.C) --
            # Compare the four-axis energy of the generated motion against
            # the minimum implied by the caption verb / parsed constraints.
            # Cheap (only joint_pos + motion_raw[:,0] + a few regex matches)
            # so we run it for every sample regardless of motion_is_alive.
            energy_actual = _compute_motion_energy(joint_pos, motion_raw)
            energy_required = _required_minimum_energy(captions[i], parsed_constraints)
            energy_gates[i] = motion_energy_gate(
                energy_actual, energy_required, floor=0.25,
            )

        # --- Combine rewards ---
        # Matching score: shifted to [0, 1] range using positive cosine similarity
        # (InfoNCE is in [-log(B), 0]; instead use raw cosine for combination)
        text_norm = F.normalize(text_emb, p=2, dim=-1)
        motion_norm = F.normalize(motion_emb, p=2, dim=-1)
        cos_sim = (text_norm * motion_norm).sum(dim=-1)  # [B] in [-1, 1]
        cos_sim_01 = (cos_sim + 1.0) / 2.0  # shift to [0, 1]

        # Unified reward composition (P0.B):
        #
        # Real-data audit (audit/reward_real_eval.py) showed the previous
        # additive composition gave structurally different reward ceilings
        # depending on which caption bucket a sample fell into -- numeric
        # captions had a higher max but their numerical/executor branches
        # were crippled by VQ-VAE foot_contact distortion, so direction-only
        # caption + wrong motion would routinely outscore numeric caption +
        # right motion. Result: matched > mismatched only 57% of the time.
        #
        # Fix: collapse caption-dependent signals into a single
        # "caption_satisfaction" score in [0,1] that's always defined, then
        # combine it with caption-agnostic signals (matching, physical) with
        # fixed weights. Every caption now has the same reward ceiling.
        #
        #   reward = w_match * cos_sim_01            # always active, [0,1]
        #          + w_phys  * physical_scores        # always active, [0,1]
        #          + w_cap   * caption_satisfaction   # always active, [0,1]
        #          - length_penalty
        #
        # caption_satisfaction is composed from numerical/direction/executor/
        # kinematic depending on which signals the caption asks for. When
        # the caption is "pure" (no parseable cues) it falls back to the
        # matching-score itself so the model still gets a learning signal.
        caption_sat = self._compose_caption_satisfaction(
            numerical_scores=numerical_scores,
            executor_scores=executor_scores,
            kinematic_scores=kinematic_scores,
            direction_scores=direction_scores,
            has_numerical=has_numerical,
            has_direction=has_direction,
            has_executor=has_executor,
            has_kinematic=has_kinematic,
            cos_sim_01=cos_sim_01,
        )

        # Weights chosen so matching (the only consistently reliable signal
        # in the real-data audit, 74% matched-vs-mismatched on its own)
        # dominates, but caption-specific evidence can still push the reward
        # up another ~0.5. Physical scores act as a sanity floor.
        w_match = 1.0
        w_phys = self.physical_weight  # default 0.5
        w_cap = self.numerical_weight  # default 1.0
        length_frac = motion_lengths.float() / motion_lengths.float().clamp(min=1).max()
        length_penalty = self.length_penalty_weight * length_frac

        # Energy-gate scaling (P0.C + P2.B post-adversarial-audit).
        # Originally the gate was applied only to physical + caption_sat to
        # preserve a "look like a human" baseline via cos_sim_01. The
        # adversarial audit showed frozen samples kept cos_sim ~0.88 (the
        # eval_wrapper's text-motion encoder is happy with any plausible
        # human pose), so leaving cos_sim ungated gave frozen samples too
        # much of the total reward. Now the gate multiplies cos_sim too:
        # the gate's 0.25 floor still preserves a baseline gradient (gate=0.25
        # means the reward keeps 25% of every branch), but a sample that
        # fully misses the prompt's energy requirement loses ~75% of total.
        cos_sim_gated = cos_sim_01 * energy_gates
        physical_gated = torch.where(
            physical_scores > 0,
            physical_scores * energy_gates,
            physical_scores,
        )
        caption_sat_gated = caption_sat * energy_gates

        rewards = (
            w_match * cos_sim_gated
            + w_phys * physical_gated
            + w_cap * caption_sat_gated
            - length_penalty
        )

        # Store stats
        self._reward_stats = {
            'pos_sim_mean': cos_sim.mean().item(),
            'neg_sim_mean': self._reward_stats.get('neg_sim_mean', 0.0),
            'physical_mean': physical_scores.mean().item(),
            'numerical_mean': (
                numerical_scores[has_numerical > 0].mean().item()
                if has_numerical.sum() > 0 else 0.0
            ),
            'numerical_frac': has_numerical.mean().item(),
            'kinematic_mean': (
                kinematic_scores[has_kinematic > 0].mean().item()
                if has_kinematic.sum() > 0 else 0.0
            ),
            'kinematic_frac': has_kinematic.mean().item(),
            'executor_mean': (
                executor_scores[has_executor > 0].mean().item()
                if has_executor.sum() > 0 else 0.0
            ),
            'executor_frac': has_executor.mean().item(),
            'direction_mean': (
                direction_scores[has_direction > 0].mean().item()
                if has_direction.sum() > 0 else 0.0
            ),
            'direction_frac': has_direction.mean().item(),
            'length_penalty_mean': length_penalty.mean().item(),
            'energy_gate_mean': energy_gates.mean().item(),
            'energy_gate_min': energy_gates.min().item(),
            'energy_gate_low_frac': (energy_gates < 0.5).float().mean().item(),
            'constraint_parser_mode': self.constraint_parser_mode,
            'constraint_parser_llm_frac': parser_source_counts.get('llm', 0) / max(batch_size, 1),
            'constraint_parser_hybrid_frac': parser_source_counts.get('hybrid', 0) / max(batch_size, 1),
            'constraint_parser_regex_frac': (
                (parser_source_counts.get('regex', 0) + parser_source_counts.get('regex_fallback', 0))
                / max(batch_size, 1)
            ),
        }

        if self.normalize_reward:
            rewards = torch.tanh(rewards)
        rewards = rewards * self.reward_scale

        if return_components:
            return rewards, {
                'matching_scores': cos_sim,
                'cos_sim_gated': cos_sim_gated,
                'physical_scores': physical_scores,
                'physical_gated': physical_gated,
                'numerical_scores': numerical_scores,
                'kinematic_scores': kinematic_scores,
                'executor_scores': executor_scores,
                'direction_scores': direction_scores,
                'caption_sat': caption_sat,
                'caption_sat_gated': caption_sat_gated,
                'energy_gates': energy_gates,
                'length_penalty': length_penalty,
                'has_numerical': has_numerical,
                'has_kinematic': has_kinematic,
                'has_executor': has_executor,
                'has_direction': has_direction,
                'reward_stats': self._reward_stats,
            }

        return rewards

    def _decode_motion_tokens(
        self,
        motion_tokens_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode motion tokens to continuous sequences using VQ-VAE decoder."""
        batch_size = len(motion_tokens_list)
        decoded_motions = [None] * batch_size
        motion_lengths = [0] * batch_size

        length_groups = {}
        for i, tokens in enumerate(motion_tokens_list):
            t_len = len(tokens)
            if t_len not in length_groups:
                length_groups[t_len] = []
            length_groups[t_len].append(i)

        for t_len, indices in length_groups.items():
            batch_tokens = torch.stack(
                [motion_tokens_list[i].unsqueeze(0) if motion_tokens_list[i].dim() == 1
                 else motion_tokens_list[i] for i in indices]
            ).to(self.device)
            if batch_tokens.dim() == 3:
                batch_tokens = batch_tokens.squeeze(1)

            try:
                batch_motion = self.vqvae.forward_decoder(batch_tokens)
                for j, idx in enumerate(indices):
                    decoded_motions[idx] = batch_motion[j]
                    motion_lengths[idx] = batch_motion.shape[1]
            except Exception:
                for idx in indices:
                    tokens = motion_tokens_list[idx]
                    if tokens.dim() == 1:
                        tokens = tokens.unsqueeze(0)
                    tokens = tokens.to(self.device)
                    try:
                        motion = self.vqvae.forward_decoder(tokens)
                        decoded_motions[idx] = motion.squeeze(0)
                        motion_lengths[idx] = motion.shape[1]
                    except Exception:
                        dummy_motion = torch.zeros(4, 263, device=self.device)
                        decoded_motions[idx] = dummy_motion
                        motion_lengths[idx] = 4

        max_len = max(motion_lengths)
        motion_dim = decoded_motions[0].shape[-1]
        padded_motions = torch.zeros(batch_size, max_len, motion_dim, device=self.device)

        for i, motion in enumerate(decoded_motions):
            cur_len = motion.shape[0]
            padded_motions[i, :cur_len] = motion

        motion_lengths = torch.tensor(motion_lengths, device=self.device)
        return padded_motions, motion_lengths

    def _encode_text(
        self,
        captions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text captions using word vectorizer."""
        batch_size = len(captions)
        word_embs_list = []
        pos_ohots_list = []
        sent_lens = []

        for caption in captions:
            words = caption.lower().split()
            tokens = ['sos/OTHER'] + [f'{word}/OTHER' for word in words] + ['eos/OTHER']
            sent_len = len(tokens)

            pos_one_hots_list = []
            word_embeddings_list = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots_list.append(pos_oh[None, :])
                word_embeddings_list.append(word_emb[None, :])

            pos_ohot = np.concatenate(pos_one_hots_list, axis=0)
            word_embs = np.concatenate(word_embeddings_list, axis=0)

            word_embs_list.append(torch.from_numpy(word_embs))
            pos_ohots_list.append(torch.from_numpy(pos_ohot))
            sent_lens.append(sent_len)

        max_len = max(sent_lens)
        word_dim = word_embs_list[0].shape[-1]
        pos_dim = pos_ohots_list[0].shape[-1]

        word_embeddings = torch.zeros(batch_size, max_len, word_dim)
        pos_one_hots = torch.zeros(batch_size, max_len, pos_dim)

        for i in range(batch_size):
            cur_len = sent_lens[i]
            word_embeddings[i, :cur_len] = word_embs_list[i]
            pos_one_hots[i, :cur_len] = pos_ohots_list[i]

        sent_lens = torch.tensor(sent_lens, device=self.device)
        return word_embeddings, pos_one_hots, sent_lens

    def _compose_caption_satisfaction(
        self,
        numerical_scores: torch.Tensor,
        executor_scores: torch.Tensor,
        kinematic_scores: torch.Tensor,
        direction_scores: torch.Tensor,
        has_numerical: torch.Tensor,
        has_direction: torch.Tensor,
        has_executor: torch.Tensor,
        has_kinematic: torch.Tensor,
        cos_sim_01: torch.Tensor,
    ) -> torch.Tensor:
        """Collapse caption-aware reward branches into one [0,1] score.

        Composition rules (n=300 real-data audit, post-P0-P3):
          numeric caption:
              0.55 * numerical_scores
            + 0.05 * executor_scores  (when has_executor)
            + 0.05 * kinematic_scores (when has_kinematic)
            + 0.35 * cos_sim_01
            -- executor weight dropped 0.15 -> 0.05 because the executor
            branch failed to discriminate matched vs mismatched in the
            audit (Δ 0.015). The released ~0.10 budget shifted to matching
            (cos_sim_01), which IS the dominant signal (Δ 0.116).
          direction-only / pure caption:
              cos_sim_01 only
              -- direction specs were too noisy to be useful as the sole
              signal (see audit log); deferring to matching alone is the
              safer trade until they can distinguish e.g. "walks left"
              from "walks forward" reliably.
        """
        B = numerical_scores.shape[0]
        sat = torch.zeros(B, device=numerical_scores.device,
                          dtype=numerical_scores.dtype)
        for i in range(B):
            has_num = bool(has_numerical[i].item() > 0)
            c = float(cos_sim_01[i].item())
            if has_num:
                n = float(numerical_scores[i].item())
                has_exec = bool(has_executor[i].item() > 0)
                has_kin = bool(has_kinematic[i].item() > 0)
                e = float(executor_scores[i].item()) if has_exec else 0.0
                k = float(kinematic_scores[i].item()) if has_kin else 0.0
                sat[i] = 0.55 * n + 0.05 * e + 0.05 * k + 0.35 * c
            else:
                # direction_only and pure both fall back to matching.
                sat[i] = c
        return sat.clamp(0.0, 1.0)

    def _compute_matching_score(
        self,
        text_emb: torch.Tensor,
        motion_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE reward. Also populates self._reward_stats."""
        text_emb_norm = F.normalize(text_emb, p=2, dim=-1)
        motion_emb_norm = F.normalize(motion_emb, p=2, dim=-1)

        sim_matrix = motion_emb_norm @ text_emb_norm.T
        B = sim_matrix.shape[0]
        positive_sim = sim_matrix.diag()

        if B > 1:
            logits = sim_matrix / self.tau
            scores = logits.diag() - torch.logsumexp(logits, dim=-1)
            mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
            negative_sim = (sim_matrix * mask).sum(dim=-1) / (B - 1)
        else:
            scores = torch.zeros(1, device=sim_matrix.device)
            negative_sim = torch.zeros(1, device=sim_matrix.device)

        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=-10.0)

        self._reward_stats = {
            'pos_sim_mean': positive_sim.mean().item(),
            'neg_sim_mean': negative_sim.mean().item(),
        }

        return scores


# ---------------------------------------------------------------------------
# Utility: standalone test
# ---------------------------------------------------------------------------

def test_reward_model(reward_model: GRPORewardModel):
    """Quick sanity check."""
    print("Testing GRPO Reward Model...")

    captions = [
        "a person walks forward",
        "a person jumps up and down three times",
        "a person takes four steps forward",
    ]

    motion_tokens = [
        torch.randint(0, 512, (64,)),
        torch.randint(0, 512, (48,)),
        torch.randint(0, 512, (32,)),
    ]

    rewards, components = reward_model.compute_reward(
        captions, motion_tokens, return_components=True,
    )

    print(f"Rewards: {rewards}")
    print(f"Matching (cos sim): {components['matching_scores']}")
    print(f"Physical scores: {components['physical_scores']}")
    print(f"Numerical scores: {components['numerical_scores']}")
    print(f"Has numerical: {components['has_numerical']}")
    print(f"Stats: {components['reward_stats']}")
    print("Test passed!")
    return rewards, components
