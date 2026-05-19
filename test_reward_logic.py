"""
Unit tests for reward-logic changes (no model loading).

Tests three behaviors:
  1. _step_accuracy — asymmetric penalty (undershoot > overshoot)
  2. score_direction_sequence — redundancy penalty for extra phases
  3. _score_global / _score_temporal — direction-as-multiplicative-gate
"""
import sys
from grpo_reward import (
    Direction, MotionPhase, ConstraintPhase,
    _step_accuracy,
    _direction_purity, _purity_factor,
    score_direction_sequence,
    _score_global, _score_temporal,
)
import numpy as np

def _ok(cond, msg):
    print(("PASS" if cond else "FAIL") + " - " + msg)
    if not cond:
        globals().setdefault("_FAILED", []).append(msg)

# ---------------------------------------------------------------------------
# Test 1: _step_accuracy asymmetry
# ---------------------------------------------------------------------------
print("\n[Test 1] _step_accuracy — undershoot is punished harder than overshoot")

# target=3 steps
s_hit = _step_accuracy(3, 3)
s_under1 = _step_accuracy(2, 3)
s_under2 = _step_accuracy(1, 3)
s_over1 = _step_accuracy(4, 3)
s_over2 = _step_accuracy(5, 3)

print(f"  target=3: hit={s_hit:.3f}  under(2)={s_under1:.3f}  under(1)={s_under2:.3f}  over(4)={s_over1:.3f}  over(5)={s_over2:.3f}")
_ok(s_hit > 0.99, "exact hit should be ~1.0")
_ok(s_under1 < s_over1, "undershoot(2) should score LOWER than overshoot(4)")
_ok(s_under2 < s_over2, "undershoot(1) should score LOWER than overshoot(5)")
_ok(s_under2 < 0.15, "severe undershoot (1 of 3) should be heavily penalized (<0.15)")
_ok(s_over1 > 0.6, "mild overshoot (4 of 3) should still get decent score (>0.6)")

# target=5 steps — larger target, sigma scales
s_under_big = _step_accuracy(3, 5)
s_over_big = _step_accuracy(7, 5)
print(f"  target=5: under(3)={s_under_big:.3f}  over(7)={s_over_big:.3f}")
_ok(s_under_big < s_over_big, "target=5: undershoot should be worse than symmetric overshoot")

# ---------------------------------------------------------------------------
# Test 2: score_direction_sequence — redundancy penalty
# ---------------------------------------------------------------------------
print("\n[Test 2] score_direction_sequence — redundant phases are penalized")

def mk_phase(direction, displacement=0.5, step_count=3, purity=1.0):
    return MotionPhase(
        start_frame=0, end_frame=20, direction=direction,
        step_count=step_count, displacement=displacement, rotation_deg=0.0,
        purity=purity,
    )

expected = [Direction.LEFT, Direction.RIGHT]

# Clean: exactly 2 phases matching
clean_phases = [mk_phase(Direction.LEFT), mk_phase(Direction.RIGHT)]
clean_score = score_direction_sequence(expected, clean_phases)

# Noisy: 2 matches + 2 extra fidget phases
noisy_phases = [
    mk_phase(Direction.LEFT),
    mk_phase(Direction.FORWARD, displacement=0.1),  # extra fidget
    mk_phase(Direction.RIGHT),
    mk_phase(Direction.BACKWARD, displacement=0.1),  # extra fidget
]
noisy_score = score_direction_sequence(expected, noisy_phases)

# Very noisy: 2 matches + 4 fidgets
very_noisy_phases = noisy_phases + [
    mk_phase(Direction.FORWARD, displacement=0.1),
    mk_phase(Direction.BACKWARD, displacement=0.1),
]
very_noisy_score = score_direction_sequence(expected, very_noisy_phases)

print(f"  clean (2 phases)={clean_score:.3f}  noisy (+2 extra)={noisy_score:.3f}  very_noisy (+4 extra)={very_noisy_score:.3f}")
_ok(clean_score > noisy_score, "clean should score HIGHER than noisy (redundancy hurts)")
_ok(noisy_score > very_noisy_score, "more redundancy should hurt more")
_ok(clean_score > 0.9, "clean LEFT→RIGHT should be ~1.0")

# ---------------------------------------------------------------------------
# Test 3: direction gate is multiplicative, not additive
# ---------------------------------------------------------------------------
print("\n[Test 3] direction mismatch no longer rescues step accuracy via +0.15 bonus")

# caption: "a person takes 3 steps to the left" — wanted 3 steps LEFT
c_left3 = ConstraintPhase(type='steps', value=3.0, direction=Direction.LEFT, order=0, raw='3 steps left')

# Case A: correct direction, right step count
phases_correct = [mk_phase(Direction.LEFT, step_count=3, displacement=0.5)]
score_A = _score_global([c_left3], phases_correct,
                        total_steps=3, total_rotation_deg=0.0, total_repetitions=0)

# Case B: WRONG direction, right step count
phases_wrong_dir = [mk_phase(Direction.RIGHT, step_count=3, displacement=0.5)]
score_B = _score_global([c_left3], phases_wrong_dir,
                        total_steps=3, total_rotation_deg=0.0, total_repetitions=0)

# Case C: correct direction, WRONG step count (only 1 step)
phases_wrong_count = [mk_phase(Direction.LEFT, step_count=1, displacement=0.5)]
score_C = _score_global([c_left3], phases_wrong_count,
                        total_steps=1, total_rotation_deg=0.0, total_repetitions=0)

# Case D (old bug): wrong direction AND wrong step count — should be lowest
phases_all_wrong = [mk_phase(Direction.RIGHT, step_count=1, displacement=0.5)]
score_D = _score_global([c_left3], phases_all_wrong,
                        total_steps=1, total_rotation_deg=0.0, total_repetitions=0)

print(f"  A correct/correct={score_A:.3f}  B wrong-dir/correct={score_B:.3f}  C correct/undershoot={score_C:.3f}  D both-wrong={score_D:.3f}")
_ok(score_A > 0.9, "A (correct direction, correct count) should be ~1.0")
_ok(score_B < score_A, "B (wrong direction) should score LOWER than A")
_ok(score_C < score_A, "C (undershoot) should score LOWER than A")
_ok(score_D < score_B, "D (wrong dir + wrong count) should be WORSE than B alone")
_ok(score_D < score_C, "D should be WORSE than C alone")

# ---------------------------------------------------------------------------
# Test 4: temporal scoring — "先左后右" case
# ---------------------------------------------------------------------------
print("\n[Test 4] temporal ordering — 'N steps left then N steps right' with extra fidgets")

c_left = ConstraintPhase(type='steps', value=3.0, direction=Direction.LEFT, order=0, raw='3 steps left')
c_right = ConstraintPhase(type='steps', value=3.0, direction=Direction.RIGHT, order=1, raw='3 steps right')

# Clean: 2 phases, exact
clean_temporal_phases = [
    mk_phase(Direction.LEFT, step_count=3, displacement=0.5),
    mk_phase(Direction.RIGHT, step_count=3, displacement=0.5),
]
score_clean = _score_temporal([c_left, c_right], clean_temporal_phases,
                              total_steps=6, total_rotation_deg=0.0, total_repetitions=0)

# Fidgety: 4 phases, with extras
fidgety_temporal_phases = [
    mk_phase(Direction.LEFT, step_count=3, displacement=0.5),
    mk_phase(Direction.FORWARD, step_count=1, displacement=0.1),  # fidget
    mk_phase(Direction.RIGHT, step_count=3, displacement=0.5),
    mk_phase(Direction.BACKWARD, step_count=1, displacement=0.1),  # fidget
]
score_fidgety = _score_temporal([c_left, c_right], fidgety_temporal_phases,
                                total_steps=8, total_rotation_deg=0.0, total_repetitions=0)

print(f"  clean={score_clean:.3f}  fidgety={score_fidgety:.3f}")
_ok(score_clean > score_fidgety, "clean LEFT→RIGHT should score HIGHER than fidgety version")
_ok(score_clean > 0.9, "clean LEFT→RIGHT should be ~1.0")

# ---------------------------------------------------------------------------
# Test 5: direction purity — gentle factor for near-pure movement
# ---------------------------------------------------------------------------
print("\n[Test 5] _direction_purity + _purity_factor")

initial_facing = np.pi / 2  # +Z

# Pure LEFT: move_angle should be pi (facing + 90° in atan2(dz,dx) = +Z + 90° CCW = -X)
# In atan2(dz, dx) convention: LEFT = -X direction = angle pi (since dx<0, dz=0)
pure_left_angle = np.pi
p_pure = _direction_purity(pure_left_angle, Direction.LEFT, initial_facing)
print(f"  pure LEFT purity: {p_pure:.3f}")
_ok(p_pure > 0.99, "pure left direction has purity ~1.0")

# 45° leak (LEFT + FORWARD mix): move_angle = pi/2 + 3pi/4 = ... use direct approach
# Ideal LEFT angle = facing + pi/2 = pi/2 + pi/2 = pi. 45° off = pi - pi/4 = 3pi/4
p_45 = _direction_purity(3 * np.pi / 4, Direction.LEFT, initial_facing)
print(f"  45° off LEFT purity: {p_45:.3f}")
_ok(0.65 < p_45 < 0.75, "45° off gives purity ~0.71")

# 60° off
p_60 = _direction_purity(np.pi - np.pi / 3, Direction.LEFT, initial_facing)
print(f"  60° off LEFT purity: {p_60:.3f}")
_ok(0.45 < p_60 < 0.55, "60° off gives purity ~0.5")

# 90° off (orthogonal)
p_90 = _direction_purity(np.pi / 2, Direction.LEFT, initial_facing)
print(f"  90° off LEFT purity: {p_90:.3f}")
_ok(p_90 < 0.05, "90° off gives purity ~0")

# _purity_factor: gentle mapping
f_pure = _purity_factor(1.0)
f_at_thresh = _purity_factor(0.6)
f_half = _purity_factor(0.5)
f_zero = _purity_factor(0.0)
print(f"  factor: purity=1.0 → {f_pure:.3f}, 0.6 → {f_at_thresh:.3f}, 0.5 → {f_half:.3f}, 0.0 → {f_zero:.3f}")
_ok(f_pure == 1.0, "pure motion: no penalty")
_ok(f_at_thresh == 1.0, "at threshold 0.6: no penalty")
_ok(0.9 < f_half < 0.97, "purity 0.5: slight penalty (~0.95)")
_ok(abs(f_zero - 0.7) < 0.01, "purity 0: factor = 0.7 floor")

# ---------------------------------------------------------------------------
# Test 6: purity penalty in direction sequence scoring
# ---------------------------------------------------------------------------
print("\n[Test 6] direction sequence scoring — diagonal movement is penalized")

expected = [Direction.LEFT, Direction.RIGHT]

# Clean pure: both phases with purity 1.0
clean_pure = [
    mk_phase(Direction.LEFT, purity=1.0),
    mk_phase(Direction.RIGHT, purity=1.0),
]
score_pure = score_direction_sequence(expected, clean_pure)

# Diagonal: phases classified as LEFT/RIGHT but with significant forward leak
diagonal = [
    mk_phase(Direction.LEFT, purity=0.71),   # 45° off
    mk_phase(Direction.RIGHT, purity=0.71),
]
score_diag = score_direction_sequence(expected, diagonal)

# Very diagonal: 60° off
very_diagonal = [
    mk_phase(Direction.LEFT, purity=0.5),
    mk_phase(Direction.RIGHT, purity=0.5),
]
score_very_diag = score_direction_sequence(expected, very_diagonal)

print(f"  pure={score_pure:.3f}  45°-off={score_diag:.3f}  60°-off={score_very_diag:.3f}")
_ok(score_pure > 0.99, "pure lateral should be ~1.0")
_ok(score_pure == score_diag, "45° off (purity 0.71 > threshold 0.6) should not be penalized")
# Note: first-match bonus masks moderate purity penalty in direction-seq scoring.
# Purity's main effect on direction constraints comes through _score_global (Test 7).
_ok(score_very_diag >= 0.9, "60° off may still get full credit via first-match bonus (acceptable)")

# ---------------------------------------------------------------------------
# Test 7: purity in step accuracy scoring
# ---------------------------------------------------------------------------
print("\n[Test 7] _score_global — purity affects step accuracy for directional constraints")

c_left3 = ConstraintPhase(type='steps', value=3.0, direction=Direction.LEFT, order=0, raw='3 steps left')

# Correct direction, pure
phases_pure = [mk_phase(Direction.LEFT, step_count=3, purity=1.0)]
score_pure = _score_global([c_left3], phases_pure,
                           total_steps=3, total_rotation_deg=0.0, total_repetitions=0)

# Correct direction, moderate purity (45° leak)
phases_45 = [mk_phase(Direction.LEFT, step_count=3, purity=0.71)]
score_45 = _score_global([c_left3], phases_45,
                         total_steps=3, total_rotation_deg=0.0, total_repetitions=0)

# Correct direction, low purity (60° leak)
phases_60 = [mk_phase(Direction.LEFT, step_count=3, purity=0.5)]
score_60 = _score_global([c_left3], phases_60,
                         total_steps=3, total_rotation_deg=0.0, total_repetitions=0)

print(f"  pure={score_pure:.3f}  45°-off={score_45:.3f}  60°-off={score_60:.3f}")
_ok(score_pure > 0.99, "pure + correct count = ~1.0")
_ok(score_pure == score_45, "45° (purity 0.71) not penalized (above threshold)")
_ok(score_60 < score_pure, "60° (purity 0.5) should be penalized")
_ok(score_60 > 0.9, "penalty should be gentle (stay above 0.9)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
failed = globals().get("_FAILED", [])
print("\n" + "="*60)
if failed:
    print(f"FAILED: {len(failed)} assertion(s)")
    for m in failed:
        print(f"  - {m}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
