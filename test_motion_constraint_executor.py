import torch

from motion_constraint_executor import MotionConstraintExecutor
from motion_step_detector import detect_steps


def _toy_motion():
    T = 40
    motion_raw = torch.zeros(T, 263)
    joints = torch.zeros(T, 22, 3)
    foot_contact = torch.zeros(T, 4)

    # Stable body frame: left side along +X, forward from cross(left, up) -> +Z.
    joints[:, 16, 0] = 0.3
    joints[:, 17, 0] = -0.3
    joints[:, 1, 0] = 0.2
    joints[:, 2, 0] = -0.2
    joints[:, 15, 1] = 1.6

    # Pelvis walks forward before the turn.
    joints[:, 0, 2] = torch.linspace(0.0, 0.6, T)

    # Two foot-contact starts.
    foot_contact[5:8, 0] = 1.0
    foot_contact[18:21, 2] = 1.0

    # Hands close briefly, then both hands go up.
    joints[:, 20] = torch.tensor([0.25, 1.1, 0.0])
    joints[:, 21] = torch.tensor([-0.25, 1.1, 0.0])
    joints[25:30, 20] = torch.tensor([0.02, 1.75, 0.0])
    joints[25:30, 21] = torch.tensor([-0.02, 1.75, 0.0])

    # Left hand touches head; right foot raises.
    joints[12:14, 20] = torch.tensor([0.0, 1.58, 0.0])
    joints[12:14, 15] = torch.tensor([0.0, 1.6, 0.0])
    joints[30:34, 11, 1] = 0.12

    # Left turn after frame 24. motion_raw[:, 0] uses half-angle convention.
    motion_raw[24:34, 0] = torch.deg2rad(torch.tensor(4.0)) / 2.0
    return motion_raw, foot_contact, joints


def _phase_toy_motion():
    T = 80
    motion_raw = torch.zeros(T, 263)
    joints = torch.zeros(T, 22, 3)
    foot_contact = torch.zeros(T, 4)

    # Body frame: left along +X, forward along +Z.
    joints[:, 16, 0] = 0.3
    joints[:, 17, 0] = -0.3
    joints[:, 1, 0] = 0.2
    joints[:, 2, 0] = -0.2
    joints[:, 15, 1] = 1.6

    # First phase moves forward, second phase moves backward.
    joints[:45, 0, 2] = torch.linspace(0.0, 0.9, 45)
    joints[45:, 0, 2] = torch.linspace(0.9, 0.55, T - 45)

    # Three steps during forward phase, one during backward phase.
    foot_contact[8:11, 0] = 1.0
    foot_contact[20:23, 2] = 1.0
    foot_contact[33:36, 0] = 1.0
    foot_contact[58:61, 2] = 1.0
    return motion_raw, foot_contact, joints


def test_signal_count_temporal_and_composite():
    motion_raw, foot_contact, joints = _toy_motion()
    executor = MotionConstraintExecutor()
    results = executor.evaluate(
        motion_raw=motion_raw,
        foot_contact=foot_contact,
        joints=joints,
        constraints=[
            {
                "id": "hands_close",
                "kind": "signal",
                "ref": {
                    "type": "signal",
                    "name": "dist",
                    "args": {"a": "l_hand", "b": "r_hand"},
                },
                "reduce": "min",
                "op": "lt",
                "value": 0.1,
            },
            {
                "id": "step_count_eq_2",
                "kind": "count",
                "ref": {
                    "type": "template",
                    "name": "step",
                    "args": {"foot": "any"},
                },
                "op": "eq",
                "value": 2,
            },
            {
                "id": "steps_before_hands_up",
                "kind": "temporal",
                "relation": "before",
                "lhs": {
                    "type": "template",
                    "name": "step",
                    "args": {"foot": "any"},
                },
                "rhs": {
                    "type": "state",
                    "name": "hands_up",
                    "args": {"mode": "both"},
                },
            },
            {
                "id": "forward_before_turn_left",
                "kind": "temporal_composite",
                "lhs": {
                    "ref": {
                        "type": "signal",
                        "name": "directional_displacement",
                        "args": {
                            "entity": "pelvis",
                            "direction": "forward",
                            "frame": "body",
                        },
                    },
                    "evidence": {
                        "measure": "displacement",
                        "op": "ge",
                        "value": 0.25,
                    },
                },
                "rhs": {
                    "ref": {
                        "type": "template",
                        "name": "turn_left",
                        "args": {"min_angle_deg": 20},
                    },
                    "evidence": {
                        "measure": "duration",
                        "op": "ge",
                        "value": 0.3,
                    },
                },
                "relation": {
                    "name": "evidence_before",
                    "rhs_anchor": "start",
                    "measure": "lhs_pre_anchor_displacement",
                    "op": "ge",
                    "value": 0.15,
                },
            },
            {
                "id": "touch_head_present",
                "kind": "count",
                "ref": {
                    "type": "template",
                    "name": "touch_head",
                    "args": {"hand": "left", "threshold": 0.18},
                },
                "op": "ge",
                "value": 1,
            },
            {
                "id": "raise_right_foot",
                "kind": "count",
                "ref": {
                    "type": "template",
                    "name": "raise_foot",
                    "args": {"foot": "right", "threshold": 0.08},
                },
                "op": "ge",
                "value": 1,
            },
            {
                "id": "avoid_right_turn",
                "kind": "absence",
                "ref": {
                    "type": "template",
                    "name": "turn_right",
                    "args": {"min_angle_deg": 20},
                },
                "value": 0,
            },
        ],
    )

    assert [result.constraint_id for result in results] == [
        "hands_close",
        "step_count_eq_2",
        "steps_before_hands_up",
        "forward_before_turn_left",
        "touch_head_present",
        "raise_right_foot",
        "avoid_right_turn",
    ]
    assert all(result.violation == 0.0 for result in results)


def test_caption_to_executor_specs_cover_common_actions():
    from grpo_reward import caption_to_executor_specs, parse_constraints_regex, constraints_to_executor_specs

    clap_specs = caption_to_executor_specs("clap twice")
    assert any(spec["id"] == "clap_count" and spec["value"] == 2.0 for spec in clap_specs)

    touch_specs = caption_to_executor_specs("touch the head with the left hand")
    assert any(
        spec["id"] == "touch_head_present"
        and spec["ref"]["args"]["hand"] == "left"
        for spec in touch_specs
    )

    raise_specs = caption_to_executor_specs("raise the right foot")
    assert any(
        spec["id"] == "raise_foot_present"
        and spec["ref"]["args"]["foot"] == "right"
        for spec in raise_specs
    )

    parsed = parse_constraints_regex("walk forward then turn left")
    specs = constraints_to_executor_specs(parsed, "walk forward then turn left")
    assert any(spec["kind"] == "temporal_composite" for spec in specs)
    assert any(spec["id"] == "avoid_extra_right_turn" for spec in specs)


def test_direction_phase_count_constraints():
    motion_raw, foot_contact, joints = _phase_toy_motion()
    executor = MotionConstraintExecutor()
    results = executor.evaluate(
        motion_raw=motion_raw,
        foot_contact=foot_contact,
        joints=joints,
        constraints=[
            {
                "id": "forward_three_steps",
                "kind": "phase_count",
                "phase_ref": {
                    "type": "template",
                    "name": "direction_phase",
                    "args": {
                        "direction": "forward",
                        "frame": "body",
                        "min_displacement": 0.2,
                        "purity_threshold": 0.45,
                    },
                },
                "count_ref": {
                    "type": "template",
                    "name": "step",
                    "args": {"foot": "any"},
                },
                "op": "eq",
                "value": 3,
                "tolerance": 0.0,
            },
            {
                "id": "backward_one_step",
                "kind": "phase_count",
                "phase_ref": {
                    "type": "template",
                    "name": "direction_phase",
                    "args": {
                        "direction": "backward",
                        "frame": "body",
                        "min_displacement": 0.1,
                        "purity_threshold": 0.45,
                    },
                },
                "count_ref": {
                    "type": "template",
                    "name": "step",
                    "args": {"foot": "any"},
                },
                "op": "eq",
                "value": 1,
                "tolerance": 0.0,
            },
        ],
    )
    assert [result.measured_value for result in results] == [3.0, 1.0]
    assert all(result.violation == 0.0 for result in results)


def test_hybrid_step_detector_validates_contact_and_landings():
    T = 32
    joints = torch.zeros(T, 22, 3)
    foot_contact = torch.zeros(T, 4)

    # Valid contact: the left foot is near the floor and almost stationary.
    foot_contact[5:8, 0] = 1.0

    # Invalid contact: the right foot is marked down while floating.
    foot_contact[12:15, 2] = 1.0
    joints[12:15, 11, 1] = 0.2

    detected = detect_steps(joints, foot_contact)
    assert detected.count == 1
    assert detected.events[0].source == "contact"
    assert detected.consistency_penalty > 0.0

    # No contact labels, but the left foot clearly swings up and lands.
    landing_contact = torch.zeros(T, 4)
    landing_joints = torch.zeros(T, 22, 3)
    landing_joints[8:13, 10, 1] = torch.tensor([0.12, 0.16, 0.12, 0.07, 0.0])
    landing_joints[8:13, 10, 2] = torch.tensor([0.0, 0.03, 0.06, 0.08, 0.09])
    landing_joints[:, 11, 1] = 0.0

    landing_detected = detect_steps(landing_joints, landing_contact)
    assert landing_detected.count == 1
    assert landing_detected.events[0].source == "landing"


def test_diagonal_direction_parser_and_specs():
    from grpo_reward import (
        Direction,
        constraints_to_executor_specs,
        parse_constraints_regex,
    )

    parsed = parse_constraints_regex("walk left-forward, then walk right-forward")
    assert parsed.direction_sequence == [Direction.LEFT_FORWARD, Direction.RIGHT_FORWARD]
    specs = constraints_to_executor_specs(parsed, "walk left-forward, then walk right-forward")
    assert any(spec["id"] == "dir_phase_left_forward_0" for spec in specs)
    assert any(spec["id"] == "dir_phase_right_forward_1" for spec in specs)
    assert not any("turn_right" in str(spec) for spec in specs)

    parsed_steps = parse_constraints_regex(
        "walk forward three steps, then walk backward one step"
    )
    step_specs = constraints_to_executor_specs(
        parsed_steps,
        "walk forward three steps, then walk backward one step",
    )
    assert any(spec["kind"] == "phase_count" and spec["value"] == 3.0 for spec in step_specs)
    assert any(spec["kind"] == "phase_count" and spec["value"] == 1.0 for spec in step_specs)
