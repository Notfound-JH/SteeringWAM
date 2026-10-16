import math

import numpy as np
import pytest

from recovery_protocol import (
    PROTOCOL_VERSION,
    SafetyLimits,
    compile_program,
    episode_duration_for_steps,
    parse_control_code,
    validate_assessment,
    validate_failure_decision,
    validate_recovery_proposal,
)


CURRENT_POSITION = [0.45, 0.0, 0.35]
CURRENT_QUATERNION = [1.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("steps,step_dt", [
    (1, 1 / 15), (16, 1 / 15), (600, 1 / 15),
    (2000, 1 / 15), (2000, 1 / 30), (2000, 0.1),
])
def test_exact_native_control_step_horizon(steps, step_dt):
    seconds = episode_duration_for_steps(steps, step_dt)
    assert math.ceil(seconds / step_dt) == steps
    assert math.isclose(seconds, steps * step_dt, rel_tol=1e-15)


@pytest.mark.parametrize("steps,step_dt", [(0, 1 / 15), (True, 1 / 15), (2000, 0), (2000, float("inf"))])
def test_reject_invalid_control_step_horizon(steps, step_dt):
    with pytest.raises(ValueError):
        episode_duration_for_steps(steps, step_dt)


def proposal(**overrides):
    value = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "token",
        "diagnosis": "The gripper stopped above the fruit.",
        "recovery_goal": "Move to a graspable pre-contact pose.",
        "correction_mode": "joint",
        "condition": "Align above the nearest orange and close only after contact.",
        "wam_execute_steps": 4,
        "program": [
            {"op": "move_ee", "position": [0.48, 0.0, 0.35], "steps": 3},
            {"op": "gripper", "value": 1, "steps": 2},
            {"op": "hold", "steps": 1},
        ],
    }
    value.update(overrides)
    return value


def test_validate_and_compile_joint_proposal():
    limits = SafetyLimits()
    validated = validate_recovery_proposal(
        proposal(), "token", 0, CURRENT_POSITION, limits
    )
    actions = compile_program(
        validated["program"], CURRENT_POSITION, CURRENT_QUATERNION, [0.08]
    )
    assert actions.shape == (6, 8)
    np.testing.assert_allclose(actions[2, :3], [0.48, 0.0, 0.35])
    assert not actions[0, -1]
    assert np.all(actions[3:, -1] == 1)
    np.testing.assert_allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1)


def test_condition_and_action_modes_are_disjoint():
    limits = SafetyLimits()
    condition = proposal(correction_mode="condition")
    condition.pop("program")
    validated = validate_recovery_proposal(
        condition, "token", 0, CURRENT_POSITION, limits
    )
    assert validated["correction_mode"] == "condition" and "program" not in validated

    action = proposal(correction_mode="action")
    action.pop("condition")
    action.pop("wam_execute_steps")
    validated = validate_recovery_proposal(
        action, "token", 0, CURRENT_POSITION, limits
    )
    assert validated["correction_mode"] == "action" and "condition" not in validated

    with pytest.raises(ValueError, match="cannot include"):
        validate_recovery_proposal(
            proposal(correction_mode="action"), "token", 0, CURRENT_POSITION, limits
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"request_id": "stale"}, "request_id"),
        ({"correction_mode": "shell"}, "correction_mode"),
        ({"program": [{"op": "exec", "steps": 1}]}, "op"),
        ({"program": [{"op": "move_ee", "position": [2, 0, 0.3]}]}, "workspace"),
        ({"program": [{"op": "move_ee", "position": [0.7, 0, 0.35]}]}, "segment"),
        ({"program": [{"op": "hold", "steps": 49}]}, "steps"),
        ({"program": [{"op": "gripper", "value": 0.4}]}, "exactly"),
    ],
)
def test_rejects_unsafe_or_invalid_proposals(change, message):
    value = proposal(**change)
    with pytest.raises(ValueError, match=message):
        validate_recovery_proposal(
            value, "token", 0, CURRENT_POSITION, SafetyLimits()
        )


def test_rejects_attempt_four_and_bad_quaternion():
    with pytest.raises(ValueError, match="limit"):
        validate_recovery_proposal(
            proposal(), "token", 3, CURRENT_POSITION, SafetyLimits()
        )
    value = proposal(program=[
        {
            "op": "move_ee",
            "position": [0.46, 0, 0.35],
            "quaternion": [2, 0, 0, 0],
        }
    ])
    with pytest.raises(ValueError, match="normalized"):
        validate_recovery_proposal(
            value, "token", 0, CURRENT_POSITION, SafetyLimits()
        )


def test_failure_and_assessment_require_current_token_and_evidence():
    failure = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "token",
        "failure_type": "missed_grasp",
        "evidence": "The fingers closed beside the fruit.",
        "subgoal": "grasp the orange",
    }
    assert validate_failure_decision(failure, "token")["subgoal"] == "grasp the orange"
    assessment = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "next",
        "local_recovery_success": False,
        "evidence": "The EE moved closer but remains laterally offset.",
    }
    assert not validate_assessment(assessment, "next")["local_recovery_success"]
    with pytest.raises(ValueError, match="Stale"):
        validate_failure_decision(failure, "other")
    assessment["local_recovery_success"] = "yes"
    with pytest.raises(ValueError, match="boolean"):
        validate_assessment(assessment, "next")


def test_literal_control_code_compiles_without_execution():
    code = "controller.move_ee(position=[0.48, 0.0, 0.35], steps=3)\ncontroller.gripper(value=0, steps=2)"
    value = proposal(control_code=code)
    value.pop("program")
    validated = validate_recovery_proposal(value, "token", 0, CURRENT_POSITION, SafetyLimits())
    assert validated["control_code"] == code
    assert len(compile_program(validated["program"], CURRENT_POSITION, CURRENT_QUATERNION, [0])) == 5


@pytest.mark.parametrize("code", [
    "import os", "controller.reset()", "controller.hold(steps=1 + 1)",
    "controller.hold(steps=__import__('os').getpid())",
    "controller.hold(**{'steps': 1})", "env.step([0])",
    "while True: controller.hold(steps=1)",
])
def test_control_code_cannot_escape_literal_api(code):
    with pytest.raises((ValueError, SyntaxError)):
        parse_control_code(code)


def test_rotation_is_bounded_before_execution():
    with pytest.raises(ValueError, match="rotation"):
        compile_program(
            [{"op": "move_ee", "position": CURRENT_POSITION, "quaternion": [0, 1, 0, 0], "steps": 16}],
            CURRENT_POSITION, CURRENT_QUATERNION, [0],
        )
