import copy

import pytest

from verify_recovery_episode import proposal_matches_validation


def proposal():
    return {
        "diagnosis": "Recorded decision",
        "program": [{
            "op": "move_ee",
            "position": [0.5, 0.3, 0.2],
            "quaternion": [0.9906497177818776, 0.011517256703152338,
                           0.13593428078631842, -0.0015364777337188295],
            "gripper": 0.0,
            "steps": 1,
        }],
    }


def test_accepts_only_quaternion_normalization_roundoff():
    saved = proposal()
    checked = copy.deepcopy(saved)
    checked["program"][0]["quaternion"][0] += 1e-16
    original = copy.deepcopy(saved)
    assert proposal_matches_validation(saved, checked)
    assert saved == original


@pytest.mark.parametrize("field,value", [
    ("position", [0.5, 0.3, 0.2000000000000001]),
    ("quaternion", [1.0, 0.0, 0.0, 0.0]),
    ("gripper", 1.0),
    ("steps", 2),
])
def test_rejects_other_operation_changes(field, value):
    saved = proposal()
    checked = copy.deepcopy(saved)
    checked["program"][0][field] = value
    assert not proposal_matches_validation(saved, checked)


def test_rejects_metadata_or_program_length_changes():
    saved = proposal()
    checked = copy.deepcopy(saved)
    checked["diagnosis"] = "Different decision"
    assert not proposal_matches_validation(saved, checked)
    checked = copy.deepcopy(saved)
    checked["program"].append(copy.deepcopy(checked["program"][0]))
    assert not proposal_matches_validation(saved, checked)


def test_accepts_identical_condition_without_program():
    assert proposal_matches_validation({"condition": "Open"}, {"condition": "Open"})
