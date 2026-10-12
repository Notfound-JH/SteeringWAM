import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import numpy as np

from blue_placement_verification import placement_sample_passes, strict_blue_placement_success
from verify_recovery_episode import verify_placement_gripper


SAMPLE = {
    "original_native_success": True,
    "object_lower": [0.25, -0.1, 0],
    "object_upper": [0.55, 0.1, 0.04],
    "bin_lower": [0.58, 0.12, 0],
    "bin_upper": [0.9, 0.4, 0.2],
    "ee_position": [0.4, 0, 0.3],
    "object_velocity": [0, 0, 0, 0, 0, 0],
    "table_contact_force": [0, 0, 1],
    "gripper_command": 0,
    "gripper_closure": 0,
}


def test_released_stable_outside_placement_passes():
    assert placement_sample_passes(SAMPLE)


@pytest.mark.parametrize("field,value", [
    ("original_native_success", False),
    ("object_upper", [0.60, 0.15, 0.04]),
    ("object_upper", [0.579, 0.15, 0.04]),
    ("ee_position", [0.4, 0, 0.2]),
    ("gripper_command", 1),
    ("gripper_closure", 0.1),
    ("table_contact_force", [0, 0, 0.09]),
    ("table_contact_force", [1, 0, 0.5]),
    ("object_velocity", [0.011, 0, 0, 0, 0, 0]),
    ("object_velocity", [0, 0, 0, 0, 0.051, 0]),
])
def test_native_label_alone_does_not_prove_placement(field, value):
    sample = copy.deepcopy(SAMPLE)
    sample[field] = value
    assert not placement_sample_passes(sample)


def test_nonfinite_sample_is_an_error():
    sample = {**SAMPLE, "object_velocity": [float("nan"), 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError, match="Nonfinite"):
        placement_sample_passes(sample)


@pytest.mark.parametrize("closure", [-0.0002684, 0.0, 0.79, 1.00003995])
def test_verifier_distinguishes_raw_joint_closure_from_clipped_observation(closure):
    robot = {
        "joint_names": ["panda_joint1", "finger_joint"],
        "measured_joint_pos": [0, closure * (np.pi / 4)],
        "gripper_pos": [float(np.clip(closure, 0, 1))],
    }
    verify_placement_gripper({"gripper_closure": closure}, robot)
    with pytest.raises(AssertionError):
        verify_placement_gripper({"gripper_closure": closure + 0.001}, robot)
    robot["gripper_pos"][0] += 0.001
    with pytest.raises(AssertionError):
        verify_placement_gripper({"gripper_closure": closure}, robot)


def test_strict_success_requires_fifteen_consecutive_stable_steps(monkeypatch):
    import torch

    world = SimpleNamespace(
        get_bbox=lambda name: (
            torch.tensor([[
                SAMPLE["object_lower" if name == "blue_hammer" else "bin_lower"],
                SAMPLE["object_upper" if name == "blue_hammer" else "bin_upper"],
            ]]), None
        ),
        get_contact_force=lambda *args, **kwargs: torch.tensor([0.0, 0.0, 1.0]),
    )
    module = ModuleType("robolab.core.world.world_state")
    module.get_world = lambda env: world
    monkeypatch.setitem(sys.modules, module.__name__, module)
    robot = SimpleNamespace(
        body_names=["base_link"], joint_names=["finger_joint"],
        data=SimpleNamespace(body_pos_w=torch.tensor([[SAMPLE["ee_position"]]]),
                             joint_pos=torch.zeros((1, 1))),
    )
    hammer = SimpleNamespace(data=SimpleNamespace(root_vel_w=torch.zeros((1, 6))))
    class Scene(dict):
        env_origins = torch.zeros((1, 3))

    env = SimpleNamespace(
        _blue_native_success_cfg=SimpleNamespace(
            func=lambda env: torch.tensor([True]), params={}
        ),
        scene=Scene(robot=robot, blue_hammer=hammer),
        action_manager=SimpleNamespace(action=torch.zeros((1, 8))),
        episode_length_buf=torch.zeros(1, dtype=torch.int64), device="cpu",
    )
    for step in range(1, 16):
        env.episode_length_buf[0] = step
        assert bool(strict_blue_placement_success(env)[0]) == (step == 15)
    hammer.data.root_vel_w[0, 0] = 0.02
    env.episode_length_buf[0] = 16
    assert not bool(strict_blue_placement_success(env)[0])
    assert env._blue_placement_sample["stable_steps"] == 0
