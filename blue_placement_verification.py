"""Stricter, opt-in completion check for the single-blue-hammer experiment."""

from __future__ import annotations

import numpy as np


STABLE_STEPS = 15


def placement_sample_passes(sample: dict) -> bool:
    lower = np.asarray(sample["object_lower"])
    upper = np.asarray(sample["object_upper"])
    bin_lower = np.asarray(sample["bin_lower"])
    bin_upper = np.asarray(sample["bin_upper"])
    ee = np.asarray(sample["ee_position"])
    velocity = np.asarray(sample["object_velocity"])
    force = np.asarray(sample["table_contact_force"])
    if velocity.shape != (6,) or any(
        value.shape != (3,) for value in (lower, upper, bin_lower, bin_upper, ee, force)
    ):
        raise ValueError("Invalid placement sample shape")
    values = np.concatenate((lower, upper, bin_lower, bin_upper, ee, velocity, force))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite placement verification geometry or velocity")
    outside = np.any(
        (upper[:2] <= bin_lower[:2] - 0.005)
        | (lower[:2] >= bin_upper[:2] + 0.005)
    )
    ee_distance = np.linalg.norm(np.maximum(np.maximum(lower - ee, ee - upper), 0))
    return bool(
        sample["original_native_success"]
        and np.linalg.norm(force) >= 0.1
        and force[2] >= np.linalg.norm(force) / np.sqrt(2)
        and outside
        and sample["gripper_command"] < 0.5
        and sample["gripper_closure"] < 0.1
        and ee_distance >= 0.20
        and np.linalg.norm(velocity[:3]) <= 0.01
        and np.linalg.norm(velocity[3:]) <= 0.05
    )


def strict_blue_placement_success(env):
    import torch
    from robolab.core.world.world_state import get_world

    native = env._blue_native_success_cfg
    original = native.func(env, **native.params)
    world = get_world(env)
    corners, _ = world.get_bbox("blue_hammer")
    bin_corners, _ = world.get_bbox("left_bin")
    corners, bin_corners = corners[0], bin_corners[0]
    robot = env.scene["robot"]
    body = robot.body_names.index("base_link")
    joint = robot.joint_names.index("finger_joint")
    sample = {
        "original_native_success": bool(original[0]),
        "object_lower": corners.amin(dim=0).cpu().tolist(),
        "object_upper": corners.amax(dim=0).cpu().tolist(),
        "bin_lower": bin_corners.amin(dim=0).cpu().tolist(),
        "bin_upper": bin_corners.amax(dim=0).cpu().tolist(),
        "ee_position": (
            robot.data.body_pos_w[0, body] - env.scene.env_origins[0]
        ).cpu().tolist(),
        "object_velocity": env.scene["blue_hammer"].data.root_vel_w[0].cpu().tolist(),
        "table_contact_force": world.get_contact_force(
            "blue_hammer", "table", env_id=0
        ).cpu().tolist(),
        "gripper_command": float(env.action_manager.action[0, -1]),
        "gripper_closure": float(robot.data.joint_pos[0, joint]) / (np.pi / 4),
    }
    previous = getattr(env, "_blue_placement_sample", {})
    step = int(env.episode_length_buf[0])
    passed = placement_sample_passes(sample)
    count = previous.get("stable_steps", 0) if previous.get("episode_step") == step - 1 else 0
    sample.update(
        sample_passed=passed,
        stable_steps=count + 1 if passed else 0,
        required_stable_steps=STABLE_STEPS,
        episode_step=step,
    )
    env._blue_placement_sample = sample
    return torch.tensor([sample["stable_steps"] >= STABLE_STEPS], device=env.device)
