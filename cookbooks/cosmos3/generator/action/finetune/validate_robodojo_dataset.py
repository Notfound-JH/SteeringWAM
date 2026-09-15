#!/usr/bin/env python3
"""Validate the RoboDojo LeRobot v3 contract used by the Cosmos recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ACTION_KEY = "action"
STATE_KEY = "observation.state"
EXPECTED_VECTOR_NAMES = (
    "l_x",
    "l_y",
    "l_z",
    "l_w",
    "l_wx",
    "l_wy",
    "l_wz",
    "l_g",
    "r_x",
    "r_y",
    "r_z",
    "r_w",
    "r_wx",
    "r_wy",
    "r_wz",
    "r_g",
)
EXPECTED_VECTOR_SIZE = len(EXPECTED_VECTOR_NAMES)


def feature_shape(feature: dict[str, Any]) -> list[int] | None:
    shape = feature.get("shape")
    if not isinstance(shape, list) or not all(isinstance(value, int) for value in shape):
        return None
    return shape


def require_vector_feature(features: dict[str, Any], key: str) -> dict[str, Any]:
    feature = features.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"missing feature: {key}")
    shape = feature_shape(feature)
    if shape != [EXPECTED_VECTOR_SIZE]:
        raise ValueError(f"{key} must have shape [{EXPECTED_VECTOR_SIZE}], got {shape}")
    if feature.get("names") != [list(EXPECTED_VECTOR_NAMES)]:
        raise ValueError(f"{key} has unexpected dimension names: {feature.get('names')!r}")
    return feature


def validate_dataset(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"missing LeRobot metadata: {info_path}")

    with info_path.open(encoding="utf-8") as info_file:
        info = json.load(info_file)

    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{info_path} does not contain a features object")

    missing_cameras = [key for key in EXPECTED_CAMERAS if key not in features]
    if missing_cameras:
        raise ValueError(f"missing camera features: {', '.join(missing_cameras)}")

    action_feature = require_vector_feature(features, ACTION_KEY)
    state_feature = require_vector_feature(features, STATE_KEY)

    fps = info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"invalid fps: {fps!r}")

    return {
        "root": str(root.resolve()),
        "fps": fps,
        "action_key": ACTION_KEY,
        "action_shape": feature_shape(action_feature),
        "action_names": list(EXPECTED_VECTOR_NAMES),
        "state_key": STATE_KEY,
        "state_shape": feature_shape(state_feature),
        "camera_keys": list(EXPECTED_CAMERAS),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()

    try:
        summary = validate_dataset(args.dataset_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()