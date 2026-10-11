"""Verify recorded execution, not merely the episode's success label."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def verify(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    for name in ("trajectory", "events", "episode_video"):
        reference = manifest[name]
        path = Path(reference["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    actual = [event for event in events if event["event_type"] == "actual_execution_response"]
    if not actual or [event["episode_step"] for event in actual] != list(range(1, len(actual) + 1)):
        raise ValueError("Missing or discontinuous actual execution")
    for event in actual:
        if event["events"]["episode_step"] != event["episode_step"]:
            raise ValueError("Simulator step discontinuity")
    with h5py.File(directory / "trajectory.hdf5", "r") as handle:
        recorded = handle["data/demo_0/actions"][:]
    np.testing.assert_array_equal(recorded, np.asarray([event["actual_action"] for event in actual]))
    for result_path in sorted((directory / "segments").glob("*/result.json")):
        result = json.loads(result_path.read_text())
        predicted = np.load(result["predicted_actions"]["path"])
        executed = np.load(result["executed_actions"]["path"])
        np.testing.assert_array_equal(executed, predicted[:len(executed)])
        future = np.load(result["predicted_future"]["array"]["path"])
        if predicted.shape != (32, 8) or future.ndim != 4 or future.dtype != np.uint8:
            raise ValueError("Invalid model output shape or dtype")
    failure_indices = [i for i, event in enumerate(events) if event["event_type"] == "failure_detected"]
    recovery_indices = [
        i for i, event in enumerate(events)
        if event["event_type"] == "actual_execution_response"
        and event["action_source"] in {"agent_correction", "agent_condition"}
    ]
    handoff_indices = [i for i, event in enumerate(events) if event["event_type"] == "control_handed_back_to_wam"]
    continuation_indices = [
        i for i, event in enumerate(events)
        if event["event_type"] == "actual_execution_response"
        and event["action_source"] == "wam_continuation"
    ]
    ordered_chain = bool(
        failure_indices and recovery_indices and handoff_indices and continuation_indices
        and failure_indices[0] < recovery_indices[0] < handoff_indices[0] < continuation_indices[0]
    )
    accepted = bool(
        ordered_chain
        and actual[-1]["action_source"] == "wam_continuation"
        and actual[-1]["events"]["native_success"]
        and manifest["native_success"]
        and not manifest["mock_execution"]
    )
    if accepted != manifest["acceptance_success"]:
        raise ValueError("Acceptance label disagrees with actual event sequence")
    return {
        "recording_verified": True,
        "actual_steps": len(actual),
        "agent_steps": sum(event["action_source"] == "agent_correction" for event in actual),
        "wam_continuation_steps": sum(event["action_source"] == "wam_continuation" for event in actual),
        "attempts": manifest["correction_attempts_preserved"],
        "native_success": manifest["native_success"],
        "acceptance_success": accepted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    result = verify(args.directory.resolve())
    (args.directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 2 if args.require_success and not result["acceptance_success"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
