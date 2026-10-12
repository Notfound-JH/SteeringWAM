"""Verify recorded execution, not merely the episode's success label."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from recovery_protocol import SafetyLimits, compile_program, validate_recovery_proposal
from blue_placement_verification import STABLE_STEPS, placement_sample_passes


def verify_agent_programs(manifest: dict, events: list[dict], actual: list[dict]) -> None:
    if any(event["action_source"] != "agent_control" for event in actual):
        raise ValueError("Agent-only baseline contains non-Agent execution")
    proposals = [event for event in events if event["event_type"] == "agent_control_proposal"]
    if len(proposals) != manifest["control_programs"]:
        raise ValueError("Agent control program count mismatch")
    authorized = [event for event in events if event["event_type"] == "action_authorized"]
    if len(authorized) != len(actual):
        raise ValueError("Missing Agent action authorization")
    for intent, response in zip(authorized, actual):
        np.testing.assert_array_equal(intent["authorized_action"], response["actual_action"])
    initial = next(event["state"] for event in events if event["event_type"] == "session_started")
    covered = 0
    limits = SafetyLimits(**manifest["safety_limits"])
    for number, event in enumerate(proposals, 1):
        if event["program_index"] != number or event["episode_step"] != covered:
            raise ValueError("Discontinuous Agent control programs")
        reference = event["proposal"]
        path = Path(reference["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("Agent proposal checksum mismatch")
        proposal = json.loads(path.read_text())
        if proposal["correction_mode"] != "action":
            raise ValueError("Agent-only baseline contains a non-action proposal")
        state = actual[covered - 1]["robot_state"] if covered else initial
        submitted = (
            {key: value for key, value in proposal.items() if key != "program"}
            if "control_code" in proposal else proposal
        )
        validated = validate_recovery_proposal(
            submitted, proposal["request_id"], 0, state["ee_pos"], limits
        )
        if validated["program"] != proposal["program"]:
            raise ValueError("Recorded program differs from its literal control code")
        proposal = validated
        compiled = compile_program(
            proposal["program"], state["ee_pos"], state["ee_quat"], state["gripper_pos"], limits
        )
        result = json.loads((path.parent / "result.json").read_text())
        count = result["executed_steps"]
        if not 1 <= count <= len(compiled):
            raise ValueError("Invalid Agent executed length")
        responses = actual[covered:covered + count]
        np.testing.assert_array_equal(
            np.load(path.parent / "compiled_cartesian_targets.npy"), compiled
        )
        np.testing.assert_array_equal(
            [item["cartesian_intent"] for item in responses], compiled[:count]
        )
        np.testing.assert_array_equal(
            np.load(result["executed_actions"]["path"]),
            [item["actual_action"] for item in responses],
        )
        covered += count
    if covered != len(actual):
        raise ValueError("Agent execution is not covered by bounded programs")


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
    agent_only = manifest.get("experiment_mode") == "agent_control"
    if agent_only:
        verify_agent_programs(manifest, events, actual)
    agent_success = bool(
        agent_only and actual[-1]["events"]["native_success"]
        and manifest["native_success"] and not manifest["mock_execution"]
    )
    if agent_success != manifest.get("agent_control_success", False):
        raise ValueError("Agent baseline success label disagrees with execution")
    stable_placement = False
    if manifest.get("success_criterion") == "native_and_stable_blue_placement":
        count = 0
        for event in actual:
            sample = event["events"]["placement_verification"]
            with np.load(event["observation"]["scene"]["path"]) as scene:
                np.testing.assert_array_equal(
                    sample["object_velocity"], scene["rigid_object/blue_hammer/root_velocity"][0]
                )
            np.testing.assert_allclose(
                sample["gripper_closure"], event["robot_state"]["gripper_pos"][0], atol=1e-6
            )
            if sample["gripper_command"] != event["actual_action"][-1]:
                raise ValueError("Placement gripper command disagrees with execution")
            passed = placement_sample_passes(sample)
            count = count + 1 if passed else 0
            if sample["sample_passed"] != passed or sample["stable_steps"] != count:
                raise ValueError("Placement stability history disagrees with measured samples")
            if sample["episode_step"] != event["episode_step"]:
                raise ValueError("Placement sample step mismatch")
        stable_placement = count >= STABLE_STEPS
        if bool(manifest["native_success"]) != stable_placement:
            raise ValueError("Strict completion label disagrees with placement history")
    return {
        "recording_verified": True,
        "actual_steps": len(actual),
        "agent_steps": sum(event["action_source"] == "agent_correction" for event in actual),
        "wam_continuation_steps": sum(event["action_source"] == "wam_continuation" for event in actual),
        "attempts": manifest["correction_attempts_preserved"],
        "native_success": manifest["native_success"],
        "acceptance_success": accepted,
        "agent_control_steps": sum(event["action_source"] == "agent_control" for event in actual),
        "agent_control_success": agent_success,
        "stable_blue_placement_verified": stable_placement,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--require-agent-success", action="store_true")
    args = parser.parse_args()
    result = verify(args.directory.resolve())
    (args.directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 2 if (
        args.require_success and not result["acceptance_success"]
        or args.require_agent_success and not result["agent_control_success"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
