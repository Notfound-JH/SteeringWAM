"""Verify recorded execution, not merely the episode's success label."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from recovery_protocol import (
    LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION, RECOVERY_EVALUATION_STEPS,
    SafetyLimits, compile_program, validate_assessment, validate_condition_change,
    validate_recovery_proposal,
)
from blue_placement_verification import STABLE_STEPS, placement_sample_passes


def verify_placement_gripper(sample: dict, robot_state: dict) -> None:
    index = robot_state["joint_names"].index("finger_joint")
    raw_closure = robot_state["measured_joint_pos"][index] / (np.pi / 4)
    np.testing.assert_allclose(sample["gripper_closure"], raw_closure, atol=1e-6)
    np.testing.assert_allclose(
        np.clip(raw_closure, 0, 1), robot_state["gripper_pos"][0], atol=1e-6
    )


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
            submitted, proposal["request_id"], 0, state["ee_pos"], limits,
            protocol_version=manifest.get("protocol_version", PROTOCOL_VERSION),
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


def verify_comparison_contract(manifest: dict, events: list[dict], actual: list[dict]) -> dict:
    if manifest.get("protocol_version", LEGACY_PROTOCOL_VERSION) == LEGACY_PROTOCOL_VERSION:
        return {"comparison_contract_verified": None}
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported recovery protocol version")
    if manifest.get("experiment_mode") == "agent_control":
        return {"comparison_contract_verified": None}
    if manifest["recovery_evaluation_steps"] != RECOVERY_EVALUATION_STEPS:
        raise ValueError("Unexpected recovery evaluation window")
    previous = None
    conditions_before = {}
    expected_changes = []
    for event in events:
        conditions_before[event["event_index"]] = previous
        if event["event_type"] != "wam_prediction":
            continue
        change = event["condition_change"]
        expected = validate_condition_change(
            previous, change["after"],
            kind=change["kind"] if previous is not None and change["after"] != previous else None,
            reason=change["reason"],
        )
        if change != expected:
            raise ValueError("WAM condition-change metadata disagrees with history")
        request = json.loads(
            Path(event["predicted_actions"]["path"]).with_name("request.json").read_text()
        )
        if (
            request["condition_change"] != expected
            or request["condition"] != expected["after"]
            or event["prompt"] != f"Overall task: {manifest['native_goal']}\nCurrent subgoal: {expected['after']}"
            or request["prompt"] != event["prompt"]
            or request["source"] != event["source"]
            or request["attempt"] != event["attempt"]
            or request["start_step"] != event["episode_step"]
        ):
            raise ValueError("WAM request disagrees with recorded condition")
        if expected["changed"]:
            expected_changes.append({
                **expected, "source": event["source"], "attempt": event["attempt"],
                "episode_step": event["episode_step"],
            })
        previous = expected["after"]
    keys = ("before", "after", "changed", "kind", "reason", "source", "attempt", "episode_step")
    changes = [
        {key: e[key] for key in keys} for e in events if e["event_type"] == "condition_changed"
    ]
    if changes != expected_changes:
        raise ValueError("Missing or inconsistent condition-change events")

    proposals = [e for e in events if e["event_type"] == "recovery_proposal"]
    if len(proposals) != manifest["attempts"]:
        raise ValueError("Recovery attempt count mismatch")
    complete = 0
    covered = set()
    for number, event in enumerate(proposals, 1):
        if event["attempt"] != number or event["episode_step"] < 1:
            raise ValueError("Discontinuous recovery attempts")
        path = Path(event["proposal"]["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != event["proposal"]["sha256"]:
            raise ValueError("Recovery proposal checksum mismatch")
        proposal = json.loads(path.read_text())
        submitted = {k: v for k, v in proposal.items() if k != "program"} if "control_code" in proposal else proposal
        before = conditions_before[event["event_index"]]
        if before is None or before != event["condition_before"]:
            raise ValueError("Recovery has no matching pre-correction condition")
        validated = validate_recovery_proposal(
            submitted, proposal["request_id"], number - 1,
            actual[event["episode_step"] - 1]["robot_state"]["ee_pos"],
            SafetyLimits(**manifest["safety_limits"]), recovery_condition=before,
        )
        if proposal != validated or event["mode_reason"] != proposal["mode_reason"]:
            raise ValueError("Recovery decision or mode rationale disagrees with proposal")
        if event["correction_mode"] != proposal["correction_mode"]:
            raise ValueError("Recovery mode disagrees with proposal")
        boundaries = {}
        for kind in ("recovery_evaluation_started", "recovery_evaluation_completed", "recovery_attempt_completed"):
            matches = [e for e in events if e["event_type"] == kind and e["attempt"] == number]
            if len(matches) != 1:
                raise ValueError(f"Expected one {kind} for attempt {number}")
            boundaries[kind] = matches[0]
        start = boundaries["recovery_evaluation_started"]
        end = boundaries["recovery_evaluation_completed"]
        completion = boundaries["recovery_attempt_completed"]
        evaluation = end["evaluation"]
        condition = proposal.get("condition", before)
        if not event["event_index"] < start["event_index"] < end["event_index"] < completion["event_index"]:
            raise ValueError("Invalid recovery evaluation ordering")
        correction = [
            e for e in actual if event["event_index"] < e["event_index"] < start["event_index"]
        ]
        if (
            any(e["action_source"] != "agent_correction" for e in correction)
            or start["episode_step"] - event["episode_step"] != len(correction)
        ):
            raise ValueError("Unexpected execution before recovery evaluation")
        if proposal["correction_mode"] == "condition":
            if correction:
                raise ValueError("Condition-only recovery executed controller actions")
        else:
            robot = actual[event["episode_step"] - 1]["robot_state"]
            compiled = compile_program(
                proposal["program"], robot["ee_pos"], robot["ee_quat"], robot["gripper_pos"],
                SafetyLimits(**manifest["safety_limits"]),
            )
            if not 1 <= len(correction) <= len(compiled):
                raise ValueError("Invalid recovery controller execution length")
            if len(correction) < len(compiled) and not (
                correction[-1]["events"]["native_terminated"] or correction[-1]["events"]["native_truncated"]
            ):
                raise ValueError("Nonterminal correction did not execute its full program")
            np.testing.assert_array_equal(
                np.load(path.parent / "compiled_cartesian_targets.npy"), compiled
            )
            np.testing.assert_array_equal(
                [e["cartesian_intent"] for e in correction], compiled[:len(correction)]
            )
            np.testing.assert_array_equal(
                np.load(path.parent / "executed_joint_actions.npy"),
                [e["actual_action"] for e in correction],
            )
        responses = [e for e in actual if start["event_index"] < e["event_index"] < end["event_index"]]
        if any(e["action_source"] != "recovery_evaluation" for e in responses):
            raise ValueError("Non-WAM execution inside recovery evaluation")
        covered.update(e["event_index"] for e in responses)
        count = len(responses)
        final_events = actual[end["episode_step"] - 1]["events"]
        interruption = (
            None if count == RECOVERY_EVALUATION_STEPS
            else "native_success" if final_events["native_success"]
            else "native_horizon" if final_events["native_truncated"]
            else "native_termination" if final_events["native_terminated"] else None
        )
        expected = {
            "condition_before": before, "condition": condition,
            "start_step": start["episode_step"], "end_step": end["episode_step"],
            "requested_steps": RECOVERY_EVALUATION_STEPS, "executed_steps": count,
            "complete": count == RECOVERY_EVALUATION_STEPS, "interrupted_by": interruption,
        }
        if (
            count > RECOVERY_EVALUATION_STEPS
            or (count < RECOVERY_EVALUATION_STEPS and interruption is None)
            or end["episode_step"] - start["episode_step"] != count
            or evaluation != expected or completion["evaluation"] != expected
            or completion["episode_step"] != end["episode_step"]
            or start["condition"] != condition or start["condition_before"] != before
            or start["correction_mode"] != proposal["correction_mode"]
            or start["requested_steps"] != RECOVERY_EVALUATION_STEPS
        ):
            raise ValueError("Recovery evaluation window or condition lock mismatch")
        predictions = [
            e for e in events if start["event_index"] < e["event_index"] < end["event_index"]
            and e["event_type"] == "wam_prediction"
        ]
        if len(predictions) != (1 if count else 0):
            raise ValueError("Recovery evaluation prediction count mismatch")
        if predictions:
            prediction = predictions[0]
            if (
                prediction["source"] != "recovery_evaluation" or prediction["attempt"] != number
                or prediction["requested_execution_length"] != RECOVERY_EVALUATION_STEPS
                or prediction["condition_change"]["after"] != condition
                or prediction["episode_step"] != start["episode_step"]
                or prediction["condition_change"]["kind"] != (
                    "unchanged" if proposal["correction_mode"] == "action" else "recovery_rewrite"
                )
                or prediction["condition_change"]["reason"] != (
                    None if proposal["correction_mode"] == "action" else proposal["mode_reason"]
                )
            ):
                raise ValueError("Recovery evaluation used an unexpected WAM condition or length")
        assessments = [
            e for e in events if e["event_type"] == "recovery_assessment" and e["attempt"] == number
        ]
        if len(assessments) > 1:
            raise ValueError("Duplicate recovery assessment")
        for assessment in assessments:
            a = assessment["assessment"]
            validate_assessment(a, a["request_id"])
            if (
                not expected["complete"] or assessment["episode_step"] != end["episode_step"]
                or assessment["event_index"] <= completion["event_index"]
                or a["evaluation_steps"] != RECOVERY_EVALUATION_STEPS
                or a["evaluation_end_step"] != end["episode_step"]
                or a["label_source"] != "agent_assessment"
            ):
                raise ValueError("Assessment is not tied to a complete evaluation window")
        complete += expected["complete"]
    if covered != {e["event_index"] for e in actual if e["action_source"] == "recovery_evaluation"}:
        raise ValueError("Unattributed recovery evaluation execution")
    return {
        "comparison_contract_verified": True, "complete_evaluation_windows": complete,
        "incomplete_evaluation_windows": len(proposals) - complete,
    }


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
        and event["action_source"] in {"agent_correction", "agent_condition", "recovery_evaluation"}
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
    comparison = verify_comparison_contract(manifest, events, actual)
    stable_placement = False
    if manifest.get("success_criterion") == "native_and_stable_blue_placement":
        count = 0
        for event in actual:
            sample = event["events"]["placement_verification"]
            with np.load(event["observation"]["scene"]["path"]) as scene:
                np.testing.assert_array_equal(
                    sample["object_velocity"], scene["rigid_object/blue_hammer/root_velocity"][0]
                )
            verify_placement_gripper(sample, event["robot_state"])
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
        **comparison,
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
