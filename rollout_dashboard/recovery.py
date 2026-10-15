"""Read-only projections of finalized recovery episodes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from urllib.parse import quote


DOWNLOADS = {
    "manifest": "manifest.json",
    "events": "events.jsonl",
    "trajectory": "trajectory.hdf5",
    "verification": "verification.json",
    "video": "episode_wrist.mp4",
}


def episode_file(directory: Path, relative: str) -> Path:
    path = (directory / "recovery" / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-episode artifact: {relative}")
    return path


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def summary(directory: Path) -> dict:
    manifest = read_json(episode_file(directory, "manifest.json"))
    if manifest.get("format") != "wam-agent-recovery-v1":
        raise ValueError("Unsupported recovery episode format")
    return {
        "id": directory.name,
        "task": manifest["task"],
        "steps": manifest["step"],
        "native_success": manifest.get("native_success"),
        "acceptance_success": manifest.get("acceptance_success"),
        "experiment_mode": manifest.get("experiment_mode", "wam_recovery"),
        "success_criterion": manifest.get("success_criterion", "native"),
        "agent_control_success": manifest.get("agent_control_success", False),
        "protocol_version": manifest.get("protocol_version", "wam-recovery-v1"),
        "recovery_evaluation_steps": manifest.get("recovery_evaluation_steps"),
    }


def load_episode(directory: Path) -> dict:
    directory = directory.resolve()
    manifest = read_json(episode_file(directory, "manifest.json"))
    result = summary(directory)
    dt = float(manifest["control_dt"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid control timestep")
    raw_events = episode_file(directory, "events.jsonl").read_bytes()
    if hashlib.sha256(raw_events).hexdigest() != manifest["events"]["sha256"]:
        raise ValueError("Event log differs from its finalized manifest")
    events = [json.loads(line) for line in raw_events.splitlines() if line.strip()]
    verification_path = directory / "recovery/verification.json"
    verification = read_json(episode_file(directory, "verification.json")) if verification_path.exists() else None
    experiment_path = directory / "experiment.json"
    experiment = read_json(episode_file(directory, "../experiment.json")) if experiment_path.exists() else {}

    def reference_path(reference: dict) -> Path:
        path = Path(reference["path"])
        path = (path if path.is_absolute() else directory / path).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Recorded artifact is missing or outside this episode")
        return path

    def media(reference: dict) -> str:
        return f"/media/{quote(directory.name, safe='')}/{quote(reference_path(reference).relative_to(directory).as_posix(), safe='/')}"

    def frames(observation: dict) -> dict:
        return {
            frame["camera"]: media(frame)
            for frame in observation["frames"]
            if not frame.get("debug_camera", False)
        }

    actual = [event for event in events if event["event_type"] == "actual_execution_response"]
    if [event["episode_step"] for event in actual] != list(range(1, manifest["step"] + 1)):
        raise ValueError("Actual execution steps are missing or discontinuous")
    initial = next(event for event in events if event["event_type"] == "session_started")
    observations = [{
        "step": 0, "source": "initial", "frames": frames(initial["observation"]),
        "robot": initial["state"],
    }]
    phases = []
    for event in actual:
        source, step = event["action_source"], event["episode_step"]
        observations.append({
            "step": step, "source": source, "frames": frames(event["observation"]),
            "robot": event["robot_state"], "action": event["actual_action"],
        })
        if phases and phases[-1]["source"] == source:
            phases[-1]["end"] = step
        else:
            phases.append({"source": source, "start": step, "end": step})

    timeline = []
    names = {
        "failure_detected": "failure",
        "recovery_proposal": "proposal",
        "agent_control_proposal": "proposal",
        "recovery_assessment": "assessment",
        "control_handed_back_to_wam": "handoff",
        "control_returned_to_wam_after_budget": "budget_handoff",
        "condition_changed": "condition_change",
        "command_failed": "command_error",
    }
    attempts = []
    failure_id = None
    for index, event in enumerate(events):
        kind = event["event_type"]
        if kind in names:
            entry = {
                "id": f"event-{index}",
                "type": names[kind], "step": event["episode_step"],
                "attempt": event.get("attempt"), "recorded_at": event["recorded_at"],
            }
            if kind == "failure_detected":
                failure_id = entry["id"]
                entry["failure"] = event["failure"]
            elif kind == "command_failed":
                entry.update(error=event["error"], operation=event.get("operation"))
            timeline.append(entry)
        if kind in {"control_handed_back_to_wam", "control_returned_to_wam_after_budget"}:
            failure_id = None
        if kind not in {"recovery_proposal", "agent_control_proposal"}:
            continue
        end = next((
            i for i in range(index + 1, len(events))
            if events[i]["event_type"] in {
                "recovery_proposal", "agent_control_proposal", "control_handed_back_to_wam",
                "control_returned_to_wam_after_budget",
            }
        ), len(events))
        responses = [
            item for item in events[index + 1:end]
            if item["event_type"] == "actual_execution_response"
            and item["action_source"] in {
                "agent_correction", "agent_condition", "recovery_evaluation", "agent_control",
            }
        ]
        assessments = [
            item["assessment"] for item in events[index + 1:]
            if item["event_type"] == "recovery_assessment"
            and item["attempt"] == event.get("attempt")
        ]
        evaluations = [
            item["evaluation"] for item in events[index + 1:]
            if item["event_type"] == "recovery_evaluation_completed"
            and item["attempt"] == event.get("attempt")
        ]
        tracking = []
        action_ranges = []
        for response in responses:
            if response["action_source"] in {"agent_correction", "agent_control"}:
                step = response["episode_step"]
                if action_ranges and action_ranges[-1]["end"] == step - 1:
                    action_ranges[-1]["end"] = step
                else:
                    action_ranges.append({"start": step, "end": step})
            if "cartesian_intent" in response:
                target = response["cartesian_intent"][:3]
                measured = response["robot_state"]["ee_pos"]
                tracking.append({
                    "step": response["episode_step"], "target": target, "measured": measured,
                    "error_mm": 1000 * math.dist(target, measured),
                })
        attempts.append({
            "number": event.get("attempt", event.get("program_index")), "start": event["episode_step"],
            "end": responses[-1]["episode_step"] if responses else event["episode_step"],
            "failure_id": failure_id, "action_ranges": action_ranges,
            "proposal": read_json(reference_path(event["proposal"])),
            "assessment": assessments[-1] if assessments else None,
            "assessments": assessments, "evaluations": evaluations,
            "tracking": tracking, "executed_steps": len(responses),
            "evaluation": evaluations[-1] if evaluations else None,
        })
    if manifest.get("native_success") is True:
        timeline.append({"type": "success", "step": manifest["step"]})
    segments = []
    for path in sorted((directory / "recovery/segments").glob("*/result.json")):
        segment = read_json(episode_file(directory, path.relative_to(directory / "recovery").as_posix()))
        request_path = episode_file(directory, (path.parent / "request.json").relative_to(directory / "recovery").as_posix())
        request = read_json(request_path)
        segments.append({
            "id": path.parent.name, "source": segment["source"],
            "start": request["start_step"], "executed_steps": segment["executed_steps"],
            "end": request["start_step"] + segment["executed_steps"],
            "condition": request.get("condition"), "subgoal": request.get("subgoal"),
            "attempt": request.get("attempt"),
            "predicted_steps": segment["prediction_length"], "prompt": segment["prompt"],
            "inference_seconds": segment["inference_seconds"],
            "input": media({"path": str(path.parent / "input.png")}),
            "prediction": media(segment["predicted_future"]["video"]),
            "condition_change": request.get("condition_change"),
        })
    downloads = {
        kind: f"/api/recovery/{quote(directory.name, safe='')}/download/{kind}"
        for kind, filename in DOWNLOADS.items()
        if (directory / "recovery" / filename).is_file()
    }
    return {
        **result, "goal": manifest["native_goal"], "dt": dt,
        "duration": (manifest["step"] + 1) * dt,
        "local_recovery_success": manifest.get("local_recovery_success"),
        "verified": bool(
            result["experiment_mode"] == "wam_recovery"
            and verification and verification.get("recording_verified") is True
            and verification.get("acceptance_success") is True
            and manifest.get("acceptance_success") is True
            and manifest.get("mock_execution") is False
        ),
        "agent_control_verified": bool(
            result["experiment_mode"] == "agent_control"
            and verification and verification.get("recording_verified") is True
            and verification.get("agent_control_success") is True
            and manifest.get("agent_control_success") is True
            and manifest.get("mock_execution") is False
        ),
        "stable_blue_placement_verified": bool(
            result["experiment_mode"] == "agent_control"
            and result["success_criterion"] == "native_and_stable_blue_placement"
            and verification and verification.get("recording_verified") is True
            and verification.get("stable_blue_placement_verified") is True
            and verification.get("agent_control_success") is True
            and manifest.get("agent_control_success") is True
            and manifest.get("mock_execution") is False
        ),
        "verification": verification, "mock_execution": manifest.get("mock_execution"),
        "comparison_contract_verified": verification.get("comparison_contract_verified") if verification else None,
        "condition_changes": [
            {key: event[key] for key in ("episode_step", "source", "attempt", "before", "after", "kind", "reason")}
            for event in events if event["event_type"] == "condition_changed"
        ],
        "agent_input_mode": experiment.get("agent_input_mode"),
        "agent_model_version": experiment.get("agent_model_version"),
        "attempt_limit": None if result["experiment_mode"] == "agent_control" else manifest["safety_limits"]["max_attempts"],
        "budget_exhaustion_policy": manifest.get("budget_exhaustion_policy", "stop"),
        "budget_fallback": manifest.get("budget_fallback", False),
        "max_episode_steps": manifest.get("max_episode_steps"),
        "failure": manifest.get("failure"),
        "video": media(manifest["episode_video"]),
        "observations": observations, "phases": phases, "timeline": timeline,
        "attempts": attempts, "segments": segments, "downloads": downloads,
        "errors": [
            {"step": event["episode_step"], "operation": event.get("operation"), "error": event["error"]}
            for event in events if event["event_type"] == "command_failed"
        ],
    }
