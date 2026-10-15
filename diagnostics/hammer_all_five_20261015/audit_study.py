"""Audit five real four-hammer episodes and summarize actual subgoal responses."""

import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))
from rollout_dashboard.preview import media_preview


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


runs = sorted((ROOT / "diagnostics").glob("recovery_all_hammers_0*_seed3407_20261015"))
assert len(runs) == 5
summaries, all_chunks, initial_audits = [], [], {}
reference_path = runs[0] / "recovery/observations/00000_initial"
for run in runs:
    directory = run / "recovery"
    manifest = json.loads((directory / "manifest.json").read_text())
    verified = json.loads((directory / "verification.json").read_text())
    metadata = json.loads((run / "experiment.json").read_text())
    spec = json.loads((run / "study_episode.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    initial = next(e for e in events if e["event_type"] == "session_started")
    actual = [e for e in events if e["event_type"] == "actual_execution_response"]
    authorized = [e for e in events if e["event_type"] == "action_authorized"]
    intents = [json.loads(line) for line in (run / "subgoal_intents.jsonl").read_text().splitlines()]
    replies = [json.loads(line) for line in (run / "subgoal_responses.jsonl").read_text().splitlines()]
    assert manifest["task"] == metadata["task"] == "ToolsPickingAllHammersTask"
    assert metadata["seed"] == 3407 and metadata["initial_genesis_sha256"] == "f39ad091d69f571bb68e02519f68f5998bd4377ac0e270cc6ea4b761b75f4d4b"
    assert manifest["native_goal"] == "Take out all the hammers and put them on the table"
    assert manifest["mock_execution"] is False and manifest["reset_after_initialization_allowed"] is False
    assert manifest["max_episode_steps"] == 600
    assert manifest["correction_attempts_preserved"] == 0
    assert verified["recording_verified"] and verified["runtime_policy_verified"]
    assert [e["episode_step"] for e in actual] == list(range(1, len(actual) + 1))
    assert len(actual) == manifest["step"] == len(authorized)
    assert set(e["action_source"] for e in actual) == {"wam"}
    for intent, response in zip(authorized, actual):
        assert intent["episode_step"] + 1 == response["episode_step"]
        np.testing.assert_array_equal(intent["authorized_action"], response["actual_action"])
    assert manifest["native_terminated"] or manifest["native_truncated"]
    assert len(actual) == 600 or manifest["native_terminated"]
    for event in actual:
        progress = event["events"]["hammer_task_progress"]
        flags = progress["per_object"]
        assert set(flags) == {"husky_hammer", "blue_hammer", "red_hammer", "wood_hammer"}
        assert progress["completed_count"] == sum(flags.values())
        assert progress["required_count"] == 4
        assert progress["remaining_objects"] == [name for name, passed in flags.items() if not passed]
        assert event["events"]["native_success"] == all(flags.values())
    assert manifest["hammer_task_progress"] == actual[-1]["events"]["hammer_task_progress"]
    with np.load(reference_path / "scene.npz") as ref, np.load(directory / "observations/00000_initial/scene.npz") as other:
        scene_difference = max(float(np.abs(ref[key] - other[key]).max()) for key in ref.files)
        assert scene_difference == 0
    image_differences = {}
    for source in sorted(reference_path.glob("*.png")):
        other = directory / "observations/00000_initial" / source.name
        a, b = np.asarray(Image.open(source)).astype(float), np.asarray(Image.open(other)).astype(float)
        image_differences[source.name] = float(np.abs(a - b).mean())
    initial_audits[run.name] = {"scene_max_abs_difference": scene_difference, "initial_image_mae_255": image_differences}

    observations = [initial["observation"]] + [e["observation"] for e in actual]
    objects = {name: [] for name in progress["per_object"]}
    for observation in observations:
        with np.load(observation["scene"]["path"]) as scene:
            for name in objects:
                objects[name].append(scene[f"rigid_object/{name}/root_pose"][0, :3])
    motion = {}
    for name, values in objects.items():
        xyz = np.array(values)
        first_pass = next((e["episode_step"] for e in actual if e["events"]["hammer_task_progress"]["per_object"][name]), None)
        motion[name] = {
            "max_displacement_mm": float(np.linalg.norm(xyz - xyz[0], axis=1).max() * 1000),
            "max_root_height_gain_mm": float((xyz[:, 2] - xyz[0, 2]).max() * 1000),
            "first_native_placement_step": first_pass,
            "final_native_placed": progress["per_object"][name],
        }
    segment_paths = sorted((directory / "segments").glob("*_wam/result.json"))
    assert len(segment_paths) == len(intents) == len(replies)
    target_windows = []
    stalled_candidates = []
    for index, (path, intent, response) in enumerate(zip(segment_paths, intents, replies)):
        result = json.loads(path.read_text())
        req = json.loads((path.parent / "request.json").read_text())
        start, end = req["start_step"], result["episode_step"]
        assert intent["episode_step"] == start and response["step"] == end
        assert intent["condition"] == req["condition"]
        assert response["progress"] == result["hammer_task_progress"]
        assert req["prompt"] == f"Overall task: {manifest['native_goal']}\nCurrent subgoal: {req['condition']}"
        commands = np.load(path.parent / "predicted_actions.npy")
        executed = np.array([e["actual_action"] for e in actual[start:end]])
        np.testing.assert_array_equal(commands[:end - start], executed)
        ee = np.array([e["robot_state"]["ee_pos"] for e in actual[start:end]])
        before_robot = actual[start - 1]["robot_state"] if start else initial["state"]
        ee = np.vstack((before_robot["ee_pos"], ee))
        xyz = np.array(objects[intent["target"]][start:end + 1])
        ee_range = float(np.linalg.norm(np.ptp(ee, axis=0)))
        target_range = float(np.linalg.norm(np.ptp(xyz, axis=0)))
        close = int((executed[:, -1] > 0.5).sum())
        tail = int((commands[16:, -1] > 0.5).sum())
        stall = end - start == 16 and ee_range <= 0.02 and target_range <= 0.01 and close == 0
        record = {
            "trial": spec["trial"], "chunk": index + 1, "first_step": start + 1, "last_step": end,
            "target": intent["target"], "condition": req["condition"],
            "change_kind": req["condition_change"]["kind"],
            "close_commands": close, "predicted_tail_close_commands": tail,
            "closure_only_in_unexecuted_tail": end - start == 16 and close == 0 and tail > 0,
            "ee_range_norm_m": ee_range, "target_range_norm_m": target_range,
            "near_static_no_close_candidate": stall,
            "completed_hammers": result["hammer_task_progress"]["completed_count"],
        }
        all_chunks.append(record)
        stalled_candidates.append(record)
        if not target_windows or target_windows[-1]["target"] != intent["target"]:
            target_windows.append({
                "target": intent["target"], "first_step": start + 1, "last_step": end,
                "close_commands": close, "first_condition": req["condition"],
                "selection_reason": intent["reason"],
            })
        else:
            target_windows[-1]["last_step"] = end
            target_windows[-1]["close_commands"] += close
    candidates = []
    for left, right in zip(stalled_candidates, stalled_candidates[1:]):
        if left["target"] == right["target"] and left["near_static_no_close_candidate"] and right["near_static_no_close_candidate"]:
            candidates.append({"first_window": [left["first_step"], left["last_step"]], "confirmation_step": right["last_step"], "target": left["target"]})
    video = Path(manifest["episode_video"]["path"])
    original_hash = sha256(video)
    preview = media_preview(run, video)
    def probe(path):
        result = subprocess.run([
            str(ROOT / "packages/ffmpeg/bin/ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,r_frame_rate:format=duration", "-of", "json", str(path),
        ], check=True, capture_output=True, text=True)
        return json.loads(result.stdout)
    source_info, preview_info = probe(video), probe(preview)
    assert int(source_info["streams"][0]["nb_frames"]) == len(actual) + 1
    assert source_info == preview_info
    assert sha256(video) == original_hash
    summaries.append({
        "trial": spec["trial"], "run": run.name, "initial_target": spec["initial_target"],
        "actual_steps": len(actual), "chunks": len(segment_paths),
        "native_success": manifest["native_success"], "native_truncated": manifest["native_truncated"],
        "completed_hammers_final": progress["completed_count"],
        "max_simultaneously_completed_hammers": max(e["events"]["hammer_task_progress"]["completed_count"] for e in actual),
        "target_windows": target_windows, "object_measurements": motion,
        "close_commands": sum(e["actual_action"][-1] > 0.5 for e in actual),
        "max_measured_closure": max(e["robot_state"]["gripper_pos"][0] for e in actual),
        "first_close_command_step": next((e["episode_step"] for e in actual if e["actual_action"][-1] > 0.5), None),
        "condition_changes": sum(e["event_type"] == "condition_changed" for e in events) - 1,
        "first_sustained_stall_candidate": candidates[0] if candidates else None,
        "stall_candidate_note": "Measured post-hoc motion/command criterion; not automatic proof of target identity, grasp, or every task-level failure.",
        "video_frames": len(actual) + 1, "preview_bytes": preview.stat().st_size,
        "native_progress_verified_every_step": True,
        "all_commands_and_prediction_prefixes_verified": True,
        "zero_controller_actions": True,
        "recovery_acceptance_not_applicable": True,
    })

log = (STUDY / "model_server.log").read_text()
assert "seed=3407 deterministic_seed=True" in log
request_counts = Counter()
seeds = []
for line in log.splitlines():
    match = re.search(r"\[robolab-policy-server\] prompt=(.*) seed=(\d+)$", line)
    if match:
        caption = json.loads(ast.literal_eval(match[1]))
        description = caption["actions"][0]["description"]
        assert description.startswith("Overall task: Take out all the hammers and put them on the table\nCurrent subgoal: ")
        request_counts[description.split("\nCurrent subgoal: ", 1)[1]] += 1
        seeds.append(int(match[2]))
assert set(seeds) == {3407}
assert len(seeds) == sum(s["chunks"] for s in summaries)
assert request_counts == Counter(c["condition"] for c in all_chunks)
report = {
    "study_id": STUDY.name, "verified_real_episodes": len(summaries),
    "total_actual_actions": sum(s["actual_steps"] for s in summaries),
    "total_model_requests": len(seeds), "all_request_seeds": sorted(set(seeds)),
    "full_task_native_successes": sum(s["native_success"] for s in summaries),
    "native_criterion": "all four objects pass unchanged native outside-bin/table-support/gripper-detached predicate",
    "same_initial_physics_states": True, "initial_image_differences": initial_audits,
    "language_condition_changes_are_interventions": True,
    "no_local_controller_actions": True, "privileged_state_assisted": True,
    "episodes": summaries,
}
(STUDY / "study_results.json").write_text(json.dumps(report, indent=2) + "\n")
write_csv(STUDY / "chunk_metrics.csv", all_chunks)
write_csv(STUDY / "trial_results.csv", [{
    k: s[k] for k in (
        "trial", "initial_target", "actual_steps", "native_success",
        "completed_hammers_final", "close_commands", "first_close_command_step",
        "max_measured_closure", "condition_changes",
    )
} for s in summaries])
print(json.dumps({k: v for k, v in report.items() if k not in ("episodes", "initial_image_differences")}, indent=2))
for summary in summaries:
    print(json.dumps(summary, indent=2))
