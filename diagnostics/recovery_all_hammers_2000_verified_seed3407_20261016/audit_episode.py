"""Audit actual full-task progression, mixed action sources, seeds and archive inputs."""

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

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from rollout_dashboard.preview import media_preview


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(RUN / "recovery/manifest.json")
verification = read(RUN / "recovery/verification.json")
events = [json.loads(line) for line in (RUN / "recovery/events.jsonl").read_text().splitlines()]
actual = [e for e in events if e["event_type"] == "actual_execution_response"]
authorized = [e for e in events if e["event_type"] == "action_authorized"]
initial = next(e for e in events if e["event_type"] == "session_started")
assert manifest["task"] == "ToolsPickingAllHammersTask"
assert manifest["max_episode_steps"] == 2000
assert manifest["mock_execution"] is False
assert manifest["reset_after_initialization_allowed"] is False
assert verification["recording_verified"] and verification["runtime_policy_verified"]
assert verification["progress_contract_verified"]
assert len(actual) == manifest["step"] == len(authorized)
stopped_by_user = (RUN / "user_stop.json").exists()
assert len(actual) <= 2000
assert len(actual) == 2000 or manifest["native_terminated"] or stopped_by_user
if stopped_by_user:
    assert read(RUN / "user_stop.json")["reason"] == "user_requested_early_stop"
    endpoint = read(RUN / "user_stop_finalize_response.json")
    endpoint = endpoint.get("state", endpoint)
    assert endpoint["status"] == "finalized" and endpoint["step"] == len(actual)
assert [e["episode_step"] for e in actual] == list(range(1, len(actual) + 1))
for command, response in zip(authorized, actual):
    assert command["episode_step"] + 1 == response["episode_step"]
    np.testing.assert_array_equal(command["authorized_action"], response["actual_action"])

names = ("husky_hammer", "blue_hammer", "red_hammer", "wood_hammer")
positions = {name: [] for name in names}
rows = []
for event in actual:
    progress = event["events"]["hammer_task_progress"]
    flags = progress["per_object"]
    assert set(flags) == set(names)
    assert progress["completed_count"] == sum(flags.values())
    assert progress["required_count"] == 4
    assert sorted(progress["remaining_objects"]) == sorted(name for name in names if not flags[name])
    assert event["events"]["native_success"] == all(flags.values())
    rows.append({
        "step": event["episode_step"], "source": event["action_source"],
        "completed_count": progress["completed_count"], **flags,
    })
for observation in [initial["observation"]] + [e["observation"] for e in actual]:
    with np.load(observation["scene"]["path"]) as scene:
        for name in names:
            positions[name].append(scene[f"rigid_object/{name}/root_pose"][0, :3])
motion = {}
for name in names:
    xyz = np.array(positions[name])
    passed = [e["episode_step"] for e in actual if e["events"]["hammer_task_progress"]["per_object"][name]]
    motion[name] = {
        "max_displacement_mm": float(np.linalg.norm(xyz - xyz[0], axis=1).max() * 1000),
        "max_root_height_gain_mm": float((xyz[:, 2] - xyz[0, 2]).max() * 1000),
        "native_placement_steps": len(passed),
        "first_native_placement_step": passed[0] if passed else None,
        "last_native_placement_step": passed[-1] if passed else None,
        "final_native_placed": rows[-1][name],
    }

predictions = sorted((RUN / "recovery/segments").glob("*/predicted_actions.npy"))
expected_prompts = Counter()
for path in predictions:
    request = read(path.with_name("request.json"))
    result = read(path.with_name("result.json"))
    start, end = request["start_step"], result["episode_step"]
    np.testing.assert_array_equal(
        np.load(path)[:end - start], np.array([e["actual_action"] for e in actual[start:end]])
    )
    prompt = request["prompt"]
    expected_prompts[prompt if prompt.endswith((".", "!", "?")) else prompt + "."] += 1
log = (RUN / "services/model/server.log").read_text()
assert "seed=3407 deterministic_seed=True" in log
observed_prompts, seeds = Counter(), []
for line in log.splitlines():
    match = re.search(r"\[robolab-policy-server\] prompt=(.*) seed=(\d+)$", line)
    if match:
        description = json.loads(ast.literal_eval(match[1]))["actions"][0]["description"]
        observed_prompts[description] += 1
        seeds.append(int(match[2]))
assert observed_prompts == expected_prompts
assert len(seeds) == len(predictions) and set(seeds) == {3407}

judge_records = []
for path in sorted(RUN.glob("judge_*_provenance.json")):
    provenance = read(path)
    assert provenance["execution_mode"] == "background"
    assert sha256(RUN / provenance["input"]) == provenance["input_sha256"]
    assert sha256(RUN / provenance["raw_output"]) == provenance["raw_sha256"]
    raw = read(RUN / provenance["raw_output"])
    submitted = read(path.with_name(path.name.replace("_provenance", "_submitted")))
    for value in (raw, submitted):
        value.pop("agent_id", None)
        value.pop("request_id", None)
    assert raw == submitted
    judge_records.append({
        "context_id": provenance["context_id"], "raw_output": provenance["raw_output"],
        "stage": raw["stage"], "effect": raw["correction_effect"],
        "recovery_success": raw["recovery_success"],
    })
assert len(judge_records) >= manifest["correction_attempts_preserved"]

video = Path(manifest["episode_video"]["path"])
original_hash = sha256(video)
preview = media_preview(RUN, video)


def probe(path):
    result = subprocess.run([
        str(ROOT / "packages/ffmpeg/bin/ffprobe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate:format=duration", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


source_info = probe(video)
assert int(source_info["streams"][0]["nb_frames"]) == len(actual) + 1
assert source_info == probe(preview)
assert sha256(video) == original_hash
report = {
    "episode": RUN.name, "actual_steps": len(actual),
    "configured_step_budget": 2000, "reached_step_budget": len(actual) == 2000,
    "stopped_by_user": stopped_by_user,
    "action_source_counts": dict(Counter(e["action_source"] for e in actual)),
    "native_success": manifest["native_success"],
    "acceptance_success": manifest["acceptance_success"],
    "final_hammer_progress": manifest["hammer_task_progress"],
    "max_simultaneously_completed_hammers": max(row["completed_count"] for row in rows),
    "object_measurements": motion,
    "model_requests": len(seeds), "request_seeds": sorted(set(seeds)),
    "prediction_prefixes_and_authorizations_verified": True,
    "native_progress_verified_every_step": True,
    "judge_verdicts_unchanged": True, "judges": judge_records,
    "distinct_judge_contexts": len({j["context_id"] for j in judge_records}),
    "video": source_info, "original_video_sha256": original_hash,
    "preview": str(preview), "preview_bytes": preview.stat().st_size,
    "interpretation": "Real V3 intervention episode; not a horizon-only comparison to prior language-only trials. Local partial contact is not retained extraction or all-four success.",
}
(RUN / "audit_results.json").write_text(json.dumps(report, indent=2) + "\n")
with (RUN / "task_progress.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(report, indent=2))
