"""Verify the completed comparison and prepare small, source-preserving previews."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent.parent
sys.path.insert(0, str(ROOT))
from rollout_dashboard.preview import media_preview


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_info(path):
    result = subprocess.run([
        str(ROOT / "packages/ffmpeg/bin/ffprobe"), "-v", "error",
        "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


design = json.loads((STUDY / "study_design.json").read_text())
expected = {}
episodes = {}
source_identities = []
for variant in design["variants"]:
    run = ROOT / "diagnostics" / variant["episode"]
    manifest = json.loads((run / "recovery/manifest.json").read_text())
    verification = json.loads((run / "recovery/verification.json").read_text())
    metadata = json.loads((run / "experiment.json").read_text())
    assert metadata["seed"] == design["seed"]
    assert metadata["initial_genesis_sha256"] == design["genesis_sha256"]
    assert manifest["native_goal"] == design["native_goal"]
    assert manifest["success_criterion"] == "native"
    assert manifest["max_episode_steps"] == 600
    assert manifest["reset_after_initialization_allowed"] is False
    assert manifest["mock_execution"] is False
    assert verification["recording_verified"] and verification["runtime_policy_verified"]
    assert verification["actual_steps"] == manifest["step"]
    assert verification["attempts"] == verification["agent_steps"] == 0
    assert verification["agent_control_steps"] == verification["wam_after_budget_steps"] == 0
    segments = list((run / "recovery/segments").glob("*_wam/result.json"))
    expected[variant["id"]] = len(segments)
    source_identities.append(manifest["source_identity"])
    source = Path(manifest["episode_video"]["path"])
    original_hash = sha256(source)
    preview = media_preview(run, source)
    original_info, preview_info = video_info(source), video_info(preview)
    assert int(original_info["streams"][0]["nb_frames"]) == manifest["step"] + 1
    assert original_info["streams"][0]["nb_frames"] == preview_info["streams"][0]["nb_frames"]
    assert original_info["streams"][0]["r_frame_rate"] == preview_info["streams"][0]["r_frame_rate"]
    assert original_info["format"]["duration"] == preview_info["format"]["duration"]
    assert sha256(source) == original_hash
    episodes[variant["id"]] = {
        "episode": variant["episode"], "steps": manifest["step"],
        "chunks": len(segments), "native_success": manifest["native_success"],
        "native_truncated": manifest["native_truncated"], "verification": verification,
        "original_video_sha256": original_hash,
        "original_video_bytes": source.stat().st_size,
        "preview_video_bytes": preview.stat().st_size,
        "video_frames": int(original_info["streams"][0]["nb_frames"]),
        "video_duration_seconds": float(original_info["format"]["duration"]),
        "preview_path": str(preview),
    }
assert all(identity == source_identities[0] for identity in source_identities)

log = (STUDY / "model_server.log").read_text()
assert "seed=3407 deterministic_seed=True" in log
requests = Counter()
seeds = []
for line in log.splitlines():
    match = re.search(r"\[robolab-policy-server\] prompt=(.*) seed=(\d+)$", line)
    if not match:
        continue
    caption = json.loads(ast.literal_eval(match[1]))
    description = caption["actions"][0]["description"]
    variants = [
        v for v in design["variants"]
        if description.startswith(
            f"Overall task: {design['native_goal']}\nCurrent subgoal: {v['condition']}"
        )
    ]
    assert len(variants) == 1, description
    requests[variants[0]["id"]] += 1
    seeds.append(int(match[2]))
assert dict(requests) == expected, (requests, expected)
assert set(seeds) == {3407}
report = {
    "all_episodes_verified": True,
    "model_request_counts": dict(requests),
    "all_request_seeds": sorted(set(seeds)),
    "total_model_requests": len(seeds),
    "total_actual_actions": sum(e["steps"] for e in episodes.values()),
    "same_runtime_source_identities": True,
    "initial_matching_audit": "initial_matching_audit.json",
    "initial_physics_states_exactly_equal": True,
    "initial_rendered_images_bitwise_equal": False,
    "episodes": episodes,
}
(STUDY / "study_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
