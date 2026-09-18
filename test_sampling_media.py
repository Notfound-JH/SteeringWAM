import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from rollout_dashboard import sampling_media as media
from rollout_dashboard.server import create_app
from fastapi.testclient import TestClient


def fixture_candidate(tmp_path):
    experiment = tmp_path / "instruction_test"
    candidate = experiment / "fruits/candidates/candidate1"
    candidate.mkdir(parents=True)
    frames = []
    for step in range(3):
        for camera in media.CAMERAS:
            path = candidate / f"{camera}_{step}.png"
            image = Image.new("RGB", (64, 40), (60 + step * 30, 90, 140))
            image.paste((200, 120, 30), (step * 5, 10, 30 + step * 5, 30))
            image.save(path)
            frames.append({"path": str(path), "camera": camera, "step": step, "sha256": media.digest(path)})
    result = candidate / "result.json"
    result.write_text(json.dumps({"status": "complete", "candidate_id": "candidate1", "executed_steps": 2,
                                 "control_dt": 1 / 15, "frames": frames}))
    (candidate / "request.json").write_text(json.dumps({"candidate": {"chunk": 0, "attempt": 0,
        "instruction": {"instruction": "Approach the visible target with the fingers open."}}}))
    return experiment, result


def test_actual_video_decode_and_cache(tmp_path):
    experiment, result = fixture_candidate(tmp_path)
    before = result.read_bytes()
    destination = media.render_candidate(experiment, result)
    manifest = media.read_json(destination / "manifest.json")
    assert manifest["unique_frames"] == 3 and manifest["executed_steps"] == 2
    assert manifest["full_decode"] == "passed" and not manifest["unexecuted_tail_included"]
    assert manifest["video_sha256"] == media.digest(destination / "actual.mp4")
    timestamp = (destination / "actual.mp4").stat().st_mtime_ns
    assert media.render_candidate(experiment, result) == destination
    assert (destination / "actual.mp4").stat().st_mtime_ns == timestamp
    assert result.read_bytes() == before


def test_missing_or_changed_frames_rejected(tmp_path):
    experiment, result = fixture_candidate(tmp_path)
    value = media.read_json(result)
    Path(value["frames"][0]["path"]).write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        media.render_candidate(experiment, result)
    value["frames"] = value["frames"][1:]
    result.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Missing"):
        media.render_candidate(experiment, result)


def test_sampling_api_shows_recorded_copilot_output(tmp_path):
    experiment, result_path = fixture_candidate(tmp_path)
    result = media.read_json(result_path)
    result["executed_actions"] = str(result_path.parent / "executed.npy")
    result["prompt"] = "ACTUAL SENT PROMPT"
    candidate = {"id": "candidate1", "chunk": 2, "attempt": 7,
                 "instruction": {"instruction": "APPROACH ONLY", "subgoal": "align"}, "result": result,
                 "review": {"verdict": "accept", "subgoal_complete": False, "evidence": "RECORDED VISUAL PROGRESS"}}
    selected = {**candidate, "selection_reason": "RECORDED CHOICE"}
    (experiment / "experiment.json").write_text("{}")
    (experiment / "fruits/session.json").write_text(json.dumps({"format": "instruction-sampling-v3-visual-review",
        "task": "FruitsOnPlate3Task", "name": "fruits", "status": "need_instruction", "chunk": 3,
        "step": 48, "rounds": [[candidate]], "candidates": [], "selected": [selected]}))
    client = TestClient(create_app(tmp_path))
    assert client.get("/sampling").status_code == 200
    response = client.get("/api/sampling/instruction_test")
    assert response.status_code == 200
    data = response.json()
    assert data["counts"]["total"] == 1
    assert data["candidates"][0]["review"]["evidence"] == "RECORDED VISUAL PROGRESS"
    assert data["candidates"][0]["selection_reason"] == "RECORDED CHOICE"
    assert data["candidates"][0]["instruction"]["instruction"] == "APPROACH ONLY"
    assert data["candidates"][0]["video"] is None
    assert client.get("/media/instruction_test/../../../.blob-sync/temp_sas_url.txt").status_code == 404
    result = media.scan_once(tmp_path)
    assert result["errors"] == [] and result["videos_ready"] == 1
    snapshot = media.read_json(experiment / "media/copilot_outputs.json")
    assert snapshot["candidates"][0]["review"]["evidence"] == "RECORDED VISUAL PROGRESS"
    html = (experiment / "media/review.html").read_text()
    assert "window.SAMPLING_SNAPSHOT=" in html and "RECORDED VISUAL PROGRESS" in html
    assert "../media/chunks/candidate1/actual.mp4" in html