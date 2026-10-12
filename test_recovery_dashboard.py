import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from rollout_dashboard.recovery import load_episode
from rollout_dashboard.server import create_app


@pytest.fixture
def episode(tmp_path):
    run = tmp_path / "episode"
    recovery = run / "recovery"
    recovery.mkdir(parents=True)
    image = recovery / "frame.png"
    image.write_bytes(b"unit-test-image")
    video = recovery / "episode_wrist.mp4"
    video.write_bytes(b"unit-test-video")
    observation = {"frames": [{"camera": "wrist_cam", "path": str(image)}]}
    robot = {"ee_pos": [0.4, 0, 0.3], "gripper_pos": [0]}
    events = [
        {"event_type": "session_started", "observation": observation, "state": robot},
        {"event_type": "actual_execution_response", "episode_step": 1,
         "action_source": "wam", "observation": observation, "robot_state": robot,
         "actual_action": [0] * 8},
    ]
    raw = "\n".join(json.dumps(event) for event in events).encode()
    (recovery / "events.jsonl").write_bytes(raw)
    manifest = {
        "format": "wam-agent-recovery-v1", "task": "UnitTestTask", "native_goal": "<script>bad()</script>",
        "step": 1, "native_success": False, "acceptance_success": False,
        "control_dt": 1 / 15, "safety_limits": {"max_attempts": 3}, "mock_execution": True,
        "events": {"sha256": hashlib.sha256(raw).hexdigest()}, "episode_video": {"path": str(video)},
    }
    (recovery / "manifest.json").write_text(json.dumps(manifest))
    return run, manifest


def test_read_only_routes_and_unverified_episode(episode):
    run, _ = episode
    with TestClient(create_app(run.parent)) as client:
        assert client.get("/recovery").status_code == 200
        assert client.get("/api/recovery").json()["episodes"][0]["id"] == run.name
        response = client.get(f"/api/recovery/{run.name}")
        assert response.status_code == 200
        data = response.json()
        assert data["verified"] is False
        assert data["agent_input_mode"] is None
        assert data["agent_model_version"] is None
        assert data["observations"][1]["step"] == 1
        assert data["attempts"] == []
        assert data["segments"] == []
        assert client.get(data["video"], headers={"Range": "bytes=0-3"}).status_code in {200, 206}
        assert client.get(data["downloads"]["manifest"]).headers["content-disposition"].startswith("attachment")
        assert client.get(f"/api/recovery/{run.name}/download/credentials").status_code == 404
        assert client.post(f"/api/recovery/{run.name}").status_code == 405


def test_corrupted_events_are_an_explicit_error(episode):
    run, _ = episode
    (run / "recovery/events.jsonl").write_text("{}\n")
    with TestClient(create_app(run.parent)) as client:
        assert client.get(f"/api/recovery/{run.name}").status_code == 500


def test_download_rejects_symlink_escape(episode, tmp_path):
    run, _ = episode
    secret = tmp_path / "not-episode.json"
    secret.write_text('{"private":true}')
    (run / "recovery/verification.json").symlink_to(secret)
    with TestClient(create_app(run.parent)) as client:
        assert client.get(f"/api/recovery/{run.name}/download/verification").status_code == 404


def test_agent_input_provenance_is_exposed_without_changing_outcomes(episode):
    run, _ = episode
    (run / "experiment.json").write_text(json.dumps({
        "agent_input_mode": "debug_ground_truth_assisted", "agent_model_version": None,
    }))
    with TestClient(create_app(run.parent)) as client:
        data = client.get(f"/api/recovery/{run.name}").json()
        assert data["agent_input_mode"] == "debug_ground_truth_assisted"
        assert data["agent_model_version"] is None
        assert data["native_success"] is False
        assert data["verified"] is False


def test_agent_provenance_rejects_symlink_escape(episode, tmp_path):
    run, _ = episode
    outside = tmp_path / "outside-experiment.json"
    outside.write_text('{"agent_input_mode":"not-episode-data"}')
    (run / "experiment.json").symlink_to(outside)
    with TestClient(create_app(run.parent)) as client:
        assert client.get(f"/api/recovery/{run.name}").status_code == 500


def test_mock_success_cannot_be_verified(episode):
    run, manifest = episode
    manifest.update(native_success=True, acceptance_success=True)
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    (run / "recovery/verification.json").write_text(json.dumps({
        "recording_verified": True, "acceptance_success": True,
    }))
    assert load_episode(run)["verified"] is False


def test_agent_baseline_is_not_a_wam_recovery_success(episode):
    run, manifest = episode
    manifest.update(
        experiment_mode="agent_control", native_success=True,
        agent_control_success=True, acceptance_success=False, mock_execution=False,
    )
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    (run / "recovery/verification.json").write_text(json.dumps({
        "recording_verified": True, "agent_control_success": True, "acceptance_success": False,
    }))
    data = load_episode(run)
    assert data["agent_control_verified"] is True
    assert data["verified"] is False
    assert data["attempt_limit"] is None
    manifest["mock_execution"] = True
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    assert load_episode(run)["agent_control_verified"] is False


def test_assessment_logged_after_handoff_is_attached_to_attempt(episode):
    run, manifest = episode
    path = run / "recovery/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    proposal = run / "recovery/proposal.json"
    proposal.write_text(json.dumps({"correction_mode": "condition", "condition": "retry grasp"}))
    response = {**events[-1], "episode_step": 2, "action_source": "agent_condition"}
    events.extend([
        {"event_type": "recovery_proposal", "episode_step": 1, "attempt": 1,
         "recorded_at": 1, "proposal": {"path": str(proposal)}},
        response,
        {"event_type": "control_handed_back_to_wam", "episode_step": 2, "recorded_at": 2},
        {"event_type": "recovery_assessment", "episode_step": 2, "attempt": 1,
         "recorded_at": 3, "assessment": {"local_recovery_success": True, "evidence": "observed"}},
    ])
    raw = "\n".join(json.dumps(event) for event in events).encode()
    path.write_bytes(raw)
    manifest.update(step=2, events={"sha256": hashlib.sha256(raw).hexdigest()})
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    attempt = load_episode(run)["attempts"][0]
    assert attempt["executed_steps"] == 1
    assert attempt["assessment"]["local_recovery_success"] is True
