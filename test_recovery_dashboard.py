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
        assert data["condition_changes"] == []
        assert data["comparison_contract_verified"] is None
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
    assert data["stable_blue_placement_verified"] is False
    manifest["mock_execution"] = True
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    assert load_episode(run)["agent_control_verified"] is False


def test_strict_placement_label_requires_matching_verified_real_run(episode):
    run, manifest = episode
    manifest.update(
        experiment_mode="agent_control", native_success=True,
        agent_control_success=True, acceptance_success=False, mock_execution=False,
        success_criterion="native_and_stable_blue_placement",
    )
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    (run / "recovery/verification.json").write_text(json.dumps({
        "recording_verified": True, "agent_control_success": True,
        "stable_blue_placement_verified": True,
    }))
    assert load_episode(run)["stable_blue_placement_verified"] is True
    manifest["success_criterion"] = "native"
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    assert load_episode(run)["stable_blue_placement_verified"] is False
    manifest.update(success_criterion="native_and_stable_blue_placement", mock_execution=True)
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    assert load_episode(run)["stable_blue_placement_verified"] is False


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


def test_new_evaluation_and_condition_change_metadata_are_projected(episode):
    run, manifest = episode
    path = run / "recovery/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    proposal = run / "recovery/proposal.json"
    proposal.write_text(json.dumps({
        "correction_mode": "condition", "condition": "regrasp",
        "mode_reason": "The target instruction is wrong; direct motion is unnecessary.",
    }))
    evaluation = {
        "condition_before": "grasp", "condition": "regrasp", "start_step": 1,
        "end_step": 2, "requested_steps": 16, "executed_steps": 1,
        "complete": False, "interrupted_by": "native_horizon",
    }
    events.extend([
        {"event_type": "recovery_proposal", "episode_step": 1, "attempt": 1,
         "recorded_at": 1, "proposal": {"path": str(proposal)}},
        {"event_type": "condition_changed", "episode_step": 1, "attempt": 1,
         "recorded_at": 1, "source": "recovery_evaluation", "before": "grasp",
         "after": "regrasp", "kind": "recovery_rewrite", "reason": "Wrong target"},
        {**events[-1], "episode_step": 2, "action_source": "recovery_evaluation"},
        {"event_type": "recovery_evaluation_completed", "episode_step": 2,
         "attempt": 1, "evaluation": evaluation},
    ])
    raw = "\n".join(json.dumps(event) for event in events).encode()
    path.write_bytes(raw)
    manifest.update(step=2, events={"sha256": hashlib.sha256(raw).hexdigest()},
                    protocol_version="wam-recovery-v2", recovery_evaluation_steps=16)
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    data = load_episode(run)
    assert data["recovery_evaluation_steps"] == 16
    assert data["condition_changes"][0]["kind"] == "recovery_rewrite"
    assert data["attempts"][0]["evaluation"] == evaluation
    assert data["attempts"][0]["assessment"] is None
    assert data["attempts"][0]["executed_steps"] == 1


def test_offline_review_embeds_video_and_preserves_unknown_outcome(episode):
    from rollout_dashboard.offline import render_episode

    run, _ = episode
    document = render_episode(run)
    assert "data:video/mp4;base64," in document
    assert "&lt;script&gt;bad()&lt;/script&gt;" in document
    assert "<script>bad()</script>" not in document
    assert "Not recorded" in document
    assert "No recorded correction attempts" in document
    assert "connect-src 'none'" in document
    assert "/api/recovery/" not in document
    assert "/media/" not in document


def test_offline_review_rejects_corruption_before_writing(episode, tmp_path):
    from rollout_dashboard.offline import export_episode

    run, _ = episode
    (run / "recovery/events.jsonl").write_text("{}\n")
    output = tmp_path / "review.html"
    with pytest.raises(ValueError, match="Event log"):
        export_episode(run, output)
    assert not output.exists()


def test_offline_review_never_overwrites_existing_files(episode, tmp_path):
    from rollout_dashboard.offline import export_episode

    run, _ = episode
    output = tmp_path / "review.html"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        export_episode(run, output)
    assert output.read_text() == "keep"


def test_offline_media_rejects_external_and_escaping_paths(episode, tmp_path):
    from rollout_dashboard.offline import _embedded_media

    run, _ = episode
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"private")
    (run / "recovery/escape.mp4").symlink_to(outside)
    for url in (
        "https://example.com/video.mp4",
        "/media/another/recovery/video.mp4",
        "/media/episode/%2e%2e/outside.mp4",
        "/media/episode/recovery/escape.mp4",
    ):
        with pytest.raises(ValueError):
            _embedded_media(run, url)


def test_offline_attempt_images_and_failure_labels_preserve_source(episode, tmp_path):
    from PIL import Image
    from rollout_dashboard.offline import export_episode

    run, manifest = episode
    Image.new("RGB", (800, 600), "blue").save(run / "recovery/frame.png")
    path = run / "recovery/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    proposal = run / "recovery/proposal.json"
    proposal.write_text(json.dumps({
        "correction_mode": "action", "control_code": "<script>untrusted()</script>",
    }))
    events.extend([
        {"event_type": "recovery_proposal", "episode_step": 1, "attempt": 1,
         "recorded_at": 1, "proposal": {"path": str(proposal)}},
        {**events[-1], "episode_step": 2, "action_source": "agent_correction"},
        {"event_type": "recovery_assessment", "episode_step": 2, "attempt": 1,
         "recorded_at": 2, "assessment": {"local_recovery_success": False, "evidence": "failed"}},
    ])
    raw = "\n".join(json.dumps(event) for event in events).encode()
    path.write_bytes(raw)
    manifest.update(step=2, events={"sha256": hashlib.sha256(raw).hexdigest()})
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    output = tmp_path / "offline.html"
    export_episode(run, output)
    document = output.read_text()
    assert "Attempt 1 / action" in document
    assert document.count("data:image/jpeg;base64,") == 2
    assert "&quot;local_recovery_success&quot;: false" in document
    assert "&lt;script&gt;untrusted()&lt;/script&gt;" in document
    assert "over_shoulder_left_camera image not recorded" in document
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_unlimited_attempt_limit_is_preserved(episode):
    run, manifest = episode
    manifest["safety_limits"]["max_attempts"] = None
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    assert load_episode(run)["attempt_limit"] is None


def test_failed_budget_handoff_is_distinct_from_successful_continuation(episode):
    run, manifest = episode
    path = run / "recovery/events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events.extend([
        {"event_type": "control_returned_to_wam_after_budget", "episode_step": 1,
         "attempt": 5, "recorded_at": 1, "local_recovery_success": False},
        {**events[-1], "episode_step": 2, "action_source": "wam_after_budget"},
    ])
    raw = "\n".join(json.dumps(e) for e in events).encode()
    path.write_bytes(raw)
    manifest.update(
        step=2, events={"sha256": hashlib.sha256(raw).hexdigest()},
        safety_limits={"max_attempts": 5}, max_episode_steps=600,
        budget_exhaustion_policy="continue_wam", budget_fallback=True,
        local_recovery_success=False,
    )
    (run / "recovery/manifest.json").write_text(json.dumps(manifest))
    data = load_episode(run)
    assert data["attempt_limit"] == 5 and data["max_episode_steps"] == 600
    assert data["budget_fallback"] and data["budget_exhaustion_policy"] == "continue_wam"
    assert data["timeline"][0]["type"] == "budget_handoff"
    assert data["observations"][-1]["source"] == "wam_after_budget"
    assert data["local_recovery_success"] is False and data["verified"] is False


def test_timeline_links_each_failure_to_its_own_attempts_and_separates_wam_actions(episode):
    run, manifest = episode
    recovery = run / "recovery"
    path = recovery / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    actual = events[-1]
    first = {"failure_type": "missed_grasp", "evidence": "open fingers", "subgoal": "grasp"}
    second = {"failure_type": "lost_grasp", "evidence": "object fell", "subgoal": "retain"}
    for number, mode in [(1, "action"), (2, "condition")]:
        (recovery / f"proposal-{number}.json").write_text(json.dumps({
            "correction_mode": mode, "diagnosis": f"diagnosis {number}",
            "mode_reason": f"reason {number}",
        }))

    def proposal(number, step):
        return {"event_type": "recovery_proposal", "episode_step": step,
                "attempt": number, "recorded_at": step,
                "proposal": {"path": str(recovery / f"proposal-{number}.json")}}

    events.extend([
        {"event_type": "failure_detected", "episode_step": 1, "recorded_at": 1, "failure": first},
        proposal(1, 1),
        {**actual, "episode_step": 2, "action_source": "agent_correction"},
        {**actual, "episode_step": 3, "action_source": "recovery_evaluation"},
        {"event_type": "control_handed_back_to_wam", "episode_step": 3, "recorded_at": 3},
        {"event_type": "recovery_assessment", "episode_step": 3, "recorded_at": 3,
         "attempt": 1, "assessment": {"local_recovery_success": True, "evidence": "pinched"}},
        {"event_type": "failure_detected", "episode_step": 3, "recorded_at": 3, "failure": second},
        proposal(2, 3),
        {**actual, "episode_step": 4, "action_source": "agent_condition"},
        {"event_type": "command_failed", "episode_step": 4, "recorded_at": 4,
         "operation": "recover", "error": "<script>untrusted error</script>"},
    ])
    raw = "\n".join(json.dumps(event) for event in events).encode()
    path.write_bytes(raw)
    manifest.update(step=4, failure=second, events={"sha256": hashlib.sha256(raw).hexdigest()})
    (recovery / "manifest.json").write_text(json.dumps(manifest))
    data = load_episode(run)
    failures = [event for event in data["timeline"] if event["type"] == "failure"]
    assert [event["step"] for event in failures] == [1, 3]
    assert [event["failure"] for event in failures] == [first, second]
    assert [attempt["failure_id"] for attempt in data["attempts"]] == [event["id"] for event in failures]
    assert data["attempts"][0]["action_ranges"] == [{"start": 2, "end": 2}]
    assert data["attempts"][1]["action_ranges"] == []
    assert [attempt["proposal"]["mode_reason"] for attempt in data["attempts"]] == ["reason 1", "reason 2"]
    error = data["timeline"][-1]
    assert error["type"] == "command_error" and error["step"] == 4
    assert error["error"] == "<script>untrusted error</script>"


@pytest.mark.parametrize("source", ["wam", "recovery_evaluation", "agent_condition", "wam_after_budget"])
def test_wam_axis_uses_executed_prefix_and_original_instruction(episode, source):
    run, _ = episode
    segment = run / "recovery/segments/000"
    segment.mkdir(parents=True)
    (segment / "input.png").write_bytes(b"unit-test-image")
    prompt = "Overall task: place\nCurrent subgoal: grasp the blue handle"
    (segment / "request.json").write_text(json.dumps({
        "start_step": 0, "condition": "grasp the blue handle", "subgoal": "grasp",
        "attempt": 1 if source != "wam" else None,
    }))
    (segment / "result.json").write_text(json.dumps({
        "source": source, "executed_steps": 1, "prediction_length": 32,
        "prompt": prompt, "inference_seconds": 0.1,
        "predicted_future": {"video": {"path": str(run / "recovery/episode_wrist.mp4")}},
    }))
    item = load_episode(run)["segments"][0]
    assert item["start"] == 0 and item["end"] == 1
    assert item["executed_steps"] == 1 and item["predicted_steps"] == 32
    assert item["condition"] == "grasp the blue handle" and item["prompt"] == prompt
    assert item["subgoal"] == "grasp" and item["source"] == source
