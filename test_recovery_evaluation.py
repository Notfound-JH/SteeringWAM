import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import recovery_client
from instruction_sampling_worker import artifact
from recovery_protocol import (
    LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION, RECOVERY_EVALUATION_STEPS,
    SafetyLimits, validate_condition_change,
)
from recovery_worker import RecoverySession
from verify_recovery_episode import verify, verify_comparison_contract, verify_runtime_policy


def events(worker):
    return [json.loads(line) for line in worker.events_path.read_text().splitlines()]


def test_unlimited_horizon_runtime_and_no_hidden_timeout(worker):
    worker.state["max_episode_steps"] = None
    worker.step = 2001
    worker.env = SimpleNamespace(episode_length_buf=[2001])
    worker._before_step("wam_initial")
    actual = [{"episode_step": 2001, "action_source": "wam_initial",
               "events": {"native_truncated": False}}]
    manifest = {**worker.state, "step": 1}
    assert verify_runtime_policy(manifest, [], actual)["runtime_policy_verified"]
    actual[0]["events"]["native_truncated"] = True
    with pytest.raises(ValueError, match="Native timeout"):
        verify_runtime_policy(manifest, [], actual)
    worker.state["native_terminated"] = True
    with pytest.raises(ValueError, match="terminated"):
        worker._before_step("wam_initial")


@pytest.fixture
def worker(tmp_path):
    w = RecoverySession.__new__(RecoverySession)
    w.directory = tmp_path / "recovery"
    w.directory.mkdir()
    w.events_path = w.directory / "events.jsonl"
    w.step = w.attempts = w.segment_index = w.event_index = w.continuation_steps = 0
    w.limits = SafetyLimits()
    w.env_cfg = SimpleNamespace(instruction="Put the block in the bin")
    w.args = SimpleNamespace(output_dir=tmp_path)
    w.stop_at = None
    w.stop_success = False
    w.on_evaluation_step = None
    w.state = {
        "protocol_version": PROTOCOL_VERSION, "status": "ready_wam", "request_id": "initial",
        "native_goal": w.env_cfg.instruction, "safety_limits": w.limits.to_dict(),
        "experiment_mode": "wam_recovery", "local_recovery_success": None,
        "recovery_evaluation_steps": RECOVERY_EVALUATION_STEPS, "recovery_evaluation": None,
        "native_success": False, "native_terminated": False, "native_truncated": False,
        "budget_exhaustion_policy": "stop", "budget_fallback": False,
        "wam_after_budget_steps": 0, "max_episode_steps": 600,
    }
    robot = {"ee_pos": [0.45, 0.0, 0.35], "ee_quat": [1, 0, 0, 0], "gripper_pos": [0]}
    w._robot_state = lambda: copy.deepcopy(robot)
    w._snapshot = lambda label: {"step": w.step, "label": label}
    w._events = lambda: {
        key: w.state[key] for key in ("native_success", "native_terminated", "native_truncated")
    } | {"episode_step": w.step}
    w.observation = None
    w.client = SimpleNamespace(
        _extract_observation=lambda obs: {},
        _pack_request=lambda extracted, prompt: {
            "observation/image": np.zeros((4, 4, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7),
            "observation/gripper_position": np.zeros(1),
        },
        _unpack_response=lambda reply: reply["actions"],
        _postprocess_chunk=lambda raw: raw,
    )
    w.transport = SimpleNamespace(request=lambda *a, **kw: {
        "actions": np.zeros((32, 8), dtype=np.float32),
        "video": np.zeros((33, 4, 4, 3), dtype=np.uint8),
    })

    def save_video(array, directory):
        path = directory / "predicted_future.npy"
        np.save(path, array)
        return {"array": artifact(path)}

    w._save_predicted_video = save_video

    def execute(commands, source, directory, cartesian=False):
        actions = []
        for command in commands:
            if source == "recovery_evaluation" and w.on_evaluation_step:
                w.on_evaluation_step()
            action = np.zeros(8, dtype=np.float32) if cartesian else command
            w._event("action_authorized", authorized_action=action.tolist(), action_source=source)
            w.step += 1
            if cartesian:
                robot.update(ee_pos=command[:3].tolist(), ee_quat=command[3:7].tolist(),
                             gripper_pos=command[7:].tolist())
            if w.stop_at == w.step:
                w.state.update(
                    native_success=w.stop_success, native_terminated=w.stop_success,
                    native_truncated=not w.stop_success,
                    status="task_success" if w.stop_success else "task_failed",
                )
            w.state["last_action_source"] = source
            if source == "wam_after_budget":
                w.state["wam_after_budget_steps"] += 1
            w._event(
                "actual_execution_response", action_source=source, actual_action=action.tolist(),
                robot_state=w._robot_state(), observation=w._snapshot("actual"),
                events=w._events(), **({"cartesian_intent": command.tolist()} if cartesian else {}),
            )
            actions.append(action)
            if w.state["native_truncated"] or w.state["native_terminated"]:
                break
        path = directory / ("executed_joint_actions.npy" if cartesian else "executed_actions.npy")
        np.save(path, np.asarray(actions))
        return {"executed_steps": len(actions), "executed_actions": artifact(path), **w._events()}

    w._execute_actions = lambda commands, source, directory: execute(commands, source, directory)
    w._execute_cartesian_targets = lambda commands, directory: execute(
        commands, "agent_correction", directory, True
    )
    w.run_wam(wam_decision(w, execute_steps=1))
    w.mark_failure({
        "protocol_version": PROTOCOL_VERSION, "request_id": w.state["request_id"],
        "failure_type": "missed_grasp", "evidence": "Closed beside the block", "subgoal": "regrasp",
    })
    return w


def wam_decision(w, **overrides):
    return {
        "protocol_version": PROTOCOL_VERSION, "request_id": w.state["request_id"],
        "condition": "Grasp the block", "subgoal": "grasp", **overrides,
    }


def proposal(w, mode="action", **overrides):
    p = {
        "protocol_version": PROTOCOL_VERSION, "request_id": w.state["request_id"],
        "diagnosis": "The grasp missed.", "recovery_goal": "Recover grasp alignment.",
        "mode_reason": "The goal is unchanged; the open gripper needs alignment.",
        "correction_mode": mode,
    }
    if mode != "condition":
        p["control_code"] = "controller.hold(steps=2)"
    if mode != "action":
        p.update(condition="Keep the block held and lift", wam_execute_steps=16,
                 mode_reason="The next-stage instruction must match the recovered grasp.")
    return {**p, **overrides}


def assess(w, success=True):
    return w.assess({
        "protocol_version": PROTOCOL_VERSION, "request_id": w.state["request_id"],
        "local_recovery_success": success, "evidence": "Observed the complete evaluation window.",
    })


def verify_contract(w, es=None):
    es = events(w) if es is None else es
    actual = [e for e in es if e["event_type"] == "actual_execution_response"]
    return verify_comparison_contract(w.state, es, actual)


def test_unlimited_legacy_recovery_keeps_failed_attempts_available(worker):
    worker.limits = SafetyLimits(max_attempts=None)
    worker.state["safety_limits"] = worker.limits.to_dict()
    for number in range(1, 8):
        worker.recover(proposal(worker, "action"))
        worker.assess({
            "protocol_version": PROTOCOL_VERSION, "request_id": worker.state["request_id"],
            "local_recovery_success": False, "evidence": "Grasp remains unestablished.",
            "continuation_reason": "Another actual correction is still required.",
        })
        assert worker.attempts == number
        assert worker.state["status"] == "ready_recovery"
        assert not worker.state["budget_fallback"]
    es = events(worker)
    actual = [e for e in es if e["event_type"] == "actual_execution_response"]
    assert verify_contract(worker)["comparison_contract_verified"]
    assert verify_runtime_policy(worker.state, es, actual)["runtime_policy_verified"]


@pytest.mark.parametrize("mode", ["action", "condition", "joint"])
def test_every_mode_runs_same_window_and_assesses_afterwards(worker, mode):
    before = worker.state["condition"]
    worker.recover(proposal(worker, mode))
    evaluation = worker.state["recovery_evaluation"]
    assert evaluation["executed_steps"] == 16
    assert evaluation["complete"] and evaluation["interrupted_by"] is None
    assert evaluation["condition"] == (before if mode == "action" else "Keep the block held and lift")
    assert worker.state["status"] == "awaiting_assessment"
    assess(worker)
    assert worker.state["status"] == "ready_wam"
    assert verify_contract(worker)["complete_evaluation_windows"] == 1


def test_action_condition_cannot_change_during_window(worker):
    attempts = []

    def try_change():
        with pytest.raises(ValueError, match="not allowed"):
            worker.run_wam(wam_decision(
                worker, condition="Different target", condition_change_kind="recovery_rewrite",
                condition_change_reason="Attempted to contaminate the action comparison",
            ))
        attempts.append(1)

    worker.on_evaluation_step = try_change
    worker.recover(proposal(worker))
    assert len(attempts) == 16
    with pytest.raises(ValueError, match="not allowed"):
        worker.run_wam(wam_decision(worker, condition="Different target"))
    assert worker.state["condition"] == "Grasp the block"


@pytest.mark.parametrize("mode,overrides,match", [
    ("action", {"mode_reason": ""}, "mode_reason"),
    ("joint", {"wam_execute_steps": 8}, "exactly 16"),
    ("condition", {"condition": "Grasp the block"}, "must change"),
    ("action", {"protocol_version": LEGACY_PROTOCOL_VERSION}, "protocol_version"),
])
def test_invalid_decisions_do_not_start_execution(worker, mode, overrides, match):
    with pytest.raises(ValueError, match=match):
        worker.recover(proposal(worker, mode, **overrides))
    assert worker.step == 1 and worker.attempts == 0
    assert worker.state["status"] == "ready_recovery"


@pytest.mark.parametrize("mode,offset,expected", [
    ("action", 1, 0), ("action", 7, 5), ("condition", 5, 5),
    ("joint", 7, 5), ("action", 18, 16),
])
@pytest.mark.parametrize("success", [False, True])
def test_native_termination_does_not_pad_or_reset_evaluation(worker, mode, offset, expected, success):
    worker.stop_at = worker.step + offset
    worker.stop_success = success
    worker.recover(proposal(worker, mode))
    evaluation = worker.state["recovery_evaluation"]
    assert evaluation["executed_steps"] == expected
    assert evaluation["complete"] == (expected == 16)
    reason = "native_success" if success else "native_horizon"
    assert evaluation["interrupted_by"] == (None if expected == 16 else reason)
    with pytest.raises(ValueError, match="not allowed"):
        assess(worker)
    assert worker.state["local_recovery_success"] is None
    assert verify_contract(worker)["incomplete_evaluation_windows"] == (expected != 16)


def test_retries_keep_episode_wide_three_attempt_budget(worker):
    for _ in range(3):
        worker.recover(proposal(worker))
        assess(worker, False)
    assert worker.attempts == 3
    assert worker.state["status"] == "recovery_failed"
    with pytest.raises(ValueError, match="not allowed"):
        worker.recover(proposal(worker))
    assert verify_contract(worker)["complete_evaluation_windows"] == 3


def exhaust_five_attempts(worker):
    worker.limits = SafetyLimits(max_attempts=5)
    worker.state.update(
        safety_limits=worker.limits.to_dict(), budget_exhaustion_policy="continue_wam",
    )
    for attempt in range(5):
        worker.recover(proposal(worker))
        assess(worker, False)
        assert worker.state["status"] == ("ready_recovery" if attempt < 4 else "ready_wam")


def test_five_failed_attempts_return_to_wam_until_exact_600_step_timeout(worker):
    exhaust_five_attempts(worker)
    assert worker.state["local_recovery_success"] is False
    assert worker.state["budget_fallback"] is True
    with pytest.raises(ValueError, match="not allowed"):
        worker.recover(proposal(worker))
    with pytest.raises(ValueError, match="limit exhausted"):
        worker.mark_failure({})
    start = worker.step
    worker.stop_at = 600
    while worker.state["status"] == "ready_wam":
        worker.run_wam(wam_decision(worker))
    assert worker.step == 600
    assert worker.state["native_truncated"] and not worker.state["native_success"]
    assert worker.state["wam_after_budget_steps"] == 600 - start
    assert worker.state["last_action_source"] == "wam_after_budget"
    with pytest.raises(ValueError, match="not allowed"):
        worker.run_wam(wam_decision(worker))
    es = events(worker)
    actual = [e for e in es if e["event_type"] == "actual_execution_response"]
    assert verify_runtime_policy(worker.state, es, actual)["runtime_policy_verified"]
    assert verify_contract(worker)["complete_evaluation_windows"] == 5


@pytest.mark.parametrize("tamper", ["label", "count", "handoff", "source", "horizon"])
def test_runtime_verifier_rejects_invalid_budget_fallback(worker, tamper):
    exhaust_five_attempts(worker)
    worker.run_wam(wam_decision(worker))
    es, manifest = events(worker), copy.deepcopy(worker.state)
    if tamper == "label":
        manifest["local_recovery_success"] = True
    elif tamper == "count":
        manifest["wam_after_budget_steps"] = 0
    elif tamper == "handoff":
        es = [e for e in es if e["event_type"] != "control_returned_to_wam_after_budget"]
    elif tamper == "source":
        next(e for e in es if e.get("action_source") == "wam_after_budget")["action_source"] = "wam_continuation"
        for e in es:
            if e["event_type"] == "actual_execution_response" and e["action_source"] == "wam_after_budget":
                e["action_source"] = "wam_continuation"
                break
    else:
        manifest["max_episode_steps"] = worker.step - 1
    actual = [e for e in es if e["event_type"] == "actual_execution_response"]
    with pytest.raises(ValueError):
        verify_runtime_policy(manifest, es, actual)


def test_success_on_fifth_attempt_is_still_a_successful_recovery(worker):
    worker.limits = SafetyLimits(max_attempts=5)
    worker.state.update(safety_limits=worker.limits.to_dict(), budget_exhaustion_policy="continue_wam")
    for index in range(5):
        worker.recover(proposal(worker))
        assess(worker, index == 4)
    worker.run_wam(wam_decision(worker))
    assert not worker.state["budget_fallback"]
    assert worker.state["local_recovery_success"]
    assert worker.state["last_action_source"] == "wam_continuation"


def test_wam_success_after_failed_budget_is_not_recovery_acceptance(worker):
    exhaust_five_attempts(worker)
    worker.stop_at, worker.stop_success = worker.step + 3, True
    worker.run_wam(wam_decision(worker))
    actions = [e["actual_action"] for e in events(worker) if e["event_type"] == "actual_execution_response"]
    with h5py.File(worker.directory / "trajectory.hdf5", "w") as handle:
        handle.create_dataset("data/demo_0/actions", data=actions)
    video = worker.directory / "episode_wrist.mp4"
    video.write_bytes(b"unit-test-video")
    manifest = {
        **worker.state, "trajectory": artifact(worker.directory / "trajectory.hdf5"),
        "events": artifact(worker.events_path), "episode_video": artifact(video),
        "mock_execution": False, "acceptance_success": False, "correction_attempts_preserved": 5,
    }
    (worker.directory / "manifest.json").write_text(json.dumps(manifest))
    result = verify(worker.directory)
    assert result["native_success"] and not result["acceptance_success"]
    assert result["wam_after_budget_steps"] == 3 and result["runtime_policy_verified"]


def test_condition_changes_require_kind_and_evidence_after_window(worker):
    worker.recover(proposal(worker))
    assess(worker)
    step = worker.step
    with pytest.raises(ValueError, match="condition_change_kind"):
        worker.run_wam(wam_decision(worker, condition="Carry to the bin"))
    assert worker.step == step
    worker.run_wam(wam_decision(
        worker, condition="Carry to the bin", execute_steps=1,
        condition_change_kind="subgoal_progression",
        condition_change_reason="The grasp stage is complete.",
    ))
    change = [e for e in events(worker) if e["event_type"] == "condition_changed"][-1]
    assert change["kind"] == "subgoal_progression" and change["attempt"] is None
    assert verify_contract(worker)["comparison_contract_verified"]


def test_condition_change_schema_distinguishes_initial_unchanged_and_rewrite():
    assert validate_condition_change(None, "grasp")["kind"] == "initial"
    assert not validate_condition_change("grasp", "grasp")["changed"]
    assert validate_condition_change(
        "grasp", "regrasp", kind="recovery_rewrite", reason="The object slipped."
    )["changed"]
    with pytest.raises(ValueError, match="requires a change"):
        validate_condition_change("grasp", "grasp", kind="subgoal_progression", reason="no change")
    with pytest.raises(ValueError, match="condition_change_reason"):
        validate_condition_change("grasp", "lift", kind="subgoal_progression")


@pytest.mark.parametrize("tamper", [
    "condition", "length", "reason", "source", "assessment", "change_event", "controller",
])
def test_verifier_rejects_comparison_contract_corruption(worker, tamper):
    worker.recover(proposal(worker))
    assess(worker)
    es = events(worker)
    if tamper in {"condition", "length"}:
        evaluation = next(e["evaluation"] for e in es if e["event_type"] == "recovery_evaluation_completed")
        evaluation["condition" if tamper == "condition" else "executed_steps"] = "wrong" if tamper == "condition" else 15
    elif tamper == "reason":
        next(e for e in es if e["event_type"] == "recovery_proposal")["mode_reason"] = "different"
    elif tamper == "source":
        next(e for e in es if e.get("action_source") == "recovery_evaluation" and e["event_type"] == "actual_execution_response")["action_source"] = "agent_correction"
    elif tamper == "assessment":
        next(e for e in es if e["event_type"] == "recovery_assessment")["assessment"]["evaluation_steps"] = 8
    elif tamper == "controller":
        next(e for e in es if e.get("action_source") == "agent_correction" and e["event_type"] == "actual_execution_response")["cartesian_intent"][0] += 0.01
    else:
        es = [e for e in es if e["event_type"] != "condition_changed"]
    with pytest.raises((ValueError, AssertionError)):
        verify_contract(worker, es)


def test_legacy_history_is_not_relabelled_as_comparable():
    assert verify_comparison_contract({"protocol_version": LEGACY_PROTOCOL_VERSION}, [], []) == {
        "comparison_contract_verified": None,
    }


def test_full_recording_verifier_accepts_new_contract_without_claiming_real_success(worker):
    worker.recover(proposal(worker, "joint"))
    assess(worker)
    es = events(worker)
    actions = [e["actual_action"] for e in es if e["event_type"] == "actual_execution_response"]
    with h5py.File(worker.directory / "trajectory.hdf5", "w") as handle:
        handle.create_dataset("data/demo_0/actions", data=actions)
    video = worker.directory / "episode_wrist.mp4"
    video.write_bytes(b"mock-video")
    manifest = {
        **worker.state, "trajectory": artifact(worker.directory / "trajectory.hdf5"),
        "events": artifact(worker.events_path), "episode_video": artifact(video),
        "mock_execution": True, "acceptance_success": False, "correction_attempts_preserved": 1,
    }
    (worker.directory / "manifest.json").write_text(json.dumps(manifest))
    result = verify(worker.directory)
    assert result["recording_verified"] and result["comparison_contract_verified"]
    assert not result["acceptance_success"]


def test_client_applies_change_metadata_only_to_first_chunk(worker, monkeypatch, tmp_path):
    worker.recover(proposal(worker))
    assess(worker)
    monkeypatch.setattr(recovery_client, "request", lambda address, operation, decision=None:
                        copy.deepcopy(worker.state) if operation == "status" else worker.run_wam(decision))
    monkeypatch.setattr("sys.argv", [
        "recovery_client.py", "--socket", "/unused", "--output", str(tmp_path / "response.json"),
        "wam", "--condition", "Carry to the bin", "--subgoal", "transport", "--chunks", "2",
        "--condition-change-kind", "subgoal_progression",
        "--condition-change-reason", "The grasp stage is complete.",
    ])
    recovery_client.main()
    changes = [e for e in events(worker) if e["event_type"] == "condition_changed"]
    assert sum(e["kind"] == "subgoal_progression" for e in changes) == 1


def test_client_rejects_missing_mode_reason_before_mutating_request(worker, monkeypatch, tmp_path):
    p = proposal(worker)
    p.pop("mode_reason")
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(p))
    calls = []

    def request(address, operation, decision=None):
        calls.append(operation)
        assert operation == "context"
        return {"state": worker.state, "robot_state": worker._robot_state()}

    monkeypatch.setattr(recovery_client, "request", request)
    monkeypatch.setattr("sys.argv", ["recovery_client.py", "--socket", "/unused", "recover", str(path)])
    with pytest.raises(ValueError, match="mode_reason"):
        recovery_client.main()
    assert calls == ["context"]
