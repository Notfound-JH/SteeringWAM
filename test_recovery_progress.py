import copy
import json

import h5py
import numpy as np
import pytest

from instruction_sampling_worker import artifact
from recovery_progress import ProgressRecoverySession
from recovery_protocol import PROGRESS_PROTOCOL_VERSION, SafetyLimits
from test_recovery_evaluation import (
    events, proposal as legacy_proposal, verify_contract, wam_decision, worker as legacy_worker,
)
from verify_recovery_episode import verify, verify_runtime_policy


def progress(**overrides):
    return {
        "target_object": "block", "target_alignment": "correct", "stage_alignment": "correct",
        "expected_stage": "grasp", "observed_stage": "grasp", "progress": "stalled",
        "evidence": "The correct block is approached but not held.", **overrides,
    }


def review(w, needs_recovery=False, **overrides):
    return w.review_progress({
        "protocol_version": PROGRESS_PROTOCOL_VERSION, "request_id": w.state["request_id"],
        "agent_role": "correction_agent", "agent_id": "corrector",
        "task_progress": progress(), "needs_recovery": needs_recovery,
        "failure_type": "missed_grasp", "evidence": "The gripper missed the block.",
        "subgoal": "grasp", **overrides,
    })


@pytest.fixture
def worker(legacy_worker):
    w = legacy_worker
    w.__class__ = ProgressRecoverySession
    w.state.update(
        protocol_version=PROGRESS_PROTOCOL_VERSION, recovery_evaluation_steps=None,
        task_progress_recovered=None, status="awaiting_progress_check",
    )
    review(w, True)
    return w


def proposal(w, mode="action", **overrides):
    return legacy_proposal(
        w, mode, protocol_version=PROGRESS_PROTOCOL_VERSION,
        agent_role="correction_agent", agent_id="corrector", task_progress=progress(),
        condition_reason="Target and stage are correct; physical alignment is missing.",
        **overrides,
    )


def wam(w, **overrides):
    return w.run_wam(wam_decision(
        w, protocol_version=PROGRESS_PROTOCOL_VERSION,
        condition=w.state.get("condition", "Grasp the block"), **overrides,
    ))


def assess(w, success=True, **overrides):
    return w.assess({
        "protocol_version": PROGRESS_PROTOCOL_VERSION, "request_id": w.state["request_id"],
        "agent_role": "recovery_judge", "agent_id": "separate-judge",
        "stage": w.state["recovery_evaluation"]["stage"],
        "correction_effect": "achieved" if success else "failed",
        "task_progress": progress(), "recovery_success": success,
        "evidence": "Compared the pre/post stage observations.",
        "continuation_reason": "Alignment supports the next native-task grasp.",
        **overrides,
    })


@pytest.mark.parametrize("mode,steps,stage", [
    ("condition", 16, "condition_progress"), ("action", 2, "action_effect"),
])
def test_mode_specific_judgment_boundary(worker, mode, steps, stage):
    start = worker.step
    worker.recover(proposal(worker, mode))
    assert worker.step == start + steps
    assert worker.state["recovery_evaluation"]["stage"] == stage
    with pytest.raises(ValueError, match="not allowed"):
        wam(worker)
    assess(worker)
    assert verify_contract(worker)["progress_contract_verified"]


def test_joint_requires_two_separate_judgments(worker):
    worker.recover(proposal(worker, "joint"))
    assert worker.step == 3
    with pytest.raises(ValueError, match="positive Action judgment"):
        worker.evaluate_condition({})
    assess(worker)
    assert worker.state["status"] == "ready_condition_evaluation"
    with pytest.raises(ValueError, match="not allowed"):
        wam(worker)
    worker.evaluate_condition({
        "protocol_version": PROGRESS_PROTOCOL_VERSION, "request_id": worker.state["request_id"],
    })
    assert worker.step == 19
    assess(worker)
    assert worker.state["status"] == "ready_wam"
    assert verify_contract(worker)["recovery_stages"] == 2


def test_failed_joint_action_does_not_execute_condition(worker):
    worker.recover(proposal(worker, "joint"))
    assess(worker, False)
    assert worker.step == 3 and worker.state["status"] == "ready_recovery"
    assert verify_contract(worker)["recovery_stages"] == 1


def test_controller_directory_contains_compiled_artifact_before_execution(worker):
    execute = worker._execute_cartesian_targets

    def check_directory(commands, directory):
        np.testing.assert_array_equal(np.load(directory / "compiled_cartesian_targets.npy"), commands)
        return execute(commands, directory)

    worker._execute_cartesian_targets = check_directory
    worker.recover(proposal(worker))
    assess(worker)
    assert verify_contract(worker)["progress_contract_verified"]


def test_chunk_progress_review_cannot_be_skipped(worker):
    worker.recover(proposal(worker))
    assess(worker)
    wam(worker)
    assert worker.state["status"] == "awaiting_progress_check"
    with pytest.raises(ValueError, match="not allowed"):
        wam(worker)
    review(worker)
    assert verify_contract(worker)["progress_reviews"] == 2


def test_user_stop_can_finalize_a_pending_final_progress_review(worker):
    worker.recover(proposal(worker))
    assess(worker)
    wam(worker)
    with pytest.raises(ValueError, match="exactly one progress review"):
        verify_contract(worker)
    worker._event(
        "session_finalized", reason="User requested early stop", final_state=copy.deepcopy(worker.state)
    )
    result = verify_contract(worker)
    assert result["unreviewed_final_wam_chunks"] == 1
    assert not worker._acceptance_success()
    incomplete = [
        e for e in events(worker)
        if e["event_type"] != "task_progress_review"
    ]
    with pytest.raises(ValueError, match="exactly one progress review"):
        verify_contract(worker, incomplete)


@pytest.mark.parametrize("tamper", ["step", "status"])
def test_pending_review_exception_requires_the_exact_finalized_boundary(worker, tamper):
    worker.recover(proposal(worker))
    assess(worker)
    wam(worker)
    worker._event(
        "session_finalized", reason="User requested early stop", final_state=copy.deepcopy(worker.state)
    )
    es = copy.deepcopy(events(worker))
    final = next(e for e in es if e["event_type"] == "session_finalized")
    final["final_state"][tamper] = worker.step - 1 if tamper == "step" else "ready_wam"
    with pytest.raises(ValueError, match="exactly one progress review"):
        verify_contract(worker, es)


@pytest.mark.parametrize("overrides,match", [
    ({"agent_id": "corrector"}, "different agent context"),
    ({"agent_role": "correction_agent"}, "agent_role"),
    ({"task_progress": progress(target_alignment="wrong")}, "aligned task progress"),
    ({"task_progress": progress(progress="uncertain")}, "aligned task progress"),
    ({"correction_effect": "uncertain"}, "aligned task progress"),
    ({"stage": "condition_progress"}, "current recovery stage"),
    ({"request_id": "stale"}, "request_id"),
])
def test_judgment_rejects_wrong_role_goal_uncertainty_or_boundary(worker, overrides, match):
    worker.recover(proposal(worker))
    with pytest.raises(ValueError, match=match):
        assess(worker, **overrides)


@pytest.mark.parametrize("mode", ["action", "joint"])
def test_condition_priority_rejects_action_for_wrong_target(worker, mode):
    p = proposal(worker, mode)
    p["task_progress"]["target_alignment"] = "wrong"
    with pytest.raises(ValueError, match="Condition takes priority"):
        worker.recover(p)
    assert worker.step == 1 and worker.attempts == 0


def test_native_success_in_condition_chunk_needs_judge_but_not_extra_wam(worker):
    worker.stop_at, worker.stop_success = 4, True
    worker.recover(proposal(worker, "condition"))
    assert not worker._acceptance_success()
    assess(worker)
    assert worker._acceptance_success() and worker.step == 4
    actions = [e["actual_action"] for e in events(worker) if e["event_type"] == "actual_execution_response"]
    with h5py.File(worker.directory / "trajectory.hdf5", "w") as handle:
        handle.create_dataset("data/demo_0/actions", data=actions)
    video = worker.directory / "episode_wrist.mp4"
    video.write_bytes(b"unit-test-video")
    manifest = {
        **worker.state, "trajectory": artifact(worker.directory / "trajectory.hdf5"),
        "events": artifact(worker.events_path), "episode_video": artifact(video),
        "mock_execution": False, "acceptance_success": True, "correction_attempts_preserved": 1,
    }
    (worker.directory / "manifest.json").write_text(json.dumps(manifest))
    assert verify(worker.directory)["acceptance_success"]


def test_achieved_local_motion_is_not_goal_aligned_recovery(worker):
    worker.recover(proposal(worker))
    assess(worker, False, correction_effect="achieved", task_progress=progress(target_alignment="wrong"))
    assert worker.state["local_recovery_success"]
    assert not worker.state["task_progress_recovered"]
    assert worker.state["status"] == "ready_recovery"


@pytest.mark.parametrize("mode,offset", [("condition", 3), ("action", 1), ("joint", 1)])
def test_native_terminal_prefix_is_judged_without_more_actions(worker, mode, offset):
    worker.stop_at = worker.step + offset
    worker.recover(proposal(worker, mode))
    assert worker.state["status"] == "awaiting_assessment"
    assess(worker, False)
    assert worker.state["status"] == "task_failed" and worker.step == worker.stop_at
    assert verify_contract(worker)["complete_recovery_stages"] == 0


def test_five_failed_attempts_continue_to_exact_600_with_progress_checks(worker):
    worker.limits = SafetyLimits(max_attempts=5)
    worker.state.update(safety_limits=worker.limits.to_dict(), budget_exhaustion_policy="continue_wam")
    for _ in range(5):
        worker.recover(proposal(worker))
        assess(worker, False)
    assert worker.state["budget_fallback"] and worker.step == 11
    worker.stop_at = 600
    while not worker.state["native_truncated"]:
        wam(worker)
        if worker.state["status"] == "awaiting_progress_check":
            review(worker, True)
    assert worker.step == 600 and worker.attempts == 5
    assert worker.state["wam_after_budget_steps"] == 589
    es = events(worker)
    actual = [e for e in es if e["event_type"] == "actual_execution_response"]
    assert verify_runtime_policy(worker.state, es, actual)["runtime_policy_verified"]
    assert verify_contract(worker)["progress_contract_verified"]


def test_regression_after_successful_final_attempt_uses_budget_fallback(worker):
    worker.limits = SafetyLimits(max_attempts=1)
    worker.state.update(safety_limits=worker.limits.to_dict(), budget_exhaustion_policy="continue_wam")
    worker.recover(proposal(worker))
    assess(worker)
    wam(worker)
    review(worker, True)
    assert worker.state["budget_fallback"] and not worker.state["task_progress_recovered"]
    wam(worker)
    review(worker, False)
    es = events(worker)
    assert verify_runtime_policy(
        worker.state, es, [e for e in es if e["event_type"] == "actual_execution_response"]
    )["runtime_policy_verified"]


@pytest.mark.parametrize("mode", ["condition", "action", "joint"])
def test_unlimited_recovery_remains_available_after_seven_failures(worker, mode):
    worker.limits = SafetyLimits(max_attempts=None)
    worker.state.update(safety_limits=worker.limits.to_dict(), budget_exhaustion_policy="continue_wam")
    for number in range(1, 8):
        overrides = {} if mode == "action" else {"condition": f"Acquire the block, attempt {number}"}
        worker.recover(proposal(worker, mode, **overrides))
        assess(worker, False)
        assert worker.attempts == number
        assert worker.state["status"] == "ready_recovery"
        assert not worker.state["budget_fallback"]
    es = events(worker)
    actual = [e for e in es if e["event_type"] == "actual_execution_response"]
    assert verify_contract(worker)["progress_contract_verified"]
    assert verify_runtime_policy(worker.state, es, actual)["runtime_policy_verified"]
    assert not any(e["event_type"] == "control_returned_to_wam_after_budget" for e in es)


def test_unlimited_recovery_handles_new_failure_after_handoff(worker):
    worker.limits = SafetyLimits(max_attempts=None)
    worker.state.update(safety_limits=worker.limits.to_dict())
    for _ in range(6):
        worker.recover(proposal(worker))
        assess(worker, False)
    worker.recover(proposal(worker))
    assess(worker)
    wam(worker)
    review(worker, True)
    assert worker.state["status"] == "ready_recovery"
    worker.recover(proposal(worker))
    assess(worker, False)
    assert worker.attempts == 8 and worker.state["status"] == "ready_recovery"
    assert not worker.state["budget_fallback"]
    assert verify_contract(worker)["progress_contract_verified"]


@pytest.mark.parametrize("tamper", ["review", "judge", "step", "source"])
def test_verifier_rejects_timing_and_identity_tampering(worker, tamper):
    worker.recover(proposal(worker))
    assess(worker)
    es = copy.deepcopy(events(worker))
    if tamper == "review":
        es = [e for e in es if e["event_type"] != "task_progress_review"]
    elif tamper == "judge":
        next(e for e in es if e["event_type"] == "recovery_assessment")["assessment"]["agent_id"] = "corrector"
    elif tamper == "step":
        next(e for e in es if e["event_type"] == "recovery_assessment")["episode_step"] += 1
    else:
        next(e for e in es if e.get("action_source") == "agent_correction"
             and e["event_type"] == "actual_execution_response")["action_source"] = "recovery_evaluation"
    with pytest.raises(ValueError):
        verify_contract(worker, es)
