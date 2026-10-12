import copy
import json
from pathlib import Path

import numpy as np
import pytest

from instruction_sampling_worker import artifact
from recovery_protocol import PROTOCOL_VERSION, SafetyLimits
from recovery_worker import RecoverySession
from verify_recovery_episode import verify_agent_programs


@pytest.fixture
def session(tmp_path):
    worker = RecoverySession.__new__(RecoverySession)
    worker.agent_control = True
    worker.directory = tmp_path
    worker.control_programs = 0
    worker.step = 0
    worker.attempts = 0
    worker.limits = SafetyLimits()
    worker.state = {"status": "ready_agent", "request_id": "initial", "control_programs": 0}
    robot = {"ee_pos": [0.5, 0, 0.35], "ee_quat": [1, 0, 0, 0], "gripper_pos": [0]}
    events = [{"event_type": "session_started", "state": copy.deepcopy(robot)}]
    worker._robot_state = lambda: copy.deepcopy(robot)
    worker._snapshot = lambda label: {"label": label}
    worker._save_state = lambda: None

    def event(kind, **payload):
        events.append({"event_type": kind, "episode_step": worker.step, **payload})

    def execute(targets, directory, source):
        actions = np.zeros_like(targets)
        for target, action in zip(targets, actions):
            event("action_authorized", authorized_action=action.tolist(), action_source=source)
            worker.step += 1
            robot.update(ee_pos=target[:3].tolist(), ee_quat=target[3:7].tolist(),
                         gripper_pos=target[7:].tolist())
            event("actual_execution_response", actual_action=action.tolist(),
                  action_source=source, cartesian_intent=target.tolist(),
                  robot_state=copy.deepcopy(robot))
        path = directory / "executed_joint_actions.npy"
        np.save(path, actions)
        return {"executed_steps": len(actions), "executed_actions": artifact(path)}

    worker._event = event
    worker._execute_cartesian_targets = execute
    return worker, events


def decision(worker, **overrides):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": worker.state["request_id"],
        "correction_mode": "action",
        "diagnosis": "Measure free-space closure before approaching clutter.",
        "recovery_goal": "Observe closure, not task success.",
        "control_code": "controller.gripper(value=1, steps=2)",
        **overrides,
    }


def test_baseline_programs_are_bounded_and_separate_from_recovery(session):
    worker, events = session
    for _ in range(4):
        old_token = worker.state["request_id"]
        worker.control(decision(worker))
        assert worker.state["request_id"] != old_token
    assert worker.control_programs == 4
    assert worker.attempts == 0
    assert worker.state["status"] == "ready_agent"
    assert not any(e["event_type"] == "recovery_proposal" for e in events)
    actual = [e for e in events if e["event_type"] == "actual_execution_response"]
    verify_agent_programs(
        {"control_programs": 4, "safety_limits": worker.limits.to_dict()}, events, actual
    )
    path = Path(events[1]["proposal"]["path"])
    proposal = json.loads(path.read_text())
    proposal["control_code"] = "controller.hold(steps=16)"
    path.write_text(json.dumps(proposal))
    with pytest.raises(ValueError, match="checksum"):
        verify_agent_programs(
            {"control_programs": 4, "safety_limits": worker.limits.to_dict()}, events, actual
        )


@pytest.mark.parametrize("change", [
    {"request_id": "stale"},
    {"correction_mode": "condition"},
    {"control_code": "import os"},
    {"control_code": "controller.hold(steps=16)\n" * 4},
])
def test_invalid_control_cannot_execute(session, change):
    worker, events = session
    with pytest.raises(ValueError):
        worker.control(decision(worker, **change))
    assert worker.control_programs == 0
    assert worker.step == 0
    assert len(events) == 1


def test_control_cannot_bypass_recovery_mode_or_terminal_state(session):
    worker, _ = session
    worker.agent_control = False
    with pytest.raises(ValueError, match="agent-control baseline"):
        worker.control(decision(worker))
    worker.agent_control = True
    worker.state["status"] = "task_success"
    with pytest.raises(ValueError, match="ready"):
        worker.control(decision(worker))
