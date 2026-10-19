import sys

import pytest

import recovery_client


@pytest.mark.parametrize("status", [
    "awaiting_assessment", "execution_unknown", "ready_wam", "ready_recovery",
])
def test_condition_evaluation_rejects_wrong_state_without_mutating_request(monkeypatch, status):
    calls = []

    def request(address, operation, decision=None):
        calls.append(operation)
        assert operation == "status"
        return {"status": status}

    monkeypatch.setattr(recovery_client, "request", request)
    monkeypatch.setattr(sys, "argv", ["recovery_client", "--socket", "/tmp/test.sock", "evaluate_condition"])
    with pytest.raises(RuntimeError, match="positive Action judgment"):
        recovery_client.main()
    assert calls == ["status"]


def test_condition_evaluation_submits_only_after_positive_joint_action(monkeypatch, capsys):
    calls = []
    state = {
        "status": "ready_condition_evaluation",
        "protocol_version": "wam-recovery-v3",
        "request_id": "current-joint-endpoint",
    }

    def request(address, operation, decision=None):
        calls.append((operation, decision))
        return state if operation == "status" else {"status": "awaiting_assessment"}

    monkeypatch.setattr(recovery_client, "request", request)
    monkeypatch.setattr(sys, "argv", ["recovery_client", "--socket", "/tmp/test.sock", "evaluate_condition"])
    recovery_client.main()
    assert calls == [
        ("status", None),
        ("evaluate_condition", {
            "protocol_version": state["protocol_version"], "request_id": state["request_id"],
        }),
    ]
    assert "awaiting_assessment" in capsys.readouterr().out
