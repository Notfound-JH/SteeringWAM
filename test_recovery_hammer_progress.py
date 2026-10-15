from types import SimpleNamespace

import pytest

from recovery_worker import RecoverySession


HAMMERS = ["husky_hammer", "blue_hammer", "red_hammer", "wood_hammer"]


def session(completed, *, native_success=False, task="ToolsPickingAllHammersTask"):
    calls = []
    params = {
        "object": HAMMERS.copy(), "container": "left_bin", "surface": "table",
        "logical": "all", "tolerance": 0.0, "require_gripper_detached": True,
    }

    def predicate(env, **kwargs):
        calls.append(kwargs)
        return [kwargs["object"][0] in completed]

    cfg = SimpleNamespace(func=predicate, params=params)
    worker = RecoverySession.__new__(RecoverySession)
    worker.args = SimpleNamespace(task=task, verify_blue_placement=False)
    worker.state = {"native_terminated": native_success, "native_truncated": False}
    worker.env = SimpleNamespace(
        episode_length_buf=[32],
        termination_manager=SimpleNamespace(
            get_term=lambda name: [native_success],
            get_term_cfg=lambda name: cfg,
        ),
    )
    return worker, cfg, calls


@pytest.mark.parametrize("count", range(5))
def test_per_hammer_uses_same_predicate_without_mutating_native_config(count):
    worker, cfg, calls = session(set(HAMMERS[:count]), native_success=count == 4)
    events = worker._events()
    progress = events["hammer_task_progress"]
    assert progress["completed_count"] == count
    assert progress["required_count"] == 4
    assert progress["remaining_objects"] == HAMMERS[count:]
    assert events["native_success"] is (count == 4)
    assert cfg.params["object"] == HAMMERS
    assert len(calls) == 4
    for index, call in enumerate(calls):
        assert call == {**cfg.params, "object": [HAMMERS[index]]}


def test_disagreement_does_not_report_success():
    worker, _, _ = session(set(HAMMERS), native_success=False)
    with pytest.raises(ValueError, match="disagrees"):
        worker._events()


def test_wrong_goal_checker_is_rejected():
    worker, cfg, _ = session(set())
    cfg.params["logical"] = "any"
    with pytest.raises(ValueError, match="unchanged"):
        worker._events()


def test_single_hammer_preserves_existing_event_shape():
    worker, _, calls = session(set(), task="ToolsPickingHammerTask")
    assert worker._events() == {
        "native_success": False, "native_terminated": False,
        "native_truncated": False, "episode_step": 32,
    }
    assert not calls
