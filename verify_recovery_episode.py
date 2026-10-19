"""Verify recorded execution, not merely the episode's success label."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from recovery_protocol import (
    LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION, PROGRESS_PROTOCOL_VERSION, RECOVERY_EVALUATION_STEPS,
    SafetyLimits, compile_program, condition_continuation_evidence, condition_reacquisition_evidence,
    validate_assessment, validate_condition_change,
    validate_recovery_proposal, validate_progress_review,
)
from blue_placement_verification import STABLE_STEPS, placement_sample_passes


def verify_placement_gripper(sample: dict, robot_state: dict) -> None:
    index = robot_state["joint_names"].index("finger_joint")
    raw_closure = robot_state["measured_joint_pos"][index] / (np.pi / 4)
    np.testing.assert_allclose(sample["gripper_closure"], raw_closure, atol=1e-6)
    np.testing.assert_allclose(
        np.clip(raw_closure, 0, 1), robot_state["gripper_pos"][0], atol=1e-6
    )


def verify_agent_programs(manifest: dict, events: list[dict], actual: list[dict]) -> None:
    if any(event["action_source"] != "agent_control" for event in actual):
        raise ValueError("Agent-only baseline contains non-Agent execution")
    proposals = [event for event in events if event["event_type"] == "agent_control_proposal"]
    if len(proposals) != manifest["control_programs"]:
        raise ValueError("Agent control program count mismatch")
    authorized = [event for event in events if event["event_type"] == "action_authorized"]
    if len(authorized) != len(actual):
        raise ValueError("Missing Agent action authorization")
    for intent, response in zip(authorized, actual):
        np.testing.assert_array_equal(intent["authorized_action"], response["actual_action"])
    initial = next(event["state"] for event in events if event["event_type"] == "session_started")
    covered = 0
    limits = SafetyLimits(**manifest["safety_limits"])
    for number, event in enumerate(proposals, 1):
        if event["program_index"] != number or event["episode_step"] != covered:
            raise ValueError("Discontinuous Agent control programs")
        reference = event["proposal"]
        path = Path(reference["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("Agent proposal checksum mismatch")
        proposal = json.loads(path.read_text())
        if proposal["correction_mode"] != "action":
            raise ValueError("Agent-only baseline contains a non-action proposal")
        state = actual[covered - 1]["robot_state"] if covered else initial
        submitted = (
            {key: value for key, value in proposal.items() if key != "program"}
            if "control_code" in proposal else proposal
        )
        validated = validate_recovery_proposal(
            submitted, proposal["request_id"], 0, state["ee_pos"], limits,
            protocol_version=manifest.get("protocol_version", PROTOCOL_VERSION),
        )
        if validated["program"] != proposal["program"]:
            raise ValueError("Recorded program differs from its literal control code")
        proposal = validated
        compiled = compile_program(
            proposal["program"], state["ee_pos"], state["ee_quat"], state["gripper_pos"], limits
        )
        result = json.loads((path.parent / "result.json").read_text())
        count = result["executed_steps"]
        if not 1 <= count <= len(compiled):
            raise ValueError("Invalid Agent executed length")
        responses = actual[covered:covered + count]
        np.testing.assert_array_equal(
            np.load(path.parent / "compiled_cartesian_targets.npy"), compiled
        )
        np.testing.assert_array_equal(
            [item["cartesian_intent"] for item in responses], compiled[:count]
        )
        np.testing.assert_array_equal(
            np.load(result["executed_actions"]["path"]),
            [item["actual_action"] for item in responses],
        )
        covered += count
    if covered != len(actual):
        raise ValueError("Agent execution is not covered by bounded programs")


def verify_condition_history(manifest: dict, events: list[dict]) -> dict:
    previous = None
    conditions_before = {}
    expected_changes = []
    for event in events:
        conditions_before[event["event_index"]] = previous
        if event["event_type"] != "wam_prediction":
            continue
        change = event["condition_change"]
        expected = validate_condition_change(
            previous, change["after"],
            kind=change["kind"] if previous is not None and change["after"] != previous else None,
            reason=change["reason"],
        )
        if change != expected:
            raise ValueError("WAM condition-change metadata disagrees with history")
        request = json.loads(
            Path(event["predicted_actions"]["path"]).with_name("request.json").read_text()
        )
        if (
            request["condition_change"] != expected
            or request["condition"] != expected["after"]
            or event["prompt"] != f"Overall task: {manifest['native_goal']}\nCurrent subgoal: {expected['after']}"
            or request["prompt"] != event["prompt"]
            or request["source"] != event["source"]
            or request["attempt"] != event["attempt"]
            or request["start_step"] != event["episode_step"]
        ):
            raise ValueError("WAM request disagrees with recorded condition")
        if expected["changed"]:
            expected_changes.append({
                **expected, "source": event["source"], "attempt": event["attempt"],
                "episode_step": event["episode_step"],
            })
        previous = expected["after"]
    keys = ("before", "after", "changed", "kind", "reason", "source", "attempt", "episode_step")
    changes = [
        {key: e[key] for key in keys} for e in events if e["event_type"] == "condition_changed"
    ]
    if changes != expected_changes:
        raise ValueError("Missing or inconsistent condition-change events")
    return conditions_before


def verify_comparison_contract(manifest: dict, events: list[dict], actual: list[dict]) -> dict:
    if manifest.get("protocol_version", LEGACY_PROTOCOL_VERSION) == LEGACY_PROTOCOL_VERSION:
        return {"comparison_contract_verified": None}
    if manifest["protocol_version"] == PROGRESS_PROTOCOL_VERSION:
        return verify_progress_contract(manifest, events, actual)
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported recovery protocol version")
    if manifest.get("experiment_mode") == "agent_control":
        return {"comparison_contract_verified": None}
    if manifest["recovery_evaluation_steps"] != RECOVERY_EVALUATION_STEPS:
        raise ValueError("Unexpected recovery evaluation window")
    conditions_before = verify_condition_history(manifest, events)
    proposals = [e for e in events if e["event_type"] == "recovery_proposal"]
    if len(proposals) != manifest["attempts"]:
        raise ValueError("Recovery attempt count mismatch")
    complete = 0
    covered = set()
    for number, event in enumerate(proposals, 1):
        if event["attempt"] != number or event["episode_step"] < 1:
            raise ValueError("Discontinuous recovery attempts")
        path = Path(event["proposal"]["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != event["proposal"]["sha256"]:
            raise ValueError("Recovery proposal checksum mismatch")
        proposal = json.loads(path.read_text())
        submitted = {k: v for k, v in proposal.items() if k != "program"} if "control_code" in proposal else proposal
        before = conditions_before[event["event_index"]]
        if before is None or before != event["condition_before"]:
            raise ValueError("Recovery has no matching pre-correction condition")
        validated = validate_recovery_proposal(
            submitted, proposal["request_id"], number - 1,
            actual[event["episode_step"] - 1]["robot_state"]["ee_pos"],
            SafetyLimits(**manifest["safety_limits"]), recovery_condition=before,
        )
        if proposal != validated or event["mode_reason"] != proposal["mode_reason"]:
            raise ValueError("Recovery decision or mode rationale disagrees with proposal")
        if event["correction_mode"] != proposal["correction_mode"]:
            raise ValueError("Recovery mode disagrees with proposal")
        boundaries = {}
        for kind in ("recovery_evaluation_started", "recovery_evaluation_completed", "recovery_attempt_completed"):
            matches = [e for e in events if e["event_type"] == kind and e["attempt"] == number]
            if len(matches) != 1:
                raise ValueError(f"Expected one {kind} for attempt {number}")
            boundaries[kind] = matches[0]
        start = boundaries["recovery_evaluation_started"]
        end = boundaries["recovery_evaluation_completed"]
        completion = boundaries["recovery_attempt_completed"]
        evaluation = end["evaluation"]
        condition = proposal.get("condition", before)
        if not event["event_index"] < start["event_index"] < end["event_index"] < completion["event_index"]:
            raise ValueError("Invalid recovery evaluation ordering")
        correction = [
            e for e in actual if event["event_index"] < e["event_index"] < start["event_index"]
        ]
        if (
            any(e["action_source"] != "agent_correction" for e in correction)
            or start["episode_step"] - event["episode_step"] != len(correction)
        ):
            raise ValueError("Unexpected execution before recovery evaluation")
        if proposal["correction_mode"] == "condition":
            if correction:
                raise ValueError("Condition-only recovery executed controller actions")
        else:
            robot = actual[event["episode_step"] - 1]["robot_state"]
            compiled = compile_program(
                proposal["program"], robot["ee_pos"], robot["ee_quat"], robot["gripper_pos"],
                SafetyLimits(**manifest["safety_limits"]),
            )
            if not 1 <= len(correction) <= len(compiled):
                raise ValueError("Invalid recovery controller execution length")
            if len(correction) < len(compiled) and not (
                correction[-1]["events"]["native_terminated"] or correction[-1]["events"]["native_truncated"]
            ):
                raise ValueError("Nonterminal correction did not execute its full program")
            np.testing.assert_array_equal(
                np.load(path.parent / "compiled_cartesian_targets.npy"), compiled
            )
            np.testing.assert_array_equal(
                [e["cartesian_intent"] for e in correction], compiled[:len(correction)]
            )
            np.testing.assert_array_equal(
                np.load(path.parent / "executed_joint_actions.npy"),
                [e["actual_action"] for e in correction],
            )
        responses = [e for e in actual if start["event_index"] < e["event_index"] < end["event_index"]]
        if any(e["action_source"] != "recovery_evaluation" for e in responses):
            raise ValueError("Non-WAM execution inside recovery evaluation")
        covered.update(e["event_index"] for e in responses)
        count = len(responses)
        final_events = actual[end["episode_step"] - 1]["events"]
        interruption = (
            None if count == RECOVERY_EVALUATION_STEPS
            else "native_success" if final_events["native_success"]
            else "native_horizon" if final_events["native_truncated"]
            else "native_termination" if final_events["native_terminated"] else None
        )
        expected = {
            "condition_before": before, "condition": condition,
            "start_step": start["episode_step"], "end_step": end["episode_step"],
            "requested_steps": RECOVERY_EVALUATION_STEPS, "executed_steps": count,
            "complete": count == RECOVERY_EVALUATION_STEPS, "interrupted_by": interruption,
        }
        if (
            count > RECOVERY_EVALUATION_STEPS
            or (count < RECOVERY_EVALUATION_STEPS and interruption is None)
            or end["episode_step"] - start["episode_step"] != count
            or evaluation != expected or completion["evaluation"] != expected
            or completion["episode_step"] != end["episode_step"]
            or start["condition"] != condition or start["condition_before"] != before
            or start["correction_mode"] != proposal["correction_mode"]
            or start["requested_steps"] != RECOVERY_EVALUATION_STEPS
        ):
            raise ValueError("Recovery evaluation window or condition lock mismatch")
        predictions = [
            e for e in events if start["event_index"] < e["event_index"] < end["event_index"]
            and e["event_type"] == "wam_prediction"
        ]
        if len(predictions) != (1 if count else 0):
            raise ValueError("Recovery evaluation prediction count mismatch")
        if predictions:
            prediction = predictions[0]
            if (
                prediction["source"] != "recovery_evaluation" or prediction["attempt"] != number
                or prediction["requested_execution_length"] != RECOVERY_EVALUATION_STEPS
                or prediction["condition_change"]["after"] != condition
                or prediction["episode_step"] != start["episode_step"]
                or prediction["condition_change"]["kind"] != (
                    "unchanged" if proposal["correction_mode"] == "action" else "recovery_rewrite"
                )
                or prediction["condition_change"]["reason"] != (
                    None if proposal["correction_mode"] == "action" else proposal["mode_reason"]
                )
            ):
                raise ValueError("Recovery evaluation used an unexpected WAM condition or length")
        assessments = [
            e for e in events if e["event_type"] == "recovery_assessment" and e["attempt"] == number
        ]
        if len(assessments) > 1:
            raise ValueError("Duplicate recovery assessment")
        for assessment in assessments:
            a = assessment["assessment"]
            validate_assessment(a, a["request_id"])
            if (
                not expected["complete"] or assessment["episode_step"] != end["episode_step"]
                or assessment["event_index"] <= completion["event_index"]
                or a["evaluation_steps"] != RECOVERY_EVALUATION_STEPS
                or a["evaluation_end_step"] != end["episode_step"]
                or a["label_source"] != "agent_assessment"
            ):
                raise ValueError("Assessment is not tied to a complete evaluation window")
        complete += expected["complete"]
    if covered != {e["event_index"] for e in actual if e["action_source"] == "recovery_evaluation"}:
        raise ValueError("Unattributed recovery evaluation execution")
    return {
        "comparison_contract_verified": True, "complete_evaluation_windows": complete,
        "incomplete_evaluation_windows": len(proposals) - complete,
    }


def verify_progress_contract(manifest: dict, events: list[dict], actual: list[dict]) -> dict:
    conditions = verify_condition_history(manifest, events)
    limits = SafetyLimits(**manifest["safety_limits"])
    proposals = [e for e in events if e["event_type"] == "recovery_proposal"]
    if len(proposals) != manifest["attempts"] or manifest["recovery_evaluation_steps"] is not None:
        raise ValueError("Invalid V3 attempt count or evaluation policy")
    reviews = [e for e in events if e["event_type"] == "task_progress_review"]
    finalizations = [e for e in events if e["event_type"] == "session_finalized"]
    unreviewed_final_chunks = 0
    reviewed = set()
    for prediction in (e for e in events if e["event_type"] == "wam_prediction"):
        if prediction["source"] == "recovery_evaluation":
            continue
        result = json.loads(Path(prediction["predicted_actions"]["path"]).with_name("result.json").read_text())
        endpoint = prediction["episode_step"] + result["executed_steps"]
        last = actual[endpoint - 1]
        next_action = next((e["event_index"] for e in actual if e["episode_step"] > endpoint), len(events))
        checks = [e for e in reviews if last["event_index"] < e["event_index"] < next_action]
        terminal = last["events"]["native_terminated"] or last["events"]["native_truncated"]
        stopped_pending_review = (
            not checks and not terminal and endpoint == len(actual) == manifest["step"]
            and any(
                e["episode_step"] == endpoint and e["event_index"] > last["event_index"]
                and e["final_state"]["status"] == "awaiting_progress_check"
                and e["final_state"]["step"] == endpoint
                for e in finalizations
            )
        )
        unreviewed_final_chunks += stopped_pending_review
        if len(checks) != (0 if terminal or stopped_pending_review else 1):
            raise ValueError("Each nonterminal WAM chunk requires exactly one progress review")
        for check in checks:
            if check["episode_step"] != endpoint:
                raise ValueError("Progress review is not at its WAM boundary")
            if validate_progress_review(check["review"], check["review"]["request_id"]) != check["review"]:
                raise ValueError("Invalid task-progress review")
            reviewed.add(check["event_index"])
    if reviewed != {e["event_index"] for e in reviews}:
        raise ValueError("Unattributed task-progress review")
    covered, judges, stage_count, complete_count = set(), set(), 0, 0
    for number, event in enumerate(proposals, 1):
        if event["attempt"] != number or event["episode_step"] < 1:
            raise ValueError("Discontinuous V3 recovery attempts")
        path = Path(event["proposal"]["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != event["proposal"]["sha256"]:
            raise ValueError("V3 proposal checksum mismatch")
        proposal = json.loads(path.read_text())
        submitted = {k: v for k, v in proposal.items() if k != "program"} if "control_code" in proposal else proposal
        robot = actual[event["episode_step"] - 1]["robot_state"]
        before = conditions[event["event_index"]]
        failed_condition = None
        partial_condition = None
        if number > 1 and (
            "target_reacquisition_after_attempt" in proposal
            or "continue_condition_after_attempt" in proposal
        ):
            previous_ends = [
                e for e in events if e["event_type"] == "recovery_evaluation_completed"
                and e["attempt"] == number - 1 and e["event_index"] < event["event_index"]
            ]
            previous_assessments = [
                e for e in events if e["event_type"] == "recovery_assessment"
                and e["attempt"] == number - 1 and e["event_index"] < event["event_index"]
            ]
            if previous_ends and previous_assessments:
                previous_proposal = json.loads(Path(proposals[number - 2]["proposal"]["path"]).read_text())
                failed_condition = condition_reacquisition_evidence(
                    previous_proposal, previous_ends[-1]["evaluation"],
                    previous_assessments[-1]["assessment"], number - 1, event["episode_step"],
                )
                partial_condition = condition_continuation_evidence(
                    previous_proposal, previous_ends[-1]["evaluation"],
                    previous_assessments[-1]["assessment"], number - 1, event["episode_step"],
                )
        validated = validate_recovery_proposal(
            submitted, proposal["request_id"], number - 1, robot["ee_pos"], limits,
            protocol_version=PROGRESS_PROTOCOL_VERSION, recovery_condition=before,
            failed_condition=failed_condition,
            partial_condition=partial_condition,
        )
        if proposal != validated or event["condition_before"] != before:
            raise ValueError("V3 proposal or pre-correction condition mismatch")
        for key in ("correction_mode", "agent_role", "agent_id", "task_progress", "condition_reason", "mode_reason"):
            if event[key] != proposal[key]:
                raise ValueError("V3 proposal event differs from the saved decision")
        starts = [e for e in events if e["event_type"] == "recovery_evaluation_started" and e["attempt"] == number]
        ends = [e for e in events if e["event_type"] == "recovery_evaluation_completed" and e["attempt"] == number]
        assessments = [e for e in events if e["event_type"] == "recovery_assessment" and e["attempt"] == number]
        if not (1 <= len(starts) == len(ends) == len(assessments) <= 2):
            raise ValueError("Missing or duplicated V3 stage/judgment")
        expected_stages = ["condition_progress"] if proposal["correction_mode"] == "condition" else ["action_effect"]
        first_terminal = actual[ends[0]["episode_step"] - 1]["events"]
        if (proposal["correction_mode"] == "joint" and assessments[0]["assessment"]["recovery_success"]
                and not first_terminal["native_terminated"] and not first_terminal["native_truncated"]):
            expected_stages.append("condition_progress")
        if [e["stage"] for e in starts] != expected_stages:
            raise ValueError("Joint stages must preserve the immediate Action judgment boundary")
        previous = event
        for start, end, judgment in zip(starts, ends, assessments):
            stage = start["stage"]
            if not previous["event_index"] < start["event_index"] < end["event_index"] < judgment["event_index"]:
                raise ValueError("Invalid V3 stage ordering")
            if start["episode_step"] != previous["episode_step"]:
                raise ValueError("Unattributed action before a recovery stage")
            responses = [e for e in actual if start["event_index"] < e["event_index"] < end["event_index"]]
            if not responses:
                raise ValueError("Recovery judgment has no actual response")
            source = "agent_correction" if stage == "action_effect" else "recovery_evaluation"
            if any(e["action_source"] != source for e in responses):
                raise ValueError("Mixed Action/WAM response in a V3 recovery stage")
            predictions = [e for e in events if e["event_type"] == "wam_prediction"
                           and start["event_index"] < e["event_index"] < end["event_index"]]
            condition = before if stage == "action_effect" else proposal["condition"]
            if stage == "action_effect":
                compiled = compile_program(
                    proposal["program"], robot["ee_pos"], robot["ee_quat"], robot["gripper_pos"], limits
                )
                requested = len(compiled)
                np.testing.assert_array_equal(np.load(path.parent / "compiled_cartesian_targets.npy"), compiled)
                np.testing.assert_array_equal(np.load(path.parent / stage / "compiled_cartesian_targets.npy"), compiled)
                np.testing.assert_array_equal([e["cartesian_intent"] for e in responses], compiled[:len(responses)])
                np.testing.assert_array_equal(
                    np.load(path.parent / stage / "executed_joint_actions.npy"),
                    [e["actual_action"] for e in responses],
                )
                if predictions:
                    raise ValueError("Action judgment was contaminated by a WAM prediction")
            else:
                requested = RECOVERY_EVALUATION_STEPS
                if len(predictions) != 1 or (
                    predictions[0]["source"] != source or predictions[0]["attempt"] != number
                    or predictions[0]["condition_change"]["after"] != condition
                    or predictions[0]["requested_execution_length"] != requested
                ):
                    raise ValueError("Condition judgment requires one correctly conditioned WAM chunk")
            evaluation = end["evaluation"]
            terminal = responses[-1]["events"]
            interruption = (
                None if len(responses) == requested else
                "native_success" if terminal["native_success"] else
                "native_horizon" if terminal["native_truncated"] else
                "native_termination" if terminal["native_terminated"] else "execution_incomplete"
            )
            expected = {
                "stage": stage, "start_step": start["episode_step"], "end_step": end["episode_step"],
                "condition_before": before, "condition": condition,
                "requested_steps": requested, "executed_steps": len(responses),
                "complete": len(responses) == requested, "interrupted_by": interruption,
                "observation_before": start["observation_before"],
                "observation_after": responses[-1]["observation"],
            }
            if (
                len(responses) > requested or interruption == "execution_incomplete"
                or end["episode_step"] - start["episode_step"] != len(responses)
                or any(evaluation.get(k) != v for k, v in expected.items() if k != "observation_after")
                or start["observation_before"]["step"] != start["episode_step"]
                or evaluation["observation_after"]["step"] != end["episode_step"]
                or start["condition"] != condition or start["requested_steps"] != requested
            ):
                raise ValueError("V3 recovery stage disagrees with actual execution")
            if any(end["event_index"] < e["event_index"] < judgment["event_index"] for e in actual):
                raise ValueError("Actions occurred before the immediate recovery judgment")
            a = judgment["assessment"]
            normalized = validate_assessment(
                a, a["request_id"], protocol_version=PROGRESS_PROTOCOL_VERSION,
                stage=stage, corrector_id=proposal["agent_id"],
            )
            if (
                any(a.get(k) != v for k, v in normalized.items())
                or a["evaluation_end_step"] != end["episode_step"] or judgment["episode_step"] != end["episode_step"]
                or a["evaluation_steps"] != len(responses) or a["label_source"] != "separate_recovery_judge"
                or json.loads((path.parent / stage / "assessment.json").read_text()) != a
            ):
                raise ValueError("V3 judgment disagrees with the evaluated endpoint")
            covered.update(e["event_index"] for e in responses)
            judges.add(a["agent_id"])
            stage_count += 1
            complete_count += evaluation["complete"]
            previous = judgment
        completion = [e for e in events if e["event_type"] == "recovery_attempt_completed" and e["attempt"] == number]
        if (len(completion) != 1 or completion[0]["event_index"] <= previous["event_index"]
                or completion[0]["episode_step"] != previous["episode_step"]
                or completion[0]["evaluations"] != [e["evaluation"] for e in ends]):
            raise ValueError("V3 attempt completion does not preserve its stage judgments")
    if covered != {e["event_index"] for e in actual if e["action_source"] in {"agent_correction", "recovery_evaluation"}}:
        raise ValueError("Unattributed V3 recovery execution")
    return {
        "comparison_contract_verified": None, "progress_contract_verified": True,
        "progress_reviews": len(reviews), "recovery_stages": stage_count,
        "unreviewed_final_wam_chunks": unreviewed_final_chunks, "recovery_stages": stage_count,
        "complete_recovery_stages": complete_count, "distinct_judge_contexts": len(judges),
    }


def verify_runtime_policy(manifest: dict, events: list[dict], actual: list[dict]) -> dict:
    if "budget_exhaustion_policy" not in manifest:
        return {"runtime_policy_verified": None}
    policy = manifest["budget_exhaustion_policy"]
    if policy not in {"stop", "continue_wam"}:
        raise ValueError("Invalid budget exhaustion policy")
    horizon = manifest["max_episode_steps"]
    if not actual or (horizon is not None and (
        type(horizon) is not int or horizon < 1 or len(actual) > horizon
    )):
        raise ValueError("Execution exceeds the configured native horizon")
    if manifest["step"] != len(actual):
        raise ValueError("Manifest step differs from actual execution")
    for event in actual:
        if event["events"]["native_truncated"] and (
            horizon is None or event["episode_step"] != horizon
        ):
            raise ValueError("Native timeout differs from the configured horizon")
    if len(actual) == horizon and not (
        actual[-1]["events"]["native_terminated"] or actual[-1]["events"]["native_truncated"]
    ):
        raise ValueError("Configured horizon did not terminate the environment")
    limit = manifest["safety_limits"]["max_attempts"]
    attempts = manifest["attempts"]
    if (
        type(attempts) is not int or attempts < 0
        or (limit is not None and (
            type(limit) is not int or not 1 <= limit <= 5 or attempts > limit
        ))
    ):
        raise ValueError("Invalid or exceeded recovery attempt budget")
    handoffs = [e for e in events if e["event_type"] == "control_returned_to_wam_after_budget"]
    assessments = [e for e in events if e["event_type"] == "recovery_assessment"]
    if manifest.get("protocol_version") == PROGRESS_PROTOCOL_VERSION:
        triggers = []
        attempts_so_far = 0
        for event in events:
            attempts_so_far += event["event_type"] == "recovery_proposal"
            if attempts_so_far != limit or not event["episode_step"]:
                continue
            native = actual[event["episode_step"] - 1]["events"]
            if native["native_terminated"] or native["native_truncated"]:
                continue
            if ((event["event_type"] == "recovery_assessment" and not event["assessment"]["recovery_success"])
                    or (event["event_type"] == "task_progress_review" and event["review"]["needs_recovery"])):
                triggers.append(event)
        expected = policy == "continue_wam" and bool(triggers)
        if len(handoffs) != int(expected) or manifest["budget_fallback"] != expected:
            raise ValueError("V3 budget handoff disagrees with task-progress judgments")
        fallback = [e for e in actual if e["action_source"] == "wam_after_budget"]
        if len(fallback) != manifest["wam_after_budget_steps"]:
            raise ValueError("V3 fallback action count mismatch")
        if handoffs:
            h, trigger = handoffs[0], triggers[0]
            if (h["episode_step"] != trigger["episode_step"] or h["event_index"] <= trigger["event_index"]
                    or h["reason"] != trigger["event_type"] or h["attempt"] != limit
                    or h["task_progress_recovered"] is not False or manifest["task_progress_recovered"] is not False
                    or [e for e in actual if e["event_index"] > h["event_index"]] != fallback):
                raise ValueError("V3 failed-budget WAM continuation was mislabeled")
        elif fallback:
            raise ValueError("V3 fallback actions have no handoff")
        return {"runtime_policy_verified": True}
    exhausted_failure = any(
        e["attempt"] == limit and e["assessment"]["local_recovery_success"] is False
        for e in assessments
    )
    expected_fallback = policy == "continue_wam" and exhausted_failure
    if len(handoffs) != int(expected_fallback) or manifest["budget_fallback"] != expected_fallback:
        raise ValueError("Budget fallback label or handoff disagrees with assessments")
    fallback_actions = [e for e in actual if e["action_source"] == "wam_after_budget"]
    if len(fallback_actions) != manifest["wam_after_budget_steps"]:
        raise ValueError("Budget fallback action count mismatch")
    if handoffs:
        handoff = handoffs[0]
        assessment = next(e for e in assessments if e["attempt"] == limit)
        if (
            handoff["attempt"] != limit
            or handoff["episode_step"] != assessment["episode_step"]
            or handoff["local_recovery_success"] is not False
            or manifest["local_recovery_success"] is not False
        ):
            raise ValueError("Failed budget handoff was mislabeled or moved")
        after = [e for e in actual if e["episode_step"] > handoff["episode_step"]]
        if after != fallback_actions or manifest["attempts"] != limit:
            raise ValueError("Non-WAM execution after the budget handoff")
    elif fallback_actions:
        raise ValueError("Budget fallback actions have no failed-budget handoff")
    return {"runtime_policy_verified": True}


def verify(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    for name in ("trajectory", "events", "episode_video"):
        reference = manifest[name]
        path = Path(reference["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")
    actual = [event for event in events if event["event_type"] == "actual_execution_response"]
    if not actual or [event["episode_step"] for event in actual] != list(range(1, len(actual) + 1)):
        raise ValueError("Missing or discontinuous actual execution")
    for event in actual:
        if event["events"]["episode_step"] != event["episode_step"]:
            raise ValueError("Simulator step discontinuity")
    with h5py.File(directory / "trajectory.hdf5", "r") as handle:
        recorded = handle["data/demo_0/actions"][:]
    np.testing.assert_array_equal(recorded, np.asarray([event["actual_action"] for event in actual]))
    for result_path in sorted((directory / "segments").glob("*/result.json")):
        result = json.loads(result_path.read_text())
        predicted = np.load(result["predicted_actions"]["path"])
        executed = np.load(result["executed_actions"]["path"])
        np.testing.assert_array_equal(executed, predicted[:len(executed)])
        future = np.load(result["predicted_future"]["array"]["path"])
        if predicted.shape != (32, 8) or future.ndim != 4 or future.dtype != np.uint8:
            raise ValueError("Invalid model output shape or dtype")
    failure_indices = [i for i, event in enumerate(events) if event["event_type"] == "failure_detected"]
    recovery_indices = [
        i for i, event in enumerate(events)
        if event["event_type"] == "actual_execution_response"
        and event["action_source"] in {"agent_correction", "agent_condition", "recovery_evaluation"}
    ]
    handoff_indices = [i for i, event in enumerate(events) if event["event_type"] == "control_handed_back_to_wam"]
    continuation_indices = [
        i for i, event in enumerate(events)
        if event["event_type"] == "actual_execution_response"
        and event["action_source"] == "wam_continuation"
    ]
    ordered_chain = bool(
        failure_indices and recovery_indices and handoff_indices and continuation_indices
        and failure_indices[0] < recovery_indices[0] < handoff_indices[0] < continuation_indices[0]
    )
    accepted = bool(
        ordered_chain
        and actual[-1]["action_source"] == "wam_continuation"
        and actual[-1]["events"]["native_success"]
        and manifest["native_success"]
        and not manifest["mock_execution"]
    )
    if manifest.get("protocol_version") == PROGRESS_PROTOCOL_VERSION:
        judgments = [e for e in events if e["event_type"] == "recovery_assessment"]
        accepted = bool(
            failure_indices and recovery_indices and judgments
            and judgments[-1]["assessment"]["recovery_success"]
            and not any(e["event_type"] == "failure_detected" and e["event_index"] > judgments[-1]["event_index"] for e in events)
            and actual[-1]["action_source"] in {"wam_continuation", "recovery_evaluation"}
            and actual[-1]["events"]["native_success"] and manifest["native_success"]
            and manifest.get("task_progress_recovered") and not manifest["mock_execution"]
        )
    if accepted != manifest["acceptance_success"]:
        raise ValueError("Acceptance label disagrees with actual event sequence")
    agent_only = manifest.get("experiment_mode") == "agent_control"
    if agent_only:
        verify_agent_programs(manifest, events, actual)
    agent_success = bool(
        agent_only and actual[-1]["events"]["native_success"]
        and manifest["native_success"] and not manifest["mock_execution"]
    )
    if agent_success != manifest.get("agent_control_success", False):
        raise ValueError("Agent baseline success label disagrees with execution")
    comparison = verify_comparison_contract(manifest, events, actual)
    runtime_policy = verify_runtime_policy(manifest, events, actual)
    stable_placement = False
    if manifest.get("success_criterion") == "native_and_stable_blue_placement":
        count = 0
        for event in actual:
            sample = event["events"]["placement_verification"]
            with np.load(event["observation"]["scene"]["path"]) as scene:
                np.testing.assert_array_equal(
                    sample["object_velocity"], scene["rigid_object/blue_hammer/root_velocity"][0]
                )
            verify_placement_gripper(sample, event["robot_state"])
            if sample["gripper_command"] != event["actual_action"][-1]:
                raise ValueError("Placement gripper command disagrees with execution")
            passed = placement_sample_passes(sample)
            count = count + 1 if passed else 0
            if sample["sample_passed"] != passed or sample["stable_steps"] != count:
                raise ValueError("Placement stability history disagrees with measured samples")
            if sample["episode_step"] != event["episode_step"]:
                raise ValueError("Placement sample step mismatch")
        stable_placement = count >= STABLE_STEPS
        if bool(manifest["native_success"]) != stable_placement:
            raise ValueError("Strict completion label disagrees with placement history")
    return {
        **comparison, **runtime_policy,
        "recording_verified": True,
        "actual_steps": len(actual),
        "agent_steps": sum(event["action_source"] == "agent_correction" for event in actual),
        "wam_continuation_steps": sum(event["action_source"] == "wam_continuation" for event in actual),
        "wam_after_budget_steps": sum(event["action_source"] == "wam_after_budget" for event in actual),
        "attempts": manifest["correction_attempts_preserved"],
        "native_success": manifest["native_success"],
        "acceptance_success": accepted,
        "agent_control_steps": sum(event["action_source"] == "agent_control" for event in actual),
        "agent_control_success": agent_success,
        "stable_blue_placement_verified": stable_placement,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--require-agent-success", action="store_true")
    args = parser.parse_args()
    result = verify(args.directory.resolve())
    (args.directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 2 if (
        args.require_success and not result["acceptance_success"]
        or args.require_agent_success and not result["agent_control_success"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
