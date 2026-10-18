"""Opt-in V3 task-progress recovery, reusing the bounded simulator execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from instruction_sampling import write_json
from instruction_sampling_worker import artifact
from recovery_protocol import (
    PROGRESS_PROTOCOL_VERSION, RECOVERY_EVALUATION_STEPS, compile_program,
    validate_assessment, validate_progress_review, validate_recovery_proposal,
)
from recovery_worker import RecoverySession


class ProgressRecoverySession(RecoverySession):
    protocol_version = PROGRESS_PROTOCOL_VERSION

    def context(self) -> dict[str, Any]:
        context = super().context()
        context["task_progress_contract"] = {
            "priority": "Condition for target/stage deviation; Action for aligned physical blockage",
            "objective": self.state["native_goal"],
            "action_judgment": "Immediately after the controller, before further WAM actions",
            "condition_judgment": "After one WAM chunk using the revised condition",
        }
        if self.state.get("active_recovery_proposal"):
            context["proposal"] = self._proposal()
        return context

    def _proposal(self) -> dict[str, Any]:
        return json.loads(Path(self.state["active_recovery_proposal"]["path"]).read_text())

    def run_wam(self, decision: dict[str, Any]) -> dict[str, Any]:
        if (
            self.state.get("condition") is not None
            and decision.get("condition") != self.state["condition"]
            and decision.get("condition_change_kind") != "subgoal_progression"
        ):
            raise ValueError("Recovery rewrites must use a budgeted Condition or Joint attempt")
        result = super().run_wam(decision)
        if not self._terminal():
            self.state["status"] = "awaiting_progress_check"
            self._save_state()
        result["state"] = self.status()
        return result

    def _terminal(self) -> bool:
        return self.state["native_terminated"] or self.state["native_truncated"]

    def review_progress(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.state["status"] != "awaiting_progress_check":
            raise ValueError("Progress review requires a newly executed WAM chunk")
        review = validate_progress_review(decision, self.state["request_id"])
        self.state["task_progress"] = review["task_progress"]
        self._event("task_progress_review", review=review,
                    observation=self.state["latest_observation"])
        self.state["status"] = "ready_wam"
        if review["needs_recovery"]:
            self.state["task_progress_recovered"] = False
            if self.limits.allows_attempt(self.attempts):
                return super().mark_failure(review)
            self.state["failure"] = review
            self._event("failure_detected", failure=review, budget_exhausted=True,
                        observation=self.state["latest_observation"])
            self._after_failed_recovery("task_progress_review")
        self._rotate_token()
        self._save_state()
        return self.status()

    def mark_failure(self, decision: dict[str, Any]) -> dict[str, Any]:
        return self.review_progress({**decision, "needs_recovery": True})

    def _after_failed_recovery(self, reason: str) -> None:
        if self.limits.allows_attempt(self.attempts):
            self.state["status"] = "ready_recovery"
        elif self.state["budget_exhaustion_policy"] == "continue_wam":
            self.state["status"] = "ready_wam"
            if not self.state["budget_fallback"]:
                self.state["budget_fallback"] = True
                self._event(
                    "control_returned_to_wam_after_budget", attempt=self.attempts,
                    local_recovery_success=False, task_progress_recovered=False,
                    reason=reason, observation=self.state["latest_observation"],
                )
        else:
            self.state["status"] = "recovery_failed"

    def recover(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.state["status"] != "ready_recovery":
            raise ValueError(f"Recovery is not allowed from {self.state['status']}")
        robot = self._robot_state()
        proposal = validate_recovery_proposal(
            decision, self.state["request_id"], self.attempts, robot["ee_pos"], self.limits,
            protocol_version=self.protocol_version, recovery_condition=self.state["condition"],
        )
        compiled = None
        if proposal["correction_mode"] != "condition":
            compiled = compile_program(
                proposal["program"], robot["ee_pos"], robot["ee_quat"],
                robot["gripper_pos"], self.limits,
            )
        self.attempts += 1
        directory = self.directory / "recovery_attempts" / f"{self.attempts:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "proposal.json", proposal)
        (directory / "control_code.txt").write_text(
            proposal.get("control_code", json.dumps(proposal.get("program", []), indent=2)) + "\n"
        )
        self.state.update(
            status="recovering", recovery_evaluation=None, recovery_stages=[],
            active_recovery_proposal=artifact(directory / "proposal.json"),
            task_progress_recovered=False,
        )
        self._event(
            "recovery_proposal", attempt=self.attempts,
            condition_before=self.state["condition"], **{
                key: proposal[key] for key in (
                    "correction_mode", "diagnosis", "recovery_goal", "mode_reason",
                    "agent_role", "agent_id", "task_progress", "condition_reason",
                )
            }, proposal=self.state["active_recovery_proposal"],
        )
        if compiled is not None:
            path = directory / "compiled_cartesian_targets.npy"
            np.save(path, compiled)
            self._event("correction_program_compiled", attempt=self.attempts,
                        intent=proposal["program"], compiled_cartesian_targets=artifact(path))
            return self._execute_stage("action_effect", compiled)
        return self._execute_stage("condition_progress")

    def evaluate_condition(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.state["status"] != "ready_condition_evaluation":
            raise ValueError("Joint Condition evaluation requires a positive Action judgment first")
        if decision.get("request_id") != self.state["request_id"]:
            raise ValueError("Stale or incorrect request_id")
        if decision.get("protocol_version") != self.protocol_version:
            raise ValueError("Condition evaluation requires wam-recovery-v3")
        return self._execute_stage("condition_progress")

    def _execute_stage(self, stage: str, compiled: np.ndarray | None = None) -> dict[str, Any]:
        if self._terminal():
            raise ValueError("Cannot evaluate further actions after native termination")
        proposal = self._proposal()
        before, start = self.state["latest_observation"], self.step
        condition_before = self.state["condition"]
        condition = proposal["condition"] if stage == "condition_progress" else condition_before
        requested = RECOVERY_EVALUATION_STEPS if compiled is None else len(compiled)
        directory = self.directory / "recovery_attempts" / f"{self.attempts:02d}" / stage
        directory.mkdir(exist_ok=False)
        self.state["status"] = "evaluating_recovery"
        self._event(
            "recovery_evaluation_started", attempt=self.attempts, stage=stage,
            condition_before=condition_before, condition=condition,
            correction_mode=proposal["correction_mode"], requested_steps=requested,
            observation_before=before,
        )
        if compiled is not None:
            np.save(directory / "compiled_cartesian_targets.npy", compiled)
            result = self._execute_cartesian_targets(compiled, directory)
        else:
            result = self._query_wam(
                condition, self.state["failure"]["subgoal"], requested, "recovery_evaluation",
                change_kind="recovery_rewrite", change_reason=proposal["condition_reason"],
            )
        write_json(directory / "result.json", result)
        self.state["latest_observation"] = self._snapshot(f"recovery_{self.attempts}_{stage}")
        evaluation = {
            "stage": stage, "condition_before": condition_before, "condition": condition,
            "start_step": start, "end_step": self.step, "requested_steps": requested,
            "executed_steps": self.step - start, "complete": self.step - start == requested,
            "interrupted_by": (
                None if self.step - start == requested else
                "native_success" if self.state["native_success"] else
                "native_horizon" if self.state["native_truncated"] else
                "native_termination" if self.state["native_terminated"] else "execution_incomplete"
            ),
            "observation_before": before, "observation_after": self.state["latest_observation"],
        }
        if not evaluation["complete"] and not self._terminal():
            raise RuntimeError("Recovery stage stopped without native termination")
        write_json(directory / "evaluation.json", evaluation)
        self.state["recovery_evaluation"] = evaluation
        self.state["recovery_stages"].append(evaluation)
        self._event("recovery_evaluation_completed", attempt=self.attempts, evaluation=evaluation)
        self.state["status"] = "awaiting_assessment"
        self._rotate_token()
        self._save_state()
        return {"state": self.status(), "results": [result]}

    def assess(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self.state["status"] != "awaiting_assessment":
            raise ValueError("Judgment requires a newly completed recovery stage")
        evaluation, proposal = self.state["recovery_evaluation"], self._proposal()
        if evaluation["end_step"] != self.step:
            raise ValueError("The observed recovery endpoint has changed")
        assessment = validate_assessment(
            decision, self.state["request_id"], protocol_version=self.protocol_version,
            stage=evaluation["stage"], corrector_id=proposal["agent_id"],
        )
        assessment.update(
            evaluation_steps=evaluation["executed_steps"], evaluation_end_step=self.step,
            label_source="separate_recovery_judge",
        )
        self.state.update(
            local_recovery_success=assessment["local_recovery_success"],
            task_progress_recovered=assessment["recovery_success"],
            task_progress=assessment["task_progress"],
        )
        write_json(
            self.directory / "recovery_attempts" / f"{self.attempts:02d}"
            / evaluation["stage"] / "assessment.json", assessment,
        )
        self._event("recovery_assessment", attempt=self.attempts, assessment=assessment)
        joint_pending = (
            proposal["correction_mode"] == "joint" and evaluation["stage"] == "action_effect"
            and assessment["recovery_success"] and not self._terminal()
        )
        if joint_pending:
            self.state.update(status="ready_condition_evaluation", task_progress_recovered=False)
        else:
            self._event(
                "recovery_attempt_completed", attempt=self.attempts,
                evaluation=evaluation, evaluations=self.state["recovery_stages"],
                observation=self.state["latest_observation"], events=self._events(),
            )
            if self._terminal():
                self.state["status"] = "task_success" if self.state["native_success"] else "task_failed"
            elif assessment["recovery_success"]:
                self.state["status"] = "ready_wam"
                self._event("control_handed_back_to_wam", observation=self.state["latest_observation"])
            else:
                self._after_failed_recovery("recovery_assessment")
        self._rotate_token()
        self._save_state()
        return self.status()

    def _acceptance_success(self) -> bool:
        return bool(
            self.state["native_success"] and self.state["failure"]
            and self.state.get("task_progress_recovered")
            and self.state.get("last_action_source") in {"wam_continuation", "recovery_evaluation"}
            and self.state["status"] == "task_success"
        )
