"""Validation and compilation for bounded WAM recovery proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
import math
from typing import Any

import numpy as np


PROTOCOL_VERSION = "wam-recovery-v2"
LEGACY_PROTOCOL_VERSION = "wam-recovery-v1"
RECOVERY_EVALUATION_STEPS = 16
CONDITION_CHANGE_KINDS = {"recovery_rewrite", "subgoal_progression"}
CORRECTION_MODES = {"condition", "action", "joint"}
ALLOWED_OPERATIONS = {"move_ee", "hold", "gripper"}


@dataclass(frozen=True)
class SafetyLimits:
    workspace_min: tuple[float, float, float] = (0.15, -0.65, 0.02)
    workspace_max: tuple[float, float, float] = (0.90, 0.65, 0.90)
    max_segment_distance_m: float = 0.12
    max_total_distance_m: float = 0.30
    max_translation_per_step_m: float = 0.02
    max_operation_steps: int = 16
    max_total_steps: int = 48
    max_wam_execute_steps: int = 16
    max_attempts: int = 3
    max_rotation_per_segment_rad: float = math.pi / 4
    max_rotation_per_step_rad: float = 0.12
    max_joint_step_rad: float = 0.12

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain exactly {size} finite numbers")
    return array


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _bounded_int(value: Any, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def normalize_quaternion(value: Any, name: str = "quaternion") -> np.ndarray:
    quaternion = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if not 0.95 <= norm <= 1.05:
        raise ValueError(f"{name} must be normalized within 5%")
    return quaternion / norm


def validate_failure_decision(value: Any, request_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Failure decision must be an object")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION!r}")
    if value.get("request_id") != request_id:
        raise ValueError("Stale or incorrect request_id")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "failure_type": _nonempty(value.get("failure_type"), "failure_type"),
        "evidence": _nonempty(value.get("evidence"), "evidence"),
        "subgoal": _nonempty(value.get("subgoal"), "subgoal"),
    }


def validate_assessment(value: Any, request_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Assessment must be an object")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {PROTOCOL_VERSION!r}")
    if value.get("request_id") != request_id:
        raise ValueError("Stale or incorrect request_id")
    if type(value.get("local_recovery_success")) is not bool:
        raise ValueError("local_recovery_success must be a boolean")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "local_recovery_success": value["local_recovery_success"],
        "evidence": _nonempty(value.get("evidence"), "evidence"),
    }


def validate_recovery_proposal(
    value: Any,
    request_id: str,
    attempt: int,
    current_position: Any,
    limits: SafetyLimits,
    *,
    protocol_version: str = PROTOCOL_VERSION,
    recovery_condition: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Recovery proposal must be an object")
    if protocol_version not in {PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION}:
        raise ValueError("Unsupported recovery protocol version")
    if value.get("protocol_version") != protocol_version:
        raise ValueError(f"protocol_version must be {protocol_version!r}")
    if value.get("request_id") != request_id:
        raise ValueError("Stale or incorrect request_id")
    if attempt >= limits.max_attempts:
        raise ValueError("Recovery attempt limit exhausted")
    mode = value.get("correction_mode")
    if mode not in CORRECTION_MODES:
        raise ValueError(f"correction_mode must be one of {sorted(CORRECTION_MODES)}")
    proposal: dict[str, Any] = {
        "protocol_version": protocol_version,
        "request_id": request_id,
        "diagnosis": _nonempty(value.get("diagnosis"), "diagnosis"),
        "recovery_goal": _nonempty(value.get("recovery_goal"), "recovery_goal"),
        "correction_mode": mode,
    }
    if recovery_condition is not None or "mode_reason" in value:
        proposal["mode_reason"] = _nonempty(value.get("mode_reason"), "mode_reason")

    if mode in {"condition", "joint"}:
        proposal["condition"] = _nonempty(value.get("condition"), "condition")
        proposal["wam_execute_steps"] = _bounded_int(
            value.get("wam_execute_steps", limits.max_wam_execute_steps),
            1,
            limits.max_wam_execute_steps,
            "wam_execute_steps",
        )
    elif "condition" in value or "wam_execute_steps" in value:
        raise ValueError("Action-only correction cannot include a WAM condition")

    if mode in {"action", "joint"}:
        if "control_code" in value and "program" in value:
            raise ValueError("Provide control_code or program, not both")
        program = value.get("program")
        if "control_code" in value:
            proposal["control_code"] = _nonempty(value["control_code"], "control_code")
            program = parse_control_code(proposal["control_code"])
        if not isinstance(program, list) or not program:
            raise ValueError("Action and joint corrections require a nonempty program")
        proposal["program"] = validate_program(program, current_position, limits)
    elif "program" in value or "control_code" in value:
        raise ValueError("Condition-only correction cannot include an action program")
    if recovery_condition is not None:
        if mode in {"condition", "joint"}:
            if proposal["condition"] == recovery_condition:
                raise ValueError("Condition correction must change the current condition")
            if proposal["wam_execute_steps"] != RECOVERY_EVALUATION_STEPS:
                raise ValueError(f"Recovery evaluation requires exactly {RECOVERY_EVALUATION_STEPS} WAM steps")
    return proposal


def validate_condition_change(
    before: str | None, after: Any, *, kind: Any = None, reason: Any = None,
) -> dict[str, Any]:
    after = _nonempty(after, "condition")
    if before is None or before == after:
        if kind is not None or reason is not None:
            raise ValueError("Condition-change metadata requires a change to an existing condition")
        kind = "initial" if before is None else "unchanged"
    else:
        if not isinstance(kind, str) or kind not in CONDITION_CHANGE_KINDS:
            raise ValueError(f"condition_change_kind must be one of {sorted(CONDITION_CHANGE_KINDS)}")
        reason = _nonempty(reason, "condition_change_reason")
    return {"before": before, "after": after, "changed": before != after,
            "kind": kind, "reason": reason}


def parse_control_code(code: str) -> list[dict[str, Any]]:
    """Parse literal controller calls, never execute Python or expose the environment."""
    if len(code) > 16000:
        raise ValueError("control_code exceeds 16000 characters")
    tree = ast.parse(code, mode="exec")
    program = []
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError("Only literal controller calls are allowed")
        call = statement.value
        if (
            not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "controller"
            or call.func.attr not in ALLOWED_OPERATIONS
            or call.args
        ):
            raise ValueError("Only controller.move_ee/hold/gripper with keyword literals are allowed")
        item: dict[str, Any] = {"op": call.func.attr}
        allowed = {
            "move_ee": {"position", "quaternion", "gripper", "steps"},
            "hold": {"steps"},
            "gripper": {"value", "steps"},
        }[call.func.attr]
        for keyword in call.keywords:
            if keyword.arg not in allowed or keyword.arg in item:
                raise ValueError("Invalid or duplicate controller argument")
            item[keyword.arg] = ast.literal_eval(keyword.value)
        program.append(item)
    return program


def validate_program(
    program: list[Any],
    current_position: Any,
    limits: SafetyLimits,
) -> list[dict[str, Any]]:
    position = _finite_vector(current_position, 3, "current_position")
    total_distance = 0.0
    total_steps = 0
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(program):
        if not isinstance(raw, dict) or raw.get("op") not in ALLOWED_OPERATIONS:
            raise ValueError(f"program[{index}].op must be one of {sorted(ALLOWED_OPERATIONS)}")
        op = raw["op"]
        item: dict[str, Any] = {"op": op}
        if op == "move_ee":
            target = _finite_vector(raw.get("position"), 3, f"program[{index}].position")
            minimum = np.asarray(limits.workspace_min)
            maximum = np.asarray(limits.workspace_max)
            if np.any(target < minimum) or np.any(target > maximum):
                raise ValueError(f"program[{index}].position is outside the configured workspace")
            distance = float(np.linalg.norm(target - position))
            if distance > limits.max_segment_distance_m:
                raise ValueError(f"program[{index}] exceeds max segment distance")
            minimum_steps = max(1, math.ceil(distance / limits.max_translation_per_step_m))
            steps = _bounded_int(
                raw.get("steps", minimum_steps), minimum_steps, limits.max_operation_steps,
                f"program[{index}].steps",
            )
            item.update(position=target.tolist(), steps=steps)
            if "quaternion" in raw:
                item["quaternion"] = normalize_quaternion(
                    raw["quaternion"], f"program[{index}].quaternion"
                ).tolist()
            if "gripper" in raw:
                item["gripper"] = _gripper(raw["gripper"], f"program[{index}].gripper")
            total_distance += distance
            position = target
        elif op == "hold":
            item["steps"] = _bounded_int(
                raw.get("steps"), 1, limits.max_operation_steps, f"program[{index}].steps"
            )
        else:
            item["value"] = _gripper(raw.get("value"), f"program[{index}].value")
            item["steps"] = _bounded_int(
                raw.get("steps", 2), 1, limits.max_operation_steps, f"program[{index}].steps"
            )
        total_steps += item["steps"]
        if total_distance > limits.max_total_distance_m:
            raise ValueError("Program exceeds max total translation distance")
        if total_steps > limits.max_total_steps:
            raise ValueError("Program exceeds max total action steps")
        validated.append(item)
    return validated


def _gripper(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not np.isfinite(value) or float(value) not in (0.0, 1.0):
        raise ValueError(f"{name} must be exactly 0 (open) or 1 (closed)")
    return float(value)


def compile_program(
    program: list[dict[str, Any]],
    current_position: Any,
    current_quaternion: Any,
    current_gripper: Any,
    limits: SafetyLimits | None = None,
) -> np.ndarray:
    limits = limits or SafetyLimits()
    position = _finite_vector(current_position, 3, "current_position")
    quaternion = normalize_quaternion(current_quaternion, "current_quaternion")
    measured_gripper = float(np.asarray(current_gripper).reshape(-1)[0])
    if not np.isfinite(measured_gripper):
        raise ValueError("current_gripper must be finite")
    gripper = float(measured_gripper > 0.5)
    actions: list[np.ndarray] = []
    for item in program:
        steps = item["steps"]
        if item["op"] == "move_ee":
            target_position = _finite_vector(item["position"], 3, "position")
            target_quaternion = normalize_quaternion(
                item.get("quaternion", quaternion), "quaternion"
            )
            if float(np.dot(quaternion, target_quaternion)) < 0:
                target_quaternion = -target_quaternion
            angle = 2 * math.acos(float(np.clip(np.dot(quaternion, target_quaternion), -1, 1)))
            if angle > limits.max_rotation_per_segment_rad or angle / steps > limits.max_rotation_per_step_rad:
                raise ValueError("Program exceeds rotation limits")
            target_gripper = float(item.get("gripper", gripper))
            start_position = position.copy()
            start_quaternion = quaternion.copy()
            for step in range(1, steps + 1):
                fraction = step / steps
                interpolated_position = start_position + fraction * (target_position - start_position)
                interpolated_quaternion = (
                    (1.0 - fraction) * start_quaternion + fraction * target_quaternion
                )
                interpolated_quaternion /= np.linalg.norm(interpolated_quaternion)
                actions.append(np.concatenate(
                    [interpolated_position, interpolated_quaternion, [target_gripper]]
                ))
            position, quaternion, gripper = target_position, target_quaternion, target_gripper
        elif item["op"] == "gripper":
            gripper = item["value"]
            actions.extend(
                np.concatenate([position, quaternion, [gripper]]) for _ in range(steps)
            )
        else:
            actions.extend(
                np.concatenate([position, quaternion, [gripper]]) for _ in range(steps)
            )
    compiled = np.asarray(actions, dtype=np.float32)
    if compiled.ndim != 2 or compiled.shape[1] != 8 or not np.isfinite(compiled).all():
        raise ValueError("Compiled controller program produced invalid actions")
    return compiled
