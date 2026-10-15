"""Local tools for an external visual Coding Agent; no policy or simulator imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket

from recovery_protocol import (
    CONDITION_CHANGE_KINDS, PROTOCOL_VERSION, SafetyLimits, compile_program,
    validate_condition_change, validate_recovery_proposal,
)


def request(address: Path, operation: str, decision: dict | None = None) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(900)
        connection.connect(str(address))
        connection.sendall((json.dumps({**(decision or {}), "op": operation}) + "\n").encode())
        with connection.makefile("rb") as reader:
            response = json.loads(reader.readline(64 * 1024 * 1024))
    if "error" in response:
        raise RuntimeError(response["error"] + "\n" + response.get("traceback", ""))
    return response["result"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Save the complete tool response")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("status", "context"):
        subparsers.add_parser(operation)
    wam = subparsers.add_parser("wam")
    wam.add_argument("--condition", required=True)
    wam.add_argument("--subgoal", required=True)
    wam.add_argument("--chunks", type=int, default=1)
    wam.add_argument("--execute-steps", type=int, default=16)
    wam.add_argument("--condition-change-kind", choices=sorted(CONDITION_CHANGE_KINDS))
    wam.add_argument("--condition-change-reason")
    for operation in ("mark_failure", "recover", "assess"):
        command = subparsers.add_parser(operation)
        command.add_argument("decision", type=Path)
    control = subparsers.add_parser("control")
    control.add_argument("decision", type=Path, nargs="?")
    control.add_argument("--code", help="Literal bounded controller calls, never unrestricted Python")
    control.add_argument("--diagnosis")
    control.add_argument("--goal")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.operation == "control":
        if bool(args.decision) == bool(args.code):
            parser.error("control requires either a decision file or --code")
        context = request(args.socket, "context")
        state, robot = context["state"], context["robot_state"]
        if state["status"] != "ready_agent":
            raise RuntimeError(f"Agent control cannot execute from {state['status']}")
        decision = json.loads(args.decision.read_text()) if args.decision else {
            "protocol_version": PROTOCOL_VERSION, "request_id": state["request_id"],
            "correction_mode": "action", "diagnosis": args.diagnosis,
            "recovery_goal": args.goal, "control_code": args.code,
        }
        limits = SafetyLimits(**state["safety_limits"])
        proposal = validate_recovery_proposal(
            decision, state["request_id"], 0, robot["ee_pos"], limits
        )
        compile_program(
            proposal["program"], robot["ee_pos"], robot["ee_quat"], robot["gripper_pos"], limits
        )
        result = request(args.socket, "control", decision)
    elif args.operation == "recover":
        context = request(args.socket, "context")
        state, robot = context["state"], context["robot_state"]
        if state["status"] != "ready_recovery":
            raise RuntimeError(f"Recovery cannot execute from {state['status']}")
        decision = json.loads(args.decision.read_text())
        limits = SafetyLimits(**state["safety_limits"])
        proposal = validate_recovery_proposal(
            decision, state["request_id"], state["attempts"], robot["ee_pos"], limits,
            recovery_condition=state["condition"],
        )
        if "program" in proposal:
            compile_program(
                proposal["program"], robot["ee_pos"], robot["ee_quat"], robot["gripper_pos"], limits
            )
        result = request(args.socket, "recover", decision)
    elif args.operation == "wam":
        if not 1 <= args.chunks <= 40:
            parser.error("chunks must be in [1, 40]")
        results = []
        for _ in range(args.chunks):
            state = request(args.socket, "status")
            if state["status"] != "ready_wam":
                if not results:
                    raise RuntimeError(f"WAM cannot execute from {state['status']}")
                break
            kind = args.condition_change_kind if not results else None
            reason = args.condition_change_reason if not results else None
            validate_condition_change(
                state.get("condition"), args.condition, kind=kind, reason=reason
            )
            reply = request(args.socket, "wam", {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": state["request_id"],
                "condition": args.condition,
                "subgoal": args.subgoal,
                "execute_steps": args.execute_steps,
                "condition_change_kind": kind,
                "condition_change_reason": reason,
            })
            results.append(reply)
            print(json.dumps({
                "status": reply["state"]["status"], "step": reply["state"]["step"],
                "native_success": reply["state"]["native_success"],
            }), flush=True)
        result = {"chunks": results, "state": results[-1]["state"]}
    elif args.operation == "finalize":
        state = request(args.socket, "status")
        result = request(args.socket, "finalize", {
            "request_id": state["request_id"], "reason": args.reason,
        })
    else:
        decision = json.loads(args.decision.read_text()) if hasattr(args, "decision") else None
        result = request(args.socket, args.operation, decision)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Full response: {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
