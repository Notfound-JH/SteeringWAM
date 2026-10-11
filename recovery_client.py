"""Local tools for an external visual Coding Agent; no policy or simulator imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket

from recovery_protocol import PROTOCOL_VERSION


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
    for operation in ("mark_failure", "recover", "assess"):
        command = subparsers.add_parser(operation)
        command.add_argument("decision", type=Path)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.operation == "wam":
        if not 1 <= args.chunks <= 40:
            parser.error("chunks must be in [1, 40]")
        results = []
        for _ in range(args.chunks):
            state = request(args.socket, "status")
            if state["status"] != "ready_wam":
                if not results:
                    raise RuntimeError(f"WAM cannot execute from {state['status']}")
                break
            reply = request(args.socket, "wam", {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": state["request_id"],
                "condition": args.condition,
                "subgoal": args.subgoal,
                "execute_steps": args.execute_steps,
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
