"""Submit parent decisions and preserve exact live inputs/responses."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from recovery_client import request
from recovery_protocol import SafetyLimits, compile_program, validate_recovery_proposal

parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["context", "wam", "review_progress", "recover", "assess", "evaluate_condition", "finalize"])
parser.add_argument("name")
parser.add_argument("--decision", type=Path)
parser.add_argument("--judge-id")
parser.add_argument("--snapshot", type=Path)
parser.add_argument("--serialize-evidence-lists", action="store_true")
parser.add_argument("--root", type=Path, default=ROOT)
args = parser.parse_args()
ROOT = args.root.resolve()
address = Path(json.loads((ROOT / "experiment.json").read_text())["socket"])
context = request(address, "context")
(ROOT / f"{args.name}_before.json").write_text(json.dumps(context, indent=2) + "\n")
if args.operation == "context":
    response = context
else:
    decision = json.loads(args.decision.read_text()) if args.decision else {}
    if args.operation == "assess":
        if not args.judge_id or not args.snapshot or not args.decision:
            raise ValueError("Assess requires raw judgment, actual Judge context ID and evaluated snapshot")
        evaluated = json.loads(args.snapshot.read_text())["state"]
        live = context["state"]
        if (evaluated["request_id"], evaluated["step"]) != (live["request_id"], live["step"]):
            raise RuntimeError("Independent judgment does not refer to the unchanged live endpoint")
        if live["status"] != "awaiting_assessment" or decision["stage"] != live["recovery_evaluation"]["stage"]:
            raise RuntimeError("Independent judgment refers to a different execution boundary")
        adapted = []
        if args.serialize_evidence_lists:
            for label, owner in (("evidence", decision), ("task_progress.evidence", decision["task_progress"])):
                value = owner["evidence"]
                if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
                    raise ValueError(f"{label} is not a nonempty evidence string list")
                owner["evidence"] = json.dumps(value)
                adapted.append(label)
        decision["agent_id"] = args.judge_id
        (ROOT / f"{args.name}_provenance.json").write_text(json.dumps({
            "agent_type": "general-purpose", "execution_mode": "sync",
            "context_id": args.judge_id,
            "input": args.snapshot.name,
            "input_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
            "raw_output": args.decision.name,
            "raw_sha256": hashlib.sha256(args.decision.read_bytes()).hexdigest(),
            "caller_modification": "Bound actual Judge context ID and verified live transport token; verdict unchanged.",
            "schema_only_list_serialization": adapted,
            "live_endpoint_verified": True,
        }, indent=2) + "\n")
    decision.update(protocol_version=context["state"]["protocol_version"], request_id=context["state"]["request_id"])
    (ROOT / f"{args.name}_submitted.json").write_text(json.dumps(decision, indent=2) + "\n")
    if args.operation == "recover":
        state, robot = context["state"], context["robot_state"]
        if state["status"] != "ready_recovery":
            raise RuntimeError(f"Cannot recover from {state['status']}")
        limits = SafetyLimits(**state["safety_limits"])
        proposal = validate_recovery_proposal(
            decision, state["request_id"], state["attempts"], robot["ee_pos"], limits,
            recovery_condition=state["condition"], protocol_version=state["protocol_version"],
        )
        if "program" in proposal:
            compile_program(proposal["program"], robot["ee_pos"], robot["ee_quat"], robot["gripper_pos"], limits)
    response = request(address, args.operation, decision)
(ROOT / f"{args.name}_response.json").write_text(json.dumps(response, indent=2) + "\n")
state = response.get("state", response)
print(json.dumps({key: state.get(key) for key in (
    "status", "step", "attempts", "native_success", "max_episode_steps",
    "hammer_task_progress", "task_progress_recovered", "budget_fallback",
)}))
