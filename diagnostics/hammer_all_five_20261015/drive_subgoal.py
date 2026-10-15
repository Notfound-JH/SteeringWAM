"""Execute a specified language subgoal to a checkpoint without controller actions."""

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recovery_client import request
from recovery_protocol import validate_condition_change


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--target", choices=["wood_hammer", "husky_hammer", "blue_hammer", "red_hammer", "cordless_drill"], required=True)
parser.add_argument("--condition", required=True)
parser.add_argument("--until-step", type=int, required=True)
parser.add_argument("--kind", choices=["subgoal_progression", "recovery_rewrite"])
parser.add_argument("--reason", required=True)
args = parser.parse_args()
run = args.run.resolve()
metadata = json.loads((run / "experiment.json").read_text())
address = Path(metadata["socket"])
assert metadata["task"] == "ToolsPickingAllHammersTask"
deadline = time.monotonic() + 300
while not address.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("Simulator socket did not become ready")
    time.sleep(2)
while True:
    state = request(address, "status")
    if state["status"] in ("task_success", "task_failed"):
        result = request(address, "finalize", {
            "request_id": state["request_id"],
            "reason": "Observed-subgoal all-four-hammers episode reached native termination or600-step budget; no local controller actions.",
        })
        (run / "finalize_response.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"finalized": True, "step": state["step"], "success": state["native_success"], "progress": state["hammer_task_progress"]}), flush=True)
        break
    if state["status"] != "ready_wam":
        raise ValueError(f"Unexpected state {state['status']}; no implicit replay or reset")
    if state["step"] >= args.until_step:
        print(json.dumps({"checkpoint": True, "step": state["step"], "progress": state["hammer_task_progress"]}), flush=True)
        break
    progress = state["hammer_task_progress"]["per_object"]
    if args.target in progress and progress[args.target]:
        print(json.dumps({"target_already_complete": args.target, "step": state["step"]}), flush=True)
        break
    changed = state.get("condition") is not None and state["condition"] != args.condition
    change = validate_condition_change(
        state.get("condition"), args.condition,
        kind=args.kind if changed else None, reason=args.reason if changed else None,
    )
    decision = {
        "protocol_version": state["protocol_version"], "request_id": state["request_id"],
        "condition": args.condition,
        "subgoal": (
            "Return the incorrectly selected drill to its bin; all four hammers remain the full goal"
            if args.target == "cordless_drill"
            else f"Extract and place {args.target}; full goal is all four hammers"
        ),
        "execute_steps": min(16, args.until_step - state["step"]),
        "condition_change_kind": change["kind"] if changed else None,
        "condition_change_reason": args.reason if changed else None,
    }
    intent = {
        "episode_step": state["step"], "target": args.target,
        "condition": args.condition, "reason": args.reason, "change": change,
        "native_progress_before": state["hammer_task_progress"],
        "observation_before": state["latest_observation"],
        "decision": decision,
        "caller": "parent_copilot_assistant_selected_subgoal",
        "executor": "programmatic_chunk_loop_no_automatic_language_replanning",
    }
    with (run / "subgoal_intents.jsonl").open("a") as handle:
        handle.write(json.dumps(intent) + "\n")
    response = request(address, "wam", decision)
    state = response["state"]
    output = {
        "step": state["step"], "target": args.target,
        "native_success": state["native_success"],
        "progress": state["hammer_task_progress"],
        "executed_steps": response["result"]["executed_steps"],
        "observation_after": state["latest_observation"],
    }
    with (run / "subgoal_responses.jsonl").open("a") as handle:
        handle.write(json.dumps(output) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "observation_after"}), flush=True)
