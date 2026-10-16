"""Bounded initial approach or authorized post-budget WAM, never new recovery."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from recovery_client import request

parser = argparse.ArgumentParser()
parser.add_argument("--until-step", type=int, required=True)
args = parser.parse_args()
deadline = time.monotonic() + 300
while not (ROOT / "recovery_ready.json").exists():
    if time.monotonic() > deadline:
        raise TimeoutError("Simulator did not become ready")
    time.sleep(2)
address = Path(json.loads((ROOT / "experiment.json").read_text())["socket"])
directory = ROOT / "programmatic_monitor"
directory.mkdir(exist_ok=True)
initial_condition = "Pick up the blue-handled hammer from the left bin and place it on the clear table in front of the bin."
while True:
    context = request(address, "context")
    state = context["state"]
    initial = state["attempts"] == 0 and args.until_step <= 48
    fallback = state["budget_fallback"] and state["attempts"] == state["safety_limits"]["max_attempts"] == 5
    if not initial and not fallback:
        raise RuntimeError("Only short initial approach or already-authorized budget fallback may run programmatically")
    if state["native_terminated"] or state["native_truncated"] or state["step"] >= args.until_step:
        (directory / f"endpoint_{state['step']:04d}.json").write_text(json.dumps(context, indent=2) + "\n")
        print(json.dumps({"endpoint": state["step"], "native_success": state["native_success"]}), flush=True)
        break
    if state["status"] == "awaiting_progress_check":
        evidence = (
            "Explicit programmatic monitoring, not a fresh visual GPT or independent recovery judgment. "
            f"Actual step={state['step']}, EE={context['robot_state']['ee_pos']}, "
            f"measured gripper={context['robot_state']['gripper_pos']}, "
            f"native per-hammer progress={state['hammer_task_progress']}. "
            "These measurements alone do not prove target identity, retained grasp or recovery. "
            + ("Pause for parent visual review at48." if initial else
               "Existing failed recovery remains unresolved; no sixth correction or condition rewrite is requested.")
        )
        decision = {
            "protocol_version": state["protocol_version"], "request_id": state["request_id"],
            "agent_role": "correction_agent",
            "agent_id": "copilot-session-9404bdf1-9de0-475d-a8af-e926d5fc471d",
            "execution_origin": "corrector_authored_programmatic_monitor",
            "needs_recovery": False,
            "subgoal": "Acquire, extract and place all four hammers without resetting",
            "evidence": evidence,
            "task_progress": {
                "target_object": "blue_hammer" if initial else "all_remaining_hammers",
                "expected_stage": "hammer acquisition followed by extraction and table placement",
                "observed_stage": "programmatically observed WAM response; semantic alignment unestablished",
                "target_alignment": "uncertain", "stage_alignment": "uncertain",
                "progress": "uncertain", "evidence": evidence,
            },
        }
        (directory / f"{state['step']:04d}_review.json").write_text(json.dumps(decision, indent=2) + "\n")
        state = request(address, "review_progress", decision)
    if state["status"] != "ready_wam":
        raise RuntimeError(f"Unexpected state {state['status']}; no implicit recovery or replay")
    decision = {
        "protocol_version": state["protocol_version"], "request_id": state["request_id"],
        "condition": state.get("condition") or initial_condition,
        "subgoal": "Acquire, extract and place all four hammers",
        "execute_steps": min(16, args.until_step - state["step"]),
    }
    response = request(address, "wam", decision)
    end = response["state"]
    if not state["step"] < end["step"] <= args.until_step:
        raise RuntimeError("Unexpected actual control-step progression")
    (directory / f"{end['step']:04d}_wam.json").write_text(json.dumps(response, indent=2) + "\n")
    print(json.dumps({
        "step": end["step"], "source": end["last_action_source"],
        "native_success": end["native_success"],
        "progress": end["hammer_task_progress"],
    }), flush=True)
