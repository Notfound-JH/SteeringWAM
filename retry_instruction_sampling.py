"""One explicitly authorized extra pool at a committed checkpoint; parent stays immutable."""

import argparse
import copy
import fcntl
import json
from pathlib import Path
import socket
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import instruction_sampling as sampling


def retry_state(parent, decision, parent_dir):
    sampling.require(parent, decision["request_id"], "stopped")
    if parent["stop_reason"] != "no_acceptable_candidate" or parent["pending"]:
        raise ValueError("Only a fully reviewed no_acceptable_candidate pool can be retried")
    if parent["chunk"] >= parent["stop_chunk"] or parent["step"] >= 1500:
        raise ValueError("Committed trajectory budget exhausted")
    candidates = parent["candidates"]
    if len(candidates) != 12 or any(
        not item.get("review") or item["review"]["verdict"] not in ("reject", "uncertain")
        or item.get("result", {}).get("status") != "complete" for item in candidates
    ):
        raise ValueError("Retry requires twelve completed, reviewed candidates with no accept")
    if any(item["checkpoint"] != parent["checkpoint"] for item in candidates):
        raise ValueError("Candidate checkpoint mismatch")
    for key in ("user_request", "reason"):
        if not isinstance(decision.get(key), str) or not decision[key].strip():
            raise ValueError(f"Explicit {key} is required")
    checkpoint_key = parent["checkpoint"]["sha256"]
    counts = dict(parent.get("extra_pool_counts", {}))
    if counts.get(checkpoint_key, 0) >= 1:
        raise ValueError("One extra pool already used at this checkpoint; no third pool")
    counts[checkpoint_key] = 1
    child = copy.deepcopy(parent)
    child["rounds"].append(child["candidates"])
    child.update(candidates=[], pending=[], instruction=None, status="initializing", stop_reason=None,
                 extra_pool_counts=counts, retry_parent=str(parent_dir),
                 retry_authorization={"user_request": decision["user_request"], "reason": decision["reason"],
                                      "checkpoint": copy.deepcopy(parent["checkpoint"]), "extra_pool": 1})
    for key in ("integrity", "review_sheet", "execution_error", "decisions"):
        child.pop(key, None)
    return child


def run(parent_root, task, decision_path):
    parent_root = parent_root.resolve()
    parent_manifest = sampling.load_manifest(parent_root)
    origin = Path(parent_manifest.get("retry_origin", str(parent_root))).resolve()
    origin_manifest = sampling.load_manifest(origin)
    raw = decision_path.read_bytes()
    decision = json.loads(raw)
    with (origin / ".services.lock").open("a") as services_lock:
        fcntl.flock(services_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with sampling.locked(parent_root / task) as parent:
            state = retry_state(parent, decision, parent_root / task)
            parent_state = parent_root / task / "session.json"
            parent_hash = sampling.digest(parent_state)
            checkpoint_hash = parent["checkpoint"]["sha256"]
            destination = origin / "retries" / f"{task}_chunk_{parent['chunk']:03d}_{checkpoint_hash[:16]}_extra1"
            if destination.exists():
                raise ValueError(f"Retry already reserved; do not rerun or delete: {destination}")
            used_gpus = {slot["gpu"] for slot in parent["slots"]}
            manifests = [(origin, origin_manifest)]
            for path in sorted((origin / "retries").glob("*/experiment.json")):
                manifests.append((path.parent, sampling.load_manifest(path.parent)))
            for base, manifest in manifests:
                for name in manifest.get("activated", []):
                    other = sampling.load_state(base / name)
                    if other["status"] != "stopped" and used_gpus & {slot["gpu"] for slot in other["slots"]}:
                        raise ValueError(f"GPU slots still owned by an unfinished task: {base / name}")
            for item in parent["candidates"]:
                sampling.validate_result(item, item["result"])
            for reference in (parent["genesis"], parent["checkpoint"]):
                if sampling.digest(reference["path"]) != reference["sha256"]:
                    raise ValueError("Start checkpoint identity mismatch")
            owned_pids = [process["pid"] for _, existing in manifests
                          for process in existing["models"] + existing["workers"]
                          if sampling.process_identity(process["pid"]) == process["start_ticks"]]
            sampling.preflight(owned_pids, allow_gpu_sharing=origin_manifest.get("allow_gpu_sharing", False))
            destination.mkdir(parents=True, exist_ok=False)
            (destination / task).mkdir()
            sockets = Path(tempfile.mkdtemp(prefix="robolab_extra_pool_"))
            state["slots"] = [{**slot, "socket": str(sockets / f"{slot['gpu']}.sock")} for slot in state["slots"]]
            manifest = {"format": sampling.FORMAT, "code": sampling.identity(), "created_at": time.time(),
                        "checkpoint": origin_manifest["checkpoint"],
                        "allow_gpu_sharing": origin_manifest.get("allow_gpu_sharing", False),
                        "slots": state["slots"], "models": [], "workers": [], "active": [task], "activated": [task],
                        "retry_origin": str(origin), "retry_parent": str(parent_root),
                        "parent_state": {"path": str(parent_state), "sha256": parent_hash},
                        "extension": {"path": str(Path(__file__).resolve()), "sha256": sampling.digest(__file__)},
                        "authorization": state["retry_authorization"], "borrowed_models": []}
            (destination / "retry_decision.json").write_bytes(raw)
            sampling.write_json(destination / "experiment.json", sampling.seal(manifest))
            sampling.save_state(destination / task, state)
            for base, other_manifest in manifests:
                for worker in other_manifest["workers"]:
                    if worker["slot"]["gpu"] in used_gpus:
                        sampling.stop_owned(worker)
            for slot in state["slots"]:
                matches = [process for _, other in manifests for process in other["models"]
                           if str(origin / "services" / f"model_{slot['gpu']}") == process["directory"]
                           or process.get("gpu") == slot["gpu"]]
                live = next((item for item in matches if sampling.process_identity(item["pid"]) == item["start_ticks"]), None)
                if live:
                    if not sampling.healthy(slot["port"]):
                        raise RuntimeError("Owned model is not healthy; do not replace a live process")
                    manifest["borrowed_models"].append(copy.deepcopy(live))
                else:
                    for port in (slot["port"], slot["master_port"]):
                        with socket.socket() as port_probe:
                            port_probe.bind(("127.0.0.1", port))
                    env = sampling.environment(slot)
                    env.update(MODEL_GPU=str(slot["gpu"]), PORT=str(slot["port"]), MASTER_PORT=str(slot["master_port"]),
                               SEED=str(slot["seed"] + 100000), ALLOW_ALL_GPUS="1", DECODE_VIDEO="0")
                    target = destination / "services" / f"model_{slot['gpu']}"
                    process = sampling.spawn(["bash", str(sampling.ROOT / "launch_cosmos3_robolab_server.sh"), str(target)], env, target)
                    manifest["models"].append({**process, "gpu": slot["gpu"]})
                sampling.write_json(destination / "experiment.json", sampling.seal(manifest))
            for process in manifest["models"]:
                slot = next(item for item in state["slots"] if item["gpu"] == process["gpu"])
                sampling.wait_ready(lambda port=slot["port"]: sampling.healthy(port), process)
            for slot in state["slots"]:
                target = destination / "services" / f"{task}_{slot['gpu']}"
                command = [str(sampling.ROOT / "RoboLab/.venv/bin/python"), str(sampling.ROOT / "probe_isaac_camera.py"),
                           "--output-dir", str(target), "--task", state["task"], "--sim-gpu", str(slot["gpu"]),
                           "--seed", "3407", "--policy-port", str(slot["port"]), "--worker-socket", slot["socket"]]
                process = sampling.spawn(command, sampling.environment(slot), target)
                manifest["workers"].append({**process, "name": task, "slot": slot})
                sampling.write_json(destination / "experiment.json", sampling.seal(manifest))
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(sampling.wait_ready,
                    lambda process=process: (Path(process["directory"]) / "ready.json").exists(), process)
                    for process in manifest["workers"]]
                for future in futures:
                    future.result()
            if sampling.digest(parent_state) != parent_hash:
                raise ValueError("Parent changed during retry reservation")
            state["status"] = "need_instruction"
            sampling.save_state(destination / task, state)
            print(json.dumps({"retry_session": str(destination), "task": task, "status": state["status"],
                              "chunk": state["chunk"], "step": state["step"], "request_id": state["request_id"],
                              "checkpoint": state["checkpoint"], "extra_pool": 1}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--task", choices=sampling.TASKS, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()
    run(args.session, args.task, args.decision)


if __name__ == "__main__":
    main()