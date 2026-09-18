"""New-session, four-way language sampling. No Agent action submission."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent
FORMAT = "instruction-sampling-v3-visual-review"
TASKS = {"fruits": "FruitsOnPlate3Task", "tools": "ToolsPickingAllHammersTask",
         "blocks": "BlocksInBinTask", "black_items": "BlackItemsInBinTask",
         "organic": "ClearOrganicObjectsTask", "toys": "CleanUpToysTask"}
ROUNDS = {name: 8 if name in ("fruits", "tools") else 10 for name in TASKS}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity():
    files = [ROOT / "instruction_sampling.py", ROOT / "instruction_sampling_worker.py",
             ROOT / "probe_isaac_camera.py", ROOT / "launch_cosmos3_robolab_server.sh"]
    for base in (ROOT / "RoboLab/robolab", ROOT / "RoboLab/policies/cosmos3"):
        files.extend(sorted(base.rglob("*.py")))
    files.append(ROOT / "packages/cosmos3/cosmos_framework/scripts/action_policy_server_robolab.py")
    return {str(path.relative_to(ROOT)): digest(path) for path in files if path.exists()}


def seal(state):
    state.pop("integrity", None)
    state["integrity"] = hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return state


def load_state(directory):
    state = json.loads((Path(directory) / "session.json").read_text())
    recorded = state.get("integrity")
    if state.get("format") != FORMAT or seal(state)["integrity"] != recorded:
        raise ValueError("Unsupported session or state integrity mismatch; no legacy migration")
    if state["code"] != identity():
        raise ValueError("Source identity changed; do not edit session hashes")
    return state


@contextmanager
def locked(directory):
    directory = Path(directory)
    with (directory / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = load_state(directory)
        yield state


def save_state(directory, state):
    state["request_id"] = uuid.uuid4().hex
    write_json(Path(directory) / "session.json", seal(state))
    (Path(directory) / "PROGRESS.md").write_text(
        f"# Progress\n\nStatus: {state['status']}\n\nChunk: {state['chunk']} / {state['stop_chunk']}\n\n"
        f"Step: {state['step']}\n\nRequest: {state['request_id']}\n\n"
        f"Candidates this round: {len(state['candidates'])} / 12\n\n"
        "Only the checked session.json is authoritative. No automatic visual review.\n")


def require(state, token, *statuses):
    if token != state["request_id"]:
        raise ValueError("Stale or already consumed request_id")
    if state["status"] not in statuses:
        raise ValueError(f"Illegal state: {state['status']}; expected {statuses}")


def validate_instruction(value):
    if not isinstance(value, dict) or not all(
        isinstance(value.get(key), str) and value[key].strip()
        for key in ("subgoal", "instruction", "completion_condition", "avoid")
    ):
        raise ValueError("Provide nonempty subgoal, instruction, completion_condition and avoid")
    return {key: value[key] for key in ("subgoal", "instruction", "completion_condition", "avoid")}


def reserve_batch(state, decision):
    require(state, decision["request_id"], "ready")
    if state["chunk"] >= state["stop_chunk"] or state["step"] >= 1500:
        raise ValueError("Committed trajectory budget exhausted")
    if len(state["candidates"]) >= 12:
        raise ValueError("Twelve-candidate budget exhausted")
    instructions = decision.get("instructions", [state["instruction"]] * 4)
    if len(instructions) != 4:
        raise ValueError("A batch requires exactly four instructions")
    instructions = [validate_instruction(value) for value in instructions]
    if any(value["subgoal"] != state["instruction"]["subgoal"] for value in instructions):
        raise ValueError("Keep the current subgoal within the common-start round")
    batch = []
    for slot, instruction in zip(state["slots"], instructions):
        candidate = {"id": uuid.uuid4().hex, "slot": slot, "instruction": instruction,
                     "action_source": "cosmos", "checkpoint": state["checkpoint"],
                     "chunk": state["chunk"], "attempt": len(state["candidates"]),
                     "request_id": decision["request_id"], "review": None}
        state["candidates"].append(candidate)
        batch.append(candidate)
    state["pending"] = [candidate["id"] for candidate in batch]
    state["status"] = "execution_unknown"
    return batch


def validate_result(candidate, result):
    if result.get("status") != "complete" or result.get("candidate_id") != candidate["id"]:
        raise ValueError("Worker result is incomplete or belongs to another candidate")
    if result.get("checkpoint") != candidate["checkpoint"]:
        raise ValueError("Worker used a different checkpoint")
    if result.get("execution_receipt"):
        receipt = result["execution_receipt"]
        if digest(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Worker completion receipt changed")
        completed = json.loads(Path(receipt["path"]).read_text())
        if completed != {"candidate_id": candidate["id"], "status": "complete", "operation_exit_code": 0}:
            raise ValueError("Worker operation did not finish successfully")
    for artifact in result["artifacts"]:
        if digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"Artifact changed: {artifact['path']}")
    generated = np.load(result["processed_actions"], allow_pickle=False)
    executed = np.load(result["executed_actions"], allow_pickle=False)
    if generated.shape != (32, 8) or not np.isfinite(generated).all():
        raise ValueError("Invalid generated actions")
    if executed.shape != (result["executed_steps"], 8) or not 1 <= len(executed) <= 32:
        raise ValueError("Invalid executed prefix")
    if not np.array_equal(generated[:len(executed)].astype(np.float32), executed):
        raise ValueError("Execution differs from the model prefix")
    return result


def review_batch(state, decision):
    require(state, decision["request_id"], "pending_review")
    reviews = decision["reviews"]
    if len(reviews) != 4 or {item["candidate_id"] for item in reviews} != set(state["pending"]):
        raise ValueError("Review exactly the four pending candidates")
    checked = []
    for review in reviews:
        candidate = next(item for item in state["candidates"] if item["id"] == review["candidate_id"])
        validate_result(candidate, candidate["result"])
        if review["verdict"] not in ("accept", "reject", "uncertain"):
            raise ValueError("Invalid verdict")
        if not isinstance(review.get("evidence"), str) or not review["evidence"].strip():
            raise ValueError("Real visual evidence is required")
        if type(review.get("subgoal_complete")) is not bool:
            raise ValueError("subgoal_complete must be a boolean")
        if review["subgoal_complete"] and review["verdict"] != "accept":
            raise ValueError("Only accepted candidates can complete a subgoal")
        required = {frame["sha256"] for frame in candidate["result"]["review_frames"]}
        supplied = set(review["reviewed_frame_sha256"])
        if supplied != required:
            raise ValueError("Review all five available times in both views; never fabricate hashes")
        checked.append((candidate, review))
    for candidate, review in checked:
        candidate["review"] = review
    state["pending"] = []
    accepted = [item for item in state["candidates"] if item["review"]["verdict"] == "accept"]
    success = any(item["result"]["native_success"] for item in accepted)
    state["status"] = "need_selection" if len(state["candidates"]) == 12 or success else "ready"
    if len(state["candidates"]) == 12 and not accepted:
        state.update(status="stopped", stop_reason="no_acceptable_candidate")


def select_candidate(state, decision):
    require(state, decision["request_id"], "need_selection")
    candidate = next((item for item in state["candidates"] if item["id"] == decision["candidate_id"]), None)
    if candidate is None or candidate["review"]["verdict"] != "accept":
        raise ValueError("Select only an executed and accepted candidate")
    if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
        raise ValueError("A visual selection reason is required")
    validate_result(candidate, candidate["result"])
    if len(state["candidates"]) != 12 and not candidate["result"]["native_success"]:
        raise ValueError("Only native task success permits early selection")
    state["selected"].append({**candidate, "selection_reason": decision["reason"]})
    state["rounds"].append(state["candidates"])
    state["step"] += candidate["result"]["executed_steps"]
    state["chunk"] += 1
    state["checkpoint"] = candidate["result"]["final_checkpoint"]
    state["candidates"] = []
    state["status"] = "need_instruction"
    reason = None
    if candidate["result"]["native_success"]:
        reason = "native_success"
    elif candidate["result"]["native_terminated"] or candidate["result"]["native_truncated"]:
        reason = "native_termination"
    elif state["chunk"] >= state["stop_chunk"] or state["step"] >= 1500:
        reason = "budget_reached"
    if reason:
        state.update(status="stopped", stop_reason=reason)


def rpc(slot, command):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1200)
        connection.connect(slot["socket"])
        connection.sendall(json.dumps(command, allow_nan=False).encode() + b"\n")
        reply = json.loads(connection.makefile("rb").readline(8 * 1024 * 1024))
    if reply.get("error"):
        raise RuntimeError(reply["error"])
    return reply


def archive_decision(directory, decision_path):
    raw = Path(decision_path).read_bytes()
    decision = json.loads(raw)
    destination = Path(directory) / "decisions" / (uuid.uuid4().hex + ".json")
    destination.parent.mkdir(exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(raw)
    return decision, {"path": str(destination), "sha256": digest(destination)}


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def spawn(command, environment, directory):
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "worker.log").open("xb") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=ROOT, start_new_session=True)
    record = {"pid": process.pid, "start_ticks": process_identity(process.pid), "command": command,
              "directory": str(directory), "started_at": time.time()}
    write_json(directory / "process.json", record)
    return record


def wait_ready(check, process, timeout=1200):
    deadline = time.monotonic() + timeout
    waiter = threading.Event()
    while time.monotonic() < deadline:
        if process_identity(process["pid"]) != process["start_ticks"]:
            raise RuntimeError(f"Owned process exited; inspect {process['directory']}/worker.log")
        if check():
            return
        waiter.wait(1)
    raise TimeoutError(f"Readiness timed out; no restart: {process['directory']}")


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def preflight(owned_pids=(), allow_gpu_sharing=False):
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                    "--format=csv,noheader,nounits"], text=True)
    busy = [row for row in csv.reader(text.splitlines()) if row and int(row[1]) not in owned_pids]
    if busy and not allow_gpu_sharing:
        raise RuntimeError(f"GPUs have unrelated compute processes; refusing to share or kill them: {busy}")
    if busy:
        print(json.dumps({"gpu_sharing_authorized": True, "existing_compute_processes": busy}), flush=True)
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    devices = {int(row[0]): row[1].strip() for row in csv.reader(text.splitlines())}
    if set(devices) != set(range(8)):
        raise RuntimeError("This experiment requires physical GPUs 0..7")
    return devices


def environment(slot):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=slot["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
               LD_LIBRARY_PATH=str(ROOT / "packages/ffmpeg/lib"),
               PATH=str(ROOT / "packages/ffmpeg/bin") + os.pathsep + env.get("PATH", ""),
               PYTHONUNBUFFERED="1", OMNI_KIT_ACCEPT_EULA="YES", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               NO_PROXY="127.0.0.1,localhost")
    return env


def task_state(manifest, name):
    slots = manifest["slots"][4:] if name == "tools" else manifest["slots"][:4]
    return {"format": FORMAT, "code": manifest["code"], "task": TASKS[name], "name": name,
            "slots": slots, "status": "queued", "request_id": uuid.uuid4().hex,
            "chunk": 0, "step": 0, "stop_chunk": ROUNDS[name], "execute_steps": 16,
            "genesis": None, "checkpoint": None, "instruction": None, "pending": [],
            "selected": [], "rounds": [], "candidates": [], "stop_reason": None}


def start_experiment(directory, base_port, master_port, allow_gpu_sharing=False):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("New experiments require a nonexistent session directory")
    devices = preflight(allow_gpu_sharing=allow_gpu_sharing)
    ports = [base_port + index for index in range(8)] + [master_port + index for index in range(8)]
    if len(set(ports)) != 16 or any(not 1024 <= port <= 65535 for port in ports):
        raise ValueError("Use two distinct valid eight-port ranges")
    for port in ports:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    directory.mkdir(parents=True)
    sockets = Path("/tmp") / ("robolab_sampling_" + uuid.uuid4().hex)
    sockets.mkdir(mode=0o700)
    manifest = {"format": FORMAT, "code": identity(), "created_at": time.time(),
                "allow_gpu_sharing": allow_gpu_sharing,
                "checkpoint": str(ROOT / "models/Cosmos3-Edge-Policy-DROID"),
                "slots": [{"gpu": index, "uuid": devices[index], "port": base_port + index,
                           "master_port": master_port + index, "seed": 3407 + index,
                           "socket": str(sockets / f"{index}.sock")} for index in range(8)],
                "models": [], "workers": [], "active": [], "activated": []}
    write_json(directory / "experiment.json", seal(manifest))
    for name in TASKS:
        task_dir = directory / name
        task_dir.mkdir()
        save_state(task_dir, task_state(manifest, name))
    for slot in manifest["slots"]:
        env = environment(slot)
        env.update(MODEL_GPU=str(slot["gpu"]), PORT=str(slot["port"]), MASTER_PORT=str(slot["master_port"]),
                   SEED=str(slot["seed"]), ALLOW_ALL_GPUS="1", DECODE_VIDEO="0")
        target = directory / "services" / f"model_{slot['gpu']}"
        process = spawn(["bash", str(ROOT / "launch_cosmos3_robolab_server.sh"), str(target)], env, target)
        manifest["models"].append(process)
        write_json(directory / "experiment.json", seal(manifest))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(wait_ready, lambda port=slot["port"]: healthy(port), process)
                   for slot, process in zip(manifest["slots"], manifest["models"])]
        for future in futures:
            future.result()
    activate_tasks(directory, manifest, ["fruits", "tools"])


def activate_tasks(directory, manifest, names):
    for name in names:
        task_dir = directory / name
        with locked(task_dir) as state:
            if state["status"] != "queued":
                raise ValueError("Task already activated; do not reinitialize it")
            state["status"] = "initializing"
            save_state(task_dir, state)
            workers = []
            for slot in state["slots"]:
                target = directory / "services" / f"{name}_{slot['gpu']}"
                command = [str(ROOT / "RoboLab/.venv/bin/python"), str(ROOT / "probe_isaac_camera.py"),
                           "--output-dir", str(target), "--task", state["task"], "--sim-gpu", str(slot["gpu"]),
                           "--seed", "3407", "--policy-port", str(slot["port"]), "--worker-socket", slot["socket"]]
                process = spawn(command, environment(slot), target)
                manifest["workers"].append({**process, "name": name, "slot": slot})
                workers.append(process)
                write_json(directory / "experiment.json", seal(manifest))
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(wait_ready, lambda proc=process: (Path(proc["directory"]) / "ready.json").exists(), process)
                           for process in workers]
                for future in futures:
                    future.result()
            reply = rpc(state["slots"][0], {"op": "initial", "code": state["code"]})
            state.update(checkpoint=reply["checkpoint"], genesis=reply["checkpoint"], status="need_instruction")
            save_state(task_dir, state)
        manifest["activated"].append(name)
        manifest["active"].append(name)
        write_json(directory / "experiment.json", seal(manifest))


def load_manifest(directory):
    manifest = json.loads((directory / "experiment.json").read_text())
    checksum = manifest.get("integrity")
    if manifest.get("format") != FORMAT or seal(manifest)["integrity"] != checksum or manifest["code"] != identity():
        raise ValueError("Experiment format, integrity or source identity mismatch")
    return manifest


def stop_owned(process):
    pid = process["pid"]
    if process_identity(pid) != process["start_ticks"]:
        return
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    waiter = threading.Event()
    while process_identity(pid) == process["start_ticks"] and time.monotonic() < deadline:
        waiter.wait(0.2)
    if process_identity(pid) == process["start_ticks"]:
        raise RuntimeError(f"Owned process {pid} has not exited; no forced kill")


def advance_queue(directory):
    manifest = load_manifest(directory)
    if any(load_state(directory / name)["status"] != "stopped" for name in manifest["active"]):
        raise ValueError("Finish and export active tasks before advancing the queue")
    for name in manifest["active"]:
        if not (directory / name / "export/manifest.json").exists():
            raise ValueError(f"Export {name} before advancing")
    for worker in manifest["workers"]:
        if worker["name"] in manifest["active"]:
            stop_owned(worker)
            Path(worker["slot"]["socket"]).unlink(missing_ok=True)
    manifest["active"] = []
    write_json(directory / "experiment.json", seal(manifest))
    owned = [process["pid"] for process in manifest["models"] if process_identity(process["pid"]) == process["start_ticks"]]
    preflight(owned, allow_gpu_sharing=manifest.get("allow_gpu_sharing", False))
    following = next((name for name in TASKS if name not in manifest["activated"]), None)
    if following:
        activate_tasks(directory, manifest, [following])
    else:
        print("All six task budgets finished; use stop-services to stop owned model servers.")


def batch_sheet(directory, batch):
    from PIL import Image

    images = []
    for candidate in batch:
        reference = candidate["result"]["sheet"]
        if digest(reference["path"]) != reference["sha256"]:
            raise ValueError("Review sheet changed")
        with Image.open(reference["path"]) as source:
            images.append(source.convert("RGB"))
    sheet = Image.new("RGB", (max(image.width for image in images), sum(image.height for image in images)), "white")
    offset = 0
    for image in images:
        sheet.paste(image, (0, offset))
        offset += image.height
    path = directory / f"review_chunk_{batch[0]['chunk']:03d}_batch_{batch[0]['attempt'] // 4}.png"
    if path.exists():
        raise ValueError("Review sheet already exists")
    sheet.save(path)
    return str(path)


def import_batch(directory, state):
    batch = [item for item in state["candidates"] if item["id"] in state["pending"]]
    results = []
    for candidate in batch:
        path = directory / "candidates" / candidate["id"] / "result.json"
        result = validate_result(candidate, json.loads(path.read_text()))
        if not result.get("execution_receipt"):
            raise ValueError("Missing worker completion receipt; do not retry execution")
        if result["source_identity"] != state["code"]:
            raise ValueError("Worker source differs from session")
        if result["executed_steps"] > min(state["execute_steps"], 1500 - state["step"]):
            raise ValueError("Worker exceeded the authorized execution prefix")
        request = json.loads((path.parent / "request.json").read_text())
        reservation = {key: value for key, value in candidate.items() if key != "result"}
        if request["candidate"] != reservation or request["genesis"] != state["genesis"]:
            raise ValueError("Worker request differs from the reserved candidate/start")
        results.append(result)
    for candidate, result in zip(batch, results):
        candidate["result"] = result
    state["review_sheet"] = batch_sheet(directory, batch)
    state["status"] = "pending_review"
    save_state(directory, state)


def run_batch(directory, state, decision):
    batch = reserve_batch(state, decision)
    for candidate in batch:
        candidate["decision"] = state["decisions"][-1]
    save_state(directory, state)

    def execute(candidate):
        output = directory / "candidates" / candidate["id"]
        return rpc(candidate["slot"], {"op": "execute", "code": state["code"], "candidate": candidate,
                   "genesis": state["genesis"], "selected": [item["result"] for item in state["selected"]],
                   "output": str(output), "execute_steps": min(state["execute_steps"], 1500 - state["step"])})

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(execute, candidate) for candidate in batch]
            for future in futures:
                future.result()
        import_batch(directory, state)
    except Exception as error:
        state["execution_error"] = str(error)
        save_state(directory, state)
        raise


def brief(state):
    return {key: state[key] for key in ("task", "status", "request_id", "chunk", "step", "stop_chunk", "stop_reason")}


def show_status(directory):
    state = load_state(directory)
    summary = brief(state)
    summary.update(instruction=state["instruction"], candidates_this_round=len(state["candidates"]),
                   review_sheet=state.get("review_sheet"))
    if state["checkpoint"]:
        checkpoint = state["checkpoint"]
        if digest(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("Checkpoint identity changed")
        value = json.loads(Path(checkpoint["path"]).read_text())
        summary["observation"] = {key: value[key] for key in ("frames", "robot", "native_goal", "control_dt")}
    summary["candidates"] = [{"id": item["id"], "attempt": item["attempt"], "instruction": item["instruction"],
                              "review": item["review"], "sheet": item.get("result", {}).get("sheet"),
                              "review_frames": item.get("result", {}).get("review_frames", [])}
                             for item in state["candidates"]]
    print(json.dumps(summary, indent=2))


def export(directory, state):
    destination = directory / "export"
    if destination.exists():
        raise ValueError("Export already exists; inspect it, do not overwrite")
    destination.mkdir()
    chunks, segments = [], []
    for candidate in state["selected"]:
        result = validate_result(candidate, candidate["result"])
        actions = np.load(result["executed_actions"], allow_pickle=False)
        start = sum(len(chunk) for chunk in chunks)
        chunks.append(actions)
        segments.append({"candidate_id": candidate["id"], "action_source": "cosmos", "start": start,
                         "length": len(actions), "predicted_length": 32, "unexecuted_tail_verified": False,
                         "raw_actions": result["raw_actions"], "review": candidate["review"],
                         "instruction": candidate["instruction"], "reason": candidate["selection_reason"]})
    np.save(destination / "actions.npy", np.concatenate(chunks) if chunks else np.empty((0, 8), dtype=np.float32))
    videos = []
    for camera in ("wrist_cam", "egocentric_mirrored_camera"):
        frames_dir = destination / camera
        frames_dir.mkdir()
        frames = [frame for candidate in state["selected"] for frame in candidate["result"]["frames"]
                  if frame["camera"] == camera and frame["step"] > 0]
        for index, frame in enumerate(frames):
            if digest(frame["path"]) != frame["sha256"]:
                raise ValueError("Export frame changed")
            (frames_dir / f"{index:06d}.png").symlink_to(frame["path"])
        if frames:
            path = destination / f"{camera}.mp4"
            fps = 1 / state["selected"][0]["result"]["control_dt"]
            subprocess.run([str(ROOT / "packages/ffmpeg/bin/ffmpeg"), "-v", "error", "-n", "-framerate", str(fps),
                            "-i", str(frames_dir / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
            subprocess.run([str(ROOT / "packages/ffmpeg/bin/ffmpeg"), "-v", "error", "-i", str(path),
                            "-f", "null", "-"], check=True)
            videos.append({"path": str(path), "sha256": digest(path)})
    completed = [item for group in state["rounds"] + [state["candidates"]] for item in group if "result" in item]
    report = {**brief(state), "segments": segments, "videos": videos, "code": state["code"],
              "candidate_count": len(completed), "candidate_steps": sum(item["result"]["executed_steps"] for item in completed),
              "replay_steps": sum(item["result"]["replay_steps"] for item in completed),
              "actions": {"path": str(destination / "actions.npy"), "sha256": digest(destination / "actions.npy")},
              "native_success": any(item["result"]["native_success"] for item in state["selected"]),
              "local_task_success": "not inferred from accept; requires separate visual evidence"}
    write_json(destination / "manifest.json", report)
    (directory / "RESULT.md").write_text(
        f"# Sampling Result\n\nTask: {state['task']}\n\nStop: {state['stop_reason'] or state['status']}\n\n"
        f"Committed chunks: {state['chunk']}; steps: {state['step']}; executed candidates: {len(completed)}.\n\n"
        f"Candidate steps: {report['candidate_steps']}; replay steps: {report['replay_steps']}.\n\n"
        f"Native checker success: {report['native_success']}.\n\n"
        "Accept means useful visual progress, not task success. Unexecuted prediction tails are unverified.\n"
        "This is rollback-based branch search, not irreversible real-robot correction.\n"
        "See export/manifest.json for the selected sequence and visual review evidence.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="New eight-GPU experiment; refuses busy GPUs")
    start.add_argument("--session", type=Path, required=True)
    start.add_argument("--base-port", type=int, default=18840)
    start.add_argument("--master-port", type=int, default=29840)
    start.add_argument("--allow-gpu-sharing", action="store_true", help="Explicitly authorize sharing GPUs with existing jobs; never stops them")
    for command in ("status", "instruct", "run-batch", "review", "select", "recover", "export", "stop-task"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--session", type=Path, required=True)
        subparser.add_argument("--task", choices=TASKS, required=True)
        if command in ("instruct", "run-batch", "review", "select", "stop-task"):
            subparser.add_argument("--decision", type=Path, required=True)
    for command in ("next-task", "stop-services"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    args.session = args.session.resolve()
    if args.command == "start":
        start_experiment(args.session, args.base_port, args.master_port, args.allow_gpu_sharing)
        return
    if args.command in ("next-task", "stop-services"):
        with (args.session / ".services.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.command == "next-task":
                advance_queue(args.session)
            else:
                manifest = load_manifest(args.session)
                for process in manifest["workers"] + manifest["models"]:
                    stop_owned(process)
        return
    directory = args.session / args.task
    if args.command == "status":
        show_status(directory)
        return
    with locked(directory) as state:
        decision = None
        if hasattr(args, "decision"):
            decision, reference = archive_decision(directory, args.decision)
            require(state, decision["request_id"], state["status"])
            state.setdefault("decisions", []).append({"command": args.command, **reference})
        if args.command == "instruct":
            require(state, decision["request_id"], "need_instruction", "ready")
            if state["candidates"]:
                raise ValueError("Use run-batch instructions to revise within a round")
            instruction = validate_instruction(decision["instruction"])
            state.update(instruction=instruction, status="ready")
        elif args.command == "run-batch":
            run_batch(directory, state, decision)
            print(json.dumps(brief(state)))
            return
        elif args.command == "review":
            review_batch(state, decision)
        elif args.command == "select":
            select_candidate(state, decision)
        elif args.command == "recover":
            if state["status"] != "execution_unknown":
                raise ValueError("Only execution_unknown needs recovery; never reruns a worker")
            import_batch(directory, state)
            return
        elif args.command == "stop-task":
            if state["status"] in ("pending_review", "need_selection"):
                raise ValueError("Finish the pending review/selection before stopping")
            if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                raise ValueError("Record the real technical blocker or explicit user stop reason")
            if state["status"] == "execution_unknown":
                raise ValueError("Resolve unknown worker execution before stopping/advancing")
            state.update(status="stopped", stop_reason=decision["reason"])
        elif args.command == "export":
            export(directory, state)
            return
        save_state(directory, state)
        print(json.dumps(brief(state)))


if __name__ == "__main__":
    main()