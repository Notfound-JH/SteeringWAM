"""Run the preregistered three-checkpoint paired pilot and blind assessment."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
import urllib.request
import uuid

import numpy as np
from PIL import Image

from instruction_sampling import write_json
from paired_instruction_experiment import prepare, export_blind, summarize, sha256


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "RoboLab/.venv/bin/python"
OLD = ROOT / "diagnostics/multi_env_six_20260918T065328Z"
SPEC = {
    "fruits": {"task": "FruitsOnPlate3Task", "pool": 5, "gpu": 0, "order": ["A", "B"],
               "rewrite": "Keep the fingers open and make a small sideways adjustment until the loose orange beside the green limes is centered between the fingertips. Stay above the fruit without descending into it."},
    "tools": {"task": "ToolsPickingAllHammersTask", "pool": 3, "gpu": 1, "order": ["B", "A"],
              "rewrite": "Raise the open hand clear of the blue and red handles, then approach the exposed pale wooden grip from its outer tabletop side. Keep the fingers open and hover above the wood."},
    "organic": {"task": "ClearOrganicObjectsTask", "pool": 7, "gpu": 4, "order": ["A", "B"],
                "rewrite": "Keep the fingers open and slide the hand sideways above the large red pomegranate until its center is between the fingertips. Maintain clearance while centering; do not descend yet."},
}


def environment():
    return os.environ | {"PATH": f"{ROOT}/packages/ffmpeg/bin:{Path.home()}/.local/bin:{os.environ['PATH']}",
                         "LD_LIBRARY_PATH": str(ROOT / "packages/ffmpeg/lib"), "OMP_NUM_THREADS": "4",
                         "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost"}


def freeze(session):
    session.mkdir(parents=True, exist_ok=False)
    seeds = np.random.default_rng(20260918).choice(2**31, size=12, replace=False).tolist()
    for name, spec in SPEC.items():
        directory = session / name
        directory.mkdir()
        historical = OLD / name / f"round_{spec['pool']:03d}"
        baseline = json.loads((historical / "instructions.json").read_text())[0]
        baseline_prompt = json.loads((historical / "candidate_00/request.json").read_text())["prompt"]
        checkpoint = Path(json.loads((historical / "result.json").read_text())["checkpoint_parent"])
        rewritten = {**baseline, "instruction": spec["rewrite"]}
        prefix, marker, suffix = baseline_prompt.partition("Current subgoal: ")
        if not marker or "\nAvoid: " not in suffix:
            raise ValueError("Unexpected historical prompt structure")
        alternate_prompt = prefix + marker + spec["rewrite"] + "\nAvoid: " + suffix.split("\nAvoid: ", 1)[1]
        for condition, instruction, prompt in (("a", baseline, baseline_prompt), ("b", rewritten, alternate_prompt)):
            write_json(directory / f"instructions_{condition}.json", [instruction] * 12)
            (directory / f"prompt_{condition}.txt").write_text(prompt)
        write_json(directory / "seeds.json", seeds)
        prepare(SimpleNamespace(task=spec["task"], checkpoint=checkpoint,
            instructions_a=directory / "instructions_a.json", instructions_b=directory / "instructions_b.json",
            prompt_a=directory / "prompt_a.txt", prompt_b=directory / "prompt_b.txt",
            seeds=directory / "seeds.json", output=directory / "plan"))
        protocol = f"""Evaluate visible contribution to this shared target/subgoal: {baseline['subgoal']}.
Completion criterion: {baseline['completion_condition']}
Constraints: {baseline['avoid']}
Score each candidate relative to its own start: -1 regression, 0 no clear change,
1 useful partial progress, 2 subgoal completed; null if views are insufficient.
unsafe: true for visible constraint violation/contact, false if clearly absent, null if uncertain.
Do not infer grasp/lift or precise clearance from ambiguous images. Use all anonymous review.png
sheets, inspect underlying real start/end frames for ambiguous cases. Ignore task-success assumptions.
Keep scoring criteria identical for every candidate. All candidates are independent trials.
Write reviews.json as an object keyed by ALL24 sample_NNN labels, values containing score, unsafe,
and concise specific visual evidence. Do not read outside this directory or seek condition/prompt/seed.
Do not call task_complete until all24 reviews are written. No shell execution is needed.
"""
        (directory / "protocol.txt").write_text(protocol)
    sources = [ROOT / "sampling_multi_env.py", ROOT / "paired_instruction_experiment.py", Path(__file__),
               ROOT / "packages/cosmos3/cosmos_framework/scripts/action_policy_server_robolab.py"]
    files = list(session.glob("*/plan/*.json")) + sources
    write_json(session / "experiment.json", {"format": "paired-pilot-v1", "tasks": SPEC,
        "created_at": datetime.now(timezone.utc).isoformat(), "sha256": {str(path): sha256(path) for path in files},
        "consistency_policy": "diagnostic differences only; retain identity, seed, shape, finite and action integrity checks",
        "candidate_budget": 72, "diagnostic_extra_requests": 3})


def consistency(directory):
    report = {"diagnostic_only": True, "input_image_mae": []}
    with np.load(directory / "run_A/scene.npz") as first, np.load(directory / "run_B/scene.npz") as second:
        if set(first.files) != set(second.files):
            raise ValueError("Initial scene fields differ")
        report["state_max_abs_difference"] = {}
        for key in first.files:
            if first[key].shape != second[key].shape or not np.isfinite(first[key]).all() or not np.isfinite(second[key]).all():
                raise ValueError("Invalid initial state structure")
            report["state_max_abs_difference"][key] = float(np.max(np.abs(first[key] - second[key])))
    for index in range(12):
        images = [np.asarray(Image.open(directory / f"run_{arm}/candidate_{index:02d}/input.png"), dtype=np.float32) for arm in ("A", "B")]
        report["input_image_mae"].append(float(np.mean(np.abs(images[0] - images[1]))))
    write_json(directory / "consistency.json", report)


def run_task(session, name):
    spec = SPEC[name]
    directory = session / name
    env = environment()
    port = 18960 + spec["gpu"]
    model = None
    try:
        write_json(directory / "status.json", {"stage": "starting_model"})
        model_dir = directory / "model"
        model_dir.mkdir()
        with (model_dir / "server.log").open("wb") as log:
            model = subprocess.Popen(["bash", str(ROOT / "launch_cosmos3_robolab_server.sh"), str(model_dir)],
                env=env | {"MODEL_GPU": str(spec["gpu"]), "PORT": str(port), "MASTER_PORT": str(29960 + spec["gpu"]),
                           "ALLOW_ALL_GPUS": "1", "DECODE_VIDEO": "0"}, stdout=log, stderr=subprocess.STDOUT)
        write_json(directory / "model_process.json", {"pid": model.pid, "port": port})
        deadline = time.monotonic() + 600
        while True:
            if model.poll() is not None:
                raise RuntimeError("Experiment model exited during startup")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Experiment model readiness timeout")
            threading.Event().wait(2)
        for index, condition in enumerate(spec["order"]):
            plan = json.loads((directory / f"plan/arm_{condition}.json").read_text())
            write_json(directory / "status.json", {"stage": "sampling", "condition": condition})
            command = [str(PYTHON), str(ROOT / "sampling_multi_env.py"), "--task", spec["task"],
                "--sim-gpu", str(spec["gpu"]), "--allow-gpu-sharing", "--policy-port", str(port),
                "--instructions", plan["instructions"]["path"], "--checkpoint", plan["checkpoint"]["path"],
                "--experiment-arm", str(directory / f"plan/arm_{condition}.json"), "--execute-steps", "16",
                "--output-dir", str(directory / f"run_{condition}")]
            if index == 0:
                command.append("--repeatability-probe")
            with (directory / f"run_{condition}.log").open("wb") as log:
                subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        model.terminate()
        model.wait(timeout=30)
        model = None
        consistency(directory)
        export_blind(SimpleNamespace(arms=[directory / "run_A", directory / "run_B"], output=directory / "blind",
            mapping=directory / "private_mapping.json", protocol=directory / "protocol.txt"))
        write_json(directory / "status.json", {"stage": "blind_review"})
        reviewer = str(uuid.uuid4())
        command = [str(Path.home() / ".local/bin/copilot"), "-C", str(directory / "blind"),
            "--session-id", reviewer, "--model", "gpt-6-astra", "--reasoning-effort", "low", "--no-auto-update",
            "--no-custom-instructions", "--disable-builtin-mcps", "--no-remote-export", "--no-ask-user",
            "--autopilot", "--max-autopilot-continues", "20", "--no-color",
            "--available-tools", "view", "apply_patch", "task_complete", "--allow-tool", "view", "--allow-tool", "write",
            "--allow-tool", "task_complete", "-p", "Read review_protocol.txt. Independently inspect ALL24 anonymous sample_000 through sample_023/review.png sheets using view, plus individual frames as needed. Write the requested complete reviews.json before finishing. No outside directory access. No inference or task execution. You are solely a blinded image reviewer."]
        write_json(directory / "reviewer.json", {"session_id": reviewer, "model": "gpt-6-astra", "effort": "low"})
        with (directory / "reviewer.log").open("wb") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        summarize(SimpleNamespace(reviews=directory / "blind/reviews.json", mapping=directory / "private_mapping.json",
                                  output=directory / "summary.json"))
        write_json(directory / "status.json", {"stage": "complete"})
    except Exception as error:
        write_json(directory / "status.json", {"stage": "failed", "error": str(error)})
        raise
    finally:
        if model is not None and model.poll() is None:
            model.terminate()
            try:
                model.wait(timeout=30)
            except subprocess.TimeoutExpired:
                model.kill()
                model.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--task", choices=list(SPEC))
    args = parser.parse_args()
    if args.task:
        run_task(args.session.resolve(), args.task)
        return
    freeze(args.session.resolve())
    children = []
    for name in SPEC:
        with (args.session / name / "worker.log").open("wb") as log:
            children.append(subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--session", str(args.session.resolve()),
                                              "--task", name], stdout=log, stderr=subprocess.STDOUT))
    codes = [child.wait() for child in children]
    write_json(args.session / "completion.json", {"exit_codes": codes, "complete": all(code == 0 for code in codes)})
    if any(codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()