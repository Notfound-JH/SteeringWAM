"""Launch six isolated Copilot controllers for twelve-environment sampling."""

import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import uuid

from instruction_sampling import TASKS, write_json


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "RoboLab/.venv/bin/python"


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session = ROOT / "diagnostics" / f"multi_env_six_{stamp}"
    session.mkdir()
    tmux = f"multi-env-six-{stamp}"
    environment = os.environ.copy()
    libraries = [str(ROOT / "packages/ffmpeg/lib")]
    for package in ("libglu-*/lib", "xorg-libxt-*/lib"):
        libraries.extend(str(path) for path in (Path.home() / ".conda/pkgs").glob(package))
    environment.update(PATH=f"{ROOT}/packages/ffmpeg/bin:{Path.home()}/.local/bin:{environment['PATH']}",
                       LD_LIBRARY_PATH=":".join(libraries),
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                       NO_PROXY="127.0.0.1,localhost", COPILOT_TASK_WAIT_TIMEOUT_SECONDS="7200")
    records = []
    for gpu, (task, task_class) in enumerate(TASKS.items()):
        directory = session / task
        directory.mkdir()
        controller_id = str(uuid.uuid4())
        port = 18860 + gpu
        limit = 6 if task == "fruits" else 8 if task == "tools" else 10
        prompt = f"""You are the sole controller for {task} ({task_class}). User explicitly authorized six concurrent
Copilot controllers, one task and twelve Isaac environments on each physical GPU. Your GPU is {gpu};
your model endpoint is ws://127.0.0.1:{port}. NEVER touch another GPU, task, training job or old session.
Use exactly gpt-6-astra with low reasoning. No fallback models or delegated agents.
ROOT={ROOT}
PYTHON={PYTHON}
OUTPUT={directory}
Model is started externally and has logs in OUTPUT/model/server.log. Do not launch another model.
First run {PYTHON} {ROOT}/sampling_multi_env.py --task {task_class} --sim-gpu {gpu} --allow-gpu-sharing --policy-port {port} --instructions {ROOT}/multi_env_instructions.example.json --output-dir {directory}/initialization --initialization-only
This instruction example is ONLY for initialization: no inference is performed. View actual initial wrist
and viewport PNGs from multiple environments, verify nonblank cameras and bent default robot pose,
and inspect initial.json for all twelve joint configurations. Never claim runtime success from mock tests.
If initialization fails, write BLOCKED.md with exact log evidence and stop; do not edit shared source.
Write ACK.md now and continually update PROGRESS.md with truthful stage and any blockers.
Then inspect real initial images to choose a small task-relevant subgoal and write instructions_000.json:
exactly twelve distinct short English instruction objects (subgoal,instruction,completion_condition,avoid).
All twelve share one subgoal. For tools use outer pale wooden hammer handle approach, raised open fingers,
avoid blue blocker/bin walls; choose a visible accessible target instead if needed. Do not blindly target an orange in other tasks.
Wait for model server ready using logs or bounded socket checks; do not rerun a failed/unknown inference automatically.
Run sampling_multi_env.py with the same flags, replacing instructions/output with the current round
(round_000, round_001...), without --initialization-only. Use --execute-steps 16 and keep videos enabled.
After each completed pool, use view on ALL twelve real review.png sheets, inspect real frames as needed.
Write review_NNN.json containing each candidate id, accepted bool, completion bool, specific visual evidence,
and chosen candidate id plus reason. Acceptance means VISIBLE CONTRIBUTION TO CURRENT SUBGOAL;
partial approach/alignment can count without grasp or lift. No fabricated reviews or hidden state scoring.
Choose the best actually executed, nonterminated endpoint and pass its checkpoint.json as --checkpoint
for the next pool. Save selected_NNN.json with checkpoint path/hash and reasons before continuing.
If all twelve unusable, one additional distinct instruction pool at the SAME checkpoint is permitted;
if that also has no useful candidate, stop task. No unlimited retry or repeat unknown actions.
Maximum {limit} selected committed chunks; stop sooner on success or blocked state. Never continue terminated checkpoints.
Only Cosmos model actions; never edit actions or move objects to help success. Checkpoint restoration by
the runner is allowed. Source files and other sessions are READ ONLY. Do not edit runtime/hash/runner code.
Write all logs, images, decisions and exports under OUTPUT only. SAS sidecar backs up the shared parent
session including your output. Do not read SAS credentials or start your own uploader. Native videos are
produced per candidate; do not claim old dashboard supports this new format.
On completion write FINAL.md with selected checkpoints and honest results. Do not stop shared backup.
"""
        (directory / "HANDOFF.md").write_text(prompt)
        model_dir = directory / "model"
        model_dir.mkdir()
        model_env = environment | {"MODEL_GPU": str(gpu), "PORT": str(port),
                                   "MASTER_PORT": str(29860 + gpu), "SEED": str(3407 + gpu),
                                   "ALLOW_ALL_GPUS": "1", "DECODE_VIDEO": "0"}
        with (model_dir / "server.log").open("ab") as log:
            model = subprocess.Popen(["bash", str(ROOT / "launch_cosmos3_robolab_server.sh"), str(model_dir)],
                                     cwd=ROOT, env=model_env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        command = [str(Path.home() / ".local/bin/copilot"), "-C", str(directory),
                   "--session-id", controller_id, "--model", "gpt-6-astra", "--reasoning-effort", "low",
                   "--no-auto-update", "--no-custom-instructions", "--disable-builtin-mcps",
                   "--no-remote-export", "--no-ask-user", "--autopilot", "--max-autopilot-continues", "100",
                   "--log-dir", str(directory / "copilot_logs"), "--log-level", "info", "--no-color",
                   "--add-dir", str(ROOT), "--add-dir", "/home/aiscuser/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu",
                   "--add-dir", "/opt/conda", "--available-tools", "view", "bash", "read_bash", "list_bash", "stop_bash", "apply_patch", "task_complete",
                   "--allow-tool", "task_complete",
                   "--allow-tool", "view", "--allow-tool", "write",
                   "--deny-tool", f"write({ROOT}/sampling_multi_env.py)",
                   "--deny-tool", f"write({ROOT}/instruction_sampling.py)",
                   "--deny-tool", f"write({ROOT}/instruction_sampling_worker.py)",
                   "--deny-tool", f"write({ROOT}/launch_cosmos3_robolab_server.sh)",
                   "--allow-tool", "read_bash", "--allow-tool", "list_bash", "--allow-tool", "stop_bash",
                   "--allow-tool", f"shell({PYTHON}:*)", "--allow-tool", "shell(cd:*)",
                   "--allow-tool", "shell(ls:*)", "--allow-tool", "shell(cat:*)", "--allow-tool", "shell(tail:*)",
                   "--allow-url", "http://127.0.0.1", "--allow-url", "http://localhost",
                   "-p", f"Read {directory}/HANDOFF.md and execute only your assigned task end to end. Write ACK.md first."]
        script = directory / "controller.sh"
        exports = "\n".join(f"export {name}={shlex.quote(environment[name])}" for name in
                            ("PATH", "LD_LIBRARY_PATH", "PYTHONUNBUFFERED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NO_PROXY", "COPILOT_TASK_WAIT_TIMEOUT_SECONDS"))
        script.write_text("#!/bin/bash\nset -uo pipefail\numask 077\n" + exports + "\nunset COPILOT_ALLOW_ALL\n" +
                          shlex.join(command) + " 2>&1 | tee -a " + shlex.quote(str(directory / "console.log")) +
                          "\nexit ${PIPESTATUS[0]}\n")
        if gpu == 0:
            subprocess.run(["tmux", "new-session", "-d", "-s", tmux, "-n", task, "bash", str(script)], check=True)
            subprocess.run(["tmux", "set-option", "-t", tmux, "remain-on-exit", "on"], check=True)
        else:
            subprocess.run(["tmux", "new-window", "-t", tmux, "-n", task, "bash", str(script)], check=True)
        records.append({"task": task, "task_class": task_class, "gpu": gpu, "port": port,
                        "model_pid": model.pid, "controller_uuid": controller_id, "limit": limit})
        write_json(session / "experiment.json", {"format": "multi-env-six-v1", "tmux": tmux, "tasks": records})
    sync = shlex.join([str(PYTHON), "-u", str(ROOT / "sync_instruction_results.py"), "--session", str(session)])
    subprocess.run(["tmux", "new-window", "-t", tmux, "-n", "sas", sync + " > " + shlex.quote(str(session / "sas.log")) + " 2>&1"], check=True)
    print(json.dumps({"session": str(session), "tmux": tmux, "tasks": records}, indent=2), flush=True)


if __name__ == "__main__":
    main()