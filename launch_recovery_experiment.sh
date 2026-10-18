#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN="${1:?Usage: bash launch_recovery_experiment.sh NEW_RUN_DIR [TASK]}"
TASK="${2:-RubiksCubeTask}"
AGENT_CONTROL="${AGENT_CONTROL:-0}"
VERIFY_BLUE_PLACEMENT="${VERIFY_BLUE_PLACEMENT:-0}"
RECOVERY_MAX_ATTEMPTS="${RECOVERY_MAX_ATTEMPTS:-3}"
RECOVERY_MAX_STEPS="${RECOVERY_MAX_STEPS:-}"
CONTINUE_WAM_AFTER_BUDGET="${CONTINUE_WAM_AFTER_BUDGET:-0}"
TASK_PROGRESS_RECOVERY="${TASK_PROGRESS_RECOVERY:-0}"
[[ "$AGENT_CONTROL" == 0 || "$AGENT_CONTROL" == 1 ]] || { echo "AGENT_CONTROL must be 0 or 1." >&2; exit 2; }
[[ "$VERIFY_BLUE_PLACEMENT" == 0 || "$VERIFY_BLUE_PLACEMENT" == 1 ]] || { echo "VERIFY_BLUE_PLACEMENT must be 0 or 1." >&2; exit 2; }
[[ "$RECOVERY_MAX_ATTEMPTS" == unlimited || "$RECOVERY_MAX_ATTEMPTS" =~ ^[1-5]$ ]] || { echo "RECOVERY_MAX_ATTEMPTS must be in 1..5 or unlimited." >&2; exit 2; }
[[ -z "$RECOVERY_MAX_STEPS" || "$RECOVERY_MAX_STEPS" == unlimited || "$RECOVERY_MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "RECOVERY_MAX_STEPS must be positive or unlimited." >&2; exit 2; }
[[ "$CONTINUE_WAM_AFTER_BUDGET" == 0 || "$CONTINUE_WAM_AFTER_BUDGET" == 1 ]] || { echo "CONTINUE_WAM_AFTER_BUDGET must be 0 or 1." >&2; exit 2; }
[[ "$TASK_PROGRESS_RECOVERY" == 0 || "$TASK_PROGRESS_RECOVERY" == 1 ]] || { echo "TASK_PROGRESS_RECOVERY must be 0 or 1." >&2; exit 2; }
if [[ "$TASK_PROGRESS_RECOVERY" == 1 && "$AGENT_CONTROL" == 1 ]]; then
    echo "Task-progress recovery requires WAM, not Agent-only mode." >&2
    exit 2
fi
if [[ "$AGENT_CONTROL" == 1 && "$CONTINUE_WAM_AFTER_BUDGET" == 1 ]]; then
    echo "Budget fallback requires WAM, not Agent-only mode." >&2
    exit 2
fi
if [[ "$VERIFY_BLUE_PLACEMENT" == 1 && ( "$AGENT_CONTROL" != 1 || "$TASK" != ToolsPickingHammerTask ) ]]; then
    echo "VERIFY_BLUE_PLACEMENT requires Agent-only ToolsPickingHammerTask." >&2
    exit 2
fi
[[ ! -e "$RUN" ]] || { echo "Use a new episode directory." >&2; exit 2; }
mkdir -p "$RUN/services/model"
RUN="$(cd "$RUN" && pwd)"
PY="$ROOT/RoboLab/.venv/bin/python"
PORT="${PORT:-18745}"
SIM_GPU="${SIM_GPU:-4}"
SOCKET="/tmp/wam-recovery-$$.sock"
MODEL_PID=""
SYNC_PID=""
cleanup() {
    [[ -z "$SYNC_PID" ]] || kill "$SYNC_PID" 2>/dev/null || true
    [[ -z "$MODEL_PID" ]] || kill "$MODEL_PID" 2>/dev/null || true
}
trap cleanup EXIT
if [[ "$AGENT_CONTROL" == 1 ]]; then
    echo "Agent-only control baseline: no WAM server will be started."
elif [[ "${REUSE_POLICY:-0}" == 1 ]]; then
    curl -fsS --max-time 5 "http://127.0.0.1:$PORT/healthz" >/dev/null
else
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
        echo "Policy port is already in use; set REUSE_POLICY=1 explicitly." >&2
        exit 2
    fi
    PORT="$PORT" bash launch_cosmos3_robolab_server.sh "$RUN/services/model" >"$RUN/services/model/server.log" 2>&1 &
    MODEL_PID=$!
    ready=0
    for _ in {1..180}; do
        kill -0 "$MODEL_PID" || { echo "Policy startup failed; inspect server.log." >&2; exit 1; }
        if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done
    [[ "$ready" == 1 ]] || { echo "Policy startup timed out." >&2; exit 1; }
fi
"$PY" - "$RUN" "$TASK" "$SOCKET" "${RECOVERY_GENESIS:-}" "${RECOVERY_GENESIS_SHA256:-}" "$AGENT_CONTROL" "$RECOVERY_MAX_ATTEMPTS" "$RECOVERY_MAX_STEPS" "$CONTINUE_WAM_AFTER_BUDGET" "$TASK_PROGRESS_RECOVERY" <<'PY'
import json
from pathlib import Path
import sys
from recovery_protocol import parse_recovery_attempt_limit
run, task, socket, genesis, genesis_sha256, agent_control, attempts, max_steps, fallback, progress = sys.argv[1:]
(Path(run) / "experiment.json").write_text(json.dumps({
    "format": "wam-agent-recovery-v1", "task": task, "socket": socket,
    "seed": 3407, "model": None if agent_control == "1" else "Cosmos3-Edge-Policy-DROID",
    "experiment_mode": "agent_control" if agent_control == "1" else "wam_recovery",
    "agent": "external visual Coding Agent via recovery_client",
    "mock_execution": False, "reset_during_recovery": False,
    "initial_genesis": genesis or None, "initial_genesis_sha256": genesis_sha256 or None,
    "recovery_max_attempts": parse_recovery_attempt_limit(attempts),
    "requested_max_episode_steps": int(max_steps) if max_steps and max_steps != "unlimited" else None,
    "unlimited_episode_steps": max_steps == "unlimited",
    "budget_exhaustion_policy": "continue_wam" if fallback == "1" else "stop",
    "protocol_version": "wam-recovery-v3" if progress == "1" else "wam-recovery-v2",
    "evaluation_policy": "task_progress" if progress == "1" else "fixed_wam_window",
}, indent=2) + "\n")
PY
if [[ "${SYNC_BLOB:-1}" == 1 ]]; then
    "$PY" sync_instruction_results.py --session "$RUN" \
        --remote-prefix jinghe/steeringwam/recovery --interval 60 >"$RUN/sync.log" 2>&1 &
    SYNC_PID=$!
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$SIM_GPU" --query-gpu=uuid --format=csv,noheader)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH="$ROOT/packages/ffmpeg/lib"
export PATH="$ROOT/packages/ffmpeg/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
GENESIS_ARGS=(--recovery-max-attempts "$RECOVERY_MAX_ATTEMPTS")
if [[ "$TASK_PROGRESS_RECOVERY" == 1 ]]; then
    GENESIS_ARGS+=(--task-progress-recovery)
fi
if [[ "$RECOVERY_MAX_STEPS" == unlimited ]]; then
    GENESIS_ARGS+=(--recovery-unlimited-steps)
elif [[ -n "$RECOVERY_MAX_STEPS" ]]; then
    GENESIS_ARGS+=(--recovery-max-steps "$RECOVERY_MAX_STEPS")
fi
if [[ "$CONTINUE_WAM_AFTER_BUDGET" == 1 ]]; then
    GENESIS_ARGS+=(--continue-wam-after-budget)
fi
if [[ "$AGENT_CONTROL" == 1 ]]; then
    GENESIS_ARGS+=(--agent-control)
fi
if [[ "$VERIFY_BLUE_PLACEMENT" == 1 ]]; then
    GENESIS_ARGS+=(--verify-blue-placement)
fi
if [[ -n "${RECOVERY_GENESIS:-}" ]]; then
    GENESIS_ARGS+=(--recovery-genesis "$RECOVERY_GENESIS" --recovery-genesis-sha256 "${RECOVERY_GENESIS_SHA256:?Provide the historical genesis SHA-256}")
elif [[ -n "${RECOVERY_GENESIS_SHA256:-}" ]]; then
    echo "RECOVERY_GENESIS_SHA256 requires RECOVERY_GENESIS." >&2
    exit 2
fi
printf 'Episode: %s\nAgent socket: %s\nSimulator log: %s/launch.log\n' "$RUN" "$SOCKET" "$RUN"
"$PY" probe_isaac_camera.py --task "$TASK" --sim-gpu "$SIM_GPU" --seed 3407 \
    --policy-port "$PORT" --recovery-worker-socket "$SOCKET" --output-dir "$RUN" "${GENESIS_ARGS[@]}" >"$RUN/launch.log" 2>&1
"$PY" verify_recovery_episode.py "$RUN/recovery"
if [[ -n "$SYNC_PID" ]]; then
    kill "$SYNC_PID"
    wait "$SYNC_PID"
    SYNC_PID=""
    "$PY" sync_instruction_results.py --session "$RUN" --remote-prefix jinghe/steeringwam/recovery --once
fi
