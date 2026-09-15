#!/usr/bin/env bash

set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
STEERING_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"
FRAMEWORK_ROOT="${COSMOS_FRAMEWORK_ROOT:-$STEERING_ROOT/packages/cosmos3}"
RECIPE_DIR="$STEERING_ROOT/cookbooks/cosmos3/generator/action/finetune"
RUN_DIR="${COSMOS3_ROBODOJO_RUN_DIR:-$RECIPE_DIR/outputs/train/cosmos3_action_robodojo/action_sft/action_policy_robodojo_edge}"
CHECKPOINT_PATH="${COSMOS3_ROBODOJO_CHECKPOINT:-}"
CONFIG_FILE="${COSMOS3_ROBODOJO_CONFIG:-$RUN_DIR/config.yaml}"
FFMPEG_ROOT="${FFMPEG_ROOT:-$STEERING_ROOT/packages/ffmpeg}"
EXECUTE_STEPS="${COSMOS3_ROBODOJO_EXECUTE_STEPS:-4}"
NUM_STEPS="${COSMOS3_ROBODOJO_NUM_STEPS:-8}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
    if [[ "$ckpt_name" == /* ]]; then
        CHECKPOINT_PATH="$ckpt_name"
    else
        CHECKPOINT_PATH="$RUN_DIR/checkpoints/$ckpt_name"
    fi
fi
if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "[Cosmos3_RoboDojo] checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[Cosmos3_RoboDojo] config not found: $CONFIG_FILE" >&2
    exit 1
fi

if [[ -x "$policy_env/bin/python" ]]; then
    PYTHON="$policy_env/bin/python"
elif [[ -x "$FRAMEWORK_ROOT/.venv/bin/python" ]]; then
    PYTHON="$FRAMEWORK_ROOT/.venv/bin/python"
else
    echo "[Cosmos3_RoboDojo] policy environment not found: $policy_env" >&2
    exit 1
fi

PYTHON_SITE_PACKAGES="$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDA_WHEEL_LIB_DIR="$PYTHON_SITE_PACKAGES/nvidia/cu13/lib"
TORCH_LIB_DIR="$PYTHON_SITE_PACKAGES/torch/lib"
export LD_LIBRARY_PATH="$CUDA_WHEEL_LIB_DIR:$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -d "$FFMPEG_ROOT/lib" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$FFMPEG_ROOT/lib"
fi
export CUDA_VISIBLE_DEVICES="$policy_gpu_id"
export PYTHONPATH="$XPL_ROOT:$BENCH_ROOT:$FRAMEWORK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec "$PYTHON" -u "$XPL_ROOT/setup_policy_server.py" \
    --config_path "$SCRIPT_DIR/deploy.yml" \
    --overrides \
        host="$policy_server_host" \
        port="$policy_server_port" \
        bench_name="$bench_name" \
        task_name="$task_name" \
        ckpt_name="$ckpt_name" \
        env_cfg_type="$env_cfg_type" \
        action_type="$action_type" \
        seed="$seed" \
        checkpoint_path="$CHECKPOINT_PATH" \
        config_file="$CONFIG_FILE" \
        execute_steps="$EXECUTE_STEPS" \
        num_steps="$NUM_STEPS"