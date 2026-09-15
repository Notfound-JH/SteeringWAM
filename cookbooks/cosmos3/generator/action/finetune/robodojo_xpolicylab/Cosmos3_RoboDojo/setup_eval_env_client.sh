#!/usr/bin/env bash

set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
STEERING_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"
UTILS_DIR="$XPL_ROOT/utils"
FFMPEG_ROOT="${FFMPEG_ROOT:-$STEERING_ROOT/packages/ffmpeg}"

source "$UTILS_DIR/resolve_eval_env_type.sh"
eval_env_mode="$(resolve_eval_env_type)"
if [[ "$eval_env_mode" != "sim" ]]; then
    echo "[Cosmos3_RoboDojo] only simulator evaluation is supported" >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$eval_env"
export CUDA_VISIBLE_DEVICES="$env_gpu_id"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$XPL_ROOT:$BENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$FFMPEG_ROOT/bin/ffmpeg" ]]; then
    export PATH="$FFMPEG_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec bash "$BENCH_ROOT/scripts/eval_policy.sh" \
    --bench_name "$bench_name" \
    --task_name "$task_name" \
    --env_cfg_type "$env_cfg_type" \
    --policy_name Cosmos3_RoboDojo \
    --host "$policy_server_ip" \
    --port "$policy_server_port" \
    --eval_batch false \
    --root_dir "$BENCH_ROOT" \
    --device_id "$env_gpu_id" \
    --additional_info "$additional_info" \
    --seed "$seed"