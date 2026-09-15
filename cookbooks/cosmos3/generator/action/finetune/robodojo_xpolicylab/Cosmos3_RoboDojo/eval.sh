#!/usr/bin/env bash

set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_env=$9
eval_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
port="$(bash "$XPL_ROOT/utils/get_free_port.sh")"

cleanup() {
	if [[ -n "${SERVER_PID:-}" ]]; then
		kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
	fi
}
trap cleanup EXIT

setsid bash "$SCRIPT_DIR/setup_eval_policy_server.sh" \
	"$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" \
	"$seed" "$policy_gpu_id" "$policy_env" "$port" localhost &
SERVER_PID=$!

bash "$XPL_ROOT/utils/wait_for_policy_server.sh" localhost "$port" "$SERVER_PID" "Cosmos3 policy server" 1200
bash "$SCRIPT_DIR/setup_eval_env_client.sh" \
	"$bench_name" "$task_name" "$ckpt_name" "$env_cfg_type" "$action_type" \
	"$seed" "$env_gpu_id" "$eval_env" "ckpt_name=$ckpt_name,action_type=$action_type" \
	"$port" localhost