#!/usr/bin/env bash

set -euo pipefail

SYNC_SCRIPT="${AZURE_SYNC_SCRIPT:-/home/aiscuser/OpenWAM-TTT/extensions/wuji_ttt/sync_azure_blob_weights.sh}"
export AZURE_BLOB_SAS_FILE="${AZURE_BLOB_SAS_FILE:-/home/aiscuser/.blob-sync/temp_sas_url.txt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${STEERING_ROOT:-$SCRIPT_DIR}"
RECIPE="${RECIPE_DIR:-$ROOT/cookbooks/cosmos3/generator/action/finetune}"
ROBODOJO="${ROBODOJO_ROOT:-$ROOT/RoboDojo}"
REMOTE="${ROBODOJO_SAS_REMOTE_ROOT:-jinghe/steeringwam/robodojo_bundle}"
LOG_DIR="${SAS_UPLOAD_LOG_DIR:-$ROOT/sas-upload-logs}"

mkdir -p "$LOG_DIR"

upload() {
    local name=$1
    local source=$2
    local destination=$3
    local marker="$LOG_DIR/$name.complete"
    local log="$LOG_DIR/$name.log"

    if [[ -f "$marker" ]]; then
        echo "[$(date --iso-8601=seconds)] $name already complete"
        return
    fi
    if [[ ! -d "$source" ]]; then
        echo "Missing upload source: $source" >&2
        return 1
    fi

    echo "[$(date --iso-8601=seconds)] starting $name: $source -> $destination" | tee "$log"
    "$SYNC_SCRIPT" upload "$source" "$destination" 2>&1 | tee -a "$log"
    touch "$marker"
    echo "[$(date --iso-8601=seconds)] completed $name" | tee -a "$log"
}

upload evaluation "$ROBODOJO/eval_result" "$REMOTE/evaluation/eval_result"
upload training_run "$RECIPE/outputs/train/cosmos3_action_robodojo/action_sft/action_policy_robodojo_edge" "$REMOTE/training/action_policy_robodojo_edge"
upload base_models "$RECIPE/checkpoints" "$REMOTE/base_models"
upload simulator_assets "$ROBODOJO/.cache/robodojo_assets_repo/Assets" "$REMOTE/simulator/Assets"
upload lerobot_v3_ee "$ROBODOJO/.cache/robodojo_data_huggingface_lerobot_v3.0_ee_repo/data/RoboDojo_ee_lerobot_v30_video" "$REMOTE/datasets/RoboDojo_ee_lerobot_v30_video"
upload hdf5 "$ROBODOJO/.cache/robodojo_data_huggingface_hdf5_repo/data/RoboDojo" "$REMOTE/datasets/RoboDojo_hdf5"

echo "[$(date --iso-8601=seconds)] all RoboDojo assets uploaded" | tee "$LOG_DIR/all.complete"
