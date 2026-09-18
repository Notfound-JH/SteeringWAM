#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="${STEERING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ROBODOJO="${ROBODOJO_ROOT:-$ROOT/RoboDojo}"
RECIPE="${RECIPE_DIR:-$ROOT/cookbooks/cosmos3/generator/action/finetune}"
AZCOPY="${AZCOPY_BIN:-$HOME/.local/bin/azcopy}"
SAS_FILE="${AZURE_BLOB_SAS_FILE:-$HOME/.blob-sync/temp_sas_url.txt}"
REMOTE="${ROBODOJO_SAS_REMOTE_ROOT:-jinghe/steeringwam/robodojo_bundle}"
LOG_DIR="${SAS_RESTORE_LOG_DIR:-$HOME/.blob-sync/restore-logs}"
TRAIN_REMOTE="${ROBODOJO_TRAIN_REMOTE:-$REMOTE/training/action_policy_robodojo_edge}"
TRAIN_LOCAL="$RECIPE/outputs/train/cosmos3_action_robodojo/action_sft/action_policy_robodojo_edge"
STAGE="${1:-all}"

case "$STAGE" in
    weights|assets|data|all) ;;
    *) printf 'Usage: %s [weights|assets|data|all]\n' "$0" >&2; exit 2 ;;
esac
[[ -x "$AZCOPY" && -r "$SAS_FILE" ]] || { echo 'AzCopy or SAS file missing.' >&2; exit 1; }
mkdir -p "$LOG_DIR"
exec 9>"$LOG_DIR/restore.lock"
flock -n 9 || { echo 'Another restore is already running.' >&2; exit 1; }
export AZCOPY_LOG_LOCATION="$LOG_DIR/azcopy"
export AZCOPY_JOB_PLAN_LOCATION="$LOG_DIR/plans"
mkdir -p "$AZCOPY_LOG_LOCATION" "$AZCOPY_JOB_PLAN_LOCATION"
SAS_URL="$(tr -d '\r\n' < "$SAS_FILE")"
CONTAINER="${SAS_URL%%\?*}"
QUERY="${SAS_URL#*\?}"

download() {
    local name=$1 remote=$2 destination=$3
    shift 3
    mkdir -p "$destination"
    printf 'Downloading %s -> %s\n' "$name" "$destination"
    if ! "$AZCOPY" copy "$CONTAINER/$remote?$QUERY" "$destination" \
        --overwrite=ifSourceNewer --check-length=true --output-type=json "$@" \
        >"$LOG_DIR/$name.jsonl" 2>"$LOG_DIR/$name.stderr"; then
        printf 'Download failed: %s. Private logs: %s\n' "$name" "$LOG_DIR" >&2
        return 1
    fi
    printf 'Completed %s\n' "$name"
}

if [[ "$STAGE" == weights || "$STAGE" == all ]]; then
    download base_models "$REMOTE/base_models/*" "$RECIPE/checkpoints" --recursive=true
    download training_metadata "$TRAIN_REMOTE/*" "$TRAIN_LOCAL" --recursive=false
    download latest_pointer "$TRAIN_REMOTE/checkpoints/latest_checkpoint.txt" "$TRAIN_LOCAL/checkpoints"
    LATEST="$(tr -d '\r\n' < "$TRAIN_LOCAL/checkpoints/latest_checkpoint.txt")"
    [[ "$LATEST" =~ ^iter_[0-9]+$ ]] || { echo 'Invalid latest checkpoint name.' >&2; exit 1; }
    download latest_checkpoint "$TRAIN_REMOTE/checkpoints/$LATEST/*" "$TRAIN_LOCAL/checkpoints/$LATEST" --recursive=true
    download evaluation "$REMOTE/evaluation/eval_result/*" "$ROBODOJO/eval_result" --recursive=true
    printf 'Latest restored checkpoint: %s\n' "$LATEST"
fi

if [[ "$STAGE" == assets || "$STAGE" == all ]]; then
    download simulator_assets "$REMOTE/simulator/Assets/*" "$ROBODOJO/.cache/robodojo_assets_repo/Assets" --recursive=true
    ln -sfnT "$ROBODOJO/.cache/robodojo_assets_repo/Assets" "$ROBODOJO/Assets"
fi

if [[ "$STAGE" == data || "$STAGE" == all ]]; then
    LEROBOT="$ROBODOJO/.cache/robodojo_data_huggingface_lerobot_v3.0_ee_repo/data/RoboDojo_ee_lerobot_v30_video"
    HDF5="$ROBODOJO/.cache/robodojo_data_huggingface_hdf5_repo/data/RoboDojo"
    download lerobot_data "$REMOTE/datasets/RoboDojo_ee_lerobot_v30_video/*" "$LEROBOT" --recursive=true
    download hdf5_data "$REMOTE/datasets/RoboDojo_hdf5/*" "$HDF5" --recursive=true
    mkdir -p "$ROBODOJO/data"
    ln -sfnT "$LEROBOT" "$ROBODOJO/data/RoboDojo_ee_lerobot_v30_video"
    ln -sfnT "$HDF5" "$ROBODOJO/data/RoboDojo"
fi