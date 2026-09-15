#!/usr/bin/env bash

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${ROBODOJO_RUN_DIR:-$RECIPE_DIR/outputs/train/cosmos3_action_robodojo/action_sft/action_policy_robodojo_edge}"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
REMOTE_ROOT="${ROBODOJO_SAS_REMOTE_ROOT:-jinghe/steeringwam/cosmos3_action_robodojo/action_policy_robodojo_edge}"
SYNC_SCRIPT="${AZURE_SYNC_SCRIPT:-/home/aiscuser/OpenWAM-TTT/extensions/wuji_ttt/sync_azure_blob_weights.sh}"
SAS_FILE="${AZURE_BLOB_SAS_FILE:-/home/aiscuser/.blob-sync/temp_sas_url.txt}"
LOCK_FILE="$RUN_DIR/.sas-sync.lock"

mkdir -p "$RUN_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "$(date --iso-8601=seconds) sync already running; skipping"
    exit 0
fi

export AZURE_BLOB_SAS_FILE="$SAS_FILE"

if [[ ! -d "$CHECKPOINT_DIR" ]] || ! find "$CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "$(date --iso-8601=seconds) no checkpoints yet; skipping weights"
    exit 0
fi

"$SYNC_SCRIPT" upload "$CHECKPOINT_DIR" "$REMOTE_ROOT/checkpoints"
echo "$(date --iso-8601=seconds) checkpoint sync complete"