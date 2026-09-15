#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/robodojo_xpolicylab/Cosmos3_RoboDojo"
STEERING_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ROBODOJO_ROOT="${1:-${ROBODOJO_ROOT:-$STEERING_ROOT/RoboDojo}}"
TARGET_DIR="$ROBODOJO_ROOT/XPolicyLab/policy/Cosmos3_RoboDojo"

if [[ ! -d "$ROBODOJO_ROOT/XPolicyLab/policy" ]]; then
    echo "RoboDojo XPolicyLab checkout not found: $ROBODOJO_ROOT" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"
cp -a "$SOURCE_DIR/." "$TARGET_DIR/"
echo "Installed Cosmos3_RoboDojo adapter to $TARGET_DIR"