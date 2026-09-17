#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
PYTHON="$WORKSPACE/packages/cosmos3/.venv/bin/python"
CONTROL_ROOT="$WORKSPACE/outputs/edge_droid_control_stability_200_seed42_ema_fix"
SCHEDULED_ROOT="$WORKSPACE/outputs/edge_droid_scheduled_stability_200_seed42"
REPORT_ROOT="$WORKSPACE/loop_reasoner/validation"
mkdir -p "$CONTROL_ROOT" "$SCHEDULED_ROOT"

PLANNING_MODE=control IMAGINAIRE_OUTPUT_ROOT="$CONTROL_ROOT" \
    bash "$WORKSPACE/loop_reasoner/run_edge_stability.sh" 2>&1 | tee -a "$CONTROL_ROOT/train.log"
PLANNING_MODE=scheduled IMAGINAIRE_OUTPUT_ROOT="$SCHEDULED_ROOT" \
    bash "$WORKSPACE/loop_reasoner/run_edge_stability.sh" 2>&1 | tee -a "$SCHEDULED_ROOT/train.log"

for MODE in control scheduled feedback; do
    EXTRA=()
    case "$MODE" in
        control) ROOT="$CONTROL_ROOT"; LABEL=B0-CONT ;;
        scheduled) ROOT="$SCHEDULED_ROOT"; LABEL=B4 ;;
        feedback)
            ROOT="$WORKSPACE/outputs/edge_droid_feedback_stability_200_seed42"
            LABEL=B5
            EXTRA=(--mechanism)
            ;;
    esac
    RUN_DIR="$ROOT/cosmos3_action/edge_droid_planning/action_policy_droid_edge_${MODE}_stability_200_seed42"
    "$PYTHON" "$WORKSPACE/packages/cosmos3/tools/diagnose_edge_planning.py" \
        --run-dir "$RUN_DIR" --checkpoint "$RUN_DIR/checkpoints/iter_000000200" \
        --data-root "$WORKSPACE/datasets/droid_plus_lerobot_640x360_20260412" \
        --episode-limit 512 --episodes 512 --windows 2 --label "$LABEL" \
        --report "$REPORT_ROOT/edge_${MODE}_200_comparison.json" "${EXTRA[@]}" \
        2>&1 | tee -a "$ROOT/comparison.log"
done

"$PYTHON" "$WORKSPACE/packages/cosmos3/tools/diagnose_edge_planning.py" \
    --compare "$REPORT_ROOT/edge_control_200_comparison.json" \
        "$REPORT_ROOT/edge_scheduled_200_comparison.json" "$REPORT_ROOT/edge_feedback_200_comparison.json" \
    --report "$REPORT_ROOT/edge_attribution_200.json"