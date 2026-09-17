#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLANNING_MODE="${PLANNING_MODE:-feedback}"
case "$PLANNING_MODE" in
    feedback|scheduled|static) LEARNING_RATE=1e-4 ;;
    control) LEARNING_RATE=2e-5 ;;
    *) printf 'Unsupported PLANNING_MODE: %s\n' "$PLANNING_MODE" >&2; exit 2 ;;
esac
JOB_NAME="action_policy_droid_edge_${PLANNING_MODE}_stability_200_seed42"
OUTPUT_SUFFIX=""
if [[ "$PLANNING_MODE" == control ]]; then
    OUTPUT_SUFFIX="_ema_fix"
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$WORKSPACE/outputs/edge_droid_smoke/cosmos3_action/edge_droid_warmstart/action_policy_droid_edge/checkpoints/iter_000000002}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$WORKSPACE/outputs/edge_droid_${PLANNING_MODE}_stability_200_seed42${OUTPUT_SUFFIX}}"
export RECIPE=examples/toml/sft_config/action_policy_droid_edge_recurrent.toml

exec bash "$WORKSPACE/loop_reasoner/run_edge_smoke.sh" \
    experiment="action_policy_droid_edge_${PLANNING_MODE}" job.name="$JOB_NAME" \
    trainer.seed=42 trainer.max_iter=200 trainer.grad_accum_iter=1 \
    trainer.validation_iter=25 trainer.max_val_iter=5 trainer.run_validation_on_start=true \
    checkpoint.save_iter=50 checkpoint.verbose=false \
    scheduler.cycle_lengths=[200] scheduler.warm_up_steps=[10] \
    scheduler.f_start=[0.0] scheduler.f_max=[1.0] scheduler.f_min=[0.1] \
    optimizer.lr="$LEARNING_RATE" trainer.callbacks.skip_nan_step.max_consecutive_nan=1 \
    "$@"