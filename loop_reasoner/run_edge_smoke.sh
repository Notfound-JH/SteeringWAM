#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=''
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export NCCL_DEBUG=WARN
export DROID_ROOT="$WORKSPACE/datasets/droid_plus_lerobot_640x360_20260412"
export WAN_VAE_PATH="$WORKSPACE/cookbooks/cosmos3/generator/action/finetune/checkpoints/wan22_vae/Wan2.2_VAE.pth"
export BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$WORKSPACE/models/Cosmos3-Edge-Policy-DROID-DCP}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$WORKSPACE/outputs/edge_droid_smoke}"
TOKENIZER="$WORKSPACE/models/Cosmos3-Edge-Policy-DROID"
cd "$WORKSPACE/packages/cosmos3"
exec bash examples/launch_sft_action_policy_droid_edge.sh -- \
    trainer.max_iter=2 trainer.grad_accum_iter=1 trainer.max_val_iter=1 \
    checkpoint.save_iter=2 trainer.validation_iter=2 \
    scheduler.cycle_lengths=[2] scheduler.warm_up_steps=[0] \
    model.config.vlm_config.tokenizer.repository=null \
    "++model.config.vlm_config.tokenizer.tokenizer_type=$TOKENIZER" \
    "model.config.vlm_config.model_name=$TOKENIZER" \
    ++dataloader_train.dataloader.datasets.droid.dataset.episode_limit=128 \
    ++dataloader_val.dataloader.datasets.droid.dataset.episode_limit=128 \
    ++dataloader_train.dataloader.datasets.droid.dataset.video_backend=pyav \
    ++dataloader_val.dataloader.datasets.droid.dataset.video_backend=pyav \
    dataloader_train.dataloader.num_workers=1 dataloader_val.dataloader.num_workers=1 \
    "$@"