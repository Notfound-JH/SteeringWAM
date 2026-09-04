#!/usr/bin/env bash

# LIBERO-10 action-policy SFT on Cosmos3-Edge (8-GPU FSDP by default).
# Run from this folder with the cosmos-framework training environment active:
#   bash launch_sft_action_policy_libero_10_edge.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RECIPE_DIR="$PWD"
TOML_FILE="$RECIPE_DIR/toml/sft_config/action_policy_libero_10_edge.toml"
TRAIN_ENTRYPOINT="$RECIPE_DIR/train_action_policy_libero_edge.py"
: "${LIBERO_ROOT:=$RECIPE_DIR/data/LIBERO_LeRobot_v3/libero_10}"
: "${BASE_CHECKPOINT_PATH:=$RECIPE_DIR/checkpoints/Cosmos3-Edge}"
: "${WAN_VAE_PATH:=$RECIPE_DIR/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

COSMOS_ROOT="$(cd "$RECIPE_DIR/../../../../.." && pwd)"
DEFAULT_FRAMEWORK_ROOT="$COSMOS_ROOT/packages/cosmos3"
: "${FFMPEG_ROOT:=$COSMOS_ROOT/packages/ffmpeg}"
if ! python -c 'import cosmos_framework' >/dev/null 2>&1; then
    if [[ -x "$DEFAULT_FRAMEWORK_ROOT/.venv/bin/python" ]]; then
        export PATH="$DEFAULT_FRAMEWORK_ROOT/.venv/bin:$PATH"
    else
        cat >&2 <<EOF
ERROR: cosmos-framework is not installed.

Install it at the cookbook's default location:
  cd "$COSMOS_ROOT"
  mkdir -p packages
  git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3
  cd packages/cosmos3
  export GIT_LFS_SKIP_SMUDGE=1
  uv sync --all-extras --group=cu130-train

Then rerun this launcher. It will discover packages/cosmos3/.venv automatically.
EOF
        exit 1
    fi
fi
FRAMEWORK_ROOT="$(python -c 'import pathlib, cosmos_framework; print(pathlib.Path(cosmos_framework.__file__).resolve().parents[1])')"
PYTHON_SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDA_WHEEL_LIB_DIR="$PYTHON_SITE_PACKAGES/nvidia/cu13/lib"
TORCH_LIB_DIR="$PYTHON_SITE_PACKAGES/torch/lib"
if [[ -d "$CUDA_WHEEL_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$CUDA_WHEEL_LIB_DIR:$TORCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -x "$FFMPEG_ROOT/bin/ffmpeg" ]]; then
    export PATH="$FFMPEG_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$FFMPEG_ROOT/lib"
fi

TAIL_OVERRIDES=()
if [[ -n "${EXTRA_TAIL_OVERRIDES:-}" ]]; then
    # EXTRA_TAIL_OVERRIDES follows the framework launcher's space-separated UX.
    # shellcheck disable=SC2206
    TAIL_OVERRIDES=(${EXTRA_TAIL_OVERRIDES})
fi

export LIBERO_ROOT
export BASE_CHECKPOINT_PATH
export WAN_VAE_PATH

# Validate the Edge experiment and TOML without downloading assets or starting training.
if [[ "${DRYRUN:-0}" == "1" ]]; then
    cd "$FRAMEWORK_ROOT"
    IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$RECIPE_DIR/outputs/train}" \
        python "$TRAIN_ENTRYPOINT" --sft-toml="$TOML_FILE" --dryrun -- "${TAIL_OVERRIDES[@]}"
    exit 0
fi

if ! python -c 'import torchcodec' >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: TorchCodec cannot load FFmpeg/CUDA shared libraries.

Install FFmpeg 6 locally with:
  conda create -y -p "$COSMOS_ROOT/packages/ffmpeg" -c conda-forge 'ffmpeg>=6,<7'
or export FFMPEG_ROOT=/path/to/an/ffmpeg/prefix containing bin/ffmpeg and lib/.
EOF
    exit 1
fi

# Stage the 20 FPS LIBERO-10 suite.
if [[ ! -f "$LIBERO_ROOT/meta/info.json" ]]; then
    echo "Downloading nvidia/LIBERO_LeRobot_v3 (libero_10) ..."
    uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 \
        --include 'libero_10/**' --local-dir "$(dirname "$LIBERO_ROOT")"
fi
if [[ ! -f "$LIBERO_ROOT/meta/info.json" ]]; then
    cat >&2 <<EOF
ERROR: missing libero_10 dataset at:
  $LIBERO_ROOT

Expected a LeRobotDataset directory containing meta/info.json. Stage it with:
  uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 \
      --include 'libero_10/**' --local-dir data/LIBERO_LeRobot_v3
or export LIBERO_ROOT=/path/to/libero_10.
EOF
    exit 1
fi

# Download the Wan2.2 VAE and convert the Edge base checkpoint to DCP.
if [[ ! -f "$WAN_VAE_PATH" ]]; then
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$(dirname "$WAN_VAE_PATH")"
fi
if [[ ! -d "$BASE_CHECKPOINT_PATH" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_dcp \
        -o "$BASE_CHECKPOINT_PATH" --checkpoint-path Cosmos3-Edge
fi

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-8}")
TORCHRUN_ARGS+=(--master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

# Edge's model config references packaged JSON files relative to this directory.
cd "$FRAMEWORK_ROOT"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$RECIPE_DIR/outputs/train}" \
    torchrun "${TORCHRUN_ARGS[@]}" "$TRAIN_ENTRYPOINT" --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}"