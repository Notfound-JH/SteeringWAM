#!/usr/bin/env bash

# RoboDojo dual-arm action-policy SFT on Cosmos3-Edge (4-GPU FSDP by default).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RECIPE_DIR="$PWD"
TOML_FILE="$RECIPE_DIR/toml/sft_config/action_policy_robodojo_edge.toml"
TRAIN_ENTRYPOINT="$RECIPE_DIR/train_action_policy_robodojo_edge.py"
VALIDATOR="$RECIPE_DIR/validate_robodojo_dataset.py"
COSMOS_ROOT="$(cd "$RECIPE_DIR/../../../../.." && pwd)"
ROBODOJO_REPO="${ROBODOJO_REPO:-$COSMOS_ROOT/RoboDojo}"
: "${ROBODOJO_ROOT:=$ROBODOJO_REPO/data/RoboDojo_ee_lerobot_v30_video}"
: "${BASE_CHECKPOINT_PATH:=$RECIPE_DIR/checkpoints/Cosmos3-Edge}"
: "${WAN_VAE_PATH:=$RECIPE_DIR/checkpoints/wan22_vae/Wan2.2_VAE.pth}"

DEFAULT_FRAMEWORK_ROOT="$COSMOS_ROOT/packages/cosmos3"
: "${FFMPEG_ROOT:=$COSMOS_ROOT/packages/ffmpeg}"
if ! python -c 'import cosmos_framework' >/dev/null 2>&1; then
    if [[ -x "$DEFAULT_FRAMEWORK_ROOT/.venv/bin/python" ]]; then
        export PATH="$DEFAULT_FRAMEWORK_ROOT/.venv/bin:$PATH"
    else
        cat >&2 <<EOF
ERROR: cosmos-framework is not installed.

Install it at $DEFAULT_FRAMEWORK_ROOT and create its training environment first.
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
    # shellcheck disable=SC2206
    TAIL_OVERRIDES=(${EXTRA_TAIL_OVERRIDES})
fi

export ROBODOJO_ROOT
export BASE_CHECKPOINT_PATH
export WAN_VAE_PATH

if [[ "${DRYRUN:-0}" == "1" ]]; then
    cd "$FRAMEWORK_ROOT"
    IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$RECIPE_DIR/outputs/train}" \
        python "$TRAIN_ENTRYPOINT" --sft-toml="$TOML_FILE" --dryrun -- "${TAIL_OVERRIDES[@]}"
    exit 0
fi

if [[ ! -f "$ROBODOJO_ROOT/meta/info.json" ]]; then
    if [[ ! -x "$ROBODOJO_REPO/scripts/RoboDojo/download_data.sh" ]]; then
        echo "ERROR: missing RoboDojo downloader under $ROBODOJO_REPO" >&2
        exit 1
    fi
    echo "Downloading RoboDojo LeRobot v3.0 EE data ..."
    bash "$ROBODOJO_REPO/scripts/RoboDojo/download_data.sh" huggingface lerobot_v3.0_ee
fi
python "$VALIDATOR" "$ROBODOJO_ROOT"

if ! python -c 'from torchcodec.decoders import VideoDecoder' >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: TorchCodec cannot load FFmpeg/CUDA shared libraries.

Install FFmpeg 6 under $COSMOS_ROOT/packages/ffmpeg or set FFMPEG_ROOT.
EOF
    exit 1
fi

if [[ ! -f "$WAN_VAE_PATH" ]]; then
    uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$(dirname "$WAN_VAE_PATH")"
fi
if [[ ! -d "$BASE_CHECKPOINT_PATH" ]]; then
    python -m cosmos_framework.scripts.convert_model_to_dcp \
        -o "$BASE_CHECKPOINT_PATH" --checkpoint-path Cosmos3-Edge
fi

TORCHRUN_ARGS=(--nproc_per_node="${NPROC_PER_NODE:-4}")
TORCHRUN_ARGS+=(--master_port="${MASTER_PORT:-50012}")
[[ -n "${NNODES:-}" ]] && TORCHRUN_ARGS+=(--nnodes="$NNODES")
[[ -n "${NODE_RANK:-}" ]] && TORCHRUN_ARGS+=(--node_rank="$NODE_RANK")
[[ -n "${MASTER_ADDR:-}" ]] && TORCHRUN_ARGS+=(--master_addr="$MASTER_ADDR")

cd "$FRAMEWORK_ROOT"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$RECIPE_DIR/outputs/train}" \
    torchrun "${TORCHRUN_ARGS[@]}" "$TRAIN_ENTRYPOINT" --sft-toml="$TOML_FILE" \
    -- "${TAIL_OVERRIDES[@]}"