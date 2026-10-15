#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_GPU="${MODEL_GPU:-5}"
PORT="${PORT:-18745}"
MASTER_PORT="${MASTER_PORT:-29745}"
SEED="${SEED:-3407}"
DETERMINISTIC_SEED="${DETERMINISTIC_SEED:-0}"
[[ "$DETERMINISTIC_SEED" == 0 || "$DETERMINISTIC_SEED" == 1 ]] || { echo 'DETERMINISTIC_SEED must be 0 or 1.' >&2; exit 2; }
RUN_DIR="${1:?Pass an output directory for this server run}"
case "$MODEL_GPU" in
    0|1|2|3) [[ "${ALLOW_ALL_GPUS:-0}" == 1 ]] || { echo 'GPU 0-3 requires ALLOW_ALL_GPUS=1.' >&2; exit 2; } ;;
    4|5|6|7) ;;
    *) echo 'MODEL_GPU must be a physical GPU index in 4,5,6,7.' >&2; exit 2 ;;
esac
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$MODEL_GPU" --query-gpu=uuid --format=csv,noheader)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MASTER_ADDR=127.0.0.1 MASTER_PORT WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export PATH="$ROOT/packages/cosmos3/.venv/bin:$ROOT/packages/ffmpeg/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/packages/ffmpeg/lib"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export NO_PROXY=127.0.0.1,localhost
printf 'Cosmos physical GPU: %s (%s); port: %s; output: %s\n' "$MODEL_GPU" "$CUDA_VISIBLE_DEVICES" "$PORT" "$RUN_DIR"
cd "$ROOT/packages/cosmos3"
VAE_SOURCE="$ROOT/cookbooks/cosmos3/generator/action/finetune/checkpoints/wan22_vae/Wan2.2_VAE.pth"
VAE_TARGET="pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"
if [[ ! -e "$VAE_TARGET" && ! -L "$VAE_TARGET" ]]; then
    [[ -f "$VAE_SOURCE" ]] || { echo 'Local Wan VAE is missing.' >&2; exit 1; }
    mkdir -p "$(dirname "$VAE_TARGET")"
    ln -s "$VAE_SOURCE" "$VAE_TARGET"
fi
[[ -f "$VAE_TARGET" ]] || { echo 'Wan VAE path is invalid.' >&2; exit 1; }
VIDEO_ARGS=(--decode-video)
if [[ "${DECODE_VIDEO:-1}" == 0 ]]; then
    VIDEO_ARGS=(--no-decode-video)
fi
SEED_ARGS=(--seed "$SEED")
if [[ "$DETERMINISTIC_SEED" == 1 ]]; then
    SEED_ARGS+=(--deterministic-seed)
fi
exec .venv/bin/python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path "$ROOT/models/Cosmos3-Edge-Policy-DROID" \
    --format-prompt-as-json True \
    --host 127.0.0.1 --port "$PORT" \
    "${SEED_ARGS[@]}" --num-steps 4 --guidance-interval 960 1001 \
    --no-guardrails --no-use-torch-compile \
    "${VIDEO_ARGS[@]}" --output-dir "$RUN_DIR/runtime"