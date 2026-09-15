#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RECIPE_DIR="$PWD"
COSMOS_ROOT="$(cd "$RECIPE_DIR/../../../../.." && pwd)"
FRAMEWORK_ROOT="$COSMOS_ROOT/packages/cosmos3"
FFMPEG_ROOT="${FFMPEG_ROOT:-$COSMOS_ROOT/packages/ffmpeg}"

if [[ -x "$FRAMEWORK_ROOT/.venv/bin/python" ]]; then
    export PATH="$FRAMEWORK_ROOT/.venv/bin:$PATH"
fi

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

python -c 'from torchcodec.decoders import VideoDecoder' >/dev/null
cd "$FRAMEWORK_ROOT"
exec python "$RECIPE_DIR/evaluate_robodojo_checkpoint.py" "$@"