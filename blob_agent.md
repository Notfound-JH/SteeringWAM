# RoboDojo Blob Asset Guide

This document describes how to upload and restore the non-Git assets used by
the Cosmos3-Edge RoboDojo training and closed-loop evaluation workflow.

## Storage layout

The Azure Blob container SAS URL is stored locally, never committed:

```text
/home/aiscuser/.blob-sync/temp_sas_url.txt
```

The current SAS expires at `2026-09-22T06:03:00Z`. Replace the file with a new
container SAS URL before that time when a transfer must continue. The SAS needs
at least read, create, write, and list permissions.

All RoboDojo artifacts live below:

```text
jinghe/steeringwam/robodojo_bundle/
|-- base_models/                         # Cosmos3-Edge DCP and Wan2.2 VAE
|-- datasets/
|   |-- RoboDojo_ee_lerobot_v30_video/   # 25 FPS LeRobot v3 EE training data
|   `-- RoboDojo_hdf5/                   # Original RoboDojo HDF5 data
|-- evaluation/eval_result/              # Closed-loop metrics and videos
|-- simulator/Assets/                    # RoboDojo Isaac Sim USD assets
`-- training/action_policy_robodojo_edge/ # Config, logs, offline eval, DCP checkpoints
```

Verified remote manifest for the upload completed on 2026-09-15:

| Group | Files | Content bytes | GiB |
| --- | ---: | ---: | ---: |
| `base_models` | 9 | 9,559,165,771 | 8.903 |
| `datasets/RoboDojo_ee_lerobot_v30_video` | 602 | 119,928,768,368 | 111.692 |
| `datasets/RoboDojo_hdf5` | 14,001 | 522,982,222,916 | 487.065 |
| `evaluation/eval_result` | 8 | 4,579,336 | 0.004 |
| `simulator/Assets` | 15,367 | 41,269,533,508 | 38.435 |
| `training/action_policy_robodojo_edge` | 91 | 153,461,672,964 | 142.922 |

File sizes above are logical content lengths. Local disk usage can be larger
because the source repositories contain sparse files and Git/LFS metadata that
are not part of the restored asset directories.

The completed remote audit contains 30,078 objects and 847,205,942,863 content
bytes across these six groups. Every remote group exactly matched its local
file count and logical byte total.

The older checkpoint-only backup remains at
`jinghe/steeringwam/cosmos3_action_robodojo/action_policy_robodojo_edge/`.
Use `robodojo_bundle/` for full recovery.

## Requirements

Install AzCopy and make it executable at `/home/aiscuser/.local/bin/azcopy`, or
set `AZCOPY_BIN` to another location. Confirm the SAS metadata without printing
its signature:

```bash
python - <<'PY'
from pathlib import Path
from urllib.parse import parse_qs, urlparse

url = Path("/home/aiscuser/.blob-sync/temp_sas_url.txt").read_text().strip()
parsed = urlparse(url)
query = parse_qs(parsed.query)
print("container:", parsed.path.strip("/"))
print("permissions:", query.get("sp", ["unknown"])[0])
print("expiry:", query.get("se", ["unknown"])[0])
PY
```

Never paste the SAS URL into Git, logs, documentation, or process arguments
that will be shared. Rotate the SAS immediately if its `sig` value is exposed.

## Upload and resume

From the SteeringWAM root, launch the versioned uploader in tmux:

```bash
chmod +x upload_robodojo_assets_to_sas.sh
tmux new-session -d -s robodojo-sas-assets \
  'cd /home/aiscuser/SteeringWAM && ./upload_robodojo_assets_to_sas.sh'
tmux attach -t robodojo-sas-assets
```

For non-default checkouts or destinations, set `STEERING_ROOT`,
`ROBODOJO_ROOT`, `RECIPE_DIR`, `ROBODOJO_SAS_REMOTE_ROOT`, or
`SAS_UPLOAD_LOG_DIR` before launching the script.

Per-stage logs and `.complete` markers are written to `sas-upload-logs/`.
AzCopy uses `--overwrite=ifSourceNewer` and `--check-length=true`, so rerunning
the command skips matching remote files. To force one stage to be checked
again, remove only its local marker and rerun the uploader:

```bash
rm sas-upload-logs/lerobot_v3_ee.complete
./upload_robodojo_assets_to_sas.sh
```

Do not run multiple AzCopy uploads concurrently on the same machine. If a SAS
expires during transfer, replace `temp_sas_url.txt`, remove the incomplete
stage marker if one exists, and rerun the uploader.

## Full restore

Clone the code first, initialize RoboDojo, and define the restore locations:

```bash
git clone https://github.com/Notfound-JH/SteeringWAM.git
cd SteeringWAM
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git RoboDojo
git -C RoboDojo submodule update --init --recursive

export STEERING_ROOT="$PWD"
export ROBODOJO_ROOT="$PWD/RoboDojo"
export RECIPE_DIR="$PWD/cookbooks/cosmos3/generator/action/finetune"
export AZCOPY_BIN="${AZCOPY_BIN:-/home/aiscuser/.local/bin/azcopy}"
export SAS_FILE="${AZURE_BLOB_SAS_FILE:-/home/aiscuser/.blob-sync/temp_sas_url.txt}"
```

Build a source URL without printing it, then restore every artifact group:

```bash
SAS_URL="$(tr -d '\r\n' < "$SAS_FILE")"
CONTAINER_URL="${SAS_URL%%\?*}"
SAS_QUERY="${SAS_URL#*\?}"
REMOTE_ROOT="$CONTAINER_URL/jinghe/steeringwam/robodojo_bundle"

download() {
  local remote_path=$1
  local local_path=$2
  mkdir -p "$local_path"
  "$AZCOPY_BIN" copy \
    "$REMOTE_ROOT/$remote_path/*?$SAS_QUERY" "$local_path" \
    --recursive=true \
    --overwrite=ifSourceNewer \
    --check-length=true
}

download base_models "$RECIPE_DIR/checkpoints"
download training/action_policy_robodojo_edge \
  "$RECIPE_DIR/outputs/train/cosmos3_action_robodojo/action_sft/action_policy_robodojo_edge"
download evaluation/eval_result "$ROBODOJO_ROOT/eval_result"
download simulator/Assets \
  "$ROBODOJO_ROOT/.cache/robodojo_assets_repo/Assets"
download datasets/RoboDojo_ee_lerobot_v30_video \
  "$ROBODOJO_ROOT/.cache/robodojo_data_huggingface_lerobot_v3.0_ee_repo/data/RoboDojo_ee_lerobot_v30_video"
download datasets/RoboDojo_hdf5 \
  "$ROBODOJO_ROOT/.cache/robodojo_data_huggingface_hdf5_repo/data/RoboDojo"
```

Recreate the paths expected by RoboDojo and install the policy adapter:

```bash
mkdir -p "$ROBODOJO_ROOT/data"
ln -sfn "$ROBODOJO_ROOT/.cache/robodojo_assets_repo/Assets" \
  "$ROBODOJO_ROOT/Assets"
ln -sfn \
  "$ROBODOJO_ROOT/.cache/robodojo_data_huggingface_lerobot_v3.0_ee_repo/data/RoboDojo_ee_lerobot_v30_video" \
  "$ROBODOJO_ROOT/data/RoboDojo_ee_lerobot_v30_video"
ln -sfn \
  "$ROBODOJO_ROOT/.cache/robodojo_data_huggingface_hdf5_repo/data/RoboDojo" \
  "$ROBODOJO_ROOT/data/RoboDojo"
bash "$RECIPE_DIR/install_robodojo_xpolicylab_adapter.sh" "$ROBODOJO_ROOT"
```

Restore software environments separately; the Blob backup intentionally does
not contain Conda environments, Isaac Sim packages, Git object databases, or
temporary caches. Follow the RoboDojo and cookbook READMEs to reinstall those
dependencies.

## Verification

Validate the recovered training dataset and checkpoint metadata:

```bash
python "$RECIPE_DIR/validate_robodojo_dataset.py" \
  "$ROBODOJO_ROOT/data/RoboDojo_ee_lerobot_v30_video"

cat "$RECIPE_DIR/outputs/train/cosmos3_action_robodojo/action_sft/action_policy_robodojo_edge/checkpoints/latest_checkpoint.txt"
find "$RECIPE_DIR/outputs/train/cosmos3_action_robodojo/action_sft/action_policy_robodojo_edge/checkpoints" \
  -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
```

List remote objects without downloading them:

```bash
"$AZCOPY_BIN" list \
  "$CONTAINER_URL/jinghe/steeringwam/robodojo_bundle?$SAS_QUERY" \
  --machine-readable
```

For a strict audit, compare local and remote file counts and total content
lengths for each top-level group. AzCopy's successful summary must report zero
failed transfers before a stage's `.complete` marker is trusted.