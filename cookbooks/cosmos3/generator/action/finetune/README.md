# Cosmos3 Action-Policy Fine-Tuning (SFT)

This example demonstrates supervised fine-tuning (SFT) of [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) or [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) into a robot action policy, using the action-policy recipe from [cosmos-framework](https://github.com/NVIDIA/cosmos-framework). Two embodiments are covered; the Nano recipes reproduce Cosmos3 paper results, while the Edge LIBERO recipe is an experimental architecture port with the same data and optimization settings:

- **DROID** — reproduces [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID): trained on real-robot DROID data, evaluated on the RoboLab simulation benchmark.
- **LIBERO-10** — the Nano recipe reproduces the Cosmos3 paper's LIBERO-10 results; the Edge port uses the same simulation benchmark and training data.
- **RoboDojo** — an experimental Cosmos3-Edge port for the official dual-arm LeRobot v3.0 EE dataset.

| Recipe | Launch shell | Base model | Dataset |
| --- | --- | --- | --- |
| Policy-DROID SFT (Nano) | `launch_sft_action_policy_droid_nano.sh` | Cosmos3-Nano | [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) success split |
| Policy-LIBERO-10 SFT (Nano) | `launch_sft_action_policy_libero_10_nano.sh` | Cosmos3-Nano | [LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) `libero_10` |
| Policy-LIBERO-10 SFT (Edge) | `launch_sft_action_policy_libero_10_edge.sh` | Cosmos3-Edge | [LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) `libero_10` |
| Policy-LIBERO-all SFT (Nano) | `launch_sft_action_policy_libero_all_nano.sh` | Cosmos3-Nano | [LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) all 4 suites |
| Policy-RoboDojo SFT (Edge) | `launch_sft_action_policy_robodojo_edge.sh` | Cosmos3-Edge | [RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) `lerobot_v3.0_ee` |

The DROID recipe uses the registered `action_policy_droid_nano` experiment: `joint_pos` 8-D actions, proprioceptive state, `concat_view` 480p video, chunk length 32, episode-shuffle streaming, JSON-formatted action prompts (`format_prompt_as_json=True`), and the optional `keep_ranges_1_0_1.json` window filter. The reference reproduction runs lr 2e-4 (cosine, cycle 100000), generator loss_scale 10, global batch 8192 (HSDP 32x8 = 256 ranks; GB200 reference, 64 nodes x 4), for 10000 iters. The action prompt is serialized as JSON at both train and eval time, so evaluation must use the matching JSON prompt format.

The LIBERO recipe uses `frame_wise_relative` rot6d 10-D actions, `quantile_rot` normalization, `concat_view` (third-person + wrist) at 20 fps, lr 5e-5 / warmup 500 / cycle 16000, and global batch 2048. Nano provides two reference presets; the Edge architecture port currently covers the `libero_10` preset:

- **libero-10** — `action_policy_libero_nano` + `launch_sft_action_policy_libero_10_nano.sh`; trains on `libero_10` alone (max_iter 2000).
- **libero-10 Edge** — the local `train_action_policy_libero_edge.py` entrypoint derives `action_policy_libero_edge` from the framework's `EDGE_MODEL_CONFIG`, then reuses the Nano LIBERO dataset and training tree. It adds Edge's `k_norm_und_for_gen` optimizer key and starts from `Cosmos3-Edge`; no reference success rate is claimed for this port.
- **libero-all** — `action_policy_libero_all_nano` + `launch_sft_action_policy_libero_all_nano.sh`; equal mix of all 4 LIBERO suites, which needs longer training (max_iter 5000).

For a runnable egocentric hand-pose data conversion example, see
[`README_egocentric_hand_action.md`](./README_egocentric_hand_action.md). It
converts a sample video and 3D hand-pose annotation pair into the raw 57D hand
Action format used by the dataset path.

The recipe uses `[job].task = "vfm"` with the registered `action_policy_droid_nano` experiment. It trains a DROID policy model with `joint_pos` 8-D actions, proprioceptive state, `concat_view` 480p video, chunk length 32, episode-shuffle streaming, JSON-formatted action prompts, and the optional `keep_ranges_1_0_1.json` window filter.

## Prerequisites

1. **Install cosmos-framework.** From the `cosmos` repository root, run `mkdir -p packages && git clone https://github.com/NVIDIA/cosmos-framework.git packages/cosmos3`, then run `uv sync --all-extras --group=cu130-train` inside `packages/cosmos3` (`cu128-train` on a CUDA 12.x driver). See the shared [cosmos-framework setup](../../../README.md#cosmos-framework).
2. **Recommended container.** For a curated CUDA + PyTorch base, NVIDIA recommends starting from the NGC PyTorch container **`nvcr.io/nvidia/pytorch:25.09-py3`** (CUDA 13; use **`:25.06-py3`** for a CUDA 12.8 driver). See the cosmos-framework [setup guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#recommended-base-image).
3. **Activate** the cosmos-framework venv from anywhere inside this checkout with `source "$(git rev-parse --show-toplevel)/packages/cosmos3/.venv/bin/activate"`. The Edge launcher also discovers this venv automatically when it exists at the default location.
4. **Hugging Face access.** The base model, Wan2.2 VAE, Cosmos3-DROID dataset, and `keep_ranges_1_0_1.json` filter are hosted on Hugging Face — authenticate once with `uvx hf@latest auth login` (or export `HF_TOKEN`) and accept any model/dataset terms first.
5. **Weights & Biases.** The Nano TOMLs follow the canonical examples and default to `wandb_mode="online"`; the experimental Edge recipe defaults to `disabled` so it runs locally without credentials. To log the Edge run online, export `WANDB_API_KEY` and set `EXTRA_TAIL_OVERRIDES="job.wandb_mode=online"`.
6. **FFmpeg for LIBERO.** TorchCodec needs FFmpeg shared libraries. If FFmpeg is not installed system-wide, run `conda create -y -p "$(git rev-parse --show-toplevel)/packages/ffmpeg" -c conda-forge 'ffmpeg>=6,<7'`; the Edge launcher discovers that prefix automatically. Set `FFMPEG_ROOT` for a different installation.
7. **Prepare the Cosmos3-DROID dataset.** The full Cosmos3-DROID dataset is large, so this cookbook expects you to stage it explicitly. By default the launcher looks for `data/Cosmos3-DROID/success/meta/info.json` under this folder. Set `DATASET_PATH=/path/to/Cosmos3-DROID/success` to use another filesystem.
8. **Run from this directory** (`cookbooks/cosmos3/generator/action/finetune/`). Converted checkpoints, the Wan2.2 VAE, the `keep_ranges_1_0_1.json` filter, and run outputs default to `checkpoints/`, `data/`, and `outputs/` here (all git-ignored).

## Quick start

Stage the dataset, then run the launcher:

```shell
# One possible layout. The dataset is large; put it on a filesystem with enough space.
uvx hf@latest download \
    nvidia/Cosmos3-DROID \
    --repo-type dataset \
    --local-dir data/Cosmos3-DROID

bash launch_sft_action_policy_droid_nano.sh
```

The launcher is a complete local wrapper for the public cookbook:

- checks that the Cosmos3-DROID success split exists
- downloads `Wan2.2_VAE.pth` if needed
- converts `Cosmos3-Nano` to a local DCP checkpoint if needed
- downloads `keep_ranges_1_0_1.json` if needed
- launches training with `action_policy_droid_nano.toml`

The script intentionally stays close to the `cosmos-framework` example launcher: `DATASET_PATH`
is bridged to `DROID_ROOT`, `BASE_CHECKPOINT_PATH` and `WAN_VAE_PATH` are exported for the TOML,
`EXTRA_TAIL_OVERRIDES` is appended after `--`, and `NPROC_PER_NODE` / `NNODES` / `NODE_RANK` /
`MASTER_ADDR` / `MASTER_PORT` map directly to `torchrun`. The extra cookbook behavior is the
reasoner-style local prep above and enabling the released `keep_ranges_1_0_1.json` filter by default.

Paths are fixed at the top of the script (under this git-ignored folder) — edit them there or export env vars to relocate data and checkpoints:

```shell
export DATASET_PATH=/scratch/Cosmos3-DROID/success
export BASE_CHECKPOINT_PATH=/scratch/checkpoints/Cosmos3-Nano
export WAN_VAE_PATH=/scratch/checkpoints/wan22_vae/Wan2.2_VAE.pth
export FILTER_PATH=/scratch/droid/keep_ranges_1_0_1.json
bash launch_sft_action_policy_droid_nano.sh
```

The committed TOML pins the GB200 reference shape (HSDP 32x8 = 256 ranks, global
batch 8192). To run on a single 8-GPU node — e.g. a short smoke test — drop the
replicate degree to 1 alongside the iteration/batch knobs:

```shell
export EXTRA_TAIL_OVERRIDES=" \
  job.wandb_mode=disabled \
  trainer.max_iter=10 \
  checkpoint.save_iter=10 \
  model.config.parallelism.data_parallel_replicate_degree=1 \
  dataloader_train.max_samples_per_batch=32 \
"
bash launch_sft_action_policy_droid_nano.sh
```

To reproduce the reference at full global batch 8192 on fewer GPUs, keep
`data_parallel_replicate_degree=1` and raise `trainer.grad_accum_iter` (32 on one
8-GPU node) instead of shrinking the batch.

## LIBERO quick start

Each launcher stages its dataset (auto-downloaded if missing), downloads the Wan
VAE, converts the base checkpoint, and trains.

**libero-10:**

```shell
bash launch_sft_action_policy_libero_10_nano.sh
# or, for Cosmos3-Edge:
bash launch_sft_action_policy_libero_10_edge.sh
```

- downloads `nvidia/LIBERO_LeRobot_v3` `libero_10` to `data/LIBERO_LeRobot_v3/libero_10` if missing
- launches Nano with `action_policy_libero_10_nano.toml`, or Edge with `action_policy_libero_10_edge.toml` through the local Edge experiment entrypoint (max_iter 2000)
- defaults Edge to 8-way FSDP with grad accumulation 2, preserving the Nano recipe's global batch 2048 on one 8-GPU node

**libero-all:**

```shell
bash launch_sft_action_policy_libero_all_nano.sh
```

- downloads all 4 suites of `nvidia/LIBERO_LeRobot_v3` to `data/LIBERO_LeRobot_v3` if missing
- launches training with `action_policy_libero_all_nano.toml` (max_iter 5000; it needs longer to converge)

All launchers download `Wan2.2_VAE.pth` and convert their selected Nano or Edge base to a local DCP checkpoint if needed.
Relocate inputs via env vars, or run a short smoke test:

```shell
export LIBERO_ROOT=/scratch/LIBERO_LeRobot_v3/libero_10   # libero-all preset: the parent dir, /scratch/LIBERO_LeRobot_v3
export EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=10 checkpoint.save_iter=10 dataloader_train.max_samples_per_batch=32"
bash launch_sft_action_policy_libero_10_nano.sh
```

On a 4-GPU node, keep the Edge global batch at 2048 by matching the FSDP shard degree and raising gradient accumulation:

```shell
export NPROC_PER_NODE=4
export EXTRA_TAIL_OVERRIDES="model.config.parallelism.data_parallel_shard_degree=4 trainer.grad_accum_iter=4"
bash launch_sft_action_policy_libero_10_edge.sh
```

Validate the Edge experiment and TOML without downloading assets or launching training:

```shell
DRYRUN=1 bash launch_sft_action_policy_libero_10_edge.sh
```

Checkpoints are saved every 500 iters.

## RoboDojo Edge quick start

The RoboDojo recipe consumes the official `RoboDojo_ee_lerobot_v30_video`
dataset at 25 FPS. Its raw state and action are 16D absolute dual-arm
end-effector commands:

```text
[left position xyz, left quaternion wxyz, left gripper,
 right position xyz, right quaternion wxyz, right gripper]
```

The three video keys are `cam_high`, `cam_left_wrist`, and
`cam_right_wrist`. The reader composes the high camera above the two wrist
cameras and preserves the raw absolute action representation. This differs
from the 14D joint-space `robodojo` domain reserved elsewhere in the framework,
so the recipe uses the dedicated `robodojo-ee` domain.

The launcher downloads and validates the dataset when it is missing:

```shell
bash launch_sft_action_policy_robodojo_edge.sh
```

To use an existing copy or another checkout of RoboDojo:

```shell
export ROBODOJO_ROOT=/path/to/RoboDojo_ee_lerobot_v30_video
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Edge-DCP
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
bash launch_sft_action_policy_robodojo_edge.sh
```

Validate metadata and the merged Hydra configuration without downloading
weights or starting training:

```shell
python validate_robodojo_dataset.py "$ROBODOJO_ROOT"
DRYRUN=1 bash launch_sft_action_policy_robodojo_edge.sh
```

For a short single-node run, reduce the number of iterations and per-rank
batch before selecting hyperparameters for a full run:

```shell
export EXTRA_TAIL_OVERRIDES="trainer.max_iter=5 checkpoint.save_iter=5 dataloader_train.max_samples_per_batch=4"
bash launch_sft_action_policy_robodojo_edge.sh
```

This is an architecture and dataset integration recipe. It does not claim a
published RoboDojo success rate, and the LIBERO-derived optimizer schedule is
only a starting point.

Run deterministic offline action evaluation against the episode-level
validation split. The primary evaluator is fixed to the 60K DCP checkpoint by
default; do not select a checkpoint using validation or test results. Pass
`--checkpoint` only for explicitly labeled diagnostic evaluations of
intermediate checkpoints. The evaluator writes aggregate and per-sample
metrics after every completed sample:

```shell
CUDA_VISIBLE_DEVICES=0 ./launch_offline_eval_robodojo.sh \
    --num-samples 32 \
    --num-steps 8
```

The report includes raw 16D action MAE/RMSE, per-arm position error in
centimeters, quaternion angular error in degrees, and per-arm gripper MAE. The
current training recipe uses the full dataset, so this deterministic split is a
sanity-check slice rather than a strictly unseen holdout set.

### RoboDojo closed-loop evaluation

Install the versioned XPolicyLab adapter into a RoboDojo checkout, then launch
the simulator evaluation through RoboDojo's standard CLI:

```shell
bash install_robodojo_xpolicylab_adapter.sh /path/to/RoboDojo

cd /path/to/RoboDojo
COSMOS3_ROBODOJO_EXECUTE_STEPS=16 \
COSMOS3_ROBODOJO_NUM_STEPS=8 \
bash scripts/robodojo.sh eval \
    --policy-dir XPolicyLab/policy/Cosmos3_RoboDojo \
    --task deposit_coin \
    --ckpt iter_000002000 \
    --policy-env /path/to/SteeringWAM/packages/cosmos3/.venv \
    --eval-env RoboDojo \
    --env-cfg arx_x5 \
    --action-type ee \
    --seed 0 \
    --policy-gpu 4 \
    --env-gpu 5 \
    --eval-num 1
```

The adapter executes 16D absolute dual-arm EE action chunks through the
XPolicyLab WebSocket protocol. It refreshes observations after every executed
action, so the three evaluation videos are recorded at the simulator's 25 FPS
rather than once per predicted chunk. See
[`robodojo_xpolicylab/Cosmos3_RoboDojo/README.md`](./robodojo_xpolicylab/Cosmos3_RoboDojo/README.md)
for simulator prerequisites and checkpoint path overrides.

## Outputs

Training writes to `outputs/train/<project>/<group>/<name>/`:

- `checkpoints/iter_<N>/` — DCP checkpoint (model / optim / scheduler / trainer state); `checkpoints/latest_checkpoint.txt` names the newest.
- `config.yaml`, launch metadata, logs, and one directory per registered callback.

## Export to Hugging Face safetensors

```shell
RUN_DIR=outputs/train/<project>/<group>/<name>
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" \
    --config-file "$RUN_DIR/config.yaml" \
    -o "$RUN_DIR/model"
```

Use the exported `$RUN_DIR/model` with the [Cosmos3-Nano-Policy-DROID inference cookbook](../run_policy_with_cosmos_framework.md).

## Convert to Diffusers

Convert the export into a Diffusers pipeline:

```shell
python -m cosmos_framework.scripts.convert_model_to_diffusers \
    --checkpoint-path "$RUN_DIR/model" -o "$RUN_DIR/diffusers"
```

The input is the Hugging Face directory produced by `export_model` above (not a raw DCP checkpoint), so run the export first. See the [Export and Convert Checkpoints](../../../../../README.md#export-and-convert-checkpoints) overview for details.

## Advanced configuration

These recipes are intentionally minimal. For the full post-training reference — raw `torchrun`, resuming, every TOML field, the `keep_ranges_1_0_1.json` filter, and advanced HSDP/multi-node parallelism — see the canonical cosmos-framework docs:

- [Cosmos3-Nano-Policy-DROID post-training guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/action_policy_droid_posttrain.md)
- [Post-Training (SFT) guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [SFT structured-TOML config reference](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/sft_config.md)
- [environment variables](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/environment_variables.md) · [FAQ / OOM during SFT](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md)

> SFT here is a multi-GPU `torchrun` job, so this cookbook ships as a launch script + this README rather than a one-click notebook.
