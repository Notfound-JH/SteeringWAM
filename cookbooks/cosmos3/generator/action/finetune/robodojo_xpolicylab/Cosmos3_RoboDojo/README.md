# Cosmos3 RoboDojo

XPolicyLab adapter for evaluating a Cosmos3-Edge checkpoint on RoboDojo's
dual-arm X5 tasks. It consumes the high, left-wrist, and right-wrist RGB
cameras and returns 16D absolute dual-arm end-effector action chunks.

## Prerequisites

From the RoboDojo repository root:

```bash
bash scripts/install.sh -i
bash scripts/init_assets.sh
python utils/update_embodiment_config_path.py
conda install -n RoboDojo -c conda-forge ffmpeg -y
```

The policy environment must be the Cosmos framework virtual environment and
must include the XPolicyLab WebSocket codec dependency:

```bash
uv pip install \
  --python ../packages/cosmos3/.venv/bin/python \
  'msgpack-numpy>=0.4.8'
```

## Single-Episode Evaluation

The default checkpoint run directory is resolved from the SteeringWAM action
finetuning output. Override it with `COSMOS3_ROBODOJO_RUN_DIR`, or set the
checkpoint and frozen training config directly with
`COSMOS3_ROBODOJO_CHECKPOINT` and `COSMOS3_ROBODOJO_CONFIG`.

```bash
COSMOS3_ROBODOJO_EXECUTE_STEPS=16 \
COSMOS3_ROBODOJO_NUM_STEPS=8 \
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Cosmos3_RoboDojo \
  --task deposit_coin \
  --ckpt iter_000002000 \
  --policy-env ../packages/cosmos3/.venv \
  --eval-env RoboDojo \
  --env-cfg arx_x5 \
  --action-type ee \
  --seed 0 \
  --policy-gpu 4 \
  --env-gpu 5 \
  --eval-num 1
```

Results and the three camera videos are written below
`eval_result/RoboDojo/<task>/Cosmos3_RoboDojo/`.

`COSMOS3_ROBODOJO_EXECUTE_STEPS` controls how many actions from each 16-step
prediction are executed before replanning. The adapter defaults to 4 for
tighter feedback; 16 is substantially faster for smoke evaluation.