#!/usr/bin/env python3

"""Evaluate a RoboDojo action-policy checkpoint on deterministic dataset windows."""

from cosmos_framework.inference.common.init import init_script

init_script()

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_robodojo_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.init import get_rank
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    disable_runtime_ema_for_frozen_config,
    maybe_init_distributed,
)

ACTION_DIM = 16
LEFT_POSITION = slice(0, 3)
LEFT_QUATERNION = slice(3, 7)
LEFT_GRIPPER = 7
RIGHT_POSITION = slice(8, 11)
RIGHT_QUATERNION = slice(11, 15)
RIGHT_GRIPPER = 15


def parse_args() -> argparse.Namespace:
    recipe_dir = Path(__file__).resolve().parent
    default_run_dir = (
        recipe_dir
        / "outputs/train/cosmos3_action_robodojo/action_sft/action_policy_robodojo_edge"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "ROBODOJO_ROOT",
                recipe_dir.parents[4] / "RoboDojo/data/RoboDojo_ee_lerobot_v30_video",
            )
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "val", "full"), default="val")
    parser.add_argument("--split-val-ratio", type=float, default=0.01)
    return parser.parse_args()


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint.expanduser().resolve()
    latest_file = run_dir / "checkpoints/latest_checkpoint.txt"
    latest = latest_file.read_text(encoding="utf-8").strip()
    candidate = Path(latest)
    if not candidate.is_absolute():
        candidate = run_dir / "checkpoints" / candidate
    return candidate.resolve()


def build_data_batch(sample: dict[str, Any]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            batch[key] = [value.unsqueeze(0)]
        else:
            batch[key] = [value]
    return batch


def quaternion_error_degrees(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = torch.nn.functional.normalize(prediction.float(), dim=-1)
    target = torch.nn.functional.normalize(target.float(), dim=-1)
    cosine = torch.sum(prediction * target, dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(cosine) * (180.0 / math.pi)


def sample_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = prediction[..., :ACTION_DIM].float()
    target = target[..., :ACTION_DIM].float()
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    error = prediction - target
    return {
        "action_mae": error.abs().mean().item(),
        "action_rmse": error.square().mean().sqrt().item(),
        "left_position_l2_cm": torch.linalg.vector_norm(
            error[..., LEFT_POSITION], dim=-1
        ).mean().mul(100.0).item(),
        "right_position_l2_cm": torch.linalg.vector_norm(
            error[..., RIGHT_POSITION], dim=-1
        ).mean().mul(100.0).item(),
        "left_rotation_degrees": quaternion_error_degrees(
            prediction[..., LEFT_QUATERNION], target[..., LEFT_QUATERNION]
        ).mean().item(),
        "right_rotation_degrees": quaternion_error_degrees(
            prediction[..., RIGHT_QUATERNION], target[..., RIGHT_QUATERNION]
        ).mean().item(),
        "left_gripper_mae": error[..., LEFT_GRIPPER].abs().mean().item(),
        "right_gripper_mae": error[..., RIGHT_GRIPPER].abs().mean().item(),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = results[0]["metrics"].keys()
    return {
        name: float(np.mean([result["metrics"][name] for result in results]))
        for name in metric_names
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    checkpoint = resolve_checkpoint(args.run_dir, args.checkpoint)
    config_file = (args.run_dir / "config.yaml").resolve()
    output = args.output or args.run_dir / "offline_eval" / f"{checkpoint.name}.json"

    maybe_init_distributed()
    setup = OmniSetupOverrides(
        checkpoint_path=str(checkpoint),
        config_file=str(config_file),
        output_dir=args.run_dir / "offline_eval/runtime",
        sampler="unipc",
        guardrails=False,
    ).build_setup()
    init_output_dir(setup.output_dir)
    setup = disable_runtime_ema_for_frozen_config(setup)
    pipeline = OmniInference.create(setup)
    model = pipeline.model
    model.eval()

    dataset = get_action_robodojo_sft_dataset(
        root=str(args.data_root),
        split=args.split,
        split_val_ratio=args.split_val_ratio,
        split_seed=args.seed,
        cfg_dropout_rate=0.0,
        iterable_shuffle=False,
    )
    sample_count = min(args.num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64).tolist()
    results: list[dict[str, Any]] = []
    started = time.monotonic()

    for sample_number, index in enumerate(indices, start=1):
        random.seed(args.seed + index)
        torch.manual_seed(args.seed + index)
        sample = dataset[index]
        target = sample["action_raw"].detach().cpu()[..., :ACTION_DIM]
        with torch.inference_mode():
            generated = model.generate_samples_from_batch(
                build_data_batch(sample),
                guidance=args.guidance,
                seed=[args.seed + index],
                num_steps=args.num_steps,
                n_sample=1,
            )
        prediction = generated["action"][0].detach().cpu()[..., :ACTION_DIM]
        metrics = sample_metrics(prediction, target)
        results.append({"sample": sample_number, "index": index, "metrics": metrics})
        if get_rank() == 0:
            report = {
                "checkpoint": str(checkpoint),
                "data_root": str(args.data_root.resolve()),
                "split": args.split,
                "split_val_ratio": args.split_val_ratio,
                "seed": args.seed,
                "num_steps": args.num_steps,
                "guidance": args.guidance,
                "completed_samples": len(results),
                "requested_samples": sample_count,
                "elapsed_seconds": time.monotonic() - started,
                "metrics": aggregate(results),
                "samples": results,
            }
            write_report(output, report)
            print(
                f"[{sample_number}/{sample_count}] index={index} "
                f"mae={metrics['action_mae']:.6f} "
                f"left_pos_cm={metrics['left_position_l2_cm']:.3f} "
                f"right_pos_cm={metrics['right_position_l2_cm']:.3f}",
                flush=True,
            )

    if get_rank() == 0:
        print(f"Offline evaluation complete: {output}")


if __name__ == "__main__":
    main()