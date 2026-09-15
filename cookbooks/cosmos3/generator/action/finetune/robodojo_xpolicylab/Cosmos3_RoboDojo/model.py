from __future__ import annotations

from cosmos_framework.inference.common.init import init_script

init_script()

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    disable_runtime_ema_for_frozen_config,
    maybe_init_distributed,
)
from XPolicyLab.model_template import ModelTemplate

CAMERA_ALIASES = {
    "top": ("cam_high", "cam_head"),
    "left": ("cam_left_wrist",),
    "right": ("cam_right_wrist",),
}


def _unwrap_observation(observation: dict[str, Any]) -> dict[str, Any]:
    if "vision" in observation:
        return observation
    if len(observation) == 1:
        candidate = next(iter(observation.values()))
        if isinstance(candidate, dict) and "vision" in candidate:
            return candidate
    raise ValueError("Expected a RoboDojo observation containing 'vision'")


def _get_image(observation: dict[str, Any], aliases: tuple[str, ...]) -> np.ndarray:
    observation = _unwrap_observation(observation)
    vision = observation["vision"]
    for name in aliases:
        if name in vision and "color" in vision[name]:
            image = np.asarray(vision[name]["color"])
            if image.ndim != 3 or image.shape[-1] < 3:
                raise ValueError(f"Camera {name!r} must be HWC RGB, got {image.shape}")
            return np.ascontiguousarray(image[..., :3].astype(np.uint8, copy=False))
    raise KeyError(f"Missing camera; expected one of {aliases}")


def compose_views(observation: dict[str, Any]) -> torch.Tensor:
    images = {
        key: torch.from_numpy(_get_image(observation, aliases).copy()).permute(2, 0, 1)
        for key, aliases in CAMERA_ALIASES.items()
    }
    top = images["top"]
    half_size = (top.shape[-2] // 2, top.shape[-1] // 2)
    left = F.interpolate(images["left"].unsqueeze(0).float(), size=half_size, mode="bilinear", align_corners=False)
    right = F.interpolate(images["right"].unsqueeze(0).float(), size=half_size, mode="bilinear", align_corners=False)
    wrists = torch.cat([left, right], dim=-1).squeeze(0).round().clamp(0, 255).to(torch.uint8)
    return torch.cat([top, wrists], dim=-2)


def action_vector_to_dict(action: np.ndarray) -> dict[str, np.ndarray]:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != (16,):
        raise ValueError(f"Expected a 16D RoboDojo action, got {vector.shape}")
    left_quaternion = vector[3:7]
    right_quaternion = vector[11:15]
    left_quaternion /= max(float(np.linalg.norm(left_quaternion)), 1e-6)
    right_quaternion /= max(float(np.linalg.norm(right_quaternion)), 1e-6)
    return {
        "left_ee_pose": np.concatenate([vector[:3], left_quaternion]),
        "left_ee_joint_state": np.clip(vector[7:8], 0.0, 1.0),
        "right_ee_pose": np.concatenate([vector[8:11], right_quaternion]),
        "right_ee_joint_state": np.clip(vector[15:16], 0.0, 1.0),
    }


def _build_data_batch(sample: dict[str, Any]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            batch[key] = [value.unsqueeze(0)]
        else:
            batch[key] = [value]
    return batch


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        if self.model_cfg.get("action_type") != "ee":
            raise ValueError("Cosmos3_RoboDojo requires action_type=ee")

        self.action_chunk_size = int(self.model_cfg.get("action_chunk_size", 16))
        self.execute_steps = int(self.model_cfg.get("execute_steps", 4))
        if not 1 <= self.execute_steps <= self.action_chunk_size:
            raise ValueError("execute_steps must be in [1, action_chunk_size]")
        self.fps = float(self.model_cfg.get("fps", 25))
        self.num_steps = int(self.model_cfg.get("num_steps", 8))
        self.guidance = float(self.model_cfg.get("guidance", 1.0))
        self.seed = int(self.model_cfg.get("seed", 42))
        self.domain_name = str(self.model_cfg.get("domain_name", "robodojo-ee"))
        self.observation: dict[str, Any] | None = None
        self.observations: list[dict[str, Any]] = []
        self.request_index = 0

        maybe_init_distributed()
        setup = OmniSetupOverrides(
            checkpoint_path=str(self.model_cfg["checkpoint_path"]),
            config_file=str(self.model_cfg["config_file"]),
            output_dir=Path(str(self.model_cfg.get("output_dir", "/tmp/cosmos3_robodojo_policy"))),
            sampler="unipc",
            guardrails=False,
        ).build_setup()
        init_output_dir(setup.output_dir)
        setup = disable_runtime_ema_for_frozen_config(setup)
        pipeline = OmniInference.create(setup)
        self.model = pipeline.model
        self.model.eval()
        self.transform = ActionTransformPipeline(
            cfg_dropout_rate=0.0,
            max_action_dim=64,
            append_viewpoint_info=True,
            append_duration_fps_timestamps=True,
            append_resolution_info=True,
            append_idle_frames=False,
            format_prompt_as_json=True,
        )

    def reset(self):
        self.observation = None
        self.observations.clear()
        self.request_index = 0

    def update_obs(self, obs):
        self.observation = _unwrap_observation(obs)

    def update_obs_batch(self, obs_list):
        self.observations = [_unwrap_observation(obs) for obs in obs_list]

    def _build_sample(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = _unwrap_observation(observation)
        first_frame = compose_views(observation)
        video = torch.zeros(
            (3, self.action_chunk_size + 1, first_frame.shape[-2], first_frame.shape[-1]),
            dtype=torch.uint8,
        )
        video[:, 0] = first_frame
        instruction = observation.get("instruction", "follow the instruction")
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0]
        sample = {
            "ai_caption": str(instruction),
            "video": video,
            "action": torch.zeros((self.action_chunk_size, 16), dtype=torch.float32),
            "conditioning_fps": torch.tensor(self.fps, dtype=torch.long),
            "mode": "wam",
            "domain_id": torch.tensor(get_domain_id(self.domain_name), dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": (
                "The top row is the high third-person camera. The bottom row contains "
                "the left and right wrist cameras."
            ),
        }
        sample = self.transform(sample, str(self.model_cfg.get("resolution", "480")))
        if isinstance(sample.get("ai_caption"), dict):
            sample["ai_caption"] = json.dumps(sample["ai_caption"])
        return sample

    def _predict(self, observation: dict[str, Any]) -> list[dict[str, np.ndarray]]:
        self.request_index += 1
        request_seed = self.seed + self.request_index
        random.seed(request_seed)
        torch.manual_seed(request_seed)
        sample = self._build_sample(observation)
        with torch.inference_mode():
            generated = self.model.generate_samples_from_batch(
                _build_data_batch(sample),
                guidance=self.guidance,
                seed=[request_seed],
                num_steps=self.num_steps,
                n_sample=1,
            )
        actions = generated["action"][0].detach().cpu().numpy()[:, :16]
        return [action_vector_to_dict(action) for action in actions[: self.execute_steps]]

    def get_action(self):
        if self.observation is None:
            raise RuntimeError("update_obs must be called before get_action")
        return self._predict(self.observation)

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list) != len(self.observations):
            raise ValueError("env_idx_list and the latest observation batch have different sizes")
        return [self._predict(observation) for observation in self.observations]