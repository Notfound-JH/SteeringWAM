#!/usr/bin/env python3
"""Register a Cosmos3-Edge LIBERO experiment and run the framework trainer."""

from __future__ import annotations

import copy
import runpy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano import (
    action_policy_libero_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG


EXPERIMENT_NAME = "action_policy_libero_edge"


def build_edge_experiment():
    experiment = copy.deepcopy(action_policy_libero_nano)
    model_config = copy.deepcopy(EDGE_MODEL_CONFIG)

    model_config["max_num_tokens_after_packing"] = 74000
    model_config["activation_checkpointing"]["mode"] = "selective"
    model_config["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    model_config["rectified_flow_training_config"]["loss_scale"] = 10.0
    model_config["rectified_flow_training_config"]["image_loss_scale"] = None
    model_config["tokenizer"]["encode_exact_durations"] = [17, 61, 73]

    experiment["model"]["config"] = model_config
    experiment["job"]["name"] = EXPERIMENT_NAME

    optimizer_keys = experiment["optimizer"]["keys_to_select"]
    if "k_norm_und_for_gen" not in optimizer_keys:
        optimizer_keys.append("k_norm_und_for_gen")

    return experiment


def main() -> None:
    ConfigStore.instance().store(
        group="experiment",
        package="_global_",
        name=EXPERIMENT_NAME,
        node=build_edge_experiment(),
    )
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()