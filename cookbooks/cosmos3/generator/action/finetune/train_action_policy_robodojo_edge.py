#!/usr/bin/env python3
"""Register a Cosmos3-Edge RoboDojo action-policy experiment."""

from __future__ import annotations

import copy
import runpy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
    get_action_robodojo_sft_dataset,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.utils.lazy_config import LazyCall as L


EXPERIMENT_NAME = "action_policy_robodojo_edge"


def build_edge_experiment():
    experiment = copy.deepcopy(action_policy_droid_nano)
    model_config = copy.deepcopy(EDGE_MODEL_CONFIG)

    model_config["max_num_tokens_after_packing"] = 74000
    model_config["num_embodiment_domains"] = 33
    if get_domain_id("robodojo-ee") >= model_config["num_embodiment_domains"]:
        raise ValueError("robodojo-ee domain ID exceeds the model projector capacity")
    model_config["activation_checkpointing"]["mode"] = "selective"
    model_config["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    model_config["rectified_flow_training_config"]["loss_scale"] = 10.0
    model_config["rectified_flow_training_config"]["image_loss_scale"] = None
    model_config["tokenizer"]["encode_exact_durations"] = [17]

    experiment["model"]["config"] = model_config
    experiment["job"]["name"] = EXPERIMENT_NAME
    experiment["dataloader_train"]["dataset_name"] = "action_robodojo"
    experiment["dataloader_train"]["dataloader"]["datasets"] = {
        "robodojo": {
            "ratio": 1,
            "dataset": L(get_action_robodojo_sft_dataset)(
                root="${oc.env:ROBODOJO_ROOT}",
                fps=25.0,
                chunk_length=16,
                mode="wam",
                viewpoint="concat_view",
                sample_stride=1,
                resolution="480",
                max_action_dim="${model.config.max_action_dim}",
                cfg_dropout_rate=0.1,
                tokenizer_config="${model.config.vlm_config.tokenizer}",
                format_prompt_as_json=True,
                iterable_shuffle=True,
                episode_shuffle_seed=42,
            ),
        },
    }

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