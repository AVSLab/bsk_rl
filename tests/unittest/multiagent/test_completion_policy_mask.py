"""PPO logits obey completion and continuation masks in every forward mode."""

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.columns import Columns
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from bsk_rl.utils.rllib.target_gnn_module import GNNModule
from examples.multiagent_imaging.config import (
    MultiAgentImagingConfig,
    GLOBAL_FEATURES,
    TARGET_FEATURES,
)
from examples.multiagent_imaging.train import target_attention_config


def test_mask_applies_to_sampling_inference_and_training_logits():
    config = MultiAgentImagingConfig(n_candidates=3)
    size = GLOBAL_FEATURES + 3 * TARGET_FEATURES
    module = RLModuleSpec(
        module_class=GNNModule,
        catalog_class=PPOCatalog,
        observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(size,), dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(8),
        model_config_dict=target_attention_config(config),
    ).build()
    observations = torch.zeros((2, size))
    # Row zero can only choose the four operational actions. Row one can continue
    # or image slot one; invalid slots stay masked regardless of learned scores.
    observations[1, 25] = 1
    observations[1, GLOBAL_FEATURES + TARGET_FEATURES + 16] = 1
    for method in (
        module._forward_train,
        module._forward_inference,
        module._forward_exploration,
    ):
        output = method({Columns.OBS: observations})
        logits = output[Columns.ACTION_DIST_INPUTS]
        assert torch.all(logits[0, 4:] < -1e8)
        assert logits[1, 4] > -1e8 and logits[1, 6] > -1e8
        assert logits[1, 5] < -1e8 and logits[1, 7] < -1e8
        assert output[Columns.ACTIONS][0] < 4
        assert output[Columns.ACTIONS][1] not in [5, 7]
