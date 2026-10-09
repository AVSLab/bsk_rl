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
from examples.multiagent_imaging.training_audit import initial_policy_gate


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
    observations[:, 1] = 0.9
    # Row zero can choose only meaningful operational actions. Row one can also
    # continue or image slot one; invalid slots stay masked regardless of scores.
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


def test_resource_mask_blocks_empty_downlink_and_new_low_battery_tasks():
    config = MultiAgentImagingConfig(n_sensors=3, n_candidates=2)
    size = GLOBAL_FEATURES + 2 * TARGET_FEATURES
    module = RLModuleSpec(
        module_class=GNNModule,
        catalog_class=PPOCatalog,
        observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(size,), dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(7),
        model_config_dict=target_attention_config(config),
    ).build()
    observations = torch.zeros((3, size))
    observations[:, GLOBAL_FEATURES:].reshape(3, 2, TARGET_FEATURES)[:, :, -1] = 1
    observations[0, 1] = 0.9  # Empty storage, otherwise healthy.
    observations[1, 0] = 0.2  # A local product makes downlink meaningful.
    observations[1, 1] = 0.9
    observations[2, 0] = 0.2  # Low battery blocks every new powered task.
    observations[2, 1] = config.minimum_task_battery_fraction
    observations[2, 25] = 1  # Continuing an already active task remains possible.

    logits = module._forward_inference({Columns.OBS: observations})[
        Columns.ACTION_DIST_INPUTS
    ]
    assert logits[0, 1] < -1e8
    assert torch.all(logits[0, [0, 2, 5, 6]] > -1e8)
    assert torch.all(logits[1, [0, 1, 2, 5, 6]] > -1e8)
    assert torch.all(logits[2, [1, 2, 3, 5, 6]] < -1e8)
    assert torch.all(logits[2, [0, 4]] > -1e8)


def test_initial_valid_action_probabilities_share_one_calibrated_scale():
    config = MultiAgentImagingConfig(n_sensors=4, n_targets=10, n_candidates=10)
    size = GLOBAL_FEATURES + 10 * TARGET_FEATURES
    torch.manual_seed(0)
    module = RLModuleSpec(
        module_class=GNNModule,
        catalog_class=PPOCatalog,
        observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(size,), dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(15),
        model_config_dict=target_attention_config(config),
    ).build()
    observations = torch.zeros((4, size))
    observations[:, 1] = 0.9
    observations[:, GLOBAL_FEATURES:].reshape(4, 10, TARGET_FEATURES)[:, :, -1] = 1
    logits = module._forward_inference({Columns.OBS: observations})[
        Columns.ACTION_DIST_INPUTS
    ]
    probabilities = torch.softmax(logits, dim=-1)
    valid = logits > -1e8
    for row, row_valid in zip(probabilities, valid):
        values = row[row_valid]
        entropy = -(values * values.log()).sum() / torch.log(
            torch.tensor(float(len(values)))
        )
        assert entropy > 0.999
        assert values.max() / values.min() < 1.10


def test_initial_gate_audits_product_and_all_directed_peer_choices():
    config = MultiAgentImagingConfig(
        n_sensors=4,
        n_targets=10,
        n_candidates=10,
        information_case="completion",
        communication_mode="directed",
    )
    size = GLOBAL_FEATURES + 10 * TARGET_FEATURES + config.n_peers * 12
    module = RLModuleSpec(
        module_class=GNNModule,
        catalog_class=PPOCatalog,
        observation_space=gym.spaces.Box(
            -np.inf, np.inf, shape=(size,), dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(18),
        model_config_dict=target_attention_config(config),
    ).build()
    observations = np.zeros((4, size), dtype=np.float32)
    observations[:, 1] = 0.9
    observations[:, GLOBAL_FEATURES : GLOBAL_FEATURES + 10 * TARGET_FEATURES].reshape(
        4, 10, TARGET_FEATURES
    )[:, :, -1] = 1

    result = initial_policy_gate(module, observations, config)

    assert result["passed"]
    for row in result["filled_and_connected"]["rows"]:
        assert row["valid_action_labels"] == [
            "charge",
            "downlink",
            "desat",
            *[f"image_{index}" for index in range(10)],
            "transmit_0",
            "transmit_1",
            "transmit_2",
        ]
