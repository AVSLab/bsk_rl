"""Metamorphic checks: padding and candidate/peer order are representation choices."""

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from bsk_rl.utils.rllib.target_gnn_module import GNNModule
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.train import target_attention_config


def module_and_obs():
    cfg = MultiAgentImagingConfig(
        n_sensors=3,
        n_candidates=3,
        communication_mode="directed",
        information_case="completion",
    )
    size = 26 + 3 * 17 + 2 * 12
    torch.manual_seed(72)
    module = RLModuleSpec(
        module_class=GNNModule,
        catalog_class=PPOCatalog,
        observation_space=gym.spaces.Box(-np.inf, np.inf, (size,), np.float32),
        action_space=gym.spaces.Discrete(10),
        model_config_dict=target_attention_config(cfg),
    ).build()
    obs = torch.randn(2, size)
    obs[:, 25] = 1
    obs[:, 26:77].reshape(2, 3, 17)[:, :, -1] = torch.tensor([1, 0, 1])
    obs[:, 77:].reshape(2, 2, 12)[:, :, -1] = 1
    return module, obs


def logits(module, obs):
    return module._forward_inference({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS]


def test_padding_contents_cannot_change_valid_logits_or_critic():
    module, obs = module_and_obs()
    changed = obs.clone()
    changed[:, 43:59] = 10000 * torch.randn(2, 16)
    torch.testing.assert_close(logits(module, obs), logits(module, changed))
    torch.testing.assert_close(module.vf(obs), module.vf(changed))


def test_target_and_peer_permutation_equivariance():
    module, obs = module_and_obs()
    changed = obs.clone()
    changed[:, 26:77] = obs[:, 26:77].reshape(2, 3, 17)[:, [2, 0, 1]].reshape(2, -1)
    changed[:, 77:] = obs[:, 77:].reshape(2, 2, 12)[:, [1, 0]].reshape(2, -1)
    permutation = [0, 1, 2, 3, 4, 7, 5, 6, 9, 8]
    torch.testing.assert_close(
        logits(module, changed),
        logits(module, obs)[:, permutation],
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(module.vf(obs), module.vf(changed), atol=1e-6, rtol=1e-5)


def test_empty_sets_keep_resource_sensitive_operations_and_finite_gradients():
    module, obs = module_and_obs()
    obs[:] = 0
    obs[1, 1] = 0.9  # Battery state remains relevant without any target or peer.
    out = module._forward_train({Columns.OBS: obs})
    values = out[Columns.ACTION_DIST_INPUTS]
    assert torch.isfinite(values).all()
    assert torch.all(values[:, 3:] < -1e8)
    assert not torch.allclose(values[0, :3], values[1, :3])
    loss = values[:, :3].square().mean() + out[Columns.VF_PREDS].square().mean()
    loss.backward()
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
