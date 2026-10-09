"""Optimizer settings must survive configuration and restore without misreporting."""

import pytest
import torch

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.optimization import PPOSettings, actual_ppo_settings
from examples.multiagent_imaging.train import build_ppo_config


def test_profiles_preserve_physical_time_and_only_declared_optimizer_changes():
    mission = MultiAgentImagingConfig()
    old = build_ppo_config(
        mission,
        train_batch_size=256,
        ppo_settings=PPOSettings.from_profile("legacy_pilot"),
    )
    cap = build_ppo_config(
        mission,
        train_batch_size=256,
        ppo_settings=PPOSettings.from_profile("value_repair"),
    )
    recovered = build_ppo_config(
        mission,
        train_batch_size=256,
        ppo_settings=PPOSettings.from_profile("learning_recovery"),
    )
    for config in (old, cap, recovered):
        config.validate()
        assert config.gamma == mission.discount_per_s
        assert config.env_config == old.env_config
        assert config.learner_config_dict == old.learner_config_dict
    a, b = actual_ppo_settings(old), actual_ppo_settings(cap)
    assert {key for key in a if a[key] != b[key]} == {"vf_clip_param"}
    assert recovered.num_sgd_iter == 10
    # Reproduce the installed Ray squared-residual cap's missing gradient.
    value = torch.tensor(0.0, requires_grad=True)
    torch.clamp((value - 8).square(), 0, old.vf_clip_param).backward()
    assert value.grad == 0
    value.grad = None
    torch.clamp((value - 8).square(), 0, cap.vf_clip_param).backward()
    assert value.grad == -16


@pytest.mark.parametrize(
    "field", ["learning_rate", "vf_clip_param", "entropy_coeff", "vf_loss_coeff"]
)
def test_nonfinite_optimizer_settings_are_rejected(field):
    with pytest.raises(ValueError, match="finite"):
        PPOSettings(**{field: float("nan")})


def test_selected_amos_profile_matches_optimizer_trace_and_model_capacity():
    from pathlib import Path
    from examples.multiagent_imaging.train import target_attention_config

    mission = MultiAgentImagingConfig.from_json(
        Path(__file__).resolve().parents[3]
        / "examples/multiagent_imaging/configs/walker4_mixed_amos_best_conflict.json"
    )
    ppo = build_ppo_config(
        mission,
        train_batch_size=4200,
        ppo_settings=PPOSettings.from_profile("amos_best"),
    )
    ppo.validate()
    assert ppo.lr == [[0, 1e-5], [40000, 5.970334799302134e-7]]
    assert ppo.num_sgd_iter == 30 and ppo.sgd_minibatch_size == 128
    assert ppo.gamma == 0.9915045428565076
    assert ppo.lambda_ == 0.8713548569911232
    assert ppo.learner_config_dict["lambda_time"] == "step"
    assert ppo.clip_param == 0.14701727973480344 and not ppo.use_kl_loss
    assert ppo.entropy_coeff == 0.023694512589767867
    assert ppo.grad_clip == 0.3104924935285628
    assert ppo.vf_clip_param == 10.0 and ppo.vf_loss_coeff == 1.0
    model = target_attention_config(mission)
    assert model["tgt_encoded_dim"] == model["attention_dim"] == 128
    assert model["width_f"] == model["critic_width_f"] == 256
    assert model["dropout"] == 0.1
    # Historical default configs/checkpoint schemas must remain byte-equivalent.
    old = MultiAgentImagingConfig().to_dict()
    assert "gae_lambda_per_decision" not in old and "policy_model_profile" not in old


def test_invalid_amos_schedule_and_per_decision_trace_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        PPOSettings(learning_rate=[[0, 1e-5], [40000, float("nan")]])
    with pytest.raises(ValueError, match="ordered"):
        PPOSettings(learning_rate=[[0, 1e-5], [0, 1e-6]])
    with pytest.raises(ValueError, match="finite"):
        MultiAgentImagingConfig(gae_lambda_per_decision=float("nan"))
