from pathlib import Path

from examples.prospectus_rfi.config import load_study_config
from examples.prospectus_rfi.environment import environment_contract


CONFIG_DIR = Path(__file__).parents[3] / "examples" / "prospectus_rfi" / "configs"


def test_selected_configs_preserve_matched_environment():
    mlp = load_study_config(CONFIG_DIR / "mlp_selected.yaml", CONFIG_DIR / "base.yaml")
    attention = load_study_config(
        CONFIG_DIR / "attention_selected.yaml", CONFIG_DIR / "base.yaml"
    )

    assert mlp.environment == attention.environment
    assert mlp.environment.episode_duration_s == 45_000.0
    assert mlp.environment.imaging_duration_s == 100.0
    assert mlp.environment.catalog_min == 100
    assert mlp.environment.catalog_max == 400
    assert mlp.environment.candidate_count == 10
    assert mlp.environment.initial_battery_fraction_min == 0.20
    assert mlp.environment.initial_battery_fraction_max == 0.60


def test_alpha_zero_is_observation_only_and_not_alphazero_algorithm():
    config = load_study_config(
        CONFIG_DIR / "mlp_selected.yaml", CONFIG_DIR / "base.yaml"
    )
    contract = environment_contract(config.environment)

    assert config.environment.alpha == 0.0
    assert config.environment.imaging_bonus == 1.0
    assert config.environment.downlink_bonus == 0.0
    assert "not AlphaZero" in contract["alpha_interpretation"]


def test_observation_and_action_contract_dimensions():
    config = load_study_config(
        CONFIG_DIR / "mlp_selected.yaml", CONFIG_DIR / "base.yaml"
    )
    contract = environment_contract(config.environment)

    assert contract["observation"]["global_features"] == 11
    assert contract["observation"]["target_slots"] == 10
    assert contract["observation"]["flattened_size"] == 91
    assert contract["action"]["total_actions"] == 13


def test_memorysafe_v2_config_limits_training_catalog_without_changing_actions():
    mlp = load_study_config(
        CONFIG_DIR / "mlp_selected.yaml",
        CONFIG_DIR / "base_memorysafe_100_200.yaml",
    )
    attention = load_study_config(
        CONFIG_DIR / "attention_selected.yaml",
        CONFIG_DIR / "base_memorysafe_100_200.yaml",
    )

    assert mlp.environment == attention.environment
    assert mlp.environment.catalog_min == 100
    assert mlp.environment.catalog_max == 200
    assert mlp.validation.catalog_sizes == (100, 150, 200)
    assert mlp.compute.evaluation_catalog_sizes == (100, 200, 300, 400)
    assert mlp.environment.episode_duration_s == 45_000.0
    assert mlp.environment.imaging_duration_s == 100.0
    assert mlp.environment.alpha == 0.0
    assert mlp.environment.action_count == 13


def test_amos2025_attention_control_restores_checkpoint_physical_regime():
    control = load_study_config(
        CONFIG_DIR / "attention_amos2025_control.yaml",
        CONFIG_DIR / "base_amos2025_attention_control.yaml",
    )
    env = control.environment
    contract = environment_contract(env)

    assert env.profile == "amos2025_checkpoint_attention_control"
    assert (env.catalog_min, env.catalog_max, env.candidate_count) == (100, 100, 10)
    assert env.imaging_duration_s == 300.0
    assert env.charge_duration_s == 300.0
    assert env.downlink_duration_s == 180.0
    assert env.desaturation_duration_s == 150.0
    assert (env.initial_battery_fraction_min, env.initial_battery_fraction_max) == (
        0.10,
        0.40,
    )
    assert control.ppo.train_batch_size == 180
    assert control.ppo.ppo_epochs == 10
    assert control.ppo.learning_rate == 1.0e-6
    assert contract["observation"]["flattened_size"] == 97
    assert contract["observation"]["target_start"] == 5
