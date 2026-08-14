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
