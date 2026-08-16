from pathlib import Path

import pandas as pd
import pytest
import yaml

from examples.prospectus_rfi.collect_legacy_policy_mc import validate_campaign
from examples.prospectus_rfi.environment import legacy_amos2025_policy_contract
from examples.prospectus_rfi.legacy_policy_mc import task_spec
from examples.prospectus_rfi.legacy_policy_mc_design import (
    CATALOG_SIZES,
    EXPECTED_MODULE_STATE_SHA256,
    METHOD,
    POLICY_BEST_ITERATION,
    TOTAL_TASKS,
)


def complete_campaign_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "catalog_size": catalog_size,
                "scenario_seed": seed,
                "scenario_fingerprint": f"fingerprint-{catalog_size}-{seed}",
                "method": METHOD,
                "candidate_count": 10,
                "shield_enabled": True,
                "episode_duration_s": 45_000.0,
                "policy_best_iteration": POLICY_BEST_ITERATION,
                "policy_module_state_sha256": EXPECTED_MODULE_STATE_SHA256,
                "policy_training_imaging_duration_s": 300.0,
                "evaluation_imaging_duration_s": 100.0,
                "observation_contract": "amos2025_obs_v2_checkpoint_exact",
            }
            for catalog_size in CATALOG_SIZES
            for seed in range(100)
        ]
    )


def test_array_mapping_assigns_each_episode_to_an_independent_task() -> None:
    pairs = [
        (task_spec(task_id).catalog_size, task_spec(task_id).seed)
        for task_id in range(TOTAL_TASKS)
    ]
    assert len(pairs) == 300
    assert len(set(pairs)) == 300
    assert set(pairs) == {
        (catalog_size, seed) for catalog_size in (100, 200, 400) for seed in range(100)
    }


@pytest.mark.parametrize("task_id", [-1, TOTAL_TASKS])
def test_array_mapping_rejects_out_of_range_tasks(task_id: int) -> None:
    with pytest.raises(ValueError, match="task_id"):
        task_spec(task_id)


def test_archived_policy_contract_is_exactly_87_to_13() -> None:
    contract = legacy_amos2025_policy_contract()
    assert contract["flattened_observation_size"] == 87
    assert contract["action_count"] == 13
    assert contract["candidate_count"] == 10
    assert contract["field_order"] == [
        "spacecraft[5]",
        "target_rows[10,7]",
        "eclipse[2]",
        "ground_station_windows[5,2]",
    ]


def test_campaign_validator_accepts_exact_transfer_design() -> None:
    completion = validate_campaign(complete_campaign_frame())
    assert completion["complete"] is True
    assert completion["episode_count"] == 300


def test_campaign_validator_rejects_missing_seed() -> None:
    with pytest.raises(ValueError, match="missing pairs"):
        validate_campaign(complete_campaign_frame().iloc[:-1])


def test_machine_readable_transfer_config_matches_campaign() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "prospectus_rfi"
        / "configs"
        / "legacy_policy_mc_100_200_400.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    assert config["evaluation_catalog_sizes"] == [100, 200, 400]
    assert config["evaluation_seed_start"] == 0
    assert config["evaluation_seed_stop_inclusive"] == 99
    assert config["evaluation_total_episodes"] == 300
    assert config["slurm_tasks"] == 300
    assert config["episodes_per_task"] == 1
    assert config["dependencies"] == "none"
