from pathlib import Path

import pandas as pd
import pytest
import yaml

from examples.prospectus_rfi.collect_heuristic_mc import validate_campaign
from examples.prospectus_rfi.heuristic_mc import (
    task_spec,
)
from examples.prospectus_rfi.heuristic_mc_independent import (
    TOTAL_INDEPENDENT_TASKS,
    independent_task_spec,
)
from examples.prospectus_rfi.heuristic_mc_design import CATALOG_SIZES, TOTAL_TASKS


def complete_campaign_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "catalog_size": catalog_size,
                "scenario_seed": seed,
                "method": "heuristic_historical",
                "heuristic_mode": "angle",
                "information_scope": "full_visible_eligible_catalog",
                "candidate_count": 10,
                "shield_enabled": True,
                "episode_duration_s": 45_000.0,
            }
            for catalog_size in CATALOG_SIZES
            for seed in range(100)
        ]
    )


def test_array_mapping_covers_each_requested_seed_once() -> None:
    pairs = [
        (task.catalog_size, seed)
        for task_id in range(TOTAL_TASKS)
        for task in [task_spec(task_id)]
        for seed in task.seeds
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


def test_campaign_validator_accepts_exact_design() -> None:
    result = validate_campaign(complete_campaign_frame())
    assert result["complete"] is True
    assert result["episode_count"] == 300


def test_campaign_validator_rejects_missing_seed() -> None:
    with pytest.raises(ValueError, match="missing pairs"):
        validate_campaign(complete_campaign_frame().iloc[:-1])


def test_machine_readable_campaign_config_matches_array_design() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "prospectus_rfi"
        / "configs"
        / "heuristic_mc_100_200_400.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    assert config["catalog_sizes"] == [100, 200, 400]
    assert config["seed_start"] == 0
    assert config["seed_stop_inclusive"] == 99
    assert config["total_episodes"] == 300
    assert config["dependencies"] == "none"


def test_independent_recovery_mapping_covers_every_pair_once() -> None:
    pairs = [
        (
            independent_task_spec(task_id).catalog_size,
            independent_task_spec(task_id).seed,
        )
        for task_id in range(TOTAL_INDEPENDENT_TASKS)
    ]
    assert len(pairs) == 300
    assert len(set(pairs)) == 300
    assert set(pairs) == {
        (catalog_size, seed) for catalog_size in (100, 200, 400) for seed in range(100)
    }


def test_recovery_submitter_uses_explicit_study_python() -> None:
    submitter = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "prospectus_rfi"
        / "submit_missing_amos2025_heuristic_mc.sh"
    ).read_text()
    assert 'PYTHON="$VENV_ROOT/bin/python"' in submitter
    assert 'MISSING_OUTPUT=$(' in submitter
    assert "mapfile" not in submitter
