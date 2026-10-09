"""Keep mission metrics and checkpoint run identity intact without network use."""

import json
import sys
from types import SimpleNamespace

import pytest

from examples.multiagent_imaging.wandb_logging import TrainingLogger


def test_required_logging_cannot_be_disabled(tmp_path):
    with pytest.raises(ValueError, match="cannot be disabled"):
        TrainingLogger(tmp_path, {}, required=True)


def test_nested_mission_metrics_and_same_run_resume(tmp_path, monkeypatch):
    calls, logged = [], []
    run = SimpleNamespace(
        id="test-id",
        project="test-project",
        entity="test-entity",
        url="https://wandb.ai/test-entity/test-project/runs/test-id",
        log=logged.append,
    )
    monkeypatch.delenv("BSK_RL_WANDB_KEY_PATH", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=lambda **kwargs: calls.append(kwargs) or run,
        ),
    )
    logger = TrainingLogger(tmp_path, {}, mode="online", required=True)
    logger.log_update(
        dict(
            iteration=1,
            policy_decisions_lifetime=42,
            complete_episode_count=1,
            gradient_l2=0.5,
            parameter_change={"l2": 0.1},
            losses={},
            learner_metrics={"vf_clamp_fraction": 0.0},
            measurement={},
            episodes=[
                dict(
                    constellation_reward=3,
                    coverage={
                        "capture_coverage_fraction": 0.5,
                        "ground_delivery_coverage_fraction": 0.25,
                    },
                    team_summary={
                        "unique_acquisition_count": 8,
                        "unique_service_count": 4,
                        "duplicate_attempt_count": 2,
                    },
                    useful_post_cooldown_revisits={"count": 1},
                    communication_summary={"radio_occupancy_s": 10},
                    action_selection_counts={"sensor_0": {"image": 5, "charge": 2}},
                )
            ],
        )
    )
    assert logged[0]["episode/ground_delivery_coverage_fraction_mean"] == 0.25
    assert logged[0]["episode/useful_delivered_visits_mean"] == 4
    assert logged[0]["actions/sensor_0/image_mean"] == 5
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    logger.save_identity(checkpoint)
    assert json.loads((checkpoint / "wandb.json").read_text())["id"] == "test-id"
    TrainingLogger(tmp_path, {}, mode="online", resume=checkpoint, required=True)
    assert calls[1]["id"] == "test-id"
    assert calls[1]["resume"] == "must"
    assert calls[1]["project"] == "test-project"
