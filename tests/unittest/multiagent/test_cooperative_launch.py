"""Submission preparation cannot spend compute or omit resource budgets."""

import json
from pathlib import Path

import pytest

from examples.multiagent_imaging.cluster.cooperative_launch import command


@pytest.fixture
def release(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(json.dumps(dict(account="ucb550_asc3", commit="abc123")))
    return path


def test_canary_plan_is_explicit_bounded_and_does_not_submit(release, tmp_path):
    plan = command(release, "canary", tmp_path / "out")
    assert not plan["submitted"]
    assert plan["allocated_cpu_hours"] == 84
    for option in (
        "--account=ucb550_asc3",
        "--partition=acpu",
        "--qos=cpu-normal",
        "--cpus-per-task=56",
        "--mem=200G",
        "--no-requeue",
        "--time=90",
    ):
        assert option in plan["argv"]
    assert plan["argv"][-3:] == ["2", "56", "120000"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(wall_minutes=1500, cpu_hour_budget=5000),
        dict(
            wall_minutes=60,
            cpu_hour_budget=56,
            iterations=10,
            episodes=100,
            decisions=100000,
        ),
        dict(
            wall_minutes=60,
            cpu_hour_budget=112,
            iterations=0,
            episodes=100,
            decisions=100000,
        ),
    ],
)
def test_training_requires_reviewed_pair_budget_and_sampling_caps(
    release, tmp_path, kwargs
):
    with pytest.raises(ValueError):
        command(release, "train", tmp_path / "out", **kwargs)


def test_pair_changes_only_the_declared_information_treatment(release, tmp_path):
    common = dict(
        wall_minutes=60,
        cpu_hour_budget=112,
        iterations=10,
        episodes=100,
        decisions=100000,
    )
    a = command(release, "train", tmp_path / "a", arm="catalog_tx", **common)
    b = command(release, "train", tmp_path / "b", arm="geometry_only", **common)
    assert a["allocated_cpu_hours"] == b["allocated_cpu_hours"] == 56
    assert a["pair_allocated_cpu_hours"] == 112
    configs = Path(__file__).parents[3] / "examples/multiagent_imaging/configs"
    for variant in ("shared", "owned", "delivery"):
        left = json.loads(
            (configs / f"cooperative_v1_{variant}_catalog_tx.json").read_text()
        )
        right = json.loads(
            (configs / f"cooperative_v1_{variant}_geometry_only.json").read_text()
        )
        assert left.pop("information_case") == "catalog_tx"
        assert right.pop("information_case") == "geometry_only"
        assert left == right


def test_evaluation_requires_explicit_unique_seeds_and_checkpoints(release, tmp_path):
    with pytest.raises(ValueError):
        command(
            release, "evaluate", tmp_path / "out", wall_minutes=60, cpu_hour_budget=56
        )
    plan = command(
        release,
        "evaluate",
        tmp_path / "out",
        wall_minutes=60,
        cpu_hour_budget=56,
        checkpoints=["initial=/checkpoint_0000", "trained=/checkpoint_0002"],
        seeds=[90001, 90002],
    )
    assert "seed=90001" in plan["argv"] and not plan["submitted"]


def test_foreign_account_fails_before_printing_sbatch(release, tmp_path):
    release.write_text(json.dumps(dict(account="ucb550_asc2", commit="abc123")))
    with pytest.raises(ValueError, match="Allocation mismatch"):
        command(release, "canary", tmp_path / "out")


def test_evaluation_logging_keeps_argmax_stochastic_and_training_experience_separate(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from examples.multiagent_imaging import cooperative_eval_logging as logging

    heldout = tmp_path / "heldout"
    heldout.mkdir()
    preflight = tmp_path / "preflight"
    preflight.mkdir()
    (preflight / "source-gate.json").write_text('{"passed": true}')
    rows = []
    for mode in ("deterministic", "stochastic"):
        rows.append(
            dict(
                policy="trained",
                mode=mode,
                seed=90001,
                capture_coverage=0.2,
                ground_coverage=0.1,
                useful_acquisitions=4,
                useful_delivered_visits=2,
                duplicate_attempts=1,
                rollout_wall_s=3,
                cooperative_tasking=dict(
                    unshaped_weighted_science=5, redundant_capture_cost=0.1
                ),
            )
        )
    path = heldout / "summary.json"
    path.write_text(
        json.dumps(
            dict(
                complete=True,
                initial_conditions_matched=True,
                expected_episodes=2,
                completed_episodes=2,
                rows=rows,
            )
        )
    )
    logs, finished, configurations = [], [], []

    def logger(output, config, **kwargs):
        assert kwargs == dict(mode="online", required=True)
        configurations.append(config)
        return SimpleNamespace(
            _log=logs.append, finish=lambda **kw: finished.append(kw)
        )

    monkeypatch.setattr(logging, "TrainingLogger", logger)
    logging.publish(
        tmp_path,
        SimpleNamespace(to_dict=lambda: dict(experiment_profile="cooperative-v1")),
    )
    assert len(logs) == 2 and finished == [dict(failed=False)]
    assert (
        logs[0]["heldout/trained/deterministic/requests/unshaped_weighted_science"] == 5
    )
    assert logs[1]["heldout/trained/stochastic/requests/unshaped_weighted_science"] == 5
    assert not any("lifetime/true_decisions" in log for log in logs)
    path.write_text(
        json.dumps(
            dict(
                complete=False,
                initial_conditions_matched=True,
                expected_episodes=2,
                completed_episodes=1,
                rows=rows[:1],
            )
        )
    )
    with pytest.raises(ValueError, match="complete"):
        logging.publish(tmp_path, SimpleNamespace())
