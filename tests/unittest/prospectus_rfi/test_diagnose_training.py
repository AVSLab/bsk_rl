import json

import pytest

from examples.prospectus_rfi.diagnose_training import diagnose_root, write_outputs


def test_diagnostic_uses_environment_steps_and_prefixed_physical_metrics(tmp_path):
    run_dir = tmp_path / "training" / "attention_k10_seed10001"
    run_dir.mkdir(parents=True)
    rows = []
    for iteration, (steps, result) in enumerate(
        [(4200, 80.0), (8400, 90.0), (12600, 100.0)], start=1
    ):
        rows.append(
            {
                "training_iteration": iteration,
                "environment_steps": steps,
                "wall_clock_h": float(iteration),
                "samples_per_second": 1.5,
                "episode_return_mean": result,
                "custom/episode_target_count": 150.0,
                "custom/successful_observation_fraction": 0.5 + iteration / 100,
                "custom/illuminated_observation_fraction": 0.4 + iteration / 100,
            }
        )
    (run_dir / "training_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    result = diagnose_root(tmp_path)[0]

    assert result.iterations == 3
    assert result.environment_steps == 12_600
    assert result.median_steps_per_iteration == 4200
    assert result.last_episode_target_count == 150.0
    assert result.last_successful_observation_fraction == pytest.approx(0.53)
    assert result.return_slope_per_10000_steps == pytest.approx(23.8095238)

    write_outputs([result], tmp_path / "analysis")
    assert (tmp_path / "analysis" / "training_diagnostic_summary.csv").is_file()
    assert (
        "PPO iteration is not a comparable unit"
        in (tmp_path / "analysis" / "TRAINING_DIAGNOSTIC.md").read_text()
    )
