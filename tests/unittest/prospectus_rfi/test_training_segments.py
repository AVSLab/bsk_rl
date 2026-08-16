import json

import pandas as pd
import pytest

from examples.prospectus_rfi.train import (
    _load_prior_progress,
    _segment_budget_seconds,
)


@pytest.mark.parametrize(
    ("prior_hours", "segment_hours", "expected_hours"),
    [
        (0.0, 23.5, 23.5),
        (23.5, 23.5, 23.5),
        (47.0, 1.5, 1.0),
        (48.0, 1.5, 0.0),
    ],
)
def test_segment_budget_respects_cumulative_48_hour_target(
    prior_hours: float, segment_hours: float, expected_hours: float
) -> None:
    budget = _segment_budget_seconds(
        target_wall_hours=48.0,
        prior_wall_seconds=prior_hours * 3600.0,
        segment_wall_hours=segment_hours,
    )
    assert budget == pytest.approx(expected_hours * 3600.0)


def test_segment_budget_rejects_nonpositive_cap() -> None:
    with pytest.raises(ValueError, match="positive"):
        _segment_budget_seconds(
            target_wall_hours=48.0,
            prior_wall_seconds=0.0,
            segment_wall_hours=0.0,
        )


def test_prior_progress_uses_latest_persisted_counters(tmp_path) -> None:
    status = {
        "cumulative_wall_clock_s": 84_600.0,
        "environment_steps": 20_000,
        "training_iteration": 4,
        "segments": [{"segment_index": 0}],
    }
    (tmp_path / "status.json").write_text(json.dumps(status))
    pd.DataFrame(
        [
            {
                "training_iteration": 5,
                "environment_steps": 25_000,
                "wall_clock_s": 85_000.0,
            }
        ]
    ).to_csv(tmp_path / "training_metrics.csv", index=False)

    progress = _load_prior_progress(tmp_path)

    assert progress["wall_clock_s"] == 85_000.0
    assert progress["environment_steps"] == 25_000
    assert progress["training_iteration"] == 5
    assert progress["segments"] == [{"segment_index": 0}]


def test_empty_run_has_zero_prior_progress(tmp_path) -> None:
    assert _load_prior_progress(tmp_path) == {
        "wall_clock_s": 0.0,
        "environment_steps": 0,
        "training_iteration": 0,
        "segments": [],
    }
