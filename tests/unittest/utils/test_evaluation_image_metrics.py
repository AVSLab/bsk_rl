import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))

from evaluation_image_metrics import (  # noqa: E402
    annotate_downlink_window_alignment,
    cooldown_diagnostic_title,
    cumulative_count_axis_limit,
    desat_availability_summary,
    decision_state_summary,
    ground_station_window_dict,
)


@pytest.mark.parametrize(
    ("cooldown_orbits", "expected"),
    [
        (2.0, None),
        (
            0.0,
            "GROUND-CONFIRMATION RE-IMAGING - seed 0, 100 targets; "
            "target availability and Desat decisions",
        ),
        (
            1.0,
            "ONE-ORBIT COOLDOWN ABLATION - seed 0, 100 targets; "
            "target availability and Desat decisions",
        ),
        (
            0.5,
            "0.5-ORBIT COOLDOWN SCENARIO - seed 0, 100 targets; "
            "target availability and Desat decisions",
        ),
    ],
)
def test_cooldown_diagnostic_title(cooldown_orbits, expected):
    assert (
        cooldown_diagnostic_title(cooldown_orbits, seed=0, target_count=100) == expected
    )


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        ([], 300.0),
        ([0, 299, 300], 300.0),
        ([0, 301], 400.0),
        ([0, 420], 500.0),
        ([0, 530], 600.0),
        ([math.nan, math.inf, 250], 300.0),
    ],
)
def test_cumulative_count_axis_limit(series, expected):
    assert cumulative_count_axis_limit(series) == expected


def test_cumulative_count_axis_limit_uses_all_plotted_series():
    assert cumulative_count_axis_limit([0, 280], [0, 420]) == 500.0


@pytest.mark.parametrize(
    ("minimum", "increment"),
    [(0.0, 100.0), (300.0, 0.0), (-1.0, 100.0), (300.0, -1.0)],
)
def test_cumulative_count_axis_limit_rejects_invalid_scale(minimum, increment):
    with pytest.raises(ValueError):
        cumulative_count_axis_limit([1], minimum=minimum, increment=increment)


def test_decision_state_summary_preserves_zero_action_metrics():
    summary = decision_state_summary(
        [
            {
                "action_category": "Desat",
                "eligible_target_count": 20,
                "imageable_eligible_count": 0,
                "wheel_speed_max_fraction_cmd": 0.03,
            },
            {
                "action_category": "Desat",
                "eligible_target_count": 24,
                "imageable_eligible_count": 2,
                "wheel_speed_max_fraction_cmd": 0.05,
            },
            {
                "action_category": "Imaging",
                "eligible_target_count": 90,
                "imageable_eligible_count": 8,
                "wheel_speed_max_fraction_cmd": 0.08,
            },
        ]
    )

    assert summary["Desat"]["decision_count"] == 2
    assert summary["Desat"]["eligible_target_count"]["median"] == 22.0
    assert summary["Desat"]["imageable_eligible_count"]["zero_fraction"] == 0.5
    assert summary["Imaging"]["decision_count"] == 1


def test_ground_station_windows_use_authoritative_opportunities():
    class Station:
        def __init__(self, name):
            self.ModelTag = name

    station = Station("Boulder")
    satellite = type(
        "Satellite",
        (),
        {
            "opportunities": [
                {"type": "ground_station", "object": station, "window": (10, 20)},
                {"type": "target", "object": object(), "window": (12, 18)},
                {"type": "ground_station", "object": station, "window": (20, 30)},
            ]
        },
    )()

    assert ground_station_window_dict(satellite) == {"Boulder": [(10.0, 30.0)]}


def test_downlink_alignment_detects_action_window_overlap():
    rows = annotate_downlink_window_alignment(
        [
            {
                "action_category": "Downlink",
                "t_cmd": 100.0,
                "t_after": 280.0,
                "storage_frac_cmd": 0.5,
                "storage_frac_after": 0.3,
                "useful_downlinks_cmd": 4,
                "useful_downlinks_after": 6,
            }
        ],
        {"Hawaii": [(250.0, 320.0)]},
    )

    assert rows[0]["ground_station_overlap_sec"] == 30.0
    assert not rows[0]["starts_in_ground_station_window"]
    assert rows[0]["ends_in_ground_station_window"]
    assert rows[0]["useful_deliveries_during_action"] == 2


def test_desat_availability_summary():
    summary = desat_availability_summary(
        [
            {
                "action_category": "Desat",
                "imageable_eligible_count": 0,
                "wheel_speed_max_fraction_cmd": 0.02,
                "sat_shadow_cmd": 0.0,
            },
            {
                "action_category": "Desat",
                "imageable_eligible_count": 2,
                "wheel_speed_max_fraction_cmd": 0.04,
                "sat_shadow_cmd": 1.0,
            },
        ]
    )

    assert summary["desat_decision_count"] == 2
    assert summary["desat_with_zero_imageable"] == 1
    assert summary["desat_with_at_most_three_imageable"] == 2
    assert summary["desat_in_observer_umbra"] == 1
