"""Pure campaign/controller invariants, with no cluster or long episode execution."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.multiagent_imaging.baseline_monte_carlo import (
    BaselineConfig,
    CELLS,
    choose_joint,
    coverage_metrics,
    duplicate_product_metrics,
    exact_regimes,
    sample_orbit,
    task_spec,
)
from examples.multiagent_imaging.environment import R_EARTH_M


def test_campaign_is_exactly_four_cells_with_fifty_matched_seeds():
    tasks = [task_spec(i) for i in range(200)]
    assert len({(x["case"], x["target_environment"], x["seed"]) for x in tasks}) == 200
    for case, environment in CELLS:
        assert [
            x["seed"]
            for x in tasks
            if x["case"] == case and x["target_environment"] == environment
        ] == list(range(50))
    for invalid in (-1, 200, 1.5, True):
        with pytest.raises(ValueError):
            task_spec(invalid)


def test_exact_amos_mix_and_repeatable_orbits():
    names = exact_regimes(100, "mixed", 31)
    assert [names.count(x) for x in ("LEO", "MEO", "GEO")] == [50, 30, 20]
    assert names == exact_regimes(100, "mixed", 31)
    assert names != exact_regimes(100, "mixed", 32)
    for regime in ("LEO", "MEO", "GEO"):
        np.random.seed(4)
        first = sample_orbit(regime)
        np.random.seed(4)
        second = sample_orbit(regime)
        assert first.a * (1 - first.e) >= R_EARTH_M + 400e3
        assert [getattr(first, p) for p in ("a", "e", "i", "Omega", "omega", "f")] == [
            getattr(second, p) for p in ("a", "e", "i", "Omega", "omega", "f")
        ]


def choice(action, target, angle, new=1):
    return dict(action=action, target_id=target, angle=angle, new=new, priority=1.0)


def test_central_assignment_excludes_duplicate_and_in_progress_targets():
    options = {
        "sensor_0": [choice(5, 0, 1), choice(6, 1, 3), choice(0, None, 0, 0)],
        "sensor_1": [choice(5, 0, 1), choice(6, 2, 20), choice(0, None, 0, 0)],
    }
    # Globally lower combined angle assigns target1 to sensor0 and target0 to1.
    assert choose_joint(options) == {"sensor_0": 6, "sensor_1": 5}
    assert choose_joint(options, reserved={0}) == {"sensor_0": 6, "sensor_1": 6}
    assert choose_joint(options, reserved={0, 1, 2}) == {"sensor_0": 0, "sensor_1": 0}


def test_central_assignment_generalizes_to_three_sensors():
    options = {
        "sensor_0": [choice(5, 0, 2), choice(6, 1, 3), choice(0, None, 0, 0)],
        "sensor_1": [choice(5, 0, 1), choice(6, 2, 4), choice(0, None, 0, 0)],
        "sensor_2": [choice(5, 1, 1), choice(6, 2, 2), choice(0, None, 0, 0)],
    }
    assert choose_joint(options) == {"sensor_0": 6, "sensor_1": 5, "sensor_2": 6}


def test_coverage_uses_catalog_union_not_service_or_sensor_count():
    def product(target, source, quality=1):
        return SimpleNamespace(
            target_id=target, source_sensor=source, quality=quality, delivery_time=20
        )

    captures = [
        product(0, "sensor_0"),
        product(0, "sensor_1"),
        product(1, "sensor_1"),
        product(1, "sensor_2"),
        product(2, "sensor_0", 0),
    ]
    services = [
        SimpleNamespace(product=captures[0]),
        SimpleNamespace(product=captures[1]),
    ]
    result = coverage_metrics(
        range(4), captures, services, 0.5, ["sensor_0", "sensor_1", "sensor_2"]
    )
    assert result["capture_coverage_fraction"] == 0.5
    assert result["ground_delivery_coverage_fraction"] == 0.25
    assert result["qualified_exposure_count"] == 4
    assert result["unqualified_exposure_count"] == 1
    assert result["cross_sensor_capture_overlap_count"] == 2
    assert result["never_captured_target_ids"] == [2, 3]


def test_duplicate_products_separate_ground_staleness_from_onboard_overlap():
    def product(record_id, target, source, capture, delivery=None):
        return SimpleNamespace(
            record_id=record_id,
            target_id=target,
            source_sensor=source,
            capture_time=capture,
            delivery_time=delivery,
            quality=1.0,
        )

    captures = [
        product("old", 0, "sensor_0", 10),
        product("new", 0, "sensor_1", 20),
        product("held-a", 1, "sensor_2", 30),
        product("held-b", 1, "sensor_0", 40),
    ]
    services = [
        SimpleNamespace(product=product("old", 0, "sensor_0", 10, 100)),
        SimpleNamespace(product=product("new", 0, "sensor_1", 20, 80)),
    ]
    result = duplicate_product_metrics(captures, services, 0.5, 200)
    assert result["stale_cross_sensor_ground_delivery_count"] == 1
    assert result["stale_cross_sensor_ground_delivery_target_count"] == 1
    assert result["causally_avoidable_stale_ground_delivery_count"] == 1
    assert result["cross_sensor_onboard_overlap_target_count"] == 2
    assert result["cross_sensor_onboard_overlap_product_count"] == 4
    assert result["cross_sensor_onboard_redundant_acquisition_count"] == 2
    assert result["cross_sensor_onboard_redundant_sensor_time_s"] == 220
    assert result["cross_sensor_onboard_overlap_sensor_time_s"] == 440


def test_campaign_never_changes_the_checkpoint_config_schema():
    config = BaselineConfig()
    independent = config.environment_config("independent", 3)
    centralized = config.environment_config("centralized_full_state", 3)
    assert independent.n_sensors == centralized.n_sensors == 3
    assert independent.n_peers == centralized.n_peers == 0
    assert independent.episode_duration_s == 45000
    assert replace(independent, information_case="ideal_completion") == centralized


def test_source_fingerprint_tracks_inputs_but_allows_execution_report_updates(
    tmp_path, monkeypatch
):
    import subprocess
    from examples.multiagent_imaging import baseline_monte_carlo as baseline

    example = tmp_path / "examples/multiagent_imaging"
    example.mkdir(parents=True)
    source = example / "baseline_monte_carlo.py"
    source.write_text("physics = 1\n")
    report = example / "EXECUTION.md"
    report.write_text("Build pending.\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setattr(baseline, "__file__", str(source))
    original = baseline.source_record()["source_fingerprint"]
    report.write_text("Build passed.\n")
    assert baseline.source_record()["source_fingerprint"] == original
    source.write_text("physics = 2\n")
    assert baseline.source_record()["source_fingerprint"] != original


def test_coverage_timeline_uses_first_qualified_completion_and_full_delivery():
    from examples.multiagent_imaging.aggregate_baseline_monte_carlo import (
        coverage_timeline,
    )

    records = [
        dict(target_id=0, quality=1, completion_time=30, delivery_time=80),
        dict(target_id=0, quality=1, completion_time=10, delivery_time=90),
        dict(target_id=1, quality=0, completion_time=5, delivery_time=6),
        dict(target_id=1, quality=1, completion_time=50, delivery_time=None),
    ]
    result = dict(
        capture_records=records,
        ground_delivery_records=records,
        sim_time_s=100,
        coverage=dict(
            catalog_target_count=4,
            capture_target_count=2,
            ground_delivery_target_count=1,
        ),
    )
    x, y = coverage_timeline(result)
    np.testing.assert_array_equal(x, [0, 10, 50, 100])
    np.testing.assert_array_equal(y, [0, 25, 50, 50])
    x, y = coverage_timeline(result, ground=True)
    np.testing.assert_array_equal(x, [0, 80, 100])
    np.testing.assert_array_equal(y, [0, 25, 25])
    result["coverage"]["ground_delivery_target_count"] = 2
    with pytest.raises(ValueError, match="endpoint coverage"):
        coverage_timeline(result, ground=True)


def test_paired_aggregation_rejects_unmatched_initial_states():
    from examples.multiagent_imaging.aggregate_baseline_monte_carlo import (
        validate_results,
    )
    from examples.multiagent_imaging.baseline_monte_carlo import (
        CAMPAIGN_VERSION,
        digest,
    )

    config = {"n_sensors": 3, "n_targets": 100}
    manifest = {
        "campaign_version": CAMPAIGN_VERSION,
        "baseline_config": config,
        "source": {"source_fingerprint": "source"},
    }

    def episode(task):
        initial = {"same": "initial state"}
        return dict(
            **task_spec(task),
            campaign_version=CAMPAIGN_VERSION,
            baseline_config=config,
            source=manifest["source"],
            manifest_sha256=digest(manifest),
            initial_conditions=initial,
            initial_conditions_sha256=digest(initial),
            pettingzoo_agents=["sensor_0", "sensor_1", "sensor_2"],
            passive_target_count=100,
            communication={"radio_action_count": 0},
            coordination={
                "communication_time_s": {
                    "sensor_0": 0.0,
                    "sensor_1": 0.0,
                    "sensor_2": 0.0,
                }
            },
            event_steps=4,
            centralized_information_audit={
                "enabled": task >= 100,
                "decision_boundaries": 4 if task >= 100 else 0,
                "sensor_state_reads": 12 if task >= 100 else 0,
                "last_snapshot_sha256": "snapshot" if task >= 100 else None,
            },
        )

    paired = [episode(0), episode(100)]
    assert (
        validate_results(manifest, paired, allow_partial=True)[
            "validated_information_pairs"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="Missing 198"):
        validate_results(manifest, paired)
    paired[1]["initial_conditions"] = {"different": "initial state"}
    paired[1]["initial_conditions_sha256"] = digest(paired[1]["initial_conditions"])
    with pytest.raises(ValueError, match="initial conditions differ"):
        validate_results(manifest, paired, allow_partial=True)


def test_pooled_pair_statistics_resample_shared_seed_blocks():
    from examples.multiagent_imaging.aggregate_baseline_monte_carlo import (
        summarize_seed_blocked,
    )

    rows = [
        {"seed": 0, "environment": "leo", "effect": 1.0},
        {"seed": 0, "environment": "mixed", "effect": 3.0},
        {"seed": 1, "environment": "leo", "effect": 5.0},
        {"seed": 1, "environment": "mixed", "effect": 7.0},
    ]
    summary = summarize_seed_blocked(rows, "effect")
    assert summary["n"] == 4
    assert summary["n_seed_blocks"] == 2
    assert summary["mean"] == 4.0
    assert summary["bootstrap_mean_ci95"] == [2.0, 6.0]


def test_flatten_separates_positive_reward_components_from_adjustments():
    from examples.multiagent_imaging.aggregate_baseline_monte_carlo import flatten

    result = {
        "task_id": 0,
        "case": "independent",
        "target_environment": "leo",
        "seed": 0,
        "sim_time_s": 10,
        "horizon_reached": True,
        "wall_time_s": 1,
        "peak_process_rss_bytes": 1,
        "simulated_seconds_per_wall_second": 10,
        "total_constellation_reward": 19.0,
        "baseline_config": {"alpha": 0.1},
        "coverage": {
            "capture_coverage_fraction": 1,
            "ground_delivery_coverage_fraction": 1,
            "capture_target_count": 1,
            "ground_delivery_target_count": 1,
            "qualified_exposure_count": 1,
            "unqualified_exposure_count": 0,
            "qualified_ground_delivery_count": 1,
            "cross_sensor_capture_overlap_count": 0,
        },
        "product_duplicates": {
            key: 0
            for key in (
                "stale_cross_sensor_ground_delivery_count",
                "stale_cross_sensor_ground_delivery_target_count",
                "causally_avoidable_stale_ground_delivery_count",
                "cross_sensor_onboard_overlap_target_count",
                "cross_sensor_onboard_overlap_product_count",
                "cross_sensor_onboard_redundant_acquisition_count",
                "cross_sensor_onboard_redundant_sensor_time_s",
                "cross_sensor_onboard_overlap_sensor_time_s",
            )
        },
        "coordination": {
            "duplicate_sensor_time_s": 0,
            "interrupted_nonduplicate_sensor_time_s": 0,
            "wasted_time_fraction": 0,
            "policy_decisions": {"sensor_0": 1},
            "communication_time_s": {"sensor_0": 0},
        },
        "team_summary": {
            "duplicate_attempt_count": 0,
            "successful_duplicate_count": 0,
            "unique_acquisition_count": 2,
            "unique_service_count": 1,
            "team_acquisition_value": 20,
            "team_value": 10,
        },
        "resource_history": {
            "sensor_0": [
                {
                    "battery_fraction": 1,
                    "storage_fraction": 0,
                    "max_wheel_fraction": 0,
                    "alive": True,
                }
            ]
        },
    }
    row = flatten(result)
    assert row["acquisition_reward_component"] == 18
    assert row["ground_delivery_reward_component"] == 1
    assert row["reward_adjustment_component"] == 0
    assert row["unique_acquisition_service_count"] == 2
    assert row["unique_ground_service_count"] == 1
