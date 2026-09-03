import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(
    "initial_priority_runner",
    ROOT / "examples" / "amos_2026" / "run_initial_priority_allocation_mc_task.py",
)
ANALYZER = load_module(
    "initial_priority_analyzer",
    ROOT / "examples" / "amos_2026" / "analyze_initial_priority_allocation_mc.py",
)


@pytest.mark.parametrize(
    ("task_id", "case", "cooldown", "seed"),
    [
        (0, "ground_confirmation", 0.0, 0),
        (1, "one_orbit", 1.0, 0),
        (98, "ground_confirmation", 0.0, 49),
        (99, "one_orbit", 1.0, 49),
    ],
)
def test_task_assignment_is_paired(task_id, case, cooldown, seed):
    assert RUNNER.task_assignment(task_id, 50) == (case, cooldown, seed)


@pytest.mark.parametrize("task_id", [-1, 100])
def test_task_assignment_rejects_out_of_range_ids(task_id):
    with pytest.raises(ValueError):
        RUNNER.task_assignment(task_id, 50)


def test_normal_targets_are_split_by_priority_into_scalable_tertiles():
    normal_priorities = np.linspace(0.01, 1.99, 160)
    frame = pd.DataFrame(
        {
            "seed": 0,
            "target_id": np.arange(200),
            "response_class": (
                ["CONTROL"] * 160 + ["HIO"] * 20 + ["SHIO"] * 20
            ),
            "initial_priority": np.r_[
                normal_priorities,
                np.linspace(0.1, 1.9, 40),
            ],
        }
    )

    assigned = ANALYZER.assign_normal_tertiles(frame)
    counts = assigned["allocation_class"].value_counts().to_dict()

    assert counts == {
        "HIO": 20,
        "SHIO": 20,
        "Normal: lower third": 54,
        "Normal: middle third": 53,
        "Normal: upper third": 53,
    }
    normal = assigned[assigned["response_class"].eq("CONTROL")]
    group_means = normal.groupby("allocation_class")["initial_priority"].mean()
    assert (
        group_means["Normal: lower third"]
        < group_means["Normal: middle third"]
        < group_means["Normal: upper third"]
    )


def test_aggregate_analysis_writes_vector_plots_and_statistics(tmp_path):
    frames = []
    for case_index, case in enumerate(ANALYZER.CASE_ORDER):
        for seed in range(3):
            frame = pd.DataFrame(
                {
                    "case": case,
                    "seed": seed,
                    "target_id": np.arange(100),
                    "response_class": (
                        ["CONTROL"] * 50 + ["HIO"] * 25 + ["SHIO"] * 25
                    ),
                    "initial_priority": np.linspace(0.01, 1.99, 100),
                    "successful_image_count_after_event": (
                        (np.arange(100) + 3 * seed) % 7 + case_index
                    ),
                    "useful_delivery_count": (np.arange(100) + seed) % 5,
                    "delivered_priority_value": np.arange(100) % 7,
                    "candidate_presentation_count": np.arange(100) + 1,
                    "eligible_visible_access_count": np.arange(100) + 2,
                }
            )
            frames.append(ANALYZER.assign_normal_tertiles(frame))
    targets = pd.concat(frames, ignore_index=True)

    summary = ANALYZER.build_seed_class_summary(targets)
    statistics = ANALYZER.hio_shio_statistics(summary)
    ANALYZER.plot_allocation(summary, tmp_path)
    ANALYZER.plot_service_metrics(summary, tmp_path)
    ANALYZER.plot_hio_shio_differences(summary, tmp_path)
    ANALYZER.write_summary(statistics, summary, tmp_path)

    assert len(summary) == 2 * 3 * 5
    assert len(statistics) == 2 * 5
    for stem in (
        "initial_priority_capture_allocation",
        "initial_priority_service_metrics",
        "initial_priority_hio_shio_differences",
    ):
        assert (tmp_path / f"{stem}.pdf").stat().st_size > 0
        assert (tmp_path / f"{stem}.png").stat().st_size > 0
    assert (tmp_path / "STATISTICAL_SUMMARY.md").stat().st_size > 0
