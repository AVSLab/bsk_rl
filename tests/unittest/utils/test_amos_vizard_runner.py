import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "amos_2026"
    / "run_vizard_episode.py"
)
SPEC = importlib.util.spec_from_file_location("amos_vizard_episode_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def runner_args(**overrides):
    values = {
        "policy_path": Path("/tmp/checkpoint"),
        "seed": 7,
        "vizard_rate_hz": 1.0,
        "n_targets": 200,
        "interest_fraction": 0.10,
        "reimage_cooldown_orbits": 0.0,
        "output_dir": Path("/tmp/vizard"),
        "plots_dir": Path("/tmp/plots"),
        "rw_display": "off",
        "no_text_hud": False,
        "no_metric_bars": False,
        "no_image_bars": False,
        "target_status_outlines": False,
        "no_hud": False,
        "quiet": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def command_value(command, flag):
    return command[command.index(flag) + 1]


def test_evaluator_environment_forces_noninteractive_plot_backend():
    environment = RUNNER.evaluator_environment(
        {"MPLBACKEND": "macosx", "EXISTING_SETTING": "preserved"}
    )

    assert environment["MPLBACKEND"] == "Agg"
    assert environment["EXISTING_SETTING"] == "preserved"


def test_default_scenario_is_200_targets_with_two_disjoint_10_percent_tiers():
    args = runner_args()
    command = RUNNER.build_command(args)

    assert command_value(command, "--n_targets") == "200"
    assert command_value(command, "--hio_count") == "20"
    assert command_value(command, "--shio_count") == "20"
    assert command_value(command, "--dynamic_priority_event_fraction") == "0.5"
    assert command_value(command, "--dynamic_priority_event_seed") == "7"
    assert command_value(command, "--reimage_cooldown_orbits") == "0.0"
    assert command[1] == "-u"
    assert "--quiet" not in command
    assert "--no_amos_vizard_metric_bars" not in command
    assert "--amos_vizard_target_status_outlines" not in command
    assert "10pctHIO_10pctSHIO_groundConfirm" in RUNNER.vizard_tag(args)


def test_target_status_outlines_are_explicitly_opt_in():
    command = RUNNER.build_command(runner_args(target_status_outlines=True))

    assert "--amos_vizard_target_status_outlines" in command


def test_action_only_recording_and_quiet_mode_are_explicitly_opt_in():
    command = RUNNER.build_command(
        runner_args(no_metric_bars=True, quiet=True)
    )

    assert "--no_amos_vizard_metric_bars" in command
    assert "--quiet" in command


def test_interest_fraction_must_map_to_an_exact_catalog_count():
    with pytest.raises(ValueError, match="not an integer target count"):
        RUNNER.interest_object_count(100, 0.015)


@pytest.mark.parametrize("fraction", [-0.01, 0.51])
def test_interest_fraction_must_leave_room_for_both_disjoint_tiers(fraction):
    with pytest.raises(ValueError, match=r"\[0, 0.5\]"):
        RUNNER.interest_object_count(200, fraction)
