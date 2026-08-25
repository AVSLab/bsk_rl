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
        "hio_count": None,
        "shio_count": None,
        "hio_fraction": None,
        "shio_fraction": None,
        "interest_fraction": None,
        "reimage_cooldown_orbits": 0.0,
        "random_imaging": False,
        "heuristic_mode": "off",
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


def test_default_scenario_uses_paper_hio_and_shio_counts():
    args = runner_args()
    command = RUNNER.build_command(args)

    assert command_value(command, "--n_targets") == "200"
    assert command_value(command, "--hio_count") == "5"
    assert command_value(command, "--shio_count") == "3"
    assert command_value(command, "--dynamic_priority_event_fraction") == "0.5"
    assert command_value(command, "--dynamic_priority_event_seed") == "7"
    assert command_value(command, "--reimage_cooldown_orbits") == "0.0"
    assert command[1] == "-u"
    assert "--quiet" not in command
    assert "--no_amos_vizard_metric_bars" not in command
    assert "--amos_vizard_target_status_outlines" not in command
    assert "HIO5_SHIO3_groundConfirm" in RUNNER.vizard_tag(args)


def test_independent_fraction_scenario_scales_each_promotion_tier():
    args = runner_args(hio_fraction=0.25, shio_fraction=0.25)
    command = RUNNER.build_command(args)

    assert command_value(command, "--hio_count") == "50"
    assert command_value(command, "--shio_count") == "50"
    assert "25pctHIO_25pctSHIO_groundConfirm" in RUNNER.vizard_tag(args)


def test_random_imaging_scenario_is_labeled_and_passed_to_evaluator():
    args = runner_args(random_imaging=True, reimage_cooldown_orbits=1.0)
    command = RUNNER.build_command(args)

    assert "--random_imaging" in command
    assert "RANDOM_IMAGING_SHIELD" in command_value(command, "--policy_name")
    assert "randomImagingShield_HIO5_SHIO3_1orbitCooldown" in RUNNER.vizard_tag(
        args
    )


@pytest.mark.parametrize(
    ("mode", "policy_label", "vizard_label"),
    [
        (
            "angle",
            "HEURISTIC_MIN_ANGLE_ELIGIBLE_SHIELD",
            "heuristicMinAngleEligibleShield",
        ),
        (
            "candidate_priority",
            "HEURISTIC_MAX_PRIORITY_CANDIDATE10_SHIELD",
            "heuristicMaxPriorityCandidate10Shield",
        ),
    ],
)
def test_shield_only_heuristic_scenarios_are_distinctly_labeled(
    mode, policy_label, vizard_label
):
    args = runner_args(heuristic_mode=mode, reimage_cooldown_orbits=1.0)
    command = RUNNER.build_command(args)

    assert command_value(command, "--heuristic_mode") == mode
    assert "--heuristic_shield_only" in command
    assert policy_label in command_value(command, "--policy_name")
    assert vizard_label in RUNNER.vizard_tag(args)


def test_random_and_heuristic_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        RUNNER.build_command(
            runner_args(random_imaging=True, heuristic_mode="angle")
        )


def test_legacy_interest_fraction_is_a_symmetric_fraction_shorthand():
    args = runner_args(interest_fraction=0.10)
    command = RUNNER.build_command(args)

    assert command_value(command, "--hio_count") == "20"
    assert command_value(command, "--shio_count") == "20"
    assert "10pctHIO_10pctSHIO_groundConfirm" in RUNNER.vizard_tag(args)


def test_target_status_outlines_are_explicitly_opt_in():
    command = RUNNER.build_command(runner_args(target_status_outlines=True))

    assert "--amos_vizard_target_status_outlines" in command


def test_action_only_recording_and_quiet_mode_are_explicitly_opt_in():
    command = RUNNER.build_command(runner_args(no_metric_bars=True, quiet=True))

    assert "--no_amos_vizard_metric_bars" in command
    assert "--quiet" in command


def test_promotion_fraction_must_map_to_an_exact_catalog_count():
    with pytest.raises(ValueError, match="does not produce an integer target count"):
        RUNNER.resolve_promotions(
            runner_args(n_targets=100, hio_fraction=0.015, shio_fraction=0.10)
        )


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_each_promotion_fraction_must_be_a_catalog_fraction(fraction):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RUNNER.resolve_promotions(
            runner_args(hio_fraction=fraction, shio_fraction=0.10)
        )


def test_disjoint_promotion_groups_cannot_exceed_the_catalog():
    with pytest.raises(ValueError, match="exceeds the 200-target catalog"):
        RUNNER.resolve_promotions(runner_args(hio_fraction=0.75, shio_fraction=0.50))


def test_count_and_fraction_modes_cannot_be_mixed():
    with pytest.raises(ValueError, match="counts or fractions, not both"):
        RUNNER.resolve_promotions(
            runner_args(
                hio_count=5, shio_count=3, hio_fraction=0.10, shio_fraction=0.10
            )
        )
