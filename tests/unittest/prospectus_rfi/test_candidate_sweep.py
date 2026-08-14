from dataclasses import replace
from pathlib import Path

import yaml

from examples.prospectus_rfi.config import load_study_config
from examples.prospectus_rfi.models import (
    build_actor_critic,
    layout_from_environment,
    parameter_count,
)


CONFIG_DIR = Path(__file__).parents[3] / "examples" / "prospectus_rfi" / "configs"


def test_candidate_sweep_is_six_single_seed_runs():
    with (CONFIG_DIR / "candidate_sweep.yaml").open() as stream:
        sweep = yaml.safe_load(stream)

    assert sweep["candidate_counts"] == [5, 10, 20]
    assert len(sweep["architectures"]) == 2
    assert sweep["training_seeds"] == [10001]
    assert sweep["run_count"] == 6


def test_attention_parameters_are_candidate_count_independent():
    config = load_study_config(
        CONFIG_DIR / "attention_selected.yaml", CONFIG_DIR / "base.yaml"
    )
    counts = []
    for candidate_count in (5, 10, 20):
        environment = replace(config.environment, candidate_count=candidate_count)
        model = build_actor_critic(
            config.architecture, layout_from_environment(environment)
        )
        counts.append(parameter_count(model))

    assert len(set(counts)) == 1


def test_monolithic_parameters_change_with_candidate_count():
    config = load_study_config(
        CONFIG_DIR / "mlp_selected.yaml", CONFIG_DIR / "base.yaml"
    )
    counts = []
    for candidate_count in (5, 10, 20):
        environment = replace(config.environment, candidate_count=candidate_count)
        model = build_actor_critic(
            config.architecture, layout_from_environment(environment)
        )
        counts.append(parameter_count(model))

    assert counts[0] < counts[1] < counts[2]
