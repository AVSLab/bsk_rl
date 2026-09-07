"""Tests for the deterministic controller used by the Vizard demo."""

import numpy as np

from examples.multiagent_imaging.config import (
    GLOBAL_FEATURES,
    MultiAgentImagingConfig,
    NON_IMAGING_ACTIONS,
    TARGET_FEATURES,
)
from examples.multiagent_imaging.evaluate import _target_action


def _observation(config, angles, *, unavailable=()):
    observation = np.zeros(GLOBAL_FEATURES + config.n_candidates * TARGET_FEATURES)
    targets = observation[GLOBAL_FEATURES:].reshape(
        config.n_candidates, TARGET_FEATURES
    )
    targets[:, 7] = angles
    targets[:, 16] = 1.0
    for index in unavailable:
        targets[index, 16] = 0.0
    return observation


def test_closest_angle_chooses_smallest_eligible_candidate():
    config = MultiAgentImagingConfig(n_targets=4, n_candidates=4)
    observation = _observation(config, [0.3, 0.1, 0.2, 0.4], unavailable=(1,))

    assert _target_action(observation, config, "closest_angle") == (
        NON_IMAGING_ACTIONS + 2
    )


def test_closest_angle_charges_when_every_candidate_is_unavailable():
    config = MultiAgentImagingConfig(n_targets=4, n_candidates=4)
    observation = _observation(config, [0.3, 0.1, 0.2, 0.4], unavailable=range(4))

    assert _target_action(observation, config, "closest_angle") == 0
