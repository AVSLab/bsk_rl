"""Real-Basilisk checks for the versioned four-sensor Walker/mixed mission."""

from collections import Counter

import numpy as np
import pytest

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.environment import build_environment


def _angle_difference(left, right):
    return (float(left) - float(right)) % (2 * np.pi)


def test_walker_421_and_mixed_population_are_episode_seeded_live_spacecraft():
    config = MultiAgentImagingConfig(
        n_sensors=4,
        n_targets=10,
        n_candidates=3,
        episode_duration_s=30,
        communication_mode="directed",
        information_case="completion",
        target_population="mixed_50_30_20",
        sensor_constellation="walker_delta",
        walker_altitude_km=700,
        walker_inclination_deg=97,
        walker_planes=2,
        walker_phasing=1,
    )
    env = build_environment(config)
    try:
        observations, _ = env.reset(seed=101)
        assert env.possible_agents == [f"sensor_{i}" for i in range(4)]
        assert len(env.passive_satellites) == 10
        assert Counter(
            target.target_population_regime for target in env.passive_satellites
        ) == {"LEO": 5, "MEO": 3, "GEO": 2}
        assert all(observation.shape == (113,) for observation in observations.values())
        assert all(sensor.action_space.n == 11 for sensor in env.sensing_satellites)

        orbits = [sensor.sat_args["oe"] for sensor in env.sensing_satellites]
        assert all(orbit.a == pytest.approx(6371e3 + 700e3) for orbit in orbits)
        assert all(orbit.e == 0 for orbit in orbits)
        assert all(orbit.i == pytest.approx(np.deg2rad(97)) for orbit in orbits)
        assert _angle_difference(orbits[1].f, orbits[0].f) == pytest.approx(np.pi)
        assert _angle_difference(orbits[3].f, orbits[2].f) == pytest.approx(np.pi)
        assert _angle_difference(orbits[2].Omega, orbits[0].Omega) == pytest.approx(
            np.pi
        )
        # Walker F=1 gives a 90-degree inter-plane argument-of-latitude offset.
        assert _angle_difference(
            orbits[2].omega + orbits[2].f,
            orbits[0].omega + orbits[0].f,
        ) == pytest.approx(np.pi / 2)

        expected_period = 2 * np.pi * np.sqrt(
            (6371e3 + 700e3) ** 3 / (398600.436e9)
        )
        assert env.rewarder.reimage_cooldown_s == pytest.approx(2 * expected_period)
        first_state = np.asarray(
            [target.dynamics.r_BN_N for target in env.passive_satellites]
        )
        env.reset(seed=101)
        np.testing.assert_allclose(
            first_state,
            [target.dynamics.r_BN_N for target in env.passive_satellites],
        )
    finally:
        env.close()
