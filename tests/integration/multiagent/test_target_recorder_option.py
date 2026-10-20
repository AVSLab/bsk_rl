"""Operational recorder selection leaves mission/checkpoint semantics unchanged."""
import numpy as np

from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.environment import build_environment
from examples.multiagent_imaging.readiness import schema


def test_diagnostic_recorders_are_opt_in_without_scientific_schema_change(monkeypatch):
    monkeypatch.delenv("BSK_RL_TARGET_RECORDERS", raising=False)
    config = MultiAgentImagingConfig(n_sensors=2, n_targets=2, n_candidates=2,
                                     episode_duration_s=30)
    scientific_config, contract = config.to_dict(), schema(config)
    states = []
    for option in (None, True):
        env = build_environment(config, target_recorders_enabled=option)
        try:
            observations, _ = env.reset(seed=19)
            active = [getattr(s.dynamics, key) is not None for s in env.passive_satellites
                      for key in ("target_state_recorder", "target_eclipse_recorder")]
            assert all(active) if option else not any(active)
            states.append(np.concatenate([s.dynamics.r_BN_N for s in env.satellites]))
            assert config.to_dict() == scientific_config and schema(config) == contract
        finally:
            env.close()
    np.testing.assert_array_equal(states[0], states[1])
