"""A matching tensor shape alone is insufficient to restore a spacecraft policy."""

from dataclasses import replace
import pytest
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.readiness import schema, validate_schema
from examples.multiagent_imaging.train import worker_seed


def test_schema_permits_matched_held_out_seeds_only():
    config = MultiAgentImagingConfig(communication_mode="directed")
    validate_schema(schema(config), replace(config, seed=99))
    for changed in (
        replace(config, communication_mode="broadcast"),
        replace(config, retasking_mode="continuous"),
        replace(config, n_candidates=5),
        replace(config, discount_per_s=0.999),
        replace(config, transmit_hold_s=20),
    ):
        with pytest.raises(ValueError, match="schema mismatch"):
            validate_schema(schema(config), changed)
    legacy = schema(config)
    legacy["version"] = "completion-v1"
    with pytest.raises(ValueError):
        validate_schema(legacy, config)


def test_worker_specific_seed_streams_are_reproducible_and_distinct():
    first = [
        worker_seed(41, worker, vector) for worker in range(8) for vector in range(4)
    ]
    assert first == [
        worker_seed(41, worker, vector) for worker in range(8) for vector in range(4)
    ]
    assert len(set(first)) == len(first)
    assert set(first).isdisjoint(
        worker_seed(42, worker, vector) for worker in range(8) for vector in range(4)
    )
