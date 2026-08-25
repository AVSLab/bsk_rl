"""Configuration schema for bounded multi-agent imaging experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MultiAgentImagingConfig:
    n_sensors: int = 2
    n_targets: int = 8
    n_candidates: int = 4
    episode_duration_s: float = 1200.0
    sim_rate_s: float = 1.0
    max_step_duration_s: float = 180.0
    imaging_duration_s: float = 180.0
    downlink_duration_s: float = 120.0
    charge_duration_s: float = 180.0
    desat_duration_s: float = 120.0
    broadcast_duration_s: float = 30.0
    min_pointing_hold_s: float = 10.0
    reimage_cooldown_orbits: float = 2.0
    alpha: float = 0.1
    information_case: str = "independent"
    perfect_metadata_delivery: bool = True
    message_ttl_s: float = 600.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_sensors < 1:
            raise ValueError("n_sensors must be positive.")
        if self.n_targets < self.n_candidates or self.n_candidates < 1:
            raise ValueError("Require n_targets >= n_candidates >= 1.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")

    @classmethod
    def from_json(cls, path: str | Path) -> "MultiAgentImagingConfig":
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)


TARGET_FEATURES = 13
NON_IMAGING_ACTIONS = 4
GLOBAL_FEATURES = 14


__all__ = [
    "GLOBAL_FEATURES",
    "MultiAgentImagingConfig",
    "NON_IMAGING_ACTIONS",
    "TARGET_FEATURES",
]
