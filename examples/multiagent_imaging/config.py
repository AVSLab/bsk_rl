"""Configuration schema for bounded multi-agent imaging experiments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


# One versioned contract is shared by observations, actions, PPO, and evaluation.
from bsk_rl.obs.completion_observations import (
    GLOBAL_FEATURES,
    TARGET_FEATURES,
    NON_IMAGING_ACTIONS,
    OBSERVATION_VERSION,
)


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
    retasking_mode: str = "conflict"
    link_mode: str = "los"
    message_delay_s: float = 0.0
    packet_loss_probability: float = 0.0
    communication_cost_per_s: float = 0.0
    # One mission-length half-life: a reward at 45,000 s retains half its weight.
    discount_per_s: float = math.exp(math.log(0.5) / 45000.0)
    gae_trace_half_life_s: float = 6000.0
    communication_mode: str = "broadcast"
    transmit_duration_s: float = 300.0
    transmit_hold_s: float = 10.0
    metadata_bitrate_bps: float | None = None
    message_ttl_s: float = 600.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.information_case not in {
            "independent",
            "ideal_completion",
            "completion",
        }:
            raise ValueError(
                "Use independent, ideal_completion, or completion; legacy intent checkpoints/configurations use a different contract."
            )
        if self.retasking_mode not in {"conflict", "continuous"}:
            raise ValueError("retasking_mode must be conflict or continuous.")
        if self.communication_mode not in {"broadcast", "directed"}:
            raise ValueError("communication_mode must be broadcast or directed.")
        if self.metadata_bitrate_bps is not None and (
            not math.isfinite(self.metadata_bitrate_bps)
            or self.metadata_bitrate_bps <= 0
        ):
            raise ValueError("metadata_bitrate_bps must be finite and positive.")
        if self.transmit_hold_s > self.transmit_duration_s:
            raise ValueError("Transmit deadline must accommodate its minimum hold.")
        if self.link_mode not in {"ideal", "los"}:
            raise ValueError("link_mode must be ideal or los.")
        for name in (
            "episode_duration_s",
            "sim_rate_s",
            "max_step_duration_s",
            "imaging_duration_s",
            "downlink_duration_s",
            "charge_duration_s",
            "desat_duration_s",
            "broadcast_duration_s",
            "min_pointing_hold_s",
            "message_ttl_s",
            "transmit_duration_s",
            "transmit_hold_s",
            "gae_trace_half_life_s",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "message_delay_s",
            "communication_cost_per_s",
            "reimage_cooldown_orbits",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if (
            not 0 <= self.packet_loss_probability <= 1
            or not 0 < self.discount_per_s <= 1
        ):
            raise ValueError("Invalid loss probability or per-second discount.")
        if self.n_sensors < 1:
            raise ValueError("n_sensors must be positive.")
        if self.n_targets < self.n_candidates or self.n_candidates < 1:
            raise ValueError("Require n_targets >= n_candidates >= 1.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")

    @property
    def n_peers(self):
        return self.n_sensors - 1 if self.communication_mode == "directed" else 0

    @property
    def discount_half_life_s(self):
        return (
            math.log(0.5) / math.log(self.discount_per_s)
            if self.discount_per_s < 1
            else None
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "MultiAgentImagingConfig":
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = [
    "GLOBAL_FEATURES",
    "MultiAgentImagingConfig",
    "NON_IMAGING_ACTIONS",
    "TARGET_FEATURES",
    "OBSERVATION_VERSION",
]
