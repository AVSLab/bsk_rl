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
    # Experimental, unit-tested opt-in objective. Native/PPO qualification and
    # independent episode-accounting support are still required before launch.
    # Old checkpoints retain source-owned returns.
    reward_credit_mode: str = "owned"
    information_case: str = "independent"
    retasking_mode: str = "conflict"
    link_mode: str = "los"
    message_delay_s: float = 0.0
    packet_loss_probability: float = 0.0
    communication_cost_per_s: float = 0.0
    # State-local action shielding.  Downlinking an empty buffer previously
    # allowed a high-entropy policy's tiny logit preference to create a rapid
    # empty-downlink loop.  The battery reserve prevents starting a new powered
    # task when only charging is operationally safe; an already active task may
    # still use the explicit continue action.
    empty_storage_mask_fraction: float = 1e-6
    minimum_task_battery_fraction: float = 0.05
    # One mission-length half-life: a reward at 45,000 s retains half its weight.
    discount_per_s: float = math.exp(math.log(0.5) / 45000.0)
    gae_trace_half_life_s: float = 6000.0
    # Explicit AMOS-parity ablation. None retains the physical-second multi-agent trace.
    gae_lambda_per_decision: float | None = None
    policy_model_profile: str = "completion_default"
    communication_mode: str = "broadcast"
    transmit_duration_s: float = 300.0
    transmit_hold_s: float = 10.0
    metadata_bitrate_bps: float | None = None
    message_ttl_s: float = 600.0
    # Mission-population and sensor-geometry fields are part of the checkpoint
    # contract.  Legacy defaults keep earlier small examples reproducible; the
    # completion-v3 pilot selects the explicit Walker/mixed values below.
    target_population: str = "all_leo"
    sensor_constellation: str = "legacy_staggered"
    walker_altitude_km: float = 700.0
    walker_inclination_deg: float = 97.0
    walker_planes: int = 2
    walker_phasing: int = 1
    seed: int = 0
    inventory_chunk_records: int = 256

    def __post_init__(self) -> None:
        if self.reward_credit_mode not in {"owned", "team_mean"}:
            raise ValueError("Unknown reward credit mode")
        if self.gae_lambda_per_decision is not None and (
            not math.isfinite(self.gae_lambda_per_decision)
            or not 0 <= self.gae_lambda_per_decision <= 1
        ):
            raise ValueError("Per-decision GAE lambda must be finite and in [0, 1].")
        if self.policy_model_profile not in {"completion_default", "amos_attention"}:
            raise ValueError("Unknown policy model profile.")
        if self.information_case not in {
            "independent",
            "ideal_completion",
            "completion", "catalog_tx", "geometry_only",
        }:
            raise ValueError(
                "Use independent, ideal_completion, or completion; legacy intent checkpoints/configurations use a different contract."
            )
        if self.inventory_chunk_records < 1:
            raise ValueError("Positive inventory chunk record limit required")
        if self.is_study and (self.communication_mode != "directed" or self.link_mode != "los"):
            raise ValueError("Study requires directed LOS navigation/transport")
        if self.retasking_mode not in {"conflict", "continuous"}:
            raise ValueError("retasking_mode must be conflict or continuous.")
        if self.communication_mode not in {"broadcast", "directed"}:
            raise ValueError("communication_mode must be broadcast or directed.")
        if self.target_population not in {"all_leo", "mixed_50_30_20"}:
            raise ValueError("target_population must be all_leo or mixed_50_30_20.")
        if self.sensor_constellation not in {"legacy_staggered", "walker_delta"}:
            raise ValueError(
                "sensor_constellation must be legacy_staggered or walker_delta."
            )
        if self.sensor_constellation == "walker_delta":
            if self.walker_planes < 1 or self.n_sensors % self.walker_planes:
                raise ValueError(
                    "Walker planes must be positive and divide the sensing-agent count."
                )
            if not 0 <= self.walker_phasing < self.walker_planes:
                raise ValueError("Walker phasing must satisfy 0 <= F < planes.")
            if not 0 < self.walker_altitude_km < 100000:
                raise ValueError("Walker altitude must be finite and positive.")
            if not 0 <= self.walker_inclination_deg <= 180:
                raise ValueError("Walker inclination must be in [0, 180] degrees.")
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
        if not 0.0 <= self.empty_storage_mask_fraction < 1.0:
            raise ValueError("empty_storage_mask_fraction must be in [0, 1).")
        if not 0.0 <= self.minimum_task_battery_fraction < 1.0:
            raise ValueError("minimum_task_battery_fraction must be in [0, 1).")

    @property
    def is_study(self): return self.information_case in {"catalog_tx", "geometry_only"}

    @property
    def peer_features(self): return 18 if self.is_study else 12

    @property
    def target_features(self): return 21 if self.is_study else TARGET_FEATURES

    @property
    def observation_version(self): return "walker-inventory-navigation-v1" if self.is_study else OBSERVATION_VERSION

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
        values = asdict(self)
        if self.reward_credit_mode == "owned":
            values.pop("reward_credit_mode")
        if not self.is_study: values.pop("inventory_chunk_records")
        # Preserve the exact semantic contract of historical checkpoints when
        # these new, opt-in experimental fields are not selected.
        if self.gae_lambda_per_decision is None:
            values.pop("gae_lambda_per_decision")
        if self.policy_model_profile == "completion_default":
            values.pop("policy_model_profile")
        return values

    @property
    def effective_gae_lambda(self):
        return (
            self.gae_lambda_per_decision
            if self.gae_lambda_per_decision is not None
            else math.exp(math.log(0.5) / self.gae_trace_half_life_s)
        )

    @property
    def gae_lambda_time(self):
        return "step" if self.gae_lambda_per_decision is not None else "second"


__all__ = [
    "GLOBAL_FEATURES",
    "MultiAgentImagingConfig",
    "NON_IMAGING_ACTIONS",
    "TARGET_FEATURES",
    "OBSERVATION_VERSION",
]
