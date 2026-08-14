"""Configuration and reproducibility helpers for the prospectus study."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EnvironmentConfig:
    """Matched physical environment shared by every comparison method."""

    episode_duration_s: float = 45_000.0
    catalog_min: int = 100
    catalog_max: int = 400
    candidate_count: int = 10
    imaging_duration_s: float = 100.0
    variable_duration_imaging: bool = False
    charge_duration_s: float = 300.0
    downlink_duration_s: float = 300.0
    desaturation_duration_s: float = 150.0
    alpha: float = 0.0
    failure_penalty: float = -100.0
    battery_capacity_ws: float = 500.0 * 3600.0
    initial_battery_fraction_min: float = 0.20
    initial_battery_fraction_max: float = 0.60
    image_storage_capacity: float = 50.0
    image_size_bits: float = 0.5 * 8e6
    instrument_baud_rate: float = 0.5 * 8e6
    transmitter_baud_rate: float = -0.5 * 8e6
    base_power_draw_w: float = -10.0
    instrument_power_draw_w: float = -30.0
    transmitter_power_draw_w: float = -25.0
    thruster_power_draw_w: float = -80.0
    panel_area_m2: float = 1.0
    max_wheel_speed_rpm: float = 6000.0
    image_attitude_error_requirement: float = 0.01
    eclipse_threshold: float = 0.5
    allow_reimaging: bool = False
    target_priority: float = 1.0
    target_orbit_distribution: str = "amos2025_leo"
    observation_layout: str = "amos2025_obs_v2_masked"
    action_order: tuple[str, ...] = (
        "image_candidate_slots",
        "charge",
        "downlink",
        "desaturate",
    )

    @property
    def imaging_bonus(self) -> float:
        return 1.0 - self.alpha

    @property
    def downlink_bonus(self) -> float:
        return self.alpha

    @property
    def non_target_actions(self) -> int:
        return 3

    @property
    def action_count(self) -> int:
        return self.candidate_count + self.non_target_actions

    @property
    def storage_capacity_bits(self) -> float:
        return self.image_storage_capacity * self.image_size_bits

    def validate(self) -> None:
        if self.catalog_min < 1 or self.catalog_min > self.catalog_max:
            raise ValueError("catalog_min must be positive and <= catalog_max")
        if self.candidate_count < 1 or self.candidate_count > self.catalog_min:
            raise ValueError(
                "candidate_count must be positive and no larger than the smallest catalog"
            )
        if self.episode_duration_s != 45_000.0:
            raise ValueError("The prospectus study requires 45,000-second episodes")
        if self.imaging_duration_s != 100.0:
            raise ValueError("The prospectus study requires 100-second imaging actions")
        if self.variable_duration_imaging:
            raise ValueError("Imaging must use a fixed decision interval")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.alpha != 0.0 or self.imaging_bonus != 1.0:
            raise ValueError("Research Focus I uses alpha=0 observation-only reward")
        if not (
            0.0
            <= self.initial_battery_fraction_min
            <= self.initial_battery_fraction_max
            <= 1.0
        ):
            raise ValueError("Initial battery fractions must lie in [0, 1]")
        if self.allow_reimaging:
            raise ValueError("The AMOS 2025 comparison does not allow re-imaging")


@dataclass(frozen=True)
class PPOStudyConfig:
    learning_rate: float = 1e-5
    train_batch_size: int = 4200
    minibatch_size: int = 128
    ppo_epochs: int = 10
    clip_parameter: float = 0.05
    entropy_coefficient: float = 0.0
    value_function_coefficient: float = 1.0
    gradient_clip: float = 0.5
    gamma: float = 0.9997
    gae_lambda: float = 0.95
    continuous_time_discount: bool = True
    reward_time: str = "step_start"

    def validate(self) -> None:
        if self.train_batch_size < self.minibatch_size:
            raise ValueError("train_batch_size must be >= minibatch_size")
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")


@dataclass(frozen=True)
class ArchitectureConfig:
    name: str = "fixed_input_monolithic_mlp"
    hidden_widths: tuple[int, ...] = (2048, 2048)
    activation: str = "relu"
    layer_norm: bool = False
    embedding_dim: int = 128
    attention_heads: int = 2
    attention_blocks: int = 1
    feed_forward_width: int = 128
    dropout: float = 0.0
    separate_value_network: bool = True

    def validate(self) -> None:
        if self.name not in {
            "fixed_input_monolithic_mlp",
            "target_set_attention",
        }:
            raise ValueError(f"Unsupported architecture: {self.name}")
        if not self.hidden_widths or any(width < 1 for width in self.hidden_widths):
            raise ValueError("hidden_widths must contain positive integers")
        if self.activation not in {"relu", "silu", "tanh"}:
            raise ValueError("activation must be relu, silu, or tanh")
        if self.name == "target_set_attention":
            if self.embedding_dim % self.attention_heads != 0:
                raise ValueError("embedding_dim must be divisible by attention_heads")
            if self.attention_blocks < 1:
                raise ValueError("attention_blocks must be positive")


@dataclass(frozen=True)
class ValidationConfig:
    seeds: tuple[int, ...] = (91_001, 91_002, 91_003, 91_004, 91_005)
    catalog_sizes: tuple[int, ...] = (100, 250, 400)
    checkpoint_interval_iterations: int = 5
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "successful_observation_fraction": 0.55,
            "illuminated_observation_fraction": 0.25,
            "survival_fraction": 0.15,
            "constraint_intervention_rate": -0.05,
        }
    )
    primary_metric: str = "successful_observation_fraction"
    practical_equivalence_margin: float = 0.02
    score_thresholds: tuple[float, ...] = (0.25, 0.50, 0.70)


@dataclass(frozen=True)
class ComputeConfig:
    campaign_phase: str = "exploratory_candidate_sweep"
    final_seeds: tuple[int, ...] = (10_001,)
    confirmatory_seeds: tuple[int, ...] = (10_001, 20_001, 30_001)
    final_wall_hours: float = 48.0
    tuning_trials_per_architecture: int = 12
    tuning_wall_hours_per_trial: float = 8.0
    evaluation_catalog_sizes: tuple[int, ...] = (100, 200, 300, 400)
    evaluation_episodes_per_size: int = 100
    evaluation_seed_start: int = 700_000
    checkpoint_interval_iterations: int = 3
    checkpoints_to_keep: int = 5

    @property
    def tuning_gpu_hours_per_architecture(self) -> float:
        return self.tuning_trials_per_architecture * self.tuning_wall_hours_per_trial

    @property
    def is_confirmatory(self) -> bool:
        return self.campaign_phase == "confirmatory"


@dataclass(frozen=True)
class StudyConfig:
    study_name: str
    architecture: ArchitectureConfig
    ppo: PPOStudyConfig
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    output_root: str = "results/prospectus_rfi"
    historical_code_commit: str = "d0bcc54c6610643cc946ce92f2ea30314659fe0e"
    public_snapshot_commit: str = "c6b9e4310a36476944cfa27b1d02e43c34362952"

    def validate(self) -> None:
        self.environment.validate()
        self.architecture.validate()
        self.ppo.validate()
        if not self.compute.final_seeds:
            raise ValueError("At least one final-training seed is required")
        if len(self.compute.confirmatory_seeds) < 3:
            raise ValueError(
                "The planned confirmatory campaign requires at least three seeds"
            )
        if self.compute.is_confirmatory and len(self.compute.final_seeds) < 3:
            raise ValueError("Confirmatory training requires at least three seeds")
        if self.compute.final_wall_hours != 48.0:
            raise ValueError("Final runs must retain the requested 48-hour wall budget")
        if self.compute.evaluation_episodes_per_size < 100:
            raise ValueError(
                "At least 100 Monte Carlo episodes per catalog size are required"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _tuple_fields(data: dict[str, Any]) -> dict[str, Any]:
    result = dict(data)
    for key in (
        "hidden_widths",
        "seeds",
        "catalog_sizes",
        "score_thresholds",
        "final_seeds",
        "confirmatory_seeds",
        "evaluation_catalog_sizes",
        "action_order",
    ):
        if key in result and isinstance(result[key], list):
            result[key] = tuple(result[key])
    return result


def load_study_config(
    architecture_path: str | Path,
    base_path: str | Path | None = None,
) -> StudyConfig:
    """Load a selected architecture config, optionally layered on a common base."""

    architecture_path = Path(architecture_path)
    payload: dict[str, Any] = {}
    if base_path is not None:
        with Path(base_path).open() as stream:
            payload = yaml.safe_load(stream) or {}
    with architecture_path.open() as stream:
        payload = _deep_merge(payload, yaml.safe_load(stream) or {})

    config = StudyConfig(
        study_name=payload["study_name"],
        architecture=ArchitectureConfig(**_tuple_fields(payload["architecture"])),
        ppo=PPOStudyConfig(**payload["ppo"]),
        environment=EnvironmentConfig(**_tuple_fields(payload.get("environment", {}))),
        validation=ValidationConfig(**_tuple_fields(payload.get("validation", {}))),
        compute=ComputeConfig(**_tuple_fields(payload.get("compute", {}))),
        output_root=payload.get("output_root", "results/prospectus_rfi"),
        historical_code_commit=payload.get(
            "historical_code_commit",
            "d0bcc54c6610643cc946ce92f2ea30314659fe0e",
        ),
        public_snapshot_commit=payload.get(
            "public_snapshot_commit",
            "c6b9e4310a36476944cfa27b1d02e43c34362952",
        ),
    )
    config.validate()
    return config


def git_metadata(repository: str | Path) -> dict[str, Any]:
    """Return commit and dirty-state metadata without mutating the repository."""

    repository = Path(repository)

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repository, text=True
        ).strip()

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def write_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
