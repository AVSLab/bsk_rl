"""Common physical metrics and validation score for every study method."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import ValidationConfig


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def episode_metrics(env) -> dict[str, float]:
    """Collect method-independent end-of-episode operational metrics."""

    # RLlib's EpisodeData wrapper is itself passed to this function.
    base = getattr(env, "env", env)
    scanner = base.satellites[0]
    scenario = base.scenario
    rewarder = base.rewarder
    reward_data = rewarder.data
    catalog_size = int(
        getattr(scenario, "sampled_catalog_size", None) or scenario.n_targets
    )
    unique_observations = len(getattr(reward_data, "imaged", []))
    illuminated_observations = len(
        set(getattr(rewarder, "imaged_illuminated_names", set()))
    )
    action_counts = getattr(scanner, "study_action_counts", {})
    total_actions = int(sum(action_counts.values()))
    interventions = int(getattr(base, "study_constraint_interventions", 0))
    storage_fraction = _safe_float(scanner.dynamics.storage_level_fraction)
    battery_fraction = _safe_float(scanner.dynamics.battery_charge_fraction)
    storage_bits = _safe_float(scanner.dynamics.storage_level)
    alive = bool(scanner.dynamics.is_alive())
    rewarder_reward = _safe_float(rewarder.cum_reward.get("SS1", 0.0))
    imaging_bonus = _safe_float(scanner.dynamics.imaging_bonus)
    downlink_bonus = _safe_float(scanner.dynamics.downlink_bonus)
    imaging_reward_component = illuminated_observations * imaging_bonus
    downlink_reward_component = (
        float(getattr(rewarder, "useful_downlinks", 0)) * downlink_bonus
    )
    operational_penalty_component = (
        rewarder_reward - imaging_reward_component - downlink_reward_component
    )
    failure_penalty_component = 0.0 if alive else _safe_float(base.failure_penalty)
    initial_battery_fraction = _safe_float(
        scanner.sat_args.get("storedCharge_Init", 0.0)
        / scanner.sat_args.get("batteryStorageCapacity", 1.0)
    )

    return {
        "episode_target_count": float(catalog_size),
        "episode_duration_s": _safe_float(base.simulator.sim_time),
        "episode_reward": rewarder_reward + failure_penalty_component,
        "rewarder_reward": rewarder_reward,
        "imaging_reward_component": imaging_reward_component,
        "downlink_reward_component": downlink_reward_component,
        "operational_penalty_component": operational_penalty_component,
        "failure_penalty_component": failure_penalty_component,
        "successful_observations": float(unique_observations),
        "illuminated_observations": float(illuminated_observations),
        "successful_observation_fraction": unique_observations / catalog_size,
        "illuminated_observation_fraction": illuminated_observations / catalog_size,
        "useful_deliveries": float(getattr(rewarder, "useful_downlinks", 0)),
        "total_downlinks": float(getattr(rewarder, "total_downlinks", 0)),
        "onboard_backlog_bits": storage_bits,
        "onboard_backlog_fraction": storage_fraction,
        "final_battery_fraction": battery_fraction,
        "initial_battery_fraction": initial_battery_fraction,
        "survival_fraction": float(alive),
        "image_action_count": float(action_counts.get("image", 0)),
        "charge_action_count": float(action_counts.get("charge", 0)),
        "downlink_action_count": float(action_counts.get("downlink", 0)),
        "desaturation_action_count": float(action_counts.get("desaturate", 0)),
        "resource_constraint_interventions": float(interventions),
        "constraint_intervention_rate": interventions / max(total_actions, 1),
        "total_action_count": float(total_actions),
    }


def satellite_metrics(env, satellite) -> dict[str, float]:
    if satellite.name != "SS1":
        return {
            "battery_fraction": 0.0,
            "storage_fraction": 0.0,
            "wheel_speed_fraction_max": 0.0,
        }
    return {
        "battery_fraction": _safe_float(satellite.dynamics.battery_charge_fraction),
        "storage_fraction": _safe_float(satellite.dynamics.storage_level_fraction),
        "wheel_speed_fraction_max": _safe_float(
            np.max(np.abs(satellite.dynamics.wheel_speeds_fraction))
        ),
    }


def physical_validation_score(
    metrics: dict[str, float], validation: ValidationConfig
) -> float:
    """Common policy-independent selection score declared before evaluation."""

    return float(
        sum(
            weight * _safe_float(metrics.get(metric, 0.0))
            for metric, weight in validation.score_weights.items()
        )
    )


__all__ = ["episode_metrics", "physical_validation_score", "satellite_metrics"]
