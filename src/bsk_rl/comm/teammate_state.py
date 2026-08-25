"""Scalable, information-bounded teammate status summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


ACTION_CATEGORIES = (
    "charge",
    "downlink",
    "desat",
    "broadcast",
    "imaging",
    "other",
)
POOLED_TEAMMATE_FEATURES = (
    "distance",
    "radial_rate",
    "battery_fraction",
    "storage_fraction",
    "wheel_speed_fraction",
    "action_remaining",
    "status_age",
    "catalog_age",
    "link_available",
)
TEAMMATE_SUMMARY_KEYS = tuple(
    [f"teammate_mean_{name}" for name in POOLED_TEAMMATE_FEATURES]
    + [f"teammate_max_{name}" for name in POOLED_TEAMMATE_FEATURES]
    + [f"teammate_action_fraction_{name}" for name in ACTION_CATEGORIES]
)


@dataclass(frozen=True, slots=True)
class TeammateStatus:
    """One teammate record available under the selected information case."""

    source_sensor: str
    creation_time: float
    position_N: tuple[float, float, float]
    velocity_N: tuple[float, float, float]
    battery_fraction: float
    storage_fraction: float
    wheel_speed_fraction: float
    action: str
    action_remaining_s: float
    catalog_update_time: Optional[float]
    target_id: Optional[int]


def action_category(action: str) -> str:
    """Map an action label into a stable, non-ordinal category."""
    label = str(action).lower()
    for category in ACTION_CATEGORIES[:-1]:
        if category in label:
            return category
    return "other"


def current_target_id(sensor) -> Optional[int]:
    """Return the target of an active imaging action, if one exists."""
    if "imag" not in str(getattr(sensor, "_current_action_label", "")).lower():
        return None
    for action in sensor.action_builder.action_spec:
        chosen = getattr(action, "chosen_target_ids", None)
        if chosen:
            return int(chosen[-1])
    return None


def action_remaining_s(sensor, sim_time: float) -> float:
    """Return scheduled time remaining for the active action."""
    if bool(getattr(sensor, "requires_retasking", True)):
        return 0.0
    terminal_time = getattr(sensor, "_timed_terminal_time", None)
    if terminal_time is None:
        return 0.0
    return max(0.0, float(terminal_time) - float(sim_time))


def latest_catalog_update_time(catalog) -> Optional[float]:
    """Return the newest finite update time in one local catalog."""
    updates = [
        float(state.last_update_time)
        for state in catalog.targets.values()
        if np.isfinite(state.last_update_time)
    ]
    return max(updates, default=None)


def status_from_sensor(sensor, sim_time: float) -> TeammateStatus:
    """Snapshot compact state directly from one sensing spacecraft."""
    wheel_speeds = np.asarray(
        getattr(sensor.dynamics, "wheel_speeds_fraction", np.zeros(3)), dtype=float
    )
    catalog = getattr(sensor, "local_catalog", None)
    return TeammateStatus(
        source_sensor=str(sensor.name),
        creation_time=float(sim_time),
        position_N=tuple(map(float, np.asarray(sensor.dynamics.r_BN_N, dtype=float))),
        velocity_N=tuple(map(float, np.asarray(sensor.dynamics.v_BN_N, dtype=float))),
        battery_fraction=float(sensor.dynamics.battery_charge_fraction),
        storage_fraction=float(sensor.dynamics.storage_level_fraction),
        wheel_speed_fraction=float(np.max(np.abs(wheel_speeds))),
        action=str(getattr(sensor, "_current_action_label", "unknown")),
        action_remaining_s=action_remaining_s(sensor, sim_time),
        catalog_update_time=(
            latest_catalog_update_time(catalog) if catalog is not None else None
        ),
        target_id=current_target_id(sensor),
    )


def earth_unoccluded(
    first_position_N,
    second_position_N,
    *,
    earth_radius_m: float = 6378.1366e3,
) -> bool:
    """Return whether the segment between two positions clears the Earth."""
    first = np.asarray(first_position_N, dtype=float)
    second = np.asarray(second_position_N, dtype=float)
    segment = second - first
    denominator = float(segment @ segment)
    if denominator == 0.0:
        return True
    fraction = float(np.clip(-(first @ segment) / denominator, 0.0, 1.0))
    closest = first + fraction * segment
    return float(np.linalg.norm(closest)) > float(earth_radius_m)


def pool_teammate_statuses(
    statuses: Iterable[TeammateStatus],
    *,
    receiver_position_N,
    receiver_velocity_N,
    sim_time: float,
    distance_norm_m: float,
    speed_norm_m_s: float,
    duration_norm_s: float,
    age_norm_s: float,
) -> dict[str, float]:
    """Return fixed-size mean/max pooling invariant to peer ordering and count."""
    ordered = sorted(statuses, key=lambda status: status.source_sensor)
    if not ordered:
        return {key: 0.0 for key in TEAMMATE_SUMMARY_KEYS}

    receiver_position = np.asarray(receiver_position_N, dtype=float)
    receiver_velocity = np.asarray(receiver_velocity_N, dtype=float)
    rows = []
    action_counts = {category: 0 for category in ACTION_CATEGORIES}
    for status in ordered:
        relative_position = (
            np.asarray(status.position_N, dtype=float) - receiver_position
        )
        relative_velocity = (
            np.asarray(status.velocity_N, dtype=float) - receiver_velocity
        )
        distance = float(np.linalg.norm(relative_position))
        radial_rate = (
            float(relative_position @ relative_velocity) / distance
            if distance > 0.0
            else 0.0
        )
        status_age = max(0.0, float(sim_time) - float(status.creation_time))
        catalog_age = (
            max(0.0, float(sim_time) - float(status.catalog_update_time))
            if status.catalog_update_time is not None
            else float(age_norm_s)
        )
        rows.append(
            [
                distance / float(distance_norm_m),
                radial_rate / float(speed_norm_m_s),
                float(np.clip(status.battery_fraction, 0.0, 1.0)),
                float(np.clip(status.storage_fraction, 0.0, 1.0)),
                float(np.clip(status.wheel_speed_fraction, 0.0, 1.0)),
                max(0.0, status.action_remaining_s) / float(duration_norm_s),
                status_age / float(age_norm_s),
                catalog_age / float(age_norm_s),
                float(earth_unoccluded(receiver_position, status.position_N)),
            ]
        )
        action_counts[action_category(status.action)] += 1

    values = np.asarray(rows, dtype=float)
    summary = {}
    for name, value in zip(POOLED_TEAMMATE_FEATURES, np.mean(values, axis=0)):
        summary[f"teammate_mean_{name}"] = float(value)
    for name, value in zip(POOLED_TEAMMATE_FEATURES, np.max(values, axis=0)):
        summary[f"teammate_max_{name}"] = float(value)
    peer_count = len(ordered)
    for category in ACTION_CATEGORIES:
        summary[f"teammate_action_fraction_{category}"] = (
            action_counts[category] / peer_count
        )
    return summary


__all__ = [
    "ACTION_CATEGORIES",
    "POOLED_TEAMMATE_FEATURES",
    "TEAMMATE_SUMMARY_KEYS",
    "TeammateStatus",
    "action_category",
    "action_remaining_s",
    "current_target_id",
    "earth_unoccluded",
    "latest_catalog_update_time",
    "pool_teammate_statuses",
    "status_from_sensor",
]
