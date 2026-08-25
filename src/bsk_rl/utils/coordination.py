"""Small coordination helpers without environment or communication dependencies."""

from __future__ import annotations

from typing import Optional

import numpy as np


def current_target_id(sensor) -> Optional[int]:
    """Return the target ID of a sensor's active imaging action, if any."""
    if "imag" not in str(getattr(sensor, "_current_action_label", "")).lower():
        return None
    for action in sensor.action_builder.action_spec:
        chosen = getattr(action, "chosen_target_ids", None)
        if chosen:
            return int(chosen[-1])
    return None


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


__all__ = ["current_target_id", "earth_unoccluded"]
