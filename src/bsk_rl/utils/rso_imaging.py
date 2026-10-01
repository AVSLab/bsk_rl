"""Shared contracts for RSO imaging execution and candidate observations."""

from dataclasses import dataclass

import numpy as np
from Basilisk.architecture import messaging


@dataclass(frozen=True)
class RSOImageRecord:
    """Provenance of one physically stored image (sizes are in bits)."""

    record_id: str
    target_id: str
    source_imager: str
    capture_time: float
    size_bits: float
    quality: float
    hold_valid_time_s: float
    delivery_time: float | None = None


@dataclass(frozen=True)
class CandidateSnapshot:
    """One imager's ordered, possibly padded decision candidates."""

    time: float
    revision: int
    targets: tuple


def target_geometry(imager, target):
    """Return relative position [m], distance [m], and elevation [deg]."""
    position = np.asarray(imager.dynamics.r_BN_N)
    relative = np.asarray(target.target_spacecraft.dynamics.r_BN_N) - position
    distance = float(np.linalg.norm(relative))
    elevation = float(
        np.degrees(
            np.arcsin(
                np.clip(
                    relative @ position / (distance * np.linalg.norm(position)),
                    -1,
                    1,
                )
            )
        )
    )
    return relative, distance, elevation


def candidate_snapshot(imager, count: int) -> CandidateSnapshot:
    """Share one selection between observations and action decoding.

    Retain AMOS ascending elevation in [-21, 90] degrees, then nearest-distance
    fill. Catalog order breaks ties. Empty slots are None, never duplicated targets.
    Physical access is evaluated separately by the acquisition gate.
    """
    scene = imager.rso_scenario
    now = float(imager.simulator.sim_time)
    cached = getattr(imager, "_rso_candidate_snapshot", None)
    if (
        cached is not None
        and cached.time == now
        and cached.revision == scene.revision
        and len(cached.targets) >= count
    ):
        return cached
    # Include all selectable slots even when an observation requests fewer rows.
    from bsk_rl.act.rso_imaging import ImageRSO

    actions = getattr(getattr(imager, "action_builder", None), "action_spec", ())
    count = max(
        [
            count,
            *(action.n_actions for action in actions if isinstance(action, ImageRSO)),
        ]
    )
    rows = []
    for index, target in enumerate(scene.targets_by_id.values()):
        _, distance, elevation = target_geometry(imager, target)
        if scene.is_eligible(target.id, now):
            rows.append((target, elevation, distance, index))
    visible = sorted(
        (row for row in rows if -21 <= row[1] <= 90), key=lambda row: (row[1], row[3])
    )
    selected = [row[0] for row in visible[:count]]
    selected_ids = {target.id for target in selected}
    remaining = sorted(
        (row for row in rows if row[0].id not in selected_ids),
        key=lambda row: (row[2], row[3]),
    )
    selected.extend(row[0] for row in remaining[: count - len(selected)])
    selected.extend([None] * (count - len(selected)))
    snapshot = CandidateSnapshot(now, scene.revision, tuple(selected))
    imager._rso_candidate_snapshot = snapshot
    return snapshot


__all__ = []


def illumination_factor(payload):
    """Read Basilisk's illuminated fraction: zero is dark, one is fully lit.

    Use the standard payload accessor because another SWIG module can return
    an equivalent payload wrapper without the Python alias on its own class.
    """
    return float(messaging.EclipseMsgPayload.illuminationFactor.fget(payload))
