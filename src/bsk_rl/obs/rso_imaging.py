"""Observations of the exact independently orbiting targets decoded by ImageRSO."""

import numpy as np

from bsk_rl.obs.observations import Observation
from bsk_rl.utils.orbital import rv2HN
from bsk_rl.utils.rso_imaging import (
    candidate_snapshot,
    illumination_factor,
    target_geometry,
)


def target_properties(imager, target):
    """Geometric/illumination features with explicit units and no name parsing."""
    if target is None:
        return dict(
            valid=0.0,
            priority=0.0,
            target_elevation_angle=0.0,
            rel_pos_vector_r_BR_H=np.zeros(3),
            rel_vel_vector_v_BR_H=np.zeros(3),
            angle_to_target=0.0,
            target_distance=0.0,
            target_illumination_factor=0.0,
        )
    relative, distance, elevation = target_geometry(imager, target)
    HN = rv2HN(imager.dynamics.r_BN_N, imager.dynamics.v_BN_N)
    # AMOS uses the difference of inertial velocities, expressed in the observer
    # Hill frame, rather than the derivative in the rotating Hill frame.
    relative_velocity = np.asarray(
        target.target_spacecraft.dynamics.v_BN_N
    ) - np.asarray(imager.dynamics.v_BN_N)
    c_hat_N = np.asarray(imager.dynamics.BN).T @ np.asarray(imager.fsw.locPoint.pHat_B)
    angle = float(np.degrees(np.arccos(np.clip(relative @ c_hat_N / distance, -1, 1))))
    target_dyn = target.target_spacecraft.dynamics
    illumination = illumination_factor(
        target_dyn.world.eclipseObject.eclipseOutMsgs[target_dyn.eclipse_index].read()
    )
    return dict(
        valid=1.0,
        priority=float(target.priority),
        target_elevation_angle=elevation,
        rel_pos_vector_r_BR_H=HN @ relative,
        rel_vel_vector_v_BR_H=HN @ relative_velocity,
        angle_to_target=angle,
        target_distance=distance,
        target_illumination_factor=illumination,
    )


class RSOTargetProperties(Observation):
    """Fixed-size spacecraft candidate rows; empty slots are zeros with valid=0.

    The default row contains priority, elevation, relative position and velocity
    in the observer Hill frame, boresight angle, distance, and illumination.
    Relative velocity is the inertial velocity difference expressed in that frame,
    rather than the derivative of rotating-frame position. Include ``valid`` as
    a custom property when a numeric mask for padded rows is useful.

    Observation and action row counts are independent. ``ImageRSO`` selects from
    the first action-count rows of this ordering; additional rows provide context.
    """

    def __init__(self, *properties, n_ahead_observe=10, name="rso_targets") -> None:
        """Select named properties and optional nonzero normalization factors."""
        super().__init__(name)
        if not isinstance(n_ahead_observe, int) or n_ahead_observe < 1:
            raise ValueError("Candidate count must be a positive integer.")
        allowed = set(target_properties(None, None))
        if not properties:
            properties = (
                dict(prop="priority", norm=2),
                dict(prop="target_elevation_angle", norm=90),
                dict(prop="rel_pos_vector_r_BR_H", norm=15960e3),
                dict(prop="rel_vel_vector_v_BR_H", norm=16000),
                dict(prop="angle_to_target", norm=90),
                dict(prop="target_distance", norm=15960e3),
                dict(prop="target_illumination_factor"),
            )
        self.properties = []
        for spec in properties:
            prop, norm = spec["prop"], float(spec.get("norm", 1))
            if prop not in allowed or not np.isfinite(norm) or norm == 0:
                raise ValueError("Unsupported RSO target property or normalization.")
            self.properties.append((prop, spec.get("name", prop), norm))
        if len({name for _, name, _ in self.properties}) != len(self.properties):
            raise ValueError("Observation property names must be unique.")
        self.n_ahead_observe = n_ahead_observe

    def reset_overwrite_previous(self) -> None:
        """Discard prior-episode candidates."""
        self.satellite._rso_candidate_snapshot = None

    def get_obs(self):
        """Build rows from the shared decision snapshot, including explicit padding."""
        snapshot = candidate_snapshot(self.satellite, self.n_ahead_observe)
        result = {}
        for index, target in enumerate(snapshot.targets[: self.n_ahead_observe]):
            values = target_properties(self.satellite, target)
            result[f"{self.name}_{index}"] = {
                name: values[prop] / norm for prop, name, norm in self.properties
            }
        return result

    def candidate_ids(self) -> tuple[str | None, ...]:
        """Return the same ordered IDs as the observation/action slots.

        This diagnostic does not change the numeric observation schema. Padded
        slots are ``None``; callers must never replace them with a real target.
        """
        return tuple(
            None if target is None else target.id
            for target in candidate_snapshot(
                self.satellite, self.n_ahead_observe
            ).targets[: self.n_ahead_observe]
        )


__doc_title__ = "Space-to-Space RSO Observations"
__all__ = ["RSOTargetProperties"]
