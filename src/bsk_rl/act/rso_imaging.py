"""Discrete actions for independent RSO spacecraft imaging."""

import warnings

import numpy as np
from Basilisk.utilities import macros

from bsk_rl.act.discrete_actions import DiscreteAction
from bsk_rl.utils.rso_imaging import candidate_snapshot


class ImageRSO(DiscreteAction):
    """Image one of the observed candidate slots, or explicitly task a catalog ID.

    Uses :class:`~bsk_rl.sim.fsw.SpaceToSpaceImagingFSWModel` to track a whole
    spacecraft. For surface inspection of a nearby RSO, see
    :class:`~bsk_rl.sim.fsw.RSOInspectorFSWModel`. An empty candidate slot raises
    ``ValueError``; it never tasks a duplicated or padded target.
    """

    def __init__(
        self,
        n_ahead_image: int,
        max_duration=300.0,
        min_pointing_hold_s=10.0,
        hold_mode="cumulative",
        require_illumination_during_hold=True,
        hold_illumination_threshold=0.5,
        variable_duration_imaging=True,
        name="image_rso",
    ) -> None:
        """Configure candidate count, deadline [s], hold [s], and quality gate."""
        if not isinstance(n_ahead_image, int) or n_ahead_image < 1:
            raise ValueError("Candidate count must be a positive integer.")
        if (
            not np.isfinite(max_duration)
            or max_duration <= 0
            or not np.isfinite(min_pointing_hold_s)
            or min_pointing_hold_s < 0
        ):
            raise ValueError("Duration must be positive and hold nonnegative.")
        if hold_mode not in ("cumulative", "continuous"):
            raise ValueError("Hold mode must be cumulative or continuous.")
        if not 0 <= hold_illumination_threshold <= 1:
            raise ValueError("Illumination threshold must be in [0, 1].")
        super().__init__(name, n_ahead_image)
        self.max_duration = max_duration
        self.min_pointing_hold_s = min_pointing_hold_s
        self.hold_mode = hold_mode
        self.require_illumination_during_hold = require_illumination_during_hold
        self.hold_illumination_threshold = hold_illumination_threshold
        self.variable_duration_imaging = variable_duration_imaging
        self.reset_overwrite_previous()

    def reset_overwrite_previous(self) -> None:
        """Drop event names and per-episode selection history."""
        self.event_name = None
        self.chosen_target_ids = []

    def _task(self, target) -> str:
        if not self.satellite.rso_scenario.is_eligible(
            target.id, self.simulator.sim_time
        ):
            raise ValueError("RSO target is pending or in cooldown.")
        if self.event_name in self.simulator.eventMap:
            self.simulator.delete_event(self.event_name)
        fsw = self.satellite.fsw
        fsw.action_image_rso(
            target,
            duration=self.max_duration,
            hold_s=self.min_pointing_hold_s,
            hold_mode=self.hold_mode,
            require_illumination=self.require_illumination_during_hold,
            illumination_threshold=self.hold_illumination_threshold,
        )
        self.chosen_target_ids.append(target.id)
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + self.max_duration,
            info="RSO imaging deadline",
            extra_actions=lambda sim: fsw.rso_capture.cancel(),
        )
        if self.variable_duration_imaging:
            self.event_name = f"rso_image_success_{self.satellite.name}"
            self.simulator.createNewEvent(
                self.event_name,
                macros.sec2nano(fsw.dynamics.dyn_rate),
                True,
                conditionFunction=lambda sim: fsw.rso_capture.check_capture(),
                actionFunction=lambda sim: setattr(
                    self.satellite, "requires_retasking", True
                ),
                terminal=self.satellite.variable_interval,
            )
        return target.id

    def set_action(self, action: int, prev_action_key=None) -> str:
        """Decode the same ordered snapshot used to construct observations."""
        if not 0 <= action < self.n_actions:
            raise ValueError("RSO imaging action index out of range.")
        snapshot = getattr(self.satellite, "_rso_candidate_snapshot", None)
        if snapshot is None:
            snapshot = candidate_snapshot(self.satellite, self.n_actions)
        if (
            snapshot.time != self.simulator.sim_time
            or snapshot.revision != self.satellite.rso_scenario.revision
        ):
            # Attribute the warning to the caller selecting this action.
            warnings.warn(
                "RSO candidates changed since the observation; attempting the "
                "original target ID. Obtain a fresh observation for current candidates.",
                stacklevel=2,
            )
        if len(snapshot.targets) < self.n_actions:
            raise ValueError("RSO snapshot does not contain all action candidates.")
        target = snapshot.targets[action]
        if target is None:
            raise ValueError("RSO imaging action selected an empty candidate slot.")
        # Preserve the observed slot's ID while using its current episode binding.
        try:
            target = self.satellite.rso_scenario.targets_by_id[target.id]
        except KeyError as error:
            raise ValueError(
                "The observed RSO target is no longer in the scenario."
            ) from error
        return self._task(target)

    def set_action_override(self, action, prev_action_key=None) -> str:
        """Task an explicitly identified target without parsing spacecraft names."""
        target_id = action if isinstance(action, str) else getattr(action, "id", None)
        if target_id is None:
            raise TypeError("Expected an RSO target ID or RSOTarget definition.")
        try:
            target = self.satellite.rso_scenario.targets_by_id[target_id]
        except KeyError as error:
            raise ValueError(f"Unknown RSO target {target_id!r}.") from error
        return self._task(target)


__doc_title__ = "Space-to-Space RSO Imaging"
__all__ = ["ImageRSO"]
