"""Continuous actions set satellite behavior based on some continuous value."""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from gymnasium import spaces

from bsk_rl.act.actions import Action, ActionBuilder

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class ContinuousActionBuilder(ActionBuilder):
    def __init__(self, satellite: "Satellite") -> None:
        """Processes actions for a continuous action space.

        Args:
            satellite: Satellite to create actions for.
        """
        self.action_spec: list[ContinuousAction]
        super().__init__(satellite)
        assert len(self.action_spec) == 1, "Only one continuous action is supported."

    @property
    def _action(self) -> "ContinuousAction":
        return self.action_spec[0]

    @property
    def action_space(self) -> spaces.Box:
        """Continuous action space."""
        return self._action.space

    @property
    def action_description(self) -> list[str]:
        """Return a human-readable description of the continuous action space."""
        return self._action.action_description()

    def set_action(self, action: np.ndarray) -> None:
        """Activate the action by setting the continuous value."""
        self._action.set_action(action)


class ContinuousAction(Action):
    builder_type = ContinuousActionBuilder

    def __init__(self, name: str = "discrete_act") -> None:
        """Base class for actions with a continuous action space.

        Args:
            name: Name of the action.
        """
        super().__init__(name=name)

    @property
    @abstractmethod
    def space(self) -> spaces.Box:
        """Return the action space."""
        pass

    @property
    @abstractmethod
    def action_description(self) -> list[str]:
        """Return a description of the action space."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Activate an action by a continuous value."""
        pass


class ImpulsiveThrust(ContinuousAction):
    def __init__(
        self,
        name: str = "thrust_act",
        max_dv: float = float("inf"),
        max_drift_duration: float = float("inf"),
        fsw_action: Optional[str] = None,
    ) -> None:
        """Perform an impulsive thrust and drift for some duration.

        Args:
            name: Name of the action.
            max_dv: Maximum delta-V that can be applied. [m/s]
            max_drift_duration: Maximum duration to drift after applying the delta-V. [s]
            fsw_action: Name of the FSW action to activate during the drift period.
        """
        super().__init__(name)
        self.max_dv = max_dv
        self.max_drift_duration = max_drift_duration
        self.fsw_action = fsw_action

    @property
    def space(self) -> spaces.Box:
        """Return the action space."""
        return spaces.Box(
            low=np.array(
                [-self.max_dv, -self.max_dv, -self.max_dv, 2 * self.simulator.sim_rate]
            ),
            high=np.array(
                [self.max_dv, self.max_dv, self.max_dv, self.max_drift_duration]
            ),
            shape=(4,),
            dtype=np.float32,
        )

    @property
    def action_description(self) -> list[str]:
        """Description of the continuous action space."""
        return ["dV_N_x", "dV_N_y", "dV_N_z", "duration"]

    def set_action(self, action: np.ndarray) -> None:
        """Thrust the satellite with a given inertial delta-V and drift for some duration.

        Args:
            action: vector of [dV_N_x, dV_N_y, dV_N_z, duration] in [m/s] and [s]
        """
        assert len(action) == 4, "Action must have 4 elements."

        # Restrict to maximum delta-V
        dv_N = action[0:3]
        if np.linalg.norm(dv_N) > self.max_dv:
            self.satellite.logger.info(
                f"Thrust clamped from {np.linalg.norm(dv_N)} m/s to {self.max_dv} m/s."
            )
            dv_N = dv_N / np.linalg.norm(dv_N) * self.max_dv

        # Restrict to duration limits
        dt_desired = action[3]
        dt = np.clip(
            dt_desired, 2 * self.satellite.simulator.sim_rate, self.max_drift_duration
        )
        if dt != dt_desired:
            self.satellite.logger.warning(
                f"Requested drift duration {dt_desired} out of range. Clamping to {dt}."
            )

        self.satellite.logger.info(
            f"Thrusting with inertial dV {dv_N} with {dt} second drift."
        )
        self.satellite.fsw.action_impulsive_thrust(dv_N)
        self.satellite.update_timed_terminal_event(
            self.satellite.simulator.sim_time + dt
        )

        # Activate the FSW action for the drift period
        # TODO: should wait until action_impulsive_thrust is done if not immediate
        if self.fsw_action is not None:
            getattr(self.satellite.fsw, self.fsw_action)()
            self.satellite.logger.info(f"FSW action {self.fsw_action} activated.")


class ImpulsiveThrustHill(ImpulsiveThrust):
    def __init__(self, chief_name, *args, **kwargs):
        """Impulsive thrusts in the Hill frame.

        Args:
            chief_name: Chief to use for Hill frame.
            *args: Passed to ``ImpulsiveThrust``.
            **kwargs: Passed to ``ImpulsiveThrust``.
        """
        self.chief_name = chief_name
        super().__init__(*args, **kwargs)

    def reset_post_sim_init(self) -> None:
        """Connect to the chief satellite.

        :meta private:
        """
        self.chief = self.satellite.simulator.get_satellite(self.chief_name)

    def set_action(self, action: np.ndarray) -> None:
        """Activate the action by setting the continuous value."""
        dv_H = action[0:3]
        dt = action[3]

        NH = self.chief.dynamics.HN.T
        dv_N = NH @ dv_H

        super().set_action(np.concatenate((dv_N, [dt])))
