"""Continuous actions set satellite behavior based on some continuous value."""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from gymnasium import spaces

from bsk_rl.act.actions import Action, ActionBuilder

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

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
        """Base class for discrete, integer-indexable actions.

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


class MagicThrust(ContinuousAction):
    # TODO set the fsw mode to carry out after action
    def __init__(
        self,
        name: str = "thrust_act",
        max_dv: float = float("inf"),
        fsw_action: Optional[str] = None,
    ) -> None:
        """Instantaneously change the satellite's velocity, and drift for some duration.

        TODO: Support specifying frame of thrust.

        Args:
            name: Name of the action.
            max_dv: Maximum delta-V that can be applied. [m/s]
        """
        super().__init__(name)
        self.max_dv = max_dv
        self.fsw_action = fsw_action

    @property
    def space(self) -> spaces.Box:
        """Return the action space."""
        return spaces.Box(
            low=np.array(
                [-self.max_dv, -self.max_dv, -self.max_dv, self.simulator.sim_rate]
            ),
            high=np.array([self.max_dv, self.max_dv, self.max_dv, 5700.0]),
            shape=(4,),
            dtype=np.float32,
        )

    @property
    def action_description(self) -> list[str]:
        """Description of the continuous action space."""
        return ["dV_N_x", "dV_N_y", "dV_N_z", "duration"]

    def set_action(self, action: np.ndarray) -> None:
        """Thrust the satellite with a given inertial delta-V and drift for some duration."""
        assert len(action) == 4, "Action must have 4 elements."
        dv_N = action[0:3]
        dt = action[3]

        self.satellite.log_info(
            f"Thrusting with inertial dV {dv_N} with {dt} second drift."
        )
        self.satellite.fsw.action_magic_thrust(dv_N)
        self.satellite.update_timed_terminal_event(
            self.satellite.simulator.sim_time + dt
        )

        # Activate the FSW action for the drift period
        getattr(self.satellite.fsw, self.fsw_action)()
        self.satellite.log_info(f"FSW action {self.fsw_action} activated.")
