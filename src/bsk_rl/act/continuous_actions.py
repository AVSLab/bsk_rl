"""Continuous actions for satellites with a single Box action space."""

import logging
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces

from bsk_rl.act.actions import Action, ActionBuilder

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class ContinuousActionBuilder(ActionBuilder):
    """Builder for a single continuous Box action."""

    def __init__(self, satellite: "Satellite") -> None:
        """Processes actions for a continuous action space.

        Args:
            satellite: Satellite to create actions for.
        """
        super().__init__(satellite)
        assert (
            len(self.action_spec) == 1
        ), "Only a single continuous action is supported."

    @property
    def action_space(self) -> spaces.Box:
        """Continuous action space."""
        return self.action_spec[0].space

    @property
    def action_description(self) -> list[str]:
        """Return a list with the action name."""
        return [self.action_spec[0].name]

    def set_action(self, action) -> None:
        """Pass the continuous action to the action spec."""
        self.satellite.disable_timed_terminal_event()
        self.action_spec[0].set_action(
            np.atleast_1d(np.asarray(action, dtype=float))
        )


class ContinuousAction(Action):
    """Base class for single continuous Box actions."""

    builder_type = ContinuousActionBuilder

    def __init__(self, name: str = "continuous_act") -> None:
        """Base class for continuous actions.

        Args:
            name: Name of the action.
        """
        super().__init__(name=name)

    @property
    def space(self) -> spaces.Box:
        """Action space of the action."""
        raise NotImplementedError

    def set_action(self, action: np.ndarray) -> None:
        """Execute the action, given a 1-d array."""
        raise NotImplementedError


class Sweep(ContinuousAction):
    """Command a target cross-track roll angle for a sweeping satellite.

    The action is a single value in [-1, 1], denormalized to a roll target
    in [-roll_max, +roll_max] and passed to
    :class:`~bsk_rl.sim.fsw.SweepFSWModel.action_sweep`, which rate-limits
    the reference slew. Requires the satellite to use
    :class:`~bsk_rl.sim.fsw.SweepFSWModel`.

    With ``control_yaw`` the action gains a second value: a yaw target in
    [-yaw_max, +yaw_max] about the boresight, orienting the scan line on the
    ground. Pair it with a scenario that derives the swath from the flown
    attitude (``SweepRiskField(attitude_swath=True)``), or the yaw has no
    effect on the reward.
    """

    def __init__(
        self,
        roll_max: float = np.radians(30.0),
        duration: float = 120.0,
        name: str = "sweep",
        *,
        control_yaw: bool = False,
        yaw_max: float = np.radians(90.0),
    ) -> None:
        """Continuous sweep action.

        Args:
            roll_max: [rad] Roll angle corresponding to action = +/-1.
            duration: [s] Decision interval; the reference slews toward the
                target (rate-limited in FSW) over this time.
            name: Name of the action.
            control_yaw: Add a second action dimension commanding the yaw
                about the boresight (rate-limited in FSW like the roll).
            yaw_max: [rad] Yaw angle corresponding to action[1] = +/-1.
        """
        super().__init__(name=name)
        self.roll_max = roll_max
        self.duration = duration
        self.control_yaw = control_yaw
        self.yaw_max = yaw_max

    @property
    def space(self) -> spaces.Box:
        """Normalized roll (and optionally yaw) target in [-1, 1]."""
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,) if self.control_yaw else (1,),
            dtype=np.float32,
        )

    def set_action(self, action: np.ndarray) -> None:
        """Denormalize the target(s) and task the FSW sweep."""
        roll_target = float(np.clip(action[0], -1.0, 1.0)) * self.roll_max
        if self.control_yaw:
            yaw_target = float(np.clip(action[1], -1.0, 1.0)) * self.yaw_max
            self.satellite.log_info(
                f"sweep tasked: roll target {np.degrees(roll_target):.1f} deg, "
                f"yaw target {np.degrees(yaw_target):.1f} deg "
                f"for {self.duration:.0f} s"
            )
            self.satellite.fsw.action_sweep(
                roll_target, self.duration, yaw_target=yaw_target
            )
        else:
            self.satellite.log_info(
                f"sweep tasked: roll target {np.degrees(roll_target):.1f} deg "
                f"for {self.duration:.0f} s"
            )
            self.satellite.fsw.action_sweep(roll_target, self.duration)
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + self.duration, info="for sweep"
        )


__doc_title__ = "Continuous Actions"
__all__ = ["ContinuousActionBuilder", "ContinuousAction", "Sweep"]
