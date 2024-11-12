"""Discrete actions are indexable by integer."""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from gymnasium import spaces

from bsk_rl.act.actions import Action, ActionBuilder

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)


class DiscreteActionBuilder(ActionBuilder):
    def __init__(self, satellite: "Satellite") -> None:
        """Processes actions for a discrete action space.

        Args:
            satellite: Satellite to create actions for.
        """
        super().__init__(satellite)
        self.prev_action_key = None

    def reset_post_sim_init(self) -> None:
        """Log previous action key."""
        super().reset_post_sim_init()
        self.prev_action_key = None

    @property
    def action_space(self) -> spaces.Discrete:
        """Discrete action space."""
        return spaces.Discrete(sum([act.n_actions for act in self.action_spec]))

    @property
    def action_description(self) -> list[str]:
        """Return a list of strings corresponding to action names."""
        actions = []
        for act in self.action_spec:
            if act.n_actions == 1:
                actions.append(act.name)
            else:
                actions.extend([f"{act.name}_{i}" for i in range(act.n_actions)])
        return actions

    def set_action(self, action: int) -> None:
        """Sets the action based on the integer index.

        If the action is not an integer, the satellite will attempt to call ``set_action_override``
        for each action, in order, until one works.
        """
        self.satellite.disable_timed_terminal_event()
        if not np.issubdtype(type(action), np.integer):
            logger.warning(
                f"Action '{action}' is not an integer. Will attempt to use compatible set_action_override method."
            )
            for act in self.action_spec:
                try:
                    self.prev_action_key = act.set_action_override(
                        action, prev_action_key=self.prev_action_key
                    )
                    return
                except AttributeError:
                    pass
                except TypeError:
                    pass
            else:
                raise ValueError(
                    f"Action '{action}' is not an integer and no compatible set_action_override method found."
                )
        index = 0
        for act in self.action_spec:
            if index + act.n_actions > action:
                self.prev_action_key = act.set_action(
                    action - index, prev_action_key=self.prev_action_key
                )
                return
            index += act.n_actions
        else:
            raise ValueError(f"Action index {action} out of range.")


class DiscreteAction(Action):
    builder_type = DiscreteActionBuilder

    def __init__(self, name: str = "discrete_act", n_actions: int = 1):
        """Base class for discrete, integer-indexable actions.

        A discrete action may represent multiple indexed actions of the same type.

        Optionally, discrete actions may have a ``set_action_override`` function defined.
        If the action passed to the satellite is not an integer, the satellite will iterate
        over the ``action_spec`` and attempt to call ``set_action_override`` on each action
        until one is successful.

        Args:
            name: Name of the action.
            n_actions: Number of actions available.
        """
        super().__init__(name=name)
        self.n_actions = n_actions

    @abstractmethod
    def set_action(self, action: int, prev_action_key=None) -> str:
        """Activate an action by local index."""
        pass


class DiscreteFSWAction(DiscreteAction):
    def __init__(
        self,
        fsw_action,
        name=None,
        duration: Optional[float] = None,
        reset_task: bool = False,
    ):
        """Discrete action to task a flight software action function.

        This action executes a function of a :class:`~bsk_rl.sim.fsw.FSWModel`
        instance that takes no arguments, typically decorated with ``@action``.

        Args:
            fsw_action: Name of the flight software function to task.
            name: Name of the action. If not specified, defaults to the ``fsw_action`` name.
            duration: Duration of the action in seconds. Defaults to a large value so that
                the :class:`~bsk_rl.GeneralSatelliteTasking` ``max_step_duration``
                controls step length.
            reset_task: If true, reset the action if the previous action was the same.
                Generally, this parameter should be false to ensure realistic, continuous
                operation of satellite modes; however, some Basilisk modules may require
                frequent resetting for normal operation.
        """
        if name is None:
            name = fsw_action
        super().__init__(name=name, n_actions=1)
        self.fsw_action = fsw_action
        self.reset_task = reset_task
        if duration is None:
            duration = 1e9
        self.duration = duration

    def set_action(self, action: int, prev_action_key=None) -> str:
        """Activate the ``fsw_action`` function.

        Args:
            action: Should always be ``1``.
            prev_action_key: Previous action key.

        Returns:
            The name of the activated action.
        """
        assert action == 0
        self.satellite.logger.info(f"{self.name} tasked for {self.duration} seconds")
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + self.duration, info=f"for {self.fsw_action}"
        )

        if self.reset_task or prev_action_key != self.fsw_action:
            getattr(self.satellite.fsw, self.fsw_action)()

        return self.fsw_action


class Charge(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to enter a sun-pointing charging mode (:class:`~bsk_rl.sim.fsw.BasicFSWModel.action_charge`).

        Charging will only occur if the satellite is in sunlight.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_charge", name=name, duration=duration)


class Drift(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to disable all FSW tasks (:class:`~bsk_rl.sim.fsw.BasicFSWModel.action_drift`).

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_drift", name=name, duration=duration)


class Desat(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to desaturate reaction wheels (:class:`~bsk_rl.sim.fsw.BasicFSWModel.action_desat`).

        This action must be called repeatedly to fully desaturate the reaction wheels.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(
            fsw_action="action_desat", name=name, duration=duration, reset_task=True
        )


class Downlink(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to transmit data from the data buffer (:class:`~bsk_rl.sim.fsw.ImagingFSWModel.action_downlink`).

        If not in range of a ground station (defined in
        :class:`~bsk_rl.sim.world.GroundStationWorldModel`), no data will
        be downlinked.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_downlink", name=name, duration=duration)


class Scan(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to collect data from a :class:`~bsk_rl.scene.UniformNadirScanning` (:class:`~bsk_rl.sim.fsw.ContinuousImagingFSWModel.action_nadir_scan`).

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_nadir_scan", name=name, duration=duration)


class Image(DiscreteAction):
    def __init__(
        self,
        n_ahead_image: int,
        max_duration: Optional[float] = None,
        name: str = "action_image",
    ):
        """Actions to image upcoming target (:class:`~bsk_rl.sim.fsw.ImagingFSWModel.action_image`).

        Adds ``n_ahead_image`` actions to the action space, corresponding to the next
        ``n_ahead_image`` unimaged targets. The action may be unsuccessful if the target
        exits the satellite's field of regard before the satellite settles on the target
        and takes an image. The action with stop as soon as the image is successfully
        taken, or when the the target exits the field of regard.

        This action implements a ``set_action_override`` that allows a target to be tasked
        based on the target's ID string or the Target object.

        Args:
            n_ahead_image: Number of unimaged, along-track targets to consider.
            max_duration: Maximum time to task the action, in seconds. If ``None``, tasks until
                the end of the next opportunity.
            name: Action name.
        """
        from bsk_rl.sats import ImagingSatellite

        self.satellite: "ImagingSatellite"
        self.max_duration = max_duration
        super().__init__(name=name, n_actions=n_ahead_image)

    def image(
        self, target: Union[int, "Target", str], prev_action_key: Optional[str] = None
    ) -> str:
        """Task or retask a satellite for imaging a target.

        Args:
            target: Target to image.
            prev_action_key: Previous action key.

        :meta private:
        """
        target = self.satellite.parse_target_selection(target)
        if target.id != prev_action_key:
            self.satellite.task_target_for_imaging(
                target, max_duration=self.max_duration
            )
        else:
            self.satellite.enable_target_window(target, max_duration=self.max_duration)

        return target.id

    def set_action(self, action: int, prev_action_key: Optional[str] = None) -> str:
        """Image a target by local index.

        Args:
            action: Index of the target to image.
            prev_action_key: Previous action key.

        :meta_private:
        """
        self.satellite.logger.info(f"target index {action} tasked")
        return self.image(action, prev_action_key)

    def set_action_override(
        self, action: Union["Target", str], prev_action_key: Optional[str] = None
    ) -> str:
        """Image a target by target index, Target, or ID.

        Args:
            action: Target to image in the form of a Target object, target ID, or target index.
            prev_action_key: Previous action key.

        :meta_private:
        """
        return self.image(action, prev_action_key)


class Broadcast(DiscreteAction):
    def __init__(
        self,
        name: str = "action_broadcast",
        duration: Optional[float] = None,
    ):
        """Action to broadcast data to all satellites in the simulation.

        Should be used with a :class:`~bsk_rl.comm.BroadcastCommunication`-derived
        communication method to limit communication to broadcasting satellites.

        Args:
            name: Action name.
            duration: [s] Time to idle before communicating.
        """
        super().__init__(name=name, n_actions=1)
        if duration is None:
            duration = 1e9
        self.duration = duration

    def reset_post_sim_init(self) -> None:
        """Log previous action key."""
        super().reset_post_sim_init()
        self.broadcast_pending = False

    def set_action(self, action: int, prev_action_key=None) -> str:
        assert action == 0
        self.broadcast_pending = True
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + self.duration, info=f"for broadcast"
        )
        return self.name


__doc_title__ = "Discrete Backend"
__all__ = ["DiscreteActionBuilder"]
