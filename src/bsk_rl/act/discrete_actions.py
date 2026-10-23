"""Discrete actions are indexable by integer."""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from Basilisk.utilities import macros as mc
from gymnasium import spaces

from bsk_rl.act.actions import Action, ActionBuilder

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.scene.targets import Target

logger = logging.getLogger(__name__)


def clean_action(action: Any, hybrid: bool = False) -> tuple[Any, Optional[float]]:
    """Normalize an action to its selection and optional continuous parameter.

    Args:
        action: A selection (integer index, target ID, or Target), or a
            ``(selection, parameter)`` tuple. The parameter may be a scalar or
            an array-like value with shape ``()`` or ``(1,)``.
        hybrid: Whether to retain the continuous parameter. Otherwise, discard it.

    Returns:
        The selection unchanged and the parameter as a Python float, or ``None``
        when absent, explicitly ``None``, or hybrid mode is disabled. The selected
        action defines its meaning: a mode duration or desired capture offset in
        seconds.

    Raises:
        ValueError: If a tuple does not contain two items, or a retained parameter
            has an unsupported shape or cannot be converted to a float.
        TypeError: If the parameter's type does not support conversion to a float.
    """
    selection, value = action if isinstance(action, tuple) else (action, None)

    if not hybrid or value is None:
        return selection, None

    value = np.asarray(value)
    if value.shape not in ((), (1,)):
        raise ValueError("Hybrid action requires a scalar or one-element parameter.")

    return selection, float(value.item())


class DiscreteActionBuilder(ActionBuilder):
    def __init__(self, satellite: "Satellite") -> None:
        """Processes actions for a discrete action space.

        Args:
            satellite: Satellite to create actions for.
        """
        super().__init__(satellite)
        self.prev_action_key = None
        self.hybrid = any([act.hybrid for act in self.action_spec])

    def reset_post_sim_init(self) -> None:
        """Clear the previous action key after simulator initialization."""
        super().reset_post_sim_init()
        self.prev_action_key = None

    @property
    def action_space(self) -> spaces.Discrete | spaces.Tuple:
        """Return a discrete space, with a continuous parameter for hybrid actions."""
        n_actions = sum([act.n_actions for act in self.action_spec])
        if self.hybrid:
            return spaces.Tuple(
                (spaces.Discrete(n_actions), spaces.Box(low=0, high=np.inf, shape=(1,)))
            )
        return spaces.Discrete(n_actions)

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

    def set_action(self, action: Union[int, tuple[int, float]]) -> None:
        """Set an action by global index, with an optional continuous parameter.

        If the action is not an integer, the satellite will attempt to call ``set_action_override``
        for each action, in order, until one works.

        Hybrid actions receive ``(local_index, parameter)``. Other actions receive
        only their local index, even when the overall action space is hybrid.
        """
        action_index, parameter = clean_action(action, hybrid=self.hybrid)
        self.satellite.disable_timed_terminal_event()
        if not np.issubdtype(type(action_index), np.integer):
            logger.warning(
                f"Action '{action_index}' is not an integer. Will attempt to use compatible set_action_override method."
            )
            for act in self.action_spec:
                payload = (action_index, parameter) if act.hybrid else action_index
                try:
                    self.prev_action_key = act.set_action_override(
                        payload, prev_action_key=self.prev_action_key
                    )
                    return
                except AttributeError:
                    pass
                except TypeError:
                    pass
            else:
                raise ValueError(
                    f"Action '{action_index}' is not an integer and no compatible set_action_override method found."
                )
        offset = 0
        for act in self.action_spec:
            if offset <= action_index < offset + act.n_actions:
                local_index = action_index - offset
                payload = (local_index, parameter) if act.hybrid else local_index
                self.prev_action_key = act.set_action(
                    payload, prev_action_key=self.prev_action_key
                )
                return
            offset += act.n_actions
        raise ValueError(f"Action index {action_index} out of range.")


class DiscreteAction(Action):
    builder_type = DiscreteActionBuilder

    def __init__(
        self, name: str = "discrete_act", n_actions: int = 1, hybrid: bool = False
    ):
        """Base class for discrete, integer-indexable actions.

        A discrete action may represent multiple indexed actions of the same type.

        Optionally, discrete actions may have a ``set_action_override`` function defined.
        If the action passed to the satellite is not an integer, the satellite will iterate
        over the ``action_spec`` and attempt to call ``set_action_override`` on each action
        until one is successful.

        Args:
            name: Name of the action.
            n_actions: Number of actions available.
            hybrid: If true, accept ``(selection, parameter)`` actions. The action
                subclass defines the meaning of the continuous parameter.
        """
        super().__init__(name=name)
        self.n_actions = n_actions
        self.hybrid = hybrid

    @abstractmethod
    def set_action(
        self, action: Union[int, tuple[int, float]], prev_action_key=None
    ) -> str:
        """Activate an action by local index, optionally with a continuous parameter."""
        pass


class DiscreteFSWAction(DiscreteAction):
    def __init__(
        self,
        fsw_action,
        name=None,
        duration: Optional[float] = None,
        reset_task: bool = False,
        hybrid: bool = False,
    ):
        """Discrete action to task a flight software action function.

        This action executes a function of a :class:`~bsk_rl.sim.fsw.FSWModelABC`
        instance that takes no arguments, typically decorated with ``@action``.

        Args:
            fsw_action: Name of the flight software function to task.
            name: Name of the action. If not specified, defaults to the ``fsw_action`` name.
            duration: Duration of the action in seconds, also used as a cap on a
                hybrid action's requested duration. At least one simulator time step
                is scheduled, even if this cap is smaller. Defaults to a large value so that
                the :class:`~bsk_rl.GeneralSatelliteTasking` ``max_step_duration``
                controls step length.
            reset_task: If true, reset the action if the previous action was the same.
                Generally, this parameter should be false to ensure realistic, continuous
                operation of satellite modes; however, some Basilisk modules may require
                frequent resetting for normal operation.
            hybrid: If true, the action duration can be set with the action.
        """
        if name is None:
            name = fsw_action
        super().__init__(name=name, n_actions=1, hybrid=hybrid)
        self.fsw_action = fsw_action
        self.reset_task = reset_task
        if duration is None:
            duration = 1e9
        self.duration = duration

    def set_action(
        self, action: Union[int, tuple[int, float]], prev_action_key=None
    ) -> str:
        """Activate the ``fsw_action`` function.

        Args:
            action: Local index ``0``, or ``(0, duration)`` in hybrid mode, where
                duration is in seconds and is capped by the configured duration.
            prev_action_key: Previous action key.

        Returns:
            The name of the activated action.
        """
        action_index, requested_duration = clean_action(action, hybrid=self.hybrid)
        assert action_index == 0
        action_duration = (
            np.min([self.duration, requested_duration])
            if requested_duration is not None
            else self.duration
        )
        action_duration = max(self.simulator.sim_rate, action_duration)
        self.satellite.logger.info(f"{self.name} tasked for {action_duration} seconds")
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + action_duration, info=f"for {self.fsw_action}"
        )

        if self.reset_task or prev_action_key != self.fsw_action:
            getattr(self.satellite.fsw, self.fsw_action)()

        return self.fsw_action


class Charge(DiscreteFSWAction):
    def __init__(
        self,
        name: Optional[str] = None,
        duration: Optional[float] = None,
        hybrid: bool = False,
    ):
        """Action to enter a sun-pointing charging mode (:class:`~bsk_rl.sim.fsw.BasicFSWModel.action_charge`).

        Charging will only occur if the satellite is in sunlight.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
            hybrid: Whether to use a hybrid action space.
        """
        super().__init__(
            fsw_action="action_charge", name=name, duration=duration, hybrid=hybrid
        )


class Drift(DiscreteFSWAction):
    def __init__(
        self,
        name: Optional[str] = None,
        duration: Optional[float] = None,
        hybrid: bool = False,
    ):
        """Action to disable all FSW tasks (:class:`~bsk_rl.sim.fsw.BaseFSWModel.action_drift`).

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
            hybrid: Whether to use a hybrid action space.
        """
        super().__init__(
            fsw_action="action_drift", name=name, duration=duration, hybrid=hybrid
        )


class NadirPoint(DiscreteFSWAction):
    def __init__(
        self,
        name: Optional[str] = None,
        duration: Optional[float] = None,
        hybrid: bool = False,
    ):
        """Action to point nadir (:class:`~bsk_rl.sim.fsw.BaseFSWModel.action_nadir_point`).

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
            hybrid: Whether to use a hybrid action space.
        """
        super().__init__(
            fsw_action="action_nadir_point", name=name, duration=duration, hybrid=hybrid
        )


class Desat(DiscreteFSWAction):
    def __init__(
        self,
        name: Optional[str] = None,
        duration: Optional[float] = None,
        hybrid: bool = False,
    ):
        """Action to desaturate reaction wheels (:class:`~bsk_rl.sim.fsw.BasicFSWModel.action_desat`).

        This action must be called repeatedly to fully desaturate the reaction wheels.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
            hybrid: Whether to use a hybrid action space.
        """
        super().__init__(
            fsw_action="action_desat",
            name=name,
            duration=duration,
            reset_task=True,
            hybrid=hybrid,
        )


class Downlink(DiscreteFSWAction):
    def __init__(
        self,
        name: Optional[str] = None,
        duration: Optional[float] = None,
        hybrid: bool = False,
    ):
        """Action to transmit data from the data buffer (:class:`~bsk_rl.sim.fsw.ImagingFSWModel.action_downlink`).

        If not in range of a ground station (defined in
        :class:`~bsk_rl.sim.world.GroundStationWorldModel`), no data will
        be downlinked.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
            hybrid: Whether to use a hybrid action space.
        """
        super().__init__(
            fsw_action="action_downlink", name=name, duration=duration, hybrid=hybrid
        )


class Scan(DiscreteFSWAction):
    def __init__(
        self,
        name: Optional[str] = None,
        duration: Optional[float] = None,
        hybrid: bool = False,
    ):
        """Action to collect data from a :class:`~bsk_rl.scene.UniformNadirScanning` (:class:`~bsk_rl.sim.fsw.ContinuousImagingFSWModel.action_nadir_scan`).

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
            hybrid: Whether to use a hybrid action space.
        """
        super().__init__(
            fsw_action="action_nadir_scan", name=name, duration=duration, hybrid=hybrid
        )


class Image(DiscreteAction):
    def __init__(
        self,
        n_ahead_image: int,
        max_duration: Optional[float] = None,
        name: str = "action_image",
        hybrid: bool = False,
    ):
        """Actions to image upcoming target (:class:`~bsk_rl.sim.fsw.ImagingFSWModel.action_image`).

        Adds ``n_ahead_image`` actions to the action space, corresponding to the next
        ``n_ahead_image`` unimaged targets. The action may be unsuccessful if the target
        exits the satellite's field of regard before the satellite settles on the target
        and takes an image. The action will stop as soon as the image is successfully
        taken, or when the target exits the field of regard.

        This action implements a ``set_action_override`` that allows a target to be tasked
        based on the target's ID string or the Target object.

        Args:
            n_ahead_image: Number of unimaged, along-track targets to consider.
            max_duration: Maximum time to task the action, in seconds. If ``None``, tasks until
                the end of the next opportunity.
            name: Action name.
            hybrid: Whether to accept ``(target_index, imaging_time)`` actions.
                The continuous parameter is the desired capture offset from the
                current simulation time in seconds, not an exposure duration.
        """
        from bsk_rl.sats import ImagingSatellite

        self.satellite: "ImagingSatellite"
        self.max_duration = max_duration
        super().__init__(name=name, n_actions=n_ahead_image, hybrid=hybrid)

    def image(
        self,
        target: Union[int, "Target", str],
        prev_action_key: Optional[str] = None,
        imaging_time: Optional[float] = None,
    ) -> str:
        """Task or retask a satellite for imaging a target.

        Args:
            target: Target to image.
            prev_action_key: Previous action key.
            imaging_time: Desired imaging time relative to the current simulation time,
                in seconds. If ``None``, retain the controller's existing imaging time
                and tolerances. Warn if either time tolerance is enabled.

        :meta private:
        """
        target = self.satellite.parse_target_selection(target)

        time_tolerance_upper = mc.NANO2SEC * getattr(
            self.satellite.fsw.insControl, "timeToleranceUpper", 0
        )
        if imaging_time is None and (
            getattr(self.satellite.fsw.insControl, "timeToleranceLower", 0) > 0
            or time_tolerance_upper > 0
        ):
            self.satellite.logger.warning(
                "No imaging_time was provided while imaging time tolerances are enabled. "
                "The controller's existing imagingTime and tolerances are retained.",
            )
        max_duration = self.max_duration
        if imaging_time is not None and time_tolerance_upper > 0.0:
            max_duration = imaging_time + time_tolerance_upper
            if self.max_duration is not None:
                max_duration = np.min([self.max_duration, max_duration])
            max_duration = max(self.simulator.sim_rate, max_duration)
        timing = []
        if imaging_time is not None:
            timing.append(
                f"requested capture at {self.simulator.sim_time + imaging_time:.1f}"
            )
        if max_duration is not None:
            timing.append(f"maximum duration {max_duration:.1f} seconds")
        if timing:
            self.satellite.logger.info(f"{target} imaging timing: " + "; ".join(timing))
        if target.id != prev_action_key:
            kwargs = {}
            if imaging_time is not None:
                kwargs["imaging_time"] = imaging_time
            self.satellite.task_target_for_imaging(
                target, max_duration=max_duration, **kwargs
            )
        else:
            self.satellite.enable_target_window(target, max_duration=max_duration)
            if imaging_time is not None:
                self.satellite.update_imaging_time(imaging_time=imaging_time)

        return target.id

    def set_action(
        self,
        action: Union[int, tuple[int, float]],
        prev_action_key: Optional[str] = None,
    ) -> str:
        """Image a target by local index.

        Args:
            action: Local target index, or ``(index, imaging_time)`` in hybrid
                mode. The imaging time is a desired capture offset in seconds
                from the current simulation time.
            prev_action_key: Previous action key.

        :meta private:
        """
        target_index, imaging_time = clean_action(action, hybrid=self.hybrid)
        self.satellite.logger.info(f"target index {target_index} tasked")
        return self.image(target_index, prev_action_key, imaging_time=imaging_time)

    def set_action_override(
        self,
        action: Union[Union["Target", str], tuple[Union["Target", str], float]],
        prev_action_key: Optional[str] = None,
    ) -> str:
        """Image a target by target index, Target, or ID.

        Args:
            action: Target object, target ID, or target index. In hybrid mode,
                also accepts ``(target, imaging_time)``, where imaging time is a
                desired capture offset from the current simulation time in seconds.
            prev_action_key: Previous action key.

        :meta private:
        """
        target, imaging_time = clean_action(action, hybrid=self.hybrid)
        return self.image(target, prev_action_key, imaging_time=imaging_time)


class Broadcast(DiscreteAction):
    def __init__(
        self,
        name: str = "action_broadcast",
        duration: Optional[float] = None,
        hybrid: bool = False,
    ):
        """Action to broadcast data to all satellites in the simulation.

        Should be used with a :class:`~bsk_rl.comm.BroadcastCommunication`-derived
        communication method to limit communication to broadcasting satellites.

        Args:
            name: Action name.
            duration: [s] Time to idle before communicating, also used as a cap
                on a hybrid action's requested duration. At least one simulator
                time step is scheduled, even if this cap is smaller.
            hybrid: Whether to use the hybrid action format.
        """
        super().__init__(name=name, n_actions=1, hybrid=hybrid)
        if duration is None:
            duration = 1e9
        self.duration = duration

    def reset_post_sim_init(self) -> None:
        """Clear pending broadcasts after simulator initialization."""
        super().reset_post_sim_init()
        self.broadcast_pending = False

    def set_action(
        self, action: Union[int, tuple[int, float]], prev_action_key=None
    ) -> str:
        """Schedule a broadcast using local index ``0`` or ``(0, duration_seconds)``."""
        action_index, requested_duration = clean_action(action, hybrid=self.hybrid)
        assert action_index == 0
        self.broadcast_pending = True
        action_duration = (
            np.min([self.duration, requested_duration])
            if requested_duration is not None
            else self.duration
        )
        action_duration = max(self.simulator.sim_rate, action_duration)
        self.satellite.logger.info(
            f"{self.name} tasked for {action_duration:.1f} seconds"
        )
        self.satellite.update_timed_terminal_event(
            self.simulator.sim_time + action_duration, info="for broadcast"
        )
        return self.name


__doc_title__ = "Discrete Backend"
__all__ = ["DiscreteActionBuilder"]
