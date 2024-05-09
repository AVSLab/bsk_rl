"""Satellite action types can be used to add actions to an agent.

To configure the observation, set the ``action_spec`` attribute of a
:class:`~bsk_rl.env.scenario.satellites.Satellite` subclass. For example:

.. code-block:: python

    class MyActionSatellite(Satellite):
        action_spec = [
            Charge(duration=60.0),
            Desat(duration=30.0),
            Downlink(duration=60.0),
            Image(n_ahead_image=10),
        ]

Actions in an ``action_spec`` should all be of the same subclass of :class:`Action`. The
following actions are currently available:

Discrete Actions: :class:`DiscreteAction`
-----------------------------------------
For integer-indexable, discrete actions.

+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| **Action**                 |**Count**| **Description**                                                                                       |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`DiscreteFSWAction` | 1       | Call an arbitrary ``@action`` decorated function in the :class:`~bsk_rl.env.simulation.fsw.FSWModel`. |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Charge`            | 1       | Point the solar panels at the sun.                                                                    |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Drift`             | 1       | Do nothing.                                                                                           |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Desat`             | 1       | Desaturate the reaction wheels with RCS thrusters. Needs to be called multiple times.                 |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Downlink`          | 1       | Downlink data to any ground station that is in range.                                                 |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+
| :class:`Image`             | ≥1      | Image one of the next ``N`` upcoming, unimaged targets once in range.                                 |
+----------------------------+---------+-------------------------------------------------------------------------------------------------------+

"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.env.types import Satellite, Simulator

import numpy as np
from gymnasium import spaces

from bsk_rl.env.scenario.environment_features import Target
from bsk_rl.utils.functional import AbstractClassProperty, bind, configurable


def select_action_builder(satellite: "Satellite") -> "ActionBuilder":
    """Identify the proper action builder based on a satellite's action spec.

    Args:
        satellite: Satellite to build actions for.

    Returns:
        action builder of the appropriate type

    :meta private:
    """
    builder_types = [spec.builder_type for spec in satellite.action_spec]
    if all([builder_type == builder_types[0] for builder_type in builder_types]):
        return builder_types[0](satellite)
    else:
        raise NotImplementedError("Heterogenous action builders not supported.")


class ActionBuilder(ABC):
    """:meta private:"""

    def __init__(self, satellite: "Satellite") -> None:
        self.satellite = satellite
        self.simulator: "Simulator"
        self.action_spec = deepcopy(self.satellite.action_spec)
        for act in self.action_spec:
            act.link_satellite(self.satellite)

    def reset_post_sim(self) -> None:
        """Perform any once-per-episode setup."""
        self.simulator = self.satellite.simulator  # already a proxy
        for act in self.action_spec:
            act.link_simulator(self.simulator)  # already a proxy
            act.reset_post_sim()

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        """Return the action space."""
        pass

    @property
    @abstractmethod
    def action_description(self) -> Any:
        """Return a description of the action space."""
        pass

    @abstractmethod
    def set_action(self, action: Any) -> None:
        """Set the action to be taken."""
        pass


class DiscreteActionBuilder(ActionBuilder):
    """:meta private:"""

    def __init__(self, satellite: "Satellite") -> None:
        super().__init__(satellite)
        self.prev_action_key = None

    def reset_post_sim(self) -> None:
        super().reset_post_sim()
        self.prev_action_key = None

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(sum([act.n_actions for act in self.action_spec]))

    @property
    def action_description(self) -> list[str]:
        actions = []
        for act in self.action_spec:
            if act.n_actions == 1:
                actions.append(act.name)
            else:
                actions.extend([f"{act.name}_{i}" for i in range(act.n_actions)])
        return actions

    def set_action(self, action: int) -> None:
        self.satellite._disable_timed_terminal_event()
        if not np.issubdtype(type(action), np.integer):
            logging.warning(
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


class Action(ABC):
    builder_type: type[ActionBuilder] = AbstractClassProperty()  #: :meta private:

    def __init__(self, name: str = "act") -> None:
        """Base class for all actions.

        Args:
            name: Name of the action.
        """
        self.name = name
        self.satellite: "Satellite"
        self.simulator: "Simulator"

    def link_satellite(self, satellite: "Satellite") -> None:
        """Link the action to a satellite.

        Args:
            satellite: Satellite to link to

        :meta private:
        """
        self.satellite = satellite  # already a proxy

    def link_simulator(self, simulator: "Simulator") -> None:
        """Link the action to a simulator.

        Args:
            simulator: Simulator to link to

        :meta private:
        """
        self.simulator = simulator  # already a proxy

    def reset_post_sim(self) -> None:  # pragma: no cover
        """Perform any once-per-episode setup."""
        pass

    @abstractmethod
    def set_action(self, action: Any) -> None:  # pragma: no cover
        """Execute code to perform an action."""
        pass


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

        This action executes a function of a :class:`~bsk_rl.env.simulation.fsw.FSWModel`
        instance that takes no arguments, typically decorated with ``@action``.

        Args:
            fsw_action: Name of the flight software function to task.
            name: Name of the action. If not specified, defaults to the ``fsw_action`` name.
            duration: Duration of the action in seconds. Defaults to a large value so that
                the :class:`~bsk_rl.env.gym_env.GeneralSatelliteTasking` ``max_step_duration``
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
        self.satellite.log_info(f"{self.name} tasked for {self.duration} seconds")
        self.satellite._update_timed_terminal_event(
            self.simulator.sim_time + self.duration, info=f"for {self.fsw_action}"
        )

        if self.reset_task or prev_action_key != self.fsw_action:
            getattr(self.satellite.fsw, self.fsw_action)()

        return self.fsw_action


class Charge(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to enter a sun-pointing charging mode (:class:`~bsk_rl.env.simulation.fsw.BasicFSWModel.action_charge`).

        Charging will only occur if the satellite is in sunlight.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_charge", name=name, duration=duration)


class Drift(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to disable all FSW tasks (:class:`~bsk_rl.env.simulation.fsw.BasicFSWModel.action_drift`).

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_drift", name=name, duration=duration)


class Desat(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to desaturate reaction wheels (:class:`~bsk_rl.env.simulation.fsw.BasicFSWModel.action_desat`).

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
        """Action to transmit data from the data buffer (:class:`~bsk_rl.env.simulation.fsw.ImagingFSWModel.action_downlink`).

        If not in range of a ground station (defined in
        :class:`~bsk_rl.env.simulation.environment.GroundStationEnvModel`), no data will
        be downlinked.

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_downlink", name=name, duration=duration)


class Scan(DiscreteFSWAction):
    def __init__(self, name: Optional[str] = None, duration: Optional[float] = None):
        """Action to collect data from a :class:`~bsk_rl.env.scenario.environment_features.UniformNadirFeature` (:class:`~bsk_rl.env.simulation.fsw.ContinuousImagingFSWModel.action_nadir_scan`).

        Args:
            name: Action name.
            duration: Time to task action, in seconds.
        """
        super().__init__(fsw_action="action_nadir_scan", name=name, duration=duration)


class Image(DiscreteAction):
    def __init__(
        self,
        n_ahead_image: int,
        name: str = "action_image",
    ):
        """Actions to image upcoming target (:class:`~bsk_rl.env.simulation.fsw.ImagingFSWModel.action_image`).

        Adds `n_ahead_image` actions to the action space, corresponding to the next
        `n_ahead_image` unimaged targets. The action may be unsuccessful if the target
        exits the satellite's field of regard before the satellite settles on the target
        and takes an image. The action with stop as soon as the image is successfully
        taken, or when the the target exits the field of regard.

        This action implements a `set_action_override` that allows a target to be tasked
        based on the target's ID string or the Target object.

        Args:
            name: Action name.
            n_ahead_image: Number of unimaged, along-track targets to consider.
        """
        from bsk_rl.env.scenario.satellites import ImagingSatellite

        self.satellite: "ImagingSatellite"
        super().__init__(name=name, n_actions=n_ahead_image)

    def image(
        self, target: Union[int, Target, str], prev_action_key: Optional[str] = None
    ) -> str:
        """:meta private:"""
        target = self.satellite.parse_target_selection(target)
        if target.id != prev_action_key:
            self.satellite.task_target_for_imaging(target)
        else:
            self.satellite.enable_target_window(target)

        return target.id

    def set_action(self, action: int, prev_action_key: Optional[str] = None) -> str:
        """Image a target by local index.

        Args:
            action: Index of the target to image.
            prev_action_key: Previous action key.

        :meta_private:
        """
        self.satellite.log_info(f"target index {action} tasked")
        return self.image(action, prev_action_key)

    def set_action_override(
        self, action: Union[Target, str], prev_action_key: Optional[str] = None
    ) -> str:
        """Image a target by target index, Target, or ID.

        Args:
            target: Target to image.
            prev_action_key: Previous action key.

        :meta_private:
        """
        return self.image(action, prev_action_key)
