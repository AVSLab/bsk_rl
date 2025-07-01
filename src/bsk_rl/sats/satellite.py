"""Satellites are the agents in the environment."""

import inspect
import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, Optional, Union
from weakref import proxy

import numpy as np
from Basilisk.utilities import macros
from deprecated import deprecated
from gymnasium import spaces

from bsk_rl.act.actions import select_action_builder
from bsk_rl.obs.observations import ObservationBuilder
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils import vizard
from bsk_rl.utils.functional import (
    AbstractClassProperty,
    Resetable,
    collect_default_args,
    safe_dict_merge,
    valid_func_name,
)
from bsk_rl.utils.orbital import TrajectorySimulator

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.act import Action
    from bsk_rl.data.base import DataStore
    from bsk_rl.obs import Observation
    from bsk_rl.sim import Simulator


SatObs = Any
SatAct = Any


class Satellite(ABC, Resetable):
    """Abstract base class for satellites."""

    dyn_type: type["dyn.DynamicsModel"] = AbstractClassProperty()
    fsw_type: type["fsw.FSWModel"] = AbstractClassProperty()
    observation_spec: list["Observation"] = AbstractClassProperty()
    action_spec: list["Action"] = AbstractClassProperty()

    @classmethod
    def default_sat_args(cls, **kwargs) -> dict[str, Any]:
        """Compile default arguments for :class:`~bsk_rl.sim.dyn.DynamicsModel` and :class:`~bsk_rl.sim.fsw.FSWModel`, replacing those specified.

        Args:
            **kwargs: Arguments to override in the default arguments.

        Returns:
            Dictionary of arguments for simulation models.
        """
        defaults = collect_default_args(cls.dyn_type)
        defaults = safe_dict_merge(defaults, collect_default_args(cls.fsw_type))
        for name in dir(cls.fsw_type):
            if inspect.isclass(getattr(cls.fsw_type, name)) and issubclass(
                getattr(cls.fsw_type, name), fsw.Task
            ):
                defaults = safe_dict_merge(
                    defaults, collect_default_args(getattr(cls.fsw_type, name))
                )

        for k, v in kwargs.items():
            if k not in defaults:
                raise KeyError(f"{k} not a valid key for sat_args")
            defaults[k] = v
        return defaults

    def __init__(
        self,
        name: str,
        sat_args: Optional[dict[str, Any]] = None,
        obs_type=np.ndarray,
        dtype: np.dtype = np.float64,
        variable_interval: bool = True,
    ) -> None:
        """The base satellite agent class.

        Args:
            name: Identifier for satellite; does not need to be unique.
            sat_args: Arguments for :class:`~bsk_rl.sim.dyn.DynamicsModel` and
                :class:`~bsk_rl.sim.fsw.FSWModel` construction. Should be in the form of
                a dictionary with keys corresponding to the arguments of the constructor
                and values that are either the desired value or a function that takes no
                arguments and returns a randomized value.
            obs_type: Observation format for the satellite. The :class:`bsk_rl.obs.observations.ObservationBuilder`
                will convert the observation to this format.
            dtype: Data type for observation np vectors.
            variable_interval: Whether to stop the simulation at terminal events. If
                False, only the ``max_step_duration`` setting in :class:`~bsk_rl.GeneralSatelliteTasking`
                will stop the simulation.
        """
        self.name = name
        self.logger = logging.getLogger(__name__).getChild(self.name)
        if sat_args is None:
            sat_args = self.default_sat_args()
        self.sat_args_generator = self.default_sat_args(**sat_args)
        self.simulator: "Simulator"
        self.fsw: "fsw.FSWModel"
        self.dynamics: "dyn.DynamicsModel"
        self.data_store: "DataStore"
        self.requires_retasking: bool
        self.variable_interval = variable_interval
        self._timed_terminal_event_name = None
        self.observation_builder = ObservationBuilder(
            self, obs_type=obs_type, dtype=dtype
        )
        self.action_builder = select_action_builder(self)

    @property
    @deprecated(reason="Use satellite.name instead")
    def id(self) -> str:
        """Unique human-readable identifier."""
        return self.name

    def generate_sat_args(self, **kwargs) -> None:
        """Instantiate sat_args from any randomizers in provided sat_args.

        Args:
            **kwargs: Arguments to override in the default arguments.
        """
        self.sat_args = {
            k: v if not callable(v) else v() for k, v in self.sat_args_generator.items()
        }
        for k, v in kwargs.items():
            if k not in self.sat_args:
                raise KeyError(f"{k} not a valid key for sat_args")
            if np.any(self.sat_args[k] != v):
                self.logger.debug(
                    f"Overwriting {k}={self.sat_args[k]} in sat_args with {v}"
                )
            self.sat_args[k] = v

        self.logger.debug(f"Satellite initialized with {self.sat_args}")

    def reset_overwrite_previous(self) -> None:
        """Overwrite attributes from previous episode."""
        self.requires_retasking = True
        self._timed_terminal_event_name = None
        self._is_alive = True
        self.time_of_death = None
        self.observation_builder.reset_overwrite_previous()
        self.action_builder.reset_overwrite_previous()

    @vizard.visualize
    def create_vizard_data(self, color, vizSupport=None) -> None:
        """Create a location to store data to be passed to enableUnityVisualization."""
        self.vizard_color = color
        self.vizard_data = dict(
            spriteList=vizSupport.setSprite("SQUARE", color=color),
        )

    def reset_pre_sim_init(self) -> None:
        """Called during environment reset, before Basilisk simulation initialization."""
        self.trajectory = TrajectorySimulator(
            utc_init=self.sat_args["utc_init"],
            rN=self.sat_args["rN"],
            vN=self.sat_args["vN"],
            oe=self.sat_args["oe"],
            mu=self.sat_args["mu"],
        )
        self.observation_builder.reset_pre_sim_init()
        self.action_builder.reset_pre_sim_init()

    def set_simulator(self, simulator: "Simulator"):
        """Set the simulator for models.

        Called during simulator initialization.

        Args:
            simulator: Basilisk simulator

        :meta private:
        """
        self.simulator = proxy(simulator)

    def set_dynamics(self, dyn_rate: float) -> "dyn.DynamicsModel":
        """Create dynamics model; called during simulator initialization.

        Args:
            dyn_rate: rate for dynamics simulation [s]

        Returns:
            Satellite's dynamics model

        :meta private:
        """
        dynamics = self.dyn_type(self, dyn_rate, **self.sat_args)
        self.dynamics = proxy(dynamics)
        return dynamics

    def set_fsw(self, fsw_rate: float) -> "fsw.FSWModel":
        """Create flight software model; called during simulator initialization.

        Args:
            fsw_rate: rate for FSW simulation [s]

        Returns:
            Satellite's FSW model

        :meta private:
        """
        fsw = self.fsw_type(self, fsw_rate, **self.sat_args)
        self.fsw = proxy(fsw)
        return fsw

    def reset_during_sim_init(self) -> None:
        """Called during environment reset, during Basilisk simulation initialization."""
        self.observation_builder.reset_during_sim_init()
        self.action_builder.reset_during_sim_init()
        return super().reset_during_sim_init()

    def reset_post_sim_init(self) -> None:
        """Called during environment reset, after Basilisk simulation initialization."""
        self.observation_builder.reset_post_sim_init()
        self.action_builder.reset_post_sim_init()

    @property
    def observation_space(self) -> spaces.Space:
        """Observation space for single satellite, determined from observation.

        Returns:
            gymanisium observation space
        """
        return self.observation_builder.observation_space

    def get_obs(self) -> SatObs:
        """Construct the satellite's observation.

        Returns:
            satellite observation
        """
        return self.observation_builder.get_obs()

    @property
    def observation_description(self) -> Any:
        """Human-interpretable description of observation space."""
        return self.observation_builder.observation_description

    @property
    def action_space(self) -> spaces.Space:
        """Action space for single satellite.

        Returns:
            gymanisium action space
        """
        return self.action_builder.action_space

    def set_action(self, action: Any) -> None:
        """Enable certain processes in the simulator to command the satellite task.

        Args:
            action: Action to take, according to the :class:`action_spec`
        """
        self.action_builder.set_action(action)

    @property
    def action_description(self) -> Any:
        """Human-interpretable description of action space."""
        return self.action_builder.action_description

    def is_alive(self, log_failure=True) -> bool:
        """Check if the satellite is violating any aliveness requirements.

        Checks aliveness checkers in dynamics and FSW models.

        Returns:
            is_alive
        """
        if not self._is_alive:
            return False

        self._is_alive = self.dynamics.is_alive(
            log_failure=log_failure
        ) and self.fsw.is_alive(log_failure=log_failure)
        if not self._is_alive:
            self.record_death(self.simulator.sim_time)
        return self._is_alive

    def record_death(self, time: float) -> None:
        """Record the time of death of the satellite, if not already recorded.

        Args:
            time: Time of death [s]
        """
        if self.time_of_death is None:
            self.time_of_death = time

    @property
    def _satellite_command(self) -> str:
        """Generate string that refers to self in simBase."""
        return f"self.get_satellite('{self.name}')"

    def _info_command(self, info: str) -> str:
        """Generate command to log to info from an event.

        Args:
            info: information to log; cannot include `'` or `"`

        Returns:
            actionList action for simBase.createNewEvent
        """
        return self._satellite_command + f".logger.info('{info}')"

    @deprecated(reason="Use satellite.logger.info instead")
    def log_info(self, info: Any) -> None:
        """Record information at the current simulation time.

        :meta private:

        Args:
            info: Information to log
        """
        self.logger.info(f"{info}")

    @deprecated(reason="Use satellite.logger.warning instead")
    def log_warning(self, warning: Any) -> None:
        """Record warning at the current simulation time.

        :meta private:

        Args:
            warning: Warning to log
        """
        self.logger.warning(f"{warning}")

    def update_timed_terminal_event(
        self,
        t_close: float,
        info: str = "",
        extra_actions: Optional[Union[list[str], callable]] = None,
    ) -> None:
        """Create a simulator event that stops the simulation a certain time.

        Args:
            t_close: Termination time [s]
            info: Additional identifying info to log at terminal time
            extra_actions: Additional actions to perform at terminal time
        """
        self.disable_timed_terminal_event()
        self.logger.info(f"setting timed terminal event at {t_close:.1f}")

        # Create new timed terminal event
        self._timed_terminal_event_name = valid_func_name(
            f"timed_terminal_{t_close}_{self.name}"
        )

        def side_effect(sim):
            self.logger.info(f"timed termination at {t_close:.1f} " + info)
            self.requires_retasking = True
            if extra_actions is not None:
                if callable(extra_actions):
                    extra_actions(sim)
                elif isinstance(extra_actions, list):
                    for action in extra_actions:
                        exec(action, {"self": sim})

        self.simulator.createNewEvent(
            self._timed_terminal_event_name,
            macros.sec2nano(self.simulator.sim_rate),
            True,
            conditionFunction=lambda sim: sim.sim_time >= t_close,
            actionFunction=side_effect,
            terminal=self.variable_interval,
        )
        self.simulator.eventMap[self._timed_terminal_event_name].eventActive = True

    def disable_timed_terminal_event(self) -> None:
        """Turn off simulator termination due to :class:`update_timed_terminal_event`."""
        if (
            self._timed_terminal_event_name is not None
            and self._timed_terminal_event_name in self.simulator.eventMap
        ):
            self.simulator.delete_event(self._timed_terminal_event_name)
