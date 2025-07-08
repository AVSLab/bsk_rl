"""Gymnasium and PettingZoo environments for satellite tasking problems."""

import functools
import logging
import os
from copy import deepcopy
from time import time_ns
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, Union

import numpy as np
from gymnasium import Env, spaces
from pettingzoo.utils.env import AgentID, ParallelEnv

from bsk_rl.comm import CommunicationMethod, NoCommunication
from bsk_rl.data import GlobalReward, NoReward
from bsk_rl.data.composition import ComposedReward
from bsk_rl.sats import Satellite
from bsk_rl.scene import Scenario
from bsk_rl.sim import Simulator
from bsk_rl.sim.world import WorldModel
from bsk_rl.utils import logging_config, vizard

logger = logging.getLogger(__name__)


SatObs = TypeVar("SatObs")
SatAct = TypeVar("SatAct")
MultiSatObs = tuple[SatObs, ...]
MultiSatAct = Iterable[SatAct]
SatArgRandomizer = Callable[[list[Satellite]], dict[Satellite, dict[str, Any]]]

NO_ACTION = int(2**31) - 1


def is_no_action(action):
    """Check if the action is a no-action placeholder."""
    if action is None:
        return True
    if isinstance(action, (int, np.integer)) and action == NO_ACTION:
        return True
    if isinstance(action, (np.ndarray, list, tuple)) and np.allclose(action, NO_ACTION):
        return True
    return False


def no_action_like(action):
    """Generate an action that is the same type and shape as the no-action placeholder."""
    if isinstance(action, (int, np.integer)):
        return NO_ACTION
    return action * 0 + NO_ACTION  # Aggressively try to convert while retaining type


class GeneralSatelliteTasking(Env, Generic[SatObs, SatAct]):
    def __init__(
        self,
        satellites: Union[Satellite, list[Satellite]],
        scenario: Optional[Scenario] = None,
        rewarder: Optional[Union[GlobalReward, list[GlobalReward]]] = None,
        world_type: Optional[type[WorldModel]] = None,
        world_args: Optional[dict[str, Any]] = None,
        communicator: Optional[CommunicationMethod] = None,
        sat_arg_randomizer: Optional[SatArgRandomizer] = None,
        sim_rate: float = 1.0,
        max_step_duration: float = 1e9,
        failure_penalty: float = -1.0,
        time_limit: Union[float, Callable] = float("inf"),
        terminate_on_time_limit: bool = False,
        generate_obs_retasking_only: bool = False,
        dtype: Optional[np.dtype] = None,
        log_level: Union[int, str] = logging.WARNING,
        log_dir: Optional[str] = None,
        vizard_dir: Optional[str] = None,
        vizard_settings: Optional[dict[str, Any]] = None,
        render_mode=None,
    ) -> None:
        """A `Gymnasium <https://gymnasium.farama.org>`_ environment adaptable to a wide range satellite tasking problems.

        These problems involve satellite(s) being tasked to complete tasks and maintain
        aliveness. These tasks often include rewards for data collection. The environment
        can be configured for any collection of satellites, including heterogenous
        constellations. Other configurable aspects are the scenario (e.g.
        imaging targets), data collection and recording, and intersatellite
        communication of data.

        The state space is a tuple containing the state of each satellite. Actions are
        assigned as a tuple of actions, one per satellite.

        Args:
            satellites: Satellite(s) to be simulated. See :ref:`bsk_rl.sats`.
            scenario: Environment the satellite is acting in; contains information
                about targets, etc. See :ref:`bsk_rl.scene`.
            rewarder: Handles recording and rewarding for data collection towards
                objectives. Can be a single rewarder or a tuple of multiple rewarders.
                See :ref:`bsk_rl.data`.
            communicator: Manages communication between satellites. See :ref:`bsk_rl.comm`.
            sat_arg_randomizer: For correlated randomization of satellites arguments. Should
                be a function that takes a list of satellites and returns a dictionary that
                maps satellites to dictionaries of satellite model arguments to be overridden.
            world_type: Type of Basilisk world model to be constructed.
            world_args: Arguments for :class:`~bsk_rl.sim.world.WorldModel` construction.
                Should be in the form of a dictionary with keys corresponding to the
                arguments of the constructor and values that are either the desired value
                or a function that takes no arguments and returns a randomized value.
            sim_rate: [s] Rate for model simulation.
            max_step_duration: [s] Maximum time to propagate sim at a step. If
                satellites are using variable interval actions, the actual step duration
                will be less than or equal to this value. It is preferable to set durations
                in the actions themselves.
            failure_penalty: Reward for satellite failure. Should be nonpositive.
            time_limit: [s] Time at which to truncate the simulation. Can also be a function
                that takes no arguments and returns a float. This function will be called
                every time the environment is reset to randomize the time limit.
            terminate_on_time_limit: Send terminations signal time_limit instead of just
                truncation.
            generate_obs_retasking_only: If True, only generate observations for satellites
                that require retasking. All other satellites will receive an observation of
                zeros.
            dtype: Data type for satellite observations. If None, the data type specified
                in the satellite.
            log_level: Logging level for the environment. Default is ``WARNING``.
            log_dir: Directory to write logs to in addition to the console.
            vizard_dir: Path to save Vizard visualization files. If None, no Vizard-related
                modules will be imported.
            vizard_settings: Settings for Vizard visualization. Set in ``vizIstance.settings``.
                Additionally, the key ``vizard_rate`` can be set to the rate at which Vizard updates.
                Valid setting can be found `here <https://hanspeterschaub.info/basilisk/Vizard/vizardAdvanced/vizardSettings.html#id1>`_.
            render_mode: Unused.
        """
        self.seed = None
        self._configure_logging(log_level, log_dir)
        if vizard_dir is not None:
            vizard.VIZARD_PATH = vizard_dir
            if vizard_settings is not None:
                logger.warning(
                    "Vizard settings provided but Vizard is not enabled. Ignoring settings."
                )
        self.vizard_settings = vizard_settings if vizard_settings is not None else {}

        if isinstance(satellites, Satellite):
            satellites = [satellites]
        self.satellites = deepcopy(satellites)

        self.dtype = dtype
        while True:
            for satellite in self.satellites:
                if [sat.name for sat in self.satellites].count(satellite.name) > 1:
                    for i, sat_rename in enumerate(
                        [sat for sat in self.satellites if sat.name == satellite.name]
                    ):
                        new_name = f"{sat_rename.name}_{i}"
                        logger.warning(
                            f"Renaming satellite {sat_rename.name} to {new_name}"
                        )
                        sat_rename.name = new_name

                # Update satellite observation dtypes
                if self.dtype is not None:
                    satellite.observation_builder.dtype = self.dtype

            # Check if all satellite names are unique
            sat_names = [sat.name for sat in self.satellites]
            if len(sat_names) == len(set(sat_names)):
                break

        self.simulator: Simulator

        if sat_arg_randomizer is None:
            sat_arg_randomizer = lambda sats: {}
        self.sat_arg_randomizer = sat_arg_randomizer

        if scenario is None:
            scenario = Scenario()

        if world_type is None:
            world_type = self._minimum_world_model()
        self.world_type = world_type
        if world_args is None:
            world_args = self.world_type.default_world_args()
        self.world_args_generator = self.world_type.default_world_args(**world_args)

        self.scenario = deepcopy(scenario)
        self.scenario.link_satellites(self.satellites)

        rewarder = deepcopy(rewarder)
        if rewarder is None:
            rewarder = NoReward()
        if (
            isinstance(rewarder, Iterable)
            and not type(rewarder).__name__ == "MagicMock"
        ):
            rewarder = ComposedReward(*rewarder)
        self.rewarder = rewarder
        self.rewarder.link_scenario(self.scenario)

        if communicator is None:
            communicator = NoCommunication()
        self.communicator = deepcopy(communicator)
        self.communicator.link_satellites(self.satellites)

        self.sim_rate = sim_rate
        self.max_step_duration = max_step_duration
        self.failure_penalty = failure_penalty
        if self.failure_penalty > 0:
            logger.warn("Failure penalty should be nonpositive")
        if callable(time_limit):
            self.time_limit_generator = time_limit
        else:
            self.time_limit_generator = lambda: time_limit
        self.terminate_on_time_limit = terminate_on_time_limit
        self.latest_step_duration = 0.0
        self.render_mode = render_mode
        self.generate_obs_retasking_only = generate_obs_retasking_only

    def _minimum_world_model(self) -> type[WorldModel]:
        """Determine the minimum world model required by the satellites."""
        types = set(
            sum(
                [satellite.dyn_type._requires_world() for satellite in self.satellites],
                [],
            )
        )
        if len(types) == 1:
            return list(types)[0]
        for test_type in types:
            if all([issubclass(test_type, other_type) for other_type in types]):
                return test_type

        # Else compose all types into a new class
        class MinimumEnv(*types):
            pass

        return MinimumEnv

    def get_satellite(self, name: str) -> "Satellite":
        """Get a satellite by name.

        Args:
            name: Name of the satellite to retrieve.

        Returns:
            The satellite object with the specified name.
        """
        for sat in self.satellites:
            if sat.name == name:
                return sat
        raise ValueError(f"Satellite with name '{name}' not found.")

    def _configure_logging(self, log_level, log_dir=None):
        if isinstance(log_level, str):
            log_level = log_level.upper()
        logger = logging.getLogger("bsk_rl")
        logger.setLevel(log_level)

        # Ensure each process has its own logger to avoid conflicts when printing
        # sim timestamps. Running multiple environments in the same process in
        # parallel will cause logging times to be incorrectly reported.
        warn_new_env = False
        for handler in logger.handlers:
            if handler.filters[0].proc_id == os.getpid():
                logger.handlers.remove(handler)
                warn_new_env = True

        ch = logging.StreamHandler()
        ch.setFormatter(logging_config.SimFormatter(color_output=True))
        ch.addFilter(logging_config.ContextFilter(env=self, proc_id=os.getpid()))
        logger.addHandler(ch)
        if warn_new_env:
            logger.warning(
                f"Creating logger for new env on PID={os.getpid()}. "
                "Old environments in process may now log times incorrectly."
            )

        if log_dir is not None:
            fh = logging.FileHandler(log_dir)
            fh.setFormatter(logging_config.SimFormatter(color_output=False))
            fh.addFilter(logging_config.ContextFilter(env=self, proc_id=os.getpid()))
            logger.addHandler(fh)

    def _generate_world_args(self) -> None:
        """Instantiate world_args from any randomizers in provided world_args."""
        self.world_args = {
            k: v if not callable(v) else v()
            for k, v in self.world_args_generator.items()
        }

    def _randomize_time_limit(self) -> None:
        time_limit = self.time_limit_generator()
        time_limit = np.ceil(time_limit / self.sim_rate) * self.sim_rate
        self.time_limit = time_limit

    def reset(
        self,
        seed: Optional[int] = None,
        options=None,
    ) -> tuple[MultiSatObs, dict[str, Any]]:
        """Reconstruct the simulator and reset the scenario.

        Satellite and world arguments get randomized on reset, if :class:`~bsk_rl.GeneralSatelliteTasking` ``.world_args``
        or :class:`~bsk_rl.sats.Satellite` ``.sat_args`` includes randomization functions.

        Certain classes in ``bsk_rl`` have a ``reset_pre_sim_init`` and/or ``reset_post_sim_init``
        method. These methods are respectively called before and after the new Basilisk
        :class:`~bsk_rl.sim.Simulator` is created. These allow for reset actions that
        feed into the underlying simulation and those that are dependent on the underlying
        simulation to be performed.

        Args:
            seed: Gymnasium environment seed.
            options: Unused.

        Returns:
            observation, info
        """
        # Explicitly delete the Basilisk simulation before creating a new one.
        self.delete_simulator()

        if seed is None:
            seed = time_ns() % 2**32
        logger.info(f"Resetting environment with seed={seed}")
        self.seed = seed
        super().reset(seed=self.seed)
        np.random.seed(self.seed)

        self._randomize_time_limit()

        self.scenario.reset_overwrite_previous()
        self.rewarder.reset_overwrite_previous()
        self.communicator.reset_overwrite_previous()
        for i, satellite in enumerate(self.satellites):
            satellite.reset_overwrite_previous()
            satellite.create_vizard_data(color=vizard.get_color(i))
        self.latest_step_duration = 0.0

        self._generate_world_args()
        overrides = self.sat_arg_randomizer(self.satellites)
        for satellite in self.satellites:
            sat_overrides = overrides.get(satellite, {})
            satellite.generate_sat_args(
                utc_init=self.world_args["utc_init"], **sat_overrides
            )

        self.scenario.utc_init = self.world_args["utc_init"]

        self.scenario.reset_pre_sim_init()
        self.rewarder.reset_pre_sim_init()
        self.communicator.reset_pre_sim_init()

        for satellite in self.satellites:
            self.rewarder.create_data_store(satellite)
            self.rewarder.data += satellite.data_store.data
            satellite.reset_pre_sim_init()

        self.simulator = Simulator(
            self.satellites,
            self.world_type,
            self.world_args,
            sim_rate=self.sim_rate,
            max_step_duration=self.max_step_duration,
            time_limit=self.time_limit,
        )
        self.simulator.setup_vizard(**self.vizard_settings)

        self.scenario.reset_during_sim_init()
        self.rewarder.reset_during_sim_init()
        self.communicator.reset_during_sim_init()

        self.simulator.finish_init()

        self.scenario.reset_post_sim_init()
        self.rewarder.reset_post_sim_init()
        self.communicator.reset_post_sim_init()

        for satellite in self.satellites:
            satellite.reset_post_sim_init()
            satellite.data_store.update_from_logs()

        observation = self._get_obs()
        info = self._get_info()
        logger.info("Environment reset")
        return observation, info

    def delete_simulator(self):
        """Delete Basilisk objects.

        Only the simulator contains strong references to BSK models, so deleting it
        will delete all Basilisk objects. Enable debug-level logging to verify that the
        simulator, FSW, dynamics, and world models are all deleted on reset.
        """
        try:
            del self.simulator
        except AttributeError:
            pass

    def _get_obs(self) -> MultiSatObs:
        """Compose satellite observations into a single observation.

        Returns:
            tuple: Joint observation
        """
        if self.generate_obs_retasking_only:
            return tuple(
                (
                    satellite.get_obs()
                    if satellite.requires_retasking
                    else satellite.observation_space.sample() * 0
                )
                for satellite in self.satellites
            )
        else:
            return tuple(satellite.get_obs() for satellite in self.satellites)

    def _get_info(self) -> dict[str, Any]:
        """Compose satellite info into a single info dict.

        Returns:
            tuple: Joint info
        """
        info: dict[str, Any] = {
            satellite.name: {"requires_retasking": satellite.requires_retasking}
            for satellite in self.satellites
        }
        info["d_ts"] = self.latest_step_duration
        return info

    def _get_reward(self):
        """Return a scalar reward for the step."""
        reward = sum(self.reward_dict.values())
        for satellite in self.satellites:
            if not satellite.is_alive(log_failure=True):
                reward += self.failure_penalty
        return reward

    def _get_terminated(self) -> bool:
        """Return the terminated flag for the step."""
        if self.terminate_on_time_limit and self._get_truncated():
            return True
        else:
            return not all(
                satellite.is_alive() and not self.rewarder.is_terminated(satellite)
                for satellite in self.satellites
            )

    def _get_truncated(self) -> bool:
        """Return the truncated flag for the step."""
        return (self.simulator.sim_time >= self.time_limit) or any(
            self.rewarder.is_truncated(satellite) for satellite in self.satellites
        )

    @property
    def action_space(self) -> spaces.Space[MultiSatAct]:
        """Compose satellite action spaces into a tuple.

        Returns:
            Joint action space
        """
        return spaces.Tuple((satellite.action_space for satellite in self.satellites))

    @property
    def observation_space(self) -> spaces.Space[MultiSatObs]:
        """Compose satellite observation spaces into a tuple.

        Note: calls ``reset()``, which can be expensive, to determine observation size.

        Returns:
            Joint observation space
        """
        try:
            self.simulator
        except AttributeError:
            logger.info("Calling env.reset() to get observation space")
            self.reset(seed=self.seed)
        return spaces.Tuple(
            [satellite.observation_space for satellite in self.satellites]
        )

    def _step(self, actions: MultiSatAct) -> None:
        logger.debug(f"Stepping environment with actions: {actions}")
        if len(actions) != len(self.satellites):
            raise ValueError("There must be the same number of actions and satellites")
        for satellite, action in zip(self.satellites, actions):
            satellite.info = []  # reset satellite info log

            if not is_no_action(action):
                satellite.requires_retasking = False
                satellite.set_action(action)
            if not satellite.is_alive():
                satellite.requires_retasking = False
            else:
                if satellite.requires_retasking:
                    satellite.logger.warning(
                        f"Requires retasking but received no task."
                    )

        previous_time = self.simulator.sim_time  # should these be recorded in simulator
        self.simulator.run()
        self.latest_step_duration = self.simulator.sim_time - previous_time

        new_data = {
            satellite.name: satellite.data_store.update_from_logs()
            for satellite in self.satellites
        }
        self.reward_dict = self.rewarder.reward(new_data)

        self.communicator.communicate()

        for satellite in self.satellites:
            if satellite.requires_retasking:
                satellite.logger.info(f"Satellite {satellite.name} requires retasking")

    def step(
        self, actions: MultiSatAct
    ) -> tuple[MultiSatObs, float, bool, bool, dict[str, Any]]:
        """Propagate the simulation, update information, and get rewards.

        Args:
            actions: Joint action for satellites

        Returns:
            observation, reward, terminated, truncated, info
        """
        logger.info("=== STARTING STEP ===")
        self._step(actions)

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()
        logger.info(f"Step reward: {reward}")
        if terminated or truncated:
            logger.info(f"Episode terminated: {terminated}")
            logger.info(f"Episode truncated: {truncated}")
        else:
            logger.debug(f"Episode terminated: {terminated}")
            logger.debug(f"Episode truncated: {truncated}")
        logger.debug(f"Step info: {info}")
        logger.debug(f"Step observation: {observation}")
        return observation, reward, terminated, truncated, info

    def render(self) -> None:  # pragma: no cover
        """No rendering implemented."""
        return None

    def close(self) -> None:
        """Try to cleanly delete everything."""
        if self.simulator is not None:
            del self.simulator


class SatelliteTasking(GeneralSatelliteTasking, Generic[SatObs, SatAct]):
    def __init__(self, satellite: Satellite, *args, **kwargs) -> None:
        """A special case of :class:`GeneralSatelliteTasking` for one satellite.

        For compatibility with standard training APIs, actions and observations are
        directly exposed for the single satellite as opposed to being wrapped in a
        tuple.

        Args:
            satellite: Satellite to be simulated.
            *args: Passed to :class:`GeneralSatelliteTasking`.
            **kwargs: Passed to :class:`GeneralSatelliteTasking`.
        """
        super().__init__(satellites=satellite, *args, **kwargs)
        if not len(self.satellites) == 1:
            raise ValueError(
                "SatelliteTasking must be initialized with a single satellite."
            )

    @property
    def action_space(self) -> spaces.Space[SatAct]:
        """Return the single satellite action space."""
        return self.satellite.action_space

    @property
    def observation_space(self) -> spaces.Box:
        """Return the single satellite observation space."""
        super().observation_space
        return self.satellite.observation_space

    @property
    def satellite(self) -> Satellite:
        """Satellite being tasked."""
        return self.satellites[0]

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Task the satellite with a single action."""
        return super().step([action])

    def _get_obs(self) -> Any:
        return self.satellite.get_obs()

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        for k, v in info[self.satellite.name].items():
            info[k] = v
        del info[self.satellite.name]
        return info


class ConstellationTasking(
    GeneralSatelliteTasking, ParallelEnv, Generic[SatObs, SatAct, AgentID]
):
    def __init__(
        self,
        *args,
        meta_agent_groupings: Optional[dict[AgentID, list[str]]] = None,
        only_retask_idle_meta_agent_members: bool = False,
        **kwargs,
    ) -> None:
        """Implements the `PettingZoo <https://pettingzoo.farama.org>`_ parallel API for the :class:`GeneralSatelliteTasking` environment.

        Args:
            *args: Passed to :class:`GeneralSatelliteTasking`.
            meta_agent_groupings: A dictionary mapping agent names to lists of satellite names.
            only_retask_idle_meta_agent_members: If True, only satellites in a meta agent
                that require retasking will receive actions. Other actions in the meta
                agent output will be ignored. This may also be useful to control in the
                training pipeline.
            **kwargs: Passed to :class:`GeneralSatelliteTasking`.
        """
        super().__init__(*args, **kwargs)

        self.only_retask_idle_meta_agent_members = only_retask_idle_meta_agent_members

        if meta_agent_groupings is None:
            meta_agent_groupings = {}

        sats_in_meta_agents = sum(meta_agent_groupings.values(), [])
        for sat in self.satellites:
            if sat.name not in sats_in_meta_agents:
                meta_agent_groupings[sat.name] = [sat.name]

        self.meta_agent_groupings: dict[AgentID, list[Satellite]] = {
            name: [self.get_satellite(member) for member in members]
            for name, members in meta_agent_groupings.items()
        }

    def _validate_meta_agent_groupings(self):
        """Validate that meta agent groupings consist of similar action spaces."""
        for name, members in self.meta_agent_groupings.items():
            if len(members) == 0:
                raise ValueError(f"Meta agent '{name}' has no members.")
            action_space_type = type(members[0].action_space)
            for member in members:
                assert isinstance(member.action_space, action_space_type), (
                    f"Meta agent '{name}' has members with different action space types."
                )
                assert isinstance(member.observation_space, spaces.Box), (
                    f"Only Box observation spaces are supported for meta agents, "
                    f"but member '{member.name}' has {type(member.observation_space)}."
                )

    def reset(
        self, seed: int | None = None, options=None
    ) -> tuple[MultiSatObs, dict[str, Any]]:
        """Reset the environment and return PettingZoo Parallel API format."""
        self.newly_dead = []
        self._agents_last_compute_time = None
        return super().reset(seed, options)

    @property
    def agents(self) -> list[AgentID]:
        """Agents currently in the environment."""
        if (
            self._agents_last_compute_time is None
            or self._agents_last_compute_time != self.simulator.sim_time
        ):
            truncated = super()._get_truncated()
            agents = [
                agent
                for agent, satellites in self.meta_agent_groupings.items()
                if all(satellite.is_alive() for satellite in satellites)
                and not truncated
            ]
            self._agents_last_compute_time = self.simulator.sim_time
            self._agents_cache = agents
            return agents
        else:
            return self._agents_cache

    @property
    def num_agents(self) -> int:
        """Number of agents currently in the environment."""
        return len(self.agents)

    @property
    def possible_agents(self) -> list[AgentID]:
        """Return the list of all possible agents."""
        return list(self.meta_agent_groupings.keys())

    @property
    def max_num_agents(self) -> int:
        """Maximum number of agents possible in the environment."""
        return len(self.possible_agents)

    @property
    def previously_dead(self) -> list[AgentID]:
        """Return the list of agents that died at least one step ago."""
        return list(set(self.possible_agents) - set(self.agents) - set(self.newly_dead))

    @property
    def observation_spaces(self) -> dict[AgentID, spaces.Box]:
        """Return the observation space for each agent."""
        super().observation_space
        self._validate_meta_agent_groupings()

        obs_spaces = {}
        for agent, satellites in self.meta_agent_groupings.items():
            if len(satellites) == 1:
                obs_spaces[agent] = satellites[0].observation_space
            else:
                obs_spaces[agent] = spaces.Box(
                    low=np.concatenate(
                        [sat.observation_space.low for sat in satellites]
                    ),
                    high=np.concatenate(
                        [sat.observation_space.high for sat in satellites]
                    ),
                    dtype=(
                        self.dtype
                        if self.dtype
                        else satellites[0].observation_space.dtype
                    ),
                )
        return obs_spaces

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> spaces.Space[SatObs]:
        """Return the observation space for a certain agent."""
        return self.observation_spaces[agent]

    @property
    def action_spaces(self) -> dict[AgentID, spaces.Space[SatAct]]:
        """Return the action space for each agent."""
        act_spaces = {}
        for agent, satellites in self.meta_agent_groupings.items():
            if len(satellites) == 1:
                act_spaces[agent] = satellites[0].action_space
            else:
                if isinstance(satellites[0].action_space, spaces.Discrete):
                    act_spaces[agent] = spaces.MultiDiscrete(
                        [sat.action_space.n for sat in satellites]
                    )
                elif isinstance(satellites[0].action_space, spaces.Box):
                    low = np.concatenate([sat.action_space.low for sat in satellites])
                    high = np.concatenate([sat.action_space.high for sat in satellites])
                    act_spaces[agent] = spaces.Box(
                        low=low,
                        high=high,
                        dtype=(
                            self.dtype
                            if self.dtype
                            else satellites[0].action_space.dtype
                        ),
                    )
        return act_spaces

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> spaces.Space[SatAct]:
        """Return the action space for a certain agent."""
        return self.action_spaces[agent]

    def _requires_retasking(self, agent: AgentID) -> bool:
        """Check if the agent requires retasking."""
        return any(
            satellite.requires_retasking
            for satellite in self.meta_agent_groupings[agent]
        )

    def _get_obs(self) -> dict[AgentID, SatObs]:
        """Format the observation per the PettingZoo Parallel API."""
        obs = {}
        for agent, satellites in self.meta_agent_groupings.items():
            # Don't generate observations for agents that are dead
            if agent in self.previously_dead:
                continue

            if self.generate_obs_retasking_only and not self._requires_retasking(agent):
                agent_obs = [
                    satellite.observation_space.sample() * 0 for satellite in satellites
                ]
            else:
                agent_obs = [satellite.get_obs() for satellite in satellites]

            if len(agent_obs) == 1:
                obs[agent] = agent_obs[0]
            else:
                obs[agent] = np.concatenate(agent_obs)

        return obs

    def _get_reward(self) -> dict[AgentID, float]:
        """Format the reward per the PettingZoo Parallel API."""
        satellite_rewards = {
            self.get_satellite(name): reward
            for name, reward in self.reward_dict.items()
        }
        for satellite in self.satellites:
            if not satellite.is_alive():
                if satellite in satellite_rewards:
                    satellite_rewards[satellite] += self.failure_penalty
                else:
                    satellite_rewards[satellite] = self.failure_penalty

        reward = {
            agent: sum(satellite_rewards[sat] for sat in sats)
            for agent, sats in self.meta_agent_groupings.items()
        }

        reward_keys = list(reward.keys())
        for agent in reward_keys:
            if agent in self.previously_dead:
                del reward[agent]

        return reward

    def _get_terminated(self) -> dict[AgentID, bool]:
        """Format terminations per the PettingZoo Parallel API."""
        if self.terminate_on_time_limit and super()._get_truncated():
            return {
                agent: True
                for agent in self.possible_agents
                if agent not in self.previously_dead
            }
        else:
            return {
                agent: any(
                    not sat.is_alive() or self.rewarder.is_terminated(sat)
                    for sat in satellites
                )
                for agent, satellites in self.meta_agent_groupings.items()
                if agent not in self.previously_dead
            }

    def _get_truncated(self) -> dict[AgentID, bool]:
        """Format truncations per the PettingZoo Parallel API."""
        truncated = super()._get_truncated()
        return {
            agent: truncated
            or any(self.rewarder.is_truncated(sat) for sat in satellites)
            for agent, satellites in self.meta_agent_groupings.items()
            if agent not in self.previously_dead
        }

    def _get_info(self) -> dict[AgentID, dict]:
        """Format info per the PettingZoo Parallel API."""
        info_per_sat = super()._get_info()

        # Group info by agent
        info = {}
        for agent, satellites in self.meta_agent_groupings.items():
            if agent not in self.previously_dead:
                info[agent] = {
                    "requires_retasking": any(
                        info_per_sat[sat.name]["requires_retasking"]
                        for sat in satellites
                    )
                }
                if len(satellites) > 1:
                    for satellite in satellites:
                        info[agent][satellite.name] = info_per_sat[satellite.name]

        # Identify common info
        common = {
            k: v
            for k, v in info_per_sat.items()
            if k not in [sat.name for sat in self.satellites]
        }

        # Pass common info to all agents and to __common__
        for agent in info.keys():
            for k, v in common.items():
                info[agent][k] = v
        info["__common__"] = common

        return info

    def _decompose_meta_action(
        self, agent: AgentID, action: SatAct
    ) -> dict[Satellite, SatAct]:
        """Decompose a meta agent action into satellite actions."""
        sat_to_action_map = {}
        i = 0
        for satellite in self.meta_agent_groupings[agent]:
            action_len = satellite.action_space.shape
            if len(action_len) == 0:
                action_len = 1
            else:
                action_len = action_len[0]

            if isinstance(action, (list, tuple, np.ndarray)):
                if (
                    not self.only_retask_idle_meta_agent_members
                    or satellite.requires_retasking
                ):
                    if action_len == 1:
                        sat_to_action_map[satellite] = action[i]
                    else:
                        sat_to_action_map[satellite] = action[i : i + action_len]
                else:
                    sat_to_action_map[satellite] = None
                i += action_len
            else:
                sat_to_action_map[satellite] = action

        return sat_to_action_map

    def step(
        self,
        actions: dict[AgentID, SatAct],
    ) -> tuple[
        dict[AgentID, SatObs],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """Step the environment and return PettingZoo Parallel API format."""
        logger.info("=== STARTING STEP ===")

        previous_alive = self.agents

        sat_to_action_map = {}
        for agent, action in actions.items():
            if len(self.meta_agent_groupings[agent]) > 1:
                logger.info(f"Decomposing action for meta agent {agent}")
            sat_to_action_map.update(self._decompose_meta_action(agent, action))

        action_vector = []
        for satellite in self.satellites:
            if satellite in sat_to_action_map:
                action_vector.append(sat_to_action_map[satellite])
            else:
                action_vector.append(None)
        self._step(action_vector)

        self.newly_dead = list(set(previous_alive) - set(self.agents))

        for agent in self.newly_dead:
            for satellite in self.meta_agent_groupings[agent]:
                for attr in [
                    "_timed_terminal_event_name",
                    "_image_event_name",
                ]:
                    event_name = getattr(satellite, attr, None)
                    if event_name is not None:
                        self.simulator.delete_event(event_name)

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()
        nonzero_reward = {k: v for k, v in reward.items() if v != 0}
        logger.info(f"Step reward: {nonzero_reward}")
        if any(terminated.values()):
            terminated_true = [k for k, v in terminated.items() if v]
            logger.info(f"Episode terminated: {terminated_true}")
        if any(truncated.values()):
            truncated_true = [k for k, v in truncated.items() if v]
            logger.info(f"Episode truncated: {truncated_true}")
        logger.debug(f"Step info: {info}")
        logger.debug(f"Step observation: {observation}")
        return observation, reward, terminated, truncated, info


__all__ = []
