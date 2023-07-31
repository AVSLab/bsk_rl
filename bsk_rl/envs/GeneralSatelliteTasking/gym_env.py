from copy import deepcopy
from typing import Any, Iterable, Optional, Union

import numpy as np
from gymnasium import Env, spaces

from bsk_rl.envs.GeneralSatelliteTasking.scenario.communication import NoCommunication
from bsk_rl.envs.GeneralSatelliteTasking.simulation.simulator import Simulator
from bsk_rl.envs.GeneralSatelliteTasking.types import (
    CommunicationMethod,
    DataManager,
    EnvironmentFeatures,
    EnvironmentModel,
    Satellite,
)

SatObs = Any
SatAct = Any
MultiSatObs = tuple[SatObs, ...]
MultiSatAct = Iterable[SatAct]


class GeneralSatelliteTasking(Env):
    def __init__(
        self,
        satellites: Union[Satellite, list[Satellite]],
        env_type: type[EnvironmentModel],
        env_features: EnvironmentFeatures,
        data_manager: DataManager,
        env_args: Optional[dict[str, Any]] = None,
        communicator: Optional[CommunicationMethod] = None,
        sim_rate: float = 1.0,
        max_step_duration: float = 600.0,
        failure_penalty: float = -100,
        time_limit: float = float("inf"),
        terminate_on_time_limit: bool = False,
        render_mode=None,
    ) -> None:
        """A Gymnasium environment adaptable to a wide range satellite tasking problems that involve satellite(s) being
        tasked to complete tasks and maintain aliveness. These tasks often include rewards for data collection. The
        environment can be configured for any collection of satellites, including heterogenous constellations. Other
        configurable aspects are environment features (e.g. imaging targets), data collection and recording, and
        intersatellite communication of data.

        The state space is a tuple containing the state of each satellite. Actions are assigned as a tuple of actions,
        one per satellite.

        The preferred method of instantiating this environment is to make the "GeneralSatelliteTasking-v1" environment
        and pass a kwargs dict with the environment configuration. In some cases (e.g. the multiprocessed Gymnasium
        vector environment), it is necessary for compatibility to instead register a new environment using the
        GeneralSatelliteTasking class and a kwargs dict. See examples/general_satellite_tasking for examples of
        environment configuration.

        New environments should be built using this framework.

        Args:
            satellites: Satellites(s) to be simulated.
            env_type: Type of environment model to be constructed.
            env_args: Arguments for environment model construction. {key: value or key: function}, where
                function is called at reset to set the value (used for randomization).
            env_features: Information about the environment.
            data_manager: Object to record and reward data collection.
            communicator: Object to manage communication between satellites
            sim_rate: Rate for model simulation [s].
            max_step_duration: Maximum time to propagate sim at a step [s].
            failure_penalty: Reward for satellite failure.
            time_limit: Time at which to truncate the simulation [s].
            terminate_on_time_limit: Send terminations signal time_limit instead of just truncation.
            render_mode: Unused.
        """
        self.seed = None
        if isinstance(satellites, Satellite):
            satellites = [satellites]
        self.satellites = satellites
        self.simulator: Simulator
        self.env_type = env_type
        if env_args is None:
            env_args = self.env_type.default_env_args()
        self.env_args_generator = self.env_type.default_env_args(**env_args)
        self.env_features = env_features
        self.data_manager = data_manager
        if communicator is None:
            communicator = NoCommunication(self.satellites)
        self.communicator = communicator
        self.sim_rate = sim_rate
        self.max_step_duration = max_step_duration
        self.failure_penalty = failure_penalty
        self.time_limit = time_limit
        self.terminate_on_time_limit = terminate_on_time_limit
        self.latest_step_duration = 0
        self.render_mode = render_mode

    def _generate_env_args(self) -> None:
        """Instantiate env_args from any randomizers in provided env_args."""
        self.env_args = {
            k: v if not callable(v) else v() for k, v in self.env_args_generator.items()
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options=None,
    ) -> tuple[MultiSatObs, dict[str, Any]]:
        """Reconstruct the simulator and wipe data records.

        Args:
            seed: Gymnasium environment seed.
            options: Unused.

        Returns:
            observation, info
        """
        self.seed = seed
        super().reset(seed=self.seed)
        np.random.seed(self.seed)
        self._generate_env_args()

        self.env_features.reset()
        self.data_manager.reset()

        # Destroy cross references to actually delete the simulator.
        for satellite in self.satellites:
            satellite.delete_simulator()
        self.delete_simulator()

        for satellite in self.satellites:
            self.data_manager.create_data_store(satellite)
            satellite.sat_args_generator["utc_init"] = self.env_args["utc_init"]
            satellite.reset()

        self.simulator = Simulator(
            self.satellites,
            self.env_type,
            self.env_args,
            sim_rate=self.sim_rate,
            max_step_duration=self.max_step_duration,
            time_limit=self.time_limit,
        )

        self.communicator.reset()

        for satellite in self.satellites:
            satellite.data_store.internal_update()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def delete_simulator(self):
        """Delete cross-references to simulator objects"""

        for attr in [
            "simulator.environment.simulator",
            "simulator.environment",
            "simulator",
        ]:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def _get_obs(self) -> MultiSatObs:
        """Compose satellite observations into a single observation.

        Returns:
            tuple: Joint observation
        """
        return tuple(satellite.get_obs() for satellite in self.satellites)

    def _get_info(self) -> dict[str, Any]:
        """Compose satellite info into a single info dict.

        Returns:
            tuple: Joint info
        """
        info: dict[str, Any] = {
            satellite.id: deepcopy(satellite.info) for satellite in self.satellites
        }
        info["d_ts"] = self.latest_step_duration
        return info

    @property
    def action_space(self) -> spaces.Space[MultiSatAct]:
        """Compose satellite action spaces

        Returns:
            Joint action space
        """
        return spaces.Tuple((satellite.action_space for satellite in self.satellites))

    @property
    def observation_space(self) -> spaces.Space[MultiSatObs]:
        """Compose satellite observation spaces. Note: calls reset(), which can be expensive, to determine observation
        size

        Returns:
            Joint observation space
        """
        try:
            self.simulator
        except AttributeError:
            print("Calling env.reset() to get observation space")
            self.reset(seed=self.seed)
        return spaces.Tuple(
            [satellite.observation_space for satellite in self.satellites]
        )

    def step(
        self, actions: MultiSatAct
    ) -> tuple[MultiSatObs, float, bool, bool, dict[str, Any]]:
        """Propagate the simulation, update information, and get rewards

        Args:
            Joint action for satellites

        Returns:
            observation, reward, terminated, truncated, info
        """
        for satellite, action in zip(self.satellites, actions):
            satellite.info = []  # reset satellite info log
            satellite.set_action(action)

        previous_time = self.simulator.sim_time  # should these be recorded in simulator
        self.simulator.run()
        self.latest_step_duration = self.simulator.sim_time - previous_time

        new_data = {
            satellite.id: satellite.data_store.internal_update()
            for satellite in self.satellites
        }
        reward = self.data_manager.reward(new_data)

        self.communicator.communicate()

        terminated = False
        for satellite in self.satellites:
            if not satellite.is_alive():
                terminated = True
                reward += self.failure_penalty

        truncated = False
        if self.simulator.sim_time >= self.time_limit:
            truncated = True
            if self.terminate_on_time_limit:
                terminated = True

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """No rendering implemented"""
        return None

    def close(self) -> None:
        """Try to cleanly delete everything"""
        if self.simulator is not None:
            del self.simulator


class SingleSatelliteTasking(GeneralSatelliteTasking):
    """A special case of the GeneralSatelliteTasking for one satellite. For compatibility with standard training
    APIs, actions and observations are directly exposed for the single satellite and are not wrapped
    in a tuple.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert len(self.satellites) == 1

    @property
    def action_space(self) -> spaces.Discrete:
        """Return the single satellite action space"""
        return self.satellite.action_space

    @property
    def observation_space(self) -> spaces.Box:
        """Return the single satellite observation space"""
        super().observation_space
        return self.satellite.observation_space

    @property
    def satellite(self) -> Satellite:
        return self.satellites[0]

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Task the satellite with a single action"""
        return super().step([action])

    def _get_obs(self) -> Any:
        return self.satellite.get_obs()
