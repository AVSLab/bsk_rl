from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bsk_rl.envs.GeneralSatelliteTasking.types import (
        Satellite,
        EnvironmentModel,
    )

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc


class Simulator(SimulationBaseClass.SimBaseClass):
    def __init__(
        self,
        satellites: list["Satellite"],
        env_type: type["EnvironmentModel"],
        env_args: dict[str, Any],
        sim_rate: float = 1.0,
        max_step_duration: float = 600.0,
        time_limit: float = float("inf"),
    ) -> None:
        """Basilisk simulator for GeneralSatelliteTasking environments.

        Args:
            satellites: Satellites to be simulated
            env_type: Type of environment model to be constructed
            env_args: Arguments for environment model construction
            sim_rate: Rate for model simulation [s].
            max_step_duration: Maximum time to propagate sim at a step [s].
            time_limit: Latest time simulation will propagate to.
        """
        super().__init__()
        self.sim_rate = sim_rate
        self.satellites = satellites
        self.max_step_duration = max_step_duration
        self.time_limit = time_limit

        self.environment: EnvironmentModel

        self._set_environment(env_type, env_args)

        self.fsw_list = {}
        self.dynamics_list = {}

        for satellite in self.satellites:
            self.dynamics_list[satellite.id] = satellite.set_dynamics(
                self, self.sim_rate
            )
            self.fsw_list[satellite.id] = satellite.set_fsw(self, self.sim_rate)

        self.InitializeSimulation()
        self.ConfigureStopTime(0)
        self.ExecuteSimulation()

    @property
    def sim_time_ns(self) -> int:
        """Simulation time in ns, tied to SimBase integrator."""
        return self.TotalSim.CurrentNanos

    @property
    def sim_time(self) -> float:
        """Simulation time in seconds, tied to SimBase integrator."""
        return self.sim_time_ns * mc.NANO2SEC

    def _set_environment(
        self, env_type: type["EnvironmentModel"], env_args: dict[str, Any]
    ) -> None:
        """Construct the simulator environment model

        Args:
            env_type: type of environment model to be constructed
            env_args: arguments for environment model construction
        """
        self.environment = env_type(self, self.sim_rate, **env_args)

    def run(self) -> None:
        """Propagate the simulator"""
        simulation_time = mc.sec2nano(
            min(self.sim_time + self.max_step_duration, self.time_limit)
        )
        self.ConfigureStopTime(simulation_time)
        self.ExecuteSimulation()