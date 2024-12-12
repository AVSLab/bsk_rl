"""Extended Basilisk SimBaseClass for GeneralSatelliteTasking environments."""

import logging
from typing import TYPE_CHECKING, Any

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.sim.world import WorldModel

logger = logging.getLogger(__name__)


class Simulator(SimulationBaseClass.SimBaseClass):
    """Basilisk simulator for GeneralSatelliteTasking environments."""

    def __init__(
        self,
        satellites: list["Satellite"],
        world_type: type["WorldModel"],
        world_args: dict[str, Any],
        sim_rate: float = 1.0,
        max_step_duration: float = 600.0,
        time_limit: float = float("inf"),
    ) -> None:
        """Basilisk simulator for satellite tasking environments.

        The simulator is reconstructed each time the environment :class:`~bsk_rl.GeneralSatelliteTasking.reset`
        is called, generating a fresh Basilisk simulation.

        Args:
            satellites: Satellites to be simulated
            world_type: Type of world model to be constructed
            world_args: Arguments for world model construction
            sim_rate: [s] Rate for model simulation.
            max_step_duration: [s] Maximum time to propagate sim at a step.
            time_limit: [s] Latest time simulation will propagate to.
        """
        super().__init__()
        self.sim_rate = sim_rate
        self.satellites = satellites
        self.max_step_duration = max_step_duration
        self.time_limit = time_limit
        self.logger = logger

        self.world: WorldModel

        self._set_world(world_type, world_args)

        self.fsw_list = {}
        self.dynamics_list = {}

        for satellite in self.satellites:
            satellite.set_simulator(self)
            self.dynamics_list[satellite.name] = satellite.set_dynamics(self.sim_rate)
            self.fsw_list[satellite.name] = satellite.set_fsw(self.sim_rate)

    def finish_init(self) -> None:
        """Finish simulator initialization."""
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

    def _set_world(
        self, world_type: type["WorldModel"], world_args: dict[str, Any]
    ) -> None:
        """Construct the simulator world model.

        Args:
            world_type: Type of world model to be constructed.
            world_args: Arguments for world model construction, passed to the world
                from the environment.
        """
        self.world = world_type(self, self.sim_rate, **world_args)

    def run(self) -> None:
        """Propagate the simulator.

        Propagates for a duration up to the ``max_step_duration``, stopping if the
        environment time limit is reached or an event is triggered.
        """
        if "max_step_duration" in self.eventMap:
            self.delete_event("max_step_duration")

        self.createNewEvent(
            "max_step_duration",
            mc.sec2nano(self.sim_rate),
            True,
            [
                f"self.TotalSim.CurrentNanos * {mc.NANO2SEC} >= {self.sim_time + self.max_step_duration}"
            ],
            ["self.logger.info('Max step duration reached')"],
            terminal=True,
        )
        self.ConfigureStopTime(mc.sec2nano(min(self.time_limit, 2**31)))
        self.ExecuteSimulation()

    def delete_event(self, event_name) -> None:
        """Remove an event from the event map.

        Makes event checking faster. Due to a performance issue in Basilisk, it is
        necessary to remove created for tasks that are no longer needed (even if it is
        inactive), or else significant time is spent processing the event at each step.
        """
        event = self.eventMap[event_name]
        self.eventList.remove(event)
        del self.eventMap[event_name]

    def __del__(self):
        """Log when simulator is deleted."""
        logger.debug("Basilisk simulator deleted")


__all__ = []
