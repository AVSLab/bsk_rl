"""Extended Basilisk SimBaseClass for GeneralSatelliteTasking environments."""

import logging
import os
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any
import numpy as np

from Basilisk.utilities import macros as mc
from Basilisk.utilities import SimulationBaseClass
from Basilisk.simulation import simpleInstrument, simpleStorageUnit, partitionedStorageUnit, spaceToGroundTransmitter
from Basilisk.simulation import groundLocation
from Basilisk.utilities import vizSupport
from Basilisk.utilities import unitTestSupport

from Basilisk.simulation import spacecraft
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody
from Basilisk.architecture import astroConstants

from bsk_rl.utils import vizard

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.sim.world import WorldModel

logger = logging.getLogger(__name__)

def strip_prefix(s, prefix="GroundStation"):
    return s[len(prefix):] if s.startswith(prefix) else s

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
        self.set_vizard_epoch()
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

    @vizard.visualize
    def setup_vizard(self, vizard_rate=None, vizSupport=None, **vizard_settings):
        """Setup Vizard for visualization."""
        save_path = Path(vizard.VIZARD_PATH)
        if not save_path.exists():
            os.makedirs(save_path, exist_ok=True)

        viz_proc_name = "VizProcess"
        viz_proc = self.CreateNewProcess(viz_proc_name, priority=400)

        # Define process name, task name and task time-step
        viz_task_name = "viz_task_name"
        if vizard_rate is None:
            vizard_rate = self.sim_rate
        viz_proc.addTask(self.CreateNewTask(viz_task_name, mc.sec2nano(vizard_rate)))

        customizers = ["spriteList", "genericSensorList"]
        list_data = {}
        for customizer in customizers:
            list_data[customizer] = [
                sat.vizard_data.get(customizer, None) for sat in self.satellites
            ]
        self.vizInstance = vizSupport.enableUnityVisualization(
            self,
            viz_task_name,
            scList=[sat.dynamics.scObject for sat in self.satellites],
            **list_data,
            saveFile=save_path / f"viz_{time()}",
        )
        viz = self.vizInstance

        for i in range(len(self.world.groundStations)):
            vizSupport.addLocation(viz, stationName=strip_prefix(self.world.groundStations[i].ModelTag)
                                   , parentBodyName=self.world.planet.displayName
                                   , r_GP_P=unitTestSupport.EigenVector3d2list(self.world.groundStations[i].r_LP_P_Init)
                                   , fieldOfView=np.radians(160.)
                                   , color='green'
                                   , range=1000.0*1000  # meters
                                   )
            viz.settings.spacecraftSizeMultiplier = 1.5
            viz.settings.showLocationCommLines = 1
            viz.settings.showLocationCones = 1
            viz.settings.showLocationLabels = 1
        for key, value in vizard_settings.items():
            setattr(self.vizInstance.settings, key, value)
        vizard.VIZINSTANCE = self.vizInstance


    @vizard.visualize
    def set_vizard_epoch(self, vizInstance=None):
        """Set the Vizard epoch."""
        vizInstance.epochInMsg.subscribeTo(self.world.gravFactory.epochMsg)

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
        # event = self.eventMap[event_name]
        # self.eventList.remove(event)
        del self.eventMap[event_name]

    def __del__(self):
        """Log when simulator is deleted."""
        logger.debug("Basilisk simulator deleted")


__all__ = []
