"""Target scenarios distribute ground targets with some distribution.

Currently, targets are all known to the satellites a priori and are available based on
the imaging requirements given by the dynamics and flight software models.
"""

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
from Basilisk.simulation import (
    ReactionWheelPower,
    extForceTorque,
    facetDragDynamicEffector,
    groundLocation,
    partitionedStorageUnit,
    simpleBattery,
    simpleInstrument,
    simpleNav,
    simplePowerSink,
    simpleSolarPanel,
    simpleStorageUnit,
    spacecraft,
    spacecraftLocation,
    spaceToGroundTransmitter,
)
# from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Optional

# from bsk_rl.sim import world
# from bsk_rl.sim.world import WorldModel
# # from bsk_rl.sats import Satellite
# from bsk_rl.sim import Simulator
#
# from bsk_rl.utils import actuator_primitives as aP
# from bsk_rl.utils.attitude import random_tumble
# from bsk_rl.utils.functional import (
#     aliveness_checker,
#     check_aliveness_checkers,
#     default_args,
# )
from Basilisk.utilities import (
    RigidBodyKinematics,
    macros,
    orbitalMotion,
    unitTestSupport,
)

# from webcolors import names

from bsk_rl.scene import Scenario
# from bsk_rl.utils.orbital import lla2ecef
# from tests.integration.comm.test_int_communication import oes_visible

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.base import Data
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


class RSOTarget:
    """Creates a target spacecraft with randomized Orbital Elements"""

    def __init__(self,target_rso, name, id: int, priority: float):
        #set name, priority, initial oe
        self.id = id
        # self.name = f"TargetSat_{id}"
        self.name = name
        self.priority = priority
        self.target_spacecraft = target_rso



    #are these two world functions (and the world imports above) needed?
    # @classmethod
    # def _requires_world(cls) -> list[type["WorldModel"]]:
    #     return [world.BasicWorldModel]
    #
    # @property
    # def simulator(self) -> "Simulator":
    #     """Reference to the episode simulator."""
    #     return self.satellite.simulator
    #
    # @property
    # def world(self) -> "WorldModel":
    #     """Reference to the episode world model."""
    #     return self.simulator.world

    def add_to_sim(self, target_rso, simulator):
        self.target_spacecraft = target_rso

        # Set up location tracking

        # Set up simple navigation




class RandomSatellites(Scenario):
    """Spacecraft target with associated priority"""

    def __init__(self, ChiefSatellite, n_targets: int) -> None:
        """Spacecraft target with associated priority and location.

        Args:
            # name: Identifier; does not need to be unique
            n_targets: Number of targets
            # priority_distribution: Function for generating target priority. Defaults
            #     to ``lambda: uniform(0, 1)`` if not specified.
            # priority: Value metric.
        """
        self.n_targets = n_targets
        # if priority_distribution is None:  #priority distribution to be added later
        #     priority_distribution = lambda: np.random.rand()  # noqa: E731
        # self.priority_distribution = priority_distribution


        # dyn_proc_name = "DynamicsProcess" + self.satellite.name
        # self.dyn_proc = self.simulator.CreateNewProcess(dyn_proc_name, priority)
        # self.dyn_rate = dyn_rate
        # self.task_name = "DynamicsTask" + self.satellite.name
        # self.dyn_proc.addTask(
        #     self.simulator.CreateNewTask(self.task_name, macros.sec2nano(self.dyn_rate))
        # )
        #
        # # Initialize all modules and write init one-time messages
        # self.scObject: spacecraft.Spacecraft
        # self._setup_dynamics_objects(**kwargs)
    def link_satellites(self, satellites: list["Satellite"]) -> None:
        super().link_satellites(satellites)
        ChiefSatellite = self.satellites[0].name
        self.ScanningSat = [satellite for satellite in self.satellites if satellite.name == ChiefSatellite][0]
        self.ScanningSat.sat_args_generator["bufferNames"] = [sc.name for sc in self.satellites]
        self.ScanningSat.sat_args_generator["transmitterNumBuffers"] = len(self.ScanningSat.sat_args_generator["bufferNames"])

    def reset_pre_sim_init(self):
        for i in range(self.n_targets):
            target_sc_name = f"target_{i}" # this name here should match the bufferName so that the data gets added to the buffer !
            sc = RSOTarget(self.satellites[i+1],target_sc_name,i, 1.0)
            # sc = RSOTarget(i, priority=self.priority_distribution(), oe) #to be implemented later with priority_distribution
            self.target_spacecrafts.append(sc)


    def reset_overwrite_previous(self) -> None:
        self.target_spacecrafts = []


    def reset_during_sim_init(self):
        for i in range(self.n_targets):
            self.satellites[0].dynamics.targetLocation.addSpacecraftToModel(self.satellites[i+1].dynamics.scObject.scStateOutMsg) # this adds all possible targets to SS.targetLocation






















