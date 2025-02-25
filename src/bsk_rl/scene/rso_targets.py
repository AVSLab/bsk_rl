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

    def __init__(self, id: int, priority: float, oe):
        #set name, priority, initial oe
        self.id = id
        self.name = f"TargetSat_{id}"
        self.priority = priority
        self.oe = oe



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

    def add_to_sim(self, simulator):
        target_rso = spacecraft.Spacecraft()
        self.spacecraft = target_rso

        # Compute state vectors
        mu = 3.986 * 10**14    # Gravitational parameter [m^3/s^2]
        rN, vN = orbitalMotion.elem2rv(mu, self.oe)


        print("Testing the reset_during_sim_init function")

        # Initialize spacecraft state
        target_rso.hub.r_CN_NInit = rN
        target_rso.hub.v_CN_NInit = vN
        target_rso.gravField.gravBodies = spacecraft.GravBodyVector(
            list(simulator.world.gravFactory.gravBodies.values())
        )

        # Set up location tracking
        targetLocation = spacecraftLocation.SpacecraftLocation()
        targetLocation.ModelTag = f"targetLocation_{id}"
        targetLocation.planetInMsg.subscribeTo(
            simulator.world.gravFactory.spiceObject.planetStateOutMsgs[simulator.world.body_index]
        )
        targetLocation.primaryScStateInMsg.subscribeTo(self.spacecraft.scStateOutMsg)
        # targetLocation.primaryScStateInMsg.subscribeTo(Scenario.satellites[0].scObject.scStateOutMsg)

        targetLocation.addSpacecraftToModel(target_rso.scStateOutMsg)
        self.targetLocation = targetLocation

        # Set up simple navigation
        simpleTargetNav = simpleNav.SimpleNav()
        self.simpleNav = simpleTargetNav

        simpleTargetNav.scStateInMsg.subscribeTo(target_rso.scStateOutMsg)

        # Add models to the simulator
        simulator.AddModelToTask(self.task_name, simpleTargetNav, ModelPriority=self.priority)
        simulator.AddModelToTask(self.task_name, target_rso, ModelPriority=self.priority)


class RandomSatellites(Scenario):
    """Spacecraft target with associated priority"""

    def __init__(self, n_targets: int) -> None:
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

    def reset_pre_sim_init(self):
        for sat in self.satellites:
            sat.sat_args["bufferNames"] = [sc.name for sc in self.target_spacecrafts] # TODO: this is will give an error since it is called before target_spacecrafts gets created
            sat.sat_args["transmitterNumBuffers"] = len(sat.sat_args["bufferNames"])
    def reset_overwrite_previous(self) -> None:
        self.target_spacecrafts = []


    def reset_during_sim_init(self):

        rLEO = 7000. * 1000    # Minimum semi-major axis (LEO) in meters
        rGEO = 42164. * 1000   # Maximum semi-major axis (GEO) in meters

        for i in range(self.n_targets):

            target_sc_name = f"TargetSat_{i}"

            oe = orbitalMotion.ClassicElements()
            oe.a = np.random.uniform(rLEO, rGEO)  # Random semi-major axis between LEO and GEO
            if oe.a < 1.5*rLEO:
                oe.e = np.random.uniform(0.0, 0.1)    # Random eccentricity (allowing less elliptical orbits when near LEO)
            else:
                oe.e = np.random.uniform(0.0, 0.2)    # Random eccentricity (allowing slightly elliptical orbits)
            oe.i = np.random.uniform(0, 180) * macros.D2R  # Random inclination up to 180 degrees
            oe.Omega = np.random.uniform(0, 360) * macros.D2R  # Random RAAN
            oe.omega = np.random.uniform(0, 360) * macros.D2R  # Random argument of perigee
            oe.f = np.random.uniform(0, 360) * macros.D2R  # Random true anomaly

            sc = RSOTarget(i, 1.0 ,oe,)
            # sc = RSOTarget(i, priority=self.priority_distribution(), oe) #to be implemented later with priority_distribution
            self.target_spacecrafts.append(sc)
            sc.add_to_sim(self.satellites[0].simulator)

            # self.target_spacecrafts.append(target_sc)  # Store spacecraft reference

























class UniformTargets(Scenario):
    """Environment with targets distributed uniformly."""

    def __init__(
        self,
        n_targets: Union[int, tuple[int, int]],
        priority_distribution: Optional[Callable] = None,
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
    ) -> None:
        """An environment with evenly-distributed static targets.

        Can be used with :class:`~bsk_rl.data.UniqueImageReward`.

        Args:
            n_targets: Number of targets to generate. Can also be specified as a range
                ``(low, high)`` where the number of targets generated is uniformly selected
                ``low ≤ n_targets ≤ high``.
            priority_distribution: Function for generating target priority. Defaults
                to ``lambda: uniform(0, 1)`` if not specified.
            radius: [m] Radius to place targets from body center. Defaults to Earth's
                equatorial radius.
        """
        self._n_targets = n_targets
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()  # noqa: E731
        self.priority_distribution = priority_distribution
        self.radius = radius

    def reset_overwrite_previous(self) -> None:
        """Overwrite target list from previous episode."""
        self.targets = []

    def reset_pre_sim_init(self) -> None:
        """Regenerate target set for new episode."""
        if isinstance(self._n_targets, int):
            self.n_targets = self._n_targets
        else:
            self.n_targets = np.random.randint(self._n_targets[0], self._n_targets[1])
        logger.info(f"Generating {self.n_targets} targets")
        self.regenerate_targets()
        for satellite in self.satellites:
            if hasattr(satellite, "add_location_for_access_checking"):
                for target in self.targets:
                    satellite.add_location_for_access_checking(
                        object=target,
                        r_LP_P=target.r_LP_P,
                        min_elev=satellite.sat_args_generator[
                            "imageTargetMinimumElevation"
                        ],  # Assume not randomized
                        type="target",
                    )

    def regenerate_targets(self) -> None:
        """Regenerate targets uniformly.

        Override this method (as demonstrated in :class:`CityTargets`) to generate
        other distributions.
        """
        self.targets = []
        for i in range(self.n_targets):
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            self.targets.append(
                Target(name=f"tgt-{i}", r_LP_P=x, priority=self.priority_distribution())
            )
