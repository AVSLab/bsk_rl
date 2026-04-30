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
        # self.eclipse_status = self.target_spacecraft.eclispe_index # TODO: fix this DHP



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

    def __init__(
        self,
        ChiefSatellite,
        n_targets: int,
        priority_mode: str = "uniform",
        priority_sum: Optional[float] = 100.0,
        rescale_priorities_to_sum: bool = True,
        priority_constant: float = 1.0,
        priority_uniform_low: float = 0.0,
        priority_uniform_high: Optional[float] = None,
        priority_gaussian_mean: Optional[float] = None,
        priority_gaussian_std: Optional[float] = None,
        priority_min: float = 0.0,
        priority_max: Optional[float] = None,
    ) -> None:
        """Spacecraft-target scenario with configurable target priority generation.

        Args:
            ChiefSatellite: Name of scanning/imaging satellite.
            n_targets: Number of targets.
            priority_mode: ``"uniform"``, ``"gaussian"``, or ``"constant"``.
            priority_sum: Desired total priority sum. If ``None``, do not rescale.
            rescale_priorities_to_sum: If True and ``priority_sum`` is set, rescale
                generated priorities to make the sum exactly ``priority_sum``.
            priority_constant: Constant priority used in ``"constant"`` mode.
            priority_uniform_low: Uniform lower bound.
            priority_uniform_high: Uniform upper bound. Defaults to
                ``2 * priority_sum / n_targets`` when ``priority_sum`` is set.
            priority_gaussian_mean: Gaussian mean. Defaults to ``priority_sum / n_targets``
                when ``priority_sum`` is set.
            priority_gaussian_std: Gaussian standard deviation. Defaults to mean/3.
            priority_min: Minimum allowed priority after generation.
            priority_max: Maximum allowed priority after generation.
        """
        self.chief_satellite_name = ChiefSatellite
        self.n_targets = int(n_targets)
        self.priority_mode = str(priority_mode).lower()
        self.priority_sum = priority_sum
        self.rescale_priorities_to_sum = bool(rescale_priorities_to_sum)
        self.priority_constant = float(priority_constant)
        self.priority_uniform_low = float(priority_uniform_low)
        self.priority_uniform_high = priority_uniform_high
        self.priority_gaussian_mean = priority_gaussian_mean
        self.priority_gaussian_std = priority_gaussian_std
        self.priority_min = float(priority_min)
        self.priority_max = priority_max

        if self.priority_mode not in {"uniform", "gaussian", "constant"}:
            raise ValueError(
                "priority_mode must be one of: 'uniform', 'gaussian', 'constant'."
            )
        if self.n_targets < 0:
            raise ValueError("n_targets must be non-negative.")

    def _generate_raw_priorities(self) -> np.ndarray:
        """Generate nonnegative target priorities according to ``priority_mode``."""
        if self.n_targets == 0:
            return np.array([], dtype=float)

        target_mean = (
            float(self.priority_sum) / float(self.n_targets)
            if self.priority_sum is not None
            else 1.0
        )

        if self.priority_mode == "uniform":
            high = (
                float(self.priority_uniform_high)
                if self.priority_uniform_high is not None
                else 2.0 * target_mean
            )
            low = self.priority_uniform_low
            if high < low:
                raise ValueError("priority_uniform_high must be >= priority_uniform_low.")
            priorities = np.random.uniform(low=low, high=high, size=self.n_targets)
        elif self.priority_mode == "gaussian":
            mean = (
                float(self.priority_gaussian_mean)
                if self.priority_gaussian_mean is not None
                else target_mean
            )
            std = (
                float(self.priority_gaussian_std)
                if self.priority_gaussian_std is not None
                else max(mean / 3.0, 1e-6)
            )
            if std <= 0.0:
                raise ValueError("priority_gaussian_std must be positive.")
            priorities = np.random.normal(loc=mean, scale=std, size=self.n_targets)
        else:  # constant
            priorities = np.full(self.n_targets, fill_value=self.priority_constant)

        priorities = np.clip(priorities, self.priority_min, None)
        if self.priority_max is not None:
            priorities = np.clip(priorities, None, float(self.priority_max))
        return priorities.astype(float)

    def _rescale_priorities(self, priorities: np.ndarray) -> np.ndarray:
        """Rescale priorities to make sum exactly ``priority_sum`` when requested."""
        if (
            not self.rescale_priorities_to_sum
            or self.priority_sum is None
            or len(priorities) == 0
        ):
            return priorities

        desired_sum = float(self.priority_sum)
        current_sum = float(np.sum(priorities))

        if current_sum <= 0.0:
            priorities = np.full(len(priorities), desired_sum / float(len(priorities)))
        else:
            priorities = priorities * (desired_sum / current_sum)

        # Preserve the sampled distribution shape, then pin the final total exactly.
        priorities[-1] += desired_sum - float(np.sum(priorities))
        return priorities

    def _generate_priorities(self) -> np.ndarray:
        """Generate priorities according to configured distribution and scaling."""
        return self._rescale_priorities(self._generate_raw_priorities())

    def link_satellites(self, satellites: list["Satellite"]) -> None:
        super().link_satellites(satellites)
        scanning_sat_name = self.chief_satellite_name
        self.ScanningSat = [
            satellite for satellite in self.satellites if satellite.name == scanning_sat_name
        ][0]
        self.ScanningSat.sat_args_generator["bufferNames"] = [
            sc.name for sc in self.satellites
        ]  # includes scanner + all target spacecraft names
        self.ScanningSat.sat_args_generator["transmitterNumBuffers"] = len(
            self.ScanningSat.sat_args_generator["bufferNames"]
        )

    def reset_pre_sim_init(self):
        priorities = self._generate_priorities()
        for i in range(self.n_targets):
            target_sc_name = f"target_{i}"  # must match buffer name
            sc = RSOTarget(self.satellites[i + 1], target_sc_name, i, float(priorities[i]))
            self.target_spacecrafts.append(sc)

        if len(priorities) > 0:
            logger.info(
                "Generated %d target priorities with mode=%s, min=%.6f, max=%.6f, sum=%.6f",
                len(priorities),
                self.priority_mode,
                float(np.min(priorities)),
                float(np.max(priorities)),
                float(np.sum(priorities)),
            )

    def reset_overwrite_previous(self) -> None:
        self.target_spacecrafts = []

    def reset_during_sim_init(self):
        for i in range(self.n_targets):
            # Add all candidate targets to scanner's target location model.
            self.satellites[0].dynamics.targetLocation.addSpacecraftToModel(
                self.satellites[i + 1].dynamics.scObject.scStateOutMsg
            )





















