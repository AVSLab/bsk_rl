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
        dynamic_priority_event_enabled: bool = False,
        dynamic_priority_event_time_sec: Optional[float] = None,
        dynamic_priority_event_fraction: float = 0.5,
        hio_count: int = 5,
        hio_priority: float = 5.0,
        shio_count: int = 3,
        shio_priority: float = 10.0,
        dynamic_priority_event_seed: Optional[int] = None,
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
            dynamic_priority_event_enabled: If True, boost a fixed set of
                episode targets during the episode.
            dynamic_priority_event_time_sec: Absolute event time in seconds. If
                provided, this takes precedence over ``dynamic_priority_event_fraction``.
            dynamic_priority_event_fraction: Fraction of the episode time limit
                at which HIO/SHIO priorities become active.
            hio_count: Number of high-interest objects to boost.
            hio_priority: Absolute priority assigned to each HIO after the event.
            shio_count: Number of super-high-interest objects to boost.
            shio_priority: Absolute priority assigned to each SHIO after the event.
            dynamic_priority_event_seed: Optional seed for selecting event targets.
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
        self.dynamic_priority_event_enabled = bool(dynamic_priority_event_enabled)
        self.dynamic_priority_event_time_sec = dynamic_priority_event_time_sec
        self.dynamic_priority_event_fraction = float(dynamic_priority_event_fraction)
        self.hio_count = int(hio_count)
        self.hio_priority = float(hio_priority)
        self.shio_count = int(shio_count)
        self.shio_priority = float(shio_priority)
        self.dynamic_priority_event_seed = dynamic_priority_event_seed
        self.priority_event_applied = False
        self.priority_event_time = None
        self.priority_event_applied_time = None
        self.hio_target_ids: list[int] = []
        self.shio_target_ids: list[int] = []

        if self.priority_mode not in {"uniform", "gaussian", "constant"}:
            raise ValueError(
                "priority_mode must be one of: 'uniform', 'gaussian', 'constant'."
            )
        if self.n_targets < 0:
            raise ValueError("n_targets must be non-negative.")
        if not 0.0 <= self.dynamic_priority_event_fraction <= 1.0:
            raise ValueError("dynamic_priority_event_fraction must be in [0, 1].")
        if (
            self.dynamic_priority_event_time_sec is not None
            and float(self.dynamic_priority_event_time_sec) < 0.0
        ):
            raise ValueError("dynamic_priority_event_time_sec must be non-negative.")
        if self.hio_count < 0 or self.shio_count < 0:
            raise ValueError("hio_count and shio_count must be non-negative.")
        if self.hio_count + self.shio_count > self.n_targets:
            raise ValueError("HIO + SHIO target counts cannot exceed n_targets.")

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

    def _reset_priority_event_state(self) -> None:
        self.priority_event_applied = False
        self.priority_event_time = None
        self.priority_event_applied_time = None
        self.hio_target_ids = []
        self.shio_target_ids = []

    def _select_dynamic_priority_targets(self) -> None:
        """Choose HIO/SHIO targets from the full episode target catalog."""
        if not self.dynamic_priority_event_enabled:
            return

        total_event_targets = self.hio_count + self.shio_count
        if total_event_targets <= 0:
            return

        rng = (
            np.random.default_rng(self.dynamic_priority_event_seed)
            if self.dynamic_priority_event_seed is not None
            else np.random.default_rng(np.random.randint(0, 2**32 - 1))
        )
        selected_ids = rng.choice(
            self.n_targets,
            size=total_event_targets,
            replace=False,
        ).astype(int)
        self.hio_target_ids = selected_ids[: self.hio_count].tolist()
        self.shio_target_ids = selected_ids[self.hio_count :].tolist()

        boost_by_id = {
            target_id: ("HIO", self.hio_priority) for target_id in self.hio_target_ids
        }
        boost_by_id.update(
            {target_id: ("SHIO", self.shio_priority) for target_id in self.shio_target_ids}
        )

        for target in self.target_spacecrafts:
            kind, boosted_priority = boost_by_id.get(int(target.id), ("", None))
            target.priority_event_kind = kind
            target.priority_event_priority = boosted_priority
            target.priority_event_original_priority = float(target.priority)
            target.priority_event_active = False
            target.priority_event_candidate_count = 0
            target.priority_event_first_candidate_time = None
            target.priority_event_last_candidate_log_time = None

    def maybe_apply_dynamic_priority_event(
        self, sim_time: float, time_limit: Optional[float] = None
    ) -> bool:
        """Activate the HIO/SHIO priority boost once per episode."""
        if not self.dynamic_priority_event_enabled:
            return False
        if self.priority_event_applied:
            return True
        if time_limit is None and self.dynamic_priority_event_time_sec is None:
            return False

        if self.dynamic_priority_event_time_sec is not None:
            self.priority_event_time = float(self.dynamic_priority_event_time_sec)
        else:
            self.priority_event_time = (
                float(time_limit) * self.dynamic_priority_event_fraction
            )
        if float(sim_time) < self.priority_event_time:
            return False

        boost_ids = set(self.hio_target_ids) | set(self.shio_target_ids)
        for target in self.target_spacecrafts:
            if int(target.id) not in boost_ids:
                continue
            boosted_priority = getattr(target, "priority_event_priority", None)
            if boosted_priority is None:
                continue
            target.priority = float(boosted_priority)
            target.priority_event_active = True
            target.priority_event_applied_time = float(sim_time)

        self.priority_event_applied = True
        self.priority_event_applied_time = float(sim_time)
        logger.info(
            "Applied dynamic priority event at t=%.3f s: HIO ids=%s, SHIO ids=%s",
            float(sim_time),
            self.hio_target_ids,
            self.shio_target_ids,
        )
        return True

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
        self._reset_priority_event_state()
        priorities = self._generate_priorities()
        for i in range(self.n_targets):
            target_sc_name = f"target_{i}"  # must match buffer name
            sc = RSOTarget(self.satellites[i + 1], target_sc_name, i, float(priorities[i]))
            self.target_spacecrafts.append(sc)

        self._select_dynamic_priority_targets()

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
        self._reset_priority_event_state()

    def reset_during_sim_init(self):
        for i in range(self.n_targets):
            # Add all candidate targets to scanner's target location model.
            self.satellites[0].dynamics.targetLocation.addSpacecraftToModel(
                self.satellites[i + 1].dynamics.scObject.scStateOutMsg
            )

















