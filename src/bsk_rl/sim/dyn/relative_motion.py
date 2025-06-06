"""Dynamics models concerning the relative motion of spacecraft."""

from Basilisk.simulation import spacecraftLocation
from Basilisk.utilities import macros

from bsk_rl.sim.dyn import BasicDynamicsModel
from bsk_rl.utils.functional import aliveness_checker, default_args, valid_func_name


class LOSCommDynModel(BasicDynamicsModel):
    """For evaluating line-of-sight connections between satellites for communication."""

    def __init__(self, *args, **kwargs) -> None:
        """Allow for line-of-sight checking between satellites.

        Necessary for :class:`~bsk_rl.comm.LOSCommunication` to function.
        """
        super().__init__(*args, **kwargs)

    def _setup_dynamics_objects(self, **kwargs) -> None:
        super()._setup_dynamics_objects(**kwargs)
        self.setup_los_comms(**kwargs)

    @default_args(losMaximumRange=-1.0)
    def setup_los_comms(
        self, losMaximumRange: float, priority: int = 500, **kwargs
    ) -> None:
        """Set up line-of-sight visibility checking between satellites.

        Args:
            losMaximumRange: [m] Maximum range for line-of-sight visibility. -1 for unlimited.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.losComms = spacecraftLocation.SpacecraftLocation()
        self.losComms.ModelTag = "losComms"
        self.losComms.primaryScStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.losComms.planetInMsg.subscribeTo(
            self.world.gravFactory.spiceObject.planetStateOutMsgs[self.world.body_index]
        )
        self.losComms.rEquator = self.simulator.world.planet.radEquator
        self.losComms.rPolar = self.simulator.world.planet.radEquator * 0.98
        self.losComms.maximumRange = losMaximumRange

        self.los_comms_ids = []

        for sat_dyn in self.simulator.dynamics_list.values():
            if sat_dyn != self and sat_dyn.satellite.name not in self.los_comms_ids:
                self.losComms.addSpacecraftToModel(sat_dyn.scObject.scStateOutMsg)
                self.los_comms_ids.append(sat_dyn.satellite.name)
                sat_dyn.losComms.addSpacecraftToModel(self.scObject.scStateOutMsg)
                sat_dyn.los_comms_ids.append(self.satellite.name)
                if len(sat_dyn.los_comms_ids) == 1:
                    sat_dyn.simulator.AddModelToTask(
                        sat_dyn.task_name, sat_dyn.losComms, ModelPriority=priority
                    )

        if len(self.los_comms_ids) > 0:
            self.simulator.AddModelToTask(
                self.task_name, self.losComms, ModelPriority=priority
            )


class ConjunctionDynModel(BasicDynamicsModel):
    """For evaluating conjunctions between satellites."""

    def __init__(self, *args, **kwargs) -> None:
        """Model that evaluates conjunctions between satellites.

        The simulation is terminated at the time of collision and a conjunction_valid failure is reported.
        """
        super().__init__(*args, **kwargs)
        self.conjunctions = []

    def _setup_dynamics_objects(self, **kwargs) -> None:
        super()._setup_dynamics_objects(**kwargs)
        self.setup_conjunctions(**kwargs)

    @aliveness_checker
    def conjunction_valid(self) -> bool:
        """Check if conjunction has not occured."""
        return len(self.conjunctions) == 0

    @default_args(conjunction_radius=10)
    def setup_conjunctions(self, conjunction_radius: float, **kwargs) -> None:
        """Set up conjunction checking between satellites.

        Args:
            conjunction_radius: [m] Minimum distance for a conjunction.
            kwargs: Passed to other setup functions.
        """
        self.conjunction_radius = conjunction_radius

        for sat_dyn in self.simulator.dynamics_list.values():
            if sat_dyn != self and isinstance(sat_dyn, ConjunctionDynModel):
                self.simulator.createNewEvent(
                    valid_func_name(
                        f"conjunction_{self.satellite.name}_{sat_dyn.satellite.name}"
                    ),
                    macros.sec2nano(self.simulator.sim_rate),
                    True,
                    [
                        f"np.linalg.norm(np.array({self.satellite._satellite_command}.dynamics.r_BN_N) - np.array({sat_dyn.satellite._satellite_command}.dynamics.r_BN_N))"
                        + " <= "
                        + f"{self.satellite._satellite_command}.dynamics.conjunction_radius + {sat_dyn.satellite._satellite_command}.dynamics.conjunction_radius"
                    ],
                    [
                        self.satellite._info_command(
                            f"collided with {sat_dyn.satellite.name}"
                        ),
                        sat_dyn.satellite._info_command(
                            f"collided with {self.satellite.name}"
                        ),
                        f"{self.satellite._satellite_command}.dynamics.conjunctions.append({sat_dyn.satellite._satellite_command})",
                        f"{sat_dyn.satellite._satellite_command}.dynamics.conjunctions.append({self.satellite._satellite_command})",
                        f"[{self.satellite._satellite_command}.logger.warning('Collision occurred at t=0, may incorrectly report failure type') if self.sim_time == 0 else None]",
                    ],
                    terminal=True,
                )


class MaxRangeDynModel(BasicDynamicsModel):
    """For evaluating a maximum range limitation between satellites."""

    def __init__(self, *args, **kwargs) -> None:
        """Model that checks for maximum range violations between satellites.

        The simulation is terminated at the time of separation and a range_valid failure is reported.
        """
        super().__init__(*args, **kwargs)
        self.out_of_ranges = []

    def _setup_dynamics_objects(self, **kwargs) -> None:
        super()._setup_dynamics_objects(**kwargs)
        self.setup_range(**kwargs)

    @aliveness_checker
    def range_valid(self) -> bool:
        """Check if conjunction has not occurred."""
        return len(self.out_of_ranges) == 0

    @default_args(max_range_radius=5000, chief_name=None)
    def setup_range(self, max_range_radius: float, chief_name: str, **kwargs) -> None:
        """Set up maximum distance checking relative to a chief satellite.

        Args:
            max_range_radius: [m] Maximum allowed range from the chief satellite.
            chief_name: Chief satellite to check range against.
            kwargs: Passed to other setup functions.
        """
        self.max_range_radius = max_range_radius
        self.chief_name = chief_name

        if self.chief_name is None:
            self.logger.warning(
                "No chief satellite specified for maximum range checking. "
                "Range checking is disabled."
            )
            return

        self.simulator.createNewEvent(
            valid_func_name(f"range_{self.satellite.name}_{self.chief_name}"),
            macros.sec2nano(self.simulator.sim_rate),
            True,
            [
                f"np.linalg.norm(np.array({self.satellite._satellite_command}.dynamics.r_BN_N)"
                + f"- np.array(self.get_satellite('{self.chief_name}').dynamics.r_BN_N))"
                + " >= "
                + f"{self.satellite._satellite_command}.dynamics.max_range_radius"
            ],
            [
                self.satellite._info_command(
                    f"Exceeded maximum range of {max_range_radius} m from {self.chief_name}"
                ),
                f"{self.satellite._satellite_command}.dynamics.out_of_ranges.append(self.get_satellite('{self.chief_name}'))",
            ],
            terminal=True,
        )


__doc_title__ = "Relative Motion"
__all__ = [
    "LOSCommDynModel",
    "ConjunctionDynModel",
    "MaxRangeDynModel",
]
