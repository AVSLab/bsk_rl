"""Basilisk world models are given in ``bsk_rl.sim.world``.

In most cases, the user does not need to specify the world model, as it is inferred from
the requirements of the :class:`~bsk_rl.sim.fsw.FSWModel`. However, the user can specify
the world model in the :class:`~bsk_rl.GeneralSatelliteTasking` constructor if desired.

Customization of the world model parameters is via the ``world_args`` parameter in the
:class:`~bsk_rl.GeneralSatelliteTasking`. As with ``sat_args``, these parameters are
passed as a dictionary of key-value or key-function pairs, with the latter called to
generate the value each time the simulation is reset.

.. code-block:: python

    world_args = dict(
        utc_init="2018 SEP 29 21:00:00.000 (UTC)",  # set the epoch
        scaleHeight=np.random.uniform(7e3, 9e3),  # randomize the atmosphere
    )

In general, ``world_args`` parameter names match those used in Basilisk. See the setup
functions for short descriptions of what parameters do and the Basilisk documentation
for more detail on their exact model effects.

"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union
from weakref import proxy

import numpy as np
from Basilisk import __path__
from Basilisk.simulation import (
    eclipse,
    ephemerisConverter,
    exponentialAtmosphere,
    groundLocation,
)
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion, simIncludeGravBody

from bsk_rl.utils.functional import collect_default_args, default_args
from bsk_rl.utils.orbital import random_epoch

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sim import Simulator

logger = logging.getLogger(__name__)

bsk_path = __path__[0]


class WorldModel(ABC):
    """Abstract Basilisk world model."""

    @classmethod
    def default_world_args(cls, **kwargs) -> dict[str, Any]:
        """Compile default arguments for the world model.

        Args:
            **kwargs: Arguments to override in the default arguments.

        Returns:
            Dictionary of arguments for simulation models.
        """
        defaults = collect_default_args(cls)
        for k, v in kwargs.items():
            if k not in defaults:
                raise KeyError(f"{k} not a valid key for world_args")
            defaults[k] = v
        return defaults

    def __init__(
        self,
        simulator: "Simulator",
        world_rate: float,
        priority: int = 300,
        **kwargs,
    ) -> None:
        """Abstract Basilisk world model.

        One WorldModel is instantiated for the environment each time a new simulator
        is created.

        Args:
            simulator: Simulator using this model.
            world_rate: Rate of world simulation [s]
            priority: Model priority.
            kwargs: Passed through to setup functions.
        """
        self.simulator: Simulator = proxy(simulator)

        world_proc_name = "WorldProcess"
        world_proc = self.simulator.CreateNewProcess(world_proc_name, priority)

        # Define process name, task name and task time-step
        self.world_task_name = "WorldTask"
        world_proc.addTask(
            self.simulator.CreateNewTask(self.world_task_name, mc.sec2nano(world_rate))
        )

        self._setup_world_objects(**kwargs)

    def __del__(self):
        """Log when world is deleted."""
        logger.debug("Basilisk world deleted")

    @abstractmethod  # pragma: no cover
    def _setup_world_objects(self, **kwargs) -> None:
        """Caller for all world objects."""
        pass


class BasicWorldModel(WorldModel):
    """Basic world with minimum necessary Basilisk world components."""

    def __init__(self, *args, **kwargs) -> None:
        """Basic world with minimum necessary Basilisk world components.

        This model includes ephemeris and SPICE-based Earth gravity and dynamics models,
        an exponential atmosphere model, and an eclipse model.

        Args:
            *args: Passed to superclass.
            **kwargs: Passed to superclass.
        """
        super().__init__(*args, **kwargs)

    @property
    def PN(self):
        """Planet relative to inertial frame rotation matrix."""
        return np.array(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
            .read()
            .J20002Pfix
        ).reshape((3, 3))

    @property
    def omega_PN_N(self):
        """Planet angular velocity in inertial frame [rad/s]."""
        PNdot = np.array(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
            .read()
            .J20002Pfix_dot
        ).reshape((3, 3))
        skew_PN_N = -np.matmul(np.transpose(self.PN), PNdot)
        return np.array([skew_PN_N[2, 1], skew_PN_N[0, 2], skew_PN_N[1, 0]])

    def _setup_world_objects(self, **kwargs) -> None:
        self.setup_gravity_bodies(**kwargs)
        self.setup_ephem_object(**kwargs)
        self.setup_atmosphere_density_model(**kwargs)
        self.setup_eclipse_object(**kwargs)

    @default_args(utc_init=random_epoch)
    def setup_gravity_bodies(
        self, utc_init: str, priority: int = 1100, **kwargs
    ) -> None:
        """Specify gravitational models to use in the simulation.

        Args:
            utc_init: UTC datetime string, in the format ``YYYY MMM DD hh:mm:ss.sss (UTC)``
            priority: Model priority.
            **kwargs: Passed to other setup functions.
        """
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.gravFactory.createSun()
        self.planet = self.gravFactory.createEarth()
        self.sun_index = 0
        self.body_index = 1

        self.planet.isCentralBody = (
            True  # ensure this is the central gravitational body
        )
        self.planet.useSphericalHarmonicsGravityModel(
            bsk_path + "/supportData/LocalGravData/GGM03S.txt", 10
        )

        # setup Spice interface for some solar system bodies
        timeInitString = utc_init
        self.gravFactory.createSpiceInterface(
            bsk_path + "/supportData/EphemerisData/", timeInitString, epochInMsg=True
        )
        self.gravFactory.spiceObject.zeroBase = "earth"

        self.simulator.AddModelToTask(
            self.world_task_name, self.gravFactory.spiceObject, ModelPriority=priority
        )

    def setup_ephem_object(self, priority: int = 988, **kwargs) -> None:
        """Set up the ephemeris object to use with the SPICE library.

        Args:
            priority: Model priority.
            **kwargs: Passed to other setup functions.
        """
        self.ephemConverter = ephemerisConverter.EphemerisConverter()
        self.ephemConverter.ModelTag = "ephemConverter"
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun_index]
        )
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        self.simulator.AddModelToTask(
            self.world_task_name, self.ephemConverter, ModelPriority=priority
        )

    @default_args(
        planetRadius=orbitalMotion.REQ_EARTH * 1e3,
        baseDensity=1.22,
        scaleHeight=8e3,
    )
    def setup_atmosphere_density_model(
        self,
        planetRadius: float,
        baseDensity: float,
        scaleHeight: float,
        priority: int = 1000,
        **kwargs,
    ) -> None:
        """Set up the exponential gravity model.

        Args:
            planetRadius: [m] Planet ground radius.
            baseDensity: [kg/m^3] Exponential model parameter.
            scaleHeight: [m] Exponential model parameter.
            priority: Model priority.
            **kwargs: Passed to other setup functions.
        """
        self.densityModel = exponentialAtmosphere.ExponentialAtmosphere()
        self.densityModel.ModelTag = "expDensity"
        self.densityModel.planetRadius = planetRadius
        self.densityModel.baseDensity = baseDensity
        self.densityModel.scaleHeight = scaleHeight
        self.densityModel.planetPosInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        self.simulator.AddModelToTask(
            self.world_task_name, self.densityModel, ModelPriority=priority
        )

    def setup_eclipse_object(self, priority: int = 988, **kwargs) -> None:
        """Set up the celestial object that is causing an eclipse message.

        Args:
            priority: Model priority.
            kwargs: Ignored
        """
        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addPlanetToModel(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        self.eclipseObject.sunInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun_index]
        )
        self.simulator.AddModelToTask(
            self.world_task_name, self.eclipseObject, ModelPriority=priority
        )

    def __del__(self) -> None:
        """Log when world is deleted and unload SPICE."""
        super().__del__()
        try:
            self.gravFactory.unloadSpiceKernels()
        except AttributeError:
            pass


class GroundStationWorldModel(BasicWorldModel):
    """Model that includes downlink ground stations."""

    def __init__(self, *args, **kwargs) -> None:
        """Model that includes downlink ground stations.

        This model includes the basic world components, as well as ground stations for
        downlinking data.

        Args:
            *args: Passed to superclass.
            **kwargs: Passed to superclass.
        """
        super().__init__(*args, **kwargs)

    def _setup_world_objects(self, **kwargs) -> None:
        super()._setup_world_objects(**kwargs)
        self.setup_ground_locations(**kwargs)

    @default_args(
        groundStationsData=[
            dict(name="Boulder", lat=40.009971, long=-105.243895, elev=1624),
            dict(name="Merritt", lat=28.3181, long=-80.6660, elev=0.9144),
            dict(name="Singapore", lat=1.3521, long=103.8198, elev=15),
            dict(name="Weilheim", lat=47.8407, long=11.1421, elev=563),
            dict(name="Santiago", lat=-33.4489, long=-70.6693, elev=570),
            dict(name="Dongara", lat=-29.2452, long=114.9326, elev=34),
            dict(name="Hawaii", lat=19.8968, long=-155.5828, elev=9),
        ],
        groundLocationPlanetRadius=orbitalMotion.REQ_EARTH * 1e3,
        gsMinimumElevation=np.radians(10.0),
        gsMaximumRange=-1,
    )
    def setup_ground_locations(
        self,
        groundStationsData: list[dict[str, Union[str, float]]],
        groundLocationPlanetRadius: float,
        gsMinimumElevation: float,
        gsMaximumRange: float,
        priority: int = 1399,
        **kwargs,
    ) -> None:
        """Specify the ground locations of interest.

        Args:
            groundStationsData: List of dictionaries of ground station data. Each dictionary
                must include keys for ``lat`` and ``long`` [deg], and may include
                ``elev`` [m], ``name``. For example:

                .. code-block:: python

                    groundStationsData=[
                        dict(name="Boulder", lat=40.009971, long=-105.243895, elev=1624),
                        dict(lat=28.3181, long=-80.6660),
                    ]

                ``groundLocationPlanetRadius``, ``gsMinimumElevation``, and ``gsMaximumRange``
                may also be specified in the dictionary to override the global values
                for those parameters for a specific ground station.

            groundLocationPlanetRadius: [m] Radius of ground locations from center of
                planet.
            gsMinimumElevation: [rad] Minimum elevation angle from station to satellite
                to be able to downlink data.
            gsMaximumRange: [m] Maximum range from station to satellite when
                downlinking. Set to ``-1`` to disable.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.groundStations = []
        self.groundLocationPlanetRadius = groundLocationPlanetRadius
        self.gsMinimumElevation = gsMinimumElevation
        self.gsMaximumRange = gsMaximumRange
        for i, groundStationData in enumerate(groundStationsData):
            self._create_ground_station(**groundStationData, priority=priority - i)

    def _create_ground_station(
        self,
        lat: float,
        long: float,
        elev: float = 0,
        name: Optional[str] = None,
        groundLocationPlanetRadius: Optional[float] = None,
        gsMinimumElevation: Optional[float] = None,
        gsMaximumRange: Optional[float] = None,
        priority: int = 1399,
    ) -> None:
        """Add a ground station with given parameters.

        Args:
            lat: [deg] Latitude.
            long: [deg] Longitude.
            elev: [m] Elevation.
            name: Ground station identifier.
            groundLocationPlanetRadius: [m] Radius of planet.
            gsMinimumElevation: [rad] Minimum elevation angle to downlink to ground station.
            gsMaximumRange: [m] Maximum range to downlink to ground station. Set to ``-1`` for infinity.
            priority: Model priority.
        """
        if name is None:
            name = str(len(self.groundStations))

        groundStation = groundLocation.GroundLocation()
        groundStation.ModelTag = "GroundStation" + name
        if groundLocationPlanetRadius:
            groundStation.planetRadius = groundLocationPlanetRadius
        else:
            groundStation.planetRadius = self.groundLocationPlanetRadius
        groundStation.specifyLocation(np.radians(lat), np.radians(long), elev)
        groundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        if gsMinimumElevation:
            groundStation.minimumElevation = gsMinimumElevation
        else:
            groundStation.minimumElevation = self.gsMinimumElevation
        if gsMaximumRange:
            groundStation.maximumRange = gsMaximumRange
        else:
            groundStation.maximumRange = self.gsMaximumRange
        self.groundStations.append(groundStation)

        self.simulator.AddModelToTask(
            self.world_task_name, groundStation, ModelPriority=priority
        )


__doc_title__ = "World Sims"
__all__ = ["WorldModel", "BasicWorldModel", "GroundStationWorldModel"]
