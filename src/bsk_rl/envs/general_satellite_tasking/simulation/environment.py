from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union
from weakref import proxy

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.envs.general_satellite_tasking.types import Simulator

import numpy as np
from Basilisk import __path__
from Basilisk.simulation import (
    eclipse,
    ephemerisConverter,
    exponentialAtmosphere,
    groundLocation,
)
from Basilisk.topLevelModules import pyswice
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion, simIncludeGravBody

from bsk_rl.envs.general_satellite_tasking.utils.debug import MEMORY_LEAK_CHECKING
from bsk_rl.envs.general_satellite_tasking.utils.functional import (
    collect_default_args,
    default_args,
)
from bsk_rl.envs.general_satellite_tasking.utils.orbital import random_epoch

bsk_path = __path__[0]


class EnvironmentModel(ABC):
    @classmethod
    def default_env_args(cls, **kwargs) -> dict[str, Any]:
        """Compile default argments for the environment model"""
        defaults = collect_default_args(cls)
        for k, v in kwargs.items():
            if k not in defaults:
                raise KeyError(f"{k} not a valid key for env_args")
            defaults[k] = v
        return defaults

    def __init__(
        self,
        simulator: "Simulator",
        env_rate: float,
        priority: int = 300,
        **kwargs,
    ) -> None:
        """Base environment model

        Args:
            simulator: Simulator using this model
            env_rate: Rate of environment simulation [s]
            priority: Model priority.
        """
        self.simulator: Simulator = proxy(simulator)

        env_proc_name = "EnvironmentProcess"
        env_proc = self.simulator.CreateNewProcess(env_proc_name, priority)

        # Define process name, task name and task time-step
        self.env_task_name = "EnvironmentTask"
        env_proc.addTask(
            self.simulator.CreateNewTask(self.env_task_name, mc.sec2nano(env_rate))
        )

        self._init_environment_objects(**kwargs)

    def __del__(self):
        if MEMORY_LEAK_CHECKING:  # pragma: no cover
            print("~~~ BSK ENVIRONMENT DELETED ~~~")

    @abstractmethod  # pragma: no cover
    def _init_environment_objects(self, **kwargs) -> None:
        """Caller for all environment objects"""
        pass


class BasicEnvironmentModel(EnvironmentModel):
    """Minimal set of Basilisk environment objects"""

    @property
    def PN(self):
        return np.array(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
            .read()
            .J20002Pfix
        ).reshape((3, 3))

    @property
    def omega_PN_N(self):
        PNdot = np.array(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
            .read()
            .J20002Pfix_dot
        ).reshape((3, 3))
        skew_PN_N = -np.matmul(np.transpose(self.PN), PNdot)
        return np.array([skew_PN_N[2, 1], skew_PN_N[0, 2], skew_PN_N[1, 0]])

    def _init_environment_objects(self, **kwargs) -> None:
        self._set_gravity_bodies(**kwargs)
        self._set_epoch_object(**kwargs)
        self._set_atmosphere_density_model(**kwargs)
        self._set_eclipse_object(**kwargs)

    @default_args(utc_init=random_epoch)
    def _set_gravity_bodies(
        self, utc_init: str, priority: int = 1100, **kwargs
    ) -> None:
        """Specify gravitational models to use in the simulation.

        Args:
            utc_init: UTC datetime string
            priority: Model priority.
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
            bsk_path + "/supportData/EphemerisData/", timeInitString
        )
        self.gravFactory.spiceObject.zeroBase = "earth"

        # Add pyswice instances
        pyswice.furnsh_c(
            self.gravFactory.spiceObject.SPICEDataPath + "de430.bsp"
        )  # solar system bodies
        pyswice.furnsh_c(
            self.gravFactory.spiceObject.SPICEDataPath + "naif0012.tls"
        )  # leap second file
        pyswice.furnsh_c(
            self.gravFactory.spiceObject.SPICEDataPath + "de-403-masses.tpc"
        )  # solar system masses
        pyswice.furnsh_c(
            self.gravFactory.spiceObject.SPICEDataPath + "pck00010.tpc"
        )  # generic Planetary Constants

        self.simulator.AddModelToTask(
            self.env_task_name, self.gravFactory.spiceObject, ModelPriority=priority
        )

    def _set_epoch_object(self, priority: int = 988, **kwargs) -> None:
        """Add the ephemeris object to use with the SPICE library.

        Args:
            priority: Model priority.
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
            self.env_task_name, self.ephemConverter, ModelPriority=priority
        )

    @default_args(
        planetRadius=orbitalMotion.REQ_EARTH * 1e3,
        baseDensity=1.22,
        scaleHeight=8e3,
    )
    def _set_atmosphere_density_model(
        self,
        planetRadius: float,
        baseDensity: float,
        scaleHeight: float,
        priority: int = 1000,
        **kwargs,
    ) -> None:
        """Add the exponential gravity model.

        Args:
            planetRadius: Planet ground radius [m]
            baseDensity: Exponential model parameter [kg/m^3]
            scaleHeight: Exponential model parameter [m]
            priority (int, optional): Model priority.
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
            self.env_task_name, self.densityModel, ModelPriority=priority
        )

    def _set_eclipse_object(self, priority: int = 988, **kwargs) -> None:
        """Specify what celestial object is causing an eclipse message.

        Args:
            priority: Model priority.
        """
        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addPlanetToModel(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        self.eclipseObject.sunInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun_index]
        )
        self.simulator.AddModelToTask(
            self.env_task_name, self.eclipseObject, ModelPriority=priority
        )

    def __del__(self) -> None:
        super().__del__()
        try:
            self.gravFactory.unloadSpiceKernels()
        except AttributeError:
            pass


class GroundStationEnvModel(BasicEnvironmentModel):
    """Model that includes downlink ground stations"""

    def _init_environment_objects(self, **kwargs) -> None:
        super()._init_environment_objects(**kwargs)
        self._set_ground_locations(**kwargs)

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
        gsMinimumElevation=10.0 * mc.D2R,
        gsMaximumRange=-1,
    )
    def _set_ground_locations(
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
            groundStationsData: Dicts with name (optional), lat (required), long
                (required), and elevation (optional).
            groundLocationPlanetRadius: Radius of ground locations from center of planet
                [m]
            gsMinimumElevation:  Minimum elevation angle from station to satellite when
                downlinking [rad]
            gsMaximumRange: Maximum range from station to satellite when downlinking. -1
                to disable. [m]
            priority: Model priority.
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
        priority: int = 1399,
    ) -> None:
        """Add a ground station with given parameters.

        Args:
            lat: Latitude [deg]
            long: Longitude [deg]
            elev: Elevation [m].
            name: Ground station identifier.
            priority: Model priority.
        """
        if name is None:
            name = str(len(self.groundStations))

        groundStation = groundLocation.GroundLocation()
        groundStation.ModelTag = "GroundStation" + name
        groundStation.planetRadius = self.groundLocationPlanetRadius
        groundStation.specifyLocation(lat * mc.D2R, long * mc.D2R, elev)
        groundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        groundStation.minimumElevation = self.gsMinimumElevation
        groundStation.maximumRange = self.gsMaximumRange
        self.groundStations.append(groundStation)

        self.simulator.AddModelToTask(
            self.env_task_name, groundStation, ModelPriority=priority
        )
