import numpy as np

from Basilisk.utilities import (
    macros as mc,
    simIncludeGravBody,
)

from Basilisk.simulation import (
    ephemerisConverter,
    groundLocation,
    eclipse,
    exponentialAtmosphere,
    extForceTorque,
)

from Basilisk import __path__

bskPath = __path__[0]


class EnvironmentModel:
    """Defines the Earth Environment."""

    def __init__(self, SimBase, envRate):
        # Define class variables
        self.imagingTargetList = []
        self.extForceTorqueObjectList = []

        # Define process name, task name and task time-step
        self.envTaskName = "EnvironmentTask"
        processTasksTimeStep = mc.sec2nano(envRate)

        # Create task
        SimBase.envProc.addTask(
            SimBase.CreateNewTask(self.envTaskName, processTasksTimeStep)
        )

        # Initialize all modules and write init one-time messages
        self.InitAllEnvObjects(SimBase)

        # Add modules to environment task
        SimBase.AddModelToTask(
            self.envTaskName, self.gravFactory.spiceObject, ModelPriority=1100
        )
        SimBase.AddModelToTask(
            self.envTaskName, self.boulderGroundStation, ModelPriority=1399
        )
        SimBase.AddModelToTask(
            self.envTaskName, self.merrittGroundStation, ModelPriority=1398
        )
        SimBase.AddModelToTask(
            self.envTaskName, self.singaporeGroundStation, ModelPriority=1397
        )
        SimBase.AddModelToTask(
            self.envTaskName, self.weilheimGroundStation, ModelPriority=1396
        )
        SimBase.AddModelToTask(
            self.envTaskName, self.santiagoGroundStation, ModelPriority=1395
        )
        SimBase.AddModelToTask(
            self.envTaskName, self.dongaraGroundStation, ModelPriority=1394
        )
        SimBase.AddModelToTask(
            self.envTaskName, self.hawaiiGroundStation, ModelPriority=1393
        )
        SimBase.AddModelToTask(self.envTaskName, self.eclipseObject, ModelPriority=988)
        SimBase.AddModelToTask(self.envTaskName, self.ephemConverter, ModelPriority=988)
        SimBase.AddModelToTask(self.envTaskName, self.densityModel, ModelPriority=1000)
        for ind in range(SimBase.n_spacecraft):
            SimBase.AddModelToTask(
                self.envTaskName, self.imagingTargetList[ind], ModelPriority=2000
            )

    def SetGravityBodies(self, SimBase):
        """
        Specify what gravitational bodies to include in the simulation.
        """
        self.gravFactory = simIncludeGravBody.gravBodyFactory()
        self.gravFactory.createSun()
        self.planet = self.gravFactory.createEarth()
        self.sun = 0
        self.earth = 1

        self.planet.isCentralBody = (
            True  # ensure this is the central gravitational body
        )
        self.planet.useSphericalHarmParams = True
        simIncludeGravBody.loadGravFromFile(
            bskPath + "/supportData/LocalGravData/GGM03S.txt", self.planet.spherHarm, 10
        )

        # setup Spice interface for some solar system bodies
        timeInitString = SimBase.initial_conditions["0"].get("utc_init")
        self.gravFactory.createSpiceInterface(
            bskPath + "/supportData/EphemerisData/", timeInitString
        )

        self.gravFactory.spiceObject.zeroBase = (
            "earth"  # Make sure that the Earth is the zero base
        )

    def SetEpochObject(self):
        """
        Add the ephemeris object to use with the SPICE library.
        """
        self.ephemConverter = ephemerisConverter.EphemerisConverter()
        self.ephemConverter.ModelTag = "ephemConverter"
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun]
        )
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )

    def SetEclipseObject(self):
        """
        Specify what celestial object is causing an eclipse message.
        """
        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addPlanetToModel(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.eclipseObject.sunInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun]
        )

    def SetAtmosphereDensityModel(self, SimBase):
        self.densityModel = exponentialAtmosphere.ExponentialAtmosphere()
        self.densityModel.ModelTag = "expDensity"
        self.densityModel.planetRadius = SimBase.initial_conditions["env_params"].get(
            "planetRadius"
        )
        self.densityModel.baseDensity = SimBase.initial_conditions["env_params"].get(
            "baseDensity"
        )
        self.densityModel.scaleHeight = SimBase.initial_conditions["env_params"].get(
            "scaleHeight"
        )
        self.densityModel.planetPosInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )

    def SetDisturbanceTorque(self, SimBase):
        # Loop through every spacecraft to get the disturbance torque
        for ind in range(SimBase.n_spacecraft):
            disturbance_magnitude = SimBase.initial_conditions[str(ind)].get(
                "disturbance_magnitude"
            )
            disturbance_vector = SimBase.initial_conditions[str(ind)].get(
                "disturbance_vector"
            )
            unit_disturbance = disturbance_vector / np.linalg.norm(disturbance_vector)
            extForceTorqueObject = extForceTorque.ExtForceTorque()
            extForceTorqueObject.ModelTag = "DisturbanceTorque"
            extForceTorqueObject.extTorquePntB_B = (
                disturbance_magnitude * unit_disturbance
            )
            self.extForceTorqueObjectList.append(extForceTorqueObject)

    def SetGroundLocations(self, SimBase):
        """
        Specify which ground locations are of interest.
        """
        # Create a Boulder-based ground station
        self.boulderGroundStation = groundLocation.GroundLocation()
        self.boulderGroundStation.ModelTag = "GroundStation1"
        self.boulderGroundStation.planetRadius = SimBase.initial_conditions[
            "env_params"
        ].get("groundLocationPlanetRadius")
        self.boulderGroundStation.specifyLocation(
            SimBase.initial_conditions["env_params"].get("boulderGroundStationLat"),
            SimBase.initial_conditions["env_params"].get("boulderGroundStationLong"),
            SimBase.initial_conditions["env_params"].get("boulderGroundStationAlt"),
        )
        self.boulderGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.boulderGroundStation.minimumElevation = SimBase.initial_conditions[
            "env_params"
        ].get("boulderMinimumElevation")
        self.boulderGroundStation.maximumRange = SimBase.initial_conditions[
            "env_params"
        ].get("boulderMaximumRange")

        # Create a Merritt-Island ground station (NASA's Near Earth Network)
        self.merrittGroundStation = groundLocation.GroundLocation()
        self.merrittGroundStation.ModelTag = "GroundStation2"
        self.merrittGroundStation.planetRadius = SimBase.initial_conditions[
            "env_params"
        ].get("groundLocationPlanetRadius")
        self.merrittGroundStation.specifyLocation(
            SimBase.initial_conditions["env_params"].get("merrittGroundStationLat"),
            SimBase.initial_conditions["env_params"].get("merrittGroundStationLong"),
            SimBase.initial_conditions["env_params"].get("merrittGroundStationAlt"),
        )
        self.merrittGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.merrittGroundStation.minimumElevation = SimBase.initial_conditions[
            "env_params"
        ].get("merrittMinimumElevation")
        self.merrittGroundStation.maximumRange = SimBase.initial_conditions[
            "env_params"
        ].get("merrittMaximumRange")

        # Create a Singapore ground station (NASA's Near Earth Network)
        self.singaporeGroundStation = groundLocation.GroundLocation()
        self.singaporeGroundStation.ModelTag = "GroundStation3"
        self.singaporeGroundStation.planetRadius = SimBase.initial_conditions[
            "env_params"
        ].get("groundLocationPlanetRadius")
        self.singaporeGroundStation.specifyLocation(
            SimBase.initial_conditions["env_params"].get("singaporeGroundStationLat"),
            SimBase.initial_conditions["env_params"].get("singaporeGroundStationLong"),
            SimBase.initial_conditions["env_params"].get("singaporeGroundStationAlt"),
        )
        self.singaporeGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.singaporeGroundStation.minimumElevation = SimBase.initial_conditions[
            "env_params"
        ].get("singaporeMinimumElevation")
        self.singaporeGroundStation.maximumRange = SimBase.initial_conditions[
            "env_params"
        ].get("singaporeMaximumRange")

        # Create a Weilheim Germany ground station (NASA's Near Earth Network)
        self.weilheimGroundStation = groundLocation.GroundLocation()
        self.weilheimGroundStation.ModelTag = "GroundStation4"
        self.weilheimGroundStation.planetRadius = SimBase.initial_conditions[
            "env_params"
        ].get("groundLocationPlanetRadius")
        self.weilheimGroundStation.specifyLocation(
            SimBase.initial_conditions["env_params"].get("weilheimGroundStationLat"),
            SimBase.initial_conditions["env_params"].get("weilheimGroundStationLong"),
            SimBase.initial_conditions["env_params"].get("weilheimGroundStationAlt"),
        )
        self.weilheimGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.weilheimGroundStation.minimumElevation = SimBase.initial_conditions[
            "env_params"
        ].get("weilheimMinimumElevation")
        self.weilheimGroundStation.maximumRange = SimBase.initial_conditions[
            "env_params"
        ].get("weilheimMaximumRange")

        # Create a Santiago, Chile ground station (NASA's Near Earth Network)
        self.santiagoGroundStation = groundLocation.GroundLocation()
        self.santiagoGroundStation.ModelTag = "GroundStation5"
        self.santiagoGroundStation.planetRadius = SimBase.initial_conditions[
            "env_params"
        ].get("groundLocationPlanetRadius")
        self.santiagoGroundStation.specifyLocation(
            SimBase.initial_conditions["env_params"].get("santiagoGroundStationLat"),
            SimBase.initial_conditions["env_params"].get("santiagoGroundStationLong"),
            SimBase.initial_conditions["env_params"].get("santiagoGroundStationAlt"),
        )
        self.santiagoGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.santiagoGroundStation.minimumElevation = SimBase.initial_conditions[
            "env_params"
        ].get("santiagoMinimumElevation")
        self.santiagoGroundStation.maximumRange = SimBase.initial_conditions[
            "env_params"
        ].get("santiagoMaximumRange")

        # Create a Dongara, Australia ground station (NASA's Near Earth Network)
        self.dongaraGroundStation = groundLocation.GroundLocation()
        self.dongaraGroundStation.ModelTag = "GroundStation6"
        self.dongaraGroundStation.planetRadius = SimBase.initial_conditions[
            "env_params"
        ].get("groundLocationPlanetRadius")
        self.dongaraGroundStation.specifyLocation(
            SimBase.initial_conditions["env_params"].get("dongaraGroundStationLat"),
            SimBase.initial_conditions["env_params"].get("dongaraGroundStationLong"),
            SimBase.initial_conditions["env_params"].get("dongaraGroundStationAlt"),
        )
        self.dongaraGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.dongaraGroundStation.minimumElevation = SimBase.initial_conditions[
            "env_params"
        ].get("dongaraMinimumElevation")
        self.dongaraGroundStation.maximumRange = SimBase.initial_conditions[
            "env_params"
        ].get("dongaraMaximumRange")

        # Create a Dongara, Australia ground station (NASA's Near Earth Network)
        self.hawaiiGroundStation = groundLocation.GroundLocation()
        self.hawaiiGroundStation.ModelTag = "GroundStation7"
        self.hawaiiGroundStation.planetRadius = SimBase.initial_conditions[
            "env_params"
        ].get("groundLocationPlanetRadius")
        self.hawaiiGroundStation.specifyLocation(
            SimBase.initial_conditions["env_params"].get("hawaiiGroundStationLat"),
            SimBase.initial_conditions["env_params"].get("hawaiiGroundStationLong"),
            SimBase.initial_conditions["env_params"].get("hawaiiGroundStationAlt"),
        )
        self.hawaiiGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.hawaiiGroundStation.minimumElevation = SimBase.initial_conditions[
            "env_params"
        ].get("hawaiiMinimumElevation")
        self.hawaiiGroundStation.maximumRange = SimBase.initial_conditions[
            "env_params"
        ].get("hawaiiMaximumRange")

    def SetImagingTarget(self, SimBase):
        # Create one imaging target per spacecraft
        for ind in range(SimBase.n_spacecraft):
            imagingTarget = groundLocation.GroundLocation()
            imagingTarget.ModelTag = "ImagingTarget"
            imagingTarget.planetRadius = SimBase.initial_conditions["env_params"].get(
                "groundLocationPlanetRadius"
            )
            # Just initialize to Hawaii... will get re-initialized with target during
            # step
            imagingTarget.specifyLocation(
                SimBase.initial_conditions["env_params"].get("hawaiiGroundStationLat"),
                SimBase.initial_conditions["env_params"].get("hawaiiGroundStationLong"),
                SimBase.initial_conditions["env_params"].get("hawaiiGroundStationAlt"),
            )
            imagingTarget.planetInMsg.subscribeTo(
                self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
            )
            imagingTarget.minimumElevation = SimBase.initial_conditions[str(ind)].get(
                "imageTargetMinimumElevation"
            )
            imagingTarget.maximumRange = SimBase.initial_conditions[str(ind)].get(
                "imageTargetMaximumRange"
            )
            self.imagingTargetList.append(imagingTarget)

    # Global call to initialize every module
    def InitAllEnvObjects(self, SimBase):
        self.SetGravityBodies(SimBase)
        self.SetEpochObject()
        self.SetEclipseObject()
        self.SetGroundLocations(SimBase)
        self.SetImagingTarget(SimBase)
        self.SetAtmosphereDensityModel(SimBase)
        self.SetDisturbanceTorque(SimBase)
