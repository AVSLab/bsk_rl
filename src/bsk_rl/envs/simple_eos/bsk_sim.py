from time import sleep

import numpy as np
from Basilisk import __path__
from Basilisk.architecture import cMsgCInterfacePy as cMsgPy
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (
    attTrackingError,
    hillPoint,
    mrpFeedback,
    rwMotorTorque,
    thrForceMapping,
    thrMomentumDumping,
    thrMomentumManagement,
)
from Basilisk.simulation import (
    ReactionWheelPower,
    eclipse,
    ephemerisConverter,
    exponentialAtmosphere,
    extForceTorque,
    facetDragDynamicEffector,
    groundLocation,
    simpleBattery,
    simpleInstrument,
    simpleNav,
    simplePowerSink,
    simpleSolarPanel,
    simpleStorageUnit,
    spacecraft,
    spaceToGroundTransmitter,
)

#   Basilisk modules
from Basilisk.utilities import SimulationBaseClass, astroFunctions
from Basilisk.utilities import macros as mc
from Basilisk.utilities import (
    orbitalMotion,
    simIncludeGravBody,
    unitTestSupport,
    vizSupport,
)
from numpy.random import uniform

from bsk_rl.utilities.effector_primitives import actuator_primitives as ap
from bsk_rl.utilities.initial_conditions import leo_orbit, sc_attitudes

bskPath = __path__[0]


class SimpleEOSSimulator(SimulationBaseClass.SimBaseClass):
    """
    Simulates a spacecraft in LEO with atmospheric drag and J2 collecting imagery of
    the Earth.

    Dynamics Components
    - Forces: J2, Atmospheric Drag w/ COM offset
    - Environment: Exponential density model; eclipse
    - Actuators: ExternalForceTorque
    - Sensors: SimpleNav
    - Power System: SimpleBattery, SimpleSink, SimpleSolarPanel
    - Data Management System: spaceToGroundTransmitter, simpleStorageUnit,
        simpleInstrument

    FSW Components:
    - MRP Feedback controller
    - inertial3d (sun pointing), hillPoint (nadir pointing)
    - Desat
    """

    def __init__(
        self, dynRate, fswRate, step_duration, initial_conditions=None, render=False
    ):
        """
        Creates the simulation, but does not initialize the initial conditions.
        """
        self.dynRate = dynRate
        self.fswRate = fswRate

        self.step_duration = step_duration

        SimulationBaseClass.SimBaseClass.__init__(self)

        self.attRefMsg = None

        self.simTime = 0.0

        # Initialize performance metrics
        self.downlinked = 0.0
        self.total_access = 0.0
        self.utilized_access = 0.0

        # If no initial conditions are defined yet, set ICs
        if initial_conditions is None:
            self.initial_conditions = self.set_ICs()
        # If ICs were passed through, use the ones that were passed through
        else:
            self.initial_conditions = initial_conditions

        self.DynModels = []
        self.FSWModels = []

        #   Initialize the dynamics and fsw task groups and modules
        self.DynamicsProcessName = "DynamicsProcess"  # Create simulation process name
        self.dynProc = self.CreateNewProcess(self.DynamicsProcessName)  # Create process
        self.dynTaskName = "DynTask"
        self.spiceTaskName = "SpiceTask"
        self.envTaskName = "EnvTask"
        self.dynTask = self.dynProc.addTask(
            self.CreateNewTask(self.dynTaskName, mc.sec2nano(self.dynRate)),
            taskPriority=200,
        )
        self.spiceTask = self.dynProc.addTask(
            self.CreateNewTask(self.spiceTaskName, mc.sec2nano(self.dynRate)),
            taskPriority=199,
        )
        self.envTask = self.dynProc.addTask(
            self.CreateNewTask(self.envTaskName, mc.sec2nano(self.dynRate)),
            taskPriority=198,
        )

        self.obs = np.zeros([23, 1])
        self.obs_full = np.zeros([23, 1])

        self.curr_step = 0
        self.max_steps = 0
        self.max_length = 0

        self.set_dynamics()
        self.setupGatewayMsgs()
        self.set_fsw()
        if render:
            self.setup_viz()
            self.clear_logs = False
        else:
            self.clear_logs = True

        self.set_logging()
        self.previousPointingGoal = "sunPointTask"

        self.modeRequest = None
        self.InitializeSimulation()

        return

    def __del__(self):
        self.close_gracefully()

        if self.clear_logs:
            self.boulderGSLog.clear()
            self.merrittGSLog.clear()
            self.singaporeGSLog.clear()
            self.weilheimGSLog.clear()
            self.santiagoGSLog.clear()
            self.dongaraGSLog.clear()
            self.hawaiiGSLog.clear()
            self.transmitterLog.clear()

        print("Destructor called, simulation deleted")

    def set_ICs(self):
        # Sample orbital parameters
        oe, rN, vN = leo_orbit.sampled_500km_boulder_gs()

        # Sample attitude and rates
        sigma_init, omega_init = sc_attitudes.random_tumble(maxSpinRate=0.00001)

        # Dict of initial conditions
        initial_conditions = {
            # Mass
            "mass": 330,  # kg
            # Orbital parameters
            "oe": oe,
            "rN": rN,
            "vN": vN,
            # Spacecraft dimensions
            "width": 1.38,
            "depth": 1.04,
            "height": 1.58,
            # Attitude and rate initialization
            "sigma_init": sigma_init,
            "omega_init": omega_init,
            # Atmospheric density
            "planetRadius": orbitalMotion.REQ_EARTH * 1000.0,
            "baseDensity": 1.22,  # kg/m^3
            "scaleHeight": 8e3,  # m
            # Disturbance Torque
            "disturbance_magnitude": 2e-3,
            "disturbance_vector": np.random.standard_normal(3),
            # Reaction Wheel speeds
            "wheelSpeeds": uniform(-4000 * mc.RPM, 4000 * mc.RPM, 3),  # rad/s
            # Solar Panel Parameters
            "nHat_B": np.array([0, 1, 0]),
            "panelArea": 2 * 1.0 * 0.5,
            "panelEfficiency": 0.20,
            # Power Sink Parameters
            "instrumentPowerDraw": -30.0,  # W, Assuming 50 W imager (Harris Spaceview)
            "transmitterPowerDraw": -15.0,  # W
            "rwBasePower": 0.4,  # W, Note the opposite convention
            "rwMechToElecEfficiency": 0.0,
            "rwElecToMechEfficiency": 0.5,
            # Battery Parameters
            "batteryStorageCapacity": 80.0 * 3600.0,
            "storedCharge_Init": np.random.uniform(30.0 * 3600.0, 50.0 * 3600.0, 1)[0],
            # Sun pointing FSW config
            "sigma_R0N": [1, 0, 0],
            # RW motor torque and thruster force mapping FSW config
            "controlAxes_B": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            # Attitude controller FSW config
            "K": 7,
            "Ki": -1.0,  # Note: make value negative to turn off integral feedback
            "P": 35,
            # Momentum dumping config
            "hs_min": 4.0,  # Nms
            # Thruster force mapping FSW module
            "thrForceSign": 1,
            # Thruster momentum dumping FSW config
            "maxCounterValue": 8,
            "thrMinFireTime": 0.002,  # Seconds
            # Ground station - Located in Boulder, CO
            "boulderGroundStationPlanetRadius": astroFunctions.E_radius * 1e3,
            "boulderGroundStationLat": np.radians(40.009971),  # 40.0150 N Latitude
            "boulderGroundStationLong": np.radians(-105.243895),  # 105.2705 W Longitude
            "boulderGroundStationAlt": 1624,  # Altitude
            "boulderMinimumElevation": np.radians(10.0),
            "boulderMaximumRange": 1e9,
            # Ground station - Located in Merritt Island, FL
            "merrittGroundStationPlanetRadius": astroFunctions.E_radius * 1e3,
            "merrittGroundStationLat": np.radians(28.3181),  # 28.3181 N Latitude
            "merrittGroundStationLong": np.radians(-80.6660),  # 80.6660 W Longitude
            "merrittGroundStationAlt": 0.9144,  # Altitude
            "merrittMinimumElevation": np.radians(10.0),
            "merrittMaximumRange": 1e9,
            # Ground station - Located in Singapore
            "singaporeGroundStationPlanetRadius": astroFunctions.E_radius * 1e3,
            "singaporeGroundStationLat": np.radians(1.3521),  # 1.3521 N Latitude
            "singaporeGroundStationLong": np.radians(103.8198),  # 103.8198 E Longitude
            "singaporeGroundStationAlt": 15,  # Altitude, m
            "singaporeMinimumElevation": np.radians(10.0),
            "singaporeMaximumRange": 1e9,
            # Ground station - Located in Weilheim, Germany
            "weilheimGroundStationPlanetRadius": astroFunctions.E_radius * 1e3,
            "weilheimGroundStationLat": np.radians(47.8407),  # 47.8407 N Latitude
            "weilheimGroundStationLong": np.radians(11.1421),  # 11.1421 E Longitude
            "weilheimGroundStationAlt": 563,  # Altitude, m
            "weilheimMinimumElevation": np.radians(10.0),
            "weilheimMaximumRange": 1e9,
            # Ground station - Located in Santiago, Chile
            "santiagoGroundStationPlanetRadius": astroFunctions.E_radius * 1e3,
            "santiagoGroundStationLat": np.radians(-33.4489),  # 33.4489 S Latitude
            "santiagoGroundStationLong": np.radians(-70.6693),  # 70.6693 W Longitude
            "santiagoGroundStationAlt": 570,  # Altitude, m
            "santiagoMinimumElevation": np.radians(10.0),
            "santiagoMaximumRange": 1e9,
            # Ground station - Located in Dongara, Australia
            "dongaraGroundStationPlanetRadius": astroFunctions.E_radius * 1e3,
            "dongaraGroundStationLat": np.radians(-29.2452),  # 29.2452 S Latitude
            "dongaraGroundStationLong": np.radians(114.9326),  # 114.9326 E Longitude
            "dongaraGroundStationAlt": 34,  # Altitude, m
            "dongaraMinimumElevation": np.radians(10.0),
            "dongaraMaximumRange": 1e9,
            # Ground station - Located in Hawaii
            "hawaiiGroundStationPlanetRadius": astroFunctions.E_radius * 1e3,
            "hawaiiGroundStationLat": np.radians(19.8968),  # 19.8968 N Latitude
            "hawaiiGroundStationLong": np.radians(-155.5828),  # 155.5828 W Longitude
            "hawaiiGroundStationAlt": 9,  # Altitude, m
            "hawaiiMinimumElevation": np.radians(10.0),
            "hawaiiMaximumRange": 1e9,
            # Data-generating instrument
            "instrumentBaudRate": 4e6,  # baud, 8e6 = 1 MB = 1 image
            # Transmitter
            "transmitterBaudRate": -4e6,  # 4 Mbits/s
            "transmitterPacketSize": -1,  # bits
            "transmitterNumBuffers": 1,
            # Data Storage Unit
            "dataStorageCapacity": 8e9,  # bits (8E9 = 1 GB)
        }

        return initial_conditions

    def set_dynamics(self):
        """
        Sets up the dynamics modules for the sim. This simulator runs:
        scObject (spacecraft dynamics simulation)
        SpiceObject
        EclipseObject (simulates eclipse for simpleSolarPanel)
        extForceTorque (attitude actuation)
        simpleNav (attitude determination/sensing)
        simpleSolarPanel (attitude-dependent power generation)
        simpleBattery (power storage)
        simplePowerNode (constant power draw)

        By default, parameters are set to those for a 6U cubesat.
        :return:
        """
        sc_number = 0

        #   Spacecraft, Planet Setup
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "spacecraft"

        # clear prior gravitational body and SPICE setup definitions
        self.gravFactory = simIncludeGravBody.gravBodyFactory()

        self.gravFactory.createSun()
        self.sun = 0

        self.planet = self.gravFactory.createEarth()
        self.earth = 1

        self.planet.isCentralBody = (
            True  # ensure this is the central gravitational body
        )
        self.planet.useSphericalHarmParams = True
        simIncludeGravBody.loadGravFromFile(
            bskPath + "/supportData/LocalGravData/GGM03S.txt", self.planet.spherHarm, 10
        )

        # attach gravity model to spaceCraftPlus
        self.scObject.gravField.gravBodies = spacecraft.GravBodyVector(
            list(self.gravFactory.gravBodies.values())
        )

        sleep(1)

        # setup Spice interface for some solar system bodies
        timeInitString = "2021 MAY 04 07:47:48.965 (UTC)"

        self.gravFactory.createSpiceInterface(
            bskPath + "/supportData/EphemerisData/", timeInitString
        )

        self.gravFactory.spiceObject.zeroBase = (
            "earth"  # Make sure that the Earth is the zero base
        )

        rN = self.initial_conditions.get("rN")
        vN = self.initial_conditions.get("vN")

        width = self.initial_conditions.get("width")
        depth = self.initial_conditions.get("depth")
        height = self.initial_conditions.get("height")

        # Grab the mass for readability in inertia computation
        mass = self.initial_conditions.get("mass")

        MOI = [
            1.0 / 12.0 * mass * (width**2.0 + depth**2.0),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * mass * (depth**2.0 + height**2.0),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * mass * (width**2.0 + height**2.0),
        ]

        self.scObject.hub.mHub = mass  # kg
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(MOI)

        self.scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(rN)
        self.scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(vN)

        sigma_init = self.initial_conditions.get("sigma_init")
        omega_init = self.initial_conditions.get("omega_init")

        self.scObject.hub.sigma_BNInit = sigma_init  # sigma_BN_B
        self.scObject.hub.omega_BN_BInit = omega_init

        # Set up density model
        self.densityModel = exponentialAtmosphere.ExponentialAtmosphere()
        self.densityModel.ModelTag = "expDensity"
        self.densityModel.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.densityModel.planetRadius = self.initial_conditions.get("planetRadius")
        self.densityModel.baseDensity = self.initial_conditions.get("baseDensity")
        self.densityModel.scaleHeight = self.initial_conditions.get("scaleHeight")
        self.densityModel.planetPosInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )

        # Set up drag effector
        self.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        self.dragEffector.ModelTag = "FacetDrag"
        #  Set up the geometry of a small satellite, starting w/ bus
        self.dragEffector.addFacet(width * depth, 2.2, [1, 0, 0], [height / 2, 0.0, 0])
        self.dragEffector.addFacet(width * depth, 2.2, [-1, 0, 0], [height / 2, 0.0, 0])
        self.dragEffector.addFacet(height * width, 2.2, [0, 1, 0], [0, depth / 2, 0])
        self.dragEffector.addFacet(height * width, 2.2, [0, -1, 0], [0, -depth / 2, 0])
        self.dragEffector.addFacet(height * depth, 2.2, [0, 0, 1], [0, 0, width / 2])
        self.dragEffector.addFacet(height * depth, 2.2, [0, 0, -1], [0, 0, -width / 2])
        # Add solar panels
        self.dragEffector.addFacet(
            self.initial_conditions.get("panelArea") / 2, 2.2, [0, 1, 0], [0, height, 0]
        )
        self.dragEffector.addFacet(
            self.initial_conditions.get("panelArea") / 2,
            2.2,
            [0, -1, 0],
            [0, height, 0],
        )
        self.dragEffector.atmoDensInMsg.subscribeTo(self.densityModel.envOutMsgs[-1])
        self.scObject.addDynamicEffector(self.dragEffector)

        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.eclipseObject.addPlanetToModel(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.eclipseObject.sunInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun]
        )

        self.ephemConverter = ephemerisConverter.EphemerisConverter()
        self.ephemConverter.ModelTag = "ephemConverter"
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun]
        )
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )

        # Disturbance Torque Setup
        disturbance_magnitude = self.initial_conditions.get("disturbance_magnitude")
        disturbance_vector = self.initial_conditions.get("disturbance_vector")
        unit_disturbance = disturbance_vector / np.linalg.norm(disturbance_vector)
        self.extForceTorqueObject = extForceTorque.ExtForceTorque()
        self.extForceTorqueObject.extTorquePntB_B = (
            disturbance_magnitude * unit_disturbance
        )
        self.extForceTorqueObject.ModelTag = "DisturbanceTorque"
        self.scObject.addDynamicEffector(self.extForceTorqueObject)

        # Add reaction wheels to the spacecraft
        self.rwStateEffector, rwFactory, initWheelSpeeds = ap.balancedHR16Triad(
            useRandom=False,
            randomBounds=(-800, 800),
            wheelSpeeds=self.initial_conditions.get("wheelSpeeds"),
        )
        rwFactory.addToSpacecraft("ReactionWheels", self.rwStateEffector, self.scObject)
        self.rwConfigMsg = rwFactory.getConfigMessage()

        # Add thrusters to the spacecraft
        self.thrusterSet, thrFactory = ap.idealMonarc1Octet()
        thrModelTag = "ACSThrusterDynamics"
        self.thrusterConfigMsg = thrFactory.getConfigMessage()
        thrFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)

        # Add simpleNav as a mock estimator to the spacecraft
        self.simpleNavObject = simpleNav.SimpleNav()
        self.simpleNavObject.ModelTag = "SimpleNav"
        self.simpleNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)

        # Power setup
        self.solarPanel = simpleSolarPanel.SimpleSolarPanel()
        self.solarPanel.ModelTag = "solarPanel" + str(sc_number)
        self.solarPanel.stateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.solarPanel.sunEclipseInMsg.subscribeTo(
            self.eclipseObject.eclipseOutMsgs[sc_number]
        )
        self.solarPanel.sunInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun]
        )
        self.solarPanel.setPanelParameters(
            unitTestSupport.np2EigenVectorXd(self.initial_conditions.get("nHat_B")),
            self.initial_conditions.get("panelArea"),
            self.initial_conditions.get("panelEfficiency"),
        )

        # Instrument power sink
        self.instrumentPowerSink = simplePowerSink.SimplePowerSink()
        self.instrumentPowerSink.ModelTag = "insPowerSink" + str(sc_number)
        self.instrumentPowerSink.nodePowerOut = self.initial_conditions.get(
            "instrumentPowerDraw"
        )  # Watts

        # Transmitter power sink
        self.transmitterPowerSink = simplePowerSink.SimplePowerSink()
        self.transmitterPowerSink.ModelTag = "transPowerSink" + str(sc_number)
        self.transmitterPowerSink.nodePowerOut = self.initial_conditions.get(
            "transmitterPowerDraw"
        )  # Watts

        # Reaction wheel power sinks
        self.rwPowerList = []
        for ind in range(rwFactory.getNumOfDevices()):
            powerRW = ReactionWheelPower.ReactionWheelPower()
            powerRW.ModelTag = self.scObject.ModelTag
            # powerRW.ModelTag = "rw" + str(c) + "PowerSink"
            powerRW.basePowerNeed = self.initial_conditions.get(
                "rwBasePower"
            )  # baseline power draw, Watts
            powerRW.rwStateInMsg.subscribeTo(self.rwStateEffector.rwOutMsgs[ind])
            powerRW.mechToElecEfficiency = self.initial_conditions.get(
                "rwMechToElecEfficiency"
            )
            powerRW.elecToMechEfficiency = self.initial_conditions.get(
                "rwElecToMechEfficiency"
            )
            self.AddModelToTask(self.dynTaskName, powerRW, ModelPriority=(987 - ind))
            self.rwPowerList.append(powerRW)

        # Battery
        self.powerMonitor = simpleBattery.SimpleBattery()
        self.powerMonitor.ModelTag = "powerMonitor"
        self.powerMonitor.storageCapacity = self.initial_conditions.get(
            "batteryStorageCapacity"
        )
        self.powerMonitor.storedCharge_Init = self.initial_conditions.get(
            "storedCharge_Init"
        )
        self.powerMonitor.addPowerNodeToModel(self.solarPanel.nodePowerOutMsg)
        self.powerMonitor.addPowerNodeToModel(self.instrumentPowerSink.nodePowerOutMsg)
        self.powerMonitor.addPowerNodeToModel(self.transmitterPowerSink.nodePowerOutMsg)
        for powerRW in self.rwPowerList:
            self.powerMonitor.addPowerNodeToModel(powerRW.nodePowerOutMsg)

        # Create a Boulder-based ground station
        self.boulderGroundStation = groundLocation.GroundLocation()
        self.boulderGroundStation.ModelTag = "GroundStation1"
        self.boulderGroundStation.planetRadius = self.initial_conditions.get(
            "boulderGroundStationPlanetRadius"
        )
        self.boulderGroundStation.specifyLocation(
            self.initial_conditions.get("boulderGroundStationLat"),
            self.initial_conditions.get("boulderGroundStationLong"),
            self.initial_conditions.get("boulderGroundStationAlt"),
        )
        self.boulderGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.boulderGroundStation.minimumElevation = self.initial_conditions.get(
            "boulderMinimumElevation"
        )
        self.boulderGroundStation.maximumRange = self.initial_conditions.get(
            "boulderMaximumRange"
        )
        self.boulderGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)

        # Create a Merritt-Island ground station (NASA's Near Earth Network)
        self.merrittGroundStation = groundLocation.GroundLocation()
        self.merrittGroundStation.ModelTag = "GroundStation2"
        self.merrittGroundStation.planetRadius = self.initial_conditions.get(
            "merrittGroundStationPlanetRadius"
        )
        self.merrittGroundStation.specifyLocation(
            self.initial_conditions.get("merrittGroundStationLat"),
            self.initial_conditions.get("merrittGroundStationLong"),
            self.initial_conditions.get("merrittGroundStationAlt"),
        )
        self.merrittGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.merrittGroundStation.minimumElevation = self.initial_conditions.get(
            "merrittMinimumElevation"
        )
        self.merrittGroundStation.maximumRange = self.initial_conditions.get(
            "merrittMaximumRange"
        )
        self.merrittGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)

        # Create a Singapore ground station (NASA's Near Earth Network)
        self.singaporeGroundStation = groundLocation.GroundLocation()
        self.singaporeGroundStation.ModelTag = "GroundStation3"
        self.singaporeGroundStation.planetRadius = self.initial_conditions.get(
            "singaporeGroundStationPlanetRadius"
        )
        self.singaporeGroundStation.specifyLocation(
            self.initial_conditions.get("singaporeGroundStationLat"),
            self.initial_conditions.get("singaporeGroundStationLong"),
            self.initial_conditions.get("singaporeGroundStationAlt"),
        )
        self.singaporeGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.singaporeGroundStation.minimumElevation = self.initial_conditions.get(
            "singaporeMinimumElevation"
        )
        self.singaporeGroundStation.maximumRange = self.initial_conditions.get(
            "singaporeMaximumRange"
        )
        self.singaporeGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)

        # Create a Weilheim Germany ground station (NASA's Near Earth Network)
        self.weilheimGroundStation = groundLocation.GroundLocation()
        self.weilheimGroundStation.ModelTag = "GroundStation4"
        self.weilheimGroundStation.planetRadius = self.initial_conditions.get(
            "weilheimGroundStationPlanetRadius"
        )
        self.weilheimGroundStation.specifyLocation(
            self.initial_conditions.get("weilheimGroundStationLat"),
            self.initial_conditions.get("weilheimGroundStationLong"),
            self.initial_conditions.get("weilheimGroundStationAlt"),
        )
        self.weilheimGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.weilheimGroundStation.minimumElevation = self.initial_conditions.get(
            "weilheimMinimumElevation"
        )
        self.weilheimGroundStation.maximumRange = self.initial_conditions.get(
            "weilheimMaximumRange"
        )
        self.weilheimGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)

        # Create a Santiago, Chile ground station (NASA's Near Earth Network)
        self.santiagoGroundStation = groundLocation.GroundLocation()
        self.santiagoGroundStation.ModelTag = "GroundStation5"
        self.santiagoGroundStation.planetRadius = self.initial_conditions.get(
            "santiagoGroundStationPlanetRadius"
        )
        self.santiagoGroundStation.specifyLocation(
            self.initial_conditions.get("santiagoGroundStationLat"),
            self.initial_conditions.get("santiagoGroundStationLong"),
            self.initial_conditions.get("santiagoGroundStationAlt"),
        )
        self.santiagoGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.santiagoGroundStation.minimumElevation = self.initial_conditions.get(
            "santiagoMinimumElevation"
        )
        self.santiagoGroundStation.maximumRange = self.initial_conditions.get(
            "santiagoMaximumRange"
        )
        self.santiagoGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)

        # Create a Dongara, Australia ground station (NASA's Near Earth Network)
        self.dongaraGroundStation = groundLocation.GroundLocation()
        self.dongaraGroundStation.ModelTag = "GroundStation6"
        self.dongaraGroundStation.planetRadius = self.initial_conditions.get(
            "dongaraGroundStationPlanetRadius"
        )
        self.dongaraGroundStation.specifyLocation(
            self.initial_conditions.get("dongaraGroundStationLat"),
            self.initial_conditions.get("dongaraGroundStationLong"),
            self.initial_conditions.get("dongaraGroundStationAlt"),
        )
        self.dongaraGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.dongaraGroundStation.minimumElevation = self.initial_conditions.get(
            "dongaraMinimumElevation"
        )
        self.dongaraGroundStation.maximumRange = self.initial_conditions.get(
            "dongaraMaximumRange"
        )
        self.dongaraGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)

        # Create a Dongara, Australia ground station (NASA's Near Earth Network)
        self.hawaiiGroundStation = groundLocation.GroundLocation()
        self.hawaiiGroundStation.ModelTag = "GroundStation7"
        self.hawaiiGroundStation.planetRadius = self.initial_conditions.get(
            "hawaiiGroundStationPlanetRadius"
        )
        self.hawaiiGroundStation.specifyLocation(
            self.initial_conditions.get("hawaiiGroundStationLat"),
            self.initial_conditions.get("hawaiiGroundStationLong"),
            self.initial_conditions.get("hawaiiGroundStationAlt"),
        )
        self.hawaiiGroundStation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
        )
        self.hawaiiGroundStation.minimumElevation = self.initial_conditions.get(
            "hawaiiMinimumElevation"
        )
        self.hawaiiGroundStation.maximumRange = self.initial_conditions.get(
            "hawaiiMaximumRange"
        )
        self.hawaiiGroundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)

        # Create an instrument
        self.instrument = simpleInstrument.SimpleInstrument()
        self.instrument.ModelTag = "instrument" + str(sc_number)
        self.instrument.nodeBaudRate = self.initial_conditions.get(
            "instrumentBaudRate"
        )  # baud
        self.instrument.nodeDataName = "Instrument" + str(sc_number)

        # Create a "transmitter"
        self.transmitter = spaceToGroundTransmitter.SpaceToGroundTransmitter()
        self.transmitter.ModelTag = "transmitter" + str(sc_number)
        self.transmitter.nodeBaudRate = self.initial_conditions.get(
            "transmitterBaudRate"
        )  # baud
        self.transmitter.packetSize = self.initial_conditions.get(
            "transmitterPacketSize"
        )  # bits
        self.transmitter.numBuffers = self.initial_conditions.get(
            "transmitterNumBuffers"
        )
        self.transmitter.addAccessMsgToTransmitter(
            self.boulderGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            self.merrittGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            self.singaporeGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            self.weilheimGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            self.santiagoGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            self.dongaraGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            self.hawaiiGroundStation.accessOutMsgs[-1]
        )

        # Create a partitionedStorageUnit and attach the instrument to it
        self.storageUnit = simpleStorageUnit.SimpleStorageUnit()
        self.storageUnit.ModelTag = "storageUnit" + str(sc_number)
        self.storageUnit.storageCapacity = self.initial_conditions.get(
            "dataStorageCapacity"
        )  # bits (1 GB)
        self.storageUnit.addDataNodeToModel(self.instrument.nodeDataOutMsg)
        self.storageUnit.addDataNodeToModel(self.transmitter.nodeDataOutMsg)

        # Add the storage unit to the transmitter
        self.transmitter.addStorageUnitToTransmitter(
            self.storageUnit.storageUnitDataOutMsg
        )

        # Initialize the observations (normed)
        # Inertial position
        self.obs[0, 0] = self.scObject.hub.r_CN_NInit[0] / np.linalg.norm(
            self.scObject.hub.r_CN_NInit
        )
        self.obs[1, 0] = self.scObject.hub.r_CN_NInit[1] / np.linalg.norm(
            self.scObject.hub.r_CN_NInit
        )
        self.obs[2, 0] = self.scObject.hub.r_CN_NInit[2] / np.linalg.norm(
            self.scObject.hub.r_CN_NInit
        )
        # Inertial velocity
        self.obs[3, 0] = self.scObject.hub.v_CN_NInit[0] / np.linalg.norm(
            self.scObject.hub.v_CN_NInit
        )
        self.obs[4, 0] = self.scObject.hub.v_CN_NInit[1] / np.linalg.norm(
            self.scObject.hub.v_CN_NInit
        )
        self.obs[5, 0] = self.scObject.hub.v_CN_NInit[2] / np.linalg.norm(
            self.scObject.hub.v_CN_NInit
        )
        # Attitude error
        self.obs[6, 0] = np.linalg.norm(self.scObject.hub.sigma_BNInit)
        # Attitude rate
        self.obs[7, 0] = np.linalg.norm(self.scObject.hub.omega_BN_BInit)
        # Wheel speeds
        self.obs[8, 0] = self.initial_conditions.get("wheelSpeeds")[0] / (mc.RPM * 6000)
        self.obs[9, 0] = self.initial_conditions.get("wheelSpeeds")[1] / (mc.RPM * 6000)
        self.obs[10, 0] = self.initial_conditions.get("wheelSpeeds")[2] / (
            mc.RPM * 6000
        )
        # Stored charge
        self.obs[11, 0] = (
            self.powerMonitor.storedCharge_Init
            / self.initial_conditions.get("batteryStorageCapacity")
        )

        # Initialize the full observations
        # Inertial position
        self.obs_full[0:3, 0] = np.asarray(self.scObject.hub.r_CN_NInit).flatten()
        # Inertial velocity
        self.obs_full[3:6, 0] = np.asarray(self.scObject.hub.v_CN_NInit).flatten()
        # Attitude error
        self.obs_full[6, 0] = np.linalg.norm(self.scObject.hub.sigma_BNInit)
        # Attitude rate
        self.obs_full[7, 0] = np.linalg.norm(self.scObject.hub.omega_BN_BInit)
        # Wheel speeds
        self.obs_full[8:11, 0] = (
            self.initial_conditions.get("wheelSpeeds")[0:3] * mc.RPM
        )
        # Stored charge
        self.obs_full[11, 0] = self.powerMonitor.storedCharge_Init / 3600.0

        # Eclipse indicator
        self.obs[12, 0] = self.obs_full[12, 0] = 0
        # Stored data
        self.obs[13, 0] = self.obs_full[13, 0] = 0
        # Transmitted data
        self.obs[14, 0] = self.obs_full[14, 0] = 0
        # Ground Station access indicators
        self.obs[15:22, 0] = self.obs_full[15:22, 0] = 0
        # Set the percentage through the planning interval
        self.obs[22] = self.obs_full[22] = 0

        self.obs = np.around(self.obs, decimals=5)

        # Add all the models to the tasks
        # Spice Task
        self.AddModelToTask(
            self.spiceTaskName, self.gravFactory.spiceObject, ModelPriority=1100
        )

        # Dyn Task
        self.AddModelToTask(self.dynTaskName, self.densityModel, ModelPriority=1000)
        self.AddModelToTask(self.dynTaskName, self.dragEffector, ModelPriority=999)
        self.AddModelToTask(self.dynTaskName, self.simpleNavObject, ModelPriority=998)
        self.AddModelToTask(self.dynTaskName, self.rwStateEffector, ModelPriority=997)
        self.AddModelToTask(self.dynTaskName, self.thrusterSet, ModelPriority=996)
        self.AddModelToTask(self.dynTaskName, self.scObject, ModelPriority=899)
        self.AddModelToTask(self.dynTaskName, self.ephemConverter, ModelPriority=987)

        # Env Task
        self.AddModelToTask(
            self.envTaskName, self.boulderGroundStation, ModelPriority=995
        )
        self.AddModelToTask(
            self.envTaskName, self.merrittGroundStation, ModelPriority=994
        )
        self.AddModelToTask(
            self.envTaskName, self.singaporeGroundStation, ModelPriority=993
        )
        self.AddModelToTask(
            self.envTaskName, self.weilheimGroundStation, ModelPriority=992
        )
        self.AddModelToTask(
            self.envTaskName, self.santiagoGroundStation, ModelPriority=991
        )
        self.AddModelToTask(
            self.envTaskName, self.dongaraGroundStation, ModelPriority=990
        )
        self.AddModelToTask(
            self.envTaskName, self.hawaiiGroundStation, ModelPriority=989
        )
        self.AddModelToTask(self.envTaskName, self.eclipseObject, ModelPriority=988)
        self.AddModelToTask(self.envTaskName, self.solarPanel, ModelPriority=898)
        self.AddModelToTask(
            self.envTaskName, self.instrumentPowerSink, ModelPriority=897
        )
        self.AddModelToTask(
            self.envTaskName, self.transmitterPowerSink, ModelPriority=896
        )
        self.AddModelToTask(self.envTaskName, self.instrument, ModelPriority=895)
        self.AddModelToTask(self.envTaskName, self.powerMonitor, ModelPriority=799)
        self.AddModelToTask(self.envTaskName, self.transmitter, ModelPriority=798)
        self.AddModelToTask(self.envTaskName, self.storageUnit, ModelPriority=699)

        return

    def set_fsw(self):
        """
        Sets up the attitude guidance stack for the simulation. This simulator runs:
        inertial3Dpoint - Sets the attitude guidance objective to point the main panel
            at the sun.
        hillPointTask: Sets the attitude guidance objective to point a "camera" angle
            towards nadir.
        attitudeTrackingError: Computes the difference between estimated and guidance
            attitudes
        mrpFeedbackControl: Computes an appropriate control torque given an attitude
            error
        """

        self.processName = self.DynamicsProcessName
        self.processTasksTimeStep = mc.sec2nano(self.fswRate)  # 0.5
        self.dynProc.addTask(
            self.CreateNewTask("sunPointTask", self.processTasksTimeStep),
            taskPriority=99,
        )
        self.dynProc.addTask(
            self.CreateNewTask("nadirPointTask", self.processTasksTimeStep),
            taskPriority=98,
        )
        self.dynProc.addTask(
            self.CreateNewTask("mrpControlTask", self.processTasksTimeStep),
            taskPriority=96,
        )
        self.dynProc.addTask(
            self.CreateNewTask("rwDesatTask", self.processTasksTimeStep),
            taskPriority=97,
        )

        # Specify the vehicle configuration message to tell things what the vehicle
        # inertia is
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        # use the same inertia in the FSW algorithm as in the simulation
        #   Set inertia properties to those of a solid 6U cubeoid:
        width = self.initial_conditions.get("width")
        depth = self.initial_conditions.get("depth")
        height = self.initial_conditions.get("height")

        # Grab the mass for readability of inertia calc
        mass = self.initial_conditions.get("mass")

        MOI = [
            1.0 / 12.0 * mass * (width**2.0 + depth**2.0),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * mass * (depth**2.0 + height**2.0),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * mass * (width**2.0 + height**2.0),
        ]

        vehicleConfigOut.ISCPntB_B = MOI
        self.vcConfigMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

        #   Sun pointing configuration
        # self.sunPointData = inertial3D.inertial3DConfig()
        self.sunPointData = hillPoint.hillPointConfig()
        self.sunPointWrap = self.setModelDataWrap(self.sunPointData)
        self.sunPointWrap.ModelTag = "sunPoint"
        cMsgPy.AttRefMsg_C_addAuthor(self.sunPointData.attRefOutMsg, self.attRefMsg)
        # self.sunPointData.sigma_R0N = self.initial_conditions.get("sigma_R0N")
        self.sunPointData.transNavInMsg.subscribeTo(self.simpleNavObject.transOutMsg)
        self.sunPointData.celBodyInMsg.subscribeTo(self.ephemConverter.ephemOutMsgs[0])

        #   Earth pointing configuration
        self.hillPointData = hillPoint.hillPointConfig()
        self.hillPointWrap = self.setModelDataWrap(self.hillPointData)
        self.hillPointWrap.ModelTag = "hillPoint"
        cMsgPy.AttRefMsg_C_addAuthor(self.hillPointData.attRefOutMsg, self.attRefMsg)
        self.hillPointData.transNavInMsg.subscribeTo(self.simpleNavObject.transOutMsg)
        self.hillPointData.celBodyInMsg.subscribeTo(self.ephemConverter.ephemOutMsgs[1])

        #   Attitude error configuration
        self.trackingErrorData = attTrackingError.attTrackingErrorConfig()
        self.trackingErrorWrap = self.setModelDataWrap(self.trackingErrorData)
        self.trackingErrorWrap.ModelTag = "trackingError"
        self.trackingErrorData.attNavInMsg.subscribeTo(self.simpleNavObject.attOutMsg)
        self.trackingErrorData.attRefInMsg.subscribeTo(self.attRefMsg)

        #   Attitude controller configuration
        self.mrpFeedbackControlData = mrpFeedback.mrpFeedbackConfig()
        self.mrpFeedbackControlWrap = self.setModelDataWrap(self.mrpFeedbackControlData)
        self.mrpFeedbackControlWrap.ModelTag = "mrpFeedbackControl"
        self.mrpFeedbackControlData.guidInMsg.subscribeTo(
            self.trackingErrorData.attGuidOutMsg
        )
        self.mrpFeedbackControlData.vehConfigInMsg.subscribeTo(self.vcConfigMsg)
        self.mrpFeedbackControlData.K = self.initial_conditions.get("K")
        self.mrpFeedbackControlData.Ki = self.initial_conditions.get("Ki")
        self.mrpFeedbackControlData.P = self.initial_conditions.get("P")
        self.mrpFeedbackControlData.integralLimit = (
            2.0 / self.mrpFeedbackControlData.Ki * 0.1
        )

        # add module that maps the Lr control torque into the RW motor torques
        self.rwMotorTorqueConfig = rwMotorTorque.rwMotorTorqueConfig()
        self.rwMotorTorqueWrap = self.setModelDataWrap(self.rwMotorTorqueConfig)
        self.rwMotorTorqueWrap.ModelTag = "rwMotorTorque"
        self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(
            self.rwMotorTorqueConfig.rwMotorTorqueOutMsg
        )
        self.rwMotorTorqueConfig.rwParamsInMsg.subscribeTo(self.rwConfigMsg)
        self.rwMotorTorqueConfig.vehControlInMsg.subscribeTo(
            self.mrpFeedbackControlData.cmdTorqueOutMsg
        )
        self.rwMotorTorqueConfig.controlAxes_B = self.initial_conditions.get(
            "controlAxes_B"
        )
        self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(
            self.rwMotorTorqueConfig.rwMotorTorqueOutMsg
        )

        #   Momentum dumping configuration
        self.thrDesatControlConfig = thrMomentumManagement.thrMomentumManagementConfig()
        self.thrDesatControlWrap = self.setModelDataWrap(self.thrDesatControlConfig)
        self.thrDesatControlWrap.ModelTag = "thrMomentumManagement"
        self.thrDesatControlConfig.hs_min = self.initial_conditions.get("hs_min")  # Nms
        self.thrDesatControlConfig.rwSpeedsInMsg.subscribeTo(
            self.rwStateEffector.rwSpeedOutMsg
        )
        self.thrDesatControlConfig.rwConfigDataInMsg.subscribeTo(self.rwConfigMsg)

        # setup the thruster force mapping module
        self.thrForceMappingConfig = thrForceMapping.thrForceMappingConfig()
        self.thrForceMappingWrap = self.setModelDataWrap(self.thrForceMappingConfig)
        self.thrForceMappingWrap.ModelTag = "thrForceMapping"
        self.thrForceMappingConfig.cmdTorqueInMsg.subscribeTo(
            self.thrDesatControlConfig.deltaHOutMsg
        )
        self.thrForceMappingConfig.thrConfigInMsg.subscribeTo(self.thrusterConfigMsg)
        self.thrForceMappingConfig.vehConfigInMsg.subscribeTo(self.vcConfigMsg)
        self.thrForceMappingConfig.controlAxes_B = self.initial_conditions.get(
            "controlAxes_B"
        )
        self.thrForceMappingConfig.thrForceSign = self.initial_conditions.get(
            "thrForceSign"
        )

        self.thrDumpConfig = thrMomentumDumping.thrMomentumDumpingConfig()
        self.thrDumpWrap = self.setModelDataWrap(self.thrDumpConfig)
        self.thrDumpConfig.deltaHInMsg.subscribeTo(
            self.thrDesatControlConfig.deltaHOutMsg
        )
        self.thrDumpConfig.thrusterImpulseInMsg.subscribeTo(
            self.thrForceMappingConfig.thrForceCmdOutMsg
        )
        self.thrusterSet.cmdsInMsg.subscribeTo(self.thrDumpConfig.thrusterOnTimeOutMsg)
        self.thrDumpConfig.thrusterConfInMsg.subscribeTo(self.thrusterConfigMsg)
        self.thrDumpConfig.maxCounterValue = self.initial_conditions.get(
            "maxCounterValue"
        )
        self.thrDumpConfig.thrMinFireTime = self.initial_conditions.get(
            "thrMinFireTime"
        )

        #   Add models to tasks
        self.AddModelToTask(
            "sunPointTask", self.sunPointWrap, self.sunPointData, ModelPriority=1201
        )

        self.AddModelToTask(
            "nadirPointTask", self.hillPointWrap, self.hillPointData, ModelPriority=1200
        )

        self.AddModelToTask(
            "mrpControlTask",
            self.mrpFeedbackControlWrap,
            self.mrpFeedbackControlData,
            ModelPriority=1198,
        )
        self.AddModelToTask(
            "mrpControlTask",
            self.trackingErrorWrap,
            self.trackingErrorData,
            ModelPriority=1199,
        )
        self.AddModelToTask(
            "mrpControlTask",
            self.rwMotorTorqueWrap,
            self.rwMotorTorqueConfig,
            ModelPriority=1196,
        )

        self.AddModelToTask(
            "rwDesatTask",
            self.thrDesatControlWrap,
            self.thrDesatControlConfig,
            ModelPriority=1195,
        )
        self.AddModelToTask(
            "rwDesatTask",
            self.thrForceMappingWrap,
            self.thrForceMappingConfig,
            ModelPriority=1194,
        )
        self.AddModelToTask(
            "rwDesatTask", self.thrDumpWrap, self.thrDumpConfig, ModelPriority=1193
        )

    def setup_viz(self):
        """
        Initializes a vizSupport instance and logs all RW/thruster/spacecraft
        state messages.
        """
        from datetime import datetime

        fileName = f"leo_nadir_env-v1_{datetime.today()}"

        self.vizInterface = vizSupport.enableUnityVisualization(
            self,
            self.dynTaskName,
            self.scObject,
            rwEffectorList=self.rwStateEffector,
            thrEffectorList=self.thrusterSet,
            saveFile=fileName,
        )
        vizSupport.addLocation(
            self.vizInterface,
            stationName="Boulder Station",
            parentBodyName=self.planet.planetName,
            r_GP_P=self.boulderGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="pink",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Merritt Station",
            parentBodyName=self.planet.planetName,
            r_GP_P=self.merrittGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="green",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Singapore Station",
            parentBodyName=self.planet.planetName,
            r_GP_P=self.singaporeGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="blue",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Weilheim Station",
            parentBodyName=self.planet.planetName,
            r_GP_P=self.weilheimGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="purple",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Santiago Station",
            parentBodyName=self.planet.planetName,
            r_GP_P=self.santiagoGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="yellow",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Dongara Station",
            parentBodyName=self.planet.planetName,
            r_GP_P=self.dongaraGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="orange",
            range=1000.0 * 1000,  # meters
        )

        vizSupport.addLocation(
            self.vizInterface,
            stationName="Hawaii Station",
            parentBodyName=self.planet.planetName,
            r_GP_P=self.hawaiiGroundStation.r_LP_P_Init,
            fieldOfView=np.radians(160.0),
            color="grey",
            range=1000.0 * 1000,  # meters
        )

        self.vizInterface.settings.spacecraftSizeMultiplier = 1.5
        self.vizInterface.settings.showLocationCommLines = 1
        self.vizInterface.settings.showLocationCones = 1
        self.vizInterface.settings.showLocationLabels = 1

    def setupGatewayMsgs(self):
        """create C-wrapped gateway messages such that different modules can write to
        this message and provide a common input msg for down-stream modules"""
        self.attRefMsg = cMsgPy.AttRefMsg_C()

        self.zeroGateWayMsgs()

    def zeroGateWayMsgs(self):
        """Zero all the FSW gateway message payloads"""
        self.attRefMsg.write(messaging.AttRefMsgPayload())

    def set_logging(self):
        """
        Logs simulation outputs to return as observations. This simulator observes:
        mrp_bn - inertial to body MRP
        error_mrp - Attitude error given current guidance objective
        power_level - current W-Hr from the battery
        r_bn - inertial position of the s/c relative to Earth
        :return:
        """

        self.boulderGSLog = self.boulderGroundStation.accessOutMsgs[-1].recorder()
        self.singaporeGSLog = self.singaporeGroundStation.accessOutMsgs[-1].recorder()
        self.merrittGSLog = self.merrittGroundStation.accessOutMsgs[-1].recorder()
        self.weilheimGSLog = self.weilheimGroundStation.accessOutMsgs[-1].recorder()
        self.santiagoGSLog = self.santiagoGroundStation.accessOutMsgs[-1].recorder()
        self.dongaraGSLog = self.dongaraGroundStation.accessOutMsgs[-1].recorder()
        self.hawaiiGSLog = self.hawaiiGroundStation.accessOutMsgs[-1].recorder()
        self.transmitterLog = self.transmitter.nodeDataOutMsg.recorder()
        self.scLog = self.scObject.scStateOutMsg.recorder()
        self.batteryLog = self.powerMonitor.batPowerOutMsg.recorder()
        self.storageUnitLog = self.storageUnit.storageUnitDataOutMsg.recorder()
        self.attGuidLog = self.trackingErrorData.attGuidOutMsg.recorder()
        self.rwSpeedLog = self.rwStateEffector.rwSpeedOutMsg.recorder()

        self.AddModelToTask(self.envTaskName, self.boulderGSLog, ModelPriority=599)
        self.AddModelToTask(self.envTaskName, self.singaporeGSLog, ModelPriority=598)
        self.AddModelToTask(self.envTaskName, self.merrittGSLog, ModelPriority=597)
        self.AddModelToTask(self.envTaskName, self.weilheimGSLog, ModelPriority=596)
        self.AddModelToTask(self.envTaskName, self.santiagoGSLog, ModelPriority=595)
        self.AddModelToTask(self.envTaskName, self.dongaraGSLog, ModelPriority=594)
        self.AddModelToTask(self.envTaskName, self.hawaiiGSLog, ModelPriority=593)
        self.AddModelToTask(self.envTaskName, self.transmitterLog, ModelPriority=592)
        self.AddModelToTask(self.envTaskName, self.scLog, ModelPriority=591)
        self.AddModelToTask(self.envTaskName, self.batteryLog, ModelPriority=590)
        self.AddModelToTask(self.envTaskName, self.storageUnitLog, ModelPriority=590)
        self.AddModelToTask(self.envTaskName, self.attGuidLog, ModelPriority=589)
        self.AddModelToTask(self.envTaskName, self.rwSpeedLog, ModelPriority=588)

        return

    def run_sim(self, action, return_obs=True):
        """
        Executes the sim for a specified duration given a mode command.
        :param action:
        :param duration:
        :return:
        """

        # Turn mode request into a string
        self.modeRequest = str(action)

        # Set the sim_over param to false
        self.sim_over = False

        currentResetTime = mc.sec2nano(self.simTime)

        self.zeroGateWayMsgs()

        # Imaging mode
        if self.modeRequest == "0":
            self.dynProc.enableAllTasks()
            self.hillPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)
            # Disable sun pointing and desat
            self.disableTask("sunPointTask")
            self.disableTask("rwDesatTask")
            # Turn off transmitter
            self.transmitter.dataStatus = 0
            self.transmitterPowerSink.powerStatus = 0
            # Turn on instrument
            self.instrument.dataStatus = 1
            self.instrumentPowerSink.powerStatus = 1
            # Turn on nadir pointing and MRP control
            self.enableTask("nadirPointTask")
            self.enableTask("mrpControlTask")

        # Battery charging mode
        elif self.modeRequest == "1":
            self.dynProc.enableAllTasks()
            self.sunPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)
            # Turn off nadir pointing and desat
            self.disableTask("nadirPointTask")
            self.disableTask("rwDesatTask")
            # Turn off transmitter
            self.transmitter.dataStatus = 0
            self.transmitterPowerSink.powerStatus = 0
            # Turn off instrument
            self.instrument.dataStatus = 0
            self.instrumentPowerSink.powerStatus = 0
            # Turn on sun pointing and MRP control
            self.enableTask("sunPointTask")
            self.enableTask("mrpControlTask")

        # Desaturation mode
        elif self.modeRequest == "2":
            self.dynProc.enableAllTasks()
            self.sunPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)
            # Reset
            self.thrDesatControlWrap.Reset(currentResetTime)
            self.thrDumpWrap.Reset(currentResetTime)
            # Turn off nadir pointing and sun pointing
            self.disableTask("nadirPointTask")
            self.disableTask("sunPointTask")
            # Turn off transmitter
            self.transmitter.dataStatus = 0
            self.transmitterPowerSink.powerStatus = 0
            # Turn off instrument
            self.instrument.dataStatus = 0
            self.instrumentPowerSink.powerStatus = 0
            # Turn on sunPoint, MRP control, and desat
            self.enableTask("sunPointTask")
            self.enableTask("mrpControlTask")
            self.enableTask("rwDesatTask")

        # Downlink mode
        elif self.modeRequest == "3":
            self.dynProc.enableAllTasks()
            self.hillPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)
            # Disable sunPoint and Desat tasks
            self.disableTask("sunPointTask")
            self.disableTask("rwDesatTask")
            # Turn off instrument
            self.instrument.dataStatus = 0
            self.instrumentPowerSink.powerStatus = 0
            # Turn on transmitter
            self.transmitter.dataStatus = 1
            self.transmitterPowerSink.powerStatus = 1
            # Turn on nadir pointing and MRP control
            self.enableTask("nadirPointTask")
            self.enableTask("mrpControlTask")

        # Increment time and the current step
        self.simTime += self.step_duration
        simulation_time = mc.sec2nano(self.simTime)
        self.curr_step += 1

        #   Execute the sim
        self.ConfigureStopTime(simulation_time)
        self.ExecuteSimulation()

        # if return_obs:
        # Compute the relevant state variables
        attErr = self.trackingErrorData.attGuidOutMsg.read().sigma_BR
        attRate = self.simpleNavObject.attOutMsg.read().omega_BN_B
        storedCharge = self.powerMonitor.batPowerOutMsg.read().storageLevel
        storedData = self.storageUnit.storageUnitDataOutMsg.read().storageLevel
        eclipseIndicator = self.eclipseObject.eclipseOutMsgs[0].read().shadowFactor
        wheelSpeeds = self.rwStateEffector.rwSpeedOutMsg.read().wheelSpeeds

        # Get the rotation matrix from the inertial to planet frame from SPICE
        dcm_PN = np.array(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.earth]
            .read()
            .J20002Pfix
        ).reshape((3, 3))

        # Get inertial position and velocity, rotate to planet-fixed frame
        inertialPos = self.scObject.scStateOutMsg.read().r_BN_N
        inertialVel = self.scObject.scStateOutMsg.read().v_BN_N
        planetFixedPos = np.matmul(dcm_PN, inertialPos)
        planetFixedVel = np.matmul(dcm_PN, inertialVel)

        accessIndicator1 = self.boulderGSLog.hasAccess[-int(self.step_duration) :]
        accessIndicator2 = self.merrittGSLog.hasAccess[-int(self.step_duration) :]
        accessIndicator3 = self.singaporeGSLog.hasAccess[-int(self.step_duration) :]
        accessIndicator4 = self.weilheimGSLog.hasAccess[-int(self.step_duration) :]
        accessIndicator5 = self.santiagoGSLog.hasAccess[-int(self.step_duration) :]
        accessIndicator6 = self.dongaraGSLog.hasAccess[-int(self.step_duration) :]
        accessIndicator7 = self.hawaiiGSLog.hasAccess[-int(self.step_duration) :]

        transmitterBaud = self.transmitterLog.baudRate[-int(self.step_duration) :]

        # Full observations (non-normalized), all rounded to five decimals
        obs_full = np.hstack(
            np.around(
                [
                    planetFixedPos[0],
                    planetFixedPos[1],
                    planetFixedPos[2],
                    planetFixedVel[0],
                    planetFixedVel[1],
                    planetFixedVel[2],
                    np.linalg.norm(attErr),
                    np.linalg.norm(attRate),
                    wheelSpeeds[1],
                    wheelSpeeds[2],
                    wheelSpeeds[3],
                    storedCharge / 3600.0,
                    eclipseIndicator,
                    storedData,
                    np.sum(transmitterBaud) * self.dynRate / 8e6,
                    np.sum(accessIndicator1) * self.dynRate,
                    np.sum(accessIndicator2) * self.dynRate,
                    np.sum(accessIndicator3) * self.dynRate,
                    np.sum(accessIndicator4) * self.dynRate,
                    np.sum(accessIndicator5) * self.dynRate,
                    np.sum(accessIndicator6) * self.dynRate,
                    np.sum(accessIndicator7) * self.dynRate,
                    self.simTime / 60,
                ],
                decimals=5,
            )
        )

        # Normalized observations, pull things from dictionary for readability
        transmitterBaudRate = self.initial_conditions.get("transmitterBaudRate")
        batteryStorageCapacity = self.initial_conditions.get("batteryStorageCapacity")
        dataStorageCapacity = self.initial_conditions.get("dataStorageCapacity")

        # Normalized observations, all rounded to five decimals
        obs_norm = np.hstack(
            np.around(
                [
                    planetFixedPos[0] / np.linalg.norm(planetFixedPos[0:3]),
                    planetFixedPos[1] / np.linalg.norm(planetFixedPos[0:3]),
                    planetFixedPos[2] / np.linalg.norm(planetFixedPos[0:3]),
                    planetFixedVel[0] / np.linalg.norm(planetFixedVel[0:3]),
                    planetFixedVel[1] / np.linalg.norm(planetFixedVel[0:3]),
                    planetFixedVel[2] / np.linalg.norm(planetFixedVel[0:3]),
                    np.linalg.norm(attErr),
                    np.linalg.norm(attRate),
                    wheelSpeeds[1] / (6000 * mc.RPM),
                    wheelSpeeds[2] / (6000 * mc.RPM),
                    wheelSpeeds[3] / (6000 * mc.RPM),
                    storedCharge / batteryStorageCapacity,
                    eclipseIndicator,
                    storedData / dataStorageCapacity,
                    np.sum(transmitterBaud)
                    * self.dynRate
                    / (transmitterBaudRate * self.step_duration),
                    np.sum(accessIndicator1) * self.dynRate / self.step_duration,
                    np.sum(accessIndicator2) * self.dynRate / self.step_duration,
                    np.sum(accessIndicator3) * self.dynRate / self.step_duration,
                    np.sum(accessIndicator4) * self.dynRate / self.step_duration,
                    np.sum(accessIndicator5) * self.dynRate / self.step_duration,
                    np.sum(accessIndicator6) * self.dynRate / self.step_duration,
                    np.sum(accessIndicator7) * self.dynRate / self.step_duration,
                    self.simTime / (self.max_length * 60),
                ],
                decimals=5,
            )
        )

        # Reshape and assign the observations
        self.obs_full = obs_full.reshape(len(obs_full), 1)
        self.obs = obs_norm.reshape(len(obs_norm), 1)

        # Check if crashed into Earth
        if np.linalg.norm(inertialPos) < (orbitalMotion.REQ_EARTH / 1000.0):
            self.sim_over = True

        # Update performance metrics
        self.downlinked += -obs_full[14]
        self.total_access += max(obs_full[15:22])
        self.utilized_access += (
            np.sum(transmitterBaud)
            * self.dynRate
            / self.initial_conditions.get("transmitterBaudRate")
        )

        if self.clear_logs:
            self.boulderGSLog.clear()
            self.merrittGSLog.clear()
            self.singaporeGSLog.clear()
            self.weilheimGSLog.clear()
            self.santiagoGSLog.clear()
            self.dongaraGSLog.clear()
            self.hawaiiGSLog.clear()
            self.transmitterLog.clear()
            self.scLog.clear()
            self.batteryLog.clear()
            self.storageUnitLog.clear()
            self.attGuidLog.clear()
            self.rwSpeedLog.clear()

        # return self.obs, self.sim_states, self.sim_over
        return self.obs, self.sim_over, self.obs_full

    def close_gracefully(self):
        """
        makes sure spice gets shut down right when we close.
        :return:
        """
        self.gravFactory.unloadSpiceKernels()

        return
