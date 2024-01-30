import Basilisk.architecture.cMsgCInterfacePy as cMsgPy
import numpy as np
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (
    attTrackingError,
    locationPointing,
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
    extForceTorque,
    groundLocation,
    simpleBattery,
    simpleNav,
    simplePowerSink,
    simpleSolarPanel,
    spacecraft,
)
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import simIncludeGravBody, unitTestSupport, vizSupport

from bsk_rl.utilities.effector_primitives import actuator_primitives as ap

bskPath = __path__[0]


class MultiSensorEOSSimulator(SimulationBaseClass.SimBaseClass):
    """
    Simulates ground observations by a single spacecraft in LEO.

    Dynamics Components
    - Forces: J2, Atmospheric Drag w/ COM offset
    - Environment: Exponential density model; eclipse
    - Actuators: ExternalForceTorque
    - Sensors: SimpleNav
    - Systems: SimpleBattery, SimpleSink, SimpleSolarPanel

    FSW Components:
    - mrpFeedback controller
    - inertial3d (sun pointing), hillPoint (nadir pointing)

    :return:
    """

    def __init__(
        self,
        dyn_step,
        fsw_step,
        mode_duration,
        max_simtime,
        init_epoch="2021 MAY 04 06:47:48.965 (UTC)",
        initial_conditions={},
        render=False,
        settings=None,
    ):
        """
        Creates the simulation, but does not initialize the initial conditions.
        """
        self.dyn_step = dyn_step
        self.fsw_step = fsw_step
        self.mode_duration = mode_duration
        self.max_simtime = max_simtime

        super().__init__()

        # define class variables that are assigned later on
        self.attRefMsg = None
        self.has_access_rec = None
        self.img_modes = settings.img_modes

        self.sim_time = 0.0
        self.sim_over = False

        # If no initial conditions are defined yet, set ICs
        if initial_conditions:
            self.initial_conditions = initial_conditions
        else:
            self.initial_conditions = settings.INITIAL_CONDITIONS

        #   Specify some simulation parameters
        self.init_epoch = self.initial_conditions["utc_init"]
        self.mass = self.initial_conditions.get("mass")  # kg
        self.powerDraw = self.initial_conditions.get("powerDraw")  # W
        self.imaged_targets = np.zeros(self.initial_conditions["n_targets"])
        self.upcoming_tgts = np.arange(0, self.initial_conditions["n_targets"])

        self.dyn_models = []
        self.fsw_models = []

        #   Initialize the dynamics+fsw task groups, modules

        self.dyn_proc_name = "DynamicsProcess"  # Create simulation process name
        self.dyn_proc = self.CreateNewProcess(self.dyn_proc_name)  # Create process
        self.dyn_task_name = "dyn_task"
        self.env_task_name = "env_task"
        self.dyn_task = self.dyn_proc.addTask(
            self.CreateNewTask(self.dyn_task_name, mc.sec2nano(self.dyn_step)),
            taskPriority=200,
        )
        self.env_task = self.dyn_proc.addTask(
            self.CreateNewTask(self.env_task_name, mc.sec2nano(self.dyn_step)),
            taskPriority=199,
        )

        self.set_env_dynamics()

        self.set_sc_dynamics()
        self.setup_gateway_msgs()
        if render:
            self.setup_viz()
        self.set_fsw()

        self.set_logging()
        self.Init_sc_mode = 2

        self.mode_request = None
        self.img_mode = None
        self.data_buffer = None
        self.InitializeSimulation()

        self._sim_state = self.get_sim_state(init=True)

        return

    def set_env_dynamics(self):
        """
        Sets up environmental dynamics for the sim, including:
        - SPICE
        - Eclipse
        - Planetary atmosphere
        - Gravity
        - Spherical harmonics
        """
        # clear prior gravitational body and SPICE setup definitions
        self.gravFactory = simIncludeGravBody.gravBodyFactory()

        self.sun = 0
        self.central_body = 1
        body_name = self.initial_conditions.get("central_body")
        gravBodies = self.gravFactory.createBodies(["sun", body_name])
        gravBodies[body_name].isCentralBody = (
            True  # ensure this is the central gravitational body
        )

        self.mu = gravBodies[body_name].mu
        self.radEquator = gravBodies[body_name].radEquator

        # setup Spice interface for some solar system bodies
        timeInitString = self.init_epoch
        self.gravFactory.createSpiceInterface(
            bskPath + "/supportData/EphemerisData/", timeInitString
        )

        self.gravFactory.spiceObject.zeroBase = (
            body_name  # Make sure that the central body is the zero base
        )

        self.ephemConverter = ephemerisConverter.EphemerisConverter()
        self.ephemConverter.ModelTag = "ephemConverter"
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun]
        )
        self.ephemConverter.addSpiceInputMsg(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.central_body]
        )

        # Create the ground location
        self.groundLocation = groundLocation.GroundLocation()
        self.groundLocation.ModelTag = "groundTarget"
        self.groundLocation.planetRadius = self.radEquator
        self.groundLocation.specifyLocationPCPF(
            unitTestSupport.np2EigenVectorXd(
                self.initial_conditions["targetLocation"][:, 0]
            )
        )
        self.groundLocation.planetInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.central_body]
        )
        self.groundLocation.minimumElevation = np.radians(
            self.initial_conditions.get("minElev")
        )
        self.groundLocation.maximumRange = self.initial_conditions.get("maxRange")
        self.AddModelToTask(self.dyn_task_name, self.groundLocation)
        self.AddModelToTask(self.dyn_task_name, self.gravFactory.spiceObject)
        self.AddModelToTask(self.dyn_task_name, self.ephemConverter)

    def set_sc_dynamics(self):
        """
        Sets up the dynamics modules for the sim. This simulator runs:
        scObject (spacecraft dynamics simulation)
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

        # attach gravity model to spaceCraftPlus
        self.scObject.gravField.gravBodies = spacecraft.GravBodyVector(
            list(self.gravFactory.gravBodies.values())
        )

        #   Make sure cross-coupling is done
        self.groundLocation.addSpacecraftToModel(self.scObject.scStateOutMsg)

        rN = self.initial_conditions.get("rN")
        vN = self.initial_conditions.get("vN")

        width = self.initial_conditions.get("width")
        depth = self.initial_conditions.get("depth")
        height = self.initial_conditions.get("height")

        self.Inertia_mat = [
            1.0 / 12.0 * self.mass * (width**2.0 + depth**2.0),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * self.mass * (depth**2.0 + height**2.0),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * self.mass * (width**2.0 + height**2.0),
        ]

        self.scObject.hub.mHub = self.mass  # kg
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(
            self.Inertia_mat
        )

        self.scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(rN)
        self.scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(vN)

        sigma_init = self.initial_conditions.get("sigma_init")
        omega_init = self.initial_conditions.get("omega_init")

        self.scObject.hub.sigma_BNInit = sigma_init  # sigma_BN_B
        self.scObject.hub.omega_BN_BInit = omega_init

        self.eclipseObject = eclipse.Eclipse()
        self.eclipseObject.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.eclipseObject.addPlanetToModel(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.central_body]
        )
        self.eclipseObject.sunInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.sun]
        )

        #   Disturbance Torque Setup
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
            useRandom=True, randomBounds=(-800, 800)
        )
        # Change the wheel speeds
        rwFactory.rwList["RW1"].Omega = self.initial_conditions.get("wheelSpeeds")[
            0
        ]  # rad/s
        rwFactory.rwList["RW2"].Omega = self.initial_conditions.get("wheelSpeeds")[
            1
        ]  # rad/s
        rwFactory.rwList["RW3"].Omega = self.initial_conditions.get("wheelSpeeds")[
            2
        ]  # rad/s
        rwFactory.addToSpacecraft("ReactionWheels", self.rwStateEffector, self.scObject)
        self.rwConfigMsg = rwFactory.getConfigMessage()

        #   Add thrusters to the spacecraft
        self.thrusterSet, thrFactory = ap.idealMonarc1Octet()
        thrModelTag = "ACSThrusterDynamics"
        self.thrusterConfigMsg = thrFactory.getConfigMessage()
        thrFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)

        #   Add simpleNav as a mock estimator to the spacecraft
        self.simpleNavObject = simpleNav.SimpleNav()
        self.simpleNavObject.ModelTag = "SimpleNav"
        self.simpleNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)

        #   Power Setup
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

        self.nominalPowerSink = simplePowerSink.SimplePowerSink()
        self.nominalPowerSink.ModelTag = "nominalPowerSink" + str(sc_number)
        self.nominalPowerSink.nodePowerOut = self.powerDraw  # Watts

        self.sensorPowerSink = simplePowerSink.SimplePowerSink()
        self.sensorPowerSink.ModelTag = "sensorPowerSink" + str(sc_number)
        self.sensorPowerSink.nodePowerOut = self.powerDraw  # Watts
        self.sensorPowerSink.powerStatus = 0

        self.rwPowerList = []
        for ind in range(rwFactory.getNumOfDevices()):
            rwPower = ReactionWheelPower.ReactionWheelPower()
            rwPower.ModelTag = f"rwPower_{ind}"
            rwPower.basePowerNeed = 1.0
            rwPower.elecToMechEfficiency = 0.9
            rwPower.mechToElecEfficiency = 0.5
            rwPower.rwStateInMsg.subscribeTo(self.rwStateEffector.rwOutMsgs[ind])
            rwPower.nodePowerOut = self.powerDraw  # Watts
            self.rwPowerList.append(rwPower)

        self.powerMonitor = simpleBattery.SimpleBattery()
        self.powerMonitor.ModelTag = "powerMonitor"
        self.powerMonitor.storageCapacity = self.initial_conditions.get(
            "storageCapacity"
        )
        self.powerMonitor.storedCharge_Init = self.initial_conditions.get(
            "storedCharge_Init"
        )
        self.powerMonitor.addPowerNodeToModel(self.solarPanel.nodePowerOutMsg)
        self.powerMonitor.addPowerNodeToModel(self.nominalPowerSink.nodePowerOutMsg)
        self.powerMonitor.addPowerNodeToModel(self.sensorPowerSink.nodePowerOutMsg)
        for rwPower in self.rwPowerList:
            self.powerMonitor.addPowerNodeToModel(rwPower.nodePowerOutMsg)

        #   Add all the models to the dynamics task
        self.AddModelToTask(self.dyn_task_name, self.scObject, ModelPriority=200)
        self.AddModelToTask(self.dyn_task_name, self.simpleNavObject, ModelPriority=199)
        self.AddModelToTask(self.dyn_task_name, self.rwStateEffector, ModelPriority=198)
        self.AddModelToTask(self.dyn_task_name, self.thrusterSet, ModelPriority=197)
        self.AddModelToTask(self.env_task_name, self.eclipseObject, ModelPriority=196)
        self.AddModelToTask(self.env_task_name, self.solarPanel, ModelPriority=195)
        self.AddModelToTask(self.env_task_name, self.powerMonitor, ModelPriority=194)
        self.AddModelToTask(
            self.env_task_name, self.nominalPowerSink, ModelPriority=193
        )
        self.AddModelToTask(self.env_task_name, self.sensorPowerSink, ModelPriority=192)
        for rwPower in self.rwPowerList:
            self.AddModelToTask(self.env_task_name, rwPower)

        return

    def setup_viz(self):
        """
        Initializes a vizSupport instance and logs all RW/thruster/spacecraft state
        messages.
        """
        from datetime import datetime

        fileName = f"earth_obs_env-v3_{datetime.today()}"

        self.vizInterface = vizSupport.enableUnityVisualization(
            self,
            self.dyn_task_name,
            self.scObject,
            rwEffectorList=self.rwStateEffector,
            thrEffectorList=self.thrusterSet,
            saveFile=fileName,
        )
        vizSupport.addLocation(
            self.vizInterface,
            stationName=self.initial_conditions.get("target_name"),
            parentBodyName=self.initial_conditions.get("central_body") + "_planet_data",
            r_GP_P=self.groundLocation.r_LP_P_Init,
            fieldOfView=2 * (np.pi / 2 - self.groundLocation.minimumElevation),
            color="pink",
            range=self.groundLocation.maximumRange,  # meters
        )
        self.vizInterface.settings.spacecraftSizeMultiplier = 1.5
        self.vizInterface.settings.showLocationCommLines = 1
        self.vizInterface.settings.showLocationCones = 1
        self.vizInterface.settings.showLocationLabels = 1

    def set_fsw(self):
        """
        Sets up the attitude guidance stack for the simulation. This simulator runs:
        inertial3Dpoint - Sets the attitude guidance objective to point the main panel
            at the sun.
        hillPointTask: Sets the attitude guidance objective to point a "camera"
            boresight towards nadir.
        attitudeTrackingError: Computes the difference between estimated and guidance
            attitudes
        mrpFeedbackControl: Computes an appropriate control torque given an attitude
            error
        :return:
        """
        self.dyn_proc.addTask(
            self.CreateNewTask("sunPointTask", mc.sec2nano(self.fsw_step)),
            taskPriority=100,
        )
        self.dyn_proc.addTask(
            self.CreateNewTask("nonSunPointTask", mc.sec2nano(self.fsw_step)),
            taskPriority=100,
        )
        self.dyn_proc.addTask(
            self.CreateNewTask("nadirPointTask", mc.sec2nano(self.fsw_step)),
            taskPriority=100,
        )
        self.dyn_proc.addTask(
            self.CreateNewTask("mrpControlTask", mc.sec2nano(self.fsw_step)),
            taskPriority=50,
        )
        self.dyn_proc.addTask(
            self.CreateNewTask("rwDesatTask", mc.sec2nano(self.fsw_step)),
            taskPriority=25,
        )

        # Specify the vehicle configuration message to tell things what the vehicle
        # inertia is
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        # use the same inertia in the FSW algorithm as in the simulation
        #   Set inertia properties to those of a solid 6U cubeoid:
        vehicleConfigOut.ISCPntB_B = self.Inertia_mat
        # adcs_config_data -> vcConfigMsg
        self.vcConfigMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

        #   Sun pointing configuration
        self.sunPointData = locationPointing.locationPointingConfig()
        self.sunPointWrap = self.setModelDataWrap(self.sunPointData)
        self.sunPointWrap.ModelTag = "sunPoint"
        cMsgPy.AttRefMsg_C_addAuthor(self.sunPointData.attRefOutMsg, self.attRefMsg)
        self.sunPointData.pHat_B = unitTestSupport.EigenVector3d2np(
            self.solarPanel.nHat_B
        )
        self.sunPointData.scTransInMsg.subscribeTo(self.simpleNavObject.transOutMsg)
        self.sunPointData.scAttInMsg.subscribeTo(self.simpleNavObject.attOutMsg)
        self.sunPointData.celBodyInMsg.subscribeTo(
            self.ephemConverter.ephemOutMsgs[self.sun]
        )
        self.sunPointData.useBoresightRateDamping = 1

        #   Non-Sun pointing configuration
        self.nonSunPointData = locationPointing.locationPointingConfig()
        self.nonSunPointWrap = self.setModelDataWrap(self.nonSunPointData)
        self.nonSunPointWrap.ModelTag = "nonSunPoint"
        cMsgPy.AttRefMsg_C_addAuthor(self.nonSunPointData.attRefOutMsg, self.attRefMsg)
        self.nonSunPointData.pHat_B = -unitTestSupport.EigenVector3d2np(
            self.solarPanel.nHat_B
        )
        self.nonSunPointData.scTransInMsg.subscribeTo(self.simpleNavObject.transOutMsg)
        self.nonSunPointData.scAttInMsg.subscribeTo(self.simpleNavObject.attOutMsg)
        self.nonSunPointData.celBodyInMsg.subscribeTo(
            self.ephemConverter.ephemOutMsgs[self.sun]
        )
        self.nonSunPointData.useBoresightRateDamping = 1

        #   Nadir pointing configuration
        self.nadirPointData = locationPointing.locationPointingConfig()
        self.nadirPointWrap = self.setModelDataWrap(self.nadirPointData)
        self.nadirPointWrap.ModelTag = "nadirPoint"
        cMsgPy.AttRefMsg_C_addAuthor(self.nadirPointData.attRefOutMsg, self.attRefMsg)
        self.nadirPointData.pHat_B = unitTestSupport.EigenVector3d2np(
            self.solarPanel.nHat_B
        )
        self.nadirPointData.scTransInMsg.subscribeTo(self.simpleNavObject.transOutMsg)
        self.nadirPointData.scAttInMsg.subscribeTo(self.simpleNavObject.attOutMsg)
        self.nadirPointData.celBodyInMsg.subscribeTo(
            self.ephemConverter.ephemOutMsgs[self.central_body]
        )
        self.nadirPointData.useBoresightRateDamping = 1

        #   Attitude error configuration
        self.trackingErrorData = attTrackingError.attTrackingErrorConfig()
        self.trackingErrorWrap = self.setModelDataWrap(self.trackingErrorData)
        self.trackingErrorWrap.ModelTag = "trackingError"
        self.trackingErrorData.attNavInMsg.subscribeTo(self.simpleNavObject.attOutMsg)
        self.trackingErrorData.attRefInMsg.subscribeTo(self.attRefMsg)
        self.trackingErrorData.sigma_R0R = rbk.C2MRP(
            self.initial_conditions.get("C_R0R")
        )

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
        self.thrForceMappingConfig.angErrThresh = 3.15

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
            "sunPointTask", self.sunPointWrap, self.sunPointData, ModelPriority=10
        )
        self.AddModelToTask(
            "nonSunPointTask",
            self.nonSunPointWrap,
            self.nonSunPointData,
            ModelPriority=10,
        )
        self.AddModelToTask(
            "nadirPointTask", self.nadirPointWrap, self.nadirPointData, ModelPriority=10
        )

        self.AddModelToTask(
            "mrpControlTask",
            self.mrpFeedbackControlWrap,
            self.mrpFeedbackControlData,
            ModelPriority=3,
        )
        self.AddModelToTask(
            "mrpControlTask",
            self.trackingErrorWrap,
            self.trackingErrorData,
            ModelPriority=9,
        )
        self.AddModelToTask(
            "mrpControlTask",
            self.rwMotorTorqueWrap,
            self.rwMotorTorqueConfig,
            ModelPriority=1,
        )

        self.AddModelToTask(
            "rwDesatTask",
            self.thrDesatControlWrap,
            self.thrDesatControlConfig,
            ModelPriority=3,
        )
        self.AddModelToTask(
            "rwDesatTask",
            self.thrForceMappingWrap,
            self.thrForceMappingConfig,
            ModelPriority=2,
        )
        self.AddModelToTask(
            "rwDesatTask", self.thrDumpWrap, self.thrDumpConfig, ModelPriority=1
        )

    def set_logging(self):
        """
        Logs simulation outputs to return as observations. This simulator observes:
        mrp_bn - inertial to body MRP
        error_mrp - Attitude error given current guidance objective
        power_level - current W-Hr from the battery
        r_bn - inertial position of the s/c relative to Earth
        :return:
        """
        # Set the sampling time to the duration of a timestep:
        samplingTime = mc.sec2nano(self.mode_duration)

        self.has_access_rec = self.groundLocation.accessOutMsgs[-1].recorder()
        self.onTimeLog = self.thrDumpConfig.thrusterOnTimeOutMsg.recorder()
        self.thrMapLog = self.thrForceMappingConfig.thrForceCmdOutMsg.recorder(
            samplingTime
        )

        self.AddModelToTask("mrpControlTask", self.thrMapLog)
        self.AddModelToTask("mrpControlTask", self.has_access_rec)
        self.AddModelToTask("mrpControlTask", self.onTimeLog)
        return

    def setup_gateway_msgs(self):
        """create C-wrapped gateway messages such that different modules can write to
        this message
        and provide a common input msg for down-stream modules"""
        self.attRefMsg = cMsgPy.AttRefMsg_C()
        self.zeroGateWayMsgs()

    def zeroGateWayMsgs(self):
        """Zero all the FSW gateway message payloads"""
        self.attRefMsg.write(messaging.AttRefMsgPayload())

    def run_sim(self, action):
        """
        Executes the sim for a specified duration given a mode command.
        :param action:
            0 - Point solar panels at the sun
            1 - Desaturate reaction wheels
            >1 - Image types
        :return:
            sim_state - simulation states generated
            sim_over - episode over flag
        """

        self.mode_request = action

        currentResetTime = mc.sec2nano(self.sim_time)
        self.zeroGateWayMsgs()
        if self.mode_request == 0:
            #   Set up a sun pointing mode
            self.dyn_proc.enableAllTasks()
            self.sunPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)

            self.disableTask("nadirPointTask")
            self.disableTask("nonSunPointTask")
            self.disableTask("rwDesatTask")

            self.enableTask("sunPointTask")
            self.enableTask("mrpControlTask")
            self.sensorPowerSink.powerStatus = 0

        elif self.mode_request == 1:
            #   Set up a desat mode
            self.dyn_proc.enableAllTasks()
            self.nonSunPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)

            self.thrDesatControlWrap.Reset(currentResetTime)
            self.thrDumpWrap.Reset(currentResetTime)
            self.thrForceMappingWrap.Reset(currentResetTime)

            self.disableTask("nadirPointTask")
            self.disableTask("sunPointTask")

            self.enableTask("nonSunPointTask")
            self.enableTask("mrpControlTask")
            self.enableTask("rwDesatTask")
            self.sensorPowerSink.powerStatus = 0

        elif (self.mode_request - 2) in range(self.img_modes):
            #   Set up a nadir pointing mode
            self.dyn_proc.enableAllTasks()
            self.nadirPointWrap.Reset(currentResetTime)
            self.trackingErrorWrap.Reset(currentResetTime)

            self.disableTask("sunPointTask")
            self.disableTask("nonSunPointTask")
            self.disableTask("rwDesatTask")

            self.enableTask("nadirPointTask")
            self.enableTask("mrpControlTask")
            self.sensorPowerSink.powerStatus = 1
            self.data_buffer += 1

        else:
            raise invalid_action

        self.sim_time += self.mode_duration
        simulationTime = mc.sec2nano(self.sim_time)

        #   Execute the sim
        self.has_access_rec.clear()  # purge the recorder history
        self.onTimeLog.clear()
        self.thrMapLog.clear()
        self.ConfigureStopTime(simulationTime)
        self.ExecuteSimulation()

        #   Simulation States
        self._sim_state = self.get_sim_state()

        #   Check if sim is over
        sc_pos_inertial = np.array(
            [
                self._sim_state.get("rx_sc_N"),
                self._sim_state.get("ry_sc_N"),
                self._sim_state.get("rz_sc_N"),
            ]
        )
        if np.linalg.norm(sc_pos_inertial) < (self.radEquator):
            self.sim_over = True
            print("Spacecraft crashed into Body.")
        elif self.sim_time >= self.max_simtime:
            self.sim_over = True

        return self._sim_state

    def get_sim_state(self, init=False):
        # Check if the target was imaged
        if any(self.has_access_rec.hasAccess) and not init:
            self.imaged_targets[self.upcoming_tgts[0]] = 1

        # Update list of upcoming targets
        self.upcoming_tgts = self.check_target_switch()

        # Update the groundLocation with the next target
        self.groundLocation.specifyLocationPCPF(
            unitTestSupport.np2EigenVectorXd(
                self.initial_conditions["targetLocation"][:, self.upcoming_tgts[0]]
            )
        )

        self.img_mode = self.initial_conditions["instrumentSpecification"][
            self.upcoming_tgts[0]
        ]

        if init:
            r_target_N = np.array(self.groundLocation.r_LP_P_Init).T[0]
            r_sc_N = self.initial_conditions.get("rN")
            att_err = np.linalg.norm(self.scObject.hub.sigma_BNInit)
            att_rate = np.linalg.norm(self.scObject.hub.omega_BN_BInit)
            wheel_speed = np.linalg.norm(self.initial_conditions.get("wheelSpeeds"))
            stored_charge = self.powerMonitor.storedCharge_Init
            sun_indicator = self.eclipseObject.eclipseOutMsgs[0].read().shadowFactor
            r_target_sc_N = r_sc_N - r_target_N
            access_indicator = int(
                np.arccos(
                    np.dot(r_target_N, r_target_sc_N)
                    / (np.linalg.norm(r_target_N) * np.linalg.norm(r_target_sc_N))
                )
                <= np.deg2rad(90.0 - self.initial_conditions.get("minElev"))
            )
            self.data_buffer = 0
            r_LN_N = [0, 0, 0]
            r_LN_N_norm = 1
        else:
            # Call update state on the ground location class to update the target
            self.groundLocation.UpdateState(mc.sec2nano(self.sim_time))
            r_target_N = self.groundLocation.currentGroundStateOutMsg.read().r_LN_N
            r_sc_N = self.scObject.scStateOutMsg.read().r_BN_N
            att_err = np.linalg.norm(
                self.trackingErrorData.attGuidOutMsg.read().sigma_BR
            )
            att_rate = np.linalg.norm(self.simpleNavObject.attOutMsg.read().omega_BN_B)
            wheel_speed = np.linalg.norm(
                self.rwStateEffector.rwSpeedOutMsg.read().wheelSpeeds[0:3]
            )
            stored_charge = self.powerMonitor.batPowerOutMsg.read().storageLevel
            sun_indicator = self.eclipseObject.eclipseOutMsgs[0].read().shadowFactor
            has_access = self.has_access_rec.hasAccess
            if any(has_access):
                access_indicator = 1
            else:
                access_indicator = 0

            r_LN_N = self.groundLocation.currentGroundStateOutMsg.read().r_LN_N
            r_LN_N_norm = np.linalg.norm(r_LN_N)

        azimuth_angle = self.groundLocation.accessOutMsgs[-1].read().azimuth
        elevation_angle = self.groundLocation.accessOutMsgs[-1].read().elevation
        azimuth_rate = self.groundLocation.accessOutMsgs[-1].read().az_dot
        elevation_rate = self.groundLocation.accessOutMsgs[-1].read().el_dot
        rx_BL_L = self.groundLocation.accessOutMsgs[-1].read().r_BL_L[0]
        ry_BL_L = self.groundLocation.accessOutMsgs[-1].read().r_BL_L[1]
        rz_BL_L = self.groundLocation.accessOutMsgs[-1].read().r_BL_L[2]
        rxy_2norm_BL_L = np.linalg.norm(
            self.groundLocation.accessOutMsgs[-1].read().r_BL_L[0:2]
        )
        xDot_BL_L = self.groundLocation.accessOutMsgs[-1].read().v_BL_L[0]
        yDot_BL_L = self.groundLocation.accessOutMsgs[-1].read().v_BL_L[1]
        zDot_BL_L = self.groundLocation.accessOutMsgs[-1].read().v_BL_L[2]
        v_can = np.sqrt(
            self.mu / self.radEquator
        )  # canonical velocity normalizing constant
        rot_ang_mom = unitTestSupport.EigenVector3d2np(
            self.scObject.totRotAngMomPntC_N
        )  # rotational angular momentum about CoM in N-frame coords
        current_sim_state = {
            "rx_target_N": r_target_N[0],
            "ry_target_N": r_target_N[1],
            "rz_target_N": r_target_N[2],
            "rx_sc_N": r_sc_N[0],
            "ry_sc_N": r_sc_N[1],
            "rz_sc_N": r_sc_N[2],
            "sc_az": azimuth_angle,
            "sc_el": elevation_angle,
            "sc_az_dot": azimuth_rate,
            "sc_el_dot": elevation_rate,
            "rx_BL_L": rx_BL_L,
            "ry_BL_L": ry_BL_L,
            "rz_BL_L": rz_BL_L,
            "rx_canonical_BL_L": rx_BL_L / self.radEquator,
            "ry_canonical_BL_L": ry_BL_L / self.radEquator,
            "rz_canonical_BL_L": rz_BL_L / self.radEquator,
            "rxy_2norm_canonical_BL_L": rxy_2norm_BL_L / self.radEquator,
            "xDot_BL_L": xDot_BL_L,
            "yDot_BL_L": yDot_BL_L,
            "zDot_BL_L": zDot_BL_L,
            "xDot_canonical_BL_L": xDot_BL_L / v_can,
            "yDot_canonical_BL_L": yDot_BL_L / v_can,
            "zDot_canonical_BL_L": zDot_BL_L / v_can,
            "att_err": att_err,
            "att_rate": att_rate,
            "wheel_speed": wheel_speed,
            "stored_charge": stored_charge,
            "sun_indicator": sun_indicator,
            "access_indicator": access_indicator,
            "sc_mode": self.mode_request,
            "rxhat_target_N": r_LN_N[0] / r_LN_N_norm,
            "ryhat_target_N": r_LN_N[1] / r_LN_N_norm,
            "rzhat_target_N": r_LN_N[2] / r_LN_N_norm,
            "rot_ang_mom_x": rot_ang_mom[0],
            "rot_ang_mom_y": rot_ang_mom[1],
            "rot_ang_mom_z": rot_ang_mom[2],
            "data_buffer": self.data_buffer,
            "img_mode": self.img_mode,
            "img_mode_norm": self.img_mode / self.img_modes,
        }

        return current_sim_state

    def check_target_switch(self):
        """
        Grabs the index(s) of the next upcoming target(s)
        """
        times = self.initial_conditions.get("target_times")
        idx = 0
        upcoming_tgts = []
        for idx, time in enumerate(times):
            # If less than simTime, add to upcoming targets
            if self.sim_time < time:
                if not self.imaged_targets[idx]:
                    upcoming_tgts.append(idx)
                # return idx

        # Check that the list has at least as many upcoming targets as n_targets
        # (num in action space)
        # If not, backfill with last target
        if len(upcoming_tgts) < self.initial_conditions["n_targets"]:
            for tgt in range(self.initial_conditions["n_targets"] - len(upcoming_tgts)):
                # Append the last target
                upcoming_tgts.append(idx)

        return upcoming_tgts

    def __del__(self):
        """
        Deletes the simulation instance and calls the spice kernel unloader
        :return:
        """
        self.close_gracefully()
        return

    def close_gracefully(self):
        """
        makes sure spice gets shut down right when we close.
        :return:
        """
        self.gravFactory.unloadSpiceKernels()
        return


class invalid_action(Exception):
    def __str__(self):
        return "Invalid mode selection"
