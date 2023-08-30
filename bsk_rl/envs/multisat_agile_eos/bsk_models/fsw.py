from Basilisk.architecture import cMsgCInterfacePy as cMsgPy
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (
    attTrackingError,
    hillPoint,
    locationPointing,
    mrpFeedback,
    rwMotorTorque,
    simpleInstrumentController,
    thrForceMapping,
    thrMomentumDumping,
    thrMomentumManagement,
)
from Basilisk.utilities import macros as mc


class FSWModel:
    """Defines the FSW class"""

    def __init__(self, SimBase, fswRate, spacecraftIndex):
        # define empty class variables
        self.spacecraftIndex = spacecraftIndex
        self.modeRequest = "0"

        self.vcConfigMsg = None
        self.thrusterConfigMsg = None
        self.fswRwConfigMsg = None
        self.attRefMsg = None
        self.attGuidMsg = None

        # Define process name and default time-step for all FSW tasks defined later on
        self.processName = SimBase.FSWProcessName[spacecraftIndex]
        self.processTasksTimeStep = mc.sec2nano(fswRate)

        # Create tasks
        SimBase.fswProc[spacecraftIndex].addTask(
            SimBase.CreateNewTask(
                "sunPointTask" + str(spacecraftIndex), self.processTasksTimeStep
            ),
            taskPriority=99,
        )
        SimBase.fswProc[spacecraftIndex].addTask(
            SimBase.CreateNewTask(
                "nadirPointTask" + str(spacecraftIndex), self.processTasksTimeStep
            ),
            taskPriority=98,
        )
        SimBase.fswProc[spacecraftIndex].addTask(
            SimBase.CreateNewTask(
                "rwDesatTask" + str(spacecraftIndex), self.processTasksTimeStep
            ),
            taskPriority=1000,
        )
        SimBase.fswProc[spacecraftIndex].addTask(
            SimBase.CreateNewTask(
                "locPointTask" + str(spacecraftIndex), self.processTasksTimeStep
            ),
            taskPriority=96,
        )
        SimBase.fswProc[spacecraftIndex].addTask(
            SimBase.CreateNewTask(
                "trackingErrTask" + str(spacecraftIndex), self.processTasksTimeStep
            ),
            taskPriority=90,
        )
        SimBase.fswProc[spacecraftIndex].addTask(
            SimBase.CreateNewTask(
                "mrpControlTask" + str(spacecraftIndex), self.processTasksTimeStep
            ),
            taskPriority=80,
        )

        # Create module data and module wraps
        # Sun pointing configuration
        self.sunPointData = locationPointing.locationPointingConfig()
        self.sunPointWrap = SimBase.setModelDataWrap(self.sunPointData)
        self.sunPointWrap.ModelTag = "sunPoint"

        # Earth pointing configuration
        self.hillPointData = hillPoint.hillPointConfig()
        self.hillPointWrap = SimBase.setModelDataWrap(self.hillPointData)
        self.hillPointWrap.ModelTag = "hillPoint"

        # Location pointing configuration
        self.locPointConfig = locationPointing.locationPointingConfig()
        self.locPointWrap = SimBase.setModelDataWrap(self.locPointConfig)
        self.locPointWrap.ModelTag = "locPoint"

        # Attitude error configuration
        self.trackingErrorData = attTrackingError.attTrackingErrorConfig()
        self.trackingErrorWrap = SimBase.setModelDataWrap(self.trackingErrorData)
        self.trackingErrorWrap.ModelTag = "trackingError"

        # Attitude controller configuration
        self.mrpFeedbackControlData = mrpFeedback.mrpFeedbackConfig()
        self.mrpFeedbackControlWrap = SimBase.setModelDataWrap(
            self.mrpFeedbackControlData
        )
        self.mrpFeedbackControlWrap.ModelTag = "mrpFeedbackControl"

        # add module that maps the Lr control torque into the RW motor torques
        self.rwMotorTorqueConfig = rwMotorTorque.rwMotorTorqueConfig()
        self.rwMotorTorqueWrap = SimBase.setModelDataWrap(self.rwMotorTorqueConfig)
        self.rwMotorTorqueWrap.ModelTag = "rwMotorTorque"

        #   Momentum dumping configuration
        self.thrDesatControlConfig = thrMomentumManagement.thrMomentumManagementConfig()
        self.thrDesatControlWrap = SimBase.setModelDataWrap(self.thrDesatControlConfig)
        self.thrDesatControlWrap.ModelTag = "thrMomentumManagement"

        self.thrDumpConfig = thrMomentumDumping.thrMomentumDumpingConfig()
        self.thrDumpWrap = SimBase.setModelDataWrap(self.thrDumpConfig)
        self.thrDumpWrap.ModelTag = "thrDump"

        # setup the thruster force mapping module
        self.thrForceMappingConfig = thrForceMapping.thrForceMappingConfig()
        self.thrForceMappingWrap = SimBase.setModelDataWrap(self.thrForceMappingConfig)
        self.thrForceMappingWrap.ModelTag = "thrForceMapping"

        # setup the simpleInstrumentController module
        self.simpleInsControlConfig = (
            simpleInstrumentController.simpleInstrumentControllerConfig()
        )
        self.simpleInsControlWrap = SimBase.setModelDataWrap(
            self.simpleInsControlConfig
        )
        self.simpleInsControlWrap.ModelTag = "instrumentController"

        # create the FSW module gateway messages
        self.setupGatewayMsgs(SimBase)

        # Initialize all modules
        self.InitAllFSWObjects(SimBase)

        # Assign initialized modules to tasks
        SimBase.AddModelToTask(
            "sunPointTask" + str(spacecraftIndex),
            self.sunPointWrap,
            self.sunPointData,
            ModelPriority=1200,
        )
        SimBase.AddModelToTask(
            "nadirPointTask" + str(spacecraftIndex),
            self.hillPointWrap,
            self.hillPointData,
            ModelPriority=1199,
        )
        SimBase.AddModelToTask(
            "locPointTask" + str(spacecraftIndex),
            self.locPointWrap,
            self.locPointConfig,
            ModelPriority=1198,
        )
        SimBase.AddModelToTask(
            "trackingErrTask" + str(spacecraftIndex),
            self.trackingErrorWrap,
            self.trackingErrorData,
            ModelPriority=1197,
        )
        SimBase.AddModelToTask(
            "mrpControlTask" + str(spacecraftIndex),
            self.mrpFeedbackControlWrap,
            self.mrpFeedbackControlData,
            ModelPriority=1196,
        )
        SimBase.AddModelToTask(
            "mrpControlTask" + str(spacecraftIndex),
            self.rwMotorTorqueWrap,
            self.rwMotorTorqueConfig,
            ModelPriority=1195,
        )
        SimBase.AddModelToTask(
            "rwDesatTask" + str(spacecraftIndex),
            self.thrDesatControlWrap,
            self.thrDesatControlConfig,
            ModelPriority=1194,
        )
        SimBase.AddModelToTask(
            "rwDesatTask" + str(spacecraftIndex),
            self.thrForceMappingWrap,
            self.thrForceMappingConfig,
            ModelPriority=1193,
        )
        SimBase.AddModelToTask(
            "rwDesatTask" + str(spacecraftIndex),
            self.thrDumpWrap,
            self.thrDumpConfig,
            ModelPriority=1192,
        )
        SimBase.AddModelToTask(
            "mrpControlTask" + str(spacecraftIndex),
            self.simpleInsControlWrap,
            self.simpleInsControlConfig,
            ModelPriority=987,
        )

        # Create events to be called for triggering GN&C maneuvers
        SimBase.fswProc[spacecraftIndex].disableAllTasks()

    # These are module-initialization methods
    def SetSunPointGuidance(self, SimBase):
        """
        Defines the Sun pointing guidance module.
        """
        self.sunPointData.pHat_B = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ]["nHat_B"]
        self.sunPointData.scAttInMsg.subscribeTo(
            SimBase.DynModels[self.spacecraftIndex].simpleNavObject.attOutMsg
        )
        self.sunPointData.scTransInMsg.subscribeTo(
            SimBase.DynModels[self.spacecraftIndex].simpleNavObject.transOutMsg
        )
        self.sunPointData.celBodyInMsg.subscribeTo(
            SimBase.EnvModel.ephemConverter.ephemOutMsgs[0]
        )
        self.sunPointData.useBoresightRateDamping = 1
        cMsgPy.AttGuidMsg_C_addAuthor(self.sunPointData.attGuidOutMsg, self.attGuidMsg)

    def SetNadirPointGuidance(self, SimBase):
        """
        Defines the nadir pointing guidance module.
        """
        self.hillPointData.transNavInMsg.subscribeTo(
            SimBase.DynModels[self.spacecraftIndex].simpleNavObject.transOutMsg
        )
        self.hillPointData.celBodyInMsg.subscribeTo(
            SimBase.EnvModel.ephemConverter.ephemOutMsgs[1]
        )
        cMsgPy.AttRefMsg_C_addAuthor(self.hillPointData.attRefOutMsg, self.attRefMsg)

    def SetLocationPointGuidance(self, SimBase):
        """
        Defines the Earth location pointing guidance module.
        """
        self.locPointConfig.pHat_B = [0, 0, 1]
        self.locPointConfig.scAttInMsg.subscribeTo(
            SimBase.DynModels[self.spacecraftIndex].simpleNavObject.attOutMsg
        )
        self.locPointConfig.scTransInMsg.subscribeTo(
            SimBase.DynModels[self.spacecraftIndex].simpleNavObject.transOutMsg
        )
        self.locPointConfig.locationInMsg.subscribeTo(
            SimBase.EnvModel.imagingTargetList[
                self.spacecraftIndex
            ].currentGroundStateOutMsg
        )
        self.locPointConfig.useBoresightRateDamping = 1
        cMsgPy.AttGuidMsg_C_addAuthor(
            self.locPointConfig.attGuidOutMsg, self.attGuidMsg
        )

    def SetMomentumDumping(self, SimBase):
        """
        Defines the momentum dumping configuration.
        """
        self.thrDesatControlConfig.hs_min = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get(
            "hs_min"
        )  # Nms
        self.thrDesatControlConfig.rwSpeedsInMsg.subscribeTo(
            SimBase.DynModels[self.spacecraftIndex].rwStateEffector.rwSpeedOutMsg
        )
        self.thrDesatControlConfig.rwConfigDataInMsg.subscribeTo(self.fswRwConfigMsg)

        self.thrDumpConfig.deltaHInMsg.subscribeTo(
            self.thrDesatControlConfig.deltaHOutMsg
        )
        self.thrDumpConfig.thrusterImpulseInMsg.subscribeTo(
            self.thrForceMappingConfig.thrForceCmdOutMsg
        )
        self.thrDumpConfig.thrusterConfInMsg.subscribeTo(self.thrusterConfigMsg)
        self.thrDumpConfig.maxCounterValue = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("maxCounterValue")
        self.thrDumpConfig.thrMinFireTime = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("thrMinFireTime")

    def SetAttitudeTrackingError(self, SimBase):
        """
        Defines the module that converts a reference message into a guidance message.
        """
        self.trackingErrorData.attNavInMsg.subscribeTo(
            SimBase.DynModels[self.spacecraftIndex].simpleNavObject.attOutMsg
        )
        self.trackingErrorData.attRefInMsg.subscribeTo(self.attRefMsg)
        cMsgPy.AttGuidMsg_C_addAuthor(
            self.trackingErrorData.attGuidOutMsg, self.attGuidMsg
        )

    def SetMRPFeedbackRWA(self, SimBase):
        """
        Defines the control properties.
        """
        self.mrpFeedbackControlData.guidInMsg.subscribeTo(self.attGuidMsg)
        self.mrpFeedbackControlData.vehConfigInMsg.subscribeTo(self.vcConfigMsg)
        self.mrpFeedbackControlData.K = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("K")
        self.mrpFeedbackControlData.Ki = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("Ki")
        self.mrpFeedbackControlData.P = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("P")
        self.mrpFeedbackControlData.integralLimit = (
            2.0 / self.mrpFeedbackControlData.Ki * 0.1
        )

    def SetThrusterMapping(self, SimBase):
        """
        Defines the thrusters mapping.
        """
        self.thrForceMappingConfig.cmdTorqueInMsg.subscribeTo(
            self.thrDesatControlConfig.deltaHOutMsg
        )
        self.thrForceMappingConfig.thrConfigInMsg.subscribeTo(self.thrusterConfigMsg)
        self.thrForceMappingConfig.vehConfigInMsg.subscribeTo(self.vcConfigMsg)
        self.thrForceMappingConfig.controlAxes_B = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("controlAxes_B")
        self.thrForceMappingConfig.thrForceSign = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("thrForceSign")
        self.thrForceMappingConfig.angErrThresh = 3.15

    def SetRWConfigMsg(self, SimBase):
        """
        Imports the RWs configuration information.
        """
        # Configure RW pyramid exactly as it is in the Dynamics
        # (i.e. FSW with perfect knowledge)
        # the same msg is used here for both spacecraft
        self.fswRwConfigMsg = SimBase.DynModels[
            self.spacecraftIndex
        ].rwFactory.getConfigMessage()

    def SetThrustersConfigMsg(self, SimBase):
        """
        Imports the thrusters configuration information.
        """
        self.thrusterConfigMsg = SimBase.DynModels[
            self.spacecraftIndex
        ].thrFactory.getConfigMessage()

    def SetVehicleConfigMsg(self, SimBase):
        """
        Set the vehicle configuration message.
        """
        # Specify the vehicle configuration message to tell things what the vehicle
        # inertia is
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()

        # Use the same inertia in the FSW algorithm as in the simulation
        vehicleConfigOut.ISCPntB_B = SimBase.DynModels[self.spacecraftIndex].I
        self.vcConfigMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    def SetRWMotorTorque(self, SimBase):
        """
        Defines the motor torque from the control law.
        """
        self.rwMotorTorqueConfig.rwParamsInMsg.subscribeTo(self.fswRwConfigMsg)
        self.rwMotorTorqueConfig.vehControlInMsg.subscribeTo(
            self.mrpFeedbackControlData.cmdTorqueOutMsg
        )
        self.rwMotorTorqueConfig.controlAxes_B = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("controlAxes_B")

    def SetInstrumentController(self, SimBase):
        """
        Defines the instrument controller.
        """
        self.simpleInsControlConfig.attErrTolerance = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("imageAttErrorRequirement")
        if (
            "imageRateErrorRequirement"
            in SimBase.initial_conditions[str(self.spacecraftIndex)].keys()
            and SimBase.initial_conditions[str(self.spacecraftIndex)][
                "imageRateErrorRequirement"
            ]
            != 1
        ):
            self.simpleInsControlConfig.useRateTolerance = 1
            self.simpleInsControlConfig.rateErrTolerance = SimBase.initial_conditions[
                str(self.spacecraftIndex)
            ].get("imageRateErrorRequirement")
        self.simpleInsControlConfig.attGuidInMsg.subscribeTo(self.attGuidMsg)
        self.simpleInsControlConfig.locationAccessInMsg.subscribeTo(
            SimBase.EnvModel.imagingTargetList[self.spacecraftIndex].accessOutMsgs[-1]
        )

    # Global call to initialize every module
    def InitAllFSWObjects(self, SimBase):
        """
        Initializes all FSW objects.
        """
        self.SetSunPointGuidance(SimBase)
        self.SetNadirPointGuidance(SimBase)
        self.SetLocationPointGuidance(SimBase)
        self.SetAttitudeTrackingError(SimBase)
        self.SetRWConfigMsg(SimBase)
        self.SetThrustersConfigMsg(SimBase)
        self.SetVehicleConfigMsg(SimBase)
        self.SetMRPFeedbackRWA(SimBase)
        self.SetThrusterMapping(SimBase)
        self.SetMomentumDumping(SimBase)
        self.SetRWMotorTorque(SimBase)
        self.SetInstrumentController(SimBase)

    def setupGatewayMsgs(self, SimBase):
        """create C-wrapped gateway messages such that different modules can write to
        this message and provide a common input msg for down-stream modules"""
        self.attRefMsg = cMsgPy.AttRefMsg_C()
        self.attGuidMsg = cMsgPy.AttGuidMsg_C()

        self.zeroGateWayMsgs()

        # connect gateway FSW effector command msgs with the dynamics
        SimBase.DynModels[
            self.spacecraftIndex
        ].rwStateEffector.rwMotorCmdInMsg.subscribeTo(
            self.rwMotorTorqueConfig.rwMotorTorqueOutMsg
        )
        SimBase.DynModels[self.spacecraftIndex].instrument.nodeStatusInMsg.subscribeTo(
            self.simpleInsControlConfig.deviceCmdOutMsg
        )
        SimBase.DynModels[self.spacecraftIndex].thrusterSet.cmdsInMsg.subscribeTo(
            self.thrDumpConfig.thrusterOnTimeOutMsg
        )

    def zeroGateWayMsgs(self):
        """Zero all the FSW gateway message payloads"""
        self.attRefMsg.write(messaging.AttRefMsgPayload())
        self.attGuidMsg.write(messaging.AttGuidMsgPayload())
