from Basilisk.simulation import (
    ReactionWheelPower,
    facetDragDynamicEffector,
    partitionedStorageUnit,
    simpleBattery,
    simpleInstrument,
    simpleNav,
    simplePowerSink,
    simpleSolarPanel,
    spacecraft,
    spacecraftLocation,
    spaceToGroundTransmitter,
)
from Basilisk.utilities import macros as mc
from Basilisk.utilities import unitTestSupport

from bsk_rl.utilities.effector_primitives import actuator_primitives as ap


class DynamicModel:
    """Defines the Dynamics class."""

    def __init__(self, SimBase, dynRate, spacecraftIndex, singleSat=False):
        # Define class variables
        self.spacecraftIndex = spacecraftIndex
        self.dynRate = dynRate

        # Define process name, task name and task time-step
        self.taskName = "DynamicsTask" + str(spacecraftIndex)
        self.processTasksTimeStep = mc.sec2nano(self.dynRate)

        # Create task
        SimBase.dynProc[spacecraftIndex].addTask(
            SimBase.CreateNewTask(self.taskName, self.processTasksTimeStep)
        )

        # Initialize all modules and write init one-time messages
        self.InitAllDynObjects(SimBase)

        # Assign initialized modules to tasks
        SimBase.AddModelToTask(self.taskName, self.dragEffector, ModelPriority=999)
        SimBase.AddModelToTask(self.taskName, self.simpleNavObject, ModelPriority=1400)
        SimBase.AddModelToTask(self.taskName, self.rwStateEffector, ModelPriority=997)
        SimBase.AddModelToTask(self.taskName, self.thrusterSet, ModelPriority=996)
        SimBase.AddModelToTask(self.taskName, self.scObject, ModelPriority=2000)
        SimBase.AddModelToTask(self.taskName, self.solarPanel, ModelPriority=898)
        SimBase.AddModelToTask(
            self.taskName, self.instrumentPowerSink, ModelPriority=897
        )
        SimBase.AddModelToTask(
            self.taskName, self.transmitterPowerSink, ModelPriority=896
        )
        SimBase.AddModelToTask(self.taskName, self.instrument, ModelPriority=895)
        SimBase.AddModelToTask(self.taskName, self.powerMonitor, ModelPriority=799)
        SimBase.AddModelToTask(self.taskName, self.transmitter, ModelPriority=798)
        SimBase.AddModelToTask(self.taskName, self.storageUnit, ModelPriority=699)
        for ind in range(self.rwFactory.getNumOfDevices()):
            SimBase.AddModelToTask(
                self.taskName, self.rwPowerList[ind], ModelPriority=(987 - ind)
            )
        if not singleSat:
            SimBase.AddModelToTask(self.taskName, self.losComms, ModelPriority=500)

    # These are module-initialization methods

    def SetSpacecraftHub(self, SimBase):
        """Defines the spacecraft object properties."""
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "sat-" + str(self.spacecraftIndex)
        # Grab the mass for readability in inertia computation
        mass = SimBase.initial_conditions[str(self.spacecraftIndex)].get("mass")
        # Get the spacecraft geometric properties
        self.width = SimBase.initial_conditions[str(self.spacecraftIndex)].get("width")
        self.depth = SimBase.initial_conditions[str(self.spacecraftIndex)].get("depth")
        self.height = SimBase.initial_conditions[str(self.spacecraftIndex)].get(
            "height"
        )

        self.I_mat = [
            1.0 / 12.0 * mass * (self.width**2.0 + self.depth**2.0),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * mass * (self.depth**2.0 + self.height**2.0),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * mass * (self.width**2.0 + self.height**2.0),
        ]

        self.scObject.hub.mHub = mass  # kg
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(self.I_mat)

        # Set the initial attitude and position
        self.scObject.hub.sigma_BNInit = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("sigma_init")
        self.scObject.hub.omega_BN_BInit = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("omega_init")
        self.scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(
            SimBase.initial_conditions[str(self.spacecraftIndex)].get("rN")
        )
        self.scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(
            SimBase.initial_conditions[str(self.spacecraftIndex)].get("vN")
        )

    def SetGravityBodies(self, SimBase):
        """Specify what gravitational bodies to include in the simulation."""
        # Attach the gravity body
        self.scObject.gravField.gravBodies = spacecraft.GravBodyVector(
            list(SimBase.EnvModel.gravFactory.gravBodies.values())
        )

    def SetDisturbanceTorque(self, SimBase):
        """Attach the disturbance torque to the spacecraft object."""
        self.scObject.addDynamicEffector(
            SimBase.EnvModel.extForceTorqueObjectList[self.spacecraftIndex]
        )

    def SetDensityModel(self, SimBase):
        """Attaches the density model effector to the spacecraft."""
        SimBase.EnvModel.densityModel.addSpacecraftToModel(self.scObject.scStateOutMsg)

    def SetDragEffector(self, SimBase):
        """Set the drag effector."""
        self.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        self.dragEffector.ModelTag = "FacetDrag"
        #  Set up the geometry of a small satellite, starting w/ bus
        self.dragEffector.addFacet(
            self.width * self.depth, 2.2, [1, 0, 0], [self.height / 2, 0.0, 0]
        )
        self.dragEffector.addFacet(
            self.width * self.depth, 2.2, [-1, 0, 0], [self.height / 2, 0.0, 0]
        )
        self.dragEffector.addFacet(
            self.height * self.width, 2.2, [0, 1, 0], [0, self.depth / 2, 0]
        )
        self.dragEffector.addFacet(
            self.height * self.width, 2.2, [0, -1, 0], [0, -self.depth / 2, 0]
        )
        self.dragEffector.addFacet(
            self.height * self.depth, 2.2, [0, 0, 1], [0, 0, self.width / 2]
        )
        self.dragEffector.addFacet(
            self.height * self.depth, 2.2, [0, 0, -1], [0, 0, -self.width / 2]
        )
        # Add solar panels
        self.dragEffector.addFacet(
            SimBase.initial_conditions[str(self.spacecraftIndex)].get("panelArea") / 2,
            2.2,
            [0, 1, 0],
            [0, self.height, 0],
        )
        self.dragEffector.addFacet(
            SimBase.initial_conditions[str(self.spacecraftIndex)].get("panelArea") / 2,
            2.2,
            [0, -1, 0],
            [0, self.height, 0],
        )
        self.dragEffector.atmoDensInMsg.subscribeTo(
            SimBase.EnvModel.densityModel.envOutMsgs[-1]
        )
        self.scObject.addDynamicEffector(self.dragEffector)

    def SetGroundLocations(self, SimBase):
        """Adds the spacecraft to the ground location modules."""
        SimBase.EnvModel.boulderGroundStation.addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )
        SimBase.EnvModel.merrittGroundStation.addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )
        SimBase.EnvModel.singaporeGroundStation.addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )
        SimBase.EnvModel.weilheimGroundStation.addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )
        SimBase.EnvModel.santiagoGroundStation.addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )
        SimBase.EnvModel.dongaraGroundStation.addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )
        SimBase.EnvModel.hawaiiGroundStation.addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )
        SimBase.EnvModel.imagingTargetList[self.spacecraftIndex].addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )

    def SetEclipseObject(self, SimBase):
        """Adds the spacecraft to the eclipse module."""
        SimBase.EnvModel.eclipseObject.addSpacecraftToModel(self.scObject.scStateOutMsg)

    def SetSimpleNavObject(self):
        """Defines the navigation module."""
        self.simpleNavObject = simpleNav.SimpleNav()
        self.simpleNavObject.ModelTag = "SimpleNav"
        self.simpleNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)

    def SetReactionWheelDynEffector(self, SimBase):
        """Defines the RW state effector."""
        self.rwStateEffector, self.rwFactory, initWheelSpeeds = ap.balancedHR16Triad(
            useRandom=False,
            randomBounds=(-800, 800),
            wheelSpeeds=SimBase.initial_conditions[str(self.spacecraftIndex)].get(
                "wheelSpeeds"
            ),
        )
        self.rwFactory.addToSpacecraft(
            "ReactionWheels", self.rwStateEffector, self.scObject
        )

    def SetThrusterDynEffector(self):
        """Defines the thruster state effector."""
        self.thrusterSet, self.thrFactory = ap.idealMonarc1Octet()
        thrModelTag = "ACSThrusterDynamics"
        self.thrFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)

    def SetSolarPanel(self, SimBase):
        """Sets the solar panel."""
        self.solarPanel = simpleSolarPanel.SimpleSolarPanel()
        self.solarPanel.ModelTag = "solarPanel" + str(self.spacecraftIndex)
        self.solarPanel.stateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.solarPanel.sunEclipseInMsg.subscribeTo(
            SimBase.EnvModel.eclipseObject.eclipseOutMsgs[self.spacecraftIndex]
        )
        self.solarPanel.sunInMsg.subscribeTo(
            SimBase.EnvModel.gravFactory.spiceObject.planetStateOutMsgs[
                SimBase.EnvModel.sun
            ]
        )
        self.solarPanel.setPanelParameters(
            unitTestSupport.np2EigenVectorXd(
                SimBase.initial_conditions[str(self.spacecraftIndex)].get("nHat_B")
            ),
            SimBase.initial_conditions[str(self.spacecraftIndex)].get("panelArea"),
            SimBase.initial_conditions[str(self.spacecraftIndex)].get(
                "panelEfficiency"
            ),
        )

    def SetInstrumentPowerSink(self, SimBase):
        """Defines the instrument power sink parameters."""
        self.instrumentPowerSink = simplePowerSink.SimplePowerSink()
        self.instrumentPowerSink.ModelTag = "insPowerSink" + str(self.spacecraftIndex)
        self.instrumentPowerSink.nodePowerOut = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get(
            "instrumentPowerDraw"
        )  # Watts

    def SetTransmitterPowerSink(self, SimBase):
        """Defines the trasmitter power sink parameters."""
        self.transmitterPowerSink = simplePowerSink.SimplePowerSink()
        self.transmitterPowerSink.ModelTag = "transPowerSink" + str(
            self.spacecraftIndex
        )
        self.transmitterPowerSink.nodePowerOut = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get(
            "transmitterPowerDraw"
        )  # Watts

    def SetReactionWheelPower(self, SimBase):
        """Defines the reaction wheel power draw."""
        self.rwPowerList = []
        for ind in range(self.rwFactory.getNumOfDevices()):
            powerRW = ReactionWheelPower.ReactionWheelPower()
            powerRW.ModelTag = "rwPower" + str(ind)
            powerRW.basePowerNeed = SimBase.initial_conditions[
                str(self.spacecraftIndex)
            ].get(
                "rwBasePower"
            )  # baseline power draw, Watts
            powerRW.rwStateInMsg.subscribeTo(self.rwStateEffector.rwOutMsgs[ind])
            powerRW.mechToElecEfficiency = SimBase.initial_conditions[
                str(self.spacecraftIndex)
            ].get("rwMechToElecEfficiency")
            powerRW.elecToMechEfficiency = SimBase.initial_conditions[
                str(self.spacecraftIndex)
            ].get("rwElecToMechEfficiency")
            self.rwPowerList.append(powerRW)

    def SetBattery(self, SimBase):
        """Sets up the battery with all the power components."""
        self.powerMonitor = simpleBattery.SimpleBattery()
        self.powerMonitor.ModelTag = "powerMonitor"
        self.powerMonitor.storageCapacity = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("batteryStorageCapacity")
        self.powerMonitor.storedCharge_Init = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("storedCharge_Init")
        self.powerMonitor.addPowerNodeToModel(self.solarPanel.nodePowerOutMsg)
        self.powerMonitor.addPowerNodeToModel(self.instrumentPowerSink.nodePowerOutMsg)
        self.powerMonitor.addPowerNodeToModel(self.transmitterPowerSink.nodePowerOutMsg)
        for powerRW in self.rwPowerList:
            self.powerMonitor.addPowerNodeToModel(powerRW.nodePowerOutMsg)

    def SetInstrument(self, SimBase):
        """Create the instrument."""
        self.instrument = simpleInstrument.SimpleInstrument()
        self.instrument.ModelTag = "instrument" + str(self.spacecraftIndex)
        self.instrument.nodeBaudRate = (
            SimBase.initial_conditions[str(self.spacecraftIndex)].get(
                "instrumentBaudRate"
            )
            / self.dynRate
        )  # baud
        self.instrument.nodeDataName = "Instrument" + str(self.spacecraftIndex)

    def SetTransmitter(self, SimBase):
        """Create the transmitter."""
        self.transmitter = spaceToGroundTransmitter.SpaceToGroundTransmitter()
        self.transmitter.ModelTag = "transmitter" + str(self.spacecraftIndex)
        self.transmitter.nodeBaudRate = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get(
            "transmitterBaudRate"
        )  # baud
        self.transmitter.packetSize = -SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get(
            "instrumentBaudRate"
        )  # bits, set packet size equal to the size of a single image
        self.transmitter.numBuffers = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get("transmitterNumBuffers")
        self.transmitter.addAccessMsgToTransmitter(
            SimBase.EnvModel.boulderGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            SimBase.EnvModel.merrittGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            SimBase.EnvModel.singaporeGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            SimBase.EnvModel.weilheimGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            SimBase.EnvModel.santiagoGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            SimBase.EnvModel.dongaraGroundStation.accessOutMsgs[-1]
        )
        self.transmitter.addAccessMsgToTransmitter(
            SimBase.EnvModel.hawaiiGroundStation.accessOutMsgs[-1]
        )

    def SetStorageUnit(self, SimBase):
        self.storageUnit = partitionedStorageUnit.PartitionedStorageUnit()
        self.storageUnit.ModelTag = "storageUnit" + str(self.spacecraftIndex)
        self.storageUnit.storageCapacity = SimBase.initial_conditions[
            str(self.spacecraftIndex)
        ].get(
            "dataStorageCapacity"
        )  # bits (1 GB)
        self.storageUnit.addDataNodeToModel(self.instrument.nodeDataOutMsg)
        self.storageUnit.addDataNodeToModel(self.transmitter.nodeDataOutMsg)
        # Add all of the targets to the data buffer
        for idx in range(
            SimBase.initial_conditions[str(self.spacecraftIndex)].get(
                "transmitterNumBuffers"
            )
        ):
            self.storageUnit.addPartition(str(idx))

        # Add the storage unit to the transmitter
        self.transmitter.addStorageUnitToTransmitter(
            self.storageUnit.storageUnitDataOutMsg
        )

    def SetLosComms(self, SimBase):
        self.losComms = spacecraftLocation.SpacecraftLocation()
        self.losComms.ModelTag = "losComms"
        self.losComms.primaryScStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.losComms.planetInMsg.subscribeTo(
            SimBase.EnvModel.gravFactory.spiceObject.planetStateOutMsgs[
                SimBase.EnvModel.earth
            ]
        )
        self.losComms.rEquator = SimBase.EnvModel.planet.radEquator
        self.losComms.rPolar = SimBase.EnvModel.planet.radEquator * 0.98
        self.losComms.maximumRange = -1.0  # m, unlimited

    # Global call to initialize every module
    def InitAllDynObjects(self, SimBase):
        """Initializes all dynamic objects."""
        self.SetSpacecraftHub(SimBase)
        self.SetGravityBodies(SimBase)
        self.SetDensityModel(SimBase)
        self.SetDragEffector(SimBase)
        self.SetReactionWheelDynEffector(SimBase)
        self.SetThrusterDynEffector()
        self.SetSimpleNavObject()
        self.SetGroundLocations(SimBase)
        self.SetEclipseObject(SimBase)
        self.SetReactionWheelPower(SimBase)
        self.SetSolarPanel(SimBase)
        self.SetInstrumentPowerSink(SimBase)
        self.SetTransmitterPowerSink(SimBase)
        self.SetBattery(SimBase)
        self.SetInstrument(SimBase)
        self.SetTransmitter(SimBase)
        self.SetStorageUnit(SimBase)
        self.SetLosComms(SimBase)
        self.SetDisturbanceTorque(SimBase)

    def ConnectLosComms(self, SatList):
        for sat in SatList:
            if sat != self.scObject:
                self.losComms.addSpacecraftToModel(sat.scStateOutMsg)
