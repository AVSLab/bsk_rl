from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from bsk_rl.envs.GeneralSatelliteTasking.types import (
        EnvironmentModel,
        Satellite,
    )

import numpy as np
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
    spacecraft,
    spacecraftLocation,
    spaceToGroundTransmitter,
)
from Basilisk.utilities import macros as mc
from Basilisk.utilities import orbitalMotion, unitTestSupport

from bsk_rl.envs.GeneralSatelliteTasking.simulation import environment
from bsk_rl.envs.GeneralSatelliteTasking.utils.functional import (
    aliveness_checker,
    check_aliveness_checkers,
    default_args,
)
from bsk_rl.utilities.effector_primitives import actuator_primitives as aP
from bsk_rl.utilities.initial_conditions import leo_orbit, sc_attitudes


class DynamicsModel(ABC):
    @classmethod
    @property
    def requires_env(cls) -> list[type["EnvironmentModel"]]:
        """Define minimum EnvironmentModels for compatibility."""
        return []

    def __init__(
        self,
        satellite: "Satellite",
        dyn_rate: float,
        priority: int = 200,
        **kwargs,
    ) -> None:
        """Base DynamicsModel

        Args:
            satellite: Satellite modelled by this model
            dyn_rate: Rate of dynamics simulation [s]
            priority: Model priority.
        """
        self.satellite = satellite
        self.simulator = satellite.simulator
        assert all(
            [
                issubclass(type(self.simulator.environment), required)
                for required in self.requires_env
            ]
        )

        dyn_proc_name = "DynamicsProcess" + self.satellite.id
        self.dyn_proc = self.simulator.CreateNewProcess(dyn_proc_name, priority)
        self.dyn_rate = dyn_rate
        self.task_name = "DynamicsTask" + self.satellite.id
        self.dyn_proc.addTask(
            self.simulator.CreateNewTask(self.task_name, mc.sec2nano(self.dyn_rate))
        )

        # Initialize all modules and write init one-time messages
        self.scObject: spacecraft.Spacecraft
        self._init_dynamics_objects(**kwargs)

    @abstractmethod
    def _init_dynamics_objects(self, **kwargs) -> None:
        """Caller for all dynamics object initialization"""
        pass

    def is_alive(self) -> bool:
        """Check if the dynamics model has failed any aliveness requirements.

        Returns:
            If the satellite dynamics are still alive
        """
        return check_aliveness_checkers(self)

    def reset_for_action(self) -> None:
        """Called whenever a FSW @action is called"""
        pass


class BasicDynamicsModel(DynamicsModel):
    """Minimal set of Basilisk dynamics objects"""

    @classmethod
    @property
    def requires_env(cls) -> list[type["EnvironmentModel"]]:
        return [environment.BasicEnvironmentModel]

    def _init_dynamics_objects(self, **kwargs) -> None:
        self._set_spacecraft_hub(**kwargs)
        self._set_drag_effector(**kwargs)
        self._set_reaction_wheel_dyn_effector(**kwargs)
        self._set_thruster_dyn_effector()
        self._set_simple_nav_object()
        self._set_eclipse_object()
        self._set_solar_panel(**kwargs)
        self._set_battery(**kwargs)
        self._set_reaction_wheel_power(**kwargs)

    @default_args(
        mass=330,
        width=1.38,
        depth=1.04,
        height=1.58,
        sigma_init=lambda: sc_attitudes.random_tumble(maxSpinRate=0.0001)[0],
        omega_init=lambda: sc_attitudes.random_tumble(maxSpinRate=0.0001)[1],
        rN=None,
        vN=None,
        oe=None,
        mu=leo_orbit.mu,
    )
    def _set_spacecraft_hub(
        self,
        mass: float,
        width: float,
        depth: float,
        height: float,
        sigma_init: Iterable[float],
        omega_init: Iterable[float],
        rN: Iterable[float],
        vN: Iterable[float],
        oe: Iterable[float],
        mu: float,
        priority: int = 2000,
        **kwargs,
    ) -> None:
        """Defines the spacecraft object properties.

        Args:
            mass: Hub mass [kg]
            width: Hub width [m]
            depth: Hub depth [m]
            height: Hub height [m]
            sigma_init: Initial MRP
            omega_init: Initial body rate [rad/s]
            rN: Initial inertial position [m]
            vN: Initial inertial velocity [m/s]
            oe: (a, e, i, AN, AP, f); alternative to rN, vN [km, rad]
            mu: Gravitational parameter (used only with oe)
            priority: Model priority.
        """
        if rN is not None and vN is not None and oe is None:
            pass
        elif oe is not None and rN is None and vN is None:
            rN, vN = orbitalMotion.elem2rv(mu, oe)
        else:
            raise (KeyError("Orbit is overspecified. Provide either (rN, vN) or oe"))

        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "sat-" + self.satellite.id

        Ixx = 1.0 / 12.0 * mass * (width**2.0 + depth**2.0)
        Iyy = 1.0 / 12.0 * mass * (depth**2.0 + height**2.0)
        Izz = 1.0 / 12.0 * mass * (width**2.0 + height**2.0)
        self.I = [Ixx, 0.0, 0.0, 0.0, Iyy, 0.0, 0.0, 0.0, Izz]

        self.scObject.hub.mHub = mass  # kg
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(self.I)

        # Set the initial attitude and position
        self.scObject.hub.sigma_BNInit = sigma_init
        self.scObject.hub.omega_BN_BInit = omega_init
        self.scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(rN)
        self.scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(vN)

        self.simulator.AddModelToTask(
            self.task_name, self.scObject, ModelPriority=priority
        )

        self._set_gravity_bodies()
        self._set_disturbance_torque(**kwargs)
        self._set_density_model()

    def _set_gravity_bodies(self) -> None:
        """Specify what gravitational bodies to include in the simulation."""
        self.scObject.gravField.gravBodies = spacecraft.GravBodyVector(
            list(self.simulator.environment.gravFactory.gravBodies.values())
        )

    @default_args(disturbance_vector=None)
    def _set_disturbance_torque(
        self, disturbance_vector: Iterable[float] | None = None, **kwargs
    ) -> None:
        """Attach the disturbance torque to the satellite.

        Args:
            disturbance_vector: Constant disturbance torque [N*m].
        """
        if disturbance_vector is None:
            disturbance_vector = np.array([0, 0, 0])
        self.extForceTorqueObject = extForceTorque.ExtForceTorque()
        self.extForceTorqueObject.ModelTag = "DisturbanceTorque"
        self.extForceTorqueObject.extTorquePntB_B = disturbance_vector
        self.scObject.addDynamicEffector(self.extForceTorqueObject)

    def _set_density_model(self) -> None:
        """Attaches the density model effector to the satellite."""
        self.simulator.environment.densityModel.addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )

    def _set_drag_effector(
        self,
        width: float,
        depth: float,
        height: float,
        panelArea: float,
        priority: int = 999,
        **kwargs,
    ) -> None:
        """Attach the drag effector to the satellite.

        Args:
            width: Hub width [m]
            depth: Hub depth [m]
            height: Hub height [m]
            panelArea: Solar panel surface area [m**2]
            priority: Model priority.
        """
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
            panelArea / 2,
            2.2,
            [0, 1, 0],
            [0, height, 0],
        )
        self.dragEffector.addFacet(
            panelArea / 2,
            2.2,
            [0, -1, 0],
            [0, height, 0],
        )
        self.dragEffector.atmoDensInMsg.subscribeTo(
            self.simulator.environment.densityModel.envOutMsgs[-1]
        )
        self.scObject.addDynamicEffector(self.dragEffector)

        self.simulator.AddModelToTask(
            self.task_name, self.dragEffector, ModelPriority=priority
        )

    def _set_simple_nav_object(self, priority: int = 1400, **kwargs) -> None:
        """Defines the navigation module.

        Args:
            priority: Model priority.
        """
        self.simpleNavObject = simpleNav.SimpleNav()
        self.simpleNavObject.ModelTag = "SimpleNav"
        self.simpleNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.simulator.AddModelToTask(
            self.task_name, self.simpleNavObject, ModelPriority=priority
        )

    @aliveness_checker
    def altitude_valid(self) -> bool:
        """Check for deorbit by checking if altitude is greater than 200km above Earth's surface."""
        return (
            np.linalg.norm(self.scObject.scStateOutMsg.read().r_BN_N)
            > (orbitalMotion.REQ_EARTH + 200) * 1e3
        )

    @default_args(
        wheelSpeeds=lambda: np.random.uniform(-1500, 1500, 3),
        maxWheelSpeed=np.inf,
        u_max=0.200,
    )
    def _set_reaction_wheel_dyn_effector(
        self,
        wheelSpeeds: Iterable[float],
        maxWheelSpeed: float,
        u_max: float,
        priority: int = 997,
        **kwargs,
    ) -> None:
        """Defines the RW state effector.

        Args:
            wheelSpeeds: Initial speeds of each wheel [RPM]
            maxWheelSpeed: Failure speed for wheels [RPM]
            priority: Model priority.
        """
        self.maxWheelSpeed = maxWheelSpeed
        self.rwStateEffector, self.rwFactory, _ = aP.balancedHR16Triad(
            useRandom=False,
            wheelSpeeds=wheelSpeeds,
        )
        for RW in self.rwFactory.rwList.values():
            RW.u_max = u_max
        self.rwFactory.addToSpacecraft(
            "ReactionWheels", self.rwStateEffector, self.scObject
        )
        self.simulator.AddModelToTask(
            self.task_name, self.rwStateEffector, ModelPriority=priority
        )

    @aliveness_checker
    def rw_speeds_valid(self) -> bool:
        """Check if any wheel speed exceeds the maximum."""
        valid = all(
            abs(speed) < self.maxWheelSpeed * mc.rpm2radsec
            for speed in self.rwStateEffector.rwSpeedOutMsg.read().wheelSpeeds
        )
        return valid

    def _set_thruster_dyn_effector(self, priority: int = 996) -> None:
        """Defines the thruster state effector.

        Args:
            priority: Model priority.
        """
        self.thrusterSet, self.thrFactory = aP.idealMonarc1Octet()
        thrModelTag = "ACSThrusterDynamics"
        self.thrFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)
        self.simulator.AddModelToTask(
            self.task_name, self.thrusterSet, ModelPriority=priority
        )

    def _set_eclipse_object(self) -> None:
        """Adds the spacecraft to the eclipse module"""
        self.simulator.environment.eclipseObject.addSpacecraftToModel(
            self.scObject.scStateOutMsg
        )
        self.eclipse_index = (
            len(self.simulator.environment.eclipseObject.eclipseOutMsgs) - 1
        )

    @default_args(
        panelArea=2 * 1.0 * 0.5,
        panelEfficiency=0.20,
        nHat_B=np.array([0, 1, 0]),
    )
    def _set_solar_panel(
        self,
        panelArea: float,
        panelEfficiency: float,
        nHat_B: Iterable[float],
        priority: int = 898,
        **kwargs,
    ) -> None:
        """Sets the solar panel for power generation.

        Args:
            panelArea: Solar panel surface area [m**2]
            panelEfficiency: Efficiency coefficient of solar to electrical power conversion
            nHat_B: Body-fixed array normal vector
            priority: Model priority.
        """
        self.solarPanel = simpleSolarPanel.SimpleSolarPanel()
        self.solarPanel.ModelTag = "solarPanel" + self.satellite.id
        self.solarPanel.stateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.solarPanel.sunEclipseInMsg.subscribeTo(
            self.simulator.environment.eclipseObject.eclipseOutMsgs[self.eclipse_index]
        )
        self.solarPanel.sunInMsg.subscribeTo(
            self.simulator.environment.gravFactory.spiceObject.planetStateOutMsgs[
                self.simulator.environment.sun_index
            ]
        )
        self.solarPanel.setPanelParameters(
            unitTestSupport.np2EigenVectorXd(nHat_B),
            panelArea,
            panelEfficiency,
        )
        self.simulator.AddModelToTask(
            self.task_name, self.solarPanel, ModelPriority=priority
        )

    @default_args(
        batteryStorageCapacity=80.0 * 3600.0,
        storedCharge_Init=lambda: np.random.uniform(30.0 * 3600.0, 70.0 * 3600.0),
    )
    def _set_battery(
        self,
        batteryStorageCapacity: float,
        storedCharge_Init: float,
        priority: int = 799,
        **kwargs,
    ) -> None:
        """Sets the battery model.

        Args:
            batteryStorageCapacity: Maximum battery charge [W*s]
            storedCharge_Init: Initial battery charge [W*s]
            priority: Model priority.
        """
        self.powerMonitor = simpleBattery.SimpleBattery()
        self.powerMonitor.ModelTag = "powerMonitor"
        self.powerMonitor.storageCapacity = batteryStorageCapacity
        self.powerMonitor.storedCharge_Init = storedCharge_Init
        self.powerMonitor.addPowerNodeToModel(self.solarPanel.nodePowerOutMsg)
        self.simulator.AddModelToTask(
            self.task_name, self.powerMonitor, ModelPriority=priority
        )

    @aliveness_checker
    def battery_valid(self) -> bool:
        """Check if the battery has charge remaining."""
        return self.powerMonitor.batPowerOutMsg.read().storageLevel > 0

    @default_args(
        rwBasePower=0.4, rwMechToElecEfficiency=0.0, rwElecToMechEfficiency=0.5
    )
    def _set_reaction_wheel_power(
        self,
        rwBasePower: float,
        rwMechToElecEfficiency: float,
        rwElecToMechEfficiency: float,
        priority: int = 987,
        **kwargs,
    ) -> None:
        """Defines the reaction wheel power draw.

        Args:
            rwBasePower: Constant power draw when operational [W]
            rwMechToElecEfficiency: Efficiency factor to convert mechanical power to electrical power
            rwElecToMechEfficiency: Efficiency factor to convert electrical power to mechanical power
            priority: Model priority.
        """
        self.rwPowerList = []
        for i_device in range(self.rwFactory.getNumOfDevices()):
            powerRW = ReactionWheelPower.ReactionWheelPower()
            powerRW.ModelTag = "rwPower" + str(i_device)
            powerRW.basePowerNeed = rwBasePower  # baseline power draw, Watts
            powerRW.rwStateInMsg.subscribeTo(self.rwStateEffector.rwOutMsgs[i_device])
            powerRW.mechToElecEfficiency = rwMechToElecEfficiency
            powerRW.elecToMechEfficiency = rwElecToMechEfficiency
            self.rwPowerList.append(powerRW)
            self.simulator.AddModelToTask(
                self.task_name, powerRW, ModelPriority=(priority - i_device)
            )
            self.powerMonitor.addPowerNodeToModel(powerRW.nodePowerOutMsg)


class LOSCommDynModel(BasicDynamicsModel):
    """For evaluating line-of-sight connections between satellites for communication"""

    def _init_dynamics_objects(self, **kwargs) -> None:
        super()._init_dynamics_objects(**kwargs)
        self._set_los_comms(**kwargs)

    def _set_los_comms(self, priority: int = 500, **kwargs) -> None:
        """Set up line-of-sight visibility checking between satellites.

        Args:
            priority: Model priority.
        """
        self.losComms = spacecraftLocation.SpacecraftLocation()
        self.losComms.ModelTag = "losComms"
        self.losComms.primaryScStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.losComms.planetInMsg.subscribeTo(
            self.simulator.environment.gravFactory.spiceObject.planetStateOutMsgs[
                self.simulator.environment.body_index
            ]
        )
        self.losComms.rEquator = self.simulator.environment.planet.radEquator
        self.losComms.rPolar = self.simulator.environment.planet.radEquator * 0.98
        self.losComms.maximumRange = -1.0  # m, unlimited

        self.los_comms_ids = []

        for sat_dyn in self.simulator.dynamics_list.values():
            if sat_dyn != self and sat_dyn.satellite.id not in self.los_comms_ids:
                self.losComms.addSpacecraftToModel(sat_dyn.scObject.scStateOutMsg)
                self.los_comms_ids.append(sat_dyn.satellite.id)
                sat_dyn.losComms.addSpacecraftToModel(self.scObject.scStateOutMsg)
                sat_dyn.los_comms_ids.append(self.satellite.id)
                if len(sat_dyn.los_comms_ids) == 1:
                    sat_dyn.simulator.AddModelToTask(
                        sat_dyn.task_name, sat_dyn.losComms, ModelPriority=priority
                    )

        if len(self.los_comms_ids) > 0:
            self.simulator.AddModelToTask(
                self.task_name, self.losComms, ModelPriority=priority
            )


class ImagingDynModel(BasicDynamicsModel):
    """Equips the satellite with an instrument, storage unit, and transmitter."""

    def _init_dynamics_objects(self, **kwargs) -> None:
        super()._init_dynamics_objects(**kwargs)
        self._set_instrument_power_sink(**kwargs)
        self._set_transmitter_power_sink(**kwargs)
        self._set_instrument(**kwargs)
        self._set_transmitter(**kwargs)
        self._set_storage_unit(**kwargs)
        self._set_imaging_target(**kwargs)

    @default_args(instrumentBaudRate=8e6)
    def _set_instrument(
        self, instrumentBaudRate: float, priority: int = 895, **kwargs
    ) -> None:
        """Create the instrument model.

        Args:
            instrumentBaudRate: Data generated in a single step by an image [bits]
            priority: Model priority.
        """
        self.instrument = simpleInstrument.SimpleInstrument()
        self.instrument.ModelTag = "instrument" + self.satellite.id
        self.instrument.nodeBaudRate = (
            instrumentBaudRate / self.dyn_rate
        )  # make imaging instantaneous
        self.instrument.nodeDataName = "Instrument" + self.satellite.id
        self.simulator.AddModelToTask(
            self.task_name, self.instrument, ModelPriority=priority
        )

    @default_args(transmitterBaudRate=8e6, transmitterNumBuffers=100)
    def _set_transmitter(
        self,
        transmitterBaudRate: float,
        instrumentBaudRate: float,
        transmitterNumBuffers: int,
        priority: int = 798,
        **kwargs,
    ) -> None:
        """Create the transmitter model.

        Args:
            transmitterBaudRate: Rate of data downlink [baud]
            instrumentBaudRate: Image size, used to set packet size [bits]
            transmitterNumBuffers: Number of transmitter buffers
            priority: Model priority.
        """
        self.transmitter = spaceToGroundTransmitter.SpaceToGroundTransmitter()
        self.transmitter.ModelTag = "transmitter" + self.satellite.id
        self.transmitter.nodeBaudRate = transmitterBaudRate  # baud
        # set packet size equal to the size of a single image
        self.transmitter.packetSize = -instrumentBaudRate  # bits
        self.transmitter.numBuffers = transmitterNumBuffers
        self.simulator.AddModelToTask(
            self.task_name, self.transmitter, ModelPriority=798
        )

    @default_args(instrumentPowerDraw=-30.0)
    def _set_instrument_power_sink(
        self, instrumentPowerDraw: float, priority: int = 897, **kwargs
    ) -> None:
        """Defines the instrument power sink parameters.

        Args:
            instrumentPowerDraw: Power draw when instrument is enabled [W]
            priority: Model priority.
        """
        self.instrumentPowerSink = simplePowerSink.SimplePowerSink()
        self.instrumentPowerSink.ModelTag = "insPowerSink" + self.satellite.id
        self.instrumentPowerSink.nodePowerOut = instrumentPowerDraw  # Watts
        self.simulator.AddModelToTask(
            self.task_name, self.instrumentPowerSink, ModelPriority=priority
        )
        self.powerMonitor.addPowerNodeToModel(self.instrumentPowerSink.nodePowerOutMsg)

    @default_args(transmitterPowerDraw=-15.0)
    def _set_transmitter_power_sink(
        self, transmitterPowerDraw: float, priority: int = 896, **kwargs
    ) -> None:
        """Defines the transmitter power sink parameters.

        Args:
            transmitterPowerDraw: Power draw when transmitter is enabled [W]
            priority: Model priority.
        """
        self.transmitterPowerSink = simplePowerSink.SimplePowerSink()
        self.transmitterPowerSink.ModelTag = "transPowerSink" + self.satellite.id
        self.transmitterPowerSink.nodePowerOut = transmitterPowerDraw  # Watts
        self.simulator.AddModelToTask(
            self.task_name, self.transmitterPowerSink, ModelPriority=priority
        )
        self.powerMonitor.addPowerNodeToModel(self.transmitterPowerSink.nodePowerOutMsg)

    @default_args(dataStorageCapacity=20 * 8e6, bufferNames=None)
    def _set_storage_unit(
        self,
        dataStorageCapacity: int,
        transmitterNumBuffers: int,
        bufferNames: Iterable[str] | None,
        priority: int = 699,
        **kwargs,
    ) -> None:
        """Configure the storage unit and its buffers.

        Args:
            dataStorageCapacity: Maximum data to be stored [bits]
            transmitterNumBuffers: Number of unit buffers
            bufferNames: List of buffer names to use. Named by number if None.
            priority: Model priority.
        """
        self.storageUnit = partitionedStorageUnit.PartitionedStorageUnit()
        self.storageUnit.ModelTag = "storageUnit" + self.satellite.id
        self.storageUnit.storageCapacity = dataStorageCapacity  # bits
        self.storageUnit.addDataNodeToModel(self.instrument.nodeDataOutMsg)
        self.storageUnit.addDataNodeToModel(self.transmitter.nodeDataOutMsg)
        # Add all of the targets to the data buffer
        if bufferNames is None:
            for buffer_idx in range(transmitterNumBuffers):
                self.storageUnit.addPartition(str(buffer_idx))
        else:
            for buffer_name in bufferNames:
                self.storageUnit.addPartition(buffer_name)

        # Add the storage unit to the transmitter
        self.transmitter.addStorageUnitToTransmitter(
            self.storageUnit.storageUnitDataOutMsg
        )

        self.simulator.AddModelToTask(
            self.task_name, self.storageUnit, ModelPriority=priority
        )

    @aliveness_checker
    def data_storage_valid(self) -> bool:
        """Check that the buffer has not run out of space."""
        return (
            self.storageUnit.storageUnitDataOutMsg.read().storageLevel
            < self.storageUnit.storageCapacity
        )

    @default_args(
        groundLocationPlanetRadius=orbitalMotion.REQ_EARTH * 1e3,
        imageTargetMinimumElevation=np.radians(45.0),
        imageTargetMaximumRange=-1,
    )
    def _set_imaging_target(
        self,
        groundLocationPlanetRadius: float,
        imageTargetMinimumElevation: float,
        imageTargetMaximumRange: float,
        priority: int = 2000,
        **kwargs,
    ) -> None:
        """Add a generic imaging target to dynamics. The target must be updated with a particular location when used.

        Args:
            groundLocationPlanetRadius: Radius of ground locations from center of planet [m]
            imageTargetMinimumElevation: Minimum elevation angle from target to satellite when imaging [rad]
            imageTargetMaximumRange: Maximum range from target to satellite when imaging. -1 to disable. [m]
            priority: Model priority.
        """
        self.imagingTarget = groundLocation.GroundLocation()
        self.imagingTarget.ModelTag = "ImagingTarget"
        self.imagingTarget.planetRadius = groundLocationPlanetRadius
        self.imagingTarget.specifyLocation(0.0, 0.0, 1000.0)
        self.imagingTarget.planetInMsg.subscribeTo(
            self.simulator.environment.gravFactory.spiceObject.planetStateOutMsgs[
                self.simulator.environment.body_index
            ]
        )
        self.imagingTarget.minimumElevation = imageTargetMinimumElevation
        self.imagingTarget.maximumRange = imageTargetMaximumRange

        self.simulator.AddModelToTask(
            self.simulator.environment.env_task_name,
            self.imagingTarget,
            ModelPriority=priority,
        )
        self.imagingTarget.addSpacecraftToModel(self.scObject.scStateOutMsg)

    def reset_for_action(self) -> None:
        """Shut off power sinks."""
        super().reset_for_action()
        self.transmitter.dataStatus = 0
        self.transmitterPowerSink.powerStatus = 0
        self.instrumentPowerSink.powerStatus = 0


class GroundStationDynModel(ImagingDynModel):
    """Model that connects satellite to environment ground stations"""

    @classmethod
    @property
    def requires_env(cls) -> list[type["EnvironmentModel"]]:
        return super().requires_env + [environment.GroundStationEnvModel]

    def _init_dynamics_objects(self, **kwargs) -> None:
        super()._init_dynamics_objects(**kwargs)
        self._set_ground_station_locations()

    def _set_ground_station_locations(self) -> None:
        """Connect the transmitter to ground stations."""
        for groundStation in self.simulator.environment.groundStations:
            groundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)
            self.transmitter.addAccessMsgToTransmitter(groundStation.accessOutMsgs[-1])


class FullFeaturedDynModel(GroundStationDynModel, LOSCommDynModel):
    pass
