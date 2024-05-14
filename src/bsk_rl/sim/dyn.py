"""Basilisk dynamics models are given in ``bsk_rl.sim.dyn``.

The dynamics model is the core of the satellite simulation, representing the physical
properties of the satellite and its interactions with the environment. The dynamics model
can be customized to represent different satellite configurations, actuator models, and
instrument models.

The dynamics model is selected using the ``dyn_type`` class property of the
:class:`~bsk_rl.sats.Satellite`. Certain environment elements may require specific
dynamics models, such as :class:`~bsk_rl.comm.LOSCommunication` requiring a dynamics
model that inherits from :class:`~bsk_rl.sim.dyn.LOSCommDynModel` or :class:`~bsk_rl.sats.ImagingSatellite`
requiring a dynamics model that inherits from :class:`~bsk_rl.sim.dyn.ImagingDynModel`.

Setting Parameters
------------------

Customization of the dynamics model parameters is achieved through the ``sat_args``
dictionary passed to the :class:`~bsk_rl.sats.Satellite` constructor. This dictionary is
passed on to the dynamics model setup functions, which are called each time the simulator
is reset.

Properties
----------

The dynamics model provides a number of properties for easy access to the satellite state.
These can be accessed directly from the dynamics model instance, or in the observation
via the :class:`~bsk_rl.obs.SatProperties` observation.


Aliveness Checking
------------------

Certain functions in the dynamics model are decorated with the :func:`~bsk_rl.utils.functional.aliveness_checker`
decorator. These functions are called at each step to check if the satellite is still
operational, returning true if the satellite is still alive.

"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Optional

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
    simpleStorageUnit,
    spacecraft,
    spacecraftLocation,
    spaceToGroundTransmitter,
)
from Basilisk.utilities import (
    RigidBodyKinematics,
    macros,
    orbitalMotion,
    unitTestSupport,
)

from bsk_rl.sim import world
from bsk_rl.utils import actuator_primitives as aP
from bsk_rl.utils.attitude import random_tumble
from bsk_rl.utils.functional import (
    aliveness_checker,
    check_aliveness_checkers,
    default_args,
)
from bsk_rl.utils.orbital import random_orbit

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.sim import Simulator
    from bsk_rl.sim.world import WorldModel


class DynamicsModel(ABC):
    """Abstract Basilisk dynamics model."""

    @classmethod
    def _requires_world(cls) -> list[type["WorldModel"]]:
        """Define minimum :class:`~bsk_rl.sim.world.WorldModel` for compatibility."""
        return []

    def __init__(
        self,
        satellite: "Satellite",
        dyn_rate: float,
        priority: int = 200,
        **kwargs,
    ) -> None:
        """The abstract base dynamics model.

        One DynamicsModel is instantiated for each satellite in the environment each
        time the environment is reset and new simulator is created.

        Args:
            satellite: Satellite represented by this model.
            dyn_rate: [s] Rate of dynamics simulation.
            priority: Model priority.
            kwargs: Passed through to setup functions.
        """
        self.satellite = satellite
        self.logger = self.satellite.logger.getChild(self.__class__.__name__)

        for required in self._requires_world():
            if not issubclass(type(self.simulator.world), required):
                raise TypeError(
                    f"{self.simulator.world} must be a subclass of {required} to "
                    + f"use dynamics model of type {self.__class__}"
                )

        dyn_proc_name = "DynamicsProcess" + self.satellite.id
        self.dyn_proc = self.simulator.CreateNewProcess(dyn_proc_name, priority)
        self.dyn_rate = dyn_rate
        self.task_name = "DynamicsTask" + self.satellite.id
        self.dyn_proc.addTask(
            self.simulator.CreateNewTask(self.task_name, macros.sec2nano(self.dyn_rate))
        )

        # Initialize all modules and write init one-time messages
        self.scObject: spacecraft.Spacecraft
        self._setup_dynamics_objects(**kwargs)

    @property
    def simulator(self) -> "Simulator":
        """Reference to the episode simulator."""
        return self.satellite.simulator

    @property
    def world(self) -> "WorldModel":
        """Reference to the episode world model."""
        return self.simulator.world

    @abstractmethod  # pragma: no cover
    def _setup_dynamics_objects(self, **kwargs) -> None:
        """Caller for all dynamics object initialization."""
        pass

    def is_alive(self, log_failure=False) -> bool:
        """Check if the dynamics model has failed any aliveness requirements.

        Returns:
            ``True`` if the satellite dynamics are still alive.
        """
        return check_aliveness_checkers(self, log_failure=log_failure)

    def reset_for_action(self) -> None:
        """Reset whenever a flight software :class:`~bsk_rl.sim.fsw.action` is called."""
        pass

    def __del__(self):
        """Log when dynamics are deleted."""
        self.logger.debug("Basilisk dynamics deleted")


class BasicDynamicsModel(DynamicsModel):
    """Basic Dynamics model with minimum necessary Basilisk components."""

    @classmethod
    def _requires_world(cls) -> list[type["WorldModel"]]:
        return [world.BasicWorldModel]

    def __init__(self, *args, **kwargs) -> None:
        """A dynamics model with a basic feature set.

        Includes the following:

        * Spacecraft hub physical properties
        * Gravity
        * Constant disturbance torque (defaults to none)
        * Aerodynamic drag
        * Eclipse checking for power generation
        * Reaction wheels
        * Momentum desaturation thrusters
        * Solar panels, battery, and power system

        Args:
            *args: Passed to superclass
            **kwargs: Passed to superclass
        """
        super().__init__(*args, **kwargs)

    @property
    def sigma_BN(self):
        """Body attitude MRP relative to inertial frame."""
        return self.scObject.scStateOutMsg.read().sigma_BN

    @property
    def BN(self):
        """Body relative to inertial frame rotation matrix."""
        return RigidBodyKinematics.MRP2C(self.sigma_BN)

    @property
    def omega_BN_B(self):
        """Body rate relative to inertial frame in body frame [rad/s]."""
        return self.scObject.scStateOutMsg.read().omega_BN_B

    @property
    def BP(self):
        """Body relative to planet freame rotation matrix."""
        return np.matmul(self.BN, self.world.PN.T)

    @property
    def r_BN_N(self):
        """Body position relative to inertial origin in inertial frame [m]."""
        return self.scObject.scStateOutMsg.read().r_BN_N

    @property
    def r_BN_P(self):
        """Body position relative to inertial origin in planet frame [m]."""
        return np.matmul(self.world.PN, self.r_BN_N)

    @property
    def v_BN_N(self):
        """Body velocity relative to inertial origin in inertial frame [m/s]."""
        return self.scObject.scStateOutMsg.read().v_BN_N

    @property
    def v_BN_P(self):
        """Planet-frame derivative of ``r_BN``."""
        omega_NP_P = np.matmul(self.world.PN, -self.world.omega_PN_N)
        return np.matmul(self.world.PN, self.v_BN_N) + np.cross(omega_NP_P, self.r_BN_P)

    @property
    def omega_BP_P(self):
        """Body angular velocity relative to planet frame in plant frame [rad/s]."""
        omega_BN_N = np.matmul(self.BN.T, self.omega_BN_B)
        omega_BP_N = omega_BN_N - self.world.omega_PN_N
        return np.matmul(self.world.PN, omega_BP_N)

    @property
    def battery_charge(self):
        """Battery charge [W*s]."""
        return self.powerMonitor.batPowerOutMsg.read().storageLevel

    @property
    def battery_charge_fraction(self):
        """Battery charge as a fraction of capacity."""
        return self.battery_charge / self.powerMonitor.storageCapacity

    @property
    def wheel_speeds(self):
        """Wheel speeds [rad/s]."""
        return np.array(self.rwStateEffector.rwSpeedOutMsg.read().wheelSpeeds)[0:3]

    @property
    def wheel_speeds_fraction(self):
        """Wheel speeds normalized by maximum allowable speed."""
        return self.wheel_speeds / (self.maxWheelSpeed * macros.rpm2radsec)

    def _setup_dynamics_objects(self, **kwargs) -> None:
        self.setup_spacecraft_hub(**kwargs)
        self.setup_drag_effector(**kwargs)
        self.setup_reaction_wheel_dyn_effector(**kwargs)
        self.setup_thruster_dyn_effector()
        self.setup_simple_nav_object()
        self.setup_eclipse_object()
        self.setup_solar_panel(**kwargs)
        self.setup_battery(**kwargs)
        self.setup_power_sink(**kwargs)
        self.setup_reaction_wheel_power(**kwargs)
        self.setup_thruster_power(**kwargs)

    @default_args(
        mass=330,
        width=1.38,
        depth=1.04,
        height=1.58,
        sigma_init=lambda: random_tumble(maxSpinRate=0.0001)[0],
        omega_init=lambda: random_tumble(maxSpinRate=0.0001)[1],
        rN=None,
        vN=None,
        oe=random_orbit,
        mu=orbitalMotion.MU_EARTH * 1e9,
    )
    def setup_spacecraft_hub(
        self,
        mass: float,
        width: float,
        depth: float,
        height: float,
        sigma_init: Iterable[float],
        omega_init: Iterable[float],
        oe: Iterable[float],
        rN: Iterable[float],
        vN: Iterable[float],
        mu: float,
        priority: int = 2000,
        **kwargs,
    ) -> None:
        """Set up the spacecraft hub physical properties and state.

        The hub is assumed to be a uniform-density rectangular prism with the center of
        mass at the center.

        Args:
            mass: [kg] Hub mass.
            width: [m] Hub width.
            depth: [m] Hub depth.
            height: [m] Hub height.
            sigma_init: Initial attitude MRP.
            omega_init: [rad/s] Initial body rate.
            oe: Orbital element tuple of (semimajor axis [km], eccentricity, inclination
                [rad], ascending node [rad], argument of periapsis [rad], initial true
                anomaly [rad]). Either ``oe`` and ``mu`` or ``rN`` and ``vN`` must be
                provided, but not both.
            mu: Gravitational parameter (used only with ``oe``).
            rN: [m] Initial inertial position.
            vN: [m/s] Initial inertial velocity.
            priority: Model priority.
            kwargs: Passed to other setup functions.
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
        self.I_mat = [Ixx, 0.0, 0.0, 0.0, Iyy, 0.0, 0.0, 0.0, Izz]

        self.scObject.hub.mHub = mass  # kg
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(self.I_mat)

        # Set the initial attitude and position
        self.scObject.hub.sigma_BNInit = sigma_init
        self.scObject.hub.omega_BN_BInit = omega_init
        self.scObject.hub.r_CN_NInit = unitTestSupport.np2EigenVectorXd(rN)
        self.scObject.hub.v_CN_NInit = unitTestSupport.np2EigenVectorXd(vN)

        self.simulator.AddModelToTask(
            self.task_name, self.scObject, ModelPriority=priority
        )

        self.setup_gravity_bodies()
        self.setup_disturbance_torque(**kwargs)
        self.setup_density_model()

    def setup_gravity_bodies(self) -> None:
        """Set up gravitational bodies from the :class:`~bsk_rl.sim.world.WorldModel` to included in the simulation."""
        self.scObject.gravField.gravBodies = spacecraft.GravBodyVector(
            list(self.world.gravFactory.gravBodies.values())
        )

    @default_args(disturbance_vector=None)
    def setup_disturbance_torque(
        self, disturbance_vector: Optional[Iterable[float]] = None, **kwargs
    ) -> None:
        """Set up a constant disturbance torque acting on the satellite.

        Args:
            disturbance_vector: [N*m] Constant disturbance torque in the body frame.
            kwargs: Passed to other setup functions.
        """
        if disturbance_vector is None:
            disturbance_vector = np.array([0, 0, 0])
        self.extForceTorqueObject = extForceTorque.ExtForceTorque()
        self.extForceTorqueObject.ModelTag = "DisturbanceTorque"
        self.extForceTorqueObject.extTorquePntB_B = disturbance_vector
        self.scObject.addDynamicEffector(self.extForceTorqueObject)

    def setup_density_model(self) -> None:
        """Set up the atmospheric density model effector."""
        self.world.densityModel.addSpacecraftToModel(self.scObject.scStateOutMsg)

    @default_args(dragCoeff=2.2)
    def setup_drag_effector(
        self,
        width: float,
        depth: float,
        height: float,
        panelArea: float,
        dragCoeff: float,
        priority: int = 999,
        **kwargs,
    ) -> None:
        """Set up the satellite drag effector.

        The drag effector causes aerodynamic forces and torques to act on the satellite.
        For purposes of this model, the satellite is assumed to be a rectangular prism
        with a solar panel on one end.

        Args:
            width: [m] Hub width.
            depth: [m] Hub depth.
            height: [m] Hub height.
            panelArea: [m^2] Solar panel surface area.
            dragCoeff: Drag coefficient.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        self.dragEffector.ModelTag = "FacetDrag"
        #  Set up the geometry of a small satellite, starting w/ bus
        self.dragEffector.addFacet(
            width * depth, dragCoeff, [1, 0, 0], [height / 2, 0.0, 0]
        )
        self.dragEffector.addFacet(
            width * depth, dragCoeff, [-1, 0, 0], [height / 2, 0.0, 0]
        )
        self.dragEffector.addFacet(
            height * width, dragCoeff, [0, 1, 0], [0, depth / 2, 0]
        )
        self.dragEffector.addFacet(
            height * width, dragCoeff, [0, -1, 0], [0, -depth / 2, 0]
        )
        self.dragEffector.addFacet(
            height * depth, dragCoeff, [0, 0, 1], [0, 0, width / 2]
        )
        self.dragEffector.addFacet(
            height * depth, dragCoeff, [0, 0, -1], [0, 0, -width / 2]
        )
        # Add solar panels
        self.dragEffector.addFacet(
            panelArea / 2,
            dragCoeff,
            [0, 1, 0],
            [0, height, 0],
        )
        self.dragEffector.addFacet(
            panelArea / 2,
            dragCoeff,
            [0, -1, 0],
            [0, height, 0],
        )
        self.dragEffector.atmoDensInMsg.subscribeTo(
            self.world.densityModel.envOutMsgs[-1]
        )
        self.scObject.addDynamicEffector(self.dragEffector)

        self.simulator.AddModelToTask(
            self.task_name, self.dragEffector, ModelPriority=priority
        )

    def setup_simple_nav_object(self, priority: int = 1400, **kwargs) -> None:
        """Set up the navigation module.

        Args:
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.simpleNavObject = simpleNav.SimpleNav()
        self.simpleNavObject.ModelTag = "SimpleNav"
        self.simpleNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.simulator.AddModelToTask(
            self.task_name, self.simpleNavObject, ModelPriority=priority
        )

    @aliveness_checker
    def altitude_valid(self) -> bool:
        """Check that satellite has not deorbited.

        Checks if altitude is greater than 200km above Earth's surface.
        """
        return np.linalg.norm(self.r_BN_N) > (orbitalMotion.REQ_EARTH + 200) * 1e3

    @default_args(
        wheelSpeeds=lambda: np.random.uniform(-1500, 1500, 3),
        maxWheelSpeed=np.inf,
        u_max=0.200,
    )
    def setup_reaction_wheel_dyn_effector(
        self,
        wheelSpeeds: Iterable[float],
        maxWheelSpeed: float,
        u_max: float,
        priority: int = 997,
        **kwargs,
    ) -> None:
        """Set the reaction wheel state effector parameters.

        Three reaction wheels modeled on the HR16 wheel are used.

        Args:
            wheelSpeeds: [rpm] Initial speeds of each wheel.
            maxWheelSpeed: [rpm] Failure speed for wheels.
            u_max: [N*m] Maximum torque producible by each wheel.
            priority: Model priority.
            kwargs: Passed to other setup functions.
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
        """Check if any wheel speed exceeds the ``maxWheelSpeed``."""
        valid = all(
            abs(speed) < self.maxWheelSpeed * macros.rpm2radsec
            for speed in self.wheel_speeds
        )
        return valid

    def setup_thruster_dyn_effector(self, priority: int = 996) -> None:
        """Set up the thruster state effector.

        Args:
            priority: Model priority.
        """
        self.thrusterSet, self.thrFactory = aP.idealMonarc1Octet()
        thrModelTag = "ACSThrusterDynamics"
        self.thrFactory.addToSpacecraft(thrModelTag, self.thrusterSet, self.scObject)
        self.simulator.AddModelToTask(
            self.task_name, self.thrusterSet, ModelPriority=priority
        )

    @default_args(thrusterPowerDraw=0.0)
    def setup_thruster_power(
        self, thrusterPowerDraw, priority: int = 899, **kwargs
    ) -> None:
        """Set up the thruster power draw.

        When momentum desaturating using wheels, power is consumed at this rate.

        Args:
            thrusterPowerDraw: [W] Constant power draw desat mode is active.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.thrusterPowerSink = simplePowerSink.SimplePowerSink()
        self.thrusterPowerSink.ModelTag = "thrustPowerSink" + self.satellite.id
        self.thrusterPowerSink.nodePowerOut = thrusterPowerDraw  # Watts
        self.simulator.AddModelToTask(
            self.task_name, self.thrusterPowerSink, ModelPriority=priority
        )
        self.powerMonitor.addPowerNodeToModel(self.thrusterPowerSink.nodePowerOutMsg)

    def setup_eclipse_object(self) -> None:
        """Add the spacecraft to the eclipse module."""
        self.world.eclipseObject.addSpacecraftToModel(self.scObject.scStateOutMsg)
        self.eclipse_index = len(self.world.eclipseObject.eclipseOutMsgs) - 1

    @default_args(
        panelArea=2 * 1.0 * 0.5,
        panelEfficiency=0.20,
        nHat_B=np.array([0, 1, 0]),
    )
    def setup_solar_panel(
        self,
        panelArea: float,
        panelEfficiency: float,
        nHat_B: Iterable[float],
        priority: int = 898,
        **kwargs,
    ) -> None:
        """Set the solar panel parameters for power generation.

        Power generation takes into account panel size and efficiency, the eclipse
        state, and the angle of solar incidence.

        Args:
            panelArea: [m^2] Solar panel area.
            panelEfficiency: Efficiency coefficient of solar to electrical power
                conversion.
            nHat_B: Body-fixed array normal vector.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.solarPanel = simpleSolarPanel.SimpleSolarPanel()
        self.solarPanel.ModelTag = "solarPanel" + self.satellite.id
        self.solarPanel.stateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.solarPanel.sunEclipseInMsg.subscribeTo(
            self.world.eclipseObject.eclipseOutMsgs[self.eclipse_index]
        )
        self.solarPanel.sunInMsg.subscribeTo(
            self.world.gravFactory.spiceObject.planetStateOutMsgs[self.world.sun_index]
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
    def setup_battery(
        self,
        batteryStorageCapacity: float,
        storedCharge_Init: float,
        priority: int = 799,
        **kwargs,
    ) -> None:
        """Set the battery model parameters.

        Args:
            batteryStorageCapacity: [W*s] Maximum battery charge.
            storedCharge_Init: [W*s] Initial battery charge.
            priority: Model priority.
            kwargs: Passed to other setup functions.
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
        """Check if the battery has charge remaining.

        Note that this check is instantaneous. If a satellite runs out of power during a
        environment step but then recharges to have positive power at the end of the step,
        the satellite will still be considered alive.
        """
        return self.battery_charge > 0

    @default_args(basePowerDraw=0.0)
    def setup_power_sink(
        self, basePowerDraw: float, priority: int = 897, **kwargs
    ) -> None:
        """Set the instrument power sink parameters.

        Args:
            basePowerDraw: [W] Baseline satellite power draw. Should be negative.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        if basePowerDraw > 0:
            self.logger.warning("basePowerDraw should probably be zero or negative.")
        self.basePowerSink = simplePowerSink.SimplePowerSink()
        self.basePowerSink.ModelTag = "basePowerSink" + self.satellite.id
        self.basePowerSink.nodePowerOut = basePowerDraw  # Watts
        self.simulator.AddModelToTask(
            self.task_name, self.basePowerSink, ModelPriority=priority
        )
        self.powerMonitor.addPowerNodeToModel(self.basePowerSink.nodePowerOutMsg)
        self.basePowerSink.powerStatus = 1

    @default_args(
        rwBasePower=0.4, rwMechToElecEfficiency=0.0, rwElecToMechEfficiency=0.5
    )
    def setup_reaction_wheel_power(
        self,
        rwBasePower: float,
        rwMechToElecEfficiency: float,
        rwElecToMechEfficiency: float,
        priority: int = 987,
        **kwargs,
    ) -> None:
        """Set the reaction wheel power draw.

        Args:
            rwBasePower: [W] Constant power draw when operational.
            rwMechToElecEfficiency: Efficiency factor to convert mechanical power to
                electrical power.
            rwElecToMechEfficiency: Efficiency factor to convert electrical power to
                mechanical power.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.rwPowerList = []
        for i_device in range(self.rwFactory.getNumOfDevices()):
            powerRW = ReactionWheelPower.ReactionWheelPower()
            powerRW.ModelTag = "rwPower" + str(i_device)
            powerRW.basePowerNeed = rwBasePower
            powerRW.rwStateInMsg.subscribeTo(self.rwStateEffector.rwOutMsgs[i_device])
            powerRW.mechToElecEfficiency = rwMechToElecEfficiency
            powerRW.elecToMechEfficiency = rwElecToMechEfficiency
            self.rwPowerList.append(powerRW)
            self.simulator.AddModelToTask(
                self.task_name, powerRW, ModelPriority=(priority - i_device)
            )
            self.powerMonitor.addPowerNodeToModel(powerRW.nodePowerOutMsg)


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

    def __init__(self, *args, **kwargs) -> None:
        """Equips the satellite with an instrument, storage unit, and transmitter.

        This dynamics model is used with :class:`~bsk_rl.sats.ImagingSatellite`. It
        provides the satellite with the ability to take images of a point target. To
        enable downlink, use :class:`GroundStationDynModel` and :class:`~bsk_rl.sim.world.GroundStationWorldModel`.
        """
        super().__init__(*args, **kwargs)

    @property
    def storage_level(self):
        """Storage level [bits]."""
        return self.storageUnit.storageUnitDataOutMsg.read().storageLevel

    @property
    def storage_level_fraction(self):
        """Storage level as a fraction of capacity."""
        return self.storage_level / self.storageUnit.storageCapacity

    def _setup_dynamics_objects(self, **kwargs) -> None:
        super()._setup_dynamics_objects(**kwargs)
        self.setup_instrument_power_sink(**kwargs)
        self.setup_transmitter_power_sink(**kwargs)
        self.setup_instrument(**kwargs)
        self.setup_transmitter(**kwargs)
        self.setup_storage_unit(**kwargs)
        self.setup_imaging_target(**kwargs)

    @default_args(instrumentBaudRate=8e6)
    def setup_instrument(
        self, instrumentBaudRate: float, priority: int = 895, **kwargs
    ) -> None:
        """Set up the instrument data collection model.

        Args:
            instrumentBaudRate: [bits] Data generated by an image.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.instrument = simpleInstrument.SimpleInstrument()
        self.instrument.ModelTag = "instrument" + self.satellite.id
        self.instrument.nodeBaudRate = (
            instrumentBaudRate / self.dyn_rate
        )  # makes imaging instantaneous
        self.instrument.nodeDataName = "Instrument" + self.satellite.id
        self.simulator.AddModelToTask(
            self.task_name, self.instrument, ModelPriority=priority
        )

    @default_args(transmitterBaudRate=-8e6, transmitterNumBuffers=100)
    def setup_transmitter(
        self,
        transmitterBaudRate: float,
        instrumentBaudRate: float,
        transmitterNumBuffers: int,
        priority: int = 798,
        **kwargs,
    ) -> None:
        """Set up the transmitter model for downlinking data.

        Args:
            transmitterBaudRate: [baud] Rate of data downlink. Should be negative.
            instrumentBaudRate: [bits] Image size, used to set packet size.
            transmitterNumBuffers: Number of transmitter buffers
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        if transmitterBaudRate > 0:
            self.logger.warning("transmitterBaudRate should probably be negative.")
        self.transmitter = spaceToGroundTransmitter.SpaceToGroundTransmitter()
        self.transmitter.ModelTag = "transmitter" + self.satellite.id
        self.transmitter.nodeBaudRate = transmitterBaudRate  # baud
        # set packet size equal to the size of a single image
        self.transmitter.packetSize = -instrumentBaudRate  # bits
        self.transmitter.numBuffers = transmitterNumBuffers
        self.simulator.AddModelToTask(
            self.task_name, self.transmitter, ModelPriority=priority
        )

    @default_args(instrumentPowerDraw=-30.0)
    def setup_instrument_power_sink(
        self, instrumentPowerDraw: float, priority: int = 897, **kwargs
    ) -> None:
        """Set the instrument power sink parameters.

        The instrument draws power when in an imaging task, representing the power cost
        of operating the instrument.

        Args:
            instrumentPowerDraw: [W] Power draw when instrument is enabled.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        if instrumentPowerDraw > 0:
            self.logger.warning(
                "instrumentPowerDraw should probably be zero or negative."
            )
        self.instrumentPowerSink = simplePowerSink.SimplePowerSink()
        self.instrumentPowerSink.ModelTag = "insPowerSink" + self.satellite.id
        self.instrumentPowerSink.nodePowerOut = instrumentPowerDraw
        self.simulator.AddModelToTask(
            self.task_name, self.instrumentPowerSink, ModelPriority=priority
        )
        self.powerMonitor.addPowerNodeToModel(self.instrumentPowerSink.nodePowerOutMsg)

    @default_args(transmitterPowerDraw=-15.0)
    def setup_transmitter_power_sink(
        self, transmitterPowerDraw: float, priority: int = 896, **kwargs
    ) -> None:
        """Set the transmitter power sink parameters.

        The transmitter draws power when in a downlink task, representing the power cost
        of downlinking data.

        Args:
            transmitterPowerDraw: [W] Power draw when transmitter is enabled.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        if transmitterPowerDraw > 0:
            self.logger.warning(
                "transmitterPowerDraw should probably be zero or negative."
            )
        self.transmitterPowerSink = simplePowerSink.SimplePowerSink()
        self.transmitterPowerSink.ModelTag = "transPowerSink" + self.satellite.id
        self.transmitterPowerSink.nodePowerOut = transmitterPowerDraw
        self.simulator.AddModelToTask(
            self.task_name, self.transmitterPowerSink, ModelPriority=priority
        )
        self.powerMonitor.addPowerNodeToModel(self.transmitterPowerSink.nodePowerOutMsg)

    @default_args(
        dataStorageCapacity=20 * 8e6,
        bufferNames=None,
        storageUnitValidCheck=False,
        storageInit=0,
    )
    def setup_storage_unit(
        self,
        dataStorageCapacity: int,
        storageUnitValidCheck: bool,
        storageInit: int,
        transmitterNumBuffers: Optional[int] = None,
        bufferNames: Optional[Iterable[str]] = None,
        priority: int = 699,
        **kwargs,
    ) -> None:
        """Configure the storage unit and its buffers.

        Separate buffers can be used to track imaging of different targets. Often, the
        buffer names will be set up by satellite based on the scenario configuration.

        Args:
            dataStorageCapacity: [bits] Maximum data that can be stored.
            transmitterNumBuffers: Number of unit buffers. Not necessary if ``bufferNames``
                are given.
            bufferNames: List of buffer names to use. Named by number if ``None``.
            storageUnitValidCheck: If ``True``, enforce that the storage level is below
                the storage capacity when checking aliveness.
            storageInit: [bits] Initial storage level.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.storageUnit = partitionedStorageUnit.PartitionedStorageUnit()
        self.storageUnit.ModelTag = "storageUnit" + self.satellite.id
        self.storageUnit.storageCapacity = dataStorageCapacity  # bits
        self.storageUnit.addDataNodeToModel(self.instrument.nodeDataOutMsg)
        self.storageUnit.addDataNodeToModel(self.transmitter.nodeDataOutMsg)
        self.storageUnitValidCheck = storageUnitValidCheck
        # Add all of the targets to the data buffer
        if bufferNames is None:
            for buffer_idx in range(transmitterNumBuffers):
                self.storageUnit.addPartition(str(buffer_idx))
        else:
            if transmitterNumBuffers is not None and transmitterNumBuffers != len(
                bufferNames
            ):
                raise ValueError(
                    "transmitterNumBuffers cannot be different than len(bufferNames)."
                )
            for buffer_name in bufferNames:
                self.storageUnit.addPartition(buffer_name)

        if storageInit != 0:
            self.storageUnit.setDataBuffer(["STORED DATA"], [int(storageInit)])

        # Add the storage unit to the transmitter
        self.transmitter.addStorageUnitToTransmitter(
            self.storageUnit.storageUnitDataOutMsg
        )

        self.simulator.AddModelToTask(
            self.task_name, self.storageUnit, ModelPriority=priority
        )

    @aliveness_checker
    def data_storage_valid(self) -> bool:
        """Check that the buffer has not run out of space.

        Only is checked if ``storageUnitValidCheck`` is ``True``; otherwise, a full storage
        unit will prevent additional data from being stored but will not cause the satellite
        to be considered dead.
        """
        storage_check = self.storageUnitValidCheck
        if storage_check:
            return self.storage_level < self.storageUnit.storageCapacity or np.isclose(
                self.storage_level, self.storageUnit.storageCapacity
            )
        else:
            return True

    @default_args(
        groundLocationPlanetRadius=orbitalMotion.REQ_EARTH * 1e3,
        imageTargetMinimumElevation=np.radians(45.0),
        imageTargetMaximumRange=-1,
    )
    def setup_imaging_target(
        self,
        groundLocationPlanetRadius: float,
        imageTargetMinimumElevation: float,
        imageTargetMaximumRange: float,
        priority: int = 2000,
        **kwargs,
    ) -> None:
        """Add a generic imaging target to dynamics.

        The target must be updated with a particular location when used.

        Args:
            groundLocationPlanetRadius: [m] Radius of ground locations from center of planet.
            imageTargetMinimumElevation: [rad] Minimum elevation angle from target to
                satellite when imaging.
            imageTargetMaximumRange: [m] Maximum range from target to satellite when
                imaging. -1 to disable.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.imagingTarget = groundLocation.GroundLocation()
        self.imagingTarget.ModelTag = "ImagingTarget"
        self.imagingTarget.planetRadius = groundLocationPlanetRadius
        self.imagingTarget.specifyLocation(0.0, 0.0, 1000.0)
        self.imagingTarget.planetInMsg.subscribeTo(
            self.world.gravFactory.spiceObject.planetStateOutMsgs[self.world.body_index]
        )
        self.imagingTarget.minimumElevation = imageTargetMinimumElevation
        self.imagingTarget.maximumRange = imageTargetMaximumRange

        self.simulator.AddModelToTask(
            self.world.world_task_name,
            self.imagingTarget,
            ModelPriority=priority,
        )
        self.imagingTarget.addSpacecraftToModel(self.scObject.scStateOutMsg)

    def reset_for_action(self) -> None:
        """Shut off power sinks unless the transmitter or instrument is being used."""
        super().reset_for_action()
        self.transmitter.dataStatus = 0
        self.transmitterPowerSink.powerStatus = 0
        self.instrumentPowerSink.powerStatus = 0


class ContinuousImagingDynModel(ImagingDynModel):
    """Equips the satellite for continuous nadir imaging."""

    def __init__(self, *args, **kwargs) -> None:
        """Equips the satellite for continuous nadir imaging.

        Equips satellite with an instrument, storage unit, and transmitter
        for continuous nadir imaging. A single data buffer is used for storage, and data
        is accumulated continuously while imaging. The imaging target is fixed at the
        center of the Earth for nadir imaging.
        """
        super().__init__(*args, **kwargs)

    @default_args(instrumentBaudRate=8e6)
    def setup_instrument(
        self, instrumentBaudRate: float, priority: int = 895, **kwargs
    ) -> None:
        """Set up the continuous instrument model.

        Args:
            instrumentBaudRate: [baud] Data generation rate step when continuously imaging.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.instrument = simpleInstrument.SimpleInstrument()
        self.instrument.ModelTag = "instrument" + self.satellite.id
        self.instrument.nodeBaudRate = instrumentBaudRate  # make imaging instantaneous
        self.instrument.nodeDataName = "Instrument" + self.satellite.id
        self.simulator.AddModelToTask(
            self.task_name, self.instrument, ModelPriority=priority
        )

    @default_args(
        dataStorageCapacity=20 * 8e6,
        storageUnitValidCheck=False,
        storageInit=0,
    )
    def setup_storage_unit(
        self,
        dataStorageCapacity: int,
        storageUnitValidCheck: bool,
        storageInit: int,
        priority: int = 699,
        **kwargs,
    ) -> None:
        """Configure the storage unit and its buffers.

        Args:
            dataStorageCapacity: [bits] Maximum data that can be stored.
            storageUnitValidCheck: If True, check that the storage level is below the
                storage capacity.
            storageInit: [bits] Initial storage level.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.storageUnit = simpleStorageUnit.SimpleStorageUnit()
        self.storageUnit.ModelTag = "storageUnit" + self.satellite.id
        self.storageUnit.storageCapacity = dataStorageCapacity  # bits
        self.storageUnit.addDataNodeToModel(self.instrument.nodeDataOutMsg)
        self.storageUnit.addDataNodeToModel(self.transmitter.nodeDataOutMsg)
        self.storageUnitValidCheck = storageUnitValidCheck
        self.storageUnit.setDataBuffer(storageInit)

        # Add the storage unit to the transmitter
        self.transmitter.addStorageUnitToTransmitter(
            self.storageUnit.storageUnitDataOutMsg
        )

        self.simulator.AddModelToTask(
            self.task_name, self.storageUnit, ModelPriority=priority
        )

    @default_args(imageTargetMaximumRange=-1)
    def setup_imaging_target(
        self,
        imageTargetMaximumRange: float = -1,
        priority: int = 2000,
        **kwargs,
    ) -> None:
        """Add a imaging target at the center of the Earth.

        Args:
            imageTargetMaximumRange: [m] Maximum range from target to satellite when
                imaging. -1 to disable.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        self.imagingTarget = groundLocation.GroundLocation()
        self.imagingTarget.ModelTag = "scanningTarget"
        self.imagingTarget.planetRadius = 1e-6
        self.imagingTarget.specifyLocation(0, 0, 0)
        self.imagingTarget.planetInMsg.subscribeTo(
            self.world.gravFactory.spiceObject.planetStateOutMsgs[self.world.body_index]
        )
        self.imagingTarget.minimumElevation = np.radians(-90)
        self.imagingTarget.maximumRange = imageTargetMaximumRange

        self.simulator.AddModelToTask(
            self.world.world_task_name,
            self.imagingTarget,
            ModelPriority=priority,
        )
        self.imagingTarget.addSpacecraftToModel(self.scObject.scStateOutMsg)


class GroundStationDynModel(ImagingDynModel):
    """Model that connects satellite to world ground stations."""

    def __init__(self, *args, **kwargs) -> None:
        """Model that connects satellite to world ground stations.

        This model enables the use of ground stations defined in :class:`~bsk_rl.sim.world.GroundStationWorldModel`
        for data downlink.
        """
        super().__init__(*args, **kwargs)

    @classmethod
    def _requires_world(cls) -> list[type["WorldModel"]]:
        return super()._requires_world() + [world.GroundStationWorldModel]

    def _setup_dynamics_objects(self, **kwargs) -> None:
        super()._setup_dynamics_objects(**kwargs)
        self.setup_ground_station_locations()

    def setup_ground_station_locations(self) -> None:
        """Connect the transmitter to ground stations."""
        for groundStation in self.world.groundStations:
            groundStation.addSpacecraftToModel(self.scObject.scStateOutMsg)
            self.transmitter.addAccessMsgToTransmitter(groundStation.accessOutMsgs[-1])


class FullFeaturedDynModel(GroundStationDynModel, LOSCommDynModel):
    """Convenience class for a satellite with ground station and line-of-sight comms."""

    def __init__(self, *args, **kwargs) -> None:
        """Convenience class for an imaging satellite with ground stations and line-of-sight communication."""
        super().__init__(*args, **kwargs)


__doc_title__ = "Dynamics Sims"
__all__ = [
    "DynamicsModel",
    "BasicDynamicsModel",
    "LOSCommDynModel",
    "ImagingDynModel",
    "ContinuousImagingDynModel",
    "GroundStationDynModel",
    "FullFeaturedDynModel",
]
