"""Basic dynamics model for BSK-RL."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
from Basilisk.simulation import (
    ReactionWheelPower,
    extForceTorque,
    facetDragDynamicEffector,
    simpleBattery,
    simpleNav,
    simplePowerSink,
    simpleSolarPanel,
    spacecraft,
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
from bsk_rl.utils.orbital import random_orbit, rv2HN, rv2omega

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

        dyn_proc_name = "DynamicsProcess" + self.satellite.name
        self.dyn_proc = self.simulator.CreateNewProcess(dyn_proc_name, priority)
        self.dyn_rate = dyn_rate
        self.task_name = "DynamicsTask" + self.satellite.name
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

    @default_args(utc_init="this value will be set by the world model")
    def _utc_init(self, utc_init: float) -> None:
        """Exists so that utc_init is registered as part of sat_args."""
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
    def HN(self):
        """Hill frame relative to inertial frame rotation matrix."""
        return rv2HN(self.r_BN_N, self.v_BN_N)

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
    def omega_BH_H(self):
        """Body angular velocity relative to Hill frame in Hill frame [rad/s]."""
        omega_BN_N = np.matmul(self.BN.T, self.omega_BN_B)
        omega_HN_N = rv2omega(self.r_BN_N, self.v_BN_N)
        omega_BH_N = omega_BN_N - omega_HN_N
        HN = rv2HN(self.r_BN_N, self.v_BN_N)
        return HN @ omega_BH_N

    def _compute_oes(self):
        if not hasattr(self, "_oe_cache_time") or (
            self._oe_cache_time != getattr(self.simulator, "time", None)
        ):
            self._oe_cache = orbitalMotion.rv2elem(
                mu=self.mu, rVec=np.array(self.r_BN_N), vVec=np.array(self.v_BN_N)
            )
            self._oe_cache_time = getattr(self.simulator, "time", None)
        return self._oe_cache

    @property
    def semi_major_axis(self):
        """Semimajor axis of the satellite's orbit [km]."""
        return self._compute_oes().a

    @property
    def eccentricity(self):
        """Eccentricity of the satellite's orbit [-]."""
        return self._compute_oes().e

    @property
    def inclination(self):
        """Inclination of the satellite's orbit [rad]."""
        return self._compute_oes().i

    @property
    def ascending_node(self):
        """Longitude of ascending node of the satellite's orbit [rad]."""
        return self._compute_oes().Omega

    @property
    def argument_of_periapsis(self):
        """Argument of periapsis of the satellite's orbit [rad]."""
        return self._compute_oes().omega

    @property
    def true_anomaly(self):
        """True anomaly of the satellite's orbit [rad]."""
        return self._compute_oes().f

    @property
    def beta_angle(self):
        """Beta angle of the satellite's orbit, between 0 and 2pi [rad].

        The angle between the angular momentum vector and the sun direction vector.
        """
        r_BN_N = self.r_BN_N
        v_BN_N = self.v_BN_N
        h_N = np.cross(r_BN_N, v_BN_N)
        r_SN_N = (
            self.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[
                self.simulator.world.sun_index
            ]
            .read()
            .PositionVector
        )

        beta = np.arccos(
            np.dot(h_N, r_SN_N) / (np.linalg.norm(h_N) * np.linalg.norm(r_SN_N))
        )
        return beta

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
        oe: Optional[Iterable[float]],
        rN: Optional[Iterable[float]],
        vN: Optional[Iterable[float]],
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
        self.mu = mu
        if rN is not None and vN is not None and oe is None:
            pass
        elif oe is not None and rN is None and vN is None:
            rN, vN = orbitalMotion.elem2rv(mu, oe)
        else:
            raise (KeyError("Orbit is overspecified. Provide either (rN, vN) or oe"))

        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = self.satellite.name

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
        self.thrusterPowerSink.ModelTag = "thrustPowerSink" + self.satellite.name
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
        nHat_B=np.array([0, 0, -1]),
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
        self.solarPanel.ModelTag = "solarPanel" + self.satellite.name
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
        if storedCharge_Init > batteryStorageCapacity or storedCharge_Init < 0:
            self.logger.warning(
                f"Battery initial charge {storedCharge_Init} incompatible with its capacity {batteryStorageCapacity}."
            )
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
        self.basePowerSink.ModelTag = "basePowerSink" + self.satellite.name
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


__doc_title__ = "Dynamics Base"

__all__ = [
    "DynamicsModel",
    "BasicDynamicsModel",
]
