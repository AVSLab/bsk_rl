"""Basilisk flight software models (FSW) are given in ``bsk_rl.sim.fsw``.

Flight software models serve as the interface between the operation of the satellite in
simulation and the Gymnasium environment. While some FSW models add additional
functionality to the satellite, such as imaging instrument control in :class:`ImagingFSWModel`,
others replace the default control laws with a more complex algorithms, such as :class:`SteeringFSWModel`
vis a vis :class:`BasicFSWModel`.

Actions
-------

Each FSW model has a number of actions that can be called to task the satellite. These
actions are decorated with the :func:`~bsk_rl.sim.fsw.action` decorator, which performs
housekeeping tasks before the action is executed. These actions are the primary way to
control the satellite simulation from other parts of the Gymnasium environment.

Properties
----------

The FSW model provides a number of properties for easy access to the satellite state.
These can be accessed directly from the dynamics model instance, or in the observation
via the :class:`~bsk_rl.obs.SatProperties` observation.

Aliveness Checking
------------------

Certain functions in the FSW models are decorated with the :func:`~bsk_rl.utils.functional.aliveness_checker`
decorator. These functions are called at each step to check if the satellite is still
operational, returning true if the satellite is still alive.


"""

from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Callable, Iterable, Optional
from weakref import proxy

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (
    attTrackingError,
    hillPoint,
    locationPointing,
    mrpFeedback,
    mrpSteering,
    rateServoFullNonlinear,
    rwMotorTorque,
    scanningInstrumentController,
    simpleInstrumentController,
    thrForceMapping,
    thrMomentumDumping,
    thrMomentumManagement,
)
from Basilisk.utilities import macros as mc

from bsk_rl.sim import dyn
from bsk_rl.utils import vizard
from bsk_rl.utils.functional import (
    AbstractClassProperty,
    check_aliveness_checkers,
    default_args,
)
from bsk_rl.utils.orbital import rv2HN

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats import Satellite
    from bsk_rl.sim import Simulator
    from bsk_rl.sim.dyn import DynamicsModel
    from bsk_rl.sim.world import WorldModel


def action(
    func: Callable[..., None],
) -> Callable[Callable[..., None], Callable[..., None]]:
    """Decorator to reset the satellite software before executing an action.

    Each time an action is called, the FSW tasks and dynamics models call their
    ``reset_for_action`` methods to ensure that the satellite is in a consistent state
    before the action is executed.

    Action functions are typically called by :ref:`bsk_rl.act` to task the satellite.
    """

    @wraps(func)
    def inner(self, *args, **kwargs) -> Callable[..., None]:
        self.fsw_proc.disableAllTasks()
        self._zero_gateway_msgs()
        self.dynamics.reset_for_action()
        for task in self.tasks:
            task.reset_for_action()
        return func(self, *args, **kwargs)

    inner.__doc__ = "*Decorated with* :class:`~bsk_rl.sim.fsw.action`\n\n" + str(
        func.__doc__
    )

    return inner


class FSWModel(ABC):
    """Abstract Basilisk flight software model."""

    @classmethod
    def _requires_dyn(cls) -> list[type["DynamicsModel"]]:
        """Define minimum :class:`~bsk_rl.sim.dyn.DynamicsModel` for compatibility."""
        return []

    def __init__(
        self, satellite: "Satellite", fsw_rate: float, priority: int = 100, **kwargs
    ) -> None:
        """The abstract base flight software model.

        One FSWModel is instantiated for each satellite in the environment each time the
        environment is reset and new simulator is created.

        Args:
            satellite: Satellite modelled by this model
            fsw_rate: [s] Rate of FSW simulation.
            priority: Model priority.
            kwargs: Passed to task creation functions
        """
        self.satellite = satellite
        self.logger = self.satellite.logger.getChild(self.__class__.__name__)

        for required in self._requires_dyn():
            if not issubclass(satellite.dyn_type, required):
                raise TypeError(
                    f"{satellite.dyn_type} must be a subclass of {required} to "
                    + f"use FSW model of type {self.__class__}"
                )

        fsw_proc_name = "FSWProcess" + self.satellite.name
        self.fsw_proc = self.simulator.CreateNewProcess(fsw_proc_name, priority)
        self.fsw_rate = fsw_rate

        self.tasks = self._make_task_list()

        for task in self.tasks:
            task.create_task()

        for task in self.tasks:
            task._create_module_data()

        self._set_messages()

        for task in self.tasks:
            task._setup_fsw_objects(**kwargs)

        self.fsw_proc.disableAllTasks()

    @property
    def simulator(self) -> "Simulator":
        """Reference to the episode simulator."""
        return self.satellite.simulator

    @property
    def world(self) -> "WorldModel":
        """Reference to the episode world model."""
        return self.simulator.world

    @property
    def dynamics(self) -> "DynamicsModel":
        """Reference to the satellite dynamics model for the episode."""
        return self.satellite.dynamics

    def _make_task_list(self) -> list["Task"]:
        return []

    @abstractmethod  # pragma: no cover
    def _set_messages(self) -> None:
        """Message setup after task creation."""
        pass

    def is_alive(self, log_failure=False) -> bool:
        """Check if the FSW model has failed any aliveness requirements.

        Returns:
            ``True`` if the satellite FSW is still alive.
        """
        return check_aliveness_checkers(self, log_failure=log_failure)

    def __del__(self):
        """Log when FSW model is deleted."""
        self.logger.debug("Basilisk FSW deleted")


class Task(ABC):
    """Abstract class for defining FSW tasks."""

    name: str = AbstractClassProperty()

    def __init__(self, fsw: FSWModel, priority: int) -> None:
        """Template class for defining FSW processes.

        Each FSW process has a task associated with it, which handle certain housekeeping
        functions.

        Args:
            fsw: FSW model task contributes to
            priority: Task priority
        """
        self.fsw: FSWModel = proxy(fsw)
        self.priority = priority

    def create_task(self) -> None:
        """Add task to FSW with a unique name."""
        self.fsw.fsw_proc.addTask(
            self.fsw.simulator.CreateNewTask(
                self.name + self.fsw.satellite.name, mc.sec2nano(self.fsw.fsw_rate)
            ),
            taskPriority=self.priority,
        )

    @abstractmethod  # pragma: no cover
    def _create_module_data(self) -> None:
        """Create module data wrappers."""
        pass

    @abstractmethod  # pragma: no cover
    def _setup_fsw_objects(self, **kwargs) -> None:
        """Initialize model parameters with satellite arguments."""
        pass

    def _add_model_to_task(self, module, priority) -> None:
        """Add a model to this task.

        Args:
            module: Basilisk module
            priority: Model priority
        """
        self.fsw.simulator.AddModelToTask(
            self.name + self.fsw.satellite.name,
            module,
            ModelPriority=priority,
        )

    def reset_for_action(self) -> None:
        """Housekeeping for task when a new action is called.

        Disables task by default, can be overridden by subclasses.
        """
        self.fsw.simulator.disableTask(self.name + self.fsw.satellite.name)


class BasicFSWModel(FSWModel):
    """Basic FSW model with minimum necessary Basilisk components."""

    @classmethod
    def _requires_dyn(cls) -> list[type["DynamicsModel"]]:
        return [dyn.BasicDynamicsModel]

    def _make_task_list(self) -> list[Task]:
        return [
            self.SunPointTask(self),
            self.NadirPointTask(self),
            self.RWDesatTask(self),
            self.TrackingErrorTask(self),
            self.MRPControlTask(self),
        ]

    def _set_messages(self) -> None:
        self._set_config_msgs()
        self._set_gateway_msgs()

    def _set_config_msgs(self) -> None:
        self._set_vehicle_config_msg()
        self._set_thrusters_config_msg()
        self._set_rw_config_msg()

    def _set_vehicle_config_msg(self) -> None:
        """Set the vehicle configuration message."""
        # Use the same inertia in the FSW algorithm as in the simulation
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        vehicleConfigOut.ISCPntB_B = self.dynamics.I_mat
        self.vcConfigMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    def _set_thrusters_config_msg(self) -> None:
        """Import the thrusters configuration information."""
        self.thrusterConfigMsg = self.dynamics.thrFactory.getConfigMessage()

    def _set_rw_config_msg(self) -> None:
        """Configure RW pyramid exactly as it is in dynamics."""
        self.fswRwConfigMsg = self.dynamics.rwFactory.getConfigMessage()

    def _set_gateway_msgs(self) -> None:
        """Create C-wrapped gateway messages."""
        self.attRefMsg = messaging.AttRefMsg_C()
        self.attGuidMsg = messaging.AttGuidMsg_C()

        self._zero_gateway_msgs()

        # connect gateway FSW effector command msgs with the dynamics
        self.dynamics.rwStateEffector.rwMotorCmdInMsg.subscribeTo(
            self.rwMotorTorque.rwMotorTorqueOutMsg
        )
        self.dynamics.thrusterSet.cmdsInMsg.subscribeTo(
            self.thrDump.thrusterOnTimeOutMsg
        )

    def _zero_gateway_msgs(self) -> None:
        """Zero all the FSW gateway message payloads."""
        self.attRefMsg.write(messaging.AttRefMsgPayload())
        self.attGuidMsg.write(messaging.AttGuidMsgPayload())

    @action
    def action_drift(self) -> None:
        """Disable all tasks and do nothing."""
        self.simulator.disableTask(
            BasicFSWModel.MRPControlTask.name + self.satellite.name
        )

    class SunPointTask(Task):
        """Task to generate sun-pointing reference."""

        name = "sunPointTask"

        def __init__(self, fsw, priority=99) -> None:  # noqa: D107
            """Task to generate a sun-pointing reference."""
            super().__init__(fsw, priority)

        def _create_module_data(self) -> None:
            self.sunPoint = self.fsw.sunPoint = locationPointing.locationPointing()
            self.sunPoint.ModelTag = "sunPoint"

        def _setup_fsw_objects(self, **kwargs) -> None:
            """Configure the solar array sun-pointing task."""
            self.setup_sun_pointing(**kwargs)

        def setup_sun_pointing(self, nHat_B: Iterable[float], **kwargs) -> None:
            """Configure the solar array sun-pointing task.

            Args:
                nHat_B: Solar array normal vector.
                kwargs: Passed to other setup functions.
            """
            self.sunPoint.pHat_B = nHat_B
            self.sunPoint.scAttInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.sunPoint.scTransInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.sunPoint.celBodyInMsg.subscribeTo(
                self.fsw.world.ephemConverter.ephemOutMsgs[self.fsw.world.sun_index]
            )
            self.sunPoint.useBoresightRateDamping = 1
            messaging.AttGuidMsg_C_addAuthor(
                self.sunPoint.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(self.sunPoint, priority=1200)

    @action
    def action_charge(self) -> None:
        """Charge battery by pointing the solar panels at the sun."""
        self.sunPoint.Reset(self.simulator.sim_time_ns)
        self.simulator.enableTask(self.SunPointTask.name + self.satellite.name)

    class NadirPointTask(Task):
        """Task to generate nadir-pointing reference."""

        name = "nadirPointTask"

        def __init__(self, fsw, priority=98) -> None:  # noqa: D107
            """Task to generate nadir-pointing reference."""
            super().__init__(fsw, priority)

        def _create_module_data(self) -> None:
            self.hillPoint = self.fsw.hillPoint = hillPoint.hillPoint()
            self.hillPoint.ModelTag = "hillPoint"

        def _setup_fsw_objects(self, **kwargs) -> None:
            """Configure the nadir-pointing task.

            Args:
                kwargs: Passed to other setup functions.
            """
            self.hillPoint.transNavInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.hillPoint.celBodyInMsg.subscribeTo(
                self.fsw.world.ephemConverter.ephemOutMsgs[self.fsw.world.body_index]
            )
            messaging.AttRefMsg_C_addAuthor(
                self.hillPoint.attRefOutMsg, self.fsw.attRefMsg
            )

            self._add_model_to_task(self.hillPoint, priority=1199)

    class RWDesatTask(Task):
        """Task to desaturate reaction wheels."""

        name = "rwDesatTask"

        def __init__(self, fsw, priority=97) -> None:  # noqa: D107
            """Task to desaturate reaction wheels using thrusters."""
            super().__init__(fsw, priority)

        def _create_module_data(self) -> None:
            """Set up momentum dumping and thruster control."""
            # Momentum dumping configuration
            self.thrDesatControl = self.fsw.thrDesatControl = (
                thrMomentumManagement.thrMomentumManagement()
            )
            self.thrDesatControl.ModelTag = "thrMomentumManagement"

            self.thrDump = self.fsw.thrDump = thrMomentumDumping.thrMomentumDumping()
            self.thrDump.ModelTag = "thrDump"

            # Thruster force mapping configuration
            self.thrForceMapping = self.fsw.thrForceMapping = (
                thrForceMapping.thrForceMapping()
            )
            self.thrForceMapping.ModelTag = "thrForceMapping"

        def _setup_fsw_objects(self, **kwargs) -> None:
            """Set up thrusters and momentum dumping.

            Args:
                kwargs: Passed to other setup functions.
            """
            self.setup_thruster_mapping(**kwargs)
            self.setup_momentum_dumping(**kwargs)

        @default_args(controlAxes_B=[1, 0, 0, 0, 1, 0, 0, 0, 1], thrForceSign=+1)
        def setup_thruster_mapping(
            self, controlAxes_B: Iterable[float], thrForceSign: int, **kwargs
        ) -> None:
            """Configure the thruster mapping.

            Args:
                controlAxes_B: Control unit axes.
                thrForceSign: Flag indicating if pos (+1) or negative (-1) thruster
                    solutions are found.
                kwargs: Passed to other setup functions.
            """
            self.thrForceMapping.cmdTorqueInMsg.subscribeTo(
                self.thrDesatControl.deltaHOutMsg
            )
            self.thrForceMapping.thrConfigInMsg.subscribeTo(self.fsw.thrusterConfigMsg)
            self.thrForceMapping.vehConfigInMsg.subscribeTo(self.fsw.vcConfigMsg)
            self.thrForceMapping.controlAxes_B = controlAxes_B
            self.thrForceMapping.thrForceSign = thrForceSign
            self.thrForceMapping.angErrThresh = 3.15

            self._add_model_to_task(self.thrForceMapping, priority=1192)

        @default_args(
            hs_min=0.0,
            maxCounterValue=4,
            thrMinFireTime=0.02,
            desatAttitude="sun",
        )
        def setup_momentum_dumping(
            self,
            hs_min: float,
            maxCounterValue: int,
            thrMinFireTime: float,
            desatAttitude: Optional[str],
            **kwargs,
        ) -> None:
            """Configure the momentum dumping algorithm.

            Args:
                hs_min: [N*m*s] Minimum RW cluster momentum for dumping.
                maxCounterValue: Control periods between firing thrusters.
                thrMinFireTime: [s] Minimum thruster firing time.
                desatAttitude: Direction to point while desaturating:

                    * ``"sun"`` points panels at sun
                    * ``"nadir"`` points instrument nadir
                    * ``None`` disables attitude control.

                kwargs: Passed to other setup functions.
            """
            self.fsw.desatAttitude = desatAttitude
            self.thrDesatControl.hs_min = hs_min  # Nms
            self.thrDesatControl.rwSpeedsInMsg.subscribeTo(
                self.fsw.dynamics.rwStateEffector.rwSpeedOutMsg
            )
            self.thrDesatControl.rwConfigDataInMsg.subscribeTo(self.fsw.fswRwConfigMsg)

            self.thrDump.deltaHInMsg.subscribeTo(self.thrDesatControl.deltaHOutMsg)
            self.thrDump.thrusterImpulseInMsg.subscribeTo(
                self.thrForceMapping.thrForceCmdOutMsg
            )
            self.thrDump.thrusterConfInMsg.subscribeTo(self.fsw.thrusterConfigMsg)
            self.thrDump.maxCounterValue = maxCounterValue
            self.thrDump.thrMinFireTime = thrMinFireTime

            self._add_model_to_task(self.thrDesatControl, priority=1193)
            self._add_model_to_task(self.thrDump, priority=1191)

        def reset_for_action(self) -> None:
            """Disable power draw for thrusters when a new action is selected."""
            super().reset_for_action()
            self.fsw.dynamics.thrusterPowerSink.powerStatus = 0

    @action
    def action_desat(self) -> None:
        """Charge while desaturating reaction wheels.

        This action maneuvers the satellite into ``desatAttitude``, turns on the thruster
        power sink, and enables the desaturation tasks. This action typically needs to be
        called multiple times to fully desaturate the wheels.
        """
        self.trackingError.Reset(self.simulator.sim_time_ns)
        self.thrDesatControl.Reset(self.simulator.sim_time_ns)
        self.thrDump.Reset(self.simulator.sim_time_ns)
        self.dynamics.thrusterPowerSink.powerStatus = 1
        self.simulator.enableTask(self.RWDesatTask.name + self.satellite.name)
        if self.desatAttitude == "sun":
            self.sunPoint.Reset(self.simulator.sim_time_ns)
            self.simulator.enableTask(self.SunPointTask.name + self.satellite.name)
        elif self.desatAttitude == "nadir":
            self.hillPoint.Reset(self.simulator.sim_time_ns)
            self.simulator.enableTask(
                BasicFSWModel.NadirPointTask.name + self.satellite.name
            )
        elif self.desatAttitude is None:
            pass
        else:
            raise ValueError(f"{self.desatAttitude} not a valid desatAttitude")
        self.simulator.enableTask(self.TrackingErrorTask.name + self.satellite.name)

    class TrackingErrorTask(Task):
        """Task to convert an attitude reference to guidance."""

        name = "trackingErrTask"

        def __init__(self, fsw, priority=90) -> None:  # noqa: D107
            """Task to convert an attitude reference to guidance."""
            super().__init__(fsw, priority)

        def _create_module_data(self) -> None:
            self.trackingError = self.fsw.trackingError = (
                attTrackingError.attTrackingError()
            )
            self.trackingError.ModelTag = "trackingError"

        def _setup_fsw_objects(self, **kwargs) -> None:
            self.trackingError.attNavInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.trackingError.attRefInMsg.subscribeTo(self.fsw.attRefMsg)
            messaging.AttGuidMsg_C_addAuthor(
                self.trackingError.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(self.trackingError, priority=1197)

    class MRPControlTask(Task):
        """Task to control the satellite attitude using reaction wheels."""

        name = "mrpControlTask"

        def __init__(self, fsw, priority=80) -> None:  # noqa: D107
            """Task to control the satellite with reaction wheels."""
            super().__init__(fsw, priority)

        def _create_module_data(self) -> None:
            # Attitude controller configuration
            self.mrpFeedbackControl = self.fsw.mrpFeedbackControl = (
                mrpFeedback.mrpFeedback()
            )
            self.mrpFeedbackControl.ModelTag = "mrpFeedbackControl"

            # add module that maps the Lr control torque into the RW motor torques
            self.rwMotorTorque = self.fsw.rwMotorTorque = rwMotorTorque.rwMotorTorque()
            self.rwMotorTorque.ModelTag = "rwMotorTorque"

        def _setup_fsw_objects(self, **kwargs) -> None:
            self.setup_mrp_feedback_rwa(**kwargs)
            self.setup_rw_motor_torque(**kwargs)

        @default_args(K=7.0, Ki=-1, P=35.0)
        def setup_mrp_feedback_rwa(
            self, K: float, Ki: float, P: float, **kwargs
        ) -> None:
            """Set the MRP feedback control properties.

            Args:
                K: Proportional gain.
                Ki: Integral gain.
                P: Derivative gain.
                kwargs: Passed to other setup functions.
            """
            self.mrpFeedbackControl.guidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.mrpFeedbackControl.vehConfigInMsg.subscribeTo(self.fsw.vcConfigMsg)
            self.mrpFeedbackControl.K = K
            self.mrpFeedbackControl.Ki = Ki
            self.mrpFeedbackControl.P = P
            self.mrpFeedbackControl.integralLimit = (
                2.0 / self.mrpFeedbackControl.Ki * 0.1
            )

            self._add_model_to_task(self.mrpFeedbackControl, priority=1196)

        def setup_rw_motor_torque(
            self, controlAxes_B: Iterable[float], **kwargs
        ) -> None:
            """Set parameters for finding motor torque from the control law.

            Args:
                controlAxes_B: Control unit axes.
                kwargs: Passed to other setup functions.
            """
            self.rwMotorTorque.rwParamsInMsg.subscribeTo(self.fsw.fswRwConfigMsg)
            self.rwMotorTorque.vehControlInMsg.subscribeTo(
                self.mrpFeedbackControl.cmdTorqueOutMsg
            )
            self.rwMotorTorque.controlAxes_B = controlAxes_B

            self._add_model_to_task(self.rwMotorTorque, priority=1195)

        def reset_for_action(self) -> None:
            """MRP control is enabled by default for all tasks."""
            self.fsw.simulator.enableTask(self.name + self.fsw.satellite.name)


class ImagingFSWModel(BasicFSWModel):
    """Extend FSW with instrument pointing and triggering control."""

    @classmethod
    def _requires_dyn(cls) -> list[type["DynamicsModel"]]:
        return super()._requires_dyn() + [dyn.ImagingDynModel]

    def __init__(self, *args, **kwargs) -> None:
        """Adds instrument pointing and triggering control to FSW."""
        super().__init__(*args, **kwargs)

    @property
    def c_hat_P(self):
        """Instrument pointing direction in the planet frame."""
        c_hat_B = self.locPoint.pHat_B
        return np.matmul(self.dynamics.BP.T, c_hat_B)

    @property
    def c_hat_H(self):
        """Instrument pointing direction in the hill frame."""
        c_hat_B = self.locPoint.pHat_B
        HN = rv2HN(self.satellite.dynamics.r_BN_N, self.satellite.dynamics.v_BN_N)
        return HN @ self.satellite.dynamics.BN.T @ c_hat_B

    def _make_task_list(self) -> list[Task]:
        return super()._make_task_list() + [self.LocPointTask(self)]

    def _set_gateway_msgs(self) -> None:
        super()._set_gateway_msgs()
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(
            self.insControl.deviceCmdOutMsg
        )

    class LocPointTask(Task):
        """Task to point at targets and trigger the instrument."""

        name = "locPointTask"

        def __init__(self, fsw, priority=96) -> None:  # noqa: D107
            """Task to point the instrument at ground targets."""
            super().__init__(fsw, priority)

        def _create_module_data(self) -> None:
            # Location pointing configuration
            self.locPoint = self.fsw.locPoint = locationPointing.locationPointing()
            self.locPoint.ModelTag = "locPoint"

            # SimpleInstrumentController configuration
            self.insControl = self.fsw.insControl = (
                simpleInstrumentController.simpleInstrumentController()
            )
            self.insControl.ModelTag = "instrumentController"

        def _setup_fsw_objects(self, **kwargs) -> None:
            self.setup_location_pointing(**kwargs)
            self.setup_instrument_controller(**kwargs)
            self.show_sensor()

        @default_args(inst_pHat_B=[0, 0, 1])
        def setup_location_pointing(
            self, inst_pHat_B: Iterable[float], **kwargs
        ) -> None:
            """Set the Earth location pointing guidance module.

            Args:
                inst_pHat_B: Instrument pointing direction.
                kwargs: Passed to other setup functions.
            """
            self.locPoint.pHat_B = inst_pHat_B
            self.locPoint.scAttInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.locPoint.scTransInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.locPoint.locationInMsg.subscribeTo(
                self.fsw.dynamics.imagingTarget.currentGroundStateOutMsg
            )
            self.locPoint.useBoresightRateDamping = 1
            messaging.AttGuidMsg_C_addAuthor(
                self.locPoint.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(self.locPoint, priority=1198)

        @default_args(imageAttErrorRequirement=0.01, imageRateErrorRequirement=None)
        def setup_instrument_controller(
            self,
            imageAttErrorRequirement: float,
            imageRateErrorRequirement: float,
            **kwargs,
        ) -> None:
            """Set the instrument controller parameters.

            The instrument controller is used to take an image when certain relative
            attitude requirements are met, along with the access requirements of the
            target (i.e. ``imageTargetMinimumElevation`` and ``imageTargetMaximumRange``
            as set in :class:`~bsk_rl.sim.dyn.ImagingDynModel.setup_imaging_target`).

            Args:
                imageAttErrorRequirement: [MRP norm] Pointing attitude error tolerance
                    for imaging.
                imageRateErrorRequirement: [rad/s] Rate tolerance for imaging. Disable
                    with ``None``.
                kwargs: Passed to other setup functions.
            """
            self.insControl.attErrTolerance = imageAttErrorRequirement
            if imageRateErrorRequirement is not None:
                self.insControl.useRateTolerance = 1
                self.insControl.rateErrTolerance = imageRateErrorRequirement
            self.insControl.attGuidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.insControl.locationAccessInMsg.subscribeTo(
                self.fsw.dynamics.imagingTarget.accessOutMsgs[-1]
            )

            self._add_model_to_task(self.insControl, priority=987)

        @vizard.visualize
        def show_sensor(self, vizInterface=None, vizSupport=None):
            """Visualize the sensor in Vizard."""
            genericSensor = vizInterface.GenericSensor()
            genericSensor.normalVector = self.locPoint.pHat_B
            genericSensor.r_SB_B = [0.0, 0.0, 0.0]
            genericSensor.fieldOfView.push_back(4 * self.insControl.attErrTolerance)
            genericSensor.color = vizInterface.IntVector(
                vizSupport.toRGBA255(self.fsw.satellite.vizard_color, alpha=0.5)
            )
            cmdInMsg = messaging.DeviceCmdMsgReader()
            cmdInMsg.subscribeTo(self.insControl.deviceCmdOutMsg)
            genericSensor.genericSensorCmdInMsg = cmdInMsg
            self.fsw.satellite.vizard_data["genericSensorList"] = [genericSensor]

        def reset_for_action(self) -> None:
            """Reset pointing controller."""
            self.fsw.dynamics.imagingTarget.Reset(self.fsw.simulator.sim_time_ns)
            self.locPoint.Reset(self.fsw.simulator.sim_time_ns)
            self.insControl.controllerStatus = 0
            return super().reset_for_action()

    @action
    def action_image(self, r_LP_P: Iterable[float], data_name: str) -> None:
        """Attempt to image a target at a location.

        This action sets the target attitude to one tracking a ground location. If the
        target is within the imaging constraints, an image will be taken and stored in
        the data buffer. The instrument power sink will be active as long as the task is
        enabled.

        Args:
            r_LP_P: [m] Planet-fixed planet relative target location.
            data_name: Data buffer to store image data to.
        """
        self.insControl.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.imagingTarget.r_LP_P_Init = r_LP_P
        self.dynamics.instrument.nodeDataName = data_name
        self.insControl.imaged = 0
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.name)

    @action
    def action_downlink(self) -> None:
        """Attempt to downlink data.

        This action points the satellite nadir and attempts to downlink data. If the
        satellite is in range of a ground station, data will be downlinked at the specified
        baud rate. The transmitter power sink will be active as long as the task is enabled.
        """
        self.hillPoint.Reset(self.simulator.sim_time_ns)
        self.trackingError.Reset(self.simulator.sim_time_ns)
        self.dynamics.transmitter.dataStatus = 1
        self.dynamics.transmitterPowerSink.powerStatus = 1
        self.simulator.enableTask(
            BasicFSWModel.NadirPointTask.name + self.satellite.name
        )
        self.simulator.enableTask(
            BasicFSWModel.TrackingErrorTask.name + self.satellite.name
        )


class ContinuousImagingFSWModel(ImagingFSWModel):
    """FSW model for continuous nadir scanning."""

    def __init__(self, *args, **kwargs) -> None:
        """FSW model for continuous nadir scanning.

        Instead of imaging point targets, this model is used to continuously scan the
        ground while pointing nadir.
        """
        super().__init__(*args, **kwargs)

    class LocPointTask(ImagingFSWModel.LocPointTask):
        """Task to point nadir and trigger the instrument."""

        def __init__(self, *args, **kwargs) -> None:
            """Task to point nadir and trigger the instrument."""
            super().__init__(*args, **kwargs)

        def _create_module_data(self) -> None:
            # Location pointing configuration
            self.locPoint = self.fsw.locPoint = locationPointing.locationPointing()
            self.locPoint.ModelTag = "locPoint"

            # scanningInstrumentController configuration
            self.insControl = self.fsw.insControl = (
                scanningInstrumentController.scanningInstrumentController()
            )
            self.insControl.ModelTag = "instrumentController"

        @default_args(imageAttErrorRequirement=0.01, imageRateErrorRequirement=None)
        def setup_instrument_controller(
            self,
            imageAttErrorRequirement: float,
            imageRateErrorRequirement: float,
            **kwargs,
        ) -> None:
            """Set the instrument controller parameters for scanning.

            As long as these two conditions are met, scanning will occur continuously.

            Args:
                imageAttErrorRequirement: [MRP norm] Pointing attitude error tolerance
                    for imaging.
                imageRateErrorRequirement: [rad/s] Rate tolerance for imaging. Disable
                    with None.
                kwargs: Passed to other setup functions.
            """
            self.insControl.attErrTolerance = imageAttErrorRequirement
            if imageRateErrorRequirement is not None:
                self.insControl.useRateTolerance = 1
                self.insControl.rateErrTolerance = imageRateErrorRequirement
            self.insControl.attGuidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.insControl.accessInMsg.subscribeTo(
                self.fsw.dynamics.imagingTarget.accessOutMsgs[-1]
            )

            self._add_model_to_task(self.insControl, priority=987)

        def reset_for_action(self) -> None:
            """Reset scanning controller."""
            self.instMsg = messaging.DeviceCmdMsg_C()
            self.instMsg.write(messaging.DeviceCmdMsgPayload())
            self.fsw.dynamics.instrument.nodeStatusInMsg.subscribeTo(self.instMsg)
            return super().reset_for_action()

    @action
    def action_nadir_scan(self) -> None:
        """Scan nadir.

        This action points the instrument nadir and continuously adds data to the buffer
        as long as attitude requirements are met. The instrument power sink is active
        as long as the action is set.
        """
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(
            self.insControl.deviceCmdOutMsg
        )
        self.insControl.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.imagingTarget.r_LP_P_Init = np.array(
            [0, 0, 0.1]
        )  # All zero causes an error
        self.dynamics.instrument.nodeDataName = "nadir"
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.name)

    @action
    def action_image(self, *args, **kwargs) -> None:
        """Disable ``action_image`` from parent class.

        :meta private:
        """
        raise NotImplementedError("Use action_nadir_scan instead")


class SteeringFSWModel(BasicFSWModel):
    """FSW extending MRP control to use MRP steering instead of MRP feedback."""

    def __init__(self, *args, **kwargs) -> None:
        """FSW extending attitude control to use MRP steering instead of MRP feedback.

        This class replaces the simple attitude feedback control law with a more
        sophisticated `MRP steering control law <https://hanspeterschaub.info/basilisk/Documentation/fswAlgorithms/attControl/mrpSteering/mrpSteering.html>`_.
        """
        super().__init__(*args, **kwargs)

    class MRPControlTask(Task):
        """Task that uses MRP steering to control reaction wheels."""

        name = "mrpControlTask"

        def __init__(self, fsw, priority=80) -> None:  # noqa: D107
            """Task to control the satellite with reaction wheels."""
            super().__init__(fsw, priority)

        def _create_module_data(self) -> None:
            # Attitude controller configuration
            self.mrpSteeringControl = self.fsw.mrpSteeringControl = (
                mrpSteering.mrpSteering()
            )
            self.mrpSteeringControl.ModelTag = "mrpSteeringControl"

            # Rate servo
            self.servo = self.fsw.servo = (
                rateServoFullNonlinear.rateServoFullNonlinear()
            )
            self.servo.ModelTag = "rateServo"

            # add module that maps the Lr control torque into the RW motor torques
            self.rwMotorTorque = self.fsw.rwMotorTorque = rwMotorTorque.rwMotorTorque()
            self.rwMotorTorque.ModelTag = "rwMotorTorque"

        def _setup_fsw_objects(self, **kwargs) -> None:
            self.setup_mrp_steering_rwa(**kwargs)
            self.setup_rw_motor_torque(**kwargs)

        @default_args(
            K1=0.25, K3=3.0, omega_max=np.radians(3), servo_Ki=5.0, servo_P=150
        )
        def setup_mrp_steering_rwa(
            self,
            K1: float,
            K3: float,
            omega_max: float,
            servo_Ki: float,
            servo_P: float,
            **kwargs,
        ) -> None:
            """Define the control properties.

            Args:
                K1: MRP steering gain.
                K3: MRP steering gain.
                omega_max: [rad/s] Maximum targetable spacecraft body rate.
                servo_Ki: Servo gain.
                servo_P: Servo gain.
                kwargs: Passed to other setup functions.
            """
            self.mrpSteeringControl.guidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.mrpSteeringControl.K1 = K1
            self.mrpSteeringControl.K3 = K3
            self.mrpSteeringControl.omega_max = omega_max
            self.mrpSteeringControl.ignoreOuterLoopFeedforward = False

            self.servo.Ki = servo_Ki
            self.servo.P = servo_P
            self.servo.integralLimit = 2.0 / self.servo.Ki * 0.1
            self.servo.knownTorquePntB_B = [0.0, 0.0, 0.0]
            self.servo.guidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.servo.vehConfigInMsg.subscribeTo(self.fsw.vcConfigMsg)
            self.servo.rwParamsInMsg.subscribeTo(self.fsw.fswRwConfigMsg)
            self.servo.rwSpeedsInMsg.subscribeTo(
                self.fsw.dynamics.rwStateEffector.rwSpeedOutMsg
            )
            self.servo.rateSteeringInMsg.subscribeTo(
                self.mrpSteeringControl.rateCmdOutMsg
            )

            self._add_model_to_task(self.mrpSteeringControl, priority=1196)
            self._add_model_to_task(self.servo, priority=1195)

        def setup_rw_motor_torque(
            self, controlAxes_B: Iterable[float], **kwargs
        ) -> None:
            """Define the motor torque from the control law.

            Args:
                controlAxes_B: Control unit axes.
                kwargs: Passed to other setup functions.
            """
            self.rwMotorTorque.rwParamsInMsg.subscribeTo(self.fsw.fswRwConfigMsg)
            self.rwMotorTorque.vehControlInMsg.subscribeTo(self.servo.cmdTorqueOutMsg)
            self.rwMotorTorque.controlAxes_B = controlAxes_B

            self._add_model_to_task(self.rwMotorTorque, priority=1194)

        def reset_for_action(self) -> None:
            """Keep MRP control enabled on action calls."""
            self.fsw.simulator.enableTask(self.name + self.fsw.satellite.name)


class SteeringImagerFSWModel(SteeringFSWModel, ImagingFSWModel):
    """Convenience type for ImagingFSWModel with MRP steering."""

    def __init__(self, *args, **kwargs) -> None:
        """Convenience type that combines the imaging FSW model with MRP steering."""
        super().__init__(*args, **kwargs)


__doc_title__ = "FSW Sims"
__all__ = [
    "action",
    "BasicFSWModel",
    "ImagingFSWModel",
    "ContinuousImagingFSWModel",
    "SteeringFSWModel",
    "SteeringImagerFSWModel",
]
