from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Iterable
from weakref import proxy

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.envs.general_satellite_tasking.types import (
        DynamicsModel,
        EnvironmentModel,
        Satellite,
        Simulator,
    )

import Basilisk.architecture.cMsgCInterfacePy as cMsgPy
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

from bsk_rl.envs.general_satellite_tasking.simulation import dynamics
from bsk_rl.envs.general_satellite_tasking.utils.debug import MEMORY_LEAK_CHECKING
from bsk_rl.envs.general_satellite_tasking.utils.functional import (
    check_aliveness_checkers,
    default_args,
)


def action(
    func: Callable[..., None]
) -> Callable[Callable[..., None], Callable[..., None]]:
    """Wrapper to do housekeeping for action functions that should be called by the
    satellite class."""

    def inner(self, *args, **kwargs) -> Callable[..., None]:
        self.fsw_proc.disableAllTasks()
        self._zero_gateway_msgs()
        self.dynamics.reset_for_action()
        for task in self.tasks:
            task.reset_for_action()
        return func(self, *args, **kwargs)

    return inner


class FSWModel(ABC):
    @classmethod
    @property
    def requires_dyn(cls) -> list[type["DynamicsModel"]]:
        """Define minimum DynamicsModels for compatibility."""
        return []

    def __init__(
        self, satellite: "Satellite", fsw_rate: float, priority: int = 100, **kwargs
    ) -> None:
        """Base FSWModel

        Args:
            satellite: Satellite modelled by this model
            fsw_rate: Rate of FSW simulation [s]
            priority: Model priority.
        """

        self.satellite = satellite

        for required in self.requires_dyn:
            if not issubclass(satellite.dyn_type, required):
                raise TypeError(
                    f"{satellite.dyn_type} must be a subclass of {required} to "
                    + f"use FSW model of type {self.__class__}"
                )

        fsw_proc_name = "FSWProcess" + self.satellite.id
        self.fsw_proc = self.simulator.CreateNewProcess(fsw_proc_name, priority)
        self.fsw_rate = fsw_rate

        self.tasks = self._make_task_list()

        for task in self.tasks:
            task.create_task()

        for task in self.tasks:
            task.create_module_data()

        self._set_messages()

        for task in self.tasks:
            task.init_objects(**kwargs)

        self.fsw_proc.disableAllTasks()

    @property
    def simulator(self) -> "Simulator":
        return self.satellite.simulator

    @property
    def environment(self) -> "EnvironmentModel":
        return self.simulator.environment

    @property
    def dynamics(self) -> "DynamicsModel":
        return self.satellite.dynamics

    def _make_task_list(self) -> list["Task"]:
        return []

    @abstractmethod  # pragma: no cover
    def _set_messages(self) -> None:
        """Message setup after task creation"""
        pass

    def is_alive(self) -> bool:
        """Check if the fsw model has failed any aliveness requirements.

        Returns:
            If the satellite fsw is still alive
        """
        return check_aliveness_checkers(self)

    def __del__(self):
        if MEMORY_LEAK_CHECKING:  # pragma: no cover
            print("~~~ BSK FSW DELETED ~~~")


class Task(ABC):
    @property
    @abstractmethod  # pragma: no cover
    def name(self) -> str:
        pass

    def __init__(self, fsw: FSWModel, priority: int) -> None:
        """Template class for defining FSW processes

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
                self.name + self.fsw.satellite.id, mc.sec2nano(self.fsw.fsw_rate)
            ),
            taskPriority=self.priority,
        )

    @abstractmethod  # pragma: no cover
    def create_module_data(self) -> None:
        """Create module data wrappers."""
        pass

    @abstractmethod  # pragma: no cover
    def init_objects(self, **kwargs) -> None:
        """Initialize model parameters with satellite arguments."""
        pass

    def _add_model_to_task(self, module, priority) -> None:
        """Add a model to this task.

        Args:
            module: Basilisk module
            priority: Model priority
        """
        self.fsw.simulator.AddModelToTask(
            self.name + self.fsw.satellite.id,
            module,
            ModelPriority=priority,
        )

    def reset_for_action(self) -> None:
        """Housekeeping for task when a new action is called; by default, disable
        task."""
        self.fsw.simulator.disableTask(self.name + self.fsw.satellite.id)


class BasicFSWModel(FSWModel):
    @classmethod
    @property
    def requires_dyn(cls) -> list[type["DynamicsModel"]]:
        return [dynamics.BasicDynamicsModel]

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
        """Configure RW pyramid exactly as it is in the Dynamics (i.e. FSW with perfect
        knowledge)."""
        self.fswRwConfigMsg = self.dynamics.rwFactory.getConfigMessage()

    def _set_gateway_msgs(self) -> None:
        """Create C-wrapped gateway messages such that different modules can write to
        this message and provide a common input msg for down-stream modules."""
        self.attRefMsg = cMsgPy.AttRefMsg_C()
        self.attGuidMsg = cMsgPy.AttGuidMsg_C()

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
        """Action to disable all tasks."""
        self.simulator.disableTask(
            BasicFSWModel.MRPControlTask.name + self.satellite.id
        )

    class SunPointTask(Task):
        """Task to generate sun-pointing reference."""

        name = "sunPointTask"

        def __init__(self, fsw, priority=99) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            self.sunPoint = self.fsw.sunPoint = locationPointing.locationPointing()
            self.sunPoint.ModelTag = "sunPoint"

        def init_objects(self, nHat_B: Iterable[float], **kwargs) -> None:
            """Configure the sun-pointing task.

            Args:
                nHat_B: Solar array normal vector
            """
            self.sunPoint.pHat_B = nHat_B
            self.sunPoint.scAttInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.sunPoint.scTransInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.sunPoint.celBodyInMsg.subscribeTo(
                self.fsw.environment.ephemConverter.ephemOutMsgs[
                    self.fsw.environment.sun_index
                ]
            )
            self.sunPoint.useBoresightRateDamping = 1
            cMsgPy.AttGuidMsg_C_addAuthor(
                self.sunPoint.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(self.sunPoint, priority=1200)

    @action
    def action_charge(self) -> None:
        """Action to charge solar panels."""
        self.sunPoint.Reset(self.simulator.sim_time_ns)
        self.simulator.enableTask(self.SunPointTask.name + self.satellite.id)

    class NadirPointTask(Task):
        """Task to generate nadir-pointing reference."""

        name = "nadirPointTask"

        def __init__(self, fsw, priority=98) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            self.hillPoint = self.fsw.hillPoint = hillPoint.hillPoint()
            self.hillPoint.ModelTag = "hillPoint"

        def init_objects(self, **kwargs) -> None:
            """Configure the nadir-pointing task."""
            self.hillPoint.transNavInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.hillPoint.celBodyInMsg.subscribeTo(
                self.fsw.environment.ephemConverter.ephemOutMsgs[
                    self.fsw.environment.body_index
                ]
            )
            cMsgPy.AttRefMsg_C_addAuthor(
                self.hillPoint.attRefOutMsg, self.fsw.attRefMsg
            )

            self._add_model_to_task(self.hillPoint, priority=1199)

    class RWDesatTask(Task):
        """Task to desaturate reaction wheels."""

        name = "rwDesatTask"

        def __init__(self, fsw, priority=97) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            """Set up momentum dumping and thruster control."""
            # Momentum dumping configuration
            self.thrDesatControl = (
                self.fsw.thrDesatControl
            ) = thrMomentumManagement.thrMomentumManagement()
            self.thrDesatControl.ModelTag = "thrMomentumManagement"

            self.thrDump = self.fsw.thrDump = thrMomentumDumping.thrMomentumDumping()
            self.thrDump.ModelTag = "thrDump"

            # Thruster force mapping configuration
            self.thrForceMapping = (
                self.fsw.thrForceMapping
            ) = thrForceMapping.thrForceMapping()
            self.thrForceMapping.ModelTag = "thrForceMapping"

        def init_objects(self, **kwargs) -> None:
            self._set_thruster_mapping(**kwargs)
            self._set_momentum_dumping(**kwargs)

        @default_args(controlAxes_B=[1, 0, 0, 0, 1, 0, 0, 0, 1], thrForceSign=+1)
        def _set_thruster_mapping(
            self, controlAxes_B: Iterable[float], thrForceSign: int, **kwargs
        ) -> None:
            """Configure the thruster mapping.

            Args:
                controlAxes_B: Control unit axes
                thrForceSign: Flag indicating if pos (+1) or negative (-1) thruster
                    solutions are found
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

        @default_args(hs_min=0.0, maxCounterValue=4, thrMinFireTime=0.02)
        def _set_momentum_dumping(
            self, hs_min: float, maxCounterValue: int, thrMinFireTime: float, **kwargs
        ) -> None:
            """Configure the momentum dumping algorithm.

            Args:
                hs_min: minimum RW cluster momentum for dumping [N*m*s]
                maxCounterValue: Control periods between firing thrusters
                thrMinFireTime: Minimum thruster firing time [s]
            """
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
            super().reset_for_action()
            self.fsw.dynamics.thrusterPowerSink.powerStatus = 0

    @action
    def action_desat(self) -> None:
        """Action to charge while desaturating reaction wheels."""
        self.sunPoint.Reset(self.simulator.sim_time_ns)
        self.trackingError.Reset(self.simulator.sim_time_ns)
        self.thrDesatControl.Reset(self.simulator.sim_time_ns)
        self.thrDump.Reset(self.simulator.sim_time_ns)
        self.dynamics.thrusterPowerSink.powerStatus = 1
        self.simulator.enableTask(self.SunPointTask.name + self.satellite.id)
        self.simulator.enableTask(self.RWDesatTask.name + self.satellite.id)
        self.simulator.enableTask(self.TrackingErrorTask.name + self.satellite.id)

    class TrackingErrorTask(Task):
        """Task to convert an attitude reference to guidance."""

        name = "trackingErrTask"

        def __init__(self, fsw, priority=90) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            self.trackingError = (
                self.fsw.trackingError
            ) = attTrackingError.attTrackingError()
            self.trackingError.ModelTag = "trackingError"

        def init_objects(self, **kwargs) -> None:
            self.trackingError.attNavInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.trackingError.attRefInMsg.subscribeTo(self.fsw.attRefMsg)
            cMsgPy.AttGuidMsg_C_addAuthor(
                self.trackingError.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(self.trackingError, priority=1197)

    class MRPControlTask(Task):
        """Task to control the satellite with reaction wheels."""

        name = "mrpControlTask"

        def __init__(self, fsw, priority=80) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            # Attitude controller configuration
            self.mrpFeedbackControl = (
                self.fsw.mrpFeedbackControl
            ) = mrpFeedback.mrpFeedback()
            self.mrpFeedbackControl.ModelTag = "mrpFeedbackControl"

            # add module that maps the Lr control torque into the RW motor torques
            self.rwMotorTorque = self.fsw.rwMotorTorque = rwMotorTorque.rwMotorTorque()
            self.rwMotorTorque.ModelTag = "rwMotorTorque"

        def init_objects(self, **kwargs) -> None:
            self._set_mrp_feedback_rwa(**kwargs)
            self._set_rw_motor_torque(**kwargs)

        @default_args(K=7.0, Ki=-1, P=35.0)
        def _set_mrp_feedback_rwa(
            self, K: float, Ki: float, P: float, **kwargs
        ) -> None:
            """Defines the control properties.

            Args:
                K: Proportional gain
                Ki: Integral gain
                P: Derivative gain
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

        def _set_rw_motor_torque(
            self, controlAxes_B: Iterable[float], **kwargs
        ) -> None:
            """Defines the motor torque from the control law.

            Args:
                controlAxes_B): Control unit axes
            """
            self.rwMotorTorque.rwParamsInMsg.subscribeTo(self.fsw.fswRwConfigMsg)
            self.rwMotorTorque.vehControlInMsg.subscribeTo(
                self.mrpFeedbackControl.cmdTorqueOutMsg
            )
            self.rwMotorTorque.controlAxes_B = controlAxes_B

            self._add_model_to_task(self.rwMotorTorque, priority=1195)

        def reset_for_action(self) -> None:
            """MRP control enabled by default for all tasks."""
            self.fsw.simulator.enableTask(self.name + self.fsw.satellite.id)


class ImagingFSWModel(BasicFSWModel):
    """Extend FSW with instrument pointing and triggering control"""

    @classmethod
    @property
    def requires_dyn(cls) -> list[type["DynamicsModel"]]:
        return super().requires_dyn + [dynamics.ImagingDynModel]

    @property
    def c_hat_P(self):
        c_hat_B = self.locPoint.pHat_B
        return np.matmul(self.dynamics.BP.T, c_hat_B)

    def _make_task_list(self) -> list[Task]:
        return super()._make_task_list() + [self.LocPointTask(self)]

    def _set_gateway_msgs(self) -> None:
        super()._set_gateway_msgs()
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(
            self.insControl.deviceCmdOutMsg
        )

    class LocPointTask(Task):
        """Task to point at targets and trigger the instrument"""

        name = "locPointTask"

        def __init__(self, fsw, priority=96) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            # Location pointing configuration
            self.locPoint = self.fsw.locPoint = locationPointing.locationPointing()
            self.locPoint.ModelTag = "locPoint"

            # SimpleInstrumentController configuration
            self.insControl = (
                self.fsw.insControl
            ) = simpleInstrumentController.simpleInstrumentController()
            self.insControl.ModelTag = "instrumentController"

        def init_objects(self, **kwargs) -> None:
            self._set_location_pointing(**kwargs)
            self._set_instrument_controller(**kwargs)

        @default_args(inst_pHat_B=[0, 0, 1])
        def _set_location_pointing(
            self, inst_pHat_B: Iterable[float], **kwargs
        ) -> None:
            """Defines the Earth location pointing guidance module.

            Args:
                inst_pHat_B: Instrument pointing direction
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
            cMsgPy.AttGuidMsg_C_addAuthor(
                self.locPoint.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(self.locPoint, priority=1198)

        @default_args(imageAttErrorRequirement=0.01, imageRateErrorRequirement=None)
        def _set_instrument_controller(
            self,
            imageAttErrorRequirement: float,
            imageRateErrorRequirement: float,
            **kwargs,
        ) -> None:
            """Defines the instrument controller parameters.

            Args:
                imageAttErrorRequirement: Pointing attitude error tolerance for imaging
                    [MRP norm]
                imageRateErrorRequirement: Rate tolerance for imaging. Disable with
                    None. [rad/s]
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

        def reset_for_action(self) -> None:
            self.fsw.dynamics.imagingTarget.Reset(self.fsw.simulator.sim_time_ns)
            self.locPoint.Reset(self.fsw.simulator.sim_time_ns)
            self.insControl.controllerStatus = 0
            return super().reset_for_action()

    @action
    def action_image(self, location: Iterable[float], data_name: str) -> None:
        """Action to image a target at a location.

        Args:
            location: PCPF target location [m]
            data_name: Data buffer to store image data to
        """
        self.insControl.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.imagingTarget.r_LP_P_Init = location
        self.dynamics.instrument.nodeDataName = data_name
        self.insControl.imaged = 0
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.id)

    @action
    def action_downlink(self) -> None:
        """Action to attempt to downlink data."""
        self.hillPoint.Reset(self.simulator.sim_time_ns)
        self.trackingError.Reset(self.simulator.sim_time_ns)
        self.dynamics.transmitter.dataStatus = 1
        self.dynamics.transmitterPowerSink.powerStatus = 1
        self.simulator.enableTask(BasicFSWModel.NadirPointTask.name + self.satellite.id)
        self.simulator.enableTask(
            BasicFSWModel.TrackingErrorTask.name + self.satellite.id
        )


class ContinuousImagingFSWModel(ImagingFSWModel):
    class LocPointTask(ImagingFSWModel.LocPointTask):
        """Task to point at targets and trigger the instrument"""

        def create_module_data(self) -> None:
            # Location pointing configuration
            self.locPoint = self.fsw.locPoint = locationPointing.locationPointing()
            self.locPoint.ModelTag = "locPoint"

            # scanningInstrumentController configuration
            self.insControl = (
                self.fsw.insControl
            ) = scanningInstrumentController.scanningInstrumentController()
            self.insControl.ModelTag = "instrumentController"

        @default_args(imageAttErrorRequirement=0.01, imageRateErrorRequirement=None)
        def _set_instrument_controller(
            self,
            imageAttErrorRequirement: float,
            imageRateErrorRequirement: float,
            **kwargs,
        ) -> None:
            """Defines the instrument controller parameters.

            Args:
                imageAttErrorRequirement: Pointing attitude error tolerance for imaging
                    [MRP norm]
                imageRateErrorRequirement: Rate tolerance for imaging. Disable with
                    None. [rad/s]
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
            self.instMsg = cMsgPy.DeviceCmdMsg_C()
            self.instMsg.write(messaging.DeviceCmdMsgPayload())
            self.fsw.dynamics.instrument.nodeStatusInMsg.subscribeTo(self.instMsg)
            return super().reset_for_action()

    @action
    def action_nadir_scan(self) -> None:
        """Action scan nadir.

        Args:
            location: PCPF target location [m]
            data_name: Data buffer to store image data to
        """
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(
            self.insControl.deviceCmdOutMsg
        )
        self.insControl.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.imagingTarget.r_LP_P_Init = np.array([0, 0, 0.1])
        self.dynamics.instrument.nodeDataName = "nadir"
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.id)

    @action
    def action_image(self, *args, **kwargs) -> None:
        raise NotImplementedError("Use action_nadir_scan instead")


class SteeringFSWModel(BasicFSWModel):
    """FSW extending MRP control to use MRP steering instesd of MRP feedback."""

    class MRPControlTask(Task):
        name = "mrpControlTask"

        def __init__(self, fsw, priority=80) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            # Attitude controller configuration
            self.mrpSteeringControl = (
                self.fsw.mrpSteeringControl
            ) = mrpSteering.mrpSteering()
            self.mrpSteeringControl.ModelTag = "mrpSteeringControl"

            # Rate servo
            self.servo = (
                self.fsw.servo
            ) = rateServoFullNonlinear.rateServoFullNonlinear()
            self.servo.ModelTag = "rateServo"

            # add module that maps the Lr control torque into the RW motor torques
            self.rwMotorTorque = self.fsw.rwMotorTorque = rwMotorTorque.rwMotorTorque()
            self.rwMotorTorque.ModelTag = "rwMotorTorque"

        def init_objects(self, **kwargs) -> None:
            self._set_mrp_steering_rwa(**kwargs)
            self._set_rw_motor_torque(**kwargs)

        @default_args(K1=0.25, K3=3.0, omega_max=3 * mc.D2R, servo_Ki=5.0, servo_P=150)
        def _set_mrp_steering_rwa(
            self,
            K1: float,
            K3: float,
            omega_max: float,
            servo_Ki: float,
            servo_P: float,
            **kwargs,
        ) -> None:
            """Defines the control properties.

            Args:
                K1: MRP steering gain
                K3: MRP steering gain
                omega_max: Maximum targetable spacecraft body rate [rad/s]
                servo_Ki: Servo gain
                servo_P: Servo gain
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

        def _set_rw_motor_torque(
            self, controlAxes_B: Iterable[float], **kwargs
        ) -> None:
            """Defines the motor torque from the control law.

            Args:
                controlAxes_B: Control unit axes
            """
            self.rwMotorTorque.rwParamsInMsg.subscribeTo(self.fsw.fswRwConfigMsg)
            self.rwMotorTorque.vehControlInMsg.subscribeTo(self.servo.cmdTorqueOutMsg)
            self.rwMotorTorque.controlAxes_B = controlAxes_B

            self._add_model_to_task(self.rwMotorTorque, priority=1194)

        def reset_for_action(self) -> None:
            # MRP control enabled by default
            self.fsw.simulator.enableTask(self.name + self.fsw.satellite.id)


class SteeringImagerFSWModel(SteeringFSWModel, ImagingFSWModel):
    pass
