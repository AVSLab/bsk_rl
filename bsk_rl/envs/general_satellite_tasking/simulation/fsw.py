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

    def _add_model_to_task(self, wrap, data, priority: int) -> None:
        """Add a model to this task.

        Args:
            wrap: Basilisk module wrapper
            data: Basilisk module data
            priority: Model priority
        """
        self.fsw.simulator.AddModelToTask(
            self.name + self.fsw.satellite.id,
            wrap,
            data,
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
            self.rwMotorTorqueConfig.rwMotorTorqueOutMsg
        )
        self.dynamics.thrusterSet.cmdsInMsg.subscribeTo(
            self.thrDumpConfig.thrusterOnTimeOutMsg
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
            self.sunPointData = (
                self.fsw.sunPointData
            ) = locationPointing.locationPointingConfig()
            self.sunPointWrap = (
                self.fsw.sunPointWrap
            ) = self.fsw.simulator.setModelDataWrap(self.sunPointData)
            self.sunPointWrap.ModelTag = "sunPoint"

        def init_objects(self, nHat_B: Iterable[float], **kwargs) -> None:
            """Configure the sun-pointing task.

            Args:
                nHat_B: Solar array normal vector
            """
            self.sunPointData.pHat_B = nHat_B
            self.sunPointData.scAttInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.sunPointData.scTransInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.sunPointData.celBodyInMsg.subscribeTo(
                self.fsw.environment.ephemConverter.ephemOutMsgs[
                    self.fsw.environment.sun_index
                ]
            )
            self.sunPointData.useBoresightRateDamping = 1
            cMsgPy.AttGuidMsg_C_addAuthor(
                self.sunPointData.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(self.sunPointWrap, self.sunPointData, priority=1200)

    @action
    def action_charge(self) -> None:
        """Action to charge solar panels."""
        self.sunPointWrap.Reset(self.simulator.sim_time_ns)
        self.simulator.enableTask(self.SunPointTask.name + self.satellite.id)

    class NadirPointTask(Task):
        """Task to generate nadir-pointing reference."""

        name = "nadirPointTask"

        def __init__(self, fsw, priority=98) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            self.hillPointData = self.fsw.hillPointData = hillPoint.hillPointConfig()
            self.hillPointWrap = (
                self.fsw.hillPointWrap
            ) = self.fsw.simulator.setModelDataWrap(self.hillPointData)
            self.hillPointWrap.ModelTag = "hillPoint"

        def init_objects(self, **kwargs) -> None:
            """Configure the nadir-pointing task."""
            self.hillPointData.transNavInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.hillPointData.celBodyInMsg.subscribeTo(
                self.fsw.environment.ephemConverter.ephemOutMsgs[
                    self.fsw.environment.body_index
                ]
            )
            cMsgPy.AttRefMsg_C_addAuthor(
                self.hillPointData.attRefOutMsg, self.fsw.attRefMsg
            )

            self._add_model_to_task(
                self.hillPointWrap, self.hillPointData, priority=1199
            )

    class RWDesatTask(Task):
        """Task to desaturate reaction wheels."""

        name = "rwDesatTask"

        def __init__(self, fsw, priority=97) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            """Set up momentum dumping and thruster control."""
            # Momentum dumping configuration
            self.thrDesatControlConfig = (
                self.fsw.thrDesatControlConfig
            ) = thrMomentumManagement.thrMomentumManagementConfig()
            self.thrDesatControlWrap = (
                self.fsw.thrDesatControlWrap
            ) = self.fsw.simulator.setModelDataWrap(self.thrDesatControlConfig)
            self.thrDesatControlWrap.ModelTag = "thrMomentumManagement"

            self.thrDumpConfig = (
                self.fsw.thrDumpConfig
            ) = thrMomentumDumping.thrMomentumDumpingConfig()
            self.thrDumpWrap = (
                self.fsw.thrDumpWrap
            ) = self.fsw.simulator.setModelDataWrap(self.thrDumpConfig)
            self.thrDumpWrap.ModelTag = "thrDump"

            # Thruster force mapping configuration
            self.thrForceMappingConfig = (
                self.fsw.thrForceMappingConfig
            ) = thrForceMapping.thrForceMappingConfig()
            self.thrForceMappingWrap = (
                self.fsw.thrForceMappingWrap
            ) = self.fsw.simulator.setModelDataWrap(self.thrForceMappingConfig)
            self.thrForceMappingWrap.ModelTag = "thrForceMapping"

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
            self.thrForceMappingConfig.cmdTorqueInMsg.subscribeTo(
                self.thrDesatControlConfig.deltaHOutMsg
            )
            self.thrForceMappingConfig.thrConfigInMsg.subscribeTo(
                self.fsw.thrusterConfigMsg
            )
            self.thrForceMappingConfig.vehConfigInMsg.subscribeTo(self.fsw.vcConfigMsg)
            self.thrForceMappingConfig.controlAxes_B = controlAxes_B
            self.thrForceMappingConfig.thrForceSign = thrForceSign
            self.thrForceMappingConfig.angErrThresh = 3.15

            self._add_model_to_task(
                self.thrForceMappingWrap, self.thrForceMappingConfig, priority=1192
            )

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
            self.thrDesatControlConfig.hs_min = hs_min  # Nms
            self.thrDesatControlConfig.rwSpeedsInMsg.subscribeTo(
                self.fsw.dynamics.rwStateEffector.rwSpeedOutMsg
            )
            self.thrDesatControlConfig.rwConfigDataInMsg.subscribeTo(
                self.fsw.fswRwConfigMsg
            )

            self.thrDumpConfig.deltaHInMsg.subscribeTo(
                self.thrDesatControlConfig.deltaHOutMsg
            )
            self.thrDumpConfig.thrusterImpulseInMsg.subscribeTo(
                self.thrForceMappingConfig.thrForceCmdOutMsg
            )
            self.thrDumpConfig.thrusterConfInMsg.subscribeTo(self.fsw.thrusterConfigMsg)
            self.thrDumpConfig.maxCounterValue = maxCounterValue
            self.thrDumpConfig.thrMinFireTime = thrMinFireTime

            self._add_model_to_task(
                self.thrDesatControlWrap, self.thrDesatControlConfig, priority=1193
            )
            self._add_model_to_task(self.thrDumpWrap, self.thrDumpConfig, priority=1191)

    @action
    def action_desat(self) -> None:
        """Action to charge while desaturating reaction wheels."""
        self.sunPointWrap.Reset(self.simulator.sim_time_ns)
        self.trackingErrorWrap.Reset(self.simulator.sim_time_ns)
        self.thrDesatControlWrap.Reset(self.simulator.sim_time_ns)
        self.thrDumpWrap.Reset(self.simulator.sim_time_ns)
        self.simulator.enableTask(self.SunPointTask.name + self.satellite.id)
        self.simulator.enableTask(self.RWDesatTask.name + self.satellite.id)
        self.simulator.enableTask(self.TrackingErrorTask.name + self.satellite.id)

    class TrackingErrorTask(Task):
        """Task to convert an attitude reference to guidance."""

        name = "trackingErrTask"

        def __init__(self, fsw, priority=90) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            self.trackingErrorData = (
                self.fsw.trackingErrorData
            ) = attTrackingError.attTrackingErrorConfig()
            self.trackingErrorWrap = (
                self.fsw.trackingErrorWrap
            ) = self.fsw.simulator.setModelDataWrap(self.trackingErrorData)
            self.trackingErrorWrap.ModelTag = "trackingError"

        def init_objects(self, **kwargs) -> None:
            self.trackingErrorData.attNavInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.trackingErrorData.attRefInMsg.subscribeTo(self.fsw.attRefMsg)
            cMsgPy.AttGuidMsg_C_addAuthor(
                self.trackingErrorData.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(
                self.trackingErrorWrap, self.trackingErrorData, priority=1197
            )

    class MRPControlTask(Task):
        """Task to control the satellite with reaction wheels."""

        name = "mrpControlTask"

        def __init__(self, fsw, priority=80) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            # Attitude controller configuration
            self.mrpFeedbackControlData = (
                self.fsw.mrpFeedbackControlData
            ) = mrpFeedback.mrpFeedbackConfig()
            self.mrpFeedbackControlWrap = (
                self.fsw.mrpFeedbackControlWrap
            ) = self.fsw.simulator.setModelDataWrap(self.mrpFeedbackControlData)
            self.mrpFeedbackControlWrap.ModelTag = "mrpFeedbackControl"

            # add module that maps the Lr control torque into the RW motor torques
            self.rwMotorTorqueConfig = (
                self.fsw.rwMotorTorqueConfig
            ) = rwMotorTorque.rwMotorTorqueConfig()
            self.rwMotorTorqueWrap = (
                self.fsw.rwMotorTorqueWrap
            ) = self.fsw.simulator.setModelDataWrap(self.rwMotorTorqueConfig)
            self.rwMotorTorqueWrap.ModelTag = "rwMotorTorque"

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
            self.mrpFeedbackControlData.guidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.mrpFeedbackControlData.vehConfigInMsg.subscribeTo(self.fsw.vcConfigMsg)
            self.mrpFeedbackControlData.K = K
            self.mrpFeedbackControlData.Ki = Ki
            self.mrpFeedbackControlData.P = P
            self.mrpFeedbackControlData.integralLimit = (
                2.0 / self.mrpFeedbackControlData.Ki * 0.1
            )

            self._add_model_to_task(
                self.mrpFeedbackControlWrap, self.mrpFeedbackControlData, priority=1196
            )

        def _set_rw_motor_torque(
            self, controlAxes_B: Iterable[float], **kwargs
        ) -> None:
            """Defines the motor torque from the control law.

            Args:
                controlAxes_B): Control unit axes
            """
            self.rwMotorTorqueConfig.rwParamsInMsg.subscribeTo(self.fsw.fswRwConfigMsg)
            self.rwMotorTorqueConfig.vehControlInMsg.subscribeTo(
                self.mrpFeedbackControlData.cmdTorqueOutMsg
            )
            self.rwMotorTorqueConfig.controlAxes_B = controlAxes_B

            self._add_model_to_task(
                self.rwMotorTorqueWrap, self.rwMotorTorqueConfig, priority=1195
            )

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
        c_hat_B = self.locPointConfig.pHat_B
        return np.matmul(self.dynamics.BP.T, c_hat_B)

    def _make_task_list(self) -> list[Task]:
        return super()._make_task_list() + [self.LocPointTask(self)]

    def _set_gateway_msgs(self) -> None:
        super()._set_gateway_msgs()
        self.dynamics.instrument.nodeStatusInMsg.subscribeTo(
            self.insControlConfig.deviceCmdOutMsg
        )

    class LocPointTask(Task):
        """Task to point at targets and trigger the instrument"""

        name = "locPointTask"

        def __init__(self, fsw, priority=96) -> None:
            super().__init__(fsw, priority)

        def create_module_data(self) -> None:
            # Location pointing configuration
            self.locPointConfig = (
                self.fsw.locPointConfig
            ) = locationPointing.locationPointingConfig()
            self.locPointWrap = (
                self.fsw.locPointWrap
            ) = self.fsw.simulator.setModelDataWrap(self.locPointConfig)
            self.locPointWrap.ModelTag = "locPoint"

            # SimpleInstrumentController configuration
            self.insControlConfig = (
                self.fsw.insControlConfig
            ) = simpleInstrumentController.simpleInstrumentControllerConfig()
            self.insControlWrap = (
                self.fsw.insControlWrap
            ) = self.fsw.simulator.setModelDataWrap(self.insControlConfig)
            self.insControlWrap.ModelTag = "instrumentController"

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
            self.locPointConfig.pHat_B = inst_pHat_B
            self.locPointConfig.scAttInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.attOutMsg
            )
            self.locPointConfig.scTransInMsg.subscribeTo(
                self.fsw.dynamics.simpleNavObject.transOutMsg
            )
            self.locPointConfig.locationInMsg.subscribeTo(
                self.fsw.dynamics.imagingTarget.currentGroundStateOutMsg
            )
            self.locPointConfig.useBoresightRateDamping = 1
            cMsgPy.AttGuidMsg_C_addAuthor(
                self.locPointConfig.attGuidOutMsg, self.fsw.attGuidMsg
            )

            self._add_model_to_task(
                self.locPointWrap, self.locPointConfig, priority=1198
            )

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
            self.insControlConfig.attErrTolerance = imageAttErrorRequirement
            if imageRateErrorRequirement is not None:
                self.insControlConfig.useRateTolerance = 1
                self.insControlConfig.rateErrTolerance = imageRateErrorRequirement
            self.insControlConfig.attGuidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.insControlConfig.locationAccessInMsg.subscribeTo(
                self.fsw.dynamics.imagingTarget.accessOutMsgs[-1]
            )

            self._add_model_to_task(
                self.insControlWrap, self.insControlConfig, priority=987
            )

        def reset_for_action(self) -> None:
            self.fsw.dynamics.imagingTarget.Reset(self.fsw.simulator.sim_time_ns)
            self.locPointWrap.Reset(self.fsw.simulator.sim_time_ns)
            self.insControlConfig.controllerStatus = 0
            return super().reset_for_action()

    @action
    def action_image(self, location: Iterable[float], data_name: str) -> None:
        """Action to image a target at a location.

        Args:
            location: PCPF target location [m]
            data_name: Data buffer to store image data to
        """
        self.insControlConfig.controllerStatus = 1
        self.dynamics.instrumentPowerSink.powerStatus = 1
        self.dynamics.imagingTarget.r_LP_P_Init = location
        self.dynamics.instrument.nodeDataName = data_name
        self.insControlConfig.imaged = 0
        self.simulator.enableTask(self.LocPointTask.name + self.satellite.id)

    @action
    def action_downlink(self) -> None:
        """Action to attempt to downlink data."""
        self.hillPointWrap.Reset(self.simulator.sim_time_ns)
        self.trackingErrorWrap.Reset(self.simulator.sim_time_ns)
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
            self.locPointConfig = (
                self.fsw.locPointConfig
            ) = locationPointing.locationPointingConfig()
            self.locPointWrap = (
                self.fsw.locPointWrap
            ) = self.fsw.simulator.setModelDataWrap(self.locPointConfig)
            self.locPointWrap.ModelTag = "locPoint"

            # scanningInstrumentController configuration
            self.insControlConfig = (
                self.fsw.insControlConfig
            ) = scanningInstrumentController.scanningInstrumentControllerConfig()
            self.insControlWrap = (
                self.fsw.simpleInsControlWrap
            ) = self.fsw.simulator.setModelDataWrap(self.insControlConfig)
            self.insControlWrap.ModelTag = "instrumentController"

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
            self.insControlConfig.attErrTolerance = imageAttErrorRequirement
            if imageRateErrorRequirement is not None:
                self.insControlConfig.useRateTolerance = 1
                self.insControlConfig.rateErrTolerance = imageRateErrorRequirement
            self.insControlConfig.attGuidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.insControlConfig.accessInMsg.subscribeTo(
                self.fsw.dynamics.imagingTarget.accessOutMsgs[-1]
            )

            self._add_model_to_task(
                self.insControlWrap, self.insControlConfig, priority=987
            )

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
            self.insControlConfig.deviceCmdOutMsg
        )
        self.insControlConfig.controllerStatus = 1
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
            self.mrpSteeringControlData = (
                self.fsw.mrpSteeringControlData
            ) = mrpSteering.mrpSteeringConfig()
            self.mrpSteeringControlWrap = (
                self.fsw.mrpSteeringControlWrap
            ) = self.fsw.simulator.setModelDataWrap(self.mrpSteeringControlData)
            self.mrpSteeringControlWrap.ModelTag = "mrpSteeringControl"

            # Rate servo
            self.servoData = (
                self.fsw.servoData
            ) = rateServoFullNonlinear.rateServoFullNonlinearConfig()
            self.servoWrap = self.fsw.servoWrap = self.fsw.simulator.setModelDataWrap(
                self.servoData
            )
            self.servoWrap.ModelTag = "rateServo"

            # add module that maps the Lr control torque into the RW motor torques
            self.rwMotorTorqueConfig = (
                self.fsw.rwMotorTorqueConfig
            ) = rwMotorTorque.rwMotorTorqueConfig()
            self.rwMotorTorqueWrap = (
                self.fsw.rwMotorTorqueWrap
            ) = self.fsw.simulator.setModelDataWrap(self.rwMotorTorqueConfig)
            self.rwMotorTorqueWrap.ModelTag = "rwMotorTorque"

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
            self.mrpSteeringControlData.guidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.mrpSteeringControlData.K1 = K1
            self.mrpSteeringControlData.K3 = K3
            self.mrpSteeringControlData.omega_max = omega_max
            self.mrpSteeringControlData.ignoreOuterLoopFeedforward = False

            self.servoData.Ki = servo_Ki
            self.servoData.P = servo_P
            self.servoData.integralLimit = 2.0 / self.servoData.Ki * 0.1
            self.servoData.knownTorquePntB_B = [0.0, 0.0, 0.0]
            self.servoData.guidInMsg.subscribeTo(self.fsw.attGuidMsg)
            self.servoData.vehConfigInMsg.subscribeTo(self.fsw.vcConfigMsg)
            self.servoData.rwParamsInMsg.subscribeTo(self.fsw.fswRwConfigMsg)
            self.servoData.rwSpeedsInMsg.subscribeTo(
                self.fsw.dynamics.rwStateEffector.rwSpeedOutMsg
            )
            self.servoData.rateSteeringInMsg.subscribeTo(
                self.mrpSteeringControlData.rateCmdOutMsg
            )

            self._add_model_to_task(
                self.mrpSteeringControlWrap, self.mrpSteeringControlData, priority=1196
            )
            self._add_model_to_task(self.servoWrap, self.servoData, priority=1195)

        def _set_rw_motor_torque(
            self, controlAxes_B: Iterable[float], **kwargs
        ) -> None:
            """Defines the motor torque from the control law.

            Args:
                controlAxes_B: Control unit axes
            """
            self.rwMotorTorqueConfig.rwParamsInMsg.subscribeTo(self.fsw.fswRwConfigMsg)
            self.rwMotorTorqueConfig.vehControlInMsg.subscribeTo(
                self.servoData.cmdTorqueOutMsg
            )
            self.rwMotorTorqueConfig.controlAxes_B = controlAxes_B

            self._add_model_to_task(
                self.rwMotorTorqueWrap, self.rwMotorTorqueConfig, priority=1194
            )

        def reset_for_action(self) -> None:
            # MRP control enabled by default
            self.fsw.simulator.enableTask(self.name + self.fsw.satellite.id)


class SteeringImagerFSWModel(SteeringFSWModel, ImagingFSWModel):
    pass
