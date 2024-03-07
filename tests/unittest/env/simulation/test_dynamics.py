from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from Basilisk.utilities import macros as mc

from bsk_rl.env.simulation import environment
from bsk_rl.env.simulation.dynamics import (
    BasicDynamicsModel,
    ContinuousImagingDynModel,
    DynamicsModel,
    GroundStationDynModel,
    ImagingDynModel,
    LOSCommDynModel,
)

module = "bsk_rl.env.simulation.dynamics."


@patch.multiple(DynamicsModel, __abstractmethods__=set())
class TestDynamicsModel:
    def test_base_class(self):
        sat = MagicMock()
        dyn = DynamicsModel(sat, 1.0)
        dyn.simulator.CreateNewProcess.assert_called_once()
        assert sat.simulator.environment == dyn.environment
        dyn.reset_for_action()

    @patch(module + "check_aliveness_checkers", MagicMock(return_value=True))
    def test_is_alive(self):
        dyn = DynamicsModel(MagicMock(), 1.0)
        assert dyn.is_alive()


basicdyn = module + "BasicDynamicsModel."


def test_basic_requires_env():
    assert environment.BasicEnvironmentModel in BasicDynamicsModel._requires_env()


@patch(basicdyn + "_requires_env", MagicMock(return_value=[]))
@patch(basicdyn + "_set_spacecraft_hub")
@patch(basicdyn + "_set_drag_effector")
@patch(basicdyn + "_set_reaction_wheel_dyn_effector")
@patch(basicdyn + "_set_thruster_dyn_effector")
@patch(basicdyn + "_set_simple_nav_object")
@patch(basicdyn + "_set_eclipse_object")
@patch(basicdyn + "_set_solar_panel")
@patch(basicdyn + "_set_battery")
@patch(basicdyn + "_set_reaction_wheel_power")
@patch(basicdyn + "_set_thruster_power")
def test_basic_init_objects(self, *args):
    BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
    for setter in args:
        setter.assert_called_once()


@patch(basicdyn + "_requires_env", MagicMock(return_value=[]))
@patch(basicdyn + "_init_dynamics_objects", MagicMock())
class TestBasicDynamicsModel:
    def test_dynamic_properties(self):
        dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.simulator.environment = MagicMock(PN=np.identity(3), omega_PN_N=np.zeros(3))
        dyn.scObject = MagicMock()
        message = dyn.scObject.scStateOutMsg.read.return_value
        message.sigma_BN = np.zeros(3)
        message.omega_BN_B = np.zeros(3)
        message.r_BN_N = np.zeros(3)
        message.v_BN_N = np.zeros(3)
        assert (dyn.sigma_BN == np.zeros(3)).all()
        assert (dyn.BN == np.identity(3)).all()
        assert (dyn.omega_BN_B == np.zeros(3)).all()
        assert (dyn.BP == np.identity(3)).all()
        assert (dyn.r_BN_N == np.zeros(3)).all()
        assert (dyn.r_BN_P == np.zeros(3)).all()
        assert (dyn.v_BN_N == np.zeros(3)).all()
        assert (dyn.v_BN_P == np.zeros(3)).all()
        assert (dyn.omega_BP_P == np.zeros(3)).all()

    def test_battery_properties(self):
        dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.powerMonitor = MagicMock()
        dyn.powerMonitor.batPowerOutMsg.read.return_value.storageLevel = 50.0
        dyn.powerMonitor.storageCapacity = 100.0
        assert dyn.battery_charge == 50.0
        assert dyn.battery_charge_fraction == 0.5

    def test_wheel_properties(self):
        dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.rwStateEffector = MagicMock()
        speeds = np.array([10.0, 20.0, 30.0]) * mc.rpm2radsec
        dyn.rwStateEffector.rwSpeedOutMsg.read.return_value.wheelSpeeds = speeds
        dyn.maxWheelSpeed = 100.0
        assert (dyn.wheel_speeds == speeds).all()
        assert (abs(dyn.wheel_speeds_fraction - np.array([0.1, 0.2, 0.3])) < 1e-9).all()

    @pytest.mark.parametrize(
        "vec,valid",
        [([1, 1, 1], False), ([1e9, 0, 1e9], True)],
    )
    def test_altitude_valid(self, vec, valid):
        dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.simulator.environment = MagicMock()
        dyn.scObject = MagicMock()
        message = dyn.scObject.scStateOutMsg.read.return_value
        message.r_BN_N = vec
        assert dyn.altitude_valid() == valid

    @pytest.mark.parametrize(
        "speeds,valid",
        [
            ([-10, 0.0, 10.0], True),
            ([200.0, 50.0, -50.0], False),
            ([-200.0, 50.0, -50.0], False),
        ],
    )
    def test_rw_speeds_valid(self, speeds, valid):
        dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.simulator.environment = MagicMock()
        dyn.rwStateEffector = MagicMock()
        speeds = np.array(speeds) * mc.rpm2radsec
        dyn.rwStateEffector.rwSpeedOutMsg.read.return_value.wheelSpeeds = speeds
        dyn.maxWheelSpeed = 100.0
        assert dyn.rw_speeds_valid() == valid

    @pytest.mark.parametrize(
        "level,valid",
        [(10, True), (0, False), (-10, False)],
    )
    def test_battery_valid(self, level, valid):
        dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.powerMonitor = MagicMock()
        dyn.powerMonitor.batPowerOutMsg.read.return_value.storageLevel = level
        dyn.powerMonitor.storageCapacity = 100.0
        assert dyn.battery_valid() == valid


class TestLOSCommDynModel:
    losdyn = module + "LOSCommDynModel."

    @patch(losdyn + "_requires_env", MagicMock(return_value=[]))
    @patch(module + "BasicDynamicsModel._init_dynamics_objects", MagicMock())
    @patch(losdyn + "_set_los_comms")
    def test_init_objects(self, *args):
        LOSCommDynModel(MagicMock(simulator=MagicMock()), 1.0)
        for setter in args:
            setter.assert_called_once()

    @patch(losdyn + "_requires_env", MagicMock(return_value=[]))
    @patch(losdyn + "_init_dynamics_objects", MagicMock())
    @patch(module + "spacecraftLocation", MagicMock())
    def test_set_los_comms(self):
        mock_sim = MagicMock()
        dyn1 = LOSCommDynModel(MagicMock(simulator=mock_sim), 1.0)
        dyn1.scObject = MagicMock()
        dyn1._set_los_comms(priority=1)
        mock_sim.dynamics_list = {1: dyn1}
        mock_sim.AddModelToTask.assert_not_called()
        assert dyn1.los_comms_ids == []

        dyn2 = LOSCommDynModel(MagicMock(simulator=mock_sim), 1.0)
        dyn2.scObject = MagicMock()
        dyn2._set_los_comms(priority=1)
        mock_sim.dynamics_list[2] = dyn2
        assert dyn1.los_comms_ids == [dyn2.satellite.id]
        assert dyn2.los_comms_ids == [dyn1.satellite.id]
        calls = [
            call(dyn1.task_name, dyn1.losComms, ModelPriority=1),
            call(dyn2.task_name, dyn2.losComms, ModelPriority=1),
        ]
        mock_sim.AddModelToTask.assert_has_calls(calls)

        dyn3 = LOSCommDynModel(MagicMock(simulator=mock_sim), 1.0)
        dyn3.scObject = MagicMock()
        dyn3._set_los_comms(priority=1)
        mock_sim.dynamics_list[3] = dyn3
        assert dyn1.los_comms_ids == [dyn2.satellite.id, dyn3.satellite.id]
        assert dyn2.los_comms_ids == [dyn1.satellite.id, dyn3.satellite.id]
        assert dyn3.los_comms_ids == [dyn1.satellite.id, dyn2.satellite.id]
        calls += [call(dyn3.task_name, dyn3.losComms, ModelPriority=1)]
        mock_sim.AddModelToTask.assert_has_calls(calls)


imdyn = module + "ImagingDynModel."


@patch(imdyn + "_requires_env", MagicMock(return_value=[]))
@patch(module + "BasicDynamicsModel._init_dynamics_objects", MagicMock())
@patch(imdyn + "_set_instrument_power_sink")
@patch(imdyn + "_set_transmitter_power_sink")
@patch(imdyn + "_set_instrument")
@patch(imdyn + "_set_transmitter")
@patch(imdyn + "_set_storage_unit")
@patch(imdyn + "_set_imaging_target")
def test_init_objects(*args):
    ImagingDynModel(MagicMock(simulator=MagicMock()), 1.0)
    for setter in args:
        setter.assert_called_once()


@patch(imdyn + "_requires_env", MagicMock(return_value=[]))
@patch(imdyn + "_init_dynamics_objects", MagicMock())
class TestImagingDynModel:
    def test_storage_properties(self):
        dyn = ImagingDynModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.storageUnit = MagicMock()
        dyn.storageUnit.storageUnitDataOutMsg.read.return_value.storageLevel = 50.0
        dyn.storageUnit.storageCapacity = 100.0
        assert dyn.storage_level == 50.0
        assert dyn.storage_level_fraction == 0.5

    @pytest.mark.parametrize(
        "level,valid_check,valid",
        [
            (10, True, True),
            (0, True, True),
            (110, True, False),
            (100.001, True, True),
            (10, False, True),
            (110, False, True),
        ],
    )
    def test_data_storage_valid(self, level, valid_check, valid):
        dyn = ImagingDynModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.storageUnit = MagicMock()
        dyn.storageUnitValidCheck = valid_check
        dyn.storageUnit.storageUnitDataOutMsg.read.return_value.storageLevel = level
        dyn.storageUnit.storageCapacity = 100.0
        assert dyn.data_storage_valid() == valid

    @patch(module + "partitionedStorageUnit", MagicMock())
    @pytest.mark.parametrize(
        "buffers,names,expected",
        [
            (3, None, ["0", "1", "2"]),
            (None, ["a", "b", "c"], ["a", "b", "c"]),
            (3, ["a", "b", "c"], ["a", "b", "c"]),
            (2, ["a", "b", "c"], ValueError),
        ],
    )
    def test_set_storage_unit(self, buffers, names, expected):
        mock_sim = MagicMock()
        dyn = ImagingDynModel(MagicMock(simulator=mock_sim), 1.0)
        dyn.instrument = MagicMock()
        dyn.transmitter = MagicMock()
        if isinstance(expected, type):
            with pytest.raises(expected):
                dyn._set_storage_unit(1000, buffers, names)
            return
        dyn._set_storage_unit(1000, buffers, names)
        dyn.storageUnit.addPartition.assert_has_calls([call(name) for name in expected])
        dyn.simulator.CreateNewProcess.assert_called_once()


class TestGroundStationDynModel:
    def test_requires_env(self):
        assert (
            environment.GroundStationEnvModel in GroundStationDynModel._requires_env()
        )

    gsdyn = module + "GroundStationDynModel."

    @patch(gsdyn + "_requires_env", MagicMock(return_value=[]))
    @patch(module + "ImagingDynModel._init_dynamics_objects", MagicMock())
    @patch(gsdyn + "_set_ground_station_locations")
    def test_init_objects(self, *args):
        GroundStationDynModel(MagicMock(simulator=MagicMock()), 1.0)
        for setter in args:
            setter.assert_called_once()


@patch(imdyn + "_requires_env", MagicMock(return_value=[]))
@patch(imdyn + "_init_dynamics_objects", MagicMock())
class TestContinuousImagingDynModel:
    def test_storage_properties(self):
        dyn = ContinuousImagingDynModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.storageUnit = MagicMock()
        dyn.storageUnit.storageUnitDataOutMsg.read.return_value.storageLevel = 50.0
        dyn.storageUnit.storageCapacity = 100.0
        assert dyn.storage_level == 50.0
        assert dyn.storage_level_fraction == 0.5
