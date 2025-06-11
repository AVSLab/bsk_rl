from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from Basilisk.utilities import macros as mc

from bsk_rl.sim import world
from bsk_rl.sim.dyn import (
    BasicDynamicsModel,
    ContinuousImagingDynModel,
    DynamicsModel,
    GroundStationDynModel,
    ImagingDynModel,
    LOSCommDynModel,
)
from bsk_rl.sim.dyn import base as dyn_module

module = "bsk_rl.sim.dyn."


@patch.multiple(DynamicsModel, __abstractmethods__=set())
class TestDynamicsModel:
    def test_base_class(self):
        sat = MagicMock()
        dyn = DynamicsModel(sat, 1.0)
        dyn.simulator.CreateNewProcess.assert_called_once()
        assert sat.simulator.world == dyn.world
        dyn.reset_for_action()

    @patch(module + "base.check_aliveness_checkers", MagicMock(return_value=True))
    def test_is_alive(self):
        dyn = DynamicsModel(MagicMock(), 1.0)
        assert dyn.is_alive()


basicdyn = module + "BasicDynamicsModel."


def test_basic_requires_world():
    assert world.BasicWorldModel in BasicDynamicsModel._requires_world()


@patch(basicdyn + "_requires_world", MagicMock(return_value=[]))
@patch(basicdyn + "setup_spacecraft_hub")
@patch(basicdyn + "setup_drag_effector")
@patch(basicdyn + "setup_reaction_wheel_dyn_effector")
@patch(basicdyn + "setup_thruster_dyn_effector")
@patch(basicdyn + "setup_simple_nav_object")
@patch(basicdyn + "setup_eclipse_object")
@patch(basicdyn + "setup_solar_panel")
@patch(basicdyn + "setup_battery")
@patch(basicdyn + "setup_power_sink")
@patch(basicdyn + "setup_reaction_wheel_power")
@patch(basicdyn + "setup_thruster_power")
def test_basic_setup_objects(self, *args):
    BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
    for setter in args:
        setter.assert_called_once()


@patch(basicdyn + "_requires_world", MagicMock(return_value=[]))
@patch(basicdyn + "_setup_dynamics_objects", MagicMock())
class TestBasicDynamicsModel:
    def test_dynamic_properties(self):
        dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.simulator.world = MagicMock(PN=np.identity(3), omega_PN_N=np.zeros(3))
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

    def test_orbit_properties(self):
        with patch.object(dyn_module, "orbitalMotion") as mocked:
            mocked.rv2elem = MagicMock(
                return_value=MagicMock(a=1.0, e=2.0, i=3.0, Omega=4.0, omega=5.0, f=6.0)
            )
            dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
            dyn.mu = 1
            dyn.scObject = MagicMock()
            dyn.scObject.scStateOutMsg.read.return_value = MagicMock(
                r_BN_N=[1, 2, 3], v_BN_N=[4, 5, 6]
            )
            dyn.simulator.time = 0.0

            # Check that call to rv2elem happens only once
            assert dyn.semi_major_axis == 1.0
            mocked.rv2elem.assert_called_once()
            assert dyn.eccentricity == 2.0
            assert dyn.inclination == 3.0
            assert dyn.ascending_node == 4.0
            assert dyn.argument_of_periapsis == 5.0
            assert dyn.true_anomaly == 6.0
            mocked.rv2elem.assert_called_once()

    def test_orbit_properties_no_cache_update(self):
        with patch.object(dyn_module, "orbitalMotion") as mocked:
            mocked.rv2elem = MagicMock(return_value=MagicMock(a=1.0))
            dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
            dyn.mu = 1
            dyn.scObject = MagicMock()
            dyn.scObject.scStateOutMsg.read.return_value = MagicMock()
            dyn.simulator.time = 0.0

            assert dyn.semi_major_axis == 1.0
            mocked.rv2elem = MagicMock(return_value=MagicMock(a=2.0))
            assert dyn.semi_major_axis == 1.0

    def test_orbit_properties_cache_update(self):
        with patch.object(dyn_module, "orbitalMotion") as mocked:
            mocked.rv2elem = MagicMock(return_value=MagicMock(a=1.0))
            dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
            dyn.mu = 1
            dyn.scObject = MagicMock()
            dyn.scObject.scStateOutMsg.read.return_value = MagicMock()
            dyn.simulator.time = 0.0

            assert dyn.semi_major_axis == 1.0
            dyn.simulator.time = 1.0
            mocked.rv2elem = MagicMock(return_value=MagicMock(a=2.0))
            assert dyn.semi_major_axis == 2.0

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
        np.testing.assert_allclose(dyn.wheel_speeds_fraction, np.array([0.1, 0.2, 0.3]))

    @pytest.mark.parametrize(
        "vec,valid",
        [([1, 1, 1], False), ([1e9, 0, 1e9], True)],
    )
    def test_altitude_valid(self, vec, valid):
        dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.simulator.world = MagicMock()
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
        dyn.simulator.world = MagicMock()
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

    @patch("Basilisk.simulation.simpleBattery.SimpleBattery", MagicMock())
    @pytest.mark.parametrize(
        "battery_capacity,init_charge,warning",
        [
            (100, 50, False),
            (100, 101, True),
            (100, 100, False),
            (100, 0.0, False),
            (100, -1, True),
        ],
    )
    def test_battery_init_warning(self, battery_capacity, init_charge, warning) -> None:
        dyn = BasicDynamicsModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.solarPanel = MagicMock()
        dyn.logger = MagicMock()
        dyn.setup_battery(
            batteryStorageCapacity=battery_capacity, storedCharge_Init=init_charge
        )
        if warning:
            dyn.logger.warning.assert_called_with(
                f"Battery initial charge {init_charge} incompatible with its capacity {battery_capacity}."
            )
        else:
            dyn.logger.warning.assert_not_called()


class TestLOSCommDynModel:
    losdyn = module + "LOSCommDynModel."

    @patch(losdyn + "_requires_world", MagicMock(return_value=[]))
    @patch(module + "BasicDynamicsModel._setup_dynamics_objects", MagicMock())
    @patch(losdyn + "setup_los_comms")
    def test_setup_objects(self, *args):
        LOSCommDynModel(MagicMock(simulator=MagicMock()), 1.0)
        for setter in args:
            setter.assert_called_once()

    @patch(losdyn + "_requires_world", MagicMock(return_value=[]))
    @patch(losdyn + "_setup_dynamics_objects", MagicMock())
    @patch(module + "relative_motion.spacecraftLocation", MagicMock())
    def test_setup_los_comms(self):
        mock_sim = MagicMock()
        dyn1 = LOSCommDynModel(MagicMock(simulator=mock_sim), 1.0)
        dyn1.scObject = MagicMock()
        dyn1.setup_los_comms(losMaximumRange=-1, priority=1)
        mock_sim.dynamics_list = {1: dyn1}
        mock_sim.AddModelToTask.assert_not_called()
        assert dyn1.los_comms_ids == []

        dyn2 = LOSCommDynModel(MagicMock(simulator=mock_sim), 1.0)
        dyn2.scObject = MagicMock()
        dyn2.setup_los_comms(losMaximumRange=-1, priority=1)
        mock_sim.dynamics_list[2] = dyn2
        assert dyn1.los_comms_ids == [dyn2.satellite.name]
        assert dyn2.los_comms_ids == [dyn1.satellite.name]
        calls = [
            call(dyn1.task_name, dyn1.losComms, ModelPriority=1),
            call(dyn2.task_name, dyn2.losComms, ModelPriority=1),
        ]
        mock_sim.AddModelToTask.assert_has_calls(calls)

        dyn3 = LOSCommDynModel(MagicMock(simulator=mock_sim), 1.0)
        dyn3.scObject = MagicMock()
        dyn3.setup_los_comms(losMaximumRange=-1, priority=1)
        mock_sim.dynamics_list[3] = dyn3
        assert dyn1.los_comms_ids == [dyn2.satellite.name, dyn3.satellite.name]
        assert dyn2.los_comms_ids == [dyn1.satellite.name, dyn3.satellite.name]
        assert dyn3.los_comms_ids == [dyn1.satellite.name, dyn2.satellite.name]
        calls += [call(dyn3.task_name, dyn3.losComms, ModelPriority=1)]
        mock_sim.AddModelToTask.assert_has_calls(calls)


imdyn = module + "ImagingDynModel."


@patch(imdyn + "_requires_world", MagicMock(return_value=[]))
@patch(module + "BasicDynamicsModel._setup_dynamics_objects", MagicMock())
@patch(imdyn + "setup_instrument_power_sink")
@patch(imdyn + "setup_transmitter_power_sink")
@patch(imdyn + "setup_instrument")
@patch(imdyn + "setup_transmitter")
@patch(imdyn + "setup_storage_unit")
@patch(imdyn + "setup_imaging_target")
def test_setup_objects(*args):
    ImagingDynModel(MagicMock(simulator=MagicMock()), 1.0)
    for setter in args:
        setter.assert_called_once()


class FullFeaturedDynModel(GroundStationDynModel, LOSCommDynModel):
    pass


@patch(imdyn + "_requires_world", MagicMock(return_value=[]))
@patch(imdyn + "_setup_dynamics_objects", MagicMock())
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

    @patch(module + "ground_imaging.partitionedStorageUnit", MagicMock())
    @pytest.mark.parametrize(
        "buffers,names,expected",
        [
            (3, None, ["0", "1", "2"]),
            (None, ["a", "b", "c"], ["a", "b", "c"]),
            (3, ["a", "b", "c"], ["a", "b", "c"]),
            (2, ["a", "b", "c"], ValueError),
        ],
    )
    def test_setup_storage_unit(self, buffers, names, expected):
        mock_sim = MagicMock()
        dyn = ImagingDynModel(MagicMock(simulator=mock_sim), 1.0)
        dyn.instrument = MagicMock()
        dyn.transmitter = MagicMock()
        if isinstance(expected, type):
            with pytest.raises(expected):
                dyn.setup_storage_unit(1000, False, 0, buffers, names)
            return
        dyn.setup_storage_unit(1000, False, 0, buffers, names)
        dyn.storageUnit.addPartition.assert_has_calls([call(name) for name in expected])
        dyn.simulator.CreateNewProcess.assert_called_once()

    @patch(
        "Basilisk.simulation.partitionedStorageUnit.PartitionedStorageUnit", MagicMock()
    )
    @pytest.mark.parametrize(
        "storage_capacity,init_storage,warning",
        [
            (100, 50, False),
            (100, 101, True),
            (100, 100, False),
            (100, 0.0, False),
            (100, -1, True),
        ],
    )
    def test_storage_init_warning(
        self, storage_capacity, init_storage, warning
    ) -> None:
        dyn = ImagingDynModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.instrument = MagicMock()
        dyn.transmitter = MagicMock()
        dyn.setup_storage_unit(
            dataStorageCapacity=storage_capacity,
            storageUnitValidCheck=False,
            storageInit=init_storage,
            transmitterNumBuffers=1,
        )
        if warning:
            dyn.logger.warning.assert_called_with(
                f"Initial storage level {init_storage} incompatible with its capacity {storage_capacity}."
            )
        else:
            dyn.logger.warning.assert_not_called()


class TestGroundStationDynModel:
    def test_requires_world(self):
        assert world.GroundStationWorldModel in GroundStationDynModel._requires_world()

    gsdyn = module + "GroundStationDynModel."

    @patch(gsdyn + "_requires_world", MagicMock(return_value=[]))
    @patch(module + "ImagingDynModel._setup_dynamics_objects", MagicMock())
    @patch(gsdyn + "setup_ground_station_locations")
    def test_setup_objects(self, *args):
        GroundStationDynModel(MagicMock(simulator=MagicMock()), 1.0)
        for setter in args:
            setter.assert_called_once()


@patch(imdyn + "_requires_world", MagicMock(return_value=[]))
@patch(imdyn + "_setup_dynamics_objects", MagicMock())
class TestContinuousImagingDynModel:
    def test_storage_properties(self):
        dyn = ContinuousImagingDynModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.storageUnit = MagicMock()
        dyn.storageUnit.storageUnitDataOutMsg.read.return_value.storageLevel = 50.0
        dyn.storageUnit.storageCapacity = 100.0
        assert dyn.storage_level == 50.0
        assert dyn.storage_level_fraction == 0.5

    @patch("Basilisk.simulation.simpleStorageUnit.SimpleStorageUnit", MagicMock())
    @pytest.mark.parametrize(
        "storage_capacity,init_storage,warning",
        [
            (100, 50, False),
            (100, 101, True),
            (100, 100, False),
            (100, 0.0, False),
            (100, -1, True),
        ],
    )
    def test_storage_init_warning(
        self, storage_capacity, init_storage, warning
    ) -> None:
        dyn = ContinuousImagingDynModel(MagicMock(simulator=MagicMock()), 1.0)
        dyn.instrument = MagicMock()
        dyn.transmitter = MagicMock()
        dyn.setup_storage_unit(
            dataStorageCapacity=storage_capacity,
            storageUnitValidCheck=False,
            storageInit=init_storage,
        )
        if warning:
            dyn.logger.warning.assert_called_with(
                f"Initial storage level {init_storage} incompatible with its capacity {storage_capacity}."
            )
        else:
            dyn.logger.warning.assert_not_called()
