from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from bsk_rl.envs.general_satellite_tasking.simulation.environment import (
    BasicEnvironmentModel,
    EnvironmentModel,
    GroundStationEnvModel,
)

module = "bsk_rl.envs.general_satellite_tasking.simulation.environment."


class TestEnvironmentModel:
    @patch(module + "collect_default_args")
    def test_default_env_args(self, mock_collect):
        mock_collect.return_value = {"a": 1, "b": 2, "c": 3}
        assert EnvironmentModel.default_env_args() == {"a": 1, "b": 2, "c": 3}

    @pytest.mark.parametrize(
        "overwrite,error", [({"c": 4}, False), ({"not_c": 4}, True)]
    )
    @patch(module + "collect_default_args")
    def test_default_sat_args_overwrote(self, mock_collect, overwrite, error):
        mock_collect.return_value = {"a": 1, "b": 2, "c": 3}
        if not error:
            assert EnvironmentModel.default_env_args(**overwrite) == {
                "a": 1,
                "b": 2,
                "c": 4,
            }
        else:
            with pytest.raises(KeyError):
                EnvironmentModel.default_env_args(**overwrite)

    @patch.multiple(EnvironmentModel, __abstractmethods__=set())
    @patch(module + "EnvironmentModel._init_environment_objects")
    def test_init(self, mock_obj_init):
        mock_sim = MagicMock()
        kwargs = dict(a=1, b=2)
        env = EnvironmentModel(mock_sim, 1.0, **kwargs)
        mock_sim.CreateNewProcess.assert_called_once()
        mock_sim.CreateNewTask.assert_called_once()
        mock_obj_init.assert_called_once_with(**kwargs)
        assert env.simulator is not mock_sim
        assert env.simulator == mock_sim


class TestBasicEnvironmentModel:
    basicenv = module + "BasicEnvironmentModel."

    @patch(basicenv + "__init__", MagicMock(return_value=None))
    def test_PN(self):
        env = BasicEnvironmentModel(MagicMock(), 1.0)
        env.gravFactory = MagicMock()
        env.body_index = 0
        msg = env.gravFactory.spiceObject.planetStateOutMsgs.__getitem__
        msg.return_value.read.return_value.J20002Pfix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        assert (env.PN == np.identity(3)).all()

    @patch(basicenv + "__init__", MagicMock(return_value=None))
    def test_omega_PN_N(self):
        env = BasicEnvironmentModel(MagicMock(), 1.0)
        env.gravFactory = MagicMock()
        env.body_index = 0
        msg = env.gravFactory.spiceObject.planetStateOutMsgs.__getitem__
        msg.return_value.read.return_value.J20002Pfix_dot = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        msg.return_value.read.return_value.J20002Pfix = [0, -1, 1, 1, 0, -1, -1, 1, 0]
        assert (env.omega_PN_N == np.array([1, 1, 1])).all()

    @patch(basicenv + "_set_gravity_bodies")
    @patch(basicenv + "_set_epoch_object")
    @patch(basicenv + "_set_atmosphere_density_model")
    @patch(basicenv + "_set_eclipse_object")
    def test_init_and_delete(self, grav_set, epoch_set, atmos_set, eclipse_set):
        env = BasicEnvironmentModel(MagicMock(), 1.0)
        for setter in (grav_set, epoch_set, atmos_set, eclipse_set):
            setter.assert_called_once()
        unload_function = MagicMock()
        env.gravFactory = MagicMock(unloadSpiceKernels=unload_function)
        del env
        unload_function.assert_called_once()

    @patch(basicenv + "_init_environment_objects", MagicMock())
    @patch(module + "simIncludeGravBody", MagicMock())
    @patch(
        module + "pyswice",
        MagicMock(),
    )
    def test_set_gravity_bodies(self):
        # Smoke test
        env = BasicEnvironmentModel(MagicMock(), 1.0)
        env.simulator = MagicMock()
        env._set_gravity_bodies(utc_init="time")
        env.simulator.AddModelToTask.assert_called_once()

    @patch(basicenv + "_init_environment_objects", MagicMock())
    @patch(module + "ephemerisConverter", MagicMock())
    def test_set_epoch_object(self):
        # Smoke test
        env = BasicEnvironmentModel(MagicMock(), 1.0)
        env.simulator = MagicMock()
        env.gravFactory = MagicMock()
        env.sun_index = 0
        env.body_index = 1
        env._set_epoch_object()
        env.simulator.AddModelToTask.assert_called_once()

    @patch(basicenv + "_init_environment_objects", MagicMock())
    @patch(module + "exponentialAtmosphere", MagicMock())
    def test_set_atmosphere_density_model(self):
        # Smoke test
        env = BasicEnvironmentModel(MagicMock(), 1.0)
        env.simulator = MagicMock()
        env.gravFactory = MagicMock()
        env.body_index = 1

        env._set_atmosphere_density_model(
            planetRadius=1.0,
            baseDensity=1.0,
            scaleHeight=1.0,
        )
        env.simulator.AddModelToTask.assert_called_once()

    @patch(basicenv + "_init_environment_objects", MagicMock())
    @patch(module + "eclipse", MagicMock())
    def test_set_eclipse_object(self):
        # Smoke test
        env = BasicEnvironmentModel(MagicMock(), 1.0)
        env.simulator = MagicMock()
        env.gravFactory = MagicMock()
        env.sun_index = 0
        env.body_index = 1
        env._set_eclipse_object()
        env.simulator.AddModelToTask.assert_called_once()


class TestGroundStationEnvModel:
    groundenv = module + "GroundStationEnvModel."

    @patch(groundenv + "_set_ground_locations")
    @patch(module + "BasicEnvironmentModel._init_environment_objects", MagicMock())
    def test_init_environment_objects(self, ground_set):
        GroundStationEnvModel(MagicMock(), 1.0)
        ground_set.assert_called_once()

    @patch(groundenv + "_init_environment_objects", MagicMock())
    @patch(groundenv + "_create_ground_station")
    def test_set_ground_locations(self, mock_gs_create):
        env = GroundStationEnvModel(MagicMock(), 1.0)
        env._set_ground_locations([dict(a=1), dict(b=2)], 1000.0, 1.0, 1000.0)
        mock_gs_create.assert_has_calls(
            [call(a=1, priority=1399), call(b=2, priority=1398)]
        )

    @patch(groundenv + "_init_environment_objects", MagicMock())
    @patch(module + "groundLocation", MagicMock())
    def test_create_ground_station(self):
        env = GroundStationEnvModel(MagicMock(), 1.0)
        env.simulator = MagicMock()
        env.gravFactory = MagicMock()
        env.groundStations = []
        env.groundLocationPlanetRadius = 10.0
        env.gsMinimumElevation = 1.0
        env.gsMaximumRange = 1000.0
        env.body_index = 1
        env._create_ground_station(0.0, 0.0)
        env.simulator.AddModelToTask.assert_called_once()
