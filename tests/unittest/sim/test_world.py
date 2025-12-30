from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from bsk_rl.sim.world import (
    AtmosphereWorldModel,
    EclipseWorldModel,
    GroundStationWorldModel,
    WorldModel,
    WorldModelABC,
)

module = "bsk_rl.sim.world."


class TestWorldModelABC:
    @patch(module + "collect_default_args")
    def test_default_world_args(self, mock_collect):
        mock_collect.return_value = {"a": 1, "b": 2, "c": 3}
        assert WorldModelABC.default_world_args() == {"a": 1, "b": 2, "c": 3}

    @pytest.mark.parametrize(
        "overwrite,error", [({"c": 4}, False), ({"not_c": 4}, True)]
    )
    @patch(module + "collect_default_args")
    def test_default_sat_args_overwrote(self, mock_collect, overwrite, error):
        mock_collect.return_value = {"a": 1, "b": 2, "c": 3}
        if not error:
            assert WorldModelABC.default_world_args(**overwrite) == {
                "a": 1,
                "b": 2,
                "c": 4,
            }
        else:
            with pytest.raises(KeyError):
                WorldModelABC.default_world_args(**overwrite)

    @patch.multiple(WorldModelABC, __abstractmethods__=set())
    @patch(module + "WorldModelABC._setup_world_objects")
    def test_init(self, mock_obj_init):
        mock_sim = MagicMock()
        kwargs = dict(a=1, b=2)
        world = WorldModelABC(mock_sim, 1.0, **kwargs)
        mock_sim.CreateNewProcess.assert_called_once()
        mock_sim.CreateNewTask.assert_called_once()
        mock_obj_init.assert_called_once_with(**kwargs)
        assert world.simulator is not mock_sim
        assert world.simulator == mock_sim


class TestWorldModel:
    baseworld = module + "WorldModel."

    @patch(baseworld + "__init__", MagicMock(return_value=None))
    def test_PN(self):
        world = WorldModel(MagicMock(), 1.0)
        world.gravFactory = MagicMock()
        world.body_index = 0
        msg = world.gravFactory.spiceObject.planetStateOutMsgs.__getitem__
        msg.return_value.read.return_value.J20002Pfix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        assert (world.PN == np.identity(3)).all()

    @patch(baseworld + "__init__", MagicMock(return_value=None))
    def test_omega_PN_N(self):
        world = WorldModel(MagicMock(), 1.0)
        world.gravFactory = MagicMock()
        world.body_index = 0
        msg = world.gravFactory.spiceObject.planetStateOutMsgs.__getitem__
        msg.return_value.read.return_value.J20002Pfix_dot = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        msg.return_value.read.return_value.J20002Pfix = [0, -1, 1, 1, 0, -1, -1, 1, 0]
        assert (world.omega_PN_N == np.array([1, 1, 1])).all()

    @patch(baseworld + "setup_gravity_bodies")
    @patch(baseworld + "setup_ephem_object")
    def test_setup_and_delete(self, grav_set, epoch_set):
        world = WorldModel(MagicMock(), 1.0)
        for setter in (grav_set, epoch_set):
            setter.assert_called_once()
        unload_function = MagicMock()
        world.gravFactory = MagicMock(unloadSpiceKernels=unload_function)
        del world
        unload_function.assert_called_once()

    @patch(baseworld + "_setup_world_objects", MagicMock())
    @patch(module + "simIncludeGravBody", MagicMock())
    def testsetup_gravity_bodies(self):
        # Smoke test
        world = WorldModel(MagicMock(), 1.0)
        world.simulator = MagicMock()
        world.setup_gravity_bodies(utc_init="time")
        world.simulator.AddModelToTask.assert_called_once()

    @patch(baseworld + "_setup_world_objects", MagicMock())
    @patch(module + "ephemerisConverter", MagicMock())
    def testsetup_epoch_object(self):
        # Smoke test
        world = WorldModel(MagicMock(), 1.0)
        world.simulator = MagicMock()
        world.gravFactory = MagicMock()
        world.sun_index = 0
        world.body_index = 1
        world.setup_ephem_object()
        world.simulator.AddModelToTask.assert_called_once()


class TestAtmosphereWorldModel:
    baseworld = module + "AtmosphereWorldModel."

    @patch(baseworld + "_setup_world_objects", MagicMock())
    @patch(module + "exponentialAtmosphere", MagicMock())
    def testsetup_atmosphere_density_model(self):
        # Smoke test
        world = AtmosphereWorldModel(MagicMock(), 1.0)
        world.simulator = MagicMock()
        world.gravFactory = MagicMock()
        world.body_index = 1

        world.setup_atmosphere_density_model(
            planetRadius=1.0,
            baseDensity=1.0,
            scaleHeight=1.0,
        )
        world.simulator.AddModelToTask.assert_called_once()


class TestEclipseWorldModel:
    baseworld = module + "EclipseWorldModel."

    @patch(baseworld + "_setup_world_objects", MagicMock())
    @patch(module + "eclipse", MagicMock())
    def testsetup_eclipse_object(self):
        # Smoke test
        world = EclipseWorldModel(MagicMock(), 1.0)
        world.simulator = MagicMock()
        world.gravFactory = MagicMock()
        world.sun_index = 0
        world.body_index = 1
        world.setup_eclipse_object()
        world.simulator.AddModelToTask.assert_called_once()


class TestGroundStationWorldModel:
    groundworld = module + "GroundStationWorldModel."

    @patch(groundworld + "setup_ground_locations")
    @patch(module + "WorldModel._setup_world_objects", MagicMock())
    def test_setup_world_objects(self, ground_set):
        GroundStationWorldModel(MagicMock(), 1.0)
        ground_set.assert_called_once()

    @patch(groundworld + "_setup_world_objects", MagicMock())
    @patch(groundworld + "_create_ground_station")
    def testsetup_ground_locations(self, mock_gs_create):
        world = GroundStationWorldModel(MagicMock(), 1.0)
        world.setup_ground_locations([dict(a=1), dict(b=2)], 1000.0, 1.0, 1000.0)
        mock_gs_create.assert_has_calls(
            [call(a=1, priority=1399), call(b=2, priority=1398)]
        )

    @patch(groundworld + "_setup_world_objects", MagicMock())
    @patch(module + "groundLocation", MagicMock())
    def test_create_ground_station(self):
        world = GroundStationWorldModel(MagicMock(), 1.0)
        world.simulator = MagicMock()
        world.gravFactory = MagicMock()
        world.groundStations = []
        world.groundLocationPlanetRadius = 10.0
        world.gsMinimumElevation = 1.0
        world.gsMaximumRange = 1000.0
        world.body_index = 1
        world._create_ground_station(0.0, 0.0)
        world.simulator.AddModelToTask.assert_called_once()
