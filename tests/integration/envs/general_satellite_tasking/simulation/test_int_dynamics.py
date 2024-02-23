import gymnasium as gym
import pytest

from bsk_rl.envs.general_satellite_tasking.scenario import data
from bsk_rl.envs.general_satellite_tasking.scenario import sat_actions as sa
from bsk_rl.envs.general_satellite_tasking.scenario import sat_observations as so
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
    StaticTargets,
)
from bsk_rl.envs.general_satellite_tasking.simulation import dynamics, environment, fsw
from bsk_rl.envs.general_satellite_tasking.utils.orbital import random_orbit

###########################
# Composed Dynamics Tests #
###########################


class TestImagingDynModelStorage:

    @pytest.mark.parametrize(
        "storage_capacity, initial_storage",
        [
            (8e7, -1),
            (8e7, 0),
            (8e7, 8e6),
            (10e7, 10e7),
            (8e7, 10e7),
        ],
    )
    def test_storageInit(self, storage_capacity, initial_storage):

        class ImageSat(
            sa.ImagingActions,
            sa.DownlinkAction,
            so.TimeState,
        ):
            dyn_type = dynamics.ImagingDynModel
            fsw_type = fsw.ImagingFSWModel

        env = gym.make(
            "SingleSatelliteTasking-v1",
            satellites=ImageSat(
                "EO-1",
                n_ahead_act=10,
                sat_args=ImageSat.default_sat_args(
                    oe=random_orbit,
                    dataStorageCapacity=storage_capacity,
                    storageInit=initial_storage,
                ),
            ),
            env_type=environment.GroundStationEnvModel,
            env_args=environment.GroundStationEnvModel.default_env_args(),
            env_features=StaticTargets(n_targets=1000),
            data_manager=data.NoDataManager(),
            sim_rate=1.0,
            time_limit=10000.0,
            max_step_duration=1e9,
            disable_env_checker=True,
        )

        env.reset()

        if initial_storage > storage_capacity or initial_storage < 0:
            assert env.satellite.dynamics.storage_level == 0
        else:
            assert env.satellite.dynamics.storage_level == initial_storage

    @pytest.mark.parametrize(
        "storage_capacity, initial_storage",
        [
            (8e7, 8e6),
            (10e7, 10e7),
        ],
    )
    def test_storageInit_downlink(self, storage_capacity, initial_storage):

        class ImageSat(
            sa.DownlinkAction,
            so.TimeState,
        ):
            dyn_type = dynamics.FullFeaturedDynModel
            fsw_type = fsw.ImagingFSWModel

        env = gym.make(
            "SingleSatelliteTasking-v1",
            satellites=ImageSat(
                "EO-1",
                n_ahead_act=10,
                sat_args=ImageSat.default_sat_args(
                    oe=random_orbit,
                    dataStorageCapacity=storage_capacity,
                    storageInit=initial_storage,
                ),
            ),
            env_type=environment.GroundStationEnvModel,
            env_args=environment.GroundStationEnvModel.default_env_args(),
            env_features=StaticTargets(n_targets=1000),
            data_manager=data.NoDataManager(),
            sim_rate=1.0,
            time_limit=10000.0,
            max_step_duration=1e9,
            disable_env_checker=True,
        )

        env.reset()

        terminated = False
        truncated = False
        while not terminated and not truncated:
            observation, reward, terminated, truncated, info = env.step(0)

        assert env.satellite.dynamics.storage_level < initial_storage
