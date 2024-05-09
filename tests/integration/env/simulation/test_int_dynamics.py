import gymnasium as gym
import pytest

from bsk_rl.env.scenario import actions as act
from bsk_rl.env.scenario import data
from bsk_rl.env.scenario import observations as obs
from bsk_rl.env.scenario import satellites as sats
from bsk_rl.env.scenario.environment_features import StaticTargets
from bsk_rl.env.simulation import dynamics, fsw
from bsk_rl.utils.orbital import random_orbit

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

        class ImageSat(sats.ImagingSatellite):
            dyn_type = dynamics.ImagingDynModel
            fsw_type = fsw.ImagingFSWModel
            observation_spec = [obs.Time()]
            action_spec = [act.Downlink(), act.Image(n_ahead_image=10)]

        env = gym.make(
            "SingleSatelliteTasking-v1",
            satellites=ImageSat(
                "EO-1",
                sat_args=ImageSat.default_sat_args(
                    oe=random_orbit,
                    dataStorageCapacity=storage_capacity,
                    storageInit=initial_storage,
                ),
            ),
            env_features=StaticTargets(n_targets=1000),
            data_manager=data.NoDataManager(),
            sim_rate=1.0,
            time_limit=10000.0,
            max_step_duration=1e9,
            disable_env_checker=True,
        )

        env.reset()

        if initial_storage > storage_capacity or initial_storage < 0:
            assert env.unwrapped.satellite.dynamics.storage_level == 0
        else:
            assert env.unwrapped.satellite.dynamics.storage_level == initial_storage

    @pytest.mark.parametrize(
        "storage_capacity, initial_storage",
        [
            (8e7, 8e6),
            (10e7, 10e7),
        ],
    )
    def test_storageInit_downlink(self, storage_capacity, initial_storage):

        class ImageSat(sats.ImagingSatellite):
            dyn_type = dynamics.FullFeaturedDynModel
            fsw_type = fsw.ImagingFSWModel
            observation_spec = [obs.Time()]
            action_spec = [act.Downlink()]

        env = gym.make(
            "SingleSatelliteTasking-v1",
            satellites=ImageSat(
                "EO-1",
                sat_args=ImageSat.default_sat_args(
                    oe=random_orbit,
                    dataStorageCapacity=storage_capacity,
                    storageInit=initial_storage,
                ),
            ),
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

        assert env.unwrapped.satellite.dynamics.storage_level < initial_storage
