import gymnasium as gym
import numpy as np
from pytest import approx

from bsk_rl.envs.general_satellite_tasking.scenario import data
from bsk_rl.envs.general_satellite_tasking.scenario import sat_actions as sa
from bsk_rl.envs.general_satellite_tasking.scenario import sat_observations as so
from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
    StaticTargets,
    UniformNadirFeature,
)
from bsk_rl.envs.general_satellite_tasking.simulation import dynamics, environment, fsw
from bsk_rl.envs.general_satellite_tasking.utils.orbital import random_orbit

#########################
# Composed Action Tests #
#########################


class TestImagingAndDownlink:
    class ImageSat(
        sa.ImagingActions,
        sa.DownlinkAction,
        so.TimeState,
    ):
        dyn_type = dynamics.GroundStationDynModel
        fsw_type = fsw.ImagingFSWModel

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=ImageSat(
            "EO-1",
            n_ahead_act=10,
            sat_args=ImageSat.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.05,
                imageRateErrorRequirement=0.05,
                instrumentBaudRate=1.0,
                dataStorageCapacity=3.0,
                transmitterBaudRate=-1.0,
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

    def test_image(self):
        self.env.reset()
        storage_init = self.env.satellite.dynamics.storage_level
        for i in range(10):
            self.env.step(i + 1)
        assert self.env.satellite.dynamics.storage_level > storage_init
        assert self.env.satellite.dynamics.storage_level == approx(3.0)

    def test_downlink(self):
        storage_init = self.env.satellite.dynamics.storage_level
        assert storage_init > 0.0  # Should be filled from previous test
        self.env.step(0)  # Should encounter a downlink opportunity before timeout
        assert self.env.satellite.dynamics.storage_level < storage_init

    def test_image_by_name(self):
        # Smoketest
        self.env.reset()
        target = self.env.satellite.upcoming_targets(10)[9]
        self.env.step(target)
        target = self.env.satellite.upcoming_targets(10)[9]
        self.env.step(target.id)
        assert True


###########################
# Individual Action Tests #
###########################


class TestChargingAction:
    class ChargeSat(
        sa.ChargingAction,
        so.TimeState,
    ):
        dyn_type = dynamics.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=ChargeSat(
            "Charger",
            sat_args=ChargeSat.default_sat_args(
                # High, inclined orbit makes eclipse unlikely
                oe=random_orbit(alt=50000, i=90),
                batteryStorageCapacity=500_000,
                storedCharge_Init=250_000,
            ),
        ),
        env_type=environment.BasicEnvironmentModel,
        env_args=environment.BasicEnvironmentModel.default_env_args(),
        env_features=StaticTargets(n_targets=0),
        data_manager=data.NoDataManager(),
        sim_rate=1.0,
        max_step_duration=300.0,
        time_limit=300.0,
        disable_env_checker=True,
    )

    def test_charging_action(self):
        self.env.reset()
        init_charge = self.env.satellite.dynamics.battery_charge
        self.env.step(0)  # Charge
        assert self.env.satellite.dynamics.battery_charge > init_charge


class TestDesatAction:
    class DesatSat(
        sa.DesatAction,
        so.TimeState,
    ):
        dyn_type = dynamics.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=DesatSat(
            "Ellite",
            sat_args=DesatSat.default_sat_args(
                oe=random_orbit,
                wheelSpeeds=[1000.0, -1000.0, 1000.0],
            ),
        ),
        env_type=environment.BasicEnvironmentModel,
        env_args=environment.BasicEnvironmentModel.default_env_args(),
        env_features=StaticTargets(n_targets=0),
        data_manager=data.NoDataManager(),
        sim_rate=1.0,
        max_step_duration=300.0,
        time_limit=300.0,
        disable_env_checker=True,
    )

    def test_desat_action(self):
        self.env.reset()
        init_speeds = self.env.satellite.dynamics.wheel_speeds
        self.env.step(0)  # Desat
        assert np.linalg.norm(
            self.env.satellite.dynamics.wheel_speeds
        ) < np.linalg.norm(init_speeds)


class TestNadirImagingActions:
    class ImageSat(
        sa.NadirImagingAction,
        so.TimeState,
    ):
        dyn_type = dynamics.ContinuousImagingDynModel
        fsw_type = fsw.ContinuousImagingFSWModel

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=ImageSat(
            "EO-1",
            n_ahead_act=10,
            sat_args=ImageSat.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.05,
                imageRateErrorRequirement=0.05,
                instrumentBaudRate=1.0,
                dataStorageCapacity=3.0,
                transmitterBaudRate=-1.0,
            ),
        ),
        env_type=environment.BasicEnvironmentModel,
        env_args=environment.BasicEnvironmentModel.default_env_args(),
        env_features=UniformNadirFeature(),
        data_manager=data.NoDataManager(),
        sim_rate=1.0,
        time_limit=10000.0,
        max_step_duration=1e9,
        disable_env_checker=True,
    )

    def test_image(self):
        self.env.reset()
        storage_init = self.env.satellite.dynamics.storage_level
        self.env.step(0)
        assert self.env.satellite.dynamics.storage_level > storage_init
