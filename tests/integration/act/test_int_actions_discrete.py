from functools import partial

import gymnasium as gym
import numpy as np
import pytest
from pytest import approx

from bsk_rl import act, data, obs, sats
from bsk_rl.scene import UniformNadirScanning, UniformTargets
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_circular_orbit, random_orbit

#########################
# Composed Action Tests #
#########################


class TestImagingAndDownlink:
    class ImageSat(sats.ImagingSatellite):
        dyn_type = dyn.GroundStationDynModel
        fsw_type = fsw.ImagingFSWModel
        observation_spec = [obs.Time()]
        action_spec = [act.Downlink(), act.Image(n_ahead_image=10)]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=ImageSat(
            "EO-1",
            sat_args=ImageSat.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.05,
                imageRateErrorRequirement=0.05,
                instrumentBaudRate=1.0,
                dataStorageCapacity=3.0,
                transmitterBaudRate=-1.0,
            ),
        ),
        scenario=UniformTargets(n_targets=1000),
        rewarder=data.UniqueImageReward(),
        sim_rate=1.0,
        time_limit=10000.0,
        max_step_duration=1e9,
        disable_env_checker=True,
    )

    def test_image(self):
        self.env.reset()
        storage_init = self.env.unwrapped.satellite.dynamics.storage_level
        for i in range(10):
            self.env.step(i + 1)
        assert self.env.unwrapped.satellite.dynamics.storage_level > storage_init
        assert self.env.unwrapped.satellite.dynamics.storage_level == approx(3.0)

    @pytest.mark.flaky(retries=5, delay=1)
    def test_downlink(self):
        self.env.reset()
        for i in range(10):
            self.env.step(i + 1)
        storage_init = self.env.unwrapped.satellite.dynamics.storage_level
        assert storage_init > 0.0  # Should be filled from random stepping
        self.env.step(0)  # Should encounter a downlink opportunity before timeout
        assert self.env.unwrapped.satellite.dynamics.storage_level < storage_init

    def test_image_by_name(self):
        # Smoketest
        self.env.reset()
        target = self.env.unwrapped.satellite.find_next_opportunities(
            n=10, types="target"
        )[9]["object"]
        self.env.step(target)
        target = self.env.unwrapped.satellite.find_next_opportunities(
            n=10, types="target"
        )[9]["object"]
        self.env.step(target.id)
        assert True


###########################
# Individual Action Tests #
###########################


class TestChargingAction:
    class ChargeSat(sats.Satellite):
        dyn_type = dyn.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [obs.Time()]
        action_spec = [act.Charge()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=ChargeSat(
            "Charger",
            sat_args=ChargeSat.default_sat_args(
                # High, inclined orbit makes eclipse unlikely
                oe=random_orbit(a=50000, i=90),
                batteryStorageCapacity=500_000,
                storedCharge_Init=250_000,
            ),
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        max_step_duration=300.0,
        time_limit=300.0,
        disable_env_checker=True,
    )

    def test_charging_action(self):
        self.env.reset()
        init_charge = self.env.unwrapped.satellite.dynamics.battery_charge
        self.env.step(0)  # Charge
        assert self.env.unwrapped.satellite.dynamics.battery_charge > init_charge


class TestDesatAction:
    class DesatSat(sats.Satellite):
        dyn_type = dyn.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [obs.Time()]
        action_spec = [act.Desat()]

    def make_env(self):
        return gym.make(
            "SatelliteTasking-v1",
            satellite=self.DesatSat(
                "Ellite",
                sat_args=self.DesatSat.default_sat_args(
                    oe=partial(random_circular_orbit, i=45),
                    wheelSpeeds=[1000.0, -1000.0, 1000.0],
                    nHat_B=np.array([0, 1, 0]),
                ),
            ),
            scenario=UniformTargets(n_targets=0),
            rewarder=data.NoReward(),
            sim_rate=1.0,
            max_step_duration=300.0,
            time_limit=1200.0,
            disable_env_checker=True,
        )

    def test_desat_action(self):
        env = self.make_env()
        env.reset()
        init_speeds = env.unwrapped.satellite.dynamics.wheel_speeds
        for _ in range(4):
            env.step(0)
            current_speeds = env.unwrapped.satellite.dynamics.wheel_speeds
            assert np.linalg.norm(current_speeds) < np.linalg.norm(init_speeds)
            init_speeds = current_speeds

    def test_desat_action_power_draw(self):
        env = self.make_env()
        env.unwrapped.satellite.sat_args_generator["thrusterPowerDraw"] = 0.0
        env.reset()
        env.step(0)  # Desat
        assert env.unwrapped.satellite.dynamics.battery_valid()

        env.unwrapped.satellite.sat_args_generator["thrusterPowerDraw"] = -10000.0
        env.reset()
        env.step(0)  # Desat
        assert not env.unwrapped.satellite.dynamics.battery_valid()

    def test_desat_action_pointing(self):
        env = self.make_env()
        env.unwrapped.satellite.sat_args_generator["desatAttitude"] = "nadir"
        env.reset(seed=0)
        env.step(0)  # Desat
        battery_level_nadir = env.unwrapped.satellite.dynamics.battery_charge

        env.unwrapped.satellite.sat_args_generator["desatAttitude"] = "sun"
        env.reset(seed=0)
        env.step(0)  # Desat
        battery_level_sun = env.unwrapped.satellite.dynamics.battery_charge
        assert battery_level_sun > battery_level_nadir


class TestNadirImagingActions:
    class ImageSat(sats.Satellite):
        dyn_type = dyn.ContinuousImagingDynModel
        fsw_type = fsw.ContinuousImagingFSWModel
        observation_spec = [obs.Time()]
        action_spec = [act.Scan()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=ImageSat(
            "EO-1",
            sat_args=ImageSat.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.05,
                imageRateErrorRequirement=0.05,
                instrumentBaudRate=1.0,
                dataStorageCapacity=3.0,
                transmitterBaudRate=-1.0,
            ),
        ),
        scenario=UniformNadirScanning(),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        time_limit=10000.0,
        max_step_duration=1e9,
        disable_env_checker=True,
    )

    def test_image(self):
        self.env.reset()
        storage_init = self.env.unwrapped.satellite.dynamics.storage_level
        self.env.step(0)
        assert self.env.unwrapped.satellite.dynamics.storage_level > storage_init
