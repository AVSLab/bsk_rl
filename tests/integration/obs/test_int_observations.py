import gymnasium as gym
import numpy as np
from pytest import approx

from bsk_rl import act, data, obs, sats
from bsk_rl.scene import UniformTargets
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit


##############################
# Composed Observation Tests #
##############################
class TestComposedState:
    class ComposedPropSat(sats.ImagingSatellite):
        dyn_type = dyn.ImagingDynModel
        fsw_type = fsw.ImagingFSWModel
        observation_spec = [
            obs.Time(),
            obs.SatProperties(
                dict(prop="r_BN_N", module="dynamics"),
                dict(prop="r_BN_N", norm=7000.0 * 1e3),
            ),
            obs.OpportunityProperties(dict(prop="priority"), n_ahead_observe=2),
            obs.Eclipse(),
        ]
        action_spec = [act.Drift()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=ComposedPropSat(
            "Explorer 1",
            sat_args=ComposedPropSat.default_sat_args(oe=random_orbit),
        ),
        scenario=UniformTargets(n_targets=1000),
        rewarder=data.UniqueImageReward(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
    )

    def test_normd_property_state(self):
        observation, info = self.env.reset()
        assert observation[0] == 0.0  # Timed observation should be first
        self.env.unwrapped.satellite.observation_builder.obs_type = dict
        eclipse = self.env.unwrapped.satellite.get_obs()["eclipse"]
        assert observation[-2] == eclipse[0]  # Eclipse should be last
        assert observation[-1] == eclipse[1]  # Eclipse should be last


################################
# Individual Observation Tests #
################################


class TestSatProperties:
    class SatPropertiesSat(sats.Satellite):
        dyn_type = dyn.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [
            obs.SatProperties(
                dict(prop="r_BN_N", module="dynamics"),
                dict(prop="r_BN_N", norm=7000.0 * 1e3),
                dict(prop="inclination", module="dynamics", norm=np.pi / 180),
            ),
        ]
        action_spec = [act.Drift()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=SatPropertiesSat(
            "Sputnik",
            sat_args=SatPropertiesSat.default_sat_args(oe=random_orbit(a=7000, i=45)),
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
    )

    def test_normd_property_state(self):
        observation, info = self.env.reset()
        assert np.linalg.norm(observation[0:3]) == approx(7000.0 * 1e3)
        assert np.linalg.norm(observation[3:6]) == approx(1.0)
        assert observation[6] == approx(45)
        observation, reward, terminated, truncated, info = self.env.step(0)
        assert observation[6] == approx(45)


class TestTime:
    class TimedSat(sats.Satellite):
        dyn_type = dyn.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [obs.Time()]
        action_spec = [act.Drift()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=TimedSat(
            "Voyager",
            sat_args=TimedSat.default_sat_args(oe=random_orbit()),
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
    )

    def test_normd_property_state(self):
        observation, info = self.env.reset()
        assert observation[0] == 0.0
        observation, reward, terminated, truncated, info = self.env.step(0)
        assert observation[0] == 0.1


class TestOpportunityProperties:
    class TargetSat(sats.ImagingSatellite):
        dyn_type = dyn.ImagingDynModel
        fsw_type = fsw.ImagingFSWModel
        observation_spec = [
            obs.OpportunityProperties(dict(prop="priority"), n_ahead_observe=2)
        ]
        action_spec = [act.Drift()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=TargetSat(
            "Bullseye",
            obs_type=dict,
            sat_args=TargetSat.default_sat_args(oe=random_orbit()),
        ),
        scenario=UniformTargets(n_targets=100),
        rewarder=data.UniqueImageReward(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
    )

    def test_target_state(self):
        observation, info = self.env.reset()
        assert "target_1" in observation["target"]
        assert "priority" in observation["target"]["target_1"]


class TestEclipse:
    class EclipseSat(sats.Satellite):
        dyn_type = dyn.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [obs.Eclipse()]
        action_spec = [act.Drift()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=EclipseSat(
            "PinkFloyd",
            obs_type=list,
            sat_args=EclipseSat.default_sat_args(oe=random_orbit()),
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        max_step_duration=5700.0,
        time_limit=5800.0,
        disable_env_checker=True,
    )

    def test_eclipse_state(self):
        observation1, info = self.env.reset()
        observation2, reward, terminated, truncated, info = self.env.step(0)
        assert (observation2[0] - observation1[0]) < 0.05
        assert (observation2[1] - observation1[1]) < 0.05


class TestGroundStationProperties:
    class GroundSat(sats.AccessSatellite):
        dyn_type = dyn.GroundStationDynModel
        fsw_type = fsw.ImagingFSWModel
        observation_spec = [
            obs.OpportunityProperties(
                dict(
                    prop="opportunity_open",
                ),
                n_ahead_observe=2,
                type="ground_station",
            ),
        ]
        action_spec = [act.Drift()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=GroundSat(
            "Satellite",
            obs_type=list,
            sat_args=GroundSat.default_sat_args(oe=random_orbit()),
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        max_step_duration=5700.0,
        time_limit=5700.0,
        disable_env_checker=True,
    )

    def test_ground_station_state(self):
        observation, info = self.env.reset()
        assert sum(observation) > 0  # Check that there are downlink opportunities


class TestResourceRewardWeight:
    class ResourceSat(sats.ImagingSatellite):
        dyn_type = dyn.ImagingDynModel
        fsw_type = fsw.ImagingFSWModel
        observation_spec = [obs.ResourceRewardWeight()]
        action_spec = [act.Drift()]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=ResourceSat(
            "ResourceSat",
            obs_type=dict,
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=(
            data.ResourceReward(
                resource_fn=lambda sat: sat.simulator.sim_time, reward_weight=0.1
            ),
            data.ResourceReward(
                resource_fn=lambda sat: sat.simulator.sim_time, reward_weight=1.0
            ),
        ),
        sim_rate=1.0,
        max_step_duration=10.0,
        disable_env_checker=True,
    )

    def test_resource_reward_weight(self):
        observation, info = self.env.reset()
        observation, reward, terminated, truncated, info = self.env.step(0)
        assert reward == 1.0 + 10.0
        assert (observation["resource_reward_weight"] == [0.1, 1.0]).all()
