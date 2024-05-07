import gymnasium as gym
import numpy as np
from pytest import approx

from bsk_rl.env.scenario import actions as act
from bsk_rl.env.scenario import data
from bsk_rl.env.scenario import observations as obs
from bsk_rl.env.scenario import satellites as sats
from bsk_rl.env.scenario.environment_features import StaticTargets
from bsk_rl.env.simulation import dynamics, fsw
from bsk_rl.utils.orbital import random_orbit


##############################
# Composed Observation Tests #
##############################
class TestComposedState:
    class ComposedPropSat(
        act.DriftAction,
        sats.ImagingSatellite,
    ):
        dyn_type = dynamics.ImagingDynModel
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

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=ComposedPropSat(
            "Explorer 1",
            sat_args=ComposedPropSat.default_sat_args(oe=random_orbit),
        ),
        env_features=StaticTargets(n_targets=1000),
        data_manager=data.NoDataManager(),
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
    class SatPropertiesSat(act.DriftAction):
        dyn_type = dynamics.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [
            obs.SatProperties(
                dict(prop="r_BN_N", module="dynamics"),
                dict(prop="r_BN_N", norm=7000.0 * 1e3),
            ),
        ]

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=SatPropertiesSat(
            "Sputnik",
            sat_args=SatPropertiesSat.default_sat_args(
                oe=random_orbit(r_body=7000, alt=0)
            ),
        ),
        env_features=StaticTargets(n_targets=0),
        data_manager=data.NoDataManager(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
    )

    def test_normd_property_state(self):
        observation, info = self.env.reset()
        assert np.linalg.norm(observation[0:3]) == approx(7000.0 * 1e3)
        assert np.linalg.norm(observation[3:6]) == approx(1.0)


class TestTime:
    class TimedSat(act.DriftAction):
        dyn_type = dynamics.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [obs.Time()]

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=TimedSat(
            "Voyager",
            sat_args=TimedSat.default_sat_args(oe=random_orbit()),
        ),
        env_features=StaticTargets(n_targets=0),
        data_manager=data.NoDataManager(),
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
    class TargetSat(act.DriftAction, sats.ImagingSatellite):
        dyn_type = dynamics.ImagingDynModel
        fsw_type = fsw.ImagingFSWModel
        observation_spec = [
            obs.OpportunityProperties(dict(prop="priority"), n_ahead_observe=2)
        ]

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=TargetSat(
            "Bullseye",
            obs_type=dict,
            sat_args=TargetSat.default_sat_args(oe=random_orbit()),
        ),
        env_features=StaticTargets(n_targets=100),
        data_manager=data.NoDataManager(),
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
    class EclipseSat(act.DriftAction):
        dyn_type = dynamics.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel
        observation_spec = [obs.Eclipse()]

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=EclipseSat(
            "PinkFloyd",
            obs_type=list,
            sat_args=EclipseSat.default_sat_args(oe=random_orbit()),
        ),
        env_features=StaticTargets(n_targets=0),
        data_manager=data.NoDataManager(),
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
    class GroundSat(act.DriftAction, sats.AccessSatellite):
        dyn_type = dynamics.GroundStationDynModel
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

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=GroundSat(
            "Satellite",
            obs_type=list,
            sat_args=GroundSat.default_sat_args(oe=random_orbit()),
        ),
        env_features=StaticTargets(n_targets=0),
        data_manager=data.NoDataManager(),
        sim_rate=1.0,
        max_step_duration=5700.0,
        time_limit=5700.0,
        disable_env_checker=True,
    )

    def test_ground_station_state(self):
        observation, info = self.env.reset()
        assert sum(observation) > 0  # Check that there are downlink opportunities
