import gymnasium as gym
import numpy as np
from pytest import approx

from bsk_rl.env.scenario import data
from bsk_rl.env.scenario import sat_actions as sa
from bsk_rl.env.scenario import sat_observations as so
from bsk_rl.env.scenario import satellites as sats
from bsk_rl.env.scenario.environment_features import StaticTargets
from bsk_rl.env.simulation import dynamics, fsw
from bsk_rl.utils.orbital import random_orbit


##############################
# Composed Observation Tests #
##############################
class TestComposedState:
    class ComposedPropSat(
        sa.DriftAction,
        so.EclipseState,
        so.TargetState.configure(n_ahead_observe=2),
        so.NormdPropertyState.configure(
            obs_properties=[
                dict(prop="r_BN_N", module="dynamics"),
                dict(prop="r_BN_N", norm=7000.0 * 1e3),
            ]
        ),
        so.TimeState,
        sats.ImagingSatellite,
    ):
        dyn_type = dynamics.ImagingDynModel
        fsw_type = fsw.ImagingFSWModel

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
        eclipse = self.env.satellite.obs_dict["eclipse_state"]
        assert observation[-2] == eclipse[0]  # Eclipse should be last
        assert observation[-1] == eclipse[1]  # Eclipse should be last


################################
# Individual Observation Tests #
################################


class TestNormdPropertyState:
    class NormdPropSat(
        sa.DriftAction,
        so.NormdPropertyState.configure(
            obs_properties=[
                dict(prop="r_BN_N", module="dynamics"),
                dict(prop="r_BN_N", norm=7000.0 * 1e3),
            ]
        ),
    ):
        dyn_type = dynamics.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=NormdPropSat(
            "Sputnik",
            sat_args=NormdPropSat.default_sat_args(oe=random_orbit(r_body=7000, alt=0)),
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


class TestTimeState:
    class TimedSat(
        sa.DriftAction,
        so.TimeState,
    ):
        dyn_type = dynamics.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel

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


class TestTargetState:
    class TargetSat(
        sa.DriftAction,
        so.TargetState,
    ):
        dyn_type = dynamics.ImagingDynModel
        fsw_type = fsw.ImagingFSWModel

    env = gym.make(
        "SingleSatelliteTasking-v1",
        satellites=TargetSat(
            "Bullseye",
            obs_type=dict,
            n_ahead_observe=2,
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
        assert "target_1" in observation["target_obs"]
        assert "priority" in observation["target_obs"]["target_1"]
        assert "location_normd" in observation["target_obs"]["target_1"]


class TestEclipseState:
    class EclipseSat(
        sa.DriftAction,
        so.EclipseState,
    ):
        dyn_type = dynamics.BasicDynamicsModel
        fsw_type = fsw.BasicFSWModel

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


class TestGroundStationState:
    class GroundSat(
        sa.DriftAction, so.GroundStationState.configure(n_ahead_observe_downlinks=2)
    ):
        dyn_type = dynamics.GroundStationDynModel
        fsw_type = fsw.ImagingFSWModel

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
