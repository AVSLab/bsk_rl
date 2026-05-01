import gymnasium as gym

from bsk_rl import act, data, obs, sats
from bsk_rl.scene import CityTargets, UniformTargets
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit


def make_env(scenario):
    class ImageSat(sats.ImagingSatellite):
        dyn_type = dyn.GroundStationDynModel
        fsw_type = fsw.ImagingFSWModel
        observation_spec = [obs.Time()]
        action_spec = [act.Image(n_ahead_image=10)]

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=ImageSat(
            "EO-1",
            sat_args=ImageSat.default_sat_args(
                oe=random_orbit,
                imageAttErrorRequirement=0.05,
                imageRateErrorRequirement=0.05,
            ),
        ),
        scenario=scenario,
        rewarder=data.UniqueImageReward(),
        sim_rate=1.0,
        time_limit=5700.0,
        max_step_duration=1e9,
        disable_env_checker=True,
        failure_penalty=0,
    )
    return env


class TestUniformTargets:
    def test_priority_dist(self):
        env = make_env(UniformTargets(n_targets=1000, priority_distribution=lambda: 1))
        env.reset()
        for i in range(10):
            observation, reward, terminated, truncated, info = env.step(i)
            assert reward in [0.0, 1.0]


class TestCityTargets:
    def test_city_distribution(self):
        # Just a smoke test
        env = make_env(
            CityTargets(n_targets=1000, n_select_from=10000, location_offset=1e3)
        )
        env.reset()
        reward_sum = 0.0
        for i in range(10):
            observation, reward, terminated, truncated, info = env.step(i)
            reward_sum += reward
        assert reward_sum > 0
