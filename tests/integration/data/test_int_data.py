import gymnasium as gym

from bsk_rl import act, data, obs, sats, scene, sim
from bsk_rl.data.composition import ComposedReward
from bsk_rl.utils.orbital import random_orbit

# For data models not tested in other tests

# NoData sufficiently checked in many cases

# UniqueImageData sufficiently checked in test_int_communication

# from ..test_int_full_environments


class FullFeaturedSatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(dict(prop="r_BN_P", module="dynamics", norm=6e6)),
        obs.Time(),
    ]
    action_spec = [act.Image(n_ahead_image=10)]


def test_multi_rewarder():
    env = gym.make(
        "GeneralSatelliteTasking-v1",
        satellites=[
            FullFeaturedSatellite(
                "Sentinel-2A",
                sat_args=FullFeaturedSatellite.default_sat_args(
                    oe=random_orbit,
                    imageAttErrorRequirement=0.01,
                    imageRateErrorRequirement=0.01,
                ),
            ),
            FullFeaturedSatellite(
                "Sentinel-2B",
                sat_args=FullFeaturedSatellite.default_sat_args(
                    oe=random_orbit,
                    imageAttErrorRequirement=0.01,
                    imageRateErrorRequirement=0.01,
                ),
            ),
        ],
        scenario=scene.UniformTargets(n_targets=1000),
        rewarder=(data.UniqueImageReward(), data.UniqueImageReward()),
        sim_rate=0.5,
        max_step_duration=1e9,
        time_limit=5700.0,
        disable_env_checker=True,
    )

    assert isinstance(env.unwrapped.rewarder, ComposedReward)

    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())


class DriftSat(sats.Satellite):
    observation_spec = [obs.Time()]
    action_spec = [act.Drift(duration=100.0)]
    dyn_type = sim.dyn.BasicDynamicsModel
    fsw_type = sim.fsw.BasicFSWModel


def test_time_penalty():
    env = gym.make(
        "SatelliteTasking-v1",
        satellite=DriftSat(
            "DriftSat",
        ),
        rewarder=data.ResourceReward(
            reward_weight=-0.1,
            resource_fn=lambda sat: sat.simulator.sim_time,
        ),
        disable_env_checker=True,
    )

    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert reward == -10.0
