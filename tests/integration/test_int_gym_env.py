from typing import ClassVar

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from bsk_rl import ConstellationTasking, act, data, obs, sats
from bsk_rl.gym import NO_ACTION
from bsk_rl.scene import UniformTargets
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit


class DoNothingSatellite(sats.ImagingSatellite):
    observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
    action_spec: ClassVar[list[act.Action]] = [act.Drift()]


class HybridModeSatellite(sats.Satellite):
    dyn_type = dyn.BasicDynamicsModel
    fsw_type = fsw.BasicFSWModel
    observation_spec: ClassVar[list[obs.Observation]] = [obs.Time()]
    action_spec: ClassVar[list[act.Action]] = [
        act.Charge(hybrid=True),
        act.Drift(hybrid=True),
    ]


@pytest.fixture
def make_hybrid_constellation():
    environments = []

    def make_env(n_satellites=2, only_retask_idle=False):
        names = [f"hybrid_{i}" for i in range(n_satellites)]
        env = ConstellationTasking(
            satellites=[HybridModeSatellite(name) for name in names],
            scenario=UniformTargets(n_targets=0),
            rewarder=data.NoReward(),
            meta_agent_groupings={"group": names},
            only_retask_idle_meta_agent_members=only_retask_idle,
            sim_rate=1.0,
            max_step_duration=100.0,
            time_limit=100.0,
        )
        environments.append(env)
        return env

    yield make_env

    for env in environments:
        env.close()


@pytest.mark.parametrize("n_satellites", [1, 2])
def test_constellation_sampled_hybrid_actions(make_hybrid_constellation, n_satellites):
    env = make_hybrid_constellation(n_satellites=n_satellites)
    env.reset(seed=0)
    space = env.action_space("group")
    space.seed(42)
    action = space.sample()

    env.step({"group": action})

    assert env.simulator.sim_time > 0
    member_actions = (action,) if n_satellites == 1 else action
    for satellite, member_action in zip(
        env.meta_agent_groupings["group"], member_actions
    ):
        assert satellite.action_builder.prev_action_key == (
            "action_charge" if member_action[0] == 0 else "action_drift"
        )


@pytest.mark.parametrize("skip_first", [False, True])
def test_constellation_hybrid_object_array_actions(
    make_hybrid_constellation, skip_first
):
    env = make_hybrid_constellation()
    env.reset(seed=0)
    first, second = env.meta_agent_groupings["group"]
    action = np.empty(2, dtype=object)
    action[0] = None if skip_first else (0, np.array([2.0]))
    action[1] = (1, np.array([5.0]))

    env.step({"group": action})

    assert env.simulator.sim_time == pytest.approx(5.0 if skip_first else 2.0)
    assert first.action_builder.prev_action_key == (
        None if skip_first else "action_charge"
    )
    assert second.action_builder.prev_action_key == "action_drift"


def test_constellation_hybrid_idle_filter_preserves_deadline(make_hybrid_constellation):
    env = make_hybrid_constellation(only_retask_idle=True)
    env.reset(seed=0)
    first, second = env.meta_agent_groupings["group"]

    env.step({"group": ((0, np.array([2.0])), (1, np.array([5.0])))})

    assert env.simulator.sim_time == pytest.approx(2.0)
    assert first.requires_retasking
    assert not second.requires_retasking

    env.step({"group": ((1, np.array([8.0])), (0, np.array([10.0])))})

    assert env.simulator.sim_time == pytest.approx(5.0)
    assert first.action_builder.prev_action_key == "action_drift"
    assert second.action_builder.prev_action_key == "action_drift"
    assert not first.requires_retasking
    assert second.requires_retasking


@pytest.mark.parametrize(
    "idle_action",
    [None, NO_ACTION, ((NO_ACTION, np.array([NO_ACTION], dtype=np.float32)),) * 2],
    ids=["none", "scalar_sentinel", "nested_sentinel"],
)
def test_constellation_hybrid_wait_preserves_running_tasks(
    make_hybrid_constellation, idle_action
):
    env = make_hybrid_constellation()
    env.reset(seed=0)
    first, second = env.meta_agent_groupings["group"]
    env.step({"group": ((0, np.array([2.0])), (1, np.array([5.0])))})
    assert env.simulator.sim_time == pytest.approx(2.0)

    env.step({"group": idle_action})

    assert env.simulator.sim_time == pytest.approx(5.0)
    assert first.action_builder.prev_action_key == "action_charge"
    assert second.action_builder.prev_action_key == "action_drift"
    assert first.requires_retasking
    assert second.requires_retasking


class TestSatelliteTasking:
    env = gym.make(
        "SatelliteTasking-v1",
        satellite=DoNothingSatellite(
            "Sputnik",
            sat_args=DoNothingSatellite.default_sat_args(oe=random_orbit),
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
        dtype=np.float32,
    )

    def test_reset(self):
        observation, _info = self.env.reset()
        assert (observation == np.array([0.0])).all()

    def test_action_space(self):
        assert self.env.action_space == spaces.Discrete(1)

    def test_observation_space(self):
        assert self.env.observation_space == spaces.Box(
            -1e16, 1e16, (1,), dtype=np.float32
        )

    def test_step(self):
        observation, _reward, _terminated, _truncated, _info = self.env.step(0)
        assert (observation == np.array([0.1])).all()

    def test_truncate(self):
        terminated = truncated = False
        while not (terminated or truncated):
            _observation, _reward, terminated, truncated, _info = self.env.step(0)
        assert truncated
        assert self.env.unwrapped.simulator.sim_time == 100.0

    def test_repeatable(self):
        self.env.reset(seed=0)
        world_args_old = self.env.unwrapped.world_args
        sat_args_old = self.env.unwrapped.satellite.sat_args
        self.env.reset(seed=0)
        assert self.env.unwrapped.world_args == world_args_old
        for val, val_old in zip(
            self.env.unwrapped.satellite.sat_args.values(), sat_args_old.values()
        ):
            if (
                isinstance(val, (np.ndarray, list, type(None)))
                or np.issubdtype(type(val), np.integer)
                or np.issubdtype(type(val), float)
            ):
                assert np.all(val == val_old)


class TestSingleSatelliteDeath:
    env = gym.make(
        "SatelliteTasking-v1",
        satellite=DoNothingSatellite(
            "Skydiver",
            sat_args=DoNothingSatellite.default_sat_args(
                rN=[0, 0, 7e6], vN=[0, 0, -100.0], oe=None
            ),
        ),
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        time_limit=1000.0,
        failure_penalty=-1000,
        disable_env_checker=True,
    )

    def test_fail(self):
        _, _ = self.env.reset()
        _observation, reward, terminated, _truncated, _info = self.env.step(0)
        assert terminated
        assert reward == -1000

    def test_close(self):
        self.env.close()


class TestGeneralSatelliteTasking:
    env = gym.make(
        "GeneralSatelliteTasking-v1",
        satellites=[
            DoNothingSatellite(
                "Sentinel_2A",
                sat_args=DoNothingSatellite.default_sat_args(oe=random_orbit),
            ),
            DoNothingSatellite(
                "Sentinel_2B",
                sat_args=DoNothingSatellite.default_sat_args(oe=random_orbit),
            ),
        ],
        scenario=UniformTargets(n_targets=0),
        rewarder=data.NoReward(),
        sim_rate=1.0,
        max_step_duration=10.0,
        time_limit=100.0,
        disable_env_checker=True,
        dtype=np.float32,
    )

    def test_reset(self):
        observation, _info = self.env.reset()
        assert (observation == np.array([0.0])).all()

    def test_action_space(self):
        assert self.env.action_space == spaces.Tuple(
            (spaces.Discrete(1), spaces.Discrete(1))
        )

    def test_observation_space(self):
        assert self.env.observation_space == spaces.Tuple(
            (
                spaces.Box(-1e16, 1e16, (1,), dtype=np.float32),
                spaces.Box(-1e16, 1e16, (1,), dtype=np.float32),
            )
        )

    def test_step(self):
        observation, _reward, _terminated, _truncated, _info = self.env.step([0, 0])
        assert (observation == np.array([0.1])).all()

    def test_truncate(self):
        terminated = truncated = False
        while not (terminated or truncated):
            _observation, _reward, terminated, truncated, _info = self.env.step([0, 0])
        assert truncated
        assert self.env.unwrapped.simulator.sim_time == 100.0
