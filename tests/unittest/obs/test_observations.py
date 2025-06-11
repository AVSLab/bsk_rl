from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from gymnasium import spaces

from bsk_rl import obs
from bsk_rl.obs.observations import ObservationBuilder, _target_angle


class TestObservationBuilder:
    def test_init(self):
        names = ["obs_A", "obs_B", "obs_A"]
        observation_spec = [MagicMock() for _ in names]
        for os, name in zip(observation_spec, names):
            os.name = name

        ob = ObservationBuilder(satellite=MagicMock(observation_spec=observation_spec))
        assert len(ob.observation_spec) == 3
        assert ob.observation_spec[2].name == "obs_A_2"

    def test_reset_post_sim_init(self):
        ob = ObservationBuilder(
            satellite=MagicMock(observation_spec=[MagicMock() for _ in range(3)])
        )
        for os in ob.observation_spec:
            os.link_satellite.assert_called_once()

    def make_mocked_observation(self, obs_type=np.ndarray):
        sat = MagicMock(simulator=MagicMock(sim_time=0.0))
        observation_spec = [
            MagicMock(get_obs=MagicMock(return_value=i)) for i in range(3)
        ]
        for i, os in enumerate(observation_spec):
            os.name = f"obs_{i}"
        sat.observation_spec = observation_spec
        ob = ObservationBuilder(sat, obs_type=obs_type)
        ob.simulator = sat.simulator
        return ob

    def test_get_obs_np(self):
        ob = self.make_mocked_observation(obs_type=np.ndarray)
        assert np.isclose(ob.get_obs(), np.array([0, 1, 2])).all()

    def test_get_obs_list(self):
        ob = self.make_mocked_observation(obs_type=list)
        assert ob.get_obs() == [0, 1, 2]

    def test_get_obs_dict(self):
        ob = self.make_mocked_observation(obs_type=dict)
        assert ob.get_obs() == {"obs_0": 0, "obs_1": 1, "obs_2": 2}

    def test_obs_array_keys(self):
        ob = self.make_mocked_observation()
        assert ob.obs_array_keys() == ["obs_0", "obs_1", "obs_2"]

    def test_obs_cache(self):
        ob = self.make_mocked_observation()
        ob.get_obs()
        # Don't step time forward, expect cached value
        ob.observation_spec[0].get_obs.return_value = 1
        assert ob.get_obs()[0] == 0
        # Step time forward, expect new value
        ob.satellite.simulator.sim_time = 1.0
        assert ob.get_obs()[0] == 1

    @pytest.mark.parametrize(
        "observation,dtype,space",
        [
            (
                np.array([1]),
                np.float32,
                spaces.Box(low=-1e16, high=1e16, shape=(1,), dtype=np.float32),
            ),
            (
                np.array([1, 2]),
                np.float64,
                spaces.Box(low=-1e16, high=1e16, shape=(2,), dtype=np.float64),
            ),
            (
                {"a": 1, "b": {"c": 1}},
                np.float32,
                spaces.Dict(
                    {
                        "a": spaces.Box(
                            low=-1e16, high=1e16, shape=(1,), dtype=np.float32
                        ),
                        "b": spaces.Dict(
                            {
                                "c": spaces.Box(
                                    low=-1e16, high=1e16, shape=(1,), dtype=np.float32
                                )
                            }
                        ),
                    }
                ),
            ),
        ],
    )
    def test_obs_space(self, observation, dtype, space):
        ob = ObservationBuilder(MagicMock(), dtype=dtype)
        ob.get_obs = MagicMock(return_value=observation)
        assert ob.observation_space == space


class TestSatProperties:
    def test_init(self):
        ob = obs.SatProperties(
            dict(prop="hello", module="dynamics", name="hello_prop"),
            dict(prop="world", norm=2.0),
        )
        assert ob.obs_properties[0]["norm"] == 1.0
        assert ob.obs_properties[1]["name"] == "world_normd"

    def test_id_property(self):
        ob = obs.SatProperties(dict(prop="world"))
        ob.satellite = MagicMock(dynamics=Mock(), fsw=Mock(world=1.0))
        del ob.satellite.dynamics.world
        ob.reset_post_sim_init()
        assert ob.obs_properties[0]["module"] == "fsw"

    def test_get_obs(self):
        ob = obs.SatProperties(
            dict(prop="hello", module="dynamics", name="hello_prop", norm=2.0),
        )
        ob.satellite = MagicMock(dynamics=MagicMock(hello=6.0))
        assert ob.get_obs() == {"hello_prop": 3.0}

    def test_get_obs_fn(self):
        mock_fn = MagicMock(return_value=3.0)
        ob = obs.SatProperties(
            dict(prop="hello", fn=mock_fn),
        )
        ob.satellite = MagicMock()
        assert ob.get_obs() == {"hello": 3.0}
        mock_fn.assert_called_once_with(ob.satellite)


class TestTime:
    def test_detect_norm(self):
        ob = obs.Time()
        ob.simulator = MagicMock(sim_time=10.0, time_limit=100.0)
        ob.reset_post_sim_init()
        assert ob.get_obs() == 0.1

    def test_manual_norm(self):
        ob = obs.Time(norm=100.0)
        ob.simulator = MagicMock(sim_time=10.0, time_limit=300.0)
        ob.reset_post_sim_init()
        assert ob.get_obs() == 0.1

    def test_time_remaining(self):
        ob = obs.Time(observe_time_remaining=True)
        ob.simulator = MagicMock(sim_time=10.0, time_limit=100.0)
        ob.reset_post_sim_init()
        assert ob.get_obs() == 0.9


class TestOpportunityProperties:
    def test_fns(self):
        fns = obs.OpportunityProperties._fn_map
        sat = MagicMock(simulator=MagicMock(sim_time=10.0))
        opp = dict(
            type="target",
            object=MagicMock(priority=1.0),
            window=[20.0, 30.0],
            r_LP_P=1.0,
        )
        assert fns["priority"](sat, opp) == 1.0
        assert fns["r_LP_P"](sat, opp) == 1.0
        assert fns["opportunity_open"](sat, opp) == 10.0
        assert fns["opportunity_close"](sat, opp) == 20.0
        assert fns["opportunity_mid"](sat, opp) == 15.0

    def test_target_angle(self):
        sat = MagicMock(
            dynamics=MagicMock(r_BN_P=np.array([1.0, 0.0, 0.0])),
            fsw=MagicMock(c_hat_P=np.array([0.0, 1.0, 0.0])),
        )
        opp = dict(r_LP_P=np.array([0.0, 0.0, 0.0]))
        assert np.isclose(_target_angle(sat, opp), np.pi / 2)

    def test_init(self):
        ob = obs.OpportunityProperties(
            dict(prop="r_LP_P", norm=2.0),
            dict(
                prop="double_priority", fn=lambda sat, opp: opp["target"].priority * 2.0
            ),
            n_ahead_observe=2,
        )
        assert ob.target_properties[0]["fn"]
        assert ob.target_properties[0]["name"] == "r_LP_P_normd"
        assert ob.target_properties[1]["norm"] == 1.0

    def test_get_obs(self):
        ob = obs.OpportunityProperties(
            dict(prop="priority", norm=2.0),
            dict(
                prop="double_priority", fn=lambda sat, opp: opp["object"].priority * 2.0
            ),
            n_ahead_observe=2,
        )
        ob.satellite = MagicMock()
        ob.satellite.find_next_opportunities.return_value = [
            {"object": MagicMock(priority=1.0), "type": "target"}
        ] * ob.n_ahead_observe
        assert ob.get_obs() == {
            "target_0": {"priority_normd": 0.5, "double_priority": 2.0},
            "target_1": {"priority_normd": 0.5, "double_priority": 2.0},
        }

    def test_init_bad(self):
        with pytest.raises(ValueError):
            obs.OpportunityProperties(dict(prop="not_a_prop"), n_ahead_observe=2)


class TestEclipse:
    def test_obs(self):
        ob = obs.Eclipse(norm=100.0)
        ob.simulator = MagicMock(sim_time=10.0)
        ob.satellite = MagicMock()
        ob.satellite.trajectory.next_eclipse.return_value = (20.0, 30.0)
        assert ob.get_obs() == [0.1, 0.2]


class TestResourceRewardWeight:
    def test_obs(self):
        ob = obs.ResourceRewardWeight()
        ob.reset_overwrite_previous()
        ob.weight_vector.append(1.0)
        assert ob.get_obs() == [1.0]
