from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bsk_rl.envs.general_satellite_tasking.scenario import sat_observations as so


@patch.multiple(so.SatObservation, __abstractmethods__=set())
@patch("bsk_rl.envs.general_satellite_tasking.scenario.satellites.Satellite.__init__")
class TestSatObservation:
    def test_init(self, sat_init):
        so.SatObservation()
        sat_init.assert_called_once()

    def make_mocked_observation(self, **kwargs):
        sat = so.SatObservation(**kwargs)
        sat.simulator = MagicMock(sim_time=0.0)
        obs_fun = MagicMock(return_value=0, __name__="obs")
        sat.add_to_observation(obs_fun)
        return sat

    def test_obs_dict(self, sat_init):
        sat = self.make_mocked_observation()
        assert sat.obs_dict == {"obs": 0}

    def test_obs_dict_cache(self, sat_init):
        sat = self.make_mocked_observation()
        sat.obs_dict
        # Don't step time forward, expect cached value
        sat.obs_fn_list[0].return_value = 1
        assert sat.obs_dict == {"obs": 0}
        # Don't step time forward, expect new value
        sat.simulator.sim_time = 1.0
        assert sat.obs_dict == {"obs": 1}

    @pytest.mark.parametrize(
        "obs_type,observation",
        [(dict, {"obs": 0}), (np.ndarray, np.array([0])), (list, [0]), (int, None)],
    )
    def test_get_obs(self, sat_init, obs_type, observation):
        sat = self.make_mocked_observation(obs_type=obs_type)
        if observation is None:
            with pytest.raises(ValueError):
                sat.get_obs()
        else:
            assert sat.get_obs() == observation


@patch.multiple(so.NormdPropertyState, __abstractmethods__=set())
@patch("bsk_rl.envs.general_satellite_tasking.scenario.satellites.Satellite.__init__")
class TestNormdPropertyState:
    def test_init(self, sat_init):
        sat = so.NormdPropertyState(obs_properties=[dict(prop="some_prop")])
        sat_init.assert_called_once()
        assert len(sat.obs_fn_list) == 1
        assert sat.obs_fn_list[0].__name__ == "some_prop"

    def make_mocked_sat(self, **kwargs):
        sat = so.NormdPropertyState(**kwargs)
        sat.simulator = MagicMock(sim_time=0.0)
        sat.dynamics = MagicMock([], some_prop=0.0)
        sat.fsw = MagicMock([], other_prop=1.0)
        return sat

    def test_add_prop_function_with_module(self, sat_init):
        sat = self.make_mocked_sat()
        sat.add_prop_function("some_prop", module="dynamics")
        sat.add_prop_function("other_prop", module="fsw")
        assert sat.obs_dict == {"some_prop": 0.0, "other_prop": 1.0}

    def test_add_prop_function_no_module(self, sat_init):
        sat = self.make_mocked_sat()
        sat.add_prop_function("some_prop")
        sat.add_prop_function("other_prop")
        assert sat.obs_dict == {"some_prop": 0.0, "other_prop": 1.0}

    def test_add_prop_function_norm(self, sat_init):
        sat = self.make_mocked_sat()
        sat.add_prop_function("other_prop", norm=10.0)
        assert sat.obs_dict == {"other_prop_normd": 0.1}

    def test_add_bad_prop(self, sat_init):
        sat = self.make_mocked_sat()
        del sat.dynamics.not_a_prop
        del sat.fsw.not_a_prop
        sat.add_prop_function("not_a_prop")
        with pytest.raises(AttributeError):
            sat.obs_dict


@patch.multiple(so.TimeState, __abstractmethods__=set())
@patch("bsk_rl.envs.general_satellite_tasking.scenario.satellites.Satellite.__init__")
class TestTimeState:
    def test_init(self, sat_init):
        sat = so.TimeState()
        sat_init.assert_called_once()
        assert len(sat.obs_fn_list) == 1
        assert sat.obs_fn_list[0].__name__ == "normalized_time"

    def test_explicit_normalization(self, sat_init):
        sat = so.TimeState(normalization_time=10.0)
        sat.simulator = MagicMock(sim_time=1.0)
        sat.reset_post_sim()
        assert sat.normalized_time() == 0.1

    def test_implicit_normalization(self, sat_init):
        sat = so.TimeState(normalization_time=None)
        sat.simulator = MagicMock(sim_time=1.0, time_limit=10.0)
        sat.reset_post_sim()
        assert sat.normalized_time() == 0.1


@patch.multiple(so.TargetState, __abstractmethods__=set())
@patch(
    "bsk_rl.envs.general_satellite_tasking.scenario.satellites.ImagingSatellite.__init__"
)
class TestTargetState:
    def test_init(self, sat_init):
        sat = so.TargetState()
        sat_init.assert_called_once()
        assert len(sat.obs_fn_list) == 1
        assert sat.obs_fn_list[0].__name__ == "target_obs"

    def test_target_state(self, sat_init):
        n_ahead = 2
        sat = so.TargetState(n_ahead_observe=n_ahead)
        sat.find_next_opportunities = MagicMock(
            return_value=[
                dict(target=MagicMock(priority=i, location=np.array([0.0, 0.0, 0.0])))
                for i in range(n_ahead)
            ]
        )
        expected = dict(
            target_0=dict(priority=0.0, location_normd=np.array([0.0, 0.0, 0.0])),
            target_1=dict(priority=1.0, location_normd=np.array([0.0, 0.0, 0.0])),
        )
        for k1, v1 in sat.target_obs().items():
            for k2, v2 in v1.items():
                assert np.all(v2 == expected[k1][k2])

    def test_target_state_normed(self, sat_init):
        n_ahead = 2
        sat = so.TargetState(
            n_ahead_observe=n_ahead,
            target_properties=[
                dict(prop="priority"),
                dict(prop="location", norm=10.0),
                dict(prop="window_open", norm=10.0),
                dict(prop="window_mid"),
                dict(prop="window_close"),
                dict(prop="target_angle"),
            ],
        )

        sat.upcoming_targets = MagicMock(
            return_value=[
                MagicMock(priority=i, location=np.array([i, 1.0, 0.0]))
                for i in range(n_ahead)
            ]
        )
        sat.opportunities = [
            dict(target=target, window=(10.0, 20.0), type="target")
            for target in sat.upcoming_targets()
        ]
        sat.simulator = MagicMock(sim_time=5.0)
        sat.dynamics = MagicMock(r_BN_P=np.array([5.0, 5.0, 0.0]))
        sat.fsw = MagicMock(c_hat_P=np.array([0.0, 1.0, 0.0]))

        vector_target_sat = [
            sat.upcoming_targets()[i].location - sat.dynamics.r_BN_P
            for i in range(n_ahead)
        ]
        vector_target_sat_hat = [
            vector_target_sat[i] / np.linalg.norm(vector_target_sat[i])
            for i in range(n_ahead)
        ]

        expected = dict(
            target_0=dict(
                priority=0.0,
                location_normd=np.array([0.0, 0.1, 0.0]),
                window_open_normd=5.0 / 10.0,
                window_mid=10.0,
                window_close=15.0,
                target_angle=np.arccos(
                    np.dot(vector_target_sat_hat[0], sat.fsw.c_hat_P)
                ),
            ),
            target_1=dict(
                priority=1.0,
                location_normd=np.array([0.1, 0.1, 0.0]),
                window_open_normd=5.0 / 10.0,
                window_mid=10.0,
                window_close=15.0,
                target_angle=np.arccos(
                    np.dot(vector_target_sat_hat[1], sat.fsw.c_hat_P)
                ),
            ),
        )
        for k1, v1 in sat.target_obs().items():
            for k2, v2 in v1.items():
                print(v2, expected[k1][k2])
                assert np.all(v2 == expected[k1][k2])

    def test_bad_target_state(self, sat_init):
        n_ahead = 2
        sat = so.TargetState(
            n_ahead_observe=n_ahead,
            target_properties=[
                dict(prop="not_a_prop"),
            ],
        )
        sat.find_next_opportunities = MagicMock(
            return_value=[
                dict(target=MagicMock(priority=i, location=np.array([0.0, 0.0, 0.0])))
                for i in range(n_ahead)
            ]
        )
        with pytest.raises(ValueError):
            sat.target_obs()


@patch.multiple(so.EclipseState, __abstractmethods__=set())
@patch("bsk_rl.envs.general_satellite_tasking.scenario.satellites.Satellite.__init__")
class TestEclipseState:
    def test_init(self, sat_init):
        sat = so.EclipseState()
        sat_init.assert_called_once()
        assert len(sat.obs_fn_list) == 1
        assert sat.obs_fn_list[0].__name__ == "eclipse_state"

    def test_eclipse_state(self, sat_init):
        sat = so.EclipseState(orbit_period=10.0)
        sat.trajectory = MagicMock(next_eclipse=MagicMock(return_value=(2.0, 3.0)))
        sat.simulator = MagicMock(sim_time=1.0)
        assert sat.eclipse_state() == [0.1, 0.2]


@patch.multiple(so.GroundStationState, __abstractmethods__=set())
@patch(
    "bsk_rl.envs.general_satellite_tasking.scenario.satellites.AccessSatellite.__init__"
)
class TestGroundStationState:
    def test_init(self, sat_init):
        sat = so.GroundStationState()
        sat_init.assert_called_once()
        assert len(sat.obs_fn_list) == 1
        assert sat.obs_fn_list[0].__name__ == "ground_station_obs"

    def test_ground_station_state(self, sat_init):
        n_ahead = 2
        sat = so.GroundStationState(n_ahead_observe_downlinks=n_ahead)
        sat.simulator = MagicMock(sim_time=0.0)
        sat.find_next_opportunities = MagicMock(
            return_value=[
                dict(
                    ground_station=f"Boulder_{i}",
                    location=np.array([0.0, 0.0, 0.0]),
                    window=(i + 0.0, i + 1.0),
                )
                for i in range(n_ahead)
            ]
        )
        expected = dict(
            ground_station_0=dict(window_open_normd=0.0, window_close_normd=1.0 / 5700),
            ground_station_1=dict(
                window_open_normd=1.0 / 5700, window_close_normd=2.0 / 5700
            ),
        )
        for k1, v1 in sat.ground_station_obs().items():
            for k2, v2 in v1.items():
                assert np.all(v2 == expected[k1][k2])

    def test_target_state_normed(self, sat_init):
        n_ahead = 2
        sat = so.GroundStationState(
            n_ahead_observe_downlinks=n_ahead,
            downlink_window_properties=[
                dict(prop="location", norm=10.0),
                dict(prop="window_open", norm=10.0),
                dict(prop="window_mid"),
                dict(prop="window_close"),
            ],
        )

        sat.opportunities = [
            dict(
                ground_station="Boulder",
                window=(10.0, 20.0),
                type="ground_station",
                location=np.array([0.0, 0.0, 0.0]),
            ),
            dict(
                ground_station="Pleasanton",
                window=(20.0, 30.0),
                type="ground_station",
                location=np.array([1.0, 1.0, 1.0]),
            ),
        ]
        sat.simulator = MagicMock(sim_time=5.0)

        expected = dict(
            ground_station_0=dict(
                location_normd=np.array([0.0, 0.0, 0.0]),
                window_open_normd=5.0 / 10.0,
                window_mid=10.0,
                window_close=15.0,
            ),
            ground_station_1=dict(
                location_normd=np.array([0.1, 0.1, 0.1]),
                window_open_normd=15.0 / 10.0,
                window_mid=20.0,
                window_close=25.0,
            ),
        )
        for k1, v1 in sat.ground_station_obs().items():
            for k2, v2 in v1.items():
                print(v2, expected[k1][k2])
                assert np.all(v2 == expected[k1][k2])

    def test_bad_ground_station_state(self, sat_init):
        n_ahead = 2
        sat = so.GroundStationState(
            n_ahead_observe_downlinks=n_ahead,
            downlink_window_properties=[
                dict(prop="not_a_prop"),
            ],
        )
        sat.find_next_opportunities = MagicMock(
            return_value=[dict(ground_station=MagicMock()) for i in range(n_ahead)]
        )
        with pytest.raises(ValueError):
            sat.ground_station_obs()


@patch.multiple(so.NormdPropertyState, __abstractmethods__=set())
@patch.multiple(so.TimeState, __abstractmethods__=set())
@patch.multiple(so.TargetState, __abstractmethods__=set())
@patch.multiple(so.EclipseState, __abstractmethods__=set())
@patch(
    "bsk_rl.envs.general_satellite_tasking.scenario.satellites.ImagingSatellite.__init__"
)
def test_combination(sat_init):
    class ComboObs(
        so.EclipseState,
        so.TargetState,
        so.TimeState,
        so.NormdPropertyState.configure(obs_properties=[dict(prop="some_prop")]),
    ):
        pass

    sat = ComboObs()
    sat_init.assert_called_once()
    assert [fn.__name__ for fn in sat.obs_fn_list] == [
        "some_prop",
        "normalized_time",
        "target_obs",
        "eclipse_state",
    ]
