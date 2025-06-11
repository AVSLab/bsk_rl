from collections.abc import Iterable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import approx

from bsk_rl import sats
from bsk_rl.scene.targets import Target
from bsk_rl.utils.functional import valid_func_name


@patch(
    "bsk_rl.sats.Satellite.__init__",
    MagicMock(),
)
@patch("bsk_rl.utils.orbital.elevation", lambda x, y: y - x)
@patch.multiple(sats.AccessSatellite, __abstractmethods__=set())
class TestAccessSatellite:
    def make_sat(self):
        sat = sats.AccessSatellite(
            "TestSat",
            sat_args={"imageTargetMinimumElevation": 1},
        )
        sat.logger = MagicMock()
        return sat

    def test_add_location_for_access_checking(self):
        sat = self.make_sat()
        sat.locations_for_access_checking = []
        target = MagicMock()
        sat.add_location_for_access_checking(
            object=target, r_LP_P=[0, 0, 0], min_elev=1.0, type="target"
        )
        assert (
            dict(
                object=target,
                target=target,
                r_LP_P=[0, 0, 0],
                min_elev=1.0,
                type="target",
            )
            in sat.locations_for_access_checking
        )

    @pytest.mark.parametrize("start", [0.0, 100.0])
    @pytest.mark.parametrize("duration", [0.0, 20.0, 500.0])
    @pytest.mark.parametrize("traj_dt", [30.0, 200.0])
    @pytest.mark.parametrize("generation_duration", [60.0, 100.0])
    def test_calculate_windows_duration(
        self, start, duration, traj_dt, generation_duration
    ):
        sat = self.make_sat()
        sat.window_calculation_time = start
        sat.generation_duration = generation_duration
        sat.trajectory = MagicMock(
            dt=traj_dt,
            r_BP_P=MagicMock(
                x=np.linspace(0, start + duration), y=np.linspace(0, start + duration)
            ),
        )
        sat.locations_for_access_checking = []
        sat.calculate_additional_windows(duration)
        if duration == 0.0:
            return
        assert sat.trajectory.extend_to.call_args[0][0] >= start + duration
        assert sat.trajectory.extend_to.call_args[0][0] - start >= traj_dt * 2

    def test_calculate_windows(self):
        tgt = Target("tgt_0", r_LP_P=[0.0, 0.0, 1.0], priority=1.0)
        sat = self.make_sat()
        sat.window_calculation_time = 0.0
        sat.opportunities = []
        sat.access_dist_threshold = 5.0
        sat.trajectory = MagicMock(
            dt=2.0,
            r_BP_P=MagicMock(
                x=np.arange(0, 100, 2),
                y=np.array([[t - 50.0, 0.0, 2.0] for t in np.arange(0, 100, 2)]),
                side_effect=(  # noqa: E731
                    lambda t: (
                        np.array([[ti - 50.0, 0.0, 2.0] for ti in t])
                        if isinstance(t, Iterable)
                        else np.array([t - 50.0, 0.0, 2.0])
                    )
                ),
            ),
        )
        sat.locations_for_access_checking = [
            dict(object=tgt, type="target", min_elev=1.3, r_LP_P=tgt.r_LP_P)
        ]
        sat.calculate_additional_windows(100.0)
        assert tgt in sat.opportunities_dict()
        assert sat.opportunities[0]["window"][0] == approx(
            50 - 0.27762037530835193, abs=1e-2
        )
        assert sat.opportunities[0]["window"][1] == approx(
            50 + 0.27762037530835193, abs=1e-2
        )

    def test_find_elevation_roots(self):
        interp = lambda t: (  # noqa: E731
            np.array([[ti, 0.0, 2.0] for ti in t])
            if isinstance(t, Iterable)
            else np.array([t, 0.0, 2.0])
        )
        loc = np.array([0.0, 0.0, 1.0])
        elev = 1.3
        times = sats.ImagingSatellite._find_elevation_roots(interp, loc, elev, (-1, 1))
        assert len(times) == 2
        assert times[0] == approx(-times[1], abs=1e-5)
        assert times[1] == approx(0.27762037530835193, abs=1e-5)
        times = sats.ImagingSatellite._find_elevation_roots(interp, loc, elev, (0, 1))
        assert len(times) == 1
        assert times[0] == approx(0.27762037530835193, abs=1e-5)

    @pytest.mark.parametrize(
        "location,times,positions,threshold,expected",
        [
            (
                np.array([2.5, 0.0]),
                np.array([0.0, 10.0, 20.0, 30.0]),
                np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
                1.0,
                [(10.0, 30.0)],
            ),
            (
                np.array([2.5, 0.0]),
                np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
                np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
                1.0,
                [(10.0, 40.0)],
            ),
            (
                np.array([0.5, 0.0]),
                np.array([0.0, 10.0, 20.0, 30.0]),
                np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
                1.0,
                [(0.0, 20.0)],
            ),
            (
                np.array([1.2, 0.0]),
                np.array([0.0, 10.0, 20.0, 30.0]),
                np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
                5.0,
                [(0.0, 30.0)],
            ),
            (
                np.array([2.5, 100.0]),
                np.array([0.0, 10.0, 20.0, 30.0]),
                np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
                1.0,
                [],
            ),
            (
                np.array([-0.1, 0.0]),
                np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
                np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [-1.0, 0.0]]),
                1.0,
                [(0.0, 10.0), (30.0, 40.0)],
            ),
        ],
    )
    def test_find_candidate_windows(
        self, location, times, positions, threshold, expected
    ):
        assert (
            sats.ImagingSatellite._find_candidate_windows(
                location, times, positions, threshold
            )
            == expected
        )

    @pytest.mark.parametrize(
        "endpoints,candidate_window,computation_window,expected",
        [
            ([2.4, 14.6], (0.0, 20.0), (0.0, 30.0), [(2.4, 14.6)]),
            ([12.4], (0.0, 20.0), (0.0, 30.0), [(0.0, 12.4)]),
            ([12.4], (10.0, 30.0), (0.0, 30.0), [(12.4, 30.0)]),
            ([2.4, 14.6, 18.8], (0.0, 20.0), (0.0, 30.0), [(0.0, 2.4), (14.6, 18.8)]),
            (
                [2.4, 14.6, 18.8, 19.3],
                (0.0, 20.0),
                (0.0, 30.0),
                [(2.4, 14.6), (18.8, 19.3)],
            ),
        ],
    )
    def test_refine_windows(
        self, endpoints, candidate_window, computation_window, expected
    ):
        assert (
            sats.ImagingSatellite._refine_window(
                endpoints, candidate_window, computation_window
            )
            == expected
        )

    @pytest.mark.skip(reason="Disabled by temporary bugfix")
    def test_refine_windows_impossible(self):
        with pytest.raises(ValueError):
            sats.ImagingSatellite._refine_window(
                [1.0, 2.0, 3.0], (0.0, 4.0), (0.5, 3.5)
            )

    tgt0 = Target("tgt_0", r_LP_P=[0.0, 0.0, 0.0], priority=1.0)
    tgt1 = Target("tgt_1", r_LP_P=[0.0, 0.0, 0.0], priority=1.0)
    tgt2 = Target("tgt_2", r_LP_P=[0.0, 0.0, 0.0], priority=1.0)

    @pytest.mark.parametrize(
        "merge_time",
        [None, 10.0],
    )
    @pytest.mark.parametrize(
        "tgt,window,expected_window",
        [
            (tgt0, (13.0, 18.0), (13.0, 18.0)),
            (tgt2, (13.0, 18.0), (13.0, 18.0)),
            (tgt0, (10.0, 18.0), (2.0, 18.0)),  # Check that merging works
        ],
    )
    def test_add_window(self, merge_time, tgt, window, expected_window):
        sat = self.make_sat()
        sat.opportunities = [
            dict(object=self.tgt1, window=(3.0, 8.0), type="target"),
            dict(object=self.tgt0, window=(2.0, 10.0), type="target"),
        ]
        sat._add_window(
            tgt, window, merge_time=merge_time, type="target", r_LP_P=np.zeros(3)
        )
        assert expected_window in sat.opportunities_dict()[tgt]

    opportunities = [
        dict(object="downObj1", window=(10, 20), type="downlink"),
        dict(object="tgtObj1", window=(20, 30), type="target"),
        dict(object="downObj1", window=(30, 40), type="downlink"),
        dict(object="downObj2", window=(35, 45), type="downlink"),
    ]

    def test_upcoming_opportunities(self):
        sat = self.make_sat()
        sat.opportunities = self.opportunities
        sat.simulator = MagicMock(sim_time=25.0)
        assert sat.upcoming_opportunities == self.opportunities[1:4]

    def test_opportunities_dict(self):
        sat = self.make_sat()
        sat.opportunities = self.opportunities
        assert sat.opportunities_dict(types="target") == dict(tgtObj1=[(20, 30)])
        assert sat.opportunities_dict(types=None) == sat.opportunities_dict(
            types=["target", "downlink"]
        )
        assert sat.opportunities_dict(types="downlink", filter=["downObj1"]) == dict(
            downObj2=[(35, 45)]
        )

    def test_upcoming_opportunities_dict(self):
        sat = self.make_sat()
        sat.opportunities = self.opportunities
        sat.simulator = MagicMock(sim_time=35.0)
        assert sat.upcoming_opportunities_dict(types="target") == {}
        assert sat.upcoming_opportunities_dict(
            types=None
        ) == sat.upcoming_opportunities_dict(types=["target", "downlink"])
        assert sat.upcoming_opportunities_dict(
            types="downlink", filter=["downObj2"]
        ) == dict(downObj1=[(30, 40)])

    def test_next_opportunities_dict(self):
        sat = self.make_sat()
        sat.opportunities = self.opportunities
        sat.simulator = MagicMock(sim_time=15.0)
        assert sat.next_opportunities_dict() == dict(
            downObj1=(10, 20), tgtObj1=(20, 30), downObj2=(35, 45)
        )
        assert sat.next_opportunities_dict(types="downlink") == dict(
            downObj1=(10, 20), downObj2=(35, 45)
        )
        assert sat.next_opportunities_dict(filter=["downObj1"]) == dict(
            tgtObj1=(20, 30), downObj2=(35, 45)
        )

    def test_find_next_opportunities(self):
        pass  # Tested in TestImagingSatellite


@patch("bsk_rl.sats.Satellite.__init__")
@patch.multiple(sats.ImagingSatellite, __abstractmethods__=set())
def test_init(mock_init):
    sats.ImagingSatellite(
        "TestSat",
        sat_args={"imageTargetMinimumElevation": 1},
    )
    mock_init.assert_called_once()


@patch(
    "bsk_rl.sats.Satellite.__init__",
    MagicMock(),
)
@patch.multiple(sats.ImagingSatellite, __abstractmethods__=set())
class TestImagingSatellite:
    def make_sat(self):
        sat = sats.ImagingSatellite(
            "TestSat",
            sat_args={"imageTargetMinimumElevation": 1},
        )
        return sat

    @patch("bsk_rl.sats.Satellite.reset_pre_sim_init")
    def test_reset_pre_sim_init(self, mock_reset):
        sat = self.make_sat()
        sat.observation_builder = MagicMock()
        sat.action_builder = MagicMock()
        sat.reset_overwrite_previous()
        targets = [MagicMock()] * 5
        for target in targets:
            sat.add_location_for_access_checking(target, np.ones(3), 1.0, "target")
        sat.sat_args = {}
        sat.reset_pre_sim_init()
        mock_reset.assert_called_once()
        assert sat.sat_args["transmitterNumBuffers"] == 5
        assert len(sat.sat_args["bufferNames"]) == 5

    @pytest.mark.parametrize(
        "gen_duration,time_limit,expected",
        [(None, float("inf"), 0), (None, 100.0, 100.0), (10.0, 100.0, 10.0)],
    )
    @patch("bsk_rl.sats.Satellite.reset_post_sim_init")
    def test_reset_post_sim_init(self, mock_reset, gen_duration, time_limit, expected):
        sat = self.make_sat()
        sat.sat_args = {}
        sat.calculate_additional_windows = MagicMock()
        sat.initial_generation_duration = gen_duration
        sat.simulator = MagicMock(time_limit=time_limit)
        sat.data_store = MagicMock()
        sat.reset_post_sim_init()
        mock_reset.assert_called_once()
        assert sat.initial_generation_duration == expected
        sat.calculate_additional_windows.assert_called_once()

    tgt0 = Target("tgt_0", r_LP_P=[0.0, 0.0, 0.0], priority=1.0)
    tgt1 = Target("tgt_1", r_LP_P=[0.0, 0.0, 0.0], priority=1.0)
    tgt2 = Target("tgt_2", r_LP_P=[0.0, 0.0, 0.0], priority=1.0)
    windows = {
        tgt0: [(0.0, 10.0), (20.0, 30.0), (40.0, 50.0)],
        tgt1: [(10.0, 20.0)],
        tgt2: [(30.0, 40.0)],
    }
    opportunities = [
        dict(object=tgt0, window=(0.0, 10.0), type="target"),
        dict(object=tgt1, window=(10.0, 20.0), type="target"),
        dict(object=tgt0, window=(20.0, 30.0), type="target"),
        dict(object=tgt2, window=(30.0, 40.0), type="target"),
        dict(object=tgt0, window=(40.0, 50.0), type="target"),
    ]

    @patch(
        "bsk_rl.sats.ImagingSatellite._disable_image_event",
        MagicMock(),
    )
    def test_update_image_event_existing(self):
        sat = self.make_sat()
        sat.name = "Sat"
        tgt = MagicMock(name="tgt")
        existing_event = valid_func_name(f"image_{sat.name}_{tgt.id}")
        sat.simulator = MagicMock(
            eventMap={existing_event: MagicMock(eventActive=False)}
        )
        sat._update_image_event(tgt)
        assert sat.simulator.eventMap[existing_event].eventActive is True

    def test_disable_image_event(self):
        sat = self.make_sat()
        sat.simulator = MagicMock(eventMap={"some_image_event": 1})
        sat._image_event_name = "some_image_event"
        sat._disable_image_event()
        sat.simulator.delete_event.assert_called_with("some_image_event")

    def test_disable_image_event_no_event(self):
        sat = self.make_sat()
        sat.simulator = MagicMock(eventMap={"some_event": 1})
        sat._image_event_name = None
        sat._disable_image_event()
        assert not sat.simulator.delete_event.called

    upcoming_targets = [Target(f"tgt_{i}", [0, 0, 0], 1.0) for i in range(20)]

    @pytest.mark.parametrize(
        "query,expected",
        [
            (0, "tgt_0"),
            (3, "tgt_3"),
            (upcoming_targets[2].id, "tgt_2"),
            (upcoming_targets[2], "tgt_2"),
        ],
    )
    def test_parse_target_selection(self, query, expected):
        sat = self.make_sat()
        sat.find_next_opportunities = lambda *args, **kwargs: [
            dict(object=target) for target in self.upcoming_targets[0 : kwargs["n"]]
        ]
        sat.data_store = MagicMock()
        sat.data_store.data.known = self.upcoming_targets
        assert expected == sat.parse_target_selection(query).name

    def test_parse_target_selection_invalid(self):
        sat = self.make_sat()
        sat.upcoming_targets = lambda x: self.upcoming_targets[0:x]
        sat.data_store = MagicMock()
        sat.data_store.data.known = self.upcoming_targets
        with pytest.raises(TypeError):
            sat.parse_target_selection(np.zeros(10))

    def test_task_target_for_imaging(self):
        sat = self.make_sat()
        sat.opportunities = self.opportunities
        sat.name = "Sat"
        sat.fsw = MagicMock()
        sat.simulator = MagicMock(sim_time=35.0)
        sat._update_image_event = MagicMock()
        sat.update_timed_terminal_event = MagicMock()
        sat.logger = MagicMock()
        sat.task_target_for_imaging(self.tgt0)
        sat.fsw.action_image.assert_called_once()
        assert sat.fsw.action_image.call_args[0][1].startswith("tgt_0")
        sat.logger.info.assert_called()
        sat._update_image_event.assert_called_once()
        assert sat._update_image_event.call_args[0][0] == self.tgt0
        sat.update_timed_terminal_event.assert_called_once()
        assert sat.update_timed_terminal_event.call_args[0][0] == 50.0
