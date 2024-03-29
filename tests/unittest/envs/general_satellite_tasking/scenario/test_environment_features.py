from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import approx

from bsk_rl.envs.general_satellite_tasking.scenario.environment_features import (
    CityTargets,
    StaticTargets,
    Target,
    UniformNadirFeature,
    lla2ecef,
)


class TestTarget:
    T1a = Target("Boulder", [1, 2, 3], 1.0)
    T1b = Target("Boulder", [1, 2, 3], 1.0)
    T2 = Target("Pleasanton", np.array([0, 0, 0]), 0.5)

    def test_id_unique(self):
        assert self.T1a.id != self.T1b.id

    def test_hash_unique(self):
        assert hash(self.T1a) != hash(self.T1b)

    def test_repr_(self):
        assert self.T2.__repr__() == "Target(Pleasanton)"

    def test_location_type(self):
        assert np.all(self.T1a.location == np.array([1, 2, 3]))
        assert not isinstance(self.T1a.location, list)
        assert np.all(self.T2.location == np.array([0, 0, 0]))


class TestStaticTargets:
    def test_init(self):
        st = StaticTargets(1)
        assert st.priority_distribution is not None

    def test_reset_constant(self):
        st = StaticTargets(10)
        st.regenerate_targets = MagicMock()
        st.reset()
        assert st.n_targets == 10
        st.regenerate_targets.assert_called_once()

    @pytest.mark.repeat(10)
    def test_reset_variable(self):
        st = StaticTargets((8, 10))
        st.regenerate_targets = MagicMock()
        st.reset()
        assert 8 <= st.n_targets <= 10

    def test_regenerate_targets(self):
        st = StaticTargets(3, radius=1.0, priority_distribution=lambda: 1)
        st.n_targets = st._n_targets
        st.regenerate_targets()
        assert len(st.targets) == 3
        for target in st.targets:
            assert np.linalg.norm(target.location) == approx(1.0)
            assert target.priority == 1

    def test_regenerate_targets_repeatable(self):
        np.random.seed(0)
        st1 = StaticTargets(3, radius=1.0)
        st1.reset()
        np.random.seed(0)
        st2 = StaticTargets(3, radius=1.0)
        st2.reset()
        for t1, t2 in zip(st1.targets, st2.targets):
            assert (t1.location == t2.location).all()


class TestCityTargets:
    @pytest.mark.parametrize(
        "lat,long,radius,expected",
        [
            (0, 0, 1, np.array([1, 0, 0])),
            (0, 0, 2, np.array([2, 0, 0])),
            (90, 0, 1, np.array([0, 0, 1])),
            (0, 180, 1, np.array([-1, 0, 0])),
        ],
    )
    def test_lla2ecef(self, lat, long, radius, expected):
        assert abs(lla2ecef(lat, long, radius) - expected < 1e-9).all()

    def mock_data(self, mock_read_csv, n_database=5):
        mock_read_csv.return_value = MagicMock(
            iloc=MagicMock(
                __getitem__=lambda self, i: {
                    "city": f"city{i}",
                    "lat": 0.0,
                    "lng": 0.0,
                }
            ),
            __len__=lambda self: n_database,
        )

    @pytest.mark.parametrize(
        "n_targets",
        [0, 2, 5, 10],
    )
    @patch("pandas.read_csv")
    def test_regenerate_targets(self, mock_read_csv, n_targets):
        n_database = 5
        self.mock_data(mock_read_csv, n_database=n_database)
        ct = CityTargets(n_targets)
        if n_targets > n_database:
            with pytest.raises(ValueError):
                ct.reset()
        else:
            ct.reset()
            assert len(ct.targets) == n_targets
            possible_names = [f"city{i}" for i in range(5)]
            for target in ct.targets:
                assert target.name in possible_names

    @pytest.mark.parametrize(
        "n_targets,n_select_from",
        [(0, 2), (1, 2), (2, 2), (3, "all")],
    )
    @patch("pandas.read_csv")
    def test_regenerate_targets_n_select_from(
        self, mock_read_csv, n_targets, n_select_from
    ):
        self.mock_data(mock_read_csv)
        ct = CityTargets(n_targets, n_select_from=n_select_from)
        ct.reset()
        assert len(ct.targets) == n_targets
        if isinstance(n_select_from, int):
            possible_names = [f"city{i}" for i in range(n_select_from)]
            for target in ct.targets:
                assert target.name in possible_names

    @patch(
        "bsk_rl.envs.general_satellite_tasking.scenario.environment_features.lla2ecef"
    )
    @patch("pandas.read_csv")
    def test_regenerate_targets_offset(self, mock_read_csv, mock_lla2ecef):
        nominal = np.array([1.0, 0.0, 0.0])
        mock_lla2ecef.return_value = nominal
        self.mock_data(mock_read_csv, n_database=10)
        n_targets = 10
        ct = CityTargets(n_targets, location_offset=0.01, radius=1.0)
        ct.reset()
        for target in ct.targets:
            assert np.linalg.norm(target.location - nominal) <= 0.03
            assert np.linalg.norm(target.location) == approx(1.0)


class TestUniformNadirFeature:
    def test_init(self):
        st = UniformNadirFeature()
        assert st.name == "NadirFeature"
