import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from Basilisk.utilities.orbitalMotion import ClassicElements

from bsk_rl.utils import orbital


class TestRandomOrbit:
    @pytest.mark.repeat(10)
    def test_random_orbit(self):
        oe = orbital.random_orbit(i=None, Omega=None, omega=None, f=None)
        assert -np.pi <= oe.i <= np.pi
        assert 0 <= oe.Omega <= 2 * np.pi
        assert 0 <= oe.omega <= 2 * np.pi
        assert 0 <= oe.f <= 2 * np.pi

    @pytest.mark.repeat(10)
    def test_random_circular_orbit(self):
        oe = orbital.random_circular_orbit(i=None, Omega=None, f=None)
        assert -np.pi <= oe.i <= np.pi
        assert 0 <= oe.Omega <= 2 * np.pi
        assert 0 <= oe.f <= 2 * np.pi

    def test_repeatable(self):
        np.random.seed(0)
        oe1 = orbital.random_orbit()
        np.random.seed(0)
        oe2 = orbital.random_orbit()
        assert oe1.f == oe2.f

    def test_units(self):
        oe = orbital.random_orbit(i=90.0, a=1500, e=0.1, Omega=90.0, omega=90.0, f=90.0)
        assert oe.a == 1500000
        assert oe.e == 0.1
        assert np.pi / 2 == oe.i == oe.Omega == oe.omega == oe.f


def classic_elements(a=0, e=0, i=0, Omega=0, omega=0, f=0):
    oe = ClassicElements()
    oe.a = a
    oe.e = e
    oe.i = i
    oe.Omega = Omega
    oe.omega = omega
    oe.f = f
    return oe


class TestWalkerDeltaArgs:
    @patch(
        "bsk_rl.utils.orbital.walker_delta",
        MagicMock(return_value=[classic_elements(), classic_elements()]),
    )
    def test_randomize_ta_and_lan(self):
        sats = [MagicMock(), MagicMock()]
        walker_delta_arg_setup = orbital.walker_delta_args(
            n_planes=1, randomize_true_anomaly=True, randomize_lan=True
        )
        sat_arg_map = walker_delta_arg_setup(sats)
        assert sat_arg_map[sats[0]]["oe"].f == sat_arg_map[sats[1]]["oe"].f
        assert sat_arg_map[sats[0]]["oe"].f != 0
        assert sat_arg_map[sats[0]]["oe"].Omega == sat_arg_map[sats[1]]["oe"].Omega
        assert sat_arg_map[sats[0]]["oe"].Omega != 0

    @patch(
        "bsk_rl.utils.orbital.walker_delta",
        MagicMock(return_value=[classic_elements(), classic_elements()]),
    )
    def test_dont_randomize_ta_and_lan(self):
        sats = [MagicMock(), MagicMock()]
        walker_delta_arg_setup = orbital.walker_delta_args(
            n_planes=1, randomize_true_anomaly=False, randomize_lan=False
        )
        sat_arg_map = walker_delta_arg_setup(sats)
        assert sat_arg_map[sats[0]]["oe"].f == sat_arg_map[sats[1]]["oe"].f
        assert sat_arg_map[sats[0]]["oe"].f == 0
        assert sat_arg_map[sats[0]]["oe"].Omega == sat_arg_map[sats[1]]["oe"].Omega
        assert sat_arg_map[sats[0]]["oe"].Omega == 0


class TestRelativeToChief:
    def test_relative_to_chief(self):
        chief_name = "ChiefSat"
        chief_orbit = classic_elements(a=1)
        deputy_relative_state = {
            "DeputySat1": np.zeros(6),
            "DeputySat2": np.array([1, 0, 0, 0, 0, 0]),
        }

        relative_orbit_arg_setup = orbital.relative_to_chief(
            chief_name, chief_orbit, deputy_relative_state
        )

        satellites = [
            MagicMock(sat_args_generator=dict(mu=1)),
            MagicMock(),
            MagicMock(),
        ]
        satellites[0].name = "ChiefSat"
        satellites[1].name = "DeputySat1"
        satellites[2].name = "DeputySat2"
        args = relative_orbit_arg_setup(satellites)
        assert chief_orbit.a == args[satellites[0]]["oe"].a
        assert chief_orbit.a == args[satellites[1]]["oe"].a
        assert chief_orbit.e == args[satellites[0]]["oe"].e
        assert chief_orbit.e == args[satellites[1]]["oe"].e
        assert chief_orbit.i == args[satellites[0]]["oe"].i
        assert chief_orbit.i == args[satellites[1]]["oe"].i

        assert np.isclose(args[satellites[2]]["oe"].a, -1 / 3)
        assert np.isclose(args[satellites[2]]["oe"].e, 7.0)
        assert np.isclose(args[satellites[2]]["oe"].i, 0.0)


class TestRandomEpoch:
    @pytest.mark.repeat(10)
    def test_random_epoch(self):
        assert (
            re.match(
                r"\d{4} [A-Z]{3} \d{2} \d{2}:\d{2}:\d{2}\.\d{3} \(UTC\)",
                orbital.random_epoch(),
            )
            is not None
        )

    def test_repeatable(self):
        np.random.seed(0)
        e1 = orbital.random_epoch()
        np.random.seed(0)
        e2 = orbital.random_epoch()
        assert e1 == e2


@pytest.mark.parametrize(
    "r_sat,r_target,expected",
    [
        (np.array([2, 0, 0]), np.array([1, 0, 0]), np.pi / 2),
        (
            np.array([[1, 1, 0], [2, 0, 0]]),
            np.array([1, 0, 0]),
            np.array([0, np.pi / 2]),
        ),
    ],
)
def test_elevation(r_sat, r_target, expected):
    assert np.all(orbital.elevation(r_sat, r_target) == expected)


class TestTrajectorySimulator:
    epoch = "2005 JUL 24 20:50:33.771 (UTC)"
    oe = orbital.random_orbit(i=0.0, omega=0, Omega=0, f=45.0)
    mu = 0.3986004415e15

    @pytest.mark.parametrize(
        "kwargs,error",
        [
            (dict(rN=[1, 0, 0], vN=[0, 1, 0]), False),
            (dict(oe=oe, mu=mu), False),
            (dict(rN=[1, 0, 0], oe=oe), True),
            (dict(rN=[1, 0, 0], oe=oe, mu=mu), True),
            (dict(rN=[1, 0, 0], vN=[0, 1, 0], oe=oe), True),
            (dict(), True),
        ],
    )
    def test_init(self, kwargs, error):
        if error:
            with pytest.raises(ValueError):
                orbital.TrajectorySimulator(self.epoch, **kwargs)
        else:
            orbital.TrajectorySimulator(self.epoch, **kwargs)

    @pytest.mark.parametrize(
        "dt,extend1,extend2",
        [(30.0, 100.0, 200.0), (40.0, 100.0, 50.0)],
    )
    def test_extend_to_and_time_properties(self, dt, extend1, extend2):
        ts = orbital.TrajectorySimulator(self.epoch, oe=self.oe, mu=self.mu, dt=dt)
        assert ts.sim_time == 0
        ts.extend_to(extend1)
        assert ts.sim_time == dt * np.floor(extend1 / dt)
        np.testing.assert_allclose(ts.times, np.arange(0, extend1, dt), atol=1e-9)
        ts.extend_to(extend2)
        assert ts.sim_time == dt * np.floor(max(extend1, extend2) / dt)
        np.testing.assert_allclose(
            ts.times, np.arange(0, max(extend1, extend2), dt), atol=1e-9
        )

    def test_eclipse(self):
        ts = orbital.TrajectorySimulator(self.epoch, oe=self.oe, mu=self.mu, dt=500.0)
        start_1, end_1 = ts.next_eclipse(0.0)
        # Verify eclipse start and end are correct (duration is less than 1/2 orbit)
        assert (end_1 - start_1 % 5700) < 5700 / 2
        # With chosen params, start out of eclipse
        assert end_1 > start_1
        # Go into eclipse
        start_2, end_2 = ts.next_eclipse(start_1 + 1.0)
        assert end_2 == end_1  # Soonest close is the same
        assert abs(start_2 - start_1 - 5700) < 500.0  # Starts are about an orbit apart

    def test_no_eclipse(self):
        ts = orbital.TrajectorySimulator(
            self.epoch,
            oe=orbital.random_orbit(a=50000, i=90.0, omega=0, Omega=0, f=45.0),
            mu=self.mu,
        )
        assert ts.next_eclipse(0, max_tries=3) == (1.0, 1.0)

    def test_interpolators(self):
        # Weak tests, could be better
        ts = orbital.TrajectorySimulator(self.epoch, oe=self.oe, mu=self.mu)
        assert (ts.r_BN_N(0) == ts.rN_init).all()
        assert (ts.r_BN_N(0) != ts.r_BN_N(1)).all()
        ts = orbital.TrajectorySimulator(self.epoch, oe=self.oe, mu=self.mu)
        assert (ts.r_BP_P(0) != ts.r_BP_P(1)).all()


class TestFunctions:
    def test_rv2HN(self):
        r = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
        HN = orbital.rv2HN(r, v)
        assert np.allclose(HN, np.eye(3))

    def test_rv2HN_nonunit(self):
        r = np.array([1, 0, 0]) * 12345
        v = np.array([0, 1, 0]) * 678.9
        HN = orbital.rv2HN(r, v)
        assert np.allclose(HN, np.eye(3))

    def test_rv2omega(self):
        r = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
        omega = orbital.rv2omega(r, v)
        assert np.allclose(omega, np.array([0, 0, 1]))

    @pytest.mark.repeat(10)
    def test_cd_hill(self):
        rc_N = np.random.rand(3)
        vc_N = np.random.rand(3)
        rd_N = np.random.rand(3)
        vd_N = np.random.rand(3)

        rho_H, rho_deriv_H = orbital.cd2hill(rc_N, vc_N, rd_N, vd_N)
        rd_N_new, vd_N_new = orbital.hill2cd(rc_N, vc_N, rho_H, rho_deriv_H)
        assert np.allclose(rd_N, rd_N_new)
        assert np.allclose(vd_N, vd_N_new)
