import numpy as np
import pytest

from bsk_rl.utils.attitude import hill_frame_attitude


class TestHillFrameAttitude:
    def test_default_is_nadir_pointing(self):
        sigma_BN, omega_BN_B = hill_frame_attitude(
            r_N=[2.0, 0.0, 0.0],
            v_N=[0.0, 4.0, 0.0],
        )

        np.testing.assert_allclose(sigma_BN, np.zeros(3), atol=1e-15)
        np.testing.assert_allclose(omega_BN_B, [0.0, 0.0, 2.0])

    def test_fixed_spin_with_custom_axis(self):
        _, omega_BN_B = hill_frame_attitude(
            r_N=[2.0, 0.0, 0.0],
            v_N=[0.0, 4.0, 0.0],
            spin_config={"type": "fixed", "rate": 0.5, "axis": [2.0, 0.0, 0.0]},
        )

        np.testing.assert_allclose(omega_BN_B, [np.radians(0.5), 0.0, 2.0])

    def test_rotated_hill_frame(self):
        sigma_BN, omega_BN_B = hill_frame_attitude(
            r_N=[0.0, 2.0, 0.0],
            v_N=[-4.0, 0.0, 0.0],
            spin_config={"type": "fixed", "rate": 0.5, "axis": "tangential"},
        )

        assert not np.allclose(sigma_BN, np.zeros(3))
        np.testing.assert_allclose(omega_BN_B, [0.0, np.radians(0.5), 2.0])

    def test_uniform_spin(self):
        np.random.seed(0)
        _, omega_BN_B = hill_frame_attitude(
            r_N=[2.0, 0.0, 0.0],
            v_N=[0.0, 4.0, 0.0],
            spin_config={
                "type": "uniform",
                "low": 1.0,
                "high": 2.0,
                "axis": "normal",
            },
        )

        relative_rate = np.degrees(omega_BN_B[2] - 2.0)
        assert 1.0 <= relative_rate <= 2.0

    def test_velocity_axis_for_eccentric_orbit(self):
        _, omega_BN_B = hill_frame_attitude(
            r_N=[2.0, 0.0, 0.0],
            v_N=[1.0, 4.0, 0.0],
            spin_config={"type": "fixed", "rate": 1.0, "axis": "velocity"},
        )

        relative_rate_H = omega_BN_B - np.array([0.0, 0.0, 2.0])
        expected_axis_H = np.array([1.0, 4.0, 0.0]) / np.sqrt(17.0)
        np.testing.assert_allclose(relative_rate_H, np.radians(1.0) * expected_axis_H)

    def test_random_axis_has_requested_rate(self):
        np.random.seed(0)
        _, omega_BN_B = hill_frame_attitude(
            r_N=[2.0, 0.0, 0.0],
            v_N=[0.0, 4.0, 0.0],
            spin_config={"type": "fixed", "rate": 1.0, "axis": "random"},
        )

        relative_rate = omega_BN_B - np.array([0.0, 0.0, 2.0])
        assert np.linalg.norm(relative_rate) == pytest.approx(np.radians(1.0))

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"r_N": [0.0, 0.0, 0.0], "v_N": [0.0, 1.0, 0.0]}, "Hill frame"),
            ({"r_N": [1.0, 0.0], "v_N": [0.0, 1.0, 0.0]}, "three-element"),
            (
                {
                    "r_N": [1.0, 0.0, 0.0],
                    "v_N": [0.0, 1.0, 0.0],
                    "spin_config": {"axis": [0.0, 0.0, 0.0]},
                },
                "nonzero",
            ),
            (
                {
                    "r_N": [1.0, 0.0, 0.0],
                    "v_N": [0.0, 1.0, 0.0],
                    "spin_config": {"type": "unknown"},
                },
                "Unknown spin type",
            ),
            (
                {
                    "r_N": [1.0, 0.0, 0.0],
                    "v_N": [0.0, 1.0, 0.0],
                    "spin_config": {"axis": "unknown"},
                },
                "Unknown spin axis",
            ),
        ],
    )
    def test_invalid_inputs(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            hill_frame_attitude(**kwargs)
