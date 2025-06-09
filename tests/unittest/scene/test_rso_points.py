from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import approx

from bsk_rl.scene import RSOPoints, SphericalRSO
from bsk_rl.scene.rso_points import logger
from bsk_rl.sim.dyn import RSODynModel, RSOInspectorDynModel

FakeRSODyn = type("FakeRSODyn", (RSODynModel,), {})
FakeInspectorDyn = type("FakeInspectorDyn", (RSOInspectorDynModel,), {})


@patch.multiple(
    RSOPoints,
    __abstractmethods__=set(),
)
class TestRSOPoints:
    def test_reset_no_rsos(self):
        rso = RSOPoints()
        rso.satellites = [MagicMock(dyn_type=FakeRSODyn)]
        logger.warning = MagicMock()
        rso.reset_pre_sim_init()
        logger.warning.assert_called_once()

    def test_reset_too_many_rsos(self):
        rso = RSOPoints()
        rso.satellites = [
            MagicMock(dyn_type=FakeRSODyn),
            MagicMock(dyn_type=FakeRSODyn),
        ]
        with pytest.raises(AssertionError):
            rso.reset_pre_sim_init()

    def test_reset_id_satellites(self):
        rso_sat = MagicMock(dyn_type=FakeRSODyn)
        inspectors = [
            MagicMock(dyn_type=FakeInspectorDyn),
            MagicMock(dyn_type=FakeInspectorDyn),
        ]
        sats = [rso_sat] + inspectors
        rso = RSOPoints()
        rso.satellites = sats
        rso.reset_pre_sim_init()
        assert rso.rso == rso_sat
        assert rso.inspectors == inspectors


class TestSphericalRSO:
    @pytest.mark.parametrize("n_points", [10, 100, 1000])
    @pytest.mark.parametrize("radius", [0.1, 1.0, 5.0])
    def test_generate_points(self, n_points, radius):
        """Test the generation of points on a sphere."""
        rso = SphericalRSO(n_points=n_points, radius=radius)
        points = rso.generate_points()

        assert len(points) == n_points
        for point in points:
            assert np.linalg.norm(point.r_PB_B) == approx(radius)
            assert np.linalg.norm(point.n_B) == approx(1.0)
