from unittest.mock import MagicMock

import pytest

from bsk_rl.data.rso_inspection import (
    RSOInspectionData,
    RSOInspectionDataStore,
    RSOInspectionReward,
    RSOInspectorDynModel,
)


class TestRSOInspectionData:
    @pytest.mark.parametrize(
        "a, b, sum",
        [
            (dict(a=True), dict(a=False), dict(a=True)),
            (dict(a=False), dict(a=False), dict(a=False)),
            (dict(), dict(a=False), dict(a=False)),
            (dict(), dict(a=True), dict(a=True)),
            (dict(a=False), dict(), dict(a=False)),
            (dict(a=True), dict(), dict(a=True)),
            (
                dict(a=True, b=False),
                dict(b=True, c=False),
                dict(a=True, b=True, c=False),
            ),
        ],
    )
    def test_add_inspect(self, a, b, sum):
        da = RSOInspectionData(point_inspect_status=a, point_illuminate_status={})
        db = RSOInspectionData(point_inspect_status=b, point_illuminate_status={})
        result = da + db
        assert sum == result.point_inspect_status
        assert result.point_illuminate_status == {}

    @pytest.mark.parametrize(
        "a, b, sum",
        [
            (dict(a=True), dict(a=False), dict(a=True)),
            (dict(a=False), dict(a=False), dict(a=False)),
            (dict(), dict(a=False), dict(a=False)),
            (dict(), dict(a=True), dict(a=True)),
            (dict(a=False), dict(), dict(a=False)),
            (dict(a=True), dict(), dict(a=True)),
            (
                dict(a=True, b=False),
                dict(b=True, c=False),
                dict(a=True, b=True, c=False),
            ),
        ],
    )
    def test_add_illuminate(self, a, b, sum):
        da = RSOInspectionData(point_inspect_status={}, point_illuminate_status=a)
        db = RSOInspectionData(point_inspect_status={}, point_illuminate_status=b)
        result = da + db
        assert sum == result.point_illuminate_status
        assert result.point_inspect_status == {}


FakeInspectorDyn = type("FakeInspectorDyn", (RSOInspectorDynModel,), {})


class TestRSOInspectionDataStore:
    @pytest.mark.parametrize(
        "storage, access, illumination, expected",
        [
            ([10, 10, 10], [[1, 0, 0]], [[1, 1, 0]], [[0, 0, 0]]),
            ([10, 11, 12], [[1, 0, 0]], [[1, 1, 0]], [[1, 0, 0]]),
            ([10], [[1]], [[1]], [[0]]),
            (
                [10, 11, 12],
                [[1, 0, 0], [1, 1, 0]],
                [[1, 1, 0], [1, 1, 1]],
                [[1, 0, 0], [1, 1, 0]],
            ),
        ],
    )
    def test_get_log_state(self, storage, access, illumination, expected):
        ds = RSOInspectionDataStore(MagicMock(dyn_type=FakeInspectorDyn))
        ds.storage_recorder = MagicMock(storageLevel=storage)
        ds.point_access_recorders = [
            MagicMock(hasAccess=acc_i, hasIllumination=ill_i)
            for acc_i, ill_i in zip(access, illumination)
        ]
        inspected, illuminated = ds.get_log_state()
        assert illuminated == illumination
        assert inspected == expected

    def test_compare_log_states(self):
        ds = RSOInspectionDataStore(MagicMock(dyn_type=FakeInspectorDyn))
        ds.data = MagicMock(
            point_inspect_status=dict(a=None, b=None),
            point_illuminate_status=dict(a=None, b=None),
        )
        data = ds.compare_log_states(
            None, ([[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 0]])
        )
        assert data.point_inspect_status == dict(a=True)
        assert data.point_illuminate_status == dict(b=True)


class TestRSOInspectionReward:
    def test_reward(self):
        r = RSOInspectionReward()
        r.scenario = MagicMock(rso_points=list(range(100)))
        r.scenario.satellites[0].simulator.sim_time = 0.0
        r.data = RSOInspectionData()
        reward = r.calculate_reward({"Sat": RSOInspectionData(dict(a=True, b=False))})
        assert reward == {"Sat": 0.01}

    def test_reward_scaled(self):
        r = RSOInspectionReward(inspection_reward_scale=10)
        r.scenario = MagicMock(rso_points=list(range(100)))
        r.scenario.satellites[0].simulator.sim_time = 0.0
        r.data = RSOInspectionData()
        reward = r.calculate_reward({"Sat": RSOInspectionData(dict(a=True, b=False))})
        assert reward == {"Sat": 0.1}

    def test_reward_already_yielded(self):
        r = RSOInspectionReward()
        r.scenario = MagicMock(rso_points=list(range(100)))
        r.scenario.satellites[0].simulator.sim_time = 0.0
        r.data = RSOInspectionData(dict(a=True, b=False))
        reward = r.calculate_reward({"Sat": RSOInspectionData(dict(a=True, b=True))})
        assert reward == {"Sat": 0.01}

    def test_time_completion(self):
        """Meet the condition of inspected > completion_threshold*illuminated at timeout"""
        r = RSOInspectionReward(
            min_time_for_completion=100.0,
            completion_bonus=10.0,
            min_illuminated_for_completion=1.0,
            completion_threshold=0.8,
        )
        r.bonus_reward_yielded = False
        r.cum_reward = {"Sat": 0.0}
        N = 10
        N_illuminated = 5
        N_seen = 4
        r.scenario = MagicMock(rso_points=list(range(N)))
        r.scenario.satellites[0].simulator.sim_time = 101.0
        r.data = RSOInspectionData({n: False for n in range(N)})
        reward = r.calculate_reward(
            {
                "Sat": RSOInspectionData(
                    {n: True for n in range(N_seen)},
                    {n: True for n in range(N_illuminated)},
                )
            }
        )
        assert reward == {"Sat": 10.0 + 0.4}

    def test_percent_completion(self):
        """Meet the condition of inspected > completion_threshold*illuminated over min illumination"""
        r = RSOInspectionReward(
            min_time_for_completion=100.0,
            completion_bonus=10.0,
            min_illuminated_for_completion=0.5,
            completion_threshold=0.8,
        )
        r.bonus_reward_yielded = False
        r.cum_reward = {"Sat": 0.0}
        N = 10
        N_illuminated = 5
        N_seen = 4
        r.scenario = MagicMock(rso_points=list(range(N)))
        r.scenario.satellites[0].simulator.sim_time = 0.0
        r.data = RSOInspectionData({n: False for n in range(N)})
        reward = r.calculate_reward(
            {
                "Sat": RSOInspectionData(
                    {n: True for n in range(N_seen)},
                    {n: True for n in range(N_illuminated)},
                )
            }
        )
        assert reward == {"Sat": 10.0 + 0.4}

    def test_termination(self):
        r = RSOInspectionReward(
            min_time_for_completion=100.0,
            completion_bonus=10.0,
            terminate_on_completion=True,
        )
        r.bonus_reward_yielded = False
        r.cum_reward = {"Sat": 0.0}
        r.scenario = MagicMock(rso_points=list(range(2)))
        r.scenario.satellites[0].simulator.sim_time = 101.0
        r.data = RSOInspectionData(dict(a=False, b=False))
        r.calculate_reward({"Sat": RSOInspectionData(dict(a=True, b=True))})
        assert r.is_terminated(MagicMock())
        assert r.bonus_reward_yielded
