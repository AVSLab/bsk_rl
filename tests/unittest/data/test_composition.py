from unittest.mock import MagicMock

import pytest

from bsk_rl.data.composition import ComposedData, ComposedDataStore, ComposedReward


class TestComposedData:
    @pytest.mark.parametrize(
        "data1, data2, expected",
        [
            ((1, 2), (3, 4), (4, 6)),
            ((), (3, 4), (3, 4)),
            ((1, 2), (), (1, 2)),
            ((), (), ()),
        ],
    )
    def test_add(self, data1, data2, expected):
        composed_data = ComposedData(*data1)
        other_composed_data = ComposedData(*data2)
        result = composed_data + other_composed_data
        assert result.data == expected

    def test_add_different_lengths(self):
        composed_data = ComposedData(1, 2)
        other_composed_data = ComposedData(3)
        with pytest.raises(ValueError):
            composed_data + other_composed_data

    def test_getattr(self):
        data1 = MagicMock()
        data1.a = 1
        del data1.b
        del data1.c
        data2 = MagicMock()
        data2.b = 2
        del data2.a
        del data2.c
        composed_data = ComposedData(data1, data2)
        assert composed_data.a == 1
        assert composed_data.b == 2
        with pytest.raises(AttributeError):
            _ = composed_data.c


class TestComposedDataStore:
    def test_pass_data(self):
        sat = MagicMock()
        ds1 = MagicMock()
        ds1_type = MagicMock(return_value=ds1)
        ds2 = MagicMock()
        ds2_type = MagicMock(return_value=ds2)
        composed_data_store = ComposedDataStore(sat, ds1_type, ds2_type)
        composed_data_store.data = ComposedData(1, 2)
        composed_data_store.pass_data()
        assert ds1.data == 1
        assert ds2.data == 2

    def test_getattr(self):
        sat = MagicMock()
        ds1 = MagicMock()
        ds1_type = MagicMock(return_value=ds1)
        ds2 = MagicMock()
        ds2_type = MagicMock(return_value=ds2)
        composed_data_store = ComposedDataStore(sat, ds1_type, ds2_type)
        ds1.a = 1
        assert composed_data_store.a == 1

    def test_get_log_state(self):
        sat = MagicMock()
        ds1 = MagicMock(get_log_state=MagicMock(return_value=1))
        ds1_type = MagicMock(return_value=ds1)
        ds2 = MagicMock(get_log_state=MagicMock(return_value=2))
        ds2_type = MagicMock(return_value=ds2)
        composed_data_store = ComposedDataStore(sat, ds1_type, ds2_type)

        log_states = composed_data_store.get_log_state()
        for ds in [ds1, ds2]:
            ds.get_log_state.assert_called_once()
        assert log_states == [1, 2]

    def test_compare_log_states(self):
        sat = MagicMock()
        ds1 = MagicMock(get_log_state=MagicMock(return_value=1))
        ds1_type = MagicMock(return_value=ds1)
        ds2 = MagicMock(get_log_state=MagicMock(return_value=2))
        ds2_type = MagicMock(return_value=ds2)
        composed_data_store = ComposedDataStore(sat, ds1_type, ds2_type)

        composed_data_store.compare_log_states([1, 2], [3, 4])
        ds1.compare_log_states.assert_called_once_with(1, 3)
        ds2.compare_log_states.assert_called_once_with(2, 4)


class TestComposedReward:
    def test_pass_data(self):
        rewarder1 = MagicMock()
        rewarder2 = MagicMock()
        composed_rewarder = ComposedReward(rewarder1, rewarder2)
        composed_rewarder.data = ComposedData(1, 2)
        composed_rewarder.pass_data()
        assert rewarder1.data == 1
        assert rewarder2.data == 2

    @pytest.mark.parametrize(
        "function",
        [
            "reset_pre_sim_init",
            "reset_post_sim_init",
            "reset_overwrite_previous",
        ],
    )
    def test_resetable(self, function):
        rewarder1 = MagicMock()
        rewarder2 = MagicMock()
        composed_rewarder = ComposedReward(rewarder1, rewarder2)
        getattr(composed_rewarder, function)()
        for rewarder in [rewarder1, rewarder2]:
            getattr(rewarder, function).assert_called_once()

    def test_initial_data(self):
        rewarder1 = MagicMock(
            initial_data=MagicMock(return_value=1),
        )
        rewarder2 = MagicMock(initial_data=MagicMock(return_value=2))
        composed_rewarder = ComposedReward(rewarder1, rewarder2)
        data = composed_rewarder.initial_data("sat")
        assert data.data == (1, 2)

    def test_data_store(self):
        sat = MagicMock()
        ds1 = MagicMock(get_log_state=MagicMock(return_value=1))
        ds1_type = MagicMock(return_value=ds1)
        rewarder1 = MagicMock(
            data_store_type=ds1_type,
            data_store_kwargs=dict(hello="world"),
        )
        ds2 = MagicMock(get_log_state=MagicMock(return_value=2))
        ds2_type = MagicMock(return_value=ds2)
        rewarder2 = MagicMock(data_store_type=ds2_type)
        composed_rewarder = ComposedReward(rewarder1, rewarder2)
        composed_rewarder.create_data_store(sat)
        assert sat.data_store.data_stores == (ds1, ds2)
        assert ds1_type.call_args.kwargs["hello"] == "world"
        assert "hello" not in ds2_type.call_args.kwargs

    def test_calculate_reward(self):
        rewarder1 = MagicMock(
            calculate_reward=MagicMock(return_value={"sat1": 1, "sat2": 2})
        )
        rewarder2 = MagicMock(
            calculate_reward=MagicMock(return_value={"sat1": 3, "sat2": 4})
        )
        composed_rewarder = ComposedReward(rewarder1, rewarder2)

        reward = composed_rewarder.calculate_reward(
            {
                "sat1": MagicMock(data=["d11", "d21"]),
                "sat2": MagicMock(data=["d12", "d22"]),
            }
        )

        assert reward == {"sat1": 4, "sat2": 6}
