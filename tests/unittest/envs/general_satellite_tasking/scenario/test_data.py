from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import approx

from bsk_rl.envs.general_satellite_tasking.scenario import data


@patch.multiple(data.DataStore, __abstractmethods__=set())
class TestDataStore:
    def test_init(self):
        # Essentially a smoketest
        data.DataStore.DataType = MagicMock
        ds = data.DataStore(MagicMock(), MagicMock())
        ds._clear_logs()
        ds._get_log_state()

    def test_internal_update(self):
        # Essentially a smoketest
        data.DataStore.DataType = MagicMock
        ds = data.DataStore(MagicMock(), MagicMock())
        ds.internal_update()
        ds.internal_update()

    def test_communication_update(self):
        data.DataStore.DataType = MagicMock
        ds = data.DataStore(MagicMock(), MagicMock())
        ds.data = []
        ds.stage_communicated_data([100])
        ds.communication_update()
        assert ds.data == [100]


@patch.multiple(data.DataManager, __abstractmethods__=set())
class TestDataManager:
    def test_reset(self):
        data.DataManager.DataStore = MagicMock()
        dm = data.DataManager(MagicMock())
        dm.reset()
        assert dm.cum_reward == 0

    def test_create_data_store(self):
        sat = MagicMock()
        data.DataManager.DataStore = MagicMock(return_value="ds")
        dm = data.DataManager(MagicMock())
        dm.create_data_store(sat)
        assert sat.data_store == "ds"

    def test_reward(self):
        dm = data.DataManager(MagicMock())
        dm._calc_reward = MagicMock(return_value=10.0)
        dm.cum_reward = 0
        assert 10.0 == dm.reward({"new": "data"})
        assert dm.cum_reward == 10.0


class TestNoData:
    def test_add(self):
        dat1 = data.NoData()
        dat2 = data.NoData()
        dat = dat1 + dat2
        assert isinstance(dat, data.NoData)


class TestNoDataStore:
    def test_compare_log_states(self):
        ds = data.NoDataStore(MagicMock(), MagicMock())
        assert isinstance(ds._compare_log_states(0, 1), data.DataType)


class TestNoDataManager:
    def test_calc_reward(self):
        dm = data.NoDataManager(MagicMock())
        reward = dm._calc_reward({"sat1": 0, "sat2": 1})
        assert reward == 0


class TestUniqueImageData:
    def test_identify_duplicates(self):
        dat1 = data.UniqueImageData([1, 1, 2])
        assert dat1.duplicates == 1

    def test_add_null(self):
        dat1 = data.UniqueImageData()
        dat2 = data.UniqueImageData()
        dat = dat1 + dat2
        assert dat.imaged == []
        assert dat.duplicates == 0

    def test_add_to_null(self):
        dat1 = data.UniqueImageData(imaged=[1, 2])
        dat2 = data.UniqueImageData()
        dat = dat1 + dat2
        assert dat.imaged == [1, 2]
        assert dat.duplicates == 0

    def test_add(self):
        dat1 = data.UniqueImageData(imaged=[1, 2])
        dat2 = data.UniqueImageData(imaged=[3, 4])
        dat = dat1 + dat2
        assert dat.imaged == [1, 2, 3, 4]
        assert dat.duplicates == 0

    def test_add_duplicates(self):
        dat1 = data.UniqueImageData(imaged=[1, 2])
        dat2 = data.UniqueImageData(imaged=[2, 3])
        dat = dat1 + dat2
        assert dat.imaged == [1, 2, 3]
        assert dat.duplicates == 1

    def test_add_duplicates_existing(self):
        dat1 = data.UniqueImageData(imaged=[1, 2], duplicates=2)
        dat2 = data.UniqueImageData(imaged=[2, 3], duplicates=3)
        dat = dat1 + dat2
        assert dat.imaged == [1, 2, 3]
        assert dat.duplicates == 6


class TestUniqueImageStore:
    def test_get_log_state(self):
        sat = MagicMock()
        sat.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData = [1, 2, 3]
        ds = data.UniqueImageStore(MagicMock(), sat)
        assert (ds._get_log_state() == np.array([1, 2, 3])).all()

    @pytest.mark.parametrize(
        "before,after,imaged",
        [
            ([0, 0, 0], [0, 0, 0], []),
            ([0, 0, 1], [0, 0, 1], []),
            ([0, 0, 1], [0, 0, 0], []),
            ([0, 0, 0], [1, 0, 0], [0]),
            ([0, 0, 0], [0, 1, 1], [1, 2]),
        ],
    )
    def test_compare_log_states(self, before, after, imaged):
        sat = MagicMock()
        targets = [MagicMock() for i in range(3)]
        ds = data.UniqueImageStore(MagicMock(), sat)
        ds.env_knowledge = MagicMock(targets=targets)
        message = sat.dynamics.storageUnit.storageUnitDataOutMsg
        message.read.return_value.storedDataName.__getitem__.side_effect = (
            lambda x: targets[x].id
        )
        dat = ds._compare_log_states(np.array(before), np.array(after))
        assert len(dat.imaged) == len(imaged)
        for i in imaged:
            assert targets[i] in dat.imaged


class TestUniqueImagingManager:
    def test_calc_reward(self):
        dm = data.UniqueImagingManager(MagicMock())
        dm.data = data.UniqueImageData([])
        reward = dm._calc_reward(
            {
                "sat1": data.UniqueImageData([MagicMock(priority=0.1)]),
                "sat2": data.UniqueImageData([MagicMock(priority=0.2)]),
            }
        )
        assert reward == approx(0.3)

    def test_calc_reward_existing(self):
        tgt = MagicMock(priority=0.2)
        dm = data.UniqueImagingManager(MagicMock())
        dm.data = data.UniqueImageData([tgt])
        reward = dm._calc_reward(
            {
                "sat1": data.UniqueImageData([MagicMock(priority=0.1)]),
                "sat2": data.UniqueImageData([tgt]),
            }
        )
        assert reward == approx(0.1)

    def test_calc_reward_custom_fn(self):
        dm = data.UniqueImagingManager(MagicMock(), reward_fn=lambda x: 1 / x)
        dm.data = data.UniqueImageData([])
        reward = dm._calc_reward(
            {
                "sat1": data.UniqueImageData([MagicMock(priority=1)]),
                "sat2": data.UniqueImageData([MagicMock(priority=2)]),
            }
        )
        assert reward == approx(1.5)
