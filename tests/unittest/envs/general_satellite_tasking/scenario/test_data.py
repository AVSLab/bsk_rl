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
        assert dm.cum_reward == {}

    def test_create_data_store(self):
        sat = MagicMock()
        data.DataManager.DataStore = MagicMock(return_value="ds")
        dm = data.DataManager(MagicMock())
        dm.reset()
        dm.create_data_store(sat)
        assert sat.data_store == "ds"
        assert sat.id in dm.cum_reward

    def test_reward(self):
        dm = data.DataManager(MagicMock())
        dm._calc_reward = MagicMock(return_value={"sat": 10.0})
        dm.cum_reward = {"sat": 5.0}
        assert {"sat": 10.0} == dm.reward({"sat": "data"})
        assert dm.cum_reward == {"sat": 15.0}


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
        assert reward == {"sat1": 0.0, "sat2": 0.0}


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
        assert reward == {"sat1": approx(0.1), "sat2": approx(0.2)}

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
        assert reward == {"sat1": approx(0.1), "sat2": 0.0}

    def test_calc_reward_repeated(self):
        tgt = MagicMock(priority=0.2)
        dm = data.UniqueImagingManager(MagicMock())
        dm.data = data.UniqueImageData([])
        reward = dm._calc_reward(
            {
                "sat1": data.UniqueImageData([tgt]),
                "sat2": data.UniqueImageData([tgt]),
            }
        )
        assert reward == {"sat1": approx(0.1), "sat2": approx(0.1)}

    def test_calc_reward_custom_fn(self):
        dm = data.UniqueImagingManager(MagicMock(), reward_fn=lambda x: 1 / x)
        dm.data = data.UniqueImageData([])
        reward = dm._calc_reward(
            {
                "sat1": data.UniqueImageData([MagicMock(priority=1)]),
                "sat2": data.UniqueImageData([MagicMock(priority=2)]),
            }
        )
        assert reward == {"sat1": approx(1.0), "sat2": 0.5}


class TestNadirScanningTimeData:
    def test_add_null(self):
        dat1 = data.NadirScanningTimeData()
        dat2 = data.NadirScanningTimeData()
        dat = dat1 + dat2
        assert dat.scanning_time == 0.0

    def test_add_to_null(self):
        dat1 = data.NadirScanningTimeData(1.0)
        dat2 = data.NadirScanningTimeData()
        dat = dat1 + dat2
        assert dat.scanning_time == 1.0

    def test_add(self):
        dat1 = data.NadirScanningTimeData(1.0)
        dat2 = data.NadirScanningTimeData(3.0)
        dat = dat1 + dat2
        assert dat.scanning_time == 4.0


class TestScanningNadirTimeStore:
    def test_get_log_state(self):
        sat = MagicMock()
        sat.dynamics.storageUnit.storageUnitDataOutMsg.read().storageLevel = 6
        ds = data.ScanningNadirTimeStore(MagicMock(), sat)
        assert ds._get_log_state() == 6.0

    @pytest.mark.parametrize(
        "before,after,new_time",
        [
            (0, 3, 1),
            (3, 6, 1),
            (1, 1, 0),
            (0, 6, 2),
        ],
    )
    def test_compare_log_states(self, before, after, new_time):
        sat = MagicMock()
        ds = data.ScanningNadirTimeStore(MagicMock(), sat)
        sat.dynamics.instrument.nodeBaudRate = 3
        dat = ds._compare_log_states(before, after)
        assert dat.scanning_time == new_time


class TestNadirScanningManager:
    def test_calc_reward(self):
        dm = data.NadirScanningManager(MagicMock())
        dm.data = data.NadirScanningTimeData([])
        dm.env_features.value_per_second = 1.0
        reward = dm._calc_reward(
            {
                "sat1": data.NadirScanningTimeData(1),
                "sat2": data.NadirScanningTimeData(2),
            }
        )
        assert reward == {"sat1": 1.0, "sat2": 2.0}

    def test_calc_reward_existing(self):
        dm = data.NadirScanningManager(MagicMock())
        dm.data = data.NadirScanningTimeData(1)
        dm.env_features.value_per_second = 1.0
        reward = dm._calc_reward(
            {
                "sat1": data.NadirScanningTimeData(2),
                "sat2": data.NadirScanningTimeData(3),
            }
        )
        assert reward == {"sat1": 2.0, "sat2": 3.0}

    def test_calc_reward_custom_fn(self):
        dm = data.NadirScanningManager(MagicMock(), reward_fn=lambda x: 1 / x)
        dm.data = data.NadirScanningTimeData([])
        reward = dm._calc_reward(
            {
                "sat1": data.NadirScanningTimeData(2),
                "sat2": data.NadirScanningTimeData(2),
            }
        )
        assert reward == {"sat1": 0.5, "sat2": 0.5}
