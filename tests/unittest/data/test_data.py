from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import approx

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.data.nadir_data import ScanningTime, ScanningTimeReward, ScanningTimeStore
from bsk_rl.data.no_data import NoData, NoDataStore, NoReward
from bsk_rl.data.resource_data import ResourceData, ResourceDataStore, ResourceReward
from bsk_rl.data.unique_image_data import (
    UniqueImageData,
    UniqueImageReward,
    UniqueImageStore,
)


@patch.multiple(DataStore, __abstractmethods__=set())
class TestDataStore:
    def test_init(self):
        # Essentially a smoketest
        DataStore.data_type = MagicMock
        ds = DataStore(MagicMock())
        ds.get_log_state()

    def test_update_from_logs(self):
        # Essentially a smoketest
        DataStore.data_type = MagicMock
        ds = DataStore(MagicMock())
        ds.update_from_logs()
        ds.update_from_logs()

    def test_update_with_communicated_data(self):
        DataStore.data_type = MagicMock
        ds = DataStore(MagicMock())
        ds.data = []
        ds.stage_communicated_data([100])
        ds.update_with_communicated_data()
        assert ds.data == [100]


@patch.multiple(GlobalReward, __abstractmethods__=set())
class TestGlobalReward:
    def test_reset(self):
        GlobalReward.data_store_type = MagicMock()
        dm = GlobalReward()
        dm.reset_overwrite_previous()
        dm.reset_pre_sim_init()
        dm.reset_post_sim_init()
        assert dm.cum_reward == {}

    def test_create_data_store(self):
        sat = MagicMock()
        data_store = MagicMock()
        GlobalReward.data_store_type = MagicMock(return_value=data_store)
        dm = GlobalReward()
        dm.data_store_kwargs = dict(hello="world")
        dm.scenario = MagicMock()
        dm.reset_overwrite_previous()
        dm.reset_pre_sim_init()
        dm.reset_post_sim_init()
        dm.create_data_store(sat)
        assert sat.data_store == data_store
        assert GlobalReward.data_store_type.call_args.kwargs["hello"] == "world"
        assert sat.name in dm.cum_reward

    def test_reward(self):
        dm = GlobalReward()
        dm.reset_overwrite_previous()
        dm.calculate_reward = MagicMock(return_value={"sat": 10.0})
        dm.cum_reward = {"sat": 5.0}
        assert {"sat": 10.0} == dm.reward({"sat": "data"})
        assert dm.cum_reward == {"sat": 15.0}


class TestNoData:
    def test_add(self):
        dat1 = NoData()
        dat2 = NoData()
        dat = dat1 + dat2
        assert isinstance(dat, NoData)


class TestNoDataStore:
    def test_compare_log_states(self):
        ds = NoDataStore(MagicMock())
        assert isinstance(ds.compare_log_states(0, 1), Data)


class TestNoGlobalReward:
    def test_calculate_reward(self):
        dm = NoReward()
        reward = dm.calculate_reward({"sat1": 0, "sat2": 1})
        assert reward == {"sat1": 0.0, "sat2": 0.0}


class TestUniqueImageData:
    def test_identify_duplicates(self):
        dat1 = UniqueImageData([1, 1, 2])
        assert dat1.duplicates == 1

    def test_add_null(self):
        dat1 = UniqueImageData()
        dat2 = UniqueImageData()
        dat = dat1 + dat2
        assert dat.imaged == []
        assert dat.duplicates == 0

    def test_add_to_null(self):
        dat1 = UniqueImageData(imaged=[1, 2])
        dat2 = UniqueImageData()
        dat = dat1 + dat2
        assert dat.imaged == [1, 2]
        assert dat.duplicates == 0

    def test_add(self):
        dat1 = UniqueImageData(imaged=[1, 2])
        dat2 = UniqueImageData(imaged=[3, 4])
        dat = dat1 + dat2
        assert dat.imaged == [1, 2, 3, 4]
        assert dat.duplicates == 0

    def test_add_duplicates(self):
        dat1 = UniqueImageData(imaged=[1, 2])
        dat2 = UniqueImageData(imaged=[2, 3])
        dat = dat1 + dat2
        assert dat.imaged == [1, 2, 3]
        assert dat.duplicates == 1

    def test_add_duplicates_existing(self):
        dat1 = UniqueImageData(imaged=[1, 2], duplicates=2)
        dat2 = UniqueImageData(imaged=[2, 3], duplicates=3)
        dat = dat1 + dat2
        assert dat.imaged == [1, 2, 3]
        assert dat.duplicates == 6


class TestUniqueImageStore:
    def test_get_log_state(self):
        sat = MagicMock()
        sat.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData = [1, 2, 3]
        ds = UniqueImageStore(sat)
        assert (ds.get_log_state() == np.array([1, 2, 3])).all()

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
        ds = UniqueImageStore(sat)
        ds.data.known = targets
        message = sat.dynamics.storageUnit.storageUnitDataOutMsg
        message.read.return_value.storedDataName.__getitem__.side_effect = (
            lambda x: targets[x].id
        )
        dat = ds.compare_log_states(np.array(before), np.array(after))
        assert len(dat.imaged) == len(imaged)
        for i in imaged:
            assert targets[i] in dat.imaged


class TestUniqueImagingManager:
    def test_calculate_reward(self):
        dm = UniqueImageReward()
        dm.data = UniqueImageData([])
        reward = dm.calculate_reward(
            {
                "sat1": UniqueImageData([MagicMock(priority=0.1)]),
                "sat2": UniqueImageData([MagicMock(priority=0.2)]),
            }
        )
        assert reward == {"sat1": approx(0.1), "sat2": approx(0.2)}

    def test_calculate_reward_existing(self):
        tgt = MagicMock(priority=0.2)
        dm = UniqueImageReward()
        dm.data = UniqueImageData([tgt])
        reward = dm.calculate_reward(
            {
                "sat1": UniqueImageData([MagicMock(priority=0.1)]),
                "sat2": UniqueImageData([tgt]),
            }
        )
        assert reward == {"sat1": approx(0.1), "sat2": 0.0}

    def test_calculate_reward_repeated(self):
        tgt = MagicMock(priority=0.2)
        dm = UniqueImageReward()
        dm.data = UniqueImageData([])
        reward = dm.calculate_reward(
            {
                "sat1": UniqueImageData([tgt]),
                "sat2": UniqueImageData([tgt]),
            }
        )
        assert reward == {"sat1": approx(0.1), "sat2": approx(0.1)}

    def test_calculate_reward_custom_fn(self):
        dm = UniqueImageReward(reward_fn=lambda x: 1 / x)
        dm.data = UniqueImageData([])
        reward = dm.calculate_reward(
            {
                "sat1": UniqueImageData([MagicMock(priority=1)]),
                "sat2": UniqueImageData([MagicMock(priority=2)]),
            }
        )
        assert reward == {"sat1": approx(1.0), "sat2": 0.5}


class TestNadirScanningTimeData:
    def test_add_null(self):
        dat1 = ScanningTime()
        dat2 = ScanningTime()
        dat = dat1 + dat2
        assert dat.scanning_time == 0.0

    def test_add_to_null(self):
        dat1 = ScanningTime(1.0)
        dat2 = ScanningTime()
        dat = dat1 + dat2
        assert dat.scanning_time == 1.0

    def test_add(self):
        dat1 = ScanningTime(1.0)
        dat2 = ScanningTime(3.0)
        dat = dat1 + dat2
        assert dat.scanning_time == 4.0


class TestScanningNadirTimeStore:
    def test_get_log_state(self):
        sat = MagicMock()
        sat.dynamics.storageUnit.storageUnitDataOutMsg.read().storageLevel = 6
        ds = ScanningTimeStore(sat)
        assert ds.get_log_state() == 6.0

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
        ds = ScanningTimeStore(sat)
        sat.dynamics.instrument.nodeBaudRate = 3
        dat = ds.compare_log_states(before, after)
        assert dat.scanning_time == new_time


class TestNadirScanningManager:
    def test_calculate_reward(self):
        dm = ScanningTimeReward()
        dm.scenario = MagicMock()
        dm.data = ScanningTime([])
        dm.scenario.value_per_second = 1.0
        reward = dm.calculate_reward(
            {
                "sat1": ScanningTime(1),
                "sat2": ScanningTime(2),
            }
        )
        assert reward == {"sat1": 1.0, "sat2": 2.0}

    def test_calculate_reward_existing(self):
        dm = ScanningTimeReward()
        dm.scenario = MagicMock()
        dm.data = ScanningTime(1)
        dm.scenario.value_per_second = 1.0
        reward = dm.calculate_reward(
            {
                "sat1": ScanningTime(2),
                "sat2": ScanningTime(3),
            }
        )
        assert reward == {"sat1": 2.0, "sat2": 3.0}

    def test_calculate_reward_custom_fn(self):
        dm = ScanningTimeReward(reward_fn=lambda x: 1 / x)
        dm.data = ScanningTime([])
        reward = dm.calculate_reward(
            {
                "sat1": ScanningTime(2),
                "sat2": ScanningTime(2),
            }
        )
        assert reward == {"sat1": 0.5, "sat2": 0.5}


class TestResourceData:
    def test_add_null(self):
        dat1 = ResourceData()
        dat2 = ResourceData()
        dat = dat1 + dat2
        assert dat.resource_accumulated == 0.0

    def test_add_to_null(self):
        dat1 = ResourceData(1.0)
        dat2 = ResourceData()
        dat = dat1 + dat2
        assert dat.resource_accumulated == 1.0

    def test_add(self):
        dat1 = ResourceData(1.0)
        dat2 = ResourceData(3.0)
        dat = dat1 + dat2
        assert dat.resource_accumulated == 4.0


class TestResourceDataStore:
    def test_get_log_state(self):
        sat = MagicMock()
        sat.resource_level = 6
        ds = ResourceDataStore(sat, resource_fn=lambda sat: sat.resource_level)
        assert ds.get_log_state() == 6.0

    @pytest.mark.repeat(10)
    def test_compare_log_states(self):
        before = np.random.randint(0, 10)
        after = np.random.randint(0, 10)
        delta = after - before
        sat = MagicMock()
        ds = ResourceDataStore(sat)
        dat = ds.compare_log_states(before, after)
        assert dat.resource_accumulated == delta


class TestResourceReward:
    def test_calculate_reward(self):
        dm = ResourceReward(reward_weight=2.0)
        dm.reset_overwrite_previous()
        dm.reset_pre_sim_init()
        reward = dm.calculate_reward(
            {
                "sat1": ResourceData(1.0),
                "sat2": ResourceData(-2.0),
            }
        )
        assert reward == {"sat1": 2.0, "sat2": -4.0}

    def test_calculate_random_reward(self):
        dm = ResourceReward(reward_weight=lambda: 2.0)
        dm.reset_overwrite_previous()
        dm.reset_pre_sim_init()
        reward = dm.calculate_reward(
            {
                "sat1": ResourceData(1.0),
                "sat2": ResourceData(-2.0),
            }
        )
        assert reward == {"sat1": 2.0, "sat2": -4.0}

    def test_read_reward(self):
        dm = ResourceReward(resource_fn=lambda sat: sat.resource_level)
        dm.reset_overwrite_previous()
        dm.reset_pre_sim_init()
        sat = MagicMock()
        dm.create_data_store(sat)
        sat.resource_level = 3.0
        assert sat.data_store.get_log_state() == 3.0
