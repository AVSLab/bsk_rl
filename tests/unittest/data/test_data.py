from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import approx

from bsk_rl.data.base import Data, DataStore, GlobalReward
from bsk_rl.data.nadir_data import ScanningTime, ScanningTimeReward, ScanningTimeStore
from bsk_rl.data.no_data import NoData, NoDataStore, NoReward
from bsk_rl.data.rso_targets_data import RSOTargetImageData, RSOTargetImageReward
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
        GlobalReward.datastore_type = MagicMock()
        dm = GlobalReward()
        dm.reset_overwrite_previous()
        dm.reset_pre_sim_init()
        dm.reset_post_sim_init()
        assert dm.cum_reward == {}

    def test_create_data_store(self):
        sat = MagicMock()
        GlobalReward.datastore_type = MagicMock(return_value="ds")
        dm = GlobalReward()
        dm.scenario = MagicMock()
        dm.reset_overwrite_previous()
        dm.reset_pre_sim_init()
        dm.reset_post_sim_init()
        dm.create_data_store(sat)
        assert sat.data_store == "ds"
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


class TestRSOTargetImageData:
    def test_pending_target_is_not_eligible_when_hidden(self):
        target = MagicMock(id=1)
        data = RSOTargetImageData(known=[target], hide_pending_targets=True)

        data.mark_target_pending(
            target,
            {
                "record_id": "capture-1",
                "target_id": 1,
                "mean_hold_shadow_factor": 0.1,
            },
        )

        assert data.target_lifecycle_state(target, sim_time=10.0) == "pending_verification"
        assert not data.is_target_eligible(target, sim_time=10.0)

    def test_failed_pending_image_can_be_reimaged_immediately(self):
        target = MagicMock(id=1)
        data = RSOTargetImageData(known=[target], hide_pending_targets=True)
        data.mark_target_pending(target, {"record_id": "capture-1", "target_id": 1})
        data.mark_target_cooldown(target, cooldown_until=100.0)

        record = data.pop_pending_record(target)
        data.mark_record_verified(record, useful=False)
        data.clear_target_cooldown(target)

        assert data.target_lifecycle_state(target, sim_time=10.0) == "eligible"
        assert data.is_target_eligible(target, sim_time=10.0)

    def test_useful_verified_image_enters_cooldown(self):
        target = MagicMock(id=1)
        data = RSOTargetImageData(known=[target], hide_pending_targets=True)
        data.mark_target_pending(target, {"record_id": "capture-1", "target_id": 1})

        record = data.pop_pending_record(target)
        data.mark_record_verified(record, useful=True)
        data.mark_target_imaged(target)
        data.mark_target_cooldown(target, cooldown_until=100.0)

        assert data.target_lifecycle_state(target, sim_time=50.0) == "cooldown"
        assert not data.is_target_eligible(target, sim_time=50.0)
        assert data.target_lifecycle_state(target, sim_time=101.0) == "eligible"
        assert target in data.imaged


class TestRSOTargetImageReward:
    @staticmethod
    def _target(target_id, name, priority=1.0):
        target = MagicMock()
        target.id = target_id
        target.name = name
        target.priority = priority
        return target

    def test_verified_cooldown_is_anchored_to_capture_time(self):
        target = self._target(1, "target_1")
        data = RSOTargetImageData(known=[target], hide_pending_targets=True)
        rewarder = RSOTargetImageReward()
        rewarder.data = data
        rewarder.reimage_cooldown_s = 20.0

        scanner = MagicMock()
        scanner.data_store.data = data
        rewarder.scenario = MagicMock(satellites=[scanner])

        cooldown_until = rewarder._start_cooldown_everywhere(
            target, capture_time=100.0
        )

        assert cooldown_until == approx(120.0)
        assert data.cooldown_until_by_id[1] == approx(120.0)
        assert data.target_lifecycle_state(target, sim_time=119.0) == "cooldown"
        assert data.target_lifecycle_state(target, sim_time=2000.0) == "eligible"

    def test_zero_cooldown_releases_target_at_ground_confirmation(self):
        target = self._target(1, "target_1")
        data = RSOTargetImageData(known=[target], hide_pending_targets=True)
        rewarder = RSOTargetImageReward(
            reimage_cooldown_orbits=0.0,
            verify_image_quality_on_downlink=True,
        )
        rewarder.data = data
        rewarder.reimage_cooldown_s = 0.0

        scanner = MagicMock()
        scanner.data_store.data = data
        rewarder.scenario = MagicMock(satellites=[scanner])
        data.mark_target_pending(target, {"record_id": "capture-1", "target_id": 1})

        assert not data.is_target_eligible(target, sim_time=200.0)
        data.pop_pending_record(target)
        rewarder._start_cooldown_everywhere(target, capture_time=100.0)

        assert data.cooldown_until_by_id[1] == approx(100.0)
        assert data.is_target_eligible(target, sim_time=200.0)

    def test_downlink_verifies_only_decreased_partition(self):
        target_0 = self._target(0, "target_0", priority=3.0)
        target_1 = self._target(1, "target_1", priority=7.0)
        rewarder = RSOTargetImageReward(verify_image_quality_on_downlink=True)
        rewarder.data = RSOTargetImageData(
            known=[target_0, target_1], hide_pending_targets=True
        )
        rewarder.reimage_cooldown_s = 20.0
        rewarder.old_state = np.array([10.0, 10.0])

        scanner = MagicMock()
        scanner.name = "SS1"
        scanner.simulator.sim_time = 2000.0
        scanner.simulator.time_limit = 100000.0
        scanner.dynamics.penalties = 0
        scanner.dynamics.eclipse_threshold_for_reward = 0.5
        scanner.dynamics.imaging_bonus = 0.0
        scanner.dynamics.downlink_bonus = 1.0
        scanner.data_store.data = RSOTargetImageData(
            known=[target_0, target_1], hide_pending_targets=True
        )
        scanner.dynamics.storageUnit.storageUnitDataOutMsg.read.return_value = (
            SimpleNamespace(
                storedData=[0.0, 10.0],
                storedDataName=["target_0", "target_1"],
            )
        )

        rewarder.scenario = MagicMock(
            satellites=[scanner], target_spacecrafts=[target_0, target_1]
        )
        pending = RSOTargetImageData(
            pending_image_records_by_id={
                0: [
                    {
                        "record_id": "capture-0",
                        "target_id": 0,
                        "target_name": "target_0",
                        "capture_time": 100.0,
                        "mean_hold_shadow_factor": 1.0,
                    }
                ],
                1: [
                    {
                        "record_id": "capture-1",
                        "target_id": 1,
                        "target_name": "target_1",
                        "capture_time": 150.0,
                        "mean_hold_shadow_factor": 1.0,
                    }
                ],
            },
            hide_pending_targets=True,
        )

        reward = rewarder._calculate_reward_with_downlink_verification({"SS1": pending})

        assert reward["SS1"] == approx(3.0)
        assert scanner.data_store.data.cooldown_until_by_id[0] == approx(120.0)
        assert not scanner.data_store.data.is_target_pending(target_0)
        assert scanner.data_store.data.is_target_pending(target_1)
        assert 1 not in scanner.data_store.data.cooldown_until_by_id


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
