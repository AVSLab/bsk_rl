from unittest.mock import MagicMock, patch

import pytest

from bsk_rl.comm import (
    BroadcastCommunication,
    CommunicationMethod,
    FreeCommunication,
    LOSCommunication,
    MultiDegreeCommunication,
    NoCommunication,
)
from bsk_rl.sim.dyn import LOSCommDynModel


@patch.multiple(CommunicationMethod, __abstractmethods__=set())
class TestCommunicationMethod:
    def test_communicate(self):
        mock_sats = [MagicMock(), MagicMock(), MagicMock()]
        mock_sats[0].simulator.sim_time = 0.0
        comms = CommunicationMethod()
        comms.last_communication_time = 0.0
        comms.link_satellites(mock_sats)
        comms.communication_pairs = MagicMock(
            return_value=[(mock_sats[0], mock_sats[1])]
        )
        comms.communicate()
        mock_sats[1].data_store.stage_communicated_data.assert_called_once_with(
            mock_sats[0].data_store.data
        )
        for sat in mock_sats:
            sat.data_store.update_with_communicated_data.assert_called_once()

    def test_min_period_elapsed(self):
        mock_sats = [MagicMock(), MagicMock(), MagicMock()]
        comms = CommunicationMethod(min_period=1.0)
        comms.link_satellites(mock_sats)
        comms.communication_pairs = MagicMock(
            return_value=[(mock_sats[1], mock_sats[0])]
        )
        comms.last_communication_time = 0.0
        mock_sats[0].simulator.sim_time = 2.0
        comms.communicate()
        for sat in mock_sats:
            sat.data_store.update_with_communicated_data.assert_called_once()

    def test_min_period_not_elapsed(self):
        mock_sats = [MagicMock(), MagicMock(), MagicMock()]
        comms = CommunicationMethod(min_period=1.0)
        comms.link_satellites(mock_sats)
        comms.communication_pairs = MagicMock(
            return_value=[(mock_sats[1], mock_sats[0])]
        )
        comms.last_communication_time = 0.0
        mock_sats[0].simulator.sim_time = 0.5
        comms.communicate()
        for sat in mock_sats:
            sat.data_store.update_with_communicated_data.assert_not_called()

    def test_override_communicate_all(self):
        mock_sats = [MagicMock() for i in range(3)]
        comms = FreeCommunication()
        comms.last_communication_time = 0.0
        mock_sats[0].simulator.sim_time = 1.0
        comms.link_satellites(mock_sats)
        comms._communicate_all = MagicMock()
        comms.communicate()
        comms._communicate_all.assert_called_once()

    def test_override_communicate_all_nocall(self):
        mock_sats = [MagicMock() for i in range(3)]
        comms = CommunicationMethod()
        comms.communication_pairs = MagicMock(
            return_value=[(mock_sats[1], mock_sats[0])]
        )
        comms.last_communication_time = 0.0
        mock_sats[0].simulator.sim_time = 1.0
        comms.link_satellites(mock_sats)
        comms._communicate_all = MagicMock()
        comms.communicate()
        comms._communicate_all.assert_not_called()


class TestNoCommunication:
    def test_communicate(self):
        mock_sats = [MagicMock(), MagicMock()]
        mock_sats[0].simulator.sim_time = 0.0
        comms = NoCommunication()
        comms.last_communication_time = 0.0
        comms.link_satellites(mock_sats)
        comms.communicate()
        mock_sats[0].data_store.stage_communicated_data.assert_not_called()


class TestFreeCommunication:
    def test_communication_pairs(self):
        mock_sats = [MagicMock() for i in range(4)]
        comms = FreeCommunication()
        comms.link_satellites(mock_sats)
        pairs = comms.communication_pairs()
        for sat1, sat2 in zip(mock_sats, mock_sats):
            if sat1 != sat2:
                assert (sat1, sat2) in pairs or (sat2, sat1) in pairs


class TestLOSCommunication:
    def test_dyn_model_check_valid(self):
        mock_sats = [MagicMock(dyn_type=LOSCommDynModel) for i in range(3)]
        comms = LOSCommunication()
        comms.link_satellites(mock_sats)

    def test_dyn_model_check_invalid(self):
        mock_sats = [MagicMock(dyn_type="NotLOSComm") for i in range(3)]
        with pytest.raises(Exception):
            comms = LOSCommunication()
            comms.link_satellites(mock_sats)

    class LOSDynMock(MagicMock, LOSCommDynModel):
        pass

    def test_reset(self):
        mock_sats = [
            MagicMock(dyn_type=LOSCommDynModel, dynamics=self.LOSDynMock())
            for i in range(3)
        ]
        comms = LOSCommunication()
        comms.link_satellites(mock_sats)
        comms.reset_post_sim_init()
        for sat1 in mock_sats:
            assert sat1 in comms.los_logs
            for sat2 in mock_sats:
                if sat2 != sat1:
                    assert (
                        comms.los_logs[sat1][sat2]
                        == sat1.dynamics.losComms.accessOutMsgs.__getitem__().recorder()
                    )

    @pytest.mark.parametrize(
        "access1,access2,access",
        [
            ([0, 0], [0, 0], False),
            ([0, 0], [0, 1], True),
            ([1, 0], [0, 0], True),
            ([1, 0], [0, 1], True),
        ],
    )
    def test_communication_pairs(self, access1, access2, access):
        mock_sats = [MagicMock(dyn_type=LOSCommDynModel) for i in range(2)]
        comms = LOSCommunication()
        comms.link_satellites(mock_sats)
        comms.los_logs = {
            mock_sats[0]: {mock_sats[1]: MagicMock(hasAccess=access1)},
            mock_sats[1]: {mock_sats[0]: MagicMock(hasAccess=access2)},
        }
        if access:
            assert len(comms.communication_pairs()) >= 1
        else:
            assert len(comms.communication_pairs()) == 0

    @patch(
        "bsk_rl.comm.CommunicationMethod.communicate",
        MagicMock(),
    )
    def test_communicate(self):
        mock_sats = [MagicMock(dyn_type=LOSCommDynModel) for i in range(2)]
        comms = LOSCommunication()
        comms.link_satellites(mock_sats)
        loggerA = MagicMock()
        loggerB = MagicMock()
        comms.los_logs = {
            mock_sats[0]: {mock_sats[1]: loggerA},
            mock_sats[1]: {mock_sats[0]: loggerB},
        }
        comms.communicate()
        loggerA.clear.assert_called_once()
        loggerB.clear.assert_called_once()


class TestMultiDegreeCommunication:
    @patch("bsk_rl.comm.CommunicationMethod.communication_pairs")
    def test_communication_pairs(self, mock_pairs):
        mock_sats = [MagicMock() for i in range(6)]
        mock_pairs.return_value = [
            (mock_sats[0], mock_sats[1]),
            (mock_sats[1], mock_sats[3]),
            (mock_sats[3], mock_sats[4]),
        ]
        comms = MultiDegreeCommunication()
        comms.link_satellites(mock_sats)
        pairs = comms.communication_pairs()

        assert (mock_sats[0], mock_sats[4]) in pairs
        assert (mock_sats[0], mock_sats[5]) not in pairs


class FreeBroadcast(BroadcastCommunication, FreeCommunication):
    pass


class NoBroadcast(BroadcastCommunication, NoCommunication):
    pass


class TestBroadcastCommunication:
    @pytest.mark.parametrize(
        "spec1", [[], [None, None], [MagicMock(broadcast_pending=False)]]
    )
    @pytest.mark.parametrize("broadcast1", [False, True])
    @pytest.mark.parametrize(
        "spec2", [[], [None, None], [MagicMock(broadcast_pending=False)]]
    )
    @pytest.mark.parametrize("broadcast2", [False, True])
    @pytest.mark.parametrize("comm_type", [FreeBroadcast, NoBroadcast])
    def test_communication_pairs(self, spec1, broadcast1, spec2, broadcast2, comm_type):
        mock_sats = [MagicMock() for i in range(2)]
        mock_sats[0].action_builder.action_spec = spec1
        if broadcast1:
            mock_sats[0].action_builder.action_spec.append(
                MagicMock(broadcast_pending=True)
            )
        mock_sats[1].action_builder.action_spec = spec2
        if broadcast2:
            mock_sats[1].action_builder.action_spec.append(
                MagicMock(broadcast_pending=True)
            )

        comms = comm_type()
        comms.link_satellites(mock_sats)
        pairs = comms.communication_pairs()

        if comm_type is NoBroadcast:
            assert len(pairs) == 0
        else:
            assert len(pairs) == broadcast1 + broadcast2
            if broadcast1:
                assert (mock_sats[0], mock_sats[1]) in pairs
            if broadcast2:
                assert (mock_sats[1], mock_sats[0]) in pairs
