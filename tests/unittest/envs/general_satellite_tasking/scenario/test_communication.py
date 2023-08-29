from unittest.mock import MagicMock, patch

import pytest

from bsk_rl.envs.general_satellite_tasking.scenario.communication import (
    CommunicationMethod,
    FreeCommunication,
    LOSCommunication,
    MultiDegreeCommunication,
    NoCommunication,
)
from bsk_rl.envs.general_satellite_tasking.simulation.dynamics import LOSCommDynModel


@patch.multiple(CommunicationMethod, __abstractmethods__=set())
class TestCommunicationMethod:
    def test_communicate(self):
        mock_sats = [MagicMock(), MagicMock()]
        comms = CommunicationMethod(mock_sats)
        comms._communication_pairs = MagicMock(
            return_value=[(mock_sats[1], mock_sats[0])]
        )
        comms.communicate()
        mock_sats[0].data_store.stage_communicated_data.assert_called_once_with(
            mock_sats[1].data_store.data
        )
        mock_sats[1].data_store.stage_communicated_data.assert_called_once_with(
            mock_sats[0].data_store.data
        )
        for sat in mock_sats:
            sat.data_store.communication_update.assert_called_once()


class TestNoCommunication:
    def test_communicate(self):
        mock_sats = [MagicMock(), MagicMock()]
        comms = NoCommunication(mock_sats)
        comms.communicate()
        mock_sats[0].data_store.stage_communicated_data.assert_not_called()


class TestFreeCommunication:
    def test_communication_pairs(self):
        mock_sats = [MagicMock() for i in range(4)]
        comms = FreeCommunication(mock_sats)
        pairs = comms._communication_pairs()
        for sat1, sat2 in zip(mock_sats, mock_sats):
            if sat1 != sat2:
                assert (sat1, sat2) in pairs or (sat2, sat1) in pairs


class TestLOSCommunication:
    def test_dyn_model_check_valid(self):
        mock_sats = [MagicMock(dyn_type=LOSCommDynModel) for i in range(3)]
        LOSCommunication(mock_sats)

    def test_dyn_model_check_invalid(self):
        mock_sats = [MagicMock(dyn_type="NotLOSComm") for i in range(3)]
        with pytest.raises(Exception):
            LOSCommunication(mock_sats)

    class LOSDynMock(MagicMock, LOSCommDynModel):
        pass

    def test_reset(self):
        mock_sats = [
            MagicMock(dyn_type=LOSCommDynModel, dynamics=self.LOSDynMock())
            for i in range(3)
        ]
        comms = LOSCommunication(mock_sats)
        comms.reset()
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
        comms = LOSCommunication(mock_sats)
        comms.los_logs = {
            mock_sats[0]: {mock_sats[1]: MagicMock(hasAccess=access1)},
            mock_sats[1]: {mock_sats[0]: MagicMock(hasAccess=access2)},
        }
        if access:
            assert len(comms._communication_pairs()) >= 1
        else:
            assert len(comms._communication_pairs()) == 0

    @patch(
        "bsk_rl.envs.general_satellite_tasking.scenario.communication.CommunicationMethod.communicate",
        MagicMock(),
    )
    def test_communicate(self):
        mock_sats = [MagicMock(dyn_type=LOSCommDynModel) for i in range(2)]
        comms = LOSCommunication(mock_sats)
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
    @patch(
        "bsk_rl.envs.general_satellite_tasking.scenario.communication.CommunicationMethod._communication_pairs"
    )
    def test_communication_pairs(self, mock_pairs):
        mock_sats = [MagicMock() for i in range(6)]
        mock_pairs.return_value = [
            (mock_sats[0], mock_sats[1]),
            (mock_sats[1], mock_sats[3]),
            (mock_sats[3], mock_sats[4]),
        ]
        comms = MultiDegreeCommunication(mock_sats)
        pairs = comms._communication_pairs()

        assert (mock_sats[0], mock_sats[4]) in pairs
        assert (mock_sats[0], mock_sats[5]) not in pairs
