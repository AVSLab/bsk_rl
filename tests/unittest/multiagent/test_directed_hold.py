"""A broken lock resets packet progress; zeroed guidance is never a lock."""

from types import SimpleNamespace as NS
from bsk_rl.act.completion_actions import TransmitCompletions
from bsk_rl.comm.completion_communication import Transmission


def test_continuous_hold_restarts_after_contact_or_pointing_loss():
    clock = NS(sim_time=0.0)
    transmission = Transmission(0, float("inf"), (), {"receiver"}, True)
    channel = NS(transmissions={"sender": transmission}, transmission_history=[])
    action = TransmitCompletions(1)
    action.satellite = NS(
        name="sender",
        completion_communicator=channel,
        dynamics=NS(transmitterPowerSink=NS(powerStatus=0)),
    )
    action.simulator, action.peer = clock, NS(name="receiver")
    action.start_time = action.last_time = action.held_s = action.radio_on_s = 0.0
    action.was_valid, action.required_hold_s, action.payload_bytes = False, 10.0, 100
    action.pointing_valid = lambda: clock.sim_time != 5
    assert not action.check_hold()  # Ignore the zeroed guidance gateway at start.
    assert action.satellite.dynamics.transmitterPowerSink.powerStatus == 0
    for time in range(1, 16):
        clock.sim_time = time
        assert not action.check_hold()
        assert transmission.end == float("inf")
        if time == 5:
            assert action.held_s == 0
            assert action.satellite.dynamics.transmitterPowerSink.powerStatus == 0
    clock.sim_time = 16
    assert action.check_hold()
    assert transmission.end == 16
    assert action.held_s == 10
    assert action.satellite.dynamics.transmitterPowerSink.powerStatus == 0
