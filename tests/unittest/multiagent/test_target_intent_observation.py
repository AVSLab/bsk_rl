from types import SimpleNamespace

import pytest

from bsk_rl.comm.typed_messages import IntentStatusMessage
from bsk_rl.obs.observations import _known_teammate_intent
from bsk_rl.sats.roles import SpacecraftRole


def observing_sensor(*, information_case, messages=(), peer_count=1):
    sensor = SimpleNamespace(
        information_case=information_case,
        intent_status_inbox=SimpleNamespace(
            latest_intent_by_sender={message.sender: message for message in messages}
        ),
    )
    peers = [
        SimpleNamespace(role=SpacecraftRole.SENSING_AGENT) for _ in range(peer_count)
    ]
    sensor.simulator = SimpleNamespace(sim_time=15.0, satellites=[sensor, *peers])
    return sensor


def intent(sender, target_id, *, created=10.0, expires=20.0):
    return IntentStatusMessage(
        sender=sender,
        sequence_number=0,
        target_id=target_id,
        action="Imaging",
        creation_time=created,
        expiry_time=expires,
    )


def test_independent_case_has_no_teammate_intent_leakage():
    sensor = observing_sensor(
        information_case="independent", messages=[intent("sensor_1", 3)]
    )
    assert _known_teammate_intent(sensor, {"object": SimpleNamespace(id=3)}) == 0.0


def test_intent_is_target_relational_and_freshness_weighted():
    sensor = observing_sensor(
        information_case="intent_status", messages=[intent("sensor_1", 3)]
    )
    opportunity = {"object": SimpleNamespace(id=3)}
    other = {"object": SimpleNamespace(id=4)}
    assert _known_teammate_intent(sensor, opportunity) == pytest.approx(0.5)
    assert _known_teammate_intent(sensor, other) == 0.0


def test_intent_pressure_scales_over_known_messages_without_fixed_peer_slots():
    messages = [
        intent("sensor_2", 3),
        intent("sensor_1", 3),
        intent("sensor_3", 4),
    ]
    sensor = observing_sensor(
        information_case="intent_status", messages=messages, peer_count=3
    )
    opportunity = {"object": SimpleNamespace(id=3)}
    expected = (0.5 + 0.5) / len(messages)
    assert _known_teammate_intent(sensor, opportunity) == pytest.approx(expected)

    sensor.intent_status_inbox.latest_intent_by_sender = dict(
        reversed(list(sensor.intent_status_inbox.latest_intent_by_sender.items()))
    )
    assert _known_teammate_intent(sensor, opportunity) == pytest.approx(expected)


def test_expired_intent_does_not_enter_observation_or_normalization():
    messages = [
        intent("sensor_1", 3),
        intent("sensor_2", 4, created=0.0, expires=14.0),
    ]
    sensor = observing_sensor(
        information_case="intent_status", messages=messages, peer_count=2
    )
    assert _known_teammate_intent(
        sensor, {"object": SimpleNamespace(id=3)}
    ) == pytest.approx(0.5)
