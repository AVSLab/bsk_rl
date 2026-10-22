"""Information suppression must retain physical transport and ideal ACK costs."""

from tests.unittest.multiagent.test_completion_communication import channel


def test_suppression_preserves_transport_timing_sender_knowledge_and_ack():
    normal, sensors_a, clock_a, record = channel(delay_s=7)
    blocked, sensors_b, clock_b, _ = channel(delay_s=7)
    blocked._frozen_eval_suppress_catalog_updates = True
    for comm, sensors, clock in ((normal, sensors_a, clock_a), (blocked, sensors_b, clock_b)):
        comm.begin_transmission(sensors[0], 20, 30)
        clock.sim_time = 50
        comm.communicate()
        assert not sensors[1].data_store.catalog.records
        assert comm.next_event_time(50) == 57
        clock.sim_time = 57
        comm.communicate()
        assert sensors[0].data_store.catalog.records == {'a': record}
        assert comm.acknowledged[('sensor_0', 'sensor_1')]['a'] == record.version
        assert comm.backlog('sensor_0') == 0
    assert sensors_a[1].data_store.catalog.records == {'a': record}
    assert sensors_b[1].data_store.catalog.records == {}
    assert normal.transmission_history == blocked.transmission_history
    for key in ('sender', 'receiver', 'sent_at', 'received_at', 'outcome', 'payload_bytes', 'record_ids', 'acknowledged_versions'):
        assert normal.delivery_history[-1][key] == blocked.delivery_history[-1][key]
    assert normal.delivery_history[-1]['changed'] > 0
    assert blocked.delivery_history[-1]['changed'] == 0
    assert blocked.delivery_history[-1]['evaluation_intervention'] == 'suppress_receiver_catalog_v1'


def test_suppression_does_not_turn_lost_packets_into_successes():
    comm, sensors, clock, _ = channel(loss_probability=1)
    comm._frozen_eval_suppress_catalog_updates = True
    comm.begin_transmission(sensors[0], 20, 30)
    clock.sim_time = 50
    comm.communicate()
    assert comm.delivery_history[-1]['outcome'] == 'lost'
    assert comm.backlog('sensor_0') == 1
    assert not sensors[1].data_store.catalog.records


def test_catalog_inventory_suppression_retains_payload_cost_and_private_ownership():
    from tests.unittest.multiagent.test_study_protocol import make_channel, product, receive
    outcomes = []
    for suppress in (False, True):
        comm, sensors, clock = make_channel()
        comm._frozen_eval_suppress_catalog_updates = suppress
        source = sensors[0].data_store.catalog
        source.record_capture(product())
        receive(comm, sensors, clock)
        receiver = sensors[1].data_store.catalog
        outcomes.append(comm.delivery_history[-1])
        assert bool(receiver.records) == (not suppress)
        assert bool(receiver.received_inventory.current(30)) == (not suppress)
        assert not sensors[1].data_store.products
        assert source.owned_inventory['a'].fact.source_sensor == 'sensor_0'
        assert comm.backlog('sensor_0') == 0
        # Suppressing peer facts must never disable later local acquisition facts.
        receiver.record_capture(product(rid='own', sensor='sensor_1', target=4, capture=31, end=32))
        assert 'own' in receiver.records
    for key in ('inventory_snapshot', 'payload_bytes', 'sent_at', 'received_at', 'outcome'):
        assert outcomes[0][key] == outcomes[1][key]
