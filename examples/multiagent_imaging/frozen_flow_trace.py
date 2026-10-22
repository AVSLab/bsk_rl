"""Bounded private-input readbacks for frozen information interventions."""
import hashlib
import json


class FrozenFlowTrace:
    """At most four TX and eight post-receipt decisions; never supply actor inputs."""
    def __init__(self, env, config):
        self.env, self.config = env, config
        self.cursor = 0
        self.pending = {}
        self.sent = self.received = 0
        self.rows = []

    def capture(self, sensor, time_s, observation, action):
        if self.sent >= 4 and self.received >= 8:
            return
        packets = self.env.communicator.delivery_history
        for packet in packets[self.cursor:]:
            self.pending.setdefault(packet['receiver'], []).append(packet)
        self.cursor = len(packets)
        receiver_packets = self.pending.get(sensor.name, [])
        tx = action >= 5 + self.config.n_candidates
        keep_tx = tx and self.sent < 4
        keep_receipt = bool(receiver_packets) and self.received < 8
        if not (keep_tx or keep_receipt):
            return
        self.sent += bool(keep_tx)
        self.received += bool(keep_receipt)
        self.pending.pop(sensor.name, None)
        from bsk_rl.obs.completion_observations import private_action_mask
        candidates = getattr(sensor.completion_candidates, 'targets', ())
        peers = getattr(sensor.completion_peers, 'peers', ())
        catalog = sensor.data_store.catalog
        row = dict(sensor=sensor.name, time_s=float(time_s), selected_action=int(action),
            private_observation=list(map(float, observation)),
            action_mask=private_action_mask(sensor).tolist(),
            candidates=[None if target is None else int(target.id) for target in candidates],
            peers=[None if peer is None else peer.name for peer in peers],
            local_record_ids=sorted(catalog.records),
            owned_record_ids=sorted(catalog.owned_inventory),
            receiver_packet_sequences=[p['sequence'] for p in receiver_packets[-4:]],
            receiver_packets_since_previous_readback=len(receiver_packets))
        if keep_tx:
            slot=action-5-self.config.n_candidates
            receiver=peers[slot].name
            row.update(intended_receiver=receiver,
                       sender_sendable_record_ids=[r.record_id for r in self.env.communicator.delta(sensor.name, receiver)])
        row['private_observation_sha256']=hashlib.sha256(json.dumps(row['private_observation'],allow_nan=False).encode()).hexdigest()
        self.rows.append(row)

    def result(self):
        return dict(version='frozen-private-information-flow-v1', rows=self.rows,
                    max_transmit_readbacks=4,max_post_receipt_readbacks=8,
                    actor_inputs_unchanged=True,
                    interpretation='Packet sequences refer to raw message_diagnostics.delivery_history. Readbacks are offline evidence, never shared actor state.')
