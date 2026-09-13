"""Actual Basilisk pointing and selected-recipient delivery with three sensors."""

import numpy as np
import pytest
from bsk_rl import NO_ACTION
from bsk_rl.data.completion_catalog import CompletionRecord
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.environment import build_environment


@pytest.mark.parametrize("selected_slot", [0, 1, 2])
def test_four_sensors_point_hold_and_deliver_only_to_selected_peer(selected_slot):
    cfg = MultiAgentImagingConfig(
        n_sensors=4,
        n_targets=4,
        n_candidates=2,
        communication_mode="directed",
        information_case="completion",
        link_mode="ideal",
        episode_duration_s=500,
        transmit_duration_s=400,
        charge_duration_s=600,
        max_step_duration_s=5,
        message_delay_s=7,
        metadata_bitrate_bps=64000.0,
    )
    env = build_environment(cfg)
    try:
        env.reset(seed=33)
        sender, *possible_receivers = env.sensing_satellites
        receiver = possible_receivers[selected_slot]
        target_id = next(iter(sender.data_store.catalog.targets))
        record = CompletionRecord(
            "fixture_exposure", sender.name, target_id, 0, 0, 0, True
        )
        sender.data_store.catalog.merge_record(record, 0)
        sender.completion_peers = None
        sender.observation_builder.obs_dict_cache = None
        observation = env._get_obs()[sender.name]
        assert observation.shape == (26 + 2 * 17 + 3 * 12,)
        assert [p.name for p in sender.completion_peers.peers] == [
            peer.name for peer in possible_receivers
        ]
        assert observation[-1] == 1
        transmit_action = 5 + cfg.n_candidates + selected_slot
        env.step(
            {
                sensor.name: transmit_action if sensor is sender else 0
                for sensor in env.sensing_satellites
            }
        )
        assert sender.completion_task.receiver == receiver.name
        assert not sender.fsw.insControl.controllerStatus
        assert sender.dynamics.instrumentPowerSink.powerStatus == 0
        while env.simulator.sim_time < 400 and not env.communicator.delivery_history:
            actions = {
                s.name: (0 if s.requires_retasking else NO_ACTION)
                for s in env.sensing_satellites
            }
            env.step(actions)
        for bystander in possible_receivers:
            if bystander is not receiver:
                assert record.record_id not in bystander.data_store.catalog.records
        assert receiver.data_store.catalog.records[record.record_id] == record
        receipt = receiver.data_store.catalog.received_at[record.record_id]
        hold = next(
            h
            for h in env.communicator.transmission_history
            if h.get("outcome") == "hold_complete"
        )
        assert hold["held_s"] >= hold["required_hold_s"]
        assert hold["required_hold_s"] == 10
        assert hold["end"] >= 10
        assert receipt >= hold["end"] + 7
        assert hold["radio_on_s"] >= 10
        assert hold["payload_bytes"] > 64
        delivery = env.communicator.delivery_history[-1]
        assert delivery["receiver"] == receiver.name
        assert delivery["record_ids"] == [record.record_id]
        assert delivery["payload_bytes"] == hold["payload_bytes"]
        assert delivery["acknowledged_versions"][record.record_id] == 1
        assert not sender.data_store.products and not receiver.data_store.products
        assert sender.dynamics.storage_level_fraction == 0
        assert env.possible_agents == [s.name for s in env.sensing_satellites]
        assert len(env.passive_satellites) == 4
        assert np.isfinite(observation).all()
    finally:
        env.close()
