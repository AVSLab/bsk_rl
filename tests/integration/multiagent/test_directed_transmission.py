"""Actual Basilisk pointing and selected-recipient delivery with three sensors."""

import numpy as np
import pytest
from bsk_rl import NO_ACTION
from bsk_rl.data.completion_catalog import CompletionRecord
from examples.multiagent_imaging.config import MultiAgentImagingConfig
from examples.multiagent_imaging.environment import build_environment


@pytest.mark.parametrize("bitrate", [None, 64.0])
def test_three_sensors_point_hold_and_deliver_only_to_selected_peer(bitrate):
    cfg = MultiAgentImagingConfig(
        n_sensors=3,
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
        metadata_bitrate_bps=bitrate,
    )
    env = build_environment(cfg)
    try:
        env.reset(seed=33)
        sender, bystander, receiver = env.sensing_satellites
        target_id = next(iter(sender.data_store.catalog.targets))
        record = CompletionRecord(
            "fixture_exposure", sender.name, target_id, 0, 0, 0, True
        )
        sender.data_store.catalog.merge_record(record, 0)
        sender.completion_peers = None
        sender.observation_builder.obs_dict_cache = None
        observation = env._get_obs()[sender.name]
        assert observation.shape == (26 + 2 * 17 + 2 * 12,)
        assert [p.name for p in sender.completion_peers.peers] == [
            bystander.name,
            receiver.name,
        ]
        assert observation[-1] == 1
        env.step({sender.name: 8, bystander.name: 0, receiver.name: 0})
        assert sender.completion_task.receiver == receiver.name
        assert not sender.fsw.insControl.controllerStatus
        assert sender.dynamics.instrumentPowerSink.powerStatus == 0
        while env.simulator.sim_time < 400 and not env.communicator.delivery_history:
            actions = {
                s.name: (0 if s.requires_retasking else NO_ACTION)
                for s in env.sensing_satellites
            }
            env.step(actions)
        assert record.record_id not in bystander.data_store.catalog.records
        assert receiver.data_store.catalog.records[record.record_id] == record
        receipt = receiver.data_store.catalog.received_at[record.record_id]
        hold = next(
            h
            for h in env.communicator.transmission_history
            if h.get("outcome") == "hold_complete"
        )
        assert hold["held_s"] >= hold["required_hold_s"]
        if bitrate:
            assert hold["required_hold_s"] == 8 * hold["payload_bytes"] / bitrate
        assert hold["end"] >= 10
        assert receipt >= hold["end"] + 7
        assert hold["radio_on_s"] >= 10
        assert hold["payload_bytes"] > 64
        assert not sender.data_store.products and not receiver.data_store.products
        assert sender.dynamics.storage_level_fraction == 0
        assert env.possible_agents == [s.name for s in env.sensing_satellites]
        assert len(env.passive_satellites) == 4
        assert np.isfinite(observation).all()
    finally:
        env.close()
