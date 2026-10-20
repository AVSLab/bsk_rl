"""Causal controls, private reception, and reward-to-transmit target tracing."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from bsk_rl import NO_ACTION
from bsk_rl.comm.completion_communication import CompletionCommunication
from bsk_rl.data.completion_catalog import CompletionCatalog
from bsk_rl.data.multiagent_rso_data import ImageProductRecord
from bsk_rl.obs.completion_observations import CompletionPeers
from bsk_rl.sats.roles import SpacecraftRole
from examples.multiagent_imaging.communication_qualification import credit_trace, receiver_action, net_value_gate


def test_controller_tracks_target_identity_after_candidate_repacking():
    mask=np.ones(18,dtype=np.int8)
    assert receiver_action([24,55,62],mask)==6
    assert receiver_action([24,62,80],mask)==5
    mask[5]=0
    assert receiver_action([24,62,80],mask)==6
    assert receiver_action([None]*10,mask)==0


def test_positive_information_value_cannot_override_a_better_physical_alternative():
    gate=net_value_gate(.313,.116,dict(image=.922,downlink=.116))
    assert gate['information_value']==pytest.approx(.197)
    assert gate['net_value_against_best_tested_alternative']==pytest.approx(-.609)
    assert not gate['passed']
    assert net_value_gate(.313,.116,dict(image=.2,downlink=.116))['passed']


def channel():
    clock=NS(sim_time=20.)
    sensors=[NS(name=f'sensor_{i}',role=SpacecraftRole.SENSING_AGENT,simulator=clock,
        is_alive=lambda:True,dynamics=NS(r_BN_N=np.array([7e6,i*1e3,0.]),
        v_BN_N=np.array([0.,7500.,0.]),BN=np.eye(3)),
        fsw=NS(locPoint=NS(pHat_B=np.array([0.,1.,0.]))),completion_peers=None,
        data_store=NS(catalog=CompletionCatalog(f'sensor_{i}',[55,24],cooldown_s=1000),products=()))
        for i in range(4)]
    comm=CompletionCommunication('catalog_tx')
    comm.link_satellites(sensors)
    comm.reset_overwrite_previous()
    comm.reset_post_sim_init()
    return comm,sensors,clock


def peers(sensor,clock):
    obs=object.__new__(CompletionPeers)
    obs.satellite=sensor
    obs.simulator=clock
    obs.norm_time=12000
    sensor.completion_peers=None
    return np.concatenate(list(obs.get_obs().values()))


def test_sender_knowledge_is_private_until_selected_receipt_and_snapshot_is_frozen():
    comm,sensors,clock=channel()
    sender,receiver=sensors[0],sensors[2]
    before={s.name:peers(s,clock) for s in sensors[1:]}
    product=ImageProductRecord('p',sender.name,55,10,None,1.,sender.name,0.,15.)
    sender.data_store.catalog.record_capture(product)
    for s in sensors[1:]:
        np.testing.assert_array_equal(before[s.name],peers(s,clock))
    comm.begin_directed(sender,receiver,20.)
    frozen=comm.transmissions[sender.name].inventory_snapshot
    # A newer owner revision cannot enter the already frozen packet.
    sender.data_store.catalog.record_drop('p',21.)
    assert frozen.entries[0].status=='onboard'
    comm.transmissions[sender.name].end=30.
    clock.sim_time=30.
    at_receipt={s.name:peers(s,clock) for s in sensors[1:]}
    comm.communicate()
    assert receiver.data_store.catalog.received_inventory.status(sender.name,'p',30)=='onboard'
    assert not receiver.data_store.products
    assert not receiver.data_store.catalog.is_eligible(55,30)
    assert receiver.data_store.catalog.is_privately_eligible(55,30)
    for s in (sensors[1],sensors[3]):
        assert not s.data_store.catalog.records
        np.testing.assert_array_equal(at_receipt[s.name],peers(s,clock))
    assert not np.array_equal(at_receipt[receiver.name],peers(receiver,clock))
    prior=peers(receiver,clock)
    assert receiver.data_store.catalog.receive_inventory(frozen,30.,600.)==0
    np.testing.assert_array_equal(prior,peers(receiver,clock))


@pytest.mark.parametrize('loss,delay,ttl,outcome',[(1.,0.,600.,'lost'),(0.,600.,600.,'expired')])
def test_failed_packet_cannot_remove_receiver_candidate(loss,delay,ttl,outcome):
    comm,sensors,clock=channel()
    comm.loss_probability=loss
    comm.delay_s=delay
    comm.ttl_s=ttl
    product=ImageProductRecord('p','sensor_0',55,10,None,1.,'sensor_0',0.,15.)
    sensors[0].data_store.catalog.record_capture(product)
    comm.begin_directed(sensors[0],sensors[2],20.)
    comm.transmissions['sensor_0'].end=30.
    clock.sim_time=30.
    comm.communicate()
    clock.sim_time=630.
    comm.communicate()
    assert comm.delivery_history[-1]['outcome']==outcome
    assert not sensors[2].data_store.catalog.records
    assert sensors[2].data_store.catalog.is_eligible(55,630)


def test_delayed_shared_benefit_reaches_prior_transmit_only_through_discounted_trace():
    gamma=.9915045428565076
    lam=.8713548569911232
    # Transmit at t=0; charge at t=120; receiver earns a unique science increment
    # at t=281. Sender gets one fourth under team_mean, no credit under owned.
    raw=dict(actions=[16,0,NO_ACTION],durations=[120.,161.,39.],rewards=[0.,.25,0.])
    result=credit_trace(raw,gamma,lam)
    tx,charge=result['rows']
    assert tx['condensed_reward']==0
    assert charge['condensed_reward']==pytest.approx(.25*gamma**161)
    assert tx['diagnostic_value_target']==pytest.approx(.25*gamma**281*lam,rel=2e-6)
    owned=credit_trace(dict(raw,rewards=[0.,0.,0.]),gamma,lam)
    assert owned['rows'][0]['diagnostic_value_target']==0
    assert result['forced_rows_removed']==1
    assert result['rows'][0]['diagnostic_value_target']<.25


def test_credit_oracle_preserves_actual_amos_objective_and_busy_boundary_invariance():
    gamma=.9915045428565076
    raw=dict(actions=[16,NO_ACTION,0],durations=[10.,20.,40.],rewards=[0.,1.,2.])
    result=credit_trace(raw,gamma,.8713548569911232)
    assert result['rows'][0]['condensed_reward']==pytest.approx(gamma**30)
    assert result['rows'][0]['duration_s']==30
    assert result['rows'][1]['duration_s']==40
    # Splitting a busy interval preserves the physically timed reward and target.
    split=dict(actions=[16,NO_ACTION,NO_ACTION,0],durations=[5.,5.,20.,40.],rewards=[0.,0.,1.,2.])
    other=credit_trace(split,gamma,.8713548569911232)
    np.testing.assert_allclose([r['diagnostic_value_target'] for r in result['rows']],
                               [r['diagnostic_value_target'] for r in other['rows']])


@pytest.mark.parametrize('durations',[[0.],[-1.],[float('nan')],[]])
def test_credit_trace_rejects_malformed_physical_time(durations):
    with pytest.raises(ValueError,match='Malformed'):
        credit_trace(dict(actions=[16],durations=durations,rewards=[1.]),.99,.8)


def test_baseline_config_does_not_adopt_shared_credit():
    from pathlib import Path
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    p=Path(__file__).parents[3]/'examples/multiagent_imaging/configs/walker4_t200_cd1_catalog_tx.json'
    baseline=MultiAgentImagingConfig.from_json(p)
    assert baseline.reward_credit_mode=='owned'
    assert 'reward_credit_mode' not in baseline.to_dict()
    assert replace(baseline,reward_credit_mode='team_mean').reward_credit_mode=='team_mean'
