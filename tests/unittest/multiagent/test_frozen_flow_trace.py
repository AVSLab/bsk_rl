"""Bounded private-input evidence and honest separation from learner advantages."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

from examples.multiagent_imaging.frozen_flow_trace import FrozenFlowTrace
from examples.multiagent_imaging.compare_learned_checkpoints import annotate_frozen_trace


def test_flow_trace_is_bounded_and_keeps_the_selected_receivers_identity(monkeypatch):
    from bsk_rl.obs import completion_observations
    monkeypatch.setattr(completion_observations,'private_action_mask',lambda s:np.ones(18))
    comm=NS(delivery_history=[],delta=lambda sender,receiver:[NS(record_id='frozen-local-fact')])
    sensor=NS(name='sender',completion_candidates=NS(targets=[NS(id=7)]),
              completion_peers=NS(peers=[NS(name='receiver')]),
              data_store=NS(catalog=NS(records={'own':1},owned_inventory={'own':1})))
    trace=FrozenFlowTrace(NS(communicator=comm),NS(n_candidates=10))
    for i in range(30):
        comm.delivery_history.append(dict(sequence=i,receiver='sender'))
        trace.capture(sensor,i,[.1,.2],15)
    assert len(trace.rows)<=12 and trace.sent==4 and trace.received==8
    assert trace.rows[0]['intended_receiver']=='receiver'
    assert trace.rows[0]['sender_sendable_record_ids']==['frozen-local-fact']
    assert trace.result()['actor_inputs_unchanged']


def test_frozen_trace_reports_observed_reward_not_a_fabricated_PPO_advantage():
    module=NS(vf=lambda batch:torch.tensor([2.]))
    result=dict(information_flow_trace=dict(rows=[dict(sensor='sender',time_s=0.,private_observation=[1.])]),
        reward_event_audit=[dict(sensor='sender',time_s=10.,reward=1.,components=dict(credited_science=1.,owned_science=0.))])
    annotate_frozen_trace(module,result,NS(discount_per_s=.9,effective_gae_lambda=.8))
    row=result['information_flow_trace']['rows'][0]
    assert row['frozen_private_critic_value']==2.
    assert row['discounted_observed_reward_tail_without_bootstrap']==pytest.approx(.9**10)
    assert row['later_shared_science']['GAE_trace_attenuation']==pytest.approx(.8**10)
    assert row['PPO_advantage'] is None
