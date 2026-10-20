"""Contradictory or unassigned samples must not become accepted learned data."""
import io
import json
import pickle
from collections import OrderedDict

import numpy as np
import pytest

from examples.multiagent_imaging.audit_failed_runs import audit_updates, compact_episode
from examples.multiagent_imaging.finish_failed_run_audit import numeric_weights, terminal_log_evidence


def episode(identity):
    return dict(lifetime_episode_id=identity, episode_id=identity,
        worker_index=1, episode_seed=10, simulated_seconds=45000,
        coverage={'catalog_target_count':200}, initial_conditions_sha256='scene',
        policy_decisions={f'sensor_{i}':1 for i in range(4)},
        action_selection_counts={f'sensor_{i}':{'image':1} for i in range(4)},
        reward_events=[dict(reward=1.)], constellation_reward=1.)


def update(ep):
    return dict(iteration=1, episodes=[ep], actual_policy_decisions=4,
        condensed_learner_rows=4, complete_episode_count=1,
        losses={'total_loss':.1}, gradient_l2=1., finite_gradients=1.,
        parameter_change=dict(actor_l2=1.,critic_l2=1.,l2=2.),
        restore_validation={'matched_actions':True},
        repeated_learner_presentations=120, policy_decisions_lifetime=4,
        checkpoint='checkpoint_0001')


def test_spooled_but_unassigned_mission_is_separate_from_accepted_update(tmp_path):
    accepted, extra = episode('accepted'), episode('extra')
    spools = {e['lifetime_episode_id']:compact_episode(e) for e in (accepted,extra)}
    result = audit_updates(io.StringIO(json.dumps([update(accepted)])),spools,'training',tmp_path/'audit.json')
    assert result['learned_missions'] == 1 and result['learned_true_decisions'] == 4
    assert result['sampled_but_unassigned_missions'] == 1
    assert result['unassigned_sampled_true_decisions'] == 4


@pytest.mark.parametrize('fault',['learner_rows','replay','conflicting_spool'])
def test_conflicting_evidence_fails_closed(tmp_path,fault):
    ep = episode('one')
    spools = {'one':compact_episode(ep)}
    rows = [update(ep)]
    if fault == 'learner_rows':
        rows[0]['condensed_learner_rows'] = 5
    elif fault == 'replay':
        rows.append(dict(update(ep),iteration=2,policy_decisions_lifetime=8))
    else:
        spools['one']['content_sha256'] = 'wrong'
    with pytest.raises(AssertionError):
        audit_updates(io.StringIO(json.dumps(rows)),spools,'training',tmp_path/'audit.json')


def test_numpy_checkpoint_decoder_accepts_only_numeric_state(tmp_path):
    path=tmp_path/'weights.pt'
    expected=OrderedDict(weight=np.arange(4,dtype=np.float32))
    path.write_bytes(pickle.dumps(expected,protocol=5))
    np.testing.assert_array_equal(numeric_weights(path)['weight'],expected['weight'])
    path.write_bytes(pickle.dumps(tmp_path,protocol=5))
    with pytest.raises(pickle.UnpicklingError,match='Non-numeric'):
        numeric_weights(path)


def test_generic_ray_oom_suggestion_is_not_a_confirmed_oom(tmp_path):
    path=tmp_path/'log'
    path.write_text('Worker PID: 42 Worker exit type: SYSTEM_ERROR possible causes OOM or SIGSEGV\n')
    result=terminal_log_evidence(path,123)
    assert not result['allocation_oom_kill_confirmed']
    assert result['unexpected_worker_exits'][0]['pid'] == 42
    path.write_text(path.read_text() + 'slurmstepd: error: Detected 1 oom_kill event in StepId=123.batch.\n')
    assert terminal_log_evidence(path,123)['allocation_oom_kill_confirmed']
    assert not terminal_log_evidence(path,456)['allocation_oom_kill_confirmed']
