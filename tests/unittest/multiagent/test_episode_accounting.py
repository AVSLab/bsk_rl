"""Missing parallel RPC replies must not undercount completed mission evidence."""
import hashlib
import json

import pytest

from examples.multiagent_imaging.episode_accounting import reconcile_episode_audits


def record(index, count=100):
    return dict(lifetime_episode_id=f'release:arm:{index}:1', worker_index=index,
                episode_seed=index+1000, policy_decisions={f'sensor_{i}': count+i for i in range(4)},
                episode_id=str(index), simulated_seconds=45000)


def spool_write(folder, row):
    name=hashlib.sha256(row['lifetime_episode_id'].encode()).hexdigest()+'.json'
    (folder/name).write_text(json.dumps(row))


def test_parallel_reply_recovers_four_missing_missions(tmp_path):
    all_rows=[record(i) for i in range(1,29)]
    for row in all_rows: spool_write(tmp_path,row)
    merged,evidence=reconcile_episode_audits(all_rows[:24],tmp_path)
    assert merged==all_rows and evidence['recovered_from_spool']==4
    assert evidence['worker_reply_records']==24 and evidence['completed_unique_episodes']==28
    assert evidence['new_sampled_true_decisions']==sum(sum(r['policy_decisions'].values()) for r in all_rows)


def test_next_update_and_checkpoint_resume_do_not_recount_old_rows(tmp_path):
    first,second=record(1),record(2)
    for row in (first,second): spool_write(tmp_path,row)
    merged,evidence=reconcile_episode_audits([first,second,second],tmp_path,[first['lifetime_episode_id']])
    assert merged==[second] and evidence['completed_unique_episodes']==1
    assert evidence['new_sampled_true_decisions']==406
    assert reconcile_episode_audits([],tmp_path,[r['lifetime_episode_id'] for r in (first,second)])[0]==[]


def test_conflicting_replay_and_wrong_filename_fail(tmp_path):
    row=record(1);spool_write(tmp_path,row)
    with pytest.raises(ValueError,match='Conflicting'):
        reconcile_episode_audits([record(1,99)],tmp_path)
    (tmp_path/'wrong.json').write_text(json.dumps(record(2)))
    with pytest.raises(ValueError,match='filename mismatch'):
        reconcile_episode_audits([],tmp_path)


def test_missing_spool_does_not_invent_episodes(tmp_path):
    merged,evidence=reconcile_episode_audits([record(1)],tmp_path/'missing')
    assert len(merged)==1 and evidence['recovered_from_spool']==0


def test_real_retained_callback_record_roundtrip_preserves_tuple_metadata(tmp_path):
    row=record(1);row['ack']=[('a','b','record',1)]
    spool_write(tmp_path,row)
    merged,evidence=reconcile_episode_audits([row],tmp_path)
    assert evidence['completed_unique_episodes']==1 and merged[0]['ack']==[['a','b','record',1]]


def test_actual_batch_agent_membership_and_padding_gates(tmp_path,monkeypatch):
    import gzip
    import numpy as np
    from examples.multiagent_imaging import fixed_batch_diagnostic as calculator
    from examples.multiagent_imaging.episode_accounting import verify_retained_learner_batch
    from bsk_rl import NO_ACTION
    rows=[record(1,1)]
    batch={'actions':np.zeros(10,dtype=int)}
    audit=dict(forced_no_action_count=0,padded_rows=0,passed=True)
    monkeypatch.setattr(calculator,'independently_recompute',lambda *a:(batch,audit))
    raw=[dict(multi_agent_episode_id='1',agent_id=s,actions=[0]*n+[NO_ACTION]) for s,n in rows[0]['policy_decisions'].items()]
    def write():
        with gzip.open(tmp_path/'raw-episodes.json.gz','wt') as f:json.dump(raw,f)
    write()
    result=verify_retained_learner_batch(tmp_path,None,rows,10)
    assert result['sampled_decisions_match'] and len(result['agent_decisions'])==4
    with pytest.raises(ValueError,match='totals disagree'):verify_retained_learner_batch(tmp_path,None,rows,9)
    audit['padded_rows']=1
    with pytest.raises(ValueError,match='padding'):verify_retained_learner_batch(tmp_path,None,rows,10)
    audit['padded_rows']=0;raw.pop();write()
    with pytest.raises(ValueError,match='agent decisions'):verify_retained_learner_batch(tmp_path,None,rows,10)
