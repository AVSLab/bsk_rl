"""Qualification and treatment gates for the future 32-mission evaluation."""
import copy
import json

import pytest

from examples.multiagent_imaging.frozen_evaluation import (
    cases, paired_summary, physical_metrics, prepare, qualify, reserved_seed_check, run, validate_plan,
)


def test_plan_has_exactly_32_cases_and_no_training_or_submission():
    plan = prepare()
    assert len(cases(plan)) == 32
    assert len({(c['condition']['label'],c['seed'],c['mode']) for c in cases(plan)}) == 32
    assert not plan['submitted'] and plan['new_PPO_updates'] == 0
    assert plan['proposed_allocation']['allocated_cpu_hours'] == 32
    assert plan['terminal_qualification_pending']
    assert plan['conditions'][0]['config'] == plan['conditions'][1]['config']
    assert plan['conditions'][2]['config'] != plan['conditions'][0]['config']


@pytest.mark.parametrize('change', ['seed','intervention','condition','source','capacity'])
def test_plan_rejects_silent_contract_changes(change):
    plan = prepare()
    if change=='seed': plan['seeds'][0]=3105
    elif change=='intervention': plan['conditions'][1]['metadata_intervention']='normal'
    elif change=='condition': plan['conditions'][2]['arm']='with_communication'
    elif change=='source': plan['source_commit']='newsource'
    else: plan['maximum_workers']=29
    with pytest.raises(ValueError): validate_plan(plan)


def test_incomplete_terminal_audit_cannot_bind_final_checkpoint():
    with pytest.raises(ValueError,match='terminal'):
        qualify(prepare(),dict(version='cooperative-pair-terminal-qualification-v1',passed=False))


def test_predeclared_seeds_are_outside_all_29_full_budget_streams():
    proof=reserved_seed_check()
    assert proof['passed'] and len(proof['worker_seed_intervals'])==29
    assert proof['terminal_observed_seed_check_still_required']


def test_existing_native_episode_reconstructs_physical_metrics_without_running_mission():
    import gzip
    from pathlib import Path
    root=Path(__file__).parents[3]
    with gzip.open(root/'examples/multiagent_imaging/cluster/evidence/cooperative-v1-20261021/native-replay-1-episode.json.gz','rt') as f:
        e=json.load(f)
    config=json.loads((root/'examples/multiagent_imaging/configs/cooperative_v1_shared_catalog_tx.json').read_text())
    metrics=physical_metrics(dict(raw_metric_evidence=e,cooperative_tasking=e['cooperative_tasking'],
        seed=e['episode_seed'],config=config,sim_time_s=e['simulated_seconds'],team_summary=e['team_summary'],
        broadcast_time_s=dict(team=e['communication_summary']['radio_occupancy_s']),action_time_s={},
        message_diagnostics=dict(packet_outcome_counts=e['communication_summary']['packet_outcome_counts']),
        cumulative_reward=e['reward_by_agent'],decision_records=e['decision_records']))
    assert metrics['unshaped_request_science']==pytest.approx(39.482125777264834)
    assert metrics['unique_useful_acquisitions']==91
    assert metrics['expired_requests']==112
    assert 0<=metrics['missed_delivery_deadlines']<=112
    assert metrics['completed_redundant_captures']>0
    assert metrics['transmission_occupied_s']>0


def test_full_evaluation_cannot_execute_on_login_or_local_host(monkeypatch,tmp_path):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ValueError,match='authorized compute'):
        run(prepare(),{},tmp_path/'never-created')
    assert not (tmp_path/'never-created').exists()


def test_heldout_seeds_cannot_overlap_training_or_reserved_stream():
    plan=prepare()
    base=dict(version='cooperative-pair-terminal-qualification-v1',passed=True,
              training_episode_seeds=[plan['seeds'][0]],
              reserved_worker_seed_intervals=[[100000*i,5600] for i in range(29)])
    with pytest.raises(ValueError,match='used for training'): qualify(plan,base)
    base['training_episode_seeds']=[1]
    base['reserved_worker_seed_intervals'][0]=[plan['seeds'][0]-1,5600]
    with pytest.raises(ValueError,match='reserved training'): qualify(plan,base)


def test_terminal_qualification_binds_each_checkpoint_to_its_own_contract(tmp_path):
    import hashlib
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.readiness import schema
    from examples.multiagent_imaging.frozen_evaluation import config_path, JOBS, SOURCE
    plan=prepare(); arms={}
    for arm,job in JOBS.items():
        config=MultiAgentImagingConfig.from_json(config_path(next(c for c in plan['conditions'] if c['arm']==arm)))
        acceptance=tmp_path/f'{arm}-acceptance.json';acceptance.write_text('{"passed":true}')
        receipt=dict(job_id=job,terminal_state='COMPLETED',exit_code='0:0',source_commit=SOURCE,
            acceptance_path=acceptance.name,acceptance_sha256=hashlib.sha256(acceptance.read_bytes()).hexdigest(),checkpoint_sha256={})
        for role,iteration in (('initial_checkpoint',0),('final_checkpoint',3)):
            checkpoint=tmp_path/f'{arm}-{role}';checkpoint.mkdir();(checkpoint/'policy.pt').write_bytes(b'frozen')
            (checkpoint/'manifest.json').write_text(json.dumps(dict(training_iteration=iteration,schema=schema(config),policy_sha256=hashlib.sha256(b'frozen').hexdigest())))
            receipt[role]=checkpoint.name
            receipt['checkpoint_sha256'][role]={name:hashlib.sha256((checkpoint/name).read_bytes()).hexdigest() for name in ('manifest.json','policy.pt')}
        arms[arm]=receipt
    q=dict(version='cooperative-pair-terminal-qualification-v1',passed=True,training_episode_seeds=[1],
           reserved_worker_seed_intervals=[[i*100000,5600] for i in range(29)],arms=arms)
    bound=qualify(plan,q,artifact_root=tmp_path)
    assert bound['without_communication']['final_checkpoint'].startswith(str(tmp_path))
    c=tmp_path/arms['without_communication']['final_checkpoint']/'manifest.json'
    m=json.loads(c.read_text());m['schema']['settings']['information_case']='catalog_tx';c.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='schema mismatch'):qualify(plan,q,artifact_root=tmp_path)


def test_paired_contrasts_are_directional_and_modes_remain_separate():
    rows=[]
    for condition,value in (('A',10),('B',8),('C',9),('D',7)):
        for mode in ('argmax','stochastic'):
            rows.append(dict(condition=condition,selection_mode=mode,seed=9000101,
                initial_conditions_sha256='same',physical={k:value for k in (
                'unshaped_request_science','unique_useful_acquisitions','verified_useful_deliveries',
                'completed_redundant_captures','transmission_occupied_s')}))
    summary=paired_summary(rows)
    assert summary['argmax/A_minus_B']['mean']['unshaped_request_science']==2
    assert summary['stochastic/A_minus_C']['mean']['unshaped_request_science']==1
    rows[1]['initial_conditions_sha256']='different'
    with pytest.raises(ValueError,match='initial conditions'): paired_summary(rows)


@pytest.mark.parametrize('mutate_checkpoint',[False,True])
def test_full_case_routing_and_failure_logging_without_native_or_PPO(monkeypatch,tmp_path,mutate_checkpoint):
    from concurrent.futures import Future
    from types import SimpleNamespace as NS
    from examples.multiagent_imaging import frozen_evaluation as module
    from examples.multiagent_imaging import wandb_logging, memory_profile, cooperative_eval_logging
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'policy.pt').write_bytes(b'frozen')
    arms={arm:dict(initial_checkpoint=str(checkpoint),final_checkpoint=str(checkpoint)) for arm in ('with_communication','without_communication')}
    monkeypatch.setenv('SLURM_JOB_ID','future-approved-test')
    monkeypatch.setattr(module,'qualify',lambda *args,**kw:arms)
    finished=[];seen=[];published=[]
    logger=NS(heartbeat=lambda *a:None,start_heartbeat=lambda **kw:None,
              check_background=lambda:None,finish=lambda **kw:finished.append(kw))
    monkeypatch.setattr(wandb_logging,'TrainingLogger',lambda *a,**kw:logger)
    monkeypatch.setattr(memory_profile,'MemoryProfile',lambda *a,**kw:NS(start=lambda:None,mark=lambda x:None,close=lambda:None))
    monkeypatch.setattr(memory_profile,'read_cgroup_memory',lambda:[])
    monkeypatch.setattr(cooperative_eval_logging,'publish',lambda *a,**kw:published.append(True))
    class Pool:
        def __init__(self,**kwargs): assert kwargs['max_workers']==8
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def submit(self,fn,case,*args):
            seen.append((case['condition']['label'],case['mode'],case['seed']))
            if mutate_checkpoint: (checkpoint/'policy.pt').write_bytes(b'changed')
            future=Future();future.set_result(dict(condition=case['condition']['label'],
                selection_mode=case['mode'],seed=case['seed'],initial_conditions_sha256='same',
                physical={k:1 for k in ('unshaped_request_science','unique_useful_acquisitions',
                'verified_useful_deliveries','completed_redundant_captures','transmission_occupied_s')}))
            return future
    monkeypatch.setattr(module,'ProcessPoolExecutor',Pool)
    if mutate_checkpoint:
        with pytest.raises(ValueError,match='checkpoint tree changed'):run(prepare(),{},tmp_path/'heldout')
        assert finished==[dict(failed=True)] and not published
    else:
        assert len(run(prepare(),{},tmp_path/'heldout'))==32
        assert finished==[dict(failed=False)] and published
    assert len(set(seen))==32
    summary=json.loads((tmp_path/'heldout/summary.json').read_text())
    assert summary['complete'] is not mutate_checkpoint
