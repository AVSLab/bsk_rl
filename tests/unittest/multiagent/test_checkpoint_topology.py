"""All-actor checkpoint gates; no Ray init, mission sampling or optimizer."""
import copy
from types import SimpleNamespace as NS
import pytest
import torch
from examples.multiagent_imaging.checkpoint_topology import collect_all_workers, validate_seed_states, WorkerTopologyError
from examples.multiagent_imaging.checkpoints import _restore_runner_seed_state, _runner_seed_state


def state(index):
    return dict(worker_index=index, base_seed=100+index, next_episode_index=index+1,
                torch_rng=torch.get_rng_state().tolist(), study_audit_state=dict(episode_hashes={}, true_decisions=0, reward=0.0),
                pending_completed_episodes=[])


class Manager:
    def __init__(self, fault=None):
        self.fault=fault; self.options=[]
    def actor_ids(self): return list(range(1,29))
    def is_actor_healthy(self,index): return not (self.fault=='unhealthy' and index==4)
    def total_num_restarts(self): return int(self.fault=='restarted')
    def foreach_actor(self,reader,**kwargs):
        self.options.append(kwargs)
        results=[]
        for i in range(1,29):
            if self.fault in {'omitted','timeout'} and i==4: continue
            error=self.fault=='rpc' and i==4
            v=reader(NS(worker_index=i))
            if self.fault=='duplicate' and i==4: v['value']['worker_index']=3
            value=RuntimeError('fixture RPC failed') if error else v
            results.append(NS(actor_id=i,ok=not error,get=lambda value=value:value))
        return results


def algorithm(fault=None):
    return NS(config=NS(num_env_runners=28),env_runner_group=NS(
        local_env_runner=NS(worker_index=0),_worker_manager=Manager(fault)))


@pytest.mark.parametrize('fault',['omitted','timeout','rpc','duplicate','unhealthy','restarted'])
def test_all_actor_gate_fails_closed(fault):
    algo=algorithm(fault)
    with pytest.raises(WorkerTopologyError) as error:
        collect_all_workers(algo,lambda r:state(r.worker_index),phase='fixture',require_accounting=True)
    assert not error.value.evidence['passed']
    assert error.value.evidence['expected_indices']==list(range(29))
    for options in algo.env_runner_group._worker_manager.options:
        assert options['healthy_only'] is False and options['mark_healthy'] is False
        assert 0<=options['timeout_seconds']<=45


def test_exact_29_unique_states_and_process_restart_detection():
    algo=algorithm()
    first,e=collect_all_workers(algo,lambda r:state(r.worker_index),phase='save',require_accounting=True)
    second,_=collect_all_workers(algo,lambda r:state(r.worker_index),phase='readback',require_accounting=True)
    assert first==second and e['passed'] and not e['missing_indices']
    algo._checkpoint_worker_identities[5]={'pid':-1}
    with pytest.raises(WorkerTopologyError) as error:
        collect_all_workers(algo,lambda r:state(r.worker_index),phase='save')
    assert error.value.evidence['changed_process_indices']==[5]


@pytest.mark.parametrize('fault',['null','duplicate','rng','accounting'])
def test_manifest_boundary_rejects_incomplete_states(fault):
    states=[state(i) for i in range(29)]
    if fault=='null':states[7]=None
    if fault=='duplicate':states[7]['worker_index']=6
    if fault=='rng':states[7]['torch_rng']=[]
    if fault=='accounting':del states[7]['study_audit_state']
    with pytest.raises(WorkerTopologyError):validate_seed_states(states,list(range(29)),require_accounting=True)


def test_seed_rng_and_lifetime_restore_exact_without_stepping():
    saved=state(7);saved['study_audit_state']['true_decisions']=600
    runner=NS(worker_index=7,env=NS(unwrapped=NS(par_env=NS(default_seed=107,
        _episode_seed_index=0,communicator=NS(information_case='catalog_tx')))))
    prior=torch.get_rng_state()
    try:
        _restore_runner_seed_state(runner,[saved])
        assert _runner_seed_state(runner)==saved
        assert runner._needs_initial_reset is True
        runner.study_audit_state['true_decisions']=1
        assert saved['study_audit_state']['true_decisions']==600
        with pytest.raises(ValueError):_restore_runner_seed_state(runner,[])
    finally:torch.set_rng_state(prior)


def test_missing_worker_save_retains_failure_without_accepted_manifest(tmp_path,monkeypatch):
    from examples.multiagent_imaging import checkpoints
    monkeypatch.setattr(checkpoints,'_runner_seed_state',lambda r:state(r.worker_index))
    calls=[]
    algo=algorithm('omitted');algo.save_to_path=lambda p:calls.append(p)
    cfg=NS(is_study=True)
    with pytest.raises(WorkerTopologyError):checkpoints.save_checkpoint(algo,cfg,tmp_path/'failed',[])
    import json
    evidence=json.loads((tmp_path/'failed/worker-save-failed.json').read_text())
    assert evidence['missing_indices']==[4] and not evidence['passed']
    assert not calls and not (tmp_path/'failed/manifest.json').exists()


@pytest.mark.parametrize('fault',['omitted','advanced'])
def test_post_save_worker_failure_never_writes_accepted_manifest(tmp_path,monkeypatch,fault):
    from examples.multiagent_imaging import checkpoints,training_audit
    algo=algorithm(); phase={'after':False}
    def read(runner):
        value=state(runner.worker_index)
        if phase['after'] and fault=='advanced' and runner.worker_index==7:
            value['next_episode_index']+=1
        return value
    def save(_):
        phase['after']=True
        if fault=='omitted':algo.env_runner_group._worker_manager.fault='omitted'
    algo.save_to_path=save;algo.training_iteration=1
    module=NS(state_dict=lambda:{'weight':torch.zeros(1)},config=NS(model_config_dict={},inference_only=False))
    algo.get_module=lambda _:module
    monkeypatch.setattr(checkpoints,'_runner_seed_state',read)
    monkeypatch.setattr(checkpoints,'schema',lambda _: {})
    monkeypatch.setattr(checkpoints,'optimizer_digest',lambda _: 'fixture')
    monkeypatch.setattr(training_audit,'frozen_policy_logits',lambda m,o:torch.zeros(len(o),18))
    cfg=NS(is_study=True,to_dict=lambda:{})
    folder=tmp_path/fault
    with pytest.raises(WorkerTopologyError):checkpoints.save_checkpoint(algo,cfg,folder,[[0.]])
    import json
    evidence=json.loads((folder/'worker-save-failed.json').read_text())
    assert not evidence['passed'] and not (folder/'manifest.json').exists()
    assert (folder/'policy.pt').exists()  # Retain the failed partial artifact.
