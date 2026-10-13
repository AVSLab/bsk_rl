"""No Ray initialization, mission rollout, or PPO optimization."""
from types import SimpleNamespace as NS
import copy
import torch
import pytest
from examples.multiagent_imaging import worker_initial_evidence as evidence
from examples.multiagent_imaging.cluster.study_gate import validate_parallel_cpu_allocation


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pi_head=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Dropout(.1))
        self.vf=torch.nn.Linear(2,1)
        self.config=NS(inference_only=True)
    def forward(self,*args):
        raise AssertionError('Readback must never forward')


def runner(module):
    return NS(worker_index=0,module={'imager':module},env=NS(unwrapped=NS(par_env=NS(
        default_seed=3105,_episode_seed_index=1,communicator=NS(information_case='catalog_tx')))))


@pytest.mark.parametrize('mode',[True,False])
def test_query_preserves_full_rng_episode_modes_and_weights(mode):
    m=Module();m.train(mode);r=runner(m)
    weights=copy.deepcopy(m.state_dict());rng=evidence._rng_digest()
    result=evidence.read_runner_module(r)
    assert result['module_evidence']['passed'] and evidence._rng_digest()==rng
    assert all(v is mode for v in result['module_evidence']['module_modes'].values())
    assert list(result['module_evidence']['dropout'].values())==[.1]
    assert r.env.unwrapped.par_env._episode_seed_index==1
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in weights.items())
    with torch.no_grad():m.vf.weight.add_(1)
    assert evidence.actor_digest(m)==result['module_evidence']['actor_sha256']


def test_driver_and_all_worker_digests_match_or_fail(monkeypatch):
    m=Module();r=runner(m);algo=NS(get_module=lambda _:m)
    def collect(a,read,**kwargs):
        assert kwargs['require_accounting'] and kwargs['phase']=='initial-module-readback'
        return [read(r)],{'passed':True}
    monkeypatch.setattr(evidence,'collect_all_workers',collect)
    assert evidence.initial_worker_evidence(algo)['driver_rng_unchanged']
    other=Module();r.module={'imager':other}
    with pytest.raises(AssertionError,match='weights differ'):evidence.initial_worker_evidence(algo)


def test_parallel_verifier_obeys_exact_allocation(tmp_path,monkeypatch):
    (tmp_path/'allocation-gate.json').write_text('{"allocated_cpus":56}')
    resources=dict(ray_cpus=56,num_env_runners=28,num_learners=0,torch_threads=1)
    monkeypatch.setenv('BSK_EXPECTED_CPUS','56')
    assert validate_parallel_cpu_allocation(resources,tmp_path)==56
    for key,value in [('ray_cpus',32),('num_env_runners',56),('num_learners',1),('torch_threads',2)]:
        with pytest.raises(AssertionError):validate_parallel_cpu_allocation(dict(resources,**{key:value}),tmp_path)
