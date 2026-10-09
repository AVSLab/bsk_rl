"""Repair gates without simulation, sampling, Ray init or optimizer updates."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
from ray.rllib.core.columns import Columns

from examples.multiagent_imaging import train, checkpoints, fixed_batch_diagnostic as fit

@pytest.mark.parametrize('iterations,resume,batch,complete,expected',[
    (0,True,4200,True,True),(0,False,4200,True,False),(-1,True,4200,True,False),
    (0,True,7,True,False),(0,True,4200,False,False),(3,False,4200,True,True)])
def test_actual_cli_validation_does_not_launch_runtime(tmp_path,monkeypatch,iterations,resume,batch,complete,expected):
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    if complete:
        (checkpoint/'manifest.json').write_text('{}');(checkpoint/'policy.pt').touch();(checkpoint/'algorithm').mkdir()
    calls=[];monkeypatch.setattr(train,'train_run',lambda *a,**k:calls.append(k))
    argv=['train','--iterations',str(iterations),'--train-batch-size',str(batch),'--config',str(Path(train.__file__).parent/'configs/walker4_mixed_amos_best_conflict.json'),'--ppo-profile','amos_best']
    if resume:argv+=['--resume',str(checkpoint)]
    monkeypatch.setattr(sys,'argv',argv)
    if expected:
        train.main();assert len(calls)==1 and calls[0]['iterations']==iterations
    else:
        with pytest.raises(SystemExit) as e:train.main()
        assert e.value.code==2 and not calls

class ProbeModule(torch.nn.Linear):
    def forward_inference(self,batch):return {Columns.ACTION_DIST_INPUTS:self(batch[Columns.OBS])}

@pytest.mark.parametrize('fault',[None,'weights','rng','optimizer'])
def test_zero_resume_readback_fails_closed(tmp_path,monkeypatch,fault):
    module=ProbeModule(2,2);obs=torch.ones(2,2);module.eval();logits=module(obs).detach()
    payload={'weights':{n:v.detach().clone() for n,v in module.state_dict().items()},'observations':obs,'logits':logits}
    torch.save(payload,tmp_path/'policy.pt')
    seed={'worker_index':0,'base_seed':10,'next_episode_index':16,'torch_rng':torch.get_rng_state().tolist()}
    (tmp_path/'manifest.json').write_text(json.dumps({'optimizer_sha256':'opt','training_iteration':3,'policy_decisions_lifetime':14938,'worker_seed_states':[seed]}))
    (tmp_path/'wandb.json').write_text(json.dumps({'id':'7yrd5jn2'}))
    runtime={'basilisk':{},'bsk_rl':{},'modules':{}}
    monkeypatch.setattr('examples.multiagent_imaging.cluster.effective_imports.require_effective_imports',lambda:runtime)
    monkeypatch.setattr(checkpoints,'optimizer_digest',lambda a:'bad' if fault=='optimizer' else 'opt')
    actual=dict(seed)
    if fault=='rng':actual['torch_rng']=[0]*len(seed['torch_rng'])
    if fault=='weights':
        with torch.no_grad():module.weight.add_(1)
    runners=SimpleNamespace(foreach_worker=lambda f:[{'seed_state':actual,'runtime':runtime}])
    algo=SimpleNamespace(get_module=lambda _:module,training_iteration=3,env_runner_group=runners,config=SimpleNamespace(num_env_runners=0))
    monkeypatch.setattr(checkpoints,'collect_all_workers',lambda *a,**k:([{'seed_state':actual,'runtime':runtime}],{'passed':True,'expected_indices':[0]}))
    if fault:
        with pytest.raises(AssertionError):checkpoints.zero_update_resume_evidence(algo,tmp_path)
    else:
        result=checkpoints.zero_update_resume_evidence(algo,tmp_path)
        assert result['passed'] and result['exact_logits'] and result['new_ppo_updates']==0
        assert result['restored_weights_sha256']==result['exported_weights_sha256']
        assert result['workers'][0]['matches_checkpoint']

@pytest.mark.parametrize('passed',[True,False])
def test_required_critic_fit_is_a_process_gate(tmp_path,monkeypatch,passed):
    inputs=tmp_path/'inputs';inputs.mkdir();output=tmp_path/'output'
    (inputs/'updates.json').write_text(json.dumps([{'episodes':[],'learner_metrics':{'vf_explained_var':0}}]))
    monkeypatch.setattr(fit,'independently_recompute',lambda *a:({}, {'physical_reward_traces':[]}))
    calls=[]
    def stub(*a,**k):calls.append(k);return {'can_fit':passed,'steps':5000,'actor_unchanged':True}
    monkeypatch.setattr(fit,'fit_critic',stub)
    config=Path(train.__file__).parent/'configs/walker4_mixed_amos_best_conflict.json'
    monkeypatch.setattr(sys,'argv',['fit','--config',str(config),'--input-root',str(inputs),'--output',str(output),'--checkpoint',str(tmp_path),'--steps','5000','--max-wall-seconds','600','--require-fit'])
    if passed:fit.main()
    else:
        with pytest.raises(SystemExit):fit.main()
    assert calls==[{'steps':5000,'max_wall_seconds':600.0}]
    assert json.loads((output/'fixed-batch-critic-fit.json').read_text())['can_fit']==passed


def test_zero_restore_environment_guard_blocks_physics_step():
    env = object.__new__(train.CompletionPettingZooEnv)
    env._restore_only_guard = True
    env._restore_step_calls = 0
    with pytest.raises(RuntimeError, match="forbidden"):
        env.step({"sensor_0": 0})
    assert env._restore_step_calls == 1
