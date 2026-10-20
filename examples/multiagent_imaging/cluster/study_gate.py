"""Compute-only readiness gates. Merely importing this module never samples."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_gate(root=ROOT):
    manifest=json.loads((root/'completion-release.json').read_text())
    from examples.multiagent_imaging.cluster.release_manifest import validate_study_manifest
    validate_study_manifest(root, manifest)
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.readiness import schema
    configs={}
    for arm in ('catalog_tx','geometry_only'):
        name=f'examples/multiagent_imaging/configs/walker4_t200_cd1_{arm}.json'
        cfg=MultiAgentImagingConfig.from_json(root/name)
        assert cfg.n_sensors==4 and cfg.n_targets==200 and cfg.n_candidates==10
        assert cfg.episode_duration_s==45000 and cfg.reimage_cooldown_orbits==1
        assert cfg.discount_per_s==.9915045428565076 and cfg.effective_gae_lambda==.8713548569911232
        assert cfg.alpha==.1 and cfg.information_case==arm and cfg.policy_model_profile=='amos_attention'
        assert cfg.walker_altitude_km==700 and cfg.walker_inclination_deg==97 and cfg.walker_planes==2 and cfg.walker_phasing==1
        assert cfg.communication_mode=='directed' and cfg.link_mode=='los' and cfg.metadata_bitrate_bps==64000
        assert digest(root/name)==manifest['config_sha256'][arm]
        configs[arm]=dict(config_sha256=digest(root/name),schema=schema(cfg))
    return dict(passed=True,source_hashes=len(manifest['source_sha256']),configs=configs,
        release=manifest['release_id'])

def preflight(output):
    from examples.multiagent_imaging.cluster.audit_runtime import audit
    from examples.multiagent_imaging.cluster.effective_imports import require_effective_imports
    from examples.multiagent_imaging.cluster.prepare_support_data import prepare
    from examples.multiagent_imaging.readiness import write_json
    source=source_gate();write_json(output/'source-gate.json',source)
    runtime=audit(allocation=True);write_json(output/'runtime.json',runtime)
    assert runtime['passed'],runtime['errors']
    assert sys.version_info[:3]==(3,11,13)
    effective=require_effective_imports();write_json(output/'effective-driver.json',effective)
    assert effective['modules']['torch']['version']=='2.4.1+cpu'
    assert effective['basilisk']['version']=='2.12.0b0'
    native=json.loads((Path(__file__).with_name('gate-parent-runtime.manifest')).read_text())['loaded']['native_sha256']
    assert runtime['loaded']['native_sha256']==native, 'Native artifact differs from passed gate'
    support=prepare(output/'support-data.json')
    expected=json.loads(Path(__file__).with_name('gate-parent-support.manifest').read_text())
    assert support==expected, 'Support cache/source hashes differ'
    import wandb
    wandb.login(key=Path(os.environ['BSK_RL_WANDB_KEY_PATH']).read_text().strip(),relogin=False)
    viewer=wandb.Api(timeout=60).viewer
    assert viewer is not None, 'Online W&B authentication/readback unavailable'
    write_json(output/'preflight.json',dict(passed=True,online_wandb=True,source=source))

def verify(output, parallel=False):
    import torch
    from examples.multiagent_imaging.study_metrics import mission_metrics
    from examples.multiagent_imaging.checkpoints import _state_digest
    from examples.multiagent_imaging.readiness import write_json
    training,heldout,weights,rows=[],{}, {}, []
    expected_hashes={};throughput={};ids={};true_total=0;training_keys={};launcher=None;batch_checks={}
    expected_runners=28 if parallel else 1
    if parallel:
        canary=json.loads((output/'communication-canary/acceptance.json').read_text())
        assert canary['passed'] and canary['new_ppo_updates']==0 and canary['native_fixture_instances']==7
    episode_cap=32 if parallel else 6
    decision_cap=60000 if parallel else 30000
    for arm in ('catalog_tx','geometry_only'):
        folder=output/arm
        resources=json.loads((folder/'launcher.json').read_text())
        assert resources['num_env_runners']==expected_runners and resources['num_learners']==0 and resources['torch_threads']==1
        if parallel: validate_parallel_cpu_allocation(resources, output)
        if launcher is not None: assert resources==launcher, 'Arms use different sampler topology'
        launcher=resources
        driver=json.loads((output/'preflight/effective-driver.json').read_text())
        initial_modules=json.loads((folder/'initial-worker-modules.json').read_text())
        assert initial_modules['passed'] and initial_modules['driver_rng_unchanged']
        assert initial_modules['topology']['passed']
        assert {r['worker_index'] for r in initial_modules['workers']}==set(range(expected_runners+1))
        assert len(initial_modules['workers'])==expected_runners+1
        assert all(r['passed'] and r['episode_state_unchanged'] and
            r['rng_before_sha256']==r['rng_after_sha256'] and
            r['actor_sha256']==initial_modules['driver_actor_sha256'] and
            r['module_modes'] and r['dropout'] and all(p==.1 for p in r['dropout'].values())
            for r in initial_modules['workers'])
        worker_files=list((folder/'effective-worker-runtime').glob('*.json'))
        assert len(worker_files)>=2, 'Local and remote effective-import evidence missing'
        assert {int(path.name.split('-')[1]) for path in worker_files}==set(range(expected_runners+1)), 'Effective evidence missing for an actual worker'
        for worker_path in worker_files:
            worker=json.loads(worker_path.read_text());assert worker['passed']
            assert worker['basilisk']==driver['basilisk'] and worker['bsk_rl']==driver['bsk_rl']
            for name,module in driver['modules'].items():
                for field in ('version','path','file_sha256'):assert worker['modules'][name][field]==module[field]
        from examples.multiagent_imaging.update_evidence import UpdateRecords
        updates=UpdateRecords(folder);assert len(updates)==1
        assert all(math.isfinite(v) for v in updates[0]['losses'].values())
        assert math.isfinite(updates[0]['gradient_l2']) and updates[0]['finite_gradients']==1
        assert all(updates[0]['parameter_change'][key]>0 for key in ('actor_l2','critic_l2','l2'))
        episodes=updates[0]['episodes'];assert 1<=len(episodes)<=episode_cap
        true=sum(sum(e['policy_decisions'].values()) for e in episodes);true_total+=true;assert true<=decision_cap
        from examples.multiagent_imaging.episode_accounting import reconcile_episode_audits, verify_retained_learner_batch
        from examples.multiagent_imaging.config import MultiAgentImagingConfig
        complete, accounting=reconcile_episode_audits(episodes,folder/'episode-spool')
        assert len(complete)==len(episodes) and accounting['new_sampled_true_decisions']==true, 'Updates omit completed spool missions'
        assert updates[0]['actual_policy_decisions']==true
        batch_checks[arm]=verify_retained_learner_batch(folder/'learner_batch',
            MultiAgentImagingConfig.from_json(ROOT/f'examples/multiagent_imaging/configs/walker4_t200_cd1_{arm}.json'),
            episodes,updates[0]['condensed_learner_rows'])
        write_json(folder/'learner-batch-verification.json',batch_checks[arm])
        training_keys[arm]=set()
        for e in episodes:
            assert e['simulated_seconds']==45000
            assert mission_metrics(e,45000)==e['mission_metrics']
            key=(e['worker_index'],e['episode_seed'])
            training_keys[arm].add(key)
            if key in expected_hashes:assert expected_hashes[key]==e['initial_conditions_sha256']
            expected_hashes[key]=e['initial_conditions_sha256']
        training+=episodes
        payload=torch.load(folder/'checkpoint_0000/policy.pt',map_location='cpu',weights_only=True)
        assert _state_digest({k:v for k,v in payload['weights'].items() if not k.startswith('vf.')})==initial_modules['driver_actor_sha256']
        weights[arm]=_state_digest(payload['weights'])
        resume=json.loads((output/(arm+'-resume')/'zero-update-resume.json').read_text())
        assert resume['passed'] and resume['new_sampled_episodes']==resume['new_ppo_updates']==0
        assert resume['training_iteration']==1 and resume['policy_decisions_lifetime']==true
        assert resume['lifetime_metrics_matches_checkpoint'] and resume['training_accounting_matches_checkpoint']
        final_manifest=json.loads((folder/'checkpoint_0001/manifest.json').read_text())
        assert final_manifest['lifetime_metrics']['true_decisions']==true
        before=json.loads((folder/'checkpoint-before.json').read_text())
        after=json.loads((folder/'checkpoint-after.json').read_text())
        assert after['passed'] and before['checkpoint_sha256']==after['checkpoint_sha256']
        assert all(x['matches_checkpoint'] for x in resume['workers'])
        assert len(resume['workers'])==expected_runners+1
        assert {x['worker_index'] for x in resume['workers']}==set(range(expected_runners+1))
        assert resume['worker_topology']['passed']
        assert len(final_manifest['worker_seed_states'])==expected_runners+1
        assert {x['worker_index'] for x in final_manifest['worker_seed_states']}==set(range(expected_runners+1))
        identity=json.loads((folder/'checkpoint_0001/wandb.json').read_text());ids[arm]=identity['id']
        import wandb
        live=wandb.Api(timeout=60).run(identity['entity']+'/'+identity['project']+'/'+identity['id'])
        assert live.id==identity['id']
        assert live.summary.get('training/iteration')==1
        assert live.summary.get('lifetime/true_decisions')==true
        for key in ['losses/policy_loss','losses/vf_loss','mission/constellation_reward','mission/useful_delivered_visits']+[f'agent/sensor_{i}/reward' for i in range(4)]+[f'agent/sensor_{i}/storage_fraction' for i in range(4)]:
            assert live.summary.get(key) is not None, 'Missing live W&B metric: '+key
        allowed=('training/','lifetime/','mission/','agent/','actions/','losses/','learner_metrics/','update/','status/')
        write_json(folder/'live-wandb-readback.json',dict(passed=True,id=live.id,url=live.url,state=live.state,
            same_id_resume=resume['wandb']['requested_resume']=='must',
            summary={k:v for k,v in dict(live.summary).items() if k.startswith(allowed)}))
        summary=json.loads((folder/'heldout/summary.json').read_text())
        assert summary['complete'] and summary['completed_episodes']==8
        for row in summary['rows']:
            path=folder/'heldout'/Path(row['episode']).name
            assert digest(path)==row['episode_sha256']
            raw=json.loads(path.read_text());assert raw['sim_time_s']==45000
            assert mission_metrics(raw['raw_metric_evidence'],45000)==raw['mission_metrics']
            key=(row['seed'],row['mode'])
            if key in heldout:assert heldout[key]==row['initial_conditions_sha256']
            heldout[key]=row['initial_conditions_sha256']
            rows.append(dict(arm=arm,**row))
        throughput[arm]=dict(training_episodes=len(episodes),true_decisions=true,
            max_heldout_wall_s=max(row['rollout_wall_s'] for row in summary['rows']),update_wall_s=updates[0]['measurement']['wall_time_s'],startup_wall_s=json.loads((folder/'startup.json').read_text())['wall_time_s'])
    assert weights['catalog_tx']==weights['geometry_only'], 'Fresh initial weights differ'
    common_training_seeds=training_keys['catalog_tx'] & training_keys['geometry_only']
    assert common_training_seeds, 'No matched training initial states observed'
    if parallel: assert training_keys['catalog_tx']==training_keys['geometry_only'], 'Parallel arms did not sample the same seed set'
    assert ids['catalog_tx']!=ids['geometry_only'], 'Arms share a W&B ID'
    assert len(training)<=2*episode_cap and len(rows)==16 and true_total<=2*decision_cap
    result=dict(source_gate=source_gate(),passed=True,ppo_updates=2,new_training_episodes=len(training),heldout_episodes=16,
        new_true_decisions=true_total,initial_weight_digest=weights,wandb_run_ids=ids,
        measured_throughput=throughput,individual_heldout=rows,
        retained_learner_batch_verification=batch_checks,
        launcher_resources=launcher,parallel_readiness=parallel,
        matched_training_seed_keys=sorted(common_training_seeds),
        training_seed_counts={arm:len(keys) for arm,keys in training_keys.items()},
        scope='Operational readiness only; no convergence or cooperation conclusion',
        long_launch_requires_separate_approval=True)
    write_json(output/'readiness-acceptance.json',result)
    return result

def validate_parallel_cpu_allocation(resources, output):
    """Tie verifier resources to the reviewed allocation, without adding samplers."""
    expected=int(os.environ.get('BSK_EXPECTED_CPUS','32'))
    actual=json.loads((Path(output)/'allocation-gate.json').read_text())['allocated_cpus']
    assert resources['ray_cpus']==actual==expected, 'Parallel CPU allocation differs'
    assert resources['num_env_runners']==28 and resources['num_learners']==0
    assert resources['torch_threads']==1
    return actual

def allocation(cpus, wall_s, cap):
    if cpus*wall_s/3600>cap:raise ValueError('Effective allocation reservation exceeds cap')
    return dict(allocated_cpus=cpus,time_limit_s=wall_s,reserved_cpu_h=cpus*wall_s/3600,cap_cpu_h=cap)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','verify'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--parallel',action='store_true')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Compute allocation required; no login-node Ray or mission')
    a.output.mkdir(parents=True,exist_ok=True)
    result=dict(passed=False,stage=a.stage)
    try:
        if a.stage=='preflight':preflight(a.output)
        else:result.update(verify(a.output,a.parallel))
        result['passed']=True
    except BaseException as exc:
        result.update(error=str(exc),exception_type=type(exc).__name__);raise
    finally:
        (a.output/(a.stage+'-result.json')).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
