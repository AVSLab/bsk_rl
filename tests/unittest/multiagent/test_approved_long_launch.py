"""Acceptance failures and resource/science invariants, entirely offline."""
import copy
import json
import os
import subprocess
from pathlib import Path
import pytest
from examples.multiagent_imaging.cluster.approved_long_evidence import validate_sync, validate_update, validate_heldout


def sync():
    rows=[dict(worker_index=i,passed=True,parameters_finite=True,actor_sha256='shared',
               rng_before_sha256='rng',rng_after_sha256='rng',dropout={'actor':.1}) for i in range(29)]
    return dict(passed=True,learner_unchanged=True,driver_rng_unchanged=True,
                authoritative_actor_sha256='shared',after=dict(passed=True,topology={'passed':True},workers=rows),
                worker_state_comparisons=[dict(state_unchanged=True) for _ in rows])


def test_29_workers_accept_then_reject_missing_divergent_rng_and_dropout():
    validate_sync(sync())
    for field,value in [('actor_sha256','other'),('rng_after_sha256','changed'),('parameters_finite',False),('dropout',{'actor':0.})]:
        record=sync();record['after']['workers'][17][field]=value
        with pytest.raises(AssertionError):validate_sync(record)
    record=sync();record['after']['workers'].pop()
    with pytest.raises(AssertionError):validate_sync(record)


def update():
    return dict(episodes=[dict(simulated_seconds=45000,policy_decisions={f'sensor_{i}':i+1 for i in range(4)})],
        losses={'policy_loss':.1,'vf_loss':.2},gradient_l2=.3,finite_gradients=1,
        parameter_change=dict(actor_l2=.1,critic_l2=.2,l2=.3),actual_policy_decisions=10,condensed_learner_rows=10)


def test_truthful_horizon_real_rows_and_learning_finiteness_required():
    assert validate_update(update())==10
    bad=[]
    r=update();r['episodes'][0]['simulated_seconds']=44999;bad.append(r)
    r=update();r['losses']['vf_loss']=float('nan');bad.append(r)
    r=update();r['parameter_change']['actor_l2']=0;bad.append(r)
    r=update();r['condensed_learner_rows']=300;bad.append(r)
    r=update();r['episodes']=[];bad.append(r)
    for r in bad:
        with pytest.raises(AssertionError):validate_update(r)


def test_direct_authorization_resources_and_inline_resume_precede_continuation():
    root=Path(__file__).resolve().parents[3]
    text=(root/'examples/multiagent_imaging/cluster/approved_24h.slurm').read_text()
    assert '#SBATCH --cpus-per-task=56' in text and '#SBATCH --mem=200G' in text
    assert '#SBATCH --time=24:00:00' in text and '#SBATCH --no-requeue' in text
    assert 'BSK_DIRECT_24H_APPROVED' in text and 'BSK_READINESS_ACCEPTANCE' not in text
    assert 'BSK_RESERVATION_CAP=1344' in text
    assert text.index('bootstrap-checkpoint-after.json') < text.index('--output "$BSK_OUTPUT/training"')
    assert 'approved_long_evidence --root "$BSK_OUTPUT" --phase bootstrap' in text
    assert '--ppo-profile amos_best' in text and '--torch-threads 1' in text
    assert '--num-env-runners 28 --num-learners 0' in text
    assert '--seeds 401 402' in text and '--workers 8' in text
    assert 'sbatch ' not in text


def test_all_eight_frozen_cases_required_and_initial_conditions_must_match():
    summary=dict(complete=True,completed_episodes=8,rows=[dict(policy=c,seed=s,mode=m,
        initial_conditions_sha256=str(s)) for c in ('initial','trained') for s in (401,402)
        for m in ('deterministic','stochastic')])
    validate_heldout(summary)
    r=copy.deepcopy(summary);r['rows'][5]['initial_conditions_sha256']='different'
    with pytest.raises(AssertionError):validate_heldout(r)
    r=copy.deepcopy(summary);r['complete']=False
    with pytest.raises(AssertionError):validate_heldout(r)
    r=copy.deepcopy(summary);r['rows'].pop()
    with pytest.raises(AssertionError):validate_heldout(r)


def submit_fixture(tmp_path):
    source=tmp_path/'source'; cluster=source/'examples/multiagent_imaging/cluster'
    cluster.mkdir(parents=True)
    (source/'completion-release.json').write_text('{}')
    (cluster/'approved_24h.slurm').write_text('# mocked sealed launcher')
    bindir=tmp_path/'bin';bindir.mkdir()
    mock=bindir/'sbatch'
    mock.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
        'open(os.environ["CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n'
        'print("99999")\n')
    mock.chmod(0o755)
    capture=tmp_path/'args.json'
    env={**os.environ,'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
         'CAPTURE':str(capture),'BSK_PROJECT_ROOT':str(source),
         'BSK_OUTPUT':str(tmp_path/'arm/diagnostic'),
         'BSK_RL_WANDB_KEY_PATH':'/private/key-path-only',
         'BSK_DIRECT_24H_APPROVED':'1','BSK_APPROVED_WALL_S':'86340','BSK_ARM':'catalog_tx'}
    script=Path(__file__).resolve().parents[3]/'examples/multiagent_imaging/cluster/submit_approved_24h.sh'
    return script,env,capture


@pytest.mark.parametrize('arm',['catalog_tx','geometry_only'])
def test_submission_passes_explicit_settings_and_retry_wall_to_sbatch(tmp_path,arm):
    script,env,capture=submit_fixture(tmp_path);env['BSK_ARM']=arm
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    args=json.loads(capture.read_text())
    exported=next(a for a in args if a.startswith('--export=')).split(',')[1:]
    settings=dict(a.split('=',1) for a in exported)
    assert settings=={k:env[k] for k in ('BSK_DIRECT_24H_APPROVED','BSK_ARM',
        'BSK_PROJECT_ROOT','BSK_OUTPUT','BSK_RL_WANDB_KEY_PATH','BSK_APPROVED_WALL_S')}
    assert '--time=23:59:00' in args
    # Replaying the command is rejected before a second scheduler call.
    capture.unlink()
    assert subprocess.run(['bash',str(script)],env=env,capture_output=True).returncode!=0
    assert not capture.exists()


@pytest.mark.parametrize('key,value',[('BSK_DIRECT_24H_APPROVED',None),
    ('BSK_DIRECT_24H_APPROVED','0'),('BSK_APPROVED_WALL_S','86460'),
    ('BSK_APPROVED_WALL_S','86341'),('BSK_APPROVED_WALL_S','-1'),
    ('BSK_ARM','unknown')])
def test_invalid_submission_never_calls_scheduler(tmp_path,key,value):
    script,env,capture=submit_fixture(tmp_path)
    if value is None:env.pop(key)
    else:env[key]=value
    assert subprocess.run(['bash',str(script)],env=env,capture_output=True).returncode!=0
    assert not capture.exists()


@pytest.mark.parametrize('wall,expected',[('86340',0),('86400',0),('86460',1),('-1',1)])
def test_real_launcher_prefix_accepts_only_bounded_wall_without_runtime(tmp_path,wall,expected):
    root=Path(__file__).resolve().parents[3]
    text=(root/'examples/multiagent_imaging/cluster/approved_24h.slurm').read_text()
    prefix=text.split('source "${BSK_PROJECT_ROOT:?}')[0]
    result=subprocess.run(['bash','-c',prefix+'\nprintf "%s" "$BSK_EXPECTED_WALL_S"'],
        env={**os.environ,'BSK_DIRECT_24H_APPROVED':'1','BSK_ARM':'catalog_tx',
             'BSK_APPROVED_WALL_S':wall},capture_output=True,text=True)
    assert (result.returncode!=0)==bool(expected)
    if not expected:assert result.stdout==wall
    assert '#SBATCH --export=ALL' in text
    assert '+int(os.environ["BSK_APPROVED_WALL_S"])-7200' in text
