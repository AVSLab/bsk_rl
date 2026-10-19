#!/usr/bin/env python3
"""Historical campaign preparation; submission and execution are retired.

The referenced alpha=.1 pair failed (OOM and worker replacement). Rebuild a
campaign after memory, objective, and evaluation acceptance. Existing sealed
packages retain their historical bytes; this source cannot submit or run them.
plan remains read-only and never imports Ray/native modules.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent

def require_accepted_campaign():
    raise RuntimeError(
        'Retired alpha sweep: jobs 33993838/33993840 failed terminal acceptance. '
        'Rebuild and qualify a new campaign after memory and learning fixes; '
        'do not reuse these jobs as accepted alpha=.1 controls.'
    )

def read(name): return json.loads((ROOT / name).read_text())
def stamp(): return datetime.now(timezone.utc).isoformat()
def write(path, obj):
    with path.open('x') as f: json.dump(obj, f, indent=2); f.write('\n')

def verify():
    for name, meta in read('package-manifest.json')['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe package path')
        data = (ROOT / path).read_bytes()
        if len(data) != meta['bytes'] or hashlib.sha256(data).hexdigest() != meta['sha256']:
            raise ValueError('Package file changed: ' + name)
    plan = read('campaign-plan.json')
    assert len(plan['tasks']) == 16
    assert all(t['alpha'] != .1 for t in plan['tasks'])
    assert [t['task_id'] for t in plan['tasks']] == list(range(16))
    return plan

def plan_text(plan):
    return dict(alpha_values=plan['alpha_values'], new_allocations=plan['new_allocations'],
                existing_alpha_0p1_jobs=['33993838','33993840'],
                variants=['with_communication','without_communication'],
                seed=3105, wall_seconds_per_task=plan['wall_seconds_per_task'],
                cpus_per_task=56, memory_gib_per_task=200, concurrency=plan['concurrency'],
                maximum_new_cpu_hours=plan['maximum_new_allocated_cpu_hours'],
                submitted=False, final_acceptance_of_existing_pair_required=True,
                campaign_status='retired', submission_enabled=False,
                existing_pair_passed_terminal_acceptance=False)

def submit(plan):
    require_accepted_campaign()
    if not sys.platform.startswith('linux') or not Path('/projects/dahu1128').is_dir():
        raise RuntimeError('Submission is intended for the human to run on Alpine')
    output = Path(os.environ['BSK_SWEEP_OUTPUT_ROOT'])
    if not output.is_absolute() or not str(output).startswith('/scratch/alpine/dahu1128/'):
        raise ValueError('Choose a fresh absolute owner scratch path')
    key_path = Path(os.environ['BSK_RL_WANDB_KEY_PATH'])
    if not key_path.is_file(): raise ValueError('Existing private W&B key path missing')
    if output.exists(): raise ValueError('Output path already exists; no replay or automatic retry')
    # Atomic guard is intentionally retained on errors/uncertain responses.
    lock = ROOT / 'submission-lock'
    lock.mkdir()
    write(lock / 'intent.json', dict(utc=stamp(), output_root=str(output), plan=plan_text(plan)))
    output.mkdir(parents=True)
    logs = output / 'slurm'; logs.mkdir()
    write(output / 'manual-submit-intent.json', dict(utc=stamp(), campaign_root=str(ROOT), plan=plan_text(plan)))
    env = dict(os.environ, BSK_SWEEP_ROOT=str(ROOT), BSK_SWEEP_OUTPUT_ROOT=str(output))
    wall = plan['wall_seconds_per_task']; hh, rem = divmod(wall,3600); mm, ss = divmod(rem,60)
    cmd = ['sbatch','--parsable','--no-requeue','--export=ALL',
           '--array=0-15%' + str(plan['concurrency']), '--time=' + f'{hh:02d}:{mm:02d}:{ss:02d}',
           '--output=' + str(logs / '%A_%a.out'), '--error=' + str(logs / '%A_%a.err'),
           str(ROOT / 'alpha-sweep.slurm')]
    # Exactly one attempt. Do not retry after timeout or an ambiguous reply.
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=90)
    record = dict(utc=stamp(), command=cmd, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    write(lock / 'scheduler-response.json', record)
    if result.returncode or not re.fullmatch(r'\d+(;[A-Za-z0-9_.-]+)?\s*', result.stdout):
        raise RuntimeError('Inspect retained scheduler response before any manual recovery')
    job_id = result.stdout.strip().split(';')[0]
    write(output / 'submission.json', dict(job_array_id=job_id, utc=stamp(), campaign_root=str(ROOT), plan=plan_text(plan)))
    print(json.dumps(dict(job_array_id=job_id, output_root=str(output), no_retry=True),indent=2))

def run(plan, task_id):
    require_accepted_campaign()
    if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('Compute allocation required')
    if int(os.environ['SLURM_ARRAY_TASK_ID']) != task_id: raise ValueError('Task identity mismatch')
    task = plan['tasks'][task_id]
    source = ROOT / task['source_root']
    manifest = json.loads((source / 'completion-release.json').read_text())
    assert manifest['alpha'] == task['alpha'] and manifest['release_id'] == task['release_id']
    outroot = Path(os.environ['BSK_SWEEP_OUTPUT_ROOT'])
    # A task can start before sbatch returns its job ID to the submitter.
    if not (outroot / 'manual-submit-intent.json').is_file(): raise RuntimeError('Missing manual submission intent')
    output = outroot / task['output_suffix'] / 'diagnostic'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise RuntimeError('Existing task output; no overwrite or automatic restart')
    env = dict(os.environ, BSK_PROJECT_ROOT=str(source), BSK_OUTPUT=str(output), BSK_ARM=task['arm'],
               BSK_ALPHA_SWEEP_RUN_APPROVED='1', BSK_APPROVED_WALL_S=str(plan['wall_seconds_per_task']))
    write(output.parent / 'task-intent.json', dict(utc=stamp(), job_id=os.environ['SLURM_JOB_ID'], task=task))
    # Bash exec ensures Slurm warning signals reach the same PID and its driver.
    os.execvpe('bash', ['bash', str(source / 'examples/multiagent_imaging/cluster/approved_24h.slurm')], env)

def main():
    plan = verify()
    mode = sys.argv[1] if len(sys.argv)>1 else 'plan'
    if mode == 'plan': print(json.dumps(plan_text(plan),indent=2))
    elif mode == 'submit': submit(plan)
    elif mode == 'run' and len(sys.argv)==3: run(plan,int(sys.argv[2]))
    else: raise SystemExit('Usage: campaign.py plan | submit | run TASK_ID')

if __name__ == '__main__': main()
