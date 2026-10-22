"""Bounded evidence-only monitor for the authorized matched 24-hour pair; never submits.

No Ray/Basilisk import or initialization. Run through Termius on Alpine.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

BASE = Path('/scratch/alpine/dahu1128/cooperative-v1-c18-pair-24h-20261022')
EVIDENCE = Path('/projects/dahu1128/cooperative-v1-c18-pair-24h-evidence-20261022')
JOBS = {'with_communication': '34768424', 'without_communication': '34768425'}
ROOT = JOB = ARM = None


def small_json(path):
    if not path.is_file():
        return None
    if path.stat().st_size > 4 * 1024**2:
        return dict(monitor_error='metadata_exceeds_4MiB', bytes=path.stat().st_size)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return dict(monitor_error='JSON_write_in_progress')


def tail(path, size):
    if not path.is_file():
        return b''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read(size)


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return dict(returncode=result.returncode, stdout=result.stdout[-16000:], stderr=result.stderr[-2000:])
    except subprocess.TimeoutExpired:
        return dict(monitor_error='scheduler_timeout')


def monitor(arm, with_wandb=False):
    global ROOT, JOB, ARM
    ARM, JOB = arm, JOBS[arm]
    ROOT = BASE / arm
    result = dict(job_id=JOB, arm=ARM, observed_utc=datetime.now(timezone.utc).isoformat(), evidence_only=True)
    result['squeue'] = command(['squeue', '-h', '-j', JOB, '-o', '%i %T %M %l %N %R'])
    result['sacct'] = command(['sacct', '-nP', '-j', JOB, '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,MaxRSS,NodeList'])
    summaries = {}
    for name in (
        'allocation-gate.json', 'preflight/source-gate.json', 'preflight/runtime.json',
        'preflight/effective-driver.json', 'preflight/preflight.json',
        'native-transport/acceptance.json', 'training/startup.json', 'training/wandb.json',
        'training/training_hyperparameters.json', 'training/training-failed.json',
        'training/initial-worker-modules.json', 'training/tranche-stop.json',
        'zero-update-restore/zero-update-resume.json', 'post-restore-memory.json',
        'acceptance.json', 'acceptance-failed.json', 'job-exit.json',
    ):
        data = small_json(ROOT / name)
        if data is None:
            continue
        summaries[name] = {k: data[k] for k in (
            'passed', 'errors', 'monitor_error', 'id', 'url', 'mode', 'wall_time_s',
            'training_iteration', 'error_type', 'error', 'exit_code', 'updates',
            'true_decisions', 'max_logit_error', 'new_ppo_updates', 'new_sampled_episodes',
            'reason', 'checkpoint', 'cgroup_peak_bytes', 'finite_gradients', 'lr',
            'entropy_coeff', 'grad_clip', 'num_sgd_iter', 'clip_param', 'gamma', 'lambda_',
        ) if k in data}
        workers = data.get('after', data).get('workers')
        if isinstance(workers, list):
            summaries[name]['worker_count'] = len(workers)
    result['gates'] = summaries
    memory = None
    for line in reversed(tail(ROOT / 'training/memory-profile.jsonl', 1024**2).splitlines()):
        try:
            memory = json.loads(line)
            break
        except json.JSONDecodeError:
            pass
    result['latest_memory_snapshot'] = memory
    if memory:
        result['memory_summary'] = dict(
            phase=memory.get('phase'),
            process_count=len(memory.get('processes', [])),
            process_RSS_sum_bytes=sum(p.get('rss_bytes') or 0 for p in memory.get('processes', [])),
            process_PSS_sum_bytes=sum(p.get('pss_bytes') or 0 for p in memory.get('processes', [])),
            process_USS_sum_bytes=sum(p.get('uss_bytes') or 0 for p in memory.get('processes', [])),
            job_cgroups=[g for g in memory.get('cgroups', []) if g['values'].get('memory.max', g['values'].get('memory.limit_in_bytes')) == '214748364800'],
        )
    topology_files = sorted((ROOT / 'training').glob('worker-topology-*.json'))
    if topology_files:
        topology = small_json(topology_files[-1])
        result['latest_topology'] = dict(file=topology_files[-1].name, **{k: topology[k] for k in ('passed', 'actor_restart_count', 'changed_process_indices', 'monitor_error') if k in topology})
    log = tail(ROOT.parent / (ARM + '.slurm-' + JOB + '.out'), 65536).decode(errors='replace')
    result['log_tail'] = log
    if with_wandb and (ROOT / 'training/wandb.json').is_file():
        import wandb
        identity = small_json(ROOT / 'training/wandb.json')
        key_path = Path('/projects/dahu1128/bsk_rl/examples/wandb_key.txt')
        wandb.login(key=key_path.read_text().strip(), relogin=False)
        run = wandb.Api(timeout=20).run('/'.join(identity[k] for k in ('entity', 'project', 'id')))
        summary = dict(run.summary)
        result['wandb'] = dict(id=run.id, url=run.url, state=run.state, summary={k: v for k, v in summary.items() if k.startswith(('training/', 'lifetime/', 'memory/', 'mission/', 'agent/', 'status/', 'losses/'))})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = EVIDENCE / ('monitor-' + ARM + '-' + stamp + '.json')
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    compact = {k: v for k, v in result.items() if k not in ('latest_memory_snapshot', 'log_tail')}
    if compact.get('memory_summary'):
        for group in compact['memory_summary']['job_cgroups']:
            group['values'].pop('memory.stat', None)
    if 'wandb' in compact:
        compact['wandb']['summary'] = {k: v for k, v in compact['wandb']['summary'].items() if k in ('training/iteration', 'lifetime/true_decisions', 'mission/requests/unshaped_weighted_science', 'memory/job_cgroup_peak_bytes')}
    compact.update(saved=str(destination), log_last_lines=log.splitlines()[-5:])
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--arm', choices=tuple(JOBS))
    args = parser.parse_args()
    for arm in ([args.arm] if args.arm else JOBS):
        monitor(arm, args.wandb)
