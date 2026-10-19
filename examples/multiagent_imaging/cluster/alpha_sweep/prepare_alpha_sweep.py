#!/usr/bin/env python3
"""Reproduce historical alpha preparation. Never contacts Alpine.

The alpha=.1 reference pair subsequently failed terminal acceptance. Generated
controllers refuse submission and execution. This is historical preparation,
not an approved or qualified next training campaign.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

PIN = '5eb4de0875897c4b48f0c20478b7de61ea5b4231'
ALPHAS = [0., .1, .2, .3, .4, .5, .7, .9, 1.]
ARMS = {'catalog_tx': 'with_communication', 'geometry_only': 'without_communication'}
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MANIFEST = REPO / 'results/multiagent_imaging/approved-24h-training-20261011/retry-export-20261012-v14/release-v14/release.json'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')

def replace_once(data, old, new):
    text = data.decode()
    if text.count(old) != 1:
        raise ValueError('Pinned source does not contain exactly one expected replacement: ' + old)
    return text.replace(old, new).encode()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, default=86340)
    parser.add_argument('--concurrency', type=int, default=2)
    args = parser.parse_args()
    if not 10800 <= args.wall_seconds <= 86400 or args.wall_seconds % 60:
        parser.error('Wall must be whole minutes between 3 and 24 hours, including evaluation')
    if not 1 <= args.concurrency <= 16:
        parser.error('Concurrency must be 1..16')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing prepared campaign')
    parent = json.loads(MANIFEST.read_text())
    source = {}
    for name, expected in parent['source_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe manifest path')
        blob = subprocess.check_output(['git', '-C', str(REPO), 'show', PIN + ':' + name])
        if sha(blob) != expected:
            raise ValueError('Pinned Git source differs from passed release: ' + name)
        source[name] = blob
    args.output.mkdir(parents=True)
    write_json(args.output / 'parent-source-manifest.json', parent['source_sha256'])
    jobs, cells = [], []
    for alpha in ALPHAS:
        tag = 'alpha_' + f'{alpha:.1f}'.replace('.', 'p')
        folder = args.output / 'sources' / tag
        blobs = dict(source)
        config_paths = {arm: f'examples/multiagent_imaging/configs/walker4_t200_cd1_{arm}.json' for arm in ARMS}
        for arm, name in config_paths.items():
            cfg = json.loads(blobs[name]); cfg['alpha'] = alpha
            blobs[name] = (json.dumps(cfg, indent=2) + '\n').encode()
        gate = 'examples/multiagent_imaging/cluster/study_gate.py'
        blobs[gate] = replace_once(blobs[gate], 'cfg.alpha==.1', 'cfg.alpha==' + repr(alpha))
        launcher = 'examples/multiagent_imaging/cluster/approved_24h.slurm'
        blobs[launcher] = replace_once(blobs[launcher], 'BSK_DIRECT_24H_APPROVED', 'BSK_ALPHA_SWEEP_RUN_APPROVED')
        blobs[launcher] = replace_once(blobs[launcher], 'Explicit direct long-launch approval required', 'User-run alpha campaign required')
        blobs[launcher] = replace_once(blobs[launcher], '${BSK_LABEL}_s3105_24h_${SLURM_JOB_ID}', '${BSK_LABEL}_' + tag + '_s3105_${SLURM_JOB_ID}')
        changes = [n for n in sorted(blobs) if blobs[n] != source[n]]
        allowed = {gate, launcher, *config_paths.values()}
        if set(changes) - allowed:
            raise ValueError('Unexpected source change')
        for name, data in blobs.items():
            target = folder / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        hashes = {name: sha(data) for name, data in blobs.items()}
        fingerprint = sha(json.dumps(hashes, sort_keys=True).encode())
        manifest = dict(release_id='multiagent-alpha-sweep-' + tag + '-' + fingerprint[:12],
            source_sha256=hashes, config_sha256={a: hashes[p] for a,p in config_paths.items()},
            parent_git_commit=PIN, parent_source_files=len(source), alpha=alpha,
            changed_files=changes, scientific_change='Reward alpha only; no other mission/model/PPO changes',
            administrative_changes=['Alpha-specific source assertion', 'New manual-campaign guard and W&B label'],
            cluster_validated=False, deployed=False, submitted=False)
        write_json(folder / 'completion-release.json', manifest)
        for arm, label in ARMS.items():
            cell = dict(alpha=alpha, arm=arm, label=label, learning_seed=3105,
                        source_root='sources/' + tag, release_id=manifest['release_id'])
            if alpha == .1:
                cell.update(action='reuse_existing_pair_after_final_acceptance',
                            job_id={'catalog_tx': '33993838', 'geometry_only': '33993840'}[arm],
                            wandb_run_id={'catalog_tx': '0fc31890', 'geometry_only': '238d5947'}[arm])
            else:
                cell.update(action='new_fresh_training', task_id=len(jobs), output_suffix=tag + '/' + label)
                jobs.append(cell)
            cells.append(cell)
    for name in ('campaign.py', 'alpha-sweep.slurm', 'submit-alpha-sweep.sh'):
        target = args.output / name; target.write_bytes((HERE / name).read_bytes())
        target.chmod(0o755)
    plan = dict(version=1, alpha_values=ALPHAS, cells=cells, tasks=jobs, source_commit=PIN,
                wall_seconds_per_task=args.wall_seconds, cpus_per_task=56, memory_gib=200,
                concurrency=args.concurrency, training_seed=3105, heldout_seeds=[401,402],
                evaluation_modes=['deterministic','stochastic'],
                evaluation_checkpoints=['initial','trained'],
                new_allocations=len(jobs), referenced_existing_allocations=2,
                maximum_new_allocated_cpu_hours=len(jobs)*56*args.wall_seconds/3600,
                submission_authorized=False, cluster_validated=False,
                campaign_status='retired', submission_enabled=False,
                existing_pair_passed_terminal_acceptance=False,
                active_jobs_modified=False, fresh_checkpoint_each_new_cell=True,
                no_automatic_retry_or_successor=True,
                gates=['source/native/runtime/support/online-W&B', '56 CPU/200 GiB/no requeue',
                       'one-update full-mission retained-row and target reconstruction',
                       'all 29 worker identities/RNG/seed/mode/accounting readbacks',
                       'full-topology zero-update restore and unchanged checkpoint bytes',
                       'finite learning and topology for every update', 'initial/final held-out argmax and stochastic'],
                limitations=['No new native simulation or cluster execution during preparation',
                             'Alpha=.1 existing cells require final terminal acceptance before use in sweep results',
                             'Two-hour evaluation reserve has not been measured for every alpha',
                             'Final training verifier reads full update evidence inside the 200 GiB compute allocation; never invoke on a login node',
                             'Single training seed supports a descriptive sweep, not uncertainty across training seeds'])
    write_json(args.output / 'campaign-plan.json', plan)
    members = {}
    for path in sorted(args.output.rglob('*')):
        if path.is_file(): members[str(path.relative_to(args.output))] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    write_json(args.output / 'package-manifest.json', dict(files=members))
    archive = args.output.with_suffix('.tar.gz')
    if archive.exists(): raise ValueError('Archive already exists; choose a fresh output path')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(args.output, arcname=args.output.name)
    write_json(args.output.parent / (args.output.name + '-archive-verification.json'),
               dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive.read_bytes()),
                    files=len(members), submitted=False, deployed=False))
    print(json.dumps(dict(package=str(args.output), archive=str(archive), new_tasks=len(jobs),
                          maximum_new_cpu_hours=plan['maximum_new_allocated_cpu_hours']), indent=2))

if __name__ == '__main__': main()
