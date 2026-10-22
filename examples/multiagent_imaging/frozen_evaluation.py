"""Prepare or execute the explicit A/B/C/D frozen-policy experiment.

Preparation does not import a simulator, load a policy, or submit a job. Execution
requires a separately collected terminal qualification and a compute allocation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics

SOURCE = 'b205fc1603b5ca887aaadcb3c34de2700d563c0c'
JOBS = {'with_communication': '34768424', 'without_communication': '34768425'}
SEEDS = [9000101, 9000102, 9000103, 9000104]
MODES = ['argmax', 'stochastic']
CONFIG_HASHES = {'with_communication': '72584bb7de636580e2a3f96e60caf09de6bb4bcb0b69b90e314f74102552be91',
                 'without_communication': 'abe250934e6c4e8a226e5948bbfb92485d4a02bd8c4499c71cfe3285fbfdf30c'}
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'examples/multiagent_imaging/configs'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')


def config_path(condition):
    path = Path(condition['config'])
    return path if path.is_absolute() else ROOT / path


def prepare():
    conditions = []
    for label, arm, selector, intervention in (
        ('A', 'with_communication', 'final_checkpoint', 'normal'),
        ('B', 'with_communication', 'final_checkpoint', 'suppress_receiver_catalog_v1'),
        ('C', 'without_communication', 'final_checkpoint', 'normal'),
        ('D', 'with_communication', 'initial_checkpoint', 'normal'),
    ):
        name = 'catalog_tx' if arm == 'with_communication' else 'geometry_only'
        config = CONFIGS / f'cooperative_v1_shared_{name}.json'
        conditions.append(dict(label=label, arm=arm, checkpoint_selector=selector,
                               config=str(config.relative_to(ROOT)), config_sha256=sha256(config),
                               metadata_intervention=intervention))
    result = dict(version='cooperative-frozen-abcd-v1', source_commit=SOURCE,
                  jobs=dict(JOBS), seeds=list(SEEDS), modes=list(MODES), conditions=conditions,
                  expected_missions=32, horizon_s=45000, new_PPO_updates=0,
                  maximum_workers=8, torch_threads=1,
                  proposed_allocation=dict(account='ucb550_asc3', partition='acpu',
                      qos='cpu-normal', cpus=16, memory_gib=64, wall_seconds=7200,
                      allocated_cpu_hours=32, no_requeue=True),
                  submitted=False, terminal_qualification_pending=True,
                  interpretation={
                      'A_minus_C': 'trained policies; report differing actual training experience',
                      'A_minus_B': 'receiver information intervention; same frozen weights and physical TX mechanism',
                      'A_minus_D': 'communication-policy training progress on unseen scenarios',
                      'B': 'Successful transport and ideal ACK remain; only receiver catalog updates are suppressed. Not packet loss.',
                      'scope': 'Four scenario seeds, one training seed per arm. Exploratory, not training-seed generalization.'})
    validate_plan(result)
    return result


def validate_plan(plan):
    if (plan['version'] != 'cooperative-frozen-abcd-v1' or plan['source_commit'] != SOURCE
        or plan['jobs'] != JOBS or plan['seeds'] != SEEDS or plan['modes'] != MODES
        or plan['expected_missions'] != 32 or plan['new_PPO_updates'] != 0
        or plan['horizon_s'] != 45000 or plan['maximum_workers'] != 8
        or plan['torch_threads'] != 1 or plan['submitted'] is not False):
        raise ValueError('Frozen experiment contract changed')
    conditions = plan['conditions']
    if [c['label'] for c in conditions] != list('ABCD'):
        raise ValueError('Exactly four labelled conditions required')
    a, b, c, d = conditions
    for item in conditions:
        if (sha256(config_path(item)) != item['config_sha256']
            or item['config_sha256'] != CONFIG_HASHES.get(item['arm'])):
            raise ValueError('Configuration bytes changed')
    left = json.loads(config_path(a).read_text())
    right = json.loads(config_path(c).read_text())
    if left.pop('information_case') != 'catalog_tx' or right.pop('information_case') != 'geometry_only' or left != right:
        raise ValueError('Conditions must retain the original matched mission/reward definitions')
    if (a['arm'] != b['arm'] or a['arm'] != d['arm'] or a['arm'] != 'with_communication'
        or c['arm'] != 'without_communication' or len({x['config_sha256'] for x in (a,b,d)}) != 1
        or [x['checkpoint_selector'] for x in conditions] != ['final_checkpoint'] * 3 + ['initial_checkpoint']
        or [x['metadata_intervention'] for x in conditions] != ['normal', 'suppress_receiver_catalog_v1', 'normal', 'normal']):
        raise ValueError('Treatment/configuration/checkpoint mapping changed')


def qualify(plan, qualification, artifact_root=None):
    """Verify local terminal-audit receipts before binding final checkpoints."""
    validate_plan(plan)
    if qualification.get('version') != 'cooperative-pair-terminal-qualification-v1' or not qualification.get('passed'):
        raise ValueError('Independent terminal pair qualification is required')
    observed = set(qualification['training_episode_seeds'])
    if not observed or len(qualification['reserved_worker_seed_intervals']) != 29 or any(
        count < 5600 for _, count in qualification['reserved_worker_seed_intervals']):
        raise ValueError('Complete observed and reserved training seed audit required')
    for seed in plan['seeds']:
        if seed in observed:
            raise ValueError('Held-out seed was used for training')
        for start, count in qualification['reserved_worker_seed_intervals']:
            if (seed - start) % 2**32 < count:
                raise ValueError('Held-out seed overlaps a reserved training stream')
    artifact_root = Path(artifact_root or '.')
    arms = {arm:dict(receipt) for arm,receipt in qualification['arms'].items()}
    for arm, job in JOBS.items():
        receipt = arms[arm]
        if (receipt['job_id'] != job or receipt['terminal_state'] != 'COMPLETED'
            or receipt['exit_code'] != '0:0' or receipt['source_commit'] != SOURCE):
            raise ValueError('Successful terminal job identity required; failures remain failures')
        acceptance = Path(receipt['acceptance_path'])
        if not acceptance.is_absolute(): acceptance = artifact_root / acceptance
        if sha256(acceptance) != receipt['acceptance_sha256'] or not json.loads(acceptance.read_text()).get('passed'):
            raise ValueError('Acceptance receipt mismatch')
        for role in ('initial_checkpoint', 'final_checkpoint'):
            checkpoint = Path(receipt[role])
            if not checkpoint.is_absolute(): checkpoint = artifact_root / checkpoint
            arms[arm][role] = str(checkpoint.resolve())
            metadata = json.loads((checkpoint / 'manifest.json').read_text())
            from examples.multiagent_imaging.config import MultiAgentImagingConfig
            from examples.multiagent_imaging.readiness import validate_schema
            condition = next(c for c in plan['conditions'] if c['arm'] == arm)
            validate_schema(metadata['schema'], MultiAgentImagingConfig.from_json(config_path(condition)))
            if sha256(checkpoint / 'policy.pt') != metadata['policy_sha256']:
                raise ValueError('Checkpoint policy bytes changed')
            if {name:sha256(checkpoint / name) for name in ('manifest.json','policy.pt')} != receipt['checkpoint_sha256'][role]:
                raise ValueError('Checkpoint differs from the independent terminal audit')
            if (role == 'initial_checkpoint') != (metadata['training_iteration'] == 0):
                raise ValueError('Initial/final checkpoint role mismatch')
    return arms


def cases(plan):
    return [dict(condition=c, seed=s, mode=m) for c in plan['conditions']
            for s in plan['seeds'] for m in plan['modes']]


def reserved_seed_check(base_seed=3105, episodes_cap=5600):
    """Conservative full-budget exclusion, including the local worker stream."""
    import numpy as np
    intervals = [[int(np.random.SeedSequence([base_seed, worker, 0, 731]).generate_state(1)[0]),
                  episodes_cap + 2] for worker in range(29)]
    if any((seed-start) % 2**32 < count for seed in SEEDS for start,count in intervals):
        raise ValueError('Predeclared held-out seeds overlap reserved training streams')
    return dict(passed=True, seeds=list(SEEDS), worker_seed_intervals=intervals,
                terminal_observed_seed_check_still_required=True)


def physical_metrics(episode):
    """Keep completed duplicates, request science and physical work separate."""
    raw = episode['raw_metric_evidence']
    task = episode['cooperative_tasking']
    captures = raw['capture_products']
    qualified = len(captures) - task['rejected_acquisitions']
    duplicates = qualified - task['unique_acquisition_requests']
    if duplicates < 0:
        raise ValueError('Inconsistent completed acquisition accounting')
    science = .9 * task['acquisition_value'] + .1 * task['verified_delivery_value']
    if not math.isclose(science, task['unshaped_weighted_science'], abs_tol=1e-8):
        raise ValueError('Unshaped science weighting changed')
    import numpy as np
    window = task['window_s']
    closed_windows = int(episode['sim_time_s'] // window)
    requests = {index: set(map(int, np.random.default_rng(np.random.SeedSequence(
        [episode['seed'], index, 731])).choice(episode['config']['n_targets'],
        task['active_request_count'], replace=False))) for index in range(closed_windows)}
    def fulfilled(product, delivered=False):
        index = int(product['capture_time'] // window)
        time = product.get('delivery_time') if delivered else product['completion_time']
        if (index in requests and product['target_id'] in requests[index]
            and product['request_epoch'] == index*window
            and product['quality'] >= task['quality_threshold']
            and time is not None and time <= (index+1)*window):
            return index, product['target_id']
        return None
    acquisitions = {value for product in captures if (value := fulfilled(product)) is not None}
    deliveries = {value for item in raw['delivered_products']
                  if (value := fulfilled(item['product'], True)) is not None}
    requested = closed_windows * task['active_request_count']
    tx_start = 5 + episode['config']['n_candidates']
    decisions = episode.get('decision_records', [])
    resources = episode.get('resource_history', {})
    return dict(unshaped_request_science=science,
                acquisition_value=task['acquisition_value'], delivery_value=task['verified_delivery_value'],
                unique_useful_acquisitions=task['unique_acquisition_requests'],
                verified_useful_deliveries=task['useful_delivered_requests'],
                physical_completed_captures=len(captures), qualified_request_captures=qualified,
                completed_redundant_captures=duplicates,
                duplicate_attempts=episode['team_summary']['duplicate_attempt_count'],
                physical_deliveries=len(raw['delivered_products']),
                redundant_capture_cost=task['redundant_capture_cost'],
                transmission_occupied_s=sum(episode['broadcast_time_s'].values()),
                expired_requests=requested,
                missed_acquisition_deadlines=requested-len(acquisitions),
                missed_delivery_deadlines=requested-len(deliveries),
                TX_selections_with_legal_image_alternative=sum(
                    d['action'] >= tx_start and any(d['action_mask'][5:tx_start]) for d in decisions),
                TX_selections_with_legal_downlink_alternative=sum(
                    d['action'] >= tx_start and d['action_mask'][1] for d in decisions),
                minimum_battery_fraction={s:min(r['battery_fraction'] for r in rows) for s,rows in resources.items() if rows},
                maximum_storage_fraction={s:max(r['storage_fraction'] for r in rows) for s,rows in resources.items() if rows},
                action_time_s=episode['action_time_s'],
                packet_outcomes=episode['message_diagnostics']['packet_outcome_counts'],
                shaped_reward=sum(episode['cumulative_reward'].values()),
                real_decisions=sum(raw['policy_decisions'].values()),
                simulated_seconds=episode['sim_time_s'])


def _case_worker(case, arms, root):
    # Native/model imports occur only during explicitly launched compute execution.
    import torch
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.compare_learned_checkpoints import compare
    torch.set_num_threads(1)
    from examples.multiagent_imaging.cluster.effective_imports import require_effective_imports
    runtime = require_effective_imports()
    condition = case['condition']
    checkpoint = Path(arms[condition['arm']][condition['checkpoint_selector']])
    before = {name: sha256(checkpoint / name) for name in ('manifest.json', 'policy.pt')}
    child = Path(root) / f"{condition['label']}_{case['mode']}_seed{case['seed']}"
    rows = compare(MultiAgentImagingConfig.from_json(config_path(condition)),
                   {condition['label']: checkpoint}, [case['seed']], child,
                   modes=(case['mode'] == 'argmax',),
                   metadata_intervention=condition['metadata_intervention'],
                   audit_information_flow=True)
    row = rows[0]
    episode = json.loads(Path(row['episode']).read_text())
    if before != {name: sha256(checkpoint / name) for name in before}:
        raise ValueError('Frozen checkpoint bytes changed during evaluation')
    row.update(condition=condition['label'], selection_mode=case['mode'],
               config_sha256=condition['config_sha256'], checkpoint_hashes=before,
               physical=physical_metrics(episode), new_PPO_updates=0)
    write(child / 'effective-worker-runtime.json', runtime)
    write(child / 'frozen-case.json', row)
    return row


def checkpoint_bytes(arms):
    paths = {str(Path(arm[key]).resolve()) for arm in arms.values()
             for key in ('initial_checkpoint','final_checkpoint')}
    return {path: {str(p.relative_to(path)):sha256(p) for p in sorted(Path(path).rglob('*'))
                   if p.is_file()} for path in sorted(paths)}


def paired_summary(rows):
    pairs = {}
    for mode in MODES:
        for contrast, right in (('A_minus_C','C'), ('A_minus_B','B'), ('A_minus_D','D')):
            values = []
            for seed in SEEDS:
                selected = {r['condition']: r for r in rows if r['seed'] == seed and r['selection_mode'] == mode}
                if 'A' not in selected or right not in selected:
                    continue
                a, b = selected['A'], selected[right]
                if a['initial_conditions_sha256'] != b['initial_conditions_sha256']:
                    raise ValueError('Matched initial conditions differ')
                values.append(dict(seed=seed, **{k: a['physical'][k]-b['physical'][k] for k in (
                    'unshaped_request_science','unique_useful_acquisitions','verified_useful_deliveries',
                    'completed_redundant_captures','transmission_occupied_s')}))
            pairs[f'{mode}/{contrast}'] = dict(n=len(values), rows=values,
                mean={k: statistics.mean(v[k] for v in values) for k in values[0] if k != 'seed'} if values else {})
    return pairs


def run(plan, qualification, output, workers=8, qualification_root=None):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Full held-out missions require a separately authorized compute allocation')
    if not 1 <= workers <= plan['maximum_workers']:
        raise ValueError('Worker cap exceeded')
    arms = qualify(plan, qualification, artifact_root=qualification_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'plan.json', plan)
    write(output / 'qualification.json', qualification)
    before = checkpoint_bytes(arms)
    write(output / 'checkpoint-bytes-before.json', before)
    # A separate online run with zero training experience. Inherited training
    # spool paths must never ingest historical missions into evaluation logging.
    os.environ.pop('BSK_STUDY_EPISODE_SPOOL', None)
    from examples.multiagent_imaging.wandb_logging import TrainingLogger
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.memory_profile import MemoryProfile, read_cgroup_memory
    from examples.multiagent_imaging.cooperative_eval_logging import publish
    config = MultiAgentImagingConfig.from_json(config_path(plan['conditions'][0]))
    logdir = output.parent / 'evaluation-wandb'
    logdir.mkdir(exist_ok=False)
    logger = TrainingLogger(logdir, dict(mission=config.to_dict(), frozen_plan=plan,
                            evidence_kind='Frozen held-out evaluation, zero PPO updates'),
                            mode='online', required=True)
    logger.heartbeat('frozen_heldout')
    logger.start_heartbeat(status='frozen_heldout')
    profile = MemoryProfile(output, interval_s=20)
    profile.start(); profile.mark('frozen-policy-evaluation')
    rows = []
    failed = True
    def save(error=None):
        try:
            contrasts = paired_summary(rows)
        except ValueError:
            if error is None:
                raise
            contrasts = {}
        write(output / 'summary.json', dict(complete=len(rows)==32 and error is None,
            expected_episodes=32, completed_episodes=len(rows), initial_conditions_matched=True if len(rows)==32 and error is None else None,
            rows=rows, contrasts=contrasts, error=error, new_PPO_updates=0))
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_case_worker, case, arms, output) for case in cases(plan)]
            for future in as_completed(futures):
                rows.append(future.result())
                logger.check_background()
                save()
        after = checkpoint_bytes(arms)
        write(output / 'checkpoint-bytes-after.json', after)
        if before != after:
            raise ValueError('Frozen full checkpoint tree changed')
        publish(output.parent, config, logger=logger, finish=False)
        failed = False
    except BaseException as error:
        save(f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            profile.mark('after-frozen-evaluation'); profile.close()
            groups = read_cgroup_memory()
            write(output.parent / 'post-evaluation-memory.json', dict(cgroups=groups))
        except BaseException:
            failed = True
            raise
        finally:
            logger.finish(failed=failed)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('run'); b.add_argument('--plan', type=Path, required=True)
    b.add_argument('--qualification', type=Path, required=True); b.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.operation == 'prepare':
        if args.output.exists(): raise FileExistsError('Use a fresh plan path')
        write(args.output, prepare())
    else:
        run(json.loads(args.plan.read_text()), json.loads(args.qualification.read_text()), args.output,
            qualification_root=args.qualification.resolve().parent)


if __name__ == '__main__':
    main()
