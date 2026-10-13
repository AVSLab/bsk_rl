"""Read-only acceptance for the explicitly approved direct 24-hour launch."""
import argparse
import json
import math
from pathlib import Path


def validate_sync(record, workers=29):
    assert record['passed'] and record['learner_unchanged'] and record['driver_rng_unchanged']
    after = record['after']
    assert after['passed'] and after['topology']['passed']
    assert len(after['workers']) == workers
    assert {r['worker_index'] for r in after['workers']} == set(range(workers))
    assert all(r['passed'] and r['parameters_finite'] and
        r['actor_sha256'] == record['authoritative_actor_sha256'] and
        r['rng_before_sha256'] == r['rng_after_sha256'] and r['dropout'] and
        all(p == .1 for p in r['dropout'].values()) for r in after['workers'])
    assert all(r['state_unchanged'] for r in record['worker_state_comparisons'])


def validate_update(update):
    assert update['episodes'], 'Empty sampler'
    assert all(math.isfinite(v) for v in update['losses'].values())
    assert math.isfinite(update['gradient_l2']) and update['finite_gradients'] == 1
    assert all(update['parameter_change'][k] > 0 for k in ('actor_l2', 'critic_l2', 'l2'))
    assert all(e['simulated_seconds'] == 45000 for e in update['episodes']), 'False horizon'
    true = sum(sum(e['policy_decisions'].values()) for e in update['episodes'])
    assert true == update['actual_policy_decisions'] == update['condensed_learner_rows']
    return true


def validate_heldout(summary):
    assert summary['complete'] and summary['completed_episodes'] == 8
    assert len(summary['rows']) == 8
    assert {(r['policy'], r['seed'], r['mode']) for r in summary['rows']} == {
        (c, s, m) for c in ('initial', 'trained') for s in (401, 402)
        for m in ('deterministic', 'stochastic')}
    hashes = {}
    for row in summary['rows']:
        key = row['seed']
        previous = hashes.setdefault(key, row['initial_conditions_sha256'])
        assert previous == row['initial_conditions_sha256'], 'Unmatched initial conditions'


def verify(root, phase, config_path):
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.episode_accounting import reconcile_episode_audits, verify_retained_learner_batch
    from examples.multiagent_imaging.study_metrics import mission_metrics
    root = Path(root); folder = root / phase
    read = lambda p: json.loads(p.read_text())
    if phase == 'heldout':
        import hashlib
        summary = read(folder / 'summary.json')
        validate_heldout(summary)
        for row in summary['rows']:
            path = folder / Path(row['episode']).name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['episode_sha256']
            raw = read(path)
            assert raw['sim_time_s'] == 45000
            assert mission_metrics(raw['raw_metric_evidence'], 45000) == raw['mission_metrics']
        return dict(passed=True, heldout_episodes=8, rows=summary['rows'],
            operational_scope='Individual argmax and stochastic evaluations; not convergence or cooperation evidence')
    if phase == 'bootstrap':
        validate_sync(read(folder / 'initial-worker-modules.json'))
    driver = read(root / 'preflight/effective-driver.json')
    runtimes = list((folder / 'effective-worker-runtime').glob('*.json'))
    assert {int(p.name.split('-')[1]) for p in runtimes} == set(range(29))
    for p in runtimes:
        worker = read(p)
        assert worker['passed'] and worker['basilisk'] == driver['basilisk'] and worker['bsk_rl'] == driver['bsk_rl']
        for name, module in driver['modules'].items():
            assert all(worker['modules'][name][k] == module[k] for k in ('version', 'path', 'file_sha256'))
    updates = read(folder / 'updates.json')
    assert updates and (phase != 'bootstrap' or len(updates) == 1)
    episodes = []
    for update in updates:
        validate_update(update)
        topology = read(folder / f"worker-topology-{update['iteration']:04d}.json")
        assert topology['passed']
        for e in update['episodes']:
            assert mission_metrics(e, 45000) == e['mission_metrics']
        episodes.extend(update['episodes'])
    recovered, accounting = reconcile_episode_audits(episodes, folder / 'episode-spool')
    assert len(recovered) == len(episodes), 'Unaccounted or duplicated complete missions'
    assert accounting['new_sampled_true_decisions'] == sum(u['actual_policy_decisions'] for u in updates)
    batch = None
    if phase == 'bootstrap':
        batch = verify_retained_learner_batch(folder / 'learner_batch',
            MultiAgentImagingConfig.from_json(config_path), episodes, updates[0]['condensed_learner_rows'])
    identity = read(folder / 'wandb.json')
    import wandb
    live = wandb.Api(timeout=60).run(identity['entity'] + '/' + identity['project'] + '/' + identity['id'])
    assert live.id == identity['id']
    assert live.summary.get('training/iteration') == updates[-1]['iteration']
    assert live.summary.get('lifetime/true_decisions') == updates[-1]['policy_decisions_lifetime']
    required = ['losses/policy_loss', 'losses/vf_loss', 'mission/constellation_reward',
        'mission/useful_delivered_visits'] + [f'agent/sensor_{i}/reward' for i in range(4)] + [f'agent/sensor_{i}/storage_fraction' for i in range(4)]
    assert all(live.summary.get(k) is not None for k in required), 'Missing live W&B metrics'
    return dict(passed=True, phase=phase, updates=len(updates), complete_missions=len(episodes),
        new_true_decisions=accounting['new_sampled_true_decisions'],
        lifetime_true_decisions=updates[-1]['policy_decisions_lifetime'],
        training_iteration=updates[-1]['iteration'], wandb_id=live.id, wandb_url=live.url,
        learner_batch_verification=batch, operational_scope='Acceptance evidence, not convergence or cooperation benefit')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['bootstrap', 'training', 'heldout'], required=True)
    parser.add_argument('--config', type=Path, required=True); args = parser.parse_args()
    record = dict(passed=False, phase=args.phase)
    try:
        record.update(verify(args.root, args.phase, args.config))
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        with (args.root / f'{args.phase}-acceptance.json').open('x') as stream:
            json.dump(record, stream, indent=2); stream.write('\n')


if __name__ == '__main__': main()
