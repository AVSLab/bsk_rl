"""Offline collector tests with fake rollouts; no physical episode is sampled."""
import concurrent.futures
import json
from pathlib import Path
import pytest
from types import SimpleNamespace
from examples.multiagent_imaging import compare_learned_checkpoints as module
from examples.multiagent_imaging.config import MultiAgentImagingConfig


@pytest.fixture
def fake_rollouts(monkeypatch):
    monkeypatch.setattr(module, 'load_policy', lambda checkpoint, config: (SimpleNamespace(label=checkpoint,training=False), {'passed': True}))
    monkeypatch.setattr(module, 'policy_callable', lambda model, deterministic, seed: (model.label, deterministic))
    def rollout(config, policy, metadata_intervention='normal', audit_information_flow=False):
        return dict(sim_time_s=45000, initial_conditions_sha256=f'seed-{config.seed}',
                    cumulative_reward={'sensor_0': 1.},
                    coverage={'capture_coverage_fraction': .5, 'ground_delivery_coverage_fraction': .25},
                    team_summary={'unique_acquisition_count': 2, 'unique_service_count': 1, 'duplicate_attempt_count': 0},
                    useful_post_cooldown_revisits={'count': 0}, action_counts={})
    monkeypatch.setattr(module, 'run_rollout', rollout)
    def pool(*, max_workers, mp_context, initializer):
        assert mp_context.get_start_method() == 'spawn'
        assert initializer is module._evaluation_worker_init
        return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    monkeypatch.setattr(module.concurrent.futures, 'ProcessPoolExecutor', pool)
    return rollout


def test_all_eight_unique_cases_and_flat_hashed_evidence(tmp_path, fake_rollouts):
    output = tmp_path/'heldout'
    rows = module.compare(MultiAgentImagingConfig(episode_duration_s=45000), {'initial': 'a', 'trained': 'b'}, [401, 402], output, workers=8)
    assert len(rows) == 8
    assert len({(r['policy'], r['mode'], r['seed']) for r in rows}) == 8
    assert all(Path(r['episode']).parent == output for r in rows)
    assert all(module.hashlib.sha256(Path(r['episode']).read_bytes()).hexdigest() == r['episode_sha256'] for r in rows)
    summary = json.loads((output/'summary.json').read_text())
    assert summary['complete'] and summary['initial_conditions_matched']
    assert len(list((output/'cases').rglob('summary.json'))) == 8


def test_hash_mismatch_is_retained_failure(tmp_path, fake_rollouts, monkeypatch):
    def changed(config, policy, metadata_intervention='normal', audit_information_flow=False):
        record = fake_rollouts(config, policy)
        if policy == ('b', False): record['initial_conditions_sha256'] += '-changed'
        return record
    monkeypatch.setattr(module, 'run_rollout', changed)
    with pytest.raises(AssertionError, match='not matched'):
        module.compare(MultiAgentImagingConfig(episode_duration_s=45000), {'initial': 'a', 'trained': 'b'}, [401], tmp_path, workers=4)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert not summary['complete'] and summary['initial_conditions_matched'] is False


def test_false_horizon_and_repeated_seed_rejected(tmp_path, fake_rollouts, monkeypatch):
    def partial(config, policy, metadata_intervention='normal', audit_information_flow=False):
        record = fake_rollouts(config, policy);record['sim_time_s'] = 44999;return record
    monkeypatch.setattr(module, 'run_rollout', partial)
    with pytest.raises(AssertionError, match='horizon'):
        module.compare(MultiAgentImagingConfig(episode_duration_s=45000), {'initial': 'a'}, [401], tmp_path, workers=2)
    assert not json.loads((tmp_path/'summary.json').read_text())['complete']
    assert list((tmp_path/'cases').rglob('failed-horizon.json'))
    with pytest.raises(ValueError, match='unique seeds'):
        module.compare(MultiAgentImagingConfig(episode_duration_s=45000), {'initial': 'a'}, [401, 401], tmp_path/'new')
