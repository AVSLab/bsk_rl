import pytest
from examples.multiagent_imaging.sampling_budget import worker_quota, budget_manifest


@pytest.mark.parametrize("runners", [1, 3, 4, 28])
@pytest.mark.parametrize("total", [0, 6, 30_000, 260_000, 600])
def test_global_cap_is_not_multiplied_by_workers(runners, total):
    quotas = [worker_quota(total, i, runners) for i in range(1, runners + 1)]
    assert sum(quotas) == total
    assert max(quotas) - min(quotas) <= 1
    assert worker_quota(total, 0, runners) is None


def test_new_invocation_replaces_previous_caps_and_restore_clears_them(monkeypatch):
    import os
    from examples.multiagent_imaging.sampling_budget import configure_environment
    for name in ['BSK_STUDY_NUM_ENV_RUNNERS','BSK_STUDY_MAX_EPISODES','BSK_STUDY_MAX_TRUE_DECISIONS']:
        monkeypatch.setenv(name,'999')
    configure_environment(6,30000,1)
    assert os.environ['BSK_STUDY_MAX_EPISODES']=='6'
    configure_environment(32,60000,28)
    assert os.environ['BSK_STUDY_NUM_ENV_RUNNERS']=='28'
    assert os.environ['BSK_STUDY_MAX_TRUE_DECISIONS']=='60000'
    report=configure_environment(None,None,28)
    assert 'BSK_STUDY_MAX_EPISODES' not in os.environ and 'BSK_STUDY_MAX_TRUE_DECISIONS' not in os.environ
    assert all(q['episodes'] is None and q['true_decisions'] is None for q in report['quotas'].values())


def test_local_sampling_and_invalid_topology():
    assert worker_quota(6, 0, 0) == 6
    assert worker_quota(None, 0, 0) is None
    for total, worker, runners in [(6, 1, 0), (6, 29, 28), (-1, 1, 1), (6, 1, -1)]:
        with pytest.raises(ValueError):
            worker_quota(total, worker, runners)


def test_job_manifest_records_all_workers_and_exact_totals():
    report = budget_manifest(600, 260_000, 28)
    assert len(report["quotas"]) == 28
    assert sum(q["episodes"] for q in report["quotas"].values()) == 600
    assert sum(q["true_decisions"] for q in report["quotas"].values()) == 260_000


def test_callbacks_use_partitioned_budget_without_charging_empty_reset(monkeypatch):
    from types import SimpleNamespace as NS
    from examples.multiagent_imaging.training_audit import CompletionAuditCallbacks
    from examples.multiagent_imaging import baseline_monte_carlo
    monkeypatch.setenv('BSK_STUDY_NUM_ENV_RUNNERS', '3')
    monkeypatch.setenv('BSK_STUDY_MAX_EPISODES', '6')
    monkeypatch.setenv('BSK_STUDY_MAX_TRUE_DECISIONS', '30000')
    monkeypatch.setattr(baseline_monte_carlo, 'initial_conditions', lambda env: {})
    callback = CompletionAuditCallbacks()
    monkeypatch.setattr(callback, '_record_resources', lambda runner: None)
    monkeypatch.setattr(callback, '_publish_progress', lambda runner: None)
    env = NS()
    runner = NS(worker_index=2, env=NS(unwrapped=NS(par_env=env)), study_sampled_episodes=1, study_new_true_decisions=9990)
    callback.on_episode_start(env_runner=runner)
    assert env.study_remaining_episode_budget == 1
    assert env.study_remaining_true_budget == 10
    callback.on_episode_start(env_runner=runner)
    assert env.study_remaining_episode_budget == 1  # Reset itself consumes no sample.


def test_exhausted_episode_budget_fails_before_first_physical_action(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from bsk_rl.completion_gym import CompletionConstellationTasking
    from bsk_rl.sats.roles import SpacecraftRole
    env = object.__new__(CompletionConstellationTasking)
    env.simulator=NS(sim_time=0);env.study_remaining_episode_budget=0
    env.satellites=[NS(role=SpacecraftRole.SENSING_AGENT,is_alive=lambda:True)]
    env.decision_counts={'sensor_0':0};env.default_seed=1;env._episode_seed_index=2;env.task_history=[]
    monkeypatch.setenv('BSK_STUDY_EPISODE_SPOOL',str(tmp_path))
    with pytest.raises(RuntimeError,match='sampling cap'):env._step([0])
    import json
    record=json.loads(next(tmp_path.glob('*.failed')).read_text())
    assert not record['reached_horizon'] and record['simulated_seconds']==0


def test_parallel_budget_matches_exposure_and_rejects_serial_timing(monkeypatch):
    from examples.multiagent_imaging.cluster import study_long_budget as module
    monkeypatch.setattr(module, 'source_gate', lambda: {'release': 'offline-test'})
    accepted=dict(passed=True,source_gate={'release':'offline-test'},
                  launcher_resources={'num_env_runners':28,'ray_cpus':32,'torch_threads':1},
                  measured_throughput={
                      'catalog_tx':dict(true_decisions=28000,training_episodes=28,update_wall_s=1800,startup_wall_s=100,max_heldout_wall_s=500),
                      'geometry_only':dict(true_decisions=29000,training_episodes=28,update_wall_s=1500,startup_wall_s=100,max_heldout_wall_s=450)})
    a=module.parallel_budget(accepted,'catalog_tx');b=module.parallel_budget(accepted,'geometry_only')
    assert a['soft_true_decision_target']==b['soft_true_decision_target']
    assert a['hard_new_sampled_decision_cap']==b['hard_new_sampled_decision_cap']
    assert a['soft_true_decision_target']%50000==0
    assert a['hard_new_sampled_decision_cap']==a['soft_true_decision_target']+2*29000
    assert a['training_start_update_cutoff_s']==b['training_start_update_cutoff_s']<86400
    accepted['launcher_resources']['num_env_runners']=1
    with pytest.raises(ValueError,match='one-runner timing is insufficient'):
        module.parallel_budget(accepted,'catalog_tx')
