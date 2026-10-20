"""One local full-mission PPO update plus offline zero-update restore.

Uses the unchanged study config and AMOS optimizer settings. One local runner
cannot qualify Alpine's 29-worker topology. W&B/network acceptance is excluded.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['BSK_RL_RAY_OBJECT_STORE_BYTES'] = str(256 * 1024**2)
    os.environ['BSK_RL_MEMORY_PROFILE_INTERVAL_S'] = '10'
    os.environ['BSK_RL_TARGET_RECORDERS'] = '0'
    import Basilisk
    import numpy as np
    import ray
    import torch
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.train import train_run, LauncherResources
    from examples.multiagent_imaging.optimization import PPOSettings
    from examples.multiagent_imaging.episode_accounting import verify_retained_learner_batch
    from examples.multiagent_imaging.checkpoints import (
        resume_checkpoint, optimizer_digest, _runner_seed_state, _state_digest,
        study_driver_restore_evidence)
    from examples.multiagent_imaging.training_audit import frozen_policy_logits
    from examples.multiagent_imaging.update_evidence import file_sha256
    from examples.multiagent_imaging.memory_profile import ray_memory_options

    path = Path(__file__).with_name('configs') / 'walker4_t200_cd1_catalog_tx.json'
    config = MultiAgentImagingConfig.from_json(path)
    resources = LauncherResources(num_env_runners=0, ray_cpus=2)
    native = Basilisk.getBuildInfo()
    started = time.monotonic()
    rows = train_run(config, output=args.output/'train', iterations=1,
        train_batch_size=4200, resources=resources,
        ppo_settings=PPOSettings.from_profile('amos_best'), wandb_mode='disabled',
        capture_learner_batch=True, max_training_episodes=6,
        max_sampled_true_decisions=15000, max_wall_seconds=3600)
    assert len(rows) == 1
    row = rows[0]
    assert row['iteration'] == 1 and all(e['simulated_seconds'] == 45000 for e in row['episodes'])
    assert row['actual_policy_decisions'] == row['condensed_learner_rows']
    assert row['finite_gradients'] == 1 and np.isfinite(row['gradient_l2'])
    assert row['parameter_change']['actor_l2'] > 0 and row['parameter_change']['critic_l2'] > 0
    audit = verify_retained_learner_batch(args.output/'train/learner_batch', config,
        row['episodes'], row['condensed_learner_rows'])
    checkpoint = Path(row['checkpoint'])
    before = {str(p.relative_to(checkpoint)): file_sha256(p) for p in checkpoint.rglob('*') if p.is_file()}
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    os.environ['BSK_ZERO_UPDATE_RESTORE'] = '1'
    algorithm = None
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, **ray_memory_options())
    try:
        algorithm = resume_checkpoint(checkpoint, config, resources=resources)
        assert algorithm.training_iteration == 1
        assert optimizer_digest(algorithm) == manifest['optimizer_sha256']
        assert _runner_seed_state(algorithm.env_runner) == manifest['worker_seed_states'][0]
        wrapped = algorithm.env_runner.env.unwrapped
        assert wrapped._restore_step_calls == 0 and wrapped.par_env.simulator.sim_time == 0
        driver = study_driver_restore_evidence(algorithm, manifest)
        payload = torch.load(checkpoint/'policy.pt', map_location='cpu', weights_only=True)
        module = algorithm.get_module('imager')
        assert all(torch.equal(v, payload['weights'][k]) for k,v in module.state_dict().items())
        assert torch.equal(frozen_policy_logits(module,payload['observations']),payload['logits'])
        dropout = {name: layer.p for name,layer in module.named_modules() if isinstance(layer,torch.nn.Dropout)}
        assert dropout and all(p == .1 for p in dropout.values())
        restore = dict(optimizer_exact=True, weights_exact=True, logits_exact=True,
            restored_weights_sha256=_state_digest(module.state_dict()),
            worker_rng_seed_index_accounting_exact=True, zero_environment_steps=True,
            dropout=dropout, **driver)
    finally:
        if algorithm is not None:
            algorithm.stop()
        ray.shutdown()
        os.environ.pop('BSK_ZERO_UPDATE_RESTORE', None)
    after = {str(p.relative_to(checkpoint)): file_sha256(p) for p in checkpoint.rglob('*') if p.is_file()}
    assert before == after and Basilisk.getBuildInfo() == native
    result = dict(passed=True, native_build=native, science_config_sha256=file_sha256(path),
        completed_native_training_missions=len(row['episodes']), new_PPO_updates=1,
        true_decisions=row['actual_policy_decisions'], losses=row['losses'],
        gradient_l2=row['gradient_l2'], parameter_change=row['parameter_change'],
        retained_target_verification=audit, restore=restore,
        unchanged_checkpoint_files=len(before), full_alpine_topology_tested=False,
        wandb_tested=False, cluster_jobs=0, wall_s=time.monotonic()-started)
    with (args.output/'qualification.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('native_build','retained_target_verification','restore')}),flush=True)


if __name__ == '__main__':
    main()
