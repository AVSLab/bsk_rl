"""Finish the offline audit: retained targets, checkpoint bytes and runtime gates.

No mission, Ray initialization, optimization, unrestricted checkpoint unpickling,
network, or native upgrade. Exported tensors use Torch's weights-only loader;
Ray's protocol-5 NumPy weights use an explicit numeric-only class allowlist. Full late
optimizer restoration remains a future compute gate, not an inferred success.
"""
import argparse
import ast
from collections import OrderedDict
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
from types import SimpleNamespace
import shutil
import tarfile

import numpy as np
import torch

from examples.multiagent_imaging.update_evidence import file_sha256


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


class NumericStateUnpickler(pickle.Unpickler):
    """Decode Ray's NumPy-only state export, rejecting arbitrary constructors."""
    def find_class(self, module, name):
        allowed = {('collections','OrderedDict'):OrderedDict,
            ('numpy','dtype'):np.dtype, ('numpy','ndarray'):np.ndarray,
            ('numpy.core.numeric','_frombuffer'):np.core.numeric._frombuffer,
            ('numpy.core.multiarray','_reconstruct'):np.core.multiarray._reconstruct}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Non-numeric checkpoint constructor: {module}.{name}')
        return allowed[module,name]


def numeric_weights(path):
    with path.open('rb') as stream:
        weights = NumericStateUnpickler(stream).load()
        assert not stream.read(1)
    assert isinstance(weights,dict) and all(isinstance(k,str) and isinstance(v,np.ndarray)
        and v.dtype.kind in 'fc' and np.isfinite(v).all() for k,v in weights.items())
    return weights


def terminal_log_evidence(path, job_id):
    """Slurm's explicit event is stronger evidence than Ray's generic causes."""
    events, worker_exits = [], []
    pattern = re.compile(rf'Detected (\d+) oom_kill event.*StepId={job_id}\.batch')
    for number, line in enumerate(path.open(), 1):
        match = pattern.search(line)
        if match:
            events.append(dict(line=number, count=int(match.group(1)), text=line.strip()))
        if 'Worker exit type: SYSTEM_ERROR' in line:
            pid = re.search(r'Worker PID: (\d+)', line)
            worker_exits.append(dict(line=number, pid=int(pid.group(1)) if pid else None,
                error_type='SYSTEM_ERROR', generic_possible_causes_are_not_pid_specific_proof=True))
    return dict(job_id=job_id, log_sha256=file_sha256(path),
        allocation_oom_kill_confirmed=any(e['count'] > 0 for e in events), slurm_oom_events=events,
        unexpected_worker_exits=worker_exits, specific_pid_oom_attribution_proven=False)


def verify_exported_logits(folder):
    """Measure cross-runtime probes; never weaken the exact restore gate.

    Float32 Linux/x86 and macOS/arm64 kernels need not produce bitwise logits.
    Report errors and argmax disagreements without promoting them to a successful
    same-runtime restore. Exported bytes, schema, finite outputs and masks must
    still agree. No tolerance is fitted to make an acceptance flag pass.
    """
    import gymnasium as gym
    from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from bsk_rl.utils.rllib.target_gnn_module import GNNModule
    from examples.multiagent_imaging.config import MultiAgentImagingConfig
    from examples.multiagent_imaging.readiness import validate_schema
    metadata = read(folder/'manifest.json')
    validate_schema(metadata['schema'],MultiAgentImagingConfig(**metadata['config']))
    assert file_sha256(folder/'policy.pt') == metadata['policy_sha256']
    payload = torch.load(folder/'policy.pt',map_location='cpu',weights_only=True)
    contract = metadata['schema']
    rng = torch.get_rng_state()
    try:
        module = RLModuleSpec(module_class=GNNModule,catalog_class=PPOCatalog,
            observation_space=gym.spaces.Box(-np.inf,np.inf,(contract['observation']['size'],),np.float32),
            action_space=gym.spaces.Discrete(contract['actions']['size']),
            model_config_dict=metadata['model_config'],inference_only=metadata['inference_only']).build()
    finally:
        torch.set_rng_state(rng)
    module.load_state_dict(payload['weights'],strict=True)
    module.eval()
    with torch.no_grad():
        actual = module.forward_inference({Columns.OBS:payload['observations']})[Columns.ACTION_DIST_INPUTS].cpu()
    expected = payload['logits']
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    invalid = expected <= -1e8
    assert torch.equal(actual <= -1e8,invalid)
    assert torch.equal(actual[invalid],expected[invalid])
    delta = (actual-expected).abs()
    mismatches = int((actual.argmax(-1)!=expected.argmax(-1)).sum())
    evidence = dict(max_logit_error=float(delta.max()),
        valid_logit_max_abs_error=float(delta[~invalid].max()),
        probe_count=int(actual.shape[0]),matched_actions=mismatches==0,argmax_mismatches=mismatches,
        logits_bitwise_equal=torch.equal(actual,expected),invalid_masks_exact=True,
        same_runtime_restore_acceptance=False,
        scope='Cross-platform numerical readback only; exact same-runtime restore gate unchanged')
    del module
    return evidence


def change(before, after):
    branches = {'actor': 0., 'critic': 0.}
    count, largest = 0, 0.
    for key in before:
        delta = after[key].numpy() - before[key].numpy()
        assert np.isfinite(delta).all()
        branches['critic' if key.startswith('vf.') else 'actor'] += float(np.square(delta).sum())
        count += int(np.count_nonzero(delta))
        largest = max(largest, float(np.max(np.abs(delta))))
    return dict(actor_l2=math.sqrt(branches['actor']), critic_l2=math.sqrt(branches['critic']),
        l2=math.sqrt(sum(branches.values())), changed_elements=count, max_absolute=largest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    complete = read(args.audit/'archive-audit.json')
    assert complete['all_archive_files_verified'] == 6884
    manifest = read(args.manifest)
    expected = {e['path']:e for e in manifest['files']}
    # A second bounded pass extracts only the 56 bootstrap spool missions needed
    # by the existing independent retained-target verifier, not all long history.
    extracted = 0
    with tarfile.open(args.archive, 'r|gz') as archive:
        for member in archive:
            name = '/'.join(Path(member.name).parts[1:])
            if '/bootstrap/episode-spool/' not in name or Path(name).parent.name != 'episode-spool':
                continue
            destination = args.audit/'selected'/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():
                with destination.open('xb') as stream:
                    shutil.copyfileobj(archive.extractfile(member),stream,1024*1024)
            assert destination.stat().st_size == expected[name]['bytes']
            assert file_sha256(destination) == expected[name]['sha256']
            extracted += 1
    assert extracted == 56
    namespace = dict(np=np,json=json,gzip=gzip,hashlib=hashlib,Path=Path,NO_ACTION=2**31-1)
    for module, function in [('fixed_batch_diagnostic','independently_recompute'),
                             ('episode_accounting','verify_retained_learner_batch')]:
        source = Path(__file__).with_name(module+'.py')
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == function)
        node.body = [n for n in node.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    arms = {}
    initial_weights = None
    initial_optimizer = None
    for arm in ('with_communication','without_communication'):
        root = args.audit/'selected'/arm/'diagnostic'
        job_id = 33993838 if arm == 'with_communication' else 33993840
        log_evidence = terminal_log_evidence(root.parent/f'slurm-{job_id}.log',job_id)
        assert log_evidence['allocation_oom_kill_confirmed']
        updates = read(args.audit/(arm+'-bootstrap-updates.json'))['updates'] + read(args.audit/(arm+'-training-updates.json'))['updates']
        by_iteration = {u['iteration']:u for u in updates}
        assert list(by_iteration) == list(range(1,33))
        bootstrap = root/'bootstrap'
        config_data = read(bootstrap/'checkpoint_0001/manifest.json')['config']
        config = SimpleNamespace(**config_data)
        config.effective_gae_lambda = config.gae_lambda_per_decision
        config.gae_lambda_time = 'step'
        episodes = [read(p) for p in sorted((bootstrap/'episode-spool').glob('*.json'))]
        target = namespace['verify_retained_learner_batch'](bootstrap/'learner_batch',config,
            episodes,by_iteration[1]['condensed_learner_rows'])
        del episodes
        restore = read(root/'bootstrap-resume/zero-update-resume.json')
        bootstrap_manifest = read(bootstrap/'checkpoint_0001/manifest.json')
        assert restore['passed'] and restore['exact_logits'] and restore['max_logit_error'] == 0
        assert restore['new_ppo_updates'] == restore['new_sampled_episodes'] == 0
        assert restore['optimizer_sha256'] == bootstrap_manifest['optimizer_sha256']
        assert restore['restored_weights_sha256'] == restore['exported_weights_sha256']
        assert restore['lifetime_metrics_matches_checkpoint'] and restore['training_accounting_matches_checkpoint']
        assert len(restore['workers']) == 29 and all(w['matches_checkpoint'] and w['no_environment_steps'] and w['sim_time_s'] == 0 for w in restore['workers'])
        assert read(root/'bootstrap-checkpoint-before.json') == read(root/'bootstrap-checkpoint-after.json')
        before = None
        checkpoints = []
        for iteration in range(33):
            folder = root/('bootstrap' if iteration < 2 else 'training')/f'checkpoint_{iteration:04d}'
            meta = read(folder/'manifest.json')
            assert meta['training_iteration'] == iteration and meta['config'] == config_data
            assert meta['worker_topology']['expected_indices'] == list(range(29))
            assert sorted(s['worker_index'] for s in meta['worker_seed_states']) == list(range(29))
            assert all(len(s['torch_rng']) == 5056 and isinstance(s['pending_completed_episodes'],list)
                       and 'episode_hashes' in s['study_audit_state'] for s in meta['worker_seed_states'])
            assert meta['policy_sha256'] == file_sha256(folder/'policy.pt')
            payload = torch.load(folder/'policy.pt',map_location='cpu',weights_only=True)
            learner_weights = numeric_weights(folder/'algorithm/learner_group/learner/rl_module/imager/module_state.pt')
            weights = payload['weights']
            assert set(weights) == set(learner_weights)
            assert all(torch.isfinite(v).all() and np.array_equal(v.numpy(),learner_weights[k]) for k,v in weights.items())
            if iteration == 0:
                if initial_weights is None:
                    initial_weights = weights
                    initial_optimizer = meta['optimizer_sha256']
                else:
                    assert initial_optimizer == meta['optimizer_sha256']
                    assert all(torch.equal(v,initial_weights[k]) for k,v in weights.items())
            saved_restore = read(folder/'restore_validation.json')
            assert saved_restore['matched_actions'] and saved_restore['max_logit_error'] == 0
            independent_logits = verify_exported_logits(folder)
            topology_before, topology_after = read(folder/'worker-save-before.json'),read(folder/'worker-save-after.json')
            assert topology_before['passed'] and topology_after['passed']
            assert topology_before['actor_restart_count'] == topology_after['actor_restart_count'] == 0
            assert topology_before['process_identities'] == topology_after['process_identities']
            assert topology_before['state_sha256'] == topology_after['state_sha256']
            if iteration:
                update = by_iteration[iteration]
                measured = change(before,weights)
                assert all(math.isclose(measured[k],update['parameter_change'][k],rel_tol=1e-6,abs_tol=1e-12) for k in measured)
                assert meta['policy_decisions_lifetime'] == update['policy_decisions_lifetime']
            before = weights
            checkpoints.append(dict(iteration=iteration,policy_sha256=meta['policy_sha256'],
                optimizer_sha256=meta['optimizer_sha256'],policy_and_learner_tensors_exact=True,
                independently_recomputed_logits=independent_logits,
                lifetime_true_decisions=meta['policy_decisions_lifetime'],full_worker_state_present=True))
        runtime_checks = 0
        driver = read(root/'preflight/effective-driver.json')
        assert driver['passed'] and driver['modules']['psutil']['version'] == '6.1.0'
        for phase in ('bootstrap','bootstrap-resume','training'):
            files = list((root/phase/'effective-worker-runtime').glob('*.json'))
            assert {int(p.name.split('-')[1]) for p in files} == set(range(29))
            for path in files:
                worker = read(path)
                assert worker['passed'] and worker['basilisk'] == driver['basilisk'] and worker['bsk_rl'] == driver['bsk_rl']
                for name,module in driver['modules'].items():
                    assert all(worker['modules'][name][k] == module[k] for k in ('version','path','file_sha256'))
                    assert all(worker['modules'][name]['native_sha256'].get(p) == sha for p,sha in module['native_sha256'].items())
                runtime_checks += 1
        topologies = sorted((root/'training').glob('worker-topology-0*.json'))
        assert len(topologies) == 31
        identities = None
        for path in topologies:
            row = read(path)
            assert row['passed'] and row['actor_restart_count'] == 0
            assert not row['changed_process_indices']
            identities = identities or row['process_identities']
            assert identities == row['process_identities']
        failed_path = root/'training/worker-topology-failed.json'
        failure = read(failed_path) if failed_path.exists() else None
        failed_training = root/'training/training-failed.json'
        arms[arm] = dict(bootstrap_target_reconstruction=target,
            terminal_log_evidence=log_evidence,
            bootstrap_full_zero_update_restore_revalidated=True,
            checkpoints=checkpoints, runtime_worker_records_verified=runtime_checks,
            passed_training_topologies=len(topologies), failed_topology=failure,
            failed_later_training_iteration=read(failed_training).get('training_iteration') if failed_training.exists() else None,
            last_trustworthy_checkpoint_iteration=32,
            late_optimizer_resume_tested=False,heldout_present=False)
        print('CHECKPOINTS/TARGETS',arm,'passed',flush=True)
    result = dict(available_archive_scientific_audit_complete=True,arms=arms,
        fresh_learner_weights_and_optimizer_identical_across_arms=True,
        terminal_states=dict(with_communication='OUT_OF_MEMORY',without_communication='CANCELLED after demonstrated worker replacement'),
        scheduler=manifest['scheduler'],
        allocated_cpu_hours=dict(with_communication=71442*56/3600,without_communication=67094*56/3600,
            prior_failed_attempts=(7+10)*56/3600,pair_including_prior=(71442+67094+7+10)*56/3600,
            separate_diagnostic=1937*56/3600),
        limitations=['No final held-out episodes exist in the archive',
            'Later raw learner batches were not retained; exact later target reconstruction is unavailable',
            'Late checkpoint optimizer bytes are verified and preserved; full 29-worker restore has not been rerun locally',
            'Cross-platform probe errors are measured separately and do not relax the exact same-runtime restore gate',
            'Sampled missions outside accepted update records may have entered a failed optimizer call; they are excluded from accepted learned totals',
            'Both allocation logs confirm an OOM kill; the archive does not bind the OOM event to each specific worker PID'],
        new_native_missions=0,new_PPO_updates=0,Ray_initialized=False)
    with (args.audit/'scientific-audit.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)


if __name__ == '__main__':
    main()
