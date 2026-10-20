"""Offline streaming audit of the preserved October 2026 failed Alpine pair.

Verify every archive member, reconstruct each spooled mission, and reconcile
each learned update without retaining full history. Extract only checkpoints,
bootstrap batches, and small operational evidence. Never initialize Ray/native.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import time

from examples.multiagent_imaging.update_evidence import JSONStream, file_sha256, peak_rss_bytes
from examples.multiagent_imaging.study_metrics import mission_metrics


class HashedReader(io.RawIOBase):
    def __init__(self, source):
        self.source, self.digest, self.count = source, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.source.read(len(buffer))
        self.digest.update(data)
        self.count += len(data)
        buffer[:len(data)] = data
        return len(data)


def content_hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def equal_metrics(actual, expected):
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(equal_metrics(v, expected[k]) for k, v in actual.items())
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(equal_metrics(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, (float, int)) and not isinstance(actual, bool):
        return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)
    return actual == expected


def compact_episode(episode):
    assert episode['simulated_seconds'] == 45000
    assert episode['coverage']['catalog_target_count'] == 200
    assert len(episode['policy_decisions']) == 4
    assert all(sum(episode['action_selection_counts'][s].values()) == count
               for s, count in episode['policy_decisions'].items())
    assert math.isclose(sum(e['reward'] for e in episode['reward_events']),
                        episode['constellation_reward'], abs_tol=1e-7)
    return dict(lifetime_episode_id=episode['lifetime_episode_id'], episode_id=episode['episode_id'],
        worker_index=episode['worker_index'], episode_seed=episode['episode_seed'],
        content_sha256=content_hash(episode), simulated_seconds=45000,
        true_decisions=sum(episode['policy_decisions'].values()),
        reward=episode['constellation_reward'],
        initial_conditions_sha256=episode['initial_conditions_sha256'])


def audit_updates(stream, spool, phase, output):
    reader, updates, learned_ids = JSONStream(stream), [], set()
    reader.expect('[')
    previous = None
    if reader.peek() != ']':
        while True:
            update, missions = {}, []
            for key in reader.keys():
                if key == 'episodes':
                    for episode in reader.array():
                        compact = compact_episode(episode)
                        identity = compact['lifetime_episode_id']
                        assert identity not in learned_ids, 'Repeated learned mission'
                        assert identity in spool and spool[identity]['content_sha256'] == compact['content_sha256'], 'Spool/update payload mismatch'
                        learned_ids.add(identity)
                        missions.append(compact)
                else:
                    update[key] = reader.value()
            assert missions
            true = sum(e['true_decisions'] for e in missions)
            assert true == update['actual_policy_decisions'] == update['condensed_learner_rows']
            assert len(missions) == update['complete_episode_count']
            assert all(math.isfinite(v) for v in update['losses'].values())
            assert math.isfinite(update['gradient_l2']) and update['finite_gradients'] == 1
            assert all(update['parameter_change'][k] > 0 for k in ('actor_l2', 'critic_l2', 'l2'))
            if previous is not None:
                assert update['iteration'] == previous['iteration'] + 1
                assert update['policy_decisions_lifetime'] - previous['policy_decisions_lifetime'] == true
            accounting = update.get('episode_accounting')
            if accounting:
                assert accounting['new_sampled_true_decisions'] == true
                assert accounting['episode_content_sha256'] == {e['lifetime_episode_id']: e['content_sha256'] for e in missions}
            assert update['restore_validation']['matched_actions']
            summary = {k: update[k] for k in ('iteration', 'losses', 'gradient_l2', 'finite_gradients',
                'parameter_change', 'actual_policy_decisions', 'condensed_learner_rows',
                'repeated_learner_presentations', 'policy_decisions_lifetime', 'checkpoint', 'restore_validation')}
            summary.update(complete_missions=len(missions), mission_ids=[e['lifetime_episode_id'] for e in missions],
                           mean_reward=sum(e['reward'] for e in missions) / len(missions))
            updates.append(summary)
            previous = summary
            if reader.peek() == ']':
                break
            reader.expect(',')
    reader.expect(']')
    reader.finish()
    assert updates
    unlearned = sorted(set(spool) - learned_ids)
    with output.open('x') as stream:
        json.dump(dict(phase=phase, updates=updates, learned_missions=len(learned_ids),
            sampled_but_unassigned_missions=unlearned,
            unassigned_semantics='Sampled but absent from accepted update records; may have entered a failed optimizer call. Never infer these rows were successfully learned.',
            unassigned_sampled_true_decisions=sum(spool[e]['true_decisions'] for e in unlearned)), stream, indent=2)
    return dict(updates=len(updates), first_iteration=updates[0]['iteration'], last_iteration=updates[-1]['iteration'],
        learned_missions=len(learned_ids), learned_true_decisions=sum(u['actual_policy_decisions'] for u in updates),
        lifetime_true_decisions=updates[-1]['policy_decisions_lifetime'], last_checkpoint=updates[-1]['checkpoint'],
        sampled_but_unassigned_missions=len(unlearned),
        unassigned_sampled_true_decisions=sum(spool[e]['true_decisions'] for e in unlearned))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    assert args.archive.stat().st_size == 3244360920
    outer = file_sha256(args.archive)
    assert outer == '03e172312485a723d59b15cf696ca2fb9bc2318448e5725549d84a89b9c2698a'
    manifest = json.loads(args.manifest.read_text())
    expected = {row['path']: row for row in manifest['files']}
    assert len(expected) == len(manifest['files']) == 6884
    seen, spools, update_summary, topology = set(), {}, {}, {}
    mission_stream = (args.output / 'individual-missions.jsonl').open('x')
    memory_summary = {}
    extracted_bytes = 0
    try:
        with tarfile.open(args.archive, 'r|gz') as archive:
            for member in archive:
                assert member.isfile() and not Path(member.name).is_absolute()
                parts = Path(member.name).parts
                assert len(parts) >= 2 and '..' not in parts
                name = '/'.join(parts[1:])
                raw = HashedReader(archive.extractfile(member))
                binary = io.BufferedReader(raw)
                if name in {'manifest.json', 'scheduler.txt'}:
                    value = binary.read()
                    assert len(value) < 4 * 1024 * 1024
                    if name == 'manifest.json':
                        assert json.loads(value) == manifest
                    else:
                        assert value.decode() == manifest['scheduler']
                    continue
                row = expected[name]
                assert name not in seen and member.size == row['bytes']
                path = Path(name)
                arm = parts[1]
                phase = path.parts[2] if len(path.parts) > 2 else ''
                key = arm + '/' + phase
                if '/episode-spool/' in name and path.parent.name == 'episode-spool':
                    episode = json.load(io.TextIOWrapper(binary, encoding='utf-8'))
                    compact = compact_episode(episode)
                    identity = compact['lifetime_episode_id']
                    assert path.name == hashlib.sha256(identity.encode()).hexdigest() + '.json'
                    phase_spool = spools.setdefault(key, {})
                    assert identity not in phase_spool
                    metric = mission_metrics(episode, 45000)
                    assert equal_metrics(metric, episode['mission_metrics']), 'Recomputed mission metric differs'
                    compact.update(useful_delivered_visits=metric['useful_delivered_visits'],
                        ground_coverage_fraction=metric['ground_coverage_fraction'],
                        redundant_execution_fraction=metric['execution_redundancy']['fraction_of_constellation_available_time'],
                        mission_metrics_recomputed=True)
                    phase_spool[identity] = compact
                    mission_stream.write(json.dumps(dict(arm=arm, phase=phase, **compact)) + '\n')
                    if len(phase_spool) % 100 == 0:
                        mission_stream.flush()
                        print('MISSIONS', key, len(phase_spool), flush=True)
                elif name.endswith('/updates.json'):
                    update_summary[key] = audit_updates(io.TextIOWrapper(binary, encoding='utf-8'),
                        spools.get(key, {}), phase, args.output / (arm + '-' + phase + '-updates.json'))
                    print('UPDATES', key, update_summary[key], flush=True)
                elif name.endswith('/memory-profile.jsonl'):
                    summary = dict(samples=0, peak_bytes=0, failcnt_max=0, first_time=None, last_time=None)
                    for line in io.TextIOWrapper(binary, encoding='utf-8'):
                        item = json.loads(line)
                        summary['samples'] += 1
                        summary['first_time'] = summary['first_time'] or item['time_unix_s']
                        summary['last_time'] = item['time_unix_s']
                        for group in item['cgroups']:
                            if group['path'].endswith(('/job_33993838', '/job_33993840')):
                                values = group['values']
                                assert int(values['memory.limit_in_bytes']) == 200 * 2**30
                                summary['peak_bytes'] = max(summary['peak_bytes'], int(values['memory.max_usage_in_bytes']))
                                summary['failcnt_max'] = max(summary['failcnt_max'], int(values['memory.failcnt']))
                    memory_summary[key] = summary
                else:
                    # Preserve checkpoints/batches plus bounded operational JSON,
                    # logs and manifests, never duplicate the 10.9 GB spool.
                    keep = ('/checkpoint_' in name or '/learner_batch/' in name or
                        (member.size <= 16 * 1024 * 1024 and '/episode-spool/' not in name))
                    destination = args.output / 'selected' / name
                    if keep:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open('xb') as stream:
                            shutil.copyfileobj(binary, stream, 1024 * 1024)
                        extracted_bytes += member.size
                        if path.name.startswith('worker-topology-') and path.suffix == '.json':
                            topology[name] = json.loads(destination.read_text())
                    else:
                        while binary.read(1024 * 1024):
                            pass
                assert raw.count == row['bytes'] and raw.digest.hexdigest() == row['sha256'], name
                seen.add(name)
                if len(seen) % 1000 == 0:
                    print('VERIFIED', len(seen), flush=True)
        assert seen == set(expected)
    finally:
        mission_stream.close()
    result = dict(archive_bytes=args.archive.stat().st_size, archive_sha256=outer,
        all_archive_files_verified=len(seen), extracted_bytes=extracted_bytes,
        updates=update_summary, memory=memory_summary, scheduler=manifest['scheduler'],
        recomputed_missions=sum(len(s) for s in spools.values()),
        sampled_true_decisions={k:sum(e['true_decisions'] for e in v.values()) for k,v in spools.items()},
        topology={k:dict(passed=v.get('passed'), actor_restart_count=v.get('actor_restart_count'),
                        errors=v.get('errors'), phase=v.get('phase')) for k,v in topology.items()},
        retained_target_scope='Only bootstrap raw learner batches were retained. Later update target arrays cannot be independently reconstructed from this archive.',
        heldout_present=any('/heldout/' in n for n in expected),
        local_peak_rss_bytes=peak_rss_bytes(),
        wall_s=time.monotonic()-started, no_Ray_or_native_initialization=True)
    with (args.output / 'archive-audit.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print('AUDIT COMPLETE', len(seen), 'files', result['recomputed_missions'], 'missions', flush=True)


if __name__ == '__main__':
    main()
