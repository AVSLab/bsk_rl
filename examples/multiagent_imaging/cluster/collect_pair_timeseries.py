"""Bounded read-only training curves for the already running cooperative pair.

Scan each immutable update journal once for summaries; never retain decision,
observation, target or product arrays. No Ray/native imports or job operations.
The raw terminal archive and independent numerical audit remain required.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import resource
import runpy
import sys

BASE = Path('/scratch/alpine/dahu1128/cooperative-v1-c18-pair-24h-20261022')
EVIDENCE = Path('/projects/dahu1128/cooperative-v1-c18-pair-24h-evidence-20261022')
SOURCE = Path('/projects/dahu1128/bsk_rl-cooperative-v1-c18-pair-b205fc16-20261022')
READER = EVIDENCE / 'comparison-reader-v1.py'
READER_SHA = 'b35662df0d4ab99fb86821ea26d7f9a42819c2596b96fc62fe2741507adaf723'
EPISODE_FIELDS = {'episode_id', 'episode_seed', 'initial_conditions_sha256',
    'simulated_seconds', 'policy_decisions', 'constellation_reward',
    'cooperative_tasking', 'communication_summary', 'team_summary',
    'action_selection_counts'}
UPDATE_FIELDS = {'iteration', 'measurement', 'losses', 'gradient_l2',
    'finite_gradients', 'parameter_change', 'actual_policy_decisions',
    'complete_episode_count', 'policy_decisions_lifetime',
    'condensed_learner_rows', 'learner_metrics'}


def project_episode(reader, skip_value, project):
    row = {}
    for key in reader.keys():
        if key in EPISODE_FIELDS:
            row[key] = reader.value()
        elif key == 'capture_products':
            row['physical_completed_captures'] = sum(1 for _ in reader.array())
        elif key == 'mission_metrics':
            row[key] = project(reader, {'agents', 'true_decisions',
                'reached_horizon', 'actual_duration_s', 'execution_redundancy'})
        else:
            skip_value(reader)
    return row


def project_update(reader, skip_value, project):
    row = {}
    for key in reader.keys():
        if key == 'episodes':
            reader.expect('[')
            episodes = []
            if reader.peek() != ']':
                while True:
                    if len(episodes) >= 56:
                        raise ValueError('Unexpected missions per update')
                    episodes.append(project_episode(reader, skip_value, project))
                    if reader.peek() == ']':
                        break
                    reader.expect(',')
            reader.expect(']')
            row['episodes'] = episodes
        elif key in UPDATE_FIELDS:
            row[key] = reader.value()
        else:
            skip_value(reader)
    return row


def collect(source_root=SOURCE, base=BASE, evidence=EVIDENCE, reader_path=READER):
    helper = runpy.run_path(str(reader_path))
    digest = helper['digest']
    if digest(reader_path) != READER_SHA:
        raise ValueError('Projection helper identity changed')
    skip_value, project = helper['skip_value'], helper['project']
    parser_path = source_root / 'examples/multiagent_imaging/update_evidence.py'
    JSONStream = runpy.run_path(str(parser_path))['JSONStream']
    result = dict(version='cooperative-pair-training-timeseries-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(),
        source_root=str(source_root), parser_sha256=digest(parser_path),
        reader_sha256=digest(Path(__file__)), projection_reader_sha256=READER_SHA,
        new_jobs=0, new_PPO_updates=0, numerical_terminal_audit=False,
        arms={}, input_manifest=[])
    for arm in ('with_communication', 'without_communication'):
        train = base / arm / 'training'
        files = sorted((train / 'update-journal').glob('update-*.json'))
        if len(files) > 200:
            raise ValueError('Authorized update cap exceeded')
        output = dict(updates=[], memory=[])
        episodes = 0
        for path in files:
            before = path.stat()
            with path.open() as stream:
                reader = JSONStream(stream)
                row = project_update(reader, skip_value, project)
                reader.finish()
            episodes += len(row['episodes'])
            if episodes > 5600:
                raise ValueError('Authorized mission cap exceeded')
            row['journal_published_unix_s'] = before.st_mtime
            output['updates'].append(row)
            record = dict(path=str(path), size_bytes=before.st_size, sha256=digest(path))
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Immutable update changed during read')
            result['input_manifest'].append(record)
        memory_path = train / 'memory-profile.jsonl'
        with memory_path.open() as stream:
            for _ in range(10001):
                line = stream.readline(1024 * 1024 + 1)
                if len(line) > 1024 * 1024:
                    raise ValueError('Oversized memory snapshot')
                if not line or not line.endswith('\n'):
                    break
                data = json.loads(line)
                groups = [g for g in data.get('cgroups', [])
                          if re.search(r'/job_\d+$', g['path'])]
                output['memory'].append(dict(time_unix_s=data['time_unix_s'],
                    elapsed_s=data['elapsed_s'], phase=data.get('phase'), cgroups=groups,
                    RSS_sum=sum(p.get('rss_bytes') or 0 for p in data.get('processes', [])),
                    PSS_sum=sum(p.get('pss_bytes') or 0 for p in data.get('processes', [])),
                    USS_sum=sum(p.get('uss_bytes') or 0 for p in data.get('processes', []))))
            else:
                raise ValueError('Memory history cap exceeded')
        monitor = sorted(evidence.glob('monitor-' + arm + '-*.json'))[-1]
        if monitor.stat().st_size > 4 * 1024**2:
            raise ValueError('Oversized monitor')
        data = json.loads(monitor.read_text())
        output['monitor'] = {k:data.get(k) for k in ('observed_utc', 'squeue', 'sacct',
            'gates', 'latest_topology', 'memory_summary', 'wandb')}
        output['monitor_file'] = monitor.name
        result['arms'][arm] = output
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    args = parser.parse_args()
    result = collect(source_root=args.source_root)
    encoded = json.dumps(result, separators=(',', ':'), allow_nan=False).encode()
    if len(encoded) > 64 * 1024**2:
        raise ValueError('Compact projection exceeds 64MiB cap')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(peak if sys.platform == 'darwin' else peak * 1024)
    result['reader_peak_RSS_bytes'] = peak
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = EVIDENCE / ('timeseries-' + stamp + '.json.gz')
    with path.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as out:
            out.write(json.dumps(result, separators=(',', ':'), allow_nan=False).encode())
    print(json.dumps(dict(path=str(path), bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), reader_peak_RSS_bytes=peak,
        counts={a:dict(updates=len(v['updates']), missions=sum(len(u['episodes'])
            for u in v['updates']), memory=len(v['memory'])) for a,v in result['arms'].items()}), indent=2))


if __name__ == '__main__':
    main()
