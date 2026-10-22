"""Read-only, bounded projection of an active pair; never import Ray or Basilisk.

Run through Termius. Large observation/decision arrays are scanned, not decoded.
Raw evidence is untouched. Numerical reconstruction belongs in the local audit.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import runpy
import resource


BASE = Path('/scratch/alpine/dahu1128/cooperative-v1-c18-pair-24h-20261022')
EVIDENCE = Path('/projects/dahu1128/cooperative-v1-c18-pair-24h-evidence-20261022')
SOURCE = Path('/projects/dahu1128/bsk_rl-cooperative-v1-c18-pair-b205fc16-20261022')
TOKENS = re.compile(r'["\\{}\[\]]')


def skip_value(reader):
    """Skip a container in fixed chunks, respecting escaped string delimiters."""
    if reader.peek() not in ('{', '['):
        reader.value()
        return
    depth, quoted, skip_first = 0, False, False
    while True:
        start = reader.position
        if skip_first:
            start += 1
            skip_first = False
        escaped_until = start
        for match in TOKENS.finditer(reader.buffer, start):
            if match.start() < escaped_until:
                continue
            char = match.group()
            if quoted:
                if char == '\\':
                    escaped_until = match.end() + 1
                    skip_first = escaped_until > len(reader.buffer)
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in '{[':
                depth += 1
            else:
                depth -= 1
                if depth == 0:
                    reader.position = match.end()
                    return
        reader.position = len(reader.buffer)
        if reader.eof:
            raise ValueError('Truncated skipped container')
        reader.fill(65536)


def project(reader, fields, arrays=None):
    arrays = arrays or {}
    result = {}
    for key in reader.keys():
        if key in arrays:
            result[key] = [arrays[key](item) for item in reader.array()]
        elif key in fields:
            result[key] = reader.value()
        else:
            skip_value(reader)
    return result


def project_array_objects(reader, fields):
    reader.expect('[')
    rows = []
    if reader.peek() != ']':
        while True:
            rows.append(project(reader, fields))
            if reader.peek() == ']':
                break
            reader.expect(',')
    reader.expect(']')
    return rows


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect(source_root=SOURCE, base=BASE, evidence=EVIDENCE):
    parser_path = source_root / 'examples/multiagent_imaging/update_evidence.py'
    JSONStream = runpy.run_path(str(parser_path))['JSONStream']
    result = dict(version='active-pair-bounded-projection-v1',
                  observed_utc=datetime.now(timezone.utc).isoformat(),
                  source_root=str(source_root), parser_sha256=digest(parser_path),
                  reader_sha256=digest(Path(__file__)), arms={}, manifest=[])
    episode_fields = {
        'episode_id', 'episode_seed', 'lifetime_episode_id', 'initial_conditions_sha256',
        'simulated_seconds', 'reimage_cooldown_s', 'policy_decisions',
        'constellation_reward', 'reward_by_agent', 'cooperative_tasking',
        'communication_summary', 'team_summary', 'mission_metrics',
        'action_selection_counts', 'coverage', 'reward_decomposition', 'agent_live_until_s',
    }
    update_fields = {'iteration', 'measurement', 'losses', 'gradient_l2',
                     'finite_gradients', 'parameter_change', 'actual_policy_decisions',
                     'complete_episode_count', 'policy_decisions_lifetime',
                     'condensed_learner_rows', 'checkpoint', 'learner_metrics'}
    for arm in ('with_communication', 'without_communication'):
        train = base / arm / 'training'
        episodes = sorted((train / 'episode-spool').glob('*.json'))
        updates = sorted((train / 'update-journal').glob('update-*.json'))
        if len(episodes) > 512 or len(updates) > 32:
            raise ValueError('Explicit review needed to expand live collection caps')
        output = dict(episodes=[], updates=[], memory=[])
        for path in episodes:
            with path.open() as stream:
                reader = JSONStream(stream)
                row = project(reader, episode_fields,
                              {'capture_products': lambda product: {
                                  k: product[k] for k in ('record_id', 'target_id',
                                  'capture_time', 'completion_time', 'request_epoch',
                                  'quality', 'priority') if k in product}})
                reader.finish()
            output['episodes'].append(row)
            result['manifest'].append(dict(path=str(path), size_bytes=path.stat().st_size,
                                           sha256=digest(path)))
        for path in updates:
            with path.open() as stream:
                reader = JSONStream(stream)
                row = {}
                for key in reader.keys():
                    if key == 'episodes':
                        row['episodes'] = project_array_objects(reader, {'episode_id', 'episode_seed'})
                    elif key in update_fields:
                        row[key] = reader.value()
                    else:
                        skip_value(reader)
                reader.finish()
            output['updates'].append(row)
            result['manifest'].append(dict(path=str(path), size_bytes=path.stat().st_size,
                                           sha256=digest(path)))
        memory_path = train / 'memory-profile.jsonl'
        if memory_path.exists():
            with memory_path.open() as stream:
                for _ in range(10001):
                    line = stream.readline(1024 * 1024 + 1)
                    if not line:
                        break
                    if len(line) > 1024 * 1024:
                        raise ValueError('Oversized memory snapshot')
                    if not line.endswith('\n'):
                        break  # Active writer; preserve only complete snapshots.
                    data = json.loads(line)
                    groups = [g for g in data.get('cgroups', [])
                              if re.search(r'/job_\d+$', g['path'])]
                    output['memory'].append(dict(
                        timestamp=data.get('timestamp'), wall_time_s=data.get('wall_time_s'),
                        time_s=data.get('time_s'), phase=data.get('phase'),
                        cgroups=groups,
                        RSS_sum=sum(p.get('rss_bytes') or 0 for p in data.get('processes', [])),
                        PSS_sum=sum(p.get('pss_bytes') or 0 for p in data.get('processes', [])),
                        USS_sum=sum(p.get('uss_bytes') or 0 for p in data.get('processes', [])),
                        time_fields={k: v for k, v in data.items()
                                     if k not in ('processes', 'cgroups') and ('time' in k or k.endswith('_s'))}))
                else:
                    raise ValueError('Memory history cap exceeded')
        monitors = sorted(evidence.glob('monitor-' + arm + '-*.json'))
        if monitors:
            p = monitors[-1]
            if p.stat().st_size > 4 * 1024**2:
                raise ValueError('Oversized monitor metadata')
            data = json.loads(p.read_text())
            output['monitor'] = {k: data.get(k) for k in ('observed_utc', 'squeue', 'sacct', 'gates',
                                                        'latest_topology', 'memory_summary', 'wandb')}
            output['monitor_file'] = p.name
        result['arms'][arm] = output
    result['reader_peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    args = parser.parse_args()
    result = collect(source_root=args.source_root)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = EVIDENCE / ('comparison-' + stamp + '.json.gz')
    with path.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            stream.write(json.dumps(result, allow_nan=False, separators=(',', ':')).encode())
    print(json.dumps(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path),
                          reader_peak_RSS_bytes=result['reader_peak_RSS_bytes'],
                          counts={a: {k: len(v[k]) for k in ('episodes', 'updates', 'memory')}
                                  for a, v in result['arms'].items()}), indent=2))


if __name__ == '__main__':
    main()
