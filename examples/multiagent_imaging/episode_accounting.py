"""Driver evidence reconciliation; never an observation or sampler input."""

import gzip
import hashlib
import json
from pathlib import Path


def content_hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def reconcile_episode_audits(worker_audits, spool, previous_ids=()):
    """Recover completed callbacks absent from a Ray worker RPC reply.

    The append-only spool is authoritative evidence of completed missions, not
    evidence that their rows were learned. Learner rows are checked separately.
    Exclude checkpointed IDs, deduplicate replies and reject conflicting replay.
    """
    previous_ids = set(previous_ids)
    previous_names = {hashlib.sha256(e.encode()).hexdigest() + '.json' for e in previous_ids}
    records, origins = {}, {}

    def add(record, origin):
        identity = record['lifetime_episode_id']
        if identity in previous_ids:
            return
        if identity in records and content_hash(records[identity]) != content_hash(record):
            raise ValueError('Conflicting completed episode evidence: ' + identity)
        records[identity] = record
        origins.setdefault(identity, set()).add(origin)

    for record in worker_audits:
        add(record, 'worker_reply')
    if spool:
        for path in sorted(Path(spool).glob('*.json')):
            if path.name not in previous_names:
                record = json.loads(path.read_text())
                expected = hashlib.sha256(record['lifetime_episode_id'].encode()).hexdigest() + '.json'
                if path.name != expected:
                    raise ValueError('Episode spool identity/filename mismatch: ' + str(path))
                add(record, 'spool')
    ordered = sorted(records.values(), key=lambda r: (r['worker_index'], r['episode_seed'], r['lifetime_episode_id']))
    evidence = dict(version='completed-episode-accounting-v1',
                    worker_reply_records=len(worker_audits),
                    completed_unique_episodes=len(ordered),
                    recovered_from_spool=sum(v == {'spool'} for v in origins.values()),
                    new_sampled_true_decisions=sum(sum(r['policy_decisions'].values()) for r in ordered),
                    episode_content_sha256={e: content_hash(r) for e, r in records.items()},
                    semantics='Completed sampled decisions; condensed learner rows and repeated epoch presentations are separate')
    return ordered, evidence


def verify_retained_learner_batch(directory, config, episodes, learner_rows):
    """Read-only reconstruction of the first actual condensed, unpadded batch."""
    from examples.multiagent_imaging.fixed_batch_diagnostic import independently_recompute
    from bsk_rl import NO_ACTION

    directory = Path(directory)
    batch, audit = independently_recompute(directory, config)
    with gzip.open(directory / 'raw-episodes.json.gz', 'rt') as stream:
        raw = json.load(stream)
    expected = {(e['episode_id'], sensor): count for e in episodes for sensor, count in e['policy_decisions'].items()}
    observed = {}
    for row in raw:
        key = (row['multi_agent_episode_id'], row['agent_id'])
        if key in observed:
            raise ValueError('Repeated raw agent episode: ' + repr(key))
        observed[key] = sum(action != NO_ACTION for action in row['actions'])
    if observed != expected:
        raise ValueError('Raw learner episodes differ from completed sampled agent decisions')
    count = len(batch['actions'])
    if count != sum(expected.values()) or count != learner_rows:
        raise ValueError('Condensed learner/sample/counter totals disagree')
    if audit['forced_no_action_count'] or audit['padded_rows']:
        raise ValueError('Forced continuation or padding reached the learner')
    # Link physical products to the exact owned primitive reward and condensed
    # target. An interval may contain several credits/costs; report its total,
    # never pretend it is the isolated reward of one image.
    import numpy as np
    by_episode = {e['episode_id']: e for e in episodes}
    traces, offset = {}, 0
    for row in raw:
        episode = by_episode[row['multi_agent_episode_id']]
        starts = [i for i, a in enumerate(row['actions']) if a != NO_ACTION]
        times = np.cumsum([i['d_ts'] for i in row.get('infos', [])])
        events = [('acquisition', p, p['completion_time']) for p in episode.get('capture_products', []) if p['quality'] >= .5]
        if getattr(config, 'is_cooperative', False):
            from bsk_rl.data.multiagent_rso_data import ImageProductRecord
            from examples.multiagent_imaging.cooperative import RequestSchedule
            schedule = RequestSchedule(config, episode['episode_seed'])
            events = [(kind, p, when) for kind, p, when in events
                      if schedule.qualifies(ImageProductRecord(**p), 'acquisition')]
        events += [('physical_delivery', e['product'], e['product']['delivery_time']) for e in episode.get('delivered_products', []) if e['credited_value'] > 0]
        for kind, product, time_s in events:
            if kind in traces or product['source_sensor'] != row['agent_id']:
                continue
            matches = [i for i in range(len(row['actions'])) if np.isclose(times[i+1], time_s, rtol=0, atol=1e-5) and row['rewards'][i] > 0]
            if not matches:
                continue
            i = matches[0]
            owned = [e for e in episode['reward_events'] if e['sensor'] == row['agent_id'] and np.isclose(e['time_s'], time_s, rtol=0, atol=1e-5)]
            if len(owned) != 1 or not np.isclose(owned[0]['reward'], row['rewards'][i], rtol=2e-5, atol=1e-5):
                raise ValueError('Physical event and raw owned reward disagree')
            if getattr(config, 'is_cooperative', False) and owned[0].get('components', {}).get('owned_science', 0) <= 0:
                continue
            j = max(j for j, start in enumerate(starts) if start <= i)
            index = offset + j
            traces[kind] = dict(kind=kind, product=product, agent=row['agent_id'], episode_id=row['multi_agent_episode_id'],
                                raw_event_index=i, event_time_s=time_s, owned_interval_reward=row['rewards'][i],
                                condensed_row=index, seconds_from_action_start=float(times[i+1]-times[starts[j]]),
                                endpoint_discount=float(config.discount_per_s**(times[i+1]-times[starts[j]])),
                                condensed_reward=float(batch['rewards'][index]), value_target=float(batch['value_targets'][index]),
                                semantics='Owned interval total can include other credits or costs; product timing is traced, not assigned fabricated isolated credit')
        offset += len(starts)
    if not getattr(config, 'is_cooperative', False) and any('capture_products' in e for e in episodes) and set(traces) != {'acquisition', 'physical_delivery'}:
        raise ValueError('Missing acquisition or physical delivery reward-to-target trace')
    if getattr(config, 'is_cooperative', False):
        # Request-gated acquisitions/deliveries can legitimately be absent in a
        # sampled batch. Never fabricate a delivery trace to satisfy a gate.
        expected_kinds = set()
        if any((e.get('cooperative_tasking') or {}).get('acquisition_value', 0)>0 for e in episodes):
            expected_kinds.add('acquisition')
        if any((e.get('cooperative_tasking') or {}).get('verified_delivery_value', 0)>0 for e in episodes):
            expected_kinds.add('physical_delivery')
        if not expected_kinds.issubset(traces):
            raise ValueError('Missing observed request science reward-to-target trace')
    return dict(audit, matched_mission_episodes=len(episodes),
                physical_product_reward_target_traces=list(traces.values()),
                agent_decisions={sensor: sum(n for (_, s), n in observed.items() if s == sensor) for sensor in sorted({s for _, s in observed})},
                learner_counter_matches=True, sampled_decisions_match=True)
