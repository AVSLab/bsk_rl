"""Local reconstruction and plots from the Termius-downloaded live projection."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'download/comparison-20261022T052235301178Z.json.gz'
SHA = '6034e6d15a6ce4dfa01f593f6edfdf0008a59642db23692c56c9fef73529107a'
EXPECTED_BYTES = 3032702


def mission(e):
    assert math.isclose(e['simulated_seconds'], 45000, abs_tol=1e-6)
    task = e['cooperative_tasking']
    active, captures = {}, []
    for p in e['capture_products']:
        index = int(max(0, p['capture_time']) // task['window_s'])
        if index not in active:
            rng = np.random.default_rng(np.random.SeedSequence([e['episode_seed'], index, 731]))
            active[index] = set(map(int, rng.choice(200, task['active_request_count'], replace=False)))
        if (p['target_id'] in active[index] and p['request_epoch'] == index * task['window_s']
            and p['quality'] >= task['quality_threshold']
            and p['completion_time'] <= (index+1) * task['window_s']):
            captures.append(p)
    assert len(e['capture_products']) - len(captures) == task['rejected_acquisitions']
    last, unique = {}, 0
    for p in sorted(captures, key=lambda p:(p['capture_time'], p['record_id'])):
        key = p['target_id'], p['request_epoch']
        if key not in last or p['capture_time'] >= last[key] + e['reimage_cooldown_s']:
            unique += 1
            last[key] = p['capture_time']
    assert unique == task['unique_acquisition_requests']
    science = .9 * task['acquisition_value'] + .1 * task['verified_delivery_value']
    assert math.isclose(science, task['unshaped_weighted_science'], abs_tol=1e-9)
    return dict(seed=e['episode_seed'], episode_id=e['episode_id'], initial_hash=e['initial_conditions_sha256'],
        science=science, acquisition_value=task['acquisition_value'], delivery_value=task['verified_delivery_value'],
        acquisitions=unique, deliveries=task['useful_delivered_requests'],
        physical_captures=len(e['capture_products']), qualified_captures=len(captures),
        completed_duplicates=len(captures)-unique,
        duplicate_fraction=(len(captures)-unique)/len(captures) if captures else None,
        duplicate_attempts=e['team_summary']['duplicate_attempt_count'],
        shaped_reward=e['constellation_reward'],
        duplicate_cost=task['redundant_capture_cost'],
        tx_seconds=e['communication_summary']['radio_occupancy_s'],
        decisions=sum(e['policy_decisions'].values()))


def means(rows):
    return {k: statistics.mean(r[k] for r in rows if r[k] is not None)
            for k in rows[0] if k not in ('seed','episode_id','initial_hash')}


def main():
    actual = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert ARCHIVE.stat().st_size == EXPECTED_BYTES and actual == SHA
    with gzip.open(ARCHIVE, 'rt') as stream:
        data = json.load(stream)
    arms = data['arms']
    result = dict(snapshot_utc=data['observed_utc'], archive_sha256=actual,
        reader_peak_RSS_bytes=data['reader_peak_RSS_bytes'], input_manifest_files=len(data['manifest']),
        new_PPO_updates=0, terminal_audit=False, heldout=False, arms={}, matched={})
    all_rows, curves, completed_seeds = {}, {}, {}
    for arm, content in arms.items():
        rows = [mission(e) for e in content['episodes']]
        assert len({r['seed'] for r in rows}) == len(rows)
        all_rows[arm] = {r['seed']:r for r in rows}
        by_id = {r['episode_id']:r for r in rows}
        curve, committed = [], set()
        for update in content['updates']:
            assert update['condensed_learner_rows'] == update['actual_policy_decisions']
            assert all(math.isfinite(v) for v in update['losses'].values())
            selected = [by_id[e['episode_id']] for e in update['episodes'] if e['episode_id'] in by_id]
            if len(selected) != update['complete_episode_count']:
                continue  # Snapshot may precede a newly published update's spool.
            committed.update(r['seed'] for r in selected)
            curve.append(dict(update=update['iteration'], policy_generation=update['iteration']-1,
                prior_decisions=update['policy_decisions_lifetime']-update['actual_policy_decisions'],
                decisions=update['policy_decisions_lifetime'], missions=len(selected),
                means=means(selected), losses=update['losses'], gradient_l2=update['gradient_l2']))
        curves[arm] = curve
        completed_seeds[arm] = committed
        result['arms'][arm] = dict(missions=len(rows), updates=len(content['updates']),
                                   means=means(rows), curves=curve)
    common = sorted(set.intersection(*(set(r) for r in all_rows.values())))
    for seed in common:
        assert all_rows['with_communication'][seed]['initial_hash'] == all_rows['without_communication'][seed]['initial_hash']
    def paired(seeds):
        comparison = {'n':len(seeds)}
        for arm, rows in all_rows.items():
            comparison[arm] = means([rows[s] for s in seeds])
        a, b = [comparison[arm] for arm in ('with_communication','without_communication')]
        comparison['with_minus_without'] = {k:a[k]-b[k] for k in a}
        comparison['with_percent_change'] = {k:100*(a[k]/b[k]-1) if b[k] else None for k in a}
        comparison['science_positive_pairs'] = sum(all_rows['with_communication'][s]['science'] > all_rows['without_communication'][s]['science'] for s in seeds)
        comparison['rows'] = [dict(seed=s, with_communication=all_rows['with_communication'][s],
                                  without_communication=all_rows['without_communication'][s]) for s in seeds]
        return comparison
    result['matched_all_spooled'] = paired(common)
    committed_common = sorted(set(common).intersection(*completed_seeds.values()))
    result['matched'] = paired(committed_common)
    result['matched']['selection'] = 'Matched seeds included in completed update journals in both arms'
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    styles = {'with_communication':('#2768a7','o','Communication'), 'without_communication':('#cf752b','s','Control')}
    fig, axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax, metric, label in zip(axes.flat,
        ('science','completed_duplicates','duplicate_fraction','deliveries'),
        ('Unshaped request science / mission','Completed redundant captures / mission',
         'Redundant / qualified completed captures','Useful verified deliveries / mission')):
        for arm, curve in curves.items():
            color, marker, name = styles[arm]
            ax.plot([r['prior_decisions']/1000 for r in curve],[r['means'][metric] for r in curve],
                    marker=marker,color=color,label=name)
        ax.set(xlabel='Training decisions before sampled batch (thousands)',ylabel=label)
        ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Early training batches, 28 missions each — not held-out evaluation')
    fig.savefig(ROOT/'mission-curves.png',dpi=180); fig.savefig(ROOT/'mission-curves.svg');plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
    for ax, metric in zip(axes,('policy_loss','vf_loss')):
        for arm, curve in curves.items():
            color, marker, name=styles[arm]
            ax.plot([r['decisions']/1000 for r in curve],[r['losses'][metric] for r in curve],marker=marker,color=color,label=name)
        ax.set(xlabel='Decisions in completed PPO updates (thousands)',ylabel=metric.replace('_',' '));ax.grid(alpha=.2)
    axes[0].legend(frameon=False);fig.suptitle('Optimization diagnostics — lower loss does not establish mission improvement')
    fig.savefig(ROOT/'ppo-losses.png',dpi=180);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,3.7),constrained_layout=True)
    for arm, content in arms.items():
        color, marker, name=styles[arm]
        points=[]
        for row in content['memory']:
            groups=row['cgroups']
            if not groups: continue
            v=groups[0]['values']; t=row['time_fields']['elapsed_s']/60
            points.append((t,int(v['memory.usage_in_bytes'])/1024**3,int(v['memory.max_usage_in_bytes'])/1024**3,row['PSS_sum']/1024**3,row['USS_sum']/1024**3))
        assert all(int(g['values']['memory.failcnt'])==0 for row in content['memory'] for g in row['cgroups'])
        result['arms'][arm]['memory'] = dict(points=len(points),elapsed_minutes=points[-1][0],
                    current_gib=points[-1][1],peak_gib=max(p[2] for p in points),PSS_gib=points[-1][3],USS_gib=points[-1][4])
        axes[0].plot([p[0] for p in points],[p[1] for p in points],color=color,label=name)
        axes[1].plot([p[0] for p in points],[p[2] for p in points],color=color,label=name)
    for ax,title in zip(axes,('Job cgroup current memory','Job cgroup peak memory')):
        ax.set(xlabel='Training driver elapsed minutes',ylabel='GiB',title=title);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.suptitle('RAM trend; each job has 200 GiB, final acceptance ≤160 GiB including restore')
    fig.savefig(ROOT/'ram-trend.png',dpi=180);fig.savefig(ROOT/'ram-trend.svg');plt.close(fig)
    result['limitations']=['Active-run snapshot, not terminal acceptance','One training seed per treatment',
        'Matched training missions are not frozen-policy held-out interventions',
        'Later policy generations and processed mission counts differ between arms',
        'Projection transfer verified; complete raw terminal download and per-file verification still pending']
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result['matched'].items() if k!='rows'},indent=2))
    print(json.dumps({a:r['memory'] for a,r in result['arms'].items()},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--expected-bytes',type=int)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.archive is not None:
        if args.sha256 is None or args.expected_bytes is None:
            parser.error('Fresh snapshots require a verified size and SHA256')
        ARCHIVE=args.archive; SHA=args.sha256;EXPECTED_BYTES=args.expected_bytes
    if args.output is not None:
        ROOT=args.output;ROOT.mkdir(parents=True,exist_ok=True)
    main()
