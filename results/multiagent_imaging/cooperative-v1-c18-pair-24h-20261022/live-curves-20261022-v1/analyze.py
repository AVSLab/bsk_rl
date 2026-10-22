"""Local plots of completed training batches, not frozen-policy evaluation."""
import argparse
from bisect import bisect_right
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def episode(e):
    assert math.isclose(e['simulated_seconds'],45000,abs_tol=1e-6)
    task=e['cooperative_tasking']; counts=e['action_selection_counts']
    decisions=sum(e['policy_decisions'].values())
    assert decisions==sum(sum(v.values()) for v in counts.values())
    tx={s:sum(n for a,n in v.items() if a.startswith('transmit_completions_')) for s,v in counts.items()}
    qualified=e['physical_completed_captures']-task['rejected_acquisitions']
    duplicate=qualified-task['unique_acquisition_requests']; assert duplicate>=0
    science=.9*task['acquisition_value']+.1*task['verified_delivery_value']
    assert math.isclose(science,task['unshaped_weighted_science'],abs_tol=1e-8)
    comm=e['communication_summary']; agents=e['mission_metrics']['agents']
    return dict(seed=e['episode_seed'],id=e['episode_id'],initial_hash=e['initial_conditions_sha256'],
        science=science, acquisition_value=task['acquisition_value'],delivery_value=task['verified_delivery_value'],
        acquisitions=task['unique_acquisition_requests'],deliveries=task['useful_delivered_requests'],
        completed_duplicates=duplicate,qualified_captures=qualified,
        duplicate_attempts=e['team_summary']['duplicate_attempt_count'],
        decisions=decisions, tx=sum(tx.values()), tx_by_agent=tx,
        decisions_by_agent=e['policy_decisions'],tx_seconds=comm['radio_occupancy_s'],
        radio_on_seconds=sum(a['radio_on_s'] for a in agents.values()),
        packets=comm['packet_outcome_counts'],payload_bytes=comm['payload_bytes_attempted'],
        payload_records=comm['payload_records_attempted'],
        available_sensor_seconds=sum(a['live_time_s'] for a in agents.values()))


def aggregate(rows):
    def total(k): return sum(r[k] for r in rows)
    n=len(rows);decisions=total('decisions');qualified=total('qualified_captures')
    result={k:total(k)/n for k in ('science','acquisition_value','delivery_value','acquisitions',
        'deliveries','completed_duplicates','qualified_captures','duplicate_attempts','tx',
        'tx_seconds','radio_on_seconds','payload_bytes','payload_records')}
    result.update(missions=n,decisions=decisions,tx_total=total('tx'),
        tx_percent=100*total('tx')/decisions,
        duplicate_percent=100*total('completed_duplicates')/qualified if qualified else None,
        tx_occupancy_percent=100*total('tx_seconds')/total('available_sensor_seconds'),
        radio_on_percent=100*total('radio_on_seconds')/total('available_sensor_seconds'),
        packets_per_mission={k:sum(r['packets'].get(k,0) for r in rows)/n
            for k in sorted(set().union(*(r['packets'] for r in rows)))},
        tx_percent_by_agent={s:100*sum(r['tx_by_agent'][s] for r in rows)/sum(r['decisions_by_agent'][s] for r in rows)
            for s in rows[0]['tx_by_agent']})
    return result


def trailing(curve,key,n=3):
    return [aggregate([e for r in curve[max(0,i-n+1):i+1] for e in r['episodes']])[key]
            for i in range(len(curve))]


def main(args):
    archive=Path(args.archive);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive.stat().st_size==args.expected_bytes and digest==args.sha256
    with gzip.open(archive,'rt') as f:data=json.load(f)
    curves={}; all_rows={}; result=dict(snapshot_utc=data['observed_utc'],archive_sha256=digest,
        source='Completed immutable update journals; active training, not held-out',
        x_axis='Cumulative actual agent decisions in completed PPO batches; no epoch multiplication',
        smoothing='Bold curves pool trailing three full 28-mission batches; pale curves are individual batches',arms={})
    for arm,content in data['arms'].items():
        curve=[];by_seed={}
        for u in content['updates']:
            assert u['actual_policy_decisions']==u['condensed_learner_rows']
            assert all(math.isfinite(v) for v in u['losses'].values())
            rows=[episode(e) for e in u['episodes']]
            assert len(rows)==u['complete_episode_count']
            assert sum(e['decisions'] for e in rows)==u['actual_policy_decisions']
            for row in rows:
                assert row['seed'] not in by_seed
                by_seed[row['seed']]=dict(row,update=u['iteration'])
            curve.append(dict(update=u['iteration'],steps=u['policy_decisions_lifetime'],
                published=u['journal_published_unix_s'],episodes=rows,metrics=aggregate(rows),
                losses=u['losses'],gradient_l2=u['gradient_l2']))
        curves[arm]=curve;all_rows[arm]=by_seed
        result['arms'][arm]=dict(updates=len(curve),steps=curve[-1]['steps'],missions=len(by_seed),
            first_3=aggregate([e for r in curve[:3] for e in r['episodes']]),
            latest_3=aggregate([e for r in curve[-3:] for e in r['episodes']]),
            latest_5=aggregate([e for r in curve[-5:] for e in r['episodes']]),
            all=aggregate(list(by_seed.values())),
            curves=[{k:v for k,v in r.items() if k!='episodes'} for r in curve])
    common=sorted(set.intersection(*(set(v) for v in all_rows.values())))
    assert all(all_rows['with_communication'][s]['initial_hash']==all_rows['without_communication'][s]['initial_hash'] for s in common)
    for name,seeds in [('all',common),('recent_5_with_batches',[s for s in common if all_rows['with_communication'][s]['update']>len(curves['with_communication'])-5])]:
        paired={a:aggregate([rows[s] for s in seeds]) for a,rows in all_rows.items()}
        a,b=paired['with_communication'],paired['without_communication']
        paired['n']=len(seeds);paired['with_percent_change']={k:100*(a[k]/b[k]-1) if b[k] else None
            for k in ('science','acquisitions','deliveries','completed_duplicates')}
        result['matched_'+name]=paired
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    styles={'with_communication':('#2768a7','Communication'),'without_communication':('#cf752b','Control')}
    def setup(ax,ylabel):
        ax.set(xlabel='True training decision steps',ylabel=ylabel);ax.grid(alpha=.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:,.0f}k'))
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=180)
        fig.savefig(out/(name+'.svg'));plt.close(fig)
        path=out/(name+'.svg');path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,6.6),constrained_layout=True)
    for ax,key,label in zip(axes.flat,('science','completed_duplicates','acquisitions','deliveries'),
        ('Unshaped request science / mission','Completed redundant\ncaptures / mission',
         'Unique useful acquisitions / mission','Verified useful deliveries / mission')):
        for arm,c in curves.items():
            color,name=styles[arm];x=[r['steps'] for r in c]
            ax.plot(x,[r['metrics'][key] for r in c],color=color,alpha=.22,lw=1)
            ax.plot(x,trailing(c,key),color=color,lw=2,label=name)
        setup(ax,label)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Training outcomes through the latest completed update; bold = trailing 3 batches')
    save(fig,'mission-vs-steps')
    fig,axes=plt.subplots(2,2,figsize=(11,6.5),constrained_layout=True)
    comm=curves['with_communication'];steps=[r['steps'] for r in comm]
    for ax,key,label in zip((axes[0,0],axes[0,1],axes[1,1]),
        ('tx_percent','tx','tx_occupancy_percent'),
        ('Transmit choices / true decisions (%)','Transmit choices / full mission',
         'Transmit task occupancy\n(% of sensor time)')):
        ax.plot(steps,[r['metrics'][key] for r in comm],color='#2768a7',alpha=.25)
        ax.plot(steps,trailing(comm,key),color='#2768a7',lw=2,label='Communication')
        if key!='tx_occupancy_percent':ax.axhline(0,color='#cf752b',ls='--',lw=1,label='Control: TX masked')
        setup(ax,label)
    for outcome,color in (('accepted','#27825b'),('lost','#bc4e45'),('expired','#a37622'),('receiver_unavailable','#8559a4')):
        values=[r['metrics']['packets_per_mission'].get(outcome,0) for r in comm]
        if any(values) or outcome=='accepted':
            axes[1,0].plot(steps,values,color=color,label=outcome)
    setup(axes[1,0],'Packet outcomes / full mission');axes[1,0].legend(frameon=False)
    axes[0,0].legend(frameon=False)
    fig.suptitle('Communication choices, completed packets and physical opportunity costs')
    save(fig,'comms-vs-steps')
    fig,axes=plt.subplots(1,2,figsize=(11,3.4),constrained_layout=True)
    for arm,c in curves.items():
        points=data['arms'][arm]['memory'];times=[r['time_unix_s'] for r in points]
        selected=[points[max(0,bisect_right(times,r['published'])-1)] for r in c]
        color,name=styles[arm]
        for ax,key in zip(axes,('memory.usage_in_bytes','memory.max_usage_in_bytes')):
            values=[max(int(g['values'][key]) for g in p['cgroups'])/2**30 for p in selected]
            ax.plot([r['steps'] for r in c],values,color=color,lw=2,label=name)
            setup(ax,'GiB')
        latest=points[-1]
        result['arms'][arm]['memory']=dict(elapsed_hours=latest['elapsed_s']/3600,
            current_gib=max(int(g['values']['memory.usage_in_bytes']) for g in latest['cgroups'])/2**30,
            peak_gib=max(int(g['values']['memory.max_usage_in_bytes']) for g in latest['cgroups'])/2**30,
            alignment='Last memory sample at or before each update journal publication',
            max_alignment_gap_s=max(r['published']-p['time_unix_s'] for r,p in zip(c,selected)))
        assert all(int(g['values']['memory.failcnt'])==0 for p in points for g in p['cgroups'])
    for ax in axes: ax.set_ylim(bottom=0)
    axes[0].set_title('Job cgroup current memory');axes[1].set_title('Job cgroup peak memory')
    axes[0].legend(frameon=False);fig.suptitle('RAM sampled at update publication; 200 GiB allocation per arm')
    save(fig,'ram-vs-steps')
    result['limitations']=['One training seed per arm','Changing training scenarios, not frozen evaluation',
        'Control has more completed experience; no unmatched right-tail advantage claim',
        'Counts reconstructed from compact episode ledgers; full raw terminal numerical audit pending',
        'TX selection is not successful packet delivery or causal mission benefit']
    (out/'statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({a:{k:v for k,v in r.items() if k in ('updates','steps','missions','first_3','latest_3','memory')} for a,r in result['arms'].items()},indent=2))
    print(json.dumps({k:{p:v for p,v in r.items() if p in ('n','with_percent_change')} for k,r in result.items() if k.startswith('matched_')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--expected-bytes',type=int,required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
