"""Static scientific templates or real saved-study metrics; never invent observations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARMS=('catalog_tx','geometry_only')
ARM_LABELS={'catalog_tx':'With communication','geometry_only':'Without communication'}
PALETTES={'catalog_tx':'plasma','geometry_only':'viridis'}
MARKERS=('o','s','^','D')
SHADES=(.22,.42,.62,.82)

def styles():
    return {arm:{f'sensor_{i}':dict(color=matplotlib.colors.to_hex(plt.get_cmap(PALETTES[arm])(SHADES[i])),
        marker=MARKERS[i],linestyle='-' if arm=='catalog_tx' else '--') for i in range(4)} for arm in ARMS}

def legend(ax):
    sty=styles();handles=[Line2D([],[],label=f'{ARM_LABELS[arm]} · spacecraft {i+1}',
        color=sty[arm][f'sensor_{i}']['color'],marker=MARKERS[i],linestyle=sty[arm][f'sensor_{i}']['linestyle'],markersize=5) for arm in ARMS for i in range(4)]
    ax.legend(handles=handles,ncol=2,loc='upper center',frameon=False,fontsize=8)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=Path,nargs='*',default=[])
    parser.add_argument('--template',action='store_true')
    args=parser.parse_args()
    if args.template==bool(args.episodes):parser.error('Select templates OR saved episode files')
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    rows=[];inputs={}
    for path in args.episodes:
        raw=json.loads(path.read_text());inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        arm=raw['config']['information_case']
        if arm not in ARMS or raw['config']['n_targets']!=200 or raw['config']['reimage_cooldown_orbits']!=1:
            raise ValueError('Historical or incompatible episode passed to new-study plots')
        mode=raw.get('selection_mode',raw.get('controller','training'))
        if rows and mode!=rows[0]['mode']:raise ValueError('Plot argmax and stochastic separately')
        metrics=raw['mission_metrics']
        for sensor,agent in metrics['agents'].items():
            rows.append(dict(arm=arm,sensor=sensor,mode=mode,seed=raw['seed'],images=agent['actual_images'],
                deliveries=agent['useful_delivered_products'],reward=agent['reward'],redundant=agent['redundant_imaging_execution_s'],
                delay=agent['useful_delivery_delay_median_s'],coverage=agent['ground_coverage_fraction']))
    combos=[('images_vs_delivered','Images acquired per sensor','Products with positive ground credit per sensor','images','deliveries'),
        ('latency_vs_coverage','Median useful delivery delay per sensor [min]','Distinct-target ground coverage per sensor [%]','delay','coverage'),
        ('redundancy_vs_reward','Redundant imaging execution per sensor [min]','Sensor episode reward','redundant','reward')]
    sty=styles()
    artifacts=[]
    for name,xlabel,ylabel,xkey,ykey in combos:
        fig,ax=plt.subplots(figsize=(10,6.6));legend(ax)
        for row in rows:
            x,y=row[xkey],row[ykey]
            if x is None or y is None:continue
            if xkey in ('delay','redundant'):x/=60
            if ykey=='coverage':y*=100
            opts=sty[row['arm']][row['sensor']]
            ax.scatter(x,y,color=opts['color'],marker=opts['marker'],alpha=.45,s=38,edgecolors='none')
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.18)
        ax.set_title('Four spacecraft · two Walker-Delta planes · 200 mixed targets\n700 km / 97° · Walker phasing 1 · one-orbit cooldown',pad=16)
        if args.template:
            ax.set_xlim(0,1);ax.set_ylim(0,1)
            ax.text(.5,.48,'PLOT TEMPLATE\nNo episode records loaded',transform=ax.transAxes,
                ha='center',va='center',fontsize=16,color='#555555')
            ax.set_xticks([]);ax.set_yticks([])
        else:
            ax.margins(y=.35)
            fig.text(.5,.01,f'{rows[0]["mode"]} · each point is one saved episode/sensor; sensor points within a seed are correlated',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,1))
        for extension in ('png','pdf'):
            path=args.output/f'{name}.{extension}';fig.savefig(path);artifacts.append(str(path))
        plt.close(fig)
    (args.output/'plot-manifest.json').write_text(json.dumps(dict(template=args.template,
        metric_version='physical-mission-v1',input_sha256=inputs,styles=sty,arm_labels=ARM_LABELS,artifacts=artifacts,
        plotted_rows=rows, unique_seed_count=len({row['seed'] for row in rows}),
        note='No artificial replicated constellation points; compare stochastic and argmax separately.'),indent=2)+'\n')
    print(json.dumps(dict(template=args.template,plots=3,simulated_episodes=0)))
if __name__=='__main__':main()
