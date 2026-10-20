"""Plot measured fixture outcomes and explicitly diagnostic credit contributions."""
import argparse
import json
from pathlib import Path


def plot(source, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with source.open() as stream:
        review=json.load(stream)
    assert review['transport_privacy_and_target_checks_passed']
    for suffix in ('.png','.pdf'):
        if destination.with_suffix(suffix).exists():
            raise FileExistsError(destination.with_suffix(suffix))
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(9,7),layout='constrained')
    comparisons=review['comparisons']
    keys=('blocked','delivered','no_comm_image','no_comm_downlink')
    names=('Lost packet','Delivered packet','No communication:\nimage instead','No communication:\ndownlink instead')
    values=[comparisons[k]['constellation_reward'] for k in keys]
    colors=('#777777','#cc4778','#21918c','#73d055')
    axes[0].bar(names,values,color=colors)
    axes[0].set_ylabel('Actual constellation API return')
    axes[0].set_ylim(0,max(values)*1.2)
    axes[0].set_title('Information helps the receiver; the imaging alternative is worth more',fontsize=12)
    for i,v in enumerate(values):
        axes[0].text(i,v+.015,f'{v:.4f}',ha='center')
    owned=review['attribution']['owned']['sender_transmit_target_delta']
    shared=review['attribution']['team_mean']['sender_transmit_target_delta']
    image=next(r['diagnostic_value_target'] for r in review['credit']['no_comm_image']['sensor_0']['rows']
               if r['start_s']==review['attribution']['owned']['transmit_start_s'])
    axes[1].bar(('Transmit benefit:\nowned','Transmit benefit:\nteam_mean','Foregone image:\nowned'),
                (owned,shared,image),color=('#888888','#cc4778','#21918c'))
    for i,v in enumerate((owned,shared,image)):
        axes[1].text(i,v+.008,f'{v:.5f}',ha='center')
    axes[1].set_ylim(0,image*1.25)
    axes[1].set_ylabel('Zero-baseline target contribution')
    axes[1].set_title('Owned gives no transmit credit; shared credit is small after delay',fontsize=12)
    axes[1].text(.03,.86,'Shared transmit benefit: 0.1971 / 4 × γ²⁸⁹ × λ = 0.00365',
                 transform=axes[1].transAxes,fontsize=10)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Constructed native fixture, seed 401 — no learned-policy comparison',fontsize=13)
    fig.supxlabel('2400-second diagnostic window; acquisition rewards only, zero ground deliveries.\n'
                  'Zero-baseline targets isolate reward contribution; no trained critic or convergence claim.',fontsize=9)
    for suffix in ('.png','.pdf'):
        fig.savefig(destination.with_suffix(suffix),dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh output filename stem')
    args=parser.parse_args()
    plot(args.input,args.output)
