"""One occupancy regime per figure; individual instances, no pooled VCG seeds."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .run import OUTPUT

METHODS=('dynamic_pslap','ga','certified_dynamic','original_seed0_ep60',
         'original_seed1_ep60','original_seed2_ep60')
LABELS=('Dynamic\nPSLAP','GA','Certified\nDynamic','VCG\nseed 0','VCG\nseed 1','VCG\nseed 2')
COLORS=('#64748b','#a78bfa','#d97706','#0369a1','#0891b2','#0f766e')


def values(row):
    m=row['metrics'];c=m['components']
    return dict(J=m['J'],mae=m['release_mae'],within=100*m['within_20'],
        wait=m['mean_observed_admission_wait'],
        rehandles=100*c['rehandles']/m['delivered'],
        steps=row['behavior']['steps_per_delivery'],
        latency=1000*float(np.mean(row['rollout']['latencies'])))


def plot(directory,fill):
    contract=json.loads((directory/'contract.json').read_text())
    cases=[c for c in contract['cases'] if c['fill']==fill]
    data={}
    rows={}
    for method in METHODS:
        selected=[]
        for case in cases:
            path=directory/f'rows/{method}/cell-{case["index"]}.json'
            if not path.exists():
                raise ValueError(f'Regime not finished: missing {path}')
            row=json.loads(path.read_text())
            if row['contract_sha256']!=contract['contract_sha256']:
                raise ValueError('Row contract mismatch')
            selected.append(row)
        rows[method]=selected
        data[method]=[values(r) if r['status']=='complete' and r['metrics']['cost_ranking_eligible'] else None
                      for r in selected]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold'})
    fig,axes=plt.subplots(2,4,figsize=(19,10))
    specs=[('J','Total objective J ↓','Cost units'),
           ('mae','Release-clock MAE ↓','Primitive steps'),
           ('within','Deliveries within ±20 ↑','Percent'),
           ('wait','Wait before admission ↓','Primitive steps / incoming job'),
           ('rehandles','Rehandles per 100 deliveries ↓','Rehandles / 100 deliveries'),
           ('steps','Steps per delivery ↓','Primitive steps / delivery'),
           ('latency','Mean decision latency ↓','Milliseconds (log scale)')]
    offsets=np.linspace(-.18,.18,len(cases))
    markers=('o','s','^')
    exported={}
    for ax,(key,title,ylabel) in zip(axes.flat,specs):
        means=[]
        for i,method in enumerate(METHODS):
            numbers=[d[key] for d in data[method] if d is not None]
            mean=float(np.mean(numbers)) if numbers else None
            means.append(mean)
            for j,d in enumerate(data[method]):
                if d is not None:
                    ax.scatter(i+offsets[j],d[key],marker=markers[j],s=34,
                        color=COLORS[i],alpha=.6,edgecolor='white',linewidth=.5,zorder=3)
            if mean is not None:
                ax.plot([i-.27,i+.27],[mean,mean],color=COLORS[i],lw=2.4,zorder=4)
                ax.scatter(i,mean,marker='D',s=42,color=COLORS[i],edgecolor='white',lw=.6,zorder=5)
                text=f'{mean:.1f}' if key!='rehandles' else f'{mean:.2g}'
                ax.annotate(text,(i,mean),xytext=(0,10),textcoords='offset points',
                    ha='center',fontsize=9,color=COLORS[i],fontweight='bold')
        ax.set(title=title,ylabel=ylabel,xticks=range(6),xticklabels=LABELS,xlim=(-.6,5.6))
        ax.grid(axis='y',color='#e2e8f0',linewidth=.7,zorder=0)
        ax.tick_params(axis='x',length=0,labelsize=9)
        if key=='latency':
            ax.set_yscale('log')
            ax.margins(y=.35)
        elif key=='within':
            ax.set_ylim(0,112)
            ax.set_yticks([0,25,50,75,100])
        else:
            top=max((d[key] for v in data.values() for d in v if d is not None),default=1.)
            ax.set_ylim(-.04*max(top,1.),1.23*max(top,1.))
        exported[key]={m:dict(instances=[d[key] if d else None for d in data[m]],mean=means[i])
                       for i,m in enumerate(METHODS)}
    ax=axes.flat[-1]
    ax.axis('off')
    completion=[sum(r['status']=='complete' for r in rows[m]) for m in METHODS]
    ax.text(0.,.98,'Completion and reading guide',fontsize=12,fontweight='bold',va='top')
    text='\n'.join(f'{LABELS[i].replace(chr(10)," ")}: {n}/{len(cases)} episodes complete'
                   for i,n in enumerate(completion))
    ax.text(0.,.86,text,va='top',fontsize=11,linespacing=1.65)
    ax.text(0.,.39,'Symbols: individual paired instances\nDiamond + line: arithmetic mean\n\nVCG seeds are shown separately.\nNo confidence intervals with n = 3.\nLatency: CPU, one thread.',va='top',fontsize=10,linespacing=1.5,color='#475569')
    fig.suptitle(f'E145 · Initial occupancy {fill:.0%} · Original 8×8 yard',
                 x=.04,y=.98,ha='left',fontsize=19,fontweight='bold')
    subtitle=('Training-distribution control, fresh instances' if fill==0 else 'Frozen original residual VCG: initial-inventory transfer')
    fig.text(.04,.936,f'{subtitle}  |  3 paired instances · 36 incoming jobs · gap 30 · Poisson-80 · incoming due = release + duration',fontsize=12,color='#475569')
    method_handles=[Line2D([0],[0],color=color,marker='D',linewidth=2,
                          markersize=6,label=label.replace('\n',' '))
                    for label,color in zip(LABELS,COLORS)]
    fig.legend(handles=method_handles,title='Color = controller',loc='upper center',
               bbox_to_anchor=(.5,.913),ncol=6,frameon=False,columnspacing=2.2)
    handles=[Line2D([0],[0],marker=marker,color='none',markerfacecolor='#64748b',
                   markeredgecolor='white',markersize=7,label=f'Instance {i+1}') for i,marker in enumerate(markers)]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.035,.045),ncol=3,frameon=False)
    fig.text(.04,.028,'J = 2T + E + 0.1L + 10R. Admission wait is reported separately, not included in J. Costs and service metrics use completed episodes only.',fontsize=10,color='#475569')
    fig.text(.04,.009,'Descriptive development pilot; not an independent test-panel result. Plain versus certified macro-library differences remain a limitation.',fontsize=9,color='#64748b')
    fig.tight_layout(rect=(.025,.075,.995,.86),h_pad=3,w_pad=2)
    folder=directory/'plots';folder.mkdir(exist_ok=True)
    stem=folder/('occupancy-'+f'{fill:g}'.replace('.','p'))
    for suffix in ('png','svg','pdf'):
        fig.savefig(stem.with_suffix('.'+suffix),dpi=180,facecolor='white')
    plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps(dict(fill=fill,cases=cases,
        contract_sha256=contract['contract_sha256'],metrics=exported,
        completion=dict(zip(METHODS,completion))),indent=2)+'\n')
    return stem.with_suffix('.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fill',type=float,choices=(0.,.4,.7,.9),default=0.)
    parser.add_argument('--directory',type=Path,default=OUTPUT)
    args=parser.parse_args()
    print(plot(args.directory,args.fill))


if __name__=='__main__':
    main()
