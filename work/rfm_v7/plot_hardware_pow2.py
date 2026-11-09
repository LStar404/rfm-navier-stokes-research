"""Export median, full range and every sample for the six-way frozen study."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT/'hardware_pow2_output'/'timing.json').read_text(encoding='utf-8'))
    assert data['status']=='complete' and len(data['rows'])==30
    assert len(data['summary'])==6 and all(r['passes'] and r['count']==5 for r in data['summary'])
    names=['rfm37_k72','leg37_k72','fd176_k80']
    labels=['RFM37','Legendre37','FD4 (n=176)']
    palette=['#17658A','#168078','#AA6533']
    fig, axes=plt.subplots(1,2,figsize=(10.4,4.4),layout='constrained')
    for ax, backend, title in zip(axes,['cpu','cuda_graph'],['CPU: single pinned thread','GPU: resident CUDA Graph']):
        for i,name in enumerate(names):
            r=next(r for r in data['summary'] if r['name']==name and r['backend']==backend)
            median=r['median_seconds']
            ax.bar(i,median,color=palette[i],width=.58,alpha=.86,zorder=2)
            ax.errorbar(i,median,yerr=[[median-r['min_seconds']],[r['max_seconds']-median]],
                        fmt='none',color='#1C2832',capsize=5,lw=1.25,zorder=4)
            # Jitter only the plotting x coordinates; all five wall times remain visible.
            ax.scatter(i+np.linspace(-.18,.18,5),r['samples_seconds'],s=21,
                       facecolors='white',edgecolors='#172C3B',linewidths=.8,zorder=5)
            ax.text(i,r['max_seconds']*1.04,f'{median:.3f} s',ha='center',va='bottom',fontsize=10)
        ax.set_xticks(range(3),labels)
        ax.set_ylabel('Full solve wall time (s)')
        ax.set_title(title,fontsize=12,pad=12)
        ax.set_ylim(0,ax.get_ylim()[1]*1.13)
        ax.grid(axis='y',color='#DCE3E8',lw=.7,zorder=0)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Re = 100 full-annulus flow: all velocity and vorticity errors <= 0.5%',fontsize=13)
    fig.supxlabel('Median bars; full-range whiskers; five randomized-block samples. Fresh in-process solvers; t = 72.',fontsize=9)
    target=ROOT.parents[1]/'outputs'
    fig.savefig(target/'RFM_NS_v7_CPU_GPU_same_accuracy.svg')
    fig.savefig(target/'RFM_NS_v7_CPU_GPU_same_accuracy.png',dpi=300)
    plt.close(fig)
    print(target/'RFM_NS_v7_CPU_GPU_same_accuracy.svg')


if __name__=='__main__':
    main()
