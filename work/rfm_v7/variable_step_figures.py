"""Static research figures from frozen field-error records, without time shifting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'variable_step_figures'


def read(path):
    return json.loads((HERE/path).read_text(encoding='utf-8'))


def by_time(rows, metrics, refs=False):
    if refs:
        rows=[r for group in rows.values() for r in group]
    times=sorted({r['time'] for r in rows})
    return np.asarray(times), np.asarray([max(r[k] for r in rows if r['time']==t for k in metrics) for t in times])


def main():
    q=read('switch_qualification_output/qualification.json')
    previous=read('r128_operator_qualification_output/qualification.json')
    startup=read('startup_replay_output/audit.json')
    if not q['passes'] or len(q['rows'])!=18:
        raise RuntimeError('Need complete five-seed switch qualification')
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.20,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.05),layout='constrained')
    for ax,name,seed,label in zip(axes,['rfm37_k72','leg37_k72'],[7331,17],['RFM37 (seed 7331)','Legendre37']):
        old=next(r for r in startup['comparisons'] if r['name']==name and r['rate']==112 and r['arm']=='cuda_graph')
        fine=next(r for r in previous['rows'] if r['job']==dict(name=name,seed=seed,arm='cuda_graph'))
        new=next(r for r in q['rows'] if r['job']==dict(name=name,seed=seed,arm='cuda_graph',role='candidate'))
        for rows,color,style,title in [(old['fields'],'#bd3f34','o-','Uniform 112'),
                                       (fine['implementation_scores'],'#70777e','--','Uniform 128'),
                                       (new['implementation_scores'],'#147c7b','.-','128 then 112')]:
            t,y=by_time(rows,['vorticity','near_vorticity'])
            m=t<=16
            ax.semilogy(t[m],np.maximum(y[m],1e-16),style,color=color,label=title,markersize=3,linewidth=1.5)
        ax.axhline(1e-5,color='#735a39',ls=':',lw=1.3,label='Implementation gate')
        ax.axvline(8,color='#bbc5c9',lw=.9)
        ax.set(xlim=(0,16.3),ylim=(1e-15,5e-3),title=label,xlabel='Time')
    axes[0].set_ylabel('CPU-CUDA vorticity difference')
    axes[1].legend(loc='upper right',fontsize=7)
    fig.savefig(OUT/'implementation.png',dpi=200)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.15,3.05),layout='constrained')
    for ax,field,title in zip(axes,['velocity','vorticity'],['Velocity','Vorticity']):
        metrics=[field,'near_'+field]
        rfm=[r for r in q['rows'] if r['job']['name']=='rfm37_k72' and r['job']['arm']=='cuda_graph']
        seed_values=[by_time(r['physical_scores'],metrics,True) for r in rfm]
        t=seed_values[0][0]; values=100*np.stack([v for _,v in seed_values])
        ax.fill_between(t,values.min(0),values.max(0),color='#127c87',alpha=.22,label='RFM, five-seed range')
        ax.plot(t,np.median(values,axis=0),'.-',color='#127c87',markersize=3,label='RFM median, switched')
        leg=next(r for r in q['rows'] if r['job']['name']=='leg37_k72' and r['job']['arm']=='cuda_graph')
        tl,yl=by_time(leg['physical_scores'],metrics,True)
        ax.plot(tl,100*yl,'.-',color='#bf7832',markersize=3,label='Legendre, switched')
        anchors=[r for r in q['anchors'] if r['name']=='rfm37_k72']
        baseline=[by_time(r['physical_scores'],metrics,True) for r in anchors]
        ax.plot(t,100*np.median(np.stack([v for _,v in baseline]),axis=0),'--',color='#6b628c',lw=1.1,
                label='RFM median, uniform 128')
        ax.axhline(.5,color='#aa4239',ls=':',lw=1.3,label='Physical gate: 0.5%')
        ax.axvline(8,color='#bbc5c9',lw=.9)
        ax.set(xlim=(0,73),ylim=(.0001,.7),yscale='log',title=title,xlabel='Time')
    axes[0].set_ylabel('Worst registered relative error (%)')
    axes[1].legend(loc='lower right',fontsize=6.7)
    fig.savefig(OUT/'physical_accuracy.png',dpi=200)
    plt.close(fig)
    print('Wrote two figures from saved scores only.')


if __name__=='__main__':
    main()
