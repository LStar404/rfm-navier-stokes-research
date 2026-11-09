"""Plot vorticity checkpoints and the three centreline wake traces."""
import argparse
import json
from pathlib import Path

import full_case as case
from diagnostic_wake import analyze, reflection_breaking_modal, spatial_cycles
from full_case import np


def select_states(solution, after, period):
    times=solution.times
    first=len(times)-1
    if period is not None:
        best_phase=None
        for i,t in enumerate(times):
            if t<after:
                continue
            j=int(np.argmin(abs(times-(t+period/2))))
            if j>i and abs((times[j]-t)/period-.5) <= .12:
                error=abs((times[j]-t)/period-.5)
                trace=solution.transverse[:,1]
                v0,v1=np.interp(times[[i,j]],solution.probe_times,trace)
                contrast=abs(v1-v0)
                score=(error>.06,-contrast,error)
                if best_phase is None or score<best_phase[0]:
                    best_phase=(score,i,j)
        if best_phase is not None:
            return best_phase[1:]
    return (first,)


def evaluate(solution, after):
    report=analyze(solution.probe_times,solution.transverse,solution.vorticity,x=case.PROBES,after=after)
    r=np.linspace(case.A,case.R,141)
    theta=np.linspace(0,2*np.pi,321)[:-1]
    s=np.log(r/case.A)
    analysis_indices=np.flatnonzero(solution.times>=after)
    selected=select_states(solution,after,report['median_period'])
    breaks=[]
    weight=r.copy()
    weight[[0,-1]]*=.5
    for i in analysis_indices:
        coefficients=solution.modal(i,s)
        breaks.append(dict(time=float(solution.times[i]),**reflection_breaking_modal(*coefficients,weight)))
    report['reflection_breaking']=breaks
    report['plotted_snapshots']=solution.times[list(selected)].tolist()
    if report['median_period'] and len(analysis_indices)>=4:
        times=solution.times[analysis_indices]
        period=report['median_period']
        if times[-1]-times[0]>=period and np.max(np.diff(times))<=period/8:
            fields=np.stack([solution.fields(i,s,theta)[2] for i in analysis_indices])
            report['spatial_cycles']=spatial_cycles(times,fields,r,theta,weight[:,None],period)
    fields=np.stack([solution.fields(i,s,theta)[2] for i in selected])
    return report,(r,theta,fields)


def draw(solution, report, field, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#223544','xtick.color':'#223544','ytick.color':'#223544'})
    r,theta,fields=field
    n=len(fields)
    theta=np.r_[theta,2*np.pi]
    x=r[:,None]*np.cos(theta)
    y=r[:,None]*np.sin(theta)
    fields=np.concatenate((fields,fields[:,:,:1]),axis=2)
    plume=(x>=1)&(x<=5.5)&(abs(y)<1.5)
    limit=max(float(np.quantile(abs(fields[:,plume]),.985)),1e-9)
    levels=np.linspace(-limit,limit,25)
    fig=plt.figure(figsize=(8.0,4.9),layout='constrained',facecolor='white')
    grids=fig.add_gridspec(2,n,height_ratios=(.95,1.1))
    probe=fig.add_subplot(grids[0,:])
    colors=('#17738a','#a64531','#525f8c')
    for j,xprobe in enumerate(case.PROBES):
        probe.plot(solution.probe_times,solution.transverse[:,j],color=colors[j],lw=1.25,label=f'x/D = {xprobe:g}')
    probe.axvspan(4,12,color='#d6a83b',alpha=.16)
    probe.axvspan(*report['analysis_interval'],color='#96b6c5',alpha=.10)
    probe.axhline(0,color='#6d7b85',lw=.5)
    probe.set(xlabel='Time  tU/D',ylabel='Cross-stream velocity  $u_y/U$',xlim=(0,solution.probe_times[-1]))
    probe.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
    probe.grid(alpha=.15,lw=.5)
    axes=[]
    for i,(data,t) in enumerate(zip(fields,report['plotted_snapshots'])):
        ax=fig.add_subplot(grids[1,i])
        axes.append(ax)
        plot=ax.contourf(x,y,data,levels=levels,cmap='RdBu_r',extend='both')
        ax.add_patch(Circle((0,0),case.A,facecolor='white',edgecolor='#223544',lw=1.))
        ax.plot(case.PROBES,np.zeros_like(case.PROBES),'o',ms=2.8,color='#263642')
        ax.set(xlim=(-.7,5.2),ylim=(-1.6,1.6),xlabel='x/D',ylabel='y/D' if i==0 else None,
               title=f'Vorticity at tU/D = {t:.2f}')
        ax.set_aspect('equal')
        ax.set_anchor('C')
    bar=fig.colorbar(plot,ax=axes,shrink=.73,pad=.025)
    bar.set_label(r'Vorticity  $\omega D/U$')
    fig.savefig(path,dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution',type=Path)
    parser.add_argument('--kind',choices=('fd','galerkin'),default='fd')
    parser.add_argument('--history',type=Path,help='Optional untouched cold-start probe history NPZ')
    parser.add_argument('--after',type=float,default=36.)
    parser.add_argument('--plot',type=Path,required=True)
    parser.add_argument('--metrics',type=Path,required=True)
    args=parser.parse_args()
    if args.kind=='fd':
        from fd_full import FullFD as Model
    else:
        from full_galerkin import FullGalerkin as Model
    solution=case.load_solution(args.solution,Model)
    if args.history:
        with np.load(args.history,allow_pickle=False) as history:
            solution.probe_times=history['probe_times']
            solution.transverse=history['transverse']
            solution.vorticity=history['vorticity']
        if (solution.probe_times[0]>args.after or solution.probe_times[-1]<solution.times[-1]
            or len(solution.probe_times)!=len(solution.transverse)
            or solution.transverse.shape!=solution.vorticity.shape):
            raise ValueError('Probe history does not span the diagnostic states')
    report,field=evaluate(solution,args.after)
    report['state_source']=args.solution.name
    report['probe_source']=(args.history or args.solution).name
    draw(solution,report,field,args.plot)
    args.metrics.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'St={report["strouhal"]}, passed={report["passed"]}, snapshots={report["plotted_snapshots"]}')


if __name__=='__main__':
    main()
