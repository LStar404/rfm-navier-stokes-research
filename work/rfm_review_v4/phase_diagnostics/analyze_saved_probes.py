"""Post-review, analysis-only diagnostics; never integrates a PDE or uses a GPU."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json, hashlib, csv, sys
import numpy as np
import scipy
from scipy.signal import detrend

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
ABL=WORK/'rfm_review_v3/ablation'
BRIDGE=WORK/'rfm_review_v3/profiles_bridge'
WINDOWS=((24.,72.),(36.,72.),(48.,72.))
PROBES=(1.5,2.5,3.5)
FIELDS=('transverse','vorticity')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rel(path):
    return path.relative_to(WORK).as_posix()
def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def crossings(t,y,trend):
    z=detrend(y,type='linear' if trend=='linear' else 'constant')
    idx=np.flatnonzero((z[:-1]<=0)&(z[1:]>0))
    c=t[idx]-z[idx]*(t[idx+1]-t[idx])/(z[idx+1]-z[idx])
    if len(c)<4:
        raise ValueError('Too few upward crossings for frequency diagnostic')
    slope,intercept=np.polyfit(np.arange(len(c),dtype=float),c,1)
    return dict(crossings=c.tolist(),crossing_count=len(c),complete_crossing_intervals=len(c)-1,
        frequency=1./float(slope),period=float(slope),crossing_time_intercept=float(intercept),
        period_std=float(np.std(np.diff(c))),
        crossing_fit_rms=float(np.sqrt(np.mean((c-intercept-slope*np.arange(len(c)))**2))),
        detrended_rms=float(np.sqrt(np.mean(z*z))))

def load_traces():
    summary=read(ABL/'summary.json'); traces={}; inputs={}
    for row in summary['results']:
        if 'source' in row:
            spec=row['source']; path=WORK/spec['trajectory'].replace('\\','/')
            assert sha(path)==spec['trajectory_sha256']
        else:
            result_path=ABL/row['result_path'].replace('\\','/')
            result=read(result_path); path=result_path.with_name('trajectory.npz')
            assert sha(path)==result['trajectory_sha256']
        key=f"{row['resolution']}:{row['role']}:{row['seed']}"
        with np.load(path) as data:
            traces[key]=dict(meta={k:row[k] for k in ('resolution','role','seed','retained_rank')},
                t=data['probe_times'].copy(),transverse=data['transverse'].copy(),vorticity=data['vorticity'].copy())
        inputs[key]=dict(path=rel(path),sha256=sha(path))
    frozen=read(WORK/'rfm_v7/switch_qualification_output/protocol.json')
    for name,rate in [('angular_fine',384),('time_fine',512)]:
        path=WORK/f'rfm_v7/cpu_formal_point1_pow2_chain_output/fd448_k144_rate{rate}_pow2.npz'
        assert sha(path)==frozen['late_reference']['reference_hashes'][rel(path)]
        with np.load(path) as data:
            traces[name]=dict(meta=dict(resolution='reference',role=name,seed=None,retained_rank=None),
                t=data['probe_times'].copy(),transverse=data['transverse'].copy(),vorticity=data['vorticity'].copy())
        inputs[name]=dict(path=rel(path),sha256=sha(path))
    expected=np.arange(577)*.125
    for trace in traces.values():
        assert np.array_equal(trace['t'],expected)
        assert all(trace[field].shape==(577,3) and np.isfinite(trace[field]).all() for field in FIELDS)
    return traces,inputs

def analyze():
    traces,inputs=load_traces(); rows=[]; indexed={}
    for key,trace in traces.items():
        for begin,end in WINDOWS:
            mask=(trace['t']>=begin)&(trace['t']<=end); t=trace['t'][mask]
            for field in FIELDS:
                for col,x in enumerate(PROBES):
                    y=trace[field][mask,col]
                    for trend in ('linear','constant'):
                        metrics=crossings(t,y,trend)
                        row=dict(trace=key,**trace['meta'],window_start=begin,window_end=end,field=field,
                            probe_x=x,trend=trend,**metrics)
                        rows.append(row); indexed[(key,begin,end,field,x,trend)]=row
    contrasts=[]
    for key,trace in traces.items():
        if trace['meta']['resolution']=='reference': continue
        for ref in ('angular_fine','time_fine'):
            for begin,end in WINDOWS:
                mask=(trace['t']>=begin)&(trace['t']<=end)
                for field in FIELDS:
                    for col,x in enumerate(PROBES):
                        for trend in ('linear','constant'):
                            row=indexed[(key,begin,end,field,x,trend)]; r=indexed[(ref,begin,end,field,x,trend)]
                            y=trace[field][mask,col]; yr=traces[ref][field][mask,col]
                            contrasts.append(dict(trace=key,reference=ref,**trace['meta'],window_start=begin,
                                window_end=end,field=field,probe_x=x,trend=trend,
                                frequency=row['frequency'],reference_frequency=r['frequency'],
                                relative_frequency_bias=row['frequency']/r['frequency']-1,
                                phase_rate_mismatch_estimate_radians=2*np.pi*(row['frequency']-r['frequency'])*(end-begin),
                                unshifted_probe_relative_l2=float(np.linalg.norm(y-yr)/np.linalg.norm(yr)),
                                detrended_rms_ratio=row['detrended_rms']/r['detrended_rms'],
                                complete_crossing_intervals=row['complete_crossing_intervals']))
    primary=[r for r in contrasts if r['window_start']==24 and r['trend']=='linear' and r['reference']=='time_fine']
    cells=[]
    for resolution in ('mid','high'):
        for role in ('random-layers','midpoint-layers','random-no-layers','midpoint-no-layers'):
            cell=[r for r in primary if r['resolution']==resolution and r['role']==role]
            cells.append(dict(resolution=resolution,role=role,unique_trajectories=len({r['trace'] for r in cell}),
                frequency_min=min(r['frequency'] for r in cell),frequency_max=max(r['frequency'] for r in cell),
                frequency_bias_min=min(r['relative_frequency_bias'] for r in cell),
                frequency_bias_max=max(r['relative_frequency_bias'] for r in cell),
                phase_rate_mismatch_min=min(r['phase_rate_mismatch_estimate_radians'] for r in cell),
                phase_rate_mismatch_max=max(r['phase_rate_mismatch_estimate_radians'] for r in cell)))
    representative=[dict(r) for r in primary if r['role'] in ('midpoint-layers','midpoint-no-layers')
                    and r['field']=='vorticity' and r['probe_x']==2.5]
    for record in representative:
        row=indexed[(record['trace'],24.,72.,'vorticity',2.5,'linear')]
        ref=indexed[('time_fine',24.,72.,'vorticity',2.5,'linear')]
        cc=np.asarray(row['crossings']); rc=np.asarray(ref['crossings'])
        j=int(np.argmin(abs(cc-rc[0]))); n=min(len(rc),len(cc)-j)
        record['last_matched_reference_crossing_time']=float(rc[n-1])
        record['last_matched_candidate_crossing_delay']=float(cc[j+n-1]-rc[n-1])
        record['matched_reference_crossings']=rc[:n].tolist()
        record['matched_crossing_delays']=(cc[j:j+n]-rc[:n]).tolist()
    ref_rows=[r for r in rows if r['trace']=='time_fine' and r['window_start']==24 and r['trend']=='linear']
    no_layers=[r for r in contrasts if 'no-layers' in r['role']]
    result=dict(status='complete',scope='Post-review descriptive analysis of saved probes, no new trajectory or timing',
        inputs=inputs,source_sha256={Path(__file__).name:sha(Path(__file__))},
        library_versions=dict(numpy=np.__version__,scipy=scipy.__version__),
        method=dict(windows=WINDOWS,probe_x=PROBES,probe_y=0,fields=FIELDS,samples=577,sample_spacing=.125,
            estimator='Subtract least-squares linear trend; linearly interpolate upward zero crossings; fit crossing time = intercept + period * index',
            primary_trend='linear',secondary_sensitivity='subtract mean only',primary_reference='time_fine',
            primary_window=[24,72],frequency_unit='cycles per nondimensional time',
            phase_rate_mismatch='2*pi*(f_candidate-f_reference)*(window_end-window_start); a fitted frequency-rate estimate, not a field phase-alignment result',
            no_phase_shifted_error_substitution=True),
        reference_primary_frequency_range=[min(r['frequency'] for r in ref_rows),max(r['frequency'] for r in ref_rows)],
        all_no_layer_window_reference_trend_frequency_bias_range=[min(r['relative_frequency_bias'] for r in no_layers),max(r['relative_frequency_bias'] for r in no_layers)],
        cells=cells,representative=representative,
        limitations=['Features removed are analytic outer-boundary enrichments, not measured inner-wall separation data.',
            'Removing features also changes retained rank; no equal-rank causal attribution.',
            'Finite-domain probe frequency is not a measured unbounded-cylinder Strouhal benchmark.',
            'A short finite-window crossing estimate has no supplied statistical confidence interval; windows, probes and detrending sensitivities are reported.',
            'A changing frequency and phase lag are consistent with long-time drift, but do not quantify what fraction of the physical field error is phase versus amplitude or shape.',
            'All original physical scores remain at the original time, without phase alignment.'])
    write(HERE/'probe_diagnostics.json',result)
    write(HERE/'crossing_records.json',dict(rows=rows))
    csv_write(HERE/'probe_frequency_contrasts.csv',contrasts)
    plot(traces,indexed)
    return result

def plot(traces,indexed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':8})
    fig,axes=plt.subplots(2,2,figsize=(6.3,5.7),layout='constrained')
    ref=traces['time_fine']; palette={'midpoint-layers':'#2166ac','midpoint-no-layers':'#b2182b'}
    for col,resolution in enumerate(('mid','high')):
        ax=axes[0,col]; mask=ref['t']>=24
        ax.plot(ref['t'][mask],ref['vorticity'][mask,1],color='#222222',ls=':',lw=1.8,label='Fine FD reference')
        r=indexed[('time_fine',24.,72.,'vorticity',2.5,'linear')]; rc=np.asarray(r['crossings'])
        for role in ('midpoint-layers','midpoint-no-layers'):
            key=f'{resolution}:{role}:7331'; trace=traces[key]
            label='Midpoint + layers' if role=='midpoint-layers' else 'Midpoint, no layers'
            ax.plot(trace['t'][mask],trace['vorticity'][mask,1],color=palette[role],lw=1.,alpha=.9,label=label)
            cc=np.asarray(indexed[(key,24.,72.,'vorticity',2.5,'linear')]['crossings'])
            # Fix the crossing index using the nearest event at the start, then keep it.
            j=int(np.argmin(np.abs(cc-rc[0]))); n=min(len(rc),len(cc)-j)
            axes[1,col].plot(rc[:n],cc[j:j+n]-rc[:n],'-o',ms=3,color=palette[role],lw=1.,label=label)
        ax.set_title(f'{resolution.capitalize()} resolution: vorticity at (2.5, 0)')
        ax.set_xlabel('Time'); ax.set_ylabel('Vorticity'); ax.grid(alpha=.2)
        axes[1,col].set_title('Matched upward-crossing delay')
        axes[1,col].set_xlabel('Reference crossing time'); axes[1,col].set_ylabel('Candidate minus reference time')
        axes[1,col].axhline(0,color='#666666',lw=.6); axes[1,col].grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=3,frameon=False)
    fig.savefig(HERE/'review_phase_probes.pdf',metadata={'CreationDate':None})
    fig.savefig(HERE/'review_phase_probes.png',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    result=analyze()
    print(json.dumps(dict(cells=result['cells'],representative=result['representative'],
                         sensitivity=result['all_no_layer_window_reference_trend_frequency_bias_range']),indent=2))
