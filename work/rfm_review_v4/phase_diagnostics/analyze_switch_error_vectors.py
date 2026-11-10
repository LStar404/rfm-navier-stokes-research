"""Evaluate deterministic error-vector identities from frozen fields, no integration."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys, json, hashlib
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
BRIDGE=WORK/'rfm_review_v3/profiles_bridge'
sys.path.insert(0,str(BRIDGE))
import review_common as common
np=common.np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    refs=common.references(); s,theta,w=common.base.grid_set()['layer144']
    results=[]; input_specs={}
    def fields(sol):
        idx=np.flatnonzero(sol.times==72.)
        assert len(idx)==1
        return sol.fields(int(idx[0]),s,theta)
    truths={name:fields(sol) for name,sol in refs.items()}
    for name in common.NAMES:
        uniform=common.control(name,'uniform'); at8=common.control(name,'combined')
        path=BRIDGE/(name+'_switch12.npz'); meta=common.read(path.with_suffix('.json'))
        assert sha(path)==meta['sha256']
        at12=common.case.load_solution(path,common.switch.RecurrenceGalerkin)
        u,early,late=fields(uniform),fields(at8),fields(at12)
        input_specs[name]=dict(archived_controls={arm:common.job(name,arm)['control'] for arm in ('uniform','combined')},
            switch12=dict(path=path.relative_to(WORK).as_posix(),sha256=sha(path)))
        for reference,truth in truths.items():
            for field,indices in [('velocity',(0,1)),('vorticity',(2,))]:
                r=np.stack([truth[i] for i in indices]); a=np.stack([early[i] for i in indices])-r
                b=np.stack([late[i] for i in indices])-r; e0=np.stack([u[i] for i in indices])-r
                delta=a-b
                def dot(x,y):
                    return float(np.einsum('s,cst,cst->',w,x,y))
                denominator=dot(r,r)
                norm=lambda x:np.sqrt(dot(x,x)/denominator)
                e8,e12,eu=map(float,(norm(a),norm(b),norm(e0)))
                cross=2*dot(b,delta)/denominator; d2=dot(delta,delta)/denominator
                lhs=dot(a,a)/denominator; base=dot(b,b)/denominator
                result=dict(name=name,reference=reference,time=72.,grid='layer144',field=field,
                    relative_error_uniform=eu,relative_error_switch8=e8,relative_error_switch12=e12,
                    norm_difference_recovery_fraction=(e8-e12)/(e8-eu),
                    error8_squared=lhs,error12_squared=base,cross_term=cross,delta_squared=d2,
                    identity_absolute_residual=abs(lhs-base-cross-d2),
                    identity_residual_relative_to_lhs=abs(lhs-base-cross-d2)/lhs,
                    norm_of_intervention_delta=float(np.sqrt(d2)),
                    cross_term_fraction_of_squared_error_difference=cross/(lhs-base),
                    error12_delta_cosine=dot(b,delta)/np.sqrt(dot(b,b)*dot(delta,delta)))
                assert result['identity_residual_relative_to_lhs']<1e-12
                results.append(result)
    summary=common.read(BRIDGE/'bridge_summary.json')
    rows=[row for row in summary['rows'] if row['name']=='rfm37_k72']
    maxima={row['arm']:row['worst']['velocity']['value'] for row in rows}
    recovery=(maxima['switch8']-maxima['switch12'])/(maxima['switch8']-maxima['uniform'])
    imported_sources={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            path=Path(file).resolve()
            if path.suffix=='.py' and path.is_relative_to(WORK):
                imported_sources[path.relative_to(WORK).as_posix()]=sha(path)
    result=dict(status='complete',scope='Post-review error-vector algebra on stored physical fields; no new trajectory or timing',
        source_sha256={Path(__file__).name:sha(Path(__file__)),
            'rfm_review_v3/profiles_bridge/review_common.py':sha(BRIDGE/'review_common.py')},
        imported_local_source_hashes=imported_sources,
        control_inputs=input_specs,reference_inputs=common.FROZEN['early_references']|{'late':common.FROZEN['late_reference']['reference_hashes']},
        common_norm='Physical-area L2 over existing layer144 grid; all terms use the same reference denominator',
        identity='e8=e12+delta; ||e8||^2=||e12||^2+2<e12,delta>+||delta||^2; deterministic, not a statistical variance decomposition',
        reviewer_arithmetic=dict(source_sha256=sha(BRIDGE/'bridge_summary.json'),rfm37_velocity_maxima=maxima,
            norm_max_recovery_fraction=recovery,
            interpretation='Fraction of the observed increase in maximum norm recovered by postponing the switch; not a causal error budget'),
        results=results,limitations=[
            'The intervention changes four time units of fine versus coarse integration, the bridge time/state/operator, and the later error history simultaneously.',
            't=8 is inside the analytic sin^4 pulse, not a nonsmooth forcing junction. t=12 is its C3 endpoint.',
            'An observed 5.6% recovery does not identify bridge-local error; the remaining 94.4% is not an isolated coarse-step truncation term.',
            'No time-refinement series or controlled bridge-only defect experiment is supplied.'])
    write(HERE/'switch_error_vector_identity.json',result)
    print(json.dumps([r for r in results if r['name']=='rfm37_k72' and r['reference']=='time_fine' and r['field']=='velocity'],indent=2))
    print('Maximum-norm descriptive fraction',recovery)

if __name__=='__main__': main()
