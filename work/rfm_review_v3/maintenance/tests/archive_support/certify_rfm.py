"""Independent numerical audit of saved formal seed-101 RFM results.

Despite the requested filename, this is NOT a rigorous PDE error certificate.
No retraining or reference generation occurs. Run only after isolated timing.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
import rfm_solver as s

ROOT=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def batches(xy,size=192):
    for start in range(0,len(xy),size):yield xy[start:start+size]


def raw_fields(problem,basis,coefficients,method,xy):
    output={}
    for points in batches(xy):
        F,Fx,Fy,lap=basis.jet(points)
        if method in ('primitive','picard'):
            cu,cv,cp=coefficients.reshape(3,basis.count)
            part=dict(u=F@cu,v=F@cv,p=F@cp,omega=Fx@cv-Fy@cu,div=Fx@cu+Fy@cv)
        else:
            a=coefficients[:basis.count]
            lift,envelope=s.lift_and_envelope(problem,points)
            psi=lift+s.multiply(np.stack((F,Fx,Fy,lap)),envelope)@a
            part=dict(psi=psi[0],u=psi[2],v=-psi[1],omega=-psi[3],div=np.zeros(len(points)))
            if method=='mixed':part['omega_independent']=F@coefficients[basis.count:]
        for key,value in part.items():output.setdefault(key,[]).append(value)
    return {key:np.concatenate(value) for key,value in output.items()}


def rms(v):return float(np.sqrt(np.mean(np.asarray(v)**2)))


def fourth_jet_preflight():
    """Differentiate lower-order jets to check the newly added fourth jets."""
    rng=np.random.default_rng(77119);checks=[]
    for case in ('cavity','kovasznay'):
        problem=s.Problem(case,40)
        normalized=rng.uniform(.1,.9,(11,2))
        good=np.all((abs(normalized-.25)>.03)&(abs(normalized-.75)>.03),axis=1)
        xy=problem.lo+problem.length*normalized[good]
        for family in ('cos','legendre'):
            basis=(s.LegendreFeatures(problem,15) if family=='legendre'
                   else s.Features(problem,13,17,'cos',2,2.))
            h=problem.length*1e-5
            center=basis.clamped_derivatives(xy)
            errors=[]
            for ij,axis in [((2,0),0),((3,0),0),((1,2),0),((0,3),1)]:
                shift=np.zeros(2);shift[axis]=h
                plus=basis.clamped_derivatives(xy+shift)[ij]
                minus=basis.clamped_derivatives(xy-shift)[ij]
                target=(ij[0]+(axis==0),ij[1]+(axis==1))
                finite=(plus-minus)/(2*h)
                err=float(np.linalg.norm(finite-center[target])/max(np.linalg.norm(center[target]),1e-30))
                errors.append(err)
            checks.append({'case':case,'family':family,'max_relative_jet_differentiation_error':max(errors)})
            if max(errors)>2e-5:raise AssertionError(f'Fourth-jet preflight failed: {checks[-1]}')
    return checks


def strong_check(problem,basis,c,method,n):
    xy,_=s.interior_boundary(problem,n)
    values={}
    for points in batches(xy,96):
        if method in ('primitive','picard'):
            F,X,Y,L=basis.jet(points);cu,cv,cp=c.reshape(3,basis.count)
            u,v=F@cu,F@cv;ux,uy,vx,vy=X@cu,Y@cu,X@cv,Y@cv
            adv=np.column_stack((u*ux+v*uy,u*vx+v*vy))
            diff=problem.nu*np.column_stack((L@cu,L@cv))
            pressure=np.column_stack((X@cp,Y@cp))
            residual=adv-diff+pressure
            part=dict(residual=residual,advection=adv,diffusion=diff,pressure_gradient=pressure,divergence=ux+vy)
        else:
            a=c[:basis.count]
            feature_jets=basis.clamped_derivatives(points)
            lift_jets=s.lifted_stream_derivatives(problem,points)
            jets={ij:lift_jets[ij]+feature_jets[ij]@a for ij in feature_jets}
            u,v,wx,wy,lap=s.transport_components(jets)
            omega=-jets[2,0]-jets[0,2]
            adv=u*wx+v*wy;diff=problem.nu*lap
            part=dict(residual=adv-diff,advection=adv,diffusion=diff)
            if method=='mixed':
                F,X,Y,L=basis.jet(points);b=c[basis.count:]
                independent=F@b;ix,iy,ilap=X@b,Y@b,L@b
                delta=omega-independent
                predicted_transport=u*ix+v*iy-problem.nu*ilap
                correction=u*(wx-ix)+v*(wy-iy)-problem.nu*(lap-ilap)
                part.update(compatibility=delta,independent_transport=predicted_transport,
                    consistency_derivative_correction=correction,
                    true_residual_identity_error=(adv-diff)-(predicted_transport+correction))
        for key,value in part.items():values.setdefault(key,[]).append(value)
    values={key:np.concatenate(value) for key,value in values.items()}
    termkeys=['advection','diffusion']+(['pressure_gradient'] if 'pressure_gradient' in values else [])
    scale=float(np.sqrt(sum(np.mean(values[key]**2) for key in termkeys)))
    result={'grid':f'{n}x{n} midpoints','points':len(xy),'raw_residual_rms':rms(values['residual']),
        'raw_residual_sampled_max':float(np.max(abs(values['residual']))),
        'component_rms_scale':scale,'relative_to_component_rms':rms(values['residual'])/max(scale,1e-30),
        'normalization':'sqrt(sum of mean-square advection, diffusion, and (primitive only) pressure-gradient components); residual is nondimensional, no training weights.'}
    for key in values:
        if key!='residual':result[key+'_rms']=rms(values[key])
    return result


def error_against_reference(fields,reference):
    if not all(k in reference for k in ('u','v','omega')):return None
    return s.field_errors(fields,reference)


def circulation_check(problem,basis,c,method,q=64):
    z,w=leggauss(q);t=problem.lo+problem.length*(z+1)/2;weights=problem.length*w/2
    x,y=np.meshgrid(t,t,indexing='ij');xy=np.column_stack((x.ravel(),y.ravel()))
    f=raw_fields(problem,basis,c,method,xy)
    area=(weights[:,None]*weights).ravel()
    integral=float(area@f['omega'])
    lo,hi=problem.lo,problem.lo+problem.length
    edge=np.concatenate((np.column_stack((t,np.full(q,lo))),np.column_stack((np.full(q,hi),t)),
        np.column_stack((t,np.full(q,hi))),np.column_stack((np.full(q,lo),t))))
    wall=raw_fields(problem,basis,c,method,edge)
    circulation=float(weights@wall['u'][:q]+weights@wall['v'][q:2*q]-weights@wall['u'][2*q:3*q]-weights@wall['v'][3*q:])
    target=-8/15 if problem.case=='cavity' else 0.
    bu,bv=problem.boundary_velocity(edge)
    result={'gauss_order_per_axis':q,'omega_area_integral':integral,'velocity_boundary_circulation':circulation,
        'area_minus_boundary':integral-circulation,'expected_from_exact_boundary_data':target,
        'area_minus_boundary_target':integral-target,'wall_velocity_sampled_max_error':float(np.max(np.hypot(wall['u']-bu,wall['v']-bv)))}
    if 'omega_independent' in f:
        result['independent_omega_integral']=float(area@f['omega_independent'])
        result['independent_minus_velocity_curl_integral']=float(area@(f['omega_independent']-f['omega']))
    return result


def weak_vector(problem,basis,c,method,cells,order,q):
    xy,(tx,ty,lap)=s.weak_quadrature(problem,cells,order,q)
    f=raw_fields(problem,basis,c,method,xy)
    r=problem.nu*(lap@f['omega'])+tx@(f['omega']*f['u'])+ty@(f['omega']*f['v'])
    return r,{'cells_per_axis':cells,'test_order_per_axis':order,'q_per_cell_axis':q,'quadrature_points':len(xy),
        'number_of_tests':len(r),'weak_residual_l2_of_vector':float(np.linalg.norm(r)),
        'weak_residual_rms_over_tests':rms(r),'weak_residual_max':float(np.max(abs(r))),
        'uses':'actual velocity curl omega=-Delta(psi), not an independent fitted omega'}


def weak_check(problem,basis,c,method,row):
    if method in ('primitive','picard'):
        return {'skipped':'This weak curl formula assumes exact divergence freedom; primitive fields are audited by momentum and divergence instead.'}
    cells=int(row.get('cells') or 3);order=int(row.get('order',3));q=int(row.get('q',9))
    # Exact same test space at the recorded and independently enriched quadrature.
    original,first=weak_vector(problem,basis,c,method,cells,order,q)
    enriched,second=weak_vector(problem,basis,c,method,cells,order,q+8)
    delta=float(np.linalg.norm(enriched-original))
    result={'same_space':[first,second],'quadrature_vector_difference_l2':delta,
        'quadrature_difference_relative_to_enriched_norm':delta/max(float(np.linalg.norm(enriched)),1e-30)}
    altered=cells+1
    if (altered*(q+8))**2>16000 and cells>1:altered=cells-1
    altered_order=min(order,8) if cells==1 else order
    if (altered*(q+8))**2<=16000 and (altered*altered_order*(q+8))**2<=3000000:
        _,third=weak_vector(problem,basis,c,method,altered,altered_order,q+8)
        if altered_order!=order:third['changed_test_order_reason']='Lower per-cell degree on subdivided layout keeps this independent audit bounded; this is a different test space, not a directly comparable scalar norm.'
        result['altered_test_space']=third
    else:
        result['altered_test_space']={'skipped':'Requested altered layout exceeds this lightweight audit memory/work limit.',
            'requested_cells':altered,'requested_order':altered_order,'requested_q':q+8}
    result['limitation']='Finite-test residuals and changed-test-space vector norms are indicators, not full dual norms; counts differ between spaces.'
    return result


def audit_record(row,formal_source,source_sha):
    problem=s.Problem(row['case'],row['reynolds']);method=row['method']
    basis=(s.LegendreFeatures(problem,row['m']) if row['activation']=='legendre'
           else s.Features(problem,row['m'],row['seed'],row['activation'],row['patches'],row['scale']))
    re=row['reynolds'];relabel=str(int(re)) if float(re).is_integer() else str(re)
    path=ROOT/f"field_{row['case']}_{relabel}_{row['label']}.npz"
    with np.load(path) as archive:
        xy=archive['xy'].copy();c=archive['coefficients'].copy()
        saved={key:archive[key].copy() for key in ('u','v','omega','psi','p','omega_independent','div') if key in archive}
        reference={key[4:]:archive[key].copy() for key in archive.files if key.startswith('ref_')}
    expected_size=row['m']*(3 if method in ('primitive','picard') else 2 if method=='mixed' else 1)
    if c.size!=expected_size:raise ValueError(f'{path.name}: expected raw coefficients of length {expected_size}, got {c.size}')
    if not np.isfinite(c).all():raise FloatingPointError('Saved coefficients contain nonfinite values')
    fields=raw_fields(problem,basis,c,method,xy)
    if not all(np.isfinite(value).all() for value in fields.values()):raise FloatingPointError('Recomputed fields contain nonfinite values')
    differences={key:float(np.max(abs(fields[key]-value))) for key,value in saved.items() if key in fields}
    lift,_=s.lift_and_envelope(problem,xy)
    lift_fields=dict(u=lift[2],v=-lift[1],omega=-lift[3])
    if problem.case=='kovasznay':reference=problem.exact(xy)
    result={'label':row['label'],'case':row['case'],'reynolds':re,'method':method,'seed':row['seed'],
        'artifact':path.name,'artifact_sha256':digest(path),'raw_coefficient_count':c.size,
        'basis_family':row['activation'],'audit_status':'completed',
        'reported_retained_unknowns':row.get('unknowns'),'whitening':row.get('whitening'),
        'formal_results_file':formal_source,'formal_source_sha256':source_sha,
        'current_source_sha256':digest(ROOT/'rfm_solver.py'),'source_hash_matches':source_sha==digest(ROOT/'rfm_solver.py'),
        'same_score_grid':{'points':len(xy),'saved_vs_recomputed_max_absolute_differences':differences,
            'recomputed_errors':error_against_reference(fields,reference),
            'lift_only_errors_without_training':error_against_reference(lift_fields,reference)},
        'independent_strong_residual_checks':[strong_check(problem,basis,c,method,n) for n in (29,41)],
        'circulation':circulation_check(problem,basis,c,method),
        'independent_weak_validation':weak_check(problem,basis,c,method,row)}
    if 'omega_independent' in fields:
        result['same_score_grid']['independent_omega_relative_disagreement']=float(np.linalg.norm(fields['omega_independent']-fields['omega'])/max(np.linalg.norm(fields['omega']),1e-30))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['all','cavity','kovasznay'],default='all')
    args=parser.parse_args();start=time.perf_counter();records=[]
    preflight=fourth_jet_preflight()
    paths=sorted(ROOT.glob('formal_*.json'))
    for path in paths:
        data=json.loads(path.read_text(encoding='utf8'))
        if data.get('stage')!='formal':continue
        for row in data['rows']:
            if row.get('seed')!=101 or (args.case!='all' and row['case']!=args.case):continue
            tick=time.perf_counter()
            try:
                record=audit_record(row,path.name,data.get('source_sha256'))
            except Exception as error:
                record={'case':row['case'],'reynolds':row['reynolds'],'label':row['label'],
                        'seed':row['seed'],'audit_status':'failed','error_type':type(error).__name__,'error':str(error)}
            record['audit_seconds']=time.perf_counter()-tick;records.append(record)
            print(json.dumps({'case':row['case'],'Re':row['reynolds'],'label':row['label'],
                'audit_status':record['audit_status'],
                'recomputed_errors':record.get('same_score_grid',{}).get('recomputed_errors'),
                'error':record.get('error'),
                'audit_seconds':record['audit_seconds']}),flush=True)
    if not records:raise RuntimeError('No formal seed101 rows found; run only when saved formal outputs are available.')
    report={'audit_kind':'Independent numerical validation; NOT a rigorous PDE error certificate.',
        'seed':101,'records':records,'seconds':time.perf_counter()-start,'fourth_jet_preflight':preflight,
        'methodology':['Reconstruct original frozen features and saved raw coefficients; never reapply Stokes-whitening transform.',
            'Recompute fields and lift-only errors on exactly the saved score grid.',
            'Evaluate true velocity-induced curl transport on independent 29/41 midpoint grids; primitive formulation uses actual momentum plus divergence.',
            'Integrate vorticity and boundary velocity circulation by 64-point tensor/edge Gauss rules.',
            'Recompute weak residual with q+8 in the same test space and an altered cell layout where feasible.'],
        'limits':['Uses the audited analytic derivative implementation; independent grids and formulas do not constitute a separate full PDE solver.',
            'Finite sampling and finite test spaces do not give rigorous uniform residual or solution-error bounds.',
            'Circulation is a structural/global consistency check, not sufficient evidence of accurate interior flow.',
            'Only seed101 artifacts are re-evaluated; multi-seed performance remains the responsibility of the formal benchmark.',
            'Audit time is separate from and must not be folded into previously recorded isolated solver times.']}
    target=ROOT/('audit_certification_seed101.json' if args.case=='all' else f'audit_certification_seed101_{args.case}.json')
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':main()
