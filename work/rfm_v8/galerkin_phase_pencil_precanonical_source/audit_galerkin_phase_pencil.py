"""Compare phase pencils and direct solves without advancing a trajectory."""

from decimal import localcontext
from itertools import combinations
from pathlib import Path
from statistics import median
from time import perf_counter

from galerkin_phase_pencil import (
    V7, case, direct_packs, mode_matrices, np, pack_step, pencil_packs, prepare_pencils,
    scaled_direct, spatial_blocks, upper_symmetric,
)
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import csr_matrix

from cpu_formal import config, digest, grid_set, read, save
from cpu_sbdf2_switch import RecurrenceGalerkin, bridge_weights, raw_terms
from fd_equilibration_precision import OracleBand, backward, complex_vector, decimal_rows, decimal_vector
from full_galerkin import inverse_spd
from radial_layers import radial_quadrature


HERE = Path(__file__).resolve().parent
PROTOCOL = V7/"prepared_qualification_output"/"protocol.json"
OUTPUT = HERE/"galerkin_phase_pencil_audit.json"
TINY = np.finfo(float).tiny
GATE = 1e-5
LABELS = ("startup","bridge","late")
METHODS = {"mid":(("rfm37_k72",7331),("leg37_k72",17)),
           "high":(("rfm41_k80",7331),("leg39_k80",17))}


def relative(actual, expected):
    return float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),TINY))


def phase_specs(schedule):
    h0,h1 = (1/schedule[k] for k in ("startup_rate","late_rate"))
    a,b0,b1,e0,e1 = bridge_weights(h1/h0)
    scale = 1.5/a
    return ((h0,1.,2.,-.5,2.,-1.),(h1*scale,scale,b0,-b1,e0,-e1),(h1,1.,2.,-.5,2.,-1.))


def commutation(model, blocks):
    _,weights = radial_quadrature(model.metadata["q"],model.metadata["split"])
    f,_,d2 = model.samples
    j = (d2-f)*(np.sqrt(weights)/model.r)[:,None]
    u = f*(np.sqrt(weights)/model.r)[:,None]
    cross = j.T@u+u.T@j
    g = u.T@u
    coefficients = dict(M0=model.M0,M1=model.M1,K0=case.NU*(j.T@j+cross+g),
                        K1=-case.NU*(cross+2*g),K2=case.NU*g)
    mid = model.k//2
    anchor = upper_symmetric(1.5*blocks[0][mid-1]+blocks[1][mid-1]/128)
    lower = cholesky(anchor,lower=True)
    whitened = {}
    for name,matrix in coefficients.items():
        first = solve_triangular(lower,upper_symmetric(matrix),lower=True)
        whitened[name] = solve_triangular(lower,first.T,lower=True).T
    return {a+"_"+b:float(np.linalg.norm(x@y-y@x)/(np.linalg.norm(x)*np.linalg.norm(y)))
            for a,b in combinations(whitened,2) for x,y in ((whitened[a],whitened[b]),)}


def measure(model, blocks, dts):
    def old():
        return [pack_step(model,blocks,h) for h in dts]

    def direct():
        return direct_packs(model,blocks,dts)

    def pencil():
        factors = prepare_pencils(model,blocks,dts)
        return pencil_packs(model,dts,*factors)

    actions = dict(v7_inverse=old,direct_cholesky=direct,anchor_pencil=pencil)
    for action in actions.values():
        action()
    samples = {key:[] for key in actions}
    outputs = {}
    keys = list(actions)
    for i in range(5):
        for key in keys[i%3:]+keys[:i%3]:
            t = perf_counter()
            outputs[key] = actions[key]()
            samples[key].append(perf_counter()-t)
    setup,apply = [],[]
    for _ in range(5):
        t = perf_counter()
        factors = prepare_pencils(model,blocks,dts)
        middle = perf_counter()
        pencil_packs(model,dts,*factors)
        end = perf_counter()
        setup.append(middle-t)
        apply.append(end-middle)
    return outputs,factors[0],dict(
        seconds={key:dict(median=median(values),samples=values) for key,values in samples.items()},
        pencil_factor_and_projection_seconds=setup,pencil_phase_reconstruction_seconds=apply,
        dense_output_bytes=sum(a.nbytes for p in outputs["v7_inverse"] for a in p))


def differences(outputs):
    keys = ("linear","radial","angular","mass_lift","future_lift")
    comparison = {}
    for method,values in outputs.items():
        if method == "v7_inverse":
            continue
        comparison[method] = {label:{name:dict(relative_l2=relative(a,b),max_abs=float(np.max(abs(a-b))),
            worst_mode_relative=(max(relative(a[...,k],b[...,k]) for k in range(a.shape[-1])) if a.ndim==3 else None))
            for name,a,b in zip(keys,p,old,strict=True)}
            for label,p,old in zip(LABELS,values,outputs["v7_inverse"],strict=True)}
    return comparison


def held_rhs(model, blocks, terms, g_future, phase):
    h,scale,b0,b1,e0,e1 = phase
    old,now = terms
    matrices = mode_matrices(model,blocks)
    rhs = []
    for k,(m,_) in enumerate(matrices):
        mw = model.mean_width

        def projections(item):
            state,g,r,a = item
            if k == 0:
                return m@state[:mw,0].real,-model.mean_test@r[:,0].real
            mass = m@state[:,k]
            if k == 1:
                mass += g*model.M_lift
            return mass,model.test_s@r[:,k]-1j*k*(model.test@a[:,k])

        m0,f0 = projections(now)
        m1,f1 = projections(old)
        b = scale*(b0*m0+b1*m1)+h*(e0*f0+e1*f1)
        if k == 1:
            b -= g_future*(1.5*model.M_lift+h*blocks[2])
        rhs.append(b)
    return rhs


def apply_pack(model, pack, terms, g_future, phase):
    _,scale,b0,b1,e0,e1 = phase
    linear,radial,angular,lift,future = pack
    result = np.zeros(model.rhs_shape,dtype=complex)
    for weights,item in (((b1,e1),terms[0]),((b0,e0),terms[1])):
        c,g,r,a = item
        b,e = weights
        result += scale*b*np.einsum("abk,bk->ak",linear,c)
        result += e*model.kernel.n*(np.einsum("aqk,qk->ak",radial,r)
                                   -1j*np.einsum("aqk,qk->ak",angular,a))
        result[:,1] += scale*b*g*lift
    result[:,1] -= g_future*future
    return result


def held_states(control, model, schedule):
    path = V7.parent/control["cpu_recurrence"]["path"]
    if digest(path) != control["cpu_recurrence"]["sha256"]:
        raise ValueError("The archived held state changed")
    saved = case.load_solution(path,RecurrenceGalerkin)
    if not np.array_equal(saved.model.transform,model.transform):
        raise ArithmeticError("Held state uses a different spatial basis")
    pairs = ((.5,1.),(schedule["checkpoints"][6],schedule["checkpoints"][7]))
    inputs = {}
    for label,times in zip(("startup_observations","adjacent_after_bridge"),pairs,strict=True):
        states = [saved.states[np.flatnonzero(saved.times==t).item()] for t in times]
        amp = case.speed(times[-1]+1/schedule["late_rate"])
        inputs[label] = [raw_terms(model,c,case.speed(t)) for t,c in zip(times,states,strict=True)],amp
    return inputs,path


def field_errors(model, held, phases, packs):
    grids = grid_set()
    results = {}
    for label,(terms,g) in held.items():
        # Store the actual lift amplitude separately so a one-step audit does
        # not misrepresent archived checkpoints as a new time trajectory.
        t = 0.
        states = {key:[apply_pack(model,p,terms,g,phase) for p,phase in zip(values,phases,strict=True)]
                  for key,values in packs.items()}
        for method,candidates in states.items():
            if method == "v7_inverse":
                continue
            results.setdefault(method,{})[label] = phase_results = {}
            for name,actual,old in zip(LABELS,candidates,states["v7_inverse"],strict=True):
                maxima = {}
                for grid,(s,theta,w) in grids.items():
                    fields = lambda c:tuple(case.reconstruct(v,theta) for v in model.modal(c,g,s))
                    u,v,o = fields(actual)
                    ur,vr,orr = fields(old)
                    e,base_v = (u-ur)**2+(v-vr)**2,ur**2+vr**2
                    en,base_o = (o-orr)**2,orr**2
                    near = w*(case.radius(s)<=2.5)
                    maxima[grid] = {key:float(np.sqrt((weight@delta.sum(1))/(weight@base.sum(1))))
                        for key,weight,delta,base in (("velocity",w,e,base_v),("vorticity",w,en,base_o),
                                                       ("near_velocity",near,e,base_v),("near_vorticity",near,en,base_o))}
                phase_results[name] = dict(coefficient_relative=relative(actual,old),grids=maxima,
                    worst_field=max(x for m in maxima.values() for x in m.values()))
    return results


def backward_float(matrix, candidate, rhs):
    residual = abs(matrix@candidate-rhs)
    norm = abs(matrix)@abs(candidate)+abs(rhs)
    return float(np.max(residual/np.maximum(norm,TINY)))


def fixed_systems(model, blocks, phases, pencils, held):
    rng = np.random.default_rng(237193)
    matrices = mode_matrices(model,blocks)
    decimal_modes = {0,1,model.k//2,model.k-1,model.k}
    output = {}
    for label,phase in zip(LABELS,phases,strict=True):
        h = phase[0]
        physical = {name:held_rhs(model,blocks,*data,phase) for name,data in held.items()}
        tests = []
        for k,((m,d),pencil) in enumerate(zip(matrices,pencils,strict=True)):
            stored = 1.5*m+h*d
            a = upper_symmetric(stored)
            x = rng.normal(size=len(a))
            b = rng.normal(size=len(a))
            if k:
                x = x+1j*rng.normal(size=len(a))
                b = b+1j*rng.normal(size=len(a))
            workloads = dict(random_rhs=b,manufactured=a@x,**{name:rhs[k] for name,rhs in physical.items()})
            inverse = inverse_spd(stored)
            solver = dict(v7_inverse=lambda v:inverse@v,direct_cholesky=lambda v:scaled_direct(stored,v),
                          anchor_pencil=lambda v:pencil.solve(1.5,h,v))
            mode = dict(mode=k,condition_2=float(np.linalg.cond(a)),
                        balanced_condition_2=float(np.linalg.cond(a/np.sqrt(np.diag(a))[:,None]/np.sqrt(np.diag(a)))),
                        asymmetric_relative=relative(stored,a),
                        whitened_condition=float(max(pencil.diagonal(1.5,h))/min(pencil.diagonal(1.5,h))),
                        samples={})
            oracle = OracleBand(decimal_rows(csr_matrix(a)),len(a)-1) if k in decimal_modes else None
            rows = decimal_rows(csr_matrix(a)) if oracle is not None else None
            for workload,rhs in workloads.items():
                truth = complex_vector(oracle.solve(decimal_vector(rhs))) if oracle is not None else None
                metrics = {}
                for name,solve in solver.items():
                    result = solve(rhs)
                    record = dict(componentwise_on_authoritative_triangle=backward_float(a,result,rhs),
                                  componentwise_on_original_assembly=backward_float(stored,result,rhs))
                    if workload == "manufactured":
                        record["known_solution_forward_relative"] = relative(result,x)
                    if truth is not None:
                        record.update(decimal_forward_relative=relative(result,truth),**backward(a,rows,result,rhs))
                    metrics[name] = record
                mode["samples"][workload] = metrics
            tests.append(mode)
        output[label] = tests
    return output


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protocol = read(PROTOCOL)
    runs = []
    sources = {PROTOCOL.relative_to(V7.parent).as_posix():digest(PROTOCOL)}
    with localcontext() as context:
        context.prec = 65
        for scale,identities in METHODS.items():
            schedule = protocol["schedules"][scale]
            phases = phase_specs(schedule)
            dts = tuple(p[0] for p in phases)
            for name,seed in identities:
                c = next(x for x in protocol["controls"] if (x["resolution"],x["name"],x["seed"])==(scale,name,seed))
                _,settings = config(protocol["specs"][name],protocol["solve_protocol"],seed)
                t = perf_counter()
                model = RecurrenceGalerkin(dt=dts[0],**settings)
                built = perf_counter()
                blocks = spatial_blocks(model)
                assembled = perf_counter()
                held,path = held_states(c,model,schedule)
                sources[path.relative_to(V7.parent).as_posix()] = digest(path)
                packs,pencils,costs = measure(model,blocks,dts)
                fixed = fixed_systems(model,blocks,phases,pencils,held)
                physical = field_errors(model,held,phases,packs)
                rates = [schedule[k] for k in ("startup_rate","late_rate")]
                runs.append(dict(scale=scale,name=name,seed=seed,rates=rates,effective_dt=dict(zip(LABELS,dts,strict=True)),
                    same_rate_anchor=dict(minimum_denominator=min(float(min(p.diagonal(1.5,p.anchor_dt))) for p in pencils),
                        maximum_denominator=max(float(max(p.diagonal(1.5,p.anchor_dt))) for p in pencils)),
                    width=model.width,modes=model.k+1,quadrature_nodes=len(model.r),
                    common_v7_model_seconds=built-t,shared_assembly_seconds=assembled-built,costs=costs,
                    normalized_cross_mode_commutators=commutation(model,blocks),pack_differences=differences(packs),
                    fixed_systems=fixed,held_physical=physical,
                    fixed_field_gate_pass=all(x["worst_field"]<=GATE for h in physical.values() for l in h.values() for x in l.values())))
                print(scale,name,{k:round(v["median"],5) for k,v in costs["seconds"].items()},
                      "worst held",max(x["worst_field"] for h in physical.values() for l in h.values() for x in l.values()),flush=True)
    for name in ("galerkin_phase_pencil.py","audit_galerkin_phase_pencil.py","theory_galerkin_phase_pencil.txt"):
        sources[(HERE/name).relative_to(V7.parent).as_posix()] = digest(HERE/name)
    for name in ("galerkin_phase_shared.py","galerkin_cuda_native.py","cpu_sbdf2_switch.py",
                 "fd_equilibration_precision.py","cpu_sbdf2_switch_decimal.py","cpu_formal.py"):
        sources[(V7/name).relative_to(V7.parent).as_posix()] = digest(V7/name)
    for name in ("full_galerkin.py","full_nonlinear.py"):
        path = V7.parent/"rfm_v6"/name
        sources[path.relative_to(V7.parent).as_posix()] = digest(path)
    save(OUTPUT,dict(status="complete",decimal_digits=65,fixed_field_gate=GATE,source_hashes=sources,runs=runs,
        scope="CPU fixed-matrix and held-state comparisons. The same binary64 upper triangle is authoritative. "
              "Manufactured RHS roundoff is distinct from exact-assembled Decimal truth. No time stepping or GPU timing. "
              "Five-sample setup diagnostics are not isolated. Startup observations are held states, not consecutive startup steps. "
              "Two archived consecutive late states test a second held input. Same-rate anchor checks use the exact denominator 1.",
        retained_failed_r112="rfm_v7/startup_replay_output/release_replay_status.json"))


if __name__ == "__main__":
    main()
