"""Post-ablation construction-only capacity audit requested by the reviewer response.

No trajectory, cutoff, quadrature or bandwidth is tuned.  Direct raw jet SVD
avoids the old constructor guard m+6<q, which is overly restrictive for this
diagnostic because the six nonzero-mode columns are absent.
"""
import json
from pathlib import Path
from ablation_model import AblationGalerkin, np
from cylinder_rfm import Trial
from radial_layers import radial_quadrature

HERE = Path(__file__).resolve().parent
protocol = json.loads((HERE / "protocol.json").read_text())
rows = []
for resolution, target in (("mid", 39), ("high", 40)):
    job = next(j for j in protocol["jobs"] if j["resolution"]==resolution and j["role"]=="midpoint-no-layers")
    model = AblationGalerkin(dt=72/job["steps"], **job["config"])
    _, weights = radial_quadrature(job["config"]["q"], True)
    for m in (41, 45, 49, 53, 57, 61):
        # This is an instance-local reconstruction, not a solver-module patch.
        model.trial = Trial(m, "rfm", 7331, 62., 0., "inner", "farfield", sampling="paired")
        model._frequency_ready = False
        raw = model.raw_jets(model.s)
        values = (raw[2]-raw[0]) * (np.sqrt(weights)/model.r)[:,None]
        singular = np.linalg.svd(values/np.linalg.norm(values,axis=0)[None,:], compute_uv=False)
        rank = int(np.count_nonzero(singular>1e-10*singular[0]))
        rows.append(dict(resolution=resolution, raw_m=m, q=job["config"]["q"],
            frequency_scale=62, xi=.5, nonzero_layers=False, relative_cutoff=1e-10,
            rank=rank, target_enriched_rank=target, same_rank=rank==target,
            normalized_singular_values=singular.tolist(),
            constructor_raw_width_guard_would_reject=m+6>=job["config"]["q"]))
result = dict(status="construction-only", candidate_m=[41,45,49,53,57,61],
    cutoff_quadrature_bandwidth_unchanged=True, trajectories_run=0, rows=rows)
(HERE / "capacity_construction.json").write_text(json.dumps(result,indent=2))
print(json.dumps([{k:v for k,v in r.items() if k!="normalized_singular_values"} for r in rows],indent=2))
