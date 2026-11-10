"""Verify stored event fits, immutable inputs, and common-norm identities."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
read=lambda path:json.loads(path.read_text(encoding='utf8'))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf8')

probe=read(HERE/'probe_diagnostics.json')
assert all(sha(WORK/record['path'])==record['sha256'] for record in probe['inputs'].values())
assert probe['source_sha256']['analyze_saved_probes.py']==sha(HERE/'analyze_saved_probes.py')
crossings=read(HERE/'crossing_records.json')['rows']
assert len(crossings)==1008
differences=[]
for row in crossings:
    c=np.asarray(row['crossings']); index=np.arange(len(c))
    # Independent centered covariance expression, rather than polyfit.
    slope=np.sum((index-index.mean())*(c-c.mean()))/np.sum((index-index.mean())**2)
    difference=abs(1/slope-row['frequency']); differences.append(difference)
    assert difference<1e-14
    assert len(c)>=4 and np.all(np.diff(c)>0)
with (HERE/'probe_frequency_contrasts.csv').open(encoding='utf8',newline='') as stream:
    contrasts=list(csv.DictReader(stream))
assert len(contrasts)==1872
assert all(float(row['relative_frequency_bias'])<0 for row in contrasts if 'no-layers' in row['role'])

vectors=read(HERE/'switch_error_vector_identity.json')
assert len(vectors['results'])==16
assert vectors['source_sha256']['analyze_switch_error_vectors.py']==sha(HERE/'analyze_switch_error_vectors.py')
assert all(sha(WORK/name)==expected for name,expected in vectors['imported_local_source_hashes'].items())
residuals=[]
for row in vectors['results']:
    residual=abs(row['error8_squared']-row['error12_squared']-row['cross_term']-row['delta_squared'])
    residuals.append(residual/row['error8_squared'])
    assert residual/row['error8_squared']<1e-12
receipt=dict(status='passed',new_PDE_trajectories=0,new_reference_trajectories=0,new_GPU_runs=0,new_timing_measurements=0,
    input_probe_arrays_hash_verified=len(probe['inputs']),probe_crossing_records=len(crossings),reference_contrasts=len(contrasts),
    independent_frequency_fit_max_absolute_difference=max(differences),no_layer_frequency_bias_negative_in_all_variants=True,
    switch_error_vector_cases=len(vectors['results']),maximum_identity_residual_relative_to_squared_error=max(residuals),
    imported_reconstruction_sources_verified=len(vectors['imported_local_source_hashes']),
    source_sha256={Path(__file__).name:sha(Path(__file__))})
write(HERE/'verification.json',receipt)
files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json')
write(HERE/'artifact_manifest.json',dict(status='complete',scope='Analysis of saved arrays only',
    files_sha256={p.relative_to(HERE).as_posix():sha(p) for p in files}))
print(json.dumps(receipt,indent=2))
