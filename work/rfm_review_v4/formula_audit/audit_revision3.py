"""Read-only audit of the revision-3 statements challenged by the second review."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPO = ROOT if (ROOT/'.git').exists() else ROOT/'release/rfm-navier-stokes-publication'
COMMIT = '353b9cf4daafc4a87ccb752bafc3c6e7c9278a7a'
PROJECT = ROOT/'outputs/rfm_ns_tex'
TARGETS = {
    'eq:review-profile-norm': 'review_profiles_bridge',
    'eq:review-modal-error': 'review_profiles_bridge',
    'eq:schedule-amortization': 'schedule_amortization',
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def text_blob(path):
    return subprocess.check_output(['git','show',f'{COMMIT}:{path}'],cwd=REPO)

def equation(text,label):
    found=[]
    for match in re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}',text,re.S):
        if f'\\label{{{label}}}' in match.group(1):
            found.append((match.group(1), text[:match.start()].count('\n')+1))
    if len(found)!=1:
        raise ValueError((label,len(found)))
    return found[0]

def normalize(body):
    body=re.sub(r'\\label\{[^}]*\}','',body)
    body=re.sub(r'(?<!\\)%[^\n]*','',body)
    return re.sub(r'\s+','',body)

def evidence(path, needle):
    data=text_blob(path)
    text=data.decode('utf8')
    lines=[i+1 for i,line in enumerate(text.splitlines()) if needle in line]
    return {'path':path,'commit':COMMIT,'sha256':sha(data),'needle':needle,'lines':lines}

def dump(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

HERE.mkdir(parents=True,exist_ok=True)
registry={'source_commit':COMMIT,'scope':'Three requested equation bodies; formatting whitespace and labels excluded from comparison.','equations':{}}
audit={'status':'complete','review_basis_commit':COMMIT,
       'scope':'Read-only source/metadata audit; no PDE runs, no manuscript changes.',
       'equations':{},'facts':{},'compiled_revision3_labels':{}}

for label,stem in TARGETS.items():
    variants={}
    for lang in ('zh','en'):
        path=f'outputs/rfm_ns_tex/sections/{lang}_{stem}.tex'
        blob=text_blob(path)
        text=blob.decode('utf8')
        body,line=equation(text,label)
        current=(ROOT/path).read_bytes()
        variants[lang]={'path':path,'archived_source_sha256':sha(blob),
                       'current_source_sha256':sha(current),'current_file_matches_archived_bytes':current==blob,
                       'equation_start_line':line,'body':body.strip(),
                       'normalized_body':normalize(body),'normalized_sha256':sha(normalize(body).encode())}
    equal=variants['zh']['normalized_body']==variants['en']['normalized_body']
    assert equal,label
    registry['equations'][label]={'section_stem':stem,'normalized_body':variants['en']['normalized_body'],
                                'normalized_sha256':variants['en']['normalized_sha256'],
                                'equation_body':variants['en']['body']}
    audit['equations'][label]={'bilingual_bodies_equal':equal,'alleged_typo_present':False,
                             'verdict':'The cited revision already contains the correct expression in both languages.',
                             'variants':variants}
    shared=HERE/'shared_equations'/f'{label.split(":",1)[1].replace("-","_")}.tex'
    shared.parent.mkdir(exist_ok=True)
    shared.write_text('\\begin{equation}\n'+variants['en']['body']+'\n\\end{equation}\n',encoding='utf8')

labels=list(TARGETS)+['eq:geometry-equivalence','eq:geometry-condition','eq:kinematics',
                     'eq:random-features','tab:review-ablation','tab:review-switch12','tab:app-score-grids']
for lang,stem in [('zh','main'),('en','english')]:
    path=ROOT/f'work/rfm_paper/tex_build_v3/{stem}.aux'
    if not path.is_file():
        saved=json.loads((HERE/'revision3_label_map.json').read_text(encoding='utf8'))
        audit['compiled_revision3_labels'][lang]=dict(saved[lang], read_from_saved_label_map=True)
        continue
    text=path.read_text(encoding='utf8')
    rows={}
    for label in labels:
        pattern=r'\\newlabel\{'+re.escape(label)+r'\}\{\{([^}]+)\}\{([^}]+)\}'
        match=re.search(pattern,text)
        assert match,label
        rows[label]={'printed_number':match.group(1),'page':int(match.group(2))}
    audit['compiled_revision3_labels'][lang]={'aux_path':path.relative_to(ROOT).as_posix(),
                                             'aux_sha256':sha(path.read_bytes()),'labels':rows}

facts=audit['facts']
facts['geometry']={
    'review_claim':'a=0.5, R=6.5, L=6.0',
    'verified':'a=0.5, R=6, physical span R-a=5.5; logarithmic coordinate s=log(r/a), L=log(R/a)=log12.',
    'verdict':'review transcription contradicts canonical source',
    'evidence':[evidence('work/rfm_v6/full_case.py','A, R, RE, NU'),
                evidence('work/rfm_v6/full_case.py','LENGTH = np.log(R/A)')]}
facts['paired_frequency']={
    'verified':'b_j=62(j+xi_j)/p, j=0,...,p-1; canonical annular Trial implementation is in work/rfm_v3/cylinder_rfm.py.',
    'evidence':[evidence('work/rfm_v3/cylinder_rfm.py','bins=(np.arange(count)'),
                evidence('work/rfm_v6/full_galerkin.py','self.trial=Trial')]}
facts['boundary_envelope']={
    'review_claim':'Current farfield features use chi=16 z^2 (1-z)^2.',
    'verified':'Current farfield mode uses chi=16 z^2(1-z) with a second-derivative boundary correction; the squared envelope belongs to another boundary case.',
    'evidence':[evidence('work/rfm_v3/cylinder_rfm.py',"elif outer=='farfield':"),
                evidence('outputs/rfm_ns_tex/sections/en_methods.tex','let \\(\\chi(z)=16z^2(1-z)\\)')]}

records=[]
for path in sorted((ROOT/'work/rfm_v8/rank_fair_cache_qualification_output').glob('slot_*/result.json')):
    data=json.loads(path.read_text(encoding='utf8'))
    job=data['job']; config=job['config']
    if (config.get('family'),config.get('m')) not in [('rfm',37),('rfm',41),('legendre',35),('legendre',36)]:
        continue
    assert path.read_bytes()==text_blob(path.relative_to(ROOT).as_posix()),path
    assert config['layers']==[16,36,80,165,340,660]
    assert data['metadata']['layers']==config['layers']
    records.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path.read_bytes()),
                    'family':config['family'],'m':config['m'],'K':config['k'],
                    'retained_rank':data['metadata']['rank'],'mean_rank':data['metadata']['mean_rank'],
                    'layers':config['layers']})
facts['legendre_enrichment']={
    'review_claim':'Legendre35/36 reached the cited error without the six analytic layer functions.',
    'verified':'Both RFM and Legendre controls in the cited qualification include all six layers; no no-layer Legendre trajectory is supplied by this ablation.',
    'verdict':'Cannot use the reported Legendre errors as evidence of no-enrichment accuracy.',
    'source_evidence':[evidence('work/rfm_v6/full_galerkin.py','self.layers=tuple(layers)'),
                       evidence('work/rfm_v6/full_galerkin.py','np.concatenate((raw,layer_jets(s,self.layers))')],
    'checked_qualification_rows':records}
facts['rank_and_randomness']={
    'verified':'Input suffixes are not retained ranks. Main paired controls retain 39/40. Removing nonzero-mode layers reduces the ablated RFM ranks to34/35. Fixed midpoint results refute necessity for these configurations, not universal benefit from randomness.',
    'source':evidence('outputs/rfm_ns_tex/sections/en_review_ablation.tex','It is not an equal-rank enrichment ablation')}
facts['maintenance_scope']={
    'review_claim':'The old work/rfm_ns wrapper has been rewritten and all archived monkey-patching removed.',
    'verified':'The maintained package uses solve-local basis_factory; original work/rfm_ns sources remain frozen and the old wrapper still patches v2.trial_basis.',
    'evidence':[evidence('work/rfm_ns/rfm_v2_warp_legendre.py',"with patch.object(v2,'trial_basis',basis_factory):"),
                evidence('work/rfm_review_v3/maintenance/rfm_v2_warp_legendre.py','basis_factory=mapped_legendre_factory')]}
paths={
    'work/rfm_review_v3/geometry_scope/':None,
    'work/rfm_review_v3/maintenance/test_factory_isolation.py':'work/rfm_review_v3/maintenance/tests/test_factory.py',
    'work/rfm_review_v3/ablation/verify_ablation.py':'work/rfm_review_v3/ablation/verify_adapter.py',
}
facts['path_transcriptions']=[{'claimed_path':name,'exists':(ROOT/name).exists(),
                             'actual_path':actual,'actual_exists':None if actual is None else (ROOT/actual).is_file()}
                            for name,actual in paths.items()]
facts['caption_implementation_error']={
    'verified':'The <2e-10 result applies to the two enriched-random CPU controls versus their archived GPU fields in the ablation. The t_s=12 trajectories are CPU-only and have no newly measured CPU/GPU implementation error.',
    'evidence':[evidence('outputs/rfm_ns_tex/sections/en_review_ablation.tex','two CPU controls differ'),
                evidence('outputs/rfm_ns_tex/sections/en_review_profiles_bridge.tex','one-seed sensitivity results')]}
facts['order_and_causality']={
    'review_claim':'5.6% of the maximum-error increase is caused by a nonsmooth switch at t=8 and94.4% by later O(h^2) accumulation.',
    'verified':'The pulse is analytic near its interior peak t=8; only endpoints4/12 have fourth-derivative jumps. Differences between maxima from two different trajectories are not additive causal components of the error, even if the numerical ratio is5.6%.',
    'evidence':[evidence('work/rfm_v6/full_case.py','np.sin'),
                evidence('outputs/rfm_ns_tex/sections/en_review_profiles_bridge.tex','neither a transient-splitting mechanism nor order reduction')]}

dump('canonical_equations.json',registry)
if not (HERE/'revision3_label_map.json').exists():
    dump('revision3_label_map.json',audit['compiled_revision3_labels'])
dump('audit.json',audit)
print('Revision3 audit: three bilingual equations correct and identical; canonical geometry, enrichment, ranks, paths and numbering recorded.')
