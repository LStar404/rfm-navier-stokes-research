"""Audit frozen v6 evidence and build a checksummed reproducibility archive."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import uuid
import zipfile


ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
OUTPUT=WORKSPACE/'outputs'
REPORT=OUTPUT/'RFM_NS_v6_全域涡脱落复测.pdf'
ARCHIVE=OUTPUT/'RFM_NS_v6_复现实验.zip'
SEARCH=(ROOT,*(ROOT.parent/p for p in ('rfm_v5','rfm_v4','rfm_v3','rfm_ns','report_v2')))
SUFFIXES={'.py','.c','.dll','.so','.json','.txt','.md','.log','.csv','.npz','.svg','.png'}
ERROR_KEYS=('velocity','vorticity','near_velocity','near_vorticity')


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read(path):
    def nonfinite(value):
        raise ValueError(f'{path.name} contains {value}')
    return json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=nonfinite)


def member(path):
    return path.resolve().relative_to(WORKSPACE).as_posix()


def include(files,path):
    require(path.is_file(),f'Missing evidence: {path}')
    files[member(path)]=path.resolve()


def check_sources(files,hashes):
    require(bool(hashes),'Missing frozen source provenance')
    for name,expected in hashes.items():
        path=WORKSPACE/name
        include(files,path)
        require(digest(path)==expected,f'Frozen input changed: {name}')


def same(a,b):
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-15)


def check_errors(errors,scores,frames):
    require(Counter((r['grid'],r['time']) for r in scores)==frames,
            'Incorrect grid and checkpoint coverage')
    for key in ERROR_KEYS:
        values=[r[key] for r in scores]
        require(all(math.isfinite(x) and x>=0 for x in values)
                and same(max(values),errors[key]),f'Invalid error maximum: {key}')


def check_formal(files):
    path=ROOT/'full_formal.json'
    include(files,path)
    data=read(path)
    require(data['status']=='complete','Formal comparison is not complete')
    check_sources(files,data['hashes'])
    specs,seeds,rows=data['specs'],data['seeds'],data['rows']
    require(read(ROOT/'full_specs.json')==specs,'Frozen specifications differ')
    names=[s['name'] for s in specs]
    require(len(names)==len(set(names)) and len(seeds)==len(set(seeds)) and seeds,
            'Repeated or empty specs/seeds')
    schedule=[(seed,names[index]) for seed,index in data['schedule']]
    require(Counter(schedule)==Counter((seed,name) for seed in seeds for name in names),
            'Incomplete formal schedule')
    require([(r['seed'],r['name']) for r in rows]==schedule,'Run rows differ from schedule')
    require(data['thread_environment']==dict.fromkeys(
        ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'),'1')
        and isinstance(data['cpu'],int) and data['cpu']>=0,'Missing pinned single-thread protocol')
    reference_scores=data['reference_scores']
    grids={r['grid'] for r in reference_scores}
    times={r['time'] for r in reference_scores}
    require(grids=={'gauss123','mid129','layer144'} and len(times)==7,'Incomplete reference score grids')
    frames=Counter((grid,t) for grid in grids for t in times)
    check_errors(data['reference_change'],reference_scores,frames)
    screens={str(g):max(data['reference_change'].values())<=.25*g for g in data['gates']}
    require(screens==data['gate_screens'],'Incorrect reference refinement screen')
    require(data['reference_wake']['sustained_alternating_probe_signs'],
            'Reference has no verified alternating wake')
    for key in ('reference','predecessor'):
        ref=ROOT/data[key]
        require(member(ref) in data['hashes'],f'Missing frozen {key} provenance')
    artifact_count=0
    failures=[]
    for row in rows:
        require(math.isfinite(row['seconds']) and row['seconds']>0,'Invalid runtime')
        if 'errors' not in row:
            require(row.get('failure') or row.get('error'),'Failed run lacks explanation')
            failures.append(dict(seed=row['seed'],name=row['name']))
            continue
        check_errors(row['errors'],row['scores'],frames)
        bounds=row['boundary_errors']
        require(set(bounds)=={'wall_ur','wall_ut','outer_ur_error','outer_vorticity'}
                and all(math.isfinite(x) and x>=0 for x in bounds.values()),
                'Missing or invalid wall/far-field errors')
        require(0<row['metadata']['setup_seconds']<=row['seconds'],'Invalid setup runtime')
        artifact=row.get('artifact')
        require(bool(artifact)==(row['seed']==seeds[0]),'First-seed state evidence differs')
        if artifact:
            path=ROOT/artifact['file']
            include(files,path)
            require(digest(path)==artifact['sha256'],f'Changed state: {artifact["file"]}')
            artifact_count+=1
    require([r['name'] for r in data['summary']]==names,'Summary differs from spec order')
    for summary in data['summary']:
        selected=[r for r in rows if r['name']==summary['name'] and 'errors' in r]
        times=[r['seconds'] for r in selected]
        require(summary['completed']==len(selected) and summary['times']==times,
                f'Incomplete summary: {summary["name"]}')
        if selected:
            require(same(summary['seconds'],statistics.median(times)),'Median runtime mismatch')
            require(all(same(max(r['errors'][k] for r in selected),summary['worst'][k])
                        for k in ERROR_KEYS),'Summary error maximum mismatch')
        shedding=len(selected)==len(seeds) and all(
            row['wake']['sustained_alternating_probe_signs'] for row in selected)
        require(summary['shedding']==shedding,'Summary shedding differs')
        numerical={str(g):shedding and max(summary['worst'].values())<=g for g in data['gates']}
        require(summary['numerical_pass']==numerical,'Numerical pass differs')
        accepted={gate:passed and screens[gate] for gate,passed in numerical.items()}
        require(accepted==summary['passes'],'Same-accuracy acceptance mismatch')
    return dict(seeds=seeds,configurations=len(names),attempts=len(rows),failures=failures,
                saved_states=artifact_count,reference=data['reference'],reference_screens=screens,
                formal_sha256=digest(ROOT/'full_formal.json'))


def add_imports(files):
    """Follow local import dependencies while preserving the workspace layout."""
    pending=[p for p in files.values() if p.suffix=='.py']
    seen=set()
    while pending:
        path=pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):
                names=[n.name.split('.')[0] for n in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module.split('.')[0]]
            else:
                continue
            for name in names:
                for directory in (path.parent,*SEARCH):
                    dep=directory/f'{name}.py'
                    if dep.is_file():
                        include(files,dep)
                        pending.append(dep.resolve())
                        break


def collect(require_report):
    files={}
    formal=check_formal(files)
    wake=read(ROOT/'diagnostic_fine.json')
    require(wake['passed'],'Missing verified dense spatial wake diagnostic')
    for name in ('diagnostic_fine.json',wake['state_source'],wake['probe_source'],
                 'figures/diagnostic_wake.png','full_energy_audit.json'):
        include(files,ROOT/name)
    for path in ROOT.rglob('*'):
        local=path.relative_to(ROOT)
        if not path.is_file() or {'__pycache__','render','scratch'}&set(local.parts):
            continue
        if path.suffix.lower() in SUFFIXES and path.name!='package_v6_receipt.json':
            include(files,path)
    if require_report:
        include(files,REPORT)
    add_imports(files)
    require((ROOT/'README.txt').is_file(),'Missing reproduction instructions')
    files['README.txt']=ROOT/'README.txt'
    return dict(sorted(files.items())),formal


def verify(path):
    with zipfile.ZipFile(path) as archive:
        members=archive.namelist()
        require(len(members)==len(set(members)),'Duplicate archive members')
        for name in members:
            target=PurePosixPath(name)
            require(not target.is_absolute() and '..' not in target.parts and '\\' not in name
                    and ':' not in name,f'Unsafe archive path: {name}')
        require(archive.testzip() is None,'Archive CRC check failed')
        manifest=json.loads(archive.read('MANIFEST.json'))
        require(set(members)=={'MANIFEST.json',*(r['path'] for r in manifest['files'])},
                'Archive manifest mismatch')
        for row in manifest['files']:
            info=archive.getinfo(row['path'])
            with archive.open(info) as handle:
                checksum=hashlib.file_digest(handle,'sha256').hexdigest()
            require(info.file_size==row['size'] and checksum==row['sha256'],
                    f'Member integrity failed: {row["path"]}')
    return dict(path=str(path),size=path.stat().st_size,sha256=digest(path),members=len(members)-1)


def package():
    if ARCHIVE.exists():
        raise FileExistsError(ARCHIVE)
    files,formal=collect(require_report=True)
    entries=[dict(path=n,size=p.stat().st_size,sha256=digest(p)) for n,p in files.items()]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),formal=formal,files=entries)
    temporary=ARCHIVE.with_name(f'{ARCHIVE.name}.{uuid.uuid4().hex}.part')
    try:
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
            for name,path in files.items():
                archive.write(path,name)
            archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False))
        verify(temporary)
        os.replace(temporary,ARCHIVE)
    finally:
        temporary.unlink(missing_ok=True)
    receipt=verify(ARCHIVE)
    receipt['formal']=formal
    (ROOT/'package_v6_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true',help='Audit without compression or a final PDF.')
    parser.add_argument('--verify',type=Path,help='Check a previously created archive.')
    args=parser.parse_args()
    if args.verify:
        result=verify(args.verify)
    elif args.preflight:
        files,formal=collect(require_report=False)
        result=dict(formal=formal,files=len(files),bytes=sum(p.stat().st_size for p in files.values()))
    else:
        result=package()
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
