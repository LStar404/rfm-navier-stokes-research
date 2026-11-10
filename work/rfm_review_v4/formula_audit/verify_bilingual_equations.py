"""Check the three audited mathematical definitions in both manuscript editions.

This is a source-equivalence regression check, not a general TeX theorem prover.
It resolves local \\input directives so editions may share one formula source.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def expand(path,project,active=()):
    path=path.resolve()
    if path in active:
        raise ValueError(f'cyclic input: {path}')
    text=path.read_text(encoding='utf8')
    def replace(match):
        rel=match.group(1)
        if not Path(rel).suffix:
            rel += '.tex'
        target=project/rel
        if not target.is_file():
            target=path.parent/rel
        return expand(target,project,active+(path,))
    return re.sub(r'\\input\{([^}]+)\}',replace,text)

def normalize(body):
    body=re.sub(r'\\label\{[^}]*\}','',body)
    body=re.sub(r'(?<!\\)%[^\n]*','',body)
    return re.sub(r'\s+','',body)

def extract(text,label):
    matches=[m.group(1) for m in re.finditer(r'\\begin\{equation\}(.*?)\\end\{equation\}',text,re.S)
             if f'\\label{{{label}}}' in m.group(1)]
    if len(matches)!=1:
        raise ValueError(f'{label}: expected one equation, got {len(matches)}')
    return normalize(matches[0])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=ROOT/'outputs/rfm_ns_tex')
    parser.add_argument('--output',type=Path,default=HERE/'bilingual_equation_check.json')
    args=parser.parse_args()
    baseline=json.loads((HERE/'canonical_equations.json').read_text(encoding='utf8'))
    rows=[]
    for label,expected in baseline['equations'].items():
        values={}
        for lang in ('zh','en'):
            path=args.project/'sections'/f'{lang}_{expected["section_stem"]}.tex'
            values[lang]=extract(expand(path,args.project),label)
        identical=values['zh']==values['en']
        canonical=all(v==expected['normalized_body'] for v in values.values())
        rows.append({'label':label,'bilingual_equal':identical,'matches_audited_revision3_math':canonical,
                     'normalized_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in values.items()}})
    result={'status':'passed' if all(r['bilingual_equal'] and r['matches_audited_revision3_math'] for r in rows) else 'failed',
            'baseline_commit':baseline['source_commit'],'equations_checked':len(rows),'results':rows,
            'scope':'Critical equation bodies after local input expansion; only labels/comments/whitespace ignored.'}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status']!='passed':
        raise SystemExit(1)

if __name__=='__main__':
    main()
