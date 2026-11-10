"""Verify the regression check catches the two alleged classes of formula error."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
registry=json.loads((HERE/'canonical_equations.json').read_text(encoding='utf8'))
records=[]
with tempfile.TemporaryDirectory(prefix='formula_check_fixture_',dir=HERE) as tmp:
    root=Path(tmp).resolve()
    assert root.is_relative_to(HERE.resolve())
    sections=root/'sections'; sections.mkdir()
    equations=root/'equations'; equations.mkdir()
    grouped={}
    for label,item in registry['equations'].items():
        content='\\begin{equation}\n'+item['equation_body']+'\n\\end{equation}\n'
        name=label.split(':',1)[1].replace('-','_')+'.tex'
        (equations/name).write_text(content,encoding='utf8')
        grouped.setdefault(item['section_stem'],[]).append(name)
    for stem,names in grouped.items():
        for lang in ('zh','en'):
            (sections/f'{lang}_{stem}.tex').write_text(''.join(f'\\input{{equations/{n}}}\n' for n in names),encoding='utf8')
    def run(name,expected,extra=None):
        output=root/'receipt.json'
        completed=subprocess.run([sys.executable,str(HERE/'verify_bilingual_equations.py'),
                                  '--project',str(root),'--output',str(output)],capture_output=True,text=True)
        result=json.loads(output.read_text(encoding='utf8'))
        assert (completed.returncode==0)==expected,(name,completed.stdout,completed.stderr)
        records.append({'case':name,'expected_pass':expected,'actual_pass':completed.returncode==0,
                        'bilingual_equal':all(r['bilingual_equal'] for r in result['results']),
                        'baseline_matches':all(r['matches_audited_revision3_math'] for r in result['results'])})
    run('shared_input_preserves_correct_baseline',True)
    profile=sections/'zh_review_profiles_bridge.tex'
    correct=profile.read_text(encoding='utf8')
    altered=(equations/'review_profile_norm.tex').read_text(encoding='utf8').replace('|e_f(r_i,\\cdot)|^2','|e_f(r_i,\\cdot)|',1)
    profile.write_text(altered+'\\input{equations/review_modal_error.tex}\n',encoding='utf8')
    run('one_language_missing_error_square',False)
    profile.write_text(correct,encoding='utf8')
    amort=equations/'schedule_amortization.tex'
    correct=amort.read_text(encoding='utf8')
    altered=correct.replace('\\Delta T(N)=\\Delta P-','\\Delta T(N)=',1)
    assert altered!=correct
    amort.write_text(altered,encoding='utf8')
    run('same_missing_preparation_term_in_both_languages',False)
    amort.write_text(correct,encoding='utf8')
    run('restored_baseline',True)
receipt={'status':'passed','cases':records,
         'scope':'Synthetic source fixtures only; no manuscript, archived data or PDE trajectory modified.'}
(HERE/'checker_selftest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
