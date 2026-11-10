"""CPU-only AST-isolated integrity tests; no report rendering or publication writes."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = HERE / 'build_v8_report.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def fixture(root, kind='derivative', include_map=True):
    area = root / 'work/rfm_v8'
    area.mkdir(parents=True)
    public = area / 'example.public.json'
    public.write_text('{"result":"unchanged public value"}\n', encoding='utf-8')
    expected = digest(public) if kind == 'historical' else hashlib.sha256(b'private original bytes').hexdigest()
    original = 'work/rfm_v8/example_original.json'
    name = 'publication_historical_source_map.json' if kind == 'historical' else 'publication_path_redaction_map.json'
    mapping = area / name
    row = dict(original_path=original, original_sha256=expected,
               public_path=public.relative_to(root).as_posix(), public_sha256=digest(public))
    schema = 'rfm-v8-historical-source-map-1' if kind == 'historical' else 'rfm-v8-public-derivative-map-1'
    save(mapping, dict(schema=schema, records=[row]))
    members = [public, mapping] if include_map else [public]
    save(root / 'MANIFEST.json', dict(files=[dict(path=p.relative_to(root).as_posix(),
        bytes=p.stat().st_size, sha256=digest(p)) for p in members]))
    return original, expected, public, mapping


def run(output):
    if output.exists():
        raise FileExistsError('Use a new fixture output directory')
    output.mkdir()
    tree = ast.parse(BUILDER.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resolve_review_source')
    code = compile(ast.Module(body=[node], type_ignores=[]), '<frozen-resolver-ast>', 'exec')
    records = []
    cases = ('public_derivative', 'exact_historical_alias', 'tampered_public', 'unmanifested_map',
             'wrong_original_sha', 'changed_manifest_after_verification', 'historical_changed_bytes', 'ambiguous_mapping')
    for name in cases:
        root = output / name
        original, expected, public, mapping = fixture(root,
            kind='historical' if name in ('exact_historical_alias', 'historical_changed_bytes') else 'derivative',
            include_map=name != 'unmanifested_map')
        env = dict(ROOT=root.resolve(), HERE=(root/'work/rfm_v8').resolve(), Path=Path,
                   hashlib=hashlib, json=json, PUBLIC_PACKAGE_VERIFIED=False,
                   PUBLIC_MANIFEST_SHA256=None, PUBLIC_RESOLUTIONS=[])
        exec(code, env)
        resolve = env['resolve_review_source']
        if name == 'tampered_public':
            public.write_text('tampered', encoding='utf-8')
        elif name == 'wrong_original_sha':
            expected = '0' * 64
        elif name in ('historical_changed_bytes', 'ambiguous_mapping'):
            data = json.loads(mapping.read_bytes())
            if name == 'historical_changed_bytes':
                expected = data['records'][0]['original_sha256'] = '0' * 64
            else:
                data['records'].append(dict(data['records'][0]))
            save(mapping, data)
            manifest = json.loads((root/'MANIFEST.json').read_bytes())
            for row in manifest['files']:
                if row['path'] == mapping.relative_to(root).as_posix():
                    row.update(bytes=mapping.stat().st_size, sha256=digest(mapping))
            save(root/'MANIFEST.json', manifest)
        elif name == 'changed_manifest_after_verification':
            resolve(original, expected)
            manifest = json.loads((root/'MANIFEST.json').read_bytes())
            manifest['changed_after_verification'] = True
            save(root/'MANIFEST.json', manifest)
        error = None
        try:
            resolved, observed = resolve(original, expected)
            assert resolved == public.resolve() and observed == digest(public)
        except ValueError as exc:
            error = str(exc)
        positive = name in ('public_derivative', 'exact_historical_alias')
        assert (error is None) == positive, (name, error)
        if positive:
            proof = env['PUBLIC_RESOLUTIONS'][-1]
            assert proof['original_sha256'] == expected and proof['public_sha256'] == digest(public)
        records.append(dict(case=name, passed=True, resolution_accepted=positive, rejection_reason=error))
    report = dict(status='passed', scope='AST-extracted resolver only; no reportlab import, GPU, report generation or release mutation.',
        builder_sha256=digest(BUILDER), verifier_sha256=digest(Path(__file__)), cases=records)
    save(output/'certificate.json', report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps(dict(status=report['status'], cases=len(report['cases']))))
