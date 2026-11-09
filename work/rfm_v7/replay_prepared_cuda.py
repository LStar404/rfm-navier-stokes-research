"""Replay a frozen prepared CUDA trajectory into new files; no administrative inputs."""
import argparse
import json
from pathlib import Path

import replay_switch_round as common
from galerkin_cuda_prepared import solve_cuda_prepared

base, case, np = common.base, common.case, common.np
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'prepared_qualification_output'

def execute(args):
    output, sidecar = args.output, args.json_output
    if output.suffix != '.npz' or (sidecar is not None and sidecar.suffix != '.json'):
        raise ValueError('Use a new .npz output and optional new .json sidecar')
    if output.exists() or (sidecar is not None and sidecar.exists()):
        raise FileExistsError('Do not overwrite an archived or previous replay')
    protocol = base.read(ARCHIVE / 'protocol.json')
    qualification = base.read(ARCHIVE / 'qualification.json')
    if (protocol['status'] != 'frozen' or qualification['status'] != 'complete'
            or qualification['protocol_sha256'] != base.digest(ARCHIVE / 'protocol.json')
            or qualification['runs_sha256'] != base.digest(ARCHIVE / 'runs.json')):
        raise ValueError('Numerical archive protocol differs')
    seed = args.seed if args.seed is not None else (7331 if args.method.startswith('rfm') else 17)
    candidates = [r for r in qualification['rows'] if r['job']['name'] == args.method
                  and r['job']['seed'] == seed and r['job']['preparation'] == args.preparation]
    if len(candidates) != 1:
        raise ValueError('Choose a registered method, seed and preparation combination')
    row = candidates[0]
    if row['job'] not in protocol['jobs']:
        raise ValueError('Selected trajectory was not preregistered')
    runs = base.read(ARCHIVE / 'runs.json')
    sidecar_name = Path(row['artifact']).with_suffix('.json').name
    if runs['raw_records'][sidecar_name] != base.digest(ARCHIVE / sidecar_name):
        raise ValueError('Archived trajectory sidecar differs')
    scale = row['job']['resolution']
    schedule = protocol['schedules'][scale]
    p = protocol['template']
    if not p['switch_time'] < args.horizon <= p['horizon']:
        raise ValueError('Replay must include the bridge and stay within T=72')
    times = common.selected_times(schedule['checkpoints'], args.horizon)
    config = base.config(protocol['specs'][args.method], protocol['solve_protocol'], seed)[1]
    solution = solve_cuda_prepared(preparation=args.preparation, startup_rate=schedule['startup_rate'],
                                  late_rate=schedule['late_rate'], switch_time=p['switch_time'],
                                  horizon=args.horizon, snapshots=times, probe_spacing=p['probe_spacing'],
                                  graph_steps=p['graph_steps'], **config)
    if not all(np.isfinite(getattr(solution, name)).all() for name in common.ARRAYS):
        raise FloatingPointError('Replay produced nonfinite arrays')
    match = dict(status='complete', artifact=row['artifact'], sha256=row['sha256'])
    comparison = common.archive_comparison(solution, common.RecurrenceGalerkin, ARCHIVE, match)
    with np.load(ARCHIVE / row['artifact'], allow_pickle=False) as stored:
        selected = np.flatnonzero(np.isin(stored['times'], solution.times))
        for name in common.ARRAYS:
            values = getattr(solution, name)
            expected = stored[name][selected] if name == 'states' else stored[name][:len(solution.probe_times)]
            comparison['arrays'][name]['bitwise'] = (values.dtype == expected.dtype and values.shape == expected.shape
                                                      and values.tobytes() == expected.tobytes())
    comparison['scope'] = ('Saved coefficient and probe array comparison at matching times; '
                           'does not re-score physical fields or establish a new PDE accuracy gate.')
    solution.row.update(replay_mode='prepared_cuda', replay_numeric_protocol_sha256=qualification['protocol_sha256'],
                        timing_scope='Numerical reproduction only; timings are diagnostic.')
    output.parent.mkdir(parents=True, exist_ok=True)
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
    common.write_new(output, lambda stream: case.store_solution(stream, solution, config))
    report = dict(status='complete' if comparison['passed'] else 'archive_comparison_failed',
                  job=row['job'], horizon=args.horizon, observations=solution.times.tolist(),
                  artifact=output.name, artifact_sha256=base.digest(output),
                  runner_sha256=base.digest(Path(__file__)), comparison=comparison, metadata=solution.row,
                  physical_scope='No new PDE accuracy qualification; low-grid archive remains physically unqualified.')
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if sidecar is not None:
        common.write_new(sidecar, lambda stream: stream.write(payload.encode('utf-8')))
    print(payload, end='')
    return 0 if comparison['passed'] else 1

if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--method', choices=('rfm35_k64', 'leg35_k64', 'rfm37_k72', 'leg37_k72',
                                          'rfm41_k80', 'leg39_k80'), default='rfm37_k72')
    cli.add_argument('--seed', type=int)
    cli.add_argument('--preparation', choices=('module_only', 'shared_only', 'combined'), default='combined')
    cli.add_argument('--horizon', type=float, default=9.)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--json-output', type=Path)
    raise SystemExit(execute(cli.parse_args()))
