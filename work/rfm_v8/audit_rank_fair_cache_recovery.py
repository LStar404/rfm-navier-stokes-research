"""Read-only CPU audit of the original formal cohort before a separate attempt."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import sys

import freeze_rank_fair_cache_formal as formal
import verify_rank_fair_cache_formal_fixture as fixture
from rank_fair_owned_transport import key


def package_identity(name):
    dist = importlib.metadata.distribution(name)
    files = {}
    for entry in dist.files or ():
        path = Path(dist.locate_file(entry)).resolve()
        if path.is_file() and path.suffix.lower() in (".py", ".pyd", ".dll"):
            files[str(entry).replace("\\", "/")] = formal.T.digest(path)
    return dict(version=dist.version, files_sha256=files,
                tree_sha256=sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def audit(protocols, qualification):
    current = formal.environment()  # nvidia-smi query only; creates no CUDA context.
    plans = [formal.read(protocols/(name+".json")) for name in formal.WINDOWS]
    for plan in plans:
        formal.verify_plan(plan, live=True)
    qp, controls, sources, proof = formal.load_qualification(qualification)
    census = formal.timing.prior.WindowsCensus()
    snapshot = census.live()
    live_keys = {key(row) for row in snapshot["live"]}
    backgrounds = plans[0]["known_background"]
    absent = [row for row in backgrounds if key(row) not in live_keys]
    result = dict(schema="rfm-v8-recovery-audit-1", status="passed", created_utc=datetime.now(timezone.utc).isoformat(),
                  gpu_contexts_created=0, flow_solves_run=0, original_sources_edited=False,
                  all_three_original_protocols_verify_live=True,
                  qualification=proof, qualified_slots=len(controls), policy_pairs=len(qp["slots"])//2,
                  bound_qualification_and_evidence_files=len(sources), current_environment=current,
                  python_executable=str(Path(sys.executable).resolve()),
                  current_package_byte_identities={name:package_identity(name) for name in ("numpy", "scipy")},
                  package_hash_scope_note="New byte baseline; original freeze bound NumPy/SciPy versions, not complete installed package byte trees.",
                  original_protocols=[dict(window=p["window"], slots=len(p["requests"]),
                       sha256=formal.T.digest(protocols/(p["window"]+".json")),
                       ordered_request_sha256=sha256(json.dumps(p["requests"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()) for p in plans],
                  original_source_hashes={name:formal.T.digest(formal.HERE/name) for name in
                       ("freeze_rank_fair_cache_formal.py", "rank_fair_cache_formal_worker.py", "verify_rank_fair_cache_formal.py", "cuda_private_scope_solver.py")},
                  original_background=backgrounds, absent_original_background=absent,
                  original_protocol_ready_to_execute=not absent,
                  original_fixture_replayed_without_overwrite=fixture.check(qualification))
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--protocols", type=Path, required=True)
    cli.add_argument("--qualification", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.protocols.resolve(), args.qualification.resolve())
    formal.T.save(args.output.resolve(), result)
    print(json.dumps({k:result[k] for k in ("status", "gpu_contexts_created", "flow_solves_run", "all_three_original_protocols_verify_live", "original_protocol_ready_to_execute", "absent_original_background")}))
