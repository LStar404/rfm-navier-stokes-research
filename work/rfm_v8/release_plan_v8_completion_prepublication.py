"""Extend the v8 publication dry run without changing the checkout or earlier inventory."""

import argparse
from collections import Counter
import json
from pathlib import Path
import struct

import release_plan_v8 as prior


WORK = prior.WORKSPACE / "work"
BASE = WORK / "rfm_v8_publication_dryrun_v3" / "inventory.json"
REDACTIONS = WORK / "rfm_v8_publication_private_context_dryrun" / "derived"
EXTENSION_PREFIXES = (
    "build_cuda_ptx_artifacts", "cuda_ptx_", "cuda_private_", "freeze_rank_fair_",
    "install_stock_ptx", "operator_bottleneck_review", "protocol_rank_fair_", "protocol_rank_parity_",
    "rank_fair_", "rank_parity_", "release_plan_v8_completion", "rfm_rank_high_screen", "theory_independent_review",
    "stock_ptx_install", "verify_cuda_ptx_", "verify_cuda_private_", "verify_rank_fair_",
)
EVIDENCE_EXTENSIONS = prior.SCOPED_EXT | {".rfmptx", ".pending"}
SYNTHETIC_RANK_DIRS = {f"rank_fair_transport_smoke_v{v}_output" for v in (1, 2, 3)}


def root_path(relative):
    path = prior.WORKSPACE / relative
    prior.require(not path.is_symlink() and path.resolve().relative_to(prior.WORKSPACE).as_posix() == relative,
                  "Unsafe workspace evidence path")
    return path


def audit_evidence(path, relative):
    if path.suffix == ".rfmptx":
        raw = path.read_bytes()
        prior.require(raw.startswith(b"RFMPTX01") and len(raw) > 12, "Unexpected PTX artifact container")
        offset = 12 + struct.unpack_from("<I", raw, 8)[0]
        prior.require(12 < offset < len(raw), "Invalid PTX artifact header")
        manifest = json.loads(raw[12:offset])
        prior.require(len(raw[offset:]) == manifest["ptx_bytes"] and
                      prior.data_hash(raw[offset:]) == manifest["ptx_sha256"], "PTX payload identity differs")
        # The fixed header includes a binary uint32, not UTF-8. Scan both
        # textual components after validating their boundary and payload hash.
        flags = prior.text_flags(raw[12:offset].decode("utf-8"))
        flags.update(prior.text_flags(raw[offset:].decode("utf-8")))
        return flags, "compiled_ptx_artifact"
    parts = Path(relative).parts
    if path.suffix == ".npz" and len(parts) > 2 and parts[2] in SYNTHETIC_RANK_DIRS:
        raw = path.read_bytes()
        if not raw.startswith(b"PK\x03\x04"):
            return prior.text_flags(prior.decoded_bytes(raw)), "synthetic_transport_payload"
    if path.suffix == ".png":
        from PIL import Image

        with Image.open(path) as figure:
            text = "\n".join(str(value) for value in figure.info.values())
            figure.verify()
        return prior.text_flags(text), "figure"
    return prior.inspect(path)[0], None


def evidence_paths(extra):
    found = set()
    for item in prior.V8.iterdir():
        if not item.name.startswith(EXTENSION_PREFIXES):
            continue
        if item.is_file() and item.suffix in EVIDENCE_EXTENSIONS:
            found.add(item)
        if item.is_dir():
            for file in item.rglob("*"):
                if item.name == "stock_ptx_install_output" and file.suffix == ".log":
                    continue
                if file.is_file() and file.suffix in EVIDENCE_EXTENSIONS and "__pycache__" not in file.parts:
                    found.add(file)
    for relative in extra:
        path = root_path(relative)
        prior.require(path.is_file(), "An explicitly requested evidence member is missing")
        found.add(path)
    return {file.relative_to(prior.WORKSPACE).as_posix():file for file in found}


def run(destination, extra, pending):
    prior.require(destination.parent == WORK and not destination.exists(), "Choose a new work-only dry-run directory")
    original = prior.read(BASE)
    old = prior.old_head(full=True)
    found = evidence_paths(extra)
    prior.require(not any(path.startswith(prior.PREFIX + "vendor/") or "usage_checkpoint" in path for path in found),
                  "Private administration or vendor payload entered the allowlist")
    mapping_path = REDACTIONS / "private_context_stderr_redaction_map.json"
    note_path = REDACTIONS / "private_context_redaction_note.md"
    redaction = prior.read(mapping_path)["records"]
    exclusions = {row["original_path"] for row in redaction}
    for row in redaction:
        source = REDACTIONS / Path(row["public_path"]).name
        prior.require(prior.digest(root_path(row["original_path"])) == row["original_sha256"] and
                      prior.digest(source) == row["public_sha256"], "Private-context redaction mapping changed")
        found[row["public_path"]] = source
    for derived in (mapping_path, note_path):
        found[prior.PREFIX + derived.name] = derived

    retained, blocked, added = [], [], []
    for row in original["member_inventory"]:
        path = root_path(row["source_path"])
        prior.require(prior.digest(path) == row["sha256"], "A prior dry-run candidate changed. Preserve its reviewed bytes.")
        retained.append(row)
        found.pop(row["path"], None)
    for relative, path in sorted(found.items()):
        if relative in exclusions:
            continue
        try:
            source_relative = path.relative_to(prior.WORKSPACE).as_posix()
            prior.require(relative not in old and root_path(source_relative) == path,
                          "Candidate overlaps old checkout or uses a symlink")
            before = prior.digest(path)
            flags, kind = audit_evidence(path, relative)
            prior.require(before == prior.digest(path), "Candidate changed during audit")
        except (OSError, ValueError, RuntimeError, UnicodeError, KeyError, prior.BadZipFile) as error:
            blocked.append(dict(path=relative,reason=type(error).__name__))
            continue
        if flags:
            blocked.append(dict(path=relative,reason="privacy_pattern",categories=sorted(flags)))
            continue
        row = dict(path=relative,source_path=path.relative_to(prior.WORKSPACE).as_posix(),
                   sha256=before,bytes=path.stat().st_size)
        if kind is not None:
            row["kind"] = kind
        added.append(row)
    files = sorted((*retained, *added), key=lambda row:row["path"])
    linked, exceptions = prior.link_checks(files, old)
    prior.old_head(full=False)
    destination.mkdir()
    record = dict(status="publication_completion_dry_run_no_copy",old=original["old"],
        prior_inventory_path=BASE.relative_to(prior.WORKSPACE).as_posix(),prior_inventory_sha256=prior.digest(BASE),
        retained_prior_candidates=len(retained),newly_proposed_candidates=len(added),proposed_member_count=len(files),
        proposed_bytes=sum(row["bytes"] for row in files),member_inventory=files,privacy_blocked=blocked,
        excluded_private_logs=sorted(exclusions),linked_hash_records_checked=linked,linked_dependency_exceptions=exceptions,
        dependency_exception_counts=dict(Counter(row["classification"] for row in exceptions)),
        git_author_privacy=prior.git_privacy(),pending=pending,
        scope="Dry run only. Existing publication checkout and remote are untouched.")
    prior.emit(destination/"inventory.json",record)
    print("DRY RUN",len(files),len(added),len(blocked),len(exceptions),flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--extra",action="append",default=[])
    cli.add_argument("--pending",action="append",default=[])
    cli.add_argument("--include-plan",type=Path,
                     help="Small reviewed JSON with explicit required_members; missing files fail closed.")
    args = cli.parse_args()
    if args.include_plan is not None:
        plan_path = args.include_plan.resolve()
        prior.require(plan_path.is_relative_to(WORK) and not plan_path.is_symlink(),
                      "Include plan must be a workspace work file")
        plan = prior.read(plan_path)
        prior.require(plan.get("status") == "planned_no_full_scan" and
                      isinstance(plan.get("required_members"),list) and
                      all(isinstance(item,str) for item in plan["required_members"]),
                      "Unexpected include-plan contract")
        args.extra.extend(plan["required_members"])
        args.extra.append(plan_path.relative_to(prior.WORKSPACE).as_posix())
    run(args.output.resolve(),args.extra,args.pending)
