"""Extend the v8 publication dry run without changing the checkout or earlier inventory."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
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
    "audit_rank_fair_", "prepare_rank_fair_", "formal_interruption_review_",
    "release_completion_",
    "prepare_recovery_", "prepare_report_final_round", "report_completion_historical_source",
)
EVIDENCE_EXTENSIONS = prior.SCOPED_EXT | {".rfmptx", ".pending", ".cs", ".exe", ".tmp"}
SYNTHETIC_RANK_DIRS = {f"rank_fair_transport_smoke_v{v}_output" for v in (1, 2, 3)}
SYNTHETIC_RANK_DIRS.add("rank_fair_isolation_cpu_fixture_v2_output")
STOP_FILE = prior.V8 / "publication_pause.flag"


class PublicationPaused(BaseException):
    pass


def check_pause():
    if STOP_FILE.exists():
        raise PublicationPaused("Publication scan paused for formal timing")


def private_admin(relative):
    parts = Path(relative).parts
    name = parts[-1].lower()
    return ("vendor" in parts or "__pycache__" in parts or
            any(word in name for word in ("usage_checkpoint", "quota", "account", "_admin")))


def root_path(relative):
    path = prior.WORKSPACE / relative
    prior.require(not path.is_symlink() and path.resolve().relative_to(prior.WORKSPACE).as_posix() == relative,
                  "Unsafe workspace evidence path")
    return path


def audit_evidence(path, relative):
    if path.suffix == ".tmp":
        prior.require(path.name.endswith(".json.tmp") and
                      (relative.startswith(prior.PREFIX+"rank_fair_slot_publish_cpu_review/") or
                       relative==prior.PREFIX+"rank_fair_cache_recovery_formal_v3_output/mid_b1_b5/capture/outer_slot_state.json.tmp"),
                      "Only explicitly preserved failed-publication JSON temporaries are evidence")
        text = prior.decoded_bytes(path.read_bytes())
        json.loads(text)
        return prior.text_flags(text), "preserved_failed_publication_temporary_json"
    if path.suffix == ".exe":
        raw = path.read_bytes()
        owned_watchdog = path.name == "rank_fair_outer_watchdog.exe" and "rank_fair_cache_recovery_watchdog_fixture" in relative
        owned_reader = (relative.startswith(prior.PREFIX+"rank_fair_slot_publish_cpu_review/") and
                        path.name in {"reader_fixture.exe","immutable_reader_fixture.exe"})
        prior.require((owned_watchdog or owned_reader) and
                      raw[:2] == b"MZ" and len(raw) > 64,
                      "Only the independently inventoried project watchdog and CPU-reader executables are allowed")
        offset = struct.unpack_from("<I", raw, 60)[0]
        prior.require(raw[offset:offset+4] == b"PE\0\0", "Expected project PE executable")
        flags = prior.text_flags(raw.decode("latin1"))
        flags.update(prior.text_flags(raw.decode("utf-16-le", errors="ignore")))
        return flags, "project_compiled_watchdog_executable" if owned_watchdog else "project_compiled_cpu_reader_fixture"
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
    isolation_fixture = len(parts)>2 and re.fullmatch(r"rank_fair_isolation_cpu_fixture_v[0-9]+_output",parts[2])
    if path.suffix == ".npz" and len(parts) > 2 and (parts[2] in SYNTHETIC_RANK_DIRS or isolation_fixture):
        raw = path.read_bytes()
        if not raw.startswith(b"PK\x03\x04"):
            if isolation_fixture:
                prior.require(raw == b"synthetic CPU transport artifact; not numerical data\n",
                              "Isolation fixture transport sentinel changed")
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
        if item.name == "release_completion_resolver_fixture_output":
            # The verifier recreates these deliberately invalid packages. Publish
            # its certificate explicitly, not fake nested package dependencies.
            continue
        if not item.name.startswith(EXTENSION_PREFIXES) or private_admin(item.relative_to(prior.WORKSPACE).as_posix()):
            continue
        if item.is_file() and item.suffix in EVIDENCE_EXTENSIONS:
            found.add(item)
        if item.is_dir():
            for file in item.rglob("*"):
                if item.name == "stock_ptx_install_output" and file.suffix == ".log":
                    continue
                if file.is_file() and file.suffix in EVIDENCE_EXTENSIONS and not private_admin(file.relative_to(prior.WORKSPACE).as_posix()):
                    found.add(file)
    for relative in extra:
        path = root_path(relative)
        prior.require(not private_admin(relative), "Private administration or vendor payload is not a publication member")
        prior.require(path.is_file(), "An explicitly requested evidence member is missing")
        found.add(path)
    return {file.relative_to(prior.WORKSPACE).as_posix():file for file in found}


def scrub_administration(value):
    """Withhold business-identification references, preserving scientific policy."""
    count = 0
    if isinstance(value,dict):
        result = {}
        for key,child in value.items():
            path_key = isinstance(key,str) and ("/" in key or "\\" in key)
            path_value = key in ("path","source_path") and isinstance(child,str)
            if key == "admin_evidence" or (path_key and private_admin(key)) or (path_value and private_admin(child)):
                count += 1
                continue
            result[key],nested = scrub_administration(child)
            count += nested
        return result,count
    if isinstance(value,list):
        result = []
        for child in value:
            public,nested = scrub_administration(child)
            result.append(public)
            count += nested
        return result,count
    return value,0


def redacted_copy(path, relative, stage):
    """Publish a separately named derivative; never relabel the original digest."""
    raw = path.read_bytes()
    text = prior.decoded_bytes(raw)
    prior.require(prior.text_flags(text) <= {"user_path"}, "Only path and administration-reference redaction is automatic")
    changed = text
    omitted = 0
    if path.suffix == ".json":
        public,omitted = scrub_administration(json.loads(text))
        if omitted:
            changed = json.dumps(public,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    for root, placeholder in ((prior.WORKSPACE,"<WORKSPACE>"),(Path.home(),"<USER_HOME>")):
        for form in sorted({str(root),root.as_posix(),str(root).replace("\\","\\\\")}, key=len, reverse=True):
            changed = re.sub(re.escape(form), lambda _:placeholder, changed, flags=re.I)
    prior.require(changed != text and not prior.text_flags(changed), "Public copy still has privacy flags")
    public_relative = str(Path(relative).with_name(Path(relative).stem + ".public" + Path(relative).suffix)).replace("\\","/")
    target = stage / public_relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(changed,encoding="utf-8",newline="")
    if path.suffix == ".json":
        prior.read(target)
    return target, dict(original_path=relative,original_bytes=len(raw),original_sha256=prior.data_hash(raw),
        public_path=public_relative,public_bytes=target.stat().st_size,public_sha256=prior.digest(target),
        original_in_checkout=False,private_administration_references_omitted=omitted,
        transform="Replace exact workspace/home path prefixes and omit private administration references when present. Preserve scientific states, numerical values and CPU budget policy. Remaining original hashes identify original evidence only.")


def project_executable_recipe(files, stage):
    records = []
    for relative,path in sorted(files.items()):
        if path.suffix != ".exe":
            continue
        if path.name in {"reader_fixture.exe","immutable_reader_fixture.exe"}:
            source = path.with_suffix(".cs")
            certificate = path.parent/"independent_diagnosis.json"
            saved = prior.read(certificate)["local_review_source_sha256"]
            prior.require(prior.digest(path)==saved[path.name] and prior.digest(source)==saved[source.name],
                          "CPU reader fixture source or executable differs from independent diagnosis")
            records.append(dict(path=relative,sha256=prior.digest(path),bytes=path.stat().st_size,
                certificate=certificate.relative_to(prior.WORKSPACE).as_posix(),
                source_snapshot=source.relative_to(prior.WORKSPACE).as_posix(),source_sha256=prior.digest(source),
                source_verified=True,source_argument=source.name,
                scope="Project-authored CPU-only diagnostic reader fixture, not a solver or integrated replacement watchdog. Source and archived binary are independently hash-bound; the documented compiler recipe does not claim byte-identical rebuild."))
            continue
        snapshot = path.parent/"source_snapshot"/"rank_fair_outer_watchdog.cs"
        certificate = path.parent/"certificate.json"
        saved = prior.read(certificate) if certificate.exists() else {}
        digest = prior.digest(path)
        if saved:
            prior.require(saved.get("executable_sha256")==digest, "Archived project executable changed")
        source_match = snapshot.exists() and prior.digest(snapshot)==saved.get("source_sha256")
        records.append(dict(path=relative,sha256=digest,bytes=path.stat().st_size,
            certificate=certificate.relative_to(prior.WORKSPACE).as_posix() if certificate.exists() else None,
            source_snapshot=snapshot.relative_to(prior.WORKSPACE).as_posix() if source_match else None,
            source_sha256=prior.digest(snapshot) if source_match else saved.get("source_sha256"),
            source_verified=source_match,
            scope="Project-authored executable; compiler is not redistributed. Exact source replay is available." if source_match else
                  "Historical executable evidence; exact source capsule is unavailable. Do not claim this binary is reproducible from current source."))
    recipe = stage/"project_watchdog_build_recipe.json"
    prior.emit(recipe,dict(status="project_source_build_recipe_and_archived_binary_hashes",compiler=dict(
        path="C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe",file_version="4.8.9221.0 built by NET481REL1LAST_25H2",
        sha256="46809206887326d2d24db1eff1f3064de972c3451abe766b49111450a5e08e00"),
        arguments=["/nologo","/target:exe","/reference:System.Web.Extensions.dll","/out:<OUTPUT_EXE>","<EXACT_PROJECT_SOURCE_CS>"],
        platform="Compiler default AnyCPU; tested in 64-bit .NET Framework 4.8.1 on Windows 11 build 26200.",
        requirements="System .NET Framework libraries and Windows 10+ PROC_THREAD_ATTRIBUTE_JOB_LIST. No third-party library or vendor binary is distributed.",
        reproducibility="Source, compiler and archived output hashes are distinct. Byte-identical compilation across paths or machines is not claimed.",
        records=records))
    return recipe


def run(destination, extra, pending, prepare_only=False, target_overrides=None):
    prior.require(destination.parent == WORK and not destination.exists(), "Choose a new work-only dry-run directory")
    check_pause()
    original = prior.read(BASE)
    original_digest = prior.digest
    digest_cache = {}
    def cached_digest(path):
        check_pause()
        stat = path.stat()
        key = (str(path.resolve()),stat.st_size,stat.st_mtime_ns)
        if key not in digest_cache:
            digest_cache[key] = original_digest(path)
        return digest_cache[key]
    prior.digest = cached_digest
    old = prior.old_head(full=True)
    found = evidence_paths(extra)
    target_overrides = target_overrides or {}
    permitted_targets = {prior.PREFIX+"publication_v8_README.md":"START_HERE_v8.md",
        prior.PREFIX+"publication_v8_github_README.md":".github/README.md"}
    for source,target in target_overrides.items():
        prior.require(permitted_targets.get(source)==target and source in found and target not in found,
                      "Only the two explicitly authorized entry-document target mappings are allowed")
        found[target] = found.pop(source)
    prior.require(not any(path.startswith(prior.PREFIX + "vendor/") or "usage_checkpoint" in path for path in found),
                  "Private administration or vendor payload entered the allowlist")
    destination.mkdir()
    stage = destination/"derived"
    stage.mkdir()
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
    recipe = project_executable_recipe(found, stage)
    found[prior.PREFIX + recipe.name] = recipe

    retained, blocked, added, new_redaction = [], [], [], []
    for row in original["member_inventory"]:
        path = root_path(row["source_path"])
        prior.require(prior.digest(path) == row["sha256"], "A prior dry-run candidate changed. Preserve its reviewed bytes.")
        retained.append(row)
        found.pop(row["path"], None)
    for relative, path in sorted(found.items()):
        check_pause()
        if relative in exclusions:
            continue
        try:
            source_relative = path.relative_to(prior.WORKSPACE).as_posix()
            permitted_replacement = (relative==".github/README.md" and
                source_relative==prior.PREFIX+"publication_v8_github_README.md" and target_overrides.get(source_relative)==relative)
            prior.require((relative not in old or permitted_replacement) and root_path(source_relative) == path,
                          "Candidate overlaps old checkout or uses a symlink")
            initial_stat = path.stat()
            before = prior.digest(path)
            flags, kind = audit_evidence(path, relative)
            prior.require((initial_stat.st_size,initial_stat.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns), "Candidate changed during audit")
            admin_references = scrub_administration(prior.read(path))[1] if path.suffix == ".json" else 0
            if (flags == {"user_path"} or admin_references) and path.suffix in prior.TEXT_EXT | {".cs"}:
                path,mapping = redacted_copy(path,relative,stage)
                new_redaction.append(mapping)
                exclusions.add(relative)
                relative = mapping["public_path"]
                before = mapping["public_sha256"]
                flags = set()
                kind = "explicit_public_path_redaction"
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
        if relative in old:
            row["replaces_old"] = {key:old[relative][key] for key in ("sha256","bytes")}
            row["replacement_authorization"] = "Explicitly authorized v8 GitHub entry README update; all experimental base members remain protected."
        added.append(row)
    all_redaction = list(original["redaction_map"])+redaction+new_redaction
    exclusions.update(row["original_path"] for row in all_redaction)
    all_mapping = stage/"publication_path_redaction_map.json"
    prior.emit(all_mapping,dict(schema="rfm-v8-public-derivative-map-1",status="public_derivative_mapping",records=all_redaction,
        scope="Two synthetic and two real device-failure logs, plus separately named path/admin-reference evidence derivatives. Private administration and business-identification files are excluded. Scientific policy thresholds and recorded outcomes are retained. Original frozen hashes identify private originals only; public hashes identify different bytes. Original files remain local. Public copies cannot satisfy original hash checks. Regenerate local runtime evidence for fresh execution."))
    note = stage/"publication_completion_README.md"
    note.write_text("# v8 publication completion evidence\n\n"
        "The path redaction map includes the two synthetic transport logs, two real private-context failure logs, and path-bearing watchdog/recovery evidence. Originals remain local. Each public derivative has a distinct filename and its own SHA-256; original hashes retained inside frozen records do not authenticate the public derivative. No recorded outcome or numerical value is changed.\n\n"
        "The project watchdog build recipe identifies the system compiler, exact source capsules where available, and archived executable hashes. The project executable is not a redistributed CUDA or vendor runtime. Historical binaries without recovered exact source are labelled accordingly. Compilation on other machines need not be byte-identical.\n\n"
        "This prepare-only inventory is not publication approval. Final report evidence, visual QA, complete formal-window verification and all unresolved dependencies remain required.\n",encoding="utf-8")
    for path in (all_mapping,note):
        added.append(dict(path=prior.PREFIX+path.name,source_path=path.relative_to(prior.WORKSPACE).as_posix(),
            sha256=prior.digest(path),bytes=path.stat().st_size))
    files = sorted((*retained, *added), key=lambda row:row["path"])
    linked, exceptions = prior.link_checks(files, old)
    redacted_pairs = {(row["original_path"],row["original_sha256"]):row for row in all_redaction}
    for row in exceptions:
        public = redacted_pairs.get((row["path"],row["referenced_sha256"]))
        if public is not None:
            row["classification"] = "explicit_private_original_public_derivative"
            row["public_path"] = public["public_path"]
            row["public_sha256"] = public["public_sha256"]
        elif (row["path"] == prior.PREFIX+"report_draft_provenance.json" and
              row["referenced_sha256"] == "23f822ab8b53643a7d3cb2374f4be03ae105ac23d0d21b1358732512976e6840" and
              row["sample_origins"] == [prior.PREFIX+"report_completion_evidence.json"]):
            row["classification"] = "historical_draft_checkpoint_not_final_dependency"
            row["scope"] = "Superseded draft provenance referenced only by an earlier progress checklist. Exact old provenance bytes were not retained; no reconstructed original is claimed. The draft is not a final report input."
    historical_map = stage/"publication_historical_source_map.json"
    aliases = [dict(original_path=row["path"],original_sha256=row["referenced_sha256"],
        public_path=row["exact_aliases"][0],public_sha256=row["referenced_sha256"],kind="exact_historical_source_alias")
        for row in exceptions if row["classification"]=="historical_exact_source_capsule"]
    prior.emit(historical_map,dict(schema="rfm-v8-historical-source-map-1",status="exact_byte_aliases",records=aliases,
        scope="An exact historical source capsule has the same bytes and SHA as the recorded original. This is distinct from path-redacted derivatives. Both this map and the target bytes must be verified through the complete publication manifest before resolving an alias."))
    files.append(dict(path=prior.PREFIX+historical_map.name,source_path=historical_map.relative_to(prior.WORKSPACE).as_posix(),
        sha256=prior.digest(historical_map),bytes=historical_map.stat().st_size))
    files.sort(key=lambda row:row["path"])
    report_ready = False
    visual_ready = False
    final_path = prior.V8/"report_final_round.json"
    review_path = prior.V8/"report_final_round_publication_review.json"
    if final_path.exists() and review_path.exists():
        final,review = prior.read(final_path),prior.read(review_path)
        report_ready = bool(final.get("reviewed_for_publication") and review.get("status")=="passed" and
            review.get("addendum_sha256")==prior.digest(final_path) and
            review.get("builder_sha256")==prior.digest(prior.V8/"build_v8_report.py"))
    visual_path = prior.V8/"report_visual_quality_review.json"
    if visual_path.exists() and review_path.exists():
        visual = prior.read(visual_path)
        visual_ready = bool(visual.get("status")=="passed_visual_and_text_checks" and
            visual.get("pdf_sha256")==prior.digest(root_path(visual["pdf"])) and
            visual.get("builder_sha256")==prior.digest(prior.V8/"build_v8_report.py") and
            visual.get("independent_content_review_sha256")==prior.digest(review_path) and
            visual.get("all_pages_visually_inspected")==list(range(1,visual["pages"]+1)))
    prior.old_head(full=False)
    record = dict(schema="rfm-v8-publication-inventory-1",status="publication_prepare_only_dry_run_no_copy" if prepare_only else "publication_completion_dry_run_no_copy",old=original["old"],
        publication_ready=bool(not prepare_only and not pending and not blocked and report_ready and visual_ready and
            not any(row["classification"]=="unresolved_required_dependency" for row in exceptions)),
        final_report_content_review_verified=report_ready,
        final_report_visual_review_verified=visual_ready,
        authorized_target_overrides=target_overrides,
        authorized_replacements=[dict(path=row["path"],before=row["replaces_old"],after=dict(sha256=row["sha256"],bytes=row["bytes"]))
            for row in files if "replaces_old" in row],
        prior_inventory_path=BASE.relative_to(prior.WORKSPACE).as_posix(),prior_inventory_sha256=prior.digest(BASE),
        retained_prior_candidates=len(retained),newly_proposed_candidates=len(added),proposed_member_count=len(files),
        proposed_bytes=sum(row["bytes"] for row in files),member_inventory=files,privacy_blocked=blocked,
        excluded_private_originals=sorted(exclusions),redaction_map=all_redaction,
        linked_hash_records_checked=linked,linked_dependency_exceptions=exceptions,
        dependency_exception_counts=dict(Counter(row["classification"] for row in exceptions)),
        git_author_privacy=prior.git_privacy(),pending=pending,
        digest_policy="One SHA read per unchanged path/size/mtime during this invocation; file identity metadata checked after privacy inspection. Final release must verify source hashes again.",
        scope="Dry run only. Existing publication checkout and remote are untouched.")
    prior.emit(destination/"inventory.json",record)
    print("DRY RUN",len(files),len(added),len(blocked),len(exceptions),flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--extra",action="append",default=[])
    cli.add_argument("--pending",action="append",default=[])
    cli.add_argument("--prepare-only",action="store_true",help="Defer missing final report members, never treat this as publication approval")
    cli.add_argument("--include-plan",type=Path,
                     help="Small reviewed JSON with explicit required_members; missing files fail closed.")
    args = cli.parse_args()
    targets = {}
    if args.include_plan is not None:
        plan_path = args.include_plan.resolve()
        prior.require(plan_path.is_relative_to(WORK) and not plan_path.is_symlink(),
                      "Include plan must be a workspace work file")
        plan = prior.read(plan_path)
        prior.require(plan.get("status") == "planned_no_full_scan" and
                      isinstance(plan.get("required_members"),list) and
                      all(isinstance(item,str) for item in plan["required_members"]),
                      "Unexpected include-plan contract")
        for member in plan["required_members"]:
            if args.prepare_only and not root_path(member).is_file():
                args.pending.append("Required final member not yet available: "+member)
            else:
                args.extra.append(member)
        args.extra.append(plan_path.relative_to(prior.WORKSPACE).as_posix())
        targets = plan.get("target_overrides",{})
    run(args.output.resolve(),args.extra,args.pending,args.prepare_only,targets)
