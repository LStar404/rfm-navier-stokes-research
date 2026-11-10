"""Read-only append inventory and privacy audit for the v8 workspace evidence."""

import argparse
from collections import defaultdict
from hashlib import file_digest, sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from zipfile import BadZipFile, ZipFile


V8 = Path(__file__).resolve().parent
WORKSPACE = V8.parents[1]
CHECKOUT = WORKSPACE/"release"/"rfm-navier-stokes-publication"
BASE_HEAD = "dcb69a05c60a4d2c18422b9018a8421145ced190"
BASE_MANIFEST = "27299292a78718b57c18d22843dcae9b6fe55f249cd0af07e2e33f2b7a7c0089"
BASE_COUNT = 1831
PREFIX = "work/rfm_v8/"
KNOWN_LOGS = tuple(f"{name}/bad_parent_birth/worker_000.stderr.log" for name in (
    "formal_transport_smoke_output","formal_transport_smoke_final_output"))
SENTINEL_FILES = {PREFIX+name+"/"+stem for name in (
    "formal_transport_smoke_output","formal_transport_smoke_final_output") for stem in (
    "attempt_preservation/trajectory_000.npz","attempt_preservation/trajectory_002.npz",
    "cadence_1000ms/trajectory_000.npz","cadence_100ms/trajectory_000.npz")}
OUTPUT_DIRS = set("""
cuda_graph_schedule_diagnostic_output cuda_pingpong_pilot_output cuda129_t72_diagnostic_output cuda129_t9_output
formal_transport_smoke_final_output formal_transport_smoke_output fresh_worker_mid_t9_output
fused_device_watchdog_audit_output fused_march_t9_output fused_nonlinear_fft_device_output galerkin_minimal_import_audit_output import_diagnostic_output
galerkin_phase_pencil_precanonical_source
lamb_linearized_fixed_audit_output lamb_symmetric_fixed_audit_output lamb_t16_qualification_output
legendre_rank_gpu_qualification_output legendre_rank_match_high_output legendre_rank_switch_output legendre_rank_uniform_output
nonlinear_stride2_pilot_output nonlinear_stride2_pilot_retry_output thin_formal_checker_fixture_output
thin_formal_high_output thin_formal_mid_output thin_t72_qualification_output
warp_observe_micro_output warp_observe_micro_retry_output warp_observe_qualification_output warp_observe_qualification_retry_output
""".split())
TOP_PREFIXES = (
    "audit_","build_cuda","build_galerkin","compile_fused","cuda","fd_cuda","formal_timing","freeze_thin","fresh_worker",
    "fused_","galerkin_","gpu_fresh","gpu_owned","import_diagnostic","lamb_","legendre_rank","make_cuda","make_fused","nonlinear_",
    "protocol_","provision_cuda","qualify_legendre","release_plan_v8","rescore_cuda","run_fused","sbdf2_schedule",
    "solver_worker","summarize_legendre","theory_galerkin","thin_formal","thin_t72","verify_","warp_observe","watchdog_","win32_",
)
TEXT_EXT = {".py",".c",".cu",".ps1",".json",".jsonl",".txt",".log",".md"}
SCOPED_EXT = TEXT_EXT | {".npz"}
ADMIN = {
    "background_python_readonly_audit.json":"Machine background-process calibration is operational, not a flow or scientific failure record.",
    "usage_checkpoint_30.json":"Private account/quota administration.",
}
PENDING = (
    "Final fused-GPU evidence and independent safety/qualification status.",
    "Final formal timing cohort, independent isolation and numerical certification.",
    "Root-approved final report, report source, text, figure inventory and PDF privacy/render check.",
)
FILTERS = {
    "user_path":re.compile(r"(?i)(?:[a-z]:[/\\]+(?:users|documents and settings)[/\\]+|/(?:home|users)/[\w.-]+)"),
    "credential":re.compile(r"(?i)-----BEGIN [^-\r\n]*PRIVATE KEY-----|github_pat_[\w]+|gh[pousr]_[a-z0-9]{20,}|(?:sk|rk)-[\w-]{25,}"),
    "credentialed_url":re.compile(r"(?i)https?://[^/\s:]+:[^/\s@]+@"),
    "email":re.compile(r"(?i)\b[a-z0-9_.%+-]+@[a-z0-9.-]+\.(?:com|edu|net|org|gov|io|cn|uk|de|jp|au|fr|ru|in)\b"),
}
HEX = re.compile(r"[a-f0-9]{64}\Z")
NOREPLY = re.compile(r"(?i)(?:[0-9]+\+)?[a-z0-9-]+@users\.noreply\.github\.com|[^\s@]+@(?:example\.(?:invalid|org|com)|localhost|invalid)")


def require(condition,reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def data_hash(content):
    return sha256(content).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def emit(path,value):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")


def git(*args, check=True):
    result = subprocess.run(["git","-C",str(CHECKOUT),*args],capture_output=True,check=check)
    return result.stdout.decode("utf-8").strip()


def old_head(full):
    manifest = CHECKOUT/"MANIFEST.json"
    require(git("rev-parse","HEAD")==git("rev-parse","main")==BASE_HEAD and
            not git("status","--porcelain","--untracked-files=all") and digest(manifest)==BASE_MANIFEST,
            "Existing publication HEAD, clean status or manifest differs")
    files = read(manifest)["files"]
    members = {row["path"]:row for row in files}
    tracked = set(git("ls-files","-z").removesuffix("\0").split("\0"))
    require(len(files)==len(members)==BASE_COUNT and tracked==set(members)|{"MANIFEST.json"},
            "Protected manifest inventory differs from tracked checkout")
    if full:
        for relative,row in members.items():
            path = CHECKOUT/relative
            require(path.resolve().relative_to(CHECKOUT).as_posix()==relative and
                    path.stat().st_size==row["bytes"] and digest(path)==row["sha256"],
                    f"Protected member differs: {relative}")
    return members


def git_privacy():
    rows = [line.split("\0") for line in git("log","--all","--format=%an%x00%ae%x00%cn%x00%ce").splitlines()]
    contacts = {row[i] for row in rows for i in (1,3)}
    known_user = os.environ.get("USERNAME","").casefold()
    identity = git("config","--get","user.email",check=False)
    return dict(commits_checked=len(rows),distinct_contact_count=len(contacts),
        non_noreply_or_placeholder_count=sum(NOREPLY.fullmatch(value) is None for value in contacts),
        display_label_matches_local_machine_account=bool(known_user and any(row[i].casefold()==known_user for row in rows for i in (0,2))),
        next_commit_contact_configured=bool(identity),next_commit_contact_is_noreply_or_placeholder=bool(NOREPLY.fullmatch(identity)))


def text_flags(value):
    hits = set()
    for kind,pattern in FILTERS.items():
        if pattern.search(value):
            hits.add(kind)
    return hits


def decoded_bytes(value):
    if value.startswith((b"\xff\xfe",b"\xfe\xff")):
        return value.decode("utf-16")
    return value.decode("utf-8-sig")


def inspect_npz(path):
    import numpy as np
    from numpy.lib import format as npy
    hits = set()
    counts = {"config":0,"metadata":0,"other_strings":0}
    with ZipFile(path) as data:
        names = data.namelist()
        require(len(names)==len(set(names)) and all(re.fullmatch(r"[\w-]+\.npy",name) for name in names),
                "Invalid NPZ member names")
        for name in names:
            with data.open(name) as member:
                version = npy.read_magic(member)
                if version==(1,0):
                    _,_,dtype = npy.read_array_header_1_0(member)
                elif version==(2,0):
                    _,_,dtype = npy.read_array_header_2_0(member)
                else:
                    raise ValueError("Unsupported NPY header version")
            require(not dtype.hasobject and dtype.names is None,"Object or structured NPZ data requires explicit review")
            if name in ("config.npy","metadata.npy"):
                require(dtype.kind in ("S","U"),"Expected textual NPZ configuration/metadata")
                counts[name.removesuffix(".npy")] += 1
            elif dtype.kind in ("S","U"):
                counts["other_strings"] += 1
            if dtype.kind in ("S","U"):
                array = np.load(BytesIO(data.read(name)),allow_pickle=False)
                for value in array.flat:
                    text = decoded_bytes(bytes(value)) if dtype.kind=="S" else str(value)
                    hits.update(text_flags(text))
        require(data.testzip() is None,"NPZ CRC check failed")
    return hits,counts


def inspect(path):
    if path.suffix==".npz":
        return inspect_npz(path)
    if path.suffix==".pdf":
        from pypdf import PdfReader
        parsed = PdfReader(path)
        values = [str(parsed.metadata or {}),*(page.extract_text() or "" for page in parsed.pages)]
        return {hit for value in values for hit in text_flags(value)},{}
    raw = path.read_bytes()
    return text_flags(decoded_bytes(raw)),{}


def candidate_files():
    found,excluded,pending = [],[],[]
    for path in sorted(V8.iterdir()):
        name = path.name
        if name in ADMIN:
            excluded.append(dict(path=PREFIX+name,reason=ADMIN[name]))
        elif name in ("vendor","__pycache__"):
            excluded.append(dict(path=PREFIX+name+"/",reason="Vendor payload or generated Python cache."))
        elif path.is_dir() and name in OUTPUT_DIRS:
            found.extend(p for p in path.rglob("*") if p.is_file() and p.suffix in SCOPED_EXT and
                         "__pycache__" not in p.parts and not p.name.endswith(".tmp.npz"))
        elif path.is_file() and path.suffix in SCOPED_EXT and name.startswith(TOP_PREFIXES):
            found.append(path)
        elif not name.startswith("report"):
            pending.append(PREFIX+name)
    return found,excluded,pending


def make_redactions(stage):
    mapping = []
    roots = ((WORKSPACE,"<WORKSPACE>"),(Path.home(),"<USER_HOME>"))
    for name in KNOWN_LOGS:
        path = V8/name
        raw = path.read_bytes()
        text = decoded_bytes(raw)
        require(text_flags(text)=={"user_path"},"Expected only the two known fixture path disclosures")
        changed = text
        for root,placeholder in roots:
            for value in {str(root),root.as_posix(),str(root).replace("\\","\\\\")}:
                changed = re.sub(re.escape(value),lambda _:placeholder,changed,flags=re.I)
        require(changed!=text and not text_flags(changed),"Synthetic stderr redaction did not remove all private paths")
        label = name.split("/")[0]
        target = stage/f"{label}.bad_parent_birth.stderr.public.txt"
        target.write_bytes(changed.encode("utf-8"))
        mapping.append(dict(original_path=PREFIX+name,original_bytes=len(raw),original_sha256=data_hash(raw),
                            public_path=PREFIX+target.name,public_bytes=target.stat().st_size,public_sha256=digest(target),
                            original_in_checkout=False,transform="Replace exact workspace and home path prefixes by <WORKSPACE> and <USER_HOME>."))
    out = stage/"synthetic_fixture_stderr_redaction_map.json"
    emit(out,dict(status="proposed_public_redaction",records=mapping,
        scope="Synthetic invalid-parent-birth tests. Frozen fixture summaries retain original source hashes, "
              "which do not authenticate these public copies. The untouched original stderr stays only in the workspace. "
              "Rerun the fixture to generate new machine-local stderr rather than requiring the archived private original."))
    return [out,*(stage/Path(row["public_path"]).name for row in mapping)],mapping


def vendor_recipe(stage):
    plan = V8/"cuda129_vendor_plan.json"
    installed = V8/"vendor"/"cuda_12_9_1"/"manifest.json"
    p,m = read(plan),read(installed)
    require(m["plan_sha256"]==digest(plan) and p["status"]=="within_limit","CUDA vendor plan differs")
    licenses = [row for row in m["installed_members"] if row["path"].lower().endswith("/license.txt")]
    require(len(licenses)==3 and len({r["sha256"] for r in licenses})==1 and
            all(digest(installed.parent/r["path"])==r["sha256"] for r in licenses),"Pinned NVIDIA license bytes differ")
    manifest = stage/"cuda129_vendor_public_manifest.json"
    emit(manifest,dict(status="official_retrieval_recipe",plan_sha256=digest(plan),local_install_manifest_sha256=digest(installed),
        packages=[{k:r[k] for k in ("package","version","url","filename","sha256","download_bytes","license")} for r in p["packages"]],
        official_documents=p["official_documents"],
        official_license_url="https://docs.nvidia.com/cuda/archive/12.9.1/eula/index.html",
        wheel_license_members=licenses,
        runtime_file_sha256={r["path"]:r["sha256"] for r in m["installed_members"] if r["path"].endswith(".dll")},
        scope="Recreate locally with provision_cuda129.py --install using cuda129_vendor_plan.json. "
              "No downloaded wheel, runtime DLL or installed package metadata is part of the proposal. "
              "The local installation manifest hash is provenance, not a published member."))
    return [manifest]


def public_readme(stage):
    path = stage/"publication_evidence_README.md"
    path.write_text(
        "# v8 evidence and fixture provenance\n\n"
        "Numerical trajectories and scientific failure records are kept as saved. "
        "Their frozen hashes continue to identify the original bytes.\n\n"
        "Two synthetic invalid-parent-birth stderr records contain local path prefixes. "
        "Their originals remain in the workspace and are omitted here. "
        "The two separately named public copies substitute path placeholders. "
        "The original frozen summaries still bind their private originals and their recorded hashes do not match the public copies. "
        "The [redaction mapping](synthetic_fixture_stderr_redaction_map.json) records both original and public hashes. "
        "Run the synthetic fixture again to generate local original-form stderr.\n\n"
        "The inventory may identify exact historical source capsules at a different path. "
        "For historical replay use those bytes at the original logical path in a disposable worktree. "
        "The released current files keep their own hashes.\n\n"
        "The [vendor recipe](cuda129_vendor_public_manifest.json) identifies the official Windows wheels, "
        "license sources and hashes. Retrieve the tools locally. This archive carries no modern vendor binaries.\n",
        encoding="utf-8")
    return path


def references(value):
    if isinstance(value,dict):
        path = value.get("path")
        sha = value.get("sha256")
        if isinstance(path,str) and isinstance(sha,str) and HEX.fullmatch(sha):
            yield path,sha
        for key,child in value.items():
            if isinstance(child,str) and HEX.fullmatch(child) and key.startswith(("work/","rfm_v3/","rfm_v5/","rfm_v6/","rfm_v7/","rfm_v8/")):
                yield key,child
            if isinstance(child,(list,dict)):
                yield from references(child)
    elif isinstance(value,list):
        for item in value:
            if isinstance(item,(list,dict)):
                yield from references(item)


def link_checks(files,old):
    index = {r["path"]:r["sha256"] for r in files} | {name:r["sha256"] for name,r in old.items()}
    by_hash = defaultdict(list)
    for path,sha in index.items():
        by_hash[sha].append(path)
    refs = defaultdict(set)
    for row in files:
        if Path(row["source_path"]).suffix!=".json":
            continue
        for path,sha in references(read(WORKSPACE/row["source_path"])):
            if path.startswith(("rfm_v3/","rfm_v5/","rfm_v6/","rfm_v7/","rfm_v8/")):
                path = "work/"+path
            if path.startswith(("work/rfm_v3/","work/rfm_v5/","work/rfm_v6/","work/rfm_v7/","work/rfm_v8/")):
                relative = PurePosixPath(path)
                if not relative.is_absolute() and ".." not in relative.parts:
                    refs[path,sha].add(row["path"])
    differences = []
    for (path,sha),origins in sorted(refs.items()):
        if index.get(path)!=sha:
            alias = [x for x in sorted(by_hash[sha]) if x.startswith(PREFIX)]
            if path.startswith(PREFIX+"vendor/"):
                status = "local_supply_chain_recipe"
            elif origins=={PREFIX+"fresh_worker_mid_t9_output/protocol.json"}:
                status = "historical_overbroad_t9_snapshot"
            elif alias:
                status = "historical_exact_source_capsule"
            else:
                status = "unresolved_required_dependency"
            differences.append(dict(path=path,referenced_sha256=sha,proposed_sha256=index.get(path),
                                    classification=status,exact_aliases=alias,reference_count=len(origins),
                                    sample_origins=sorted(origins)[:5]))
    return len(refs),differences


def run(destination,reports):
    require(destination.resolve().parent==WORKSPACE/"work" and not destination.exists(),"Choose a new work-only dry-run directory")
    baseline = old_head(full=True)
    destination.mkdir()
    stage = destination/"derived"
    stage.mkdir()
    candidates,excluded,pending_files = candidate_files()
    public,redaction = make_redactions(stage)
    public += vendor_recipe(stage)
    public.append(public_readme(stage))
    replacements = {row["original_path"] for row in redaction}
    files,blocked,counts = [],[],defaultdict(int)
    pairs = [(path.relative_to(WORKSPACE).as_posix(),path) for path in candidates]
    pairs += [(PREFIX+path.name,path) for path in public]
    for report in reports:
        path = (WORKSPACE/report).resolve()
        require(path.relative_to(WORKSPACE).as_posix()==report and path.suffix in TEXT_EXT|{".pdf"},
                "Unsafe or unsupported explicit report input")
        pairs.append((report,path))
    for name,path in sorted(pairs):
        require(name not in baseline and not path.is_symlink(),"A proposed addition would overlap protected history or use a symlink")
        if name in replacements:
            excluded.append(dict(path=name,reason="Original synthetic stderr contains a local user path. See redaction mapping."))
            continue
        try:
            sha = digest(path)
            sentinel = name in SENTINEL_FILES
            if sentinel:
                require(path.read_bytes()==b"V8_FORMAL_SYNTHETIC_TRANSPORT_ONLY\n","Synthetic transport sentinel changed")
                flags,arrays = set(),{}
            else:
                flags,arrays = inspect(path)
            require(digest(path)==sha,"File changed during the privacy audit")
        except (BadZipFile,ValueError,UnicodeError,KeyError,RuntimeError) as exc:
            blocked.append(dict(path=name,reason=type(exc).__name__))
            continue
        if flags:
            blocked.append(dict(path=name,reason="privacy_pattern",categories=sorted(flags)))
            continue
        if sentinel:
            counts["synthetic_plain_text_sentinels_with_npz_suffix"] += 1
        elif path.suffix==".npz":
            counts["npz_files"] += 1
            for key,n in arrays.items():
                counts["npz_"+key] += n
        else:
            counts["text_or_pdf_files"] += 1
        row = dict(path=name,source_path=path.relative_to(WORKSPACE).as_posix(),sha256=sha,bytes=path.stat().st_size)
        if sentinel:
            row["kind"] = "plain_text_synthetic_transport_sentinel"
        files.append(row)
    linked,missing = link_checks(files,baseline)
    dependency_classes = defaultdict(int)
    for item in missing:
        dependency_classes[item["classification"]] += 1
    old_head(full=False)
    record = dict(status="provisional_dry_run_no_copy",old=dict(head=BASE_HEAD,manifest_sha256=BASE_MANIFEST,
        members_verified=BASE_COUNT,tracked_including_manifest=BASE_COUNT+1,clean=True),
        proposed_member_count=len(files),proposed_bytes=sum(row["bytes"] for row in files),
        counts=dict(counts),member_inventory=files,privacy_blocked=blocked,excluded=excluded,
        explicit_report_inputs=reports,pending=list(PENDING),pending_unclassified_paths=pending_files,
        linked_hash_records_checked=linked,linked_dependency_exceptions=missing,dependency_exception_counts=dict(dependency_classes),
        git_author_privacy=git_privacy(),
        redaction_map=redaction,scope="Workspace staging evidence only. No publication checkout, Git history, or remote mutation.")
    emit(destination/"inventory.json",record)
    print("DRY RUN",len(files),counts["npz_files"],len(blocked),len(missing),"old",BASE_COUNT,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report",action="append",default=[])
    args = parser.parse_args()
    run(args.output.resolve(),args.report)
