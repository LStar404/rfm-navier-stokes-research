"""Validate or stage an explicitly approved v8 inventory; never commit or push.

Default operation is read-only. Root publisher must supply --stage explicitly.
Only the fixed publication checkout is writable. Baseline bytes, approved input
hashes, public-derivative maps and final coverage are checked independently.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import uuid

WORKSPACE = Path(__file__).resolve().parents[2]
CHECKOUT = WORKSPACE / "release/rfm-navier-stokes-publication"
BASE_HEAD = "dcb69a05c60a4d2c18422b9018a8421145ced190"
BASE_MANIFEST = "27299292a78718b57c18d22843dcae9b6fe55f249cd0af07e2e33f2b7a7c0089"
BASE_COUNT = 1831
ENTRY = ".github/README.md"
HISTORY = "publication_history/v8_baseline_" + BASE_HEAD
ARCHIVES = {HISTORY + "/github_README.md": ENTRY, HISTORY + "/MANIFEST.json": "MANIFEST.json"}
TARGET_OVERRIDES = {"work/rfm_v8/publication_v8_README.md": "START_HERE_v8.md",
                    "work/rfm_v8/publication_v8_github_README.md": ENTRY}
MAP_PATH = "work/rfm_v8/publication_path_redaction_map.json"
HISTORY_MAP_PATH = "work/rfm_v8/publication_historical_source_map.json"
ALLOWED_DEPENDENCIES = {"local_supply_chain_recipe", "historical_overbroad_t9_snapshot", "historical_exact_source_capsule",
                        "explicit_private_original_public_derivative", "historical_draft_checkpoint_not_final_dependency"}
HEX = re.compile(r"[0-9a-f]{64}\Z")
RESERVED = re.compile(r"(?i)(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?\Z")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative(value):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value and
            not any(ord(c) < 32 for c in value), "Invalid relative publication path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value and all(
        part not in ("", ".", "..") and part == part.rstrip(" .") and not RESERVED.fullmatch(part) for part in path.parts),
        "Noncanonical or reserved publication path")
    require(path.parts[0].casefold() != ".git", "Git internals cannot be inventory members")
    return path


def safe(root, name, *, exists=False):
    parts = relative(name).parts
    root = root.resolve()
    require(root == WORKSPACE or root == CHECKOUT.resolve(), "Unexpected path authority")
    path = root
    for part in parts:
        path = path / part
        if path.exists() or path.is_symlink():
            info = path.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400),
                    "Symlink, junction or reparse point is forbidden")
    require(path.resolve().is_relative_to(root) and path.resolve().relative_to(root).as_posix() == name,
            "Resolved path escaped its intended root")
    if exists:
        require(path.is_file(), "Required regular file is missing")
    return path


def git(*args):
    allowed = {"rev-parse", "status", "ls-files", "symbolic-ref"}
    require(args and args[0] in allowed, "Only read-only Git queries are permitted")
    return subprocess.run(["git", "-C", str(CHECKOUT), *args], check=True, capture_output=True).stdout


def files_under_checkout():
    files = {}
    for current, directories, names in os.walk(CHECKOUT, followlinks=False):
        parent = Path(current)
        if parent == CHECKOUT:
            directories[:] = [n for n in directories if n != ".git"]
        for name in directories:
            child = parent / name
            info = child.lstat()
            require(not child.is_symlink() and not (getattr(info, "st_file_attributes", 0) & 0x400), "Checkout contains a linked directory")
        for name in names:
            path = parent / name
            rel = path.relative_to(CHECKOUT).as_posix()
            safe(CHECKOUT, rel, exists=True)
            files[rel] = path
    return files


def verify_bytes(path, row):
    require(type(row.get("bytes")) is int and row["bytes"] >= 0 and HEX.fullmatch(row.get("sha256", "")), "Invalid hash or byte count")
    require(path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], "File size or SHA-256 mismatch: " + path.name)


def baseline():
    safe(WORKSPACE, "release/rfm-navier-stokes-publication/MANIFEST.json", exists=True)
    safe(WORKSPACE, "release/rfm-navier-stokes-publication/.git/config", exists=True)
    require(git("rev-parse", "HEAD").decode().strip() == git("rev-parse", "main").decode().strip() == BASE_HEAD,
            "Publication checkout is not at the authorized baseline HEAD")
    require(not git("status", "--porcelain=v1", "-z", "--untracked-files=all"), "Publication checkout is not clean")
    manifest = safe(CHECKOUT, "MANIFEST.json", exists=True)
    require(digest(manifest) == BASE_MANIFEST, "Baseline manifest bytes changed")
    rows = read(manifest)["files"]
    old = {r["path"]: r for r in rows}
    require(len(rows) == len(old) == BASE_COUNT and len({n.casefold() for n in old}) == BASE_COUNT, "Baseline manifest membership changed")
    tracked = set(git("ls-files", "-z").decode("utf-8").rstrip("\0").split("\0"))
    require(tracked == set(old) | {"MANIFEST.json"} == set(files_under_checkout()), "Baseline tracked or physical coverage differs")
    for name, row in old.items():
        verify_bytes(safe(CHECKOUT, name, exists=True), row)
    return old, manifest.read_bytes(), safe(CHECKOUT, ENTRY, exists=True).read_bytes()


def appended_v8(name):
    return name.startswith("work/rfm_v8/") or name.startswith("outputs/rfm_v8_figures/") or (name.startswith("outputs/") and "v8" in PurePosixPath(name).name.casefold()) or name == "START_HERE_v8.md"


def private_name(name):
    parts = PurePosixPath(name).parts
    return any(p.casefold() in {"vendor", "__pycache__", "private-original", "private_original", "private_originals"} for p in parts) or any(
        x in PurePosixPath(name).name.casefold() for x in ("usage_checkpoint", "quota", "_admin", "account"))


def verify_maps(document, members, old):
    require(MAP_PATH in members and HISTORY_MAP_PATH in members, "Explicit public and historical maps are required")
    public_map = read(safe(WORKSPACE, members[MAP_PATH]["source_path"], exists=True))
    historical_map = read(safe(WORKSPACE, members[HISTORY_MAP_PATH]["source_path"], exists=True))
    require(public_map["schema"] == "rfm-v8-public-derivative-map-1" and historical_map["schema"] == "rfm-v8-historical-source-map-1", "Unexpected mapping schema")
    require(public_map["records"] == document["redaction_map"], "Canonical public mapping differs from approved inventory")
    excluded = set(document["excluded_private_originals"])
    require({r["original_path"] for r in public_map["records"]} <= excluded, "Public map omitted private-original exclusions")
    index = {n: r["sha256"] for n, r in old.items()} | {n: r["sha256"] for n, r in members.items()}
    public_pairs = {}
    for r in public_map["records"]:
        relative(r["original_path"])
        relative(r["public_path"])
        require(r.get("original_in_checkout") is False and r["original_path"] != r["public_path"] and
                r["original_path"] not in members and r["public_path"] in members and index[r["public_path"]] == r["public_sha256"] and
                members[r["public_path"]]["bytes"] == r["public_bytes"] and HEX.fullmatch(r["original_sha256"]), "Invalid private-original/public-derivative mapping")
        public_pairs[r["original_path"], r["original_sha256"]] = r
    historical_pairs = {}
    for r in historical_map["records"]:
        relative(r["original_path"])
        relative(r["public_path"])
        require(r["original_sha256"] == r["public_sha256"] == index.get(r["public_path"]), "Historical alias is not exact bytes")
        historical_pairs[r["original_path"], r["original_sha256"]] = r
    for item in document["linked_dependency_exceptions"]:
        classification = item["classification"]
        require(classification in ALLOWED_DEPENDENCIES, "Unresolved or unapproved dependency exception")
        if classification == "explicit_private_original_public_derivative":
            mapping = public_pairs.get((item["path"], item["referenced_sha256"]))
            require(mapping and mapping["public_path"] == item["public_path"] and mapping["public_sha256"] == item["public_sha256"], "Dependency lacks its explicit public derivative")
        elif classification == "historical_exact_source_capsule":
            mapping = historical_pairs.get((item["path"], item["referenced_sha256"]))
            require(mapping and mapping["public_path"] in item["exact_aliases"] and index.get(mapping["public_path"]) == item["referenced_sha256"],
                    "Missing explicitly mapped exact historical dependency")
        elif classification == "local_supply_chain_recipe":
            require(item["path"].startswith("work/rfm_v8/vendor/"), "Supplier exception exceeded its scope")
        elif classification == "historical_overbroad_t9_snapshot":
            require(item["sample_origins"] == ["work/rfm_v8/fresh_worker_mid_t9_output/protocol.json"], "Historical T9 exception exceeded its scope")
        elif classification == "historical_draft_checkpoint_not_final_dependency":
            require(item.get("scope") and item["sample_origins"] == ["work/rfm_v8/report_completion_evidence.json"], "Draft exception exceeded its scope")
    for row in members.values():
        require(row["path"] not in excluded and row["source_path"] not in excluded, "Private original must never be copied")


def verify_entry_links(members, old):
    from urllib.parse import unquote, urlsplit
    import posixpath
    available = set(members) | set(old) | set(ARCHIVES) | {"MANIFEST.json"}
    for target in (ENTRY, "START_HERE_v8.md"):
        text = safe(WORKSPACE, members[target]["source_path"], exists=True).read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            link = link.strip().strip("<>")
            require(not urlsplit(link).scheme and not link.startswith("//"), "Entry document must use repository-relative links")
            path = unquote(link.split("#", 1)[0])
            if not path:
                continue
            joined = posixpath.normpath(posixpath.join(posixpath.dirname(target), path))
            require(joined in available, "Entry document refers to an unpublished file: " + joined)


def preflight(inventory_path, expected_sha):
    require(HEX.fullmatch(expected_sha), "Supply the reviewed inventory SHA-256")
    inventory_path = inventory_path.resolve()
    require(inventory_path.is_relative_to(WORKSPACE / "work"), "Inventory must be a workspace work artifact")
    safe(WORKSPACE, inventory_path.relative_to(WORKSPACE).as_posix(), exists=True)
    require(digest(inventory_path) == expected_sha, "Inventory differs from the reviewed input hash")
    document = read(inventory_path)
    require(document.get("schema") == "rfm-v8-publication-inventory-1" and document.get("status") == "publication_completion_dry_run_no_copy" and
            document.get("publication_ready") is True and document.get("final_report_content_review_verified") is True and
            document.get("pending") == [] and document.get("privacy_blocked") == [], "Final privacy, content or completion approval is missing")
    require(document["old"]["head"] == BASE_HEAD and document["old"]["manifest_sha256"] == BASE_MANIFEST and
            document["old"]["members_verified"] == BASE_COUNT and document["old"]["clean"] is True, "Inventory baseline differs")
    require(document["dependency_exception_counts"].get("unresolved_required_dependency", 0) == 0, "Required dependencies remain unresolved")
    actual_counts = dict(Counter(r["classification"] for r in document["linked_dependency_exceptions"]))
    require(actual_counts == document["dependency_exception_counts"], "Dependency counters do not match the detailed exceptions")
    old, old_manifest, old_entry = baseline()
    rows = document["member_inventory"]
    members = {r["path"]: r for r in rows}
    require(len(rows) == len(members) == document["proposed_member_count"] and len({n.casefold() for n in members}) == len(members), "Duplicate inventory targets")
    require(sum(r["bytes"] for r in rows) == document["proposed_bytes"], "Inventory byte count differs")
    require(document["authorized_target_overrides"] == TARGET_OVERRIDES and {n for n in members if n in old} == {ENTRY}, "Unexpected historical replacement")
    require(len(document["authorized_replacements"]) == 1, "Exactly one entry replacement is allowed")
    replacement = document["authorized_replacements"][0]
    require(replacement == dict(path=ENTRY, before={k: old[ENTRY][k] for k in ("sha256", "bytes")},
                               after={k: members[ENTRY][k] for k in ("sha256", "bytes")}), "Entry replacement digest differs")
    occupied = {n.casefold() for n in old} | {"manifest.json"}
    for target, row in members.items():
        relative(target)
        relative(row["source_path"])
        require(target == ENTRY or appended_v8(target), "Only new v8 files and the authorized entry may be staged")
        require(not private_name(target) and not private_name(row["source_path"]), "Private administration or supplier payload cannot be copied")
        source = safe(WORKSPACE, row["source_path"], exists=True)
        require(not source.is_relative_to(CHECKOUT), "Reviewed inputs must be outside the publication checkout")
        destination = safe(CHECKOUT, target)
        if target != ENTRY:
            require(target.casefold() not in occupied and not destination.exists(), "Append target conflicts with history")
        verify_bytes(source, row)
    for source, target in TARGET_OVERRIDES.items():
        require(members[target]["source_path"] == source, "Entry source mapping changed")
    for name in ARCHIVES:
        require(not safe(CHECKOUT, name).exists() and name not in members and name.casefold() not in occupied, "Historical backup target already exists")
    verify_maps(document, members, old)
    verify_entry_links(members, old)
    require(digest(inventory_path) == expected_sha, "Inventory changed during verification")
    return document, members, old, old_manifest, old_entry


def serialized(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def copy_exclusive(source, destination, row):
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe(CHECKOUT, destination.relative_to(CHECKOUT).as_posix())
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(1024 * 1024), b""):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    verify_bytes(destination, row)


def replace_expected(destination, data, expected):
    require(digest(destination) == expected, "Historical target changed before its authorized replacement")
    temporary = destination.with_name(destination.name + ".v8-stage-" + uuid.uuid4().hex)
    safe(CHECKOUT, temporary.relative_to(CHECKOUT).as_posix())
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        require(digest(destination) == expected, "Historical target changed during replacement preparation")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def stage(inventory_path, expected_sha):
    document, members, old, old_manifest, old_entry = preflight(inventory_path, expected_sha)
    # This lock is operational metadata under .git, never a publication member.
    lock = CHECKOUT / ".git/rfm_v8_stage.lock"
    require((CHECKOUT / ".git").is_dir() and not lock.exists(), "A staging operation may already be active")
    with lock.open("xb") as handle:
        handle.write(expected_sha.encode("ascii"))
    created, mutations = {}, []
    try:
        # Reconfirm cleanliness after acquiring the one-publisher lock.
        require(not git("status", "--porcelain=v1", "-z", "--untracked-files=all") and digest(inventory_path) == expected_sha, "Input or checkout changed before staging")
        for name, original in ARCHIVES.items():
            data = old_entry if original == ENTRY else old_manifest
            path = safe(CHECKOUT, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            created[name] = dict(path=name, bytes=len(data), sha256=sha256(data).hexdigest())
            verify_bytes(path, created[name])
        for name, row in members.items():
            if name == ENTRY:
                continue
            path = safe(CHECKOUT, name)
            copy_exclusive(safe(WORKSPACE, row["source_path"], exists=True), path, row)
            created[name] = {k: row[k] for k in ("path", "sha256", "bytes")}
        # None of the other 1,830 old members may have changed while copying.
        for name, row in old.items():
            verify_bytes(safe(CHECKOUT, name, exists=True), row)
        entry_source = safe(WORKSPACE, members[ENTRY]["source_path"], exists=True)
        verify_bytes(entry_source, members[ENTRY])
        entry_bytes = entry_source.read_bytes()
        require(sha256(entry_bytes).hexdigest() == members[ENTRY]["sha256"], "Entry changed while being read")
        replace_expected(safe(CHECKOUT, ENTRY, exists=True), entry_bytes, old[ENTRY]["sha256"])
        mutations.append((ENTRY, old_entry, members[ENTRY]["sha256"]))
        expected = {n: {k: r[k] for k in ("path", "sha256", "bytes")} for n, r in old.items()}
        expected.update(created)
        expected[ENTRY] = {k: members[ENTRY][k] for k in ("path", "sha256", "bytes")}
        new_manifest = dict(format=2, case="Re100 finite full annulus; preserved v6/v7 history and completed v8 evidence without a new isolated speed ranking",
            parent_head=BASE_HEAD, parent_manifest_sha256=BASE_MANIFEST, publication_inventory_sha256=expected_sha,
            preserved_unchanged_old_members=BASE_COUNT - 1, archived_prior_entry=HISTORY + "/github_README.md", archived_prior_manifest=HISTORY + "/MANIFEST.json",
            files=[expected[n] for n in sorted(expected)])
        require("MANIFEST.json" not in expected and set(files_under_checkout()) == set(expected) | {"MANIFEST.json"}, "Staged physical coverage differs before manifest replacement")
        for name, row in expected.items():
            verify_bytes(safe(CHECKOUT, name, exists=True), row)
        manifest_bytes = serialized(new_manifest)
        replace_expected(safe(CHECKOUT, "MANIFEST.json", exists=True), manifest_bytes, BASE_MANIFEST)
        mutations.append(("MANIFEST.json", old_manifest, sha256(manifest_bytes).hexdigest()))
        require(read(CHECKOUT / "MANIFEST.json") == new_manifest and set(files_under_checkout()) == set(expected) | {"MANIFEST.json"}, "Final manifest coverage failed")
        for name, row in expected.items():
            verify_bytes(safe(CHECKOUT, name, exists=True), row)
        require(digest(inventory_path) == expected_sha and git("rev-parse", "HEAD").decode().strip() == BASE_HEAD, "Input or HEAD changed during staging")
        return dict(status="staged_not_committed_or_pushed", inventory_sha256=expected_sha, unchanged_historical_members=1830,
                    new_members=len(created), manifest_members=len(expected), manifest_sha256=digest(CHECKOUT / "MANIFEST.json"),
                    original_entry_backup_sha256=digest(CHECKOUT / HISTORY / "github_README.md"), original_manifest_backup_sha256=digest(CHECKOUT / HISTORY / "MANIFEST.json"))
    except BaseException:
        # Undo only our two authorized replacements if their current bytes are
        # still exactly ours. Never overwrite a concurrently edited file.
        for name, original_bytes, installed_sha in reversed(mutations):
            path = safe(CHECKOUT, name, exists=True)
            if digest(path) == installed_sha:
                replace_expected(path, original_bytes, installed_sha)
        # Leave appended files as visible, uncommitted evidence for root review;
        # do not recursively delete data or automatically retry a partial stage.
        raise
    finally:
        if lock.exists() and lock.read_bytes() == expected_sha.encode("ascii"):
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--inspect", action="store_true", help="Read-only verification; this is also the default")
    action.add_argument("--stage", action="store_true", help="Root publisher only: copy the approved inventory; default is read-only")
    args = parser.parse_args()
    if args.stage:
        result = stage(args.inventory, args.inventory_sha256)
    else:
        _, members, _, _, _ = preflight(args.inventory, args.inventory_sha256)
        result = dict(status="verified_no_copy", approved_members=len(members), inventory_sha256=args.inventory_sha256,
                      checkout_changed=False, commit_or_push_performed=False)
    print(json.dumps(result, ensure_ascii=False))
