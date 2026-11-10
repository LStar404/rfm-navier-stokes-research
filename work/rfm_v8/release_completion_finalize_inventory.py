"""Audit a small, explicit successor to the completed v8 full privacy scan.

This updates only small reviewed metadata. Publication staging independently
rehashes every source and the complete protected baseline before any copying.
It does not write a publication checkout, run a solver, commit, or push.
"""
from collections import Counter
from pathlib import Path
import ast
import copy
import json

import release_plan_v8_completion as completion
import stage_v8_publication as staging

p = completion.prior
FULL = p.WORKSPACE / "work/rfm_v8_publication_completion_final_v1/inventory.json"
FULL_SHA = "2bad47f79b5bfa3d0f32e5f20402abf5bbda8a46b2fc9c24a45d81f89a22cc97"
OUT = p.WORKSPACE / "work/rfm_v8_publication_completion_final_v2"
TEMP = p.PREFIX + "rank_fair_isolation_cpu_fixture_v2_output/before_ready/raw/outer_slot_state.json.tmp"
CAPSULE = p.V8 / "rank_fair_cache_recovery_formal_v3_output/pre_augmentation_source_snapshot"
UPDATES = {
    p.PREFIX + "release_plan_v8_completion.py": p.PREFIX + "release_plan_v8_completion.py",
    "START_HERE_v8.md": p.PREFIX + "publication_v8_README.md",
    ".github/README.md": p.PREFIX + "publication_v8_github_README.md",
}


def run():
    p.require(p.digest(FULL) == FULL_SHA and not OUT.exists(), "Expected immutable full scan or new destination missing")
    original = p.read(FULL)
    p.require(original["privacy_blocked"] == [dict(path=TEMP, reason="ValueError")] and
              not original["pending"] and original["final_report_content_review_verified"] and
              original["final_report_visual_review_verified"], "Full-scan residual scope changed")
    p.require(original["dependency_exception_counts"].get("unresolved_required_dependency") == 3,
              "Expected exactly three historical request dependencies")
    old = p.old_head(full=False)
    OUT.mkdir()
    derived = OUT / "derived"
    derived.mkdir()
    rows = {r["path"]: copy.deepcopy(r) for r in original["member_inventory"]}
    redactions = copy.deepcopy(original["redaction_map"])
    excluded = set(original["excluded_private_originals"])
    changes = []

    def audit(source, target=None):
        source = completion.root_path(source)
        target = target or source.relative_to(p.WORKSPACE).as_posix()
        before = p.digest(source)
        info = source.stat()
        flags, kind = completion.audit_evidence(source, target)
        admin = completion.scrub_administration(p.read(source))[1] if source.suffix == ".json" else 0
        if (flags == {"user_path"} or admin) and source.suffix in p.TEXT_EXT | {".cs"}:
            source, mapping = completion.redacted_copy(source, target, derived)
            redactions.append(mapping)
            excluded.add(target)
            target, before, flags = mapping["public_path"], mapping["public_sha256"], set()
            kind = "explicit_public_path_redaction"
        p.require(not flags, "Incremental source has a privacy flag")
        row = dict(path=target, source_path=source.relative_to(p.WORKSPACE).as_posix(), sha256=before, bytes=source.stat().st_size)
        if kind is not None:
            row["kind"] = kind
        if target in old:
            p.require(target == ".github/README.md" and row["source_path"] == UPDATES[target], "Unexpected historical overwrite")
            row["replaces_old"] = {k: old[target][k] for k in ("sha256", "bytes")}
            row["replacement_authorization"] = "Explicitly authorized v8 GitHub entry README update; all experimental base members remain protected."
        previous = rows.get(target)
        rows[target] = row
        changes.append(dict(target=target, previous_sha256=previous["sha256"] if previous else None,
                            public_sha256=row["sha256"], bytes=row["bytes"]))
        return row

    for target, source in UPDATES.items():
        audit(source, target)
    ast.parse((p.V8 / "release_plan_v8_completion.py").read_text(encoding="utf-8"))
    audit(TEMP)
    audit(p.PREFIX + "release_completion_final_v1_source_snapshot/release_plan_v8_completion.py")
    p.require(p.digest(p.V8 / "release_completion_final_v1_source_snapshot/release_plan_v8_completion.py") ==
              next(r["sha256"] for r in original["member_inventory"] if r["path"] == p.PREFIX + "release_plan_v8_completion.py"),
              "Full scan source capsule is not byte-exact")
    audit(Path(__file__).resolve().relative_to(p.WORKSPACE).as_posix())
    for source in sorted(CAPSULE.glob("*.json")):
        audit(source.relative_to(p.WORKSPACE).as_posix())
    redactions += completion.historical_request_derivatives(redactions)
    excluded.update(r["original_path"] for r in redactions)
    p.require(len({(r["original_path"], r["original_sha256"]) for r in redactions}) == len(redactions),
              "Ambiguous original-path/hash public mapping")

    # Canonical maps are regenerated from audited rows and become manifest members.
    map_path = derived / "publication_path_redaction_map.json"
    p.emit(map_path, dict(schema="rfm-v8-public-derivative-map-1", status="public_derivative_mapping", records=redactions,
        scope="Original frozen hashes identify local private originals; public hashes identify separately named path/admin-redacted bytes. Three exact recovered historical request capsules remain private; their original logical paths and hashes resolve only to explicitly different public derivatives. Original states and numerical results are unchanged. Public evidence must not be executed as a local runtime request."))
    audit(map_path.relative_to(p.WORKSPACE).as_posix(), p.PREFIX + map_path.name)
    note_path = derived / "publication_completion_README.md"
    note_path.write_text("# v8 publication completion evidence\n\n"
        "The canonical path-redaction map covers two synthetic transport logs, two real private-context failure logs, and path-bearing watchdog/recovery evidence. Original files stay local. Public derivatives have distinct filenames and different SHA-256 values; original hashes do not authenticate their public bytes. Recorded numerical values and outcomes are unchanged.\n\n"
        "Three historical watchdog requests were recovered by removing only the three documented prospective supervision-input additions. Every recovered byte stream matched its independently recorded old SHA. These exact private capsules stay local; separately named public derivatives and explicit original-path/original-hash mappings close their publication dependencies.\n\n"
        "The project executable recipe binds system compiler, available source capsules and archived binaries separately. Vendor runtimes are excluded; historical failed binaries without exact recovered source are identified honestly. Cross-machine byte-identical rebuilding is not claimed.\n\n"
        "This note is provenance, not independent permission to publish. The final inventory requires content review, visual review, dependency closure and privacy checks. The staging tool verifies every source and protected baseline hash again before copying.\n", encoding="utf-8", newline="")
    audit(note_path.relative_to(p.WORKSPACE).as_posix(), p.PREFIX + note_path.name)

    # Link checks read JSON metadata only, never the numerical archives again.
    linked, exceptions = p.link_checks(list(rows.values()), old)
    pairs = {(r["original_path"], r["original_sha256"]):r for r in redactions}
    previous_exceptions = {(r["path"],r["referenced_sha256"]):r for r in original["linked_dependency_exceptions"]}
    for item in exceptions:
        public = pairs.get((item["path"],item["referenced_sha256"]))
        if public:
            item.update(classification="explicit_private_original_public_derivative", public_path=public["public_path"], public_sha256=public["public_sha256"])
        elif previous_exceptions.get((item["path"],item["referenced_sha256"]),{}).get("classification") == "historical_draft_checkpoint_not_final_dependency":
            previous = previous_exceptions[item["path"],item["referenced_sha256"]]
            p.require(item["sample_origins"] == previous["sample_origins"], "Draft exception scope changed")
            item.update(classification=previous["classification"],scope=previous["scope"])
    p.require(all(r["classification"] in staging.ALLOWED_DEPENDENCIES for r in exceptions), "Dependency remains unresolved")
    historical_path = derived / "publication_historical_source_map.json"
    p.emit(historical_path,dict(schema="rfm-v8-historical-source-map-1",status="exact_byte_aliases",records=[
        dict(original_path=r["path"],original_sha256=r["referenced_sha256"],public_path=r["exact_aliases"][0],public_sha256=r["referenced_sha256"],kind="exact_historical_source_alias")
        for r in exceptions if r["classification"] == "historical_exact_source_capsule"],
        scope="Only byte-exact historical aliases. Authenticate this map and all targets through the complete manifest before use. Path-redacted historical requests are exclusively in the other canonical map."))
    audit(historical_path.relative_to(p.WORKSPACE).as_posix(),p.PREFIX+historical_path.name)

    final, review, visual = (p.read(p.V8 / name) for name in ("report_final_round.json","report_final_round_publication_review.json","report_visual_quality_review.json"))
    p.require(final["reviewed_for_publication"] and review["status"] == "passed" and review["addendum_sha256"] == p.digest(p.V8 / "report_final_round.json") and
              review["builder_sha256"] == p.digest(p.V8 / "build_v8_report.py") == visual["builder_sha256"] and
              visual["independent_content_review_sha256"] == p.digest(p.V8 / "report_final_round_publication_review.json") and
              visual["pdf_sha256"] == p.digest(completion.root_path(visual["pdf"])) and
              visual["all_pages_visually_inspected"] == list(range(1,visual["pages"]+1)), "Final report gates changed")
    document = copy.deepcopy(original)
    document.update(publication_ready=True, privacy_blocked=[],member_inventory=sorted(rows.values(),key=lambda r:r["path"]),
        redaction_map=redactions,excluded_private_originals=sorted(excluded),linked_hash_records_checked=linked,
        linked_dependency_exceptions=exceptions,dependency_exception_counts=dict(Counter(r["classification"] for r in exceptions)),
        proposed_member_count=len(rows),proposed_bytes=sum(r["bytes"] for r in rows.values()),
        newly_proposed_candidates=len(rows)-original["retained_prior_candidates"],
        prior_full_scan=dict(path=FULL.relative_to(p.WORKSPACE).as_posix(),sha256=FULL_SHA,members=original["proposed_member_count"],
            method="Inherit complete hash/privacy review of unchanged rows. Audit only explicitly listed small metadata changes; final staging must rehash every source and the entire baseline."),
        incremental_changes=changes)
    document["authorized_replacements"]=[dict(path=".github/README.md",before=rows[".github/README.md"]["replaces_old"],
        after={k:rows[".github/README.md"][k] for k in ("sha256","bytes")})]
    staging.verify_maps(document,rows,old)
    staging.verify_entry_links(rows,old)
    for target,row in rows.items():
        source=completion.root_path(row["source_path"])
        p.require(source.stat().st_size==row["bytes"] and target not in excluded and row["source_path"] not in excluded,
                  "Unchanged source size changed or a private original entered the inventory")
    p.old_head(full=False)
    p.require(p.digest(FULL)==FULL_SHA,"Full scan changed during incremental audit")
    p.emit(OUT / "inventory.json",document)
    print(json.dumps(dict(status="approved_inventory_no_checkout_write",path=(OUT/"inventory.json").relative_to(p.WORKSPACE).as_posix(),
        sha256=p.digest(OUT/"inventory.json"),members=len(rows),bytes=document["proposed_bytes"],redactions=len(redactions),
        dependency_exception_counts=document["dependency_exception_counts"])))


if __name__ == "__main__":
    run()
