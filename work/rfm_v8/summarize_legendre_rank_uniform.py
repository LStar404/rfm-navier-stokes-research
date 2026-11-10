"""Reconcile the original uniform case records without importing a solver."""

import hashlib
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent/"legendre_rank_uniform_output"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    path = OUT/"protocol.json"
    sha = digest(path)
    protocol = read(path)
    rows = []
    records = {}
    for job in protocol["jobs"]:
        sidecar = OUT/(job["name"]+".json")
        row = read(sidecar)
        if row["job"] != job or row["protocol_sha256"] != sha:
            raise RuntimeError("A registered Legendre result changed")
        if "artifact" in row and digest(OUT/row["artifact"]) != row["artifact_sha256"]:
            raise RuntimeError("A saved Legendre trajectory changed")
        rows.append(row)
        records[sidecar.name] = digest(sidecar)
    payload = dict(status="complete",protocol_sha256=sha,rows=rows,raw_records=records,
                   scope=protocol["template"]["scope"],summary_builder_sha256=digest(Path(__file__)),
                   switch_candidates=[r["job"]["name"] for r in rows if r["qualified"]],
                   physically_eligible_without_added_cross_rank_guard=[r["job"]["name"] for r in rows
                       if r.get("physical_pass") and r.get("wake",{}).get("sustained_alternating_probe_signs")],
                   timing_claim_supported=False)
    with (OUT/"summary.json").open("x",encoding="utf-8") as handle:
        json.dump(payload,handle,indent=2,allow_nan=False)
    print(digest(OUT/"summary.json"),flush=True)


if __name__ == "__main__":
    main()
