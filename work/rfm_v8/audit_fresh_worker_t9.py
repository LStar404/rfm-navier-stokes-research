"""Independently audit the saved v8 T9 import-ablation data."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
OUT = HERE/"fresh_worker_mid_t9_output"
V7 = WORK/"rfm_v7"/"prepared_formal_timing_mid_output"
EVIDENCE = HERE/"fresh_worker_mid_t9_independent_audit.json"
ARRAYS = ("states","times","probe_times","transverse","vorticity")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        from hashlib import file_digest
        return file_digest(stream,"sha256").hexdigest()


def fingerprint(data):
    return {k:dict(shape=list(data[k].shape),dtype=data[k].dtype.str,sha256=sha256(data[k].tobytes()).hexdigest())
            for k in ARRAYS}


def non_time_metadata(data):
    return {k:v for k,v in data.items() if k != "seconds" and not k.endswith("_seconds")}


def main():
    p,journal,summary = (read(OUT/n) for n in ("protocol.json","runs.json","summary.json"))
    old = read(V7/"timing.json")
    if not (summary["protocol_sha256"] == digest(OUT/"protocol.json") and summary["rows_sha256"] == digest(OUT/"runs.json")):
        raise ValueError("Summary and raw run bindings differ")
    source_drift = [name for name,sha in p["source_hashes"].items() if not (WORK/name).is_file() or digest(WORK/name)!=sha]
    groups,records,arc = {},[],[]
    for i,(request,row) in enumerate(zip(p["requests"],journal,strict=True)):
        rpath = OUT/f"request_{i:03d}.json"
        trajectory = OUT/request["trajectory"]
        if not (row["index"] == request["index"] == i and read(rpath) == request and
                digest(rpath) == row["request_sha256"] and digest(trajectory) == row["trajectory_sha256"]):
            raise ValueError(f"Request or trajectory binding differs at {i}")
        clocks = [row[k] for k in ("launch_monotonic_ns","ready_received_monotonic_ns","go_sent_monotonic_ns",
                                   "return_received_monotonic_ns","save_sent_monotonic_ns","exit_monotonic_ns")]
        if not (clocks == sorted(clocks) and 0 < row["solver_seconds"] <= (clocks[3]-clocks[2])/1e9 and
                row["solver_seconds"] == row["solver_core_seconds"]+row["model_release_seconds"] and
                row["fresh_process_return_seconds"] == (clocks[3]-clocks[0])/1e9 and
                row["fresh_process_overhead_seconds"] == row["fresh_process_return_seconds"]-row["solver_seconds"] and
                (i == 0 or journal[i-1]["exit_monotonic_ns"] < clocks[0])):
            raise ValueError(f"Dual clocks or process order differ at {i}")
        identity = row["child_identity"]
        terminal = row["terminal_identity"]
        for marker in (row["ready"],row["marker"]):
            if not (marker["index"] == i and marker["request_sha256"] == row["request_sha256"] and
                    marker["run_token"] == request["run_token"] and marker["inherited_affinity_mask"] == request["affinity_mask"] and
                    all(marker[k] == identity[k] == terminal[k] for k in ("pid","birth_filetime_100ns"))):
                raise ValueError(f"Fresh worker identity marker differs at {i}")
        if not terminal["exit_filetime_100ns"] > identity["birth_filetime_100ns"] or row["exit_code"] != 0:
            raise ValueError(f"Fresh worker terminal evidence differs at {i}")
        signature = row["marker"]["spatial_sha256"]
        key = request["name"]+"/"+request["arm"]
        with np.load(trajectory,allow_pickle=False) as raw:
            if not (json.loads(str(raw["config"])) == request["config"] and
                    np.array_equal(raw["times"],request["solve_args"]["snapshots"]) and
                    np.array_equal(raw["probe_times"],np.arange(0.,9.+.001,.125)) and
                    all(np.isfinite(raw[n]).all() for n in ARRAYS)):
                raise ValueError(f"Fresh trajectory metadata/finite check differs at {i}")
            arrays = fingerprint(raw)
            physical_metadata = non_time_metadata(json.loads(str(raw["metadata"])))
            group = dict(arrays=arrays,signature=signature,non_time_metadata=physical_metadata)
            if groups.setdefault(key,group) != group or summary["groups"][key]["reference"] != arrays:
                raise ValueError(f"Import policy changes arrays, signature or metadata at {i}")
            if request["block"] == 0 and request["imports"] == "minimal":
                arm = "switch_both" if request["arm"] == "combined" else request["arm"]
                expected = next(v for v in old["rows"] if v["block"]==1 and v["name"]==request["name"] and v["arm"]==arm)
                archive = V7/expected["artifact"]
                if digest(archive) != expected["sha256"] or signature != expected["marker"]["spatial_sha256"]:
                    raise ValueError("Archived numerical source or spatial signature differs")
                with np.load(archive,allow_pickle=False) as stored:
                    matched = {name:raw[name].shape==stored[name][:len(raw[name])].shape and
                               raw[name].dtype==stored[name].dtype and
                               raw[name].tobytes()==stored[name][:len(raw[name])].tobytes() for name in ARRAYS}
                if not all(matched.values()):
                    raise ValueError("T9 output differs from archived T72 prefix")
                arc.append(dict(name=key,path=archive.relative_to(WORK).as_posix(),sha256=expected["sha256"],
                                arrays_byte_exact=matched,spatial_signature_exact=True))
        records.append(dict(index=i,request_sha256=row["request_sha256"],trajectory_sha256=row["trajectory_sha256"],
                            identity_and_dual_clock_order=True,non_time_metadata_exact_within_group=True))
    if len(records)!=30 or len(groups)!=5 or len(arc)!=5:
        raise ValueError("Incomplete fresh-worker audit inventory")
    evidence = dict(status="passed",source_sha256=digest(Path(__file__)),
        inputs={n:digest(OUT/n) for n in ("protocol.json","runs.json","summary.json")},
        v7_timing_sha256=digest(V7/"timing.json"),source_hashes_observed_match=len(p["source_hashes"])-len(source_drift),
        source_hashes_observed_drift=source_drift,archived_prefixes=arc,rows=records,
        known_unbound_runtime_dependency=dict(path="rfm_v6/fd_full_band.dll",sha256=digest(WORK/"rfm_v6/fd_full_band.dll")),
        limitations=["Diagnostic runs have no fixed-window isolation record and only two blocks.",
            "Sources were hashed before the original run and are checked again here. The original run did not perform its own closing hash check.",
            "Original source map included unrelated files and omitted the native FD DLL and the generating v7 protocol JSON.",
            "Source audit does not establish the stability of unrecorded system library binaries.",
            "Prefix agreement and within-solver equality do not by themselves qualify the complete thin T72 implementation."])
    with EVIDENCE.open("x",encoding="utf-8") as stream:
        json.dump(evidence,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print("PASS",len(records),len(arc),digest(EVIDENCE),flush=True)


if __name__ == "__main__":
    main()
