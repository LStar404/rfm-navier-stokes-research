"""512 immutable records read concurrently with exact frozen v6 Read; no GPU or production edits."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import formal_timing_coordinator as original


def main():
    output=HERE/"immutable_publication_cpu_output";output.mkdir()
    frozen=HERE.parent/"rank_fair_cache_recovery_watchdog_fixture_v6_output/rank_fair_outer_watchdog.exe"
    process=subprocess.Popen([str(HERE/"immutable_reader_fixture.exe"),str(output),str(frozen)],stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    until=time.perf_counter()+3
    while not (output/"reader_ready.json").exists() and time.perf_counter()<until and process.poll() is None:time.sleep(.005)
    assert (output/"reader_ready.json").exists()
    for sequence in range(512):
        original.save(output/f"state_{sequence:04d}.json",dict(sequence=sequence,index=sequence//2,
            phase="active" if sequence%2==0 else "terminal",payload=str(sequence)+":"+"x"*4096))
    process.wait(timeout=15)
    reader=json.loads((output/"reader_result.json").read_text(encoding="utf-8-sig"))
    assert process.returncode==0 and reader["consumed"]==512 and reader["errors"]==0
    paths=sorted(output.glob("state_*.json"));assert len(paths)==512
    before={p.name:sha256(p.read_bytes()).hexdigest() for p in paths}
    try:
        original.save(paths[0],dict(sequence="forbidden_overwrite"))
        raise AssertionError("Existing immutable name unexpectedly overwritten")
    except FileExistsError as error:
        negative=dict(type=type(error).__name__,errno=error.errno,winerror=error.winerror)
    assert {p.name:sha256(p.read_bytes()).hexdigest() for p in paths}==before
    report=dict(status="passed_targeted_cpu_only",gpu_contexts_created=0,flow_solves_run=0,
        immutable_records=512,slot_pairs=256,writer="Unchanged original save(replace=False); complete temporary then rename to unused name",
        reader=reader,overwrite_rejected=negative,all_record_hashes=before,
        frozen_reader_executable_sha256=sha256(frozen.read_bytes()).hexdigest(),
        production_integration=False,T72_qualification=False,
        limitation="Proves only the tested concurrent immutable-file publication/read sequence. Future integration must retain slot authentication, deadlines, complete terminal evidence, fail-closed behavior and requalification.")
    original.save(output/"certificate.json",report)
    print(json.dumps({k:report[k] for k in ("status","immutable_records","slot_pairs","production_integration","T72_qualification")}))


if __name__=="__main__":main()
