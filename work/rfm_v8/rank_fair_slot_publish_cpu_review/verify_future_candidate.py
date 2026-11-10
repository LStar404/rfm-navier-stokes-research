"""CPU candidate regression; not authorization for another formal/GPU attempt."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import time
from hashlib import sha256

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import formal_timing_coordinator as original
from atomic_publish_replacefile_future import save_existing_observed


def main():
    root=HERE/"future_candidate_cpu_output";root.mkdir()
    helper=HERE/"reader_fixture.exe"
    exe=HERE.parent/"rank_fair_cache_recovery_watchdog_fixture_v6_output/rank_fair_outer_watchdog.exe"
    records=[]
    for mode in ("hold_default","hold_share_delete","poll_exact_v6"):
        folder=root/mode;folder.mkdir();target=folder/"state.json"
        original.save(target,dict(sequence=-1))
        ready,stop,report=(folder/name for name in ("ready.json","stop.flag","reader_report.json"))
        p=subprocess.Popen([str(helper),mode,str(target),str(ready),str(stop),str(report),str(exe)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
        until=time.perf_counter()+3
        while not ready.exists() and time.perf_counter()<until and p.poll() is None:time.sleep(.005)
        assert ready.exists()
        success=0;failure=None
        try:
            for i in range(1 if mode=="hold_default" else 512):
                try:
                    save_existing_observed(target,dict(sequence=i),folder/"failure.json")
                    success+=1
                except Exception as error:
                    failure=json.loads((folder/"failure.json").read_bytes());break
        finally:
            stop.write_text("stop",encoding="ascii");p.wait(timeout=3)
        if mode=="hold_default":
            assert success==0 and failure["winerror"]==32 and failure["retries"]==0 and failure["no_next_slot"]
        else:
            assert success==512 and failure is None and json.loads(target.read_bytes())==dict(sequence=511)
        records.append(dict(mode=mode,successful_replacements=success,failure=failure,
                            reader=json.loads(report.read_text(encoding="utf-8-sig"))))
    result=dict(status="passed_tested_cpu_cases_only",gpu_contexts_created=0,flow_solves_run=0,records=records,
        source_sha256=sha256((HERE/"atomic_publish_replacefile_future.py").read_bytes()).hexdigest(),
        not_integrated_into_frozen_sources=True,formal_or_gpu_requalification=False,
        limitation="No transaction or hard-real-time guarantee. ReplaceFileW failure may partially change names/metadata; any failure must remain publication_unconfirmed. Reader IOException under concurrent replacement remains possible and is not converted into publication success.")
    original.save(root/"certificate.json",result)
    print(json.dumps({k:result[k] for k in ("status","gpu_contexts_created","flow_solves_run","not_integrated_into_frozen_sources")}))


if __name__=="__main__":main()
