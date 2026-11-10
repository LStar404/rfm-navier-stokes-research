"""Verify byte-identical artifacts from two independently empty CPU builds."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


def require(ok,message):
    if not ok:
        raise ValueError(message)


def verify(primary,independent):
    paths=[directory/"manifest.json" for directory in (primary,independent)]
    raw=[path.read_bytes() for path in paths]
    a,b=map(json.loads,raw)
    require(primary.resolve()!=independent.resolve(),"Rebuild directories must be distinct")
    for key in ("compiler","manifest_key","method_mapping"):
        require(a[key]==b[key],f"Independent build differs in {key}")
    rows=[]
    for first,second in zip(a["modules"],b["modules"],strict=True):
        require(first["manifest"]==second["manifest"],"Independent PTX manifests differ")
        x,y=(row["cpu_cold_build"] for row in (first,second))
        require(x["mode"]==y["mode"]=="miss" and x["artifact"]==y["artifact"],"Expected two empty-cache builds")
        original=(primary/"artifacts"/x["artifact"]).read_bytes()
        rebuild=(independent/"artifacts"/y["artifact"]).read_bytes()
        require(original==rebuild and sha256(original).hexdigest()==x["artifact_sha256"]==y["artifact_sha256"],
                f"Independent artifact bytes differ for {x['artifact']}")
        rows.append(dict(family=first["family"],engine=first["engine"],ntheta_recipe=first["ntheta_recipe"],
                         artifact=x["artifact"],bytes=len(original),artifact_sha256=x["artifact_sha256"],
                         ptx_sha256=first["manifest"]["ptx_sha256"]))
    require(len(rows)==8,"Expected eight independent module pairs")
    return dict(status="passed",scope="Independent empty-directory CPU NVRTC rebuild. No GPU use or solve timing.",
                inputs={f"{path.parent.name}/{path.name}":sha256(data).hexdigest() for path,data in zip(paths,raw)},
                module_manifest_key=a["manifest_key"],all_bytes_equal=True,modules=rows,
                total_bytes=sum(row["bytes"] for row in rows))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("primary",type=Path)
    cli.add_argument("independent",type=Path)
    cli.add_argument("--record",type=Path)
    args=cli.parse_args()
    result=json.dumps(verify(args.primary,args.independent),indent=2,allow_nan=False)+"\n"
    if args.record:
        with args.record.open("x",encoding="utf-8",newline="\n") as handle:
            handle.write(result)
    else:
        print(result,end="")
