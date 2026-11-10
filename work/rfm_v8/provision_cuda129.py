"""Provision three pinned NVIDIA Windows runtime wheels inside the workspace."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path, PurePosixPath
from struct import Struct
from tempfile import TemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
DESTINATION = HERE/"vendor"/"cuda_12_9_1"
PLAN = HERE/"cuda129_vendor_plan.json"
LIMIT = 1_000_000_000
COMPONENTS = {"nvidia-cuda-nvrtc-cu12":"12.9.86","nvidia-cufft-cu12":"11.4.1.4","nvidia-nvjitlink-cu12":"12.9.86"}
DOCUMENTATION = (
    "https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html",
    "https://docs.nvidia.com/cuda/archive/12.9.1/nvrtc/index.html",
    "https://docs.nvidia.com/cuda/archive/12.9.1/eula/index.html",
    "https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-microsoft-windows/index.html",
)
EOCD = Struct("<4s4H2LH")
ENTRY = Struct("<4s6H3I5H2I")


def check(condition,message):
    if not condition:
        raise ValueError(message)


def record(path,data):
    with path.open("x",encoding="utf-8") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write("\n")


def range_get(url,start,stop,total):
    request = Request(url,headers={"Range":f"bytes={start}-{stop}"})
    with urlopen(request,timeout=35) as response:
        check(response.status==206 and response.headers.get("Content-Range")==f"bytes {start}-{stop}/{total}",
              "Wheel host did not honor the bounded central-directory request")
        value = response.read(stop-start+2)
    check(len(value)==stop-start+1,"Truncated wheel central-directory range")
    return value


def wheel_members(url,total):
    tail_start = max(0,total-65557)
    tail = range_get(url,tail_start,total-1,total)
    eocd = tail.rindex(b"PK\x05\x06")
    signature,disk,cd_disk,local,entries,size,offset,comment = EOCD.unpack_from(tail,eocd)
    check(signature==b"PK\x05\x06" and disk==cd_disk==0 and local==entries and
          offset+size==tail_start+eocd and eocd+EOCD.size+comment==len(tail),
          "Unsupported or inconsistent wheel ZIP directory")
    block = tail[offset-tail_start:offset-tail_start+size] if offset>=tail_start else range_get(url,offset,offset+size-1,total)
    at = 0
    members = []
    for _ in range(entries):
        fields = ENTRY.unpack_from(block,at)
        check(fields[0]==b"PK\x01\x02","Invalid wheel central-directory member")
        length,extra,comment = fields[10:13]
        name = block[at+ENTRY.size:at+ENTRY.size+length].decode("utf-8")
        members.append(dict(path=name,bytes=fields[9]))
        at += ENTRY.size+length+extra+comment
    check(at==len(block),"Unexpected trailing wheel directory bytes")
    return members


def selected(name):
    path = PurePosixPath(name)
    check(not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name,
          "Unsafe wheel member path")
    return (path.suffix.lower()==".dll" or
            any(part.endswith(".dist-info") for part in path.parts) and not name.endswith("/"))


def make_plan():
    entries = []
    for package,version in COMPONENTS.items():
        metadata_url = f"https://pypi.org/pypi/{package}/{version}/json"
        with urlopen(metadata_url,timeout=35) as response:
            metadata = json.load(response)
        candidates = [row for row in metadata["urls"] if row["filename"].endswith("win_amd64.whl")]
        check(len(candidates)==1,"Missing exact Windows wheel")
        wheel = candidates[0]
        check(wheel["url"].startswith("https://files.pythonhosted.org/packages/"),"Unexpected official wheel host")
        members = wheel_members(wheel["url"],wheel["size"])
        files = [row for row in members if selected(row["path"])]
        check(any(row["path"].lower().endswith(".dll") for row in files) and
              any("license" in row["path"].lower() for row in files),"Missing runtime DLL or NVIDIA license")
        entries.append(dict(package=package,version=version,metadata_url=metadata_url,url=wheel["url"],
            filename=wheel["filename"],sha256=wheel["digests"]["sha256"],download_bytes=wheel["size"],
            unpacked_bytes=sum(r["bytes"] for r in members),installed_bytes=sum(r["bytes"] for r in files),
            license=metadata["info"]["license"],requires_dist=metadata["info"]["requires_dist"],installed_members=files))
    entries.sort(key=lambda r:r["download_bytes"],reverse=True)
    resident,peak = 0,0
    for item in entries:
        resident += item["installed_bytes"]
        peak = max(peak,resident+item["download_bytes"])
    total = sum(r["download_bytes"] for r in entries)
    return dict(status="within_limit" if max(total,peak)<=LIMIT else "manifest_only_size_limit",
        limit_bytes=LIMIT,download_bytes=total,workspace_installed_bytes=resident,
        largest_workspace_footprint_bytes=peak,official_documents=DOCUMENTATION,packages=entries,
        scope="Workspace-local native runtime DLLs and wheel license/metadata. Scratch wheel removed after each verified extraction.")


def install(plan):
    check(plan["status"]=="within_limit" and not DESTINATION.exists(),"Preserve existing vendor runtime or size cap")
    DESTINATION.mkdir(parents=True)
    installed = []
    for package in plan["packages"]:
        with TemporaryFile(dir=DESTINATION) as scratch:
            with urlopen(package["url"],timeout=45) as response:
                count = 0
                while data := response.read(1<<20):
                    count += len(data)
                    check(count<=package["download_bytes"],"Wheel exceeded registered compressed size")
                    scratch.write(data)
            check(count==package["download_bytes"],"Incomplete wheel download")
            scratch.seek(0)
            check(file_digest(scratch,"sha256").hexdigest()==package["sha256"],"Official wheel SHA-256 mismatch")
            scratch.seek(0)
            with ZipFile(scratch) as wheel:
                for item in package["installed_members"]:
                    data = wheel.read(item["path"])
                    check(len(data)==item["bytes"],"Extracted wheel member differs from planned size")
                    target = DESTINATION.joinpath(*PurePosixPath(item["path"]).parts)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open("xb") as output:
                        output.write(data)
                    installed.append(dict(item,sha256=sha256(data).hexdigest(),package=package["package"]))
        print("INSTALLED",package["filename"],flush=True)
    record(DESTINATION/"manifest.json",dict(status="installed",plan_sha256=sha256(PLAN.read_bytes()).hexdigest(),
           source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),installed_members=installed,**{
           k:plan[k] for k in ("official_documents","packages","download_bytes","workspace_installed_bytes",
                               "largest_workspace_footprint_bytes")}))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--install",action="store_true")
    args = cli.parse_args()
    if args.install:
        install(json.loads(PLAN.read_text(encoding="utf-8")))
    else:
        plan = make_plan()
        record(PLAN,plan)
        print(plan["status"],plan["download_bytes"],plan["workspace_installed_bytes"],
              plan["largest_workspace_footprint_bytes"],flush=True)
