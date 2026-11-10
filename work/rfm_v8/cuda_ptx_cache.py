"""Explicit content-addressed NVRTC PTX artifacts. No GPU context is needed to build."""

from contextlib import contextmanager
import ctypes as C
import errno
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import struct
from tempfile import NamedTemporaryFile
from threading import Lock
import time


MAGIC=b"RFMPTX01"
SCHEMA="rfm-nvrtc-ptx-v1"
FLAGS=("--gpu-architecture=compute_52",)
LIBRARIES=("nvrtc64_112_0.dll","nvrtc-builtins64_118.dll")
ENTRY=re.compile(rb"\.entry\s+([A-Za-z_]\w*)\(")
THREAD_LOCKS={}
LOCKS_GUARD=Lock()


class InvalidArtifact(ValueError):
    pass


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("ascii")


def file_sha(path):
    with path.open("rb") as handle:
        return file_digest(handle,"sha256").hexdigest()


def bind(library,name,types):
    function=getattr(library,name)
    function.argtypes=types
    function.restype=C.c_int
    return function


def check(function,*args):
    status=function(*args)
    if status:
        raise RuntimeError(f"{function.__name__} returned {status}")


class Nvrtc:
    def __init__(self,toolkit):
        self.bin=Path(toolkit)/"bin"
        with os.add_dll_directory(str(self.bin)):
            self.lib=C.WinDLL(str(self.bin/LIBRARIES[0]))
        major,minor=C.c_int(),C.c_int()
        check(bind(self.lib,"nvrtcVersion",[C.POINTER(C.c_int),C.POINTER(C.c_int)]),C.byref(major),C.byref(minor))
        self.identity=dict(api_version=[major.value,minor.value],
                           library_sha256={name:file_sha(self.bin/name) for name in LIBRARIES})

    def compile(self,source,spec):
        validate_source(source,spec)
        if spec["compiler"]!=self.identity:
            raise ValueError("The installed NVRTC compiler differs from the artifact contract")
        p,char,size=C.c_void_p,C.c_char_p,C.c_size_t
        create=bind(self.lib,"nvrtcCreateProgram",[C.POINTER(p),char,char,C.c_int,p,p])
        compile_program=bind(self.lib,"nvrtcCompileProgram",[p,C.c_int,C.POINTER(char)])
        get_log_size=bind(self.lib,"nvrtcGetProgramLogSize",[p,C.POINTER(size)])
        get_log=bind(self.lib,"nvrtcGetProgramLog",[p,p])
        get_ptx_size=bind(self.lib,"nvrtcGetPTXSize",[p,C.POINTER(size)])
        get_ptx=bind(self.lib,"nvrtcGetPTX",[p,p])
        destroy=bind(self.lib,"nvrtcDestroyProgram",[C.POINTER(p)])
        options=tuple(flag.encode("ascii") for flag in spec["flags"])
        with os.add_dll_directory(str(self.bin)):
            program=p()
            check(create,C.byref(program),source,spec["filename"].encode("ascii"),0,None,None)
            try:
                status=compile_program(program,len(options),(char*len(options))(*options))
                if status:
                    length=size()
                    check(get_log_size,program,C.byref(length))
                    log=C.create_string_buffer(length.value)
                    check(get_log,program,log)
                    raise RuntimeError(log.value.decode("utf-8",errors="replace"))
                length=size()
                check(get_ptx_size,program,C.byref(length))
                ptx=C.create_string_buffer(length.value)
                check(get_ptx,program,ptx)
                return ptx.value
            finally:
                check(destroy,C.byref(program))


def contract(source,filename,entries,compiler,dependencies,flags=FLAGS):
    # This limited prototype accepts self-contained sources. Any new include
    # must become an explicit hashed NVRTC header input in a new schema.
    if re.search(rb"^\s*#\s*include\b",source,re.MULTILINE):
        raise ValueError("The PTX prototype requires a self-contained translation unit")
    options=tuple(flags)
    targets=[flag.split("=",1)[1] for flag in options if flag.startswith("--gpu-architecture=")]
    if len(targets)!=1 or not targets[0].startswith("compute_"):
        raise ValueError("Register exactly one virtual architecture for a PTX artifact")
    return dict(schema=SCHEMA,format="ptx",source_sha256=sha256(source).hexdigest(),filename=filename,
                entry_points=sorted(entries),target=targets[0],flags=list(options),
                compiler=compiler,source_dependencies_sha256=dict(sorted(dependencies.items())))


def validate_source(source,spec):
    if spec["schema"]!=SCHEMA or spec["format"]!="ptx" or sha256(source).hexdigest()!=spec["source_sha256"]:
        raise ValueError("Translation unit bytes do not match the artifact contract")


def key_for(spec):
    return sha256(canonical(spec)).hexdigest()


def entry_points(ptx):
    return sorted(set(name.decode("ascii") for name in ENTRY.findall(ptx)))


def validate_ptx(ptx,spec):
    if not ptx or b"\0" in ptx or b".version" not in ptx or entry_points(ptx)!=spec["entry_points"]:
        raise InvalidArtifact("PTX format or registered entry points differ")


def pack(spec,ptx):
    validate_ptx(ptx,spec)
    manifest=dict(key=key_for(spec),contract=spec,ptx_sha256=sha256(ptx).hexdigest(),ptx_bytes=len(ptx))
    header=canonical(manifest)
    return MAGIC+struct.pack("<I",len(header))+header+ptx,manifest


def unpack(raw,spec):
    if not raw.startswith(MAGIC) or len(raw)<len(MAGIC)+4:
        raise InvalidArtifact("Artifact header is absent")
    boundary=len(MAGIC)+4
    size,=struct.unpack_from("<I",raw,len(MAGIC))
    if not 0<size<=1024*1024 or boundary+size>=len(raw):
        raise InvalidArtifact("Artifact header length is invalid")
    try:
        manifest=json.loads(raw[boundary:boundary+size])
    except (UnicodeError,ValueError) as error:
        raise InvalidArtifact("Artifact manifest is invalid") from error
    ptx=raw[boundary+size:]
    if not isinstance(manifest,dict) or manifest!={"key":key_for(spec),"contract":spec,
            "ptx_sha256":sha256(ptx).hexdigest(),"ptx_bytes":len(ptx)}:
        raise InvalidArtifact("Artifact manifest or PTX digest differs")
    validate_ptx(ptx,spec)
    return ptx,manifest


@contextmanager
def exclusive(path,timeout):
    import msvcrt

    with LOCKS_GUARD:
        thread_lock=THREAD_LOCKS.setdefault(path.resolve(),Lock())
    end=time.monotonic()+timeout
    if not thread_lock.acquire(timeout=timeout):
        raise TimeoutError("PTX cache thread lock expired")
    try:
        with path.open("a+b") as handle:
            fd=handle.fileno()
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(fd,msvcrt.LK_NBLCK,1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK):
                        raise
                    if time.monotonic()>=end:
                        raise TimeoutError("PTX cache process lock expired") from error
                    time.sleep(.02)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(fd,msvcrt.LK_UNLCK,1)
    finally:
        thread_lock.release()


def atomic_write(path,payload):
    tmp=None
    try:
        with NamedTemporaryFile(mode="wb",dir=path.parent,prefix=path.stem+".",suffix=".tmp",delete=False) as handle:
            tmp=Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp,path)
        tmp=None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def artifact(root,source,spec,*,compiler=None,strict_hit=False,lock_timeout=30):
    """Return PTX, its verified manifest and observed cache/build timings."""
    start=time.perf_counter()
    validate_source(source,spec)
    root=Path(root)
    if not strict_hit:
        root.mkdir(parents=True,exist_ok=True)
    key=key_for(spec)
    path=root/(key+".rfmptx")
    with exclusive(root/(key+".lock"),lock_timeout):
        try:
            raw=path.read_bytes()
            ptx,manifest=unpack(raw,spec)
            mode,compile_seconds="hit",0.
        except (FileNotFoundError,InvalidArtifact) as error:
            if strict_hit or compiler is None:
                raise
            mode="miss" if isinstance(error,FileNotFoundError) else "repaired_invalid"
            started=time.perf_counter()
            ptx=compiler.compile(source,spec)
            compile_seconds=time.perf_counter()-started
            raw,manifest=pack(spec,ptx)
            atomic_write(path,raw)
            ptx,manifest=unpack(path.read_bytes(),spec)
        return ptx,manifest,dict(mode=mode,cache_seconds=time.perf_counter()-start,nvrtc_seconds=compile_seconds,
                                 artifact=path.name,artifact_sha256=sha256(raw).hexdigest())


def load_device(driver,ptx):
    """Load verified PTX into an owned CUDA context and measure the driver phase."""
    start=time.perf_counter()
    module=C.c_void_p()
    check(driver.cuModuleLoadData,C.byref(module),C.create_string_buffer(ptx))
    return module,time.perf_counter()-start
