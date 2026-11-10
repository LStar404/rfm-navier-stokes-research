"""Unintegrated Windows-only future candidate; one ReplaceFile call, fail closed, no retries."""
import ctypes as C
import json
import os


def save_existing_observed(path, value, failure_record):
    """Caller must treat *any* failure as unconfirmed publication, including partial API effects."""
    temporary=path.with_name(path.name+".tmp")
    operation="create_and_fsync_temporary"
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(value,ensure_ascii=True,allow_nan=False,separators=(",",":"))+"\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        operation="ReplaceFileW_existing_target"
        kernel=C.WinDLL("kernel32",use_last_error=True)
        replace=kernel.ReplaceFileW
        replace.argtypes=(C.c_wchar_p,C.c_wchar_p,C.c_wchar_p,C.c_uint,C.c_void_p,C.c_void_p)
        replace.restype=C.c_int
        # No unsupported WRITE_THROUGH and no flags that ignore ACL/metadata errors.
        if not replace(str(path),str(temporary),None,0,None,None):
            raise C.WinError(C.get_last_error())
    except Exception as error:
        report=dict(status="publication_unconfirmed",operation=operation,exception_type=type(error).__name__,
                    errno=getattr(error,"errno",None),winerror=getattr(error,"winerror",None),
                    target_name=path.name,temporary_name=temporary.name,retries=0,
                    original_target_preserved_not_assumed=True,temporary_preserved_not_assumed=True,
                    no_next_slot=True)
        with failure_record.open("x",encoding="utf-8") as handle:
            json.dump(report,handle,allow_nan=False,separators=(",",":"));handle.write("\n")
            handle.flush();os.fsync(handle.fileno())
        raise
