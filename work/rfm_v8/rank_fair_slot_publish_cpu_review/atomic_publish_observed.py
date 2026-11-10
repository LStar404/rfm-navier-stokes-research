"""Future CPU-only proposal: preserve original fail-closed publish semantics, record failing stage."""
import json
import os


def save_observed(path, value, *, replace, failure_record):
    temporary = path.with_name(path.name+".tmp")
    stage = "open_exclusive_temporary"
    try:
        with temporary.open("xb") as handle:
            stage = "write_temporary"
            handle.write((json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))+"\n").encode("ascii"))
            stage = "flush_and_fsync_temporary"
            handle.flush()
            os.fsync(handle.fileno())
        stage = "replace_target" if replace else "rename_new_target"
        if replace:
            temporary.replace(path)
        else:
            temporary.rename(path)
    except Exception as error:
        # No retry, no deletion, no amended scientific evidence, no success promotion.
        record = dict(status="failed", operation=stage, exception_type=type(error).__name__,
                      errno=getattr(error, "errno", None), winerror=getattr(error, "winerror", None),
                      filename=getattr(error, "filename", None), filename2=getattr(error, "filename2", None),
                      target_name=path.name, temporary_name=temporary.name, retries=0)
        with failure_record.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, allow_nan=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        raise
