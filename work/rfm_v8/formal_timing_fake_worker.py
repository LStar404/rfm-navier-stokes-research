"""Exercise the actual formal worker transport without numerical or GPU imports."""

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import solver_worker_formal as formal


def select(request):
    def fake_solve(seconds, fail=False):
        time.sleep(seconds)
        if fail:
            raise ArithmeticError("Synthetic solver failure")
        return SimpleNamespace(model=object())
    return fake_solve, nullcontext()


def save(stream, solution, config):
    stream.write(b"V8_FORMAL_SYNTHETIC_TRANSPORT_ONLY\n")


if __name__ == "__main__":
    formal.select_solver = select
    formal.model_signature = lambda model,family: sha256(b"fake-model").hexdigest()
    sys.modules["full_case"] = SimpleNamespace(store_solution=save)
    raise SystemExit(formal.worker(Path(sys.argv[1]).resolve(), sys.argv[2]))
