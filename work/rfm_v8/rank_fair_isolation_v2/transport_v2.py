"""Version-2 transport: distinguish inherited parent CPU from effective solver CPU."""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from rank_fair_owned_transport import *


def require_identity(value, event, request, request_sha, identity):
    require(value["event"] == event and value["index"] == request["index"] and value["run_token"] == request["run_token"] and
            value["request_sha256"] == request_sha and key(value) == key(identity) and
            value["inherited_affinity_mask"] == request["inherited_affinity_mask"] == 256 and
            value["effective_affinity_mask"] == request["effective_affinity_mask"] == request["affinity_mask"] == 16 and
            len(value["affinity_bootstrap_sha256"]) == 64,
            "V2 latch does not authenticate the exact worker and distinct inherited/effective CPUs")
