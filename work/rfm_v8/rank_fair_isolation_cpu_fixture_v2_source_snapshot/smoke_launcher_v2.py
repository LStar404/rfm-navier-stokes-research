"""Protected T9 entry-equivalence capture only; never a formal timing cohort."""
import argparse
from pathlib import Path

import coordinator_v2 as coordinator
from cpu_fixture_v2 import assert_outer_owner
from transport_v2 import read, require


def execute(protocol, output, outer):
    assert_outer_owner(outer)
    plan = read(protocol)
    require(plan["mode"] == "equivalence_only" and plan["schema"] == coordinator.SCHEMA and plan["status"] == "frozen" and
            all(r["solve_args"]["horizon"] == 9 and max(r["solve_args"]["snapshots"]) == 9 for r in plan["requests"]),
            "Smoke launcher accepts only explicitly registered T9 captures")
    summary = coordinator.run(protocol, output, plan, protocol.read_bytes(), worker=coordinator.SOLVER_WORKER)
    require(not summary["isolated_speed_claim_supported"], "Equivalence capture promoted formal timing")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-evidence", type=Path, required=True)
    args = parser.parse_args()
    execute(args.execute.resolve(), args.output.resolve(), args.outer_evidence.resolve())
