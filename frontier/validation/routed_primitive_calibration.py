"""Validate exact routed primitive profiles and emit an exact cost table."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from frontier.profiling.runtime.sglang_moe import MOE_ROUTED_PRIMITIVES
from frontier.profiling.runtime.sglang_moe_routes import read_routed_queries
from frontier.runtime_cost.routed_primitives import ExactRoutedCalibration
from frontier.runtime_cost.sglang import ExactRuntimeCostTable, RuntimeIdentity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--query-plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--primitives", choices=MOE_ROUTED_PRIMITIVES, nargs="+",
                        default=list(MOE_ROUTED_PRIMITIVES))
    args = parser.parse_args()
    payloads = [json.loads(path.read_text())
                for path in sorted(args.profile_dir.glob("rank*.json"))]
    if not payloads:
        raise ValueError("No exact routed rank profiles")
    identity = RuntimeIdentity(**payloads[0]["identity"])
    calibration = ExactRoutedCalibration(payloads, identity=identity)
    report = calibration.report()
    if not report["all_queries_passed"]:
        raise ValueError("One or more exact routed queries failed quality admission")
    queries = read_routed_queries(args.query_plan, identity, tuple(args.primitives))
    rows = [{"query": asdict(query), "estimate": asdict(calibration(query))}
            for query in queries]
    table = {"schema_version": 1, "identity": asdict(identity), "rows": rows}
    ExactRuntimeCostTable(table, identity=identity)
    for path, payload in ((args.report, report), (args.table, table)):
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
