"""Independently sum native source rows and check the comparison's counts."""
import argparse
import json
from collections import Counter
from pathlib import Path


def check(path):
    result = json.loads(path.read_text())
    phases, formal, unassigned = Counter(), Counter(), Counter()
    nodes = []
    for source in sorted(Path(result["native_artifacts"]).glob("nodes/*/serving/rank0.json")):
        data = json.loads(source.read_text())
        role = data["role"]
        nodes.append(str(source))
        for row in data["counters"]["rows"]:
            count = row.get("successful_forwards") or 0
            phases[(role, row.get("phase"), row["kind"])] += count
            if row["kind"] not in ("serving", "dummy"):
                continue
            dispatch = row["dispatch_evidence"]["dispatch"]
            key = (role, dispatch["inputs"]["dp_rank"], "dummy" if row["kind"] == "dummy" else "real",
                   dispatch["descriptor"]["cg_mode"])
            if row.get("phase") == result["native_counters"]["serving_phase"]:
                formal[key] += count
            elif row.get("phase") is None:
                unassigned[key] += count
    actual_phases = Counter({(row["role"], row["phase"], row["kind"]): row["count"]
                             for row in result["native_counters"]["phase_counts"]})
    actual_formal, actual_unassigned = Counter(), Counter()
    for dimension, target in (("census", actual_formal), ("unassigned_phase_census", actual_unassigned)):
        for row in result[dimension]["forwards"]:
            role, dp, _, _, real, mode = row["key"]
            target[(role, dp, real, mode)] += row["native"] or 0
    native_window = sum(source["forwards"] for source in result["native_runtime_sources"])
    checks = {"all_nodes_read": len(nodes) == len(result["native_counters"]["sources"]),
              "every_phase_kind_count": actual_phases == phases,
              "formal_per_lane_real_mode_counts": actual_formal == formal,
              "unassigned_per_lane_real_mode_counts": actual_unassigned == unassigned,
              "native_window_counts": sum(row["native_window_forwards"] for row in result["time_by_step_class"]) == native_window,
              "json_and_text_exist": path.is_file() and path.with_suffix(".txt").is_file()}
    return {"checks": {name: "PASS" if value else "FAIL" for name, value in checks.items()},
            "nodes": len(nodes), "all_native_observations": sum(phases.values()),
            "formal_observations": sum(formal.values()), "unassigned_observations": sum(unassigned.values()),
            "excluded_observations": sum(phases.values()) - sum(formal.values()) - sum(unassigned.values()),
            "native_profiler_window_forwards": native_window, "comparison": str(path),
            "notes": "Small eight-request scenario; not an A11 reading of either full A10 replay."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(args.comparison.resolve())
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if all(value == "PASS" for value in result["checks"].values()) else 1)
