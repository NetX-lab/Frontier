"""Summarize P1h graph-bridge traces per mode, case and scope (diagnostic only).

Reruns the saved attribution of each mode's replay traces and reports, per
(case, scope), the kernel count and the kernel-only time per measured step,
summed over layers, plus the share of each scope's kernels that ran as graph
replay. Usage: scope_summary.py BRIDGE_ROOT OUTPUT_JSON.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from frontier.profiling.attention.dsa_kernel_trace import load_trace_events
from frontier.profiling.attention.dsv4_graph_replay_runner import CONTAINER_SCOPES
from frontier.profiling.attention.graph_capture import CaptureRecord
from frontier.profiling.attention.graph_replay_trace import attribute_replay_trace

MODES = ("NONE", "PIECEWISE", "FULL_AND_PIECEWISE")


def summarize(root: Path) -> dict:
    result = {}
    for mode in MODES:
        capture = json.loads((root / mode / "raw" / "dsv4_capture_e0.json").read_text(encoding="utf-8"))
        record = CaptureRecord.from_json(capture["record"])
        step_modes = {}
        for line in (root / mode / "raw" / "dsv4_raw_e0.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["kind"] == "step":
                step_modes[row["step_index"]] = (row["case_id"], row["cudagraph_dispatch"]["cg_mode"])
        cases = {}
        for path in sorted((root / mode / "traces").glob("*.json")):
            case_id = path.stem.split("-", 1)[1]
            attributed = attribute_replay_trace(load_trace_events(path), record, CONTAINER_SCOPES)
            measured = sorted(attributed.outside_kernel_counts)
            count = defaultdict(int)
            time_ms = defaultdict(float)
            graph = defaultdict(int)
            for interval in attributed.intervals:
                if interval.activity != "kernel":
                    continue
                scope = "<outside>" if interval.owner is None else f"L{interval.owner[0]}:{interval.owner[1]}"
                count[scope] += 1
                time_ms[scope] += (interval.end_us - interval.start_us) / 1000.0
                graph[scope] += interval.kind == "graph"
            steps = len(measured)
            cases[case_id] = {
                "measured_steps": steps,
                "step_modes": sorted({step_modes[step] for step in measured}),
                "scopes": {
                    scope: {
                        "kernels_per_step": count[scope] / steps,
                        "kernel_ms_per_step": time_ms[scope] / steps,
                        "graph_share": graph[scope] / count[scope],
                    }
                    for scope in sorted(count)
                },
            }
        result[mode] = cases
    return result


def main() -> int:
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.write_text(json.dumps(summarize(root), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
