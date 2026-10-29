"""A10 operator-path client reading against the persisted native reference.

Usage: a10_gate.py <case> <frontier run dir> <output dir>   # p2_32 or p2_64
"""

import json
import subprocess
import sys
from pathlib import Path

from a10_common import REPO, TASK, load_module

h4a4 = load_module("h4a4_gate", TASK / "h4a_p0_diagnostic/h4a4_gate.py")
CASES = {name: h4a4.CASES[name] for name in ("p2_32", "p2_64")}
GATE_TIMEOUT_S = 300
ROUTING_REASON = (
    "R0 and A13 (b), H3o design §3.7: pinned native balanced_topk and "
    "determine_expert_map give the same per-expert and per-EP-rank counts "
    "as Frontier at DP2 / EP16 and DP4 / EP32 on synchronized SP-local rows."
)


def matrix(case, cell_id, out, receipt):
    # h4a4.matrix is diagnostic-specific. Keep its cell once here, changing
    # only the reading label and the routing admission.
    entry = {"architecture": "pd-disaggregation",
             "routing": {"status": "MATCH", "reason": ROUTING_REASON}}
    cell = {
        "cell_id": cell_id, "case_id": case["case_id"], "status": "claimed", "gate": "client",
        "label": "A10 operator-path client reading",
        "trace": {"request_ids": h4a4.record(h4a4.TRACE / "request_ids.json"),
                  "workload": h4a4.record(h4a4.TRACE / "manifest.json")},
        "run_check": h4a4.record(receipt),
    }
    path = out / "matrix.json"
    path.write_text(json.dumps({"cases": {case["case_id"]: entry}, "cells": [cell]}, indent=1) + "\n")
    return path


def print_metrics(reading):
    for row in reading["metrics"]:
        native, frontier = row["vllm"], row["frontier"]
        absolute = abs(frontier - native) if native is not None and frontier is not None else None
        print(f"{row['metric']}: native={native} frontier={frontier} "
              f"absolute_error={absolute} relative_error={row['relative_error']}")


def main(name, frontier_run, out):
    case = CASES[name]
    cell_id = f"{case['case_id']}/a10_operator_path"
    out.mkdir(parents=True, exist_ok=False)
    run, receipt = h4a4.native_layout(name, case, out)
    matrix_path = matrix(case, cell_id, out, receipt)
    gate = [sys.executable, str(REPO / "tests/comparison/e2e_gate/run_gate.py")]
    common = ["--matrix", str(matrix_path), "--cell", cell_id]
    subprocess.run([*gate, "normalize", *common, "--artifacts-root", str(out), "--vllm-run-dir", str(run),
                    "--frontier-run", f"a10={frontier_run}", "--output-dir", str(out / "rows")],
                   cwd=REPO, check=True, timeout=GATE_TIMEOUT_S)
    subprocess.run([*gate, "gate", *common, "--rows", str(out / "rows"), "--output", str(out / "reading.json")],
                   cwd=REPO, check=True, timeout=GATE_TIMEOUT_S)
    reading = json.loads((out / "reading.json").read_text())
    print_metrics(reading)
    return reading


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in CASES:
        raise SystemExit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
