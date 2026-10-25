"""H4a-4: client-gate reading of one Frontier run against the native06 R4 reference.

The native side is rebuilt from persisted task copies (D-INPUT-PERSISTENCE), never from
/data/ycfeng/tmp: the run files from inputs/native06 and the R4 run-check result from
native06_acceptance_20261023/frozen_r4_result.json, whose digest primary_acceptance_r4.json pins.
The layout puts that result where run_gate.py binds a run check to its run directory. The
task-local matrix holds one client cell over the frozen 68f891b workload.

Usage: h4a4_gate.py <frontier run dir> <output dir>
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
REPO = TASK.parents[1]
TAG = "dsv4-h800-p0-canonical-20261021-06"
CASE_ID = "dsv4-h800-p0-canonical-fixed128-qps1"
CELL_ID = f"{CASE_ID}/h4a_step_cost"
TRACE = REPO / "data/processed_traces/canonical_fixed_6144_1024_req128_qps1"
ACCEPTANCE = TASK / "native06_acceptance_20261023"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    return {"path": str(path), "sha256": sha256(path)}


def native_layout(out):
    """<case>/analysis/run_check_status.groundtruth_clean.json and <case>/runs/groundtruth_clean/<TAG>/."""
    acceptance = json.loads((ACCEPTANCE / "primary_acceptance_r4.json").read_text())
    result = ACCEPTANCE / "frozen_r4_result.json"
    if acceptance["status"] != "ACCEPTED_NATIVE_REFERENCE_R4" or sha256(result) != acceptance["result_sha256"]:
        raise SystemExit("native06 R4 acceptance does not pin frozen_r4_result.json")
    case = out / "native06_r4_case"
    receipt = case / "analysis/run_check_status.groundtruth_clean.json"
    receipt.parent.mkdir(parents=True)
    shutil.copy2(result, receipt)
    run = case / "runs/groundtruth_clean" / TAG
    manifest = json.loads((HERE / "inputs/manifest.json").read_text())["native06"]["files"]
    for entry in manifest:
        source = HERE / "inputs/native06" / entry["path"]
        if sha256(source) != entry["sha256"]:
            raise SystemExit(f"persisted copy differs from its manifest: {source}")
        target = run / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)  # keeps mtimes: run files must be newer than run_manifest.json
    return run, receipt


def matrix(out, receipt):
    case = {
        "architecture": "pd-disaggregation",
        "routing": {
            "status": "NOT_APPLICABLE",
            "reason": "Frontier prices whole native lane steps from dsv4_step_cost.json, measured on "
                      "profile02 with the same balanced_topk routing as native06; Frontier's routing "
                      "model does not enter this prediction (MoE and EP costs are zero).",
        },
    }
    cell = {
        "cell_id": CELL_ID, "case_id": CASE_ID, "status": "claimed", "gate": "client",
        "label": "DIAGNOSTIC: H4a, counts toward neither the 30% promotion gate nor the 10% final gate",
        "trace": {"request_ids": record(TRACE / "request_ids.json"), "workload": record(TRACE / "manifest.json")},
        "run_check": record(receipt),
    }
    path = out / "matrix.json"
    path.write_text(json.dumps({"cases": {CASE_ID: case}, "cells": [cell]}, indent=1) + "\n")
    return path


def main():
    frontier_run, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    run, receipt = native_layout(out)
    matrix_path = matrix(out, receipt)
    gate = [sys.executable, str(REPO / "tests/comparison/e2e_gate/run_gate.py")]
    common = ["--matrix", str(matrix_path), "--cell", CELL_ID]
    subprocess.run([*gate, "normalize", *common, "--artifacts-root", str(out), "--vllm-run-dir", str(run),
                    "--frontier-run", f"h4a4={frontier_run}", "--output-dir", str(out / "rows")],
                   cwd=REPO, check=True)
    subprocess.run([*gate, "gate", *common, "--rows", str(out / "rows"), "--output", str(out / "reading.json")],
                   cwd=REPO, check=True)


if __name__ == "__main__":
    main()
