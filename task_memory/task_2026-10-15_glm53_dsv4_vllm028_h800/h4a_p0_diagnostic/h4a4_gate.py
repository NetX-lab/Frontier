"""H4a-4: client-gate reading of one Frontier run against its R4-accepted native reference.

The native side is rebuilt from persisted task copies (D-INPUT-PERSISTENCE), never from
/data/ycfeng/tmp: the run files from the case's input set, checked against that set's
manifest.json, and the R4 run-check result pinned in CASES. For native06 that result is
native06_acceptance_20261023/frozen_r4_result.json, whose digest primary_acceptance_r4.json pins.
The layout puts the result where run_gate.py binds a run check to its run directory. The
task-local matrix holds one client cell over the frozen 68f891b workload.

Usage: h4a4_gate.py <case> <frontier run dir> <output dir>     # case: native06, p2_32 or p2_64
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
TRACE = REPO / "data/processed_traces/canonical_fixed_6144_1024_req128_qps1"
ACCEPTANCE = TASK / "native06_acceptance_20261023"
H200_INPUTS = TASK / "h200_mainline/diagnostic/inputs"


def routing_reason(profile, reference):
    return (f"Frontier prices whole native lane steps from dsv4_step_cost.json, measured on "
            f"{profile} with the same balanced_topk routing as {reference}; Frontier's routing "
            f"model does not enter this prediction (MoE and EP costs are zero).")


CASES = {
    "native06": dict(
        tag="dsv4-h800-p0-canonical-20261021-06", case_id="dsv4-h800-p0-canonical-fixed128-qps1",
        inputs=HERE / "inputs", input_name="native06",
        r4=ACCEPTANCE / "frozen_r4_result.json",
        r4_sha256="71bc110b8162af3672460d59a4ff95e9e8b224ad45605eb43753b983b83c5faa",
        routing=routing_reason("profile02", "native06")),
    "p2_32": dict(
        tag="dsv4-h200-p2-canonical-20261025-01", case_id="dsv4-h200-p2-canonical-fixed128-qps1",
        inputs=H200_INPUTS, input_name="p2_32r",
        r4=H200_INPUTS / "p2_32r/run_check_status.groundtruth_clean.json",
        r4_sha256="8cc662e69a6f10c093e703e9280b6336fe910374e44953b48f2f8d1a119204a1",
        routing=routing_reason("P2-32p", "P2-32r")),
    "p2_64": dict(
        tag="dsv4-h200-p2-64-canonical-20261025-01", case_id="dsv4-h200-p2-64-canonical-fixed128-qps1",
        inputs=H200_INPUTS, input_name="p2_64r",
        r4=H200_INPUTS / "p2_64r/run_check_status.groundtruth_clean.json",
        r4_sha256="e1da1a98b36aa051b062ac902217238069ab083162d6440122c913e848ce3b00",
        routing=routing_reason("P2-64p", "P2-64r")),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    return {"path": str(path), "sha256": sha256(path)}


def native_layout(name, case, out):
    """<case>/analysis/run_check_status.groundtruth_clean.json and <case>/runs/groundtruth_clean/<tag>/."""
    if name == "native06":
        acceptance = json.loads((ACCEPTANCE / "primary_acceptance_r4.json").read_text())
        if acceptance["status"] != "ACCEPTED_NATIVE_REFERENCE_R4" or acceptance["result_sha256"] != case["r4_sha256"]:
            raise SystemExit("native06 R4 acceptance does not pin frozen_r4_result.json")
    if sha256(case["r4"]) != case["r4_sha256"] or \
            json.loads(case["r4"].read_text())["runs"][case["tag"]]["status"] != "PASS":
        raise SystemExit(f"{case['r4']} is not the pinned R4 PASS result of {case['tag']}")
    root = out / f"{name}_r4_case"
    receipt = root / "analysis/run_check_status.groundtruth_clean.json"
    receipt.parent.mkdir(parents=True)
    shutil.copy2(case["r4"], receipt)
    run = root / "runs/groundtruth_clean" / case["tag"]
    manifest = json.loads((case["inputs"] / "manifest.json").read_text())[case["input_name"]]["files"]
    for entry in manifest:
        source = case["inputs"] / case["input_name"] / entry["path"]
        if sha256(source) != entry["sha256"]:
            raise SystemExit(f"persisted copy differs from its manifest: {source}")
        target = run / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)  # keeps mtimes: run files must be newer than run_manifest.json
    return run, receipt


def matrix(case, cell_id, out, receipt):
    entry = {
        "architecture": "pd-disaggregation",
        "routing": {"status": "NOT_APPLICABLE", "reason": case["routing"]},
    }
    cell = {
        "cell_id": cell_id, "case_id": case["case_id"], "status": "claimed", "gate": "client",
        "label": "DIAGNOSTIC: H4a, counts toward neither the 30% promotion gate nor the 10% final gate",
        "trace": {"request_ids": record(TRACE / "request_ids.json"), "workload": record(TRACE / "manifest.json")},
        "run_check": record(receipt),
    }
    path = out / "matrix.json"
    path.write_text(json.dumps({"cases": {case["case_id"]: entry}, "cells": [cell]}, indent=1) + "\n")
    return path


def main(name, frontier_run, out):
    case = CASES[name]
    cell_id = f"{case['case_id']}/h4a_step_cost"
    out.mkdir(parents=True, exist_ok=False)
    run, receipt = native_layout(name, case, out)
    matrix_path = matrix(case, cell_id, out, receipt)
    gate = [sys.executable, str(REPO / "tests/comparison/e2e_gate/run_gate.py")]
    common = ["--matrix", str(matrix_path), "--cell", cell_id]
    subprocess.run([*gate, "normalize", *common, "--artifacts-root", str(out), "--vllm-run-dir", str(run),
                    "--frontier-run", f"h4a4={frontier_run}", "--output-dir", str(out / "rows")],
                   cwd=REPO, check=True)
    subprocess.run([*gate, "gate", *common, "--rows", str(out / "rows"), "--output", str(out / "reading.json")],
                   cwd=REPO, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in CASES:
        raise SystemExit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
