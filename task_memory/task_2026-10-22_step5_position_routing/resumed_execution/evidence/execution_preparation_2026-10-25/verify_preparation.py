"""Check prepared bindings and inject output corruption without pricing a wave."""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from joint_timing import check_bindings, expected_output, read_references, require, validate_output


DIRECTORY = Path(__file__).resolve().parent
SCRATCH = Path(sys.argv[1])
SCRATCH.mkdir(parents=True, exist_ok=False)
bindings = json.loads((DIRECTORY/"joint_bindings.json").read_bytes())
plan = json.loads((DIRECTORY/"worker_plan.json").read_bytes())
source = check_bindings(bindings, DIRECTORY)
for filename, expected in plan["scripts"].items():
    require(hashlib.sha256((DIRECTORY/filename).read_bytes()).hexdigest() == expected, filename)
require(hashlib.sha256((DIRECTORY/"joint_bindings.json").read_bytes()).hexdigest() ==
        plan["joint_bindings_sha256"], "Worker joint binding changed")
syntax = []
for filename in (*bindings["scripts"], *plan["scripts"]):
    if filename.endswith(".py"):
        compile((DIRECTORY/filename).read_bytes(), str(DIRECTORY/filename), "exec")
        syntax.append(filename)
setup_tree = ast.parse((DIRECTORY/"joint_setup.py").read_bytes())
require(not any(isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
        {"store", "phases", "timing", "plan"} for t in node.targets) for node in setup_tree.body),
        "Old emission, pricing or reporting setup entered the prefix")
require(bindings["proposal"]["labels"] == ["cold", "warm"] and
        bindings["proposal"]["production_pricing_calls"] == 2, "Proposal is not two calls")
require(plan["resource"]["gpu"] == 0 and plan["lifecycle"]["job_limit"] == 1 and
        plan["workload_in_this_job"]["pricing_calls"] == 0 and
        plan["workload_in_this_job"]["full_cli"] is False, "Readiness resource or workload changed")
_, ledger, operators = read_references(bindings)
identities = [{"ep_id": row["ep_id"], "batch_id": row["batch_id"] + 16,
               "request_ids": [str(int(value)+352) for value in row["request_ids"]],
               "num_tokens": [0]*len(row["request_ids"]),
               "source_batch_ids": row["source_batch_ids"]} for row in ledger]
expected_ledger, expected_operators = expected_output(identities, ledger, operators)
simulation_info = {"evidence": "D29_CURRENT_JOINT_TIMING", "label": "cold",
                   "source_commit": bindings["source_commit"]}
header = {"meta": {"timestamp": 0.0, "simulation_info": simulation_info, "version": "1.0"}}


def write_outputs(rows, events, trace_header):
    (SCRATCH/"frontier_ep_wave_lane_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True)+"\n" for row in rows))
    (SCRATCH/"op_traces.jsonl").write_text(json.dumps(trace_header)+"\n"+
        "".join(json.dumps(row)+"\n" for row in events))


write_outputs(expected_ledger, expected_operators, header)
positive = validate_output(SCRATCH, identities, ledger, operators, simulation_info)
controls = []
for name in ("missing_ledger_row", "ledger_order", "compute_proof", "operator_duration",
             "extra_operator", "stale_cold_batch_ids", "captured_source_identity",
             "trace_evidence", "trace_label", "trace_source_commit"):
    rows, events = copy.deepcopy(expected_ledger), copy.deepcopy(expected_operators)
    trace_header = copy.deepcopy(header)
    if name == "missing_ledger_row":
        rows.pop()
    elif name == "ledger_order":
        rows[0], rows[1] = rows[1], rows[0]
    elif name == "compute_proof":
        proofs = rows[0]["phases"]["routed_compute"]["compute_proofs"]
        proofs[next(iter(proofs))]["duration_ms"] += 1.0
    elif name == "operator_duration":
        events[0]["duration_ms"] += 1.0
    elif name == "extra_operator":
        events.append(copy.deepcopy(events[-1]))
    elif name == "stale_cold_batch_ids":
        rows[0]["batch_id"] -= 16
    elif name == "captured_source_identity":
        rows[0]["source_batch_ids"] = [7053]
    elif name.startswith("trace_"):
        trace_header["meta"]["simulation_info"][name.removeprefix("trace_")] = "wrong_identity"
    write_outputs(rows, events, trace_header)
    try:
        validate_output(SCRATCH, identities, ledger, operators, simulation_info)
    except AssertionError as error:
        controls.append({"control": name, "status": "PASS_EXPECTED_REJECTION", "error": str(error)})
    else:
        raise AssertionError(f"Corrupt output accepted: {name}")
write_outputs(expected_ledger, expected_operators, header)
validate_output(SCRATCH, identities, ledger, operators, simulation_info)
dry = subprocess.run([sys.executable, "-B", str(DIRECTORY/"launch_worker.py"),
        "--plan", str(DIRECTORY/"worker_plan.json")], capture_output=True, text=True, check=True)
(SCRATCH/"worker_dry_run.stdout").write_text(dry.stdout)
require(json.loads(dry.stdout)["status"] == "DRY_RUN_NO_PLATFORM_OPERATIONS", "Unexpected dry run")
require(check_bindings(bindings, DIRECTORY) == source, "Binding changed during static checks")
report = {"status": "PASS_PREPARATION_AND_VALIDATOR_CONTROLS_ONLY",
    "collected_utc": datetime.now(timezone.utc).isoformat(), "source": source,
    "python": sys.version, "executable": sys.executable, "cwd": str(Path.cwd()),
    "syntax_files": syntax, "wave_AST": "UNCHANGED", "old_tail_excluded": True,
    "positive": positive, "negative_controls": controls, "worker_dry_run": "PASS",
    "new_pricing_calls": 0, "new_shape_calls": 0, "new_platform_operations": 0,
    "limitations": ["Only retained-reference output controls were executed.",
                    "Simulator, cached-model setup and worker checks were not executed."]}
(SCRATCH/"preparation.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({"status": report["status"], "negative_controls": len(controls),
                  "new_pricing_calls": 0, "new_platform_operations": 0}))
