#!/usr/bin/env python3
"""Run one frozen native-stack observation without changing the trace audit.

Usage: /usr/bin/python3 -u -B v4_trace_audit_native_stack.py EXECUTION_ALLOCATION_JSON
The fresh control and exact source/input bindings precede this invocation.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


allocation = Path(sys.argv[1])
plan = json.loads(allocation.read_text())
assert plan["status"] == "FROZEN_READY_ONCE_ONLY_NATIVE_DIAGNOSTIC"
assert sha(__file__) == plan["execution_driver_sha256"]
for pin in plan["source_pins"] + plan["runtime_pins"]:
    assert sha(pin["path"]) == pin["sha256"], pin["path"]
for name in ("source_review", "accepted_tiny", "new_control", "allocation", "download_inventory"):
    assert sha(plan[name + "_path"]) == plan[name + "_sha256"], name
assert sha(plan["gdb_source_path"]) == plan["gdb_source_sha256"]
for name, value in plan["selected_environment"].items():
    assert os.environ.get(name) == value, name
for kind in ("trace", "profile", "inventory"):
    path = Path(plan["input_root"]) / plan["input_bindings"][kind + "_path"]
    assert path.stat().st_size == plan["input_bindings"][kind + "_bytes"]
    assert sha(path) == plan["input_bindings"][kind + "_sha256"]
prefix = Path(plan["prefix"])
command = plan["command_prefix"] + ["--command=" + plan["gdb_source_path"], "--args",
    "/usr/bin/python3", "-u", "-B", plan["timing_source_path"],
    plan["input_bindings"]["trace_path"], str(prefix)]
assert command == plan["command"], "Frozen command mismatch"
receipt_path = prefix.with_suffix(".receipt.json")
receipt = dict(status="STARTING", decision=plan["decision"], attempt="1/1", command=command,
    cwd=plan["workspace"], controller_python=sys.version, allocation_sha256=sha(allocation),
    driver_sha256=sha(__file__), input_bindings=plan["input_bindings"],
    source_pins=plan["source_pins"], runtime_pins=plan["runtime_pins"],
    selected_environment=plan["selected_environment"],
    start_utc=datetime.now(timezone.utc).isoformat())
with receipt_path.open("x") as stream:
    json.dump(receipt, stream, indent=2)
started = time.monotonic()
with prefix.with_suffix(".stdout").open("x") as out, prefix.with_suffix(".stderr").open("x") as err:
    result = subprocess.run(command, cwd=plan["workspace"], stdout=out, stderr=err)
enclosing_seconds = time.monotonic() - started
end_utc = datetime.now(timezone.utc).isoformat()
markers = [json.loads(line.removeprefix("V4_NATIVE_STATUS "))
           for line in prefix.with_suffix(".stdout").read_text().splitlines()
           if line.startswith("V4_NATIVE_STATUS ")]
cgroup_path = prefix.with_suffix(".cgroup.json")
receipt.update(gdb_command_returncode=result.returncode,
    enclosing_seconds=enclosing_seconds, end_utc=end_utc,
    native_status_markers=markers,
    actual_cgroup=json.loads(cgroup_path.read_text()) if cgroup_path.exists() else None,
    status="DIAGNOSTIC_COMPLETE_AWAITING_INSPECTION")
receipt["outputs"] = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
    for p in sorted(prefix.parent.glob(prefix.name + ".*")) if p != receipt_path]
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
normal_exit = result.returncode == 0 and len(markers) == 1 and markers[0]["inferior_exit_code"] == 0 and markers[0]["inferior_exited"] and markers[0]["observed_stop_signal"] is None and markers[0]["run_error"] is None
print(json.dumps(dict(gdb_command_returncode=result.returncode,
    enclosing_seconds=receipt["enclosing_seconds"], native_status_markers=markers)), flush=True)
sys.exit(0 if normal_exit else 1)
