"""Verify this mounted CPU runtime and one append control; do not run pricing or full CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import shutil
from functools import partial
import subprocess
import sys
import time
import traceback


def limit_address_space(maximum):
    resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))


def run_child(command, label, output, environment, limits):
    started = time.monotonic()
    with (output / f"{label}.stdout").open("w") as stdout, (output / f"{label}.stderr").open("w") as stderr:
        try:
            completed = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr,
                timeout=limits["child_timeout_s"],
                preexec_fn=partial(limit_address_space, limits["child_address_space_bytes"]))
            result = {"returncode": completed.returncode, "timed_out": False}
        except subprocess.TimeoutExpired:
            result = {"returncode": None, "timed_out": True}
    result.update(command=command, cwd=str(Path.cwd()), wall_s=time.monotonic()-started,
        configured_address_space_bytes=limits["child_address_space_bytes"])
    (output / f"{label}.execution.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["returncode"] != 0:
        raise RuntimeError(f"{label} failed: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    directory = args.plan.resolve().parent
    source = Path(plan["source_root"])
    output = Path(plan["local_output_root"]) / "worker"
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"task_id": plan["task_id"], "status": "IN_PROGRESS",
        "started_utc": datetime.now(timezone.utc).isoformat(), "worker_hostname": socket.gethostname(),
        "source_root": str(source), "pricing_calls": 0, "required_shape_calls": 0,
        "full_cli_calls": 0, "gpu_request": plan["resource"]["gpu"], "full_workload_memory_verified": False}
    try:
        from joint_timing import check_bindings
        bindings = json.loads(Path(plan["joint_bindings"]).read_bytes())
        receipt["source"] = check_bindings(bindings, directory)
        for name, expected in plan["scripts"].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
                raise AssertionError(f"Worker script changed: {name}")
        if hashlib.sha256(Path(plan["runtime_probe"]).read_bytes()).hexdigest() != plan["runtime_probe_sha256"]:
            raise AssertionError("Original runtime probe changed")
        if not Path(plan["runtime_base_prefix"]).is_dir():
            raise AssertionError("Copied base prefix is not in the source mount")
        mount_lines = Path("/proc/self/mountinfo").read_text().splitlines()
        receipt["mounts"] = [line for line in mount_lines
            if f' {plan["local_mount_target"]} ' in line or " /mnt/codesign-exp " in line]
        if not any(" /mnt/codesign-exp " in line for line in receipt["mounts"]):
            raise AssertionError("Expected cloud-volume mount is absent")
        gpu_program = shutil.which("nvidia-smi")
        receipt["nvidia_smi"] = {"available": gpu_program is not None,
            "qualification": "NOT_APPLICABLE_ZERO_GPU_REQUEST"}
        if gpu_program is not None:
            gpu = subprocess.run([gpu_program, "-L"], capture_output=True, text=True,
                timeout=plan["inspection_request_timeout_s"])
            receipt["nvidia_smi"].update(command=[gpu_program, "-L"], returncode=gpu.returncode,
                stdout=gpu.stdout, stderr=gpu.stderr)
        environment = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1",
            "WANDB_DISABLED": "true", "VIDUR_DISABLE_WANDB": "1", "PYTHONNOUSERSITE": "1",
            **plan["math_thread_env"]}
        for name in ("FRONTIER_TEST_CPU_WRITER_FILE", "FRONTIER_TEST_CPU_WRITER_SHA256",
                     "FRONTIER_STEP5_PROFILE_MANIFEST"):
            environment.pop(name, None)
        os.chdir(source)
        receipt["runtime"] = run_child([plan["python"], "-B", plan["runtime_probe"],
            str(Path(plan["python"]).parent.parent)], "runtime_probe", output, environment, plan["append"])
        probe = json.loads((output / "runtime_probe.stdout").read_bytes())
        if probe["status"] != "PASS_CPU_IMPORTS_ONLY" or probe["simulator_constructed"]:
            raise AssertionError("Worker runtime probe did not retain its original predicate")
        receipt["append"] = run_child([plan["python"], "-B", str(directory/"worker_append.py"),
            "--plan", str(args.plan.resolve()), "--receipt", str(output/"append.json")],
            "append", output, environment, plan["append"])
        append = json.loads((output / "append.json").read_bytes())
        if append["status"] != "PASS_SERIALIZED_REFERENCE_APPEND_ONLY" or not append["cleanup"].startswith("PASS_"):
            raise AssertionError("Append read-back or scoped cleanup failed")
        if check_bindings(bindings, directory) != receipt["source"]:
            raise AssertionError("Source binding changed during worker checks")
        receipt["status"] = "PASS_WORKER_IMPORT_AND_SERIALIZED_APPEND_ONLY"
        receipt["append_bytes_per_s"] = append["append_bytes_per_s"]
    except BaseException as error:
        receipt["status"] = "FAIL"
        receipt["error"] = f"{type(error).__name__}: {error}"
        (output / "failure.log").write_text(traceback.format_exc())
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "readiness.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in
        ("status", "worker_hostname", "pricing_calls", "full_cli_calls", "append_bytes_per_s")}))


if __name__ == "__main__":
    main()
