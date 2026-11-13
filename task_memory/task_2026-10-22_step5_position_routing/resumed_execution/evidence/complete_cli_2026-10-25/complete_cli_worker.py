"""Run the unchanged complete CPU-fixture CLI and record its actual worker memory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_plan(path):
    raw = path.read_bytes()
    plan = json.loads(raw)
    directory = path.resolve().parent
    for name, expected in plan["scripts"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Complete CLI script changed: {name}")
    assets = {plan["fixture_manifest"]: plan["fixture_manifest_sha256"],
              plan["joint_bindings"]: plan["joint_bindings_sha256"], **plan["git"]["files"],
              plan["sdk_submission_observation"]["source"]: plan["sdk_submission_observation"]["source_sha256"]}
    for filename, expected in assets.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Complete CLI input changed: {filename}")
    expected_command = [plan["python"], "-B", str(Path(plan["source_root"]) /
        "tests/integration/run_step5_measured_main.py"), "--fixture-manifest", plan["fixture_manifest"],
        "--output-root", plan["cloud_output_root"], "--mode", "all"]
    if plan["command"] != expected_command:
        raise AssertionError("Use the unchanged complete offline/online CLI command")
    if plan["main_child_timeout_s"] != 2400 or plan["outer_timeout_s"] is not None:
        raise AssertionError("Retain the 2400s main-child limit with validation outside it")
    if plan["resource"]["gpu"] != 0 or plan["job_limit"] != 1 or plan["automatic_retries"] != 0:
        raise AssertionError("This finite execution uses one zero-GPU worker without retries")
    bindings = json.loads(Path(plan["joint_bindings"]).read_bytes())
    if (plan["source_root"], plan["source_commit"]) != (bindings["source_root"], bindings["source_commit"]):
        raise AssertionError("The CLI source differs from its admitted source bindings")
    return plan, hashlib.sha256(raw).hexdigest()


def controller_paths():
    groups = [line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()]
    locations = {}
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        mount, filesystem = line.split(" - ", 1)
        fields, options = mount.split(), filesystem.split()
        for hierarchy, controllers, group in groups:
            if options[0] == "cgroup2" and hierarchy == "0":
                names = ("memory.current", "memory.peak", "memory.max", "memory.events")
            elif options[0] == "cgroup" and "memory" in controllers.split(",") and "memory" in options[2].split(","):
                names = ("memory.usage_in_bytes", "memory.max_usage_in_bytes", "memory.limit_in_bytes", "memory.failcnt", "memory.oom_control")
            else:
                continue
            relative = Path(group).relative_to(Path(fields[3]))
            directory = Path(fields[4]) / relative
            locations.update({name: str(directory/name) for name in names})
    return locations


def memory_reading(paths, runner_pid=None):
    reading = {"observed_utc": datetime.now(timezone.utc).isoformat(), "controller": {}, "main_children": []}
    for name, filename in paths.items():
        try:
            reading["controller"][name] = Path(filename).read_text().strip()
        except OSError as error:
            reading["controller"][name] = {"unavailable": f"{type(error).__name__}: {error}"}
    if runner_pid is not None:
        try:
            children = Path(f"/proc/{runner_pid}/task/{runner_pid}/children").read_text().split()
            for child in children:
                command = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\x00")
                if b"frontier.main" not in command:
                    continue
                status = Path(f"/proc/{child}/status").read_text().splitlines()
                mode = command[command.index(b"--simulation_mode")+1].decode()
                reading["main_children"].append({"pid": int(child), "mode": mode,
                    "status_memory_fields": [line for line in status if line.startswith(("VmRSS:", "VmHWM:", "VmSize:"))]})
        except (OSError, ValueError) as error:
            reading["main_sample_unavailable"] = f"{type(error).__name__}: {error}"
    return reading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan, plan_sha256 = read_plan(args.plan)
    source = Path(plan["source_root"])
    preparation = Path(plan["preparation_directory"])
    output = Path(plan["local_output_root"])/"worker"
    output.mkdir(parents=True, exist_ok=False)
    cloud = Path(plan["cloud_output_root"]).resolve()
    if not cloud.is_relative_to(Path(plan["cloud_personal_root"]).resolve()):
        raise AssertionError("Full output must stay inside the personal codesign directory")
    receipt = {"task_id": plan["task_id"], "status": "IN_PROGRESS", "job_name": os.environ.get("KUBEBRAIN_RJOB_NAME"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "worker_hostname": socket.gethostname(),
        "command": plan["command"], "cwd": str(source), "gpu_request": 0, "ordinary_joint_controls": 0,
        "required_shape_controls": 0, "native_collections": 0, "main_child_timeout_s": 2400,
        "outer_timeout_s": None, "full_workload_memory_qualified": False,
        "plan_sha256": plan_sha256, "python": sys.version, "executable": sys.executable,
        "math_thread_env": plan["math_thread_env"]}
    write_json(output/"worker_started.json", receipt)
    result = None
    started = time.monotonic()
    try:
        os.environ["PATH"] = str(Path(plan["git"]["path"]).parent) + os.pathsep + os.environ["PATH"]
        os.environ["GIT_CONFIG_GLOBAL"] = plan["git"]["config"]
        sys.path.insert(0, str(preparation))
        from joint_timing import check_bindings
        bindings = json.loads(Path(plan["joint_bindings"]).read_bytes())
        receipt["source_before"] = check_bindings(bindings, preparation)
        launch = json.loads((output.parent / "launcher.json").read_bytes())
        if launch["source"] != receipt["source_before"] or launch["plan_sha256"] != plan_sha256:
            raise AssertionError("Worker source/plan differs from the local submission")
        receipt["mounts"] = [line for line in Path("/proc/self/mountinfo").read_text().splitlines()
            if f' {plan["local_mount_target"]} ' in line or " /mnt/codesign-exp " in line]
        expected_nfs = f'{launch["nfs_server"]}:{plan["local_mount_source"]}'
        if not any(f" - nfs4 {expected_nfs} " in line for line in receipt["mounts"]):
            raise AssertionError("The worker NFS source differs from the local submission")
        if not any(" /mnt/codesign-exp " in line and "JuiceFS:codesign-exp" in line
                   for line in receipt["mounts"]):
            raise AssertionError("The personal cloud-volume mount is absent")
        if cloud.exists():
            raise AssertionError("Full CLI output directory already exists")
        cloud.parent.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1", "TMPDIR": plan["worker_tmp_root"], "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1", **plan["math_thread_env"]}
        Path(plan["worker_tmp_root"]).mkdir(parents=True, exist_ok=False)
        for name in ("FRONTIER_TEST_CPU_WRITER_FILE", "FRONTIER_TEST_CPU_WRITER_SHA256", "FRONTIER_STEP5_PROFILE_MANIFEST"):
            environment.pop(name, None)
        os.chdir(source)
        try:
            paths = controller_paths()
        except (OSError, ValueError) as error:
            paths = {}
            receipt["controller_discovery_error"] = f"{type(error).__name__}: {error}"
        receipt["controller_paths"] = paths
        receipt["memory_before"] = memory_reading(paths)
        write_json(output/"complete_cli.json", receipt)
        with (output/"runner.stdout").open("w") as stdout, (output/"runner.stderr").open("w") as stderr:
            runner = subprocess.Popen(plan["command"], cwd=source, env=environment, stdout=stdout, stderr=stderr)
            receipt["runner_pid"] = runner.pid
            write_json(output/"complete_cli.json", receipt)
            with (output/"memory_samples.jsonl").open("w") as samples:
                while True:
                    sample = memory_reading(paths, runner.pid)
                    samples.write(json.dumps(sample)+"\n")
                    samples.flush()
                    result = runner.poll()
                    if result is not None:
                        break
                    time.sleep(plan["memory_sample_period_s"])
        receipt["runner_returncode"] = result
        receipt["memory_after"] = memory_reading(paths)
        receipt["completed_descendant_ru_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        receipt["memory_limits"] = ["Controller peak includes charged page cache and the container lifetime.",
            "Sampled main VmHWM may miss the final peak; descendant ru_maxrss includes runner and validation.",
            "A failed or truncated workload supplies partial-run memory, never full-workload qualification."]
        summary = cloud/"summary.json"
        receipt["summary"] = json.loads(summary.read_bytes()) if summary.exists() else None
        receipt["source_after"] = check_bindings(bindings, preparation)
        if hashlib.sha256(args.plan.read_bytes()).hexdigest() != plan_sha256:
            raise AssertionError("The complete CLI plan changed during execution")
        if receipt["source_after"] != receipt["source_before"]:
            raise AssertionError("Source/input binding changed during complete CLI")
        passed = result == 0 and receipt["summary"] is not None and len(receipt["summary"]) == 2
        passed = passed and [row["mode"] for row in receipt["summary"]] == ["offline", "online"]
        passed = passed and all(row["status"] == "PASS" for row in receipt["summary"])
        receipt["status"] = "PASS_COMPLETE_CPU_FIXTURE_CLI" if passed else "FAIL_COMPLETE_CPU_FIXTURE_CLI"
    except BaseException as error:
        receipt["status"] = "FAIL"
        receipt["error"] = f"{type(error).__name__}: {error}"
        (output/"failure.log").write_text(traceback.format_exc())
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["worker_wall_s"] = time.monotonic()-started
        write_json(output/"complete_cli.json", receipt)
    print(json.dumps({key: receipt.get(key) for key in ("status", "job_name", "runner_returncode", "worker_wall_s", "completed_descendant_ru_maxrss_bytes")}))
    return 0 if receipt["status"] == "PASS_COMPLETE_CPU_FIXTURE_CLI" else 1


if __name__ == "__main__":
    sys.exit(main())
