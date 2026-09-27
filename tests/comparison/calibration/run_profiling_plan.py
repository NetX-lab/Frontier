"""Run a profiling plan on a GPU worker and publish its CSVs.

The plan (JSON) names the worker environment fixes, the packages to install,
the imports that must succeed, and one entry per profiler invocation. Each
invocation writes under its own output root, so two runs of the same profiler
(for example two MoE gating contexts) do not overwrite each other's CSVs.

The run is published twice, to the cloud-volume archive and to the evidence
directory on the mounted workspace. Each copy is assembled beside its target
and renamed into place, and `COMPLETE` is written last. Neither target may
exist before the run.

Exit status: 0 every invocation passed, 2 an output already exists, 3 the
environment check failed, 4 at least one invocation failed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ENVIRONMENT_PROBE = """
import importlib, json, sys
modules = json.loads(sys.argv[1])
report = {"python": sys.version.split()[0], "imports": {}}
for name in modules:
    try:
        module = importlib.import_module(name)
        report["imports"][name] = getattr(module, "__version__", "present")
    except Exception as exc:
        report["imports"][name] = None
        report.setdefault("import_errors", {})[name] = f"{type(exc).__name__}: {exc}"
import torch
report["cuda_available"] = torch.cuda.is_available()
report["torch_cuda"] = torch.version.cuda
report["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps(report))
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frontier-tree", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    return parser.parse_args()


def child_environment(environment: dict, frontier_tree: Path) -> dict:
    libcuda_dir = next(
        (d for d in environment["libcuda_dirs"] if Path(d, "libcuda.so.1").exists()),
        None,
    )
    if libcuda_dir is None:
        raise RuntimeError(f"libcuda.so.1 not found in {environment['libcuda_dirs']}")
    env = dict(os.environ)
    library_path = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = libcuda_dir + (f":{library_path}" if library_path else "")
    env["PYTHONPATH"] = str(frontier_tree)
    env.update(PYTHONDONTWRITEBYTECODE="1", WANDB_DISABLED="true", VIDUR_DISABLE_WANDB="1")
    return env


def run_logged(command: list[str], log_path: Path, timeout_s: float, **kwargs) -> dict:
    started = time.monotonic()
    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        try:
            returncode = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, **kwargs
            ).returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"
    return {"returncode": returncode, "wall_s": round(time.monotonic() - started, 1)}


def csv_summary(path: Path, run_dir: Path) -> dict:
    with path.open(newline="") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    return {
        "path": str(path.relative_to(run_dir)),
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def publish(source: Path, target: Path, status: int) -> None:
    staging = target.with_name(target.name + ".partial")
    shutil.copytree(source, staging)
    staging.rename(target)
    (target / "COMPLETE").write_text(f"status={status}\n")


def main() -> int:
    args = parse_args()
    for target in (args.archive_dir, args.evidence_dir):
        if target.exists():
            print(f"WORKER_STATUS=2 output already exists: {target}", flush=True)
            return 2
    plan = json.loads(args.plan.read_text())
    environment = plan["environment"]
    run_dir = args.work_dir / "run"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    shutil.copy(args.plan, run_dir / "profiling_plan.json")
    manifest = {"plan": str(args.plan), "frontier_tree": str(args.frontier_tree)}

    status = 0
    try:
        env = child_environment(environment, args.frontier_tree)
    except RuntimeError as exc:
        manifest["environment_error"] = str(exc)
        status = 3
    if status == 0:
        manifest["pip"] = run_logged(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--quiet",
             "--root-user-action=ignore", *environment["pip_install_args"],
             *environment["pip_packages"]],
            logs / "pip.log", environment["pip_timeout_s"],
        )
        probe = subprocess.run(
            [sys.executable, "-c", ENVIRONMENT_PROBE,
             json.dumps(environment["required_imports"])],
            env=env, capture_output=True, text=True,
        )
        (logs / "environment.log").write_text(probe.stdout + probe.stderr)
        report = json.loads(probe.stdout) if probe.returncode == 0 else None
        manifest["environment"] = report
        if (
            manifest["pip"]["returncode"] != 0
            or report is None
            or report.get("import_errors")
            or not report["cuda_available"]
        ):
            status = 3
        elif "CUDA_VISIBLE_DEVICES" not in env:
            # The profilers discover GPUs from the visibility variables or
            # nvidia-smi, which some worker images lack; name the devices the
            # probe found.
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(len(report["devices"]))))
            manifest["cuda_visible_devices"] = env["CUDA_VISIBLE_DEVICES"]

    if status == 0:
        manifest["invocations"] = []
        for invocation in plan["invocations"]:
            output_root = run_dir / "out" / invocation["name"]
            result = run_logged(
                [sys.executable, "-m", invocation["module"], *plan["common_args"],
                 *invocation["args"], "--output_dir", str(output_root)],
                logs / f"{invocation['name']}.log", invocation["timeout_s"],
                cwd=args.frontier_tree, env=env,
            )
            result["name"] = invocation["name"]
            result["csvs"] = [
                csv_summary(p, run_dir) for p in sorted(output_root.rglob("*.csv"))
            ]
            manifest["invocations"].append(result)
            print(f"INVOCATION {json.dumps(result)}", flush=True)
            if result["returncode"] != 0:
                status = 4

    manifest["status"] = status
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    args.archive_dir.parent.mkdir(parents=True, exist_ok=True)
    publish(run_dir, args.archive_dir, status)
    publish(run_dir, args.evidence_dir, status)
    # The worker writes the mounted workspace as root; hand the evidence back to
    # the owner of the directory that holds it.
    owner = args.evidence_dir.parent.stat()
    for path in [args.evidence_dir, *args.evidence_dir.rglob("*")]:
        os.chown(path, owner.st_uid, owner.st_gid)
    print(f"ENVIRONMENT {json.dumps(manifest.get('environment'))}", flush=True)
    print(f"WORKER_STATUS={status}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
