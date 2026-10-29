"""Prepare committed G3h MoE sources for the accepted H200 producer launcher.

--check-plan only runs CPU dry-run checks on the current working tree. Without
it, the lead builds a fresh package from source_revision after committing B12 and B12b.
The plan uses native GateLinear and int64 balanced routed ids; remaining
qualifications and source evidence are recorded in native_kernel_notes.md.
This builder performs no platform, image, installation or GPU operation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SNAPSHOT_PATHS = ("frontier", "data/config")
WORKSPACE_LOCATOR = "WORKSPACE = Path('/data/ycfeng/Frontier/.worktrees/v4-balanced-decode-run')"
CPU_TIMEOUT_SECONDS = 120


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_package(path: Path, digest: str) -> dict:
    if sha(path / "manifest.json") != digest:
        raise ValueError(f"Base manifest mismatch: {path}")
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(path / name) != expected:
            raise ValueError(f"Base input mismatch: {path / name}")
    return manifest


def topology_args(spec: dict, topology: dict) -> list[str]:
    return [*spec["common_args"], "--expert_parallel_sizes", str(topology["expert_parallel_size"]),
            "--dp_lane_local_rows", *(
                ",".join(map(str, rows)) for rows in topology["dp_lane_local_rows"])]


def task_key(task: dict) -> list:
    return [task["expert_parallel_size"], task["num_tokens"],
            [task[name] for name in ("local_routed_rows", "local_experts_with_extra_row",
                                    "base_rows_per_expert")],
            task["gating_num_tokens"], task["ep_rank"]]


def check_plan(spec: dict, source: Path, *, python: str = sys.executable) -> dict:
    expected = json.loads((HERE / spec["expected_tasks"]).read_text())["tasks"]
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1",
                       WANDB_DISABLED="true", VIDUR_DISABLE_WANDB="1")
    plans = {}
    for method in spec["methods"]:
        for topology in spec["topologies"]:
            ep = topology["expert_parallel_size"]
            result = subprocess.run(
                [python, "-m", "frontier.profiling.moe.main", *topology_args(spec, topology),
                 "--profile_method", method, "--dry_run"],
                cwd=source, env=environment, capture_output=True, text=True,
                check=True, timeout=CPU_TIMEOUT_SECONDS,
            )
            # Existing producer import diagnostics precede the JSON object.
            plan = json.loads(result.stdout[result.stdout.index("{"):])
            model = plan["models"][spec["model"]]
            tasks = model["rank_tasks"]
            selected = [row for row in expected if row[0] == ep]
            if ([task_key(task) for task in tasks] != selected
                    or model["profiled_steps"] != len(selected)):
                raise ValueError(f"{method}/EP{ep}: producer rank tasks differ from expected_tasks")
            if (plan["cases_per_parallel_point"] != 1
                    or plan["load_distributions"] != ["balanced"]
                    or model["tensor_parallel_sizes"] != [1]):
                raise ValueError(f"{method}/EP{ep}: producer method/domain changed")
            plans[f"{method}_ep{ep}"] = plan
    return plans


def snapshot(repo: Path, revision: str, paths: list | tuple, destination: Path) -> None:
    archive = subprocess.check_output(["git", "archive", revision, *paths], cwd=repo,
                                      timeout=CPU_TIMEOUT_SECONDS)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isfile():
                target = destination / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(item).read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-plan", action="store_true")
    args = parser.parse_args()
    spec = json.loads((HERE / "package_plan.json").read_text())
    if args.check_plan:
        plans = check_plan(spec, ROOT)
        print(json.dumps({"plans": len(plans), "tasks_per_method": {
            str(topology["expert_parallel_size"]): plans[
                f"cuda_event_ep{topology['expert_parallel_size']}"
            ]["models"][spec["model"]]["profiled_steps"] for topology in spec["topologies"]},
            "package_built": False, "gpu_or_platform_operations": 0}))
        return

    revision, native = spec["source_revision"], spec["native_revision"]
    for repo, ref, paths in ((ROOT, revision, SNAPSHOT_PATHS),
                             (Path(spec["native_repo"]), native, spec["native_paths"])):
        committed = subprocess.check_output(["git", "rev-parse", "--verify", ref + "^{commit}"],
                                            cwd=repo, text=True, timeout=CPU_TIMEOUT_SECONDS).strip()
        if ref != committed:
            raise ValueError("source_revision and native_revision must be full committed revisions")
        if subprocess.check_output(["git", "diff", ref, "--", *paths], cwd=repo,
                                   timeout=CPU_TIMEOUT_SECONDS):
            raise RuntimeError(f"Source bytes differ from the pinned revision: {repo}")
    base = Path(spec["base_package"])
    verify_package(base, spec["base_manifest_sha256"])
    package = HERE / spec["destination"]
    if package.exists():
        raise RuntimeError("Fresh generation only; never alter an existing package")
    package.mkdir()
    for name in ("glm_producer_jobs.py", "glm_producer_worker.py", "launch_glm_producers.sh"):
        target = package / "launch" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / "launch" / name, target)
    launcher = package / "launch/glm_producer_jobs.py"
    text = launcher.read_text()
    if text.count(WORKSPACE_LOCATOR) != 1:
        raise ValueError("Accepted launcher workspace locator changed")
    launcher.write_text(text.replace(WORKSPACE_LOCATOR, f"WORKSPACE = Path({str(ROOT)!r})"))
    shutil.copytree(base / "launch/wheelhouse", package / "launch/wheelhouse")
    source = package / "source" / revision
    snapshot(ROOT, revision, SNAPSHOT_PATHS, source)
    native_source = package / "native_source" / native
    snapshot(Path(spec["native_repo"]), native, spec["native_paths"], native_source)
    native_hashes = {}
    for path in native_source.rglob("*"):
        if path.is_file():
            name = str(path.relative_to(native_source))
            if path.read_bytes() != (Path(spec["native_repo"]) / name).read_bytes():
                raise ValueError(f"Native source differs from committed bytes: {name}")
            native_hashes[name] = sha(path)
    for name, symbols in spec["native_symbols"].items():
        if any(symbol not in (native_source / name).read_text() for symbol in symbols):
            raise ValueError(f"Required native MoE symbol missing: {name}")
    for path in (Path(__file__), HERE / "package_plan.json", HERE / spec["expected_tasks"],
                 HERE / spec["native_kernel_notes"]):
        shutil.copy2(path, package / path.name)
    tools = runpy.run_path(str(launcher))
    for key, constant in (("image", "IMAGE"), ("cuda_compat_path", "CUDA_COMPAT_PATH"),
                          ("cloud_mount", "CLOUD_MOUNT")):
        if spec[key] != tools[constant]:
            raise ValueError(f"{key} differs from the accepted H200 launcher")
    plans = check_plan(spec, source)
    expected_outputs, phases = {}, []
    for method, filename in spec["methods"].items():
        phase = spec["name"] + "_" + method.upper()
        local = "/tmp/glm_producers/out/" + phase
        commands = []
        for topology in spec["topologies"]:
            ep = topology["expert_parallel_size"]
            plan = plans[f"{method}_ep{ep}"]
            write(package / f"plans/{method}_ep{ep}.json", plan)
            count = plan["models"][spec["model"]]["profiled_steps"]
            argv = ["{python}", "-m", "frontier.profiling.moe.main", *topology_args(spec, topology),
                    "--profile_method", method, "--source_revision", revision,
                    "--model_config_revision", spec["model_config_revision"],
                    "--output_dir", f"{local}/ep{ep}"]
            table = f"ep{ep}/compute/h200/{spec['model']}/{filename}"
            commands.append(tools["_step"](tools["_command"](f"{method}_ep{ep}", argv)))
            expected_outputs[f"{method}_ep{ep}"] = {"rows": count, "phase": phase, "table": table,
                "cloud_table": f"{spec['output_dir']}/{phase}/{table}"}
        phases.append({"name": phase, "steps": commands, "expected_tables": [
            {"path": output["table"], "rows": output["rows"]}
            for key, output in expected_outputs.items() if key.startswith(method + "_")]})
    job = {key: spec[key] for key in ("task_id", "name", "job_tag", "output_dir", "image",
        "cuda_compat_path", "cloud_mount", "gpu_type", "charged_group", "gpu", "cpu", "mem_gb",
        "preemptible", "timeout_seconds", "planning_gpu_hours_cap", "source_revision",
        "model_config_revision")}
    job.update(producer="moe", source_snapshot=str(source),
               worker=str(package / "launch/glm_producer_worker.py"),
               wheelhouse=str(package / "launch/wheelhouse"), phases=phases,
               formal_coverage=False, qualification=spec["qualification"])
    write(package / "jobs/moe.json", job)
    write(package / "runtime_lock.json", {"native_revision": native, "image": spec["image"],
        "native_source_hashes": native_hashes, "native_symbols": spec["native_symbols"],
        "actual_image_checked": False, "native_source_is_reference_not_worker_overlay": True})
    files = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob("*"))
             if path.is_file()}
    write(package / "manifest.json", {"task_id": spec["task_id"], "files": files,
        "source_revision": revision, "native_revision": native,
        "base_manifest_sha256": spec["base_manifest_sha256"],
        "builder_sha256": sha(Path(__file__)), "package_plan_sha256": sha(HERE / "package_plan.json"),
        "expected_tasks_sha256": sha(HERE / spec["expected_tasks"]), "expected_outputs": expected_outputs,
        "gpu_time_estimate": spec["gpu_time_estimate"], "planning_gpu_hours_cap": spec["planning_gpu_hours_cap"],
        "qualification": spec["qualification"], "sealed": False, "submitted": False,
        "actual_image_checked": False, "actual_GPU_checked": False})
    verify_package(package, sha(package / "manifest.json"))
    verify_package(base, spec["base_manifest_sha256"])
    print(json.dumps({"package": str(package), "expected_outputs": expected_outputs,
                      "gpu_or_platform_operations": 0}))


if __name__ == "__main__":
    main()
