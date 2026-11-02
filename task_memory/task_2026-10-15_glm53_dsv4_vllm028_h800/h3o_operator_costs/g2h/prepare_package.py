"""Check G2h plans on CPU; the lead builds a fresh package after committing B13.

--check-plan performs no package construction, installation, GPU or platform
operation. A build snapshots only committed source and reuses the accepted
H200 launcher, image and sealed base package.
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
GRID_PATH = str(HERE.relative_to(ROOT) / "operator_grid.json")
SNAPSHOT_PATHS = ("frontier", "data/config", GRID_PATH)
WORKSPACE_LOCATOR = "WORKSPACE = Path('/data/ycfeng/Frontier/.worktrees/v4-balanced-decode-run')"
CPU_TIMEOUT_SECONDS = 120


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_package(path, digest):
    if sha(path / "manifest.json") != digest:
        raise ValueError(f"Base manifest mismatch: {path}")
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(path / name) != expected:
            raise ValueError(f"Base input mismatch: {path / name}")
    return manifest


def invocations(spec, source):
    for method, filename in spec["methods"].items():
        yield {
            "name": "G2H_OPERATORS_" + method.upper(), "method": method,
            "filename": filename, "plan_name": None,
            "args": [*spec["common_args"], "--deepseek_v4_operator_grid", str(source / GRID_PATH)],
        }
        for topology in spec["topologies"]:
            ep = topology["expert_parallel_size"]
            yield {
                "name": f"G2H_SHARED_EXPERT_EP{ep}_{method.upper()}",
                "method": method, "filename": filename, "plan_name": f"shared_expert_ep{ep}",
                "args": [*spec["common_args"], "--sequence_parallel_moe_topology",
                         *(str(topology[key]) for key in (
                             "attention_tp_size", "attention_dp_size", "expert_parallel_size")),
                         "--num_tokens_list", *map(str, spec["shared_expert_rows"])],
            }


def check_plan(spec, source, *, python=sys.executable):
    expected = json.loads((HERE / spec["expected_rows"]).read_text())["phases"]
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1",
                       WANDB_DISABLED="true", VIDUR_DISABLE_WANDB="1")
    plans = {}
    for invocation in invocations(spec, source):
        result = subprocess.run(
            [python, "-m", "frontier.profiling.linear_op.main", *invocation["args"],
             "--profile_method", invocation["method"], "--dry_run"],
            cwd=source, env=environment, capture_output=True, text=True,
            check=True, timeout=CPU_TIMEOUT_SECONDS,
        )
        plan = json.loads(result.stdout[result.stdout.index("{"):])
        model = plan["models"][spec["model"]]
        actual = []
        for point in model["plans"]:
            axis = point.get("row_axis", "sp_local_rows")
            rows = point.get("num_tokens", plan["num_tokens"])
            for operator in point["enabled_ops"]:
                contract = point["operator_contracts"][operator]
                actual.append({
                    "plan": point.get("plan_name", invocation["plan_name"]),
                    "operator": operator, "axis": axis, "rows": rows,
                    "tp_size": contract["selected_tensor_parallel_size"],
                })
        want = expected[invocation["name"]]
        if (actual != want["plans"] or model["profiled_steps"] != want["csv_rows"]
                or sum(len(row["rows"]) for row in actual) != want["operator_rows"]):
            raise ValueError(f"{invocation['name']}: planned rows differ from expected_rows.json")
        if plan["profile_method"] != invocation["method"]:
            raise ValueError(f"{invocation['name']}: profiling method changed")
        plans[invocation["name"]] = plan
    if set(plans) != set(expected):
        raise ValueError("Producer phases differ from expected_rows.json")
    return plans


def snapshot(repo, revision, paths, destination):
    archive = subprocess.check_output(["git", "archive", revision, *paths], cwd=repo,
                                      timeout=CPU_TIMEOUT_SECONDS)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isfile():
                target = destination / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(item).read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-plan", action="store_true")
    args = parser.parse_args()
    spec = json.loads((HERE / "package_plan.json").read_text())
    expected = json.loads((HERE / spec["expected_rows"]).read_text())["phases"]
    if args.check_plan:
        plans = check_plan(spec, ROOT)
        print(json.dumps({"invocations": len(plans), "phases": {
            name: {key: expected[name][key] for key in ("csv_rows", "operator_rows")}
            for name in plans}, "package_built": False, "gpu_or_platform_operations": 0}))
        return
    revision = spec["source_revision"]
    committed = subprocess.check_output(["git", "rev-parse", "--verify", revision + "^{commit}"],
                                        cwd=ROOT, text=True, timeout=CPU_TIMEOUT_SECONDS).strip()
    if revision != committed:
        raise ValueError("source_revision must be a full committed revision")
    if subprocess.check_output(["git", "diff", revision, "--", *SNAPSHOT_PATHS],
                               cwd=ROOT, timeout=CPU_TIMEOUT_SECONDS):
        raise RuntimeError("Source bytes differ from the pinned committed revision")
    check_plan(spec, ROOT)
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
    tools = runpy.run_path(str(launcher))
    for key, constant in (("image", "IMAGE"), ("cuda_compat_path", "CUDA_COMPAT_PATH"),
                          ("cloud_mount", "CLOUD_MOUNT")):
        if spec[key] != tools[constant]:
            raise ValueError(f"{key} differs from the accepted H200 launcher")
    plans = check_plan(spec, source)
    outputs, phases = {}, []
    for invocation in invocations(spec, source):
        phase = invocation["name"]
        local = "/tmp/glm_producers/out/" + phase
        table = f"compute/{spec['common_args'][spec['common_args'].index('--device') + 1]}/{spec['model']}/{invocation['filename']}"
        argv = ["{python}", "-m", "frontier.profiling.linear_op.main", *invocation["args"],
                "--profile_method", invocation["method"], "--source_revision", revision,
                "--model_config_revision", spec["model_config_revision"], "--output_dir", local]
        count = expected[phase]["csv_rows"]
        phases.append({"name": phase, "steps": [tools["_step"](tools["_command"](phase, argv))],
                       "expected_tables": [{"path": table, "rows": count}]})
        outputs[phase] = {"rows": count, "operator_rows": expected[phase]["operator_rows"],
                          "phase": phase, "table": table,
                          "cloud_table": f"{spec['output_dir']}/{phase}/{table}"}
        write(package / f"plans/{phase}.json", plans[phase])
    job = {key: spec[key] for key in (
        "task_id", "name", "job_tag", "output_dir", "image", "cuda_compat_path", "cloud_mount",
        "gpu_type", "charged_group", "gpu", "cpu", "mem_gb", "preemptible", "timeout_seconds",
        "planning_gpu_hours_cap", "source_revision", "model_config_revision")}
    job.update(producer="linear_op", source_snapshot=str(source),
               worker=str(package / "launch/glm_producer_worker.py"),
               wheelhouse=str(package / "launch/wheelhouse"), phases=phases,
               formal_coverage=False, qualification=spec["qualification"])
    write(package / "jobs/linear_op.json", job)
    for name in ("prepare_package.py", "package_plan.json", "operator_grid.json",
                 "expected_rows.json", "native_kernel_notes.md"):
        shutil.copy2(HERE / name, package / name)
    write(package / "runtime_lock.json", {"native_revision": spec["native_revision"],
          "image": spec["image"], "actual_image_checked": False,
          "native_source_is_reference_not_worker_overlay": True})
    files = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob("*"))
             if path.is_file()}
    write(package / "manifest.json", {"task_id": spec["task_id"], "files": files,
          "source_revision": revision, "native_revision": spec["native_revision"],
          "base_manifest_sha256": spec["base_manifest_sha256"], "expected_outputs": outputs,
          "gpu_time_estimate": spec["gpu_time_estimate"],
          "planning_gpu_hours_cap": spec["planning_gpu_hours_cap"],
          "qualification": spec["qualification"], "sealed": False, "submitted": False,
          "actual_image_checked": False, "actual_GPU_checked": False})
    verify_package(package, sha(package / "manifest.json"))
    verify_package(base, spec["base_manifest_sha256"])
    print(json.dumps({"package": str(package), "expected_outputs": outputs,
                      "gpu_or_platform_operations": 0}))


if __name__ == "__main__":
    main()
