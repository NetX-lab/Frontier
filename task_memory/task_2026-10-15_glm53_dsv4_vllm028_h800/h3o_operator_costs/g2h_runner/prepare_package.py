"""Plan-check two isolated jobs on CPU; reuse P1h snapshot construction.

Default invocation constructs a fresh package after the lead commits and pins
source_revision. --check-plan/--dry_run never creates the package directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPLAY_PHASE = "G2H_RUNNER_REPLAY"
BLOCK_PHASE = "G2H_BLOCK_OPERATORS"
JOB_FIELDS = (
    "image", "cuda_compat_path", "cloud_mount", "gpu_type", "charged_group",
    "gpu", "cpu", "mem_gb", "preemptible", "timeout_seconds",
    "planning_gpu_hours_cap", "source_revision",
)


def job_definitions(spec, package):
    """Use the same job commands for CPU validation and package generation."""
    jobs = []
    for entry in spec["jobs"]:
        commands = [{"name": "bridge_run" if entry["phase"] == REPLAY_PHASE else "block_operators",
                     "argv": ["{python}", "-I", str(package / "native_dummy_bootstrap.py"),
                              "--config", str(package / entry["config"])]}]
        steps = [{"commands": commands}]
        if entry["phase"] == REPLAY_PHASE:
            steps.append({"commands": [{"name": "bridge_report", "argv": [
                "{python}", str(package / "dsv4_graph_bridge_check.py"),
                "--config", str(package / entry["config"]), "--report",
            ]}]})
        job = {key: spec[key] for key in JOB_FIELDS}
        job.update({key: entry[key] for key in ("name", "job_tag", "output_dir")})
        job.update(source_snapshot=str(package), worker=str(package / "launch/glm_producer_worker.py"),
                   wheelhouse=str(package / "launch/wheelhouse"),
                   phases=[{"name": entry["phase"], "steps": steps}])
        jobs.append(job)
    return jobs


def load_builder():
    path = HERE.parent / "p1h_bridge/prepare_package.py"
    spec = importlib.util.spec_from_file_location("g2h_p1h_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    return module


def check_plan(spec):
    from frontier.profiling.attention.dsv4_attention_main import build_parser, build_plan
    from frontier.profiling.model_runner.dsv4_block_operators import load_grid, direct_engine_plan
    g1h = json.loads((HERE.parent / "g1h/package_plan.json").read_text())
    for key in ("runs", "plan_args", "cudagraph_capture_sizes", "image", "cuda_compat_path", "cloud_mount"):
        if spec[key] != g1h[key]:
            raise ValueError(f"G2h replay {key} differs from the unchanged G1h plan")
    hf_path = ROOT / "data/config/models/deepseek-ai__DeepSeek-V4-Flash.json"
    grid = load_grid(HERE / spec["block_operator_grid"], json.loads(hf_path.read_text()))
    expected = json.loads((HERE / "expected_rows.json").read_text())
    if sorted(json.dumps(row, sort_keys=True) for row in grid.rows()) != sorted(
        json.dumps(row, sort_keys=True) for row in expected["path_b_rows"]
    ):
        raise ValueError("Path B planned rows differ from expected_rows.json")
    observed_cases = {}
    for run in spec["runs"]:
        plan = build_plan(build_parser().parse_args([
            "plan", *spec["plan_args"], "--model_config_path", str(hf_path),
            "--pilot_cases", str(HERE / spec["pilot_cases"]), "--engine_policy", "graph_replay",
            "--cudagraph_mode", run["mode"], "--cudagraph_capture_sizes", *map(str, spec["cudagraph_capture_sizes"]),
        ]))
        known = {case.case_id for case in plan.cases_for_engine(run["engine_index"])}
        if not set(run["case_ids"]) <= known:
            raise ValueError("Path A references unplanned G1h cases")
        observed_cases[str(run["engine_index"])] = run["case_ids"]
    if observed_cases != expected["path_a_cases"] or sum(map(len, observed_cases.values())) != expected["path_a_case_count"]:
        raise ValueError("Path A case IDs differ from expected_rows.json")
    # Plan the 43-layer startup through the same CLI, without loading vLLM.
    direct_args = [*spec["plan_args"], "--pilot_cases", str(HERE / spec["pilot_cases"]),
                   "--engine_policy", "graph_replay", "--cudagraph_mode", spec["runs"][0]["mode"],
                   "--cudagraph_capture_sizes", *map(str, spec["cudagraph_capture_sizes"])]
    direct_plan = direct_engine_plan(direct_args, hf_path, grid)
    if direct_plan.hf_overrides["num_hidden_layers"] != grid.payload["num_hidden_layers"]:
        raise ValueError("Direct engine plan did not retain the native layers")
    probe = '''import importlib, importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'fasteners','sklearn','torch','vllm','cuda'}: raise RuntimeError('Blocked import: '+fullname)
sys.meta_path.insert(0,Block())
for name in sys.argv[1:]:
 if name.endswith('.py'): runpy.run_path(name,run_name='probe')
 else: importlib.import_module(name)
print('worker import probe PASS')
'''
    modules = ["frontier.profiling.attention.dsv4_attention_main", "frontier.profiling.attention.dsv4_graph_replay_runner",
               "frontier.profiling.model_runner.dsv4_block_operators", "frontier.profiling.model_runner.dsv4_runner_rows",
               str(HERE / "profile_dsv4_block_operators.py"),
               str(ROOT / "tests/comparison/calibration/profile_dsv4_prefill.py"),
               str(ROOT / "tests/comparison/calibration/dsv4_graph_bridge_check.py")]
    result = subprocess.run([sys.executable, "-B", "-c", probe, *modules], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError("Worker import probe failed: " + result.stderr)
    jobs = job_definitions(spec, HERE / spec["destination"])
    if [job["phases"][0]["name"] for job in jobs] != [REPLAY_PHASE, BLOCK_PHASE]:
        raise ValueError("G2h requires one replay job and one block-operator job")
    for key in ("name", "job_tag", "output_dir"):
        if spec[key] != jobs[0][key]:
            raise ValueError(f"P1h {key} must select the replay job")
        if len({job[key] for job in jobs}) != len(jobs):
            raise ValueError(f"G2h jobs require distinct {key} values")
    job_reports = []
    for job in jobs:
        count = sum(Path(command["argv"][2]).name == "native_dummy_bootstrap.py"
                    for step in job["phases"][0]["steps"] for command in step["commands"])
        if count != 1:
            raise ValueError(f"{job['name']} must bootstrap exactly once")
        job_reports.append(dict(job, file=str(HERE / spec["destination"] / "jobs" / f"{job['name']}.json"),
                                bootstrap_calls=count))
    return {"package_built": False, "source_revision": spec["source_revision"], "phases": spec["phases"],
            "jobs": job_reports,
            "path_a_cases_per_engine": {key: len(values) for key, values in observed_cases.items()},
            "path_b_rows_per_method": expected["row_counts_per_method"], "path_b_total_rows": len(grid.rows()),
            "expected_layout": grid.layout, "import_probe": result.stdout.strip(),
            "gpu_time_estimate": spec["gpu_time_estimate"], "gpu_or_platform_operations": 0}


def build_package(spec):
    builder = load_builder()
    builder.main()
    package = HERE / spec["destination"]
    revision = spec["source_revision"]
    for name in ("block_operator_grid.json", "expected_rows.json", "native_kernel_notes.md", "README.md", "profile_dsv4_block_operators.py"):
        source = str((HERE / name).relative_to(ROOT))
        (package / name).write_bytes(subprocess.check_output(["git", "show", f"{revision}:{source}"], cwd=ROOT, timeout=60))
    builder.write(package / "block_plan_args.json", [*spec["plan_args"], "--pilot_cases", str(package / "pilot_cases.json"),
        "--engine_policy", "graph_replay", "--cudagraph_mode", spec["runs"][0]["mode"],
        "--cudagraph_capture_sizes", *map(str, spec["cudagraph_capture_sizes"])])
    bridge_path = package / "bridge_config.json"
    bridge = json.loads(bridge_path.read_text())
    bridge["output_root"] = f"{spec['worker_output_root']}/G2H_RUNNER_REPLAY/bridge"
    bootstrap = dict(bridge["bootstrap"])
    bootstrap.update(script=str(package / "profile_dsv4_block_operators.py"), profile_args=[
        "--grid", str(package / "block_operator_grid.json"), "--model_config_path", str(package / "model/config.json"),
        "--plan_args", str(package / "block_plan_args.json"), "--model_dir", str(package / "model"),
        "--output_dir", f"{spec['worker_output_root']}/G2H_BLOCK_OPERATORS",
        "--seed", str(spec["seed"]), "--gpu_memory_utilization", str(spec["gpu_memory_utilization"]),
    ])
    # Both bootstraps verify the added worker/grid before either phase executes.
    extra_hashes = {str(package / name): builder.sha(package / name)
                    for name in ("block_operator_grid.json", "block_plan_args.json", "profile_dsv4_block_operators.py")}
    for config in (bridge, bridge["bootstrap"], bootstrap):
        config["tool_hashes"] = dict(config["tool_hashes"], **extra_hashes)
    builder.write(bridge_path, bridge)
    builder.write(package / "block_config.json", bootstrap)
    for job in job_definitions(spec, package):
        builder.write(package / f"jobs/{job['name']}.json", job)
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    phases=spec["phases"], jobs=spec["jobs"], expected_rows=json.loads((package / "expected_rows.json").read_text()))
    manifest["files"] = {str(path.relative_to(package)): builder.sha(path) for path in sorted(package.rglob("*"))
                         if path.is_file() and path != manifest_path and not path.name.endswith(".pyc")}
    builder.write(manifest_path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-plan", "--dry_run", action="store_true")
    args = parser.parse_args(argv)
    spec = json.loads((HERE / "package_plan.json").read_text())
    report = check_plan(spec)
    if args.check_plan:
        print(json.dumps(report, indent=2))
    else:
        build_package(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
