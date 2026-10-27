"""Stage the P1h graph-bridge package from the accepted V4 package and one commit.

The package reuses the accepted C26 bootstrap, producer worker, launcher, model
directory and native pins of ``h800_decode_balanced_02`` unchanged, except the
launcher's workspace locator. The committed ``frontier`` and ``data/config``
trees are extracted at the package root, because the accepted bootstrap puts
only the package directory on ``PYTHONPATH`` before it runs the bridge driver.
The driver then runs one profiling process per graph mode. A fresh destination
is required; an existing package is never altered.
"""

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DRIVER = "tests/comparison/calibration/dsv4_graph_bridge_check.py"
PROFILE_SCRIPT = "tests/comparison/calibration/profile_dsv4_prefill.py"
ROUTING = "frontier/profiling/moe/balanced_routing.py"
SNAPSHOT_PATHS = ("frontier", "data/config")
BASE_FILES = ("launch/glm_producer_jobs.py", "launch/glm_producer_worker.py",
              "launch/launch_glm_producers.sh", "native_dummy_bootstrap.py", "overlay_model.py",
              "official_model.py")
WORKSPACE_LOCATOR = "WORKSPACE = Path('/data/ycfeng/Frontier/.worktrees/v4-balanced-decode-run')"
PROFILE_KEYS = ("expected_source_hashes", "moe_gating_routing_runtime_path", "native_package")


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


def main():
    spec = json.loads((HERE / "package_plan.json").read_text())
    revision = spec["source_revision"]
    if subprocess.run(["git", "diff", "--quiet", revision, "--", *SNAPSHOT_PATHS, DRIVER, PROFILE_SCRIPT],
                      cwd=ROOT).returncode:
        raise RuntimeError("Snapshot paths differ from the pinned revision")
    base = Path(spec["base_package"])
    verify_package(base, spec["base_manifest_sha256"])
    package = HERE / spec["destination"]
    if package.exists():
        raise RuntimeError("Fresh generation only; never alter an existing package")
    package.mkdir()
    for name in BASE_FILES:
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / name, package / name)
    launcher = package / "launch/glm_producer_jobs.py"
    source = launcher.read_text()
    if source.count(WORKSPACE_LOCATOR) != 1:
        raise ValueError("Accepted launcher workspace locator changed")
    launcher.write_text(source.replace(WORKSPACE_LOCATOR, f"WORKSPACE = Path({str(ROOT)!r})"))
    shutil.copytree(base / "launch/wheelhouse", package / "launch/wheelhouse")
    shutil.copytree(base / "model", package / "model")
    for name in (DRIVER, PROFILE_SCRIPT, ROUTING):
        (package / Path(name).name).write_bytes(
            subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT))
    archive = subprocess.check_output(["git", "archive", revision, *SNAPSHOT_PATHS], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
        for item in snapshot.getmembers():
            if item.isfile():
                target = package / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(snapshot.extractfile(item).read())

    sys.path.insert(0, str(package))
    from frontier.profiling.attention.dsv4_attention_main import build_parser, build_plan
    from frontier.profiling.attention.dsv4_attention_rows import BALANCED_ROUTING_SHA256

    if sha(package / "balanced_routing.py") != BALANCED_ROUTING_SHA256:
        raise ValueError("Accepted balanced routing strategy mismatch")
    plan_args = [*spec["plan_args"], "--model_config_path", str(package / "model/config.json")]
    expected = {}
    for mode in spec["modes"]:
        sizes = [] if mode == "NONE" else ["--cudagraph_capture_sizes",
                                           *map(str, spec["cudagraph_capture_sizes"])]
        plan = build_plan(build_parser().parse_args(
            ["plan", *plan_args, "--engine_policy", "graph_replay", "--cudagraph_mode", mode, *sizes]))
        cases = {case.case_id: case for case in plan.cases if case.engine_index == spec["engine_index"]}
        missing = sorted(set(spec["case_ids"]) - cases.keys())
        if missing:
            raise ValueError(f"{mode}: planned engine {spec['engine_index']} lacks cases {missing}")
        write(package / f"plans/{mode}.json", plan.to_json())
        expected[mode] = {case_id: cases[case_id].estimated_steps for case_id in spec["case_ids"]}

    overlay = json.loads((base / "overlay.json").read_text())
    overlay["staged_dir"] = str(package)
    overlay["files"]["models/deepseek_v4/nvidia/model.py"]["source"] = str(package / "overlay_model.py")
    write(package / "overlay.json", overlay)
    base_config = json.loads((base / "config.json").read_text())
    native_hashes = dict(base_config["expected_source_hashes"])
    for name in spec["graph_replay_native_sources"]:
        content = subprocess.check_output(["git", "show", f"{spec['native_revision']}:{name}"],
                                          cwd=spec["native_repo"])
        native_hashes[name] = hashlib.sha256(content).hexdigest()
    tool_names = ("dsv4_graph_bridge_check.py", "profile_dsv4_prefill.py", "balanced_routing.py",
                  "overlay.json", "overlay_model.py", "native_dummy_bootstrap.py")
    tool_hashes = {str(package / name): sha(package / name) for name in tool_names}
    for directory in ("frontier", "data", "model", "plans"):
        tool_hashes.update({str(path): sha(path) for path in (package / directory).rglob("*")
                            if path.is_file()})
    profile = {key: base_config[key] for key in PROFILE_KEYS}
    profile.update(expected_source_hashes=native_hashes, script=str(package / "profile_dsv4_prefill.py"),
                   overlay_manifest=str(package / "overlay.json"), source_snapshot=str(package),
                   worker_wheel_site=spec["worker_wheel_site"], tool_hashes=tool_hashes)
    phase_root = f"{spec['worker_output_root']}/{spec['name']}"
    bridge = {
        "script": str(package / "dsv4_graph_bridge_check.py"),
        "overlay_manifest": str(package / "overlay.json"),
        "expected_source_hashes": native_hashes, "tool_hashes": tool_hashes, "bootstrap": profile,
        "plan_args": plan_args,
        "run_args": ["--model_dir", str(package / "model"),
                     "--expected_attention_backend", spec["expected_attention_backend"],
                     "--gpu_memory_utilization", str(spec["gpu_memory_utilization"]),
                     "--seed", str(spec["seed"])],
        "engine_index": spec["engine_index"], "case_ids": spec["case_ids"], "modes": spec["modes"],
        "cudagraph_capture_sizes": spec["cudagraph_capture_sizes"],
        "output_root": f"{phase_root}/bridge",
    }
    write(package / "bridge_config.json", bridge)
    bridge_config = str(package / "bridge_config.json")
    job = {key: spec[key] for key in ("job_tag", "output_dir", "image", "cuda_compat_path", "cloud_mount",
                                      "gpu_type", "charged_group", "gpu", "cpu", "mem_gb", "preemptible",
                                      "timeout_seconds", "planning_gpu_hours_cap", "source_revision")}
    job.update(name=spec["name"], source_snapshot=str(package), worker=str(package / "launch/glm_producer_worker.py"),
               wheelhouse=str(package / "launch/wheelhouse"), phases=[{"name": spec["name"], "steps": [
                   {"commands": [{"name": "bridge_run", "argv": [
                       "{python}", "-I", str(package / "native_dummy_bootstrap.py"), "--config", bridge_config]}]},
                   {"commands": [{"name": "bridge_report", "argv": [
                       "{python}", str(package / "dsv4_graph_bridge_check.py"), "--config", bridge_config,
                       "--report"]}]}]}])
    write(package / f"jobs/{spec['name']}.json", job)
    files = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob("*"))
             if path.is_file() and not path.name.endswith(".pyc")}
    write(package / "manifest.json", {
        "task_id": spec["task_id"], "source_revision": revision, "native_revision": spec["native_revision"],
        "base_manifest_sha256": spec["base_manifest_sha256"], "builder_sha256": sha(Path(__file__)),
        "package_plan_sha256": sha(HERE / "package_plan.json"), "files": files,
        "estimated_steps_by_mode_and_case": expected, "modes": spec["modes"],
        "cudagraph_capture_sizes": spec["cudagraph_capture_sizes"],
        "planning_gpu_hours_cap": spec["planning_gpu_hours_cap"],
        "timeout_gpu_hours": spec["gpu"] * spec["timeout_seconds"] / 3600})
    verify_package(base, spec["base_manifest_sha256"])
    print(json.dumps({"package": str(package), "source_revision": revision, "files": len(files),
                      "estimated_steps_by_mode_and_case": expected, "gpu_or_platform_operations": 0}))


if __name__ == "__main__":
    main()
