"""Prepare a fresh local-MoE package; leave image gates, sealing and submission to root."""

import hashlib
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREPARE = HERE.parent / "balanced_decode/prepare_package.py"
UTILS = runpy.run_path(str(PREPARE))
sha, write, verify_package = (UTILS[name] for name in ("sha", "write", "verify_package"))


def snapshot(repo, revision, paths, destination):
    archive = subprocess.check_output(["git", "archive", revision, *paths], cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isfile():
                target = destination / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(item).read())


def main():
    spec = json.loads((HERE / "profile_plan.json").read_text())
    revision, native = spec["source_revision"], spec["native_revision"]
    source_paths = ("frontier", "data/config")
    for repo, ref, paths in ((ROOT, revision, source_paths),
                             (spec["native_repo"], native, spec["native_paths"])):
        if subprocess.check_output(["git", "diff", ref, "--", *paths], cwd=repo):
            raise RuntimeError(f"Source bytes differ from the pinned revision: {repo}")
    base = Path(spec["base_package"])
    original = verify_package(base, spec["base_manifest_sha256"])
    package = HERE / spec["destination"]
    if package.exists():
        raise RuntimeError("Fresh generation only; never alter an existing package")
    if not spec["output_dir"].startswith("/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/"):
        raise ValueError("The explicit cloud destination must remain inside the task boundary")
    package.mkdir()
    launch = package / "launch"
    launch.mkdir()
    copied = {}
    for name in ("glm_producer_jobs.py", "glm_producer_worker.py", "launch_glm_producers.sh"):
        source, target = base / "launch" / name, launch / name
        shutil.copy2(source, target)
        copied[name] = {"original_sha256": sha(source), "copied_sha256": sha(target)}
    launcher = launch / "glm_producer_jobs.py"
    text = launcher.read_text()
    locator = "WORKSPACE = LAUNCH_DIR.parents[3]"
    if text.count(locator) != 1:
        raise ValueError("The existing launcher workspace locator changed")
    launcher.write_text(text.replace(locator, "WORKSPACE = Path(" + repr(str(ROOT)) + ")"))
    copied[launcher.name].update(copied_sha256=sha(launcher), workspace_locator_only=True)
    shutil.copytree(base / "launch/wheelhouse", launch / "wheelhouse")
    source = package / "source" / revision
    snapshot(ROOT, revision, source_paths, source)
    native_source = package / "native_source" / native
    snapshot(spec["native_repo"], native, spec["native_paths"], native_source)
    native_hashes = {}
    for path in native_source.rglob("*"):
        if path.is_file():
            name = str(path.relative_to(native_source))
            if path.read_bytes() != (Path(spec["native_repo"]) / name).read_bytes():
                raise ValueError(f"Native source differs from committed bytes: {name}")
            native_hashes[name] = sha(path)
    for name, symbols in spec["native_symbols"].items():
        content = (native_source / name).read_text()
        if any(symbol not in content for symbol in symbols):
            raise ValueError(f"Required native MoE symbol missing: {name}")
    inputs = package / "inputs"
    inputs.mkdir()
    for path in (Path(__file__), HERE / "profile_plan.json", PREPARE):
        shutil.copy2(path, inputs / path.name)
    root_inputs = {}
    for name in ("requirements.md", "plan.md", "moe_grid_design.md"):
        path = Path(spec["root_task"]) / name
        shutil.copy2(path, inputs / ("root_" + name))
        root_inputs[str(path)] = sha(path)
    overlay = json.loads((base / "overlay.json").read_text())
    config = json.loads((base / "config.json").read_text())
    for name, expected in config["expected_source_hashes"].items():
        if name in native_hashes and native_hashes[name] != expected:
            raise ValueError(f"Inherited native image source pin mismatch: {name}")
    write(package / "runtime_lock.json", {
        "native_repo": spec["native_repo"], "native_revision": native,
        "image": spec["image"], "native_package": config["native_package"],
        "native_source_hashes": native_hashes, "native_symbols": spec["native_symbols"],
        "expected_source_hashes": {name: value for name, value in native_hashes.items()
                                   if name.startswith("vllm/")},
        "binary_sha256": overlay["binary_sha256"], "actual_image_checked": False,
        "binary_pin_origin": str(base / "overlay.json"), "binary_pin_origin_sha256": sha(base / "overlay.json"),
        "native_source_is_reference_not_worker_overlay": True})
    tools = runpy.run_path(str(launcher))
    worker = runpy.run_path(str(launch / "glm_producer_worker.py"))
    if spec["image"] != tools["IMAGE"]:
        raise ValueError("The requested image differs from the existing locked launcher image")
    expected, phases = {}, []
    for method, filename in spec["methods"].items():
        phase = spec["name"] + "_" + method.upper()
        local = str(worker["LOCAL_ROOT"] / "out" / phase)
        argv = ["{python}", "-m", "frontier.profiling.moe.main", *spec["common_args"],
                "--profile_method", method, "--source_revision", revision,
                "--model_config_revision", spec["model_config_revision"], "--output_dir", local]
        output = tools["_run_snapshot"](source, [*argv[1:], "--dry_run"])
        plan = tools["_json_from_output"](output)
        model = plan["models"][spec["model"]]
        if (sorted(plan["num_tokens"]) != [16, 32, 6152, 12288]
                or plan["cases_per_parallel_point"] != 1 or plan["load_distributions"] != ["balanced"]
                or model["tensor_parallel_sizes"] != [1] or model["num_experts"] != 256
                or model["num_experts_per_tok"] != 6 or model["profiled_steps"] != 4
                or model["local_experts_by_expert_parallel_size"] != {"16": 16}):
            raise ValueError("The existing producer did not plan the exact selected four-point grid")
        write(package / ("plans/" + method + ".json"), plan)
        relative = "compute/h800/" + spec["model"] + "/" + filename
        expected[method] = {"rows": 4, "phase": phase, "table": relative,
                            "cloud_table": spec["output_dir"] + "/" + phase + "/" + relative,
                            "trace_directory": "profiler_traces", "distribution_seed": 0}
        phases.append({"name": phase, "expected_tables": [{"path": relative, "rows": 4}],
                       "steps": [tools["_step"](tools["_command"](method, argv))]})
    job = {key: spec[key] for key in ("task_id", "name", "job_tag", "output_dir", "image", "gpu_type",
           "charged_group", "gpu", "cpu", "mem_gb", "preemptible", "timeout_seconds",
           "planning_gpu_hours_cap", "source_revision", "model_config_revision")}
    job.update(producer="moe", source_snapshot=str(source), worker=str(launch / "glm_producer_worker.py"),
               wheelhouse=str(launch / "wheelhouse"), cloud_mount=tools["CLOUD_MOUNT"],
               cuda_compat_path=tools["CUDA_COMPAT_PATH"], phases=phases, formal_coverage=False,
               physical_TP=1, local_EP_rank=0, expert_parallel_size=16,
               status="CPU_PREPARATION_ONLY_ROOT_IMAGE_SDK_SEAL_SUBMISSION_REQUIRED")
    write(package / "jobs/moe.json", job)
    write(package / "originals.json", {"package": str(base), "manifest_sha256": spec["base_manifest_sha256"],
          "input_count": len(original["files"]), "copied_runtime": copied, "root_inputs": root_inputs})
    files = {str(path.relative_to(package)): sha(path) for path in package.rglob("*") if path.is_file()}
    write(package / "manifest.json", {
        "task_id": spec["task_id"], "files": files, "source_revision": revision, "native_git": native,
        "builder_sha256": sha(Path(__file__)), "profile_plan_sha256": sha(HERE / "profile_plan.json"),
        "builder_dependency_sha256": sha(PREPARE), "expected_outputs": expected,
        "copied_runtime": copied, "routing_runtime_path": "balanced_topk", "load_distribution": "balanced",
        "selected_scope": spec["selected_scope"], "conditional_population": 6152,
        "retain_all_columns_and_traces": True, "qualification": spec["qualification"],
        "root_image_sdk_seal_required": True, "sealed": False, "submitted": False,
        "actual_image_checked": False, "actual_GPU_checked": False})
    verify_package(package, sha(package / "manifest.json"))
    verify_package(base, spec["base_manifest_sha256"])
    print(json.dumps({"package": str(package), "manifest_sha256": sha(package / "manifest.json"),
                      "input_count": len(files), "expected_outputs": expected, "GPU_platform_seal_operations": 0}))


if __name__ == "__main__":
    main()
