"""Stage a fresh balanced q1 package using the accepted producer and bootstrap."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_package(path, digest):
    if sha(path / "manifest.json") != digest:
        raise ValueError(f"Original manifest mismatch: {path}")
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(path / name) != expected:
            raise ValueError(f"Original input mismatch: {path / name}")
    return manifest


def main():
    spec = json.loads((HERE / "package_plan.json").read_text())
    revision, native = spec["source_revision"], spec["native_revision"]
    source_paths = ("frontier", "data/config", "tests/comparison/calibration/profile_dsv4_prefill.py")
    changed = subprocess.check_output(["git", "diff", revision, "--", *source_paths], cwd=ROOT)
    if changed:
        raise RuntimeError("Accepted source paths differ from the pinned revision")
    base, prefill = Path(spec["base_package"]), Path(spec["balanced_prefill_package"])
    original = verify_package(base, spec["base_manifest_sha256"])
    verify_package(prefill, spec["balanced_prefill_manifest_sha256"])
    package = HERE / spec["destination"]
    if package.exists():
        raise RuntimeError("Fresh generation only; never alter an existing package")
    package.mkdir()
    launch = package / "launch"
    launch.mkdir()
    copied = {}
    names = ("launch/glm_producer_jobs.py", "launch/glm_producer_worker.py",
             "launch/launch_glm_producers.sh", "native_dummy_bootstrap.py", "overlay_model.py",
             "official_model.py", "verify_bootstrap.py")
    for name in names:
        shutil.copy2(base / name, package / name)
        copied[name] = {"base_sha256": sha(base / name), "copied_sha256": sha(package / name)}
    launcher = launch / "glm_producer_jobs.py"
    launcher_source = launcher.read_text()
    locator = "WORKSPACE = LAUNCH_DIR.parents[3]"
    if launcher_source.count(locator) != 1:
        raise ValueError("Accepted launcher workspace locator changed")
    launcher.write_text(launcher_source.replace(locator, "WORKSPACE = Path(" + repr(str(ROOT)) + ")"))
    copied["launch/glm_producer_jobs.py"].update(copied_sha256=sha(launcher), workspace_locator_only=True)
    shutil.copytree(base / "launch/wheelhouse", launch / "wheelhouse")
    for source, target in ((source_paths[2], "profile_dsv4_prefill.py"),
                           ("frontier/profiling/moe/balanced_routing.py", "balanced_routing.py")):
        (package / target).write_bytes(subprocess.check_output(["git", "show", revision + ":" + source], cwd=ROOT))
    snapshot = package / "source" / revision
    snapshot.mkdir(parents=True)
    archive = subprocess.check_output(["git", "archive", revision, "frontier", "data/config"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isfile():
                target = snapshot / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(item).read())
    shutil.copytree(base / "model", package / "model")
    sys.path.insert(0, str(snapshot))
    from frontier.profiling.attention.dsv4_attention_main import build_parser, build_plan
    from frontier.profiling.attention.dsv4_attention_plan import expected_scopes
    from frontier.profiling.attention.dsv4_attention_rows import BALANCED_ROUTING_SHA256
    from frontier.profiling.attention.dsv4_attention_runner import ROUTING_NATIVE_MODULES

    common = [*spec["common_args"], "--model_config_path", str(package / "model/config.json")]
    plan = build_plan(build_parser().parse_args(["plan", *common]))
    case, = plan.cases
    if case.case_id != spec["case_id"] or case.phase != "decode" or case.batch_size != 1:
        raise ValueError("The package must contain the existing q1 decode case")
    contexts = range(case.prompt_len + case.warmup, case.prompt_len + case.warmup + case.measured)
    expected = {"steps": {}, "calls": {}, "rows": {}}
    for method in ("cuda_event", "kernel_only"):
        expected["steps"][method] = case.measured
        expected["rows"][method] = len({(layer.attention_variant, layer.plans_tile_schedule)
                                       for layer in plan.shape.layers}) * case.measured
        expected["calls"][method] = sum(len(expected_scopes(layer.attention_variant,
            plan.shape.index_scores_executed(context + 1), annotated_only=method == "kernel_only"))
            for context in contexts for layer in plan.shape.layers)
    write(package / "plan.json", plan.to_json())
    local = "/tmp/glm_producers/out/" + spec["name"]
    run = ["run", *common, "--engine_index", "0", "--model_dir", str(package / "model"),
           "--raw_dir", local + "/raw", "--trace_dir", local + "/profiler_traces", "--keep_traces",
           "--case_ids", case.case_id, "--expected_attention_backend", "FLASHMLA_SPARSE_DSV4",
           "--gpu_memory_utilization", str(spec["gpu_memory_utilization"]), "--seed", str(spec["seed"])]
    overlay = json.loads((base / "overlay.json").read_text())
    overlay["staged_dir"] = str(package)
    overlay["files"]["models/deepseek_v4/nvidia/model.py"]["source"] = str(package / "overlay_model.py")
    write(package / "overlay.json", overlay)
    config = json.loads((prefill / "config.json").read_text())
    config.update(script=str(package / "profile_dsv4_prefill.py"), overlay_manifest=str(package / "overlay.json"),
                  source_snapshot=str(snapshot), profile_args=run, worker_wheel_site="/tmp/glm_producers/site")
    if sha(package / "balanced_routing.py") != BALANCED_ROUTING_SHA256:
        raise ValueError("Accepted balanced routing strategy mismatch")
    for module in ROUTING_NATIVE_MODULES:
        name = module.replace(".", "/") + ".py"
        content = subprocess.check_output(["git", "show", native + ":" + name], cwd=spec["native_repo"])
        if config["expected_source_hashes"][name] != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Accepted native routing pin mismatch: {name}")
    config["tool_hashes"] = {str(package / name): sha(package / name) for name in
                            ("profile_dsv4_prefill.py", "balanced_routing.py", "overlay.json", "overlay_model.py",
                             "native_dummy_bootstrap.py", "plan.json")}
    for directory in (snapshot, package / "model"):
        config["tool_hashes"].update({str(path): sha(path) for path in directory.rglob("*") if path.is_file()})
    write(package / "config.json", config)
    aggregate = ["{python}", "-m", "frontier.profiling.attention.dsv4_attention_main", "aggregate", *common,
                 "--raw_dir", local + "/raw", "--output_root", local, "--device", "h800",
                 "--source_revision", revision, "--model_config_revision", spec["model_config_revision"]]
    job = json.loads((base / "jobs/Q1_WARM_6143_7166.json").read_text())
    job.pop("expected_container_seconds_range")
    job.update({key: spec[key] for key in ("name", "job_tag", "output_dir", "image", "gpu_type", "charged_group",
                "gpu", "cpu", "mem_gb", "preemptible", "aggregate_H800_ceiling", "timeout_seconds",
                "planning_gpu_hours_cap", "prior_container_seconds", "source_revision", "model_config_revision")})
    job.update(source_snapshot=str(snapshot), worker=str(launch / "glm_producer_worker.py"),
               wheelhouse=str(launch / "wheelhouse"), effective_engine_max_num_seqs=plan.engines[0].max_num_seqs,
               timeout_cap_requires_primary_acceptance=False, prior_runtime_is_reference_not_prediction=True,
               status="PREPARED_CPU_CHECK_ONLY_ROOT_IMAGE_SEAL_QUOTA_SUBMIT_REQUIRED", formal_coverage=False)
    job["phases"] = [{"name": spec["name"], "expected_tables": [
        {"path": "compute/h800/deepseek-ai/DeepSeek-V4-Flash/" + filename, "rows": expected["rows"][method]}
        for filename, method in (("attention.csv", "cuda_event"), ("attention_kernel_only.csv", "kernel_only"))],
        "steps": [{"commands": [{"name": "accepted_c26_q1_balanced", "argv": ["env", "VLLM_USE_V2_MODEL_RUNNER=1",
            "{python}", "-I", str(package / "native_dummy_bootstrap.py"), "--config", str(package / "config.json")]}]},
        {"commands": [{"name": "aggregate_q1_balanced", "argv": aggregate}]}]}]
    write(package / ("jobs/" + spec["name"] + ".json"), job)
    files = {str(path.relative_to(package)): sha(path) for path in package.rglob("*")
             if path.is_file() and not path.name.endswith(".pyc")}
    write(package / "manifest.json", {
        "task_id": spec["task_id"], "source_revision": revision, "native_git": native, "files": files,
        "prior_manifest_sha256": spec["base_manifest_sha256"],
        "balanced_prefill_manifest_sha256": spec["balanced_prefill_manifest_sha256"],
        "builder_sha256": sha(Path(__file__)), "package_plan_sha256": sha(HERE / "package_plan.json"),
        "copied_runtime_files": copied, "native_C26_sha256": sha(package / "overlay_model.py"),
        "routing_runtime_path": "balanced_topk", "routing_strategy_sha256": BALANCED_ROUTING_SHA256,
        "expected_outputs": expected, "expected_rows": expected["rows"], "case_id": case.case_id,
        "native_steps_per_method_including_seed_warmup": case.estimated_steps,
        "unmeasured_decode_warmups_per_method": case.warmup, "local_seed_prompt_tokens": case.prompt_len,
        "measured_context_min": contexts.start, "measured_context_max": contexts.stop - 1,
        "physical_engine": job["physical_engine"], "planner_max_num_seqs": plan.limits.max_num_seqs,
        "effective_engine_max_num_seqs": plan.engines[0].max_num_seqs,
        "cli_chunk_budget": plan.limits.chunk_budgets[0], "planned_LLM_max_num_batched_tokens": plan.engines[0].chunk_budget,
        "actual_native_max_num_batched_tokens": None, "formal_coverage": False,
        "physical_TP_parity_verified": False, "actual_H800_admission_verified": False,
        "planning_gpu_hours_cap": spec["planning_gpu_hours_cap"], "prior_container_seconds": spec["prior_container_seconds"],
        "prior_runtime_is_reference_not_prediction": True, "root_seal_required": True})
    if any(row["base_sha256"] != row["copied_sha256"] for name, row in copied.items()
           if name != "launch/glm_producer_jobs.py"):
        raise ValueError("Copied accepted runtime bytes changed")
    verify_package(base, spec["base_manifest_sha256"])
    verify_package(prefill, spec["balanced_prefill_manifest_sha256"])
    print(json.dumps({"package": str(package), "source_revision": revision, "input_hashes": len(files),
                      "prior_input_hashes": len(original["files"]), "expected_outputs": expected,
                      "gpu_or_platform_operations": 0}))


if __name__ == "__main__":
    main()
