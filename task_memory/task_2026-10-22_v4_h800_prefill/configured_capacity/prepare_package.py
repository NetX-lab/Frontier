"""Prepare exact warm-prefill with the retained builder and finalized source."""

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[1]
PACKAGE = HERE / "h800_prefill_capacity128"
BASE = Path("/data/ycfeng/Frontier/.worktrees/v4-q1-warm-remeasure/task_memory/task_2026-10-21_v4_q1_warm_remeasure/h800_decode")
REVISION = "9b8494c770f0b61fd728558f07ce34cdcb515295"
NATIVE = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
NAME = "PREFILL_WARM_6144_C0_CAP128"
LOCAL = "/tmp/glm_producers/out/" + NAME
CASE = "e0-prefill-p6144"
PRIOR_MANIFEST_SHA256 = "7929afaef9deced67543262ad9a3eec398048b68914fd0322b00c97416cbe360"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != REVISION:
        raise RuntimeError("Prepare only from the finalized committed prefill source")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Tracked source must be clean before package generation")
    if PACKAGE.exists():
        raise RuntimeError("Fresh generation only; never alter an existing package")
    assert sha(BASE / "manifest.json") == PRIOR_MANIFEST_SHA256
    PACKAGE.mkdir()
    launch = PACKAGE / "launch"
    launch.mkdir()
    copied = {}
    names = ("launch/glm_producer_jobs.py", "launch/glm_producer_worker.py",
              "launch/launch_glm_producers.sh", "native_dummy_bootstrap.py", "overlay_model.py",
              "official_model.py", "verify_bootstrap.py")
    for name in names:
        shutil.copyfile(BASE / name, PACKAGE / name)
        copied[name] = {"base_sha256": sha(BASE / name), "copied_sha256": sha(PACKAGE / name)}
    shutil.copytree(BASE / "launch/wheelhouse", launch / "wheelhouse")
    for source, target in (("tests/comparison/calibration/profile_dsv4_prefill.py", "profile_dsv4_prefill.py"),
                           ("frontier/profiling/moe/balanced_routing.py", "balanced_routing.py")):
        (PACKAGE / target).write_bytes(subprocess.check_output(["git", "show", REVISION + ":" + source], cwd=ROOT))
    snapshot = PACKAGE / "source" / REVISION
    snapshot.mkdir(parents=True)
    archive = subprocess.check_output(["git", "archive", REVISION, "frontier", "data/config"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isfile():
                target = snapshot / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(item).read())
    model = PACKAGE / "model"
    shutil.copytree(BASE / "model", model)
    sys.path.insert(0, str(snapshot))
    from frontier.profiling.attention.dsv4_attention_main import build_parser, build_plan

    common = ["--model_name", "deepseek-ai/DeepSeek-V4-Flash", "--model_config_path", str(model / "config.json"),
              "--target_tensor_parallel_size", "8", "--num_layers", "6", "--block_size", "256",
              "--kv_cache_dtype", "fp8", "--max_context_len", "7168", "--max_kv_tokens_per_lane", "7168",
              "--max_num_seqs", "128", "--chunk_budgets", "8192", "--prefill_prompt_len", "6144",
              "--prefill_warmup_repeats", "2", "--prefill_measured_repeats", "3"]
    plan = build_plan(build_parser().parse_args(["plan", *common]))
    case, = plan.cases
    assert (case.case_id, case.phase, case.prompt_len, case.max_tokens, case.warmup, case.measured) == (
        CASE, "prefill", 6144, 1, 2, 3)
    assert (plan.engines[0].max_model_len, plan.engines[0].chunk_budget,
            plan.limits.max_num_seqs, plan.engines[0].max_num_seqs) == (7168, 8192, 128, 128)
    write(PACKAGE / "plan.json", plan.to_json())
    run = ["run", *common, "--engine_index", "0", "--model_dir", str(model), "--raw_dir", LOCAL + "/raw",
           "--trace_dir", LOCAL + "/profiler_traces", "--keep_traces", "--case_ids", CASE,
           "--expected_attention_backend", "FLASHMLA_SPARSE_DSV4", "--gpu_memory_utilization", "0.85", "--seed", "1234"]
    overlay = json.loads((BASE / "overlay.json").read_text())
    overlay["staged_dir"] = str(PACKAGE)
    overlay["files"]["models/deepseek_v4/nvidia/model.py"]["source"] = str(PACKAGE / "overlay_model.py")
    write(PACKAGE / "overlay.json", overlay)
    old = json.loads((BASE / "config.json").read_text())
    config = {"script": str(PACKAGE / "profile_dsv4_prefill.py"), "overlay_manifest": str(PACKAGE / "overlay.json"),
              "native_package": overlay["installed_package"], "source_snapshot": str(snapshot), "profile_args": run,
              "worker_wheel_site": "/tmp/glm_producers/site", "expected_source_hashes": old["expected_source_hashes"],
              "moe_gating_routing_runtime_path": "balanced_topk",
              "tool_hashes": {str(PACKAGE / name): sha(PACKAGE / name) for name in
                              ("profile_dsv4_prefill.py", "balanced_routing.py", "overlay.json", "overlay_model.py",
                               "native_dummy_bootstrap.py", "plan.json")}}
    from frontier.profiling.attention.dsv4_attention_rows import BALANCED_ROUTING_SHA256
    from frontier.profiling.attention.dsv4_attention_runner import ROUTING_NATIVE_MODULES

    assert sha(PACKAGE / "balanced_routing.py") == BALANCED_ROUTING_SHA256
    native_root = Path("/data/ycfeng/tmp/opencode/vllm-v0.28.0")
    for module in ROUTING_NATIVE_MODULES:
        name = module.replace(".", "/") + ".py"
        content = subprocess.check_output(["git", "show", NATIVE + ":" + name], cwd=native_root)
        config["expected_source_hashes"][name] = hashlib.sha256(content).hexdigest()
    for directory in (snapshot, model):
        config["tool_hashes"].update({str(path): sha(path) for path in directory.rglob("*") if path.is_file()})
    write(PACKAGE / "config.json", config)
    aggregate = ["{python}", "-m", "frontier.profiling.attention.dsv4_attention_main", "aggregate", *common,
                 "--raw_dir", LOCAL + "/raw", "--output_root", LOCAL, "--device", "h800", "--source_revision", REVISION,
                 "--model_config_revision", "0d992cbeef1f2d4bd92a8ae67a97a43df0c1bce5"]
    job = json.loads((BASE / "jobs/Q1_WARM_6143_7166.json").read_text())
    job.pop("expected_container_seconds_range")
    job.update(name=NAME, gpu_type="H800", charged_group="codesign", gpu=1, preemptible=False,
               aggregate_H800_ceiling=128, source_revision=REVISION, source_snapshot=str(snapshot),
               worker=str(launch / "glm_producer_worker.py"), wheelhouse=str(launch / "wheelhouse"),
               job_tag="v4-h800-prefill-warm-6144-c0-cap128-20261022-9b8494c",
               output_dir="/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h800_v4_prefill/9b8494c_capacity128_warm2/" + NAME,
               status="PREPARED_CHECK_ONLY_ROOT_SOURCE_IMAGE_QUOTA_SUBMIT_REQUIRED", timeout_seconds=900,
               prior_container_seconds=664.0, reference_container_seconds_is_not_prefill_prediction=True,
               timeout_cap_requires_primary_acceptance=False, planning_gpu_hours_cap=0.25,
               effective_engine_max_num_seqs=plan.engines[0].max_num_seqs,
               capacity_input_is_fixture_lower_bound_not_H800_capacity=True, formal_coverage=False)
    job["phases"] = [{"name": NAME, "expected_tables": [
        {"path": "compute/h800/deepseek-ai/DeepSeek-V4-Flash/attention.csv", "rows": 6},
        {"path": "compute/h800/deepseek-ai/DeepSeek-V4-Flash/attention_kernel_only.csv", "rows": 6}], "steps": [
        {"commands": [{"name": "accepted_c26_prefill", "argv": ["env", "VLLM_USE_V2_MODEL_RUNNER=1", "{python}", "-I",
            str(PACKAGE / "native_dummy_bootstrap.py"), "--config", str(PACKAGE / "config.json")]}]},
        {"commands": [{"name": "aggregate_prefill", "argv": aggregate}]}]}]
    write(PACKAGE / ("jobs/" + NAME + ".json"), job)
    files = {str(path.relative_to(PACKAGE)): sha(path) for path in PACKAGE.rglob("*")
             if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".pyc")}
    write(PACKAGE / "manifest.json", {
        "task_id": "W128-V4-PREFILL-CONFIGURED-CAPACITY", "source_revision": REVISION, "native_git": NATIVE,
        "provenance_field": "source_revision", "prior_manifest_sha256": PRIOR_MANIFEST_SHA256,
        "native_C26_sha256": sha(PACKAGE / "overlay_model.py"), "files": files, "copied_runtime_files": copied,
        "expected_rows": {"cuda_event": 6, "kernel_only": 6}, "formal_coverage": False,
        "prefill_prompt_len": 6144, "cached_context": 0, "native_steps_per_method": 5,
        "unmeasured_prefill_warmups_per_method": 2, "measured_prefill_repeats_per_method": 3,
        "physical_engine": job["physical_engine"], "planner_max_num_seqs": 128,
        "effective_engine_max_num_seqs": plan.engines[0].max_num_seqs,
        "prior_container_seconds": 664.0, "planning_gpu_hours_cap": 0.25,
        "physical_TP_parity_verified": False, "actual_H800_admission_verified": False,
        "H200_measurements_used_as_H800_evidence": False,
        "timeout_reservation_seconds_not_measured_runtime": 900, "archive_budget_GiB_not_measured_storage": 10,
        "max_kv_tokens_per_lane_is_fixture_math_not_device_capacity": 7168,
        "reduced_effective_geometry_verified": False})
    assert all(row["base_sha256"] == row["copied_sha256"] for row in copied.values())
    assert sha(BASE / "manifest.json") == PRIOR_MANIFEST_SHA256
    print(json.dumps({"job": str(PACKAGE / ("jobs/" + NAME + ".json")), "source_revision": REVISION,
                      "input_hashes": len(files), "expected_rows": 12, "gpu_or_platform_operations": 0}))


if __name__ == "__main__":
    main()
