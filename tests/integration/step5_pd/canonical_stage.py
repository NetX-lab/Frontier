"""Build or verify an immutable native qualification/canonical package offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tests.integration.step5_pd import build_stage, canonical_replay, formal_release, submit_native_pd  # noqa: E402

CASES = json.loads(Path(__file__).with_name("priority_cases.json").read_text())
RELEASE_FILES = (
    "tests/comparison/e2e_gate/admission.py", "tests/comparison/e2e_gate/gate.py",
    "tests/comparison/e2e_gate/normalize.py", "tests/comparison/e2e_gate/dp_pp_modes.py",
    "tests/comparison/e2e_gate/cases/step5_priority.json",
    "tests/comparison/calibration/client_metrics.py", "tests/comparison/calibration/e2e_metrics_gap.py",
    "tests/integration/step5_pd/formal_release.py",
)


def case_definition(definition: dict, case_id: str) -> dict:
    case = CASES[case_id]
    definition = json.loads(json.dumps(definition))
    definition["case_id"] = case_id
    definition["resources"].update({key: case[key] for key in (
        "gpu_type", "charged_group", "replica", "gpu", "preemptible")})
    definition["engine"]["data_parallel_size"] = case["data_parallel_size"]
    definition["description"] = (
        f"D21 {case['priority']} {case_id}: symmetric native P/D, each TP8/"
        f"DP{case['data_parallel_size']}/EP{8 * case['data_parallel_size']}/PP1; frozen common workload.")
    return definition

RUNTIME_SOURCES = (
    "config/scheduler.py", "config/vllm.py", "config/compilation.py",
    "config/observability.py", "engine/arg_utils.py",
    "entrypoints/openai/completion/protocol.py", "entrypoints/openai/completion/serving.py",
    "v1/worker/gpu_worker.py", "v1/worker/gpu/model_runner.py",
    "v1/worker/gpu/dp_utils.py", "v1/worker/dp_utils.py", "v1/worker/gpu/cudagraph_utils.py",
    "v1/worker/gpu/async_utils.py", "v1/outputs.py", "v1/executor/multiproc_executor.py",
    "v1/core/sched/scheduler.py", "v1/engine/core.py", "v1/engine/__init__.py",
    "v1/engine/async_llm.py", "v1/metrics/stats.py", "v1/metrics/loggers.py",
    "compilation/cuda_graph.py", "envs.py",
    "models/step5/nvidia/model.py", "models/step5/nvidia/sparse_attention.py",
    "distributed/kv_transfer/kv_connector/v1/base.py",
)


def verify(stage: Path) -> dict:
    config = json.loads((stage / "config.json").read_text())
    for name, expected in config["tool_hashes"].items():
        if build_stage.sha256(stage / name) != expected:
            raise ValueError(f"Staged tool hash mismatch: {name}")
    if set(config["model_hashes"]) != {"config.json", "tokenizer_config.json", "tokenizer.json"}:
        raise ValueError("Unexpected model metadata member")
    for name, expected in config["model_hashes"].items():
        if build_stage.sha256(Path(config["model_dir"]) / name) != expected:
            raise ValueError(f"Staged metadata hash mismatch: {name}")
    canonical_replay.canonical_workload(stage / "trace/workload")
    expected_engine = json.loads((stage / "engine.json").read_text())
    if expected_engine != config["engine"]:
        raise ValueError("Staged engine differs from worker configuration")
    plan = json.loads((stage / "trace/replay_plan.json").read_text())
    definition = json.loads((stage / "run_definition.json").read_text())
    if (config.get("canonical_replay") != definition["canonical_replay"]
            or plan["kind"] != config["canonical_replay"]["kind"]
            or plan["dp_size"] != expected_engine["data_parallel_size"]):
        raise ValueError("Staged replay kind/topology differs from the frozen run definition")
    case = CASES[config["case_id"]]
    if any(config.get(key) != case[key] for key in (
            "gpu_type", "charged_group", "replica", "gpu", "preemptible")):
        raise ValueError("D21 hardware, topology or preemptible configuration mismatch")
    dp = case["data_parallel_size"]
    if ("pd_state_observation" in config or "serve_observers" in config
            or "role_num_gpu_blocks_override" in config or (stage / "r2_bootstrap").exists()):
        raise ValueError("Clean serving contains diagnostic hooks or block limits")
    for key, expected in {"max_model_len": 7168, "max_num_batched_tokens": 8192, "max_num_seqs": 128,
                          "tensor_parallel_size": 8, "data_parallel_size": dp, "pipeline_parallel_size": 1,
                          "enable_expert_parallel": True, "enforce_eager": False,
                          "cudagraph_metrics": True,
                          "enable_chunked_prefill": False, "enable_prefix_caching": False}.items():
        if expected_engine.get(key) != expected:
            raise ValueError(f"D18 engine mismatch: {key}")
    model = json.loads((Path(config["model_dir"]) / "config.json").read_text())
    if model.get("quantization_config", {}).get("quant_method") != "fp8":
        raise ValueError("Native FP8 metadata missing")
    if expected_engine.get("compilation_config") != {
            "cudagraph_mode": "FULL_AND_PIECEWISE", "max_cudagraph_capture_size": 256}:
        raise ValueError("Unexpected native graph policy")
    rows = build_stage.preflight(config, Path(config["model_dir"]))
    if rows != json.loads((stage / "preflight.json").read_text()):
        raise ValueError("Preflight does not reproduce the staged role argv/environment")
    expected_lanes = [(role, lane) for role in ("prefill", "decode") for lane in range(dp)]
    if [(row["role"], row["lane"]) for row in rows] != expected_lanes:
        raise ValueError("Missing native DP lane")
    for row in rows:
        command = row["server_command"]
        if "--speculative-config" in command or any("PD_STATE" in key for key in row["server_env_set"]):
            raise ValueError("Speculation or R2 observation enabled")
    release = formal_release.validate(config)
    return {"task_id": "D21-RUNS", "case_id": config["case_id"], "offline_verified": True,
            "stage": str(stage), "source": config["source"], "files_verified": len(config["tool_hashes"]),
            "manifest_sha256": canonical_replay.MANIFEST_SHA256,
            "config_sha256": build_stage.sha256(stage / "config.json"),
             "runtime_qualification": "UNVERIFIED", "reference_admission": "UNVERIFIED",
             "formal_release": release,
            "resources": submit_native_pd.resource_arguments(config),
            "worker_command": submit_native_pd.job_command(config, stage / "config.json")}


def build(args) -> dict:
    source = build_stage.committed_source()
    definition = case_definition(json.loads(args.run_definition.read_text()), args.case)
    definition["canonical_replay"]["kind"] = args.kind
    if args.kind == "qualification":
        definition["description"] += " Qualification only: 16 warmups then eight concurrent completions; no formal reference admission."
        definition["resources"]["timeout_seconds"] = 3600
        definition["replay_timeout_s"] = 1200
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    model = workspace / "reference_inputs"
    if not model.exists():
        shutil.copytree(args.model_dir, model)
    elif ({path.name: build_stage.sha256(path) for path in model.iterdir()}
          != {path.name: build_stage.sha256(path) for path in args.model_dir.iterdir()}):
        raise ValueError("Existing staged reference metadata differs; never overwrite it")
    config = build_stage.build(definition, stage=args.stage_dir.resolve(), workspace=workspace,
        output_dir=args.output_dir, model_dir=model, image_source_hashes=args.image_source_hashes,
        probe_dir=args.probe_dir, job_tag=args.job_tag, source=source)
    stage = args.stage_dir.resolve()
    bindings = {}
    for relative in RUNTIME_SOURCES:
        key = "vllm/" + relative
        digest = build_stage.sha256(args.runtime_source_root / key)
        if key in config["expected_source_hashes"] and config["expected_source_hashes"][key] != digest:
            raise ValueError(f"Exported native source conflicts with prior locked pin: {key}")
        config["expected_source_hashes"][key] = digest
        bindings[key] = digest
    (stage / "runtime_source_bindings.json").write_text(json.dumps({
        "exported_source_root": str(args.runtime_source_root), "sha256": bindings,
        "expected_runner": "MRv2; conditional on effective native environment and Triton availability",
        "graph_metrics": "MRv2 source path verified; FULL rows plus healthy completion required, with explicit lane/rank attribution",
        "installed_identity": "Worker must confirm these exported bytes in the locked image before serving"}, indent=1))
    (stage / "run_definition.json").write_text(json.dumps(definition, indent=1))
    for relative in RELEASE_FILES:
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
    if args.formal_inputs:
        config["formal_inputs"] = json.loads(args.formal_inputs.read_text())
        if "accuracy" in config["formal_inputs"]:
            accuracy = config["formal_inputs"]["accuracy"]
            [truth] = (Path(accuracy["frontier_run_directory"]) / "metrics").rglob("metrics_ground_truth.jsonl")
            accuracy["frontier_artifacts"] = [formal_release.pin(path) for path in (truth, truth.with_name("config.json"))]
        (stage / "formal_inputs.json").write_text(json.dumps(config["formal_inputs"], indent=1))
    formal_release.validate(config)
    for name in ("build_stage.py", "canonical_stage.py", "submit_native_pd.py", "priority_cases.json"):
        shutil.copy2(Path(__file__).with_name(name), stage / "tests/integration/step5_pd" / name)
    shutil.copy2(args.launcher_scope, stage / "launcher_scope.py")
    (stage / "submit.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"python3 {stage}/tests/integration/step5_pd/canonical_stage.py verify --stage-dir {stage}\n"
        "source /data/ycfeng/tmp/step5_native_pd_r1/stepmind_env.sh\n"
        "export STEPMIND_BACKEND=rjob\n"
        f'exec "$STEPMIND_PYTHON" {stage}/tests/integration/step5_pd/submit_native_pd.py '
        f"--config {stage}/config.json --launcher-scope {stage}/launcher_scope.py\n")
    config["tool_hashes"] = {str(path.relative_to(stage)): build_stage.sha256(path)
                             for path in sorted(stage.rglob("*")) if path.is_file() and path.name != "config.json"}
    (stage / "config.json").write_text(json.dumps(config, indent=1))
    return verify(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--stage-dir", type=Path, required=True)
    create = sub.add_parser("build")
    for name in ("run-definition", "stage-dir", "workspace", "output-dir", "model-dir",
                 "image-source-hashes", "probe-dir", "runtime-source-root", "launcher-scope"):
        create.add_argument("--" + name, type=Path, required=True)
    create.add_argument("--job-tag", required=True)
    create.add_argument("--case", choices=tuple(CASES), required=True)
    create.add_argument("--formal-inputs", type=Path, help="Explicit readiness and (for64) same-hardware32 accuracy artifact inputs")
    create.add_argument("--kind", choices=("qualification", "canonical"), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.stage_dir.resolve()) if args.action == "verify" else build(args), indent=1))


if __name__ == "__main__":
    main()
