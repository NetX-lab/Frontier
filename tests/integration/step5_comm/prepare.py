"""Freeze a Step collective package without constructing a GPU platform client."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from frontier.profiling.collectives.vllm_forward_collectives import (
    ModelShape, ServingParallelism, forward_collectives, scenarios_from_grid,
)

REPO = Path(__file__).resolve().parents[3]
CLOUD_ROOT = Path("/mnt/codesign-exp/ycfeng/frontier/step5_comm")
MODEL_CONFIG = "data/config/models/stepfun-ai__Step-5-Preview.json"
HARDWARE_GROUPS = {"H800": "codesign", "H200": "step_main"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(definition: dict, stage: Path, run_id: str, *, repository: Path = REPO) -> dict:
    """Bind the existing benchmark to the declared placement and count its rows."""
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be one path component")
    repository, stage = repository.resolve(), stage.resolve()
    if not stage.is_relative_to(repository):
        raise ValueError("The package must be inside the locally mounted repository")
    status = subprocess.check_output(["git", "status", "--porcelain"],
                                     cwd=repository, text=True)
    if status.strip():
        raise ValueError("Commit the source before preparing a submission package")
    resources = definition["resources"]
    parallel = ServingParallelism.from_engine(definition["engine"])
    if (resources["gpu"] != definition["gpus_per_node"] or
            resources["gpu"] * resources["replica"] != parallel.world_size):
        raise ValueError("Node GPU counts and replicas must equal TP x DP")
    if definition["gpus_per_node"] % parallel.tensor_parallel_size:
        raise ValueError("Every TP group must fit within one node")
    tokens = definition["native_probe"]["tokens_per_dp_rank"]
    if len(tokens) != parallel.data_parallel_size or any(t <= 0 for t in tokens):
        raise ValueError("The native probe needs one positive physical token count per DP owner")
    if definition["native_probe"]["layers"] != [3, 88]:
        raise ValueError("The native probe observes exactly layers 3 and 88")
    if resources["replica"] > 1:
        ports = definition["rendezvous"]
        if not (0 < ports["master_port"] < 65536 and 0 < ports["probe_port"] < 65536
                and ports["master_port"] != ports["probe_port"]):
            raise ValueError("Benchmark and probe require distinct valid rendezvous ports")
    if (len(resources["positive_tags"]) != 1 or
            HARDWARE_GROUPS.get(resources["positive_tags"][0]) != resources["charged_group"]):
        raise ValueError("Step communication stages require a supported hardware/charged-group pair")
    model = ModelShape.from_hf_config(json.loads((repository / MODEL_CONFIG).read_text()))
    scenarios = scenarios_from_grid(parallel, definition["grid"])
    cases = sum(len(forward_collectives(model, parallel, scenario,
                native_call_names=definition["grid"].get("native_call_names"))) for scenario in scenarios)
    output = CLOUD_ROOT / run_id
    benchmark = {key: definition[key] for key in (
        "engine", "gpus_per_node", "warmup_calls", "repeats", "calls_per_repeat",
        "passes", "shuffle_seed", "grid")}
    benchmark.update(model_config=str(repository / MODEL_CONFIG), output_dir=str(output / "collectives"))
    if "frozen_model_config" in definition:
        frozen = (repository / definition["frozen_model_config"]).resolve()
        if not frozen.is_relative_to(repository):
            raise ValueError("Frozen model configuration must use the existing repository mount")
        benchmark.update(frozen_model_config=str(frozen),frozen_model_config_sha256=sha256(frozen))
    stage.mkdir(parents=True, exist_ok=False)
    benchmark_path = stage / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark, indent=2) + "\n")
    model_path = stage / "model_metadata"
    model_path.mkdir()
    (model_path / "config.json").write_bytes((repository / MODEL_CONFIG).read_bytes())
    # Hash the existing producer and its native probe imports, not an independent
    # source snapshot or a copy of the image's vLLM package.
    source_paths = [
        "frontier/profiling", "frontier/config", "frontier/logger.py",
        "frontier/model_architectures.py", MODEL_CONFIG, "tests/integration/step5_comm"]
    if "canonical_workload" in definition:
        source_paths.append(definition["canonical_workload"])
    names = subprocess.check_output([
        "git", "ls-files", *source_paths], cwd=repository, text=True).splitlines()
    hashes = {name: sha256(repository / name) for name in names if (repository / name).is_file()}
    job = {"description": definition["description"], "run_id": run_id,
           "repository": str(repository), "image": definition["image"],
            "resources": resources, "benchmark": str(benchmark_path),
            "engine": definition["engine"], "gpus_per_node": definition["gpus_per_node"],
            "rendezvous": definition.get("rendezvous", {}),
           "benchmark_sha256": sha256(benchmark_path), "output_dir": str(output),
           "native_probe": {**definition["native_probe"], "model_path": str(model_path),
                            "config_sha256": sha256(model_path / "config.json"),
                            "disable_custom_all_reduce": definition["engine"]["disable_custom_all_reduce"]},
           "source_hashes": hashes,
           "source_commit": subprocess.check_output([
               "git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
           "expected": {"scenarios": len(scenarios), "cases": cases,
                        "measurements": cases * definition["passes"],
                        "kernel_rows": cases,
                         "wall_minutes": definition["expected_wall_minutes"],
                         "gpu_hours": definition["expected_gpu_hours"]}}
    if "local_work" in definition["grid"]:
        job["expected"]["local_work_rows"] = len(definition["grid"]["local_work"]) * definition["passes"]
    launcher = stage / "submit_with_env.sh"
    script = ("#!/usr/bin/env bash\nset -euo pipefail\nset +x\n"
              f"source {shlex.quote(definition['credential_environment'])}\n"
              f"cd {shlex.quote(str(repository))}\n"
              f'exec "$STEPMIND_PYTHON" -u {shlex.quote(str(repository / "tests/integration/step5_comm/submit.py"))} '
              f"--config {shlex.quote(str(stage / 'job.json'))}\n")
    launcher.write_text(script)
    job["submission_script"] = str(launcher)
    job["submission_script_sha256"] = sha256(launcher)
    if "canonical_workload" in definition:
        job["canonical_workload"] = definition["canonical_workload"]
        job["coverage"] = definition["coverage"]
    (stage / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, default=Path(__file__).with_name("s0_run.json"))
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = prepare(json.loads(args.definition.read_text()), args.stage_dir, args.run_id)
    print(json.dumps({"job": str(args.stage_dir / "job.json"), **result["expected"]}, indent=2))
