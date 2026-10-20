"""Prepare one S0 package locally, without constructing a GPU platform client."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from frontier.profiling.collectives.vllm_forward_collectives import (
    ModelShape, ServingParallelism, forward_collectives, scenarios_from_grid,
)

REPO = Path(__file__).resolve().parents[3]
CLOUD_ROOT = Path("/mnt/codesign-exp/ycfeng/frontier/step5_comm")
MODEL_CONFIG = "data/config/models/stepfun-ai__Step-5-Preview.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(definition: dict, stage: Path, run_id: str, *, repository: Path = REPO) -> dict:
    """Bind the existing benchmark to a one-node job and record offline row counts."""
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
    if resources["replica"] != 1 or resources["gpu"] != parallel.world_size:
        raise ValueError("S0 requires a single node and GPU count equal to TP x DP")
    if (parallel.tensor_parallel_size, parallel.data_parallel_size) != (4, 2):
        raise ValueError("S0's native dispatch probe requires TP4/DP2/EP8")
    if (resources["positive_tags"], resources["charged_group"]) != (["H800"], "codesign"):
        raise ValueError("S0 is H800/codesign only")
    model = ModelShape.from_hf_config(json.loads((repository / MODEL_CONFIG).read_text()))
    scenarios = scenarios_from_grid(parallel, definition["grid"])
    cases = sum(len(forward_collectives(model, parallel, scenario)) for scenario in scenarios)
    output = CLOUD_ROOT / run_id
    benchmark = {key: definition[key] for key in (
        "engine", "gpus_per_node", "warmup_calls", "repeats", "calls_per_repeat",
        "passes", "shuffle_seed", "grid")}
    benchmark.update(model_config=str(repository / MODEL_CONFIG), output_dir=str(output / "collectives"))
    stage.mkdir(parents=True, exist_ok=False)
    benchmark_path = stage / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark, indent=2) + "\n")
    model_path = stage / "model_metadata"
    model_path.mkdir()
    (model_path / "config.json").write_bytes((repository / MODEL_CONFIG).read_bytes())
    # Hash the existing producer and its native probe imports, not an independent
    # source snapshot or a copy of the image's vLLM package.
    names = subprocess.check_output([
        "git", "ls-files", "frontier/profiling", "frontier/config", "frontier/logger.py",
        "frontier/model_architectures.py",
        MODEL_CONFIG, "tests/integration/step5_comm"], cwd=repository, text=True).splitlines()
    hashes = {name: sha256(repository / name) for name in names if (repository / name).is_file()}
    job = {"description": definition["description"], "run_id": run_id,
           "repository": str(repository), "image": definition["image"],
           "resources": resources, "benchmark": str(benchmark_path),
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
