"""Write and submit the GLM-5.3 H800 profiling producer jobs.

``write`` exports the committed tree at HEAD into ``source/<revision>/`` and
writes one job config per producer into ``runs/``. Workers run that snapshot,
so the recorded source revision names exactly the code that ran, however the
shared worktree moves while a job waits in the queue. ``write`` also runs the
GPU-free plan and dry runs from the snapshot and stores them next to the
configs. Run it with the simulator environment's Python.

``submit --config`` sends one job config through the StepMind Python
``RJobBackend`` and blocks until the job ends. Keep that process alive for the
whole job: its exit handler can stop the worker. Run it through
``launch_glm_producers.sh``, which sets up the personal credentials.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

LAUNCH_DIR = Path(__file__).resolve().parent
WORKSPACE = Path('/data/ycfeng/Frontier/.worktrees/v4-moe-local-run')
TASK_DIR = LAUNCH_DIR.parents[1]

MODEL = "zai-org/GLM-5.3"
MODEL_CONFIG = "data/config/models/zai-org__GLM-5.3.json"
# Hugging Face revision of the checked-in model config.
MODEL_CONFIG_REVISION = "082a2aa294ad5f4056895c42b297b4af24703384"
MODEL_DIR = TASK_DIR / "runtime_lock/model_metadata/zai-org__GLM-5.3" / MODEL_CONFIG_REVISION / "model"
SNAPSHOT_PATHS = ("frontier", "data/config")

# Image, CUDA compatibility path, mount, and resources of the live G0B runs
# (runtime_lock/observed/g0b_glm53_05/config.json).
IMAGE = (
    "hub.i.basemind.com/vllm-openai/vllm-openai@sha256:"
    "2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
)
CUDA_COMPAT_PATH = "/usr/local/cuda-13.0/compat"
CLOUD_MOUNT = "juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp"
CLOUD_ROOT = "/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/w4_producers"
GPU_TYPE = "H800"
CHARGED_GROUP = "codesign"
GPUS, CPUS, MEM_GB = 8, 64, 512

WORKER_OUT = "/tmp/glm_producers/out"
PROFILE_METHODS = ("cuda_event", "record_function")

# Target limits; provenance in w4_glm_producers_20261020/notes.md.
DSA_PLAN_ARGS = [
    "--model_name", MODEL, "--model_config_path", MODEL_CONFIG,
    "--target_tensor_parallel_size", "8", "--block_size", "64",
    "--max_context_len", "33024", "--max_kv_tokens_per_lane", "205632",
    "--max_num_seqs", "1024",
    "--chunk_budgets", "256", "512", "1024", "2048", "4096", "8192",
]
DSA_RUN_ARGS = [
    "--model_dir", str(MODEL_DIR),
    "--expected_attention_backend", "FLASH_ATTN_MLA_SPARSE",
    "--gpu_memory_utilization", "0.85", "--seed", "1234",
]
LINEAR_ARGS = [
    "--disable_ray", "--device", "h800", "--models", MODEL,
    "--num_tensor_parallel_workers", "1", "8",
    "--sequence_parallel_moe_topology", "8", "4", "32",
    "--max_tokens", "8192",
]
# One EP rank receives the tokens of every DP lane in its EP group: 4 lanes x
# the 8192-token chunk budget.
MOE_ARGS = [
    "--disable_ray", "--device", "h800", "--models", MODEL,
    "--num_tensor_parallel_workers", "1", "--max_tokens", "32768",
]
MOE_EXPERT_PARALLEL_SIZES = ["8", "16", "32"]
SMOKE_NUM_TOKENS = ["1", "256"]
TIMEOUT_SECONDS = {"dsa": 3 * 3600, "linear_op": 3 * 3600, "moe": 4 * 3600}


def _command(name: str, argv: list[str], gpu: str | None = None) -> dict:
    command = {"name": name, "argv": argv}
    if gpu is not None:
        command["cuda_visible_devices"] = gpu
    return command


def _step(*commands: dict) -> dict:
    return {"commands": list(commands)}


def _revision_args(revision: str) -> list[str]:
    return ["--source_revision", revision, "--model_config_revision", MODEL_CONFIG_REVISION]


def _dsa_run(engine: int, out: str, gpu: str, case_ids: list[str] | None = None) -> dict:
    argv = ["{python}", "-m", "frontier.profiling.attention.dsa_main", "run", *DSA_PLAN_ARGS,
            *DSA_RUN_ARGS, "--engine_index", str(engine), "--raw_dir", f"{out}/raw",
            "--trace_dir", f"{out}/traces/e{engine}"]
    if case_ids:
        argv += ["--case_ids", *case_ids]
    return _command(f"dsa_e{engine}", argv, gpu)


def _dsa_aggregate(out: str, revision: str, partial: bool) -> dict:
    argv = ["{python}", "-m", "frontier.profiling.attention.dsa_main", "aggregate",
            *DSA_PLAN_ARGS, "--raw_dir", f"{out}/raw", "--output_root", out,
            "--device", "h800", *_revision_args(revision)]
    if partial:
        argv.append("--allow_partial_engines")
    return _command("dsa_aggregate", argv)


def _dsa_smoke_cases(plan: dict) -> tuple[int, list[str]]:
    """Return the decode engine with its shortest prefill and smallest decode case."""

    decode_cases = [case for case in plan["cases"] if case["phase"] == "decode"]
    engine = decode_cases[0]["engine_index"]
    prefill = min(
        (case for case in plan["cases"] if case["engine_index"] == engine and case["phase"] == "prefill"),
        key=lambda case: case["prompt_len"],
    )
    decode = min(
        (case for case in decode_cases if case["engine_index"] == engine),
        key=lambda case: (case["batch_size"], case["prompt_len"]),
    )
    return engine, [prefill["case_id"], decode["case_id"]]


def _dsa_phases(plan: dict, revision: str) -> list[dict]:
    engine, case_ids = _dsa_smoke_cases(plan)
    smoke, full = f"{WORKER_OUT}/smoke", f"{WORKER_OUT}/full"
    engines = [entry["engine_index"] for entry in plan["engines"]]
    if len(engines) > GPUS:
        raise ValueError(f"the DSA plan has {len(engines)} engines for {GPUS} GPUs")
    return [
        {"name": "smoke", "steps": [
            _step(_dsa_run(engine, smoke, "0", case_ids)),
            _step(_dsa_aggregate(smoke, revision, partial=True)),
        ]},
        {"name": "full", "steps": [
            _step(*(_dsa_run(index, full, str(gpu)) for gpu, index in enumerate(engines))),
            _step(_dsa_aggregate(full, revision, partial=False)),
        ]},
    ]


def _table_phases(module: str, args: list[str], smoke_args: list[str], full_args: list[str],
                  revision: str) -> list[dict]:
    phases = []
    for phase, extra, num_gpus in (("smoke", smoke_args, "1"), ("full", full_args, str(GPUS))):
        out = f"{WORKER_OUT}/{phase}"
        phases.append({"name": phase, "steps": [
            _step(_command(
                f"{module.rsplit('.', 2)[-2]}_{method}",
                ["{python}", "-m", module, *args, *extra, "--num_gpus", num_gpus,
                 "--profile_method", method, *_revision_args(revision),
                 "--output_dir", out, "--yes"],
            ))
            for method in PROFILE_METHODS
        ]})
    return phases


def _run_snapshot(snapshot: Path, argv: list[str]) -> str:
    env = {**os.environ, "PYTHONPATH": str(snapshot), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, *argv], cwd=snapshot, env=env, check=True,
                          capture_output=True, text=True).stdout


def _json_from_output(output: str) -> dict:
    return json.loads(output[output.index("{\n"):])


def write() -> None:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, check=True,
                              capture_output=True, text=True).stdout.strip()
    snapshot = LAUNCH_DIR / "source" / revision
    if not snapshot.is_dir():
        archive = subprocess.run(["git", "archive", revision, *SNAPSHOT_PATHS], cwd=WORKSPACE,
                                 check=True, capture_output=True).stdout
        partial = snapshot.with_name(f"{revision}.partial")
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(partial)
        partial.rename(snapshot)
    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"the DSA engines need the model config directory {MODEL_DIR}")

    runs = LAUNCH_DIR / "runs" / revision
    runs.mkdir(parents=True, exist_ok=True)
    plan_path = runs / "dsa_plan.json"
    _run_snapshot(snapshot, ["-m", "frontier.profiling.attention.dsa_main", "plan",
                             *DSA_PLAN_ARGS, "--output", str(plan_path)])
    plan = json.loads(plan_path.read_text())
    linear_dry = _json_from_output(_run_snapshot(snapshot, [
        "-m", "frontier.profiling.linear_op.main", *LINEAR_ARGS, "--dry_run",
        "--profile_method", "cuda_event"]))
    moe_dry = _json_from_output(_run_snapshot(snapshot, [
        "-m", "frontier.profiling.moe.main", *MOE_ARGS,
        "--expert_parallel_sizes", *MOE_EXPERT_PARALLEL_SIZES, "--dry_run",
        "--profile_method", "cuda_event"]))
    (runs / "linear_op_dry_run.json").write_text(json.dumps(linear_dry, indent=2))
    (runs / "moe_dry_run.json").write_text(json.dumps(moe_dry, indent=2))

    stamp = time.strftime("%Y%m%d%H%M", time.gmtime())
    phases = {
        "dsa": _dsa_phases(plan, revision),
        "linear_op": _table_phases(
            "frontier.profiling.linear_op.main", LINEAR_ARGS,
            ["--num_tokens_list", *SMOKE_NUM_TOKENS], [], revision),
        "moe": _table_phases(
            "frontier.profiling.moe.main", MOE_ARGS,
            ["--num_tokens_list", *SMOKE_NUM_TOKENS, "--expert_parallel_sizes",
             MOE_EXPERT_PARALLEL_SIZES[-1], "--num_samples_per_distribution", "1"],
            ["--expert_parallel_sizes", *MOE_EXPERT_PARALLEL_SIZES], revision),
    }
    for producer, producer_phases in phases.items():
        job_tag = f"glm53-w4-{producer.replace('_', '')}-{stamp}"
        config = {
            "job_tag": job_tag, "producer": producer, "source_revision": revision,
            "model_config_revision": MODEL_CONFIG_REVISION,
            "source_snapshot": str(snapshot), "wheelhouse": str(LAUNCH_DIR / "wheelhouse"),
            "worker": str(LAUNCH_DIR / "glm_producer_worker.py"),
            "output_dir": f"{CLOUD_ROOT}/{job_tag}", "image": IMAGE,
            "cuda_compat_path": CUDA_COMPAT_PATH, "cloud_mount": CLOUD_MOUNT,
            "gpu_type": GPU_TYPE, "charged_group": CHARGED_GROUP, "gpu": GPUS, "cpu": CPUS,
            "mem_gb": MEM_GB, "timeout_seconds": TIMEOUT_SECONDS[producer],
            "phases": producer_phases,
        }
        path = runs / f"{producer}.json"
        path.write_text(json.dumps(config, indent=2))
        print(path)


def submit(config_path: Path) -> None:
    from steptron.exp.base_exp import ResourceConfig
    from steptron.utils.stepmind import spawn_tasks

    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if os.environ.get("STEPMIND_BACKEND") != "rjob":
        raise RuntimeError("submit requires STEPMIND_BACKEND=rjob; use launch_glm_producers.sh")
    os.chdir(WORKSPACE)
    os.environ["EXP_ID"] = config["job_tag"]
    command = (
        f"export LD_LIBRARY_PATH={shlex.quote(config['cuda_compat_path'])}"
        "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
        + shlex.join(["timeout", str(config["timeout_seconds"]), "python3", config["worker"],
                      "--config", str(config_path)])
    )
    resource = ResourceConfig(
        cpu=config["cpu"], gpu=config["gpu"], mem_gb=config["mem_gb"], replica=1,
        image=config["image"], positive_tags=[config["gpu_type"]], extra_requirements=[],
        mounts=[config["cloud_mount"]],
        envs={"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
              "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
              "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
              "VLLM_CUDA_COMPATIBILITY_PATH": config["cuda_compat_path"]},
        command="{COMMAND}", task_specs={"default": {"is_critical": True}},
    )
    workers = spawn_tasks(resource, command="bash -c " + shlex.quote(command),
                          charged_group=config["charged_group"], use_image=True,
                          code_mount_point=str(WORKSPACE))
    print("RJOB_NAME", workers.rjob.meta.name, flush=True)
    print("FINAL_STATUS", workers.poll(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("write")
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.action == "write":
        write()
    else:
        submit(arguments.config)
