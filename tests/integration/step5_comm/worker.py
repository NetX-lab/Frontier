"""Run S0 in the locked Step image; use the existing collective benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


def collective_command(job: dict) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={job['resources']['gpu']}", "-m",
            "frontier.profiling.collectives.vllm_collective_benchmark",
            "--config", job["benchmark"]]


def probe_command(job: dict) -> list[str]:
    probe = job["native_probe"]
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={job['resources']['gpu']}", "-m",
               "frontier.profiling.collectives.step5_dispatch_probe",
               "--model-path", probe["model_path"], "--output-dir",
               str(Path(job["output_dir"]) / "native_dispatch"),
               "--tokens-per-dp-rank", *map(str, probe["tokens_per_dp_rank"])]
    if probe["disable_custom_all_reduce"]:
        command.append("--disable-custom-all-reduce")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.config.read_text())
    root = Path(job["output_dir"])
    if not root.is_relative_to("/mnt/codesign-exp/ycfeng/frontier/step5_comm"):
        raise ValueError("Outputs must remain inside the personal communication directory")
    root.mkdir(parents=True, exist_ok=False)
    log = (root / "worker.log").open("w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    scratch = Path(tempfile.mkdtemp(prefix="step5_comm_"))
    for name in ("HF_HOME", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                 "FLASHINFER_WORKSPACE_BASE"):
        os.environ[name] = str(scratch / name.lower())
    for relative, expected in job["source_hashes"].items():
        path = Path(job["repository"]) / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed after preparation: {relative}")
    if hashlib.sha256(Path(job["benchmark"]).read_bytes()).hexdigest() != job["benchmark_sha256"]:
        raise ValueError("The prepared benchmark configuration changed")
    metadata = Path(job["native_probe"]["model_path"]) / "config.json"
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != job["native_probe"]["config_sha256"]:
        raise ValueError("The staged native metadata changed")
    import torch
    import vllm

    manifest = {"job": job, "host": socket.gethostname(), "python": sys.version,
                "torch": torch.__version__, "vllm": vllm.__version__,
                "nccl": torch.cuda.nccl.version(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "nccl_environment": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    manifest["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("STEP5_COMM_START", job["run_id"], manifest["host"], flush=True)
    command = collective_command(job)
    print("COLLECTIVE_COMMAND", command, flush=True)
    result = subprocess.run(command)
    (root / "collective_exit.json").write_text(json.dumps({"returncode": result.returncode}))
    if result.returncode:
        raise SystemExit(result.returncode)
    print("STEP5_COMM_COLLECTIVES_DONE", job["run_id"], flush=True)
    command = probe_command(job)
    print("NATIVE_PROBE_COMMAND", command, flush=True)
    result = subprocess.run(command)
    (root / "native_probe_exit.json").write_text(json.dumps({"returncode": result.returncode}))
    if result.returncode:
        raise SystemExit(result.returncode)
    report = json.loads((root / "native_dispatch/dispatch_probe.json").read_text())
    # A diagnostic schema mismatch keeps all evidence and is explicitly not PASS.
    verdict = {"execution_completed": True, "native_dispatch_status": report["status"],
               "target_topology_qualified": False, "cost_data_qualified": False,
               "expected": job["expected"]}
    (root / "result.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print("STEP5_COMM_DONE", json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
