"""Submission lock and per-node runtime checks for the G5h H200 benchmark.

No GPU or launcher dependencies are imported. The benchmark and NCCL checks
use the stage's gate; the image vLLM source comparison is always soft.
"""

import argparse
import hashlib
import json
import shlex
import socket
import subprocess
from pathlib import Path

TASK_ROOT = Path("task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800")
K5_NODE_ENTRY = TASK_ROOT / "w5_glm_comm_20261020/launch/node_entry.py"
BENCHMARK_MODULES = (
    "vllm_forward_collectives.py", "vllm_dsv4_forward_collectives.py",
    "vllm_collective_benchmark.py",
)
VLLM_SOURCES = (
    "models/common/ops/sequence_parallel.py",
    "distributed/device_communicators/cuda_communicator.py",
    "distributed/device_communicators/custom_all_reduce.py",
    "distributed/device_communicators/all2all.py",
    "distributed/parallel_state.py",
)
NCCL_LIBRARY = "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lock(config, config_path):
    """Hash committed mounted sources and the pinned P2/native reference files."""
    workspace = Path(config["workspace"]).resolve()
    config_path = Path(config_path).resolve()
    sources = [workspace / "frontier/profiling/collectives" / name
               for name in BENCHMARK_MODULES]
    sources += [config_path.parent / "runtime_check.py", workspace / K5_NODE_ENTRY, config_path]
    relative = [str(path.relative_to(workspace)) for path in sources]
    status = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain", "--untracked-files=all",
         "--", *relative], check=True, capture_output=True, text=True, timeout=30).stdout
    if status.strip():
        raise RuntimeError("Uncommitted benchmark sources; commit before submission:\n" +
                           status.rstrip())
    head = subprocess.run(["git", "-C", str(workspace), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    overlay = json.loads(Path(config["p2_overlay"]).read_text())
    native = Path(config["native_checkout"])
    installed = Path(overlay["installed_package"])
    return {
        "workspace_head": head,
        "benchmark_sources": {str(path): sha256(path) for path in sources},
        "nccl": {NCCL_LIBRARY: overlay["binary_sha256"][NCCL_LIBRARY]},
        "vllm_sources": {str(installed / name): sha256(native / name) for name in VLLM_SOURCES},
    }


def worker_command(config, config_path, lock_path):
    """Worker shell command, also inspectable without importing steptron."""
    config_path = Path(config_path)
    check = shlex.join(["python3", str(config_path.parent / "runtime_check.py"),
                        "--config", str(config_path), "--lock", str(lock_path)])
    entry = shlex.join(["python3", "-u", str(Path(config["workspace"]) / K5_NODE_ENTRY),
                       "--config", str(config_path)])
    command = shlex.join(["timeout", str(config["timeout_seconds"]), "bash", "-c",
                          check + " && " + entry])
    return (f"export LD_LIBRARY_PATH={shlex.quote(config['cuda_compat_path'])}"
            "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
            f"export PYTHONPATH={shlex.quote(config['workspace'])}"
            "${PYTHONPATH:+:$PYTHONPATH}; " + command)


def check_runtime(config, lock):
    """Record every path comparison; return 1 only for a failed hard check.

    Successful checks never create the node directory: K5 node_entry creates
    it with exist_ok=False. Hard failures publish its failed.json for peers.
    """
    gate = config["runtime_check_gate"]
    if gate not in ("hard", "soft"):
        raise ValueError(f"Unknown runtime_check_gate {gate!r}")
    host = socket.gethostname()
    checks = []
    for name, paths, check_gate in (("nccl", lock["nccl"], gate),
                                   ("benchmark_source", lock["benchmark_sources"], gate),
                                   ("vllm_source", lock["vllm_sources"], "soft")):
        for path, expected in paths.items():
            row = {"name": name, "path": path, "expected": expected,
                   "actual": None, "pass": False, "gate": check_gate}
            try:
                row["actual"] = sha256(path)
                row["pass"] = row["actual"] == expected
            except OSError as error:
                row["error"] = str(error)
            checks.append(row)
    failed = any(not row["pass"] and row["gate"] == "hard" for row in checks)
    report = {"host": host, "workspace_head": lock["workspace_head"],
              "runtime_check_gate": gate, "checks": checks, "returncode": int(failed)}
    output = Path(config["output_dir"])
    runtime_dir = output / "runtime_checks"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    report_path = runtime_dir / f"{host}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        node = output / "nodes" / host
        node.mkdir(parents=True, exist_ok=True)
        (node / "failed.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"runtime checks: {sum(row['pass'] for row in checks)}/{len(checks)} passed; "
          f"exit {int(failed)}; {report_path}", flush=True)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    return check_runtime(json.loads(args.config.read_text()), json.loads(args.lock.read_text()))


if __name__ == "__main__":
    raise SystemExit(main())
