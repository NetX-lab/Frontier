"""Run one sealed DeepSeek-V4 profiling unit on an H200 worker.

The job config lists phases (smoke, then full); each phase is a list of steps,
and a step is one command or a group of commands that run concurrently on
separate GPUs. Commands run from the committed source snapshot named in the
config. Producers write to worker-local disk; after each phase the tables,
raw records, logs, and completed profiler traces are copied to the cloud output
directory, together with ``status.json``. A failed phase stops the job.

Only the standard library is used, so the image's ``python3`` runs it as is.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

LOCAL_ROOT = Path("/tmp/glm_producers")
CACHE_VARIABLES = {
    "HF_HOME": "hf",
    "VLLM_CACHE_ROOT": "vllm",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
}
TRACE_DIRECTORIES = {"profiler_traces", "traces"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True))
    partial.replace(path)


def _python_path_with_wheels(wheelhouse: Path, env: dict) -> list[str]:
    """Unpack the wheelhouse when the image lacks a module the producers import."""

    if importlib.util.find_spec("pandas") is not None:
        return []
    target = LOCAL_ROOT / "site"
    for wheel in sorted(wheelhouse.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)
    env["PYTHONPATH"] = f"{target}:{env['PYTHONPATH']}"
    return [wheel.name for wheel in sorted(wheelhouse.glob("*.whl"))]


def _runtime_report(env: dict) -> dict:
    probe = (
        "import json, sys, pandas, yaml, tqdm, torch, vllm, flashinfer;"
        "print(json.dumps({'python': sys.version.split()[0], 'pandas': pandas.__version__,"
        " 'torch': torch.__version__, 'vllm': vllm.__version__,"
        " 'flashinfer': flashinfer.__version__, 'cuda_available': torch.cuda.is_available(),"
        " 'device_count': torch.cuda.device_count(),"
        " 'device_name': torch.cuda.get_device_name(0)}))"
    )
    output = subprocess.run(
        [sys.executable, "-c", probe], env=env, check=True, capture_output=True, text=True
    ).stdout
    report = json.loads(output.strip().splitlines()[-1])
    report["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return report


def _run_step(step: dict, env: dict, cwd: Path, log_dir: Path) -> None:
    """Run one command, or a concurrent group with one GPU per command."""

    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for command in step["commands"]:
        command_env = dict(env)
        if "cuda_visible_devices" in command:
            command_env["CUDA_VISIBLE_DEVICES"] = command["cuda_visible_devices"]
        argv = [sys.executable if part == "{python}" else part for part in command["argv"]]
        log_path = log_dir / f"{command['name']}.log"
        print(f"[{time.strftime('%H:%M:%S')}] start {command['name']}: {shlex.join(argv)}", flush=True)
        log = log_path.open("w")
        processes.append((command["name"], subprocess.Popen(
            argv, cwd=cwd, env=command_env, stdout=log, stderr=subprocess.STDOUT
        ), log, log_path))
    failed = []
    for name, process, log, log_path in processes:
        returncode = process.wait()
        log.close()
        print(f"[{time.strftime('%H:%M:%S')}] end {name}: exit {returncode}", flush=True)
        if returncode != 0:
            tail = log_path.read_text(errors="replace").splitlines()[-40:]
            print("\n".join(f"  {name}| {line}" for line in tail), flush=True)
            failed.append(f"{name} exited {returncode}")
    if failed:
        raise RuntimeError("; ".join(failed))


def _copy_results(local_phase: Path, cloud_phase: Path) -> dict:
    """Archive completed phase files, including traces, using the existing copy."""

    copied, trace_bytes = [], 0
    for path in sorted(local_phase.rglob("*")):
        relative = path.relative_to(local_phase)
        if not path.is_file():
            continue
        if TRACE_DIRECTORIES.intersection(relative.parts):
            trace_bytes += path.stat().st_size
        destination = cloud_phase / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        copied.append(str(relative))
    return {"copied_files": copied, "trace_bytes_archived": trace_bytes,
            "trace_bytes_left_on_worker": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())

    source = Path(config["source_snapshot"])
    cloud = Path(config["output_dir"])
    status = {"job_tag": config["job_tag"], "source_revision": config["source_revision"], "phases": {}}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable, name in CACHE_VARIABLES.items():
        env[variable] = str(LOCAL_ROOT / "cache" / name)
    try:
        status["unpacked_wheels"] = _python_path_with_wheels(Path(config["wheelhouse"]), env)
        status["runtime"] = _runtime_report(env)
        _write_json(cloud / "status.json", status)
        for phase in config["phases"]:
            local_phase = LOCAL_ROOT / "out" / phase["name"]
            started = time.time()
            status["phases"][phase["name"]] = {"state": "running"}
            _write_json(cloud / "status.json", status)
            state = "failed"
            try:
                for step in phase["steps"]:
                    _run_step(step, env, source, local_phase / "logs")
                state = "succeeded"
            finally:
                status["phases"][phase["name"]] = {
                    **_copy_results(local_phase, cloud / phase["name"]),
                    "state": state,
                    "seconds": round(time.time() - started, 1),
                }
                _write_json(cloud / "status.json", status)
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        _write_json(cloud / "status.json", status)
        raise


if __name__ == "__main__":
    main()
