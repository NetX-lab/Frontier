#!/usr/bin/env python3
"""Verify an immutable Step small-probe stage without invoking platform tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def dry_run(command, *, stage, env):
    text = subprocess.check_output(command, cwd=stage, env=env, text=True)
    lines = text.splitlines()
    json_start = lines.index("{")
    return json.loads("\n".join(lines[json_start:])), lines[:json_start]


def verify(stage):
    stage = Path(stage).resolve(strict=True)
    manifest = json.loads((stage / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((stage / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Staged payload checksum differs: {name}")
    expected_files = set(manifest["sha256"]) | {"manifest.json"}
    actual_files = {str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("Staged file inventory differs from manifest")
    assert manifest["purpose"] == "SMALL_OPERATOR_PROBE_ONLY_NOT_CANONICAL_SERVING_QUALIFICATION"
    assert (manifest["gpu_tag"], manifest["charged_group"], manifest["gpus"]) == ("H200", "step_main", 1)
    assert type(manifest["preemptible"]) is bool and manifest["timeout_seconds"] == 3600
    assert ("--preemptible" in shlex.split(manifest["submission_command"])) == manifest["preemptible"]
    assert manifest["cloud_output"].startswith("/mnt/codesign-exp/ycfeng/")
    assert manifest["source_mount"] == str(stage)
    assert shlex.split(manifest["submission_command"])[-1] == manifest["worker_command"]
    for name in ("submit.sh", "submit_with_env.sh"):
        subprocess.run(["bash", "-n", str(stage / name)], check=True)
    subprocess.run(["bash", "-n"], input=manifest["worker_command"], text=True, check=True)
    env = {**os.environ, "PYTHONPATH": f"{stage}:{stage / 'offline_site'}",
           "PYTHONDONTWRITEBYTECODE": "1", "VLLM_TARGET_DEVICE": "cpu"}
    command = [sys.executable, "-B", "-m", "frontier.profiling.step5.main",
               *manifest["attention_arguments"], "--dry-run"]
    attention, attention_logs = dry_run(command, stage=stage, env=env)
    assert (attention["rows"], attention["gpu_active_rows"]) == (18, 4)
    assert attention["native_max_num_seqs"] == 2
    assert attention["native_max_num_batched_tokens"] == 14336
    assert attention["cudagraph_capture_sizes"] == [1, 2]
    assert not attention["enable_chunked_prefill"]
    dry_runs = {}
    logs = {"attention": attention_logs}
    for line in manifest["worker_command"].splitlines():
        if "-m frontier.profiling.linear_op.main" not in line:
            continue
        args = shlex.split(line.split(" 2>&1 | tee ", 1)[0])
        args = args[args.index("-m"):]
        result, messages = dry_run([sys.executable, "-B", *args, "--dry_run"], stage=stage, env=env)
        method = args[args.index("--profile_method") + 1]
        planned = result["models"]["stepfun-ai/Step-5-Preview"]["plans"]
        assert sorted(result["num_tokens"]) == manifest["linear_plan"]["num_tokens"]
        assert planned == json.loads(json.dumps(manifest["linear_plan"]["plans"]))
        dry_runs[method] = "PASS"
        logs[method] = messages
    assert set(dry_runs) == {"cuda_event", "record_function"}
    workload = stage / manifest["workload"]["path"]
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=workload, check=True,
                   stdout=subprocess.PIPE, text=True)
    return {"status": "PASS", "staged_files": len(manifest["sha256"]),
            "stage": str(stage), "manifest_sha256": hashlib.sha256(
                (stage / "manifest.json").read_bytes()).hexdigest(),
            "attention_event_rows": attention["rows"],
            "attention_gpu_active_rows": attention["gpu_active_rows"],
            "linear_rows": manifest["expected_linear_rows"], "linear_dry_runs": dry_runs,
            "dry_run_logs": logs,
            "platform_operations": 0, "admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.stage)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
