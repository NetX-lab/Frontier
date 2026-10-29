#!/usr/bin/env python3
"""Offline package/preflight for the one-cell H200 boundary experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from boundary_probe import CONFIG_PATH, describe
from frontier.profiling.common.step5_job import IMAGE, describe_job, stage_package
from frontier.profiling.step5.job import REPO_ROOT, WORKLOAD, describe_probe
from verify_fixed_probe import dry_run


RUN_ID = "d18_boundary_probe_h200_01"
SCRIPTS = Path("tests/integration/step5_profiling")
FILES = ("boundary_probe.py", "boundary_probe_job.py", "boundary_probe_config.json", "verify_fixed_probe.py")
SOURCE_FILES = (
    "frontier/profiling/common/step5_job.py", "frontier/profiling/step5/vllm_wrapper.py",
    "frontier/profiling/step5/replay.py", "frontier/profiling/step5/timing.py",
    "frontier/profiling/step5/inputs.py", "frontier/profiling/step5/fixed_probe_inputs.json",
    "frontier/profiling/step5/device_activity.py", "frontier/profiling/utils/record_function_tracer.py",
    "data/config/models/stepfun-ai__Step-5-Preview.json",
    *(str(SCRIPTS / name) for name in FILES),
    str(SCRIPTS / "submit_profiling_job.py"), str(SCRIPTS / "run_on_worker.sh"),
)


def source_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
    return commit, not dirty.strip()


def description(stage, run_id=RUN_ID):
    stage = Path(stage).resolve()
    settings = json.loads(CONFIG_PATH.read_text())
    plan = describe(settings)
    # Reuse the existing immutable workload validation and native-policy source.
    previous = describe_probe(stage, run_id)
    output = Path(previous["cloud_output"])
    args = [str(SCRIPTS / "boundary_probe.py"), "--config", str(SCRIPTS / CONFIG_PATH.name),
            "--model-path", f"/data/ycfeng/tmp/{run_id}/model", "--output-dir", str(output)]
    q = shlex.quote
    worker = "\n".join([
        "set -euo pipefail", f"export TMPDIR={q('/data/ycfeng/tmp/' + run_id)}",
        "export PYTHONDONTWRITEBYTECODE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1",
        'export PYTHONPATH="$PWD:$PWD/offline_site${PYTHONPATH:+:$PYTHONPATH}"',
        shlex.join(["python3", "-B", str(SCRIPTS / "boundary_probe_job.py"), "--stage", str(stage), "--verify"]),
        f"test ! -e {q(str(output))}", f"mkdir -p {q(str(output.parent))}", f"mkdir {q(str(output))}",
        'mkdir -p "$TMPDIR/model"',
        'cp data/config/models/stepfun-ai__Step-5-Preview.json "$TMPDIR/model/config.json"',
        shlex.join(["bash", str(SCRIPTS / "run_on_worker.sh"), *args])
            + f" 2>&1 | tee {q(str(output / 'worker.log'))}",
    ])
    manifest = describe_job(stage, run_id, worker_command=worker, gpus=1,
                            gpu_tag="H200", charged_group="step_main", timeout_seconds=600)
    commit, clean = source_state()
    return {**manifest, "purpose": "CAUSAL_BOUNDARY_DIAGNOSTIC_ONLY_NOT_PROFILE_ADMISSION",
            "source_commit": commit, "source_clean": clean, "settings": settings, "experiment": plan,
            "workload": previous["workload"], "native_probe_policy": previous["native_probe_policy"],
            "experiment_arguments": args, "verification_scripts": list(FILES),
            "expected_outputs": ["windows.jsonl", "results.json", "traces/cohort_A.json", "traces/cohort_B.json", "worker.log"],
            "expected_minutes": [2, 5], "max_gpu_hours": 1 / 6, "admitted": False}


def stage_experiment(stage, *, dependency_site, run_id=RUN_ID):
    manifest = description(stage, run_id)
    if not manifest["source_clean"]:
        raise ValueError("Commit the verified experiment; staging requires a clean source worktree")
    manifest["committed_source_sha256"] = {name: hashlib.sha256(subprocess.check_output(
        ["git", "show", f"{manifest['source_commit']}:{name}"], cwd=REPO_ROOT)).hexdigest() for name in SOURCE_FILES}
    if any(hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() != digest
           for name, digest in manifest["committed_source_sha256"].items()):
        raise ValueError("Current experiment source differs from its committed bytes")
    return stage_package(stage, manifest, repo_root=REPO_ROOT,
                          dependency_site=dependency_site, additional_paths=(WORKLOAD,))


def verify(stage):
    stage = Path(stage).resolve(strict=True)
    manifest = json.loads((stage / "manifest.json").read_text())
    expected = set(manifest["sha256"]) | {"manifest.json"}
    actual = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Staged inventory differs from the sealed manifest")
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((stage / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Staged checksum differs: {name}")
    assert manifest["purpose"] == "CAUSAL_BOUNDARY_DIAGNOSTIC_ONLY_NOT_PROFILE_ADMISSION"
    assert manifest["image"] == IMAGE
    assert (manifest["gpus"], manifest["gpu_tag"], manifest["charged_group"], manifest["timeout_seconds"]) == (1, "H200", "step_main", 600)
    assert manifest["source_clean"] and not manifest["admitted"]
    assert manifest["source_mount"] == str(stage)
    assert shlex.split(manifest["submission_command"])[-1] == manifest["worker_command"]
    for name, digest in manifest["committed_source_sha256"].items():
        assert manifest["sha256"][name] == digest
    settings = json.loads((stage / SCRIPTS / CONFIG_PATH.name).read_text())
    assert settings == manifest["settings"]
    env = {**os.environ, "PYTHONPATH": f"{stage}:{stage / 'offline_site'}",
           "PYTHONDONTWRITEBYTECODE": "1", "VLLM_TARGET_DEVICE": "cpu"}
    result, logs = dry_run([sys.executable, "-B", *manifest["experiment_arguments"], "--dry-run"],
                           stage=stage, env=env)
    assert result == manifest["experiment"]
    assert (result["measured_replays"], result["trace_replays"], result["event_handles"], result["native_graphs"]) == (300, 30, 18, 1)
    assert (stage / manifest["workload"]["path"] / "manifest.json").is_file()
    for name, digest in manifest["workload"]["sha256"].items():
        assert hashlib.sha256((stage / manifest["workload"]["path"] / name).read_bytes()).hexdigest() == digest
    for name in ("submit.sh", "submit_with_env.sh"):
        subprocess.run(["bash", "-n", str(stage / name)], check=True)
    subprocess.run(["bash", "-n"], input=manifest["worker_command"], text=True, check=True)
    return {"status": "PASS", "source_commit": manifest["source_commit"], "staged_files": len(expected) - 1,
            "committed_sources": len(manifest["committed_source_sha256"]), "experiment": result,
            "manifest_sha256": hashlib.sha256((stage / "manifest.json").read_bytes()).hexdigest(),
            "dry_run_logs": logs, "platform_operations": 0, "admitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--dependency-site", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verify", action="store_true")
    actions.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.stage)
    elif args.dry_run:
        result = description(args.stage, args.run_id)
    else:
        if args.dependency_site is None:
            parser.error("staging requires --dependency-site")
        result = stage_experiment(args.stage, dependency_site=args.dependency_site, run_id=args.run_id)
    print(json.dumps({k: v for k, v in result.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
