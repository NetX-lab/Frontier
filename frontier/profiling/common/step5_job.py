"""Shared immutable source packages for Step-image profiling jobs."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from pathlib import Path


IMAGE = ("a.z01.normandyfunc.com/docker-public/vllm/vllm-openai"
         "@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20")
CLOUD_MOUNT = "juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp"
CLOUD_ROOT = Path("/mnt/codesign-exp/ycfeng/frontier/step5_profiling")
STEPMIND_ENV = Path("/data/ycfeng/tmp/step5_native_pd_r1/stepmind_env.sh")


def describe_job(stage, run_id, *, worker_command, gpus=1,
                  cloud_root=CLOUD_ROOT, credential_env=STEPMIND_ENV,
                  gpu_tag="H800", charged_group="codesign"):
    """Describe the existing personal RJob launcher without invoking it."""
    stage = Path(stage).resolve()
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single nonempty path component")
    q = shlex.quote
    return {
        "image": IMAGE, "gpu_tag": gpu_tag, "charged_group": charged_group,
        "gpus": gpus, "cpus": 8 * gpus, "mem_gb": 32 * gpus,
        "mounts": [f"local_nfs:{stage}:{stage}", CLOUD_MOUNT],
        "source_mount": str(stage), "cloud_output": str(Path(cloud_root) / run_id),
        "timeout_seconds": 3600, "preemptible": False,
        "credential_env": str(credential_env),
        "submit_wrapper_command": f"bash {q(str(stage / 'submit_with_env.sh'))}",
        "worker_command": worker_command,
        "submission_command": (
            "/data/ycfeng/tmp/stepmind-env/bin/python -u -B "
            f"{q(str(stage / 'tests/integration/step5_profiling/submit_profiling_job.py'))} "
            f"--job-tag {q(run_id)} --gpus {gpus} --cpus {8*gpus} --mem-gb {32*gpus} "
            f"--gpu-tag {q(gpu_tag)} --charged-group {q(charged_group)} "
            f"--timeout-seconds 3600 {q(worker_command)}"),
    }


def stage_package(stage, manifest, *, repo_root, dependency_site, additional_paths=()):
    """Copy sources and offline dependencies once, hashing every payload file."""
    stage = Path(stage).resolve()
    repo_root = Path(repo_root)
    dependency_site = Path(dependency_site).resolve(strict=True)
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copytree(repo_root / "frontier", stage / "frontier",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    shutil.copytree(repo_root / "data/config", stage / "data/config")
    for relative_path in additional_paths:
        shutil.copytree(repo_root / relative_path, stage / relative_path)
    shutil.copytree(dependency_site, stage / "offline_site",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    scripts = stage / "tests/integration/step5_profiling"
    scripts.mkdir(parents=True)
    for name in ("submit_profiling_job.py", "run_on_worker.sh", *manifest.get("verification_scripts", ())):
        shutil.copy2(repo_root / "tests/integration/step5_profiling" / name, scripts / name)
    (stage / "submit.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" +
                                    manifest["submission_command"] + "\n")
    (stage / "submit_with_env.sh").write_text("\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail", "set +x",
        f"source {shlex.quote(manifest['credential_env'])}",
        ': "${BRAINPP_ACCESS_KEY:?personal BRAINPP_ACCESS_KEY is not set}"',
        ': "${BRAINPP_SECRET_KEY:?personal BRAINPP_SECRET_KEY is not set}"',
        "export STEPMIND_BACKEND=rjob", "export TMPDIR=/data/ycfeng/tmp",
        'export PYTHONPATH="/data/ycfeng/steptron${PYTHONPATH:+:$PYTHONPATH}"',
        f"exec bash {shlex.quote(str(stage / 'submit.sh'))}", "",
    ]))
    manifest = dict(manifest)
    manifest["sha256"] = {
        str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(stage.rglob("*")) if path.is_file()}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
