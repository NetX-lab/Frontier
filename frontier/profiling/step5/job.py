"""Stage an immutable smoke package; submission remains a separate parent action."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGE = ("a.z01.normandyfunc.com/docker-public/vllm/vllm-openai"
         "@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20")
CLOUD_MOUNT = "juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp"
CLOUD_ROOT = Path("/mnt/codesign-exp/ycfeng/frontier/step5_profiling")


def describe_smoke(stage, run_id, *, gpus=1):
    stage = Path(stage).resolve()
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single nonempty path component")
    if gpus != 1:
        raise ValueError("The smoke package profiles one isolated TP8 rank on one GPU")
    output = CLOUD_ROOT / run_id
    q = shlex.quote
    worker = "\n".join([
        "set -euo pipefail", "export TMPDIR=/data/ycfeng/tmp/step5_attention_worker",
        'mkdir -p "$TMPDIR/model"', f"test ! -e {q(str(output))}",
        'cp data/config/models/stepfun-ai__Step-5-Preview.json "$TMPDIR/model/config.json"',
        'export PYTHONPATH="$PWD/offline_site${PYTHONPATH:+:$PYTHONPATH}"',
        "python3 -c 'import pandas, dateutil, pytz, tzdata, yaml, tqdm, six; "
        "print(\"OFFLINE_DEPENDENCIES_READY\", pandas.__version__, pandas.__file__)'",
        "nvidia-smi -L",
        "bash tests/integration/step5_profiling/run_on_worker.sh "
        "-m frontier.profiling.step5.main --model-path \"$TMPDIR/model\" "
        "--tensor-parallel-size 8 --max-model-len 8192 "
        "--prefill-tokens 16 64 --prefixes 0 4104 --decode-batches 1 "
        "--decode-contexts 512 4104 --graph-modes none full "
        f"--output-dir {q(str(output))}",
    ])
    return {"image": IMAGE, "gpu_tag": "H800", "charged_group": "codesign",
            "gpus": gpus, "cpus": 8 * gpus, "mem_gb": 32 * gpus,
            "mounts": [f"local_nfs:{stage}:{stage}", CLOUD_MOUNT],
            "source_mount": str(stage), "cloud_output": str(output), "expected_rows": 16,
            "timeout_seconds": 3600, "preemptible": False,
            "worker_command": worker,
            "submission_command": (
                "/data/ycfeng/tmp/stepmind-env/bin/python -u -B "
                f"{q(str(stage / 'tests/integration/step5_profiling/submit_profiling_job.py'))} "
                f"--job-tag {q(run_id)} --gpus {gpus} --cpus {8*gpus} --mem-gb {32*gpus} "
                f"--timeout-seconds 3600 {q(worker)}")}


def stage_smoke(stage, run_id, *, dependency_site, gpus=1):
    stage = Path(stage).resolve()
    manifest = describe_smoke(stage, run_id, gpus=gpus)
    dependency_site = Path(dependency_site).resolve(strict=True)
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copytree(REPO_ROOT / "frontier", stage / "frontier",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    shutil.copytree(REPO_ROOT / "data/config", stage / "data/config")
    shutil.copytree(dependency_site, stage / "offline_site",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    scripts = stage / "tests/integration/step5_profiling"
    scripts.mkdir(parents=True)
    for name in ("submit_profiling_job.py", "run_on_worker.sh"):
        shutil.copy2(REPO_ROOT / "tests/integration/step5_profiling" / name, scripts / name)
    manifest["sha256"] = {
        str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(stage.rglob("*")) if path.is_file()}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (stage / "submit.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" +
                                    manifest["submission_command"] + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dependency-site", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    manifest = stage_smoke(args.stage, args.run_id, dependency_site=args.dependency_site, gpus=args.gpus)
    print(json.dumps({**{k: v for k, v in manifest.items() if k != "sha256"},
                      "staged_files": len(manifest["sha256"])}, indent=2))


if __name__ == "__main__":
    main()
