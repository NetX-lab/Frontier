#!/usr/bin/env python3
"""Submit one Step-5-Preview profiling job to an H800 worker in the Step image.

Run this launcher with the StepMind Python after setting up the personal
authentication environment described in
``/data/ycfeng/stepfun-env-handbook/stepmind-python-rjob.md`` section 2.
``spawn_tasks`` waits for the job and its exit handler stops the worker, so
the launcher must stay alive until the job ends.

The repository root is mounted on the worker at the same path. The worker
command runs from that directory under ``timeout``.
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
# The Step-5-Preview vLLM image pinned by its linux/amd64 manifest digest.
STEP_IMAGE = (
    "a.z01.normandyfunc.com/docker-public/vllm/vllm-openai"
    "@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20"
)
CLOUD_MOUNT = "juicefs+s3://oss.i.shaipower.com/codesign-exp:/mnt/codesign-exp"
GPU_TAG = "H800"
CHARGED_GROUP = "codesign"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-tag", required=True, help="Experiment id recorded on the job.")
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--mem-gb", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument(
        "worker_command",
        help="Shell command the worker runs from the repository root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("STEPMIND_BACKEND") != "rjob":
        raise SystemExit("STEPMIND_BACKEND must be rjob")
    if not (os.environ.get("BRAINPP_ACCESS_KEY") and os.environ.get("BRAINPP_SECRET_KEY")):
        raise SystemExit("personal BRAINPP credentials are not set")

    from steptron.exp.base_exp import ResourceConfig
    from steptron.utils.stepmind import spawn_tasks

    command = (
        f"timeout {args.timeout_seconds} bash -c {shlex.quote(args.worker_command)}"
    )
    print("WORKER_COMMAND", command, flush=True)
    # spawn_tasks mounts the current directory as the code source.
    os.chdir(REPO_ROOT)
    os.environ["EXP_ID"] = args.job_tag
    resources = ResourceConfig(
        cpu=args.cpus,
        gpu=args.gpus,
        mem_gb=args.mem_gb,
        replica=1,
        image=STEP_IMAGE,
        positive_tags=[GPU_TAG],
        extra_requirements=[],
        mounts=[CLOUD_MOUNT],
        custom_resources=[],
        envs={},
        command="{COMMAND}",
        preemptible=False,
        task_specs={"default": {"is_critical": True}},
    )
    workers = spawn_tasks(
        resources,
        command=command,
        charged_group=CHARGED_GROUP,
        use_image=True,
        code_mount_point=str(REPO_ROOT),
    )
    print("JOB_ID", workers.rjob.meta.name, flush=True)
    print("FINAL_STATUS", workers.poll(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
