"""Submit a prepared S0 package once; only the parent agent runs this command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex


def job_command(job: dict, config_path: Path) -> str:
    worker = Path(job["repository"]) / "tests/integration/step5_comm/run_on_worker.sh"
    return shlex.join(["timeout", str(job["resources"]["timeout_seconds"]),
                       "bash", str(worker), str(config_path)])


def resource_arguments(job: dict) -> dict:
    resources = job["resources"]
    return dict(cpu=resources["cpu"], gpu=resources["gpu"], mem_gb=resources["mem_gb"],
                replica=resources["replica"], image=job["image"],
                positive_tags=resources["positive_tags"], extra_requirements=[],
                mounts=[resources["cloud_mount"]], preemptible=False,
                envs={"DISTRIBUTED_JOB": "true", "NCCL_DEBUG": "INFO",
                      "NCCL_DEBUG_SUBSYS": "INIT,NET", "PYTHONDONTWRITEBYTECODE": "1"},
                task_anti_affinities=["hard/default"], command="{COMMAND}",
                task_specs={"default": {"is_critical": True}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    job = json.loads(config.read_text())
    command, resources = job_command(job, config), resource_arguments(job)
    if args.dry_run:
        print(json.dumps({"resources": resources, "command": command,
                          "source_mount": job["repository"], "output_dir": job["output_dir"],
                          "expected": job["expected"]}, indent=2))
        return
    if os.environ.get("STEPMIND_BACKEND") != "rjob":
        raise ValueError("Initialize the personal StepMind environment first")
    if any(not os.environ.get(name) for name in ("BRAINPP_ACCESS_KEY", "BRAINPP_SECRET_KEY")):
        raise ValueError("Personal credentials must be set before creating the client")
    # An exclusive local marker prevents accidentally duplicating this package.
    with config.with_name("submission_attempt.json").open("x") as stream:
        json.dump({"run_id": job["run_id"], "config": str(config)}, stream)
    os.chdir(job["repository"])
    os.environ["EXP_ID"] = job["run_id"]
    from steptron.exp.base_exp import ResourceConfig
    from steptron.utils.stepmind import spawn_tasks

    workers = spawn_tasks(ResourceConfig(**resources), command=command,
                          charged_group=job["resources"]["charged_group"], use_image=True,
                          code_mount_point=job["repository"])
    print("JOB_ID", workers.rjob.meta.name, "FINAL_STATUS", workers.poll(), flush=True)


if __name__ == "__main__":
    main()
