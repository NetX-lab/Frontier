"""Submit one DeepSeek-V4 G5h communication benchmark stage as an H200 RJob.

Run from the local workspace after the personal-auth initialization of
stepmind-python-rjob.md section 2, and keep this launcher alive until the job
ends: RJobBackend stops the workers when the launcher exits. The launcher
records its start and exit next to this script under ``receipts/``.
"""

import argparse
import json
import os
import shlex
import time
from pathlib import Path

from runtime_check import build_lock, worker_command

from steptron.exp.base_exp import ResourceConfig
from steptron.utils.stepmind import spawn_tasks

LAUNCH = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    config_path = parser.parse_args().config.resolve()
    config = json.loads(config_path.read_text())
    assert os.environ["STEPMIND_BACKEND"] == "rjob"
    assert (config["gpu_type"], config["charged_group"]) == ("H200", "step_main")
    assert config["gpu"] * config["replica"] <= 64
    os.chdir(config["workspace"])
    os.environ["EXP_ID"] = config["job_tag"]
    receipts = LAUNCH / "receipts"
    receipts.mkdir(exist_ok=True)
    status = {"stage": config["stage"], "config": str(config_path), "pid": os.getpid(),
              "start": time.time(), "gpus": config["gpu"] * config["replica"]}
    (receipts / f"{config['job_tag']}_started.json").write_text(json.dumps(status, indent=2))
    lock_path = receipts / f"{config['job_tag']}_lock.json"
    distributed = config["replica"] > 1
    resource = ResourceConfig(
        cpu=config["cpu"], gpu=config["gpu"], mem_gb=config["mem_gb"],
        replica=config["replica"], image=config["image"], positive_tags=[config["gpu_type"]],
        extra_requirements=[], mounts=[config["cloud_mount"]],
        envs={"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
              "TRANSFORMERS_OFFLINE": "1", "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
              "VLLM_CUDA_COMPATIBILITY_PATH": config["cuda_compat_path"],
              **({"DISTRIBUTED_JOB": "true"} if distributed else {}), **config["envs"]},
        task_anti_affinities=["hard/default"] if distributed else [],
        command="{COMMAND}", task_specs={"default": {"is_critical": True}})
    try:
        lock_path.write_text(json.dumps(build_lock(config, config_path), indent=2) + "\n")
        workers = spawn_tasks(resource, command="bash -c " + shlex.quote(
            worker_command(config, config_path, lock_path)), charged_group=config["charged_group"],
            use_image=True, code_mount_point=config["workspace"])
        status["rjob_name"] = workers.rjob.meta.name
        print("RJOB_NAME", status["rjob_name"], flush=True)
        status["final_status"] = str(workers.poll())
        print("FINAL_STATUS", status["final_status"], flush=True)
    finally:
        status["end"] = time.time()
        (receipts / f"{config['job_tag']}_exit.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
