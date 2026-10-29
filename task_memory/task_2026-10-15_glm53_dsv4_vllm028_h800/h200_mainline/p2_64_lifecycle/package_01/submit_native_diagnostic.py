"""Submit a task-local native diagnostic with the locked image and offline metadata."""

import argparse
import json
import os
import shlex
from pathlib import Path

from steptron.exp.base_exp import ResourceConfig
from steptron.utils.stepmind import spawn_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    path = parser.parse_args().config.resolve()
    config = json.loads(path.read_text())
    assert os.environ["STEPMIND_BACKEND"] == "rjob"
    os.chdir(config["workspace"])
    os.environ["EXP_ID"] = config["job_tag"]
    command = (f"export LD_LIBRARY_PATH={shlex.quote(config['cuda_compat_path'])}"
               "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
               f"export PYTHONPATH={shlex.quote(str(Path(config['script']).parent))}:"
               f"{shlex.quote(str(path.parent))}"
               "${PYTHONPATH:+:$PYTHONPATH}; " + shlex.join([
                   "timeout", str(config["timeout_seconds"]), "python3",
                   config["script"], "--config", str(path)]))
    if "observer_bootstrap" in config:
        bootstrap = Path(config["observer_bootstrap"])
        if not bootstrap.is_absolute():
            raise ValueError("Observer bootstrap path must be absolute")
        command = (f"export LD_LIBRARY_PATH={shlex.quote(config['cuda_compat_path'])}"
                   "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; " + shlex.join([
                       "timeout", str(config["timeout_seconds"]), "python3", "-I", "-B",
                       str(bootstrap), "--config", str(path)]))
    resource = ResourceConfig(
        cpu=config["cpu"], gpu=config["gpu"], mem_gb=config["mem_gb"],
        replica=config["replica"], image=config["image"], positive_tags=[config["gpu_type"]],
        extra_requirements=[], mounts=[config["cloud_mount"]],
        envs={"VLLM_LOGGING_LEVEL": "INFO",
               "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
              "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
              "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
              "VLLM_CUDA_COMPATIBILITY_PATH": config["cuda_compat_path"],
              **({"DISTRIBUTED_JOB": "true"} if config["replica"] > 1 else {})},
        task_anti_affinities=["hard/default"] if config["replica"] > 1 else [],
        command="{COMMAND}", task_specs={"default": {"is_critical": True}},
    )
    workers = spawn_tasks(resource, command="bash -c " + shlex.quote(command),
                          charged_group=config["charged_group"], use_image=True,
                          code_mount_point=config["workspace"])
    print("RJOB_NAME", workers.rjob.meta.name, flush=True)
    try:
        from native_pd_worker import write_launcher_receipt
        write_launcher_receipt(config, workers.rjob.meta.name, path)
    except Exception as error:
        print("LAUNCHER_OBSERVATION_ERROR", repr(error), flush=True)

    print("FINAL_STATUS", workers.poll(), flush=True)
