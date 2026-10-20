"""Submit one staged native P/D run as a multi-node RJob through the personal Python backend.

Initialize the personal StepMind environment first (`STEPMIND_BACKEND=rjob` and the personal
access keys). The submitter changes into the staged workspace, which `spawn_tasks` mounts at
the same path, and submits once through the reused launcher scope, which records the attempt
before the SDK call and the job name right after it, and removes the SDK's exit-time stop.
It then waits in the SDK until the platform reports a terminal state. Evidence goes to
`launcher/` in the stage directory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex


def job_command(config: dict, config_path: Path) -> str:
    stage = Path(config["script"]).parent
    inner = (f"export LD_LIBRARY_PATH={shlex.quote(config['cuda_compat_path'])}"
             "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; "
             f"export PYTHONPATH={shlex.quote(str(stage))}${{PYTHONPATH:+:$PYTHONPATH}}; "
             + shlex.join(["timeout", str(config["timeout_seconds"]), "python3", "-B",
                           config["script"], "--config", str(config_path)]))
    return "bash -c " + shlex.quote(inner)


def resource_arguments(config: dict) -> dict:
    return dict(
        cpu=config["cpu"], gpu=config["gpu"], mem_gb=config["mem_gb"], replica=config["replica"],
        image=config["image"], positive_tags=[config["gpu_type"]], extra_requirements=[],
        mounts=[config["cloud_mount"]], preemptible=False,
        envs={"PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
              "HF_DATASETS_OFFLINE": "1", "VLLM_ENABLE_CUDA_COMPATIBILITY": "1",
              "DISTRIBUTED_JOB": "true"},
        task_anti_affinities=["hard/default"], command="{COMMAND}",
        task_specs={"default": {"is_critical": True}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--launcher-scope", type=Path, required=True,
                        help="launcher_scope.py providing own_submit_callback")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if os.environ.get("STEPMIND_BACKEND") != "rjob":
        raise ValueError("Initialize the personal Python RJob backend first")
    if any(not os.environ.get(name) for name in ("BRAINPP_ACCESS_KEY", "BRAINPP_SECRET_KEY")):
        raise ValueError("Personal credentials are required before SDK construction")
    evidence = config_path.parent / "launcher"
    evidence.mkdir(exist_ok=False)
    own_submit_callback = runpy.run_path(str(args.launcher_scope))["own_submit_callback"]
    command = job_command(config, config_path)
    arguments = resource_arguments(config)
    (evidence / "payload.json").write_text(json.dumps({
        "resource_constructor_arguments": arguments,
        "spawn_arguments": {"command": command, "charged_group": config["charged_group"],
                            "use_image": True, "code_mount_point": config["workspace"]},
        "launcher_scope": {"path": str(args.launcher_scope.resolve()),
                           "sha256": hashlib.sha256(args.launcher_scope.read_bytes()).hexdigest()},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}, indent=1))
    os.chdir(config["workspace"])
    os.environ["EXP_ID"] = config["job_tag"]
    from steptron.exp.base_exp import ResourceConfig
    from steptron.utils.stepmind import spawn_tasks
    from steptron.utils.stepmind_control import RJobBackend

    with own_submit_callback(RJobBackend, evidence, config["job_tag"]):
        workers = spawn_tasks(ResourceConfig(**arguments), command=command,
                              charged_group=config["charged_group"], use_image=True,
                              code_mount_point=config["workspace"])
        print("JOB_ID", workers.rjob.meta.name, "FINAL_STATUS", workers.poll(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
