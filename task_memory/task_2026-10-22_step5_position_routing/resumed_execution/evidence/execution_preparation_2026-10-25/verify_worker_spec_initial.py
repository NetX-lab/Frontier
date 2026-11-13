"""Materialize the original SDK resource/task objects without creating a client or job."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import socket
import sys

from steptron.exp.base_exp import ResourceConfig
from steptron.utils.stepmind import RJob

base = Path(__file__).resolve().parent
plan = json.loads((base / "worker_plan.json").read_bytes())
os.environ["EXP_ID"] = plan["experiment_id"] + "-spec-only"
server = socket.gethostbyname(socket.gethostname())
mount = f'{server}:{plan["local_mount_source"]}:{plan["local_mount_target"]}'
config = ResourceConfig(**plan["resource"], image=plan["image"], mounts=plan["mounts"],
    envs={"DISTRIBUTED_JOB": "false", "PYTHONPATH": plan["local_mount_target"],
          "PYTHONDONTWRITEBYTECODE": "1", **plan["math_thread_env"]},
    annotations={"kubebrain.brainpp.cn/filestore-volume": mount}, pull_code=False)
command = shlex.join([plan["python"], "-B", str(base / "worker_entry.py"),
                     "--plan", str(base / "worker_plan.json")])
job = RJob(name="D29-prepared-spec-only", charged_group=plan["charged_group"], cfg=config)
for task_name, task_config in config:
    entrypoint = task_config.command.format(TORCHRUN="/jobutils/scripts/torchrun.sh", COMMAND=command)
    entrypoint = "bash -lc " + shlex.quote(f'cd {shlex.quote(plan["local_mount_target"])} && exec {entrypoint}')
    job.add(command=entrypoint, task_name=task_name, cfg=task_config, charged_group=plan["charged_group"])
assert len(job.tasks) == 1
report = {"status": "PASS_ORIGINAL_SDK_OBJECTS_NO_CLIENT_OR_SUBMISSION",
    "collected_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
    "nfs_annotation": mount, "task_objects": {}, "new_platform_operations": 0}
for name, task in job.tasks.items():
    template = task.template
    assert template.image == plan["image"]
    assert task.replicas == plan["resource"]["replica"] and task.is_critical
    resources = template.resources
    report["task_objects"][name] = {"template_image": template.image,
        "resources": str(resources), "replicas": task.replicas, "is_critical": task.is_critical,
        "private_machine": str(task.private_machine), "mount_config": task.mount_config,
        "affinity": str(template.affinity), "command": template.command,
        "container_count": len(template.containers)}
    assert str(resources.cpu) == str(plan["resource"]["cpu"])
    assert str(resources.gpu) == str(plan["resource"]["gpu"])
    assert task.mount_config == plan["mounts"]
print(json.dumps(report, indent=2))
