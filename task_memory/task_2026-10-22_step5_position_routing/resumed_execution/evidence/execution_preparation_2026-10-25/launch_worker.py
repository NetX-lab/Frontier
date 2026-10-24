"""Submit the one recorded zero-GPU readiness worker from this local NFS source."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    directory = args.plan.resolve().parent
    if plan["resource"]["gpu"] != 0 or plan["lifecycle"]["job_limit"] != 1:
        raise AssertionError("This launcher is restricted to one zero-GPU readiness worker")
    for name, expected in plan["scripts"].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Prepared worker script changed: {name}")
    command = shlex.join([plan["python"], "-B", str(directory/"worker_entry.py"),
                          "--plan", str(args.plan.resolve())])
    if not args.submit:
        print(json.dumps({"status": "DRY_RUN_NO_PLATFORM_OPERATIONS", "resource": plan["resource"],
            "image": plan["image"], "charged_group": plan["charged_group"], "command": command,
            "nfs_source": plan["local_mount_source"], "mounts": plan["mounts"]}, indent=2))
        return
    if socket.gethostname() != plan["submitted_from"]:
        raise AssertionError("Submission must originate on the current authorized local host")
    if os.environ.get("STEPMIND_BACKEND") != "rjob":
        raise AssertionError("Use the original Python RJobBackend")
    if not os.environ.get("BRAINPP_ACCESS_KEY") or not os.environ.get("BRAINPP_SECRET_KEY"):
        raise AssertionError("Personal credentials must be configured before client creation")
    from joint_timing import check_bindings
    bindings = json.loads(Path(plan["joint_bindings"]).read_bytes())
    identity = check_bindings(bindings, directory)
    output = Path(plan["local_output_root"])
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "SUBMITTING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "launcher_pid": os.getpid(), "source": identity, "command": command,
        "nfs_server": socket.gethostbyname(socket.gethostname()),
        "nfs_source": plan["local_mount_source"], "nfs_target": plan["local_mount_target"],
        "resource": plan["resource"], "image": plan["image"], "charged_group": plan["charged_group"]}
    (output/"launcher.json").write_text(json.dumps(receipt, indent=2) + "\n")
    started = time.monotonic()
    try:
        sys.path.insert(0, plan["steptron_root"])
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import get_rjob_client, spawn_tasks
        os.environ["EXP_ID"] = plan["experiment_id"]
        os.chdir(plan["local_mount_source"])
        config = ResourceConfig(**plan["resource"], image=plan["image"], mounts=plan["mounts"],
            envs={"DISTRIBUTED_JOB": "false", "PYTHONDONTWRITEBYTECODE": "1",
                  **plan["math_thread_env"]})
        workers = spawn_tasks(config, command=command, charged_group=plan["charged_group"],
            use_image=True, code_mount_point=plan["local_mount_target"])
        name = workers.rjob.meta.name
        receipt["job_name"] = name
        print("RJOB_NAME", name, flush=True)
        client = get_rjob_client()
        job = client.api.get_namespaced_custom_object(group=client.group, version=client.version,
            namespace=client.namespace, plural=client.plural, name=name, _request_timeout=plan["inspection_request_timeout_s"])
        metadata = job["metadata"]
        receipt["creator"] = metadata["labels"].get("kubebrain.brainpp.cn/creator")
        receipt["platform_status"] = job.get("status", {})
        receipt["actual_nfs_mounts"] = {name: task["template"]["metadata"].get("annotations", {}).get(
            "kubebrain.brainpp.cn/filestore-volume") for name, task in job["spec"]["taskSpecs"].items()}
        receipt["actual_images"] = {task_name: [container["image"]
            for container in task["template"]["spec"]["containers"]]
            for task_name, task in job["spec"]["taskSpecs"].items()}
        receipt["actual_resources"] = {task_name: [container.get("resources", {})
            for container in task["template"]["spec"]["containers"]]
            for task_name, task in job["spec"]["taskSpecs"].items()}
        receipt["replicas"] = [{"task": task_name, "process_id": replica.process_id,
                                "state": str(replica.state)}
            for record in client.list([name]) for task_name, task in record.spec.tasks.items()
            for replica in task.replicaStatus or []]
        if not receipt["actual_images"] or any(images != [plan["image"]]
                for images in receipt["actual_images"].values()):
            raise AssertionError("Platform image differs from the pinned preparation image")
        if receipt["creator"] != plan["principal"]:
            raise AssertionError("Wrong platform creator")
        expected_mount = f'{receipt["nfs_server"]}:{plan["local_mount_source"]}:{plan["local_mount_target"]}'
        if set(receipt["actual_nfs_mounts"].values()) != {expected_mount}:
            raise AssertionError("Platform source mount differs from this local source")
        receipt["final_poll"] = workers.poll()
        readiness = json.loads((output/"worker/readiness.json").read_bytes())
        if receipt["final_poll"] != "succeeded" or readiness["status"] != "PASS_WORKER_IMPORT_AND_SERIALIZED_APPEND_ONLY":
            raise AssertionError("Worker business or terminal status did not pass")
        receipt["status"] = "PASS_PLATFORM_AND_WORKER_READINESS_ONLY"
    except BaseException as error:
        receipt["status"] = "FAIL"
        receipt["error"] = f"{type(error).__name__}: {error}"
        (output/"launcher.failure.log").write_text(traceback.format_exc())
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["launcher_wall_s"] = time.monotonic()-started
        (output/"launcher.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "job_name": receipt["job_name"],
                      "creator": receipt["creator"], "final_poll": receipt["final_poll"]}))


if __name__ == "__main__":
    main()
