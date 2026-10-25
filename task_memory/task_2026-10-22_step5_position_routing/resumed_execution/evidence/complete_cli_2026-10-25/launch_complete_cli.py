"""Submit one complete CPU-fixture CLI worker through the original personal RJobBackend."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import time
import traceback

from complete_cli_worker import read_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    plan, plan_sha256 = read_plan(args.plan)
    directory = args.plan.resolve().parent
    command = shlex.join([plan["python"], "-B", str(directory/"complete_cli_worker.py"), "--plan", str(args.plan.resolve())])
    if not args.submit:
        print(json.dumps({"status": "DRY_RUN_NO_PLATFORM_OR_WORKLOAD_CALLS", "command": command,
            "full_cli_command": plan["command"], "resource": plan["resource"], "image": plan["image"],
            "nfs_source": plan["local_mount_source"], "mounts": plan["mounts"], "outer_timeout_s": None,
            "plan_sha256": plan_sha256}, indent=2))
        return
    if socket.gethostname() != plan["submitted_from"] or os.environ.get("STEPMIND_BACKEND") != "rjob":
        raise AssertionError("Use original local RJobBackend on the authorized host")
    if not os.environ.get("BRAINPP_ACCESS_KEY") or not os.environ.get("BRAINPP_SECRET_KEY"):
        raise AssertionError("Personal credentials must be configured before creating a client")
    preparation = Path(plan["preparation_directory"])
    sys.path.insert(0, str(preparation))
    from joint_timing import check_bindings
    bindings = json.loads(Path(plan["joint_bindings"]).read_bytes())
    identity = check_bindings(bindings, preparation)
    output = Path(plan["local_output_root"])
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "SUBMITTING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "launcher_pid": os.getpid(), "source": identity, "command": command, "resource": plan["resource"],
        "image": plan["image"], "nfs_server": socket.gethostbyname(socket.gethostname()),
        "nfs_source": plan["local_mount_source"], "nfs_target": plan["local_mount_target"],
        "plan_sha256": plan_sha256, "charged_group": plan["charged_group"]}
    (output/"launcher.json").write_text(json.dumps(receipt, indent=2)+"\n")
    started = time.monotonic()
    launch_error = None
    try:
        sys.path.insert(0, plan["steptron_root"])
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import get_rjob_client, spawn_tasks
        os.environ["EXP_ID"] = plan["experiment_id"]
        os.chdir(plan["local_mount_source"])
        config = ResourceConfig(**plan["resource"], image=plan["image"], mounts=list(plan["mounts"]),
            envs={"DISTRIBUTED_JOB": "false", "PYTHONDONTWRITEBYTECODE": "1", **plan["math_thread_env"]})
        try:
            workers = spawn_tasks(config, command=command, charged_group=plan["charged_group"],
                use_image=True, code_mount_point=plan["local_mount_target"])
            receipt["job_name"] = workers.rjob.meta.name
            receipt["final_poll"] = workers.poll()
        except Exception as error:
            launch_error = error
            (output/"launcher.failure.log").write_text(traceback.format_exc())
            start_record = output/"worker/worker_started.json"
            if start_record.exists():
                receipt["job_name"] = json.loads(start_record.read_bytes())["job_name"]
        if receipt.get("job_name"):
            client = get_rjob_client()
            job = client.api.get_namespaced_custom_object(group=client.group, version=client.version,
                namespace=client.namespace, plural=client.plural, name=receipt["job_name"],
                _request_timeout=plan["inspection_request_timeout_s"])
            receipt["creator"] = job["metadata"]["labels"].get("kubebrain.brainpp.cn/creator")
            receipt["platform_status"] = job.get("status", {})
            receipt["platform_labels"] = {key: job["metadata"]["labels"].get(key)
                for key in ("quotagroup.brainpp.cn/quotagroup", "quotagroup.brainpp.cn/preemptible")}
            receipt["actual_nfs_mounts"] = {name: task["template"]["metadata"].get("annotations", {}).get(
                "kubebrain.brainpp.cn/filestore-volume") for name, task in job["spec"]["taskSpecs"].items()}
            receipt["actual_images"] = {name: [container["image"] for container in task["template"]["spec"]["containers"]]
                for name, task in job["spec"]["taskSpecs"].items()}
            receipt["actual_resources"] = {name: [container.get("resources", {}) for container in task["template"]["spec"]["containers"]]
                for name, task in job["spec"]["taskSpecs"].items()}
            expected_mount = f'{receipt["nfs_server"]}:{plan["local_mount_source"]}:{plan["local_mount_target"]}'
            if receipt["creator"] != plan["principal"] or set(receipt["actual_nfs_mounts"].values()) != {expected_mount}:
                raise AssertionError("The terminal job identity/source differs from the approved plan")
            if not receipt["actual_images"] or any(images != [plan["image"]] for images in receipt["actual_images"].values()):
                raise AssertionError("The terminal image differs from the pinned plan")
            expected_resources = {"cpu": str(plan["resource"]["cpu"]),
                "memory": f'{plan["resource"]["mem_gb"] * 1000}Mi',
                "nvidia.com/gpu": str(plan["resource"]["gpu"])}
            if receipt["actual_resources"] != {"default": [{"limits": expected_resources,
                                                           "requests": expected_resources}]}:
                raise AssertionError("The terminal resource requests/limits differ from the plan")
            if set(job["spec"]["taskSpecs"]) != {"default"} or job["spec"]["taskSpecs"]["default"]["replicas"] != 1:
                raise AssertionError("The complete CLI requires one default replica")
            if receipt["platform_labels"] != {"quotagroup.brainpp.cn/quotagroup": plan["charged_group"],
                                               "quotagroup.brainpp.cn/preemptible": "no"}:
                raise AssertionError("The terminal quota/preemptible configuration differs from the plan")
            from brainpp.rjob.struct import label_rjobname
            replicas = client.api.list_namespaced_custom_object(group=client.group, version=client.version,
                namespace=client.namespace, plural="replicas", label_selector=f'{label_rjobname}={receipt["job_name"]}',
                limit=2, _request_timeout=plan["inspection_request_timeout_s"])["items"]
            if len(replicas) != 1:
                raise AssertionError("Expected exactly one terminal replica")
            replica = replicas[0]
            receipt["replica_name"] = replica["metadata"]["name"]
            receipt["replica_status"] = replica.get("status", {})
        result_path = output/"worker/complete_cli.json"
        business = json.loads(result_path.read_bytes()) if result_path.exists() else None
        receipt["business_status"] = business["status"] if business else "UNAVAILABLE"
        passed = launch_error is None and receipt.get("final_poll") == "succeeded"
        passed = passed and receipt["business_status"] == "PASS_COMPLETE_CPU_FIXTURE_CLI"
        if passed:
            if receipt["platform_status"].get("phase") != "Succeeded" or receipt["replica_status"].get("phase") != "Succeeded":
                raise AssertionError("Platform and replica must both be terminal Succeeded")
            containers = receipt["replica_status"].get("containerStatuses", [])
            if len(containers) != 1:
                raise AssertionError("Expected one terminal container")
            container = containers[0]
            if (container["state"].get("terminated", {}).get("exitCode") != 0 or
                    container["state"]["terminated"].get("reason") != "Completed" or
                    container["restartCount"] != 0 or container["imageID"] != plan["image"]):
                raise AssertionError("The terminal container did not complete cleanly on the pinned image")
            if business["source_before"] != business["source_after"] or business["source_before"] != identity:
                raise AssertionError("The completed worker source differs from the local submission")
            if receipt["replica_status"]["nodeName"] != business["worker_hostname"]:
                raise AssertionError("The terminal replica node differs from the business worker")
            if business["plan_sha256"] != plan_sha256:
                raise AssertionError("The completed worker plan differs from the local submission")
        receipt["status"] = "PASS_COMPLETE_CPU_FIXTURE_PLATFORM_AND_BUSINESS" if passed else "FAIL"
        if launch_error is not None:
            receipt["error"] = f"{type(launch_error).__name__}: {launch_error}"
    except BaseException as error:
        receipt["status"] = "FAIL"
        receipt["error"] = f"{type(error).__name__}: {error}"
        (output/"launcher.failure.log").write_text(traceback.format_exc())
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["launcher_wall_s"] = time.monotonic()-started
        (output/"launcher.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({key:receipt.get(key) for key in ("status", "job_name", "creator", "business_status", "final_poll")}))
    if receipt["status"] != "PASS_COMPLETE_CPU_FIXTURE_PLATFORM_AND_BUSINESS":
        raise RuntimeError("Complete CPU-fixture CLI did not pass; inspect the retained exact receipts")


if __name__ == "__main__":
    main()
