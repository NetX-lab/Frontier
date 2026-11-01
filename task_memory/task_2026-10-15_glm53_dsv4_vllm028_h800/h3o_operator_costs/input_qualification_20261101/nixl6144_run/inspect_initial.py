"""Read-only snapshot of the submitted NIXL6144 RJob identity on the platform."""
import datetime
import json
import os
import socket
from pathlib import Path

from steptron.utils.stepmind import get_rjob_client

JOB = os.environ["NIXL6144_RJOB_NAME"]
OUT = Path(os.environ["NIXL6144_RUN_DIR"]) / "platform_initial.json"

client = get_rjob_client()
job = client.api.get_namespaced_custom_object(
    group=client.group, version=client.version, namespace=client.namespace,
    plural=client.plural, name=JOB, _request_timeout=30,
)
metadata = job["metadata"]
spec = job["spec"]
status = job.get("status") or {}
annotations = metadata.get("annotations") or {}
labels = metadata.get("labels") or {}

tasks = {}
for name, task in spec["taskSpecs"].items():
    pod = task["template"]["spec"]
    containers = []
    for container in pod["containers"]:
        limits = container.get("resources", {}).get("limits", {})
        containers.append({
            "name": container.get("name"),
            "image": container.get("image"),
            "limits": {k: v for k, v in limits.items() if k in (
                "cpu", "memory", "nvidia.com/gpu",
                "mellanox.com/mlnx_rdma", "rdma/mlnx_shared")},
            "env_names": sorted(entry.get("name") for entry in container.get("env") or []),
            "exp_id": next((entry.get("value") for entry in container.get("env") or []
                            if entry.get("name") == "EXP_ID"), None),
        })
    tasks[name] = {
        "replicas": int(task.get("replicas", task.get("replica", 1))),
        "positive_tags": task.get("positiveTags"),
        "filestore_volume": task["template"]["metadata"].get("annotations", {}).get(
            "kubebrain.brainpp.cn/filestore-volume"),
        "mount_annotation": task["template"]["metadata"].get("annotations", {}).get(
            "mount.brainpp.cn/juicefs"),
        "volumes": pod.get("volumes"),
        "containers": containers,
    }

record = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "operation": "read only",
    "host": socket.gethostname(),
    "name": metadata["name"],
    "uid": metadata.get("uid"),
    "creation_timestamp": metadata.get("creationTimestamp"),
    "creator": labels.get("kubebrain.brainpp.cn/creator"),
    "real_creator_id": labels.get("kubebrain.brainpp.cn/real-creator-id"),
    "preemptible_label": labels.get("quotagroup.brainpp.cn/preemptible"),
    "preemptible_spec": spec.get("preemptible"),
    "queue_name": annotations.get("stepmind.com/queue-name"),
    "queue_status": labels.get("stepmind.com/queue-status"),
    "charged_group_label": labels.get("quotagroup.brainpp.cn/quotagroup"),
    "charged_group_spec": spec.get("chargedGroup"),
    "workspace_label": labels.get("workspace.brainpp.cn/workspace"),
    "mount_annotation": tasks[next(iter(tasks))]["mount_annotation"],
    "stop_annotations": {k: v for k, v in annotations.items()
                         if any(w in k.lower() for w in ("stop", "reason", "manual"))},
    "phase": status.get("phase"),
    "tasks": tasks,
    "total_gpus": sum(t["replicas"] * sum(
        float(c["limits"].get("nvidia.com/gpu", 0)) for c in t["containers"])
        for t in tasks.values()),
}

for task in tasks.values():
    for container in task["containers"]:
        container["limits"].setdefault("mellanox.com/mlnx_rdma", None)
        container["limits"].setdefault("rdma/mlnx_shared", None)

assert record["creator"] == "i-fengyicheng", record["creator"]
assert record["preemptible_label"] == "yes", record["preemptible_label"]
assert record["total_gpus"] == 16, record["total_gpus"]
assert all(t["replicas"] == 2 for t in tasks.values())
assert "codesign-exp:/mnt/codesign-exp" in (record["mount_annotation"] or "")
assert all(t["positive_tags"] == ["H200"] for t in tasks.values())
assert all(t["containers"][0]["limits"]["mellanox.com/mlnx_rdma"] == "1"
           and t["containers"][0]["limits"]["rdma/mlnx_shared"] == "8"
           for t in tasks.values())
OUT.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
