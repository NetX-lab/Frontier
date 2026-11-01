"""Record the terminal platform state of the submitted NIXL6144 RJob (read only)."""
import datetime
import json
import os
from pathlib import Path

from steptron.utils.stepmind import get_rjob_client

JOB = os.environ["NIXL6144_RJOB_NAME"]
RUN = Path(os.environ["NIXL6144_RUN_DIR"])

client = get_rjob_client()
job = client.api.get_namespaced_custom_object(
    group=client.group, version=client.version, namespace=client.namespace,
    plural=client.plural, name=JOB, _request_timeout=30,
)
metadata = job["metadata"]
status = job.get("status") or {}
annotations = metadata.get("annotations") or {}
labels = metadata.get("labels") or {}

replicas = []
for record in client.list([JOB]):
    for task_name, task in record.spec.tasks.items():
        for replica in task.replicaStatus or []:
            replicas.append({
                "task": task_name,
                "replica": replica.name,
                "process_id": replica.process_id,
                "state": replica.state.name if hasattr(replica.state, "name") else replica.state,
                "ip": replica.ip,
                "node": replica.node_name,
                "node_ip": replica.node_ip,
            })

times = {key: annotations.get(key) for key in (
    "kubebrain.brainpp.cn/filestore-volume",
    "rjob.brainpp.cn/rjob-queue-approved",
    "rjob.brainpp.cn/rjob-pod-created",
    "rjob.brainpp.cn/rjob-scheduled",
    "rjob.brainpp.cn/rjob-running",
    "stepmind.com/queue-approved-stage",
    "stepmind.com/estimated-resource",
    "stepmind.com/final-resource",
    "stepmind.com/queue-reason",
)}
finish = status.get("finishTime")
started = times["rjob.brainpp.cn/rjob-running"]


def parse_time(value):
    """Parse the platform timestamps, whose fractional part has up to 9 digits."""
    if not value:
        raise ValueError("missing timestamp")
    body, _, offset = value.partition("Z")
    if "." in body:
        head, fraction = body.split(".", 1)
        body = f"{head}.{fraction[:6]}"
    return datetime.datetime.fromisoformat(body + "+00:00" if offset else body)


try:
    running = parse_time(started)
    finished = parse_time(finish)
    running_seconds = (finished - running).total_seconds()
    created = parse_time(metadata["creationTimestamp"])
    queued_seconds = (finished - created).total_seconds()
except (TypeError, ValueError):
    running_seconds = queued_seconds = None

record = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "operation": "read only",
    "name": JOB,
    "uid": metadata.get("uid"),
    "creator": labels.get("kubebrain.brainpp.cn/creator"),
    "preemptible_label": labels.get("quotagroup.brainpp.cn/preemptible"),
    "queue_name": annotations.get("stepmind.com/queue-name"),
    "queue_status": labels.get("stepmind.com/queue-status"),
    "phase": status.get("phase"),
    "phase_reason": status.get("reason"),
    "phase_message": status.get("message"),
    "creation_timestamp": metadata.get("creationTimestamp"),
    "job_finish_time": finish,
    "timeline_annotations": times,
    "running_seconds": running_seconds,
    "created_to_finish_seconds": queued_seconds,
    "container_gpu_hours": round(16 * running_seconds / 3600.0, 6) if running_seconds else None,
    "stop_annotations": {k: v for k, v in annotations.items()
                         if any(w in k.lower() for w in ("stop", "manual"))
                         or "manual-stop" in k.lower()},
    "task_statuses": status.get("taskStatuses"),
    "replicas": replicas,
    "replica_count": len(replicas),
    "replica_states": sorted({r["state"] for r in replicas}),
    "nodes": sorted({r["node"] for r in replicas}),
    "container_exit_codes": {
        "source": "replica state from the RJob CR",
        "value": {r["process_id"]: ("0 (replica SUCCEED; job Succeeded implies zero exit)"
                                    if r["state"] == "SUCCEED" else r["state"])
                  for r in replicas},
        "note": ("The personal credential is forbidden (HTTP 403) on pod list/read in this "
                 "namespace, so raw container exitCode/restartCount are not readable. "
                 "Restart policy is Never and the platform reports both replicas SUCCEED."),
    },
    "launcher_exit_receipt": json.loads(
        (RUN.parent.parent / "g5h/launch/receipts/"
         f"{os.environ['NIXL6144_JOB_TAG']}_exit.json").read_text()),
}
(RUN / "platform_terminal.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k: record[k] for k in (
    "name", "phase", "creation_timestamp", "job_finish_time", "running_seconds",
    "container_gpu_hours", "replica_states", "nodes", "stop_annotations")}, indent=2))
print(json.dumps(record["timeline_annotations"], indent=1))
