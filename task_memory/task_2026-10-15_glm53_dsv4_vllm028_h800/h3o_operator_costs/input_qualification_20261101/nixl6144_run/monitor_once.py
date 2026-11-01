"""Append one read-only platform snapshot of the NIXL6144 RJob to monitor.jsonl."""
import datetime
import json
import os
import sys
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
snapshot = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "phase": status.get("phase"),
    "queue_status": (metadata.get("labels") or {}).get("stepmind.com/queue-status"),
    "queue_reason": annotations.get("stepmind.com/queue-reason"),
    "start_time": status.get("startTime"),
    "finish_time": status.get("finishTime"),
    "conditions": status.get("conditions"),
    "stop_annotations": {k: v for k, v in annotations.items()
                         if any(w in k.lower() for w in ("stop", "reason", "manual"))},
    "replicas": [],
}
for replica in client.get_rjob_infos(JOB).get(JOB, []) or []:
    replica_status = replica.get("status") or {}
    entry = {
        "name": replica.get("name") or replica.get("metadata", {}).get("name"),
        "phase": replica_status.get("phase"),
        "node": replica_status.get("nodeName")
        or replica.get("spec", {}).get("nodeName")
        or (replica.get("metadata") or {}).get("labels", {}).get("kubebrain.brainpp.cn/node"),
        "start_time": replica_status.get("startTime"),
        "containers": [],
    }
    for container in (
        (replica.get("spec") or {}).get("containers")
        or (replica.get("status") or {}).get("containerStatuses")
        or []
    ):
        state = (container.get("state") or {}) if "state" in container else {}
        last = (container.get("lastState") or {}) if "lastState" in container else {}
        entry["containers"].append({
            "name": container.get("name"),
            "restart_count": container.get("restartCount"),
            "state": {k: {kk: vv for kk, vv in (v or {}).items()
                          if kk in ("reason", "exitCode", "signal", "startedAt", "finishedAt")}
                      for k, v in state.items()},
            "last_state": {k: {kk: vv for kk, vv in (v or {}).items()
                               if kk in ("reason", "exitCode", "signal", "startedAt", "finishedAt")}
                           for k, v in last.items()},
        })
    snapshot["replicas"].append(entry)

with (RUN / "monitor.jsonl").open("a") as handle:
    handle.write(json.dumps(snapshot) + "\n")
print(json.dumps({k: snapshot[k] for k in (
    "observed_utc", "phase", "queue_status", "start_time", "finish_time")}),
    [{"name": r["name"], "phase": r["phase"], "node": r["node"],
      "containers": [(c["name"], c["restart_count"], c["state"], c["last_state"]) for c in r["containers"]]}
     for r in snapshot["replicas"]], flush=True)
sys.exit(0 if snapshot["phase"] not in ("Succeeded", "Failed", "Stopped") else 9)
