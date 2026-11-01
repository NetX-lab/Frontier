"""Record the task-4 monitoring checkpoint while the job is still queued.

Read-only: queries the submitted job and the personal live-job list, then writes
platform_checkpoint.json. platform_terminal.json stays absent until the job is
terminal.
"""
import datetime
import json
import os
from pathlib import Path

from steptron.utils.stepmind import get_rjob_client

JOB = os.environ["NIXL6144_RJOB_NAME"]
RUN = Path(os.environ["NIXL6144_RUN_DIR"])
LAUNCHER = 2600807

client = get_rjob_client()
job = client.api.get_namespaced_custom_object(
    group=client.group, version=client.version, namespace=client.namespace,
    plural=client.plural, name=JOB, _request_timeout=30,
)
status = job.get("status") or {}
metadata = job["metadata"]
annotations = metadata.get("annotations") or {}

live = []
page = client.api.list_namespaced_custom_object(
    group=client.group, version=client.version, namespace=client.namespace,
    plural=client.plural, label_selector="kubebrain.brainpp.cn/creator=i-fengyicheng",
    limit=100, _request_timeout=30,
)
for item in page["items"]:
    item_status = item.get("status") or {}
    if item_status.get("phase") in ("Succeeded", "Failed", "Stopped"):
        continue
    labels = item["metadata"].get("labels") or {}
    live.append({"name": item["metadata"]["name"], "phase": item_status.get("phase"),
                 "preemptible": labels.get("quotagroup.brainpp.cn/preemptible")})

monitor_rows = [json.loads(line) for line in (RUN / "monitor.jsonl").read_text().splitlines()]
receipts = RUN.parent.parent / "g5h/launch/receipts"
record = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checkpoint_reason": "monitoring budget of 120 minutes from 2026-11-01T19:00:22Z exhausted",
    "job": JOB,
    "phase": status.get("phase"),
    "queue_status": (metadata.get("labels") or {}).get("stepmind.com/queue-status"),
    "queue_reason": annotations.get("stepmind.com/queue-reason"),
    "start_time": status.get("startTime"),
    "finish_time": status.get("finishTime"),
    "stop_annotations": {k: v for k, v in annotations.items()
                         if any(w in k.lower() for w in ("stop", "reason", "manual"))},
    "replicas": [],
    "monitor_snapshots": len(monitor_rows),
    "first_snapshot_utc": monitor_rows[0]["observed_utc"],
    "last_snapshot_utc": monitor_rows[-1]["observed_utc"],
    "phase_history": sorted({row["phase"] for row in monitor_rows}),
    "launcher_pid": LAUNCHER,
    "launcher_alive": Path(f"/proc/{LAUNCHER}").exists(),
    "launcher_rjob_name_logged": "RJOB_NAME" in Path(
        "/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/launcher.log").read_text(),
    "receipts": {suffix: (receipts / f"{os.environ['NIXL6144_JOB_TAG']}_{suffix}.json").exists()
                 for suffix in ("started", "lock", "exit")},
    "live_personal_jobs": live,
    "next_action": ("Continue polling the same job; do not resubmit. When it is terminal, "
                    "write platform_terminal.json, download the cloud artifacts, and run "
                    "verify_nixl6144.py."),
}
(RUN / "platform_checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
