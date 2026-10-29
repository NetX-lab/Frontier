"""Read current personal H800/codesign quota, node capacity and own job names."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import requests
from brainpp.rjob import RJobClient

assert os.environ["STEPMIND_BACKEND"] == "rjob"
client = RJobClient(access_key=os.environ["BRAINPP_ACCESS_KEY"], secret_key=os.environ["BRAINPP_SECRET_KEY"])
base = client.cluster_entry.rstrip("/")
output = Path(__file__).with_name("platform_snapshot.json")

def numeric(value):
    if isinstance(value, dict):
        return {k: numeric(v) for k, v in value.items() if not any(s in k.lower() for s in ("secret", "token", "password", "member", "user", "key", "auth"))}
    if isinstance(value, list):
        return [numeric(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return value if isinstance(value, str) and len(value) < 40 and any(c.isdigit() for c in value) else None

def get(path, params=None):
    session = requests.Session()
    session.auth = (client.username, client.password)
    reply = session.get(base + path, params=params, timeout=25)
    reply.raise_for_status()
    return reply.status_code, reply.json()

def own_jobs():
    items, continuation = [], None
    while True:
        options = {"_continue": continuation} if continuation else {}
        response = client.api.list_namespaced_custom_object(group=client.group, version=client.version, namespace=client.namespace, plural=client.plural, label_selector="kubebrain.brainpp.cn/creator=i-fengyicheng", limit=100, _request_timeout=25, **options)
        for job in response.get("items", []):
            meta = job["metadata"]
            labels = meta.get("labels", {})
            tasks = job.get("spec", {}).get("taskSpecs", {})
            mounts = {name: task["template"]["metadata"].get("annotations", {}).get("kubebrain.brainpp.cn/filestore-volume") for name, task in tasks.items()}
            items.append({"name": meta["name"], "creator": labels.get("kubebrain.brainpp.cn/creator"), "phase": job.get("status", {}).get("phase"), "charged_group": labels.get("quotagroup.brainpp.cn/quotagroup"), "preemptible": labels.get("quotagroup.brainpp.cn/preemptible"), "mounts": mounts})
        continuation = response.get("metadata", {}).get("continue")
        if not continuation:
            return items

began = datetime.now(timezone.utc).isoformat()
with ThreadPoolExecutor(max_workers=3) as pool:
    quota_future = pool.submit(get, "/kapis/quota.brainpp.cn/v1alpha1/tenants/shai/projects/shai-core/quotagroups/shai:shai-core:codesign")
    node_future = pool.submit(get, "/kapis/node.brainpp.cn/v1alpha1/nodes/details", {"tenant": "shai", "project": "shai-core", "group": "codesign", "page": 1, "pageSize": 200, "sortBy": "gpu:des"})
    jobs_future = pool.submit(own_jobs)
    quota_http, quota_body = quota_future.result()
    node_http, node_body = node_future.result()
    jobs = jobs_future.result()
quota = quota_body["data"]["quotagroup"]
node_data = node_body["data"]
machines = node_data.get("machines", [])
def gpu_count(resources):
    return sum(float(v) for k, v in (resources or {}).items() if "gpu" in k.lower() and str(v).replace(".", "", 1).isdigit())
workspace = "/data/ycfeng/Frontier/.worktrees/v4-scope-lifetime-package/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/scope_lifetime/h800_scope_lifetime_probe_01"
matching = [row for row in jobs if any(workspace in (mount or "") for mount in row["mounts"].values())]
receipt = {"status": "PASS_FRESH_READS", "start_utc": began, "end_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "creator": "i-fengyicheng", "group": "codesign", "tag": "H800", "quota_http": quota_http, "quota": {k: numeric(quota.get(k)) for k in ("totalQuota", "usedQuota", "nonPreemptibleUsedQuota", "preemptibleUsedQuota", "gpuCardNum", "queueInfo")}, "node_http": node_http, "nodes": {"total_reported": node_data.get("total"), "returned": len(machines), "gpu_total_in_returned_nodes": sum(gpu_count(m.get("total")) for m in machines), "gpu_usable_in_returned_nodes": sum(gpu_count(m.get("usable")) for m in machines), "nodes_with_at_least_one_usable_gpu": sum(gpu_count(m.get("usable")) >= 1 for m in machines)}, "personal_jobs_checked": len(jobs), "personal_list_complete": True, "matching_scope_workspace_jobs": matching, "personal_live_jobs": [row for row in jobs if (row["phase"] or "").lower() not in ("succeeded", "failed", "stopped", "terminated")], "submissions_or_controls": 0, "limits": "Current quota and node snapshots are not reservations; actual placement follows submission. Zero predicted capacity does not exclude the authorized FIFO queue."}
assert not matching, matching
with output.open("x") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
