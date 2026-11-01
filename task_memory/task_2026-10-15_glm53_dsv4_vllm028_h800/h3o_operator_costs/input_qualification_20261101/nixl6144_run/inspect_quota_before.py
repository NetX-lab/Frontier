"""Read-only listing of personal live RJobs before submitting the NIXL6144 job.

Records phase, GPU reservations, and preemptibility labels per live job so the
non-preemptible H200 ceiling of 32 GPUs can be checked, and confirms that no
live job already carries the target job tag.
"""
import datetime
import json
import os
from pathlib import Path

from steptron.utils.stepmind import get_rjob_client

TAG = os.environ["NIXL6144_JOB_TAG"]
OUT = Path(os.environ["NIXL6144_RUN_DIR"]) / "quota_before.json"

client = get_rjob_client()
result = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "operation": "read only",
    "creator_filter": "i-fengyicheng",
    "job_tag": TAG,
    "live_jobs": [],
    "tag_matches": [],
    "pages": 0,
}
continuation = ""
while True:
    page = client.api.list_namespaced_custom_object(
        group=client.group, version=client.version, namespace=client.namespace,
        plural=client.plural, label_selector="kubebrain.brainpp.cn/creator=i-fengyicheng",
        limit=100, _continue=continuation, _request_timeout=20,
    )
    result["pages"] += 1
    for job in page["items"]:
        metadata = job["metadata"]
        status = job.get("status") or {}
        phase = status.get("phase")
        spec = job.get("spec") or {}
        labels = metadata.get("labels") or {}
        annotations = metadata.get("annotations") or {}
        gpus = 0
        env_values = []
        task_tags = set()
        for task in (spec.get("taskSpecs") or {}).values():
            task_tags.update(task.get("positiveTags") or [])
            pod = task["template"]["spec"]
            per_pod = sum(
                float(container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
                for container in pod["containers"]
            )
            replicas = int(task.get("replicas", task.get("replica", 1)))
            gpus += per_pod * replicas
            for container in pod["containers"]:
                for entry in container.get("env") or []:
                    if entry.get("name") in ("EXP_ID", "STEPMIND_STAGE"):
                        env_values.append({entry["name"]: entry.get("value")})
        record = {
            "name": metadata["name"],
            "phase": phase,
            "gpu_reservations": gpus,
            "positive_tags": sorted(task_tags),
            "preemptible_label": labels.get("quotagroup.brainpp.cn/preemptible"),
            "preemptible_spec": spec.get("preemptible"),
            "queue_name": annotations.get("stepmind.com/queue-name"),
            "charged_group": labels.get("quotagroup.brainpp.cn/quotagroup")
            or annotations.get("quotagroup.brainpp.cn/quotagroup"),
            "exp_id_envs": env_values,
        }
        if phase not in ("Succeeded", "Failed", "Stopped"):
            result["live_jobs"].append(record)
        if TAG in json.dumps(env_values) or TAG in metadata["name"]:
            result["tag_matches"].append(record)
    continuation = (page.get("metadata") or {}).get("continue", "")
    if not continuation:
        break

live = result["live_jobs"]
result["live_job_count"] = len(live)
result["live_gpu_total"] = sum(job["gpu_reservations"] for job in live)
result["live_h200_jobs"] = [job for job in live if "H200" in (job["positive_tags"] or [])]
result["live_nonpreemptible_h200_gpus"] = sum(
    job["gpu_reservations"] for job in result["live_h200_jobs"]
    if job["preemptible_label"] != "yes"
)
result["live_preemptible_h200_gpus"] = sum(
    job["gpu_reservations"] for job in result["live_h200_jobs"]
    if job["preemptible_label"] == "yes"
)
assert not result["tag_matches"], "target job tag already present on the platform"

try:
    quota = client.api.get_cluster_custom_object(
        group="quotagroup.brainpp.cn", version="v1", plural="quotagroups",
        name="step_main", _request_timeout=20,
    )
    result["quota"] = {
        "spec": {k: v for k, v in (quota.get("spec") or {}).items() if "quota" in k.lower()},
        "status": quota.get("status") or {},
    }
except Exception as error:  # quota view is informational only
    result["quota_error"] = {"type": type(error).__name__, "status": getattr(error, "status", None)}

OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "quota"}, indent=2))
print("quota_status_keys", list((result.get("quota") or {}).get("status", {})))
