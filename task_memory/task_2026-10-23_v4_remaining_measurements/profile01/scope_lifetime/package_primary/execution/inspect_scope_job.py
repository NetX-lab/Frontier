"""Read one personal job's exact resource/status fields and bounded redacted logs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from brainpp.rjob import RJobClient

job_name, output_prefix = sys.argv[1:]
prefix = Path(output_prefix)
client = RJobClient(access_key=os.environ["BRAINPP_ACCESS_KEY"], secret_key=os.environ["BRAINPP_SECRET_KEY"])
job = client.api.get_namespaced_custom_object(group=client.group, version=client.version, namespace=client.namespace, plural=client.plural, name=job_name, _request_timeout=25)
meta, status = job["metadata"], job.get("status", {})
labels = meta.get("labels", {})
assert labels.get("kubebrain.brainpp.cn/creator") == "i-fengyicheng"
def redact(text):
    for key in ("BRAINPP_ACCESS_KEY", "BRAINPP_SECRET_KEY"):
        if os.environ.get(key):
            text = text.replace(os.environ[key], "<REDACTED>")
    return re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1<REDACTED>@", text)
tasks = {}
for name, task in job["spec"]["taskSpecs"].items():
    template = task["template"]
    annotations = template["metadata"].get("annotations", {})
    spec = template["spec"]
    tasks[name] = {"replicas": task.get("replicas"), "filestore_volume": annotations.get("kubebrain.brainpp.cn/filestore-volume"), "restart_policy": spec.get("restartPolicy"), "security_context": spec.get("securityContext"), "containers": [{"name": c.get("name"), "image": c.get("image"), "resources": c.get("resources"), "security_context": c.get("securityContext")} for c in spec.get("containers", [])]}
replicas = []
for rows in client.get_rjob_infos(job_name).values():
    for replica in rows:
        pod = client.api.get_namespaced_custom_object(group=client.group, version=client.version, namespace=client.namespace, plural="replicas", name=replica["name"], _request_timeout=25)
        state = pod.get("status", {})
        item = {"name": replica["name"], "phase": state.get("phase"), "node_name": state.get("nodeName"), "container_statuses": state.get("containerStatuses", []), "start_time": state.get("startTime"), "finish_time": state.get("finishTime"), "reason": state.get("reason"), "message": redact(state.get("message", ""))}
        response = client.logs_replica(replica["name"], tail_lines=5000)
        body = response.json()
        item["log_code"] = body.get("code")
        lines = body.get("data", [])
        log = redact("\n".join(line.get("message", line.get("line", "")) for line in lines))
        path = prefix.parent / (prefix.name + "." + replica["name"] + ".log")
        path.write_text(log + ("\n" if log else ""))
        item.update(log_path=str(path), log_rows=len(lines), gpu_names=re.findall(r"GPU \d+: ([^\n]+)", log))
        replicas.append(item)
receipt = {"observed_utc": datetime.now(timezone.utc).isoformat(), "job_name": job_name, "creator": labels.get("kubebrain.brainpp.cn/creator"), "charged_group": labels.get("quotagroup.brainpp.cn/quotagroup"), "preemptible": labels.get("quotagroup.brainpp.cn/preemptible"), "creation_timestamp": meta.get("creationTimestamp"), "phase": status.get("phase"), "start_time": status.get("startTime"), "finish_time": status.get("finishTime"), "reason": status.get("reason"), "message": redact(status.get("message", "")), "stop_annotations": {k: redact(str(v)) for k, v in meta.get("annotations", {}).items() if any(s in k.lower() for s in ("stop", "reason", "manual"))}, "tasks": tasks, "replicas": replicas, "operation": "read_only_personal_python_api", "controls_or_submissions": 0}
with prefix.with_suffix(".json").open("x") as output:
    json.dump(receipt, output, indent=2, sort_keys=True)
    output.write("\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
