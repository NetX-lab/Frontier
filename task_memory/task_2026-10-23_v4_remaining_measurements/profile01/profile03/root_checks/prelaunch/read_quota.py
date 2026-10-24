"""Read-only quota and node-capacity snapshot for H800/codesign and H200/step_main.

Emits numeric quota fields, per-group node GPU totals, and the caller's own live
RJob allocations. No submission, stop or job control.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import requests
from brainpp.rjob import RJobClient
AUTH = Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')
OUTPUT = Path('/data/ycfeng/tmp/glm53_dsv4/profile03_prelaunch_dnatg1eo/quota.json')
GROUPS = {'codesign': 'H800'}
QUOTA_FIELDS = ('totalQuota', 'usedQuota', 'nonPreemptibleUsedQuota', 'preemptibleUsedQuota', 'gpuCardNum', 'queueInfo')
SENSITIVE = ('secret', 'token', 'password', 'member', 'user', 'key', 'auth')
client = RJobClient(access_key=(AUTH / 'accesskey_id').read_text().strip(), secret_key=(AUTH / 'accesskey_secret').read_text().strip())
session = requests.Session()
session.auth = (client.username, client.password)
base = client.cluster_entry.rstrip('/')

def numbers(value):
    if isinstance(value, dict):
        return {k: numbers(v) for (k, v) in value.items() if not any((s in k.lower() for s in SENSITIVE))}
    if isinstance(value, list):
        return [numbers(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str) and len(value) < 40 and any((ch.isdigit() for ch in value)):
        return value
    return None

def gpu_count(resources):
    return sum((float(v) for (k, v) in (resources or {}).items() if 'gpu' in k.lower() and str(v).replace('.', '', 1).isdigit()))
receipt = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version.split()[0], 'operation': 'read_only', 'submissions_or_controls': 0, 'groups': {}}
for (group, tag) in GROUPS.items():
    row = {'gpu_tag': tag}
    response = session.get(f'{base}/kapis/quota.brainpp.cn/v1alpha1/tenants/shai/projects/shai-core/quotagroups/shai:shai-core:{group}', timeout=20)
    row['quota_http_status'] = response.status_code
    if response.status_code == 200:
        quota = response.json().get('data', {}).get('quotagroup', {})
        row['quota'] = {k: numbers(quota.get(k)) for k in QUOTA_FIELDS}
    response = session.get(f'{base}/kapis/node.brainpp.cn/v1alpha1/nodes/details', params={'tenant': 'shai', 'project': 'shai-core', 'group': group, 'page': 1, 'pageSize': 200, 'sortBy': 'gpu:des'}, timeout=30)
    row['node_http_status'] = response.status_code
    if response.status_code == 200:
        data = response.json().get('data', {})
        machines = data.get('machines', [])
        row['nodes'] = {'total_reported': data.get('total'), 'returned': len(machines), 'gpu_total': sum((gpu_count(m.get('total')) for m in machines)), 'gpu_usable': sum((gpu_count(m.get('usable')) for m in machines)), 'nodes_with_8_usable_gpus': sum((gpu_count(m.get('usable')) >= 8 for m in machines))}
    receipt['groups'][group] = row
items = client.api.list_namespaced_custom_object(group=client.group, version=client.version, namespace=client.namespace, plural=client.plural, label_selector='kubebrain.brainpp.cn/creator=i-fengyicheng', limit=250, _request_timeout=25)
live = []
for job in items.get('items', []):
    (meta, status) = (job['metadata'], job.get('status', {}))
    if status.get('phase', '').lower() in ('succeeded', 'failed', 'stopped', 'terminated'):
        continue
    gpus = 0
    for task in job.get('spec', {}).get('taskSpecs', {}).values():
        containers = task.get('template', {}).get('spec', {}).get('containers', [])
        gpus += (task.get('replicas') or 0) * sum((gpu_count(c.get('resources', {}).get('limits')) for c in containers))
    labels = meta.get('labels', {})
    live.append({'name': meta['name'], 'phase': status.get('phase'), 'charged_group': labels.get('quotagroup.brainpp.cn/quotagroup'), 'preemptible': labels.get('quotagroup.brainpp.cn/preemptible'), 'gpus': gpus})
receipt['personal_live_jobs'] = live
receipt['personal_list_continues'] = bool(items.get('metadata', {}).get('continue'))
receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
receipt['limits'] = ['Quota and node usage are snapshots, not reservations.', 'Scheduling after submission is the only proof of capacity.']
with OUTPUT.open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(receipt, indent=2, sort_keys=True))
