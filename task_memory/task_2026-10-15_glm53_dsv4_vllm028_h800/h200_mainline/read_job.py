"""Read the exact submitted RJob and its replicas once; save redacted identity, spec and mount evidence.

Usage: read_job.py <job name> <output json>. Read-only; no secrets, env values or raw spec dump.
Combines the generation04 identity and mount verifiers into one read and adds the GPU tag and durations.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from brainpp.rjob import RJobClient

AUTH = Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')
JOB = sys.argv[1]
OUT = Path(sys.argv[2])
MOUNT_WORDS = ('mount', 'filestore', 'storage', 'juicefs')


def selected(mapping, words):
    return {k: v for k, v in (mapping or {}).items() if any(w in k.lower() for w in words)}


client = RJobClient(access_key=(AUTH / 'accesskey_id').read_text().strip(),
                    secret_key=(AUTH / 'accesskey_secret').read_text().strip())
record = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'job_requested': JOB,
          'operation': 'read_only', 'errors': []}
try:
    job = client.api.get_namespaced_custom_object(
        group=client.group, version=client.version, namespace=client.namespace,
        plural=client.plural, name=JOB, _request_timeout=(6, 25))
except Exception as error:
    record['errors'].append({'call': 'get_job', 'error': repr(error)})
    job = None

if job:
    meta = job.get('metadata') or {}
    labels = meta.get('labels') or {}
    annotations = meta.get('annotations') or {}
    status = job.get('status') or {}
    spec = job.get('spec') or {}
    record['job'] = {
        'name': meta.get('name'), 'uid': meta.get('uid'), 'namespace': meta.get('namespace'),
        'creationTimestamp': meta.get('creationTimestamp'),
        'creator': labels.get('kubebrain.brainpp.cn/creator'),
        'preemptible': labels.get('quotagroup.brainpp.cn/preemptible'),
        'quotagroup': labels.get('quotagroup.brainpp.cn/quotagroup'),
        'restartable': labels.get('brainpp.cn/restartable'),
        'queue_name': annotations.get('stepmind.com/queue-name'),
        'queue_status': labels.get('stepmind.com/queue-status'),
        'job_annotations': selected(annotations, MOUNT_WORDS + ('queue', 'tag')),
        'stop_annotations': selected(annotations, ('stop', 'manual', 'reason')),
        'phase': status.get('phase'), 'startTime': status.get('startTime'),
        'finishTime': status.get('finishTime'),
        'autoDeleteDuration': spec.get('autoDeleteDuration'),
        'backoffLimit': spec.get('backoffLimit'),
    }
    tasks = {}
    for name, task in (spec.get('taskSpecs') or {}).items():
        template = task.get('template') or {}
        tmeta = template.get('metadata') or {}
        tspec = template.get('spec') or {}
        info = {
            'replicas': task.get('replicas'),
            'positiveTags': task.get('positiveTags'),
            'maxRunningDuration': task.get('maxRunningDuration'),
            'maxWaitDuration': task.get('maxWaitDuration'),
            'task_annotations': selected(tmeta.get('annotations'), MOUNT_WORDS),
            'restartPolicy': tspec.get('restartPolicy'),
            'volumes': [{'name': v.get('name'), 'type': sorted(k for k in v if k != 'name')}
                        for v in (tspec.get('volumes') or [])],
            'containers': [],
        }
        for container in tspec.get('containers') or []:
            joined = ' '.join(str(part) for part in
                              list(container.get('command') or []) + list(container.get('args') or []))
            info['containers'].append({
                'name': container.get('name'), 'image': container.get('image'),
                'limits': (container.get('resources') or {}).get('limits') or {},
                'command_timeouts_s': [int(v) for v in re.findall(r'\btimeout\s+(\d+)', joined)],
                'volumeMounts': [m.get('mountPath') for m in container.get('volumeMounts') or []],
                'env_names': sorted(e.get('name') for e in container.get('env') or []),
            })
        tasks[name] = info
    record['job']['taskSpecs'] = tasks

record['replicas'] = {'rows': [], 'errors': []}
try:
    for rows in client.get_rjob_infos(JOB).values():
        for row in rows:
            record['replicas']['rows'].append({
                'name': row.get('name'), 'phase': row.get('phase'), 'ip': row.get('ip'),
                'node': row.get('node') or row.get('nodeName'),
                'created': row.get('created') or row.get('creationTimestamp'),
            })
except Exception as error:
    record['replicas']['errors'].append(repr(error))

record['finished_utc'] = datetime.now(timezone.utc).isoformat()
OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print('read_job', JOB, 'phase', (record.get('job') or {}).get('phase'), 'errors', len(record['errors']),
      'replicas', len(record['replicas']['rows']), 'replica_errors', len(record['replicas']['errors']))
