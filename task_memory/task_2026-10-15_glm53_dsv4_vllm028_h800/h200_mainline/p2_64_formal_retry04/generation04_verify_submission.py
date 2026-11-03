"""Read the exact submitted RJob and its replicas; save redacted identity/spec evidence.

Read-only. Bounded by the caller's timeout. No secrets, no env dumps.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from brainpp.rjob import RJobClient

AUTH = Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')
JOB = sys.argv[1]
OUT = Path(sys.argv[2])

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
        'stop_annotations': {k: v for k, v in annotations.items()
                             if any(w in k.lower() for w in ('stop', 'manual', 'reason'))},
        'phase': status.get('phase'), 'startTime': status.get('startTime'),
        'finishTime': status.get('finishTime'),
        'preemptible_spec': spec.get('preemptible'),
        'private_machine': spec.get('privateMachine'),
    }
    tasks = {}
    for name, task in (spec.get('taskSpecs') or {}).items():
        template = task.get('template') or {}
        tmeta = template.get('metadata') or {}
        containers = (template.get('spec') or {}).get('containers') or []
        info = {
            'replicas': task.get('replicas'),
            'filestore_volume': (tmeta.get('annotations') or {}).get(
                'kubebrain.brainpp.cn/filestore-volume'),
            'restartPolicy': (template.get('spec') or {}).get('restartPolicy'),
            'containers': [],
        }
        for container in containers:
            limits = ((container.get('resources') or {}).get('limits') or {})
            command = container.get('command') or []
            args = container.get('args') or []
            joined = ' '.join([str(part) for part in list(command) + list(args)])
            info['containers'].append({
                'name': container.get('name'), 'image': container.get('image'),
                'limits': limits,
                'has_timeout_3600': 'timeout 3600' in joined,
                'command_mentions_codesign_exp': '/mnt/codesign-exp' in joined,
                'command_prefix_len': len(joined),
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
print(json.dumps({k: record[k] for k in ('observed_utc', 'finished_utc', 'errors')}))
print('phase', (record.get('job') or {}).get('phase'),
      'replicas', len(record['replicas']['rows']),
      'replica_errors', record['replicas']['errors'])
