"""Second bounded read of the exact RJob: mounts, volumes, GPU tag/affinity fields.

Read-only, redacted: no env, no secrets, no raw spec dump.
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
job = client.api.get_namespaced_custom_object(
    group=client.group, version=client.version, namespace=client.namespace,
    plural=client.plural, name=JOB, _request_timeout=(6, 25))
meta = job.get('metadata') or {}
spec = job.get('spec') or {}
record = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'job': JOB,
          'name': meta.get('name'), 'uid': meta.get('uid'),
          'annotations': {k: v for k, v in (meta.get('annotations') or {}).items()
                          if any(w in k.lower() for w in ('mount', 'filestore', 'storage',
                                                          'juicefs', 'queue', 'tag'))},
          'tasks': {}}
for name, task in (spec.get('taskSpecs') or {}).items():
    template = task.get('template') or {}
    tmeta = template.get('metadata') or {}
    tspec = template.get('spec') or {}
    containers = tspec.get('containers') or []
    record['tasks'][name] = {
        'task_annotations': {k: v for k, v in (tmeta.get('annotations') or {}).items()
                             if any(w in k.lower() for w in ('mount', 'filestore', 'storage',
                                                             'juicefs'))},
        'nodeSelector': tspec.get('nodeSelector'),
        'affinity': tspec.get('affinity'),
        'tolerations': tspec.get('tolerations'),
        'volumes': [{'name': v.get('name'),
                     'type': sorted(k for k in v if k != 'name'),
                     'claimName': (v.get('persistentVolumeClaim') or {}).get('claimName'),
                     'csi_driver': ((v.get('csi') or {}).get('driver')),
                     'hostPath': (v.get('hostPath') or {}).get('path')}
                    for v in (tspec.get('volumes') or [])],
        'containers': [{'name': c.get('name'),
                        'volumeMounts': c.get('volumeMounts'),
                        'env_names': sorted(e.get('name') for e in (c.get('env') or []))}
                       for c in containers],
    }
record['finished_utc'] = datetime.now(timezone.utc).isoformat()
OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print('mount_read_ok', record['name'], 'tasks', list(record['tasks']))
