"""Read one personal RJob and its replica pods; keep only filtered, credential-free fields."""
import datetime
import json
from pathlib import Path
import sys

from brainpp.rjob import RJobClient

name, output = sys.argv[1], Path(sys.argv[2])
credentials = Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')
client = RJobClient(access_key=(credentials / 'accesskey_id').read_text().strip(),
                    secret_key=(credentials / 'accesskey_secret').read_text().strip())
job = client.api.get_namespaced_custom_object(group=client.group, version=client.version,
                                              namespace=client.namespace, plural=client.plural,
                                              name=name, _request_timeout=25)
meta = job['metadata']
tasks = {}
for task_name, task in job['spec']['taskSpecs'].items():
    pod = task['template']['spec']
    tasks[task_name] = {
        'replicas': task.get('replicas', task.get('replica')),
        'nfs': task['template']['metadata'].get('annotations', {}).get('kubebrain.brainpp.cn/filestore-volume'),
        'containers': [{'image': c.get('image'), 'resources': c.get('resources', {})} for c in pod['containers']],
    }
replicas = []
for rows in client.get_rjob_infos(name).values():
    for replica in rows:
        pod = client.api.get_namespaced_custom_object(group=client.group, version=client.version,
                                                      namespace=client.namespace, plural='replicas',
                                                      name=replica['name'], _request_timeout=25)
        status = pod.get('status') or {}
        replicas.append({
            'name': pod['metadata']['name'], 'uid': pod['metadata']['uid'],
            'owner_references': pod['metadata'].get('ownerReferences'),
            'node': (pod.get('spec') or {}).get('nodeName') or status.get('nodeName'),
            'phase': status.get('phase'),
            'container_statuses': [{
                'name': c.get('name'), 'image': c.get('image'), 'restart_count': c.get('restartCount'),
                'state': c.get('state'), 'last_state': c.get('lastState'),
            } for c in status.get('containerStatuses') or []],
        })
result = {
    'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'name': meta['name'], 'uid': meta['uid'],
    'creator': meta['labels'].get('kubebrain.brainpp.cn/creator'),
    'labels': {k: v for k, v in meta['labels'].items() if k.startswith('quotagroup.brainpp.cn/')},
    'queue': meta.get('annotations', {}).get('stepmind.com/queue-name'),
    'stop_annotations': {k: v for k, v in (meta.get('annotations') or {}).items()
                         if any(w in k.lower() for w in ('stop', 'manual'))},
    'status': job.get('status', {}), 'tasks': tasks, 'replicas': replicas,
}
with output.open('x') as stream:
    json.dump(result, stream, indent=2, default=str)
    stream.write('\n')
print(json.dumps({'name': result['name'], 'creator': result['creator'], 'labels': result['labels'],
                  'phase': result['status'].get('phase'),
                  'replicas': [(r['name'], r['node'], r['phase']) for r in replicas]}, default=str))
