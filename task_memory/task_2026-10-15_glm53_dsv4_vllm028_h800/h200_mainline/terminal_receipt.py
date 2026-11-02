"""Write the platform terminal receipt of one finished RJob, in the form native06 retained.

    python terminal_receipt.py <rjob name> <output json>

Read-only, with the personal StepMind client. The R2/R3/R4 `terminal_status` predicate reads
this file next to the run's `launcher_receipt.json`. Only the fields below are taken from the
job and its replicas; the pod spec (commands, environment) is never copied.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from brainpp.rjob import RJobClient

AUTH = Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')


def seconds(start, finish):
    parse = lambda value: datetime.fromisoformat(value.replace('Z', '+00:00'))
    return (parse(finish) - parse(start)).total_seconds()


def container(status):
    terminated = (status.get('state') or {}).get('terminated')
    row = dict(name=status['name'], restartCount=status['restartCount'], imageID=status['imageID'],
               state=dict(terminated={key: terminated.get(key) for key in
                                      ('exitCode', 'reason', 'startedAt', 'finishedAt', 'containerID')})
               if terminated else {key: None for key in status.get('state') or {}})
    if terminated:
        row['duration_seconds'] = seconds(terminated['startedAt'], terminated['finishedAt'])
    return row


def main(name, output):
    client = RJobClient(access_key=(AUTH / 'accesskey_id').read_text().strip(),
                        secret_key=(AUTH / 'accesskey_secret').read_text().strip())
    job = client.api.get_namespaced_custom_object(group=client.group, version=client.version,
                                                  namespace=client.namespace, plural=client.plural, name=name,
                                                  _request_timeout=40)
    labels, status = job['metadata'].get('labels') or {}, job.get('status') or {}
    tasks = {task: dict(replicas=spec['replicas'],
                        resources=[{key: container_spec['resources'].get(key) for key in ('limits', 'requests')}
                                   for container_spec in spec['template']['spec']['containers']])
             for task, spec in job['spec']['taskSpecs'].items()}
    replicas = []
    for rows in client.get_rjob_infos(name).values():
        for row in rows:
            pod = client._get_replica_kapis(row['name'])
            replicas.append(dict(name=row['name'], phase=pod['status']['phase'], node=pod['status'].get('nodeName'),
                                 containers=[container(c) for c in pod['status'].get('containerStatuses') or []]))
    gpus = {task: int(row['resources'][0]['limits']['nvidia.com/gpu']) for task, row in tasks.items()}
    [gpu] = set(gpus.values())
    receipt = dict(
        observed_utc=datetime.now(timezone.utc).isoformat(), job=name,
        creator=labels.get('kubebrain.brainpp.cn/creator'),
        charged_group=labels.get('quotagroup.brainpp.cn/quotagroup'),
        preemptible_label=labels.get('quotagroup.brainpp.cn/preemptible'),
        status={key: status.get(key) for key in ('phase', 'startTime', 'finishTime')},
        tasks=tasks, replicas=sorted(replicas, key=lambda replica: replica['name']),
        container_gpu_hours=sum(c.get('duration_seconds', 0) for r in replicas for c in r['containers']) * gpu / 3600)
    Path(output).write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps(dict(job=name, phase=receipt['status']['phase'], replicas=len(replicas),
                          container_gpu_hours=receipt['container_gpu_hours'])))


if __name__ == '__main__':
    main(*sys.argv[1:3])
