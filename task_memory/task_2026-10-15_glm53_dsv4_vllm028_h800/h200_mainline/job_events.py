"""One bounded read-only evidence pass for one exact RJob and its exact replicas.

Usage (with the environment of stepmind_env.sh):
    python job_events.py <job name> <job uid> <output dir>

Sources: live Kubernetes Events, historical Events (full pagination), all-operation audit
(rjobs + each replica, full pagination). Every query records PASS/FAIL separately from the
job status. No DeepScan, no Sherlock, no other job.
"""
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests
from brainpp.rjob import RJobClient

AUTH = Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')
JOB, JOB_UID, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
PAGE_SIZE = 100
OUT.mkdir(parents=True, exist_ok=True)

credentials = [(AUTH / name).read_text().strip() for name in ('accesskey_id', 'accesskey_secret')]
client = RJobClient(access_key=credentials[0], secret_key=credentials[1])
tenant = client.namespace.split('-', 1)[0]
scope = f'/{tenant}/{client.namespace}'
base = client.cluster_entry
event_path = f'/kapis/v1/tenants/{tenant}/projects/{client.namespace}/events'
audit_path = '/kapi/auditlogger.kubebrain.io/api/v1alpha1/audit'

query_log = []


def now():
    return datetime.now(timezone.utc).isoformat()


def redact(value):
    text = json.dumps(value, ensure_ascii=False, default=str)
    for credential in credentials:
        text = text.replace(credential, '<REDACTED>')
    text = re.sub(r'(https?://)[^\s/@]+:[^\s/@]+@', r'\1<REDACTED>@', text)
    return json.loads(text)


def save(name, value):
    (OUT / name).write_text(json.dumps(redact(value), indent=1, ensure_ascii=False, default=str) + '\n')


def record(query, target, status, **extra):
    row = dict(query=query, target=target, status=status, observed_utc=now())
    row.update(extra)
    query_log.append(row)
    return row


def http_get(path, parameters):
    session = requests.Session()
    session.auth = (client.username, client.password)
    session.headers['X-KB-Scope'] = scope
    try:
        response = session.get(base + path, params=parameters, verify=client.verifyssl,
                               timeout=(5, 20))
        response.raise_for_status()
        return response.json()
    finally:
        session.close()


# --- identity -------------------------------------------------------------
job_object = client.api.get_namespaced_custom_object(
    group=client.group, version=client.version, namespace=client.namespace,
    plural=client.plural, name=JOB, _request_timeout=(5, 25))
identity = dict(
    name=job_object['metadata']['name'], uid=job_object['metadata']['uid'],
    creator=(job_object['metadata'].get('labels') or {}).get('kubebrain.brainpp.cn/creator'),
    phase=(job_object.get('status') or {}).get('phase'),
    finishTime=(job_object.get('status') or {}).get('finishTime'),
    stopTimestamp=(job_object.get('spec') or {}).get('stopTimestamp'))
if identity['name'] != JOB or identity['uid'] != JOB_UID or identity['creator'] != 'i-fengyicheng':
    raise SystemExit(f'identity mismatch: {identity}')
record('identity', JOB, 'PASS', detail=identity)

# --- actual replica names and UIDs (no invented names) --------------------
replica_info = client.get_rjob_infos(JOB)
replica_targets = []
for task, rows in replica_info.items():
    for row in rows:
        name = row['name']
        pod = client._get_replica_kapis(name)
        owners = (pod.get('metadata') or {}).get('ownerReferences') or []
        if not any(o.get('uid') == JOB_UID and o.get('kind') == 'RJob' for o in owners):
            raise SystemExit(f'replica ownership mismatch for {name}')
        replica_targets.append(dict(task=task, name=name, uid=(pod.get('metadata') or {}).get('uid'),
                                    phase=(pod.get('status') or {}).get('phase')))
record('replica_identity', JOB, 'PASS', replicas=replica_targets)

# --- live Kubernetes Events ----------------------------------------------
live = {}
for kind, name in [('RJob', JOB)] + [('Replica', r['name']) for r in replica_targets]:
    pages, continuation, count = [], None, 0
    try:
        while True:
            params = {'field_selector': 'involvedObject.name=' + name, 'limit': PAGE_SIZE,
                      '_request_timeout': (5, 20)}
            if continuation:
                params['_continue'] = continuation
            response = client.corev1_api.list_namespaced_event(client.namespace, **params).to_dict()
            pages.append(response)
            count += len(response.get('items') or [])
            continuation = (response.get('metadata') or {}).get('_continue')
            if not continuation:
                break
        live[name] = dict(kind=kind, count=count, pages=pages, pagination_complete=True)
        record('live_events', name, 'PASS', kind=kind, count=count, pages=len(pages))
    except Exception as exc:  # noqa: BLE001
        live[name] = dict(kind=kind, error=dict(type=type(exc).__name__, message=str(exc),
                                                traceback=traceback.format_exc()))
        record('live_events', name, 'FAIL', kind=kind, error=f'{type(exc).__name__}: {exc}')
save('live_events.json', dict(observed_utc=now(), job=JOB, job_uid=JOB_UID, sources=live))

# --- historical Events ----------------------------------------------------
historical = {}
for kind, name, uid in [('RJob', JOB, JOB_UID)] + [('Replica', r['name'], r['uid']) for r in replica_targets]:
    pages, page, count, total = [], 1, 0, None
    try:
        while True:
            parameters = {'name': name, 'kind': kind, 'uid': uid,
                          'sortBy': 'lastTimestamp:desc', 'page': page, 'pageSize': PAGE_SIZE}
            body = http_get(event_path, parameters)
            data = body.get('data') or {}
            rows = data.get('data') or []
            total = data.get('total')
            pages.append(dict(parameters=parameters, total=total, rows=rows))
            count += len(rows)
            if total is None:
                raise ValueError('historical response has no total')
            if count >= total:
                break
            if not rows:
                raise ValueError('pagination ended before reported total')
            page += 1
        historical[name] = dict(kind=kind, uid=uid, count=count, total=total, pages=pages,
                                pagination_complete=True)
        record('historical_events', name, 'PASS', kind=kind, count=count, total=total, pages=len(pages))
    except Exception as exc:  # noqa: BLE001
        historical[name] = dict(kind=kind, uid=uid, total=total, count=count, pages=pages,
                                error=dict(type=type(exc).__name__, message=str(exc),
                                           traceback=traceback.format_exc()))
        record('historical_events', name, 'FAIL', kind=kind, total=total, count=count,
               error=f'{type(exc).__name__}: {exc}')
save('historical_events.json', dict(observed_utc=now(), job=JOB, job_uid=JOB_UID,
                                    request=dict(path=event_path, sortBy='lastTimestamp:desc',
                                                 pageSize=PAGE_SIZE),
                                    sources=historical))

# --- all-operation audit (no operationType filter) ------------------------
audit = {}


def audit_read(label, name, resource_type):
    pages, page, count, total = [], 1, 0, None
    while True:
        parameters = {'resourceType': resource_type, 'resourceName': name,
                      'projectName': client.namespace, 'pageType': client.namespace,
                      'page': page, 'pageSize': PAGE_SIZE}
        body = http_get(audit_path, parameters)
        rows = body.get('data') or []
        total = body.get('total')
        if not isinstance(rows, list):
            raise ValueError('audit data is not a record list')
        if any(row.get('resource_name') != name for row in rows):
            raise ValueError('audit row belongs to a different resource name')
        pages.append(dict(parameters=parameters, total=total, rows=rows))
        count += len(rows)
        if total is None:
            raise ValueError('audit response has no total')
        if count >= total:
            break
        if not rows:
            raise ValueError('audit pagination ended before reported total')
        page += 1
    audit[label] = dict(name=name, resource_type=resource_type, count=count, total=total,
                        pages=pages, pagination_complete=True)
    record('all_operation_audit', name, 'PASS', resource_type=resource_type, count=count,
           total=total, pages=len(pages))


for label, name, resource_type in [('rjob', JOB, 'rjobs')] + [
        (r['name'], r['name'], 'replicas') for r in replica_targets]:
    try:
        audit_read(label, name, resource_type)
    except Exception as exc:  # noqa: BLE001
        audit[label] = dict(name=name, resource_type=resource_type,
                            error=dict(type=type(exc).__name__, message=str(exc),
                                       traceback=traceback.format_exc()))
        record('all_operation_audit', name, 'FAIL', resource_type=resource_type,
               error=f'{type(exc).__name__}: {exc}')

save('audit_all_operations.json', dict(observed_utc=now(), job=JOB, job_uid=JOB_UID,
                                       request=dict(path=audit_path, pageSize=PAGE_SIZE,
                                                    note='no operationType filter'),
                                       resources=audit))
save('query_log.json', dict(observed_utc=now(), job=JOB, job_uid=JOB_UID,
                            queries=query_log))

summary = dict(job=JOB, phase=identity['phase'], finishTime=identity['finishTime'],
               live={k: v.get('count', v.get('error')) for k, v in live.items()},
               historical={k: (v.get('count'), v.get('total'), 'error' if 'error' in v else 'ok')
                           for k, v in historical.items()},
               audit={k: (v.get('count'), v.get('total'), 'error' if 'error' in v else 'ok')
                      for k, v in audit.items()},
               failed=[q for q in query_log if q['status'] != 'PASS'])
print(json.dumps(summary, indent=1, default=str)[:6000])
