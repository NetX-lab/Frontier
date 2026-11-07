"""Archive the generation06 terminal evidence and derive its terminal acceptance; local reads only.

Usage:
    python3 lead_terminal_audit.py

Copies the platform receipt, platform events and audit, monitor reads, replica logs and the launcher
records into this directory with sha256, then writes lead_terminal_acceptance.json. No container started,
so there is no cloud output to copy.
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'package_01'
TERMINAL = Path('/data/ycfeng/tmp/ds4_runs/p64-formal6-terminal-1107l/scratch')
MONITOR = Path('/data/ycfeng/tmp/ds4_runs/p64-formal6-monitor-1107k/scratch/monitor')
SUBMIT = Path('/data/ycfeng/tmp/ds4_runs/p64-formal6-submit-1107j/scratch')
JOB, UID = 'exp-1107-144321-305543', 'ceaa0653-8a7b-424b-b3e0-9c5bb52b7abd'
ADMISSION_FAILED = '8f14e45f'
GPUS = 64

SOURCES = [
    (TERMINAL / 'platform_terminal_receipt.json', 'platform_terminal_receipt.json'),
    (TERMINAL / 'terminal_receipt.out', 'terminal_receipt.out'),
    (TERMINAL / 'events.out', 'events.out'),
    (TERMINAL / 'events.err', 'events.err'),
    (TERMINAL / 'replica_logs.out', 'replica_logs.out'),
    (SUBMIT / 'launcher.log', 'launcher.log'),
    (PACKAGE / 'launcher_started.json', 'launcher_started.json'),
    (PACKAGE / 'launcher_exit.json', 'launcher_exit.json'),
]
SOURCES += [(p, f'terminal_events/{p.name}') for p in sorted((TERMINAL / 'events').iterdir())]
SOURCES += [(p, f'monitor/{p.name}') for p in sorted(MONITOR.iterdir()) if p.is_file()]
SOURCES += [(p, f'replica_logs/{p.name}') for p in sorted((TERMINAL / 'logs_1').iterdir())]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00'))


def archive():
    files = []
    for source, name in SOURCES:
        destination = HERE / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha256(source)
        assert sha256(destination) == digest, name
        files.append({'source': str(source), 'destination': name, 'bytes': source.stat().st_size, 'sha256': digest})
    (HERE / 'archive_manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    return files


def audit_rows():
    audit = json.loads((HERE / 'terminal_events/audit_all_operations.json').read_text())
    rows = []
    for resource in audit['resources'].values():
        for page in resource['pages']:
            for row in page['rows']:
                rows.append({'resource': row['resource_name'], 'operation': row['operation'],
                             'utc': datetime.fromtimestamp(row['timestamp'], timezone.utc).isoformat(),
                             'subject': row['subject']['subject_name'], 'subject_type': row['subject']['subject_type']})
    return rows


def platform_events():
    events = {}
    for name in ('live_events.json', 'historical_events.json'):
        for source in json.loads((HERE / 'terminal_events' / name).read_text())['sources'].values():
            for page in source['pages']:
                for item in page.get('items', []):
                    stamp = item.get('last_timestamp') or item.get('first_timestamp')
                    key = (item['metadata']['name'], stamp, item.get('reason'))
                    events[key] = {'utc': utc(stamp.replace(' ', 'T')).isoformat() if stamp else None,
                                   'object': item['involved_object']['name'], 'reason': item.get('reason'),
                                   'host': (item.get('source') or {}).get('host'), 'message': item.get('message') or ''}
    return sorted(events.values(), key=lambda e: e['utc'] or '')


def main():
    files = archive()
    receipt = json.loads((HERE / 'platform_terminal_receipt.json').read_text())
    final = json.loads((HERE / 'monitor/final.json').read_text())['job']
    launcher_exit = json.loads((HERE / 'launcher_exit.json').read_text())
    queries = json.loads((HERE / 'terminal_events/query_log.json').read_text())['queries']
    rows = audit_rows()
    events = platform_events()
    replicas = {r['name'].rsplit('-', 1)[1]: r for r in receipt['replicas']}
    stops = [r for r in rows if r['operation'] == 'stop']
    user_rows = [r for r in rows if r['subject_type'] == 'user']
    admission = [e for e in events if e['reason'] == 'UnexpectedAdmissionError']
    closes = [e for e in events if e['reason'] == 'Close replica']
    replica_events = [e for e in events if e['object'] != JOB and e['utc']]
    creates = sorted(r['utc'] for r in rows if r['operation'] == 'create' and r['resource'] != JOB)
    window_end = max([r['utc'] for r in stops] + [e['utc'] for e in replica_events])
    bound_s = (utc(window_end) - utc(creates[0])).total_seconds()
    checks = {
        'job_identity': receipt['job'] == JOB and receipt['creator'] == 'i-fengyicheng',
        'job_failed_without_start': receipt['status']['phase'] == 'Failed' and receipt['status']['startTime'] is None,
        'system_stop_reason_critical_task_failed':
            final['stop_annotations'].get('rjob.brainpp.cn/system-stop-reason') == 'Some critical tasks failed.',
        'only_user_operation_is_create': [r['operation'] for r in user_rows] == ['create'],
        'admission_error_on_failed_replica': bool(admission) and all(
            e['object'].endswith(ADMISSION_FAILED) or e['object'] == JOB for e in admission) and all(
            'nvidia.com/spot-gpu' in e['message'] for e in admission),
        'only_failed_replica_has_no_container': [s for s, r in replicas.items() if r['phase'] == 'Failed']
            == [ADMISSION_FAILED] and replicas[ADMISSION_FAILED]['containers'] == [],
        'no_container_started': all(c['state'].get('running') is None and c['state'].get('terminated') is None
                                    for r in replicas.values() for c in r['containers']),
        'replica_logs_empty': all(p.stat().st_size <= 1 for p in (HERE / 'replica_logs').iterdir()),
        'stops_by_rjob_controller': bool(stops) and all(r['subject'].endswith('rjob-controller') for r in stops),
        'closes_after_admission_error': bool(closes) and min(e['utc'] for e in closes) >= max(
            e['utc'] for e in admission),
        'launcher_exited_after_job_end': utc(launcher_exit['end_utc']) > utc(receipt['status']['finishTime']),
        'platform_queries_pass': all(q['status'] == 'PASS' for q in queries),
    }
    record = {
        'status': 'PASS_TERMINAL_PLATFORM_CAUSE_NO_REFERENCE' if all(checks.values()) else 'FAIL',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0],
        'command': 'python3 h200_mainline/p2_64_formal_retry06/terminal_20261107/lead_terminal_audit.py',
        'script_sha256': sha256(Path(__file__)),
        'job': JOB, 'uid': UID, 'finish_utc': receipt['status']['finishTime'],
        'checks': checks,
        'cause': {
            'classification': 'PLATFORM_SPOT_GPU_DEVICE_ADMISSION_FAILURE',
            'failed_replica': ADMISSION_FAILED,
            'admission_events': admission,
            'basis': 'While the replicas waited for spot GPUs to be released by preemption, the kubelet on the failed '
                     'replica\'s node rejected the pod at admission (8 spot GPUs requested, 7 available). The pod never '
                     'created a container, so no native code ran; the any-replica fail policy then closed the other '
                     'replicas through rjob-controller. The only user operation is the create.',
            'later_teardown_events': sorted({e['reason'] for e in events if e['utc'] and e['utc'] > max(
                x['utc'] for x in closes)}),
        },
        'accounting': {
            'known_container_gpu_hours': receipt['known_container_gpu_hours'],
            'container_clocks_complete': receipt['container_clocks_complete'],
            'upper_bound_window_utc': [creates[0], window_end],
            'upper_bound_seconds': bound_s,
            'counted_container_gpu_hours': round(GPUS * bound_s / 3600, 2),
            'rule': 'D-P2-64-FIFTH: without complete container clocks, 64 GPUs x scheduled wall time counts toward 128. '
                    'The window ends at the later of the last stop and the last replica event (image pull on a pod '
                    'still tearing down).',
        },
        'business': {'complete_native_reference': False, 'containers_started': 0},
        'not_run': {'cloud_copy': 'No container started, so no cloud output exists',
                    'pair_coverage': 'No native reference', 'R5': 'No native reference',
                    'metrics_and_warm_reading': 'No native reference'},
        'archived_files': len(files),
        'next': 'Platform-cause resubmission 2 of 2 (generation07), the last one under D-P2-64-FIFTH.',
    }
    (HERE / 'lead_terminal_acceptance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('status', 'checks', 'archived_files')}, indent=1))
    print(json.dumps(record['accounting'], indent=1))
    print(json.dumps(record['cause']['later_teardown_events']))
    return 0 if record['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
