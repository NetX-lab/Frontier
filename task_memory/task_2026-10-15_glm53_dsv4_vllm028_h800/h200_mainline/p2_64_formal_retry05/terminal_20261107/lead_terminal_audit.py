"""Archive the generation05 terminal evidence and derive its terminal acceptance; local reads only.

Usage:
    python3 lead_terminal_audit.py

Copies the platform receipt, platform events and audit, monitor reads, replica logs and the
hash-identical cloud copy into this directory with sha256, then writes lead_terminal_acceptance.json.
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'package_01'
TERMINAL = Path('/data/ycfeng/tmp/ds4_runs/p64-formal5-terminal-1107i/scratch')
MONITOR = Path('/data/ycfeng/tmp/ds4_runs/p64-formal5-monitor-1107f/scratch/monitor')
LOGS = Path('/data/ycfeng/tmp/ds4_runs/p64-formal5-logs-1107h/scratch/logs_1')
COPY = Path('/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261104_05')
RUN_TAG = 'dsv4-h200-p2-64-formal-20261104-05'
JOB, UID = 'exp-1107-135759-122907', 'a2d12f21-f384-411e-9dc4-f596e0f3e86a'
EXTERNAL_STOPS = {'8f14e45f', 'a87ff679', 'c81e728d'}
GPUS = 64

SOURCES = [
    (TERMINAL / 'platform_terminal_receipt.json', 'platform_terminal_receipt.json'),
    (TERMINAL / 'terminal_receipt.out', 'terminal_receipt.out'),
    (TERMINAL / 'copy_launcher.log', 'copy_launcher.log'),
    (TERMINAL / 'events.out', 'events.out'),
    (TERMINAL / 'events.err', 'events.err'),
    (PACKAGE / 'launcher_started.json', 'launcher_started.json'),
    (PACKAGE / 'launcher_exit.json', 'launcher_exit.json'),
]
SOURCES += [(p, f'terminal_events/{p.name}') for p in sorted((TERMINAL / 'events').iterdir())]
SOURCES += [(p, f'monitor/{p.name}') for p in sorted(MONITOR.iterdir()) if p.is_file()]
SOURCES += [(p, f'replica_logs/{p.name}') for p in sorted(LOGS.iterdir())]
SOURCES += [(COPY / 'launcher.json', 'cloud_copy/launcher.json')]
SOURCES += [(p, f'cloud_copy/{p.relative_to(COPY / RUN_TAG)}') for p in sorted((COPY / RUN_TAG).rglob('*')) if p.is_file()]


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


def main():
    files = archive()
    receipt = json.loads((HERE / 'platform_terminal_receipt.json').read_text())
    launcher_exit = json.loads((HERE / 'launcher_exit.json').read_text())
    download = json.loads((HERE / 'cloud_copy/download_receipt.json').read_text())
    rows = audit_rows()
    replicas = {r['name'].rsplit('-', 1)[1]: r for r in receipt['replicas']}
    replica_stops = {suffix: [r for r in rows if r['resource'].endswith(suffix) and r['operation'] == 'stop']
                     for suffix in replicas}
    user_rows = [r for r in rows if r['subject_type'] == 'user']
    creates = sorted(r['utc'] for r in rows if r['operation'] == 'create' and r['resource'] != JOB)
    last_stop = max(r['utc'] for r in rows if r['operation'] == 'stop')
    bound_s = (utc(last_stop) - utc(creates[0])).total_seconds()
    failed = {s: r for s, r in replicas.items() if r['phase'] == 'Failed'}
    checks = {
        'job_identity': receipt['job'] == JOB and receipt['creator'] == 'i-fengyicheng',
        'job_failed': receipt['status']['phase'] == 'Failed',
        'only_user_operation_is_create': [r['operation'] for r in user_rows] == ['create'],
        'external_stops_have_no_stop_operation': all(not replica_stops[s] for s in EXTERNAL_STOPS),
        'external_stops_phase_stopped': all(replicas[s]['phase'] == 'Stopped' for s in EXTERNAL_STOPS),
        'other_replicas_stopped_by_rjob_controller': all(
            replica_stops[s] and all(r['subject'].endswith('rjob-controller') for r in replica_stops[s])
            for s in replicas if s not in EXTERNAL_STOPS | set(failed)),
        'two_failed_containers_exit_1': len(failed) == 2 and all(
            r['containers'][0]['state']['terminated']['exitCode'] == 1 for r in failed.values()),
        'launcher_exited_after_job_end': utc(launcher_exit['end_utc']) > utc(receipt['status']['finishTime']),
        'download_hash_identical': download['status'] == 'HASH_IDENTICAL_DOWNLOAD',
    }
    record = {
        'status': 'PASS_TERMINAL_PLATFORM_CAUSE_NO_REFERENCE' if all(checks.values()) else 'FAIL',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0],
        'command': 'python3 h200_mainline/p2_64_formal_retry05/terminal_20261107/lead_terminal_audit.py',
        'script_sha256': sha256(Path(__file__)),
        'job': JOB, 'uid': UID, 'finish_utc': receipt['status']['finishTime'],
        'checks': checks,
        'cause': {
            'classification': 'PLATFORM_EXTERNAL_REPLICA_STOPS (inferred)',
            'externally_stopped_replicas': sorted(EXTERNAL_STOPS),
            'audit_rows': len(rows),
            'queue_controller_job_updates': sorted(r['utc'] for r in rows if r['resource'] == JOB
                                                   and r['subject'].endswith('queue-controller')),
            'basis': 'The three replicas entered phase Stopped with no stop operation in the audit and no native error; '
                     'the only user operation is the create; queue-controller job updates coincide with the stops; '
                     'the job is preemptible. No explicit preemption event exists, so the classification is an inference.',
            'cascade': 'Two decode DP lanes lost the DP head TCPStore, their workers exited 1 (Failed), and the '
                       'any-replica fail policy stopped the remaining replicas through rjob-controller.',
        },
        'accounting': {
            'known_container_gpu_hours': receipt['known_container_gpu_hours'],
            'container_clocks_complete': receipt['container_clocks_complete'],
            'upper_bound_window_utc': [creates[0], last_stop],
            'upper_bound_seconds': bound_s,
            'counted_container_gpu_hours': round(GPUS * bound_s / 3600, 2),
            'rule': 'D-P2-64-FIFTH: without complete container clocks, 64 GPUs x scheduled wall time counts toward 128.',
        },
        'business': {'complete_native_reference': False,
                     'download_files': download['files'], 'download_bytes': download['bytes']},
        'not_run': {'pair_coverage': 'No native reference', 'R5': 'No native reference',
                    'metrics_and_warm_reading': 'No native reference'},
        'archived_files': len(files),
        'next': 'Platform-cause resubmission 1 of 2 (generation06) under D-P2-64-FIFTH.',
    }
    (HERE / 'lead_terminal_acceptance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('status', 'checks', 'archived_files')}, indent=1))
    print(json.dumps(record['accounting'], indent=1))
    return 0 if record['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
