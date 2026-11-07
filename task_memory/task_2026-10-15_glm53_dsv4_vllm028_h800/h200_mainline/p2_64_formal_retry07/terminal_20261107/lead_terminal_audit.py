"""Archive the generation07 terminal evidence and derive its terminal acceptance; local reads only.

Usage:
    python3 lead_terminal_audit.py

Copies the platform receipt, platform events and audit, monitor reads, replica logs, the launcher records,
the two mid-run reads and the read-only summary of the native run tree into this directory with sha256, then
writes lead_terminal_acceptance.json. The job ended without a native reference, so no cloud copy job ran; the
native run tree stays at the cloud path recorded below.
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'package_01'
TERMINAL = Path('/data/ycfeng/tmp/ds4_runs/p64-formal7-terminal-1107p/scratch')
INSPECT = Path('/data/ycfeng/tmp/ds4_runs/p64-formal7-inspect-1107o/scratch')
SUBMIT = Path('/data/ycfeng/tmp/ds4_runs/p64-formal7-submit-1107m/scratch')
MONITOR = SUBMIT / 'monitor'
CLOUD = ('/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h200_p2_64_formal_20261107_07/case/runs/'
         'groundtruth_clean/dsv4-h200-p2-64-formal-20261107-07')
JOB, UID = 'exp-1107-150132-396145', '45849203-b2c0-4de5-9ef3-e3b4ad716168'
STOPPED, STOPPED_NODE = 'cfcd2084', 'gpu-h200-0726.lgcm.sh.istep.fun'
DECODE_HEAD = 'gpu-h200-0300.lgcm.sh.istep.fun'
GPUS = 64
COUNTED_BEFORE = 27.09

SOURCES = [
    (TERMINAL / 'platform_terminal_receipt.json', 'platform_terminal_receipt.json'),
    (TERMINAL / 'terminal_receipt.out', 'terminal_receipt.out'),
    (TERMINAL / 'events.out', 'events.out'),
    (TERMINAL / 'events.err', 'events.err'),
    (TERMINAL / 'replica_logs.out', 'replica_logs.out'),
    (TERMINAL / 'native_tree.json', 'native_tree.json'),
    (TERMINAL / 'native_tree.err', 'native_tree.err'),
    (INSPECT / 'remote.txt', 'midrun/remote_0746.txt'),
    (INSPECT / 'remote_2.txt', 'midrun/remote_0748.txt'),
    (INSPECT / 'live_receipt_2.json', 'midrun/live_receipt_0748.json'),
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
    """Live events list `items` (snake_case); historical replica events list `rows` (camelCase)."""
    events = {}
    for name in ('live_events.json', 'historical_events.json'):
        for target, source in json.loads((HERE / 'terminal_events' / name).read_text())['sources'].items():
            for page in source['pages']:
                for item in page.get('items', []) + page.get('rows', []):
                    stamp = (item.get('last_timestamp') or item.get('first_timestamp') or item.get('lastTimestamp')
                             or item.get('firstTimestamp'))
                    obj = (item.get('involved_object') or item.get('involvedObject') or {}).get('name') or target
                    key = (obj, stamp, item.get('reason'), item.get('message'))
                    events[key] = {'utc': utc(stamp.replace(' ', 'T')).isoformat() if stamp else None,
                                   'object': obj, 'reason': item.get('reason'),
                                   'host': (item.get('source') or {}).get('host'), 'message': item.get('message') or ''}
    return sorted(events.values(), key=lambda e: e['utc'] or '')


def main():
    files = archive()
    receipt = json.loads((HERE / 'platform_terminal_receipt.json').read_text())
    final = json.loads((HERE / 'monitor/final.json').read_text())['job']
    launcher_exit = json.loads((HERE / 'launcher_exit.json').read_text())
    queries = json.loads((HERE / 'terminal_events/query_log.json').read_text())['queries']
    tree = json.loads((HERE / 'native_tree.json').read_text())
    reads = [json.loads(line) for line in (HERE / 'monitor/monitor.jsonl').read_text().splitlines() if line.strip()]
    rows = audit_rows()
    events = platform_events()
    replicas = {r['name'].rsplit('-', 1)[1]: r for r in receipt['replicas']}
    nodes = tree['nodes']
    stopped_node = nodes[STOPPED_NODE]
    plans = {host: node.get('result', {}).get('plan', {}) for host, node in nodes.items()}
    decode_lanes = sorted(p['lane'] for p in plans.values() if p.get('role') == 'decode')
    failed_errors = {host[:13]: node['failed.json']['error'].strip().splitlines()[-1]
                     for host, node in nodes.items() if 'failed.json' in node}
    user_rows = [r for r in rows if r['subject_type'] == 'user']
    stops = [r for r in rows if r['operation'] == 'stop']
    stopped_rows = [r for r in rows if r['resource'].endswith(STOPPED)]
    stopped_events = [e for e in events if e['object'].endswith(STOPPED)]
    failed = {s: r for s, r in replicas.items() if r['phase'] == 'Failed'}
    replica_events = [e for e in events if e['object'] != JOB and e['utc']]
    creates = sorted(r['utc'] for r in rows if r['operation'] == 'create' and r['resource'] != JOB)
    window_end = max([r['utc'] for r in stops] + [e['utc'] for e in replica_events])
    bound_s = (utc(window_end) - utc(creates[0])).total_seconds()
    counted = round(GPUS * bound_s / 3600, 2)
    seen_stopped = [r['observed_utc'] for r in reads if 'Stopped' in r['replica_phases'] and r['phase'] == 'Running']
    all_running = [r['observed_utc'] for r in reads if r['replica_phases'] and set(r['replica_phases']) == {'Running'}]
    stopped_files = sorted(f[0] for f in tree['files'] if f[0].startswith(f'nodes/{STOPPED_NODE}/'))
    checks = {
        'job_identity': receipt['job'] == JOB and receipt['creator'] == 'i-fengyicheng',
        'job_failed_by_any_fail_policy': receipt['status']['phase'] == 'Failed'
            and final['stop_annotations'].get('rjob.brainpp.cn/system-stop-reason')
            == 'Task fail policy [any] caused undesired running replicas.',
        'only_user_operation_is_create': [r['operation'] for r in user_rows] == ['create'],
        'stopped_replica_phase_stopped_with_stale_running_container':
            replicas[STOPPED]['phase'] == 'Stopped' and replicas[STOPPED]['node'] == STOPPED_NODE
            and list(replicas[STOPPED]['containers'][0]['state']) == ['running'],
        'stopped_replica_has_no_stop_operation': [r['operation'] for r in stopped_rows] == ['create'],
        'stopped_replica_events_end_at_image_pull':
            [e['reason'] for e in stopped_events] == ['FailedScheduling', 'Pulling'],
        'stopped_node_record_ends_at_rendezvous': stopped_node['result']['stage'] == 'rendezvous'
            and 'failed.json' not in stopped_node and 'server.log_tail' not in stopped_node
            and stopped_files == [f'nodes/{STOPPED_NODE}/{n}' for n in ('ready.json', 'result.json', 'worker.log')],
        'stopped_node_is_missing_decode_lane_2': decode_lanes == [0, 1, 3]
            and not plans[STOPPED_NODE].get('role'),
        'stop_seen_between_monitor_reads': bool(all_running) and bool(seen_stopped)
            and max(all_running) < min(seen_stopped),
        'decode_head_startup_timeout': failed_errors.get(DECODE_HEAD[:13])
            == 'TimeoutError: Native P/D rendezvous or workload deadline exceeded',
        'prefill_head_reached_server_ready': 'server_ready.json' in nodes['gpu-h200-0023.lgcm.sh.istep.fun'],
        'no_replay_started': 'replay_finished.json' not in tree
            and not any('replay.log_tail' in node for node in nodes.values()),
        'two_failed_containers_exit_1': sorted(failed) == ['1679091c', 'eccbc87e'] and all(
            r['containers'][0]['state']['terminated']['exitCode'] == 1 for r in failed.values()),
        'teardown_stops_by_rjob_controller': bool(stops) and all(r['subject'].endswith('rjob-controller')
            and r['utc'] >= receipt['status']['finishTime'].replace('Z', '+00:00') for r in stops),
        'platform_ping_helper_error_on_every_replica': all(
            "No such file or directory: 'ip'" in p.read_text() for p in (HERE / 'replica_logs').iterdir()),
        'launcher_exited_after_job_end': utc(launcher_exit['end_utc']) > utc(receipt['status']['finishTime']),
        'platform_queries_pass': all(q['status'] == 'PASS' for q in queries),
    }
    record = {
        'status': 'PASS_TERMINAL_PLATFORM_CAUSE_NO_REFERENCE' if all(checks.values()) else 'FAIL',
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0],
        'command': 'python3 h200_mainline/p2_64_formal_retry07/terminal_20261107/lead_terminal_audit.py',
        'script_sha256': sha256(Path(__file__)),
        'job': JOB, 'uid': UID, 'finish_utc': receipt['status']['finishTime'], 'cloud_run_root': CLOUD,
        'checks': checks,
        'cause': {
            'classification': 'PLATFORM_EXTERNAL_REPLICA_STOP (inferred)',
            'stopped_replica': STOPPED, 'stopped_node': STOPPED_NODE,
            'stop_window_utc': [max(all_running), min(seen_stopped)],
            'stopped_replica_events': stopped_events,
            'node_failures': failed_errors,
            'basis': 'Replica cfcd2084 on gpu-h200-0726 wrote ready.json and its rendezvous record at 07:26:35Z and '
                     'nothing after; it reached phase Stopped between the monitor reads at 07:26:50Z and 07:31:52Z '
                     'with no stop operation in the audit, no event after the image pull and no native error, and its '
                     'container state stayed at the last reported running. The only user operation is the create; '
                     'the job is preemptible. No event names preemption or node loss, so the classification is an '
                     'inference. The p2p_ping helper error is present on all 8 replicas and is not specific.',
            'cascade': 'The other 7 nodes had planned gpu-h200-0726 as decode lane 2. The decode DP head never became '
                       'healthy and raised its 1800 s startup timeout at 07:56:45Z; decode lane 1 then saw its '
                       'headless server exit 0, decode lane 3 and prefill lane 2 saw the peer failure, and the '
                       'any-replica fail policy stopped the rest through rjob-controller. No replay started.',
        },
        'accounting': {
            'known_container_gpu_hours': receipt['known_container_gpu_hours'],
            'container_clocks_complete': receipt['container_clocks_complete'],
            'upper_bound_window_utc': [creates[0], window_end],
            'upper_bound_seconds': bound_s,
            'counted_container_gpu_hours': counted,
            'counted_before': COUNTED_BEFORE,
            'cumulative_counted': round(COUNTED_BEFORE + counted, 2),
            'cap': 128,
            'rule': 'D-P2-64-FIFTH: without complete container clocks, 64 GPUs x scheduled wall time counts toward 128. '
                    'The window runs from the first replica create to the later of the last stop and the last '
                    'replica event.',
        },
        'business': {'complete_native_reference': False, 'replay_started': False},
        'not_run': {'cloud_copy': 'No native reference; the read-only run-tree summary is archived instead',
                    'pair_coverage': 'No native reference', 'R5': 'No native reference',
                    'metrics_and_warm_reading': 'No native reference'},
        'archived_files': len(files),
        'next': 'D-P2-64-FIFTH is exhausted (generation05 and both platform-cause resubmissions used). Any further '
                'GPU job needs a new user decision.',
    }
    (HERE / 'lead_terminal_acceptance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('status', 'checks', 'archived_files')}, indent=1))
    print(json.dumps(record['accounting'], indent=1))
    print(json.dumps({k: record['cause'][k] for k in ('stop_window_utc', 'node_failures')}, indent=1))
    return 0 if record['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
