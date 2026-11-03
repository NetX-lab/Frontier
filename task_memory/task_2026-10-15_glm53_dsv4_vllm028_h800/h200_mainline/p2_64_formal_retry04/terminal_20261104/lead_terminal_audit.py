from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import json
import sys

root = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
task = root / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800'
base = task / 'h200_mainline/p2_64_formal_retry04/terminal_20261104'

def read(name):
    return json.loads((base / name).read_text())

def date(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

job = read('job_object.json')
receipt = read('platform_terminal_receipt.json')
expected_name = 'exp-1103-131952-259161'
expected_uid = '9d02faf7-621e-4368-ae73-4110312c6ae2'
assert job['name'] == receipt['job'] == expected_name
assert job['uid'] == expected_uid
assert job['creator'] == receipt['creator'] == 'i-fengyicheng'
assert job['status']['phase'] == receipt['status']['phase'] == 'Stopped'
assert job['status']['startTime'] is receipt['status']['startTime'] is None
assert job['replica_count'] == 0 and job['replicas'] == receipt['replicas'] == []
assert job['preemptible'] == receipt['preemptible_label'] == 'yes'
assert job['quotagroup'] == receipt['charged_group'] == 'step_main'
resources = job['tasks']['default']
assert resources['replicas'] == 8
assert int(resources['resources'][0]['requests']['nvidia.com/gpu']) == 8

audit = read('terminal_events/audit_all_operations.json')
assert audit['job'] == expected_name and audit['job_uid'] == expected_uid
source = audit['resources']['rjob']
rows = [row for page in source['pages'] for row in page['rows']]
assert len(source['pages']) == 2 and len(rows) == source['total'] == source['count'] == 178
assert len({row['receipt'] for row in rows}) == 178
assert {row['resource_name'] for row in rows} == {expected_name}
stop, = [row for row in rows if row['operation'] == 'stop']
assert stop['result'] == 'succeeded'
assert stop['subject']['subject_name'] == job['stop_annotations']['kubebrain.brainpp.cn/stopped-by'] == 'alimasiyilixiati'
assert job['stop_annotations']['rjob.brainpp.cn/manual-stop-reason'] == 'release_quota'
finish = date(job['status']['finishTime'])
assert finish == date(receipt['status']['finishTime']) == datetime.fromtimestamp(stop['timestamp'], timezone.utc)
assert finish == date('2026-11-03T10:00:43Z')

hist = read('terminal_events/historical_events.json')
assert hist['job_uid'] == expected_uid
source = hist['sources'][expected_name]
events = [row for page in source['pages'] for row in page['rows']]
assert len(events) == source['count'] == source['total'] == 65
assert Counter(row['reason'] for row in events) == {'QueueWaiting': 65}
query = read('terminal_events/query_log.json')
assert all(row['status'] == 'PASS' for row in query['queries'])
assert len(query['queries']) == 5

launcher = read('launcher_exit.json')
local_receipt = read('launcher_receipt.json')
assert launcher['launcher_pid'] == 81539 and launcher['exit_code'] == 0
assert not Path('/proc/81539').exists()
assert local_receipt['rjob_name'] == expected_name
log = (base / 'launcher.log').read_text()
observed = date('2026-11-03T10:00:46.504Z')
cleanup = date('2026-11-03T10:00:47.218Z')
assert '2026-11-03 18:00:46.504' in log and 'Job stopped:' in log
assert '2026-11-03 18:00:47.218' in log and 'Job Stop:' in log
assert finish < observed < date(local_receipt['recorded_at_utc']) < date(launcher['end_utc']) < cleanup

cloud = read('cloud_query_receipts.json')['records']
assert len(cloud) == 2 and all(not row['is_error'] for row in cloud)
assert all(row['structured_content']['exit_code'] == 0 for row in cloud)
assert cloud[0]['result'] == 'i-fengyicheng\nDIR_ABSENT\n'
assert 'dsv4_h200_p2_64_formal_20261103_04' in cloud[0]['command']
assert 'No such file or directory' in cloud[1]['result']
assert receipt['container_gpu_hours'] is None
assert receipt['known_container_gpu_hours'] == 0
assert receipt['container_clocks_complete'] is False
wall_seconds = (finish - date(job['creationTimestamp'])).total_seconds()
assert wall_seconds == 16851
for item in read('archive_manifest.json')['files']:
    assert sha(base / item['destination']) == item['sha256'] == sha(Path(item['source']))

result = dict(
    status='PASS_TERMINAL_CAUSE_NO_REFERENCE', observed_utc=datetime.now(timezone.utc).isoformat(),
    python=sys.version.split()[0], command=f'python3 {Path(__file__)}', script_sha256=sha(Path(__file__)),
    job=expected_name, uid=expected_uid, creator=job['creator'], finish_utc=finish.isoformat(),
    cause=dict(actor=stop['subject']['subject_name'], reason='release_quota', audit_rows=178, audit_pages=2),
    historical_events=dict(count=65, reasons={'QueueWaiting': 65}),
    chronology=dict(platform_before_launcher=True, launcher_exit_code=0, launcher_absent=True),
    business=dict(complete_native_reference=False, cloud_directory_absent=True, raw_cloud_read_receipts=2,
                  never_scheduled='Inference supported by null startTime, zero retained replicas, complete available queue history, and absent cloud output.'),
    accounting=dict(container_gpu_hours=None, known_container_gpu_hours=0.0, container_clocks_complete=False,
                    queue_wall_seconds=wall_seconds, queue_excluded_from_gpuh=True,
                    zero_container_consumption='Supported inference, not a complete-clock receipt.'),
    not_run=dict(copy='No native output', pair_coverage='No native reference', R5='No native reference',
                 metrics_and_warm_reading='No native reference'),
    limitations=['Current API has no replicas; historical queue projections are not container clocks.',
                 'No preemption classification and no further retry authority inferred.',
                 'Generation01/02 incomplete container clocks remain open.'],
)
(base / 'lead_terminal_acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
