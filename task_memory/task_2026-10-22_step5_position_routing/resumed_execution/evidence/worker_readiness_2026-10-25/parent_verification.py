"""Verify and archive the completed D29 readiness receipts without a runtime or platform call."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

started = time.monotonic()
repo = Path('/data/ycfeng/Frontier/.worktrees/step5-preview')
task = repo / 'task_memory/task_2026-10-22_step5_position_routing'
preparation = task / 'resumed_execution/evidence/execution_preparation_2026-10-25'
business = Path('/data/ycfeng/tmp/d29_worker_readiness_20261025_01')
launcher = Path('/data/ycfeng/tmp/d29_worker_readiness_launcher_20261025_01')
archive = task / 'resumed_execution/evidence/worker_readiness_2026-10-25'

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def read_json(path):
    return json.loads(path.read_bytes())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(path, *arguments):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(path), *arguments], text=True).strip()

plan = read_json(preparation / 'worker_plan.json')
execution = read_json(launcher / 'execution.json')
submission = read_json(business / 'launcher.json')
worker = read_json(business / 'worker/readiness.json')
runtime = read_json(business / 'worker/runtime_probe.stdout')
append = read_json(business / 'worker/append.json')
first = read_json(launcher / 'platform01.json')
terminal = read_json(launcher / 'platform02_terminal.json')
review = read_json(preparation / 'preparation_review_followup.json')
require(execution['exit_code'] == 0 and execution['launcher_attempt'] == execution['job_limit'] == 1,
        'The original launcher did not complete its single permitted attempt')
require(execution['automatic_retries'] == 0, 'Unexpected readiness retry')
require(execution['worker_plan_sha256'] == digest(preparation / 'worker_plan.json') == review['worker_plan_sha256'],
        'The submitted worker plan differs from the reviewed plan')
for name, expected in review['script_hashes'].items():
    require(digest(preparation / name) == expected, f'Reviewed script changed: {name}')
for filename, expected in plan['git']['files'].items():
    require(digest(Path(filename)) == expected, f'Mounted Git asset changed: {filename}')
require(digest(Path(plan['runtime_probe'])) == plan['runtime_probe_sha256'], 'Original runtime probe changed')
require(digest(Path(plan['joint_bindings'])) == plan['joint_bindings_sha256'], 'Joint bindings changed')
sys.path.insert(0, str(preparation))
from joint_timing import check_bindings
identity = check_bindings(read_json(Path(plan['joint_bindings'])), preparation)
require(identity == worker['source'] == submission['source'], 'Before/after worker and current source identities differ')
require(identity['records_head'] == execution['records_head'], 'Records HEAD moved during the worker')
for reference in [plan['source_commit'], '1461573de388b72450189bc0c1b88b4ac32491f0']:
    require(not git(repo, 'diff', '--name-only', reference, 'HEAD', '--', '.', ':(exclude)task_memory'),
            'Current non-task source differs from admitted source')
job = submission['job_name']
require(job == first['job_name'] == terminal['job_name'] == (launcher/'job_name.txt').read_text().strip(),
        'Platform receipts refer to different jobs')
require(submission['status'] == 'PASS_PLATFORM_AND_WORKER_READINESS_ONLY' and submission['final_poll'] == 'succeeded',
        'Original platform/business verdict failed')
require(submission['creator'] == first['creator'] == terminal['creator'] == plan['principal'], 'Creator differs')
require(first['labels']['quotagroup.brainpp.cn/quotagroup'] == submission['charged_group'] == plan['charged_group'],
        'Charged group differs')
require(submission['image'] == plan['image'] and submission['actual_images']['default'] == [plan['image']], 'Image differs')
expected_resources = {'cpu': '4', 'memory': '64000Mi', 'nvidia.com/gpu': '0'}
require(submission['actual_resources']['default'] == [{'limits': expected_resources, 'requests': expected_resources}],
        'Actual platform resource requests/limits differ')
require(first['tasks']['default']['replicas'] == 1, 'Replica count differs')
nfs = f"{submission['nfs_server']}:{plan['local_mount_source']}:{plan['local_mount_target']}"
require(submission['actual_nfs_mounts'] == {'default': nfs} and first['tasks']['default']['nfs'] == nfs,
        'Platform NFS source differs from the local launcher')
require(any(f" - nfs4 {submission['nfs_server']}:{plan['local_mount_source']} " in line for line in worker['mounts']),
        'Worker does not report the expected local NFS source')
require(any(' /mnt/codesign-exp ' in line and 'JuiceFS:codesign-exp' in line for line in worker['mounts']),
        'Worker does not report the expected personal-volume mount')
require(terminal['platform_status']['phase'] == terminal['replica_status']['phase'] == 'Succeeded',
        'Platform or replica terminal status is not Succeeded')
containers = terminal['replica_status']['containerStatuses']
require(len(containers) == 1, 'Unexpected container count')
container = containers[0]
require(container['state']['terminated']['exitCode'] == 0 and container['state']['terminated']['reason'] == 'Completed'
        and container['restartCount'] == 0 and container['imageID'] == plan['image'], 'Container terminal evidence differs')
require(terminal['replica_status']['nodeName'] == worker['worker_hostname'], 'Worker node identity differs')
require(terminal['log_status_code'] == 200 and terminal['log_rows'] == 3, 'Bounded replica log receipt differs')
require(worker['status'] == 'PASS_WORKER_IMPORT_AND_SERIALIZED_APPEND_ONLY', 'Worker business verdict failed')
require(all(worker[field] == 0 for field in ['pricing_calls', 'required_shape_calls', 'full_cli_calls', 'gpu_request']),
        'Readiness worker performed an out-of-scope workload')
require(not worker['full_workload_memory_verified'], 'Readiness receipt makes an unsupported full-memory claim')
require(runtime['status'] == 'PASS_CPU_IMPORTS_ONLY' and runtime['binary_numerical_checks'] == 'PASS'
        and not runtime['simulator_constructed'] and runtime['pricing_wave_calls'] == runtime['gpu_calls'] == 0,
        'Original CPU runtime predicate differs')
require(runtime['python'].split()[0] == '3.10.6' and runtime['executable'] == plan['python'], 'Worker interpreter differs')
require(runtime['base_prefix'] == plan['runtime_base_prefix'], 'Worker base prefix differs')
require(runtime['package_versions'] == read_json(preparation/'runtime_acceptance.json')['package_versions'],
        'Worker package versions differ from accepted runtime')
for module, metadata in runtime['modules'].items():
    owner = repo if module.startswith('frontier.') else Path(runtime['prefix'])
    require(Path(metadata['origin']).is_relative_to(owner), f'Unexpected module origin: {module}')
for label in ['runtime_probe', 'append']:
    child = read_json(business / f'worker/{label}.execution.json')
    require(child == worker['runtime' if label == 'runtime_probe' else 'append'], 'Child execution receipt differs')
    require(child['returncode'] == 0 and not child['timed_out']
            and child['configured_address_space_bytes'] == plan['append']['child_address_space_bytes'],
            'Child execution/limit predicate differs')
require(append['status'] == 'PASS_SERIALIZED_REFERENCE_APPEND_ONLY' and append['reference_only']
        and append['cleanup'] == 'PASS_TWO_COMPLETED_TEST_FILES_REMOVED', 'Append or cleanup verdict failed')
require(append['production_pricing_calls'] == append['production_emission_calls'] == 0, 'Unexpected production append call')
require(append['cloud_root'] == plan['cloud_test_root'] and Path(append['cloud_root']).is_relative_to(Path(plan['cloud_personal_root'])),
        'Append path is outside the authorized personal directory')
ledger_spec, operators_spec = plan['append']['reference_ledger'], plan['append']['reference_operators']
require(digest(Path(ledger_spec['path'])) == ledger_spec['sha256'], 'Ledger reference changed')
ledger_rows = read_json(Path(ledger_spec['path']))
ledger_bytes = b''.join((json.dumps(row, sort_keys=True)+'\n').encode() for row in ledger_rows)
with Path(operators_spec['path']).open('rb') as stream:
    for _ in range(operators_spec['skip_header_lines']):
        require(bool(stream.readline()), 'Missing reference header')
    operator_lines = [stream.readline() for _ in range(operators_spec['record_count'])]
operator_bytes = b''.join(operator_lines)
require(len(ledger_rows) == 16 and len(operator_lines) == 208 and all(operator_lines), 'Incomplete reference wave')
require(hashlib.sha256(operator_bytes).hexdigest() == operators_spec['sha256'], 'Operator segment changed')
repetitions = append['repetitions']
wave_bytes = len(ledger_bytes) + len(operator_bytes)
require(0 < repetitions <= plan['append']['max_bytes']//wave_bytes, 'Append exceeds the finite byte bound')
require(append['bytes'] == repetitions*wave_bytes and append['ledger_rows'] == repetitions*16
        and append['operator_rows'] == repetitions*208 and append['trace_buffer_events'] == 5000, 'Complete row counts differ')
for record, contents, filename in zip(append['files'], [ledger_bytes, operator_bytes], ['ledger.jsonl', 'operators.jsonl']):
    expected = hashlib.sha256()
    for _ in range(repetitions):
        expected.update(contents)
    require(record['path'] == str(Path(append['cloud_root'])/filename) and record['bytes'] == len(contents)*repetitions
            and record['sha256'] == expected.hexdigest() and record['read_back'] == 'PASS', 'Full replay/read-back digest differs')
require(len(append['files']) == 2 and sum(row['bytes'] for row in append['files']) == append['bytes'], 'File inventory differs')
require(math.isclose(append['append_bytes_per_s'], append['bytes']/append['append_and_close_s'], rel_tol=1e-12)
        and math.isclose(append['append_with_fsync_bytes_per_s'], append['bytes']/(append['append_and_close_s']+append['fsync_s']), rel_tol=1e-12),
        'Append rate does not match measured bytes and wall times')
worktrees = []
for name in ['step5-preview', 'step5-d29-ledger-stream', 'd29-ledger-final-1f6b4314', 'd29-snapshot-copy-finish', 'd29-init-preparation-finish']:
    path = repo.parent/name
    head = git(path, 'rev-parse', 'HEAD')
    status = git(path, 'status', '--porcelain=v1')
    require(name == 'step5-preview' or not status, f'Satellite worktree is dirty: {name}')
    require(subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',head,'HEAD'], capture_output=True).returncode == 0,
            f'Current worktree is not consolidated: {name}')
    worktrees.append({'name': name, 'path': str(path), 'head': head, 'branch': git(path,'branch','--show-current'),
                     'status': status, 'ancestor_of_primary': True})
selected = [(business/name, 'business/'+name) for name in ['launcher.json', 'worker/append.execution.json',
    'worker/append.json', 'worker/append.stderr', 'worker/append.stdout', 'worker/readiness.json',
    'worker/runtime_probe.execution.json', 'worker/runtime_probe.stderr', 'worker/runtime_probe.stdout']]
selected += [(launcher/name, 'launcher/'+name) for name in ['execution.json', 'job_name.txt', 'platform01.json',
    'platform02_terminal.json', 'replica600.log', 'stdout.redacted', 'stderr.redacted']]
secrets = [(Path('/data/ycfeng/tmp/brainctl-auth-i-fengyicheng')/name).read_bytes().strip()
           for name in ['accesskey_id','accesskey_secret']]
for path, _ in selected:
    contents = path.read_bytes()
    require(not any(secret and secret in contents for secret in secrets), f'Secret present; no archive allowed: {path.name}')
    require(not re.search(rb'[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/\"\x27]+:[^\s/\"\x27]+@', contents),
            f'URL credentials present; no archive allowed: {path.name}')
archive.mkdir(exist_ok=False)
artifacts = []
for source, relative in selected:
    destination = archive/relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    require(digest(source) == digest(destination), f'Archive bytes differ: {relative}')
    artifacts.append({'source': str(source), 'archive': str(destination.relative_to(repo)),
                      'bytes': source.stat().st_size, 'sha256': digest(source)})
shutil.copy2(Path(__file__), archive/'parent_verification.py')
receipt = {'task_id': 'D29-WORKER-READINESS-PARENT', 'status': 'PASS_SELECTED_RECEIPTS_SOURCE_AND_ARCHIVE_ONLY',
    'collected_utc': datetime.now(timezone.utc).isoformat(), 'job_name': job, 'source': identity,
    'worker_plan_sha256': execution['worker_plan_sha256'], 'reviewed_script_hashes': review['script_hashes'],
    'checks': {'source_and_inputs': 'PASS_25_SOURCE_6_INPUTS_262_METADATA', 'reviewed_scripts': 'PASS_9_UNCHANGED',
        'platform_and_container': 'PASS_SUCCEEDED_EXIT0_RESTART0', 'worker_mounts': 'PASS_LOCAL_NFS_AND_JUICEFS',
        'runtime': 'PASS_ORIGINAL_CPU_PROBE', 'append': 'PASS_COMPLETE_REFERENCE_BYTE_COUNTS_AND_DIGESTS',
        'cleanup': append['cleanup'], 'secret_scan': 'PASS_SELECTED_LOCAL_EVIDENCE_ONLY'},
    'platform': {'phase': terminal['platform_status']['phase'], 'replica_phase': terminal['replica_status']['phase'],
        'container_exit': 0, 'restart_count': 0, 'replica_name': terminal['replica_name'],
        'worker_hostname': worker['worker_hostname'], 'pod_read': terminal['pod_read'],
        'creator': submission['creator'], 'charged_group': submission['charged_group'],
        'resource_requests_and_limits': expected_resources, 'image': plan['image'], 'nfs': nfs},
    'observed': {'launcher_wrapper_wall_s': execution['wall_s'], 'original_launcher_wall_s': submission['launcher_wall_s'],
        'runtime_wall_s': worker['runtime']['wall_s'], 'runtime_peak_rss_bytes': runtime['process_peak_rss_bytes'],
        'append': append}, 'worktrees': worktrees, 'artifacts': artifacts,
    'archive_files': len(artifacts), 'archive_bytes': sum(item['bytes'] for item in artifacts),
    'environment': {'python': sys.version, 'executable': sys.executable, 'bytecode_disabled': sys.dont_write_bytecode,
        'uid': os.getuid(), 'math_thread_env': {name: os.environ.get(name) for name in plan['math_thread_env']},
        'configured_memory_max_bytes': 32212254720, 'effective_cgroup_peak': 'not measured'},
    'duration_s': time.monotonic()-started, 'new_runtime_or_platform_calls': 0, 'new_pricing_calls': 0,
    'new_required_shape_calls': 0, 'new_full_cli_calls': 0,
    'limits': ['One completed readiness job only; no resubmission, manual stop or independent Pod permission change.',
        'Original source admission and126-binding budget are retained, not recomputed by this receipt.',
        'Expected replay hashes are recomputed from retained reference bytes; deleted cloud files cannot be reread here.',
        'Append/import RSS and serialized-reference throughput do not qualify production pricing/emission or full workload.',
        'Original automatic backend exit-handler Job Stop follows succeeded; annotations alone do not exclude external actions.',
        'Raw launcher logs remain permission-restricted in local scratch; only screened redacted copies are archived.']}
(archive/'parent_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'job_name':job,'files':receipt['archive_files'],
                  'bytes':receipt['archive_bytes'],'duration_s':receipt['duration_s'],'new_runtime_or_platform_calls':0}))
