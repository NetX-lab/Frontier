"""Submit one approved P2-64 package exactly once and verify the created RJob.

Usage (through submit_package.sh, which sets the StepMind environment):
    submit_package.sh <package dir> <launch wrapper> <scratch dir>

Steps: approval/seal/resource check, launch-evidence guard, check-only, one quota snapshot, one
detached launcher start, job-name capture from the launcher log, one read of the exact job, and
submission_acceptance.json. The launcher stays alive after this script exits; this script never
stops or signals anything.
Exit codes: 0 launched and all checks pass; 2 launched with failed or unknown checks;
3 launch evidence already existed, so nothing was launched; 1 a pre-launch check failed.
"""
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(sys.argv[1]).resolve(strict=True)
WRAPPER = Path(sys.argv[2]).resolve(strict=True)
SCRATCH = Path(sys.argv[3]).resolve(strict=True)
JOB_READER = Path(__file__).resolve().parent / 'read_job.py'
BUILD_PYTHON = '/data/ycfeng/envs/openmopd-py312/bin/python'
STEPMIND_PYTHON = '/data/ycfeng/tmp/stepmind-env/bin/python'
QUOTA_READER = '/data/ycfeng/tmp/glm53_dsv4/h200/monitor/read_quota.py'
WORKTREE = '/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028'
CHECK_ONLY_PASS = 'INPUT_HASHES_PASS_NO_SUBMISSION'
JOB_CREATE = re.compile(r'Job Create: \S+\((exp-\d+-\d+-\d+)\)')
PREEMPTIBLE_QUEUE = 'quotagroup-global-default'
# RDMA requests of the accepted H200 launch plan (generation04 submission acceptance).
EXPECTED_RDMA = {'mellanox.com/mlnx_rdma': '1', 'rdma/mlnx_shared': '8'}
JOB_NAME_WAIT_S = 180
READ_TIMEOUT_S = 60


def now():
    return datetime.now(timezone.utc).isoformat()


def write(name, value):
    (SCRATCH / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run(command, log_name, timeout, **kwargs):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as error:
        (SCRATCH / log_name).write_text(f'TIMEOUT after {timeout}s: {error}\n')
        return None, ''
    (SCRATCH / log_name).write_text(done.stdout + done.stderr)
    return done.returncode, done.stdout


def live_launchers():
    """Processes running this package's launcher or its wrapper, matched on exact argv elements."""
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:
            continue
        runs_launcher = str(PACKAGE / 'launch.py') in argv and '--check-only' not in argv
        runs_wrapper = len(argv) > 1 and Path(argv[0]).name == 'bash' and argv[1] == str(WRAPPER)
        if runs_launcher or runs_wrapper:
            found.append({'pid': int(entry.name), 'cmdline': ' '.join(argv).strip()})
    return found


def approval_checks(receipt, config):
    resources = receipt['resources']
    manifest_sha = hashlib.sha256((PACKAGE / 'manifest.json').read_bytes()).hexdigest()
    checks = {
        'receipt_approved': receipt['status'].startswith('APPROVED'),
        'receipt_package_matches': Path(receipt['package']) == PACKAGE,
        'manifest_sha256_matches': receipt['manifest_sha256'] == manifest_sha,
        'config_replicas': config['replica'] == resources['replicas'],
        'config_gpus_per_replica': config['gpu'] == resources['gpu_per_replica'],
        'config_gpu_type': config['gpu_type'] == resources['gpu_type'],
        'config_preemptible': config['preemptible'] is resources['preemptible'] is True,
        'config_charged_group': config['charged_group'] == resources['charged_group'],
        'config_timeout': config['timeout_seconds'] == resources['timeout_s'],
    }
    return checks, manifest_sha


def job_checks(identity, config, launcher_log):
    job = identity.get('job') or {}
    tasks = job.get('taskSpecs') or {}
    task = next(iter(tasks.values()), {})
    container = next(iter(task.get('containers') or []), {})
    limits = container.get('limits') or {}
    annotations = {**(job.get('job_annotations') or {}), **(task.get('task_annotations') or {})}
    filestore = annotations.get('kubebrain.brainpp.cn/filestore-volume', '')
    nfs_source = filestore.split(':', 1)[0]
    host_ips = sorted(set(subprocess.run(['hostname', '-I'], capture_output=True, text=True, timeout=10).stdout.split())
                      | set(socket.gethostbyname_ex(socket.gethostname())[2]))
    checks = {
        'creator_is_i_fengyicheng': job.get('creator') == 'i-fengyicheng',
        'one_task_spec': len(tasks) == 1,
        'replicas': task.get('replicas') == config['replica'],
        'gpus_per_replica': limits.get('nvidia.com/gpu') == str(config['gpu']),
        'gpu_tag': config['gpu_type'] in (task.get('positiveTags') or []),
        'preemptible_label': job.get('preemptible') == 'yes',
        'charged_group': job.get('quotagroup') == config['charged_group'],
        'queue_name': job.get('queue_name') == PREEMPTIBLE_QUEUE,
        'auto_restart_disabled': job.get('restartable') == 'false' and task.get('restartPolicy') == 'Never',
        'command_timeout': config['timeout_seconds'] in (container.get('command_timeouts_s') or []),
        'company_image': container.get('image') == config['image'],
        'rdma': all(limits.get(k) == v for k, v in EXPECTED_RDMA.items()),
        'cloud_mount': config['cloud_mount'].removeprefix('juicefs+') in annotations.get('mount.brainpp.cn/juicefs', ''),
        'nfs_source_is_current_host': nfs_source in host_ips,
        'nfs_paths_are_package': filestore == f'{nfs_source}:{PACKAGE}:{PACKAGE}',
        'job_created_once': len(JOB_CREATE.findall(launcher_log)) == 1,
        'identity_read_errors_zero': bool(job) and not identity.get('errors'),
    }
    observed = {
        'creator': job.get('creator'), 'replicas': task.get('replicas'), 'limits': limits,
        'positiveTags': task.get('positiveTags'), 'preemptible': job.get('preemptible'),
        'quotagroup': job.get('quotagroup'), 'queue_name': job.get('queue_name'),
        'restartable': job.get('restartable'), 'restartPolicy': task.get('restartPolicy'),
        'command_timeouts_s': container.get('command_timeouts_s'), 'image': container.get('image'),
        'juicefs': annotations.get('mount.brainpp.cn/juicefs'), 'filestore_volume': filestore,
        'host_ips': host_ips, 'maxRunningDuration': task.get('maxRunningDuration'),
        'maxWaitDuration': task.get('maxWaitDuration'), 'autoDeleteDuration': job.get('autoDeleteDuration'),
        'backoffLimit': job.get('backoffLimit'),
    }
    return checks, observed


def finish(record, result, code):
    record['result'] = result
    record['recorded_utc'] = now()
    write('submission_acceptance.json', record)
    print(result, json.dumps({k: record.get(k) for k in
                              ('job_name', 'launch', 'current_phase', 'failed_checks')}), flush=True)
    return code


def main():
    record = {'started_utc': now(), 'package': str(PACKAGE), 'wrapper': str(WRAPPER),
              'scratch': str(SCRATCH), 'controls_issued': 0}
    receipt = json.loads((PACKAGE.parent / 'approval_receipt.json').read_text())
    config = json.loads((PACKAGE / 'config.json').read_text())
    checks, manifest_sha = approval_checks(receipt, config)
    record['approval'] = {'decision': receipt['decision_id'], 'manifest_sha256': manifest_sha, 'checks': checks}

    markers = {name: (PACKAGE / name).exists()
               for name in ('launcher_started.json', 'launcher_exit.json', 'case/runs')}
    markers['scratch_launch_record'] = (SCRATCH / 'launch_record.json').exists()
    live = live_launchers()
    record['pre_launch_evidence'] = {'markers': markers, 'live_launchers': live}
    if any(markers.values()) or live:
        return finish(record, 'LAUNCH_EVIDENCE_EXISTS_NOT_LAUNCHED', 3)
    if not all(checks.values()):
        return finish(record, 'APPROVAL_CHECK_FAILED_NOT_LAUNCHED', 1)

    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    code, out = run([BUILD_PYTHON, '-B', str(PACKAGE / 'launch.py'), '--check-only'],
                    'check_only.txt', 300, cwd=PACKAGE, env=env)
    record['check_only'] = {'returncode': code, 'pass': CHECK_ONLY_PASS in out}
    if code != 0 or not record['check_only']['pass']:
        return finish(record, 'CHECK_ONLY_FAILED_NOT_LAUNCHED', 1)

    code, _ = run([STEPMIND_PYTHON, QUOTA_READER, str(SCRATCH / 'quota_snapshot.json')],
                  'quota_read.log', READ_TIMEOUT_S)
    record['quota_read_returncode'] = code

    log_path = SCRATCH / 'launcher.log'
    with open(log_path, 'ab') as log:
        launcher = subprocess.Popen(['bash', str(WRAPPER)], cwd=WORKTREE, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record['launch'] = {'launched_utc': now(), 'pid': launcher.pid, 'wrapper': str(WRAPPER),
                        'cwd': WORKTREE, 'start_new_session': True, 'stdin': 'DEVNULL',
                        'stdout': str(log_path)}
    write('launch_record.json', record['launch'])
    print('LAUNCHED pid', launcher.pid, flush=True)

    record['job_name'] = None
    deadline = time.time() + JOB_NAME_WAIT_S
    while time.time() < deadline and launcher.poll() is None:
        match = JOB_CREATE.search(log_path.read_text(errors='replace'))
        if match:
            record['job_name'] = match.group(1)
            break
        time.sleep(2)
    if record['job_name'] is None:
        record['launcher_returncode'] = launcher.poll()
        return finish(record, 'LAUNCHED_JOB_NAME_UNKNOWN', 2)

    time.sleep(5)
    identity_path = SCRATCH / 'submission_identity.json'
    code, _ = run([STEPMIND_PYTHON, str(JOB_READER), record['job_name'], str(identity_path)],
                  'read_job.log', READ_TIMEOUT_S)
    identity = json.loads(identity_path.read_text()) if code == 0 else {}
    checks, observed = job_checks(identity, config, log_path.read_text(errors='replace'))
    checks['launcher_alive'] = launcher.poll() is None
    job = identity.get('job') or {}
    record.update({
        'job_uid': job.get('uid'), 'job_creation_timestamp': job.get('creationTimestamp'),
        'current_phase': job.get('phase'), 'start_time': job.get('startTime'),
        'replicas_reported': len((identity.get('replicas') or {}).get('rows') or []),
        'read_returncode': code,
        'read_errors': {'identity': identity.get('errors'),
                        'replicas': (identity.get('replicas') or {}).get('errors')},
        'checks': checks, 'observed': observed,
        'failed_checks': sorted(k for k, ok in checks.items() if not ok),
    })
    if record['failed_checks']:
        return finish(record, 'SUBMITTED_CHECKS_FAILED', 2)
    return finish(record, 'SUBMITTED_ALL_CHECKS_PASS', 0)


if __name__ == '__main__':
    sys.exit(main())
