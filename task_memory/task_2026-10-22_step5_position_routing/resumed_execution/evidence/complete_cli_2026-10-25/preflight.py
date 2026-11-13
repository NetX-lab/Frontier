"""Check the single complete-CLI plan without any platform or workload call."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

started = time.monotonic()
repo = Path('/data/ycfeng/Frontier/.worktrees/step5-preview')
base = repo / 'task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/complete_cli_2026-10-25'
sys.path.insert(0, str(base))
from complete_cli_worker import controller_paths, memory_reading, read_plan
plan_path = base / 'complete_cli_plan.json'
plan, plan_sha256 = read_plan(plan_path)
for name in ('complete_cli_worker.py', 'launch_complete_cli.py'):
    compile((base / name).read_bytes(), str(base / name), 'exec')
shell = subprocess.run(['bash', '-n', str(base / 'run_complete_cli.sh')], capture_output=True, text=True)
assert shell.returncode == 0, shell.stderr
runtime_env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
for name in ('BRAINPP_ACCESS_KEY', 'BRAINPP_SECRET_KEY'):
    runtime_env.pop(name, None)
dry = subprocess.run([sys.executable, '-B', str(base / 'launch_complete_cli.py'), '--plan', str(plan_path)],
                     env=runtime_env, capture_output=True, text=True, check=True)
(base / 'dry_run.stdout').write_text(dry.stdout)
assert not dry.stderr, dry.stderr
dry_report = json.loads(dry.stdout)
assert dry_report['status'] == 'DRY_RUN_NO_PLATFORM_OR_WORKLOAD_CALLS'
assert dry_report['plan_sha256'] == plan_sha256 and dry_report['full_cli_command'] == plan['command']
assert dry_report['resource'] == plan['resource'] and dry_report['outer_timeout_s'] is None
assert not Path(plan['local_output_root']).exists(), 'The new worker output was already created'
preparation = Path(plan['preparation_directory'])
sys.path.insert(0, str(preparation))
from joint_timing import check_bindings
bindings = json.loads(Path(plan['joint_bindings']).read_bytes())
identity = check_bindings(bindings, preparation)
assert hashlib.sha256(Path(plan['sdk_implicit_mount_observation']['source']).read_bytes()).hexdigest() == plan['sdk_implicit_mount_observation']['source_sha256']
sys.path.insert(0, plan['steptron_root'])
from steptron.exp.base_exp import ResourceConfig
from steptron.utils.stepmind import RJob
os.environ['EXP_ID'] = plan['experiment_id'] + '-spec-only'
server = socket.gethostbyname(socket.gethostname())
mount = f'{server}:{plan["local_mount_source"]}:{plan["local_mount_target"]}'
config = ResourceConfig(**plan['resource'], image=plan['image'], mounts=list(plan['mounts']),
    envs={'DISTRIBUTED_JOB': 'false', 'PYTHONDONTWRITEBYTECODE': '1',
          'PYTHONPATH': plan['local_mount_target'], **plan['math_thread_env']},
    annotations={'kubebrain.brainpp.cn/filestore-volume': mount}, pull_code=False)
job = RJob(name='D29-main09-spec-only', charged_group=plan['charged_group'], cfg=config)
command = dry_report['command']
for task_name, task_config in config:
    entrypoint = task_config.command.format(TORCHRUN='/jobutils/scripts/torchrun.sh', COMMAND=command)
    entrypoint = 'bash -lc ' + shlex.quote(f'cd {shlex.quote(plan["local_mount_target"])} && exec {entrypoint}')
    job.add(command=entrypoint, task_name=task_name, cfg=task_config, charged_group=plan['charged_group'])
assert set(job.tasks) == {'default'}
assert job.spec.preemptible == 'no'
task = job.tasks['default']
template = task.template
expected_mounts = set(plan['mounts']) | set(plan['sdk_implicit_mounts'])
assert set(task.mount_config) == expected_mounts and len(task.mount_config) == len(expected_mounts)
assert task.extend_annotations['kubebrain.brainpp.cn/filestore-volume'] == mount
assert template.image == plan['image'] and len(template.containers) == 1
assert template.affinity.positiveTags == ['H800']
assert task.replicas == 1 and task.is_critical and task.privateMachine.value == 'group'
for resource in (template.resources, template.containers[0].resources, template.containers[0].requests):
    assert str(resource.cpu) == '4' and str(resource.gpu) == '0'
    assert resource.memoryInMB == 64000 and resource.custom_resources == []
assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == plan_sha256, 'The SDK mutated the frozen plan'
assert check_bindings(bindings, preparation) == identity
try:
    paths = controller_paths()
    memory = memory_reading(paths)
except (OSError, ValueError) as error:
    memory = {'unavailable': f'{type(error).__name__}: {error}'}
report = {'status': 'PASS_COMPLETE_CLI_STATIC_DRY_RUN_AND_ORIGINAL_SDK_ONLY',
    'collected_utc': datetime.now(timezone.utc).isoformat(), 'source': identity,
    'plan_sha256': plan_sha256, 'script_sha256': plan['scripts'], 'python': sys.version,
    'syntax': 'PASS_TWO_PYTHON_AND_BASH_N', 'dry_run': dry_report,
    'sdk': {'resource_cpu': 4, 'resource_gpu': 0, 'resource_memory_mi': 64000, 'replicas': 1,
            'image': template.image, 'nfs_annotation': mount, 'mounts': task.mount_config,
            'command': template.command, 'private_machine': task.privateMachine.value},
    'local_memory_reader_only': memory, 'new_platform_calls': 0, 'new_workload_calls': 0,
    'ordinary_joint_calls': 0, 'required_shape_calls': 0, 'wall_s': time.monotonic() - started}
(base / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ('status', 'plan_sha256', 'syntax', 'new_platform_calls', 'new_workload_calls', 'wall_s')}, indent=2))
