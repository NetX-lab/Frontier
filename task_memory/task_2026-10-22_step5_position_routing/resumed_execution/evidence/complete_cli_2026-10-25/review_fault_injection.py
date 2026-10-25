"""Reproduce the two launcher evidence defects using the original SDK log and fake APIs."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from loguru import logger

started = time.monotonic()
base = Path('/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/complete_cli_2026-10-25')
real_plan = json.loads((base / 'complete_cli_plan.json').read_bytes())
scratch = Path('/data/ycfeng/tmp/d29_main09_review_faults_20261025_01')
scratch.mkdir(exist_ok=False)
sys.path.insert(0, str(base))
sdk_path = Path(real_plan['sdk_submission_observation']['source'])
assert hashlib.sha256(sdk_path.read_bytes()).hexdigest() == real_plan['sdk_submission_observation']['source_sha256']
controller_ast = next(node for node in ast.parse(sdk_path.read_bytes()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'RJobController')
identity = {'source_commit': real_plan['source_commit'], 'records_head': 'FAULT_TEST_ONLY',
            'source_paths': 25, 'fixture_metadata_paths': 262}
results = []

def run_case(label, initial, started_worker, query_failure, success):
    plan = copy.deepcopy(real_plan)
    plan['local_output_root'] = str(scratch / label)
    job_name = 'exp-1025-fault-only-000001'
    calls = []
    phase = 'Succeeded' if success else 'Failed'
    resources = {'cpu': '4', 'memory': '64000Mi', 'nvidia.com/gpu': '0'}
    job = {'metadata': {'labels': {'kubebrain.brainpp.cn/creator': plan['principal'],
           'quotagroup.brainpp.cn/quotagroup': plan['charged_group'],
           'quotagroup.brainpp.cn/preemptible': 'no'}}, 'status': {'phase': phase},
           'spec': {'taskSpecs': {'default': {'replicas': 1, 'template': {
               'metadata': {'annotations': {'kubebrain.brainpp.cn/filestore-volume':
                   f'{socket.gethostbyname(socket.gethostname())}:/data/ycfeng:/data/ycfeng'}},
               'spec': {'containers': [{'image': plan['image'],
                   'resources': {'requests': resources, 'limits': resources}}]}}}}}}
    replica = {'metadata': {'name': job_name + '-replica'}, 'status': {'phase': phase,
        'nodeName': 'fault-worker-only', 'containerStatuses': [{'imageID': plan['image'],
        'restartCount': 0, 'state': {'terminated': {'exitCode': 0 if success else 1,
        'reason': 'Completed' if success else 'Error'}}}]}}
    class FakeAPI:
        def get_namespaced_custom_object(self, **kwargs):
            calls.append(['get', kwargs['name']])
            if query_failure:
                raise ValueError('FOLLOWUP_QUERY_FAILURE')
            return job
        def list_namespaced_custom_object(self, **kwargs):
            calls.append(['list', kwargs['label_selector']])
            return {'items': [replica]}
    client = SimpleNamespace(api=FakeAPI(), group='fake', version='fake', namespace='fake', plural='rjobs',
                             submit=lambda value, no_packaging: job_name)
    namespace = {'__name__': 'steptron.utils.stepmind', 'get_rjob_client': lambda: client,
                 'logger': logger, 'RJob': object, 'ApiException': Exception}
    exec(compile(ast.Module(body=[controller_ast], type_ignores=[]), str(sdk_path), 'exec'), namespace)
    created_before_wait = []
    def fake_spawn(*args, **kwargs):
        namespace['RJobController']().submit(SimpleNamespace(
            meta=SimpleNamespace(name=job_name), name=plan['experiment_id'], job=object()))
        output = Path(plan['local_output_root'])
        created_before_wait.append(json.loads((output / 'launcher.json').read_bytes()).get('job_name'))
        if started_worker or success:
            (output / 'worker').mkdir()
            (output / 'worker/worker_started.json').write_text(json.dumps({'job_name': job_name}))
        if success:
            business = {'status': 'PASS_COMPLETE_CPU_FIXTURE_CLI', 'source_before': identity,
                'source_after': identity, 'worker_hostname': 'fault-worker-only', 'plan_sha256': 'FAULT_PLAN_ONLY'}
            (output / 'worker/complete_cli.json').write_text(json.dumps(business))
            return SimpleNamespace(rjob=SimpleNamespace(meta=SimpleNamespace(name=job_name)), poll=lambda: 'succeeded')
        raise RuntimeError('INITIAL_WAIT_FAILURE')
    sdk_module = ModuleType('steptron.utils.stepmind')
    sdk_module.spawn_tasks = fake_spawn
    sdk_module.get_rjob_client = lambda: client
    resources_module = ModuleType('steptron.exp.base_exp')
    resources_module.ResourceConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    bindings_module = ModuleType('joint_timing')
    bindings_module.check_bindings = lambda *args: identity
    struct_module = ModuleType('brainpp.rjob.struct')
    struct_module.label_rjobname = 'rjob-name'
    modules = {name: ModuleType(name) for name in ('steptron', 'steptron.exp', 'steptron.utils', 'brainpp', 'brainpp.rjob')}
    modules.update({'steptron.exp.base_exp': resources_module, 'steptron.utils.stepmind': sdk_module,
                    'joint_timing': bindings_module, 'brainpp.rjob.struct': struct_module})
    path = base / ('launch_complete_cli_initial.py' if initial else 'launch_complete_cli.py')
    spec = importlib.util.spec_from_file_location('fault_' + label, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.read_plan = lambda path: (plan, 'FAULT_PLAN_ONLY')
    cwd = Path.cwd()
    caught = None
    try:
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {
            'STEPMIND_BACKEND': 'rjob', 'BRAINPP_ACCESS_KEY': 'UNUSED_FAULT_TEST',
            'BRAINPP_SECRET_KEY': 'UNUSED_FAULT_TEST'}), patch.object(sys, 'argv', [
            str(path), '--plan', str(base / 'complete_cli_plan.json'), '--submit']):
            try:
                module.main()
            except BaseException as error:
                caught = f'{type(error).__name__}: {error}'
    finally:
        os.chdir(cwd)
    output = Path(plan['local_output_root'])
    receipt = json.loads((output / 'launcher.json').read_bytes())
    logs = {p.name: p.read_text() for p in output.glob('*.failure.log')}
    if (output / 'launcher.failure.log').is_file():
        logs['launcher.failure.log'] = (output / 'launcher.failure.log').read_text()
    row = {'case': label, 'launcher_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'creation_identity_at_wait': created_before_wait, 'fake_api_calls': calls,
           'caught': caught, 'receipt': receipt, 'failure_logs': logs}
    if initial and not started_worker:
        assert created_before_wait == [None] and not calls and 'job_name' not in receipt
        row['observed'] = 'RED_REPRODUCED_MISSING_PRE_WORKER_TERMINAL_EVIDENCE'
    elif initial:
        assert 'FOLLOWUP_QUERY_FAILURE' in logs['launcher.failure.log']
        assert 'INITIAL_WAIT_FAILURE' not in logs['launcher.failure.log']
        assert 'launch_error' not in receipt
        row['observed'] = 'RED_REPRODUCED_INITIAL_TRACEBACK_OVERWRITE'
    else:
        assert created_before_wait == [job_name] and receipt['job_name'] == job_name
        if query_failure:
            assert 'INITIAL_WAIT_FAILURE' in logs['launcher.failure.log']
            assert 'FOLLOWUP_QUERY_FAILURE' in logs['launcher.followup.failure.log']
            assert 'INITIAL_WAIT_FAILURE' in receipt['launch_error']
            assert 'FOLLOWUP_QUERY_FAILURE' in receipt['error']
        elif success:
            assert caught is None and receipt['status'] == 'PASS_COMPLETE_CPU_FIXTURE_PLATFORM_AND_BUSINESS'
            assert len(calls) == 2
        else:
            assert len(calls) == 2 and receipt['platform_status']['phase'] == 'Failed'
            assert receipt['replica_status']['containerStatuses'][0]['state']['terminated']['exitCode'] == 1
            assert receipt['business_status'] == 'UNAVAILABLE' and receipt['status'] == 'FAIL'
            assert 'INITIAL_WAIT_FAILURE' in receipt['launch_error']
        row['observed'] = 'GREEN'
    results.append(row)

run_case('initial_no_worker', True, False, False, False)
run_case('initial_query_overwrite', True, True, True, False)
run_case('final_no_worker', False, False, False, False)
run_case('final_query_failure', False, False, True, False)
run_case('final_success', False, False, False, True)
report = {'status': 'PASS_TWO_REPRODUCED_DEFECTS_AND_THREE_REPAIRED_SCENARIOS',
          'collected_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
          'original_sdk_source_sha256': real_plan['sdk_submission_observation']['source_sha256'],
          'new_platform_calls': 0, 'new_workload_calls': 0, 'ordinary_joint_calls': 0,
          'required_shape_calls': 0, 'results': results, 'wall_s': time.monotonic() - started}
(base / 'review_fault_injection.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ('status', 'new_platform_calls', 'new_workload_calls', 'wall_s')}, indent=2))
