"""Fake-only TP16 caller, supported-worker identity and census checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(os.environ.get('STEP5_TP16_TEST_SOURCE_ROOT', HERE))
sys.path.insert(0, str(SOURCE_ROOT))
import check_role_tp16 as caller


@pytest.fixture(autouse=True)
def trusted_task_locator(monkeypatch):
    monkeypatch.setattr(caller, 'TASK', HERE.parents[2]/'task_memory/task_2026-10-15_step5_preview_h800')


@pytest.fixture
def config(tmp_path):
    return dict(task_root=str(caller.TASK), generation='fake-tp16',
        resources=dict(nodes=2, gpu_per_node=8, cpu_per_node=32, memory_mi_per_node=128000,
                       worker_seconds=2400, gpu_type='H800', charged_group='codesign'),
        image='a.z01.normandyfunc.com/docker-public/vllm/vllm-openai'
              '@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20',
        vllm_version=caller.VERSION, workload=dict(stage='capacity_initialization'),
        engine=dict(max_model_len=8192, max_num_batched_tokens=512, max_num_seqs=1),
        rpc_seconds=20, record_bytes=1024*1024, archive_bytes=4*1024*1024,
        archive_reserve_seconds=300,
        marker_root=str(caller.TASK/'fake_unused_tp16_marker_generation'),
        worker_local_root=str(tmp_path/'native_local'),
        archive_root='/mnt/codesign-exp/ycfeng/fake_unused_tp16_generation', native_cli='/fake/bin/vllm',
        caller_sha256=caller.sha256(SOURCE_ROOT/'check_role_tp16.py'),
        observer_sha256=caller.sha256(SOURCE_ROOT/'role_capacity_observation.py'),
        supervisor_sha256=caller.sha256(SOURCE_ROOT/'role_capacity_supervision.py'))


@pytest.fixture(autouse=True)
def forbid_real_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Fake CPU test attempted a real child, GPU query, socket or native import')
    monkeypatch.setattr(caller.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(caller.subprocess, 'run', forbidden)
    monkeypatch.setattr(caller.supervision.socket, 'socket', forbidden)
    monkeypatch.setattr(caller.os, 'execv', forbidden)
    monkeypatch.setattr(caller.os, 'kill', forbidden)
    import builtins
    original_import = builtins.__import__
    def fake_only_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm', 'steptron', 'stepmind'):
            forbidden()
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', fake_only_import)


def nodes(config, tmp_path):
    return [dict(node_rank=rank, generation=config['generation'], hostname=f'fake-{rank}',
        boot_id=f'boot-{rank}', pid=100+rank, pid_namespace=f'pid:[{rank}]',
        observer_sha256=config['observer_sha256'], gpu_uuids=[f'GPU-{rank}-{i}' for i in range(8)],
        endpoint=dict(address='10.1.2.3', port=6000, port_source='environment'),
        workload_deadline=2310, local_output=str(tmp_path/f'local_{rank}')) for rank in (0, 1)]


def worker_row(node, local):
    return dict(generation=node['generation'], boot_id=node['boot_id'], node_rank=node['node_rank'],
        pid_namespace=node['pid_namespace'], observer_sha256=node['observer_sha256'],
        pid=1000+node['node_rank']*8+local, start_time=333+local,
        parent_pid=400+node['node_rank'], parent_start_time=200,
        rank=node['node_rank']*8+local, local_rank=local)


def snapshots(node_rows):
    expected = caller.expected_census()
    rows = []
    for node in node_rows:
        for local in range(8):
            row = worker_row(node, local)
            row.update(gpu_uuid=node['gpu_uuids'][local], status='metadata_only_not_G2',
                completed_stages=list(caller.observation.STAGES), resource_evidence_errors=[],
                parallel_config=dict(tensor_parallel_size=16, data_parallel_size=1,
                    pipeline_parallel_size=1, enable_expert_parallel=False, nnodes=2, node_rank=node['node_rank']),
                configuration=dict(model_dtype='torch.bfloat16', gpu_memory_utilization=0.92,
                                   speculative=False, prefix_caching=False),
                module_classes={'fake.'+name: count for name, count in expected['module_counts'].items()},
                routed_experts=[dict(name=f'layer{i}', count=expected['experts_per_layer'])
                                for i in range(expected['routed_layers'])])
            rows.append(row)
    return rows


def test_preflight_is_inert_and_locks_real_sources(config):
    before = set(sys.modules)
    row = caller.preflight(config)
    assert row['scope'] == 'inert_preflight_not_native_qualification'
    assert 'vllm/entrypoints/cli/serve.py' in row['sources']
    assert len(row['sources']) >= 38 and 'changed_input_all_rank_graph_replay' in row['remaining_g2']
    assert not any(name.split('.')[0] in ('torch', 'vllm', 'steptron', 'stepmind') for name in set(sys.modules)-before)
    assert not Path(config['marker_root']).exists()


@pytest.mark.parametrize('key,value', [('rpc_seconds', 0), ('rpc_seconds', float('inf')),
    ('rpc_seconds', float('nan')), ('record_bytes', 1.2), ('archive_bytes', -1),
    ('generation', '../reuse'), ('vllm_version', 'alternate'), ('native_cli', 'vllm'),
    ('observer_sha256', 'old'), ('caller_sha256', 'old'), ('supervisor_sha256', 'old'),
    ('image', 'alternate'), ('worker_local_root', '/mnt/codesign-exp/ycfeng/highfreq')])
def test_explicit_configuration_rejects_drift(config, key, value):
    config[key] = value
    with pytest.raises(ValueError): caller.preflight(config)


@pytest.mark.parametrize('key,value', [('nodes', 1), ('gpu_per_node', 16), ('cpu_per_node', 64),
    ('memory_mi_per_node', 128), ('worker_seconds', 2401), ('charged_group', 'step_main')])
def test_resource_scope_is_not_expanded(config, key, value):
    config['resources'][key] = value
    with pytest.raises(ValueError): caller.preflight(config)


def test_no_unimplemented_g2_workload_claim(config):
    config['workload'] = dict(stage='graph_replay')
    with pytest.raises(ValueError, match='faithful G2'): caller.preflight(config)


def test_node0_llm_node1_headless_geometry(config, tmp_path):
    node_rows = nodes(config, tmp_path)
    kwargs = caller.engine_kwargs(config, node_rows[0])
    assert kwargs['node_rank'] == 0 and kwargs['nnodes'] == 2
    assert (kwargs['tensor_parallel_size'], kwargs['data_parallel_size'], kwargs['pipeline_parallel_size']) == (16, 1, 1)
    assert kwargs['skip_tokenizer_init'] is False and kwargs['load_format'] == 'dummy'
    assert all(key not in kwargs for key in ('dtype', 'attention_backend', 'distributed_executor_backend',
        'enable_expert_parallel', 'quantization', 'gpu_memory_utilization', 'compilation_config'))
    command = caller.headless_argv(config, node_rows[1])
    assert command[:4] == ['/fake/bin/vllm', 'serve', str(caller.TASK/'reference_inputs'), '--headless']
    assert command[command.index('--node-rank')+1] == '1'
    assert command[command.index('--data-parallel-size')+1] == '1'
    assert '--data-parallel-rank' not in command and '--skip-tokenizer-init' not in command
    with pytest.raises(ValueError): caller.headless_argv(config, node_rows[0])
    with pytest.raises(ValueError): caller.execute_leader(config, node_rows[1], {})


def test_explicit_master_port_no_guessed_default(config):
    with pytest.raises(ValueError, match='MASTER_PORT'):
        caller.node_record(config, dict(NODE_RANK='0'))


@pytest.mark.parametrize('mode', ['complete', 'partial', 'missing', 'malformed', 'wrongboot', 'duplicate',
                                 'wrongrank', 'wrongnamespace', 'child', 'supervisor'])
def test_supported_worker_inventory_handoff(config, tmp_path, mode):
    node = nodes(config, tmp_path)[0]; output = tmp_path/'workers'; output.mkdir()
    count = 8 if mode not in ('partial', 'missing') else 3 if mode == 'partial' else 0
    for local in range(count):
        row = worker_row(node, local)
        if local == 0:
            if mode == 'wrongboot': row['boot_id'] = 'foreign'
            if mode == 'wrongrank': row['rank'] = 8
            if mode == 'wrongnamespace': row['pid_namespace'] = 'foreign'
            if mode == 'child': row['pid'] = 444
            if mode == 'supervisor': row['pid'] = node['pid']
        if local == 1 and mode == 'duplicate': row['pid'] = 1000
        path = output/f"worker_{row['rank']}_{row['pid']}.json"
        path.write_text('{' if mode == 'malformed' and local == 0 else json.dumps(row))
    inv = caller.WorkerInventory(output, node); inv.child_pid = 444
    original = inv.workers
    record = inv.snapshot()
    assert inv.workers is original
    assert record['complete'] == (mode == 'complete')
    if mode == 'partial': assert len(record['workers']) == 3 and record['errors'] == []
    if mode == 'complete': assert inv.workers == [(1000+i, 333+i) for i in range(8)]
    if mode not in ('complete', 'partial', 'missing'): assert record['errors']
    if mode in ('child', 'supervisor'):
        assert all(pid not in (node['pid'], inv.child_pid) for pid, start in record['workers'])


def test_inventory_poll_retains_raw_native_return_and_error(config, tmp_path):
    node = nodes(config, tmp_path)[0]; output = tmp_path/'workers'; output.mkdir()
    inv = caller.WorkerInventory(output, node); events = []
    error = RuntimeError('native poll failure')
    child = SimpleNamespace(pid=42, poll=lambda: 9, send_signal=lambda number: events.append(number))
    wrapper = inv.spawn(['FAKE'], popen=lambda argv: child)
    (output/'worker_0_1000.json').write_text(json.dumps(worker_row(node, 0)))
    assert wrapper.poll() == 9 and inv.workers == [(1000, 333)]
    wrapper.send_signal(signal.SIGTERM); assert events == [signal.SIGTERM]
    def fail(): raise error
    child.poll = fail
    with pytest.raises(RuntimeError) as caught: wrapper.poll()
    assert caught.value is error


@pytest.mark.parametrize('fault', ['none', 'missing', 'duplicate', 'boot', 'namespace', 'uuid', 'start', 'phase',
    'reduced_layer', 'reduced_expert', 'wrapper', 'tp', 'ep', 'dtype', 'resource'])
def test_all_sixteen_rank_capacity_census_not_g2(config, tmp_path, fault):
    node_rows = nodes(config, tmp_path); rows = snapshots(node_rows)
    if fault == 'missing': rows.pop()
    elif fault == 'duplicate': rows[15] = copy.deepcopy(rows[0])
    elif fault == 'boot': node_rows[1]['boot_id'] = node_rows[0]['boot_id']
    elif fault == 'namespace': rows[0]['pid_namespace'] = 'foreign'
    elif fault == 'uuid': rows[0]['gpu_uuid'] = rows[1]['gpu_uuid']
    elif fault == 'start': rows[0]['start_time'] = None
    elif fault == 'phase': rows[0]['completed_stages'].remove('initialize_from_config')
    elif fault == 'reduced_layer': rows[0]['module_classes']['fake.Step5DecoderLayer'] -= 1
    elif fault == 'reduced_expert': rows[0]['routed_experts'][0]['count'] -= 1
    elif fault == 'wrapper': rows[0]['module_classes']['fake.Step5ForConditionalGeneration'] = 0
    elif fault == 'tp': rows[0]['parallel_config']['tensor_parallel_size'] = 8
    elif fault == 'ep': rows[0]['parallel_config']['enable_expert_parallel'] = True
    elif fault == 'dtype': rows[0]['configuration']['model_dtype'] = 'torch.float16'
    elif fault == 'resource': rows[0]['resource_evidence_errors'].append('missing lifecycle')
    if fault != 'none':
        with pytest.raises((ValueError, KeyError)): caller.all_rank_census(rows, node_rows, caller.expected_census())
    else:
        result = caller.all_rank_census(rows, node_rows, caller.expected_census())
        assert result['ranks'] == 16 and result['status'].endswith('not_G2')
        assert set(result['remaining_g2'].values()) == {'UNMEASURED'}


@pytest.mark.parametrize('fault', ['none', 'source', 'native', 'rpc', 'census', 'diagnostic', 'terminal'])
def test_leader_terminal_actual_child_and_raw_failure(config, tmp_path, monkeypatch, fault):
    node_rows = nodes(config, tmp_path); node = node_rows[0]
    root = tmp_path/'markers'; root.mkdir(); config['marker_root'] = str(root)
    Path(node['local_output']).mkdir()
    for item in node_rows: (root/f"node_{item['node_rank']}_inventory.json").write_text(json.dumps(item))
    calls, terminals = [], []
    error = RuntimeError('original native failure')
    rows = snapshots(node_rows)
    if fault == 'census': rows.pop()
    def rpc(method, timeout):
        calls.append(('rpc', method, timeout))
        if fault == 'rpc': raise error
        return rows
    llm = SimpleNamespace(collective_rpc=rpc)
    def factory(**kwargs):
        calls.append(('LLM', kwargs))
        if fault == 'native': raise error
        return llm
    def verify(*args):
        calls.append(('source',))
        if fault == 'source': raise error
    def terminal(config, node, result, reason):
        # Exercise the existing private marker writer, not a fake supervisor publisher.
        if fault == 'terminal': raise OSError('diagnostic terminal failure')
        terminals.append(caller.publish_terminal(config, node, result, reason))
    if fault == 'diagnostic':
        monkeypatch.setattr(caller, 'save', lambda *args: (_ for _ in ()).throw(OSError('record disk full')))
    if fault in ('source', 'native', 'rpc'):
        with pytest.raises(RuntimeError) as caught:
            caller.execute_leader(config, node, {}, llm_factory=factory, verify=verify, clock=lambda: 3, publish=terminal)
        assert caught.value is error and terminals[0]['result'] == 'FAIL'
    else:
        assert caller.execute_leader(config, node, {}, llm_factory=factory, verify=verify,
                                     clock=lambda: 3, publish=terminal) is llm
        if fault != 'terminal': assert terminals[0]['result'] == ('PASS' if fault == 'none' else 'FAIL')
    if terminals:
        assert terminals[0]['pid'] == os.getpid() and terminals[0]['pid'] != node['pid']
        assert terminals[0]['node_rank'] == 0 and not terminals[0].get('supervisor_fallback')
    assert not (root/'node_1_workload_terminal.json').exists()
    assert sum(call[0] == 'LLM' for call in calls) <= 1


def test_import_is_inert_without_real_children_or_gpu_queries():
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location('tp16_inert_import', HERE/'check_role_tp16.py')
    imported = importlib.util.module_from_spec(spec); spec.loader.exec_module(imported)
    assert not any(name.split('.')[0] in ('torch', 'vllm', 'steptron', 'stepmind') for name in set(sys.modules)-before)


@pytest.mark.parametrize('gpu_type,charged_group', [('H800', 'codesign'), ('H200', 'step_main')])
def test_gpu_uuid_inventory_uses_only_explicit_fake_query(config, gpu_type, charged_group):
    config['resources'].update(gpu_type=gpu_type, charged_group=charged_group)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stderr='',
            stdout=''.join(f'{i}, GPU-{i}, NVIDIA {gpu_type}\n' for i in reversed(range(8))))
    assert caller.gpu_inventory(config['resources'], run) == [f'GPU-{i}' for i in range(8)]
    assert calls[0][1]['timeout'] == 10
    with pytest.raises(ValueError):
        caller.gpu_inventory(config['resources'], lambda *a, **k: SimpleNamespace(returncode=1, stderr='fake error'))



@pytest.mark.parametrize('gpu_type,charged_group', [('H800', 'codesign'), ('H200', 'step_main')])
def test_preflight_accepts_only_authorized_hardware_pair_without_changing_native_work(config, gpu_type, charged_group):
    config['resources'].update(gpu_type=gpu_type, charged_group=charged_group)
    original = copy.deepcopy(config)
    result = caller.preflight(config)
    assert result['resources'] == original['resources'] and config == original
    assert caller.engine_kwargs(config, {'node_rank': 0, 'endpoint': {'address': '10.1.2.3', 'port': 6000}})['load_format'] == 'dummy'


@pytest.mark.parametrize('gpu_type,charged_group', [('H200', 'codesign'), ('H800', 'step_main'), ('B200', 'step_main')])
def test_preflight_rejects_wrong_hardware_quota_pairs(config, gpu_type, charged_group):
    config['resources'].update(gpu_type=gpu_type, charged_group=charged_group)
    with pytest.raises(ValueError, match='GPU and quota-group'):
        caller.preflight(config)


@pytest.mark.parametrize('fault', ['wrong_type', 'mixed_type', 'missing', 'duplicate_uuid', 'duplicate_index', 'invalid_uuid'])
@pytest.mark.parametrize('gpu_type', ['H800', 'H200'])
def test_gpu_inventory_rejects_mismatched_actual_hardware(config, gpu_type, fault):
    config['resources']['gpu_type'] = gpu_type
    rows = [[i, f'GPU-{i}', f'NVIDIA {gpu_type}'] for i in range(8)]
    if fault in ('wrong_type', 'mixed_type'):
        for row in rows if fault == 'wrong_type' else rows[:1]:
            row[2] = 'NVIDIA H200' if gpu_type == 'H800' else 'NVIDIA H800'
    elif fault == 'missing':
        rows.pop()
    elif fault == 'duplicate_uuid':
        rows[1][1] = rows[0][1]
    elif fault == 'duplicate_index':
        rows[1][0] = rows[0][0]
    else:
        rows[0][1] = 'not-a-physical-gpu'
    def run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stderr='', stdout=''.join(', '.join(map(str, row))+'\n' for row in rows))
    with pytest.raises(ValueError, match='Allocated GPU'):
        caller.gpu_inventory(config['resources'], run)

def test_partial_inventory_retains_other_valid_live_workers(config, tmp_path):
    node = nodes(config, tmp_path)[0]; output = tmp_path/'workers'; output.mkdir()
    (output/'worker_0_1000.json').write_text('{')
    for local in range(1, 8):
        row = worker_row(node, local)
        (output/f"worker_{row['rank']}_{row['pid']}.json").write_text(json.dumps(row))
    inv = caller.WorkerInventory(output, node)
    row = inv.snapshot()
    assert len(row['workers']) == 7 and row['errors'] and not row['complete']


def test_actual_gpu_mapping_allows_cuda_visible_device_permutation(config, tmp_path):
    node_rows = nodes(config, tmp_path); rows = snapshots(node_rows)
    for row in rows:
        row['gpu_uuid'] = node_rows[row['node_rank']]['gpu_uuids'][7-row['local_rank']]
    assert caller.all_rank_census(rows, node_rows, caller.expected_census())['ranks'] == 16


def capacity03_census():
    path = HERE.parents[1]/'fixtures/step5_environment/tp16_capacity03_uuid_census.json'
    fixture = json.loads(path.read_text())
    rows = [copy.deepcopy(dict(fixture['common_worker_fields'],
        parallel_config=fixture['nodes'][row['node_rank']]['worker_parallel_config'], **row))
        for row in fixture['workers']]
    return fixture, rows, copy.deepcopy(fixture['nodes'])


@pytest.mark.parametrize('representation', ['bare', 'prefixed', 'mixed'])
def test_actual_capacity03_uuid_representations_preserve_raw_metadata(representation):
    fixture, rows, node_rows = capacity03_census()
    for row in rows:
        if representation == 'prefixed' or representation == 'mixed' and row['rank'] % 2:
            row['gpu_uuid'] = 'GPU-'+row['gpu_uuid']
    before = copy.deepcopy((rows, node_rows))
    for node in node_rows:
        physical = [node['gpu_uuids'].index('GPU-'+row['gpu_uuid'].removeprefix('GPU-'))
                    for row in rows if row['node_rank'] == node['node_rank']]
        assert physical == fixture['physical_visibility_order'][node['node_rank']]
    result = caller.all_rank_census(rows, node_rows, caller.expected_census())
    assert result['ranks'] == 16 and result['status'].endswith('not_G2')
    assert set(result['remaining_g2'].values()) == {'UNMEASURED'}
    assert (rows, node_rows) == before


@pytest.mark.parametrize('fault', ['missing_rank', 'duplicate_uuid', 'foreign_uuid', 'local_rank',
    'boot', 'namespace', 'pid', 'start', 'parent', 'parent_start', 'reduced_layer',
    'reduced_expert', 'phase', 'tp', 'ep', 'dtype', 'resource'])
def test_actual_capacity03_census_rejects_physical_binding_and_model_faults(fault):
    fixture, rows, node_rows = capacity03_census()
    if fault == 'missing_rank': rows.pop()
    elif fault == 'duplicate_uuid': rows[0]['gpu_uuid'] = 'GPU-'+rows[1]['gpu_uuid']
    elif fault == 'foreign_uuid': rows[0]['gpu_uuid'] = node_rows[1]['gpu_uuids'][0]
    elif fault == 'local_rank': rows[0]['local_rank'] = 1
    elif fault == 'boot': rows[0]['boot_id'] = node_rows[1]['boot_id']
    elif fault == 'namespace': rows[0]['pid_namespace'] = node_rows[1]['pid_namespace']
    elif fault == 'pid': rows[0]['pid'] = node_rows[0]['pid']
    elif fault == 'start': rows[0]['start_time'] = None
    elif fault == 'parent': rows[0]['parent_pid'] = rows[1]['parent_pid']+1
    elif fault == 'parent_start': rows[0]['parent_start_time'] = None
    elif fault == 'reduced_layer':
        key = next(name for name in rows[0]['module_classes'] if name.endswith('.Step5DecoderLayer'))
        rows[0]['module_classes'][key] -= 1
    elif fault == 'reduced_expert': rows[0]['routed_experts'][0]['count'] -= 1
    elif fault == 'phase': rows[0]['completed_stages'].pop()
    elif fault == 'tp': rows[0]['parallel_config']['tensor_parallel_size'] = 8
    elif fault == 'ep': rows[0]['parallel_config']['enable_expert_parallel'] = True
    elif fault == 'dtype': rows[0]['configuration']['model_dtype'] = 'torch.float16'
    elif fault == 'resource': rows[0]['resource_evidence_errors'].append('missing lifecycle')
    with pytest.raises(ValueError):
        caller.all_rank_census(rows, node_rows, caller.expected_census())


@pytest.mark.parametrize('scope', ['within_node', 'across_nodes'])
def test_actual_capacity03_allocation_uuid_aliases_are_one_physical_device(scope):
    fixture, rows, node_rows = capacity03_census()
    for row in rows:
        row['gpu_uuid'] = 'GPU-'+row['gpu_uuid']
    target = node_rows[0] if scope == 'within_node' else node_rows[1]
    original = target['gpu_uuids'][1]
    alias = node_rows[0]['gpu_uuids'][0].removeprefix('GPU-')
    target['gpu_uuids'][1] = alias
    for row in rows:
        if row['node_rank'] == target['node_rank'] and row['gpu_uuid'] == original:
            row['gpu_uuid'] = alias
    with pytest.raises(ValueError, match='unique allocated GPU'):
        caller.all_rank_census(rows, node_rows, caller.expected_census())


def test_finite_local_deadline_is_not_a_remote_marker_clock(config, tmp_path):
    node_rows = nodes(config, tmp_path); node = node_rows[0]
    node['workload_deadline'] = 5
    calls = []
    llm = SimpleNamespace(collective_rpc=lambda *a, **k: pytest.fail('RPC after workload deadline'))
    with pytest.raises(TimeoutError):
        caller.execute_leader(config, node, {}, llm_factory=lambda **k: llm,
            verify=lambda *a: None, clock=lambda: 5, publish=lambda *a: calls.append(a))
    assert calls[0][2] == 'FAIL'


def test_runtime_source_failure_preserves_original_exception(config, tmp_path):
    error = RuntimeError('native source identity failure')
    def package(name): raise error
    with pytest.raises(RuntimeError) as caught: caller.installed_sources(config, {}, distribution=package)
    assert caught.value is error


def test_worker_node_handoff_has_callback_and_keeps_nccl_defaults(config, tmp_path, monkeypatch):
    node = nodes(config, tmp_path)[1]
    config['worker_local_root'] = str(tmp_path/'worker')
    config['marker_root'] = str(tmp_path/'markers')
    config['archive_root'] = str(tmp_path/'fake_archive')
    env = dict(NODE_RANK='1', NODE_COUNT='2', PROC_PER_NODE='8', MASTER_ADDR='10.1.2.3',
               MASTER_PORT='6000', NCCL_IB_DISABLE='0', NCCL_SOCKET_IFNAME='original')
    monkeypatch.setattr(caller, 'node_record', lambda *a, **k: node)
    monkeypatch.setattr(caller.os.path, 'ismount', lambda *a: True)
    monkeypatch.setattr(caller, 'gpu_inventory', lambda *a: node['gpu_uuids'])
    events = []
    def popen(argv, **kwargs):
        assert argv[3] == 'headless' and kwargs['env']['NCCL_IB_DISABLE'] == '0'
        assert kwargs['env']['NCCL_SOCKET_IFNAME'] == 'original'
        assert kwargs['env']['STEP5_CAPACITY_NODE_FILE'].endswith('/node.json')
        assert kwargs['env']['XDG_CACHE_HOME'].endswith('/cache')
        events.append(('popen', argv))
        output = Path(node['observer_output'])
        for local in range(8):
            row = worker_row(node, local)
            row['parent_pid'] = 444
            (output/f"worker_{row['rank']}_{row['pid']}.json").write_text(json.dumps(row))
        return SimpleNamespace(pid=444, poll=lambda: -signal.SIGTERM, send_signal=lambda number: None)
    def supervise(root, generation, passed_env, engine, argv, workers, uuids, **kwargs):
        assert workers == []
        child = kwargs['popen'](argv)
        assert child.poll() == -signal.SIGTERM
        snapshot = kwargs['worker_inventory']()
        assert snapshot['complete'] and len(workers) == 8
        events.append(('inventory', snapshot))
        status = dict(result='PASS', return_code=0, reasons=[], raw_exit_code=-signal.SIGTERM,
                      worker_inventory=snapshot)
        assert kwargs['deadline'] == node['workload_deadline']+90
        status['evidence_finalization'] = kwargs['finalize_evidence'](status)
        return status
    result = caller.run_node(config, {}, tmp_path/'fake_config.json', env=env, popen=popen,
                             supervise=supervise, run=lambda *a, **k: None)
    assert result['return_code'] == 0 and len(events) == 2
    assert (tmp_path/'fake_archive/node_1/workers/worker_8_1008.json').exists()
    assert not (tmp_path/'fake_archive/node_1/cache').exists()


@pytest.mark.parametrize('node_rank,parent', [(0, 444), (0, 100), (1, 999)])
def test_worker_parent_is_not_interchanged_with_supervisor_or_direct_child(config, tmp_path, node_rank, parent):
    node = nodes(config, tmp_path)[node_rank]; output = tmp_path/'workers'; output.mkdir()
    for local in range(8):
        row = worker_row(node, local); row['parent_pid'] = parent
        (output/f"worker_{row['rank']}_{row['pid']}.json").write_text(json.dumps(row))
    inv = caller.WorkerInventory(output, node); inv.child_pid = 444
    assert not inv.snapshot()['complete'] and inv.errors


@pytest.mark.parametrize('fault', ['observer_error', 'missing_rank', 'late', 'raw_write'])
def test_review_raw_rpc_payload_saved_before_census_validation(config, tmp_path, monkeypatch, fault):
    node_rows = nodes(config, tmp_path); node = node_rows[0]
    root = tmp_path/'rpc_markers'; root.mkdir(); config['marker_root'] = str(root)
    output = Path(node['local_output']); output.mkdir()
    for item in node_rows:
        (root/f"node_{item['node_rank']}_inventory.json").write_text(json.dumps(item))
    payload = snapshots(node_rows)
    if fault == 'observer_error':
        payload[0] = dict(rank=0, pid=1000, status='UNVERIFIED', observer_error='FAKE original observer payload')
    if fault == 'missing_rank': payload.pop()
    events, terminals = [], []
    original_save, original_census = caller.save, caller.all_rank_census
    def save(path, value):
        events.append(('save', path.name))
        if fault == 'raw_write' and path.name == 'all_rank_snapshots.json':
            raise OSError('FAKE raw RPC storage unavailable')
        return original_save(path, value)
    def census(*args):
        events.append(('validate',))
        assert (output/'all_rank_snapshots.json').exists(), 'Raw RPC receipt must precede census validation'
        return original_census(*args)
    monkeypatch.setattr(caller, 'save', save)
    monkeypatch.setattr(caller, 'all_rank_census', census)
    now = [1]
    def rpc(*args, **kwargs):
        if fault == 'late': now[0] = node['workload_deadline']
        return payload
    llm = SimpleNamespace(collective_rpc=rpc)
    assert caller.execute_leader(config, node, {}, llm_factory=lambda **k: llm, verify=lambda *a: None,
        clock=lambda: now[0], publish=lambda *args: terminals.append(args)) is llm
    raw = output/'all_rank_snapshots.json'
    if fault != 'raw_write':
        assert raw.exists(), 'Failed census must retain the entire returned RPC payload'
        assert json.loads(raw.read_text())['workers'] == payload
    if fault == 'raw_write':
        assert events == [('save', 'all_rank_snapshots.json'), ('save', 'census_unverified.json')]
    assert terminals[0][2] == 'FAIL', 'Unavailable raw receipt cannot admit PASS'
    assert (output/'census_unverified.json').exists()


class ReviewClock:
    def __init__(self, now=0): self.now = now
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


def review_node_attempt(config, tmp_path, monkeypatch, *, raw_code=0, timeout=False, setup_seconds=0,
                        peer_completion=True):
    clock = ReviewClock(100)
    rank = 1 if raw_code == -signal.SIGTERM else 0
    node = nodes(config, tmp_path)[rank]
    config['worker_local_root'] = str(tmp_path/'review_local')
    config['marker_root'] = str(tmp_path/'review_markers')
    config['archive_root'] = str(tmp_path/'review_archive')
    def identify(*args, **kwargs):
        clock.now += setup_seconds
        return node
    monkeypatch.setattr(caller, 'node_record', identify)
    monkeypatch.setattr(caller.os.path, 'ismount', lambda *a: True)
    monkeypatch.setattr(caller, 'gpu_inventory', lambda *a: node['gpu_uuids'])
    monkeypatch.setattr(caller.time, 'monotonic', clock)
    monkeypatch.setattr(caller.supervision, 'node_identity', lambda *a, **k: node)
    peer = dict(node_rank=1-rank, pid=101-rank, boot_id=f'boot-{1-rank}', result='PASS')
    def preparation(root, generation, passed_node, *args, **kwargs):
        if not timeout:
            leader = dict(peer if rank == 1 else node, node_rank=0)
            if rank == 1:
                caller.supervision.publish_marker(root, generation, leader, 'native_started', 'PASS',
                    'fake child started', clock=clock, child_pid=42)
            caller.supervision.publish_marker(root, generation, dict(leader, pid=42 if rank else 444),
                'workload_terminal', 'PASS', 'fake capacity returned', clock=clock)
            if peer_completion:
                caller.supervision.publish_marker(root, generation, peer, 'completion', 'PASS',
                    'fake peer done', clock=clock, final_status=dict(result='PASS'))
        return dict(result='PASS', peer=peer, reason='', timeout=False)
    monkeypatch.setattr(caller.supervision, 'preparation_exchange', preparation)
    child = SimpleNamespace(pid=444, exit_at=clock(), signals=[])
    if rank == 1 or timeout: child.exit_at = float('inf')
    def signal_child(number):
        child.signals.append(number)
        if not timeout: child.exit_at = clock()+1
    child.send_signal = signal_child
    child.poll = lambda: raw_code if clock() >= child.exit_at else None
    def popen(argv, **kwargs):
        records = Path(node['observer_output'])
        kwargs['stdout'].write(b'FAKE raw stdout\xff\n'); kwargs['stdout'].flush()
        kwargs['stderr'].write(b'FAKE raw stderr\xfe\n'); kwargs['stderr'].flush()
        for local in range(8):
            row = worker_row(node, local)
            if rank == 1: row['parent_pid'] = child.pid
            (records/f"worker_{row['rank']}_{row['pid']}.json").write_text(json.dumps(row))
            if timeout:
                stat = tmp_path/'review_proc'/str(row['pid'])/'stat'; stat.parent.mkdir(parents=True)
                stat.write_text(str(row['pid'])+' (fake worker) S '+'0 '*18+str(row['start_time']))
        return child
    events = []
    original = caller.supervision.supervise
    def supervise(*args, **kwargs):
        events.append(('supervise', clock()))
        return original(*args, **kwargs, clock=clock, sleep=clock.sleep)
    def query(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    status = caller.run_node(config, {}, tmp_path/'fake_config.json', env=dict(MASTER_PORT='6000'),
        run=query, popen=popen, supervise=supervise, proc_root=tmp_path/'review_proc')
    return status, clock, node, events


@pytest.mark.parametrize('setup_seconds', [0, 50])
def test_review_timeout_archive_starts_with_explicit_reserve(config, tmp_path, monkeypatch, setup_seconds):
    observed = []
    original = caller.archive_records
    def archive(*args, **kwargs):
        observed.append(caller.time.monotonic())
        return original(*args, **kwargs)
    monkeypatch.setattr(caller, 'archive_records', archive)
    status, clock, node, events = review_node_attempt(config, tmp_path, monkeypatch,
        timeout=True, setup_seconds=setup_seconds)
    assert observed and observed[0] <= 2200, 'Archive must begin with 300s left in entry+2400 ceiling'
    assert node['workload_deadline'] == 2110, 'Setup must not restart the approved clock'
    assert status['timeouts']['workload'] and status['timeouts']['shutdown'] and status['timeouts']['residual']
    assert clock() <= 2500 and status['return_code'] == 1
    assert (tmp_path/'review_archive/node_0/native.stdout').read_bytes() == b'FAKE raw stdout\xff\n'


@pytest.mark.parametrize('raw_code', [0, 9, -signal.SIGTERM])
def test_review_archive_failure_precedes_durable_completion_and_status(config, tmp_path, monkeypatch, raw_code):
    def archive(*args, **kwargs): raise OSError('FAKE archive copy failure')
    monkeypatch.setattr(caller, 'archive_records', archive)
    status, clock, node, events = review_node_attempt(config, tmp_path, monkeypatch, raw_code=raw_code)
    root = Path(config['marker_root']); rank = node['node_rank']
    completion = json.loads((root/f'node_{rank}_completion.json').read_text())
    durable = json.loads((root/f'node_{rank}/status.json').read_text())
    assert completion['result'] == 'FAIL', 'Archive failure must be decided before completion publication'
    assert durable['result'] == 'FAIL' and durable['evidence_verdict'] == 'UNVERIFIED'
    assert status['return_code'] == 1 and status['raw_exit_code'] == raw_code
    assert durable['raw_exit_code'] == raw_code and completion['final_status']['raw_exit_code'] == raw_code
    assert status['raw_signal'] == (signal.SIGTERM if raw_code < 0 else None)


@pytest.mark.parametrize('raw_code', [0, -signal.SIGTERM])
@pytest.mark.parametrize('peer_already_complete', [True, False])
def test_review_successful_archive_reserve_keeps_peer_completion_inside_total_budget(
        config, tmp_path, monkeypatch, raw_code, peer_already_complete):
    original_archive = caller.archive_records
    original_read = caller.supervision.read_marker
    def archive(*args, **kwargs):
        # A legal late success uses archive time after the native/cleanup deadline.
        caller.time.monotonic.now += 250 if peer_already_complete else 180
        return original_archive(*args, **kwargs)
    def read(root, generation, rank, kind, *args, **kwargs):
        if kind == 'completion' and not peer_already_complete and caller.time.monotonic() >= 2255:
            path = Path(root)/f'node_{rank}_completion.json'
            if not path.exists():
                peer = dict(node_rank=rank, pid=100+rank, boot_id=f'boot-{rank}')
                caller.supervision.publish_marker(root, generation, peer, 'completion', 'PASS',
                    'fake peer archive completed', clock=caller.time.monotonic, final_status=dict(result='PASS'))
        return original_read(root, generation, rank, kind, *args, **kwargs)
    monkeypatch.setattr(caller, 'archive_records', archive)
    monkeypatch.setattr(caller.supervision, 'read_marker', read)
    status, clock, node, events = review_node_attempt(config, tmp_path, monkeypatch, raw_code=raw_code,
        setup_seconds=1900, peer_completion=peer_already_complete)
    assert status['result'] == 'PASS', status['reasons']
    assert status['return_code'] == 0 and status['raw_exit_code'] == raw_code
    assert status['evidence_verdict'] == 'PASS' and not any(status['timeouts'].values())
    assert status['peer_final_status']['result'] == 'PASS'
    assert 2200 < clock() < 2500, 'Exchange must use only the original entry+2400 window'
    durable = json.loads((Path(config['marker_root'])/f"node_{node['node_rank']}"/'status.json').read_text())
    assert durable['result'] == 'PASS' and durable['raw_exit_code'] == raw_code


@pytest.mark.parametrize('rank,parent', [(0, 444), (0, 100), (1, 999)])
def test_review_bad_parent_binding_keeps_known_real_workers_for_residual(config, tmp_path, monkeypatch, rank, parent):
    node = nodes(config, tmp_path)[rank]; output = tmp_path/'review_workers'; output.mkdir()
    for local in range(8):
        row = worker_row(node, local); row['parent_pid'] = parent
        (output/f"worker_{row['rank']}_{row['pid']}.json").write_text(json.dumps(row))
    inventory = caller.WorkerInventory(output, node); inventory.child_pid = 444
    snapshot = inventory.snapshot()
    assert snapshot['complete'] is False and snapshot['errors']
    assert snapshot['workers'] == [(1000+rank*8+i, 333+i) for i in range(8)], \
        'Parent admission fault must not discard parsed real PID/start_time pairs'
    query = lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr='')
    residual = caller.supervision.verify_residual(snapshot['workers'], node['gpu_uuids'], 30,
        proc_root=tmp_path/'fake_proc', clock=lambda: 0, sleep=lambda *a: None, run=query)
    assert [row['pid'] for row in residual['workers']] == [pair[0] for pair in snapshot['workers']]


@pytest.mark.parametrize('value', [0, -1, True, float('inf'), float('nan'), '300', 2310, 2400])
def test_review_archive_reserve_is_explicit_positive_and_leaves_rpc_budget(config, value):
    config['archive_reserve_seconds'] = value
    with pytest.raises(ValueError): caller.preflight(config)


def test_review_archive_reserve_missing_or_rpc_exhausts_remaining_budget(config):
    config['rpc_seconds'] = 2010
    with pytest.raises(ValueError, match='workload ceiling'): caller.preflight(config)
    del config['archive_reserve_seconds']
    with pytest.raises(ValueError, match='explicitly configured'): caller.preflight(config)


@pytest.mark.parametrize('fault', ['parent_missing', 'parent_self', 'parent_start', 'duplicate_rank', 'duplicate_pid', 'drift'])
def test_review_identity_admission_faults_keep_only_actual_worker_pairs(config, tmp_path, fault):
    node = nodes(config, tmp_path)[0]; output = tmp_path/'known_pairs'; output.mkdir()
    row = worker_row(node, 0); first_path = output/'worker_0_1000.json'
    first_path.write_text(json.dumps(row))
    inventory = caller.WorkerInventory(output, node); inventory.child_pid = 444
    assert inventory.snapshot()['workers'] == [(1000, 333)]
    changed = dict(row)
    if fault == 'parent_missing': del changed['parent_pid']
    elif fault == 'parent_self': changed['parent_pid'] = changed['pid']
    elif fault == 'parent_start': changed['parent_start_time'] = None
    elif fault == 'duplicate_rank': changed['pid'] = 2000
    elif fault == 'duplicate_pid': changed.update(rank=1, local_rank=1)
    elif fault == 'drift': changed['start_time'] = 500
    (output/f"worker_{changed['rank']}_{changed['pid']}.json").write_text(json.dumps(changed))
    snapshot = inventory.snapshot()
    assert snapshot['complete'] is False and snapshot['errors']
    assert (1000, 333) in snapshot['workers']
    assert (changed['pid'], changed['start_time']) in snapshot['workers']
    assert all(pid not in (node['pid'], inventory.child_pid) for pid, start in snapshot['workers'])


@pytest.mark.parametrize('escape', ['file', 'directory', 'destination'])
def test_review_archive_rejects_reachable_symlink_escape(tmp_path, escape):
    local, outside, destination = tmp_path/'local', tmp_path/'outside', tmp_path/'archive'
    local.mkdir(); outside.mkdir(); (outside/'secret.raw').write_bytes(b'FAKE outside not archive input')
    if escape == 'destination':
        destination.symlink_to(outside, target_is_directory=True)
    else:
        (local/'escape').symlink_to(outside/'secret.raw' if escape == 'file' else outside,
                                  target_is_directory=escape == 'directory')
    with pytest.raises(ValueError, match='canonical|symlink|escaped'):
        caller.archive_records(local, destination, 4096, deadline=10, clock=lambda: 0)
    assert sorted(path.name for path in outside.iterdir()) == ['secret.raw']


def test_review_archive_manifest_lists_exact_bytes_and_excludes_future_evidence(tmp_path, monkeypatch):
    local, destination = tmp_path/'local', tmp_path/'archive'; local.mkdir()
    raw = local/'native.stdout'; raw.write_bytes(b'FAKE raw\xff')
    (local/'cache').mkdir(); (local/'cache/never_archive').write_bytes(b'FAKE cache')
    real_open = Path.open; appended = []
    def open_path(path, mode='r', *args, **kwargs):
        if path == raw and mode == 'rb' and not appended:
            appended.append(True)
            with real_open(raw, 'ab') as stream: stream.write(b'FAKE future native tail')
        return real_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_path)
    receipt = caller.archive_records(local, destination, 4096, deadline=10, clock=lambda: 0)
    manifest = json.loads((destination/'archive_manifest.json').read_text())
    assert receipt['retained_files'] == manifest['retained_files'] == [dict(path='native.stdout', bytes=9)]
    assert receipt['retained_bytes'] == manifest['retained_bytes'] == 9
    assert (destination/'native.stdout').read_bytes() == b'FAKE raw\xff'
    assert 'verdict' not in manifest and 'result' not in manifest
    assert 'not_final_status' in manifest['scope'] and len(manifest['omitted']) == 3
    assert not (destination/'cache').exists()
    assert receipt['manifest_bytes'] == (destination/'archive_manifest.json').stat().st_size


@pytest.mark.parametrize('fault', ['byte_cap', 'deadline_before', 'deadline_during_copy'])
def test_review_archive_copy_limits_keep_original_raw_bytes(tmp_path, monkeypatch, fault):
    local, destination = tmp_path/'local', tmp_path/'archive'; local.mkdir()
    raw = local/'native.stderr'; content = b'FAKE raw stderr\xff\xfe'; raw.write_bytes(content)
    clock = ReviewClock(10 if fault == 'deadline_before' else 0)
    if fault == 'deadline_during_copy':
        real_open = Path.open
        class Source:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def read(self, count=-1): clock.now = 10; return self.stream.read(count)
        def open_path(path, mode='r', *args, **kwargs):
            stream = real_open(path, mode, *args, **kwargs)
            return Source(stream) if path == raw and mode == 'rb' else stream
        monkeypatch.setattr(Path, 'open', open_path)
    with pytest.raises(ValueError if fault == 'byte_cap' else TimeoutError):
        caller.archive_records(local, destination, 1 if fault == 'byte_cap' else 4096, deadline=10, clock=clock)
    assert raw.read_bytes() == content
    assert not (destination/'archive_manifest.json').exists()


def test_review_main_preflight_time_is_inside_absolute_node_deadline(config, tmp_path, monkeypatch):
    path = tmp_path/'fake_main_config.json'; config['native_sources'] = {}; path.write_text(json.dumps(config))
    clock = ReviewClock(100); monkeypatch.setattr(caller.time, 'monotonic', clock)
    monkeypatch.setattr(caller.sys, 'argv', ['fake', 'node', '--config', str(path)])
    def preflight(*args, **kwargs): clock.now += 80; return dict(sources={})
    monkeypatch.setattr(caller, 'preflight', preflight)
    observed = []
    def run_node(*args, **kwargs): observed.append(kwargs['deadline']); return dict(return_code=0)
    monkeypatch.setattr(caller, 'run_node', run_node)
    assert caller.main() == 0 and observed == [2500]


def test_review_setup_exhaustion_never_starts_native(config, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='Node setup consumed'):
        review_node_attempt(config, tmp_path, monkeypatch, setup_seconds=2000)


@pytest.mark.parametrize('fault', ['pre_status_write', 'archive_mount'])
def test_review_finalization_fault_keeps_raw_copy_and_durable_unverified_result(config, tmp_path, monkeypatch, fault):
    original = caller.save
    if fault == 'pre_status_write':
        def save(path, value):
            if path.name == 'pre_completion_status.json': raise OSError('FAKE pre-status unavailable')
            return original(path, value)
        monkeypatch.setattr(caller, 'save', save)
    else:
        def save(path, value):
            result = original(path, value)
            if path.name == 'pre_completion_status.json':
                monkeypatch.setattr(caller.os.path, 'ismount', lambda *a: False)
            return result
        monkeypatch.setattr(caller, 'save', save)
    status, clock, node, events = review_node_attempt(config, tmp_path, monkeypatch, raw_code=9)
    assert status['result'] == 'FAIL' and status['evidence_verdict'] == 'UNVERIFIED' and status['raw_exit_code'] == 9
    durable = json.loads((Path(config['marker_root'])/'node_0/status.json').read_text())
    assert durable['raw_exit_code'] == 9 and durable['evidence_verdict'] == 'UNVERIFIED'
    if fault == 'pre_status_write':
        assert (tmp_path/'review_archive/node_0/native.stderr').read_bytes() == b'FAKE raw stderr\xfe\n'
    else:
        assert (tmp_path/'review_local/node_0/native.stderr').read_bytes() == b'FAKE raw stderr\xfe\n'


def test_review_completion_happens_after_copy_and_manifest_never_claims_future_pass(config, tmp_path, monkeypatch):
    observed = []
    original_archive = caller.archive_records
    def archive(*args, **kwargs):
        assert not (Path(config['marker_root'])/'node_0_completion.json').exists()
        observed.append(caller.time.monotonic())
        return original_archive(*args, **kwargs)
    monkeypatch.setattr(caller, 'archive_records', archive)
    status, clock, node, events = review_node_attempt(config, tmp_path, monkeypatch)
    assert status['result'] == 'PASS' and observed
    retained = tmp_path/'review_archive/node_0'
    pre_status = json.loads((retained/'pre_completion_status.json').read_text())
    assert pre_status['result'] == 'UNVERIFIED' and pre_status['final_result_retained'] is False
    assert not (retained/'status.json').exists() and not (retained/'completion.json').exists()
    manifest = json.loads((retained/'archive_manifest.json').read_text())
    assert manifest['retained_bytes'] == sum((retained/item['path']).stat().st_size for item in manifest['retained_files'])
    assert 'result' not in manifest and 'verdict' not in manifest
