"""Fault-injection checks for transparent native capacity observations."""
import importlib.util
from pathlib import Path

import pytest

from test_role_model_input_observation import native

PATH = Path(__file__).with_name('role_capacity_observation.py')
spec = importlib.util.spec_from_file_location('capacity_observation', PATH)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


@pytest.mark.parametrize('stage', module.STAGES)
def test_native_return_and_arguments_are_preserved(stage):
    calls, records, result = [], [], object()
    def native(self, *args, **kwargs):
        calls.append((args, kwargs)); return result
    base = type('Native', (), {name: native for name in module.STAGES})
    worker = module.observed_worker(base, lambda *row: records.append(row))()
    assert getattr(worker, stage)(1, enabled=False) is result
    assert calls == [((1,), {'enabled': False})]
    assert [row[2] for row in records] == ['before', 'after']


@pytest.mark.parametrize('stage', module.STAGES)
def test_observer_failure_cannot_mask_native_exception(stage):
    failure, calls = ValueError('original native failure'), []
    def native(self):
        calls.append(stage); raise failure
    def broken(*args):
        raise OSError('observer recording failure')
    base = type('Native', (), {name: native for name in module.STAGES})
    with pytest.raises(ValueError) as caught:
        getattr(module.observed_worker(base, broken)(), stage)()
    assert caught.value is failure and calls == [stage]


def test_closed_stdout_cannot_change_native_outcome(monkeypatch):
    def closed(*args, **kwargs): raise BrokenPipeError('closed stdout')
    monkeypatch.setattr('builtins.print', closed)
    def broken(*args): raise OSError('recording failed')
    result, failure = object(), ValueError('native failure')
    base = type('Native', (), {name: lambda *a: result for name in module.STAGES})
    assert module.observed_worker(base, broken)().load_model() is result
    def fail(*args): raise failure
    base.load_model = fail
    with pytest.raises(ValueError) as caught: module.observed_worker(base, broken)().load_model()
    assert caught.value is failure


def test_module_import_does_not_load_torch_or_vllm(monkeypatch):
    import builtins
    original = builtins.__import__
    def import_module(name, *args, **kwargs):
        assert name.split('.')[0] not in ('torch', 'vllm'), 'Inert import attempted native packages'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', import_module)
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)


def fake_proc(tmp_path, monkeypatch, pid=321, start=765):
    root = tmp_path/'proc'
    (root/'self/ns').mkdir(parents=True)
    (root/'self/stat').write_text(f'{pid} (worker ) with spaces) S '+'0 '*18+str(start))
    (root/'self/ns/pid').symlink_to('pid:[123]')
    (root/'sys/kernel/random').mkdir(parents=True)
    (root/'sys/kernel/random/boot_id').write_text('boot-0')
    monkeypatch.setattr(module.os, 'getpid', lambda: pid)
    monkeypatch.setattr(module.os, 'getppid', lambda: 222)
    (root/'222').mkdir()
    (root/'222/stat').write_text('222 (native EngineCore) S '+'0 '*18+'99')
    return root


def test_live_start_time_capture_precedes_native_init(tmp_path, monkeypatch):
    root = fake_proc(tmp_path, monkeypatch)
    events = []
    identity = module.worker_identity
    def capture(worker):
        events.append('identity')
        worker._capacity_identity = identity(worker, root)
        worker._capacity_completed_stages = []
    monkeypatch.setattr(module, 'identity_record', capture)
    def native(self):
        events.append('native')
        assert self._capacity_identity['pid'] == 321
        assert self._capacity_identity['start_time'] == 765
        assert self._capacity_identity['pid_namespace'] == 'pid:[123]'
        return 'native returned'
    base = type('Native', (), {name: native for name in module.STAGES})
    worker = module.observed_worker(base, lambda *a: events.append(a[2]))()
    worker.rank, worker.local_rank = 0, 0
    assert worker.init_device() == 'native returned'
    assert events == ['identity', 'before', 'native', 'after']
    assert worker._capacity_completed_stages == ['init_device']


def test_private_identity_published_to_local_observer_file(tmp_path, monkeypatch):
    import json
    root = fake_proc(tmp_path, monkeypatch)
    output = tmp_path/'records'; output.mkdir()
    node = dict(generation='g2', node_rank=1, boot_id='boot-0', pid_namespace='pid:[123]',
        observer_sha256='source', gpu_uuids=[f'GPU-{i}' for i in range(8)], observer_output=str(output))
    path = tmp_path/'node.json'; path.write_text(json.dumps(node))
    monkeypatch.setenv('STEP5_CAPACITY_NODE_FILE', str(path))
    original = module.worker_identity
    monkeypatch.setattr(module, 'worker_identity', lambda worker: original(worker, root))
    worker = type('Worker', (), dict(rank=10, local_rank=2))()
    module.identity_record(worker)
    row = json.loads((output/'worker_10_321.json').read_text())
    assert row['start_time'] == 765 and row['parent_pid'] == 222
    assert row['parent_start_time'] == 99
    assert row['node_rank'] == 1 and row['generation'] == 'g2'
    with pytest.raises(FileExistsError): module.identity_record(worker)
    assert not list(output.glob('.*.json'))


@pytest.mark.parametrize('fault', ['stat', 'node', 'record'])
def test_live_identity_fault_is_diagnostic_not_native_failure(tmp_path, monkeypatch, fault):
    failure = ValueError('original init failure')
    events = []
    def bad(worker):
        raise OSError(f'FAKE {fault} identity unavailable')
    monkeypatch.setattr(module, 'identity_record', bad)
    def native(self):
        events.append('native'); raise failure
    base = type('Native', (), {name: native for name in module.STAGES})
    with pytest.raises(ValueError) as caught:
        module.observed_worker(base, lambda *a: None)().init_device()
    assert caught.value is failure and events == ['native']


def test_worker_snapshot_uses_supported_method_and_metadata_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    node = dict(pid=321, start_time=7, rank=0, local_rank=0)
    model_type = type('Model', (), {})
    model = model_type()
    model.named_modules = lambda: [('', model)]
    worker = SimpleNamespace(_capacity_identity=node, _capacity_completed_stages=list(module.STAGES),
        _capacity_evidence_errors=[], rank=0, device='cuda:0',
        model_runner=SimpleNamespace(model=model, model_memory_usage=123),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=16,
            data_parallel_size=1, pipeline_parallel_size=1, enable_expert_parallel=False, nnodes=2, node_rank=0),
            model_config=SimpleNamespace(dtype='torch.bfloat16'),
            cache_config=SimpleNamespace(gpu_memory_utilization=0.92, enable_prefix_caching=False),
            speculative_config=None))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda device: SimpleNamespace(uuid='GPU-0')))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    monkeypatch.setattr(module, 'worker_identity', lambda value: dict(node))
    row = module.capacity_snapshot(worker)
    assert row['status'] == 'metadata_only_not_G2' and row['gpu_uuid'] == 'GPU-0'
    assert row['model_memory_usage'] == 123 and len(row['module_classes']) == 1
    worker._capacity_identity['start_time'] = 8
    monkeypatch.setattr(module, 'worker_identity', lambda value: dict(node, start_time=7))
    assert module.capacity_snapshot(worker)['status'] == 'UNVERIFIED'


def allocator_case(*, backend='native'):
    from types import SimpleNamespace
    calls = []
    stats = {f'{kind}.all.{metric}': value for kind, value in
        [('allocated_bytes', 2048), ('reserved_bytes', 4096), ('active_bytes', 3072),
         ('requested_bytes', 2000), ('inactive_split_bytes', 512)] for metric in ('current', 'peak')}
    def stats_read(device):
        calls.append(('stats', device)); return dict(stats)
    def global_read(device):
        calls.append(('global', device)); return 8192, 16384
    class Tensor:
        def __init__(self, device, pointer, size):
            self.device, self.pointer, self.size = device, pointer, size
        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: self.pointer, nbytes=lambda: self.size)
    cuda = SimpleNamespace(is_initialized=lambda: True, is_current_stream_capturing=lambda: False,
        memory=SimpleNamespace(get_allocator_backend=lambda: backend), memory_stats=stats_read,
        mem_get_info=global_read)
    model = SimpleNamespace(_parameters={'weight': Tensor('cuda:0', 7, 1024),
        'alias': Tensor('cuda:0', 7, 1024), 'host': Tensor('cpu', 7, 8192)},
        _buffers={'other': Tensor('cuda:1', 7, 32)}, _non_persistent_buffers_set=set())
    model.named_modules = lambda: [('', model)]
    runner = SimpleNamespace(device='cuda:0', model=model)
    torch = SimpleNamespace(cuda=cuda, Tensor=Tensor)
    return SimpleNamespace(runner=runner, torch=torch, calls=calls, stats=stats, Tensor=Tensor)


def test_allocator_totals_deduplicate_aliases_and_exclude_other_devices():
    case = allocator_case()
    row = module.allocator_observation(case.runner, case.torch,
        {'cuda:0:7': 1024, 'cuda:0:8': 512, 'cpu:7': 8192}, 16)
    assert row['status'] == 'OBSERVED_NATIVE_TOTALS_AND_STORAGE_REFERENCES_NOT_ALLOCATION_CENSUS'
    assert row['known_cuda_storage_bytes'] == 1536 and row['known_cuda_storages'] == 2
    assert row['other_device_storage_bytes'] == 8224
    assert row['allocated_minus_known_cuda_storage_bytes'] == 512
    assert row['reserved_minus_allocated_bytes'] == 2048
    assert row['global_used_bytes'] == 8192
    assert case.calls == [('stats', 'cuda:0'), ('global', 'cuda:0')]
    assert row['allocation_membership'] == row['allocation_history'] == 'UNMEASURED'
    assert row['peaks'] == 'SINCE_LAST_NATIVE_RESET_NOT_THIS_SCOPE'
    assert row['observation_finished_ns'] >= row['observation_started_ns']


@pytest.mark.parametrize('fault', ['uninitialized', 'capture', 'device'])
def test_allocator_ineligible_boundaries_do_not_initialize_or_query_native_state(fault):
    case = allocator_case()
    if fault == 'uninitialized': case.torch.cuda.is_initialized = lambda: False
    if fault == 'capture': case.torch.cuda.is_current_stream_capturing = lambda: True
    if fault == 'device': case.runner.device = 'cuda'
    row = module.allocator_observation(case.runner, case.torch, {}, 16)
    assert row['status'] == 'UNVERIFIED' and row['errors'] and not case.calls
    assert 'known_cuda_storage_bytes' not in row


@pytest.mark.parametrize('fault', ['missing', 'negative', 'boolean', 'global', 'alias',
    'cap', 'model', 'residual', 'peak', 'reserved', 'unsupported', 'api', 'interrupt'])
def test_allocator_faults_preserve_unverified_raw_evidence(fault):
    case = allocator_case(backend='cudaMallocAsync' if fault == 'unsupported' else 'native')
    limit = 16
    known = {}
    if fault == 'missing': del case.stats['active_bytes.all.peak']
    if fault == 'negative': case.stats['requested_bytes.all.current'] = -1
    if fault == 'boolean': case.stats['active_bytes.all.current'] = True
    if fault == 'global': case.torch.cuda.mem_get_info = lambda device: (11, 10)
    if fault == 'alias': known['cuda:0:7'] = 12
    if fault == 'cap': limit = 1
    if fault == 'model': del case.runner.model
    if fault == 'residual': known['cuda:0:8'] = 4096
    if fault == 'peak': case.stats['reserved_bytes.all.peak'] = 1
    if fault == 'reserved': case.stats['allocated_bytes.all.current'] = 8192
    if fault in ('api', 'interrupt'):
        def fail(device):
            raise KeyboardInterrupt('native API interrupted') if fault == 'interrupt' else RuntimeError('native API failed')
        case.torch.cuda.memory_stats = fail
    row = module.allocator_observation(case.runner, case.torch, known, limit)
    assert row['status'] == 'UNVERIFIED' and row['errors']
    assert row['allocation_membership'] == row['allocation_history'] == 'UNMEASURED'
    if fault == 'unsupported':
        assert row['backend'] == 'cudaMallocAsync'
        assert 'allocated_minus_known_cuda_storage_bytes' not in row
    if fault == 'residual':
        assert row['allocated_minus_known_cuda_storage_bytes'] == -3072
    if fault == 'boolean':
        assert row['statistics']['active_bytes.all.current'] is True


def test_allocator_metadata_fault_does_not_mask_an_original_native_exception(monkeypatch):
    case = allocator_case()
    def fail(device): raise OSError('allocator read unavailable')
    case.torch.cuda.memory_stats = fail
    original = ValueError('original native failure')
    records = []
    def observe(worker, phase, boundary, error=None):
        records.append((boundary, module.allocator_observation(case.runner, case.torch, {}, 16)))
    def native(self): raise original
    base = type('Native', (), {name: native for name in module.STAGES})
    with pytest.raises(ValueError) as caught:
        module.observed_worker(base, observe)().load_model()
    assert caught.value is original
    assert [row[0] for row in records] == ['before', 'error']
    assert all(row[1]['status'] == 'UNVERIFIED' for row in records)


@pytest.mark.parametrize('boundary', ['start', 'finish'])
def test_allocator_clock_fault_cannot_escape_into_native_execution(monkeypatch, boundary):
    case = allocator_case()
    count = 0
    def clock():
        nonlocal count
        count += 1
        if count == (1 if boundary == 'start' else 2):
            raise RuntimeError('monotonic clock unavailable')
        return count
    monkeypatch.setattr(module.time, 'monotonic_ns', clock)
    row = module.allocator_observation(case.runner, case.torch, {}, 16)
    assert row['status'] == 'UNVERIFIED' and row['errors']


def test_known_storage_identity_rejects_conflicting_alias_sizes():
    from types import SimpleNamespace
    storages = {}
    value = SimpleNamespace(device='cuda:0', shape=(1,), dtype='torch.float32',
        stride=lambda: (1,), storage_offset=lambda: 0,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 71, nbytes=lambda: 16))
    module.storage_record('first', 'reference', value, storages)
    value.untyped_storage = lambda: SimpleNamespace(data_ptr=lambda: 71, nbytes=lambda: 32)
    with pytest.raises(ValueError, match='inconsistent sizes'):
        module.storage_record('second', 'reference', value, storages)
    assert storages == {'cuda:0:71': 16}


@pytest.mark.parametrize('native_raises', [False, True])
def test_allocator_occupancy_uses_existing_before_after_scope_records(native, monkeypatch, native_raises):
    from types import SimpleNamespace
    from test_role_model_input_observation import runner, record, work
    import role_capacity_observation as capacity
    native.sites.torch = SimpleNamespace(Tensor=native.module.Tensor)
    observed, graph, _ = runner(native)
    calls = []
    def observe(selected, torch, storages, limit):
        calls.append(selected)
        return dict(status='UNVERIFIED', test_boundary=len(calls), allocation_membership='UNMEASURED')
    monkeypatch.setattr(capacity, 'allocator_observation', observe)
    failure = ValueError('original execution failure')
    if native_raises:
        graph.failure = failure
        with pytest.raises(ValueError) as caught:
            observed.execute_model(work())
        assert caught.value is failure
    else:
        observed.execute_model(work())
    records = [row for row in record(observed)['events'] if row['kind'].startswith('known_state_storage')]
    assert [row['kind'] for row in records] == ['known_state_storage', 'known_state_storage_after']
    assert [row['allocator_occupancy']['test_boundary'] for row in records] == [1, 2]
    assert calls == [observed, observed]
