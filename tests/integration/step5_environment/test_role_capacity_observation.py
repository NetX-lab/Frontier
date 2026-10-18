"""Fault-injection checks for transparent native capacity observations."""
import importlib.util
from pathlib import Path

import pytest

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
