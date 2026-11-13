"""Fake-only regression checks for timed local attention evidence ordering."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).with_name('local_attention_evidence.py')
spec = importlib.util.spec_from_file_location('attention_evidence', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Value:
    dtype = 'torch.bfloat16'
    shape = (2, 2)

    def __init__(self, values, events):
        self.values, self.events = values, events

    def numel(self):
        return 4

    def element_size(self):
        return 2

    def stride(self):
        return (2, 1)

    def detach(self):
        return self

    def cpu(self):
        self.events.append('copy')
        return Value(list(self.values), self.events)


def setup(kind, error=None):
    events, active, owners = [], [], SimpleNamespace(enabled=False, calls=[])
    values = [0]

    @contextmanager
    def scope(name):
        events.append(('scope_enter', name))
        active.append(name)
        try:
            yield
        finally:
            active.pop()
            events.append(('scope_exit', name))

    def sync():
        assert not active
        events.append('sync')

    def save(value, stream):
        assert not active
        events.append('save')
        stream.write(json.dumps(value.values).encode())

    def observe():
        assert not active and not owners.enabled
        events.append('observe')
        return {'started_ns': 10, 'finished_ns': 11, 'scope': 'fake_only'}

    def forward(argument):
        assert argument == 'prepared'
        events.append('forward')
        if error is not None:
            raise error
        if owners.enabled:
            owners.calls.extend(module.attention.OWNERS[kind])
        values[0] += 1
        return Value(values, events)

    def prepare():
        assert not active and not owners.enabled
        events.append('prepare')
        return ('prepared',), 1.25

    def restore():
        assert not active and not owners.enabled
        events.append('restore')

    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=sync), save=save,
        profiler=SimpleNamespace(record_function=scope))
    return torch, owners, forward, prepare, restore, observe, events


def run(tmp_path, kind='swa', **overrides):
    torch, owners, forward, prepare, restore, observe, events = setup(kind)
    arguments = dict(repeats=2, tensor_copy_bytes=64, evidence_bytes=65536, observe=observe)
    arguments.update(overrides)
    result = module.timed_passes(torch, tmp_path, kind, owners, forward, prepare, restore, **arguments)
    return result, events, owners


@pytest.mark.parametrize('kind', ['swa', 'dsa'])
def test_each_actual_timed_output_is_retained_outside_price_and_after_drain(tmp_path, kind):
    result, events, owners = run(tmp_path, kind)
    assert not result['evidence_errors']
    assert result['status'] == 'raw_unadmitted'
    assert [row['pass'] for row in result['passes']] == ['owner_pass', 'forward_pass']*2
    assert [json.loads(Path(row['output']['path']).read_text()) for row in result['passes']] == [[1], [2], [3], [4]]
    assert events.count('forward') == 4 and events.count('observe') == 8
    assert events.count('copy') == events.count('save') == 4
    assert events.count('sync') == 8
    for index, event in enumerate(events):
        if event == 'copy':
            assert events[index-2:index] == ['sync', 'observe']
    assert owners.enabled is False
    for row in result['passes']:
        saved = json.loads((tmp_path/'timed_outputs'/f"{row['repeat']:04d}_{row['pass']}.json").read_text())
        assert saved == row and saved['output']['logical_bytes'] == 8
        assert row['forward_started_ns'] <= row['forward_finished_ns'] <= row['drained_ns']
        assert row['content_status'] == 'retained_not_numerically_verified'
    assert result['continuous_identity_exclusivity'] == result['clean_timing'] == 'UNVERIFIED'
    assert result['device_copy_activity'] == result['device_gaps'] == 'UNMEASURED'


@pytest.mark.parametrize('failure', [RuntimeError('native failure'), KeyboardInterrupt('native interrupt')])
def test_original_native_failure_propagates_with_partial_raw_record(tmp_path, failure):
    torch, owners, forward, prepare, restore, observe, events = setup('swa', failure)
    with pytest.raises(type(failure)) as caught:
        module.timed_passes(torch, tmp_path, 'swa', owners, forward, prepare, restore,
            repeats=2, tensor_copy_bytes=64, evidence_bytes=65536, observe=observe)
    assert caught.value is failure
    assert events.count('forward') == 1 and not owners.enabled
    assert not list((tmp_path/'timed_outputs').glob('*.pt'))
    saved = json.loads((tmp_path/'timed_outputs'/'0000_owner_pass_failure.json').read_text())
    assert str(failure) in saved['native_error']


@pytest.mark.parametrize('failure', [RuntimeError('native failure'), KeyboardInterrupt('native interrupt')])
def test_record_failure_cannot_replace_native_exception(tmp_path, failure):
    torch, owners, forward, prepare, restore, observe, events = setup('swa', failure)
    with pytest.raises(type(failure)) as caught:
        module.timed_passes(torch, tmp_path, 'swa', owners, forward, prepare, restore,
            repeats=2, tensor_copy_bytes=64, evidence_bytes=1, observe=observe)
    assert caught.value is failure and events.count('forward') == 1


def test_resource_failure_is_diagnostic_and_does_not_drop_timed_outputs(tmp_path):
    def observe():
        raise OSError('resource query failed')
    result, events, owners = run(tmp_path, observe=observe)
    assert len(result['evidence_errors']) == 8
    assert len(list((tmp_path/'timed_outputs').glob('*.pt'))) == 4
    assert all(row['before'] is row['after'] is None for row in result['passes'])
    assert events.count('forward') == 4 and not owners.enabled


@pytest.mark.parametrize('budget,match', [(1, 'copy budget'), (7, 'copy budget')])
def test_oversize_output_is_unverified_without_copy_or_native_rewrite(tmp_path, budget, match):
    result, events, _ = run(tmp_path, tensor_copy_bytes=budget)
    assert events.count('forward') == 4 and 'copy' not in events
    assert all(row['content_status'] == 'UNVERIFIED' for row in result['passes'])
    assert all(match in row['error'] for row in result['evidence_errors'])


def test_serialized_evidence_budget_includes_existing_sample_files(tmp_path):
    (tmp_path/'existing.trace').write_bytes(b'x'*65536)
    result, events, _ = run(tmp_path, evidence_bytes=65536)
    assert events.count('forward') == 4
    assert not list((tmp_path/'timed_outputs').glob('*.pt'))
    assert all(row['output'] is None for row in result['passes'])
    assert len(result['evidence_errors']) == 8  # output and pass-record writes
    assert (tmp_path/'existing.trace').stat().st_size == 65536


@pytest.mark.parametrize('field,value', [('repeats', 1), ('tensor_copy_bytes', 0), ('evidence_bytes', 0)])
def test_invalid_budget_or_repeat_rejected_before_forward(tmp_path, field, value):
    with pytest.raises(ValueError):
        run(tmp_path, **{field: value})
    assert not (tmp_path/'timed_outputs').exists()


def test_exclusive_new_output_does_not_overwrite_old_evidence(tmp_path):
    run(tmp_path)
    files = {p.name: p.read_bytes() for p in (tmp_path/'timed_outputs').iterdir()}
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert files == {p.name: p.read_bytes() for p in (tmp_path/'timed_outputs').iterdir()}


def test_wrong_owner_sequence_is_not_accepted(tmp_path):
    torch, owners, forward, prepare, restore, observe, events = setup('swa')
    def wrong(*args):
        value = forward(*args)
        owners.calls.pop()
        return value
    with pytest.raises(ValueError, match='owner order'):
        module.timed_passes(torch, tmp_path, 'swa', owners, wrong, prepare, restore,
            repeats=2, tensor_copy_bytes=64, evidence_bytes=65536, observe=observe)
    assert events.count('forward') == 1 and not owners.enabled


def test_resource_queries_retain_individual_intervals_and_failures(monkeypatch):
    monkeypatch.setattr(module, 'process_tree', lambda pid: [{'pid': pid}])
    monkeypatch.setattr(module, 'cgroup_memory', lambda: {'memory': 1})
    def gpu(*args, **kwargs):
        if '--query-compute-apps' in args[0][1]:
            raise TimeoutError('query timed out')
        return SimpleNamespace(args=args[0], returncode=3, stdout='', stderr='gpu query failed')
    monkeypatch.setattr(module.subprocess, 'run', gpu)
    row = module.resource_snapshot(0.25)
    assert row['pid_start_time'] > 0 and row['parent_start_time'] > 0
    assert row['boot_id'] and row['pid_namespace'].startswith('pid:[')
    assert set(row['queries']) == {'processes', 'cgroup', 'nic', 'physical_gpu', 'compute_apps'}
    assert row['queries']['physical_gpu']['value']['exit_code'] == 3
    assert 'query timed out' in row['queries']['compute_apps']['error']
    for query in row['queries'].values():
        assert row['started_ns'] <= query['started_ns'] <= query['finished_ns'] <= row['finished_ns']
    assert row['scope'] == 'bracketing_queries_not_continuous_exclusivity'


@pytest.mark.parametrize('kind', ['swa', 'dsa'])
def test_sample_orchestration_reuses_native_setup_and_retains_only_timed_outputs(tmp_path, monkeypatch, kind):
    from contextlib import nullcontext
    import sys
    from types import ModuleType
    from test_local_attention_measurement import fake_vllm
    class Tensor:
        dtype = 'torch.bfloat16'
        shape = (2, 4)
        def clone(self): return Tensor()
        def copy_(self, other): pass
        def cuda(self): return self
        def cpu(self): return self
        def detach(self): return self
        def long(self): return self
        def stride(self): return (4, 1)
        def numel(self): return 8
        def element_size(self): return 2
        def __getitem__(self, other): return self
        def __floordiv__(self, other): return self
        def __mod__(self, other): return self
        def __mul__(self, other): return self
        def __add__(self, other): return self
    owners, calls = SimpleNamespace(enabled=False, calls=[]), []
    class Attention:
        attn = SimpleNamespace(layer_name='fake.attention')
        def __call__(self, positions, inputs):
            calls.append('native')
            if owners.enabled: owners.calls.extend(module.attention.OWNERS[kind])
            return Tensor()
    class Tracer:
        def __init__(self, path): self.trace_path = str(Path(path)/'profiler_traces/fake.json')
        def __enter__(self): return self
        def __exit__(self, *args): Path(self.trace_path).write_text('FAKE_TRACE')
        def get_operation_time_stats(self):
            stats = {s: {'count': 2, 'mean': 1.0, 'kernel_count': 1} for s in module.attention.OWNERS[kind]}
            stats[module.attention.WHOLE_FORWARD] = {'count': 2, 'mean': float(len(stats)), 'kernel_count': len(stats)}
            return stats
    tracer = ModuleType('frontier.profiling.utils.record_function_tracer')
    tracer.RecordFunctionTracer = Tracer
    monkeypatch.setitem(sys.modules, tracer.__name__, tracer)
    fake_vllm(monkeypatch, {
        'vllm.forward_context': {'set_forward_context': lambda *a, **k: nullcontext()},
        'vllm.v1.attention.backend': {'CommonAttentionMetadata': lambda **k: SimpleNamespace(**k)}})
    monkeypatch.setattr(module, 'resource_snapshot', lambda seconds: {'scope': 'fake_only'})
    torch = SimpleNamespace(Generator=lambda **k: SimpleNamespace(manual_seed=lambda seed: object()),
        randn=lambda *a, **k: Tensor(), arange=lambda *a, **k: Tensor(), tensor=lambda *a, **k: Tensor(),
        bfloat16='bf16', int32='int32', device=lambda *a: nullcontext(),
        cuda=SimpleNamespace(synchronize=lambda: None),
        profiler=SimpleNamespace(record_function=lambda *a: nullcontext()),
        save=lambda value, stream: stream.write(b'FAKE_TIMED_OUTPUT'))
    selected = [{'label': 'sample', 'content_seed': 1, 'prefix': 512, 'query': 2}]
    rows = module.measure_samples(torch, tmp_path/'new',
        SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=4))),
        kind, selected, {'geometry': {'W': 512}}, Attention(), SimpleNamespace(block_size=16),
        [Tensor()], [Tensor()], SimpleNamespace(build=lambda *a, **k: SimpleNamespace()), Tensor(),
        {'backend': 'fake'}, {}, owners, 1, 2, 1024, 65536, 0.1)
    assert len(calls) == 6  # one prefix, one warmup, four timed passes
    assert len(rows) == 1 and rows[0]['schedule'] == [512, 2]
    assert len(list((tmp_path/'new/sample/timed_outputs').glob('*.pt'))) == 4
    saved = json.loads((tmp_path/'new/sample/observation.json').read_text())
    assert saved['status'] == 'raw_unadmitted'
    assert saved['kernel_only_reconciliation']['status'] == 'KERNELS_RECONCILED'
    assert saved['evidence']['continuous_identity_exclusivity'] == 'UNVERIFIED'


def test_import_is_inert_without_native_packages_or_queries():
    import os
    import subprocess
    import sys
    code = '''import importlib.abc, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'steptron'}:
            raise AssertionError('Native import attempted: '+fullname)
sys.meta_path.insert(0, BlockNative())
sys.path.insert(0, sys.argv[1])
import local_attention_evidence
print('INERT_IMPORT')
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(PATH.parent)],
        capture_output=True, text=True, timeout=15,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'INERT_IMPORT'


@pytest.fixture
def measured_sample(tmp_path, monkeypatch):
    """Run measure_samples with the unchanged tracer and fake device boundaries."""
    from contextlib import nullcontext
    import sys
    from types import ModuleType
    from test_local_attention_measurement import fake_vllm

    def run_sample(*, native_error=None, cleanup_phase=None, cleanup_error=None,
                   async_error=None, trace_bytes=None, evidence_bytes=65536):
        calls, synchronizations, exports = [], [], []
        owners = SimpleNamespace(enabled=False, calls=[])
        events = []
        for repeat in range(2):
            for ordinal, scope in enumerate(module.attention.OWNERS['swa']+
                                             (module.attention.WHOLE_FORWARD,)):
                start = (repeat*6+ordinal)*100
                events.append({'cat': 'user_annotation', 'name': 'vidur_'+scope,
                               'ts': start, 'dur': 90})
                kernels = 5 if scope == module.attention.WHOLE_FORWARD else 1
                for kernel in range(kernels):
                    correlation = start+kernel
                    events.extend([
                        {'cat': 'cuda_runtime', 'ts': start+kernel+1, 'dur': 1,
                         'args': {'correlation': correlation}},
                        {'cat': 'kernel', 'ts': start+20+kernel, 'dur': 10,
                         'args': {'correlation': correlation}}])
        trace_payload = json.dumps({'traceEvents': events}).encode()
        if trace_bytes is not None:
            trace_payload += b' '*max(0, trace_bytes-len(trace_payload))

        class Tensor:
            dtype = 'torch.bfloat16'
            shape = (2, 4)
            def clone(self): return Tensor()
            def copy_(self, other): pass
            def cuda(self): return self
            def cpu(self): return self
            def detach(self): return self
            def long(self): return self
            def stride(self): return (4, 1)
            def numel(self): return 8
            def element_size(self): return 2
            def __getitem__(self, other): return self
            def __floordiv__(self, other): return self
            def __mod__(self, other): return self
            def __mul__(self, other): return self
            def __add__(self, other): return self

        class NativeAttention:
            attn = SimpleNamespace(layer_name='fake.attention')
            def __call__(self, positions, inputs):
                calls.append('native')
                if owners.enabled:
                    owners.calls.extend(module.attention.OWNERS['swa'])
                if native_error is not None and len(calls) == 3:
                    raise native_error
                return Tensor()

        class Profile:
            def __enter__(self): return self
            def __exit__(self, *exception):
                if cleanup_phase == 'stop':
                    raise cleanup_error
            def export_chrome_trace(self, path):
                exports.append(Path(path))
                Path(path).write_bytes(trace_payload)
                if cleanup_phase == 'export':
                    raise cleanup_error

        def synchronize():
            synchronizations.append(len(calls))
            # Warmup drain, tracer priming, timed pre-drain, then native drain.
            if async_error is not None and len(synchronizations) == 4:
                raise async_error
            if cleanup_phase == 'drain' and len(synchronizations) == (5 if async_error else 4):
                raise cleanup_error

        torch = ModuleType('torch')
        torch.__dict__.update(Generator=lambda **k: SimpleNamespace(manual_seed=lambda seed: object()),
            randn=lambda *a, **k: Tensor(), arange=lambda *a, **k: Tensor(),
            tensor=lambda *a, **k: Tensor(), empty=lambda *a, **k: Tensor(),
            bfloat16='bf16', int32='int32', device=lambda *a: nullcontext(),
            cuda=SimpleNamespace(synchronize=synchronize),
            profiler=SimpleNamespace(record_function=lambda *a: nullcontext(), profile=lambda **k: Profile(),
                ProfilerActivity=SimpleNamespace(CPU='cpu', CUDA='cuda')),
            save=lambda value, stream: stream.write(b'FAKE_TIMED_OUTPUT'))
        monkeypatch.setitem(sys.modules, 'torch', torch)
        tracer_path = PATH.parents[3]/'frontier/profiling/utils/record_function_tracer.py'
        tracer_spec = importlib.util.spec_from_file_location(
            'frontier.profiling.utils.record_function_tracer', tracer_path)
        tracer_module = importlib.util.module_from_spec(tracer_spec)
        tracer_spec.loader.exec_module(tracer_module)
        monkeypatch.setitem(sys.modules, tracer_spec.name, tracer_module)
        fake_vllm(monkeypatch, {
            'vllm.forward_context': {'set_forward_context': lambda *a, **k: nullcontext()},
            'vllm.v1.attention.backend': {'CommonAttentionMetadata': lambda **k: SimpleNamespace(**k)}})
        monkeypatch.setattr(module, 'resource_snapshot', lambda seconds: {
            'pid': 123, 'pid_start_time': 456, 'scope': 'fake_only'})
        output = tmp_path/'new'
        arguments = (torch, output,
            SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=4))),
            'swa', [{'label': 'sample', 'content_seed': 1, 'prefix': 512, 'query': 2}],
            {'geometry': {'W': 512}}, NativeAttention(), SimpleNamespace(block_size=16),
            [Tensor()], [Tensor()], SimpleNamespace(build=lambda *a, **k: SimpleNamespace()), Tensor(),
            {'backend': 'fake'}, {}, owners, 1, 2, 1024, evidence_bytes, 0.1)
        return SimpleNamespace(execute=lambda: module.measure_samples(*arguments), output=output,
            calls=calls, owners=owners, exports=exports, trace_payload=trace_payload)
    return run_sample


@pytest.mark.parametrize('native_error', [RuntimeError('native forward failed'),
                                         KeyboardInterrupt('native forward interrupted')])
@pytest.mark.parametrize('cleanup_phase', ['drain', 'stop', 'export'])
@pytest.mark.parametrize('cleanup_error', [RuntimeError('tracer cleanup failed'),
                                          KeyboardInterrupt('tracer cleanup interrupted')])
def test_measure_samples_preserves_native_exception_and_retains_tracer_cleanup_error(
        measured_sample, native_error, cleanup_phase, cleanup_error):
    sample = measured_sample(native_error=native_error, cleanup_phase=cleanup_phase,
                             cleanup_error=cleanup_error)
    with pytest.raises(type(native_error)) as caught:
        sample.execute()
    assert caught.value is native_error
    assert len(sample.calls) == 3 and sample.owners.enabled is False
    partial = json.loads((sample.output/'sample/timed_outputs/0000_owner_pass_failure.json').read_text())
    assert str(native_error) in partial['native_error']
    cleanup_paths = list(sample.output.parent.rglob('tracer_cleanup_error.json'))
    assert len(cleanup_paths) == 1
    cleanup = json.loads(cleanup_paths[0].read_text())
    assert str(cleanup_error) in cleanup['tracer_cleanup_error']
    assert str(native_error) in cleanup['body_error']


@pytest.mark.parametrize('cleanup_phase', [None, 'drain', 'stop', 'export'])
def test_measure_samples_records_async_native_failure_before_rethrow(measured_sample, cleanup_phase):
    native_error = RuntimeError('native asynchronous drain failed')
    sample = measured_sample(async_error=native_error, cleanup_phase=cleanup_phase,
        cleanup_error=RuntimeError('secondary tracer cleanup failed'))
    with pytest.raises(RuntimeError) as caught:
        sample.execute()
    assert caught.value is native_error
    assert len(sample.calls) == 3 and sample.owners.enabled is False
    failure = sample.output/'sample/timed_outputs/0000_owner_pass_failure.json'
    assert failure.is_file(), 'Post-forward native drain failure must retain a partial pass record'
    partial = json.loads(failure.read_text())
    assert partial['kind'] == 'swa' and partial['sample'] == 'sample'
    assert partial['before']['pid'] == 123 and partial['before']['pid_start_time'] == 456
    assert partial['forward_started_ns'] <= partial['forward_finished_ns']
    assert partial['forward_started_monotonic_ns'] <= partial['forward_finished_monotonic_ns']
    assert partial['host_profiled_launch_ms'] >= 0
    assert partial['host_metadata_build_ms'] >= 0
    assert partial['owner_calls'] == list(module.attention.OWNERS['swa'])
    assert partial['native_failure_phase'] == 'post_forward_device_drain'
    assert str(native_error) in partial['native_error']
    assert not list((sample.output/'sample/timed_outputs').glob('*.pt'))


def test_measure_samples_keeps_oversized_export_outside_retained_sample(measured_sample):
    sample = measured_sample(trace_bytes=65536)
    with pytest.raises(ValueError, match='budget'):
        sample.execute()
    retained = sample.output/'sample'
    used = sum(path.stat().st_size for path in retained.rglob('*') if path.is_file())
    assert used <= 65536, 'Raw profiler export must not overshoot the retained sample cap'
    assert len(sample.exports) == 1
    staged = sample.exports[0]
    assert not staged.is_relative_to(retained) and staged.read_bytes() == sample.trace_payload
    assert not list((retained/'profiler_traces').iterdir())
    assert not (retained/'observation.json').exists()
    manifest = json.loads((staged.parent.parent/'trace_staging.json').read_text())
    assert manifest['status'] == 'raw_unadmitted'
    assert manifest['retained_sample_cap_applies'] is False


def test_measure_samples_cleanup_record_write_failure_does_not_replace_native_error(
        measured_sample, monkeypatch, capsys):
    native_error = KeyboardInterrupt('native forward interrupted')
    cleanup_error = RuntimeError('tracer export failed')
    sample = measured_sample(native_error=native_error, cleanup_phase='export',
                             cleanup_error=cleanup_error)
    original_open = Path.open
    def open_path(path, *args, **kwargs):
        if path.name == 'tracer_cleanup_error.json':
            raise OSError('cleanup record write failed')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', open_path)
    with pytest.raises(KeyboardInterrupt) as caught:
        sample.execute()
    assert caught.value is native_error
    stderr = capsys.readouterr().err
    assert str(native_error) in stderr and str(cleanup_error) in stderr
    assert 'cleanup record write failed' in stderr
    assert sample.exports[0].read_bytes() == sample.trace_payload


@pytest.mark.parametrize('cleanup_phase', ['stop', 'export'])
def test_measure_samples_cleanup_failure_without_native_failure_stays_visible(measured_sample, cleanup_phase):
    cleanup_error = RuntimeError('tracer cleanup failed')
    sample = measured_sample(cleanup_phase=cleanup_phase, cleanup_error=cleanup_error)
    with pytest.raises(RuntimeError) as caught:
        sample.execute()
    assert caught.value is cleanup_error and len(sample.calls) == 6
    assert not (sample.output/'sample/observation.json').exists()
    if cleanup_phase == 'export':
        assert sample.exports[0].read_bytes() == sample.trace_payload
    else:
        assert not sample.exports


def test_measure_samples_record_budget_failure_preserves_staged_and_copied_trace(measured_sample):
    sample = measured_sample(evidence_bytes=12000)
    with pytest.raises(ValueError, match='sample record exceed'):
        sample.execute()
    retained = sample.output/'sample'
    used = sum(path.stat().st_size for path in retained.rglob('*') if path.is_file())
    assert used <= 12000
    assert not (retained/'observation.json').exists()
    copied = list((retained/'profiler_traces').iterdir())
    assert len(copied) == 1
    assert copied[0].read_bytes() == sample.exports[0].read_bytes() == sample.trace_payload


@pytest.mark.parametrize('failure_phase', ['forward', 'drain'])
@pytest.mark.parametrize('cleanup_phase', [None, 'export'])
def test_partial_record_interrupt_cannot_replace_existing_native_error(
        measured_sample, monkeypatch, failure_phase, cleanup_phase):
    native_error = RuntimeError('original native failure')
    sample = measured_sample(
        native_error=native_error if failure_phase == 'forward' else None,
        async_error=native_error if failure_phase == 'drain' else None,
        cleanup_phase=cleanup_phase, cleanup_error=OSError('secondary export failure'))
    original_open = Path.open

    def open_path(path, *args, **kwargs):
        if path.name == '0000_owner_pass_failure.json':
            raise KeyboardInterrupt('partial record interrupted')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_path)
    with pytest.raises(BaseException) as caught:
        sample.execute()
    assert caught.value is native_error
    assert len(sample.calls) == 3 and not sample.owners.enabled
    assert not (sample.output/'sample/timed_outputs/0000_owner_pass_failure.json').exists()
    if cleanup_phase:
        cleanup = json.loads((sample.output.parent/'new_trace_staging/sample/tracer_cleanup_error.json').read_text())
        assert str(native_error) in cleanup['body_error']
        assert 'RuntimeError: original native failure' in cleanup['body_error']


def test_measure_samples_stats_read_bounded_copy_of_unchanged_staged_trace(measured_sample):
    sample = measured_sample()
    rows = sample.execute()
    assert len(sample.calls) == 6
    row = rows[0]
    retained = sample.output/'sample'
    copied, staged = Path(row['trace']), sample.exports[0]
    assert copied.is_relative_to(retained/'profiler_traces')
    assert not staged.is_relative_to(retained)
    assert copied.read_bytes() == staged.read_bytes() == sample.trace_payload
    assert row['kernel_only_stats']['swa_attention']['mean'] == pytest.approx(0.01)
    assert row['kernel_only_stats']['attention_forward']['mean'] == pytest.approx(0.05)
    assert row['kernel_only_reconciliation']['status'] == 'KERNELS_RECONCILED'
    assert sum(path.stat().st_size for path in retained.rglob('*') if path.is_file()) <= 65536
    assert row['trace_staging']['status'] == 'raw_unadmitted'
    assert row['trace_staging']['retained_sample_cap_applies'] is False
    assert Path(row['trace_staging']['trace']) == staged
