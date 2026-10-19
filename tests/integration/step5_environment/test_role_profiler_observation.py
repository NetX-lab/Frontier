"""CPU failure and identity checks for the original native profiler observation."""
import ast
import copy
import ctypes
from enum import IntEnum
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from test_role_execution_driver import (CpuLLM, CpuSampling, config, cpu_only,
                                       original_cpu_method, startup_fixture)
from test_role_model_input_observation import (census, census_runner, events, native,
                                              record, runner, work)
from test_role_execution_observation import output
import check_role_execution as caller
import role_execution_observation as execution
import role_model_input_observation as model
import role_profiler_observation as observation


class NativeExporter:
    """Retained Kineto writes path.tmp and renames it after export."""
    def __init__(self, payload, *, failure=None, before=None):
        self.payload, self.failure, self.before = payload, failure, before
        self.calls, self.thread = [], None

    def export_chrome_trace(self, path):
        self.calls.append(path)
        self.thread = threading.get_ident()
        if self.before is not None:
            self.before()
        temporary = Path(path+'.tmp')
        with temporary.open('wb', buffering=0) as stream:
            offset = 0
            while offset < len(self.payload):
                offset += stream.write(self.payload[offset:offset+8192])
        os.replace(temporary, path)
        if self.failure is not None:
            raise self.failure


def trace_payload(events=None):
    return json.dumps(dict(traceEvents=events or [dict(ph='X', cat='user_annotation',
        name='step5_native:execute:1:1:call', pid=os.getpid(), tid=1, ts=1.0, dur=2.0)]),
        separators=(',', ':')).encode()


@pytest.mark.parametrize('size, cap', [(1000, 4096), (262144, 31), (1000, 1000)])
def test_native_export_drains_on_calling_thread_and_caps_one_file(tmp_path, size, cap):
    exporter = NativeExporter(b'x'*size)
    path = tmp_path/'trace.json'
    result = observation.export_trace(exporter, path, cap, time.monotonic()+5)
    assert exporter.thread == threading.get_ident() and len(exporter.calls) == 1
    assert result['bytes_seen'] == size and result['retained_bytes'] == min(size, cap)
    assert path.read_bytes() == b'x'*min(size, cap)
    assert result['reader_complete'] and not result['deadline_exceeded']
    assert result['status'] == ('EXPORTED_WITHIN_BYTE_AND_DEADLINE_LIMITS' if size <= cap else 'UNVERIFIED')
    assert list(tmp_path.iterdir()) == [path]


def test_native_export_exception_remains_unverified_after_reader_completion(tmp_path):
    failure = RuntimeError('original exporter failure')
    exporter = NativeExporter(b'x'*65536, failure=failure)
    path = tmp_path/'trace.json'
    result = observation.export_trace(exporter, path, 65536, time.monotonic()+5)
    assert result['status'] == 'UNVERIFIED' and result['reader_complete']
    assert repr(failure) in result['error'] and len(exporter.calls) == 1
    assert result['bytes_seen'] == 65536 and path.stat().st_size <= 65536
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('fault', ['write', 'short_write', 'clock', 'read'])
def test_reader_failures_do_not_leave_native_exporter_blocked(tmp_path, monkeypatch, fault):
    path = tmp_path/'trace.json'
    original_open, original_read = Path.open, os.read
    reader_clock_failed = threading.Event()

    class Sink:
        def __init__(self, stream):
            self.stream = stream
        def write(self, payload):
            if fault == 'short_write':
                return self.stream.write(payload[:1])
            raise OSError('original trace write failure')
        def close(self):
            self.stream.close()

    def open_file(current, *args, **kwargs):
        stream = original_open(current, *args, **kwargs)
        return Sink(stream) if current == path and args == ('xb',) and fault in ('write', 'short_write') else stream

    def read(fd, count):
        if threading.current_thread().name == 'step5-native-trace-export' and fault == 'read':
            raise OSError('original trace read failure')
        return original_read(fd, count)

    def clock():
        if fault == 'clock' and threading.current_thread().name == 'step5-native-trace-export':
            reader_clock_failed.set()
            raise RuntimeError('original trace clock failure')
        return time.monotonic()

    monkeypatch.setattr(Path, 'open', open_file)
    monkeypatch.setattr(os, 'read', read)
    exporter = NativeExporter(b'x'*262144)
    started = time.monotonic()
    result = observation.export_trace(exporter, path, 4096, started+3, clock=clock)
    assert time.monotonic()-started < 3
    assert result['status'] == 'UNVERIFIED' and result['error'] is not None
    assert result['reader_complete'] and path.stat().st_size <= 4096
    assert len(exporter.calls) == 1 and list(tmp_path.iterdir()) == [path]
    if fault == 'clock':
        assert reader_clock_failed.is_set()


def test_deadline_exhaustion_stops_retention_and_continues_native_drain(tmp_path):
    current = [1.0]
    exporter = NativeExporter(b'x'*262144, before=lambda: current.__setitem__(0, 10.0))
    path = tmp_path/'trace.json'
    result = observation.export_trace(exporter, path, 31, 5.0, clock=lambda: current[0])
    assert result['status'] == 'UNVERIFIED' and result['deadline_exceeded']
    assert path.stat().st_size <= 31
    for thread in threading.enumerate():
        if thread.name == 'step5-native-trace-export':
            thread.join(timeout=1)
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('existing', ['output', 'native', 'pipe'])
def test_export_never_removes_preexisting_outputs_or_native_temporary(tmp_path, existing):
    path = tmp_path/'trace.json'
    protected = {'output': path, 'native': path.with_name(path.name+'.native'),
                 'pipe': path.with_name(path.name+'.native.tmp')}[existing]
    protected.write_bytes(b'old artifact')
    exporter = NativeExporter(b'new')
    with pytest.raises(FileExistsError):
        observation.export_trace(exporter, path, 4096, time.monotonic()+3)
    assert protected.read_bytes() == b'old artifact' and exporter.calls == []


def profiler_sources():
    owners = dict(wrapper=('TorchProfilerWrapper.__init__', 'TorchProfilerWrapper._start',
        'TorchProfilerWrapper._stop', 'TorchProfilerWrapper.annotate_context_manager'),
        worker=('Worker.profile', 'Worker._get_cudagraph_capture_context'),
        torch_graphs=('CUDAGraph.raw_cuda_graph_exec',), cuda_runtime=(), cupti=())
    return {'/native/'+relative: dict(sha256=digest, code_owners=[dict(name=owner, first_line=index+1)
        for index, owner in enumerate(owners[key])])
        for key, (relative, digest) in observation.SOURCE_PINS.items()}


def profiler_case(output, *, exporter=None):
    records, _ = output
    identity = dict(rank=0, local_rank=0, pid=os.getpid(), start_time=1, parent_pid=2,
        parent_start_time=3, generation='cpu-profiler', node_rank=0, boot_id='cpu-boot',
        pid_namespace='pid:[cpu]', observer_sha256=execution.SOURCE['capacity_observer_sha256'])
    (records/f'worker_0_{os.getpid()}.json').write_text(json.dumps(identity))
    worker = SimpleNamespace(rank=0, local_rank=0, _capacity_identity=identity,
                             _request_observation=execution.RequestObservation())
    state = worker._request_observation
    observed = observation.NativeProfilerObservation(worker, observation.ProfilerObservationLimits(8192, 100),
        records, time.monotonic()+5, execution.write_record, state, profiler_sources())
    wrapper = SimpleNamespace(is_running=False, _active=False,
        profiler=exporter or NativeExporter(trace_payload()), annotate_context_manager=lambda name: None)
    observed.profiler = worker.profiler = wrapper
    state.profiler = observed
    return worker, observed, wrapper


def start_window(observed, wrapper, label):
    observed.before_start(label)
    wrapper.is_running = wrapper._active = True
    observed.current['native_start'] = 'ORIGINAL_START_RETURNED'
    observed.after_start()


def stop_window(observed, wrapper):
    observed.trace_ready(wrapper.profiler)
    observed.current['native_stop'] = 'ORIGINAL_STOP_RETURNED'
    wrapper.is_running = wrapper._active = False
    observed.after_stop()


def test_original_windows_trace_completion_and_single_reservation(output):
    worker, observed, wrapper = profiler_case(output)
    for label in observation.WINDOWS:
        start_window(observed, wrapper, label)
        pending = execution.device_reservation_bytes(output[0])
        assert pending == observed.limits.trace_bytes
        stop_window(observed, wrapper)
        assert execution.device_reservation_bytes(output[0]) == 0
    summary = observed.summary()
    assert summary['errors'] == [] and summary['current'] is None
    assert [row['label'] for row in summary['windows']] == list(observation.WINDOWS)
    assert all(row['status'] == 'COMPLETE_DIAGNOSTIC_WINDOW_NOT_G2' for row in summary['windows'])
    assert summary['G2'] == 'UNVERIFIED'
    assert len(list(output[0].glob('device_reservation_rank_*'))) == 1
    assert len(list(output[0].glob('profile_rank_*.json'))) == 2
    records = [json.loads(line) for line in Path(worker._request_observation.record_file).read_text().splitlines()]
    assert len(records) == 2 and all(row['window']['handler_calls'] == 1 for row in records)


@pytest.mark.parametrize('fault', ['missing_handler', 'duplicate_handler', 'malformed_json', 'event_limit',
                                   'swallowed_stop', 'substituted_wrapper'])
def test_trace_completion_faults_finalize_unverified_window(output, fault):
    payload = b'bad JSON' if fault == 'malformed_json' else trace_payload()
    worker, observed, wrapper = profiler_case(output, exporter=NativeExporter(payload))
    if fault == 'event_limit':
        observed.limits = observation.ProfilerObservationLimits(8192, 1)
        wrapper.profiler = NativeExporter(trace_payload([dict(cat='cpu_op'), dict(cat='cpu_op')]))
    start_window(observed, wrapper, 'capture')
    if fault != 'missing_handler':
        observed.trace_ready(wrapper.profiler)
        if fault == 'duplicate_handler':
            observed.trace_ready(wrapper.profiler)
    if fault == 'substituted_wrapper':
        worker.profiler = SimpleNamespace(is_running=False, _active=False)
    if fault != 'swallowed_stop':
        observed.current['native_stop'] = 'ORIGINAL_STOP_RETURNED'
    wrapper.is_running = wrapper._active = False
    observed.after_stop()
    assert observed.current is None and len(observed.windows) == 1
    assert observed.windows[0]['status'] == 'UNVERIFIED' and observed.errors
    assert observed.summary()['G2'] == 'UNVERIFIED'
    assert worker._request_observation.unverified


def test_partial_trace_does_not_double_charge_rank_and_node_reservation(output):
    worker, observed, wrapper = profiler_case(output)
    start_window(observed, wrapper, 'capture')
    trace = output[0]/f'profile_rank_0_pid_{os.getpid()}_capture.json'
    trace.write_bytes(b'x'*1024)
    assert execution.pending_profiler_bytes(worker._request_observation) == 8192-1024
    assert execution.device_reservation_bytes(output[0]) == 8192-1024
    used = execution.rank_record_bytes(output[0], 0)
    assert used+execution.device_reservation_bytes(output[0]) == used-1024+8192
    execution.reserve_device_bytes(worker, 128, device_preparation=True)
    assert worker._request_observation.device_prepared
    assert execution.device_reservation_bytes(output[0]) == 128+8192-1024
    with pytest.raises(ValueError, match='only once'):
        execution.reserve_device_bytes(worker, 128)
    with pytest.raises(ValueError, match='all eight node workers'):
        execution.device_reservation_bytes(output[0], require_complete=True)
    for rank in range(1, 8):
        identity = dict(worker._capacity_identity, rank=rank, local_rank=rank, pid=os.getpid()+rank)
        (output[0]/f"worker_{rank}_{identity['pid']}.json").write_text(json.dumps(identity))
        reservation = dict(kind='device_observation_reservation', rank=rank, local_rank=rank,
            pid=identity['pid'], worker_identity=identity, source=dict(execution.SOURCE),
            status='NOT_ALLOCATED_PREPARATION_FAILED', owned_bytes=0)
        path = output[0]/f"device_reservation_rank_{rank}_pid_{identity['pid']}.json"
        path.write_text(json.dumps(reservation))
    with pytest.raises(execution.PendingTraceReservation, match='remains pending') as failure:
        execution.device_reservation_bytes(output[0], require_complete=True)
    assert failure.value.reservation_bytes == 128+8192-1024


def test_reservation_update_is_atomic_and_preserves_old_record_on_write_failure(output, monkeypatch):
    worker, observed, wrapper = profiler_case(output)
    start_window(observed, wrapper, 'capture')
    path = Path(worker._request_observation.device_reservation_file)
    old = path.read_bytes()
    original_open = Path.open
    class FailedWrite:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def write(self, payload):
            self.stream.write(payload[:5])
            raise OSError('reservation write failed')
    def open_file(current, *args, **kwargs):
        stream = original_open(current, *args, **kwargs)
        return FailedWrite(stream) if current.name.endswith('.json.pending') else stream
    monkeypatch.setattr(Path, 'open', open_file)
    with pytest.raises(OSError, match='reservation write failed'):
        execution.reserve_device_bytes(worker, 128)
    assert path.read_bytes() == old
    assert execution.device_reservation_bytes(output[0]) == 8192
    assert not worker._request_observation.device_prepared
    identity_bytes = (output[0]/f'worker_0_{os.getpid()}.json').stat().st_size
    assert execution.rank_record_bytes(output[0], 0) == len(old)+5+identity_bytes


def test_physical_stop_failure_still_finalizes_without_second_native_stop(output):
    worker, observed, wrapper = profiler_case(output)
    stop_failure = RuntimeError('native stop failed')
    calls = []
    def profile(self, **kwargs):
        calls.append(kwargs)
        raise stop_failure
    base = type('NativeWorker', (), dict(profile=profile, execute_model=lambda self, *a, **k: None,
        **{stage: lambda self: None for stage in execution.capacity.STAGES}))
    actual = execution.observed_worker(base)()
    actual.__dict__.update(worker.__dict__)
    actual.model_runner = SimpleNamespace()
    observed.worker = actual
    start_window(observed, wrapper, 'capture')
    result = actual.physical_profile('capture', False)
    assert calls == [dict(is_start=False)]
    assert result['profiler_observation']['current'] is None
    assert result['profiler_observation']['windows'][0]['status'] == 'UNVERIFIED'
    assert result['profiler_observation']['errors'][0]['error'] == str(stop_failure)


def profiler_response(native, *, started=False):
    identity_keys = ('rank', 'local_rank', 'pid', 'start_time', 'parent_pid', 'parent_start_time',
                     'generation', 'node_rank', 'boot_id', 'pid_namespace', 'observer_sha256')
    identity = {key: native[key] for key in identity_keys}
    sources = profiler_sources()
    windows = []
    for index, label in enumerate(observation.WINDOWS):
        start = 1.0+index*5
        trace = dict(status='COMPLETE_NATIVE_TRACE_NOT_G2', reader_complete=True, error=None,
            deadline_exceeded=False, byte_limit=8192, bytes_seen=1024, retained_bytes=1024,
            events=10, annotations=1, graph_launches=1, kernels=1, sha256='a'*64,
            path=f'/records/profile_rank_{native["rank"]}_pid_{native["pid"]}_{label}.json',
            started=start+1, finished=start+2, inspection_started=start+2,
            inspection_finished=start+3, deadline=20.0, seconds=1.0)
        windows.append(dict(label=label, status='COMPLETE_DIAGNOSTIC_WINDOW_NOT_G2', handler_calls=1,
            native_start='ORIGINAL_PROFILER_RUNNING', native_stop='ORIGINAL_PROFILER_STOPPED_AND_TRACE_COMPLETE',
            profiler_id=123, wrapper_source=sources, trace=trace, start=start, stop=start+4))
    current = None
    if started:
        current = dict(windows.pop(), status='COLLECTING_DIAGNOSTIC_TRACE_NOT_G2', handler_calls=0,
                       native_stop='UNMEASURED', trace=None)
    return dict(rank=native['rank'], local_rank=native['local_rank'], pid=native['pid'],
        worker_identity=identity, source=dict(execution.SOURCE), profiler_observation=dict(
            status='NATIVE_PROFILER_DIAGNOSTIC_ONLY_NOT_G2', worker_identity=identity,
            observer_sha256=observation.SOURCE_SHA256, errors=[],
            limits=dict(trace_bytes=8192, trace_events=100), sources=sources, windows=windows, current=current))


@pytest.mark.parametrize('started', [True, False])
def test_caller_requires_complete_original_workers_and_trace_boundaries(config, tmp_path, started):
    _, initial = startup_fixture(config, tmp_path)
    config['physical_observation'] = dict(trace_bytes=8192, trace_events=100)
    rows = [profiler_response(row, started=started) for row in initial]
    verdict = caller.profiler_rows(rows, initial, config, started=started)
    assert verdict['ranks'] == list(range(16))
    assert verdict['status'].endswith('ONLY_NOT_G2')
    assert verdict['physical_kernel_attribution'] == 'UNVERIFIED'


@pytest.mark.parametrize('fault', ['identity', 'summary_identity', 'source', 'code_owner', 'hash', 'start',
    'stop', 'reader', 'bytes', 'events', 'deadline', 'path', 'current', 'duplicates', 'missing'])
def test_caller_rejects_false_profiler_completion(config, tmp_path, fault):
    _, initial = startup_fixture(config, tmp_path)
    config['physical_observation'] = dict(trace_bytes=8192, trace_events=100)
    rows = [profiler_response(row) for row in initial]
    observed = rows[0]['profiler_observation']
    window = observed['windows'][0]
    trace = window['trace']
    if fault == 'identity':
        rows[0]['worker_identity'] = dict(rows[0]['worker_identity'], start_time=999)
    elif fault == 'summary_identity':
        observed['worker_identity'] = dict(observed['worker_identity'], parent_pid=999)
    elif fault == 'source':
        next(iter(observed['sources'].values()))['sha256'] = 'b'*64
    elif fault == 'code_owner':
        next(iter(observed['sources'].values()))['code_owners'].pop()
    elif fault == 'hash':
        trace['sha256'] = 'z'*64
    elif fault == 'start':
        window['native_start'] = 'UNMEASURED'
    elif fault == 'stop':
        window['native_stop'] = 'ORIGINAL_METHOD_RAISED'
    elif fault == 'reader':
        trace['reader_complete'] = False
    elif fault == 'bytes':
        trace['bytes_seen'] += 1
    elif fault == 'events':
        trace['events'] = 101
    elif fault == 'deadline':
        trace['inspection_finished'] = 100
    elif fault == 'path':
        trace['path'] = '/other/profile_rank_1_pid_1_capture.json'
    elif fault == 'current':
        observed['current'] = window
    elif fault == 'duplicates':
        rows[1] = rows[0]
    else:
        rows.pop()
    with pytest.raises(ValueError):
        caller.profiler_rows(rows, initial, config, started=False)


def test_retained_native_profiler_pins_and_loaded_owners_match_exact_sources():
    for key, (relative, pin) in observation.SOURCE_PINS.items():
        if key in ('torch_graphs', 'cuda_runtime', 'cupti'):
            continue  # Exact-image Torch source binding is checked by the CPU API script.
        base = 'observed_generation5' if key == 'wrapper' else 'observed'
        path = caller.capacity.TASK/'native_role_sources'/base/relative
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == pin
        compiled = compile(body, str(path), 'exec', dont_inherit=True)
        names = ('__init__', '_start', '_stop', 'annotate_context_manager') if key == 'wrapper' else (
            'profile', '_get_cudagraph_capture_context')
        owner = 'TorchProfilerWrapper' if key == 'wrapper' else 'Worker'
        for name in names:
            code = model.nested_code(compiled, owner, name)
            assert code.co_firstlineno > 0
        ast.parse(body)


class AnnotationContext:
    def __init__(self, name, calls, stack, failure=None):
        self.name, self.calls, self.stack, self.failure = name, calls, stack, failure

    def __enter__(self):
        self.calls.append(('enter', self.name))
        if self.failure == 'enter':
            raise RuntimeError('original annotation enter failure')
        self.stack.append(self.name)
        return self

    def __exit__(self, *exception):
        self.calls.append(('exit', self.name))
        assert self.stack.pop() == self.name
        if self.failure == 'exit':
            raise RuntimeError('original annotation exit failure')
        return False


def annotate_runner(observed, failure=None):
    calls, stack = [], []

    def annotate(name):
        return AnnotationContext(name, calls, stack, failure)

    observed._step5_profiler_observation = SimpleNamespace(annotate=annotate,
        capture_nodes=lambda previous: (dict(nodes={}), dict(nodes=[])),
        graph_identity=lambda graph: dict(cuda_graph_exec_handle=id(graph),
            cuda_graph_exec_identity='CPU_FIXTURE_ONLY_NOT_NATIVE'))
    return calls, stack


@pytest.mark.parametrize('mode', ['NONE', 'FULL', 'PIECEWISE'])
def test_annotations_follow_actual_nested_native_calls_and_restore_hooks(census, mode):
    observed, graph, _ = census_runner(census, mode)
    previous = sys.getprofile(), sys.gettrace()
    calls, stack = annotate_runner(observed)
    observed.execute_model(work())
    assert (sys.getprofile(), sys.gettrace()) == previous and stack == []
    assert calls[0][0] == 'enter' and calls[-1] == ('exit', calls[0][1])
    rows = record(observed)['events']
    annotated = [row for row in rows if 'trace_annotation' in row]
    assert annotated
    assert all(row['trace_annotation'] in [name for kind, name in calls if kind == 'enter'] for row in annotated)
    if mode == 'NONE':
        assert [layer.calls for layer in observed.model.layers] == [1, 1]
        assert any(row['kind'] == 'native_call' for row in annotated)
    else:
        assert graph.calls == 1
        assert [row['kind'] for row in annotated] == ['replay_attempt']
    assert record(observed)['status'] == 'OBSERVED_CPU_METADATA_ONLY_NOT_G2'


@pytest.mark.parametrize('fault', ['enter', 'exit', 'record_limit', 'native_exception'])
def test_annotation_faults_preserve_original_outcome_and_close_owned_contexts(census, fault):
    observed, _, _ = census_runner(census)
    native_error = RuntimeError('exact native layer failure')
    if fault == 'native_exception':
        observed.model.layers[1].failure = native_error
    calls, stack = annotate_runner(observed, fault if fault in ('enter', 'exit') else None)
    previous = sys.getprofile(), sys.gettrace()
    if fault == 'record_limit':
        original = model.ObservationScope.update
        count = [0]

        def exhaust(scope, row, **values):
            if 'trace_annotation' in values:
                count[0] += 1
                if count[0] == 2:
                    raise ValueError('annotation update exceeds original record allowance')
            return original(scope, row, **values)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(model.ObservationScope, 'update', exhaust)
            observed.execute_model(work())
    elif fault == 'native_exception':
        with pytest.raises(RuntimeError) as caught:
            observed.execute_model(work())
        assert caught.value is native_error
    else:
        observed.execute_model(work())
    assert previous == (sys.getprofile(), sys.gettrace()) and stack == []
    assert [layer.calls for layer in observed.model.layers] == [1, 1]
    if fault != 'native_exception':
        assert observed._model_input_observation.stopped
        assert record(observed)['status'] == 'UNVERIFIED'
    if fault == 'enter':
        assert calls == [('enter', 'step5_native:execute:1:0:scope')]


def test_annotation_duplicate_boundary_rejects_before_second_native_enter(native):
    observed, _, _ = runner(native)
    calls, stack = annotate_runner(observed)
    state = model.ModelInputObservation(native.sites, model.ObservationLimits(), execution.batch_snapshot,
                                        lambda row: None, invocations=1)
    scope = model.ObservationScope(observed, state, 'execute')
    first = scope.emit('native_call', frontiers=False)
    scope.start_annotation(('call', 1), first, 'call')
    with pytest.raises(ValueError, match='already active'):
        scope.start_annotation(('call', 1), first, 'call')
    scope.finish_annotation(('call', 1))
    assert len(calls) == 2 and stack == []


def correlation_case(pid=7):
    name = observation.annotation_name('execute', 1, 1, 'replay')
    graph = dict(path='FULL', owner_id=11, entry_id=12, graph_id=13,
                 continuity=dict(serial=1), cuda_graph_exec_handle=14, cupti_graph_exec_id=15,
                 cuda_graph_exec_identity='OBSERVED_READ_ONLY_HANDLE_NOT_NODE_BINDING')
    records = [dict(lifecycle='capture', sequence=1, events=[dict(copy.deepcopy(graph), kind='capture_entry_stored', event=1)]),
        dict(lifecycle='execute', sequence=1, events=[dict(copy.deepcopy(graph), event=1, kind='replay_attempt',
            trace_annotation=name), dict(event=2, kind='replay_outcome', replay_event=1,
                                        outcome='returned_submission_not_completion')])]
    events = [dict(ph='X', cat='user_annotation', name=name, pid=pid, tid=3, ts=10.0, dur=5.0),
              dict(ph='X', cat='cuda_runtime', name='cudaGraphLaunch', pid=pid, tid=3,
                   ts=11.0, dur=1.0, args=dict(correlation=41)),
              dict(ph='X', cat='kernel', name='repeated_native_kernel', pid=0, tid=9,
                   ts=20.0, dur=1.0, args=dict(correlation=41)),
              dict(ph='X', cat='kernel', name='repeated_native_kernel', pid=0, tid=9,
                   ts=21.0, dur=1.0, args=dict(correlation=41))]
    return events, records


def test_correlation_reuses_existing_chain_producer_and_keeps_physical_limits(monkeypatch):
    import frontier.profiling.kernel_gap.main as existing
    native_events, records = correlation_case()
    original, calls = existing.graph_launch_chains, []

    def chains(events):
        calls.append(events)
        return original(events)

    monkeypatch.setattr(existing, 'graph_launch_chains', chains)
    joined = observation.correlate_trace(native_events, records)
    assert calls == [native_events]
    assert joined['status'] == 'OBSERVED_REPLAY_LAUNCH_CORRELATION_ONLY_NOT_NODE_ATTRIBUTION'
    assert joined['replay_launches'][0]['kernel_trace_events'] == [2, 3]
    assert joined['replay_launches'][0]['capture_binding'] == dict(lifecycle='capture', sequence=1, event=1)
    assert joined['unmatched_kernels'] == 0
    assert joined['physical_kernel_attribution'] == 'UNVERIFIED_NO_CAPTURE_NODE_BINDING'
    assert joined['transient_scale_content'] == 'UNMEASURED' and joined['G2'] == 'UNVERIFIED'


@pytest.mark.parametrize('fault', ['missing_time', 'negative_duration', 'missing_host_pid',
    'missing_correlation', 'wrong_thread', 'launch_outside_annotation', 'repeated_launch',
    'cross_pid_correlation', 'duplicate_annotation', 'nested_annotations', 'missing_owner',
    'duplicate_owner', 'missing_outcome', 'raised_outcome', 'no_launch'])
def test_correlation_does_not_infer_owners_from_kernel_names_or_order(fault):
    native_events, records = correlation_case()
    if fault == 'missing_time':
        native_events[1].pop('ts')
    elif fault == 'negative_duration':
        native_events[2]['dur'] = -1
    elif fault == 'missing_host_pid':
        native_events[0].pop('pid')
    elif fault == 'missing_correlation':
        native_events[2]['args'] = {}
    elif fault == 'wrong_thread':
        native_events[1]['tid'] = 19
    elif fault == 'launch_outside_annotation':
        native_events[1]['ts'] = 100
    elif fault in ('repeated_launch', 'cross_pid_correlation'):
        native_events.append(copy.deepcopy(native_events[1]))
        if fault == 'cross_pid_correlation':
            native_events[-1]['pid'] = 8
    elif fault in ('duplicate_annotation', 'nested_annotations'):
        native_events.insert(1, copy.deepcopy(native_events[0]))
        if fault == 'nested_annotations':
            native_events[1]['name'] = observation.annotation_name('execute', 2, 1, 'replay')
    elif fault == 'missing_owner':
        records[1]['events'][0]['graph_id'] = None
    elif fault == 'duplicate_owner':
        records.append(copy.deepcopy(records[1]))
    elif fault == 'missing_outcome':
        records[1]['events'].pop()
    elif fault == 'raised_outcome':
        records[1]['events'][1]['outcome'] = 'raised'
    else:
        native_events.pop(1)
    joined = observation.correlate_trace(native_events, records)
    assert joined['status'] == 'UNVERIFIED' and joined['replay_launches'] == []
    assert joined['unmatched_kernels'] == 2
    assert joined['physical_kernel_attribution'] == 'UNVERIFIED_NO_CAPTURE_NODE_BINDING'


def test_correlation_rejects_changed_annotation_identity_and_capture_serial():
    native_events, records = correlation_case()
    records[0]['events'][0]['continuity']['serial'] = 2
    joined = observation.correlate_trace(native_events, records)
    assert joined['replay_launches'][0]['capture_binding'] == 'UNVERIFIED_CAPTURE_BINDING'
    records[1]['events'][0]['trace_annotation'] = 'step5_native:execute:99:1:replay'
    with pytest.raises(ValueError, match='original scope/event identity'):
        observation.correlate_trace(native_events, records)


@pytest.mark.parametrize('fault', ['changed_handle', 'missing_capture_handle', 'missing_replay_handle',
    'bool_capture_handle', 'zero_replay_handle', 'unverified_capture_getter',
    'unverified_replay_getter', 'bool_capture_serial'])
def test_capture_binding_requires_the_same_observed_executable_identity(fault):
    native_events, records = correlation_case()
    captured, replayed = records[0]['events'][0], records[1]['events'][0]
    if fault == 'changed_handle':
        replayed['cuda_graph_exec_handle'] += 1
    elif fault == 'missing_capture_handle':
        captured.pop('cuda_graph_exec_handle')
    elif fault == 'missing_replay_handle':
        replayed.pop('cuda_graph_exec_handle')
    elif fault == 'bool_capture_handle':
        captured['cuda_graph_exec_handle'] = True
        replayed['cuda_graph_exec_handle'] = 1
    elif fault == 'zero_replay_handle':
        captured['cuda_graph_exec_handle'] = replayed['cuda_graph_exec_handle'] = 0
    elif fault == 'unverified_capture_getter':
        captured['cuda_graph_exec_identity'] = 'UNVERIFIED_NOT_TORCH_CUDAGRAPH'
    elif fault == 'unverified_replay_getter':
        replayed['cuda_graph_exec_identity'] = 'UNVERIFIED_NOT_TORCH_CUDAGRAPH'
    else:
        captured['continuity']['serial'] = True
    joined = observation.correlate_trace(native_events, records)
    assert joined['replay_launches'][0]['capture_binding'] == 'UNVERIFIED_CAPTURE_BINDING'
    assert joined['replay_launches'][0]['kernel_trace_events'] == [2, 3]
    assert joined['physical_kernel_attribution'] == 'UNVERIFIED_NO_CAPTURE_NODE_BINDING'
    assert joined['G2'] == 'UNVERIFIED'


def offline_fixture(config, tmp_path):
    _, initial = startup_fixture(config, tmp_path)
    config['physical_observation'] = dict(trace_bytes=8192, trace_events=100)
    root = tmp_path/'detached'
    for node_rank in (0, 1):
        directory = root/f'node_{node_rank}'
        (directory/'workers').mkdir(parents=True)
        (directory/'node.json').write_text(json.dumps(dict(node_rank=node_rank,
            generation=config['generation'], workload_deadline=20.0,
            observer_output=f'/native/node_{node_rank}/workers')))
    responses = []
    for native_row in initial:
        row = profiler_response(native_row)
        worker_directory = root/f'node_{native_row["node_rank"]}'/'workers'
        identity = row['worker_identity']
        (worker_directory/f'worker_{row["rank"]}_{row["pid"]}.json').write_text(json.dumps(identity))
        records = []
        for window in row['profiler_observation']['windows']:
            label = window['label']
            lifecycle = 'capture' if label == 'capture' else 'execute'
            name = observation.annotation_name(lifecycle, 1, 0, 'scope')
            events = [dict(ph='X', cat='user_annotation', name=name, pid=row['pid'], tid=3,
                           ts=1.0, dur=1.0)]
            payload = trace_payload(events)
            trace = window['trace']
            trace.update(path=f'/native/node_{native_row["node_rank"]}/workers/'+Path(trace['path']).name,
                bytes_seen=len(payload), retained_bytes=len(payload), events=1,
                annotations=1, graph_launches=0, kernels=0, sha256=hashlib.sha256(payload).hexdigest())
            (worker_directory/Path(trace['path']).name).write_bytes(payload)
            common = dict(rank=row['rank'], local_rank=row['local_rank'], pid=row['pid'], worker_identity=identity)
            records.append(dict(common, scope='capture_cpu_metadata_only_not_G2', source=dict(execution.SOURCE),
                source_pinned_model_observation=dict(status='OBSERVED_CPU_METADATA_ONLY_NOT_G2',
                                                     lifecycle=lifecycle, sequence=1, events=[])))
            records.append(dict(common, scope='native_profiler_completion_only_not_G2',
                observer_sha256=config['profiler_observer_sha256'], window=copy.deepcopy(window), errors=[]))
        (worker_directory/f'request_rank_{row["rank"]}_pid_{row["pid"]}.jsonl').write_text(
            '\n'.join(json.dumps(record) for record in records)+'\n')
        responses.append(row)
    (root/'node_0'/'all_rank_snapshots.json').write_text(json.dumps(dict(workers=initial)))
    (root/'node_0'/'profiler_stop.json').write_text(json.dumps(dict(workers=responses)))
    return root, responses


def test_detached_profiler_analysis_reads_actual_trace_hashes_without_native_execution(config, tmp_path):
    root, _ = offline_fixture(config, tmp_path)
    result = caller.analyze_profiler(config, root)
    assert sorted(row['rank'] for row in result['workers']) == list(range(16))
    assert result['status'] == 'RETAINED_NATIVE_PROFILER_ANALYSIS_ONLY_NOT_G2'
    assert all(len(row['windows']) == 2 for row in result['workers'])
    assert all(window['correlation']['status'] == 'UNVERIFIED' for row in result['workers'] for window in row['windows'])
    assert result['G2'] == 'UNVERIFIED'


@pytest.mark.parametrize('fault', ['trace_bytes', 'trace_hash', 'worker_identity', 'node_deadline',
    'trace_node_path', 'missing_trace', 'duplicate_completion', 'record_source', 'record_identity'])
def test_detached_analysis_rejects_replaced_trace_and_receipt_artifacts(config, tmp_path, fault):
    root, responses = offline_fixture(config, tmp_path)
    row = responses[0]
    directory = root/'node_0'/'workers'
    trace_path = directory/Path(row['profiler_observation']['windows'][0]['trace']['path']).name
    records_path = directory/f'request_rank_{row["rank"]}_pid_{row["pid"]}.jsonl'
    if fault in ('trace_bytes', 'trace_hash'):
        payload = trace_path.read_bytes()
        trace_path.write_bytes(payload+b' ' if fault == 'trace_bytes' else payload.replace(b'1.0', b'2.0', 1))
    elif fault == 'worker_identity':
        path = directory/f'worker_{row["rank"]}_{row["pid"]}.json'
        path.write_text(json.dumps(dict(row['worker_identity'], start_time=999)))
    elif fault in ('node_deadline', 'trace_node_path'):
        path = root/'node_0'/'node.json'
        node = json.loads(path.read_bytes())
        node.update(workload_deadline=19.0) if fault == 'node_deadline' else node.update(observer_output='/other')
        path.write_text(json.dumps(node))
    elif fault == 'missing_trace':
        trace_path.rename(trace_path.with_suffix('.missing'))
    else:
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        if fault == 'duplicate_completion':
            records.append(records[1])
        elif fault == 'record_source':
            records[0]['source']['observer_sha256'] = 'f'*64
        else:
            records[0]['pid'] += 1
        records_path.write_text('\n'.join(json.dumps(record) for record in records)+'\n')
    with pytest.raises(ValueError):
        caller.analyze_profiler(config, root)


@pytest.mark.parametrize('fault', ['none', 'generate', 'stop_rpc', 'stop_receipt', 'final_rpc',
                                  'generate_and_cleanup_interrupt', 'cleanup_system_exit'])
def test_caller_stops_once_before_final_receipt_and_keeps_native_exception(config, tmp_path, monkeypatch, fault):
    nodes, initial = startup_fixture(config, tmp_path)
    config['physical_observation'] = dict(trace_bytes=8192, trace_events=100)
    llm = CpuLLM(initial, [])
    calls, publications = [], []
    original_rpc, original_generate = llm.collective_rpc, llm.generate
    native_error = RuntimeError('exact original native failure')
    capacity_calls = [0]

    def rpc(method, timeout, args=()):
        if method == 'physical_profile':
            calls.append(('start' if args[1] else 'stop', timeout))
            if not args[1] and fault == 'stop_rpc':
                raise OSError('original profiler stop failure')
            if not args[1] and fault == 'generate_and_cleanup_interrupt':
                raise KeyboardInterrupt('observer cleanup interrupt')
            if not args[1] and fault == 'cleanup_system_exit':
                raise SystemExit('observer cleanup exit')
            return [profiler_response(row, started=args[1]) for row in initial]
        if method == 'capacity_snapshot':
            capacity_calls[0] += 1
            if fault == 'final_rpc' and capacity_calls[0] == 2:
                raise native_error
        return original_rpc(method, timeout)

    def generate(*args, **kwargs):
        if fault in ('generate', 'generate_and_cleanup_interrupt'):
            raise native_error
        return original_generate(*args, **kwargs)

    llm.collective_rpc, llm.generate = rpc, generate
    original_save = caller.capacity.save
    saves = []

    def save(path, value):
        saves.append((path.name, copy.deepcopy(value)))
        if fault == 'stop_receipt' and path.name == 'profiler_stop.json':
            raise OSError('original stop receipt write failure')
        return original_save(path, value)

    monkeypatch.setattr(caller.capacity, 'save', save)
    kwargs = dict(llm_factory=lambda **kwargs: llm, sampling_factory=CpuSampling, verify=lambda *args: None,
        clock=lambda: 10, publish=lambda *args: publications.append(args))
    if fault in ('generate', 'final_rpc', 'generate_and_cleanup_interrupt'):
        with pytest.raises(RuntimeError) as caught:
            caller.execute_leader(config, nodes[0], {}, **kwargs)
        assert caught.value is native_error
    else:
        assert caller.execute_leader(config, nodes[0], {}, **kwargs) is llm
    assert [kind for kind, _ in calls] == ['start', 'stop']
    receipt = next(value for name, value in reversed(saves) if name == 'request_evidence.json')
    if fault == 'none':
        assert receipt['result'] == 'PASS' and receipt['completed_requests'] == ['A', 'B']
    else:
        assert receipt['result'] == 'FAIL' and publications[-1][2] == 'FAIL'
        if fault in ('stop_rpc', 'stop_receipt', 'generate_and_cleanup_interrupt', 'cleanup_system_exit'):
            assert receipt['evidence_errors']
        if fault in ('generate', 'final_rpc', 'generate_and_cleanup_interrupt'):
            assert str(native_error) in receipt['failure']['traceback']


@pytest.mark.parametrize('fault', ['clock', 'record', 'reservation'])
def test_stop_finalizes_original_window_when_auxiliary_finalization_fails(output, monkeypatch, fault):
    worker, observed, wrapper = profiler_case(output)
    start_window(observed, wrapper, 'capture')
    observed.trace_ready(wrapper.profiler)
    observed.current['native_stop'] = 'ORIGINAL_STOP_RETURNED'
    wrapper.is_running = wrapper._active = False
    failure = RuntimeError('original finalization '+fault+' failure')

    def fail(*args, **kwargs):
        raise failure

    if fault == 'clock':
        observed.clock = fail
    elif fault == 'record':
        observed.record = fail
    else:
        monkeypatch.setattr(execution, 'reserve_device_bytes', fail)
    observed.after_stop()
    assert observed.current is None and len(observed.windows) == 1
    assert observed.windows[0]['status'] == 'UNVERIFIED'
    assert observed.errors and worker._request_observation.unverified
    assert observed.summary()['G2'] == 'UNVERIFIED'


@pytest.mark.parametrize('category', ['cuda_runtime', 'cuda_driver'])
def test_shared_correlation_with_other_native_api_is_ambiguous(category):
    native_events, records = correlation_case()
    native_events.append(dict(ph='X', cat=category, name='cudaLaunchKernel', pid=7, tid=3,
                              ts=12.0, dur=1.0, args=dict(correlation=41)))
    joined = observation.correlate_trace(native_events, records)
    assert joined['status'] == 'UNVERIFIED' and joined['replay_launches'] == []
    assert len(joined['ambiguous_launches']) == 1 and joined['unmatched_kernels'] == 2


@pytest.mark.parametrize('fault', ['null_launch_args', 'null_kernel_args', 'null_name', 'scalar_event'])
def test_malformed_native_trace_is_rejected_before_correlation(fault):
    native_events, records = correlation_case()
    if fault == 'null_launch_args':
        native_events[1]['args'] = None
    elif fault == 'null_kernel_args':
        native_events[2]['args'] = None
    elif fault == 'null_name':
        native_events[0]['name'] = None
    else:
        native_events.append(3)
    with pytest.raises(ValueError, match='Native trace'):
        observation.correlate_trace(native_events, records)


@pytest.mark.parametrize('fault', ['none', 'container', 'getter', 'zero', 'bool', 'substituted_profiler'])
def test_graph_executable_identity_uses_only_the_original_read_getter(output, monkeypatch, fault):
    worker, observed, wrapper = profiler_case(output)
    observed.cupti = CpuCupti()
    calls = []

    class Graph:
        def raw_cuda_graph_exec(self):
            calls.append('read_original_handle')
            if fault == 'getter':
                raise RuntimeError('original getter failure')
            return 0 if fault == 'zero' else True if fault == 'bool' else 1234

        def __getattr__(self, name):
            raise AssertionError('No native graph mutation or CUDA operation is allowed: '+name)

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph)))
    graph = object() if fault == 'container' else Graph()
    if fault == 'substituted_profiler':
        worker.profiler = object()
    if fault in ('getter', 'zero', 'bool', 'substituted_profiler'):
        with pytest.raises((ValueError, RuntimeError)):
            observed.graph_identity(graph)
    else:
        result = observed.graph_identity(graph)
        assert result['cuda_graph_exec_handle'] == (None if fault == 'container' else 1234)
        assert result['cuda_graph_exec_identity'] == ('UNVERIFIED_NOT_TORCH_CUDAGRAPH' if fault == 'container'
                                                    else 'OBSERVED_READ_ONLY_HANDLE_NOT_NODE_BINDING')
    assert calls == ([] if fault in ('container', 'substituted_profiler') else ['read_original_handle'])
    assert observed.summary()['G2'] == 'UNVERIFIED'


@pytest.mark.parametrize('fault', ['none', 'preparation', 'native_acquire', 'native_enter',
    'native_body', 'native_exit', 'start_summary', 'stop_summary', 'stop_summary_and_native_body'])
def test_original_capture_context_keeps_native_outcome_on_auxiliary_control_failure(output, monkeypatch, fault):
    from contextlib import contextmanager

    worker, observed, wrapper = profiler_case(output)
    calls = []
    native_error = RuntimeError('exact original native capture failure')
    auxiliary_error = MemoryError('native profiler control summary unavailable')

    def original_context(self):
        calls.append('native_acquire')
        if fault == 'native_acquire':
            raise native_error

        @contextmanager
        def context():
            calls.append('native_enter')
            if fault == 'native_enter':
                raise native_error
            try:
                yield
            finally:
                calls.append('native_exit')
                if fault == 'native_exit':
                    raise native_error
        return context()

    def profile(self, *, is_start=True, **kwargs):
        calls.append('native_start' if is_start else 'native_stop')
        if is_start:
            wrapper.is_running = wrapper._active = True
            observed.current['native_start'] = 'ORIGINAL_START_RETURNED'
        else:
            observed.trace_ready(wrapper.profiler)
            observed.current['native_stop'] = 'ORIGINAL_STOP_RETURNED'
            wrapper.is_running = wrapper._active = False

    base = type('NativeWorker', (), dict(profile=profile, _get_cudagraph_capture_context=original_context,
        execute_model=lambda self, *args, **kwargs: None,
        **{stage: lambda self: None for stage in execution.capacity.STAGES}))
    actual = execution.observed_worker(base)()
    actual.__dict__.update(worker.__dict__)
    actual.model_runner = SimpleNamespace()
    actual.profiler = None
    actual._request_observation.profiler = None
    observed.worker = actual

    def prepare(native_worker, state, record):
        assert native_worker is actual and state is actual._request_observation
        calls.append('observer_prepare')
        if fault == 'preparation':
            raise auxiliary_error
        actual.profiler = wrapper
        return observed

    original_summary = observed.summary
    def summary():
        if (fault == 'start_summary' and wrapper.is_running or
                fault in ('stop_summary', 'stop_summary_and_native_body') and not wrapper.is_running):
            raise auxiliary_error
        return original_summary()

    monkeypatch.setenv('STEP5_PHYSICAL_OBSERVATION', '{}')
    monkeypatch.setattr(observation, 'create_observation', prepare)
    monkeypatch.setattr(observed, 'summary', summary)
    expected_error = fault in ('native_acquire', 'native_enter', 'native_body', 'native_exit',
                               'stop_summary_and_native_body')
    result = object()
    def capture():
        with actual._get_cudagraph_capture_context():
            calls.append('native_body')
            if fault in ('native_body', 'stop_summary_and_native_body'):
                raise native_error
            return result
    if expected_error:
        with pytest.raises(RuntimeError) as caught:
            capture()
        assert caught.value is native_error
    else:
        assert capture() is result
    assert calls.count('native_acquire') == 1
    assert calls.count('native_enter') == (fault != 'native_acquire')
    assert calls.count('native_body') == (fault not in ('native_acquire', 'native_enter'))
    assert calls.count('native_exit') == (fault not in ('native_acquire', 'native_enter'))
    if fault not in ('native_acquire', 'preparation'):
        assert calls.count('native_start') == calls.count('native_stop') == 1
        assert observed.current is None and len(observed.windows) == 1
    if fault in ('preparation', 'start_summary', 'stop_summary', 'stop_summary_and_native_body'):
        assert actual._request_observation.unverified
        if fault != 'preparation':
            assert observed.windows[0]['status'] == 'UNVERIFIED' and observed.errors


class CpuCupti:
    """Exercise the installed getters' C argument widths without calling CUDA."""
    def __init__(self):
        self.calls, self.failure, self.duplicate = [], False, False

    def read(self, name, handle, output):
        self.calls.append((name, handle.value, type(output._obj)))
        output._obj.value = 9 if self.duplicate else handle.value + 100
        return 1 if self.failure else 0

    def cuptiGetGraphId(self, handle, output):
        return self.read('graph', handle, output)

    def cuptiGetGraphNodeId(self, handle, output):
        return self.read('node', handle, output)

    def cuptiGetGraphExecId(self, handle, output):
        return self.read('exec', handle, output)


class CpuGraphRuntime:
    """CUDA metadata responses only; any unlisted API use fails the CPU test."""
    class cudaStreamCaptureStatus(IntEnum):
        cudaStreamCaptureStatusActive = 1

    class NodeType(IntEnum):
        cudaGraphNodeTypeKernel = 0
        cudaGraphNodeTypeGraph = 4

    @staticmethod
    def cudaStream_t(*, init_value):
        return init_value

    def __init__(self):
        self.nodes, self.calls = [11], []
        self.capture, self.graph, self.status = 1, 2, self.cudaStreamCaptureStatus(1)
        self.failure, self.after_fill = None, None

    def cudaStreamGetCaptureInfo(self, stream):
        self.calls.append(('capture', stream))
        return (1 if self.failure == 'capture' else 0, self.status,
                self.capture, self.graph, [], [], 0)

    def cudaGraphGetNodes(self, graph, numNodes=0):
        self.calls.append(('nodes', graph, numNodes))
        nodes, count = list(self.nodes[:numNodes]), len(self.nodes)
        if numNodes and self.after_fill is not None:
            self.after_fill()
        return 1 if self.failure == 'nodes' else 0, nodes, count

    def cudaGraphNodeGetType(self, node):
        self.calls.append(('type', node))
        return 1 if self.failure == 'type' else 0, self.NodeType(node % 2 * 4)


def capture_api_case(output, monkeypatch):
    worker, observed, wrapper = profiler_case(output)
    start_window(observed, wrapper, 'capture')
    observed.runtime, observed.cupti = CpuGraphRuntime(), CpuCupti()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        current_stream=lambda: SimpleNamespace(cuda_stream=7))))
    return worker, observed, observed.runtime, observed.cupti


def test_capture_records_only_additions_and_cupti_widths_while_graph_is_live(output, monkeypatch):
    _, observed, runtime, cupti = capture_api_case(output, monkeypatch)
    previous, first = observed.capture_nodes(None)
    assert first['baseline'] and first['cupti_graph_id'] == 102
    assert first['nodes'] == [dict(cupti_node_id=111, node_type=4, node_type_name='cudaGraphNodeTypeGraph')]
    runtime.nodes = [13, 11, 12]
    current, second = observed.capture_nodes(previous)
    assert current is previous and not second['baseline']
    assert {row['cupti_node_id'] for row in second['nodes']} == {112, 113}
    assert current['nodes'] == {11: 111, 12: 112, 13: 113}
    _, third = observed.capture_nodes(current)
    assert third['nodes'] == []
    assert second['replay_node_mapping'] == 'UNVERIFIED'
    assert second['gpu_completion'] == 'UNMEASURED'
    assert cupti.calls == [('graph', 2, ctypes.c_uint32), ('node', 11, ctypes.c_uint64),
                           ('node', 13, ctypes.c_uint64), ('node', 12, ctypes.c_uint64)]
    assert observed.cupti_id('cuptiGetGraphExecId', 8, ctypes.c_uint32) == 108
    assert cupti.calls[-1] == ('exec', 8, ctypes.c_uint32)
    assert observed.summary()['G2'] == 'UNVERIFIED'


@pytest.mark.parametrize('fault', ['inactive', 'capture_changed', 'graph_changed', 'disappeared',
    'duplicate_handle', 'null_handle', 'event_limit', 'count_changed', 'capture_api', 'nodes_api',
    'type_api', 'cupti_api', 'duplicate_id', 'deadline', 'late_api', 'profiler_changed'])
def test_invalid_capture_evidence_never_dereferences_ended_or_unadmitted_nodes(output, monkeypatch, fault):
    worker, observed, runtime, cupti = capture_api_case(output, monkeypatch)
    previous, _ = observed.capture_nodes(None)
    before = copy.deepcopy(previous)
    runtime.calls.clear()
    cupti.calls.clear()
    runtime.nodes.append(12)
    if fault == 'inactive':
        runtime.status = 0
    elif fault == 'capture_changed':
        runtime.capture += 1
    elif fault == 'graph_changed':
        runtime.graph += 1
    elif fault == 'disappeared':
        runtime.nodes = [12]
    elif fault == 'duplicate_handle':
        runtime.nodes = [11, 11]
    elif fault == 'null_handle':
        runtime.nodes.append(0)
    elif fault == 'event_limit':
        observed.limits = observation.ProfilerObservationLimits(8192, 1)
    elif fault == 'count_changed':
        runtime.after_fill = lambda: runtime.nodes.append(13)
    elif fault in ('capture_api', 'nodes_api', 'type_api'):
        runtime.failure = fault.split('_')[0]
    elif fault == 'cupti_api':
        cupti.failure = True
    elif fault == 'duplicate_id':
        runtime.nodes.append(13)
        cupti.duplicate = True
    elif fault == 'deadline':
        observed.deadline = 0
    elif fault == 'late_api':
        runtime.after_fill = lambda: setattr(observed, 'deadline', 0)
    else:
        worker.profiler = object()
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        observed.capture_nodes(previous)
    assert previous == before
    if fault in ('inactive', 'capture_changed', 'graph_changed', 'capture_api', 'deadline', 'profiler_changed'):
        assert not any(call[0] == 'nodes' for call in runtime.calls)
    if fault == 'event_limit':
        assert not any(call[0] == 'nodes' and call[2] for call in runtime.calls)
    if fault not in ('type_api', 'cupti_api', 'duplicate_id'):
        assert cupti.calls == []


@pytest.mark.parametrize('fault', ['none', 'api', 'bytes', 'native'])
def test_capture_interval_hooks_preserve_native_calls_and_clear_live_handles(census, monkeypatch, fault):
    observed, _, records = census_runner(census, 'FULL')
    annotations, stack = annotate_runner(observed)
    snapshots, scopes = [], []
    original_scope = model.ObservationScope.capture_nodes
    native_failure = RuntimeError('unchanged original capture failure')

    def record_scope(scope, *args, **kwargs):
        if not scopes or scopes[-1] is not scope:
            scopes.append(scope)
        return original_scope(scope, *args, **kwargs)

    monkeypatch.setattr(model.ObservationScope, 'capture_nodes', record_scope)

    def capture_nodes(previous):
        snapshots.append(previous)
        if fault == 'api':
            raise RuntimeError('unavailable original CUDA API')
        if fault == 'native':
            observed.model.layers[1].failure = native_failure
        values = dict(nodes=[dict(cupti_node_id=len(snapshots))], baseline=previous is None)
        if fault == 'bytes':
            values['nodes'][0]['oversized'] = 'x' * (observed._model_input_observation.limits.scope_bytes+1)
        return dict(nodes={len(snapshots): 100}), values

    observed._step5_profiler_observation.capture_nodes = capture_nodes
    previous_hooks = sys.getprofile(), sys.gettrace()
    if fault == 'native':
        with pytest.raises(RuntimeError) as caught:
            observed.capture_model()
        assert caught.value is native_failure
    else:
        observed.capture_model()
    assert previous_hooks == (sys.getprofile(), sys.gettrace()) and stack == []
    assert observed.model.calls == 2
    assert scopes and all(scope.capture_node_state is None for scope in scopes)
    if fault == 'none':
        rows = records[-1]['events']
        intervals = [row for row in rows if row['kind'] == 'capture_node_interval']
        assert intervals[0]['boundary'] == 'capture_begin' and intervals[-1]['boundary'] == 'capture_end'
        calls = {row['event']: row for row in rows if row['kind'] == 'native_call'}
        assert all(row['call_event'] is None or row['call_event'] in calls for row in intervals)
        assert all(row['capture_binding_event'] == intervals[0]['capture_binding_event'] for row in intervals)
        assert snapshots[0] is None and all(value is not None for value in snapshots[1:])
        assert records[-1]['status'] == 'OBSERVED_CPU_METADATA_ONLY_NOT_G2'
    elif fault in ('api', 'bytes'):
        assert observed._model_input_observation.stopped
        assert records[-1]['status'] == 'UNVERIFIED'


@pytest.mark.parametrize('value', [None, True, 0, 16])
def test_correlation_requires_matching_positive_cupti_executable_id(value):
    native_events, records = correlation_case()
    records[1]['events'][0]['cupti_graph_exec_id'] = value
    joined = observation.correlate_trace(native_events, records)
    assert joined['replay_launches'][0]['capture_binding'] == 'UNVERIFIED_CAPTURE_BINDING'
    assert joined['physical_kernel_attribution'] == 'UNVERIFIED_NO_CAPTURE_NODE_BINDING'


@pytest.mark.parametrize('key', ['cuda_runtime', 'cupti'])
def test_graph_api_source_mismatch_rejects_before_loading_cupti(output, monkeypatch, tmp_path, key):
    from types import ModuleType
    _, observed, _ = profiler_case(output)
    sources = copy.deepcopy(observed.sources)
    paths = {}
    for name in ('cuda_runtime', 'cupti'):
        relative, _ = observation.SOURCE_PINS[name]
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        paths[name] = path
        monkeypatch.setitem(observation.SOURCE_PINS, name,
            (relative, hashlib.sha256(name.encode()).hexdigest()))
    paths[key].write_bytes(b'wrong installed bytes')
    cuda, bindings = ModuleType('cuda'), ModuleType('cuda.bindings')
    cuda.bindings = bindings
    bindings.runtime = SimpleNamespace(__file__=str(paths['cuda_runtime']))
    monkeypatch.setitem(sys.modules, 'cuda', cuda)
    monkeypatch.setitem(sys.modules, 'cuda.bindings', bindings)
    monkeypatch.setattr(observation.importlib.metadata, 'distribution',
        lambda name: SimpleNamespace(locate_file=lambda relative: tmp_path/relative))
    loaded = []
    monkeypatch.setattr(observation.ctypes, 'CDLL', lambda path: loaded.append(path))
    with pytest.raises(ValueError, match='locked source: '+key):
        observed.bind_graph_api()
    assert loaded == [] and observed.runtime is None and observed.cupti is None
    assert all(observed.sources[path] == value for path, value in sources.items())


def node_correlation_case():
    native_events, records = correlation_case()
    stored = records[0]['events'][0]
    stored.update(event=20, capture_binding_event=1)
    common = dict(kind='capture_node_interval', capture_binding_event=1, capture_id=1,
                  graph_handle=16, stream_handle=7, cupti_graph_id=4)
    records[0]['events'] = [
        dict(kind='capture_forward', event=1, owner_id=stored['owner_id'], native_capture_region=True),
        dict(common, event=2, boundary='capture_begin', baseline=True, total_nodes=0, call_event=None, nodes=[]),
        dict(kind='native_call', event=3, capture_binding_event=1, site='triton_gemm', layer_id=8, owner_id=42),
        dict(common, event=4, boundary='call_return', baseline=False, total_nodes=2, call_event=3,
             nodes=[dict(cupti_node_id=(4 << 32)+index, node_type=0) for index in (0, 1)]),
        dict(common, event=5, boundary='capture_end', baseline=False, total_nodes=2, call_event=None, nodes=[]), stored]
    for index, kernel in enumerate(native_events[2:]):
        kernel['args'].update({'graph id': 15, 'graph node id': (15 << 32)+index})
    return native_events, records


def test_correlation_uses_native_node_ids_and_keeps_interval_ownership_limit():
    native_events, records = node_correlation_case()
    result = observation.correlate_trace(native_events, records)
    nodes = result['replay_launches'][0]['capture_node_correlation']
    assert nodes['status'] == 'OBSERVED_CAPTURE_INTERVAL_KERNEL_CORRELATION'
    assert nodes['ownership'] == 'HOST_CALL_INTERVAL_ONLY_NOT_EXCLUSIVE_GPU_OWNERSHIP'
    assert [row['cupti_replay_node_id'] for row in nodes['kernels']] == [(15 << 32), (15 << 32)+1]
    assert [row['cupti_capture_node_id'] for row in nodes['kernels']] == [(4 << 32), (4 << 32)+1]
    assert all(row['call_event'] == 3 and row['layer_id'] == 8 for row in nodes['kernels'])
    assert result['G2'] == 'UNVERIFIED'
    assert result['status'] == 'OBSERVED_REPLAY_AND_CAPTURE_INTERVAL_CORRELATION'
    assert result['physical_kernel_attribution'] == 'UNVERIFIED_EXCLUSIVE_NODE_OWNERSHIP'


@pytest.mark.parametrize('fault', ['missing_begin', 'missing_end', 'other_forward', 'wrong_owner',
    'count', 'duplicate_node', 'wrong_encoding', 'graph_changed', 'sequence', 'unknown_call',
    'wrong_call_capture', 'graph_id', 'missing_node_id', 'repeated_kernel', 'child_graph', 'bool_type'])
def test_node_correlation_rejects_unwitnessed_or_ambiguous_mapping(fault):
    native_events, records = node_correlation_case()
    rows = records[0]['events']
    if fault == 'missing_begin':
        rows.pop(1)
    elif fault == 'missing_end':
        rows.pop(-2)
    elif fault == 'other_forward':
        rows[0]['native_capture_region'] = False
    elif fault == 'wrong_owner':
        rows[0]['owner_id'] += 1
    elif fault == 'count':
        rows[3]['total_nodes'] += 1
    elif fault == 'duplicate_node':
        rows[3]['nodes'][1]['cupti_node_id'] = rows[3]['nodes'][0]['cupti_node_id']
    elif fault == 'wrong_encoding':
        rows[3]['nodes'][0]['cupti_node_id'] = (3 << 32)
    elif fault == 'graph_changed':
        rows[3]['capture_id'] += 1
    elif fault == 'sequence':
        rows[3]['event'] = 30
    elif fault == 'unknown_call':
        rows[3]['call_event'] = 99
    elif fault == 'wrong_call_capture':
        rows[2]['capture_binding_event'] += 1
    elif fault == 'graph_id':
        native_events[2]['args']['graph id'] = 16
    elif fault == 'missing_node_id':
        native_events[2]['args'].pop('graph node id')
    elif fault == 'repeated_kernel':
        native_events[3]['args']['graph node id'] = native_events[2]['args']['graph node id']
    elif fault == 'child_graph':
        rows[3]['nodes'][0]['node_type'] = 4
    else:
        rows[3]['nodes'][0]['node_type'] = False
    if fault in ('count', 'duplicate_node', 'wrong_encoding', 'graph_changed', 'sequence',
                  'unknown_call', 'wrong_call_capture', 'bool_type'):
        with pytest.raises(ValueError):
            observation.correlate_trace(native_events, records)
    else:
        result = observation.correlate_trace(native_events, records)
        assert result['replay_launches'][0]['capture_node_correlation']['status'] == 'UNVERIFIED'
        assert result['G2'] == 'UNVERIFIED'
