"""Fake-only request observer tests: no native, device, caller or platform execution."""
import ast
import builtins
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import role_execution_observation as observation

TASK = HERE.parents[2]/'task_memory/task_2026-10-15_step5_preview_h800'


@pytest.fixture(autouse=True)
def fake_only(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm', 'stepmind', 'steptron'):
            raise AssertionError('CPU fake tests must not import native or platform packages')
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU fake tests must not launch children or send signals')

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(os, 'kill', forbidden)
    monkeypatch.setattr(os, 'execv', forbidden)


@pytest.fixture
def output(tmp_path, monkeypatch):
    records = tmp_path/'records'
    records.mkdir()
    node = tmp_path/'node.json'
    node.write_text(json.dumps(dict(observer_output=str(records), record_bytes=8*1024*1024)))
    monkeypatch.setenv('STEP5_CAPACITY_NODE_FILE', str(node))
    return records, node


def batch():
    return SimpleNamespace(
        req_ids=['decode', 'prefill'], num_reqs=2, num_reqs_after_padding=4,
        num_tokens=4, num_tokens_after_padding=8,
        idx_mapping_np=np.array([6, 2], dtype=np.intp),
        num_scheduled_tokens=np.array([1, 3], dtype=np.int32),
        query_start_loc_np=np.array([0, 1, 4, 4, 4], dtype=np.int32),
        num_computed_tokens_np=np.array([12, 5], dtype=np.int32),
        prefill_len_np=np.array([8, 9], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([8, 5], dtype=np.int32),
        is_prefilling_np=np.array([False, True]),
        prefill_runs_as_decode_np=np.array([False, False]))


def scheduler(total=4):
    return SimpleNamespace(total_num_scheduled_tokens=total,
                           num_scheduled_tokens={'prefill': 3, 'decode': 1} if total else {})


class FakeRunner:
    def __init__(self, input_batch):
        self.execute_model_state = SimpleNamespace(input_batch=input_batch)


class UnreadableStateRunner:
    @property
    def execute_model_state(self):
        raise AssertionError('Zero-work and failed forwards must not read post-execution state')


class AsyncSentinel:
    def get_output(self):
        raise AssertionError('Only the native executor may materialize async output')


def make_worker(output, *, result=None, failure=None, input_batch=None, record=observation.write_record,
                native_effect=None):
    input_batch = input_batch if input_batch is not None else batch()
    calls = []

    def native(self, *args, **kwargs):
        calls.append((args, kwargs))
        if failure is not None:
            raise failure
        if native_effect is not None:
            native_effect(self, *args, **kwargs)
        return result

    def sample(self, *args, **kwargs):
        self.sample_calls.append((args, kwargs))
        current_batch = self.model_runner.execute_model_state.input_batch
        current_batch.req_ids[:] = ['reused']
        for name in observation.CPU_ARRAY_FIELDS:
            value = getattr(current_batch, name)
            if value is not None:
                value[:] = 0
        self.model_runner.execute_model_state = None
        return self.sample_result

    def prohibited(self, *args, **kwargs):
        raise AssertionError('Observation cannot prepare or recompute native inputs')

    base = type('FakeNativeWorker', (), dict(
        execute_model=native, sample_tokens=sample, prepare_inputs=prohibited,
        recompute=prohibited, **{name: lambda self: 'capacity-native' for name in observation.capacity.STAGES}))
    worker = observation.observed_worker(base, record)()
    worker.rank, worker.local_rank = 10, 2
    worker._capacity_identity = dict(rank=10, local_rank=2, pid=os.getpid(), start_time=765,
        parent_pid=222, parent_start_time=99, boot_id='fake-boot', pid_namespace='pid:[123]',
        generation='fake-request', observer_sha256=observation.SOURCE['observer_sha256'])
    worker._capacity_evidence_errors = []
    worker._capacity_completed_stages = []
    worker.model_runner = FakeRunner(input_batch)
    worker.sample_calls, worker.sample_result = [], object()
    return worker, calls, base


def request_file(worker, output):
    return output[0]/f'request_rank_{worker.rank}_pid_{os.getpid()}.jsonl'


def rows(worker, output):
    return [json.loads(line) for line in request_file(worker, output).read_text().splitlines()]


@pytest.mark.parametrize('result', [object(), None, AsyncSentinel()])
@pytest.mark.parametrize('keyword', [False, True])
def test_factory_preserves_result_arguments_one_call_and_default_writer(output, result, keyword):
    worker, calls, base = make_worker(output, result=result)
    work = scheduler()
    args, kwargs = ((), {'scheduler_output': work}) if keyword else ((work,), {})
    assert worker.execute_model(*args, **kwargs) is result
    assert len(calls) == 1 and calls[0] == (args, kwargs)
    assert (calls[0][1]['scheduler_output'] if keyword else calls[0][0][0]) is work
    assert type(worker).sample_tokens is base.sample_tokens
    assert type(worker).prepare_inputs is base.prepare_inputs
    assert type(worker).recompute is base.recompute
    assert 'sample_tokens' not in type(worker).__dict__
    assert type(worker).capacity_snapshot is observation.capacity.capacity_snapshot
    row = rows(worker, output)[0]
    assert row['native_outcome'] == 'returned' and row['native_error'] is None
    assert row['invocation'] == 1 and row['prepared_batch_status'] == 'OBSERVED_CPU_METADATA'
    assert row['runner'] == __name__+'.FakeRunner'
    assert row['worker_identity'] == worker._capacity_identity
    assert (row['pid'], row['rank'], row['local_rank']) == (os.getpid(), 10, 2)
    assert row['source'] == observation.SOURCE
    assert row['source']['native_source_status'] == 'UNVERIFIED'
    assert worker.request_snapshot()['request_metadata']['records_written'] == 1
    assert worker._capacity_evidence_errors == []


def test_positive_snapshot_is_before_native_sampling_copied_and_in_batch_order(output):
    input_batch = batch()
    worker, calls, _ = make_worker(output, input_batch=input_batch)
    work = scheduler()
    worker.execute_model(work)
    before = request_file(worker, output).read_bytes()
    work.num_scheduled_tokens.clear()
    assert worker.sample_tokens(None) is worker.sample_result
    assert worker.sample_calls == [((None,), {})]
    row = rows(worker, output)[0]
    assert row['scheduler']['num_scheduled_tokens'] == [
        dict(req_id='prefill', count=3), dict(req_id='decode', count=1)]
    prepared = row['prepared_batch']
    assert prepared['req_ids'] == ['decode', 'prefill']
    assert prepared['idx_mapping_np'] == [6, 2]
    assert prepared['num_scheduled_tokens'] == [1, 3]
    assert prepared['query_start_loc_np'] == [0, 1, 4, 4, 4]
    assert prepared['num_computed_tokens_np'] == [12, 5]
    assert prepared['prefill_len_np'] == [8, 9]
    assert prepared['num_computed_prefill_tokens_np'] == [8, 5]
    assert prepared['is_prefilling_np'] == [False, True]
    assert prepared['prefill_runs_as_decode_np'] == [False, False]
    assert prepared['num_tokens'] == 4 and prepared['num_tokens_after_padding'] == 8
    assert prepared['num_reqs'] == 2 and prepared['num_reqs_after_padding'] == 4
    assert before == request_file(worker, output).read_bytes() and len(calls) == 1
    assert input_batch.req_ids == ['reused']


def test_injected_recorder_receives_immutable_serialized_cpu_snapshot(output):
    payloads = []
    worker, _, _ = make_worker(output, record=lambda worker, payload: payloads.append(payload))
    worker.execute_model(scheduler())
    worker.sample_tokens(None)
    assert isinstance(payloads[0], str)
    assert json.loads(payloads[0])['prepared_batch']['idx_mapping_np'] == [6, 2]
    assert worker.request_snapshot()['request_metadata']['metadata_snapshots'] == 1


@pytest.mark.parametrize('optional', [None, np.array([False, True])])
def test_prefill_as_decode_optional_values_not_phase_inference(output, optional):
    input_batch = batch()
    input_batch.prefill_runs_as_decode_np = optional
    worker, _, _ = make_worker(output, input_batch=input_batch)
    worker.execute_model(scheduler())
    prepared = rows(worker, output)[0]['prepared_batch']
    assert prepared['prefill_runs_as_decode_np'] == (None if optional is None else [False, True])
    assert prepared['is_prefilling_np'] == [False, True]
    assert worker.request_snapshot()['status'] == 'OBSERVED_CPU_METADATA_ONLY_NOT_G2'


def test_zero_work_return_none_never_reads_post_state(output):
    worker, calls, _ = make_worker(output)
    worker.model_runner = UnreadableStateRunner()
    assert worker.execute_model(scheduler(0)) is None
    row = rows(worker, output)[0]
    assert row['prepared_batch'] is None
    assert row['prepared_batch_status'] == 'NOT_OBSERVED_ZERO_WORK'
    assert len(calls) == 1
    assert worker.request_snapshot()['status'] == 'UNMEASURED'
    assert worker.request_snapshot()['request_metadata']['metadata_snapshots'] == 0


@pytest.mark.parametrize('failure', [ValueError('native'), KeyboardInterrupt('native'), SystemExit(7)])
def test_native_baseexception_and_original_traceback_preserved_without_state_reads(output, failure):
    worker, calls, _ = make_worker(output, failure=failure)
    worker.model_runner = UnreadableStateRunner()
    with pytest.raises(type(failure)) as caught:
        worker.execute_model(scheduler())
    assert caught.value is failure and len(calls) == 1
    trace = caught.value.__traceback__
    while trace.tb_next:
        trace = trace.tb_next
    assert trace.tb_frame.f_code.co_name == 'native'
    row = rows(worker, output)[0]
    assert row['native_outcome'] == 'raised' and 'native' in row['native_error']
    assert row['prepared_batch_status'] == 'NOT_OBSERVED_NATIVE_ERROR'
    assert row['prepared_batch'] is None and 'observer_error' not in row
    summary = worker.request_snapshot()
    assert summary['errors']['native_failures'] == 1 and summary['status'] == 'UNVERIFIED'
    assert summary['errors']['observer'] is None
    assert worker._capacity_evidence_errors == []


@pytest.mark.parametrize('record_failure', [OSError('record failed'), KeyboardInterrupt('observer interrupted')])
def test_native_baseexception_preserved_even_when_recording_and_stdout_fail(output, monkeypatch, record_failure):
    def broken(*args):
        raise record_failure

    def closed(*args, **kwargs):
        raise BrokenPipeError('closed stdout')

    monkeypatch.setattr(builtins, 'print', closed)
    original = SystemExit('original native exit')
    worker, calls, _ = make_worker(output, failure=original, record=broken)
    worker.model_runner = UnreadableStateRunner()
    with pytest.raises(SystemExit) as caught:
        worker.execute_model(scheduler())
    assert caught.value is original and len(calls) == 1
    summary = worker.request_snapshot()
    assert summary['errors']['native']['error_type'] == 'builtins.SystemExit'
    assert summary['errors']['observer']['kind'] == 'record'
    assert summary['request_metadata']['collection_stopped'] is True
    assert not request_file(worker, output).exists()
    assert worker._capacity_evidence_errors == []


@pytest.mark.parametrize('missing', ['execute_model_state', 'input_batch', 'idx_mapping_np', 'num_tokens'])
def test_missing_metadata_unverified_not_G2_and_native_remains_healthy(output, missing):
    worker, calls, _ = make_worker(output)
    if missing == 'execute_model_state':
        worker.model_runner.execute_model_state = None
    elif missing == 'input_batch':
        worker.model_runner.execute_model_state = SimpleNamespace()
    else:
        delattr(worker.model_runner.execute_model_state.input_batch, missing)
    assert worker.execute_model(scheduler()) is None
    row = rows(worker, output)[0]
    assert row['prepared_batch_status'] == 'UNVERIFIED'
    assert row['prepared_batch'] is None and 'observer_error' in row
    assert set(row['evidence_status'].values()) == {'UNMEASURED'}
    summary = worker.request_snapshot()
    assert summary['status'] == 'UNVERIFIED' and summary['errors']['native_failures'] == 0
    assert summary['request_metadata']['metadata_snapshots'] == 0
    assert len(calls) == 1 and worker._capacity_evidence_errors == []


def test_device_fields_hidden_graph_and_model_inputs_are_never_read(output):
    class CPUBatch:
        def __getattr__(self, name):
            raise AssertionError('Unapproved field read: '+name)

    input_batch = CPUBatch()
    input_batch.__dict__.update(vars(batch()))

    class CPUOnlyState:
        input_batch = None

        def __getattr__(self, name):
            raise AssertionError('Unapproved state read: '+name)

    state = CPUOnlyState()
    state.input_batch = input_batch
    worker, _, _ = make_worker(output)
    worker.model_runner.execute_model_state = state
    assert worker.execute_model(scheduler()) is None
    row = rows(worker, output)[0]
    assert row['prepared_batch_status'] == 'OBSERVED_CPU_METADATA'
    assert set(row['evidence_status'].values()) == {'UNMEASURED'}
    assert worker.request_snapshot()['evidence_status'] == row['evidence_status']


def test_non_cpu_array_rejected_without_device_copy(output):
    class DeviceArray:
        def copy(self):
            raise AssertionError('Cannot copy a device array')

        def tolist(self):
            raise AssertionError('Cannot materialize device contents')

    input_batch = batch()
    input_batch.idx_mapping_np = DeviceArray()
    worker, _, _ = make_worker(output, input_batch=input_batch)
    assert worker.execute_model(scheduler()) is None
    assert worker.request_snapshot()['status'] == 'UNVERIFIED'
    assert rows(worker, output)[0]['prepared_batch'] is None


def test_summary_bounded_detached_and_capacity_snapshot_errors_unchanged(output, monkeypatch):
    capacity_calls = []
    capacity_result = {'status': 'metadata_only_not_G2', 'resource_evidence_errors': ['capacity-error']}

    def snapshot(worker):
        capacity_calls.append(worker)
        return capacity_result

    monkeypatch.setattr(observation.capacity, 'capacity_snapshot', snapshot)
    worker, _, _ = make_worker(output)
    capacity_errors = [dict(phase='load_model', error='retained capacity error')]
    worker._capacity_evidence_errors = capacity_errors
    worker.execute_model(scheduler())
    assert worker.capacity_snapshot() is capacity_result
    assert capacity_calls == [worker]
    summary = worker.request_snapshot()
    assert summary['errors']['observer'] is None and summary['errors']['native'] is None
    assert summary['status'] == 'OBSERVED_CPU_METADATA_ONLY_NOT_G2'
    assert summary['worker_identity'] == worker._capacity_identity
    assert (summary['pid'], summary['rank'], summary['local_rank']) == (os.getpid(), 10, 2)
    assert summary['source'] == observation.SOURCE
    assert worker._capacity_evidence_errors is capacity_errors
    assert capacity_errors == [dict(phase='load_model', error='retained capacity error')]
    assert len(json.dumps(summary).encode()) < 4096
    assert 'req_ids' not in json.dumps(summary)
    summary['request_metadata']['last_batch_counts']['num_tokens'] = 999
    summary['worker_identity']['start_time'] = 999
    summary['source']['native_source_status'] = 'PASS'
    summary['request_metadata']['last_invocation']['runner'] = 'changed'
    summary['evidence_status']['G2'] = 'PASS'
    again = worker.request_snapshot()
    assert again['request_metadata']['last_batch_counts']['num_tokens'] == 4
    assert again['worker_identity']['start_time'] == 765
    assert again['source']['native_source_status'] == 'UNVERIFIED'
    assert again['request_metadata']['last_invocation']['runner'] == __name__+'.FakeRunner'
    assert again['evidence_status']['G2'] == 'UNMEASURED'
    assert capacity_result['status'] == 'metadata_only_not_G2'


def test_summary_before_execution_inert_and_unmeasured(output):
    worker, calls, _ = make_worker(output)
    assert worker.request_snapshot()['status'] == 'UNMEASURED'
    assert worker.request_snapshot()['request_metadata']['invocations'] == 0
    assert calls == [] and not request_file(worker, output).exists()
    assert not hasattr(worker, '_request_observation')


@pytest.mark.parametrize('over_by', [0, 1])
def test_exact_budget_counts_original_jsonl_identity_and_new_file(output, monkeypatch, over_by):
    monkeypatch.setattr(observation.time, 'time_ns', lambda: 123)
    captured = []
    recorder = observation.write_record

    def capture(worker, payload):
        captured.append(payload.encode())
        recorder(worker, payload)

    worker, _, _ = make_worker(output, record=capture)
    worker.execute_model(scheduler())
    previous = request_file(worker, output).read_bytes()
    inherited = output[0]/f'rank_{worker.rank}_pid_{os.getpid()}.jsonl'
    identity = output[0]/f'worker_{worker.rank}_{os.getpid()}.json'
    inherited.write_bytes(b'original-capacity-evidence\n')
    identity.write_bytes(b'{"identity":"original"}\n')
    captured.clear()
    used = len(previous)+inherited.stat().st_size+identity.stat().st_size
    # The payload differs only by the single-digit ordinal; exact bytes are known.
    next_payload = previous.replace(b'"invocation":1,', b'"invocation":2,')
    limit = used+len(next_payload)-over_by
    node = json.loads(output[1].read_text())
    node['record_bytes'] = 8*limit+7
    output[1].write_text(json.dumps(node))
    assert worker.execute_model(scheduler()) is None
    assert captured == [next_payload]
    summary = worker.request_snapshot()
    assert inherited.read_bytes() == b'original-capacity-evidence\n'
    assert identity.read_bytes() == b'{"identity":"original"}\n'
    if over_by:
        assert request_file(worker, output).read_bytes() == previous
        assert summary['status'] == 'UNVERIFIED'
        assert summary['request_metadata']['collection_stopped'] is True
    else:
        assert request_file(worker, output).read_bytes() == previous+next_payload
        assert summary['request_metadata']['rank_limit'] == limit
        assert summary['request_metadata']['rank_bytes_at_last_write'] == limit
        assert summary['request_metadata']['records_written'] == 2


def test_single_budget_excess_stops_evidence_without_stopping_healthy_native(output, monkeypatch):
    def closed(*args, **kwargs):
        raise BrokenPipeError('closed stdout')

    monkeypatch.setattr(builtins, 'print', closed)
    node = json.loads(output[1].read_text())
    node['record_bytes'] = 8
    output[1].write_text(json.dumps(node))
    result = AsyncSentinel()
    worker, calls, _ = make_worker(output, result=result)
    assert worker.execute_model(scheduler()) is result
    first = worker.request_snapshot()
    assert first['status'] == 'UNVERIFIED' and first['request_metadata']['collection_stopped']
    worker.model_runner = UnreadableStateRunner()
    assert worker.execute_model(scheduler()) is result
    second = worker.request_snapshot()
    assert len(calls) == 2 and not request_file(worker, output).exists()
    assert second['request_metadata']['invocations'] == 2
    assert second['errors']['observer'] == first['errors']['observer']
    assert worker._capacity_evidence_errors == []


def test_snapshot_observes_native_preparation_not_pre_call_state_and_preserves_extra_args(output):
    work = scheduler()
    marker = object()

    def prepare(worker, received, extra, *, enabled):
        assert received is work and extra is marker and enabled is False
        work.num_scheduled_tokens.clear()
        worker.model_runner.execute_model_state.input_batch.num_computed_tokens_np[:] = [22, 11]

    worker, calls, _ = make_worker(output, native_effect=prepare)
    assert worker.execute_model(work, marker, enabled=False) is None
    assert calls == [((work, marker), {'enabled': False})]
    row = rows(worker, output)[0]
    assert row['scheduler']['num_scheduled_tokens'] == [
        dict(req_id='prefill', count=3), dict(req_id='decode', count=1)]
    assert row['prepared_batch']['num_computed_tokens_np'] == [22, 11]


def test_disabled_collection_reads_neither_scheduler_nor_post_state(output):
    node = json.loads(output[1].read_text())
    node['record_bytes'] = 8
    output[1].write_text(json.dumps(node))
    worker, calls, _ = make_worker(output)
    worker.execute_model(scheduler())
    worker.model_runner = UnreadableStateRunner()

    class UnreadableScheduler:
        def __getattr__(self, name):
            raise AssertionError('Stopped evidence collection cannot read scheduler metadata')

    assert worker.execute_model(UnreadableScheduler()) is None
    assert len(calls) == 2
    assert worker.request_snapshot()['errors']['observer']['invocation'] == 1


def test_default_writer_exclusive_first_file_and_no_cloud_fallback(output, monkeypatch):
    worker, _, _ = make_worker(output)
    request_file(worker, output).write_bytes(b'earlier generation\n')
    assert worker.execute_model(scheduler()) is None
    assert request_file(worker, output).read_bytes() == b'earlier generation\n'
    assert worker.request_snapshot()['request_metadata']['collection_stopped'] is True
    monkeypatch.delenv('STEP5_CAPACITY_NODE_FILE')
    worker, _, _ = make_worker(output)
    assert worker.execute_model(scheduler()) is None
    assert worker.request_snapshot()['status'] == 'UNVERIFIED'
    assert worker._capacity_evidence_errors == []


def test_invocation_ordinals_and_native_errors_are_separate_from_metadata(output):
    worker, calls, _ = make_worker(output)
    worker.execute_model(scheduler(0))
    worker.execute_model(scheduler())
    retained = rows(worker, output)
    assert [row['invocation'] for row in retained] == [1, 2]
    assert [row['prepared_batch_status'] for row in retained] == [
        'NOT_OBSERVED_ZERO_WORK', 'OBSERVED_CPU_METADATA']
    assert len(calls) == 2 and worker.request_snapshot()['errors']['native_failures'] == 0
    assert worker.request_snapshot()['request_metadata']['metadata_snapshots'] == 1


def test_module_import_inert_without_torch_vllm_or_lazy_native_entry():
    spec = importlib.util.spec_from_file_location('request_observation_inert', HERE/'role_execution_observation.py')
    imported = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = imported
    spec.loader.exec_module(imported)
    assert '__getattr__' not in vars(imported)
    assert imported.SOURCE == observation.SOURCE


def test_selected_metadata_fields_exist_in_exact_retained_input_batch_class():
    source = TASK/'native_role_sources/observed/vllm/v1/worker/gpu/input_batch.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'InputBatch')
    declarations = {node.target.id: node for node in cls.body if isinstance(node, ast.AnnAssign)}
    expected = {'req_ids', *observation.CPU_ARRAY_FIELDS, *observation.BATCH_COUNT_FIELDS}
    assert expected <= declarations.keys()
    optional = declarations['prefill_runs_as_decode_np']
    assert isinstance(optional.value, ast.Constant) and optional.value.value is None
    assert 'None' in ast.unparse(optional.annotation)
    for name in set(observation.CPU_ARRAY_FIELDS)-{'prefill_runs_as_decode_np'}:
        assert 'np.ndarray' == ast.unparse(declarations[name].annotation)



def test_exhausted_worker_budget_stops_model_observation_without_stopping_native(output):
    node = json.loads(output[1].read_text())
    node['record_bytes'] = 8
    output[1].write_text(json.dumps(node))
    worker, calls, _ = make_worker(output)
    observed = SimpleNamespace(last_record=None, stopped=False, device=None,
        summary=lambda: {'collection_stopped': observed.stopped, 'native_compatibility': 'UNVERIFIED'})
    worker.model_runner._model_input_observation = observed
    worker.execute_model(scheduler())
    assert worker._request_observation.collection_stopped is True
    assert observed.stopped is True
    assert worker.execute_model(scheduler()) is None
    assert len(calls) == 2
    assert worker.request_snapshot()['source_pinned_model_observation']['collection_stopped'] is True
    assert worker.request_snapshot()['evidence_status']['G2'] == 'UNMEASURED'


def test_model_observer_setup_error_is_visible_without_changing_cpu_or_capacity_labels(output):
    worker, calls, _ = make_worker(output)
    error = dict(status='UNVERIFIED', kind='observer_setup', error_type='builtins.ValueError',
                 native_compatibility='UNVERIFIED', evidence_status={'G2': 'UNMEASURED'})
    worker.model_runner._model_input_observation_error = error
    worker.execute_model(scheduler())
    assert len(calls) == 1
    row = rows(worker, output)[0]
    assert row['prepared_batch_status'] == 'OBSERVED_CPU_METADATA'
    assert row['model_input_observer_error'] == error
    summary = worker.request_snapshot()
    assert summary['status'] == 'OBSERVED_CPU_METADATA_ONLY_NOT_G2'
    assert summary['model_input_observer_error'] == error
    summary['model_input_observer_error']['evidence_status']['G2'] = 'changed'
    assert worker.request_snapshot()['model_input_observer_error']['evidence_status']['G2'] == 'UNMEASURED'
    assert worker._capacity_evidence_errors == []


def test_prior_model_record_cannot_be_attributed_to_a_new_native_invocation(output):
    worker, calls, _ = make_worker(output)
    observed = SimpleNamespace(last_record={'lifecycle': 'capture', 'events': ['earlier']}, stopped=False, device=None)
    worker.model_runner._model_input_observation = observed
    worker.execute_model(scheduler())
    assert len(calls) == 1 and observed.last_record is None
    assert 'source_pinned_model_observation' not in rows(worker, output)[0]


@pytest.mark.parametrize('failed_drain', [False, True])
def test_complete_device_rpc_response_respects_cap_including_error_payload(output, failed_drain):
    worker, calls, _ = make_worker(output)
    worker.execute_model(scheduler())
    class Device:
        owned_bytes = 0
        limits = SimpleNamespace(rpc_bytes=256)
        stopped = False
        def summary(self):
            return dict(status='UNVERIFIED' if self.stopped else 'OBSERVED_DIAGNOSTIC_DEVICE_WITNESSES_NOT_G2')
        def drain(self, record):
            if failed_drain:
                raise ValueError('Long diagnostic: '+'x'*8192)
            return self.summary()
        def error(self, kind, failure):
            self.stopped = True
    device = Device()
    worker.model_runner._model_input_observation = SimpleNamespace(
        device=device, stopped=False, summary=lambda: dict(status='CPU_METADATA_ONLY'))
    worker._request_observation.device_rpc_bytes = device.limits.rpc_bytes
    result = worker.request_snapshot()
    assert len((json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n').encode()) <= 256
    assert result['status'] == 'UNVERIFIED'
    assert worker._request_observation.collection_stopped
    assert device.stopped
    assert len(calls) == 1 and worker._capacity_evidence_errors == []


def test_failed_preparation_reply_is_bounded_and_records_zero_allocation(output, monkeypatch):
    worker, _, _ = make_worker(output)
    worker.execute_model(scheduler())
    worker.model_runner._model_input_observation = SimpleNamespace(device=None, sites=None, limits=None)
    def fail(*args, **kwargs):
        raise ValueError('Preparation failure: '+'x'*8192)
    monkeypatch.setattr(observation.model_observation, 'DeviceObservation', fail)
    preparation = dict(limits=dict(steps=1, scan_rows=1, witness_rows=1, owned_bytes=4096, rpc_bytes=512),
                       requests=[], sampling_fields={})
    result = observation.prepare_device_observation(worker, preparation)
    assert result['status'] == 'UNVERIFIED'
    assert len((json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n').encode()) <= 512
    assert worker._request_observation.device_owned_bytes == 0
    reservation = output[0]/f'device_reservation_rank_{worker.rank}_pid_{os.getpid()}.json'
    assert json.loads(reservation.read_text())['owned_bytes'] == 0
