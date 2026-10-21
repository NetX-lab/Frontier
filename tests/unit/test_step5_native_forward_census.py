"""Native host-seam and fault-injection checks; no GPU qualification is implied."""

import ast
import copy
from dataclasses import dataclass
from enum import Enum
import importlib
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from tests.unit.test_step5_native_pd_r2_observation import PD, configuration, events, r2


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(PD))
    return importlib.import_module('r2_runtime')


@pytest.fixture
def host_runtime(tmp_path, monkeypatch, runtime):
    source = os.environ.get('STEP5_NATIVE_SOURCE_ROOT')
    if not source:
        pytest.skip('Requires the locked native source root')
    path = Path(source)/'vllm/v1/worker/gpu/async_utils.py'
    if not path.exists():
        path = Path(source)/'v1/worker/gpu/async_utils.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'StepTimingCollector')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'record_batch')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    timing_class = type('StepTimingCollector', (), {'record_batch': namespace['record_batch']})
    timing_class.record_batch.__module__ = __name__
    timing_class.__module__ = __name__

    class Mode(Enum):
        FULL = 2

    @dataclass
    class Descriptor:
        cg_mode: Mode = Mode.FULL
        num_tokens: int = 8
        num_reqs: int = 8

    module = ModuleType('test_native_host_runner')
    calls = []

    def dispatch(manager, num_reqs, num_tokens):
        calls.append(('dispatch', num_reqs, num_tokens))
        return Descriptor(), None

    module.dispatch_cg_and_sync_dp = dispatch
    module.StepTimingCollector = timing_class
    batch = NS(num_tokens=8, num_tokens_after_padding=8, num_reqs=8, num_reqs_after_padding=8,
        num_scheduled_tokens=np.ones(8, dtype=np.int32), num_computed_tokens_np=np.zeros(8, dtype=np.int32),
        req_ids=[f'dummy-{index}' for index in range(8)], has_prefill=False)

    class ModelState:
        def prepare_attn(self, input_batch, block_tables, slot_mappings, for_capture=False):
            calls.append(('metadata', for_capture))
            return {'layer0': object(), 'layer1': object()}

    class Runner:
        def __init__(self):
            self.timing = timing_class()
            self.model_state = ModelState()
            self.fail = False
            self.calls = 0

        def execute_model(self, *, dummy_run=False, is_profile=False):
            self.calls += 1
            module.dispatch_cg_and_sync_dp(None, 1, 1)
            self.model_state.prepare_attn(batch, (NS(shape=(8, 448)),), NS(shape=(1, 8)), for_capture=True)
            self.timing.record_batch(batch, True)
            if self.fail:
                raise LookupError('native forward failed')
            return batch

    module.GPUModelRunner = Runner
    observer = r2.Observation(configuration(tmp_path))
    observer.config['schema_source_hashes'] = {name: 'fixture' for name in (*runtime.SOURCE_MODULES, __name__)}
    monkeypatch.setattr(observer, 'verify_module', lambda *args: None)
    monkeypatch.setattr(observer, 'identity', lambda owner: dict(role='decode', dp_rank=0, tp_rank=0, engine_id='D'))
    # Only module identity is substituted; the real native record_batch body runs.
    for name in runtime.SOURCE_MODULES:
        monkeypatch.setitem(__import__('sys').modules, name, module)
    runtime.install(module, observer, r2.wrap)
    runner = Runner()
    runtime.bind_model_state(NS(model_runner=runner), observer, r2.wrap)
    return observer, runner, batch, calls


def test_native_host_boundary_retains_post_dp_width_and_metadata(host_runtime):
    observer, runner, batch, calls = host_runtime
    observer.local.worker_execution_kind = 'idle_dp'
    assert runner.execute_model(dummy_run=True) is batch
    start, end = events(observer)
    assert [start['event'], end['event']] == ['native_forward_start', 'native_forward_end']
    assert start['local_num_tokens'] == 1 and start['num_tokens_after_padding'] == 8
    assert start['query_lengths'] == [1]*8 and start['context_lengths'] == [0]*8
    assert start['metadata'] == dict(layer_count=2, layers_with_metadata=2, for_capture=True,
                                    block_table_shapes=[[8, 448]], slot_mapping_shape=[1, 8])
    assert runner.timing._batch == (True, 8, 8)
    assert calls == [('dispatch', 1, 1), ('metadata', True)] and runner.calls == 1
    assert observer.local.runtime is None and not observer.disabled


def test_startup_capture_is_not_serving_census(host_runtime):
    observer, runner, batch, _ = host_runtime
    assert runner.execute_model(dummy_run=True) is batch
    assert events(observer) == [] and runner.calls == 1


@pytest.fixture
def worker_runtime(host_runtime, monkeypatch):
    observer, runner, batch, _ = host_runtime
    batch.req_ids = [f'_warmup_{index}' for index in range(batch.num_reqs)]
    output = NS(num_scheduled_tokens=dict.fromkeys(batch.req_ids, 1), finished_req_ids=set(),
                scheduled_new_reqs=[], scheduled_cached_reqs=NS(req_ids=[]))

    class Worker:
        def __init__(self):
            self.model_runner = runner
            self.compilations = 0
            self.fail_compilation = False

        def load_model(self):
            return None

        def compile_or_warm_up_model(self):
            self.compilations += 1
            result = self.execute_model(output)
            if self.fail_compilation:
                raise LookupError('native compilation failed')
            return result

        def execute_model(self, scheduler_output):
            return runner.execute_model()

        def execute_dummy_batch(self):
            return runner.execute_model(dummy_run=True)

    module = ModuleType('vllm.v1.worker.gpu_worker')
    module.Worker = Worker
    observer.config['source_hashes'][module.__name__] = 'fixture'
    observer.config['resident_identity'] = {}
    monkeypatch.setattr(observer, 'runner_info', lambda worker: None)
    r2.instrument(module, observer)
    return observer, Worker(), runner, batch, output


@pytest.mark.parametrize('failed_compilation', [False, True])
def test_startup_worker_execute_is_excluded_without_losing_later_serving(
        worker_runtime, failed_compilation):
    observer, worker, runner, batch, output = worker_runtime
    worker.fail_compilation = failed_compilation
    assert getattr(observer.local, 'worker_execution_kind', None) is None
    if failed_compilation:
        with pytest.raises(LookupError, match='native compilation failed'):
            worker.compile_or_warm_up_model()
    else:
        assert worker.compile_or_warm_up_model() is batch
    forwards = [row for row in events(observer)
                if row['event'] in ('native_forward_start', 'native_forward_end')]
    assert forwards == [], 'startup worker calls contaminated serving census'
    assert worker.compilations == runner.calls == 1

    # The same request labels outside compilation are not a filtering rule.
    assert worker.execute_model(output) is batch
    assert worker.execute_dummy_batch() is batch
    starts = [row for row in events(observer) if row['event'] == 'native_forward_start']
    ends = [row for row in events(observer) if row['event'] == 'native_forward_end']
    assert [row['forward_index'] for row in starts] == [1, 2]
    assert [row['forward_index'] for row in ends] == [1, 2]
    assert [row['worker_execution_kind'] for row in starts] == ['scheduled', 'idle_dp']
    assert [row['dummy_run'] for row in starts] == [False, True]
    assert starts[0]['native_request_ids'] == batch.req_ids
    assert runner.calls == 3 and not observer.disabled


def test_locked_startup_method_uses_worker_execute_callback():
    source = os.environ.get('STEP5_NATIVE_SOURCE_ROOT')
    if not source:
        pytest.skip('Requires the locked native source root')
    path = Path(source) / 'v1/worker/gpu_worker.py'
    if not path.exists():
        path = Path(source) / 'vllm/v1/worker/gpu_worker.py'
    tree = ast.parse(path.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Worker')
    method = next(node for node in worker.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'compile_or_warm_up_model')
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    warmup = next(call for call in calls if isinstance(call.func, ast.Name)
                  and call.func.id == 'warmup_kernels')
    assert ast.unparse(warmup.args[1]) == 'self.execute_model'


def test_native_exception_is_not_swallowed_or_reported_as_completed(host_runtime):
    observer, runner, _, _ = host_runtime
    observer.local.worker_execution_kind = 'idle_dp'
    runner.fail = True
    with pytest.raises(LookupError, match='native forward failed'):
        runner.execute_model(dummy_run=True)
    assert [row['event'] for row in events(observer)] == ['native_forward_start']
    assert observer.local.runtime is None and runner.calls == 1


def test_observer_error_is_soft_and_native_call_still_runs_once(host_runtime, monkeypatch):
    observer, runner, batch, _ = host_runtime
    observer.local.worker_execution_kind = 'idle_dp'
    def broken(*args, **kwargs):
        raise ValueError('injected observer failure')
    monkeypatch.setattr(observer, 'emit', broken)
    assert runner.execute_model(dummy_run=True) is batch
    assert observer.disabled and runner.calls == 1 and observer.local.runtime is None


def census_rows():
    rows = []
    for role in ('prefill', 'decode'):
        for dp in range(2):
            for tp in range(8):
                rank = dict(role=role, dp_rank=dp, tp_rank=tp)
                request = f'{role}-{dp}'
                rows.append(dict(rank, event='source_compute' if role == 'prefill' else 'target_compute',
                                 request_id=request, num_scheduled_tokens=1))
                for index, dummy in ((1, False), (2, True)):
                    rows.append(dict(rank, event='native_forward_start', forward_index=index,
                        dummy_run=dummy, is_profile=False, query_lengths=[1], context_lengths=[0 if dummy else 6144],
                        native_request_ids=[f'dummy-rank-{tp}' if dummy else request], num_reqs=1,
                        num_tokens=1, num_tokens_after_padding=1, descriptor=dict(cg_mode='FULL', num_tokens=1),
                        full_cudagraph=True, observer_fence=False,
                        metadata=dict(layers_with_metadata=92, layer_count=92)))
                    rows.append(dict(rank, event='native_forward_end', forward_index=index))
    return rows


@pytest.mark.parametrize('fault', [None, 'missing_rank', 'missing_end', 'duplicate', 'wrong_width',
    'wrong_context', 'metadata_absent', 'wrong_full', 'wrong_useful', 'tp_divergence', 'fence', 'startup'])
def test_census_rejects_missing_or_contradictory_native_accounting(runtime, fault):
    verdict = importlib.import_module('r2_verdict')
    rows = census_rows()
    row = rows[1]
    if fault == 'missing_rank':
        rows = [item for item in rows if not (item['role'] == 'decode' and item['tp_rank'] == 7)]
    elif fault == 'missing_end':
        rows.pop(2)
    elif fault == 'duplicate':
        rows.append(copy.deepcopy(row))
    elif fault == 'wrong_width':
        row['descriptor']['num_tokens'] = 2
    elif fault == 'wrong_context':
        rows[3]['context_lengths'] = [1]
    elif fault == 'metadata_absent':
        row['metadata'] = None
    elif fault == 'wrong_full':
        row['full_cudagraph'] = False
    elif fault == 'wrong_useful':
        rows[0]['num_scheduled_tokens'] = 2
    elif fault == 'tp_divergence':
        row['context_lengths'] = [6145]
    elif fault == 'fence':
        row['observer_fence'] = True
    elif fault == 'startup':
        row['is_profile'] = True
    check = verdict.Check()
    runtime.judge(rows, {'engine': dict(data_parallel_size=2, tensor_parallel_size=8)}, check)
    result = check.result()
    assert (result['status'] == 'PASS') is (fault is None), result
