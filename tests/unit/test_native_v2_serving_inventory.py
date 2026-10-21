"""Exercise supported V2 call sites with native-shaped CPU types, never GPU admission."""

from dataclasses import dataclass
from enum import Enum
import __future__
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from tests.unit.test_native_serving_inventory import native, scheduled
from tests.unit import test_native_serving_inventory as base


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@dataclass(frozen=True)
class Descriptor:
    cg_mode: Mode
    num_tokens: int
    num_reqs: int
    num_active_loras: int = 0


class Tensor:
    def __init__(self, length):
        self.shape = (length,)

    def __getitem__(self, key):
        return Tensor(key.stop)


@pytest.fixture
def v2(native, monkeypatch):
    state, worker = native.state, native.worker

    class ModelCudaGraphManager:
        def __init__(self):
            self.cudagraph_mode = SimpleNamespace(name='FULL_AND_PIECEWISE')
            self._graphs_captured = True
            self.graphs = {Descriptor(Mode.FULL, 1, 1): native.graph()}

        def capture(self, *args):
            return worker.model_runner.model(input_ids=Tensor(8), positions=Tensor(8))

        def run_fullgraph(self, desc):
            native.graph().replay()
            if state.forward_error:
                raise state.forward_error
            return state.result

        def run_pw_graph(self, model, model_inputs):
            if getattr(state, 'breakable', False):
                # Native breakable graphs bypass the outer model Python call.
                native.graph().replay()
                return state.result
            return model(**model_inputs)

    class GPUModelRunner:
        def __init__(self):
            self.model = type(worker.model_runner.model)()
            self.input_buffers = SimpleNamespace(input_ids=Tensor(8192), positions=Tensor(8192))
            self.cudagraph_manager = ModelCudaGraphManager()

        def execute_model(self, scheduler_output, intermediate_tensors=None,
                          dummy_run=False, skip_attn_for_dummy_run=False,
                          is_profile=False, context_len=0):
            tokens = scheduler_output.total_num_scheduled_tokens
            if tokens == 0:
                return state.result
            if state.mode == 'FULL':
                return self.cudagraph_manager.run_fullgraph(Descriptor(Mode.FULL, tokens, 1))
            if state.mode == 'PIECEWISE':
                return self.cudagraph_manager.run_pw_graph(self.model,
                    dict(input_ids=Tensor(tokens), positions=Tensor(tokens)))
            return self.model(input_ids=Tensor(tokens), positions=Tensor(tokens))

        def _dummy_run(self, num_tokens, *, is_profile=False):
            return self.execute_model(SimpleNamespace(total_num_scheduled_tokens=num_tokens),
                dummy_run=True, is_profile=is_profile)

    module = ModuleType('vllm.v1.worker.gpu.cudagraph_utils')
    module.ModelCudaGraphManager = ModelCudaGraphManager
    monkeypatch.setitem(sys.modules, module.__name__, module)
    worker.use_v2_model_runner = True
    worker.model_runner = GPUModelRunner()
    assert worker.load_model() is state.result
    return native


@pytest.mark.parametrize('mode,replays', [('NONE', 0), ('PIECEWISE', 1), ('FULL', 1)])
def test_v2_successful_calls_return_original_and_attribute_real_ids(v2, mode, replays):
    v2.state.mode = mode
    v2.state.replay = mode == 'PIECEWISE'
    # Native FULL replay has no forward-context manager at this entry point.
    if mode == 'FULL':
        v2.state.context_error = AssertionError('No V2 FULL forward context')
    assert v2.worker.execute_model(scheduled(tokens=1 if mode == 'FULL' else 6144)) is v2.state.result
    assert v2.observer.forwards == 1
    assert v2.observer.replays == replays
    assert not v2.observer.errors
    rows = list(v2.observer.rows.values())
    assert all(row['kind'] == 'serving' and row['phase'] == 'formal' and row['repeat'] == 1 for row in rows)
    assert {row['graph_mode'] for row in rows} == {mode}
    assert {tuple(row['input_shapes']['input_ids']) for row in rows} == {(1 if mode == 'FULL' else 6144,)}
    assert v2.observer.requests[(1, 'formal')] == set(scheduled().num_scheduled_tokens)


@pytest.mark.parametrize('mode', ['NONE', 'FULL'])
def test_v2_forward_failure_preserves_native_exception_and_does_not_count(v2, mode):
    error = RuntimeError('Native forward failed')
    v2.state.mode, v2.state.forward_error = mode, error
    with pytest.raises(RuntimeError) as caught:
        v2.worker.execute_model(scheduled())
    assert caught.value is error
    assert v2.observer.forwards == 0
    # The FULL replay itself returned successfully before native output failed.
    assert v2.observer.replays == int(mode == 'FULL')


def test_v2_failed_replay_is_not_a_successful_call(v2):
    v2.state.mode = 'FULL'
    error = RuntimeError('Native replay failed')
    v2.state.replay_error = error
    with pytest.raises(RuntimeError) as caught:
        v2.worker.execute_model(scheduled(tokens=1))
    assert caught.value is error
    assert v2.observer.forwards == v2.observer.replays == 0


def test_v2_dummy_profile_capture_and_noop_are_not_serving(v2):
    runner = v2.worker.model_runner
    runner._dummy_run(8, is_profile=True)
    runner._dummy_run(8)
    runner.cudagraph_manager.capture()
    v2.worker.execute_model(SimpleNamespace(num_scheduled_tokens={}, total_num_scheduled_tokens=0))
    rows = list(v2.observer.rows.values())
    assert {row['kind'] for row in rows} == {'profile', 'dummy', 'capture'}
    assert v2.observer.forwards == 3
    assert not v2.observer.requests


def test_v2_inventory_and_ordered_emission_use_actual_manager(v2):
    assert v2.worker.compile_or_warm_up_model() is v2.state.result
    path = v2.directory / 'serving/rank7.json'
    payload = json.loads(path.read_text())
    graph = payload['inventory']['effective_graph']
    assert graph['keys_initialized'] is True
    assert graph['captured_descriptors'][0]['cg_mode'] == 'FULL'
    v2.worker.execute_model(scheduled())
    assert v2.worker.shutdown() is v2.state.result
    assert v2.state.shutdown_snapshot['stage'] == 'pre_shutdown'
    assert v2.state.shutdown_snapshot['counters']['successful_forwards'] == 2
    assert not v2.state.shutdown_snapshot['observation_errors']


def test_v2_auxiliary_failure_remains_soft_unverified(v2, monkeypatch):
    def broken(*args):
        raise OSError('Injected counter failure')
    monkeypatch.setattr(v2.observer, 'count', broken)
    assert v2.worker.execute_model(scheduled()) is v2.state.result
    assert v2.observer.errors[0]['where'] == 'forward_counter'


def test_v2_load_hook_is_idempotent(v2):
    before = list(v2.observer.hooks)
    v2.worker.load_model()
    assert v2.observer.hooks == before
    v2.worker.execute_model(scheduled())
    assert v2.observer.forwards == 1


def test_v2_breakable_piecewise_counts_outer_forward_without_model_python_call(v2):
    v2.state.mode = 'PIECEWISE'
    v2.state.breakable = True
    assert v2.worker.execute_model(scheduled()) is v2.state.result
    assert v2.observer.forwards == v2.observer.replays == 1
    assert 'model' not in v2.state.events
    assert not v2.observer.errors


NATIVE_REVISION = '2cf0a6915ce544dc493a0990f2ea38d81601128a'


def native_function(relative, name, namespace, owner=None):
    """Run a pinned native function body without importing its CUDA dependencies."""
    root = os.environ.get('FRONTIER_VLLM_BS_ROOT')
    if not root:
        pytest.skip('Set FRONTIER_VLLM_BS_ROOT to check the pinned native git objects')
    path = Path(root) / relative
    source = ast.parse(subprocess.check_output(
        ['git', '-C', root, 'show', f'{NATIVE_REVISION}:{relative}'], text=True))
    body = source.body if owner is None else next(
        node.body for node in source.body if isinstance(node, ast.ClassDef) and node.name == owner)
    node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
        str(path), 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


@pytest.fixture
def native_warmup_boundary(monkeypatch):
    """Use unedited native Worker/warmup bodies; only GPU operations/types are CPU carriers."""
    import numpy as np

    class Output(SimpleNamespace):
        @classmethod
        def make_empty(cls):
            return cls(num_scheduled_tokens={}, total_num_scheduled_tokens=0,
                scheduled_cached_reqs=None, scheduled_new_reqs=[])

        @classmethod
        def from_request(cls, request, **kwargs):
            return cls(req_id=request.req_id, **kwargs)

    namespace = dict(np=np, SchedulerOutput=Output, CachedRequestData=Output,
        NewRequestData=Output, Request=lambda req_id, *args, **kwargs: SimpleNamespace(req_id=req_id),
        SamplingParams=SimpleNamespace(for_sampler_warmup=lambda: None), GrammarOutput=SimpleNamespace,
        CompilationMode=SimpleNamespace(NONE=0, VLLM_COMPILE=3), CompilationTimes=SimpleNamespace,
        set_random_seed=lambda seed: None, kernel_warmup=lambda worker: None,
        freeze_gc_heap=lambda: None, maybe_attach_gc_debug_callback=lambda: None,
        enable_gpu_sync_check=lambda: None, set_torch_threads_for_runtime=lambda: None)
    warmup = 'vllm/v1/worker/gpu/warmup.py'
    native_function(warmup, '_warmup_block_counter', namespace)
    native_function(warmup, 'warmup_kernels', namespace)
    actual_compile = native_function('vllm/v1/worker/gpu_worker.py',
        'compile_or_warm_up_model', namespace, 'Worker')
    actual_context = native_function('vllm/forward_context.py', 'get_forward_context',
        dict(_forward_context=None))
    monitor = ModuleType('vllm.utils.jit_monitor')
    monitor.activate = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, monitor.__name__, monitor)
    original_load = base.load_collector

    def load_at_native_boundary():
        worker_class = sys.modules['vllm.v1.worker.gpu_worker'].Worker
        worker_class.compile_or_warm_up_model = actual_compile
        sys.modules['vllm.forward_context'].get_forward_context = actual_context
        namespace['torch'] = sys.modules['torch']
        namespace['torch'].accelerator = SimpleNamespace(synchronize=lambda: None)
        return original_load()

    monkeypatch.setattr(base, 'load_collector', load_at_native_boundary)
    return namespace


@pytest.fixture
def native_kernel_warmup(native_warmup_boundary, v2):
    worker, runner, state = v2.worker, v2.worker.model_runner, v2.state
    runner.is_encoder_only = runner.is_encoder_decoder = runner.is_pooling_model = False
    runner.is_last_pp_rank = True
    runner.num_speculative_steps, runner.decode_query_len = 0, 1
    runner.max_model_len = 7168
    runner.scheduler_config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=8)
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[], num_blocks=64)
    runner.model_state, runner.kv_block_zeroer = SimpleNamespace(), None
    runner.kv_connector = SimpleNamespace(set_disabled=lambda value: state.events.append(('disabled', value)))
    runner.lora_config = None
    runner.maybe_remove_all_loras = lambda config: None
    # CUDA capture/model execution and cache tensors are CPU stand-ins. The
    # original native Worker and warmup scheduling bodies are executed intact.
    runner.capture_model = lambda: 0
    runner.vllm_config = SimpleNamespace(num_lookahead_tokens=0)
    worker.compilation_config = worker.vllm_config.compilation_config
    worker.compilation_config.mode = 0
    worker.compilation_config.compilation_time = worker.compilation_config.encoder_compilation_time = 0
    worker.model_config = SimpleNamespace(enforce_eager=False, seed=0)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
    worker.observability_config = SimpleNamespace(jit_monitor_mode='warn', jit_monitor_verbose=False)
    worker.sample_tokens = lambda output: None
    runner.model_config = SimpleNamespace(get_vocab_size=lambda: 32)
    execute = worker.execute_model
    v2.canonical_execute = execute
    state.native_scheduler_outputs = []
    state.execute_threads = []

    def execute_native_output(output):
        state.native_scheduler_outputs.append(output)
        state.execute_threads.append(threading.get_ident())
        # The GPU stand-in reaches the actual FULL observation boundary on the
        # native-generated cached decode. Prefill/cache cleanup do no GPU work.
        worker.noop = output.scheduled_cached_reqs is None
        state.mode = 'FULL'
        return execute(output)

    worker.execute_model = execute_native_output
    return v2


@pytest.mark.parametrize('role', ['prefill', 'decode'])
def test_native_kernel_warmup_has_no_serving_context_or_cascading_errors(native_kernel_warmup, role):
    v2 = native_kernel_warmup
    v2.observer.role = role
    result = v2.worker.compile_or_warm_up_model()
    assert result.language_model == result.encoder == 0
    assert [list(output.num_scheduled_tokens) for output in v2.state.native_scheduler_outputs] == [
        ['_warmup_0_'], ['_warmup_0_'], []]
    assert v2.observer.forwards == v2.observer.replays == 1
    assert not v2.observer.errors
    assert not v2.observer.requests
    rows = list(v2.observer.rows.values())
    assert len(rows) == 1
    assert rows[0]['kind'] == 'dummy'
    assert rows[0]['repeat'] is rows[0]['phase'] is None
    assert set(v2.state.execute_threads) == {threading.get_ident()}
    payload = json.loads((v2.directory / 'serving/rank7.json').read_text())
    assert payload['role'] == role
    assert payload['inventory']['worker_rank'] == 7
    assert not payload['observation_errors']


def test_native_cold_warmup_then_all_canonical_phases_remain_separate(native_kernel_warmup):
    v2 = native_kernel_warmup
    v2.worker.compile_or_warm_up_model()
    v2.worker.noop = False
    for repeat in (1, 2, 3):
        for phase in ('warmup', 'formal'):
            assert v2.canonical_execute(scheduled(phase, repeat, tokens=1)) is v2.state.result
    assert v2.observer.forwards == v2.observer.replays == 7
    assert set(v2.observer.requests) == {(r, p) for r in (1, 2, 3) for p in ('warmup', 'formal')}
    rows = list(v2.observer.rows.values())
    assert len(rows) == 7
    assert len([row for row in rows if row['kind'] == 'dummy']) == 1
    assert len([row for row in rows if row['kind'] == 'serving']) == 6
    assert all('_warmup_' not in name for names in v2.observer.requests.values() for name in names)
    assert v2.observer.local.context is None
    assert v2.observer.local.startup_warmup is False
    assert not v2.observer.errors


def test_native_startup_ids_outside_proven_lifecycle_remain_unverified(native_kernel_warmup):
    v2 = native_kernel_warmup
    v2.worker.compile_or_warm_up_model()
    v2.worker.noop = False
    # Reuse the unedited native-generated decode, not a guessed ID pattern.
    assert v2.canonical_execute(v2.state.native_scheduler_outputs[1]) is v2.state.result
    assert v2.observer.forwards == v2.observer.replays == 2
    assert any(error['where'] == 'scheduler_provenance' for error in v2.observer.errors)
    assert not v2.observer.requests
    assert all(row['kind'] != 'serving' for row in v2.observer.rows.values())


def test_native_warmup_exception_is_original_and_lifecycle_restored(native_kernel_warmup):
    v2 = native_kernel_warmup
    error = RuntimeError('Native warmup output failed after successful replay')
    v2.state.forward_error = error
    with pytest.raises(RuntimeError) as caught:
        v2.worker.compile_or_warm_up_model()
    assert caught.value is error
    assert v2.observer.forwards == 0 and v2.observer.replays == 1
    assert v2.observer.local.context is None
    assert v2.observer.local.startup_warmup is False
    assert not (v2.directory / 'serving/rank7.json').exists()
    assert not v2.observer.errors


def test_native_warmup_auxiliary_dump_failure_is_soft(native_kernel_warmup, monkeypatch):
    v2 = native_kernel_warmup
    def broken(*args):
        raise OSError('Injected post-warmup write failure')
    monkeypatch.setattr(v2.observer, 'emit', broken)
    result = v2.worker.compile_or_warm_up_model()
    assert result.language_model == result.encoder == 0
    assert v2.observer.forwards == v2.observer.replays == 1
    assert v2.observer.local.context is None
    assert v2.observer.local.startup_warmup is False
    assert v2.observer.errors == [{'where': 'post_warmup_dump',
        'error': 'OSError: Injected post-warmup write failure'}]
