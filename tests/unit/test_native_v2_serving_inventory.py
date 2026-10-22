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
import numpy as np

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
    uniform_token_count: int | None = None
    max_query_len: int | None = None
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
            reqs = len(getattr(scheduler_output, 'num_scheduled_tokens', {0: tokens}))
            uniform = runner_module.get_uniform_decode_token_count(reqs, tokens, tokens,
                getattr(state, 'has_prefill', tokens != 1))
            runner_module.dispatch_cg_and_sync_dp(self.cudagraph_manager, reqs, tokens,
                uniform, 2, 0, max_query_len=tokens, need_eager=is_profile)
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
    runner_module = ModuleType('vllm.v1.worker.gpu.model_runner')
    dp_module = ModuleType('vllm.v1.worker.gpu.dp_utils')
    dp_module.dist = SimpleNamespace(all_reduce=lambda *args, **kwargs: None)

    def classify(num_reqs, num_tokens, max_query_len, has_prefill):
        return max_query_len if not has_prefill and num_tokens == num_reqs * max_query_len else None

    def dispatch(cudagraph_manager, num_reqs, num_tokens, uniform_token_count, dp_size,
                 dp_rank, max_query_len=None, need_eager=False, num_active_loras=0):
        class CPUVector(np.ndarray):
            device = SimpleNamespace(type='cpu')
        vector = np.array([[num_tokens] * dp_size, [Mode[state.mode].value] * dp_size,
            [uniform_token_count or 0] * dp_size, [max_query_len or -1] * dp_size]).view(CPUVector)
        dp_module.dist.all_reduce(vector)
        return Descriptor(Mode[state.mode], num_tokens, 1, uniform_token_count,
            max_query_len), vector[0]

    runner_module.get_uniform_decode_token_count = classify
    runner_module.dispatch_cg_and_sync_dp = dispatch
    GPUModelRunner.__module__ = runner_module.__name__
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    monkeypatch.setitem(sys.modules, dp_module.__name__, dp_module)
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
    assert v2.observer.errors[0]['where'] == 'v2_forward_association'


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


def dispatch_checker():
    import importlib.util
    path = Path(__file__).parents[2] / ('task_memory/task_2026-10-22_v4_native_dispatch/'
        'check_groundtruth_run_successor_r3.py')
    spec = importlib.util.spec_from_file_location('dispatch_successor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dispatch_oracle():
    checker = dispatch_checker()
    root = os.environ.get('FRONTIER_VLLM_BS_ROOT')
    if not root:
        pytest.skip('Set FRONTIER_VLLM_BS_ROOT for source-backed dispatch controls')
    engine = dict(enforce_eager=False, tensor_parallel_size=8, data_parallel_size=2,
                  max_num_seqs=128, max_num_batched_tokens=8192)
    manifest = {'groundtruth_checkout_path': root}
    manager, _ = checker.native_v2_graph_dispatch(manifest, engine)
    return checker, engine, manager, checker.native_dispatch_oracle(manifest, manager)


def dispatch_row(oracle, prefill, peer_uniform=1, peer_tokens=1):
    import dataclasses
    import torch
    checker, engine, manager, native = oracle
    ci = dict(num_reqs=1, num_tokens=1, max_query_len=1, has_prefill=prefill)
    uniform = native['get_uniform_decode_token_count'](**ci)
    inputs = dict(num_reqs=1, num_tokens=1, uniform_token_count=uniform, dp_size=2,
                  dp_rank=0, max_query_len=1, need_eager=False, num_active_loras=0)
    desired = manager.dispatch(1, 1, uniform, 0, max_query_len=1)
    vector = [[1, peer_tokens], [desired.cg_mode.value, 2], [uniform or 0, peer_uniform], [1, 1]]
    native['dist'] = SimpleNamespace(all_reduce=lambda tensor, **kwargs:
        tensor.copy_(torch.tensor(vector, dtype=tensor.dtype)))
    desc, padded = native['dispatch_cg_and_sync_dp'](manager, **inputs)
    full = desc.cg_mode.name == 'FULL'
    return dict(scheduled_num_requests=1, scheduled_num_tokens=1, graph_mode=desc.cg_mode.name,
        successful_forwards=1, successful_replays=1,
        batch_descriptor=dict(num_tokens=desc.num_tokens, num_reqs=desc.num_reqs if full else None,
            uniform=None if full else False, has_lora=False, num_active_loras=0),
        dispatch_evidence=dict(classifier=dict(inputs=ci, uniform_token_count=uniform),
            dispatch=dict(inputs=inputs, descriptor=dataclasses.asdict(desc) | {'cg_mode': desc.cg_mode.name},
                num_tokens_across_dp=padded.tolist()), dp_post_all_reduce=vector))


@pytest.mark.parametrize('prefill,peer_uniform,peer_tokens,mode,tokens', [
    (True, 1, 1, 'PIECEWISE', 1), (False, 1, 1, 'FULL', 1),
    (False, 0, 1, 'PIECEWISE', 1), (False, 1, 2, 'FULL', 2)])
def test_actual_native_classification_and_dp_dispatch(dispatch_oracle, prefill, peer_uniform, peer_tokens, mode, tokens):
    checker, engine, manager, native = dispatch_oracle
    row = dispatch_row(dispatch_oracle, prefill, peer_uniform, peer_tokens)
    checker.verify_dispatch_row(row, engine, manager, native, 0)
    assert row['graph_mode'] == mode and row['batch_descriptor']['num_tokens'] == tokens


@pytest.mark.parametrize('fault', ['missing', 'circular', 'classifier', 'owner', 'dp_local',
    'dp_shape', 'ambiguous', 'descriptor', 'mode', 'padding', 'failed_forward', 'boolean_token', 'query_bound'])
def test_dispatch_successor_rejects_incomplete_or_wrong_operands(dispatch_oracle, fault):
    checker, engine, manager, native = dispatch_oracle
    row = dispatch_row(dispatch_oracle, fault == 'query_bound', peer_tokens=2 if fault == 'query_bound' else 1)
    ev = row['dispatch_evidence']
    if fault == 'missing': del row['dispatch_evidence']
    if fault == 'circular': del ev['classifier']['inputs']['has_prefill']
    if fault == 'classifier': ev['classifier']['uniform_token_count'] = None
    if fault == 'owner': ev['dispatch']['inputs']['dp_rank'] = 1
    if fault == 'dp_local': ev['dp_post_all_reduce'][0][0] = 2
    if fault == 'dp_shape': ev['dp_post_all_reduce'].pop()
    if fault == 'ambiguous': ev['extra_dispatch'] = ev['dispatch']
    if fault == 'descriptor': ev['dispatch']['descriptor']['num_tokens'] = 2
    if fault == 'mode': row['graph_mode'] = 'PIECEWISE'
    if fault == 'padding': ev['dispatch']['num_tokens_across_dp'] = [2, 2]
    if fault == 'failed_forward': row['successful_forwards'] = 0
    if fault == 'boolean_token': ev['dispatch']['inputs']['num_tokens'] = True
    if fault == 'query_bound':
        row['scheduled_num_tokens'] = ev['classifier']['inputs']['num_tokens'] = ev['dispatch']['inputs']['num_tokens'] = 2
        ev['dp_post_all_reduce'][0][0] = 2
    with pytest.raises((ValueError, KeyError)):
        checker.verify_dispatch_row(row, engine, manager, native, 0)


def test_frozen_r2_non_graph_predicates_and_reader_guards_remain_identical():
    checker = dispatch_checker()
    old, new = ast.parse(checker.R2_PATH.read_text()), ast.parse(checker.successor_source())
    funcs = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    before, after = funcs(old), funcs(new)
    assert {n for n in before if before[n] != after[n]} == {'native_rank_graph_evidence'}


@pytest.mark.parametrize('fault', ['snapshot', 'ambiguous', 'capacity'])
def test_dispatch_observer_errors_are_soft_and_counts_reconcile(v2, monkeypatch, fault):
    module = sys.modules['vllm.v1.worker.gpu.model_runner']
    v2.state.mode = 'FULL'
    if fault == 'snapshot': monkeypatch.setattr(v2.module.dataclasses, 'asdict', lambda obj: 1 / 0)
    if fault == 'capacity': v2.observer.MAX_ROWS = 0
    if fault == 'ambiguous':
        original = module.get_uniform_decode_token_count
        module.get_uniform_decode_token_count = lambda *a, **k: (original(*a, **k), original(*a, **k))[0]
    assert v2.worker.execute_model(scheduled(tokens=1)) is v2.state.result
    assert v2.observer.forwards == v2.observer.replays == 1
    assert v2.observer.errors
    assert v2.observer.local.dispatch_evidence is None
    assert sum(r['successful_forwards'] for r in v2.observer.rows.values()) == int(fault != 'capacity')
    assert sum(r['successful_replays'] for r in v2.observer.rows.values()) == int(fault != 'capacity')


def test_dispatch_observer_keeps_prefill_and_decode_separate(v2):
    v2.state.mode = 'FULL'
    for prefill in (True, False):
        v2.state.has_prefill = prefill
        v2.worker.execute_model(scheduled(tokens=1))
    rows = list(v2.observer.rows.values())
    assert len(rows) == 2
    assert {r['dispatch_evidence']['classifier']['inputs']['has_prefill'] for r in rows} == {True, False}
    assert sum(r['successful_forwards'] for r in rows) == v2.observer.forwards == 2
    assert sum(r['successful_replays'] for r in rows) == v2.observer.replays == 2


def test_native_cpu_dp_vector_is_copied_before_padding_and_exceptions_pass_unchanged(dispatch_oracle, monkeypatch, tmp_path):
    import torch
    _, _, manager, native = dispatch_oracle
    monkeypatch.delenv('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR', raising=False)
    observer = base.load_collector().NativeServingInventory(tmp_path, 'cpu-test', 'decode')
    namespace, dp = ModuleType('native.cpu.runner'), ModuleType('native.cpu.dp')
    for name in ('get_uniform_decode_token_count', 'dispatch_cg_and_sync_dp'):
        setattr(namespace, name, native[name])
    vector = [[1, 2], [2, 2], [1, 1], [1, 1]]
    returned = object()
    def reduce(tensor, **kwargs):
        tensor.copy_(torch.tensor(vector, dtype=tensor.dtype))
        return returned
    dp.dist = SimpleNamespace(all_reduce=reduce, untouched=returned)
    observer.install_v2_dispatch(namespace, dp)
    native['dist'] = dp.dist
    observer.local.dispatch_evidence = {'forwards': 0, 'replays': 0}
    uniform = namespace.get_uniform_decode_token_count(1, 1, 1, False)
    desc, padded = namespace.dispatch_cg_and_sync_dp(manager, 1, 1, uniform, 2, 0, max_query_len=1)
    assert desc.num_tokens == 2 and padded.tolist() == [2, 2]
    assert observer.local.dispatch_evidence['dp_post_all_reduce'] == vector
    assert dp.dist.untouched is returned and not observer.errors
    row = dispatch_row(dispatch_oracle, False, peer_tokens=2)
    row.pop('dispatch_evidence')
    row.pop('successful_forwards')
    row.pop('successful_replays')
    observer.count('replay', row)
    observer.count('forward', row)
    observer.finish_v2_dispatch()
    checker, engine, _, _ = dispatch_oracle
    checker.verify_dispatch_row(next(iter(observer.rows.values())), engine, manager, native, 0)
    error = RuntimeError('Native CPU all-reduce failed')
    def fail(*args, **kwargs):
        raise error
    observer.local.dispatch_evidence = {'forwards': 0}
    # Mutate only the CPU transport stand-in, not the installed observer.
    native_dist = SimpleNamespace(all_reduce=fail)
    namespace2, dp2 = ModuleType('native.cpu.runner2'), ModuleType('native.cpu.dp2')
    for name in ('get_uniform_decode_token_count', 'dispatch_cg_and_sync_dp'):
        setattr(namespace2, name, native[name])
    dp2.dist = native_dist
    observer.install_v2_dispatch(namespace2, dp2)
    native['dist'] = dp2.dist
    with pytest.raises(RuntimeError) as caught:
        namespace2.dispatch_cg_and_sync_dp(manager, 1, 1, 1, 2, 0, max_query_len=1)
    assert caught.value is error and 'dispatch' not in observer.local.dispatch_evidence


@pytest.mark.parametrize('fault', ['classifier', 'dispatch', 'finish'])
def test_observation_metadata_failure_never_changes_native_call(v2, monkeypatch, fault):
    v2.state.mode = 'FULL'
    if fault == 'finish':
        monkeypatch.setattr(v2.observer, 'finish_v2_dispatch', lambda: 1 / 0)
    else:
        import inspect
        bind = inspect.Signature.bind
        target = 'has_prefill' if fault == 'classifier' else 'need_eager'
        def broken(signature, *args, **kwargs):
            if target in signature.parameters:
                raise RuntimeError('Injected operand read failure')
            return bind(signature, *args, **kwargs)
        monkeypatch.setattr(inspect.Signature, 'bind', broken)
    assert v2.worker.execute_model(scheduled(tokens=1)) is v2.state.result
    assert v2.observer.forwards == v2.observer.replays == 1
    assert v2.observer.errors and v2.observer.local.dispatch_evidence is None


def test_frozen_dependency_corruption_is_rejected(tmp_path):
    checker = dispatch_checker()
    path = tmp_path / 'not_frozen_r2.py'
    path.write_text(checker.R2_PATH.read_text() + '\n# Changed dependency.\n')
    with pytest.raises(ValueError, match='Frozen R2 dependency hash mismatch'):
        checker.successor_source(path)


def test_failed_forward_after_same_key_success_remains_unverified(v2):
    v2.state.mode = 'FULL'
    v2.worker.execute_model(scheduled(tokens=1))
    v2.state.forward_error = RuntimeError('Failed outer forward on an existing aggregate key')
    with pytest.raises(RuntimeError): v2.worker.execute_model(scheduled(tokens=1))
    assert v2.observer.forwards == 1 and v2.observer.replays == 2
    assert v2.observer.errors[0]['where'] == 'v2_forward_association'
