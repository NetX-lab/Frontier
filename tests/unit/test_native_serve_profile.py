"""Execute this worktree's collector with CPU-native-shaped calls and real launch seams."""

import importlib.util
import inspect
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.unit.test_native_serving_inventory import native, scheduled
from tests.unit.test_native_v2_serving_inventory import v2, native_function
from tests.unit.test_dp_placement_harness import native_engine

SOURCE = Path(__file__).parents[1] / 'comparison/calibration/native_serve_profile.py'


def load():
    spec = importlib.util.spec_from_file_location('profile_fixture', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == SOURCE.resolve()
    return module


@pytest.fixture
def profile(v2, monkeypatch, request):
    monkeypatch.setitem(sys.modules, 'native_full_structure', v2.module)
    monkeypatch.setenv('FRONTIER_NATIVE_PROFILE_LIMITS', json.dumps(dict(records=10000, bytes=10000000)))
    entered, stack = [], []

    @contextmanager
    def record_function(name):
        entered.append((name, tuple(stack)))
        stack.append(name)
        try:
            yield
        finally:
            stack.pop()

    torch = sys.modules['torch']
    profiler = SimpleNamespace(record_function=record_function)
    monkeypatch.setitem(sys.modules, 'torch.profiler', profiler)
    streams = [SimpleNamespace(cuda_stream=i, device_index=0, stream_id=i + 100) for i in (10, 20, 30)]
    current = [streams[0]]

    @contextmanager
    def stream(selected):
        previous, current[0] = current[0], selected
        try:
            yield
        finally:
            current[0] = previous

    class Event:
        def __init__(self):
            self.cuda_event = 0

        def record(self, stream=None):
            self.cuda_event = id(self)

        def wait(self, stream=None):
            pass

    torch.cuda.Event, torch.cuda.current_stream, torch.cuda.stream = Event, lambda: current[0], stream
    torch.cuda.is_current_stream_capturing = lambda: False
    monkeypatch.setattr(type(v2.worker), 'sample_tokens', lambda self: v2.state.result, raising=False)
    class Profiler:
        is_running = False
        iterations = 0

        def step(self):
            self.iterations += 1
            if self.iterations > v2.worker.profiler_config.max_iterations:
                self.is_running = False

    def worker_profile(worker, is_start=True, profile_prefix=None):
        if worker.profiler is None:
            worker.profiler = Profiler()
        worker.profiler.is_running = is_start
        return v2.state.result

    native_execute = type(v2.worker).execute_model

    def execute(worker, *args, **kwargs):
        if worker.profiler is not None:
            worker.profiler.step()
        return native_execute(worker, *args, **kwargs)

    monkeypatch.setattr(type(v2.worker), 'profile', worker_profile, raising=False)
    monkeypatch.setattr(type(v2.worker), 'execute_model', execute)
    v2.worker.profiler = None
    v2.worker.profiler_config = SimpleNamespace(profiler='torch', max_iterations=3)
    module = load()
    collector = module._PROFILE
    monkeypatch.setattr(collector, 'trace_stream', lambda handle: handle + 1000)
    namespaces = {}
    for path, cls, method, category, scope in module.METHODS:
        namespace = namespaces.setdefault(path, sys.modules[path] if path in (
            'vllm.v1.worker.gpu.model_runner', 'vllm.v1.worker.gpu.cudagraph_utils') else ModuleType(path))
        if cls == 'GPUModelRunner':
            namespace.GPUModelRunner = type(v2.worker.model_runner)
        if not hasattr(namespace, cls):
            setattr(namespace, cls, type(cls, (), {}))
        if not hasattr(getattr(namespace, cls), method):
            setattr(getattr(namespace, cls), method, lambda self, *args, **kwargs: v2.state.result)
        if scope == 'slot_mapping_preparation':
            setattr(getattr(namespace, cls), method, native_function(
                'vllm/v1/worker/gpu/model_runner.py', method, {}, cls))
    for path, name in module.FUNCTIONS:
        namespace = namespaces.setdefault(path, ModuleType(path))
        setattr(namespace, name, lambda *args, pdl_kwargs=None, **kwargs: v2.state.result)
    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: v2.state.result

    for path, name in module.LAUNCHES:
        setattr(namespaces.setdefault(path, ModuleType(path)), name, Kernel())
    attn_module = namespaces[module.ATTENTION_MODULE]
    breakable = ModuleType('vllm.compilation.breakable_cudagraph')

    class BreakableCUDAGraphCapture:
        @staticmethod
        def is_active():
            return (getattr(v2.observer.local, 'context', None) or {}).get('kind') == 'capture'

        def replay(self):
            return attention._prepare_and_attn()

    breakable.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
    monkeypatch.setitem(sys.modules, breakable.__name__, breakable)
    for name in ('maybe_execute_in_parallel', 'execute_in_parallel'):
        setattr(attn_module, name, native_function('vllm/utils/multi_stream_utils.py', name,
            dict(torch=torch, Any=object, Callable=object)))
    for path, namespace in namespaces.items():
        monkeypatch.setitem(sys.modules, path, namespace)
    core = namespaces[module.MODEL_MODULE].DeepseekV4Model()
    decoder = namespaces[module.MODEL_MODULE].DeepseekV4DecoderLayer()
    attention = attn_module.DeepseekV4Attention()
    core.embed_tokens = SimpleNamespace(forward=lambda: v2.state.result)
    core.norm = SimpleNamespace(forward=lambda: v2.state.result)
    type(core).forward = lambda self, *args, **kwargs: decoder.forward()
    type(decoder).forward = lambda self: attention.forward()
    type(attention).forward = lambda self: self._prepare_and_attn()
    type(attention)._prepare_and_attn = lambda self: self._run_parallel_input_projections()
    model = v2.worker.model_runner.model
    model.model = core
    model.named_modules = lambda: iter([('model', core), ('model.layers.3', decoder), ('model.layers.3.attn', attention)])
    model.modules = lambda: iter([core, decoder, attention])
    attention._prepare_and_attn_fn = attention._prepare_and_attn
    v2.worker.vllm_config.kernel_config = 'native kernel config'
    v2.worker.vllm_config.profiler_config = 'torch profiler'
    manager = v2.worker.model_runner.cudagraph_manager
    monkeypatch.setattr(type(v2.worker), 'initialize_kv_cache',
        lambda worker: setattr(worker.model_runner, 'cudagraph_manager', manager), raising=False)
    if getattr(request, 'param', None) == 'manager_later':
        v2.worker.model_runner.cudagraph_manager = None
    assert v2.worker.load_model() is v2.state.result
    assert not collector.errors
    return SimpleNamespace(module=module, collector=collector, native=v2,
        core=core, attention=attention, namespaces=namespaces, entered=entered,
        streams=streams, Event=Event, graph=BreakableCUDAGraphCapture)


@pytest.mark.parametrize('profile', ['manager_later'], indirect=True)
def test_manager_none_at_load_keeps_independent_hooks_and_facts(profile):
    p = profile
    assert p.native.worker.model_runner.cudagraph_manager is None
    assert p.collector.facts['kernel_config'] == 'native kernel config'
    assert 'BreakableCUDAGraphCapture.replay' in p.collector.hooks
    p.native.worker.initialize_kv_cache()
    p.native.state.mode = 'FULL'
    assert p.native.worker.execute_model(scheduled(tokens=1)) is p.native.state.result
    assert any(r['scope'] == 'model_dispatch_run_fullgraph' for r in p.collector.annotations)


def test_metadata_owns_native_early_slot_mapping_boundary(profile):
    p = profile
    tables = p.namespaces['vllm.v1.worker.gpu.block_table'].BlockTables()
    tables.gather_block_tables = lambda *args, **kwargs: None
    p.native.worker.model_runner.pcp_manager, p.native.worker.model_runner.block_tables = None, tables
    batch = SimpleNamespace(idx_mapping=None, query_start_loc=None, positions=None,
        num_reqs_after_padding=16, num_tokens_after_padding=16)
    assert p.native.worker.model_runner.prepare_attn(batch)[1] is p.native.state.result
    p.namespaces['vllm.v1.worker.gpu.model_states.default'].DefaultModelState().prepare_attn()
    assert [r['scope'] for r in p.collector.annotations] == [
        'slot_mapping_preparation', 'slot_mappings', 'attention_metadata']
    assert 'slot_mapping_preparation' in p.entered[1][1][0]


@pytest.mark.parametrize('path,flags', [
    ('vllm.models.common.ops.fused_qk_rmsnorm', {'launch_pdl': True}),
    ('vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant', {'USE_GDC': True, 'launch_pdl': True}),
])
def test_native_launch_arguments_are_observed_at_indexed_call(profile, path, flags):
    class Tensor:
        shape, ndim, device = (1, 8), 2, 'cpu'

        def stride(self, dim):
            return 1

        def is_contiguous(self):
            return True

        def as_strided(self, shape, strides):
            return self

    namespace = vars(profile.namespaces[path]) | dict(current_platform=SimpleNamespace(is_arch_support_pdl=lambda: True),
        torch=SimpleNamespace(empty_like=lambda tensor: Tensor(), empty=lambda *a, **k: Tensor(),
            float8_e4m3fn=object(), int32=object(), float32=object()), triton=SimpleNamespace(next_power_of_2=lambda n: n))
    function = 'fused_q_kv_rmsnorm' if 'qk_rmsnorm' in path else '_fused_inv_rope_fp8_quant_kernel_impl'
    native_call = native_function(path.replace('.', '/') + '.py', function, namespace)
    result = (native_call(Tensor(), Tensor(), Tensor(), Tensor(), 1e-6) if 'qk_rmsnorm' in path else
        native_call(Tensor(), Tensor(), Tensor(), *([1] * 12)))
    assert all(isinstance(tensor, Tensor) for tensor in result)
    assert profile.collector.annotations[-1]['launch_arguments'] == flags


def test_cuda_head512_local_import_target_is_owned(profile):
    from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import compress_norm_rope_store_cutedsl
    assert compress_norm_rope_store_cutedsl(pdl_kwargs={'launch_pdl': False}) is profile.native.state.result
    assert profile.collector.annotations[-1]['scope'] == 'compress_norm_rope_store_cutedsl'
    assert profile.collector.annotations[-1]['launch_arguments'] == {'pdl_kwargs': {'launch_pdl': False}}


def test_step_enrichment_cannot_exceed_retention_bytes(profile):
    c = profile.collector
    initial = dict(step=0, request_ids=['req'], local_num_requests=1, local_num_tokens=1)
    c.limits = dict(records=1000, bytes=len(json.dumps(initial).encode()) + 8)
    initial['dispatch_evidence'] = dict(forwards=1, replays=0,
        classifier=dict(inputs=dict(has_prefill=False)), dispatch=dict(descriptor=dict(num_tokens=16, cg_mode='FULL')))
    c.finish_step(initial)
    assert c.steps == [] and c.dropped == 1
    assert c.used <= c.limits['bytes']
    c.local.step = dict(step=1)
    c.inventory.local.dispatch_evidence = initial['dispatch_evidence']
    c.context()
    assert c.local.step == dict(step=1)
    c.limits['bytes'] = 10000
    retained = c.finish_step(initial)
    initial['dispatch_evidence']['dispatch']['descriptor']['num_tokens'] = 999
    assert retained['synchronized_descriptor']['num_tokens'] == 16
    assert c.used == len(json.dumps(retained).encode())


def test_nesting_uses_separate_parser_prefix_and_keyword_calls(profile, monkeypatch):
    p = profile
    monkeypatch.setattr(type(p.native.worker.model_runner.model), '__call__',
        lambda model, **kwargs: p.core.forward(**kwargs))
    assert p.native.worker.execute_model(scheduled(tokens=4)) is p.native.state.result
    rows = p.collector.annotations
    assert [row['scope'] for row in rows] == ['model_forward', 'decoder_layer',
        'attention_forward', 'attention_eager_segment', 'input_projections']
    assert all(row['token_phase'] == 'prefill' and row['has_prefill'] is True for row in rows)
    assert all(row['request_phase'] == 'formal' and row['repeat'] == 1 for row in rows)
    assert [len(parents) for name, parents in p.entered] == [0, 1, 2, 3, 4]
    from frontier.profiling.attention.dsa_kernel_trace import parse_annotation_name
    assert all(parse_annotation_name(row['annotation']) is None for row in rows)
    assert rows[-1]['layer'] == 3


def test_keyword_only_and_original_exception_passthrough(profile):

    class Router:
        def select_experts(self, *, hidden_states, router_logits):
            assert hidden_states is profile.native.state.result
            raise router_logits

    error = RuntimeError('native keyword-only exception')
    profile.collector.wrap(Router, 'select_experts', 'ownership', 'routing')
    assert 'hidden_states' in inspect.signature(Router.select_experts).parameters
    with pytest.raises(RuntimeError) as caught:
        Router().select_experts(hidden_states=profile.native.state.result, router_logits=error)
    assert caught.value is error
    assert not profile.collector.errors


@pytest.mark.parametrize('where', ['enter', 'exit', 'append'])
def test_annotation_failures_are_soft_with_traceback(profile, monkeypatch, where, capsys):
    class Broken:
        def __enter__(self):
            if where == 'enter':
                raise RuntimeError('instrumentation enter')

        def __exit__(self, *args):
            if where == 'exit':
                raise RuntimeError('instrumentation exit')

    monkeypatch.setattr(sys.modules['torch.profiler'], 'record_function', lambda name: Broken())
    if where == 'append':
        monkeypatch.setattr(profile.collector, 'append', lambda *args: (_ for _ in ()).throw(OSError('record failure')))
    for _ in range(100):
        assert profile.core.forward() is profile.native.state.result
    assert capsys.readouterr().err.count('NATIVE_PROFILE_UNVERIFIED') == 64
    profile.collector.emit('test')
    payload = json.loads((profile.native.directory / 'serving_profile/rank7.json').read_text())
    assert payload['status'] == 'UNVERIFIED'
    assert payload['suppressed_errors'] == 436
    assert 'Traceback' in payload['errors'][0]['traceback']


def test_actual_native_parallel_event_calls_and_generations(profile):
    p, c = profile, profile.collector
    namespace = p.namespaces[p.module.ATTENTION_MODULE]
    start, done = p.Event(), p.Event()
    marker = object()
    for _ in range(2):
        assert namespace.maybe_execute_in_parallel(fn0=lambda: marker, fn1=lambda: marker,
            event0=start, event1=done, aux_stream=p.streams[1]) == (marker, marker)
    rows = c.synchronization
    assert [r['operation'] for r in rows] == ['record', 'wait', 'record', 'wait'] * 2
    assert [r['generation'] for r in rows] == [1] * 4 + [2] * 4
    assert rows[1]['source_stream'] == '0:10' and rows[1]['destination_stream'] == '0:20'
    assert rows[3]['source_stream'] == '0:20' and rows[3]['destination_stream'] == '0:10'
    assert c.streams['0:20']['trace_stream_id'] == 1020
    before = len(rows)
    namespace.execute_in_parallel(default_fn=lambda: marker, aux_fns=[None, lambda: marker],
        start_event=start, done_events=[p.Event(), done], aux_streams=p.streams[1:], enable=True)
    assert len(rows) == before + 4
    namespace.execute_in_parallel(lambda: marker, [lambda: marker], start, [done], p.streams[1:], enable=False)
    assert len(rows) == before + 4


def test_capture_replay_eager_labels_and_launch_pdl_facts(profile):
    p, c = profile, profile.collector
    p.native.observer.local.context = dict(kind='capture')
    p.core.forward()
    assert all(r['execution'] == 'capture' for r in c.annotations)
    p.native.observer.local.context = dict(kind='serving', phase='formal', repeat=2)
    p.graph().replay()
    assert c.annotations[-1]['execution'] == 'replay_eager_segment'
    p.core.forward()
    assert c.annotations[-1]['execution'] == 'eager_serving'
    namespace = p.namespaces[p.module.COMPRESSOR_MODULE]
    namespace.save_partial_states(pdl_kwargs={'launch_pdl': False})
    assert c.annotations[-1]['launch_arguments'] == {'pdl_kwargs': {'launch_pdl': False}}
    c.emit('test')
    payload = json.loads((p.native.directory / 'serving_profile/rank7.json').read_text())
    assert 'GPU CUPTI restart' in payload['unknowns']


@pytest.mark.parametrize('tokens', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('mode', ['FULL', 'PIECEWISE'])
def test_synced_descriptor_joins_existing_dispatch_without_collective(profile, tokens, mode):
    p = profile
    p.native.state.mode, p.native.state.has_prefill = mode, False
    output = scheduled(tokens=tokens)
    if tokens == 16:
        output.num_scheduled_tokens = {next(iter(output.num_scheduled_tokens)): 1}
        output.total_num_scheduled_tokens = 1
        namespace = sys.modules['vllm.v1.worker.gpu.model_runner']
        dispatched = namespace.dispatch_cg_and_sync_dp
        namespace.dispatch_cg_and_sync_dp = lambda *a, **k: dispatched(a[0], a[1], 16, *a[3:], **k)
    p.native.worker.execute_model(output)
    step = p.collector.steps[-1]
    assert step['local_num_requests'] == 1 and step['local_num_tokens'] == (1 if tokens == 16 else tokens)
    assert step['has_prefill'] is False and step['synchronized_mode'] == mode
    assert step['synchronized_descriptor']['num_tokens'] == tokens
    inventory_row = next(iter(p.native.observer.rows.values()))
    assert step['dispatch_evidence'] == inventory_row['dispatch_evidence']
    assert 'dp_post_all_reduce' in step['dispatch_evidence']


def test_inventory_rpc_and_records_remain_unchanged(profile):
    p = profile
    assert p.module.NativeServeProfile.full_model_inventory is p.native.module.FullModelInventory.full_model_inventory
    p.native.worker.compile_or_warm_up_model()
    p.native.worker.execute_model(scheduled())
    p.native.worker.shutdown()
    payload = p.native.state.shutdown_snapshot
    assert payload['schema'] == 'native_serving_v1' and payload['stage'] == 'pre_shutdown'
    assert payload['counters']['successful_forwards'] == 2
    assert {r['kind'] for r in payload['counters']['rows']} == {'dummy', 'serving'}
    assert not payload['observation_errors']


def test_not_selected_and_capacity_leave_native_calls_unchanged(profile, monkeypatch):
    original = type(profile.native.worker).execute_model
    monkeypatch.setattr(profile.native.module, '_NATIVE_SERVING_INVENTORY', None)
    assert load()._PROFILE is None
    assert type(profile.native.worker).execute_model is original
    profile.collector.limits = dict(records=1, bytes=1)
    assert profile.core.forward() is profile.native.state.result
    assert profile.collector.dropped > 0


def test_missing_stream_mapping_retains_raw_handles_and_no_invented_edges(profile, monkeypatch):
    c = profile.collector
    monkeypatch.setattr(c, 'trace_stream', lambda handle: (_ for _ in ()).throw(OSError('CUPTI unavailable')))
    assert c.stream(profile.streams[0]) == '0:10'
    assert c.streams['0:10']['cuda_stream'] == 10 and c.streams['0:10']['trace_stream_id'] is None
    assert c.errors and not c.synchronization


def test_window_budget_survives_startup_and_tracks_native_iteration_stop(profile, monkeypatch):
    p, c, worker = profile, profile.collector, profile.native.worker
    monkeypatch.setattr(type(worker.model_runner.model), '__call__', lambda model, **kw: p.core.forward(**kw))
    c.limits = dict(records=1, bytes=100000)
    worker.execute_model(scheduled(tokens=4))
    assert c.dropped > 0
    start_step, start_annotation = c.step_index, c.sequence
    c.limits = dict(records=10000, bytes=10000000)
    assert worker.profile(profile_prefix='CPU-only-window') is p.native.state.result
    directory = p.native.directory / 'serving_profile'
    startup_path = directory / 'rank7.before_profile.json'
    startup_bytes = startup_path.read_bytes()
    startup = json.loads(startup_bytes)
    assert startup['dropped'] > 0 and c.dropped == c.used == c.records == 0
    assert c.step_index == start_step and c.sequence == start_annotation
    for _ in range(5):
        assert worker.execute_model(scheduled(tokens=4)) is p.native.state.result
    assert [step['step'] for step in c.steps] == list(range(start_step, start_step + 3))
    assert len(c.annotations) == 15 and c.dropped == 0
    assert all(row['id'] >= start_annotation for row in c.annotations)
    assert c.used == sum(len(json.dumps(row).encode()) for rows in
                        (c.steps, c.annotations, c.synchronization) for row in rows)
    assert worker.profile(is_start=False) is p.native.state.result
    frozen = (directory / 'rank7.profile.json').read_bytes()
    p.core.forward()
    worker.shutdown()
    assert (directory / 'rank7.profile.json').read_bytes() == frozen
    assert startup_path.read_bytes() == startup_bytes
    payload = json.loads(frozen)
    assert payload['window']['start_step'] == start_step and payload['window']['open'] is False
    assert payload['status'] == 'UNVERIFIED' and payload['limits'] == c.limits


@pytest.mark.parametrize('stage', ['before_profile', 'profile_stopped'])
def test_window_archive_failures_preserve_native_rpc_and_workload(profile, monkeypatch, stage):
    p, c = profile, profile.collector
    c.limits = dict(records=64, bytes=1000000)
    for _ in range(20):
        p.core.forward()
    assert c.dropped > 0
    emit = c.emit

    def broken(current, **kwargs):
        if current == stage:
            raise OSError('Injected window archive failure')
        return emit(current, **kwargs)

    monkeypatch.setattr(c, 'emit', broken)
    assert p.native.worker.profile() is p.native.state.result
    assert c.window['open'] and c.records == c.used == c.dropped == 0
    for _ in range(5):
        assert p.native.worker.execute_model(scheduled()) is p.native.state.result
    assert len(c.steps) == 3 and c.dropped == 0
    assert p.native.worker.profile(False) is p.native.state.result
    assert not c.window['open']
    if stage == 'before_profile':
        snapshot = json.loads((p.native.directory / 'serving_profile/rank7.profile.json').read_text())
        assert len(snapshot['steps']) == 3 and snapshot['dropped'] == 0
    assert any('Injected window archive failure' in row['traceback'] for row in c.errors)
    assert not p.native.observer.errors


def test_profile_rpc_preserves_original_exception_and_running_state_failure(profile, monkeypatch):
    p, c = profile, profile.collector
    error = RuntimeError('native profiler start failure')

    def fail(worker, is_start=True, profile_prefix=None):
        raise error

    cls = type('NativeProfileFailure', (), {'profile': fail})
    c.install_profile(cls)
    with pytest.raises(RuntimeError) as caught:
        cls.profile(p.native.worker)
    assert caught.value is error

    class Broken:
        @property
        def is_running(self):
            raise OSError('Injected profiler-state observer failure')

    p.native.worker.profiler = Broken()
    assert p.core.forward() is p.native.state.result
    assert c.errors[-1]['where'] == 'profile_running_state'


def test_saved_attention_alias_and_warmup_profile_dummy_capture(profile):
    p = profile
    assert p.attention._prepare_and_attn_fn() is p.native.state.result
    for kind in ('profile', 'dummy', 'capture', 'serving'):
        p.native.observer.local.context = dict(kind=kind, repeat=1, phase='warmup')
        p.native.observer.local.startup_warmup = kind == 'dummy'
        p.core.forward()
        assert p.collector.annotations[-1]['kind'] == kind
        assert p.collector.annotations[-1]['startup_warmup'] == (kind == 'dummy')


def test_real_launch_seams_turn_green_and_legacy_default_is_preserved(tmp_path):
    from tests.comparison.dp_placement_pp.vllm_replay import server_env, server_command, NATIVE_WORKER_EXTENSIONS
    from tests.comparison.calibration.pd_replay import instance_env
    from tests.comparison.calibration import native_pd_worker
    engine = native_engine() | dict(worker_extension_cls='native_serve_profile.NativeServeProfile',
        moe_gating_routing_runtime_path='balanced_topk', profiler_config=dict(profiler='torch', torch_profiler_dir='/trace'))
    env, changed = server_env('clean', engine, tmp_path, {'PATH': '/usr/bin'})
    assert env['VLLM_MOE_ROUTING_SIMULATION_STRATEGY'] == 'balanced_topk'
    engine['kv_transfer'] = dict(kv_buffer_device='cuda', side_channel_host='10.0.0.1',
        prefill_side_channel_port=6200, decode_side_channel_port=6300)
    env, changed = instance_env('clean', engine, tmp_path, {'PATH': '/usr/bin'}, '0,1,2,3,4,5,6,7', role='prefill')
    assert Path(env['PYTHONPATH']) == SOURCE.parent
    (tmp_path / 'config.json').write_text(json.dumps(dict(architectures=['DeepseekV4ForCausalLM'], vocab_size=129280)))
    command = server_command(engine, tmp_path, 8100)
    assert json.loads(command[command.index('--profiler-config') + 1]) == engine['profiler_config']
    assert command[command.index('--worker-extension-cls') + 1] in NATIVE_WORKER_EXTENSIONS
    assert 'NATIVE_WORKER_EXTENSIONS' in inspect.getsource(native_pd_worker)
    legacy, changed = server_env('clean', native_engine(), tmp_path, {'PATH': '/usr/bin'})
    assert legacy == {'PATH': '/usr/bin'} and changed == {}
