"""CPU-only native-worker observation contracts; no runtime qualification."""

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SOURCE = Path(__file__).parents[1] / 'comparison/calibration/native_full_structure.py'


def load_collector():
    spec = importlib.util.spec_from_file_location('native_serving_fixture', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native(monkeypatch, tmp_path):
    state = SimpleNamespace(mode='NONE', replay=False, forward_error=None,
        replay_error=None, context_error=None, events=[], result=object())
    descriptor = SimpleNamespace(num_tokens=6144, num_reqs=1, uniform=False,
        has_lora=False, num_active_loras=0)
    context = SimpleNamespace(cudagraph_runtime_mode=SimpleNamespace(name='NONE'),
        batch_descriptor=descriptor)

    def get_context():
        if state.context_error:
            raise state.context_error
        context.cudagraph_runtime_mode.name = state.mode
        return context

    class CUDAGraph:
        def replay(self):
            state.events.append('replay')
            if state.replay_error:
                raise state.replay_error
            return state.result

    class Model:
        def __call__(self, **kwargs):
            state.events.append('model')
            if state.forward_error:
                raise state.forward_error
            if state.replay:
                CUDAGraph().replay()
            return state.result

    class GPUModelRunner:
        def __init__(self):
            self.model = Model()
            self.cudagraph_dispatcher = SimpleNamespace(
                cudagraph_mode=SimpleNamespace(name='FULL_DECODE_ONLY'), keys_initialized=True)

        def _model_forward(self, **kwargs):
            return self.model(**kwargs)

        def _dummy_run(self, num_tokens, is_profile=False, is_graph_capturing=False):
            if num_tokens == 0:
                return state.result
            return self.model(input_ids=SimpleNamespace(shape=(num_tokens,)),
                positions=SimpleNamespace(shape=(num_tokens,)), inputs_embeds=None)

    class Worker:
        def __init__(self):
            self.rank, self.device = 7, 'cuda:0'
            self.use_v2_model_runner = False
            self.model_runner = GPUModelRunner()
            self.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(
                cudagraph_mode=SimpleNamespace(name='FULL_DECODE_ONLY'),
                cudagraph_capture_sizes=[1, 8], max_cudagraph_capture_size=8))
            self.noop = False

        def load_model(self, *, load_dummy_weights=False):
            return state.result

        def compile_or_warm_up_model(self):
            self.model_runner._dummy_run(8)
            state.events.append('warmup_return')
            return state.result

        def execute_model(self, scheduler_output):
            if self.noop:
                return state.result
            if self.use_v2_model_runner:
                return self.model_runner.execute_model(scheduler_output)
            if scheduler_output.total_num_scheduled_tokens == 0:
                return self.model_runner._dummy_run(1)
            return self.model_runner._model_forward(
                input_ids=SimpleNamespace(shape=(6144,)),
                positions=SimpleNamespace(shape=(6144,)), inputs_embeds=None)

        def shutdown(self):
            state.events.append('shutdown_native')
            state.shutdown_snapshot = json.loads((tmp_path / 'serving/rank7.json').read_text())
            if getattr(state, 'shutdown_error', None):
                raise state.shutdown_error
            return state.result

    for name in ('vllm', 'vllm.v1', 'vllm.v1.worker', 'vllm.forward_context',
                 'vllm.v1.worker.gpu_worker', 'vllm.v1.worker.gpu_model_runner', 'torch'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['vllm.v1.worker.gpu_worker'].Worker = Worker
    sys.modules['vllm.v1.worker.gpu_model_runner'].GPUModelRunner = GPUModelRunner
    sys.modules['vllm.forward_context'].get_forward_context = get_context
    sys.modules['torch'].cuda = SimpleNamespace(CUDAGraph=CUDAGraph,
        get_device_properties=lambda device: SimpleNamespace(name='CPU fake GPU', uuid='GPU-fake'),
        get_device_capability=lambda device: (9, 0))
    monkeypatch.setenv('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR', str(tmp_path))
    monkeypatch.setenv('FRONTIER_NATIVE_RUN_GENERATION', 'generation-test')
    monkeypatch.setenv('FRONTIER_NATIVE_ROLE', 'prefill')
    module = load_collector()
    monkeypatch.setattr(module.FullModelInventory, 'full_model_inventory',
        lambda worker: json.dumps(dict(worker_rank=worker.rank, groups={'world': {'rank': 7}})))
    return SimpleNamespace(module=module, observer=module._NATIVE_SERVING_INVENTORY,
        worker=Worker(), state=state, graph=CUDAGraph, runner=GPUModelRunner,
        directory=tmp_path, context=context)


def scheduled(phase='formal', repeat=1, index=0, tokens=6144):
    name = f'cmpl-fixed-6144-1024-req128-qps1-{phase}-r{index:03d}-repeat{repeat:02d}-0'
    return SimpleNamespace(num_scheduled_tokens={name: tokens}, total_num_scheduled_tokens=tokens)


def test_no_environment_leaves_standalone_import_and_native_types_untouched(monkeypatch):
    monkeypatch.delenv('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR', raising=False)
    module = load_collector()
    assert module._NATIVE_SERVING_INVENTORY is None
    assert module._install_native_serving_inventory() is None
    assert set(vars(module.FullModelInventory)) - {
        '__module__', '__dict__', '__weakref__', '__doc__'} == {'full_model_inventory'}


def test_import_installs_once_without_extension_native_name_collisions(native):
    worker = native.worker
    original = type(worker).execute_model
    again = load_collector()
    assert again._NATIVE_SERVING_INVENTORY is native.observer
    assert type(worker).execute_model is original
    assert len(native.observer.hooks) == 7
    extension = native.module.FullModelInventory
    assert not hasattr(type(worker), 'full_model_inventory')
    assert '__init_subclass__' not in vars(extension)
    assert native.observer.generation == 'generation-test'


def test_post_warmup_and_pre_shutdown_inventory_and_original_return(native):
    worker, state = native.worker, native.state
    assert worker.compile_or_warm_up_model() is state.result
    path = native.directory / 'serving/rank7.json'
    first = json.loads(path.read_text())
    assert first['schema'] == 'native_serving_v1'
    assert first['stage'] == 'post_warmup'
    assert first['run_generation'] == 'generation-test'
    assert first['role'] == 'prefill'
    assert first['inventory']['device_identity']['uuid'] == 'GPU-fake'
    assert first['inventory']['device_identity']['capability'] == [9, 0]
    assert first['inventory']['device_identity']['pid'] > 0
    assert first['inventory']['effective_graph']['max_capture_size'] == 8
    assert first['inventory']['model_runner'].endswith('.GPUModelRunner')
    assert first['counters']['successful_forwards'] == 1
    assert state.events[-1] == 'warmup_return'
    assert worker.execute_model(scheduled()) is state.result
    assert worker.shutdown() is state.result
    assert state.shutdown_snapshot['stage'] == 'pre_shutdown'
    assert state.shutdown_snapshot['counters']['successful_forwards'] == 2
    assert not list(path.parent.glob('*.tmp'))


def test_none_serving_forward_does_not_require_unsupported_replay(native):
    native.worker.execute_model(scheduled())
    native.worker.execute_model(scheduled('warmup', 2))
    observer = native.observer
    assert observer.forwards == 2 and observer.replays == 0
    rows = list(observer.rows.values())
    assert {(r['repeat'], r['phase'], r['graph_mode']) for r in rows} == {
        (1, 'formal', 'NONE'), (2, 'warmup', 'NONE')}
    assert all(r['kind'] == 'serving' and r['successful_replays'] == 0 for r in rows)
    assert rows[0]['batch_descriptor']['num_tokens'] == 6144
    assert rows[0]['input_shapes']['input_ids'] == [6144]
    assert rows[0]['scheduled_num_tokens'] == 6144
    assert not observer.errors


def test_actual_replay_success_and_mode_are_separate_from_scheduling(native):
    native.state.mode, native.state.replay = 'FULL', True
    native.context.batch_descriptor.num_tokens = 8
    native.worker.noop = True
    native.worker.execute_model(scheduled())
    assert native.observer.forwards == native.observer.replays == 0
    native.worker.noop = False
    native.worker.execute_model(scheduled())
    rows = list(native.observer.rows.values())
    assert len(rows) == 1
    assert rows[0]['successful_forwards'] == rows[0]['successful_replays'] == 1
    assert rows[0]['graph_mode'] == 'FULL'
    assert rows[0]['batch_descriptor']['num_tokens'] == 8
    assert native.state.events == ['model', 'replay']


@pytest.mark.parametrize('failure', ['forward_error', 'replay_error'])
def test_native_exceptions_preserved_and_failed_calls_not_counted(native, failure):
    error = RuntimeError('native error')
    setattr(native.state, failure, error)
    native.state.replay = True
    with pytest.raises(RuntimeError) as caught:
        native.worker.execute_model(scheduled())
    assert caught.value is error
    assert native.observer.forwards == native.observer.replays == 0
    assert native.observer.local.context is None
    assert native.observer.local.shapes is None


def test_profile_dummy_capture_count_actual_calls_not_dummy_method_returns(native):
    runner, observer = native.worker.model_runner, native.observer
    runner._dummy_run(0)
    assert observer.forwards == 0
    runner._dummy_run(8, is_profile=True)
    runner._dummy_run(8, is_graph_capturing=True)
    runner._dummy_run(8)
    native.worker.execute_model(SimpleNamespace(num_scheduled_tokens={}, total_num_scheduled_tokens=0))
    assert observer.forwards == 4
    assert {row['kind'] for row in observer.rows.values()} == {'profile', 'capture', 'dummy'}
    assert not observer.requests
    assert all(row['repeat'] is None and row['phase'] is None for row in observer.rows.values())
    assert not observer.errors


@pytest.mark.parametrize('fault', ['malformed', 'mixed', 'range', 'total', 'negative'])
def test_invalid_scheduler_provenance_is_unverified_not_guessed(native, fault):
    output = scheduled()
    if fault == 'malformed':
        output.num_scheduled_tokens = {'not-a-canonical-id': 6144}
    elif fault == 'mixed':
        output.num_scheduled_tokens.update(scheduled('warmup').num_scheduled_tokens)
        output.total_num_scheduled_tokens *= 2
    elif fault == 'range':
        output = scheduled(index=128)
    elif fault == 'total':
        output.total_num_scheduled_tokens += 1
    elif fault == 'negative':
        output.num_scheduled_tokens = {'bad': -1}
    assert native.worker.execute_model(output) is native.state.result
    assert native.observer.forwards == 1
    assert list(native.observer.rows.values())[0]['kind'] == 'unattributed'
    assert not native.observer.requests
    assert native.observer.errors[0]['where'] == 'scheduler_provenance'


def test_metadata_and_dump_failures_do_not_change_native_results(native, monkeypatch):
    native.state.context_error = ValueError('context observation failure')
    assert native.worker.execute_model(scheduled()) is native.state.result
    assert native.observer.forwards == 1
    assert not native.observer.rows
    native.worker.compile_or_warm_up_model()

    def fail(*args, **kwargs):
        raise OSError('dump failed')

    monkeypatch.setattr(native.observer, 'emit', fail)
    assert native.worker.compile_or_warm_up_model() is native.state.result
    assert native.worker.shutdown() is native.state.result
    assert {'forward_context', 'post_warmup_dump', 'pre_shutdown_dump'} <= {
        row['where'] for row in native.observer.errors}


def test_pre_shutdown_emission_happens_before_native_exception(native):
    native.worker.compile_or_warm_up_model()
    native.state.shutdown_error = RuntimeError('native shutdown failure')
    with pytest.raises(RuntimeError) as caught:
        native.worker.shutdown()
    assert caught.value is native.state.shutdown_error
    assert native.state.shutdown_snapshot['stage'] == 'pre_shutdown'


def test_native_warmup_failure_is_unchanged_and_does_not_emit(native):
    error = RuntimeError('native warmup failed')
    native.state.forward_error = error
    with pytest.raises(RuntimeError) as caught:
        native.worker.compile_or_warm_up_model()
    assert caught.value is error
    assert not (native.directory / 'serving').exists()
    assert native.observer.forwards == 0


def test_inventory_failure_marks_emission_unverified_without_blocking(native, monkeypatch):
    def fail(worker):
        raise ValueError('inventory failed')

    monkeypatch.setattr(native.observer, 'collect_inventory', fail)
    assert native.worker.compile_or_warm_up_model() is native.state.result
    row = json.loads((native.directory / 'serving/rank7.json').read_text())
    assert row['inventory'] is None
    assert row['observation_errors'][0]['where'] == 'inventory'
    assert row['counters']['successful_forwards'] == 1


def test_replay_original_return_preserved_without_inventing_a_serving_phase(native):
    assert native.graph().replay() is native.state.result
    assert native.observer.replays == 1
    row = list(native.observer.rows.values())[0]
    assert row['kind'] == 'unattributed'
    assert row['phase'] is None and row['repeat'] is None


def test_counter_observation_failure_does_not_change_native_forward(native, monkeypatch):
    def fail(*args):
        raise ValueError('counter failed')

    monkeypatch.setattr(native.observer, 'count', fail)
    assert native.worker.execute_model(scheduled()) is native.state.result
    assert native.observer.errors[0]['where'] == 'forward_counter'


@pytest.mark.parametrize('signal', [KeyboardInterrupt, SystemExit])
def test_native_control_signals_are_not_suppressed_during_observation(native, signal):
    error = signal('native control signal')

    def interrupted():
        raise error

    with pytest.raises(signal) as caught:
        native.observer.observe('interrupted', interrupted)
    assert caught.value is error


def test_missing_provenance_and_bounded_observation_errors(native):
    observer = native.module.NativeServingInventory(native.directory, '', 'unknown')
    assert observer.errors[0]['where'] == 'provenance'
    for index in range(observer.MAX_ERRORS + 10):
        observer.error('test', ValueError(str(index)))
    assert len(observer.errors) == observer.MAX_ERRORS
    observer.MAX_ROWS = 1
    native.worker.execute_model(scheduled())
    for phase in ('warmup', 'formal'):
        row = dict(kind='serving', phase=phase)
        observer.count('forward', row)
    assert observer.forwards == 2
    assert observer.dropped == 1
    assert len(observer.rows) == 1


def test_standalone_request_contract_matches_canonical_generator(native):
    from tests.integration.canonical_case import build_fixed_manifest as canonical
    observer = native.observer
    assert observer.REQUEST_NAMESPACE == canonical.NAMESPACE
    assert observer.REQUEST_LIMITS == {
        'formal': canonical.NUM_REQUESTS, 'warmup': canonical.WARMUP_REQUESTS}
    assert observer.REPEATS == tuple(range(1, canonical.REPETITIONS + 1))


@pytest.fixture
def pinned_native():
    """Read the optional pinned checkout without importing its GPU dependencies."""
    try:
        git = subprocess.run(
            ['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
            cwd=SOURCE.parents[3], text=True, capture_output=True, check=False)
    except FileNotFoundError:
        pytest.skip('git is required to locate the external pinned native checkout')
    if git.returncode:
        pytest.skip('a git common directory is required for the external native checkout')
    checkout = (Path(git.stdout.strip()).parent / '.real-engine' / 'vLLM-BS'
        / '.worktrees' / 'glm-h5-clean-2cf0a69')
    if not checkout.is_dir():
        pytest.skip('external pinned native checkout is not available')
    revision = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
        text=True, capture_output=True, check=True)
    assert revision.stdout.strip() == '2cf0a6915ce544dc493a0990f2ea38d81601128a'
    return checkout


def native_class(checkout, relative_path, name):
    path = checkout / relative_path
    classes = [node for node in ast.parse(path.read_text(), filename=str(path)).body
        if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(classes) == 1, f'{relative_path} does not define exactly one {name}'
    return classes[0]


def native_method(cls, name):
    methods = [node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    assert len(methods) == 1, f'{cls.name} does not define exactly one {name}'
    return methods[0]


def test_pinned_native_collector_imports_and_hook_signatures(pinned_native):
    """Catch invented class imports even when the behavioural fixture agrees."""
    classes = {
        'vllm.v1.worker.gpu_worker': native_class(pinned_native,
            'vllm/v1/worker/gpu_worker.py', 'Worker'),
        'vllm.v1.worker.gpu_model_runner': native_class(pinned_native,
            'vllm/v1/worker/gpu_model_runner.py', 'GPUModelRunner'),
        'vllm.v1.worker.gpu.cudagraph_utils': native_class(pinned_native,
            'vllm/v1/worker/gpu/cudagraph_utils.py', 'ModelCudaGraphManager'),
    }
    collector = ast.parse(SOURCE.read_text())
    imports = {node.module: node.names for node in ast.walk(collector)
        if isinstance(node, ast.ImportFrom) and node.module in classes}
    assert set(imports) == set(classes)
    for module, aliases in imports.items():
        assert [(alias.name, alias.asname) for alias in aliases] == [(classes[module].name, None)]
    by_name = {cls.name: cls for cls in classes.values()}
    by_name['cls'] = native_class(pinned_native,
        'vllm/v1/worker/gpu/model_runner.py', 'GPUModelRunner')
    hooks = {(node.args[0].id, node.args[1].value) for node in ast.walk(collector)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'install' and isinstance(node.args[0], ast.Name)}
    assert hooks == {('Worker', 'execute_model'), ('Worker', 'shutdown'),
        ('Worker', 'compile_or_warm_up_model'), ('GPUModelRunner', '_model_forward'),
        ('GPUModelRunner', '_dummy_run'), ('Worker', 'load_model'),
        ('ModelCudaGraphManager', 'capture'), ('ModelCudaGraphManager', 'run_fullgraph'),
        ('ModelCudaGraphManager', 'run_pw_graph'),
        ('cls', 'execute_model')}
    for owner, name in hooks:
        cls = by_name[owner]
        if (owner, name) == ('ModelCudaGraphManager', 'run_pw_graph'):
            assert [ast.unparse(base) for base in cls.bases] == ['CudaGraphManager']
            cls = native_class(pinned_native,
                'vllm/v1/worker/gpu/cudagraph_utils.py', 'CudaGraphManager')
        assert native_method(cls, name).args.args[0].arg == 'self'
    worker, runner = by_name['Worker'], by_name['GPUModelRunner']
    assert any(isinstance(base, ast.Name) and base.id == 'WorkerBase' for base in worker.bases)
    for name, arguments in (
            ('compile_or_warm_up_model', ['self']), ('shutdown', ['self']),
            ('execute_model', ['self', 'scheduler_output'])):
        method = native_method(worker, name)
        assert [arg.arg for arg in method.args.args] == arguments
        assert not method.args.kwonlyargs and method.args.kwarg is None
    forward = native_method(runner, '_model_forward')
    assert [arg.arg for arg in forward.args.args] == [
        'self', 'input_ids', 'positions', 'intermediate_tensors', 'inputs_embeds']
    assert forward.args.kwarg.arg == 'model_kwargs'
    assert [ast.literal_eval(value) for value in forward.args.defaults] == [None] * 4
    assert any(isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == 'self.model' for node in ast.walk(forward))
    dummy = native_method(runner, '_dummy_run')
    assert [arg.arg for arg in dummy.args.args] == [
        'self', 'num_tokens', 'cudagraph_runtime_mode', 'force_attention',
        'uniform_decode', 'allow_microbatching', 'skip_eplb', 'is_profile',
        'create_mixed_batch', 'remove_lora', 'is_graph_capturing',
        'num_active_loras', 'profile_seq_lens', 'randomize_inputs']
    defaults = dict(zip([arg.arg for arg in dummy.args.args][-len(dummy.args.defaults):],
        dummy.args.defaults))
    assert ast.literal_eval(defaults['is_profile']) is False
    assert ast.literal_eval(defaults['is_graph_capturing']) is False
    assert any(isinstance(node, ast.Call) and ast.unparse(node.func) == 'self.model'
        for node in ast.walk(dummy))


def source_function(method, namespace):
    """Execute only the inspected function body, with no native module import."""
    method = copy.deepcopy(method)
    method.decorator_list = []
    future = ast.parse('from __future__ import annotations').body[0]
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    exec(compile(module, '<pinned-native-function>', 'exec'), namespace)
    return namespace[method.name]


@pytest.mark.parametrize('randomization_disabled', [False, True])
@pytest.mark.parametrize('header_override', [False, True])
def test_pinned_native_transport_ids_bind_actual_phase(
        pinned_native, monkeypatch, tmp_path, randomization_disabled, header_override):
    base = native_method(native_class(pinned_native,
        'vllm/entrypoints/serve/engine/serving.py', 'BaseServing'), '_base_request_id')
    serving = native_class(pinned_native,
        'vllm/entrypoints/openai/completion/serving.py', 'OpenAIServingCompletion')
    completion = native_method(serving, '_create_completion')
    assert any(isinstance(node, ast.Call) and ast.unparse(node.func) == 'self._create_completion'
        for node in ast.walk(native_method(serving, 'create_completion')))
    assign = native_method(native_class(pinned_native,
        'vllm/v1/engine/input_processor.py', 'InputProcessor'), 'assign_request_id')
    add = native_method(native_class(pinned_native,
        'vllm/v1/engine/async_llm.py', 'AsyncLLM'), 'add_request')
    assert any(isinstance(node, ast.Call)
        and ast.unparse(node.func) == 'self.input_processor.assign_request_id'
        and [ast.unparse(arg) for arg in node.args] == ['request'] for node in ast.walk(add))
    assert any(isinstance(node, ast.Call) and ast.unparse(node.func) == 'self.engine_client.generate'
        and ast.unparse(node.args[2]) == 'request_id_item' for node in ast.walk(completion))
    uuid = '0123456789abcdef'
    namespace = dict(random_uuid=lambda: uuid,
        envs=SimpleNamespace(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=randomization_disabled),
        logger=SimpleNamespace(warning_once=lambda *args: None))
    select_id = source_function(base, namespace)
    internal_id = source_function(assign, namespace)
    body_id = 'fixed-6144-1024-req128-qps1-formal-r000-repeat01'
    header_id = 'fixed-6144-1024-req128-qps1-warmup-r003-repeat02'
    raw = SimpleNamespace(headers={'X-Request-Id': header_id} if header_override else {})
    assert select_id(None, body_id) == body_id
    assert select_id(None, None) == uuid
    assert select_id(raw, body_id) == (header_id if header_override else body_id)
    namespace.update(self=SimpleNamespace(_base_request_id=select_id), raw_request=raw,
        request=SimpleNamespace(request_id=body_id), i=0)
    for name in ('request_id', 'request_id_item'):
        assignments = [node for node in ast.walk(completion) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        assert len(assignments) == 1
        exec(compile(ast.Module(body=[assignments[0]], type_ignores=[]),
            '<pinned-completion-request-id>', 'exec'), namespace)
    request = SimpleNamespace(request_id=namespace['request_id_item'], external_req_id=None)
    internal_id(request)
    expected_base = f'cmpl-{header_id if header_override else body_id}-0'
    assert request.external_req_id == expected_base
    assert request.request_id == expected_base + ('' if randomization_disabled else '-01234567')
    monkeypatch.delenv('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR', raising=False)
    observer = load_collector().NativeServingInventory(tmp_path, 'source-test', 'decode')
    output = SimpleNamespace(num_scheduled_tokens={request.request_id: 1}, total_num_scheduled_tokens=1)
    observed = observer.scheduler_context(output)
    assert observed['kind'] == 'serving'
    assert (observed['repeat'], observed['phase']) == (
        (2, 'warmup') if header_override else (1, 'formal'))
    assert next(iter(observer.requests.values())) == {request.request_id}
    for suffix in ('-0123456', '-012345678', '-nothex!!', '-01234567-extra'):
        output.num_scheduled_tokens = {expected_base + suffix: 1}
        with pytest.raises(ValueError, match='Unrecognized native request ID'):
            observer.scheduler_context(output)
