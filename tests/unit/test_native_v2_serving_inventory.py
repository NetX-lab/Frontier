"""Exercise supported V2 call sites with native-shaped CPU types, never GPU admission."""

from dataclasses import dataclass
from enum import Enum
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tests.unit.test_native_serving_inventory import native, scheduled


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
