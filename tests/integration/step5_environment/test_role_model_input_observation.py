"""CPU fixtures prove observer behavior only, never native vLLM compatibility."""
import ast
import builtins
from dataclasses import replace
import dis
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import role_execution_observation as execution
import role_model_input_observation as observation

TASK = HERE.parents[2]/'task_memory/task_2026-10-15_step5_preview_h800'

# This small CPU program exposes the same owned locals and call boundaries. It
# contains no native runner/graph bodies and supplies its own explicit test pins.
FAKE_SOURCE = '''from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

class Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2

@dataclass(frozen=True)
class Descriptor:
    cg_mode: Mode
    num_tokens: int = 8
    num_reqs: int = 1

class Storage:
    def data_ptr(self):
        return 1000
    def nbytes(self):
        return 8192

class Tensor:
    shape = (8, 2)
    device = 'cuda:0'
    dtype = 'torch.float32'
    def __init__(self, pointer=1008):
        self.pointer = pointer
    def untyped_storage(self):
        return Storage()
    def stride(self):
        return (2, 1)
    def storage_offset(self):
        return 2
    def data_ptr(self):
        return self.pointer
    def __getattr__(self, name):
        raise AssertionError('Device contents or synchronization are forbidden: '+name)

class Graph:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = 0
    def replay(self):
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return None

class CaptureContext:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False

class StepTimingCollector:
    def __init__(self):
        self.calls = 0
    def forward_start(self):
        self.calls += 1

class CudaGraphManager:
    def __init__(self, desc):
        self.graphs = {}
        self.capture_descs = [desc]
        self.capture_calls = 0
        self.graphs_stored = 0
    def capture(self, create_forward_fn, progress_bar_desc='fake'):
        self.capture_calls += 1
        for desc in self.capture_descs:
            forward_fn = create_forward_fn(desc, warmup=True)
            forward_fn(Mode.NONE)
            forward_fn = create_forward_fn(desc, warmup=False)
            graph = Graph()
            with CaptureContext():
                forward_fn(Mode.NONE)  # FULL_CAPTURE_FORWARD
            self.graphs[desc] = graph
            self.graphs_stored += 1  # FULL_CAPTURE_STORE
    def run_fullgraph(self, desc):
        self.graphs[desc].replay()  # FULL_REPLAY
    def release_graphs(self):
        self.graphs.clear()

class ModelCudaGraphManager(CudaGraphManager):
    def capture(self, model, model_state):
        def create_forward_fn(desc, warmup):
            model_inputs = dict(model_state)
            def forward_fn(cg_mode):
                self.forward_descriptor = desc
                self.forward_runtime_mode = cg_mode
                model(**model_inputs)
            return forward_fn
        super().capture(create_forward_fn)
    def run_fullgraph(self, desc):
        super().run_fullgraph(desc)
        return 'full-result'

@dataclass
class Entry:
    batch_descriptor: Descriptor
    cudagraph: object = None
    output: object = 'piece-result'

class CUDAGraphWrapper:
    def __init__(self, desc, runnable):
        self.descriptor, self.runnable = desc, runnable
        self.concrete_cudagraph_entries = {}
        self.captures = 0
        self.eager = False
    def __call__(self, *args, **kwargs):
        if self.eager:
            return self.runnable(*args, **kwargs)
        batch_descriptor = self.descriptor
        entry = self.concrete_cudagraph_entries[batch_descriptor]
        if entry.cudagraph is None:
            cudagraph = Graph()
            with CaptureContext():
                output = self.runnable(*args, **kwargs)  # PW_CAPTURE
            entry.output = output
            entry.cudagraph = cudagraph
            self.captures += 1  # PW_CAPTURE_STORE
            return output
        entry.cudagraph.replay()  # PW_REPLAY
        return entry.output
    def clear_graphs(self):
        self.concrete_cudagraph_entries.clear()

@dataclass
class BreakEntry:
    batch_descriptor: Descriptor
    capture: object = None
    output: object = 'break-result'

class BreakableCUDAGraphCapture:
    def __init__(self):
        self.segments = []
    def replay(self):
        for r in self.segments:
            r()  # BREAK_SEGMENT

class BreakableCUDAGraphWrapper:
    def __init__(self, desc, runnable):
        self.descriptor, self.runnable = desc, runnable
        self.entries = {}
        self.key_value = 'selected'
        self.captures = 0
    def __call__(self, *args, **kwargs):
        batch_descriptor = self.descriptor
        key = (batch_descriptor, self.key_value)
        entry = self.entries[key]
        if entry.capture is None:
            return self._capture(entry, args, kwargs)  # BREAK_CAPTURE_DISPATCH
        return self._replay(entry, args, kwargs)  # BREAK_DISPATCH
    def _capture(self, entry, args, kwargs):
        capture = BreakableCUDAGraphCapture()
        with CaptureContext():
            output = self.runnable(*args, **kwargs)  # BREAK_CAPTURE
        entry.capture = capture
        entry.output = output
        self.captures += 1  # BREAK_CAPTURE_STORE
        return output
    def _replay(self, entry, args, kwargs):
        assert entry.capture is not None
        entry.capture.replay()  # BREAK_REPLAY
        return entry.output
    def clear_graphs(self):
        self.entries.clear()

class GPUModelRunner:
    def __init__(self, vllm_config, device):
        self.vllm_config, self.device = vllm_config, device
        self.vllm_config.constructor_calls += 1
        self.step_timing = StepTimingCollector()
        self.cudagraph_manager = ModelCudaGraphManager(vllm_config.desc)
        self.input_batch = vllm_config.input_batch
        self.model_state = vllm_config.model_inputs
        self.model_calls = 0
        self.execution_calls = 0
        self.capture_calls = 0
        self.result = object()
        self.graph_wrapper = None
    def model(self, *args, **kwargs):
        self.model_calls += 1
        self.last_model_args = args
        self.last_model_kwargs = kwargs
        return self.result
    def execute_model(self, scheduler_output, *, dummy_run=False):
        self.execution_calls += 1
        input_batch = self.input_batch
        batch_desc = self.vllm_config.desc
        model_inputs = dict(self.model_state)
        self.step_timing.forward_start()  # FINAL_INPUTS
        if batch_desc.cg_mode is Mode.FULL:
            result = self.cudagraph_manager.run_fullgraph(batch_desc)
        elif batch_desc.cg_mode is Mode.PIECEWISE:
            result = self.graph_wrapper(**model_inputs)
        else:
            result = self.model(**model_inputs)
        self.execute_model_state = SimpleNamespace(input_batch=input_batch)
        return result
    def capture_model(self, *, profile_only=False):
        self.capture_calls += 1
        self.cudagraph_manager.capture(self.model, self.model_state)
        return 41
    def sample_tokens(self, *args, **kwargs):
        raise AssertionError('Observer must never sample or materialize output')

class Worker:
    def _make_model_runner(self):
        self.native_factory_calls += 1
        return GPUModelRunner(self.vllm_config, self.device)
    def execute_model(self, *args, **kwargs):
        self.worker_calls += 1
        return self.model_runner.execute_model(*args, **kwargs)
'''

MARKERS = {name.lower(): index for index, line in enumerate(FAKE_SOURCE.splitlines(), 1)
           for name in ('FINAL_INPUTS', 'FULL_REPLAY', 'FULL_CAPTURE_STORE', 'FULL_CAPTURE_FORWARD',
                        'PW_REPLAY', 'PW_CAPTURE', 'PW_CAPTURE_STORE', 'BREAK_DISPATCH',
                        'BREAK_CAPTURE_DISPATCH', 'BREAK_REPLAY', 'BREAK_SEGMENT',
                        'BREAK_CAPTURE', 'BREAK_CAPTURE_STORE')
           if '# '+name == line[line.find('# '):]}


@pytest.fixture(autouse=True)
def fake_only(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm', 'stepmind', 'steptron'):
            raise AssertionError('CPU fixtures cannot import native or platform packages')
        return original_import(name, *args, **kwargs)

    def prohibited(*args, **kwargs):
        raise AssertionError('CPU fixtures cannot launch processes, signal, or perform API operations')

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(subprocess, 'Popen', prohibited)
    monkeypatch.setattr(subprocess, 'run', prohibited)
    monkeypatch.setattr(os, 'kill', prohibited)


def input_batch():
    return SimpleNamespace(req_ids=['internal-decode', 'internal-prefill'],
        num_reqs=2, num_reqs_after_padding=4, num_tokens=4, num_tokens_after_padding=8,
        idx_mapping_np=np.array([6, 2]), num_scheduled_tokens=np.array([1, 3]),
        query_start_loc_np=np.array([0, 1, 4, 4, 4]),
        num_computed_tokens_np=np.array([12, 5]), prefill_len_np=np.array([8, 9]),
        num_computed_prefill_tokens_np=np.array([8, 5]),
        is_prefilling_np=np.array([False, True]), prefill_runs_as_decode_np=None)


@pytest.fixture
def native(tmp_path):
    path = tmp_path/'cpu_observation_fixture.py'
    path.write_text(FAKE_SOURCE)
    module = ModuleType('cpu_observation_fixture_'+str(id(path)))
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(FAKE_SOURCE, str(path), 'exec', dont_inherit=True), module.__dict__)
    for name in execution.capacity.STAGES:
        setattr(module.Worker, name, lambda self, *args, **kwargs: None)
    facade = SimpleNamespace(runner=module, graphs=module, piecewise=module,
        breakable=module, timing=module, tensor_type=module.Tensor, graph_type=module.Graph)
    pins = dict.fromkeys(observation.SOURCE_PINS, hashlib.sha256(path.read_bytes()).hexdigest())
    sites = observation.ObservationSites.bind(facade, module.Worker._make_model_runner,
                                            pins=pins, lines=MARKERS)
    yield SimpleNamespace(module=module, facade=facade, pins=pins, sites=sites, path=path)
    del sys.modules[module.__name__]


def runner(native, mode='FULL', *, limits=observation.ObservationLimits(), capture_record=None):
    module = native.module
    config = SimpleNamespace(desc=module.Descriptor(getattr(module.Mode, mode)),
        constructor_calls=0, input_batch=input_batch(),
        model_inputs=dict(input_ids=None, positions=module.Tensor(7000),
                          inputs_embeds=module.Tensor(8000), intermediate_tensors=None,
                          override=None))
    records = []
    observed = observation.observed_runner(module.GPUModelRunner, native.sites,
        execution.batch_snapshot, capture_record or records.append, limits=limits)(config, 'cuda:0')
    graph = module.Graph()
    observed.cudagraph_manager.graphs[config.desc] = graph
    if mode == 'PIECEWISE':
        observed.graph_wrapper = module.CUDAGraphWrapper(config.desc, observed.model)
        observed.graph_wrapper.concrete_cudagraph_entries[config.desc] = module.Entry(config.desc, graph)
    return observed, graph, records


def work():
    return SimpleNamespace(total_num_scheduled_tokens=4,
        num_scheduled_tokens={'internal-prefill': 3, 'internal-decode': 1})


def record(observed):
    return observed._model_input_observation.last_record


def events(observed, kind):
    return [row for row in record(observed)['events'] if row['kind'] == kind]


def test_pinned_retained_sources_and_executable_statement_anchors():
    roots = dict(worker='observed', runner='observed_generation2', graphs='observed_generation2',
                 piecewise='observed_generation5', breakable='observed_generation5', timing='observed_generation5')
    anchors = {
        'runner': [(('GPUModelRunner', 'execute_model'), ('final_inputs',))],
        'graphs': [(('CudaGraphManager', 'run_fullgraph'), ('full_replay',)),
                   (('CudaGraphManager', 'capture'), ('full_capture_store', 'full_capture_forward'))],
        'piecewise': [(('CUDAGraphWrapper', '__call__'), ('pw_replay', 'pw_capture', 'pw_capture_store'))],
        'breakable': [(('BreakableCUDAGraphWrapper', '__call__'), ('break_dispatch', 'break_capture_dispatch')),
                      (('BreakableCUDAGraphWrapper', '_replay'), ('break_replay',)),
                      (('BreakableCUDAGraphWrapper', '_capture'), ('break_capture', 'break_capture_store')),
                      (('BreakableCUDAGraphCapture', 'replay'), ('break_segment',))],
    }
    for name, relative in observation.SOURCE_PATHS.items():
        path = TASK/'native_role_sources'/roots[name]/relative
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == observation.SOURCE_PINS[name]
        ast.parse(body)
        compiled = compile(body, str(path), 'exec', dont_inherit=True)
        for owner, statements in anchors.get(name, ()):
            code = observation.nested_code(compiled, *owner)
            executable = {line for _, line in dis.findlinestarts(code)}
            assert all(observation.SOURCE_LINES[statement] in executable for statement in statements)
    path = TASK/'native_role_sources/observed_generation2'/observation.SOURCE_PATHS['runner']
    assert path.read_text().splitlines()[observation.SOURCE_LINES['final_inputs']-1].strip() == 'self.step_timing.forward_start()'
    graphs = (TASK/'native_role_sources/observed_generation2'/observation.SOURCE_PATHS['graphs']).read_text().splitlines()
    assert graphs[observation.SOURCE_LINES['full_replay']-1].strip() == 'self.graphs[desc].replay()'
    stored = observation.SOURCE_LINES['full_capture_store']
    assert graphs[stored-2].strip() == 'self.graphs[desc] = graph'
    assert graphs[stored-1].strip() == 'compilation_counter.num_cudagraph_captured += 1'
    pw = (TASK/'native_role_sources/observed_generation5'/observation.SOURCE_PATHS['piecewise']).read_text().splitlines()
    assert pw[observation.SOURCE_LINES['pw_replay']-1].strip() == 'entry.cudagraph.replay()'
    br = (TASK/'native_role_sources/observed_generation5'/observation.SOURCE_PATHS['breakable']).read_text().splitlines()
    assert br[observation.SOURCE_LINES['break_replay']-1].strip() == 'entry.capture.replay()'


@pytest.mark.parametrize('fault', ['hash', 'loaded_code', 'line'])
def test_hash_code_and_statement_delivery_bindings_reject_changes(native, monkeypatch, fault):
    pins, lines = dict(native.pins), dict(MARKERS)
    if fault == 'hash':
        pins['runner'] = '0'*64
    elif fault == 'loaded_code':
        original = native.module.GPUModelRunner.execute_model
        monkeypatch.setattr(original, '__code__', original.__code__.replace(co_firstlineno=1))
    else:
        lines['final_inputs'] = 99999
    with pytest.raises(ValueError):
        observation.ObservationSites.bind(native.facade, native.module.Worker._make_model_runner,
                                         pins=pins, lines=lines)


def test_actual_finalized_kwargs_none_overrides_storage_and_frontier_association(native):
    observed, graph, _ = runner(native)
    assert observed.execute_model(work()) == 'full-result'
    row = events(observed, 'finalized_model_inputs')[0]
    assert row['model_inputs']['input_ids'] is None
    assert row['model_inputs']['intermediate_tensors'] is None and row['model_inputs']['override'] is None
    positions = row['model_inputs']['positions']
    assert (positions['data_ptr'], positions['storage_ptr'], positions['storage_offset']) == (7000, 1000, 2)
    assert positions['shape'] == [8, 2] and positions['stride'] == [2, 1]
    assert positions['device'] == 'cuda:0' and positions['dtype'] == 'torch.float32'
    assert [entry['req_id'] for entry in row['request_frontiers']] == ['internal-decode', 'internal-prefill']
    assert [entry['idx_mapping_np'] for entry in row['request_frontiers']] == [6, 2]
    assert [entry['query_end'] for entry in row['request_frontiers']] == [1, 4]
    observed.model_state['positions'].pointer = 9000
    observed.input_batch.req_ids[:] = ['mutated', 'mutated']
    assert row['model_inputs']['positions']['data_ptr'] == 7000
    assert row['request_frontiers'][0]['req_id'] == 'internal-decode'
    assert observed.execution_calls == 1 and graph.calls == 1
    assert observed.vllm_config.constructor_calls == 1
    assert type(observed).sample_tokens is native.module.GPUModelRunner.sample_tokens
    assert set(record(observed)['evidence_status'].values()) == {'UNMEASURED'}
    assert record(observed)['native_compatibility'] == 'UNVERIFIED'


def test_full_replay_observation_does_not_require_or_claim_a_model_forward(native):
    observed, graph, _ = runner(native)
    observed.execute_model(work())
    assert observed.model_calls == 0 and graph.calls == 1
    attempt = events(observed, 'replay_attempt')[0]
    assert attempt['path'] == 'FULL' and attempt['graph_id'] == id(graph)
    assert attempt['input_observation'] == events(observed, 'finalized_model_inputs')[0]['event']
    assert attempt['args'] is None and attempt['kwargs'] is None
    assert events(observed, 'replay_outcome')[0]['outcome'] == 'returned_submission_not_completion'
    assert events(observed, 'replay_outcome')[0]['gpu_completion'] == 'UNMEASURED'


@pytest.mark.parametrize('mode', ['FULL', 'PIECEWISE'])
def test_exact_selected_entry_and_graph_receiver_ignores_first_table_candidate(native, mode):
    observed, graph, _ = runner(native, mode)
    decoy = native.module.Graph()
    wrong = native.module.Descriptor(getattr(native.module.Mode, mode), 16)
    if mode == 'FULL':
        class ExactLookup(dict):
            reads = 0
            def __getitem__(self, key):
                self.reads += 1
                if self.reads != 1:
                    raise AssertionError('Observer must not query graph table instead of the actual replay receiver')
                return super().__getitem__(key)
        observed.cudagraph_manager.graphs = ExactLookup({wrong: decoy, observed.vllm_config.desc: graph})
        expected_entry = graph
    else:
        wrapper = observed.graph_wrapper
        expected_entry = wrapper.concrete_cudagraph_entries[observed.vllm_config.desc]
        wrapper.concrete_cudagraph_entries = {wrong: native.module.Entry(wrong, decoy),
                                              observed.vllm_config.desc: expected_entry}
    observed.execute_model(work())
    row = events(observed, 'replay_attempt')[0]
    assert row['entry_id'] == id(expected_entry) and row['graph_id'] == id(graph)
    assert graph.calls == 1 and decoy.calls == 0 and row['descriptor']['num_tokens'] == 8
    if mode == 'PIECEWISE':
        assert row['kwargs']['input_ids'] is None
        assert row['kwargs']['positions']['data_ptr'] == 7000


def test_breakable_actual_key_segments_and_selected_capture_are_observed(native):
    observed, _, _ = runner(native, 'PIECEWISE')
    module = native.module
    wrapper = module.BreakableCUDAGraphWrapper(observed.vllm_config.desc, observed.model)
    capture, graph = module.BreakableCUDAGraphCapture(), module.Graph()
    eager_calls = []
    capture.segments = [graph.replay, lambda: eager_calls.append('native-eager')]
    selected = module.BreakEntry(observed.vllm_config.desc, capture)
    decoy = module.BreakEntry(observed.vllm_config.desc, module.BreakableCUDAGraphCapture())
    wrapper.entries = {(observed.vllm_config.desc, 'wrong'): decoy,
                       (observed.vllm_config.desc, 'selected'): selected}
    observed.graph_wrapper = wrapper
    assert observed.execute_model(work()) == 'break-result'
    row = events(observed, 'replay_attempt')[0]
    assert row['path'] == 'BREAKABLE_PIECEWISE'
    assert row['entry_id'] == id(selected) and row['graph_id'] == id(capture)
    assert row['selection_key'][1] == 'selected'
    assert [row['segment_kind'] for row in events(observed, 'breakable_segment_attempt')] == ['GRAPH_REPLAY', 'EAGER_BREAK']
    assert graph.calls == 1 and eager_calls == ['native-eager'] and observed.model_calls == 0
    assert events(observed, 'replay_outcome')[0]['outcome'] == 'returned_submission_not_completion'


@pytest.mark.parametrize('mode', ['FULL', 'PIECEWISE'])
def test_identity_continuity_is_only_at_observed_boundaries_and_detects_replacement(native, mode):
    observed, graph, _ = runner(native, mode)
    observed.execute_model(work())
    first = events(observed, 'replay_attempt')[0]['continuity']
    observed.execute_model(work())
    second = events(observed, 'replay_attempt')[0]['continuity']
    assert first['status'] == 'FIRST_OBSERVATION'
    assert second['status'] == 'SAME_OBJECTS_AT_OBSERVED_BOUNDARIES'
    assert first['serial'] == second['serial']
    assert second['residency_between_scopes'] == 'UNVERIFIED'
    replacement = native.module.Graph()
    if mode == 'FULL':
        observed.cudagraph_manager.graphs[observed.vllm_config.desc] = replacement
    else:
        observed.graph_wrapper.concrete_cudagraph_entries[observed.vllm_config.desc].cudagraph = replacement
    observed.execute_model(work())
    third = events(observed, 'replay_attempt')[0]['continuity']
    assert third['status'] == 'REPLACED_OR_RELEASED' and third['serial'] != second['serial']
    assert graph.calls == 2 and replacement.calls == 1


@pytest.mark.parametrize('mode', ['FULL', 'PIECEWISE'])
@pytest.mark.parametrize('failure', [ValueError('native-replay'), KeyboardInterrupt('native-replay'), SystemExit(7)])
def test_native_replay_failure_identity_one_delegation_and_hook_restoration(native, mode, failure):
    observed, graph, _ = runner(native, mode)
    graph.failure = failure
    profiles, traces = [], []
    def previous_profile(frame, event, arg):
        if frame.f_code is native.sites.codes['execute']:
            profiles.append(event)
    def previous_trace(frame, event, arg):
        if frame.f_code is native.sites.codes['execute']:
            traces.append(event)
            return previous_trace
        return None
    original_profile, original_trace = sys.getprofile(), sys.gettrace()
    sys.setprofile(previous_profile)
    sys.settrace(previous_trace)
    try:
        with pytest.raises(type(failure)) as caught:
            observed.execute_model(work())
        assert caught.value is failure and observed.execution_calls == 1 and graph.calls == 1
        assert sys.getprofile() is previous_profile and sys.gettrace() is previous_trace
        assert profiles.count('call') == 1 and 'exception' in traces
        outcome = events(observed, 'replay_outcome')[0]
        assert outcome['outcome'] == 'raised' and outcome['error_type'] == observation.class_name(failure)
        assert record(observed)['native_outcome'] == 'raised'
    finally:
        sys.settrace(original_trace)
        sys.setprofile(original_profile)


def test_prior_trace_local_replacement_and_none_return_semantics_are_preserved(native):
    observed, _, _ = runner(native, 'PIECEWISE')
    calls = []
    target = native.sites.codes['pw']
    def second(frame, event, arg):
        calls.append(('second', event))
        return None
    def first(frame, event, arg):
        calls.append(('first', event))
        return second
    def global_trace(frame, event, arg):
        return first if frame.f_code is target else None
    previous = sys.gettrace()
    sys.settrace(global_trace)
    try:
        observed.execute_model(work())
        assert sys.gettrace() is global_trace
        assert sum(name == 'first' for name, _ in calls) == 1
        assert sum(name == 'second' for name, _ in calls) > 1
        assert ('second', 'return') in calls
    finally:
        sys.settrace(previous)


@pytest.mark.parametrize('fault', ['batch_snapshot', 'metadata', 'capture_writer'])
def test_observation_errors_isolated_from_native_execution_and_capture(native, fault):
    def broken(*args):
        raise KeyboardInterrupt('observer-only-failure')
    observed, graph, _ = runner(native, capture_record=broken if fault == 'capture_writer' else None)
    if fault == 'batch_snapshot':
        # State is initialized by one scope; no original execution is duplicated.
        observed.execute_model(work())
        observed._model_input_observation.snapshot_batch = broken
        observed.execute_model(work())
        assert graph.calls == 2 and observed.execution_calls == 2
    elif fault == 'metadata':
        observed.model_state['positions'] = SimpleNamespace(opaque=True)
        observed.execute_model(work())
        assert graph.calls == 1 and observed.execution_calls == 1
        assert events(observed, 'finalized_model_inputs')[0]['metadata_complete'] is False
    else:
        assert observed.capture_model(profile_only=False) == 41
        assert observed.capture_calls == 1 and observed.cudagraph_manager.capture_calls == 1
    assert observed._model_input_observation.summary()['native_compatibility'] == 'UNVERIFIED'
    if fault != 'metadata':
        assert observed._model_input_observation.last_summary['status'] == 'UNVERIFIED'


def test_full_capture_keeps_none_runtime_mode_and_cpu_only_input_graph_association(native):
    observed, _, records = runner(native)
    assert observed.capture_model(profile_only=True) == 41
    assert observed.capture_calls == 1 and observed.cudagraph_manager.capture_calls == 1
    assert observed.cudagraph_manager.graphs_stored == 1 and len(records) == 1
    forward = events(observed, 'capture_forward')
    assert len(forward) == 2 and all(row['forward_runtime_mode'] == 'NONE' for row in forward)
    assert [row['native_capture_region'] for row in forward] == [False, True]
    stored = events(observed, 'capture_entry_stored')[0]
    graph = observed.cudagraph_manager.graphs[observed.vllm_config.desc]
    assert stored['graph_id'] == id(graph) and graph.calls == 0
    assert all(row['device_operations_added'] is False for row in forward)
    assert record(observed)['lifecycle'] == 'capture'
    assert set(record(observed)['evidence_status'].values()) == {'UNMEASURED'}
    assert stored['gpu_completion'] == 'UNMEASURED'
    observed.execute_model(work())
    assert events(observed, 'replay_attempt')[0]['continuity']['status'] == 'SAME_OBJECTS_AT_OBSERVED_BOUNDARIES'
    assert graph.calls == 1


@pytest.mark.parametrize('breakable', [False, True])
def test_piecewise_capture_in_execute_is_cpu_metadata_only_and_not_a_replay(native, breakable):
    observed, _, _ = runner(native, 'PIECEWISE')
    module = native.module
    if breakable:
        wrapper = module.BreakableCUDAGraphWrapper(observed.vllm_config.desc, observed.model)
        wrapper.entries[(observed.vllm_config.desc, 'selected')] = module.BreakEntry(observed.vllm_config.desc)
        observed.graph_wrapper = wrapper
    else:
        wrapper = observed.graph_wrapper
        wrapper.concrete_cudagraph_entries[observed.vllm_config.desc].cudagraph = None
    assert observed.execute_model(work()) is observed.result
    assert wrapper.captures == 1 and observed.model_calls == 1
    assert len(events(observed, 'wrapper_capture_forward')) == 1
    assert len(events(observed, 'capture_entry_stored')) == 1
    assert events(observed, 'replay_attempt') == []
    assert events(observed, 'wrapper_capture_forward')[0]['kwargs']['input_ids'] is None
    assert events(observed, 'wrapper_capture_forward')[0]['device_operations_added'] is False


@pytest.mark.parametrize('limit', ['scope_bytes', 'scope_events', 'metadata_items', 'tracked_entries'])
def test_explicit_limits_stop_only_observations_without_native_retries(native, limit):
    limits = replace(observation.ObservationLimits(), **{limit: 1})
    observed, graph, _ = runner(native, limits=limits)
    observed.execute_model(work())
    if limit == 'tracked_entries':
        second_descriptor = native.module.Descriptor(native.module.Mode.FULL, 16)
        observed.vllm_config.desc = second_descriptor
        observed.cudagraph_manager.graphs[second_descriptor] = graph
        observed.execute_model(work())
        assert observed.execution_calls == graph.calls == 2
    else:
        assert observed.execution_calls == graph.calls == 1
    assert observed._model_input_observation.last_summary['status'] == 'UNVERIFIED'
    assert observed._model_input_observation.stopped is True
    before = graph.calls
    observed.execute_model(work())
    assert graph.calls == before+1 and record(observed)['events'] == []


def test_missing_finalized_callback_delivery_is_unverified_not_forward_inferred(native):
    sites = replace(native.sites, codes=dict(native.sites.codes, execute=(lambda: None).__code__))
    original_sites, native.sites = native.sites, sites
    try:
        observed, graph, _ = runner(native)
        observed.execute_model(work())
        assert observed.execution_calls == graph.calls == 1 and observed.model_calls == 0
        assert record(observed)['status'] == 'UNVERIFIED_NO_FINAL_INPUT_CALLBACK'
        assert record(observed)['native_compatibility'] == 'UNVERIFIED'
    finally:
        native.sites = original_sites


def test_native_disabling_hooks_is_detected_and_prior_hooks_restored(native, monkeypatch):
    observed, graph, _ = runner(native)
    original = native.module.Graph.replay
    def remove_hooks(self):
        sys.setprofile(None)
        sys.settrace(None)
        return original(self)
    monkeypatch.setattr(native.module.Graph, 'replay', remove_hooks)
    original_profile, original_trace = sys.getprofile(), sys.gettrace()
    def previous_profile(*args):
        pass
    def previous_trace(*args):
        return None
    sys.setprofile(previous_profile)
    sys.settrace(previous_trace)
    try:
        observed.execute_model(work())
        assert graph.calls == observed.execution_calls == 1
        assert sys.getprofile() is previous_profile and sys.gettrace() is previous_trace
        assert record(observed)['status'] == 'UNVERIFIED'
        assert record(observed)['hook_delivery'] == 'UNVERIFIED'
    finally:
        sys.settrace(original_trace)
        sys.setprofile(original_profile)


def test_noncallable_prior_profiler_is_preserved_and_native_delegates_once(native, monkeypatch):
    observed, graph, _ = runner(native)
    sentinel = object()
    monkeypatch.setattr(observation.sys, 'getprofile', lambda: sentinel)
    observed.execute_model(work())
    assert graph.calls == observed.execution_calls == 1
    assert record(observed)['status'] == 'UNVERIFIED'
    assert record(observed)['observer_errors'][0]['kind'] == 'hook_compatibility'


def test_source_binding_failure_keeps_native_factory_once_and_explicit_gap(native):
    module = native.module
    def absent(*args):
        raise ValueError('Native source does not match the pinned observer')
    worker = execution.observed_worker(module.Worker, record=lambda *args: None,
                                       prepare_runner=absent)()
    worker.native_factory_calls = 0
    worker.vllm_config = SimpleNamespace(constructor_calls=0, desc=module.Descriptor(module.Mode.FULL),
        input_batch=input_batch(), model_inputs={})
    worker.device = 'cuda:0'
    result = worker._make_model_runner()
    assert type(result) is module.GPUModelRunner
    assert worker.native_factory_calls == 1 and worker.vllm_config.constructor_calls == 1
    assert worker._runner_observer_error['status'] == 'UNVERIFIED'
    assert worker._runner_observer_error['evidence_status']['G2'] == 'UNMEASURED'


def test_observing_worker_factory_constructor_execute_and_capture_keep_original_labels(native):
    module, retained = native.module, []
    worker = execution.observed_worker(module.Worker, record=lambda worker, payload: retained.append(json.loads(payload)),
        prepare_runner=lambda worker, factory: (module.GPUModelRunner, native.sites))()
    worker.native_factory_calls, worker.worker_calls = 0, 0
    worker.rank, worker.local_rank = 0, 0
    worker._capacity_identity = {'pid': os.getpid(), 'start_time': 123}
    worker.vllm_config = SimpleNamespace(constructor_calls=0, desc=module.Descriptor(module.Mode.FULL),
        input_batch=input_batch(), model_inputs=dict(input_ids=None, positions=module.Tensor(), inputs_embeds=None))
    worker.device = 'cuda:0'
    worker.model_runner = worker._make_model_runner()
    assert worker.vllm_config.constructor_calls == 1 and worker.native_factory_calls == 0
    assert worker.model_runner.capture_model(profile_only=False) == 41
    worker.execute_model(work())
    assert worker.worker_calls == worker.model_runner.execution_calls == 1
    assert [row['scope'] for row in retained] == ['capture_cpu_metadata_only_not_G2', 'cpu_request_metadata_only_not_G2']
    assert set(retained[-1]['evidence_status'].values()) == {'UNMEASURED'}
    summary = worker.request_snapshot()
    assert summary['status'] == 'OBSERVED_CPU_METADATA_ONLY_NOT_G2'
    assert summary['source_pinned_model_observation']['native_compatibility'] == 'UNVERIFIED'
    assert summary['request_metadata']['capture_metadata_records'] == 1
    assert summary['evidence_status']['G2'] == 'UNMEASURED'
    assert type(worker).capacity_snapshot is execution.capacity.capacity_snapshot
    assert type(worker.model_runner).sample_tokens is module.GPUModelRunner.sample_tokens



def test_trace_selection_error_cannot_escape_into_native_replay(native, monkeypatch):
    observed, graph, _ = runner(native)
    original = observation.ObservationScope.replay_parent

    def broken(scope, parent):
        if parent is not None and parent.f_code is native.sites.codes['execute']:
            raise KeyboardInterrupt('observer frame selection failed')
        return original(scope, parent)

    monkeypatch.setattr(observation.ObservationScope, 'replay_parent', broken)
    prior_profile, prior_trace = sys.getprofile(), sys.gettrace()
    assert observed.execute_model(work()) == 'full-result'
    assert observed.execution_calls == graph.calls == 1
    assert sys.getprofile() is prior_profile and sys.gettrace() is prior_trace
    assert record(observed)['status'] == 'UNVERIFIED'


def test_hook_inspection_failure_still_restores_both_hooks_in_finally(native, monkeypatch):
    observed, graph, _ = runner(native)
    actual_getprofile = sys.getprofile
    prior_profile, prior_trace = actual_getprofile(), sys.gettrace()

    def broken_getprofile():
        if sys._getframe(1).f_code.co_name == '__exit__':
            raise RuntimeError('observer hook inspection failed')
        return actual_getprofile()

    monkeypatch.setattr(observation.sys, 'getprofile', broken_getprofile)
    try:
        assert observed.execute_model(work()) == 'full-result'
        assert observed.execution_calls == graph.calls == 1
        assert actual_getprofile() is prior_profile and sys.gettrace() is prior_trace
        assert record(observed)['status'] == 'UNVERIFIED'
    finally:
        sys.settrace(prior_trace)
        sys.setprofile(prior_profile)


def test_partial_hook_installation_restores_prior_hooks_and_delegates_once(native, monkeypatch):
    observed, graph, _ = runner(native)
    actual_settrace = sys.settrace
    prior_profile, prior_trace = sys.getprofile(), sys.gettrace()

    def fail_installation(hook):
        if isinstance(getattr(hook, '__self__', None), observation.ObservationScope):
            raise KeyboardInterrupt('observer trace installation failed')
        return actual_settrace(hook)

    monkeypatch.setattr(observation.sys, 'settrace', fail_installation)
    assert observed.execute_model(work()) == 'full-result'
    assert observed.execution_calls == graph.calls == 1
    assert sys.getprofile() is prior_profile and sys.gettrace() is prior_trace
    assert record(observed)['observer_errors'][0]['kind'] == 'hook_installation'
    assert record(observed)['status'] == 'UNVERIFIED'


@pytest.mark.parametrize('mode', ['FULL', 'PIECEWISE'])
def test_absent_actual_replay_callback_is_explicitly_unverified(native, monkeypatch, mode):
    observed, graph, _ = runner(native, mode)
    original = native.module.Graph.replay
    monkeypatch.setattr(native.module.Graph, 'replay', lambda *args: original(*args))
    observed.execute_model(work())
    assert observed.execution_calls == graph.calls == 1
    assert len(events(observed, 'finalized_model_inputs')) == 1
    assert events(observed, 'replay_attempt') == []
    assert record(observed)['status'] == 'UNVERIFIED'
    assert record(observed)['observation_status']['selected_graph_entry'] == 'UNVERIFIED_NO_REPLAY_CALLBACK'
    assert record(observed)['native_compatibility'] == 'UNVERIFIED'
    assert record(observed)['evidence_status']['gpu_completion'] == 'UNMEASURED'


def test_missing_replay_return_delivery_does_not_claim_observed_success(native, monkeypatch):
    observed, graph, _ = runner(native)
    monkeypatch.setattr(observation.ObservationScope, 'finish_replay', lambda *args, **kwargs: None)
    observed.execute_model(work())
    assert observed.execution_calls == graph.calls == 1
    assert events(observed, 'replay_attempt')[0]['outcome'] == 'UNVERIFIED_MISSING_REPLAY_RETURN'
    assert record(observed)['status'] == 'UNVERIFIED'
    assert record(observed)['observation_status']['replay_outcomes'] == 'UNVERIFIED_MISSING_REPLAY_RETURN'


def test_caught_replay_exception_followed_by_return_is_a_submission_only(native, monkeypatch):
    observed, graph, _ = runner(native)

    def caught_exception(self):
        self.calls += 1
        try:
            raise ValueError('caught inside the original CPU replay fixture')
        except ValueError:
            return None

    monkeypatch.setattr(native.module.Graph, 'replay', caught_exception)
    observed.execute_model(work())
    assert observed.execution_calls == graph.calls == 1
    outcome = events(observed, 'replay_outcome')[0]
    assert outcome['outcome'] == 'returned_submission_not_completion' and outcome['error_type'] is None
    assert outcome['gpu_completion'] == 'UNMEASURED'


def test_breakable_continuity_is_associated_with_the_actual_selection_key(native):
    observed, _, _ = runner(native, 'PIECEWISE')
    module = native.module
    wrapper = module.BreakableCUDAGraphWrapper(observed.vllm_config.desc, observed.model)
    first, second = module.BreakableCUDAGraphCapture(), module.BreakableCUDAGraphCapture()
    wrapper.entries = {(observed.vllm_config.desc, 'selected'): module.BreakEntry(observed.vllm_config.desc, first),
                       (observed.vllm_config.desc, 'other'): module.BreakEntry(observed.vllm_config.desc, second)}
    observed.graph_wrapper = wrapper
    observed.execute_model(work())
    first_serial = events(observed, 'replay_attempt')[0]['continuity']['serial']
    wrapper.key_value = 'other'
    observed.execute_model(work())
    assert events(observed, 'replay_attempt')[0]['continuity']['status'] == 'FIRST_OBSERVATION'
    wrapper.key_value = 'selected'
    observed.execute_model(work())
    continuity = events(observed, 'replay_attempt')[0]['continuity']
    assert continuity['status'] == 'SAME_OBJECTS_AT_OBSERVED_BOUNDARIES'
    assert continuity['serial'] == first_serial and continuity['residency_between_scopes'] == 'UNVERIFIED'


def test_breakable_capture_records_the_native_selected_key_without_table_reads(native):
    observed, _, _ = runner(native, 'PIECEWISE')
    module = native.module
    wrapper = module.BreakableCUDAGraphWrapper(observed.vllm_config.desc, observed.model)
    entry = module.BreakEntry(observed.vllm_config.desc)
    wrapper.entries[(observed.vllm_config.desc, 'selected')] = entry
    observed.graph_wrapper = wrapper
    observed.execute_model(work())
    assert observed.model_calls == wrapper.captures == 1
    forward = events(observed, 'wrapper_capture_forward')[0]
    stored = events(observed, 'capture_entry_stored')[0]
    assert forward['selection_key'][1] == stored['selection_key'][1] == 'selected'
    assert stored['entry_id'] == id(entry) and stored['graph_id'] == id(entry.capture)
    observed.execute_model(work())
    assert events(observed, 'replay_attempt')[0]['continuity']['status'] == 'SAME_OBJECTS_AT_OBSERVED_BOUNDARIES'


def test_capture_writer_failure_marks_last_record_and_summary_unverified(native):
    def failed_write(row):
        raise OSError('observer capture record cannot be retained')
    observed, _, _ = runner(native, capture_record=failed_write)
    assert observed.capture_model() == 41
    assert observed.capture_calls == observed.cudagraph_manager.capture_calls == 1
    assert record(observed)['status'] == 'UNVERIFIED'
    assert observed._model_input_observation.summary()['last_scope']['status'] == 'UNVERIFIED'
    assert record(observed)['observer_errors'][0]['kind'] == 'capture_record'



def test_capture_entry_metadata_cannot_claim_absent_argument_callbacks(native):
    sites = replace(native.sites, codes=dict(native.sites.codes,
        create_capture=(lambda: None).__code__, capture_forward=(lambda: None).__code__))
    original_sites, native.sites = native.sites, sites
    try:
        observed, _, _ = runner(native)
        assert observed.capture_model() == 41
        assert observed.capture_calls == observed.cudagraph_manager.capture_calls == 1
        assert len(events(observed, 'capture_entry_stored')) == 1
        assert events(observed, 'capture_forward') == events(observed, 'capture_inputs_prepared') == []
        assert record(observed)['status'] == 'UNVERIFIED'
        assert record(observed)['observation_status']['kwargs_metadata'] == 'UNVERIFIED_NO_INPUT_CALLBACK'
        assert record(observed)['native_compatibility'] == 'UNVERIFIED'
    finally:
        native.sites = original_sites



def test_worker_observer_class_preparation_error_preserves_one_native_factory(native, monkeypatch):
    module = native.module
    def failed_preparation(*args, **kwargs):
        raise KeyboardInterrupt('observer class preparation failed')
    monkeypatch.setattr(observation, 'observed_runner', failed_preparation)
    worker = execution.observed_worker(module.Worker, record=lambda *args: None,
        prepare_runner=lambda worker, factory: (module.GPUModelRunner, native.sites))()
    worker.native_factory_calls = 0
    worker.vllm_config = SimpleNamespace(constructor_calls=0, desc=module.Descriptor(module.Mode.FULL),
        input_batch=input_batch(), model_inputs={})
    worker.device = 'cuda:0'
    result = worker._make_model_runner()
    assert type(result) is module.GPUModelRunner
    assert worker.native_factory_calls == worker.vllm_config.constructor_calls == 1
    assert worker._runner_observer_error['status'] == 'UNVERIFIED'
    assert worker._runner_observer_error['evidence_status']['G2'] == 'UNMEASURED'


def test_native_constructor_failure_is_never_retried_through_the_factory(native, monkeypatch):
    module = native.module
    failure = SystemExit('native constructor failed')
    calls = []
    def failed_constructor(self, config, device):
        calls.append((config, device))
        raise failure
    monkeypatch.setattr(module.GPUModelRunner, '__init__', failed_constructor)
    worker = execution.observed_worker(module.Worker, record=lambda *args: None,
        prepare_runner=lambda worker, factory: (module.GPUModelRunner, native.sites))()
    worker.native_factory_calls = 0
    worker.vllm_config, worker.device = object(), 'cuda:0'
    with pytest.raises(SystemExit) as caught:
        worker._make_model_runner()
    assert caught.value is failure and calls == [(worker.vllm_config, worker.device)]
    assert worker.native_factory_calls == 0


def test_model_observation_summary_cannot_mutate_retained_status_or_errors(native):
    observed, _, _ = runner(native)
    observed.model_state['positions'] = SimpleNamespace(opaque=True)
    observed.execute_model(work())
    summary = observed._model_input_observation.summary()
    summary['last_scope']['observation_status']['kwargs_metadata'] = 'changed'
    summary['last_scope']['observer_errors'].append({'kind': 'changed'})
    again = observed._model_input_observation.summary()
    assert again['last_scope']['observation_status']['kwargs_metadata'] == 'UNVERIFIED_UNSUPPORTED_VALUE_TYPE'
    assert again['last_scope']['observer_errors'] == []
    assert record(observed)['status'] == 'UNVERIFIED'
