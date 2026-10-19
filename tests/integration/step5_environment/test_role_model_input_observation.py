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
    def add_requests(self, scheduler_output):
        return None
    def execute_model(self, scheduler_output, *, dummy_run=False):
        self.execution_calls += 1
        self.add_requests(scheduler_output)
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
    state = observed._model_input_observation
    assert state.stopped and state.summary()['status'] == 'UNVERIFIED'
    if limit == 'scope_bytes':
        assert state.last_record is None and state.last_summary is None
    else:
        assert state.last_summary['status'] == 'UNVERIFIED'
    before = graph.calls
    observed.execute_model(work())
    assert graph.calls == before+1
    if limit == 'scope_bytes':
        assert state.last_record is None and state.last_summary is None
    else:
        assert record(observed)['events'] == []


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


class DeviceCpuTensor:
    """CPU fault model queues operations so host reads cannot race owned copies."""
    def __init__(self, values, api, dtype, device, *, pinned=False):
        self.values, self.api, self.dtype, self.device = values, api, dtype, device
        self.pinned = pinned

    @property
    def shape(self):
        return self.values.shape

    def stride(self):
        return tuple(value//self.values.itemsize for value in self.values.strides)

    def storage_offset(self):
        return 0

    def data_ptr(self):
        return self.values.__array_interface__['data'][0]

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=self.data_ptr, nbytes=lambda: self.values.nbytes)

    def __getitem__(self, key):
        return DeviceCpuTensor(self.values[key], self.api, self.dtype, self.device, pinned=self.pinned)

    def copy_(self, source, *, non_blocking):
        assert non_blocking is True
        if self.device == 'cpu':
            assert self.pinned
        self.api.queued.append(lambda: np.copyto(self.values, source.values))
        self.api.operations.append(('copy', self.device, tuple(self.shape), self.api.stream.cuda_stream))
        return self

    def view(self, shape):
        if shape is self.api.uint8:
            return DeviceCpuTensor(self.values.view(np.uint8), self.api, self.api.uint8, self.device)
        return DeviceCpuTensor(self.values.reshape(shape), self.api, self.dtype, self.device, pinned=self.pinned)

    def numpy(self):
        assert self.api.host_reads_allowed, 'Host contents are unavailable before post-generate drain'
        assert self.device == 'cpu'
        return self.values

    def __getattr__(self, name):
        if name in ('cpu', 'item', 'tolist', 'synchronize'):
            raise AssertionError('No device host read, synchronization or recomputation is allowed')
        raise AttributeError(name)


class DeviceCpuApi:
    def __init__(self):
        from contextlib import nullcontext
        self.nullcontext = nullcontext
        self.queued, self.operations, self.events = [], [], []
        self.capturing, self.host_reads_allowed, self.complete_count = False, False, 0
        self.stream = SimpleNamespace(cuda_stream=71, device='cuda:0')
        self.float32 = SimpleNamespace(is_floating_point=True, name='float32')
        self.bfloat16 = SimpleNamespace(is_floating_point=True, name='bfloat16')
        self.int32, self.int64, self.bool, self.uint8 = [DeviceCpuDtype(name) for name in
                                                      ('int32', 'int64', 'bool', 'uint8')]
        self.cuda = SimpleNamespace(is_current_stream_capturing=lambda: self.capturing,
                                    current_stream=lambda device: self.stream, Event=self.event)
        self.fail_copy, self.fail_event = False, False

    def inference_mode(self):
        return self.nullcontext()

    def finfo(self, dtype):
        return SimpleNamespace(bits=16 if dtype is self.bfloat16 else 32)

    def empty(self, shape, *, dtype, device, pin_memory=False):
        assert not self.capturing
        self.operations.append(('allocate', device, tuple(shape)))
        kind = ('float32' if dtype is self.float32 else 'uint16' if dtype is self.bfloat16 else dtype.name)
        return DeviceCpuTensor(np.zeros(shape, dtype=kind), self, dtype, device, pinned=pin_memory)

    def abs(self, tensor, *, out):
        self.queued.append(lambda: np.abs(tensor.values, out=out.values))
        self.operations.append(('abs', tuple(tensor.shape)))

    def lt(self, tensor, value, *, out):
        self.queued.append(lambda: np.less(tensor.values, value, out=out.values))
        self.operations.append(('finite', tuple(tensor.shape)))

    def sum(self, tensor, *, dim, dtype, out):
        assert dim == (0, 1) and dtype is self.int64
        self.queued.append(lambda: np.copyto(out.values, np.sum(tensor.values, dtype=np.int64)))
        self.operations.append(('sum', tuple(tensor.shape)))

    def event(self, *, enable_timing):
        assert not enable_timing
        api = self
        class Event:
            def record(self, stream):
                if api.fail_event:
                    raise ValueError('injected event record failure')
                assert stream is api.stream and not api.capturing
                self.ready_at = len(api.queued)
                api.operations.append(('event', stream.cuda_stream, self.ready_at))
            def query(self):
                api.operations.append(('query',))
                return self.ready_at <= api.complete_count
            def synchronize(self):
                raise AssertionError('Observer may not wait or synchronize')
        result = Event()
        self.events.append(result)
        return result

    def complete(self):
        for operation in self.queued[self.complete_count:]:
            operation()
        self.complete_count = len(self.queued)
        self.host_reads_allowed = True


class DeviceCpuDtype:
    is_floating_point = False
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return 'torch.'+self.name


@pytest.fixture
def device_case():
    api = DeviceCpuApi()
    tokens = [31, 32, 33, 34]
    state = SimpleNamespace(req_id_to_index={'full-internal-A': 2}, index_to_req_id={2: 'full-internal-A'},
        prompt_len=SimpleNamespace(np=np.array([0, 0, 4])), prefill_len=SimpleNamespace(np=np.array([0, 0, 4])))
    runner = SimpleNamespace(device='cuda:0', main_stream=api.stream, use_pp=False, pcp_manager=None,
        num_speculative_steps=0, is_first_pp_rank=True, is_last_pp_rank=True, max_num_tokens=8, max_num_reqs=2,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=3, fp32_residual_connection=True),
                                     dtype=api.bfloat16), req_states=state,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=['layer.0', 'layer.1'])]))
    cpu = dict(num_reqs=1, num_reqs_after_padding=2, num_tokens=4, num_tokens_after_padding=8,
        req_ids=['full-internal-A'], idx_mapping_np=[2], num_scheduled_tokens=[4],
        query_start_loc_np=[0, 4, 4], num_computed_tokens_np=[0], prefill_len_np=[4],
        num_computed_prefill_tokens_np=[0], is_prefilling_np=[True], prefill_runs_as_decode_np=None)
    def tensor(values, dtype):
        result = api.empty(np.array(values).shape, dtype=dtype, device='cuda:0')
        result.values[...] = values
        return result
    batch = SimpleNamespace(input_ids=tensor(tokens+[91]*4, api.int32),
        positions=tensor(list(range(8)), api.int64), is_padding=tensor([False]*4+[True]*4, api.bool),
        query_start_loc=tensor([0, 4, 4], api.int32), seq_lens=tensor([4, 0], api.int32),
        idx_mapping=tensor([2], api.int64))
    runner.model_state = SimpleNamespace(request_state=tensor([2, -1], api.int64))
    embed = tensor(np.arange(24).reshape(8, 3), api.bfloat16)
    slots = tensor(list(range(8)), api.int64)
    hidden = tensor(np.arange(24).reshape(8, 3), api.float32)
    hidden.values[4:] = np.nan
    runner.execute_model_state = SimpleNamespace(input_batch=batch, hidden_states=hidden)
    sites = SimpleNamespace(torch=api, tensor_type=DeviceCpuTensor, intermediate_type=None, padding_mask_used=True)
    limits = observation.DeviceObservationLimits(steps=3, scan_rows=2, witness_rows=1, owned_bytes=65536, rpc_bytes=65536)
    requests = [dict(name='A', prompt_token_ids=tokens), dict(name='B', prompt_token_ids=[31, 32, 33, 35])]
    observed = observation.DeviceObservation(runner, sites, limits, requests,
        dict(n=1, max_tokens=2, min_tokens=2, seed=None), 65536)
    req = SimpleNamespace(req_id='full-internal-A', prefill_token_ids=tokens, prompt_len=4,
        num_computed_tokens=0, sampling_params=SimpleNamespace(n=1, max_tokens=2, min_tokens=2, seed=None))
    observed.registrations(SimpleNamespace(scheduled_new_reqs=[req]))
    values = dict(input_batch=batch, model_inputs=dict(input_ids=None, positions=batch.positions,
        inputs_embeds=embed, intermediate_tensors=None), slot_mappings_by_layer={'layer.0': slots, 'layer.1': slots})
    return SimpleNamespace(api=api, observed=observed, runner=runner, cpu=cpu, values=values,
                           hidden=hidden, embed=embed, req=req, tensor=tensor)


def test_device_destinations_are_owned_and_complete_before_native_source_reuse(device_case):
    case = device_case
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    case.observed.output(index, None, [])
    assert index == 0 and not case.api.host_reads_allowed
    assert not any(row[0] == 'query' for row in case.api.operations)
    with pytest.raises(ValueError, match='not ready'):
        case.observed.drain(lambda row: pytest.fail('Unready data must not be published'))
    case.api.complete()
    case.hidden.values[:] = -99
    case.values['input_batch'].input_ids.values[:] = 77
    records = []
    result = case.observed.drain(records.append)
    row = records[0]['rows'][0]
    assert row['values']['input_ids'][:4] == [31, 32, 33, 34]
    assert row['values']['hidden_states']['raw_hex'] == np.array([[9, 10, 11]], dtype=np.float32).tobytes().hex()
    assert row['finite_elements_observed'] == row['finite_elements_expected'] == 12
    assert row['padding_finite'] == 'NOT_SCANNED_NOT_VALID_TOKEN_WORK'
    assert result['drained_steps'] == 1 and result['clean_performance'] == result['G2'] == 'UNVERIFIED'
    assert 'raw_hex' not in result['steps'][0]['values']['hidden_states']
    assert result['native_async_output_consumed_by_observer'] is False


@pytest.mark.parametrize('fault', ['id', 'index', 'prompt', 'sampling', 'duplicate'])
def test_worker_registration_has_complete_native_identity_and_sampling(device_case, fault):
    case = device_case
    case.observed.registered.clear()
    if fault == 'id':
        case.runner.req_states.index_to_req_id[2] = 'different-same-suffix-A'
    elif fault == 'index':
        case.runner.req_states.req_id_to_index.clear()
    elif fault == 'prompt':
        case.req.prefill_token_ids = [31, 32, 33, 999]
    elif fault == 'sampling':
        case.req.sampling_params.seed = 0
    reqs = [case.req, case.req] if fault == 'duplicate' else [case.req]
    case.observed.registrations(SimpleNamespace(scheduled_new_reqs=reqs))
    assert case.observed.stopped and case.observed.errors


@pytest.mark.parametrize('fault', ['stream', 'capture', 'capacity', 'index', 'missing_slots',
                                  'slot_view', 'positions', 'embedding_shape', 'no_input'])
def test_final_device_input_failures_stop_only_observation(device_case, fault):
    case = device_case
    if fault == 'stream':
        case.api.stream = SimpleNamespace(cuda_stream=72, device='cuda:0')
    elif fault == 'capture':
        case.api.capturing = True
    elif fault == 'capacity':
        case.observed.entries = [{}]*case.observed.limits.steps
    elif fault == 'index':
        case.runner.req_states.req_id_to_index['full-internal-A'] = 1
    elif fault == 'missing_slots':
        del case.values['slot_mappings_by_layer']['layer.1']
    elif fault == 'slot_view':
        case.values['slot_mappings_by_layer']['layer.1'] = case.tensor([0]*8, case.api.int64)
    elif fault == 'positions':
        case.values['model_inputs']['positions'] = case.tensor([0]*4, case.api.int64)
    elif fault == 'embedding_shape':
        case.values['model_inputs']['inputs_embeds'] = case.tensor([[0]*3]*4, case.api.bfloat16)
    elif fault == 'no_input':
        case.values['model_inputs']['inputs_embeds'] = None
    assert case.observed.inputs(case.values, case.cpu, 1, 1) is None
    assert case.observed.stopped and case.observed.errors
    assert not any(row[0] == 'query' for row in case.api.operations)


@pytest.mark.parametrize('fault', ['native', 'observer', 'missing_state', 'hidden_shape', 'hidden_dtype', 'event'])
def test_output_failure_never_materializes_native_async_output_or_waits(device_case, fault):
    case = device_case
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    native_error, errors = None, []
    if fault == 'native':
        native_error = RuntimeError
    elif fault == 'observer':
        errors = [dict(kind='trace_failure')]
    elif fault == 'missing_state':
        case.runner.execute_model_state = None
    elif fault == 'hidden_shape':
        case.runner.execute_model_state.hidden_states = case.tensor([[0]*3]*4, case.api.float32)
    elif fault == 'hidden_dtype':
        case.runner.execute_model_state.hidden_states = case.tensor([[0]*3]*8, case.api.bfloat16)
    elif fault == 'event':
        case.api.fail_event = True
    if fault in ('native', 'observer'):
        case.observed.output(index, native_error, errors)
        assert case.observed.entries[index]['status'] == 'UNVERIFIED_NATIVE_OR_OBSERVER_ERROR'
    else:
        with pytest.raises((ValueError, AttributeError)):
            case.observed.output(index, native_error, errors)
    assert not any(row[0] == 'query' for row in case.api.operations)


def test_finite_scan_covers_every_valid_hidden_element_and_preserves_nonfinite_evidence(device_case):
    case = device_case
    case.hidden.values[1, 1] = np.inf
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    case.observed.output(index, None, [])
    case.api.complete()
    records = []
    result = case.observed.drain(records.append)
    assert records[0]['rows'][0]['finite_elements_observed'] == 11
    assert result['status'] == 'UNVERIFIED'
    assert not case.observed.stopped


def test_device_drain_storage_failure_does_not_reuse_owned_destinations(device_case):
    case = device_case
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    case.observed.output(index, None, [])
    case.api.complete()
    def fail(row):
        raise OSError('injected exclusive-write failure')
    with pytest.raises(OSError, match='exclusive-write'):
        case.observed.drain(fail)
    assert case.observed.drained == 0 and case.observed.entries[0]['index'] == 0


def test_device_preparation_rejects_capture_and_actual_aggregate_allocation_overflow(device_case):
    case = device_case
    args = (case.runner, case.observed.sites, case.observed.limits, case.observed.requests,
            case.observed.sampling)
    with pytest.raises(ValueError, match='remaining rank'):
        observation.DeviceObservation(*args, 1)
    case.api.capturing = True
    with pytest.raises(ValueError, match='outside CUDA capture'):
        observation.DeviceObservation(*args, 65536)


def test_device_rpc_cap_is_checked_before_publication_or_drain_acceptance(device_case):
    case = device_case
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    case.observed.output(index, None, [])
    case.api.complete()
    case.observed.limits = replace(case.observed.limits, rpc_bytes=5)
    with pytest.raises(ValueError, match='RPC byte cap'):
        case.observed.drain(lambda row: pytest.fail('Over-cap RPC evidence cannot be published'))
    assert case.observed.drained == 0


def test_second_request_reuses_native_index_but_keeps_distinct_owned_device_snapshots(device_case):
    case = device_case
    index = case.observed.inputs(case.values, case.cpu, 1, 1)
    case.observed.output(index, None, [])
    case.api.complete()
    records = []
    first = case.observed.drain(records.append)
    case.api.host_reads_allowed = False
    case.runner.req_states.req_id_to_index = {'full-internal-B': 2}
    case.runner.req_states.index_to_req_id = {2: 'full-internal-B'}
    case.req.req_id = 'full-internal-B'
    case.req.prefill_token_ids = [31, 32, 33, 35]
    case.observed.registrations(SimpleNamespace(scheduled_new_reqs=[case.req]))
    case.cpu['req_ids'] = ['full-internal-B']
    case.values['input_batch'].input_ids.values[3] = 35
    case.hidden.values[:4] = 101
    index = case.observed.inputs(case.values, case.cpu, 2, 1)
    case.observed.output(index, None, [])
    case.api.complete()
    second = case.observed.drain(records.append)
    assert second['drained_steps'] == 2 and len(second['registered_requests']) == 2
    assert first['steps'][0]['values']['input_ids'][3] == 34
    assert second['steps'][0]['values']['input_ids'][3] == 34
    assert second['steps'][1]['values']['input_ids'][3] == 35
    assert second['steps'][1]['values']['idx_mapping'] == second['steps'][0]['values']['idx_mapping'] == [2]
    assert records[0]['rows'][0]['values']['hidden_states']['sha256'] != records[1]['rows'][0]['values']['hidden_states']['sha256']


def test_device_observation_callbacks_compose_existing_native_runner_without_sample_override(native, monkeypatch):
    observed, _, _ = runner(native)
    calls = []
    class Device:
        def registrations(self, scheduler):
            calls.append(('registration', scheduler))
        def inputs(self, values, batch, invocation, input_id, *, descriptor):
            calls.append(('inputs', invocation, input_id, descriptor['cg_mode']))
            return 0
        def output(self, index, native_error, errors, *, events):
            calls.append(('output', index, native_error, list(errors)))
        def error(self, kind, failure):
            calls.append(('error', kind, failure))
    observed.execute_model(work())
    observed._model_input_observation.device = Device()
    observed.execute_model(work())
    assert [row[0] for row in calls] == ['registration', 'inputs', 'output']
    assert calls[1][1:3] == (2, 1) and calls[1][3] == 'FULL'
    assert calls[2][1:] == (0, None, [])
    assert type(observed).sample_tokens is native.module.GPUModelRunner.sample_tokens
    assert observed.execution_calls == 2



def test_absolute_scan_scratch_matches_the_observed_hidden_dtype(device_case):
    case = device_case
    case.runner.model_config.hf_text_config.fp32_residual_connection = False
    observed = observation.DeviceObservation(case.runner, case.observed.sites, case.observed.limits,
        case.observed.requests, case.observed.sampling, 65536)
    assert observed.absolute.dtype is case.api.bfloat16
    assert observed.absolute.dtype is observed.hidden_dtype


@pytest.mark.parametrize('fault', ['allocation', 'event'])
def test_failed_device_allocation_keeps_preallocation_reservation(device_case, tmp_path, monkeypatch, fault):
    case = device_case
    output = tmp_path/'workers'; output.mkdir()
    node = tmp_path/'node.json'
    node.write_text(json.dumps(dict(observer_output=str(output), record_bytes=8*65536)))
    monkeypatch.setenv('STEP5_CAPACITY_NODE_FILE', str(node))
    observed = SimpleNamespace(device=None, sites=case.observed.sites, limits=case.observed.cpu_limits,
                               stopped=False)
    case.runner._model_input_observation = observed
    worker = SimpleNamespace(rank=0, local_rank=0, model_runner=case.runner,
        _capacity_identity=dict(rank=0, local_rank=0, pid=os.getpid()))
    allocations, original_empty = [], case.api.empty
    def allocate(*args, **kwargs):
        reservation = output/f'device_reservation_rank_0_pid_{os.getpid()}.json'
        assert json.loads(reservation.read_text())['owned_bytes'] == case.observed.owned_bytes
        allocations.append(args)
        if fault == 'allocation' and len(allocations) == 2:
            raise MemoryError('Injected partial allocation failure')
        return original_empty(*args, **kwargs)
    monkeypatch.setattr(case.api, 'empty', allocate)
    case.api.fail_event = fault == 'event'
    preparation = dict(limits=vars(case.observed.limits), requests=case.observed.requests,
                       sampling_fields=case.observed.sampling)
    result = execution.prepare_device_observation(worker, preparation)
    assert result['status'] == 'UNVERIFIED'
    assert worker._request_observation.device_owned_bytes == case.observed.owned_bytes
    assert worker._request_observation.collection_stopped and observed.stopped
    assert observed.device is None and len(allocations) >= 2
    assert len((json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n').encode()) <= case.observed.limits.rpc_bytes


@pytest.mark.parametrize('rpc_bytes', [5, 8])
def test_prepared_device_response_is_capped_after_allocation(device_case, tmp_path, monkeypatch, rpc_bytes):
    case = device_case
    output = tmp_path/'workers'; output.mkdir()
    node = tmp_path/'node.json'
    node.write_text(json.dumps(dict(observer_output=str(output), record_bytes=8*65536)))
    monkeypatch.setenv('STEP5_CAPACITY_NODE_FILE', str(node))
    observed = SimpleNamespace(device=None, sites=case.observed.sites, limits=case.observed.cpu_limits,
                               stopped=False)
    case.runner._model_input_observation = observed
    worker = SimpleNamespace(rank=0, local_rank=0, model_runner=case.runner,
        _capacity_identity=dict(rank=0, local_rank=0, pid=os.getpid()))
    limits = dict(vars(case.observed.limits), rpc_bytes=rpc_bytes)
    result = execution.prepare_device_observation(worker, dict(limits=limits,
        requests=case.observed.requests, sampling_fields=case.observed.sampling))
    assert result is None
    assert len((json.dumps(result)+'\n').encode()) <= limits['rpc_bytes']
    assert observed.device is not None and observed.device.stopped
    assert observed.stopped and worker._request_observation.collection_stopped
    assert worker._request_observation.device_owned_bytes == observed.device.owned_bytes


@pytest.mark.parametrize('rpc_bytes', [1, 2, 3, 4])
def test_device_limits_reject_unencodable_rpc_cap(rpc_bytes):
    with pytest.raises(ValueError, match='encode a JSON response'):
        observation.DeviceObservationLimits(steps=1, scan_rows=1, witness_rows=1,
                                            owned_bytes=4096, rpc_bytes=rpc_bytes)


CENSUS_SOURCE = """class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Step5MLP(Module):
    reduce_in_fp32 = True
    shard_sequence_parallel = False
    limit = None
    def forward(self, hidden_states):
        return hidden_states

class Step5Attention(Module):
    num_heads = 12
    num_kv_heads = 1
    head_dim = 192
    params_dtype = 'torch.bfloat16'
    def forward(self, positions, hidden_states):
        return hidden_states

class Step5RMSNorm(Module):
    variance_epsilon = 1e-6
    zero_centered = False
    def __init__(self):
        self.weight = Tensor()
    def forward(self, x, residual=None, output=None, fp16_out=False, *, out_dtype=None):
        return x

class GateLinear(Module):
    out_dtype = 'torch.float32'
    allow_ll_bf16_gemm = False
    allow_fp32_router_gemm = True
    allow_bf16x3_router_gemm = False
    allow_cublas_router_gemm = False
    def __init__(self):
        self.weight = Tensor()
    def forward(self, x):
        return x, None

class FusedMoEKernelModularImpl:
    is_dp_ep = False
    def _prepare(self, hidden_states, topk_weights, topk_ids, global_num_experts,
                 expert_map, apply_router_weight_on_input):
        return hidden_states, None, None, topk_ids, topk_weights

class TritonExperts:
    quant_dtype = 'torch.float8_e4m3fn'
    block_shape = [128, 64]
    per_act_token_quant = False
    quantization_emulation = False
    a1_scale = None
    a2_scale = None
    def __init__(self):
        self.w1_scale = Tensor(9010)
        self.w2_scale = Tensor(9020)
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
              global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
              workspace2, expert_tokens_meta, apply_router_weight_on_input):
        def _base_w13_fn():
            return invoke_fused_moe_triton_kernel(hidden_states, w1, output,
                a1q_scale, self.w1_scale, None, topk_ids, topk_ids, topk_ids,
                False, 8, {'BLOCK_SIZE_M': 16}, 'bf16', self.quant_dtype is not None, False, False,
                False, False, self.block_shape)
        _base_w13_fn()
        def _base_w2_fn():
            return invoke_fused_moe_triton_kernel(hidden_states, w2, output,
                a2_scale, self.w2_scale, topk_weights, topk_ids, topk_ids, topk_ids,
                True, 1, {'BLOCK_SIZE_M': 16}, 'bf16', self.quant_dtype is not None, False, False,
                False, False, self.block_shape)
        return _base_w2_fn()

def invoke_fused_moe_triton_kernel(A, B, C, A_scale, B_scale, topk_weights,
    sorted_token_ids, expert_ids, num_tokens_post_padded, mul_routed_weight,
    top_k, config, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16,
    use_int4_w4a16, per_channel_quant, block_shape=None, B_bias=None):
    return C

class Fp8MoEMethod:
    weight_block_size = [128, 128]
    moe_block_shape = [128, 64]
    weight_scale_refine = (1, 2)
    weight_scale_name = 'weight_scale_inv'
    fp8_backend = 'TRITON'
    def __init__(self):
        self.moe_kernel = SimpleNamespace(impl=FusedMoEKernelModularImpl(),
            fused_experts=TritonExperts(), prepare_finalize=SimpleNamespace())
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        a, scale, meta, ids, weights = self.moe_kernel.impl._prepare(
            x, topk_weights, topk_ids, 352, None, False)
        return self.moe_kernel.fused_experts.apply(x, a, layer.w13_weight,
            layer.w2_weight, weights, ids, None, 352, None, scale, None,
            x, x, meta, False)

class UnquantizedFusedMoEMethod:
    unquantized_backend = 'TRITON'
    def __init__(self):
        experts = TritonExperts()
        experts.quant_dtype, experts.block_shape = None, None
        experts.w1_scale, experts.w2_scale = None, None
        self.moe_kernel = SimpleNamespace(impl=FusedMoEKernelModularImpl(),
            fused_experts=experts, prepare_finalize=SimpleNamespace())
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return self.forward_native(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
    def forward_native(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return self.moe_kernel.fused_experts.apply(x, x, layer.w13_weight,
            layer.w2_weight, topk_weights, topk_ids, None, 352, None, None, None,
            x, x, None, False)

class RoutedExperts:
    global_num_experts = 352
    local_num_experts = 352
    intermediate_size_per_partition = 64
    w13_input_scale = None
    w2_input_scale = None
    expert_map = None
    def __init__(self):
        self.quant_method = Fp8MoEMethod()
        self.w13_weight, self.w2_weight = Tensor(8010), Tensor(8020)
        self.w13_weight_scale_inv = self.quant_method.moe_kernel.fused_experts.w1_scale
        self.w2_weight_scale_inv = self.quant_method.moe_kernel.fused_experts.w2_scale
    def forward_modular(self, x, topk_weights, topk_ids,
                        shared_experts=None, shared_experts_input=None):
        return self.quant_method.apply(self, x, topk_weights, topk_ids,
                                       shared_experts, shared_experts_input)

class MoERunner:
    def __init__(self):
        self.routed_experts = RoutedExperts()
    def _apply_quant_method(self, hidden_states, router_logits,
        shared_experts_input, input_ids=None, shared_experts_overlapping=False):
        return None, self.routed_experts.forward_modular(hidden_states, router_logits, Tensor())

class Step5MoE(Module):
    n_routed_experts = 352
    n_local_physical_experts = 352
    fuse_all_reduce = True
    use_mega_moe = False
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.share_expert, self.gate, self.experts = Step5MLP(), GateLinear(), MoERunner()
    def forward(self, hidden_states, padding=None):
        shared = self.share_expert(hidden_states)
        logits, _ = self.gate(hidden_states)
        _, routed = self.experts._apply_quant_method(hidden_states, logits, None)
        return shared, routed

class Step5DecoderLayer(Module):
    fp32_residual_connection = True
    use_fused_all_reduce = True
    use_sequence_parallel = False
    def __init__(self, layer_idx, use_moe):
        self.layer_idx, self.use_moe = layer_idx, use_moe
        self.input_layernorm, self.post_attention_layernorm = Step5RMSNorm(), Step5RMSNorm()
        self.self_attn = Step5Attention()
        if use_moe:
            self.moe = Step5MoE(layer_idx)
        else:
            self.mlp = Step5MLP()
        self.calls, self.failure = 0, None
    def _forward_ffn(self, hidden_states, padding=None):
        return self.moe(hidden_states, padding)[1] if self.use_moe else self.mlp(hidden_states)
    def forward(self, positions, hidden_states, residual=None, padding=None):
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        x = self.input_layernorm(hidden_states, residual)
        x = self.self_attn(positions, x)
        x = self.post_attention_layernorm(x, hidden_states)
        return self._forward_ffn(x, padding), hidden_states

class Step5Model(Module):
    fp32_residual_connection = True
    use_sequence_parallel = False
    def __init__(self, choices=(False, True)):
        self.start_layer, self.end_layer = 0, len(choices)
        self.layers = [Step5DecoderLayer(i, choice) for i, choice in enumerate(choices)]
        self.calls = 0
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        self.calls += 1
        hidden_states, residual = inputs_embeds, None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual)  # CENSUS_LAYER
        return hidden_states
    def named_modules(self):
        yield 'language_model.model', self
        for i, layer in enumerate(self.layers):
            path = 'language_model.model.layers.'+str(i)
            yield path, layer
            yield path+'.input_layernorm', layer.input_layernorm
            yield path+'.post_attention_layernorm', layer.post_attention_layernorm
            yield path+'.self_attn', layer.self_attn
            if layer.use_moe:
                yield path+'.moe', layer.moe
                yield path+'.moe.share_expert', layer.moe.share_expert
                yield path+'.moe.gate', layer.moe.gate
                yield path+'.moe.experts', layer.moe.experts
                yield path+'.moe.experts.routed_experts', layer.moe.experts.routed_experts
            else:
                yield path+'.mlp', layer.mlp
"""


@pytest.fixture
def census(native, monkeypatch, tmp_path):
    path = tmp_path/'cpu_census_fixture.py'
    path.write_text(CENSUS_SOURCE)
    module = ModuleType('cpu_census_fixture_'+str(id(path)))
    module.__file__ = str(path)
    module.Tensor, module.SimpleNamespace = native.module.Tensor, SimpleNamespace
    sys.modules[module.__name__] = module
    exec(compile(CENSUS_SOURCE, str(path), 'exec', dont_inherit=True), module.__dict__)
    sources = {name: (module.__name__, hashlib.sha256(path.read_bytes()).hexdigest())
               for name in observation.CENSUS_SOURCES}
    calls = {name: row for name, row in observation.CENSUS_CALLS.items()
             if len(row[1]) == 1 and hasattr(module, row[1][0]) or
                len(row[1]) == 2 and hasattr(module, row[1][0]) and hasattr(getattr(module, row[1][0]), row[1][1])}
    monkeypatch.setattr(observation, 'CENSUS_SOURCES', sources)
    monkeypatch.setattr(observation, 'CENSUS_CALLS', calls)
    monkeypatch.setattr(observation, 'CENSUS_LAYER_LINE', next(i for i, line in
        enumerate(CENSUS_SOURCE.splitlines(), 1) if '# CENSUS_LAYER' in line))
    yield SimpleNamespace(module=module, path=path, native=native)
    sys.modules.pop(module.__name__, None)


def census_runner(census, mode='NONE', *, limits=observation.ObservationLimits(), choices=(False, True)):
    observed, graph, records = runner(census.native, mode, limits=limits)
    observed.vllm_config.model_inputs.pop('override')
    observed.model = census.module.Step5Model(choices)
    return observed, graph, records


def test_native_layer_paths_dense_shared_router_and_real_gemm_owners(census):
    observed, _, _ = census_runner(census)
    result = observed.execute_model(work())
    assert result is observed.vllm_config.model_inputs['inputs_embeds']
    assert [layer.calls for layer in observed.model.layers] == [1, 1]
    calls = events(observed, 'native_call')
    assert record(observed)['call_census']['returned_layer_ids'] == [0, 1]
    assert record(observed)['call_census']['physical_kernel_census'] == 'UNMEASURED'
    assert all(row['request_frontiers'] is None for row in calls)
    assert all(row['input_observation'] == events(observed, 'finalized_model_inputs')[0]['event'] for row in calls)
    assert len([row for row in calls if row['site'] == 'mlp']) == 2
    gemms = [row for row in calls if row['site'] == 'triton_gemm']
    assert [row['gemm'] for row in gemms] == ['GEMM1', 'GEMM2']
    tensors = {row['tensor_id']: row for row in record(observed)['call_tensor_metadata']}
    assert [tensors[row['arguments']['B_scale']['tensor_id']]['data_ptr'] for row in gemms] == [9010, 9020]
    assert all(row['arguments']['A_scale'] is None and row['layer_id'] == 1 for row in gemms)
    inventory = events(observed, 'resident_model_bindings')[0]
    owner = next(row for row in inventory['owners'] if row['module_path'].endswith('routed_experts'))
    assert owner['attributes']['w13_input_scale'] is None
    assert owner['attributes']['w13_weight_scale_inv']['data_ptr'] == 9010
    assert [row['layer_id'] for row in inventory['layers']] == [0, 1]
    assert len(events(observed, 'native_layer_dispatch')) == 2
    assert observed._model_input_observation.sites.sources[str(census.path)]['sha256'] == hashlib.sha256(census.path.read_bytes()).hexdigest()


def test_full_replay_does_not_receive_python_layer_coverage(census):
    observed, graph, _ = census_runner(census, 'FULL')
    observed.execute_model(work())
    assert graph.calls == 1 and observed.model.calls == 0
    summary = record(observed)['call_census']
    assert summary['returned_layer_ids'] == []
    assert summary['layer_coverage'] == 'FULL_REPLAY_BYPASSES_PYTHON_LAYER_FRAMES'
    assert summary['physical_kernel_census'] == 'UNMEASURED'
    assert summary['transient_scale_content'] == 'UNMEASURED'


def test_capture_excludes_warmup_but_records_actual_full_owner(census):
    observed, _, records = census_runner(census, 'FULL')
    observed.capture_model()
    capture = records[-1]
    calls = [row for row in capture['events'] if row['kind'] == 'native_call']
    assert observed.model.calls == 2 and len([row for row in calls if row['site'] == 'backbone']) == 1
    binding = next(row for row in capture['events'] if row['kind'] == 'capture_forward' and row['native_capture_region'])
    assert all(row['capture_binding_event'] == binding['event'] for row in calls)
    assert capture['call_census']['excluded_capture_calls']['backbone'] == 1
    assert capture['call_census']['G2'] == 'UNVERIFIED'


@pytest.mark.parametrize('fault', ['hash', 'loaded_code', 'layer_line', 'missing_import'])
def test_census_binding_failure_preserves_native_execution_with_an_explicit_gap(census, monkeypatch, fault):
    observed, _, _ = census_runner(census)
    if fault == 'hash':
        census.path.write_text(CENSUS_SOURCE+'\n# different retained source\n')
    elif fault == 'loaded_code':
        function = census.module.Step5DecoderLayer.forward
        monkeypatch.setattr(function, '__code__', function.__code__.replace(co_firstlineno=1))
    elif fault == 'layer_line':
        monkeypatch.setattr(observation, 'CENSUS_LAYER_LINE', 999999)
    else:
        monkeypatch.delitem(sys.modules, census.module.__name__)
    result = observed.execute_model(work())
    assert result is observed.vllm_config.model_inputs['inputs_embeds']
    assert observed.model.calls == 1
    assert record(observed)['call_census']['source_gaps']
    assert record(observed)['call_census']['G2'] == 'UNVERIFIED'
    if fault == 'missing_import':
        monkeypatch.setitem(sys.modules, census.module.__name__, census.module)


def test_native_exception_and_hook_restoration_do_not_become_returned_layers(census):
    observed, _, _ = census_runner(census)
    failure = RuntimeError('native layer fails')
    observed.model.layers[1].failure = failure
    profile, trace = sys.getprofile(), sys.gettrace()
    with pytest.raises(RuntimeError) as caught:
        observed.execute_model(work())
    assert caught.value is failure
    assert sys.getprofile() is profile and sys.gettrace() is trace
    assert observed.model.calls == 1 and [layer.calls for layer in observed.model.layers] == [1, 1]
    layer_calls = [row for row in events(observed, 'native_call') if row['site'] == 'layer']
    assert layer_calls[1]['outcome'] == 'UNVERIFIED_NATIVE_UNWIND'
    assert record(observed)['call_census']['returned_layer_ids'] == [0]


def test_call_return_growth_is_charged_and_cannot_mutate_a_full_record(census):
    observed, _, _ = census_runner(census)
    state = observation.ModelInputObservation(census.native.sites,
        observation.ObservationLimits(scope_bytes=700), execution.batch_snapshot, lambda row: None)
    scope = observation.ObservationScope(observed, state, 'execute')
    row = scope.emit('native_call', frontiers=False, outcome='CALL_ENTERED_NOT_COMPLETED')
    before, used = dict(row), scope.used
    with pytest.raises(ValueError, match='call return'):
        scope.update(row, result='x'*1000, outcome='RETURNED_CPU_CALL_NOT_GPU_COMPLETION')
    assert row == before and scope.used == used
    scope.update(row, result=None, outcome='RETURNED_CPU_CALL_NOT_GPU_COMPLETION')
    assert scope.used == sum(len(json.dumps(item, allow_nan=False, separators=(',', ':')).encode()) for item in scope.events)


@pytest.mark.parametrize('limit', ['scope_events', 'scope_bytes', 'metadata_items'])
def test_census_limits_stop_only_observation(census, limit):
    limits = observation.ObservationLimits(**{limit: 1})
    observed, _, _ = census_runner(census, limits=limits)
    result = observed.execute_model(work())
    assert result is observed.vllm_config.model_inputs['inputs_embeds']
    assert observed.model.calls == 1 and [layer.calls for layer in observed.model.layers] == [1, 1]
    state = observed._model_input_observation
    assert state.stopped and state.summary()['status'] == 'UNVERIFIED'
    if limit == 'scope_bytes':
        assert state.last_record is None and state.last_summary is None
    else:
        assert record(observed)['status'] == 'UNVERIFIED'


@pytest.mark.parametrize('kind', ['execute', 'capture'])
def test_complete_scope_envelope_exhaustion_cannot_reach_a_writer(census, kind):
    observed, _, _ = census_runner(census)
    state = observation.ModelInputObservation(census.native.sites,
        observation.ObservationLimits(scope_bytes=1200), execution.batch_snapshot, lambda row: None)
    state.sites.census_bound = True
    state.sites.sources = {'cpu-fixture-\u6c49': dict(sha256='x'*2000, code_owners=[])}
    written = []
    state.capture_record = written.append
    scope = observation.ObservationScope(observed, state, kind)
    row = scope.emit('fixture_only', frontiers=False)
    assert scope.used < state.limits.scope_bytes
    assert scope.__exit__(None, None, None) is False
    assert state.stopped and state.last_summary['status'] == 'UNVERIFIED'
    assert state.last_record is None or len((json.dumps(state.last_record,
        allow_nan=False, separators=(',', ':'))+'\n').encode()) <= state.limits.scope_bytes
    assert state.last_record is None or row not in state.last_record['events']
    assert all(len((json.dumps(item, allow_nan=False, separators=(',', ':'))+'\n').encode())
               <= state.limits.scope_bytes for item in written)


@pytest.mark.parametrize('scope_bytes', [1, 2, 3, 4])
def test_unencodable_scope_failure_is_omitted_without_a_capture_write(census, scope_bytes):
    observed, _, _ = census_runner(census)
    written = []
    state = observation.ModelInputObservation(census.native.sites,
        observation.ObservationLimits(scope_bytes=scope_bytes), execution.batch_snapshot, written.append)
    scope = observation.ObservationScope(observed, state, 'capture')
    assert scope.__exit__(None, None, None) is False
    assert state.last_record is None and written == [] and state.stopped
    assert state.last_summary is None
    assert state.summary()['collection_stopped'] is True
    assert state.summary()['status'] == 'UNVERIFIED'


def test_native_return_classifier_disassembles_each_observed_code_once(census, monkeypatch):
    observed, _, _ = census_runner(census)
    original = observation.dis.get_instructions
    counts = {}
    def inspected(code, *args, **kwargs):
        counts[code] = counts.get(code, 0)+1
        return original(code, *args, **kwargs)
    monkeypatch.setattr(observation.dis, 'get_instructions', inspected)
    observed.execute_model(work())
    code = census.module.Step5DecoderLayer.forward.__code__
    assert [layer.calls for layer in observed.model.layers] == [1, 1]
    assert counts[code] == 1
    assert record(observed)['call_census']['returned_layer_ids'] == [0, 1]


@pytest.mark.parametrize('kind', ['execute', 'capture'])
@pytest.mark.parametrize('offset', [-1, 0])
def test_complete_scope_record_accepts_only_the_exact_json_boundary(census, kind, offset):
    observed, _, _ = census_runner(census)
    state = observation.ModelInputObservation(census.native.sites,
        observation.ObservationLimits(), execution.batch_snapshot, lambda row: None)
    state.sites.census_bound = True
    state.sites.sources = {'cpu-fixture-\u6c49': dict(sha256='a'*64, code_owners=[])}
    scope = observation.ObservationScope(observed, state, kind)
    scope.emit('fixture_only', frontiers=False, value='\u6c49')
    scope.__exit__(None, None, None)
    baseline = state.last_record
    size = len((json.dumps(baseline, allow_nan=False, separators=(',', ':'))+'\n').encode())
    state.limits = observation.ObservationLimits(scope_bytes=size+offset)
    written = []
    state.capture_record = written.append
    scope = observation.ObservationScope(observed, state, kind)
    scope.emit('fixture_only', frontiers=False, value='\u6c49')
    scope.__exit__(None, None, None)
    actual = state.last_record
    assert len((json.dumps(actual, allow_nan=False, separators=(',', ':'))+'\n').encode()) <= state.limits.scope_bytes
    if offset == 0:
        assert actual == baseline and state.failures == 0 and not state.stopped
    else:
        assert actual['status'] == 'UNVERIFIED' and actual['events'] == [] and state.stopped
        assert actual['observer_errors'] == [dict(kind='scope_record_bytes')]
    assert written == ([actual] if kind == 'capture' else [])


@pytest.mark.parametrize('kind', ['execute', 'capture'])
@pytest.mark.parametrize('native_raises', [False, True])
def test_scope_byte_exhaustion_preserves_native_outcome_and_hooks(census, kind, native_raises):
    limits = observation.ObservationLimits(scope_bytes=1 if not native_raises else 1200)
    observed, _, records = census_runner(census, limits=limits)
    failure = RuntimeError('native fixture failure')
    if native_raises:
        observed.model.layers[1].failure = failure
    profile, trace = sys.getprofile(), sys.gettrace()
    action = observed.execute_model if kind == 'execute' else observed.capture_model
    args = (work(),) if kind == 'execute' else ()
    if native_raises:
        with pytest.raises(RuntimeError) as caught:
            action(*args)
        assert caught.value is failure
    else:
        result = action(*args)
        assert result is observed.vllm_config.model_inputs['inputs_embeds'] if kind == 'execute' else result == 41
    assert sys.getprofile() is profile and sys.gettrace() is trace
    state = observed._model_input_observation
    assert state.stopped and state.summary()['status'] == 'UNVERIFIED'
    if native_raises:
        assert state.last_record['native_outcome'] == 'raised'
        assert state.last_record['native_error_type'] == 'builtins.RuntimeError'
    else:
        assert state.last_record is None and state.last_summary is None and records == []
    expected_calls = 2 if kind == 'capture' and not native_raises else 1
    assert observed.model.calls == expected_calls
    assert (observed.execution_calls if kind == 'execute' else observed.capture_calls) == 1


@pytest.mark.parametrize('fail_writer', [False, True])
@pytest.mark.parametrize('near_limit', [False, True])
def test_device_output_precedes_final_record_cap_and_capture_writer(census, fail_writer, near_limit):
    observed, _, _ = census_runner(census)
    state = observation.ModelInputObservation(census.native.sites,
        observation.ObservationLimits(), execution.batch_snapshot, lambda row: None)
    order = []
    class Device:
        def output(self, index, native_error, observer_errors, *, events):
            assert index == 3 and native_error is None and observer_errors == []
            order.append('device_output')
        def error(self, *args):
            raise AssertionError('No device observer error expected')
    state.device = Device()
    scope = observation.ObservationScope(observed, state, 'capture')
    scope.device_step = 3
    scope.__exit__(None, None, None)
    size = len((json.dumps(state.last_record, allow_nan=False, separators=(',', ':'))+'\n').encode())
    if near_limit:
        state.limits = observation.ObservationLimits(scope_bytes=size+1)
    order.clear()
    written = []
    def writer(row):
        assert order == ['device_output']
        assert len((json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n').encode()) <= state.limits.scope_bytes
        written.append(json.loads(json.dumps(row)))
        order.append('capture_writer')
        if fail_writer:
            raise KeyboardInterrupt('x'*10000)
    state.capture_record = writer
    scope = observation.ObservationScope(observed, state, 'capture')
    scope.device_step = 3
    assert scope.__exit__(None, None, None) is False
    assert order == ['device_output', 'capture_writer']
    assert state.last_summary['status'] == state.last_record['status']
    if fail_writer:
        assert state.last_record['status'] == 'UNVERIFIED' and state.stopped
        expected_error = 'scope_record_bytes' if near_limit else 'capture_record'
        assert state.last_record['observer_errors'][0]['kind'] == expected_error
        assert written[0]['observer_errors'] == []
    else:
        state.limits = observation.ObservationLimits(scope_bytes=size-1)
        order.clear()
        scope = observation.ObservationScope(observed, state, 'capture')
        scope.device_step = 3
        scope.__exit__(None, None, None)
        assert order == ['device_output', 'capture_writer']
        assert state.last_record['events'] == [] and state.last_record['status'] == 'UNVERIFIED'
    assert len((json.dumps(state.last_record, allow_nan=False, separators=(',', ':'))+'\n').encode()) <= state.limits.scope_bytes


@pytest.mark.parametrize('offset', [-1, 0])
def test_rejected_record_itself_obeys_the_same_encoded_byte_cap(census, offset):
    observed, _, _ = census_runner(census)
    state = observation.ModelInputObservation(census.native.sites,
        observation.ObservationLimits(scope_bytes=1000), execution.batch_snapshot, lambda row: None)
    oversized = dict(native_outcome='returned', native_error_type=None, content='x'*2000)
    rejected = observation.ObservationScope(observed, state, 'capture').bounded_record(oversized)
    size = len((json.dumps(rejected, allow_nan=False, separators=(',', ':'))+'\n').encode())
    state.limits = observation.ObservationLimits(scope_bytes=size+offset)
    actual = observation.ObservationScope(observed, state, 'capture').bounded_record(oversized)
    assert actual == rejected if offset == 0 else actual is None
    assert state.stopped and state.summary()['status'] == 'UNVERIFIED'


def test_full_model_size_cpu_evidence_fits_existing_scope_cap(census):
    choices = tuple(i >= 3 and i < 91 for i in range(92))
    observed, _, _ = census_runner(census, choices=choices)
    observed.execute_model(work())
    result = record(observed)
    assert not result['observer_errors']
    assert result['call_census']['returned_layer_ids'] == list(range(92))
    encoded = len((json.dumps(result, allow_nan=False, separators=(',', ':'))+'\n').encode())
    assert encoded <= observed._model_input_observation.limits.scope_bytes
    assert all(layer.calls == 1 for layer in observed.model.layers)


def test_call_metadata_reuses_only_complete_equal_descriptors(census):
    observed, _, _ = census_runner(census)
    state = observation.ModelInputObservation(census.native.sites,
        observation.ObservationLimits(), execution.batch_snapshot, lambda row: None)
    scope = observation.ObservationScope(observed, state, 'execute')
    metadata = observation.Metadata(census.native.sites, state.limits,
                                     tensor_record=scope.tensor_reference)
    first = census.native.module.Tensor(7000)
    assert metadata.value(first) == metadata.value(census.native.module.Tensor(7000))
    changed = census.native.module.Tensor(7000)
    changed.dtype = 'torch.bfloat16'
    assert metadata.value(changed) != metadata.value(first)
    assert len(scope.call_tensors) == 2
    assert all(row['content'] == 'UNMEASURED' for row in scope.call_tensors)
    assert scope.used == sum(len(json.dumps(row, allow_nan=False, separators=(',', ':')).encode())
                             for row in scope.call_tensors)


def test_call_tensor_pool_exhaustion_preserves_existing_descriptors(census):
    observed, _, _ = census_runner(census)
    state = observation.ModelInputObservation(census.native.sites,
        observation.ObservationLimits(scope_bytes=1), execution.batch_snapshot, lambda row: None)
    scope = observation.ObservationScope(observed, state, 'execute')
    metadata = observation.Metadata(census.native.sites, state.limits,
                                     tensor_record=scope.tensor_reference)
    with pytest.raises(ValueError, match='tensor metadata'):
        metadata.value(census.native.module.Tensor())
    assert scope.call_tensors == [] and scope.tensor_keys == {} and scope.used == 0


def test_unquantized_resident_method_and_no_scale_dispatch_are_observed(census):
    observed, _, _ = census_runner(census)
    routed = observed.model.layers[1].moe.experts.routed_experts
    routed.quant_method = census.module.UnquantizedFusedMoEMethod()
    routed.w13_weight_scale_inv, routed.w2_weight_scale_inv = None, None
    observed.execute_model(work())
    calls = events(observed, 'native_call')
    assert [row['site'] for row in calls if row['site'] in ('fp8_apply', 'unquantized_apply')] == ['unquantized_apply']
    assert all(row['arguments']['B_scale'] is None and row['arguments']['use_fp8_w8a8'] is False
               for row in calls if row['site'] == 'triton_gemm')
    assert record(observed)['call_census']['returned_layer_ids'] == [0, 1]


def test_same_code_on_an_unbound_layer_cannot_inherit_resident_layer_credit(census):
    observed, _, _ = census_runner(census)
    observed.execute_model(work())
    observed.model.layers[1] = census.module.Step5DecoderLayer(1, True)
    observed.execute_model(work())
    layer_calls = [row for row in events(observed, 'native_call') if row['site'] == 'layer']
    assert layer_calls[1]['owner_binding'] == 'UNVERIFIED_OWNER_NOT_RESIDENT'
    assert layer_calls[1]['layer_id'] is None
    assert record(observed)['call_census']['returned_layer_ids'] == [0]
    assert record(observed)['call_census']['layer_coverage'] == 'UNVERIFIED_INCOMPLETE_PYTHON_LAYER_RETURNS'


def test_inherited_original_native_method_binds_to_actual_subclass_owner(census):
    observed, _, _ = census_runner(census)
    class DerivedLayer(census.module.Step5DecoderLayer):
        pass
    observed.model.layers[1] = DerivedLayer(1, True)
    observed.execute_model(work())
    assert record(observed)['call_census']['returned_layer_ids'] == [0, 1]
    owner = next(row for row in events(observed, 'resident_model_bindings')[0]['owners']
                 if row['module_path'] == 'language_model.model.layers.1')
    assert owner['cls'].endswith('DerivedLayer')


def test_duplicate_resident_layer_ids_remain_an_explicit_source_gap(census):
    observed, _, _ = census_runner(census)
    observed.model.layers[1].layer_idx = 0
    observed.execute_model(work())
    assert any(row['reason'] == 'DUPLICATE_RESIDENT_LAYER_IDS'
               for row in record(observed)['call_census']['source_gaps'])
    assert record(observed)['call_census']['G2'] == 'UNVERIFIED'
