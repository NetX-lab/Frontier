"""Source-pinned MRv2 metadata observations; device execution remains unqualified.

The scoped hooks read native frames and actual replay callees. They never replace
runner, graph, sampler, or capture functions. All records are CPU metadata, even
when a hook is delivered during native GPU capture. Hook delivery, compilation
compatibility, device contents, finite values, and completion require a later
separately frozen native request run.
"""
from dataclasses import dataclass, field
import copy
import dis
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import threading
from types import CodeType, SimpleNamespace
import weakref


SOURCE_PINS = {
    'worker': 'dc0c35412670477f73200f935a761091fca455489fd21c665afd95e05b1b8f7e',
    'runner': '71642a34ca3cee5954ba8834916807810a2252d5e749151369fa31f5bf943624',
    'graphs': 'dfa356ffe5b88656945de785945115cc69b99578e77046101dc2611ba89f707d',
    'piecewise': '6e34824d2e62cd3bbd2eaee615ebfae3db542ce2d6be9b14d1b012398087fe2e',
    'breakable': 'c2ebbdcaf9b44e5fd2c9790399e7f6d4840425d66ae7eec138aa320d37d48d00',
    'timing': '1797a8abdb46bba9d30d0507d725796ec72ad836f6778a1ece4590fd74b911c9',
}
SOURCE_PATHS = {
    'worker': 'vllm/v1/worker/gpu_worker.py',
    'runner': 'vllm/v1/worker/gpu/model_runner.py',
    'graphs': 'vllm/v1/worker/gpu/cudagraph_utils.py',
    'piecewise': 'vllm/compilation/cuda_graph.py',
    'breakable': 'vllm/compilation/breakable_cudagraph.py',
    'timing': 'vllm/v1/worker/gpu/async_utils.py',
}
# Each line is a statement in the pinned original code, never a guessed table key.
SOURCE_LINES = dict(final_inputs=2022, full_replay=563, full_capture_store=508,
                    pw_replay=361, pw_capture=320, pw_capture_store=340,
                    break_dispatch=349, break_capture_dispatch=348, break_replay=434, break_segment=217,
                    break_capture=394, break_capture_store=408, full_capture_forward=497)
NATIVE_READINESS_CHECKS = (
    'Verify installed source hashes and loaded code objects on every actual worker.',
    'Verify finalized kwargs and actual replay callee delivery in the native worker thread.',
    'Compare unchanged compilation, capture, graph policy and sampling with hooks enabled.',
    'Verify coexistence with installed profiler and trace hooks and restoration on native errors.',
    'Join internal request IDs and CPU frontiers to accepted public inputs and completed outputs.',
    'Add separately verified device-content, finite, stream-completion and physical-scope witnesses.',
    'Freeze observer bounds and aggregate rank/node archive budgets for the distinct request run.',
)
UNMEASURED = dict.fromkeys(('device_inputs', 'content', 'finite', 'gpu_completion',
                          'physical_scope_census', 'public_request_join', 'G2'), 'UNMEASURED')


def class_name(value):
    return type(value).__module__+'.'+type(value).__qualname__


def nested_code(code, *names):
    for name in names:
        matches = [value for value in code.co_consts
                   if isinstance(value, CodeType) and value.co_name == name]
        if len(matches) != 1:
            raise ValueError('Native code owner is absent or ambiguous: '+name)
        code = matches[0]
    return code


def bind_code(function, owner, pin, sources):
    """Bind loaded original bytecode to the compiled, hash-pinned source without executing it."""
    original = inspect.unwrap(function)
    code = original.__code__
    path = Path(inspect.getsourcefile(original))
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != pin:
        raise ValueError('Native source hash differs: '+str(path))
    compiled = compile(body, code.co_filename, 'exec', dont_inherit=True,
                       optimize=sys.flags.optimize)
    expected = nested_code(compiled, *owner)
    if code != expected:
        raise ValueError('Loaded native code differs from pinned source: '+'.'.join(owner))
    row = sources.setdefault(str(path), dict(sha256=digest, code_owners=[]))
    row['code_owners'].append(dict(name='.'.join(owner), first_line=code.co_firstlineno))
    return code


@dataclass(frozen=True)
class ObservationLimits:
    """Explicit CPU evidence caps; exhaustion stops observations, never native work."""
    scope_bytes: int = 4*1024**2
    scope_events: int = 4096
    metadata_items: int = 4096
    metadata_depth: int = 6
    tracked_entries: int = 4096

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError('Observation limits must be positive integers')


FRONTEND_SOURCE_PINS = dict(
    engine='b3a9df98e2f0f1cd29fcbb7b079c292f680902a9b578d0d1002015a9a7d6cf15',
    output='98919ecea8e40d70fb08e13ae69ece4b9a7ad7ede56169aeb3b45b76dd87208f')
FRONTEND_SOURCE_LINES = dict(registration=297, registered_stats=642)


@dataclass
class AdmissionSites:
    codes: dict
    lines: dict
    sources: dict

    @classmethod
    def bind(cls, engine, *, pins=FRONTEND_SOURCE_PINS, lines=FRONTEND_SOURCE_LINES):
        """Bind original frontend methods on the actual resident engine and processor."""
        sources = {}
        processor = engine.output_processor
        codes = dict(engine=bind_code(engine.add_request, ('LLMEngine', 'add_request'),
                                      pins['engine'], sources),
                     registration=bind_code(processor.add_request, ('OutputProcessor', 'add_request'),
                                            pins['output'], sources),
                     stats=bind_code(processor._update_admission_stats,
                                     ('OutputProcessor', '_update_admission_stats'),
                                     pins['output'], sources))
        for name, line in (('engine', 'registration'), ('registration', 'registered_stats')):
            if lines[line] not in {number for _, number in dis.findlinestarts(codes[name])}:
                raise ValueError('Pinned frontend statement is absent: '+line)
        return cls(codes, dict(lines), sources)


class AdmissionScope:
    """Detach one original n=1 text registration before native EngineCore dispatch.

    This scope observes Python call boundaries only. It preserves the native
    generate path and existing profile hook, and performs no device operation,
    evidence write, or synchronization while native code executes.
    """
    def __init__(self, engine, sites, context, submitted, sampling_fields, identity,
                 *, limits=ObservationLimits()):
        self.engine, self.sites = engine, sites
        self.processor = engine.output_processor
        self.context, self.submitted = copy.deepcopy(context), submitted
        self.sampling_fields, self.identity = copy.deepcopy(sampling_fields), dict(identity)
        self.limits, self.records, self.errors = limits, [], []
        self.engine_frame, self.registration_frame, self.pending = None, None, None
        self.engine_calls, self.registration_calls = 0, 0
        self.active, self.installed, self.stopped = False, False, False
        self.previous_profile = sys.getprofile()
        self.profile_hook = self.profile
        self.opcodes = {code: {instruction.offset: instruction.opname
                             for instruction in dis.get_instructions(code)}
                        for code in sites.codes.values()}
        self.record = None

    def error(self, kind, failure):
        self.stopped = True
        if not self.errors:
            self.errors.append(dict(kind=kind, error_type=class_name(failure), error=str(failure)))

    def returned(self, frame):
        return self.opcodes[frame.f_code].get(frame.f_lasti) in ('RETURN_VALUE', 'RETURN_CONST')

    def registered(self, frame):
        values = frame.f_locals
        request, state = values['request'], values['req_state']
        params = request.sampling_params
        if self.pending is not None:
            raise ValueError('Duplicate native registration boundary')
        if (values['parent_req'] is not None or values['request_index'] != 0 or
                request.prompt_embeds is not None or request.pooling_params is not None or
                request.resumable or params is None or params.n != 1):
            raise ValueError('Admission observation supports one non-streaming text completion only')
        internal, external = request.request_id, request.external_req_id
        if any(type(value) is not str or not value for value in (internal, external)):
            raise ValueError('Both original native request IDs are required')
        if (self.processor.request_states.get(internal) is not state or
                state.request_id != internal or state.external_req_id != external or
                state.parent_req is not None or state.request_index != 0):
            raise ValueError('Original registered request and RequestState identities disagree')
        mapping = list(self.processor.external_req_ids[external])
        if mapping != [internal]:
            raise ValueError('Original n=1 external/internal mapping is absent or ambiguous')
        tokens = list(request.prompt_token_ids)
        if (tokens != self.context['prompt_token_ids'] or
                list(state.prompt_token_ids) != tokens or state.prompt_embeds is not None):
            raise ValueError('Original registered text prompt differs from the active manifest')
        resolved = {name: copy.deepcopy(getattr(params, name)) for name in self.sampling_fields}
        parent = frame.f_back
        if (parent.f_locals['request'] is not request or parent.f_locals['params'] is not params or
                parent.f_locals['request_id'] != external):
            raise ValueError('Original LLMEngine request and resolved parameters disagree')
        row = dict(kind='native_frontend_registration', native_registration_outcome='PENDING_RETURN',
            engine_core_submission='UNMEASURED', public_request_id=external, internal_request_id=internal,
            external_internal_mapping=mapping, prompt_token_ids=tokens,
            resolved_sampling_fields=resolved, submitted_sampling_fields=copy.deepcopy(self.sampling_fields),
            resolved_matches_submitted=resolved == self.sampling_fields,
            request_object_id=id(request), request_state_id=id(state), processor_id=id(self.processor),
            engine_id=id(self.engine), resolved_sampling_id=id(params), submitted_sampling_id=id(self.submitted),
            resolved_is_submitted=params is self.submitted,
            state=dict(request_id=state.request_id, external_req_id=state.external_req_id,
                       parent_req=None, request_index=state.request_index, max_tokens=state.max_tokens_param,
                       n=state.n, temperature=state.temperature, top_p=state.top_p),
            context={name: copy.deepcopy(value) for name, value in self.context.items()
                     if name != 'prompt_token_ids'}, frontend_identity=dict(self.identity),
            thread_id=threading.get_ident(), registration_line=frame.f_lineno,
            engine_registration_line=parent.f_lineno, device_operations_added=False)
        if len((json.dumps(row, indent=2, allow_nan=False)+'\n').encode('utf-8')) > self.limits.scope_bytes:
            raise ValueError('Frontend admission exceeds the CPU observation byte limit')
        self.pending = row

    def profile(self, frame, event, arg):
        if self.previous_profile is not None:
            self.previous_profile(frame, event, arg)
        if not self.active or self.stopped:
            return
        try:
            codes = self.sites.codes
            if event == 'call' and frame.f_code is codes['engine'] and frame.f_locals['self'] is self.engine:
                self.engine_calls += 1
                if self.engine_calls != 1:
                    raise ValueError('Exactly one original engine admission is required per generate call')
                self.engine_frame = frame
            elif (event == 'call' and frame.f_code is codes['registration'] and
                    frame.f_locals['self'] is self.processor):
                self.registration_calls += 1
                parent = frame.f_back
                if (self.registration_calls != 1 or parent is not self.engine_frame or
                        parent.f_code is not codes['engine'] or
                        parent.f_lineno != self.sites.lines['registration']):
                    raise ValueError('Registration must be owned by the original n=1 engine admission')
                self.registration_frame = frame
                request = frame.f_locals['request']
                if self.processor.request_states.get(request.request_id) is not None:
                    raise ValueError('Existing native RequestState is a streaming update, not new admission')
            elif event == 'call' and frame.f_code is codes['stats'] and frame.f_locals['self'] is self.processor:
                parent = frame.f_back
                if (parent is self.registration_frame and
                        parent.f_lineno == self.sites.lines['registered_stats']):
                    self.registered(parent)
            elif event == 'return' and frame is self.registration_frame:
                if self.pending is None:
                    raise ValueError('Original successful registration boundary was not observed')
                self.pending['native_registration_outcome'] = ('returned' if self.returned(frame)
                                                               else 'raised_or_interrupted')
                self.records.append(self.pending)
                self.pending = self.registration_frame = None
            elif event == 'return' and frame is self.engine_frame:
                if len(self.records) != 1:
                    raise ValueError('Original engine admission has no unique registered request')
                row = self.records[0]
                row['engine_returned_internal_id'] = arg if self.returned(frame) else None
                row['engine_core_submission'] = ('ORIGINAL_ENGINE_RETURNED_ID' if
                    self.returned(frame) and arg == row['internal_request_id'] else 'UNVERIFIED')
                self.engine_frame = None
        except BaseException as failure:
            self.error('frontend_callback', failure)

    def __enter__(self):
        if self.previous_profile is not None and not callable(self.previous_profile):
            self.error('profile_compatibility', TypeError('Prior profile hook cannot be composed'))
            return self
        try:
            self.active = True
            sys.setprofile(self.profile_hook)
            self.installed = True
        except BaseException as failure:
            self.error('profile_installation', failure)
            self.active = False
            try:
                sys.setprofile(self.previous_profile)
            except BaseException as restore_failure:
                self.error('profile_restoration', restore_failure)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.active = False
        try:
            if self.installed:
                try:
                    if sys.getprofile() is not self.profile_hook:
                        raise RuntimeError('Native execution replaced the scoped frontend hook')
                finally:
                    sys.setprofile(self.previous_profile)
        except BaseException as failure:
            self.error('profile_restoration', failure)
        if (self.engine_calls != 1 or self.registration_calls != 1 or len(self.records) != 1 or
                self.pending is not None or self.engine_frame is not None or
                self.registration_frame is not None):
            self.error('admission_delivery', ValueError('Original frontend admission is missing or incomplete'))
        if self.records and (self.records[0]['native_registration_outcome'] != 'returned' or
                self.records[0]['engine_core_submission'] != 'ORIGINAL_ENGINE_RETURNED_ID'):
            self.error('admission_return', ValueError('Original native admission did not complete'))
        if self.records and not self.records[0]['resolved_matches_submitted']:
            self.error('resolved_sampling', ValueError('Original resolved sampling fields changed'))
        self.record = dict(scope='source_pinned_frontend_admission_only_not_G2',
            status='UNVERIFIED' if self.errors or exc_type is not None else 'OBSERVED_FRONTEND_ADMISSION_ONLY_NOT_G2',
            records=self.records, observer_errors=list(self.errors),
            native_outcome='raised' if exc_type is not None else 'returned',
            native_error_type=exc_type.__module__+'.'+exc_type.__qualname__ if exc_type else None,
            source_status='LOADED_CODE_MATCHES_PINNED_SOURCE', sources=self.sites.sources,
            observer_scope='current_frontend_thread_call_boundaries_only',
            native_compatibility='UNVERIFIED', device_operations_added=False,
            evidence_status=dict(device_content='UNMEASURED', gpu_completion='UNMEASURED', G2='UNMEASURED'))
        if len((json.dumps(self.record, indent=2, allow_nan=False)+'\n').encode('utf-8')) > self.limits.scope_bytes:
            self.error('admission_byte_limit', ValueError('Final frontend record exceeds the CPU byte limit'))
            self.record = dict(status='UNVERIFIED', records=[],
                               observer_errors=[dict(kind='admission_byte_limit')])
            if len((json.dumps(self.record, indent=2)+'\n').encode('utf-8')) > self.limits.scope_bytes:
                self.record = None
        self.engine_frame = self.registration_frame = self.pending = None
        return False


@dataclass
class ObservationSites:
    codes: dict
    lines: dict
    sources: dict
    tensor_type: type
    graph_type: type
    intermediate_type: type | None = None

    @classmethod
    def bind(cls, native, worker_factory, *, pins=SOURCE_PINS, lines=SOURCE_LINES):
        sources, codes = {}, {}
        functions = {
            'worker': (worker_factory, ('Worker', '_make_model_runner'), 'worker'),
            'execute': (native.runner.GPUModelRunner.execute_model,
                        ('GPUModelRunner', 'execute_model'), 'runner'),
            'capture': (native.runner.GPUModelRunner.capture_model,
                        ('GPUModelRunner', 'capture_model'), 'runner'),
            'timing': (native.timing.StepTimingCollector.forward_start,
                       ('StepTimingCollector', 'forward_start'), 'timing'),
            'full': (native.graphs.CudaGraphManager.run_fullgraph,
                     ('CudaGraphManager', 'run_fullgraph'), 'graphs'),
            'manager_capture': (native.graphs.CudaGraphManager.capture,
                                ('CudaGraphManager', 'capture'), 'graphs'),
            'model_capture': (native.graphs.ModelCudaGraphManager.capture,
                              ('ModelCudaGraphManager', 'capture'), 'graphs'),
            'pw': (native.piecewise.CUDAGraphWrapper.__call__,
                   ('CUDAGraphWrapper', '__call__'), 'piecewise'),
            'break_dispatch': (native.breakable.BreakableCUDAGraphWrapper.__call__,
                               ('BreakableCUDAGraphWrapper', '__call__'), 'breakable'),
            'break_replay': (native.breakable.BreakableCUDAGraphWrapper._replay,
                             ('BreakableCUDAGraphWrapper', '_replay'), 'breakable'),
            'break_capture': (native.breakable.BreakableCUDAGraphWrapper._capture,
                              ('BreakableCUDAGraphWrapper', '_capture'), 'breakable'),
            'segments': (native.breakable.BreakableCUDAGraphCapture.replay,
                         ('BreakableCUDAGraphCapture', 'replay'), 'breakable'),
            'full_clear': (native.graphs.CudaGraphManager.release_graphs,
                           ('CudaGraphManager', 'release_graphs'), 'graphs'),
            'pw_clear': (native.piecewise.CUDAGraphWrapper.clear_graphs,
                         ('CUDAGraphWrapper', 'clear_graphs'), 'piecewise'),
            'break_clear': (native.breakable.BreakableCUDAGraphWrapper.clear_graphs,
                            ('BreakableCUDAGraphWrapper', 'clear_graphs'), 'breakable'),
        }
        for name, (function, owner, source) in functions.items():
            codes[name] = bind_code(function, owner, pins[source], sources)
        codes['create_capture'] = nested_code(codes['model_capture'], 'create_forward_fn')
        codes['capture_forward'] = nested_code(codes['create_capture'], 'forward_fn')
        required = {
            'execute': ('final_inputs',), 'full': ('full_replay',),
            'manager_capture': ('full_capture_store', 'full_capture_forward'),
            'pw': ('pw_replay', 'pw_capture', 'pw_capture_store'),
            'break_dispatch': ('break_dispatch', 'break_capture_dispatch'), 'break_replay': ('break_replay',),
            'break_capture': ('break_capture', 'break_capture_store'),
            'segments': ('break_segment',),
        }
        for name, anchors in required.items():
            executable = {line for _, line in dis.findlinestarts(codes[name])}
            if any(lines[anchor] not in executable for anchor in anchors):
                raise ValueError('Pinned statement is absent from loaded code: '+name)
        return cls(codes, dict(lines), sources, native.tensor_type, native.graph_type,
                   getattr(native, 'intermediate_type', None))


def installed_sites(worker, worker_factory):
    """Prepare the supported Worker factory branch without constructing a native runner."""
    if not worker.use_v2_model_runner or worker.vllm_config.is_mm_encoder_only:
        raise ValueError('Actual worker does not select the text MRv2 factory branch')
    native = SimpleNamespace(**{name: importlib.import_module(module) for name, module in (
        ('runner', 'vllm.v1.worker.gpu.model_runner'),
        ('graphs', 'vllm.v1.worker.gpu.cudagraph_utils'),
        ('piecewise', 'vllm.compilation.cuda_graph'),
        ('breakable', 'vllm.compilation.breakable_cudagraph'),
        ('timing', 'vllm.v1.worker.gpu.async_utils'),
    )})
    native.tensor_type = native.runner.torch.Tensor
    native.graph_type = native.runner.torch.cuda.CUDAGraph
    native.intermediate_type = native.runner.IntermediateTensors
    return native.runner.GPUModelRunner, ObservationSites.bind(native, worker_factory)


class Metadata:
    """Read only CPU tensor descriptors and ordinary containers, never tensor values."""
    def __init__(self, sites, limits):
        self.sites, self.limits, self.items = sites, limits, 0
        self.complete = True

    def value(self, value, depth=0):
        self.items += 1
        if self.items > self.limits.metadata_items or depth > self.limits.metadata_depth:
            raise ValueError('CPU metadata traversal exceeds observation limits')
        if value is None or type(value) in (str, int, float, bool):
            return value
        if isinstance(value, self.sites.tensor_type):
            storage = value.untyped_storage()
            return dict(kind='tensor_metadata_only', type=class_name(value),
                        shape=list(value.shape), stride=list(value.stride()),
                        dtype=str(value.dtype), device=str(value.device),
                        data_ptr=value.data_ptr(), storage_ptr=storage.data_ptr(),
                        storage_bytes=storage.nbytes(), storage_offset=value.storage_offset(),
                        content='UNMEASURED')
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError('Metadata dictionary keys must be strings')
            return {key: self.value(item, depth+1) for key, item in value.items()}
        if type(value) in (list, tuple):
            return [self.value(item, depth+1) for item in value]
        if getattr(type(value), '__dataclass_fields__', None) is not None:
            return self.descriptor(value)
        if self.sites.intermediate_type is not None and isinstance(value, self.sites.intermediate_type):
            return dict(kind='intermediate_tensor_metadata_only',
                        tensors=self.value(value.tensors, depth+1))
        self.complete = False
        return dict(kind='opaque_metadata_only', type=class_name(value), object_id=id(value),
                    status='UNVERIFIED_UNSUPPORTED_VALUE_TYPE', content='UNMEASURED')

    def descriptor(self, value):
        # Both native descriptor classes are dataclasses with CPU scalar fields.
        fields = getattr(type(value), '__dataclass_fields__', None)
        if fields is None:
            raise ValueError('Selected native descriptor is not a dataclass')
        result = {}
        for name in fields:
            item = getattr(value, name)
            if name == 'cg_mode':
                item = item.name
            result[name] = self.value(item)
        return result


def request_frontiers(batch):
    """Join sorted native rows by internal request ID, never by scheduler order."""
    rows = []
    names = ('idx_mapping_np', 'num_scheduled_tokens', 'num_computed_tokens_np',
             'prefill_len_np', 'num_computed_prefill_tokens_np', 'is_prefilling_np')
    count = batch['num_reqs']
    if len(batch['req_ids']) != count or any(len(batch[name]) != count for name in names):
        raise ValueError('Native CPU request rows do not have a complete association')
    if len(batch['query_start_loc_np']) < count+1:
        raise ValueError('Native CPU query frontiers are incomplete')
    for index, req_id in enumerate(batch['req_ids']):
        rows.append(dict(req_id=req_id, **{name: batch[name][index] for name in names},
                         query_start=batch['query_start_loc_np'][index],
                         query_end=batch['query_start_loc_np'][index+1]))
    return rows


@dataclass
class ModelInputObservation:
    sites: ObservationSites
    limits: ObservationLimits
    snapshot_batch: object
    capture_record: object
    invocations: int = 0
    captures: int = 0
    failures: int = 0
    last_record: dict | None = None
    last_summary: dict = field(default_factory=dict)
    identities: dict = field(default_factory=dict)
    identity_serial: int = 0
    stopped: bool = False

    def continuity(self, owner, entry, graph, selection_key, path, *, captured=False):
        key = (path, id(owner), json.dumps(selection_key, sort_keys=True, allow_nan=False))
        previous = self.identities.get(key)
        same = previous is not None and all(reference() is value for reference, value in
                   zip(previous['references'], (owner, entry, graph)))
        if not same or captured:
            for old_key in list(self.identities):
                if any(reference() is None for reference in self.identities[old_key]['references']):
                    del self.identities[old_key]
            if key not in self.identities and len(self.identities) >= self.limits.tracked_entries:
                raise ValueError('Graph identity observations exceed the entry limit')
            self.identity_serial += 1
            try:
                references = tuple(weakref.ref(value) for value in (owner, entry, graph))
            except TypeError:
                return dict(status='UNVERIFIED_WEAK_REFERENCE_UNAVAILABLE',
                            residency_between_scopes='UNVERIFIED')
            self.identities[key] = dict(references=references, serial=self.identity_serial)
        return dict(status=('CAPTURE_OBSERVED' if captured else
                            'SAME_OBJECTS_AT_OBSERVED_BOUNDARIES' if same else
                            'REPLACED_OR_RELEASED' if previous else 'FIRST_OBSERVATION'),
                    serial=self.identities[key]['serial'], residency_between_scopes='UNVERIFIED')

    def summary(self):
        return dict(status='CPU_METADATA_ONLY_NATIVE_COMPATIBILITY_UNVERIFIED',
                    source_status='LOADED_CODE_MATCHES_PINNED_SOURCE',
                    native_compatibility='UNVERIFIED', invocations=self.invocations,
                    captures=self.captures, observer_failures=self.failures,
                    collection_stopped=self.stopped, last_scope=copy.deepcopy(self.last_summary),
                    evidence_status=dict(UNMEASURED),
                    readiness_checks=list(NATIVE_READINESS_CHECKS))


class ObservationScope:
    def __init__(self, runner, state, kind):
        self.runner, self.state, self.kind = runner, state, kind
        self.sites, self.events, self.used = state.sites, [], 0
        self.input_id, self.frontiers = None, None
        self.pending, self.local_hooks, self.errors = {}, {}, []
        self.active, self.stopped, self.installed = False, False, False
        self.opcodes = {}
        self.profile_hook, self.trace_hook = self.profile, self.trace
        self.previous_profile, self.previous_trace = sys.getprofile(), sys.gettrace()

    def error(self, kind, failure):
        self.state.failures += 1
        self.stopped = True
        if not self.errors:
            self.errors.append(dict(kind=kind, error_type=class_name(failure), error=str(failure)))

    def emit(self, kind, **row):
        if self.stopped:
            return None
        event = dict(event=len(self.events)+1, kind=kind, lifecycle=self.kind,
                     input_observation=self.input_id, request_frontiers=self.frontiers,
                     cpu_frontier_semantics='optimistic_upper_bounds_not_device_progress',
                     **row)
        payload = json.dumps(event, allow_nan=False, separators=(',', ':'))
        if (len(self.events) >= self.state.limits.scope_events or
                self.used+len(payload.encode()) > self.state.limits.scope_bytes):
            raise ValueError('Model observation scope exceeds the CPU evidence limit')
        self.events.append(event)
        self.used += len(payload.encode())
        return event

    def metadata(self):
        return Metadata(self.sites, self.state.limits)

    def final_inputs(self, frame):
        values = frame.f_locals
        metadata = self.metadata()
        batch = self.state.snapshot_batch(values['input_batch'])
        self.frontiers = request_frontiers(batch) if not values['dummy_run'] else None
        row = self.emit('finalized_model_inputs', model_inputs=metadata.value(values['model_inputs']),
                        metadata_complete=metadata.complete, prepared_cpu_batch=batch,
                        descriptor=metadata.descriptor(values['batch_desc']),
                        request_association=('NO_REQUEST_DUMMY_RUN' if values['dummy_run'] else
                                             'INTERNAL_REQUEST_IDS_AND_CPU_FRONTIERS'),
                        source_line=frame.f_lineno,
                        valid_ranges='prepared_cpu_batch_not_device_content')
        if row is None:
            return
        self.input_id = row['event']
        row['input_observation'] = self.input_id

    def selection(self, parent, graph, path):
        values, metadata = parent.f_locals, self.metadata()
        owner = values['self']
        if path == 'FULL':
            entry, descriptor = graph, values['desc']
            args, kwargs = None, None
        else:
            entry = values['entry']
            descriptor = entry.batch_descriptor
            selected = entry.cudagraph if path == 'PIECEWISE' else entry.capture
            if selected is not graph:
                raise ValueError('Actual replay receiver differs from the selected local entry')
            args, kwargs = metadata.value(values['args']), metadata.value(values['kwargs'])
        descriptor = metadata.descriptor(descriptor)
        key = (self.breakable_key(parent, entry, metadata) if path == 'BREAKABLE_PIECEWISE'
               else descriptor)
        return dict(path=path, selection_key=key, owner_id=id(owner), entry_id=id(entry), graph_id=id(graph),
                    descriptor=descriptor, args=args, kwargs=kwargs,
                    metadata_complete=metadata.complete,
                    continuity=self.state.continuity(owner, entry, graph, key, path),
                    source_line=parent.f_lineno,
                    gpu_completion='UNMEASURED', replay_evidence='NATIVE_CALL_ONLY')

    def breakable_key(self, frame, entry, metadata):
        dispatch = frame.f_back
        line = ('break_capture_dispatch' if frame.f_code is self.sites.codes['break_capture']
                else 'break_dispatch')
        if (dispatch is None or dispatch.f_code is not self.sites.codes['break_dispatch']
                or dispatch.f_lineno != self.sites.lines[line]
                or dispatch.f_locals['entry'] is not entry):
            raise ValueError('Actual breakable selection key is not observable')
        return metadata.value(dispatch.f_locals['key'])

    def replay_parent(self, parent):
        if parent is None:
            return None
        codes, lines = self.sites.codes, self.sites.lines
        for name, line, path in (('full', 'full_replay', 'FULL'),
                                 ('pw', 'pw_replay', 'PIECEWISE'),
                                 ('break_replay', 'break_replay', 'BREAKABLE_PIECEWISE')):
            if parent.f_code is codes[name] and parent.f_lineno == lines[line]:
                if name == 'full' and parent.f_locals['self'] is not self.runner.cudagraph_manager:
                    return None
                return path
        return None

    def start_replay(self, key, parent, graph):
        path = self.replay_parent(parent)
        if path is None:
            return
        row = self.emit('replay_attempt', **self.selection(parent, graph, path))
        self.pending[key] = dict(row=row, exception=None)

    def finish_replay(self, key, outcome, error=None):
        pending = self.pending.pop(key, None)
        if pending is not None:
            self.emit('replay_outcome', replay_event=pending['row']['event'],
                      outcome=outcome, error_type=error, gpu_completion='UNMEASURED')

    def capture_inputs(self, frame, kind):
        values, metadata = frame.f_locals, self.metadata()
        row = dict(descriptor=metadata.descriptor(values['desc']),
                   owner_id=id(values['self']), model_inputs=metadata.value(values['model_inputs']),
                   metadata_complete=metadata.complete, request_association='NO_REQUEST_CAPTURE_ONLY',
                   device_operations_added=False, source_line=frame.f_lineno)
        if kind == 'capture_inputs_prepared':
            row['warmup'] = values['warmup']
        else:
            row['forward_runtime_mode'] = values['cg_mode'].name
            parent = frame.f_back
            row['capture_owner_line'] = parent.f_lineno if parent is not None else None
            row['native_capture_region'] = bool(parent is not None and
                parent.f_code is self.sites.codes['manager_capture'] and
                parent.f_lineno == self.sites.lines['full_capture_forward'] and values['desc'].cg_mode.name == 'FULL')
        self.emit(kind, **row)

    def profile(self, frame, event, arg):
        if self.previous_profile is not None:
            self.previous_profile(frame, event, arg)
        if not self.active or self.stopped:
            return
        try:
            codes = self.sites.codes
            if event == 'call':
                parent = frame.f_back
                if any(frame.f_code is codes[name] for name in ('full_clear', 'pw_clear', 'break_clear')):
                    owner = frame.f_locals['self']
                    for old_key in list(self.state.identities):
                        if self.state.identities[old_key]['references'][0]() is owner:
                            del self.state.identities[old_key]
                    self.emit('graph_lifecycle_clear', owner_id=id(owner), source_line=frame.f_lineno,
                              outcome='CLEAR_METHOD_ENTERED_NOT_COMPLETION')
                if (frame.f_code is codes['timing'] and parent is not None and
                        parent.f_code is codes['execute'] and
                        parent.f_locals.get('self') is self.runner and
                        parent.f_lineno == self.sites.lines['final_inputs']):
                    self.final_inputs(parent)
                path = self.replay_parent(parent)
                if path is not None:
                    graph = frame.f_locals.get('self')
                    if path == 'BREAKABLE_PIECEWISE':
                        if frame.f_code is not codes['segments']:
                            raise ValueError('Actual breakable replay code was not delivered')
                    elif not isinstance(graph, self.sites.graph_type):
                        return
                    else:
                        replay = inspect.getattr_static(type(graph), 'replay')
                        if getattr(replay, '__code__', None) is not frame.f_code:
                            return
                    self.start_replay(('python', id(frame)), parent, graph)
                if self.kind == 'capture' and frame.f_code is codes['capture_forward']:
                    if frame.f_locals['self'] is self.runner.cudagraph_manager:
                        self.capture_inputs(frame, 'capture_forward')
            elif event == 'return':
                if self.kind == 'capture' and frame.f_code is codes['create_capture']:
                    if frame.f_locals['self'] is self.runner.cudagraph_manager and arg is not None:
                        self.capture_inputs(frame, 'capture_inputs_prepared')
                key = ('python', id(frame))
                if key in self.pending:
                    opcodes = self.opcodes.setdefault(frame.f_code, {
                        instruction.offset: instruction.opname for instruction in dis.get_instructions(frame.f_code)})
                    returned = opcodes.get(frame.f_lasti) in ('RETURN_VALUE', 'RETURN_CONST')
                    error = self.pending[key]['exception']
                    self.finish_replay(key, 'returned_submission_not_completion' if returned else
                                       'raised' if error else 'UNVERIFIED_RETURN_OR_EXCEPTION',
                                       None if returned else error)
            elif event == 'c_call':
                path = self.replay_parent(frame)
                if path is not None and isinstance(getattr(arg, '__self__', None), self.sites.graph_type):
                    self.start_replay(('c', id(frame), id(arg.__self__)), frame, arg.__self__)
            elif event in ('c_return', 'c_exception'):
                self.finish_replay(('c', id(frame), id(getattr(arg, '__self__', None))),
                                   'returned_submission_not_completion' if event == 'c_return' else
                                   'raised_native_c_call')
        except BaseException as failure:
            self.error('profile_observation', failure)

    def trace(self, frame, event, arg):
        previous = self.previous_trace(frame, event, arg) if self.previous_trace is not None else None
        if not self.active or self.stopped:
            return previous
        # Global tracing is used only to select frames; line work stays in pinned frames.
        try:
            selected = ('manager_capture', 'pw', 'break_capture', 'segments')
            relevant = (any(frame.f_code is self.sites.codes[name] for name in selected)
                        or self.replay_parent(frame.f_back) is not None)
            if not relevant:
                return previous
            self.local_hooks[frame] = previous
        except BaseException as failure:
            self.error('trace_selection', failure)
            return previous

        def local(current, current_event, current_arg):
            old = self.local_hooks.get(current)
            if old is not None:
                replacement = old(current, current_event, current_arg)
                if replacement is not None:
                    self.local_hooks[current] = replacement
            if self.active and not self.stopped:
                try:
                    self.trace_observation(current, current_event, current_arg)
                except BaseException as failure:
                    self.error('trace_observation', failure)
            if current_event == 'return':
                self.local_hooks.pop(current, None)
            return local
        return local

    def trace_observation(self, frame, event, arg):
        key = ('python', id(frame))
        if event == 'exception' and key in self.pending:
            self.pending[key]['exception'] = arg[0].__module__+'.'+arg[0].__qualname__
        if event != 'line':
            return
        codes, lines, values = self.sites.codes, self.sites.lines, frame.f_locals
        if (frame.f_code is codes['manager_capture'] and frame.f_lineno == lines['full_capture_store']
                and values['self'] is self.runner.cudagraph_manager):
            self.capture_stored(values['self'], values['graph'], values['graph'], values['desc'], 'FULL', frame)
        elif frame.f_code is codes['pw']:
            if frame.f_lineno == lines['pw_capture']:
                self.wrapper_capture(frame, 'PIECEWISE', values['cudagraph'])
            elif frame.f_lineno == lines['pw_capture_store']:
                self.capture_stored(values['self'], values['entry'], values['entry'].cudagraph,
                                    values['entry'].batch_descriptor, 'PIECEWISE', frame)
        elif frame.f_code is codes['break_capture']:
            if frame.f_lineno == lines['break_capture']:
                self.wrapper_capture(frame, 'BREAKABLE_PIECEWISE', values['capture'])
            elif frame.f_lineno == lines['break_capture_store']:
                self.capture_stored(values['self'], values['entry'], values['entry'].capture,
                                    values['entry'].batch_descriptor, 'BREAKABLE_PIECEWISE', frame)
        elif frame.f_code is codes['segments'] and frame.f_lineno == lines['break_segment']:
            segment = values['r']
            receiver = getattr(segment, '__self__', None)
            self.emit('breakable_segment_attempt', capture_id=id(values['self']),
                      segment_id=id(segment),
                      segment_kind='GRAPH_REPLAY' if isinstance(receiver, self.sites.graph_type) else 'EAGER_BREAK',
                      graph_id=id(receiver) if isinstance(receiver, self.sites.graph_type) else None,
                      source_line=frame.f_lineno, outcome='ATTEMPT_ONLY', gpu_completion='UNMEASURED')

    def capture_stored(self, owner, entry, graph, descriptor, path, frame):
        metadata = self.metadata()
        descriptor = metadata.descriptor(descriptor)
        key = (self.breakable_key(frame, entry, metadata) if path == 'BREAKABLE_PIECEWISE'
               else descriptor)
        self.emit('capture_entry_stored', path=path, owner_id=id(owner), entry_id=id(entry),
                  graph_id=id(graph), descriptor=descriptor, selection_key=key, source_line=frame.f_lineno,
                  continuity=self.state.continuity(owner, entry, graph, key, path, captured=True),
                  device_operations_added=False, gpu_completion='UNMEASURED')

    def wrapper_capture(self, frame, path, graph):
        values, metadata = frame.f_locals, self.metadata()
        descriptor = metadata.descriptor(values['entry'].batch_descriptor)
        key = (self.breakable_key(frame, values['entry'], metadata) if path == 'BREAKABLE_PIECEWISE'
               else descriptor)
        self.emit('wrapper_capture_forward', path=path, owner_id=id(values['self']),
                  entry_id=id(values['entry']), graph_id=id(graph),
                  descriptor=descriptor, selection_key=key,
                  args=metadata.value(values['args']), kwargs=metadata.value(values['kwargs']),
                  metadata_complete=metadata.complete, device_operations_added=False,
                  source_line=frame.f_lineno, gpu_completion='UNMEASURED')

    def __enter__(self):
        if self.state.stopped:
            self.stopped = True
            return self
        if any(hook is not None and not callable(hook) for hook in (self.previous_profile, self.previous_trace)):
            self.error('hook_compatibility', TypeError('Prior non-callable hook cannot be composed or restored'))
            return self
        try:
            sys.setprofile(self.profile_hook)
            sys.settrace(self.trace_hook)
            self.installed = self.active = True
        except BaseException as failure:
            self.error('hook_installation', failure)
            try:
                self.restore()
            except BaseException as restore_failure:
                self.error('hook_restoration', restore_failure)
        return self

    def restore(self):
        self.active = False
        try:
            for frame, previous in list(self.local_hooks.items()):
                frame.f_trace = previous
        finally:
            self.local_hooks.clear()
            try:
                sys.settrace(self.previous_trace)
            finally:
                sys.setprofile(self.previous_profile)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.installed:
                try:
                    if sys.getprofile() is not self.profile_hook or sys.gettrace() is not self.trace_hook:
                        self.error('hook_delivery', RuntimeError('Native execution replaced scoped hooks'))
                finally:
                    self.restore()
        except BaseException as failure:
            self.error('hook_restoration', failure)
        for pending in self.pending.values():
            pending['row']['outcome'] = 'UNVERIFIED_MISSING_REPLAY_RETURN'
        inputs = [row for row in self.events if row['kind'] == 'finalized_model_inputs']
        attempts = [row for row in self.events if row['kind'] == 'replay_attempt']
        captures = [row for row in self.events if row['kind'] == 'capture_entry_stored']
        outcomes = [row for row in self.events if row['kind'] == 'replay_outcome']
        graph_mode = inputs[-1]['descriptor']['cg_mode'] if inputs else None
        argument_rows = [row for row in self.events if row['kind'] in (
            'finalized_model_inputs', 'capture_inputs_prepared', 'capture_forward', 'wrapper_capture_forward')]
        observation_status = dict(
            finalized_model_inputs='OBSERVED_CPU_METADATA' if inputs else
                'NOT_APPLICABLE_CAPTURE' if self.kind == 'capture' else 'UNVERIFIED_NO_FINAL_INPUT_CALLBACK',
            selected_graph_entry='OBSERVED_NATIVE_ENTRY_METADATA' if attempts or captures else
                'NOT_APPLICABLE_EAGER' if graph_mode == 'NONE' else
                'UNVERIFIED_NO_REPLAY_CALLBACK' if graph_mode is not None else 'UNMEASURED',
            replay_outcomes='UNVERIFIED_MISSING_REPLAY_RETURN' if self.pending else
                'UNVERIFIED_RETURN_OR_EXCEPTION' if any(row['outcome'].startswith('UNVERIFIED') for row in outcomes) else
                'OBSERVED_NATIVE_CALL_OUTCOMES_ONLY' if outcomes else 'UNMEASURED',
            kwargs_metadata='UNVERIFIED_UNSUPPORTED_VALUE_TYPE' if any(
                row.get('metadata_complete') is False for row in self.events) else
                'OBSERVED_CPU_METADATA' if argument_rows else
                'UNVERIFIED_NO_INPUT_CALLBACK' if captures else 'UNMEASURED')
        if self.stopped or self.errors or self.pending:
            status = 'UNVERIFIED'
        elif self.kind == 'execute' and self.input_id is None:
            status = 'UNVERIFIED_NO_FINAL_INPUT_CALLBACK'
        elif any(value.startswith('UNVERIFIED') for value in observation_status.values()):
            status = 'UNVERIFIED'
        else:
            status = 'OBSERVED_CPU_METADATA_ONLY_NOT_G2' if self.events else 'UNMEASURED'
        record = dict(scope='source_pinned_model_metadata_only_not_G2', lifecycle=self.kind,
                      status=status, observation_status=observation_status,
                      events=self.events, observer_errors=self.errors,
                      native_outcome='raised' if exc_type is not None else 'returned',
                      native_error_type=exc_type.__module__+'.'+exc_type.__qualname__ if exc_type else None,
                      native_compatibility='UNVERIFIED', evidence_status=dict(UNMEASURED),
                      source_status='LOADED_CODE_MATCHES_PINNED_SOURCE',
                      sources=self.sites.sources, hook_delivery='UNVERIFIED' if self.errors else
                      'OBSERVED_CALLBACKS' if self.events else 'NOT_OBSERVED')
        self.state.last_record = record
        self.state.last_summary = dict(status=status, lifecycle=self.kind, events=len(self.events),
                                       observation_status=dict(observation_status),
                                       final_inputs_observed=self.input_id is not None,
                                       replay_attempts=sum(row['kind'] == 'replay_attempt' for row in self.events),
                                       native_outcome=record['native_outcome'],
                                       observer_errors=list(self.errors))
        if self.stopped:
            self.state.stopped = True
        if self.kind == 'capture':
            try:
                self.state.capture_record(record)
            except BaseException as failure:
                self.error('capture_record', failure)
                self.state.stopped = True
                record['status'] = self.state.last_summary['status'] = 'UNVERIFIED'
                self.state.last_summary['observer_errors'] = list(self.errors)
        return False


def observed_runner(base, sites, snapshot_batch, capture_record, *, limits=ObservationLimits()):
    """Subclass only the two supported runner entry points; constructor and native bodies stay inherited."""
    def state(runner):
        current = getattr(runner, '_model_input_observation', None)
        if current is None:
            current = runner._model_input_observation = ModelInputObservation(
                sites, limits, snapshot_batch, capture_record)
        return current

    def observe(name, kind):
        original = getattr(base, name)
        def call(runner, *args, **kwargs):
            scope = None
            try:
                current = state(runner)
                if kind == 'execute':
                    current.invocations += 1
                else:
                    current.captures += 1
                scope = ObservationScope(runner, current, kind)
                scope.__enter__()
            except BaseException as failure:
                if scope is not None:
                    scope.error('observer_setup', failure)
                runner._model_input_observation_error = dict(status='UNVERIFIED',
                    kind='observer_setup', error_type=class_name(failure))
            try:
                return original(runner, *args, **kwargs)
            finally:
                if scope is not None:
                    try:
                        scope.__exit__(*sys.exc_info())
                    except BaseException as failure:
                        scope.error('observer_finalization', failure)
                        runner._model_input_observation_error = dict(status='UNVERIFIED',
                            kind='observer_finalization', error_type=class_name(failure))
        return call

    return type('InputObservedMRv2', (base,), dict(
        execute_model=observe('execute_model', 'execute'),
        capture_model=observe('capture_model', 'capture')))
