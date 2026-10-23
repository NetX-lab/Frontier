"""Python-visible native V4 ownership and synchronization; never a GPU cost verdict."""

import ctypes
import functools
import importlib
import inspect
import json
import os
import re
import sys
import threading
import traceback
from contextlib import contextmanager

import native_full_structure as inventory

MODEL_MODULE = 'vllm.models.deepseek_v4.nvidia.model'
ATTENTION_MODULE = 'vllm.models.deepseek_v4.attention'
COMPRESSOR_MODULE = 'vllm.models.deepseek_v4.compressor'
METHODS = (
    (MODEL_MODULE, 'DeepseekV4Model', 'forward', 'outer', 'model_forward'),
    (MODEL_MODULE, 'DeepseekV4DecoderLayer', 'forward', 'outer', 'decoder_layer'),
    (ATTENTION_MODULE, 'DeepseekV4Attention', 'forward', 'outer', 'attention_forward'),
    (ATTENTION_MODULE, 'DeepseekV4Attention', '_prepare_and_attn', 'outer', 'attention_eager_segment'),
    (ATTENTION_MODULE, 'DeepseekV4Attention', '_sparse_indexer_and_attn', 'outer', 'sparse_eager_segment'),
    (ATTENTION_MODULE, 'DeepseekV4Attention', '_run_parallel_input_projections', 'component', 'input_projections'),
    (ATTENTION_MODULE, 'DeepseekV4Attention', '_fused_qnorm_rope_kv_insert', 'component', 'qnorm_rope_kv_insert'),
    ('vllm.models.deepseek_v4.nvidia.flashmla', 'DeepseekV4FlashMLAAttention', 'forward_mqa', 'component', 'sparse_attention'),
    ('vllm.models.deepseek_v4.nvidia.flashmla', 'DeepseekV4FlashMLAAttention', '_o_proj', 'component', 'output_projection'),
    (ATTENTION_MODULE, 'DeepseekV4Indexer', 'forward', 'ownership', 'indexer'),
    (COMPRESSOR_MODULE, 'DeepseekCompressor', 'forward', 'ownership', 'compressor'),
    (MODEL_MODULE, 'DeepseekV4MoE', 'forward', 'ownership', 'ffn_moe'),
    (MODEL_MODULE, 'DeepseekV4MLP', 'forward', 'ownership', 'ffn'),
    (MODEL_MODULE, 'DeepseekV4ForCausalLM', 'compute_logits', 'ownership', 'logits'),
    ('vllm.v1.worker.gpu.model_states.default', 'DefaultModelState', 'prepare_attn', 'metadata', 'attention_metadata'),
)
FUNCTIONS = (
    (MODEL_MODULE, 'mhc_pre_tilelang'), (MODEL_MODULE, 'mhc_pre_broadcast_tilelang'),
    (MODEL_MODULE, 'mhc_fused_post_pre_tilelang'), (MODEL_MODULE, 'mhc_post_tilelang'),
    (MODEL_MODULE, 'hc_head_fused_kernel_tilelang'), (ATTENTION_MODULE, 'fused_q_kv_rmsnorm'),
    (COMPRESSOR_MODULE, 'save_partial_states'), (COMPRESSOR_MODULE, 'compress_norm_rope_store_triton'),
    (COMPRESSOR_MODULE, 'compress_norm_rope_store_two_stage_triton'),
)


class ServingProfile:
    def __init__(self, observer):
        self.inventory = observer
        self.local = threading.local()
        self.steps, self.annotations, self.synchronization = [], [], []
        self.errors, self.hooks, self.streams, self.events, self.layers = [], [], {}, {}, {}
        self.rank, self.worker, self.sequence = None, None, 0
        self.status = 'UNVERIFIED'
        self.used, self.dropped = 0, 0
        self.records = 0
        self.step_index = 0
        self.limits = self.observe('limits', self.read_limits)

    def read_limits(self):
        limits = json.loads(os.environ['FRONTIER_NATIVE_PROFILE_LIMITS'])
        if set(limits) != {'records', 'bytes'} or any(type(v) is not int or v <= 0 for v in limits.values()):
            raise ValueError('Profile limits require positive records and bytes')
        return limits

    def error(self, where):
        try:
            row = dict(where=where, traceback=traceback.format_exc())
            if len(self.errors) < 64:
                self.errors.append(row)
            print('NATIVE_PROFILE_UNVERIFIED', row, file=sys.stderr, flush=True)
        except BaseException:
            pass

    def observe(self, where, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except BaseException:
            self.error(where)

    def append(self, collection, row):
        size = len(json.dumps(row).encode())
        if self.limits is None or self.records >= self.limits['records'] or self.used + size > self.limits['bytes']:
            self.dropped += 1
            return None
        self.used += size
        self.records += 1
        collection.append(row)
        return row

    def context(self):
        from torch import cuda
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        native = self.inventory.local
        dispatch = getattr(native, 'dispatch_evidence', None)
        step = getattr(self.local, 'step', None)
        if dispatch is None and step is not None:
            dispatch = step.get('dispatch_evidence')
        if step is not None and dispatch is not None and 'dispatch_evidence' not in step:
            step['dispatch_evidence'] = dispatch
        context = getattr(native, 'context', None) or ({} if step is None else step.get('context', {}))
        if step is not None:
            step['context'] = dict(context)
        has_prefill = None if dispatch is None else dispatch.get('classifier', {}).get('inputs', {}).get('has_prefill')
        kind = context.get('kind', 'unattributed')
        capture_active = BreakableCUDAGraphCapture.is_active()
        stream_capturing = cuda.is_current_stream_capturing()
        execution = ('capture' if kind == 'capture' or capture_active or stream_capturing else
            'replay_eager_segment' if getattr(self.local, 'replaying', False) else
            'eager_serving' if kind == 'serving' else kind)
        return dict(step=None if step is None else step['step'], kind=kind,
            repeat=context.get('repeat'), request_phase=context.get('phase'),
            has_prefill=has_prefill, token_phase=None if has_prefill is None else
                'prefill' if has_prefill else 'decode', execution=execution,
            graph_mode=None if dispatch is None else dispatch.get('dispatch', {}).get('descriptor', {}).get('cg_mode'),
            startup_warmup=getattr(native, 'startup_warmup', False),
            breakable_capture_active=capture_active, stream_capturing=stream_capturing,
            layer=getattr(self.local, 'layer', None))

    @contextmanager
    def range(self, category, scope, owner=None, arguments=None):
        previous = getattr(self.local, 'layer', None)
        previous_scope = getattr(self.local, 'scope', None)
        annotation = None
        try:
            if owner is not None and id(owner) in self.layers:
                self.local.layer = self.layers[id(owner)]
            row = self.context() | dict(category=category, scope=scope, id=self.sequence)
            self.sequence += 1
            row['annotation'] = 'frontier_native_' + category + '|' + str(row['id']) + '|' + scope
            if arguments is not None:
                row['launch_arguments'] = {k: v for k, v in arguments.items()
                    if 'pdl' in k.lower() or 'gdc' in k.lower()}
                if category == 'metadata':
                    row['for_capture'] = arguments.get('for_capture', False)
            self.append(self.annotations, row)
            self.local.scope = scope
            from torch.profiler import record_function
            annotation = record_function(row['annotation'])
            annotation.__enter__()
        except BaseException:
            self.error('annotation_enter:' + scope)
            annotation = None
        try:
            yield
        finally:
            if annotation is not None:
                self.observe('annotation_exit:' + scope, annotation.__exit__, None, None, None)
            self.local.layer, self.local.scope = previous, previous_scope

    def wrap(self, owner, name, category, scope):
        original = getattr(owner, name)
        if getattr(original, '_native_profile', None) is self:
            return
        signature = inspect.signature(original)
        decorated = name == '_sparse_indexer_and_attn' and hasattr(original, '__wrapped__')
        native = original.__wrapped__ if decorated else original

        @functools.wraps(original)
        def call(*args, **kwargs):
            bound = self.observe('arguments:' + scope, signature.bind, *args, **kwargs)
            with self.range(category, scope, args[0] if args else None,
                    None if bound is None else bound.arguments):
                return native(*args, **kwargs)

        if decorated:
            call = importlib.import_module('vllm.compilation.breakable_cudagraph').eager_break_during_capture(call)
        call._native_profile = self
        setattr(owner, name, call)
        self.hooks.append(str(getattr(owner, '__name__', type(owner).__name__)) + '.' + name)

    def stream(self, stream):
        handle = int(stream.cuda_stream)
        device = int(stream.device_index)
        key = f'{device}:{handle}'
        if key not in self.streams:
            self.streams[key] = dict(cuda_stream=handle, device_index=device,
                torch_stream_id=stream.stream_id, trace_stream_id=None)
            identifier = self.observe('trace_stream_mapping', self.trace_stream, handle)
            self.streams[key]['trace_stream_id'] = identifier
        return key

    def trace_stream(self, handle):
        # CUPTI's ID, unlike torch's stream_id, is the profiler trace stream ID.
        driver, cupti = ctypes.CDLL('libcuda.so.1'), ctypes.CDLL('libcupti.so')
        context, identifier = ctypes.c_void_p(), ctypes.c_uint32()
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cupti.cuptiGetStreamIdEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint8, ctypes.POINTER(ctypes.c_uint32)]
        if driver.cuCtxGetCurrent(ctypes.byref(context)) or not context.value:
            raise RuntimeError('No existing CUDA context for stream identity')
        code = cupti.cuptiGetStreamIdEx(context, ctypes.c_void_p(handle), 0, ctypes.byref(identifier))
        if code:
            raise RuntimeError(f'cuptiGetStreamIdEx failed: {code}')
        return identifier.value

    def event(self, operation, event, stream):
        from torch import cuda
        actual = cuda.current_stream() if stream is None else stream
        key = self.stream(actual)
        identity = str(id(event))
        state = self.events.setdefault(identity, dict(generation=0, source=None))
        if operation == 'record':
            state.update(generation=state['generation'] + 1, source=key)
        self.append(self.synchronization, self.context() | dict(operation=operation,
            event_object=identity, event_handle=int(event.cuda_event),
            generation=state['generation'], source_stream=state['source'],
            destination_stream=key if operation == 'wait' else None,
            recording_stream=key if operation == 'record' else None,
            scope=getattr(self.local, 'scope', None),
            owner_scope=getattr(self.local, 'parallel_scope', None),
            parallel_call=getattr(self.local, 'parallel_call', None)))

    def install_events(self, cuda):
        for operation in ('record', 'wait'):
            original = getattr(cuda.Event, operation)

            def factory(original, operation):
                @functools.wraps(original)
                def call(event, stream=None):
                    if not getattr(self.local, 'parallel_call', None):
                        return original(event, stream=stream)
                    with self.range('sync', operation):
                        result = original(event, stream=stream)
                        self.observe('event:' + operation, self.event, operation, event, stream)
                        return result
                return call

            setattr(cuda.Event, operation, factory(original, operation))
            self.hooks.append('torch.cuda.Event.' + operation)

    def parallel(self, original, arguments):
        previous = getattr(self.local, 'parallel_call', None)
        previous_scope = getattr(self.local, 'parallel_scope', None)
        self.local.parallel_scope = getattr(self.local, 'scope', None)
        self.local.parallel_call = self.sequence + 1
        try:
            with self.range('parallel', original.__name__):
                return original(**arguments)
        finally:
            self.local.parallel_call = previous
            self.local.parallel_scope = previous_scope

    def install_parallel(self, namespace):
        maybe, execute = namespace.maybe_execute_in_parallel, namespace.execute_in_parallel

        def maybe_execute_in_parallel(fn0, fn1, event0, event1, aux_stream=None):
            return self.parallel(maybe, dict(fn0=fn0, fn1=fn1, event0=event0,
                event1=event1, aux_stream=aux_stream))

        def execute_in_parallel(default_fn, aux_fns, start_event, done_events,
                                aux_streams=None, enable=False):
            return self.parallel(execute, dict(default_fn=default_fn, aux_fns=aux_fns,
                start_event=start_event, done_events=done_events,
                aux_streams=aux_streams, enable=enable))

        namespace.maybe_execute_in_parallel = maybe_execute_in_parallel
        namespace.execute_in_parallel = execute_in_parallel
        self.hooks.extend([namespace.__name__ + '.' + name for name in
            ('maybe_execute_in_parallel', 'execute_in_parallel')])

    def install_target(self, path, cls, name, category, scope):
        namespace = importlib.import_module(path)
        self.wrap(namespace if cls is None else getattr(namespace, cls), name, category, scope)

    def install_model(self, worker):
        if self.worker is worker:
            return
        self.worker, self.rank = worker, worker.rank
        model = worker.model_runner.model
        for name, module in model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', name)
            if match:
                self.layers[id(module)] = int(match[1])
        for path, cls, name, category, scope in METHODS:
            self.observe('install:' + cls + '.' + name, self.install_target,
                path, cls, name, category, scope)
        for path, name in FUNCTIONS:
            self.observe('install:' + name, self.install_target, path,
                None, name, 'ownership', name)
        for module in model.modules():
            if hasattr(module, '_prepare_and_attn_fn'):
                module._prepare_and_attn_fn = module._prepare_and_attn
        core = model.model
        for name, scope in (('embed_tokens', 'embedding'), ('norm', 'final_norm')):
            self.observe('install:' + scope, self.wrap, getattr(core, name),
                'forward', 'ownership', scope)
        self.install_parallel(importlib.import_module(ATTENTION_MODULE))
        manager = worker.model_runner.cudagraph_manager
        for name in ('run_fullgraph', 'run_pw_graph'):
            self.wrap(type(manager), name, 'outer', 'model_dispatch_' + name)
        self.install_replay(importlib.import_module('vllm.compilation.breakable_cudagraph').BreakableCUDAGraphCapture)
        self.facts = dict(environment={k: v for k, v in os.environ.items() if
            any(word in k for word in ('PDL', 'GDC', 'BREAKABLE_CUDAGRAPH', 'CUPTI'))},
            kernel_config=str(worker.vllm_config.kernel_config),
            compilation_config=str(worker.vllm_config.compilation_config),
            profiler_config=str(worker.vllm_config.profiler_config))
        self.facts['source_modules'] = {path: dict(origin=sys.modules[path].__file__)
            for path in {row[0] for row in METHODS + FUNCTIONS}
            if getattr(sys.modules.get(path), '__file__', None) is not None}

    def install_replay(self, cls):
        original = cls.replay

        @functools.wraps(original)
        def replay(graph, *args, **kwargs):
            previous = getattr(self.local, 'replaying', False)
            self.local.replaying = True
            try:
                with self.range('graph', 'replay'):
                    return original(graph, *args, **kwargs)
            finally:
                self.local.replaying = previous

        cls.replay = replay
        self.hooks.append(cls.__name__ + '.replay')

    def emit(self, stage):
        directory = self.inventory.directory / 'serving_profile'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f'rank{self.rank}.json'
        payload = dict(schema='native_serving_profile_v1', status=self.status, stage=stage,
            rank=self.rank, run_generation=self.inventory.generation, role=self.inventory.role,
            steps=self.steps, annotations=self.annotations, synchronization=self.synchronization,
            streams=self.streams, errors=self.errors, hooks=self.hooks,
            limits=self.limits, retained_bytes=self.used, dropped=self.dropped,
            facts=getattr(self, 'facts', {}), inventory_join='serving/rank' + str(self.rank) + '.json',
            unknowns=['per-kernel PDL attributes not observable here',
                'captured-segment kernels have no per-replay Python owner',
                'annotation/event survival under compile and breakable capture',
                'GPU CUPTI restart'],
            coverage='Observed Python calls only; no timing, completeness or GPU verification verdict')
        if self.worker is not None:
            payload['inventory_observation_errors'] = list(self.inventory.errors)
        temporary = target.with_suffix(f'.json.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(payload, sort_keys=True) + '\n')
        os.replace(temporary, target)

    def install_worker(self, cls):
        for name in ('load_model', 'execute_model', 'sample_tokens', 'compile_or_warm_up_model', 'shutdown'):
            original = getattr(cls, name)

            def factory(original, name):
                @functools.wraps(original)
                def call(worker, *args, **kwargs):
                    self.rank = worker.rank
                    if name == 'shutdown':
                        self.observe('pre_shutdown', self.emit, 'pre_shutdown')
                    previous = getattr(self.local, 'step', None)
                    if name == 'sample_tokens':
                        self.local.step = getattr(self.local, 'last_step', None)
                    if name == 'execute_model':
                        def begin():
                            output = args[0] if args else kwargs['scheduler_output']
                            scheduled = output.num_scheduled_tokens
                            row = dict(step=self.step_index, request_ids=[k for k, v in scheduled.items() if v > 0],
                                local_num_requests=sum(v > 0 for v in scheduled.values()),
                                local_num_tokens=output.total_num_scheduled_tokens)
                            if not row['local_num_tokens']:
                                self.local.last_step = None
                            self.step_index += 1
                            return self.append(self.steps, row)
                        self.local.step = self.observe('step', begin)
                    try:
                        result = original(worker, *args, **kwargs)
                    finally:
                        if name == 'execute_model' and getattr(self.local, 'step', None) is not None:
                            self.observe('step_finalize', self.finish_step, self.local.step)
                            if self.local.step['local_num_tokens']:
                                self.local.last_step = self.local.step
                        self.local.step = previous
                    if name == 'load_model':
                        self.observe('model_installation', self.install_model, worker)
                    if name == 'compile_or_warm_up_model':
                        self.observe('post_warmup', self.emit, 'post_warmup')
                    return result
                return call

            setattr(cls, name, factory(original, name))
            self.hooks.append(cls.__name__ + '.' + name)

    def finish_step(self, step):
        dispatch = step.get('dispatch_evidence')
        if dispatch is not None:
            step['dispatch_forwards'], step['dispatch_replays'] = dispatch['forwards'], dispatch['replays']
            step['dispatch_evidence'] = {k: v for k, v in dispatch.items() if k not in ('forwards', 'replays')}
            step['has_prefill'] = dispatch.get('classifier', {}).get('inputs', {}).get('has_prefill')
            step['synchronized_descriptor'] = dispatch.get('dispatch', {}).get('descriptor')
            step['synchronized_mode'] = (step['synchronized_descriptor'] or {}).get('cg_mode')
            self.used += len(json.dumps(step['dispatch_evidence']).encode())


class NativeServeProfile(inventory.FullModelInventory):
    """Preserve the full inventory RPC while composing its installed observer."""


_PROFILE = None
if inventory._NATIVE_SERVING_INVENTORY is not None:
    _PROFILE = ServingProfile(inventory._NATIVE_SERVING_INVENTORY)
    try:
        from torch import cuda
        from vllm.v1.worker.gpu_worker import Worker
        _PROFILE.install_events(cuda)
        _PROFILE.install_worker(Worker)
    except BaseException:
        _PROFILE.error('installation')
