"""Run full-structure native GLM startup and retain unqualified G0-B observations."""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import socket
import sys
import tempfile
import traceback
from pathlib import Path


def identity(value):
    cls = value if isinstance(value, type) else type(value)
    return cls.__module__ + "." + cls.__name__


def balanced_routing_inventory(model):
    """Observe the loaded strategy and every router without device reads."""
    selected = os.environ.get('VLLM_MOE_ROUTING_SIMULATION_STRATEGY')
    if selected != 'balanced_topk':
        return None
    import vllm.envs as envs
    from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import RoutingSimulator
    strategy = RoutingSimulator._routing_strategies[selected]
    origin = Path(sys.modules[type(strategy).__module__].__file__)
    if str(origin) != strategy.route_tokens.__func__.__code__.co_filename:
        raise ValueError('Loaded routing strategy code origin differs from its module')
    return dict(environment=selected, runtime_path=envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY,
        strategy=identity(strategy), origin=str(origin), sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
        routers={name: identity(module.router) for name, module in model.named_modules()
                 if hasattr(module, 'router')})


def cutlass_state(module):
    """Read selected consumer metadata without rewriting or validating device output."""
    import torch
    method = module.quant_method
    backend = method.fp8_backend.name
    if backend != "FLASHINFER_CUTLASS":
        return {"status": "unverified", "backend": backend}
    quant, moe = method.moe_quant_config, method.moe
    experts = method.moe_kernel.fused_experts

    def layout(tensor):
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "device": str(tensor.device), "stride": list(tensor.stride()),
                "contiguous": tensor.is_contiguous(), "offset": tensor.storage_offset()}

    weights = {"w13": module.w13_weight, "w2": module.w2_weight}
    scales = {"w1": getattr(module, "w13_" + method.weight_scale_name),
              "w2": getattr(module, "w2_" + method.weight_scale_name)}
    aliases = {key: {"quant_object": getattr(quant, key + "_scale") is scale,
        "expert_object": getattr(experts, key + "_scale") is scale,
        "quant_storage": getattr(quant, key + "_scale").untyped_storage().data_ptr() == scale.untyped_storage().data_ptr(),
        "expert_storage": getattr(experts, key + "_scale").untyped_storage().data_ptr() == scale.untyped_storage().data_ptr()}
        for key, scale in scales.items()}
    block = module.weight_block_size
    e, h, i = moe.num_local_experts, moe.hidden_dim, moe.intermediate_size_per_partition
    expected = {"w13": [e, moe.w13_num_shards * i, h], "w2": [e, h, i]}
    grids = {key: [shape[0], (shape[1] + block[0] - 1) // block[0],
                           (shape[2] + block[1] - 1) // block[1]]
             for key, shape in expected.items()}
    checks = {
        "quant_config_alias": experts.quant_config is quant,
        "moe_config_aliases": method.moe is module.moe_config and experts.moe_config is moe,
        "scale_aliases": all(all(row.values()) for row in aliases.values()),
        "block_domain": quant.block_shape == block == method.weight_block_size and
            method.block_quant and quant.is_block_quantized,
        "weight_layout": all(t.dtype == torch.float8_e4m3fn and
            list(t.shape) == expected[key] for key, t in weights.items()),
        "scale_layout": all(t.dtype == torch.float32 and list(t.shape) == grids[key] and
            t.device == weights[key].device for key, t in
            (("w13", scales["w1"]), ("w2", scales["w2"]))),
        "activation_config": quant.quant_dtype == torch.float8_e4m3fn and
            quant.weight_quant_dtype == torch.float8_e4m3fn and quant.use_fp8_w8a8 and
            experts.expects_unquantized_inputs and experts.use_deepseek_fp8_block_scale,
        "input_scales_absent": all(value is None for value in
            (module.w13_input_scale, module.w2_input_scale, quant.a1_scale, quant.a2_scale,
             experts.a1_scale, experts.a2_scale)),
        "expert_dimensions": experts.num_experts == e and experts.out_dtype == moe.in_dtype,
    }
    return {"backend": backend, "status": "observed; apply remains unverified",
        "quant_config_alias": experts.quant_config is quant, "scale_aliases": aliases,
        "weights": {key: layout(t) for key, t in weights.items()},
        "scales": {key: layout(t) for key, t in scales.items()},
        "block_shape": block, "quant_block_shape": quant.block_shape,
        "dimensions": {"local_experts": e, "hidden": h, "intermediate": i},
        "in_dtype": str(moe.in_dtype), "out_dtype": str(experts.out_dtype),
        "activation": str(moe.activation), "has_bias": moe.has_bias,
        "expert_parallel": {key: getattr(experts, key) for key in
            ("tp_rank", "tp_size", "ep_rank", "ep_size", "use_dp")},
        "quant_flags": {key: getattr(quant, key) for key in
            ("is_block_quantized", "use_fp8_w8a8", "per_act_token_quant", "per_out_ch_quant")},
        "quant_dtype": str(quant.quant_dtype), "weight_quant_dtype": str(quant.weight_quant_dtype),
        "consumer_flags": {key: getattr(experts, key) for key in
            ("expects_unquantized_inputs", "use_deepseek_fp8_block_scale")},
        "input_scale_present": {key: getattr(owner, field) is not None for key, owner, field in
            (("layer_w13", module, "w13_input_scale"), ("layer_w2", module, "w2_input_scale"),
             ("quant_a1", quant, "a1_scale"), ("quant_a2", quant, "a2_scale"),
             ("expert_a1", experts, "a1_scale"), ("expert_a2", experts, "a2_scale"))},
        "auxiliary_present": {key: getattr(experts, key) is not None for key in
            ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit", "w1_bias", "w2_bias")},
        "structural_checks": checks,
        "activation_call": "unverified; constructor state is not apply arguments",
        "scale_domain": "unverified here; match recorded extrema to pinned consumer",
        "w31_content_order": "unverified; metadata does not establish tensor ordering"}


class CutlassCallInventory:
    """Retain selected callable host facts without reading device contents."""
    def __init__(self, limits):
        if any(type(limits[k]) is not int or limits[k] <= 0
               for k in ('records', 'bytes')):
            raise ValueError('Call inventory limits must be positive integers')
        self.limits, self.rows, self.used = limits, [], 0
        self.dropped, self.errors, self.installed, self.rank = 0, 0, False, None
        self.snapshot()

    def install(self, namespace, layers, rank, attention_layers=None):
        import weakref
        self.rank = rank
        sources = [(name, weakref.ref(a), weakref.ref(b)) for name, a, b in layers]
        original = namespace.flashinfer_cutlass_fused_moe

        def metadata(tensor):
            if tensor is None:
                return None
            return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                device=str(tensor.device), stride=list(tensor.stride()),
                offset=tensor.storage_offset(), object_id=id(tensor),
                storage_address=tensor.untyped_storage().data_ptr())

        def context_metadata(layer):
            module = sys.modules.get('vllm.forward_context')
            ctx = None if module is None else vars(module).get('_forward_context')
            if ctx is None:
                return {'status': 'unavailable'}
            fields = vars(ctx)
            key = attention_layers.get(layer)
            raw = fields['attn_metadata']
            attention = raw.get(key) if type(raw) is dict and key is not None else None

            def scalars(obj, names):
                values = {} if obj is None else vars(obj)
                return {k: v if type(v) in (int, bool) or v is None else 'unsupported'
                        for k in names for v in [values.get(k)]}

            dp = fields['dp_metadata']
            sizes = None if dp is None else vars(dp)['local_sizes']
            if sizes is not None:
                sizes = (list(sizes) if type(sizes) is list
                    and len(sizes) <= rank['world_size']
                    and all(type(n) is int for n in sizes) else 'unsupported')
            return dict(attention_layer=key,
                attention_status='observed' if attention is not None else 'unavailable',
                attention=scalars(attention, ('num_reqs', 'num_actual_tokens',
                    'num_decodes', 'num_prefills', 'num_decode_tokens',
                    'prefill_max_seq_len')),
                dp_local_sizes=sizes, graph_mode=fields['cudagraph_runtime_mode'].name,
                batch_descriptor=scalars(fields['batch_descriptor'], ('num_tokens',
                    'num_reqs', 'uniform', 'has_lora', 'num_active_loras')))

        def observed(*args, **kwargs):
            if len(self.rows) >= self.limits['records']:
                self.dropped += 1
                return original(*args, **kwargs)
            row = None
            try:
                scales = kwargs['quant_scales']
                layer = next((name for name, a, b in sources if len(scales) == 2
                    and a() is scales[0] and b() is scales[1]), None)
                row = dict(rank=rank, layer=layer,
                    tensors={k: metadata(kwargs[k]) for k in
                    ('input', 'output', 'fc1_expert_weights', 'fc2_expert_weights',
                     'token_selected_experts', 'token_final_scales')},
                    quant_scales=[metadata(t) for t in scales],
                    input_sf=metadata(kwargs['input_sf']),
                    scalars={k: kwargs[k] for k in ('tp_size', 'tp_rank', 'ep_size',
                        'ep_rank', 'use_deepseek_fp8_block_scale',
                        'use_mxfp8_act_scaling', 'use_w4_group_scaling')},
                    output_dtype=str(kwargs['output_dtype']),
                    activation_type=str(kwargs['activation_type']),
                    device_completion='unobserved')
                if attention_layers is not None:
                    try:
                        row['forward_context'] = context_metadata(layer)
                    except Exception:
                        self.errors += 1
                        row['forward_context'] = {'status': 'error'}
            except Exception:
                self.errors += 1
            try:
                returned = original(*args, **kwargs)
            except BaseException as error:
                self.admit(row, 'native_exception', type(error).__name__)
                raise
            self.admit(row, 'host_return', None)
            return returned

        namespace.flashinfer_cutlass_fused_moe = observed
        self.installed = True

    def admit(self, row, outcome, error):
        if row is None:
            return
        try:
            row.update(native_outcome=outcome, native_error=error)
            encoded = json.dumps(row, separators=(',', ':'))
            size = len(encoded.encode())
            if (len(self.rows) >= self.limits['records']
                    or self.used + size > self.limits['bytes']):
                self.dropped += 1
                return
            self.rows.append(encoded)
            self.used += size
        except Exception:
            self.errors += 1

    def snapshot(self):
        while True:
            payload = json.dumps(dict(rank=self.rank,
                records=[json.loads(row) for row in self.rows],
                retained_bytes=self.used, dropped=self.dropped, errors=self.errors,
                installed=self.installed, run_verified=False,
                coverage='Python-visible calls; prior capture and replay unobserved'),
                separators=(',', ':'))
            if len(payload.encode()) <= self.limits['bytes']:
                return payload
            if not self.rows:
                raise ValueError('Call inventory byte limit cannot hold host footer')
            self.used -= len(self.rows.pop().encode())
            self.dropped += 1


class FullModelInventory:
    def full_model_inventory(self):
        limits = os.environ.get('FRONTIER_CUTLASS_CALL_LIMITS')
        if limits is not None:
            if hasattr(self, '_cutlass_calls'):
                return self._cutlass_calls.snapshot()
            self._cutlass_calls = CutlassCallInventory(json.loads(limits))
        import torch
        from native_dummy_initialization import inspect_model
        from vllm.distributed import get_world_group, get_tp_group, get_dp_group, get_ep_group
        from vllm.v1.worker.workspace import current_workspace_manager
        runner, config = self.model_runner, self.vllm_config
        result = {"host": socket.gethostname(), "worker_rank": self.rank,
                  "device": str(self.device), "request_owner": config.parallel_config.data_parallel_rank}
        result["groups"] = {name: {"rank": group.rank, "ranks": group.ranks,
            "world_size": group.world_size, "rank_in_group": group.rank_in_group,
            "communicator": identity(group.device_communicator)} for name, group in
            (("world", get_world_group()), ("tp", get_tp_group()),
             ("dp", get_dp_group()), ("ep", get_ep_group()))}
        result["resolved"] = {name: str(getattr(config, name)) for name in
            ("parallel_config", "scheduler_config", "compilation_config", "kernel_config")}
        result["scheduler_class"] = identity(config.scheduler_config.get_scheduler_cls())
        result["moe_routing"] = balanced_routing_inventory(self.get_model())
        result["moe_backends"] = {name: {"method": identity(method), "backend": method.fp8_backend.name,
            "experts": identity(method.moe_kernel.fused_experts),
            "prepare_finalize": identity(method.moe_kernel.prepare_finalize),
            "block_shape": method.moe_quant_config.block_shape, "scale_validity": "unverified"}
            for name, module in self.get_model().named_modules()
            if (method := getattr(module, "quant_method", None)) is not None and type(method).__name__ == "Fp8MoEMethod"}
        result["cutlass_state"] = {}
        for name, module in self.get_model().named_modules():
            method = getattr(module, "quant_method", None)
            if method is not None and type(method).__name__ == "Fp8MoEMethod":
                try:
                    result["cutlass_state"][name] = cutlass_state(module)
                except Exception:
                    result["cutlass_state"][name] = {"error": traceback.format_exc()}
        result["cache"] = dataclasses.asdict(runner.kv_cache_config)
        result["attention_groups"] = [{"layers": g.layer_names, "backend": identity(g.backend),
            "builders": [identity(b) for b in g.metadata_builders]} for groups in runner.attn_groups for g in groups]
        result["memory"] = {"model_load_delta_bytes": runner.model_memory_usage,
            "allocator_high_water_bytes": torch.cuda.max_memory_allocated(self.device),
            "allocator_reserved_high_water_bytes": torch.cuda.max_memory_reserved(self.device),
            "current_allocated_bytes": torch.cuda.memory_allocated(self.device),
            "native_scalars": {field: getattr(self, field, None) for field in
                ("requested_memory", "total_consumed", "peak_activation_memory",
                 "cudagraph_memory_estimate", "available_kv_cache_memory_bytes")},
            "independent_stage_peaks": "unverified; these scalars are not separate stage peaks"}
        workspace = current_workspace_manager()
        result["workspace_bytes"] = [0 if t is None else t.numel() * t.element_size()
                                     for t in workspace._current_workspaces]
        try:
            result["initialization"] = inspect_model(self.get_model(), config)
        except Exception:
            result["initialization_error"] = traceback.format_exc()
        if limits is not None:
            try:
                layers = [(name, method.moe_kernel.fused_experts.w1_scale,
                    method.moe_kernel.fused_experts.w2_scale)
                    for name, module in self.get_model().named_modules()
                    if (method := getattr(module, 'quant_method', None)) is not None
                    and type(method).__name__ == 'Fp8MoEMethod'
                    and method.fp8_backend.name == 'FLASHINFER_CUTLASS']
                key = ('vllm.model_executor.layers.fused_moe.experts.'
                       'flashinfer_cutlass_moe')
                owned_attention = []
                for decoder in self.get_model().modules():
                    if type(decoder).__name__ == 'DeepseekV2DecoderLayer':
                        if (hasattr(decoder.mlp, 'experts')
                                and hasattr(decoder.self_attn, 'mla_attn')):
                            owned_attention.append((decoder.mlp.experts.routed_experts,
                                decoder.self_attn.mla_attn.mla_attn.layer_name))
                attention_layers = {name: key
                    for name, module in self.get_model().named_modules()
                    for owner, key in owned_attention if owner is module}
                self._cutlass_calls.install(sys.modules[key], layers,
                    result['groups']['world'], attention_layers)
            except Exception:
                self._cutlass_calls.errors += 1
            result['cutlass_calls'] = json.loads(self._cutlass_calls.snapshot())
        return json.dumps(result, default=str)


class NativeServingInventory:
    """Bounded host observations, never a verdict on device completion or timing.

    The evidence directory is the node directory. Both emissions atomically
    update ``serving/rank<worker.rank>.json``; pre_shutdown is cumulative.
    Counter rows distinguish native serving, profile, dummy and capture calls.
    Only original Python calls returning successfully increment counters.
    """

    MAX_ROWS = 4096
    MAX_ERRORS = 64
    # The deployed module is standalone. Keep this fixed-workload ID contract
    # source-tested against build_fixed_manifest rather than importing tests.
    REQUEST_NAMESPACE = 'fixed-6144-1024-req128-qps1'
    REQUEST_LIMITS = {'formal': 128, 'warmup': 16}
    REPEATS = (1, 2, 3)

    def __init__(self, directory, generation, role):
        import threading
        self.directory = Path(directory)
        self.generation, self.role = generation, role
        self.local = threading.local()
        self.rows, self.requests, self.errors, self.hooks = {}, {}, [], []
        self.forwards = self.replays = self.dropped = 0
        self.inventory = None
        if not generation or role not in ('prefill', 'decode'):
            self.error('provenance', ValueError('Missing run generation or invalid native role'))

    def error(self, where, error):
        # Even a broken stderr or error formatter must not affect native work.
        try:
            row = {'where': where, 'error': f'{type(error).__name__}: {error}'}
            if len(self.errors) < self.MAX_ERRORS and row not in self.errors:
                self.errors.append(row)
                print('NATIVE_SERVING_UNVERIFIED', row, file=sys.stderr, flush=True)
        except Exception:
            pass

    def observe(self, where, operation, *args):
        try:
            return operation(*args)
        except Exception as error:
            self.error(where, error)
            return None

    def scheduler_context(self, output):
        import re
        scheduled = output.num_scheduled_tokens
        if type(scheduled) is not dict or any(
                type(n) is not int or n < 0 for n in scheduled.values()):
            raise ValueError('Malformed native num_scheduled_tokens')
        active = [name for name, n in scheduled.items() if n > 0]
        if sum(scheduled.values()) != output.total_num_scheduled_tokens:
            raise ValueError('Native scheduled token total disagrees with request map')
        context = dict(kind='unattributed', repeat=None, phase=None,
            scheduled_num_requests=len(active), scheduled_num_tokens=sum(scheduled.values()))
        if not active:
            return context
        phases = set()
        for name in active:
            # Native completion adds -0; InputProcessor normally appends an
            # 8-hex suffix, omitted only when native randomization is disabled.
            match = re.fullmatch(r'cmpl-' + re.escape(self.REQUEST_NAMESPACE)
                + r'-(warmup|formal)-r(\d{3})-repeat(\d{2})-0(?:-[0-9a-f]{8})?', name)
            if match is None:
                raise ValueError(f'Unrecognized native request ID: {name!r}')
            phase, index, repeat = match.groups()
            if (int(index) >= self.REQUEST_LIMITS[phase]
                    or int(repeat) not in self.REPEATS):
                raise ValueError(f'Out-of-contract native request ID: {name!r}')
            phases.add((int(repeat), phase))
        if len(phases) != 1:
            raise ValueError('Mixed repeat/phase IDs violate the drained phase contract')
        repeat, phase = phases.pop()
        self.requests.setdefault((repeat, phase), set()).update(active)
        return context | dict(kind='serving', repeat=repeat, phase=phase)

    def forward_metadata(self, shapes=None):
        full = getattr(self.local, 'full_graph_metadata', None)
        if full is not None:
            return full
        from vllm.forward_context import get_forward_context
        ctx = get_forward_context()
        mode = ctx.cudagraph_runtime_mode.name
        if mode not in ('NONE', 'PIECEWISE', 'FULL'):
            raise ValueError(f'Unsupported native runtime graph mode: {mode}')
        batch = ctx.batch_descriptor
        descriptor = None if batch is None else {
            key: getattr(batch, key) for key in
            ('num_tokens', 'num_reqs', 'uniform', 'has_lora', 'num_active_loras')}
        if descriptor is not None and any(
                value is not None and type(value) not in (int, bool)
                for value in descriptor.values()):
            raise ValueError('Non-scalar native batch descriptor')
        context = getattr(self.local, 'context', None) or dict(
            kind='unattributed', repeat=None, phase=None,
            scheduled_num_requests=None, scheduled_num_tokens=None)
        return context | dict(graph_mode=mode, batch_descriptor=descriptor,
            input_shapes=shapes if shapes is not None else getattr(self.local, 'shapes', None))

    def count(self, event, metadata):
        if event == 'forward':
            self.forwards += 1
        else:
            self.replays += 1
        if metadata is None:
            return
        dispatch = getattr(self.local, 'dispatch_evidence', None)
        if dispatch is not None:
            if event == 'forward':
                dispatch['forwards'] += 1
                if dispatch['forwards'] != 1:
                    raise ValueError('Ambiguous V2 dispatch-to-forward association')
            else:
                dispatch['replays'] += 1
            metadata = metadata | {'dispatch_evidence': {
                k: v for k, v in dispatch.items() if k not in ('forwards', 'replays')}}
        key = json.dumps(metadata, sort_keys=True, separators=(',', ':'))
        if key not in self.rows:
            if len(self.rows) >= self.MAX_ROWS:
                self.dropped += 1
                self.error('counter_capacity', ValueError('Native counter row limit exceeded'))
                return
            self.rows[key] = metadata | dict(successful_forwards=0, successful_replays=0)
        self.rows[key]['successful_' + ('forwards' if event == 'forward' else 'replays')] += 1

    def model_call(self, original, owner, args, kwargs):
        previous = getattr(self.local, 'shapes', None)

        def shapes():
            # The pinned _model_forward and dummy model call pass keyword tensors.
            return {key: None if kwargs.get(key) is None else list(kwargs[key].shape)
                for key in ('input_ids', 'positions', 'inputs_embeds')}

        current = self.observe('input_shapes', shapes)
        self.local.shapes = current
        metadata = self.observe('forward_context', self.forward_metadata, current)
        try:
            result = original(owner, *args, **kwargs)
            self.observe('forward_counter', self.count, 'forward', metadata)
            return result
        finally:
            self.local.shapes = previous

    def install_dummy_call(self, runner):
        # _dummy_run bypasses _model_forward in pinned V1 (line 6266).
        # Observe its actual outer model call, not the dummy scheduler return.
        cls = type(runner.model)
        original = cls.__call__
        if getattr(original, '_frontier_native_dummy_call', False):
            return
        observer = self

        def call(model, *args, **kwargs):
            if model is getattr(observer.local, 'dummy_model', None):
                return observer.model_call(original, model, args, kwargs)
            return original(model, *args, **kwargs)

        call._frontier_native_dummy_call = True
        cls.__call__ = call

    def install_v2_model_call(self, runner):
        cls = type(runner.model)
        original = cls.__call__
        if getattr(original, '_frontier_native_v2_call', False):
            return
        observer = self

        def call(model, *args, **kwargs):
            if model is runner.model and not getattr(observer.local, 'v2_managed_forward', False):
                return observer.model_call(original, model, args, kwargs)
            return original(model, *args, **kwargs)

        call._frontier_native_v2_call = True
        cls.__call__ = call

    def install_v2_dispatch(self, namespace, dp_module):
        """Observe actual host operands within the existing execution scope."""
        import functools
        import inspect
        observer = self

        def record(key, value):
            state = getattr(observer.local, 'dispatch_evidence', None)
            if state is None:
                return
            if key in state:
                raise ValueError('Ambiguous V2 ' + key + ' observation')
            state[key] = value

        def descriptor(desc):
            return dataclasses.asdict(desc) | {'cg_mode': desc.cg_mode.name}

        def wrap(name, after):
            original = getattr(namespace, name)
            signature = inspect.signature(original)

            def arguments(args, kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                return bound.arguments

            @functools.wraps(original)
            def call(*args, **kwargs):
                inputs = observer.observe(name + '_inputs', arguments, args, kwargs)
                result = original(*args, **kwargs)
                observer.observe(name + '_return', after, inputs, result)
                return result

            setattr(namespace, name, call)
            observer.hooks.append(namespace.__name__ + '.' + name)

        def classified(inputs, result):
            record('classifier', dict(inputs=inputs, uniform_token_count=result))

        def dispatched(inputs, result):
            values = {k: v for k, v in inputs.items() if k != 'cudagraph_manager'}
            desc, sizes = result
            if sizes is not None and sizes.device.type != 'cpu':
                raise ValueError('V2 dispatch sizes are not CPU operands')
            record('dispatch', dict(inputs=values, descriptor=descriptor(desc),
                num_tokens_across_dp=None if sizes is None else sizes.tolist()))

        native_dist = dp_module.dist

        class Dist:
            def __getattr__(self, name):
                return getattr(native_dist, name)

            def all_reduce(self, tensor, *args, **kwargs):
                result = native_dist.all_reduce(tensor, *args, **kwargs)

                def snapshot():
                    state = getattr(observer.local, 'dispatch_evidence', None)
                    if state is None:
                        return
                    if tensor.device.type != 'cpu' or len(tensor.shape) != 2 or tensor.shape[0] != 4:
                        raise ValueError('V2 DP synchronization is not a CPU 4-by-DP tensor')
                    record('dp_post_all_reduce', tensor.tolist())

                observer.observe('dp_post_all_reduce', snapshot)
                return result

        wrap('get_uniform_decode_token_count', classified)
        wrap('dispatch_cg_and_sync_dp', dispatched)
        dp_module.dist = Dist()
        observer.hooks.append(dp_module.__name__ + '.dist.all_reduce')

    def finish_v2_dispatch(self):
        evidence = self.local.dispatch_evidence
        if evidence['replays'] and evidence['forwards'] != 1:
            raise ValueError('Successful replay lacks one successful outer forward')

    def collect_inventory(self, worker):
        import torch
        row = json.loads(FullModelInventory.full_model_inventory(worker))
        props = torch.cuda.get_device_properties(worker.device)
        row['device_identity'] = dict(pid=os.getpid(), device=str(worker.device),
            name=props.name, uuid=str(props.uuid),
            capability=list(torch.cuda.get_device_capability(worker.device)))
        runner, config = worker.model_runner, worker.vllm_config.compilation_config
        row['model_runner'] = identity(runner)
        if worker.use_v2_model_runner:
            manager = runner.cudagraph_manager
            row['effective_graph'] = dict(configured_mode=config.cudagraph_mode.name,
                dispatcher_mode=manager.cudagraph_mode.name,
                keys_initialized=manager._graphs_captured,
                capture_sizes=list(config.cudagraph_capture_sizes or []),
                max_capture_size=config.max_cudagraph_capture_size,
                captured_descriptors=[dataclasses.asdict(desc) | {'cg_mode': desc.cg_mode.name}
                    for desc in manager.graphs])
        else:
            dispatcher = runner.cudagraph_dispatcher
            row['effective_graph'] = dict(configured_mode=config.cudagraph_mode.name,
                dispatcher_mode=dispatcher.cudagraph_mode.name,
                keys_initialized=dispatcher.keys_initialized,
                capture_sizes=list(config.cudagraph_capture_sizes or []),
                max_capture_size=config.max_cudagraph_capture_size)
        return row

    def emit(self, worker, stage):
        if stage == 'post_warmup':
            self.inventory = self.observe('inventory', self.collect_inventory, worker)
        elif self.inventory is not None and self.inventory.get('moe_routing') is not None:
            self.inventory['moe_routing'] = self.observe('routing_inventory', balanced_routing_inventory, worker.get_model())
        payload = dict(schema='native_serving_v1', run_generation=self.generation,
            role=self.role, host=socket.gethostname(), stage=stage,
            inventory=self.inventory, observation_errors=list(self.errors),
            hooks=list(self.hooks),
            counters=dict(successful_forwards=self.forwards, successful_replays=self.replays,
                rows=list(self.rows.values()), dropped_rows=self.dropped,
                request_ids=[dict(repeat=repeat, phase=phase, request_ids=sorted(names))
                    for (repeat, phase), names in sorted(self.requests.items())]),
            coverage='Successful Python host returns only; no device completion or timing claim')
        directory = self.directory / 'serving'
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f'rank{worker.rank}.json'
        temporary = target.with_suffix(f'.json.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(payload, sort_keys=True) + '\n')
        os.replace(temporary, target)


def _install_native_serving_inventory():
    """Install once per physical worker, before native worker initialization."""
    directory = os.environ.get('FRONTIER_NATIVE_SERVING_EVIDENCE_DIR')
    if directory is None:
        return None
    observer = NativeServingInventory(directory,
        os.environ.get('FRONTIER_NATIVE_RUN_GENERATION'), os.environ.get('FRONTIER_NATIVE_ROLE'))
    try:
        import functools
        import inspect
        import torch
        from vllm.v1.worker.gpu_worker import Worker
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        installed = getattr(Worker.compile_or_warm_up_model,
            '_frontier_native_serving_observer', None)
        if installed is not None:
            return installed

        def install(cls, name, wrapper):
            original = getattr(cls, name)
            wrapped = functools.wraps(original)(wrapper(original))
            wrapped._frontier_native_serving_observer = observer
            setattr(cls, name, wrapped)
            observer.hooks.append(identity(cls) + '.' + name)

        def warmup(original):
            def call(worker, *args, **kwargs):
                previous = getattr(observer.local, 'context', None)
                previous_warmup = getattr(observer.local, 'startup_warmup', False)
                # Native V2 kernel warmup uses ordinary SchedulerOutputs, not
                # dummy_run=True. Its caller, not request-ID spelling, proves
                # these forwards are startup work rather than client serving.
                observer.local.context = dict(kind='dummy', repeat=None, phase=None,
                    scheduled_num_requests=None, scheduled_num_tokens=None)
                observer.local.startup_warmup = True
                try:
                    result = original(worker, *args, **kwargs)
                finally:
                    observer.local.context = previous
                    observer.local.startup_warmup = previous_warmup
                observer.observe('post_warmup_dump', observer.emit, worker, 'post_warmup')
                return result
            return call

        def shutdown(original):
            def call(worker, *args, **kwargs):
                observer.observe('pre_shutdown_dump', observer.emit, worker, 'pre_shutdown')
                return original(worker, *args, **kwargs)
            return call

        def execute(original):
            def call(worker, *args, **kwargs):
                previous = getattr(observer.local, 'context', None)
                output = args[0] if args else kwargs.get('scheduler_output')
                if not getattr(observer.local, 'startup_warmup', False):
                    observer.local.context = observer.observe(
                        'scheduler_provenance', observer.scheduler_context, output)
                try:
                    return original(worker, *args, **kwargs)
                finally:
                    observer.local.context = previous
            return call

        def forward(original):
            def call(runner, *args, **kwargs):
                return observer.model_call(original, runner, args, kwargs)
            return call

        def dummy(original):
            signature = inspect.signature(original)

            def context(runner, args, kwargs):
                bound = signature.bind(runner, *args, **kwargs)
                bound.apply_defaults()
                kind = ('capture' if bound.arguments['is_graph_capturing'] else
                    'profile' if bound.arguments['is_profile'] else 'dummy')
                return dict(kind=kind, repeat=None, phase=None,
                    scheduled_num_requests=None, scheduled_num_tokens=None)

            def call(runner, *args, **kwargs):
                previous = getattr(observer.local, 'context', None)
                previous_model = getattr(observer.local, 'dummy_model', None)
                observer.local.context = observer.observe('dummy_context', context, runner, args, kwargs)
                observer.local.dummy_model = getattr(runner, 'model', None)
                observer.observe('dummy_model_call_install', observer.install_dummy_call, runner)
                try:
                    return original(runner, *args, **kwargs)
                finally:
                    observer.local.context = previous
                    observer.local.dummy_model = previous_model
            return call

        def replay(original):
            def call(graph, *args, **kwargs):
                metadata = observer.observe('replay_context', observer.forward_metadata)
                result = original(graph, *args, **kwargs)
                observer.observe('replay_counter', observer.count, 'replay', metadata)
                return result
            return call

        def install_v2(runner):
            from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
            cls = type(runner)
            if getattr(cls.execute_model, '_frontier_native_serving_observer', None) is observer:
                return
            execute_signature = inspect.signature(cls.execute_model)

            def v2_execute(original):
                def arguments(model_runner, args, kwargs):
                    bound = execute_signature.bind(model_runner, *args, **kwargs)
                    bound.apply_defaults()
                    return bound.arguments

                def call(model_runner, *args, **kwargs):
                    previous = getattr(observer.local, 'context', None)
                    previous_runner = getattr(observer.local, 'runner', None)
                    previous_dispatch = getattr(observer.local, 'dispatch_evidence', None)
                    bound = observer.observe('v2_execution_arguments', arguments, model_runner, args, kwargs)
                    if bound is not None and bound['dummy_run']:
                        observer.local.context = dict(kind='profile' if bound['is_profile'] else 'dummy',
                            repeat=None, phase=None, scheduled_num_requests=None, scheduled_num_tokens=None)
                    observer.local.runner = model_runner
                    observer.local.dispatch_evidence = {'forwards': 0, 'replays': 0}
                    try:
                        return original(model_runner, *args, **kwargs)
                    finally:
                        observer.observe('v2_forward_association', observer.finish_v2_dispatch)
                        observer.local.context = previous
                        observer.local.runner = previous_runner
                        observer.local.dispatch_evidence = previous_dispatch
                return call

            def capture(original):
                def call(manager, *args, **kwargs):
                    previous = getattr(observer.local, 'context', None)
                    observer.local.context = dict(kind='capture', repeat=None, phase=None,
                        scheduled_num_requests=None, scheduled_num_tokens=None)
                    try:
                        return original(manager, *args, **kwargs)
                    finally:
                        observer.local.context = previous
                return call

            def full(original):
                def metadata(desc):
                    active = observer.local.runner
                    buffers = active.input_buffers
                    context = observer.local.context
                    return context | dict(graph_mode=desc.cg_mode.name,
                        batch_descriptor=dict(num_tokens=desc.num_tokens, num_reqs=desc.num_reqs,
                            uniform=None, has_lora=desc.num_active_loras > 0,
                            num_active_loras=desc.num_active_loras),
                        input_shapes={key: list(getattr(buffers, key)[:desc.num_tokens].shape)
                            if key in ('input_ids', 'positions') else None
                            for key in ('input_ids', 'positions', 'inputs_embeds')})

                def call(manager, desc, *args, **kwargs):
                    previous = getattr(observer.local, 'full_graph_metadata', None)
                    data = observer.observe('v2_full_context', metadata, desc)
                    observer.local.full_graph_metadata = data
                    try:
                        result = original(manager, desc, *args, **kwargs)
                        observer.observe('v2_full_forward', observer.count, 'forward', data)
                        return result
                    finally:
                        observer.local.full_graph_metadata = previous
                return call

            def piecewise(original):
                def call(manager, model, model_inputs):
                    previous = getattr(observer.local, 'v2_managed_forward', False)
                    previous_shapes = getattr(observer.local, 'shapes', None)

                    def shapes():
                        return {key: None if model_inputs.get(key) is None else list(model_inputs[key].shape)
                            for key in ('input_ids', 'positions', 'inputs_embeds')}

                    observer.local.shapes = observer.observe('v2_piecewise_shapes', shapes)
                    data = observer.observe('v2_piecewise_context', observer.forward_metadata)
                    observer.local.v2_managed_forward = True
                    try:
                        result = original(manager, model, model_inputs)
                        observer.observe('v2_piecewise_forward', observer.count, 'forward', data)
                        return result
                    finally:
                        observer.local.v2_managed_forward = previous
                        observer.local.shapes = previous_shapes
                return call

            install(cls, 'execute_model', v2_execute)
            install(ModelCudaGraphManager, 'capture', capture)
            install(ModelCudaGraphManager, 'run_fullgraph', full)
            install(ModelCudaGraphManager, 'run_pw_graph', piecewise)
            observer.install_v2_model_call(runner)
            observer.install_v2_dispatch(sys.modules[cls.__module__],
                sys.modules['vllm.v1.worker.gpu.dp_utils'])

        def load(original):
            def call(worker, *args, **kwargs):
                result = original(worker, *args, **kwargs)
                if worker.use_v2_model_runner:
                    observer.observe('v2_installation', install_v2, worker.model_runner)
                return result
            return call

        install(Worker, 'execute_model', execute)
        install(GPUModelRunner, '_model_forward', forward)
        install(GPUModelRunner, '_dummy_run', dummy)
        install(torch.cuda.CUDAGraph, 'replay', replay)
        install(Worker, 'shutdown', shutdown)
        install(Worker, 'compile_or_warm_up_model', warmup)
        install(Worker, 'load_model', load)
    except Exception as error:
        observer.error('installation', error)
    return observer


if os.environ.get('VLLM_MOE_ROUTING_SIMULATION_STRATEGY') == 'balanced_topk':
    from balanced_routing import register_balanced_topk
    register_balanced_topk()

_NATIVE_SERVING_INVENTORY = _install_native_serving_inventory()


def routing_rows(array, prompt_tokens, output_tokens, hf):
    import numpy as np
    expected = (prompt_tokens + output_tokens - 1, hf["num_hidden_layers"], hf["num_experts_per_tok"])
    if array.shape != expected:
        raise ValueError(f"Routing shape {array.shape} differs from expected {expected}")
    layers = [i for i, kind in enumerate(hf["mlp_layer_types"]) if kind == "sparse"]
    rows = []
    for phase, values in (("prompt_inputs", array[:prompt_tokens]),
                          ("decode_inputs", array[prompt_tokens:])):
        for layer in layers:
            ids = values[:, layer].reshape(-1)
            if not np.issubdtype(ids.dtype, np.integer) or not ((ids >= 0) & (ids < hf["n_routed_experts"])).all():
                raise ValueError(f"Invalid routing IDs in layer {layer}, phase {phase}")
            rows.append({"phase": phase, "layer": layer, "token_inputs": len(values),
                         "counts": np.bincount(ids.astype(np.int64), minlength=hf["n_routed_experts"]).tolist()})
    return {"rows": rows, "non_moe_layers": [i for i in range(expected[1]) if i not in layers],
            "execution_shape_coverage": "unverified; final request arrays do not encode native batch shapes"}


def run(config):
    import numpy as np
    from probe_nixl_read import SYSFS_INFINIBAND, gid_table, roce_v2_ipv4_ports
    from probe_runtime_inventory import primary_ipv4, read_text, sha256, wait_for, weight_candidates, write_json
    root = Path(config["output_dir"])
    if not root.is_relative_to(Path("/mnt/codesign-exp/ycfeng")):
        raise ValueError("Output must remain in the personal cloud directory")
    node = root / "nodes" / socket.gethostname()
    node.mkdir(parents=True, exist_ok=False)
    with (node / "worker.log").open("w", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    result = {"scope": "G0-B full structure; diagnostic observations, not complete qualification",
              "configuration": config, "passed": False, "observation_errors": []}

    def stage(name):
        result["stage"] = name
        write_json(node / "result.json", result)
        print("G0B_STAGE", name, flush=True)

    try:
        for path, digest in config["tool_hashes"].items():
            if sha256(path) != digest:
                raise RuntimeError(f"Tool hash mismatch: {path}")
        scratch = Path(tempfile.mkdtemp(prefix="g0b_"))
        for name, child in config["cache_environment"].items():
            os.environ[name] = str(scratch / child)
        stage("rendezvous")
        write_json(node / "ready.json", {"host": socket.gethostname(), "ip": primary_ipv4()})
        if not wait_for(lambda: len(list(root.glob("nodes/*/ready.json"))) == config["replica"], config["peer_wait_seconds"]):
            raise TimeoutError("Not all distinct physical nodes arrived")
        peers = sorted((json.loads(p.read_text()) for p in root.glob("nodes/*/ready.json")), key=lambda p: p["host"])
        rank = [p["host"] for p in peers].index(socket.gethostname())
        result["peers"], result["request_owner"] = peers, rank
        os.environ.update(VLLM_DP_SIZE=str(config["replica"]), VLLM_DP_RANK=str(rank),
            VLLM_DP_RANK_LOCAL="0", VLLM_DP_MASTER_IP=peers[0]["ip"], VLLM_DP_MASTER_PORT=str(config["dp_master_port"]))
        rdma = {d.name: {p.name: {"state": read_text(p / "state"), "link_layer": read_text(p / "link_layer"),
            "gids": gid_table(p)} for p in (d / "ports").glob("*")} for d in SYSFS_INFINIBAND.glob("*")}
        result["ucx_devices"] = roce_v2_ipv4_ports(rdma)
        if not result["ucx_devices"]:
            raise RuntimeError("No ACTIVE RoCE v2 IPv4 port")
        os.environ["UCX_NET_DEVICES"] = ",".join(result["ucx_devices"])
        import torch
        import vllm
        from vllm import LLM, SamplingParams
        result["versions"] = {"python": sys.version, "torch": torch.__version__, "vllm": vllm.__version__}
        result["gpus"] = [{"name": torch.cuda.get_device_name(i), "capability": torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]
        source = Path(vllm.__file__).parent.parent
        if vllm.__version__ != config["expected_vllm_version"] or any(sha256(source / p) != h for p, h in config["expected_source_hashes"].items()):
            raise RuntimeError("Pinned runtime/source identity mismatch")
        model = Path(config["llm"]["model"])
        if {p.name: sha256(p) for p in model.iterdir()} != config["model_hashes"]:
            raise RuntimeError("Metadata allowlist/hash mismatch")
        if 'cutlass_call_limits' in config:
            CutlassCallInventory(config['cutlass_call_limits'])
            os.environ['FRONTIER_CUTLASS_CALL_LIMITS'] = json.dumps(
                config['cutlass_call_limits'])
        stage("full_initialization")
        llm = LLM(**config["llm"], worker_extension_cls="native_full_structure.FullModelInventory")
        stage("inventory")
        try:
            result["inventory"] = [json.loads(row) for row in llm.collective_rpc("full_model_inventory")]
        except Exception:
            result["observation_errors"].append(traceback.format_exc())
        stage("generation")
        hf = json.loads((model / "config.json").read_text())
        prompts = [{"prompt_token_ids": [config["input_token_id"]] * n} for n in config["prompt_lengths"][rank]]
        responses = llm.generate(prompts, SamplingParams(temperature=0, ignore_eos=True,
            max_tokens=config["decode_tokens"], logprobs=1), use_tqdm=False)
        result["responses"] = []
        for index, response in enumerate(responses):
            output = response.outputs[0]
            row = {"request_id": response.request_id, "prompt_tokens": len(response.prompt_token_ids),
                   "token_ids": output.token_ids, "finish_reason": output.finish_reason,
                   "logprobs_finite": all(math.isfinite(v.logprob) for token in output.logprobs for v in token.values())}
            row["routing"] = {"status": "unverified", "reason": "native routing-return observation disabled"}
            if config["llm"]["enable_return_routed_experts"]:
                try:
                    np.save(node / f"routing_{index}.npy", output.routed_experts, allow_pickle=False)
                    row["routing"] = routing_rows(output.routed_experts, row["prompt_tokens"], len(output.token_ids), hf)
                except Exception:
                    result["observation_errors"].append(traceback.format_exc())
            result["responses"].append(row)
        if 'cutlass_call_limits' in config:
            try:
                result['cutlass_calls'] = [json.loads(row) for row in
                    llm.collective_rpc('full_model_inventory')]
            except Exception:
                result['observation_errors'].append(traceback.format_exc())
        result["checks"] = {"prompt_counts": [r["prompt_tokens"] for r in result["responses"]] == config["prompt_lengths"][rank],
            "decode_counts": all(len(r["token_ids"]) == config["decode_tokens"] for r in result["responses"]),
            "finite_logprobs": all(r["logprobs_finite"] for r in result["responses"])}
        result["weight_audit"] = weight_candidates([scratch / "hf", model], tuple(config["weight_suffixes"]))
        result["checks"]["no_weight_files"] = not result["weight_audit"]
        result["passed"] = all(result["checks"].values())
        stage("generation_returned")
        write_json(node / "drained.json", {"request_owner": rank})
        if not wait_for(lambda: len(list(root.glob("nodes/*/drained.json"))) == config["replica"], config["peer_wait_seconds"]):
            raise TimeoutError("Not all DP owners returned from paired generation")
        llm.llm_engine.engine_core.shutdown()
        stage("shutdown_returned")
        return 0 if result["passed"] else 1
    except Exception:
        result["exception"] = traceback.format_exc()
        stage("failed")
        print(result["exception"], flush=True)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    sys.exit(run(json.loads(parser.parse_args().config.read_text())))
