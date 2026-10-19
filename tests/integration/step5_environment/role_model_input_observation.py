"""Source-pinned MRv2 metadata observations; device execution remains unqualified.

The scoped hooks read native frames and actual replay callees. They never replace
runner, graph, sampler, or capture functions. Capture records are CPU metadata only. Separately prepared diagnostic device
witnesses use owned destinations outside capture and are read only after the
native request returns. Native compatibility and complete G2 qualification
require separately frozen runtime evidence.
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


# The finite source table belongs to this native observer, not Frontier's model or
# operator registries. It reads only already imported, constructed native owners.
CENSUS_SOURCES = {
    'model': ('vllm.models.step5.nvidia.model', '6e8574c31c421b4ebeaff36a8faa3a219cd92744251c477d43df1912a8dfa6f0'),
    'norm': ('vllm.models.step5.nvidia.layernorm', 'f6213b9466d2bfe62fa9021dc8404b3c8413ff4a08e964b5b43dbe110d31b510'),
    'linear': ('vllm.model_executor.layers.linear', '7a9b90937865fa35d955ffc2017f23d3997bc4fba954e2667870b1b4854a38e0'),
    'gate': ('vllm.model_executor.layers.fused_moe.router.gate_linear', 'ba05b95f4bece910a2df67c7ebfbaf9519b629bb65ba953afc3a820539cab98b'),
    'moe_runner': ('vllm.model_executor.layers.fused_moe.runner.moe_runner', '20b69fe514bebd6723a1378f89a564c8fec4144c30244101f629c3f32c188183'),
    'routed': ('vllm.model_executor.layers.fused_moe.routed_experts', '7b74a9792ce2da26d503185bae7fb4607a6ea41130c98cc820d4e39d38273c43'),
    'fp8': ('vllm.model_executor.layers.quantization.fp8', '37db0145189cd8870d8ac53a6eeb59ac070af51fe78e933597f51e8948250403'),
    'unquantized': ('vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method', '1ab96e74cce2d1a8540af0eb5518cf8ff77d02c7b770d55c2556cdb34c74bc1c'),
    'modular': ('vllm.model_executor.layers.fused_moe.modular_kernel', '82a5d100f42d3df3d20baea4a86e5147d4d5c17e3775ed7b5a1ce4bfb433fe0a'),
    'triton_experts': ('vllm.model_executor.layers.fused_moe.experts.triton_moe', 'ee6b83b108a0460e6308e35447434165d0778a053aa39f0c6c8ed7f36a3ddbab'),
    'triton_dispatch': ('vllm.model_executor.layers.fused_moe.fused_moe', '21860340e4b409b01666e2ba0f0edae66b0f97b859756b174b573bab895cfaf7'),
    'runner': ('vllm.v1.worker.gpu.model_runner', SOURCE_PINS['runner']),
    'request_state': ('vllm.v1.worker.gpu.states', '5ca84b57d4f4233ba4013ed42f44b7917cdd7a14b6ea4dabf4ba8de5c9d5f4d6'),
    'block_table': ('vllm.v1.worker.gpu.block_table', '822b65f3d7bd4cad47693c101ba5134923876b9c81d5a609afd6bca6dee38ba8'),
    'summary': ('vllm.models.step5.nvidia.sparse_summary_cache', '35d41f0d11d1407f688dd1d0000a73e84195e49e5da9cefb7fde1fb762411047'),
    'worker_utils': ('vllm.v1.worker.utils', '64e8f82545c63cd9026e30731d3460f420e623dfafc4ba0e3b63035c96b2856c'),
    'workspace': ('vllm.v1.worker.workspace', '681c99f19bac3d8c445c887113e61c75754c9ae615530abfe5014e7b7b47ef10'),
    'prepare_no_dp_ep': ('vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep', '263c30ea61e756d389dfbb25927feb3c0fde5936001f1f00215bd15913af0b98'),
}
# source, original code owner, observed arguments, resident owner attributes
CENSUS_CALLS = {
    'backbone': ('model', ('Step5Model', 'forward'), ('input_ids', 'positions', 'inputs_embeds', 'intermediate_tensors'), ('start_layer', 'end_layer', 'fp32_residual_connection', 'use_sequence_parallel')),
    'layer': ('model', ('Step5DecoderLayer', 'forward'), ('positions', 'hidden_states', 'residual', 'padding'), ('layer_idx', 'use_moe', 'fp32_residual_connection', 'use_fused_all_reduce', 'use_sequence_parallel')),
    'ffn': ('model', ('Step5DecoderLayer', '_forward_ffn'), ('hidden_states', 'padding'), ()),
    'attention': ('model', ('Step5Attention', 'forward'), ('positions', 'hidden_states'), ('num_heads', 'num_kv_heads', 'head_dim', 'params_dtype')),
    'mlp': ('model', ('Step5MLP', 'forward'), ('hidden_states',), ('reduce_in_fp32', 'shard_sequence_parallel', 'limit')),
    'moe': ('model', ('Step5MoE', 'forward'), ('hidden_states', 'padding'), ('layer_idx', 'n_routed_experts', 'n_local_physical_experts', 'fuse_all_reduce', 'use_mega_moe')),
    'norm': ('norm', ('Step5RMSNorm', 'forward'), ('x', 'residual', 'output', 'out_dtype'), ('weight', 'variance_epsilon', 'zero_centered')),
    'router': ('gate', ('GateLinear', 'forward'), ('x',), ('weight', 'out_dtype', 'allow_ll_bf16_gemm', 'allow_fp32_router_gemm', 'allow_bf16x3_router_gemm', 'allow_cublas_router_gemm')),
    'router_dispatch': ('gate', ('fp32_router_gemm_dispatch_impl',), ('x', 'weight', 'allow_bf16x3_router_gemm'), ()),
    'linear_replicated': ('linear', ('ReplicatedLinear', 'forward'), ('x',), ('weight', 'bias', 'return_bias')),
    'linear_column': ('linear', ('ColumnParallelLinear', 'forward'), ('input_',), ('weight', 'bias', 'gather_output')),
    'linear_row': ('linear', ('RowParallelLinear', 'forward'), ('input_',), ('weight', 'bias', 'reduce_results', 'input_is_parallel')),
    'moe_routing': ('moe_runner', ('MoERunner', '_apply_quant_method'), ('hidden_states', 'router_logits', 'shared_experts_input'), ()),
    'routed_modular': ('routed', ('RoutedExperts', 'forward_modular'), ('x', 'topk_weights', 'topk_ids'), ('w13_weight', 'w2_weight', 'w13_weight_scale_inv', 'w2_weight_scale_inv', 'w13_input_scale', 'w2_input_scale', 'expert_map', 'global_num_experts', 'local_num_experts', 'intermediate_size_per_partition')),
    'routed_monolithic': ('routed', ('RoutedExperts', 'forward_monolithic'), ('x', 'router_logits', 'input_ids'), ()),
    'fp8_apply': ('fp8', ('Fp8MoEMethod', 'apply'), ('x', 'topk_weights', 'topk_ids'), ('weight_block_size', 'moe_block_shape', 'weight_scale_refine', 'weight_scale_name', 'fp8_backend')),
    'unquantized_apply': ('unquantized', ('UnquantizedFusedMoEMethod', 'forward_native'), ('x', 'topk_weights', 'topk_ids'), ('unquantized_backend',)),
    'moe_prepare': ('modular', ('FusedMoEKernelModularImpl', '_prepare'), ('hidden_states', 'topk_weights', 'topk_ids', 'expert_map'), ('is_dp_ep',)),
    'triton_experts': ('triton_experts', ('TritonExperts', 'apply'), ('hidden_states', 'w1', 'w2', 'a1q_scale', 'a2_scale', 'topk_ids', 'topk_weights'), ('quant_dtype', 'block_shape', 'w1_scale', 'w2_scale', 'a1_scale', 'a2_scale', 'per_act_token_quant', 'quantization_emulation')),
    'triton_gemm': ('triton_dispatch', ('invoke_fused_moe_triton_kernel',), ('A', 'B', 'C', 'A_scale', 'B_scale', 'topk_weights', 'sorted_token_ids', 'expert_ids', 'num_tokens_post_padded', 'top_k', 'config', 'compute_type', 'use_fp8_w8a8', 'block_shape'), ()),
    'prepare_no_dp_ep': ('prepare_no_dp_ep', ('MoEPrepareAndFinalizeNoDPEPModular', 'prepare'), ('a1', 'topk_weights', 'topk_ids', 'expert_map', 'quant_config', 'defer_input_quant'), ()),
    'prepare_quantize': ('prepare_no_dp_ep', ('_quantize_input',), ('a1', 'quant_config', 'defer_input_quant'), ()),
    'request_finish': ('runner', ('GPUModelRunner', 'finish_requests'), (), ()),
    'request_remove': ('runner', ('GPUModelRunner', '_remove_request'), ('req_id',), ()),
    'request_add': ('runner', ('GPUModelRunner', 'add_requests'), (), ()),
    'request_update': ('runner', ('GPUModelRunner', 'update_requests'), (), ()),
    'request_state_add': ('request_state', ('RequestState', 'add_request'), ('req_id', 'prompt_len', 'num_computed_tokens', 'max_tokens'), ()),
    'request_state_remove': ('request_state', ('RequestState', 'remove_request'), ('req_id',), ()),
    'block_append': ('block_table', ('BlockTables', 'append_block_ids'), ('req_index', 'new_block_ids', 'overwrite'), ('block_sizes', 'redirect_writes_to_null_block')),
    'block_apply': ('block_table', ('BlockTables', 'apply_staged_writes'), (), ()),
    'block_gather': ('block_table', ('BlockTables', 'gather_block_tables'), ('idx_mapping', 'num_reqs_padded'), ()),
    'kv_zero': ('worker_utils', ('KVBlockZeroer', 'zero_block_ids'), ('block_ids', 'block_ids_by_group'), ()),
    'kv_raw_copy': ('worker_utils', ('copy_kv_cache_blocks_inplace',), ('kv_caches', 'num_blocks', 'kv_cache_block_copies'), ()),
    'kv_side_copy': ('worker_utils', ('copy_kv_cache_side_storage_blocks',), ('kv_cache_block_copies_by_group', 'num_blocks'), ()),
    'summary_zero': ('summary', ('Step5SparseSummaryCacheLayer', 'zero_kv_cache_side_storage'), ('block_ids',), ('main_layer_name',)),
    'summary_release': ('summary', ('Step5SparseSummaryCacheLayer', 'release_kv_cache_side_storage'), ('block_ids',), ('main_layer_name',)),
    'summary_runtime_reset': ('summary', ('Step5SparseSummaryCacheLayer', 'reset_kv_cache_side_storage_runtime_state'), (), ('main_layer_name',)),
    'summary_copy': ('summary', ('Step5SparseSummaryCacheLayer', 'copy_kv_cache_side_storage'), ('block_copies', 'num_blocks'), ('main_layer_name',)),
    'summary_reset': ('summary', ('Step5SparseSummaryCache', 'reset'), ('block_ids', 'clear_mean'), ('pages', 'block_size', 'request_journal_width')),
    'summary_cache_copy': ('summary', ('Step5SparseSummaryCache', 'copy'), ('copies',), ()),
    'workspace_views': ('workspace', ('WorkspaceManager', 'get_simultaneous'), (), ('_locked', '_num_ubatches', '_num_lanes')),
}
CENSUS_REQUEST_SITES = frozenset((
    'request_finish', 'request_remove', 'request_add', 'request_update',
    'request_state_add', 'request_state_remove', 'block_append', 'block_apply', 'block_gather'))
CENSUS_LAYER_LINE = 2139


@dataclass
class ObservationSites:
    codes: dict
    lines: dict
    sources: dict
    tensor_type: type
    graph_type: type
    intermediate_type: type | None = None
    torch: object | None = None
    padding_mask_used: bool | None = None
    census_names: dict = field(default_factory=dict)
    census_owners: dict = field(default_factory=dict)
    census_inventory: dict = field(default_factory=dict)
    census_bound: bool = False

    @classmethod
    def bind(cls, native, worker_factory, *, pins=SOURCE_PINS, lines=SOURCE_LINES):
        sources, codes = {}, {}
        functions = {
            'worker': (worker_factory, ('Worker', '_make_model_runner'), 'worker'),
            'execute': (native.runner.GPUModelRunner.execute_model,
                        ('GPUModelRunner', 'execute_model'), 'runner'),
            'add_requests': (native.runner.GPUModelRunner.add_requests,
                             ('GPUModelRunner', 'add_requests'), 'runner'),
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
                   getattr(native, 'intermediate_type', None), getattr(native, 'torch', None),
                   getattr(native, 'padding_mask_used', None))

    def bind_census(self, runner, limits):
        """Read resident modules and loaded code; never import or construct native owners."""
        if self.census_bound:
            return
        self.census_bound = True
        gaps, modules, layers = [], [], []
        model = getattr(runner, 'model', None)
        if model is None or not callable(getattr(model, 'named_modules', None)):
            self.census_inventory = dict(status='UNVERIFIED_NO_RESIDENT_MODULE_TRAVERSAL',
                                         gaps=['resident model named_modules unavailable'], owners=[], layers=[])
            return
        for name, (source, owner, arguments, attributes) in CENSUS_CALLS.items():
            module_name, pin = CENSUS_SOURCES[source]
            module = sys.modules.get(module_name)
            if module is None:
                gaps.append(dict(site=name, reason='NATIVE_MODULE_NOT_ALREADY_IMPORTED'))
                continue
            try:
                function = module
                for part in owner:
                    function = getattr(function, part)
                code = bind_code(function, owner, pin, self.sources)
                argument_names = code.co_varnames[:code.co_argcount+code.co_kwonlyargcount]
                if any(argument not in argument_names for argument in arguments):
                    raise ValueError('Pinned call argument is absent: '+name)
                self.codes['census_'+name] = code
                self.census_names[id(code)] = name
            except BaseException as failure:
                gaps.append(dict(site=name, reason='UNVERIFIED_SOURCE_BINDING',
                                 error_type=class_name(failure), error=str(failure)))
        if 'census_backbone' in self.codes:
            code = self.codes['census_backbone']
            if CENSUS_LAYER_LINE not in {line for _, line in dis.findlinestarts(code)}:
                gaps.append(dict(site='backbone', reason='PINNED_LAYER_DISPATCH_LINE_ABSENT'))
            else:
                self.lines['census_layer'] = CENSUS_LAYER_LINE
        if 'census_triton_experts' in self.codes:
            for name, nested in (('gemm1', '_base_w13_fn'), ('gemm2', '_base_w2_fn')):
                try:
                    self.codes['census_'+name] = nested_code(self.codes['census_triton_experts'], nested)
                except BaseException as failure:
                    gaps.append(dict(site=name, reason='UNVERIFIED_NESTED_DISPATCH_OWNER', error=str(failure)))
        for path, module in model.named_modules():
            modules.append((path, module))
            if 'layer' in self.matching_census_sites(module):
                layer = getattr(module, 'layer_idx')
                if len(layers) >= limits.tracked_entries:
                    raise ValueError('Resident layer bindings exceed the existing entry cap')
                if type(layer) is not int:
                    raise ValueError('Resident layer index must be an integer')
                layers.append(dict(path=path, layer_id=layer))
        objects = list(modules)+[('runner', runner)]
        # These are existing constructed owners, not allocation/configuration factories.
        for path, module in modules:
            if 'routed_modular' not in self.matching_census_sites(module):
                continue
            method = module.quant_method
            objects.append((path+'.quant_method', method))
            kernel = getattr(method, 'moe_kernel', None)
            if kernel is not None:
                objects.extend(((path+'.quant_method.moe_kernel.impl', kernel.impl),
                                (path+'.quant_method.moe_kernel.fused_experts', kernel.fused_experts),
                                (path+'.quant_method.moe_kernel.prepare_finalize', kernel.prepare_finalize)))
        rows = []
        for path, owner in objects:
            names = self.matching_census_sites(owner)
            if not names:
                continue
            matching = [layer for layer in layers if path == layer['path'] or path.startswith(layer['path']+'.')]
            layer = max(matching, key=lambda item: len(item['path']))['layer_id'] if matching else None
            if len(rows) >= limits.tracked_entries:
                raise ValueError('Resident native owner bindings exceed the existing entry cap')
            row = self.bind_census_owner(path, owner, names, layer, limits, gaps)
            rows.append(row)
        if len({row['layer_id'] for row in layers}) != len(layers):
            gaps.append(dict(site='layer', reason='DUPLICATE_RESIDENT_LAYER_IDS'))
        self.census_inventory = dict(status='LOADED_CPU_BINDINGS_NOT_EXECUTION_QUALIFICATION',
            owners=rows, layers=layers, gaps=gaps, source_status='PINNED_BOUND_SITES_ONLY',
            sites={name: dict(source=CENSUS_SOURCES[source][0], code_owner='.'.join(owner),
                   first_line=self.codes['census_'+name].co_firstlineno)
                   for name, (source, owner, _, _) in CENSUS_CALLS.items() if 'census_'+name in self.codes},
            device_content='UNMEASURED', physical_replay='UNMEASURED')


    def matching_census_sites(self, owner):
        names = []
        for name, (_, code_owner, _, _) in CENSUS_CALLS.items():
            if len(code_owner) != 2:
                continue
            method = inspect.getattr_static(type(owner), code_owner[-1], None)
            original = inspect.unwrap(method) if method is not None else None
            code = self.codes.get('census_'+name)
            if code is not None and getattr(original, '__code__', None) is code:
                names.append(name)
        return names

    def bind_census_owner(self, path, owner, names, layer, limits, gaps):
        metadata, attributes = Metadata(self, limits), {}
        for name in names:
            for attr in CENSUS_CALLS[name][3]:
                if attr in attributes or not hasattr(owner, attr):
                    continue
                value = getattr(owner, attr)
                if value is not None and (attr.endswith('dtype') or attr.endswith('backend')):
                    value = str(value)
                attributes[attr] = metadata.value(value)
        row = dict(owner_id=id(owner), module_path=path, cls=class_name(owner), layer_id=layer,
                   sites=list(names), attributes=attributes, metadata_complete=metadata.complete)
        try:
            self.census_owners[id(owner)] = (weakref.ref(owner), row)
        except TypeError:
            gaps.append(dict(owner=path, reason='NATIVE_OWNER_WEAK_REFERENCE_UNAVAILABLE'))
        return row

    def refresh_state_bindings(self, runner, limits):
        """Observe native replacement of profiling owners; preserve old records."""
        objects = [('runner.'+attr, getattr(runner, attr, None), None)
                   for attr in ('block_tables', 'req_states', 'kv_block_zeroer')]
        workspace = sys.modules.get(CENSUS_SOURCES['workspace'][0])
        objects.append(('workspace._manager', getattr(workspace, '_manager', None), None))
        for reference, row in list(self.census_owners.values()):
            layer = reference()
            if layer is not None and 'summary_zero' in row['sites']:
                objects.append((row['module_path']+'._cache', getattr(layer, '_cache', None), row['layer_id']))
        rows, gaps = [], []
        for path, owner, layer in objects:
            previous = [(key, binding) for key, binding in self.census_owners.items()
                        if binding[1]['module_path'] == path]
            if previous and previous[0][1][0]() is owner:
                continue
            for key, _ in previous:
                del self.census_owners[key]
            if owner is None:
                if previous:
                    rows.append(dict(module_path=path, previous_owner_id=previous[0][0],
                                     status='NATIVE_STATE_OWNER_NO_LONGER_PRESENT'))
                continue
            names = self.matching_census_sites(owner)
            if not names:
                gaps.append(dict(owner=path, reason='UNVERIFIED_NATIVE_STATE_OWNER_CODE'))
                continue
            if len(self.census_owners) >= limits.tracked_entries:
                raise ValueError('Known native owner bindings exceed the existing entry cap')
            row = self.bind_census_owner(path, owner, names, layer, limits, gaps)
            rows.append(dict(row, previous_owner_id=previous[0][0] if previous else None,
                             status='OBSERVED_NATIVE_STATE_OWNER_BINDING_NOT_LIFETIME_PROOF'))
        return rows, gaps


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
    native.torch = native.runner.torch
    native.padding_mask_used = native.runner.envs.VLLM_MOE_SKIP_PADDING
    native.tensor_type = native.torch.Tensor
    native.graph_type = native.runner.torch.cuda.CUDAGraph
    native.intermediate_type = native.runner.IntermediateTensors
    return native.runner.GPUModelRunner, ObservationSites.bind(native, worker_factory)


class Metadata:
    """Read only CPU tensor descriptors and ordinary containers, never tensor values."""
    def __init__(self, sites, limits, *, tensor_record=None):
        self.sites, self.limits, self.items = sites, limits, 0
        self.complete, self.tensor_record = True, tensor_record

    def value(self, value, depth=0):
        self.items += 1
        if self.items > self.limits.metadata_items or depth > self.limits.metadata_depth:
            raise ValueError('CPU metadata traversal exceeds observation limits')
        if value is None or type(value) in (str, int, float, bool):
            return value
        if isinstance(value, self.sites.tensor_type):
            storage = value.untyped_storage()
            row = dict(kind='tensor_metadata_only', type=class_name(value),
                        shape=list(value.shape), stride=list(value.stride()),
                        dtype=str(value.dtype), device=str(value.device),
                        data_ptr=value.data_ptr(), storage_ptr=storage.data_ptr(),
                        storage_bytes=storage.nbytes(), storage_offset=value.storage_offset(),
                        content='UNMEASURED')
            return self.tensor_record(row) if self.tensor_record is not None else row
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


@dataclass(frozen=True)
class DeviceObservationLimits:
    """A diagnostic allocation within the existing rank evidence allowance."""
    steps: int
    scan_rows: int
    witness_rows: int
    owned_bytes: int
    rpc_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError('Device observation limits must be positive integers')
        if self.rpc_bytes < len((json.dumps(None, separators=(',', ':'))+'\n').encode('utf-8')):
            raise ValueError('RPC byte cap must encode a JSON response')


class DeviceObservation:
    """Own snapshots before native buffer reuse; never wait in an execution hook.

    Preparation allocates every destination and reduction scratch outside capture.
    Copies and reductions run on the original execution stream. The existing
    post-generate worker RPC may read a destination only after its event is ready.
    Native AsyncOutput and its original consumption remain separate owners.
    """
    def __init__(self, runner, sites, limits, requests, sampling_fields, rank_remaining,
                 *, metadata_limits=ObservationLimits(), reserve=None, deadline=None):
        self.runner, self.sites, self.limits = runner, sites, limits
        self.torch, self.cpu_limits = sites.torch, metadata_limits
        self.padding_mask_used = sites.padding_mask_used
        if type(self.padding_mask_used) is not bool:
            raise ValueError('Actual native padding-mask policy is required')
        self.requests, self.sampling = copy.deepcopy(requests), copy.deepcopy(sampling_fields)
        self.entries, self.registered, self.errors = [], {}, []
        self.stopped, self.drained = False, 0
        if self.torch is None:
            raise ValueError('The installed runner Torch API is required for device witnesses')
        if self.torch.cuda.is_current_stream_capturing():
            raise ValueError('Device witness preparation must be outside CUDA capture')
        if (runner.use_pp or runner.pcp_manager is not None or runner.num_speculative_steps or
                not runner.is_first_pp_rank or not runner.is_last_pp_rank):
            raise ValueError('Device witnesses require the approved PP1, PCP-off, speculation-off scope')
        if not requests or len(requests) > metadata_limits.tracked_entries:
            raise ValueError('Device preparation requires bounded explicit request manifests')
        self.max_tokens, self.max_reqs = runner.max_num_tokens, runner.max_num_reqs
        self.width = runner.model_config.hf_text_config.hidden_size
        self.groups = [list(group.layer_names) for group in runner.kv_cache_config.kv_cache_groups]
        self.layers = [name for group in self.groups for name in group]
        if (len(self.layers) != len(set(self.layers)) or not self.groups or any(not group for group in self.groups) or
                limits.witness_rows != 1 or limits.scan_rows > self.max_tokens or
                limits.steps > self.cpu_limits.tracked_entries):
            raise ValueError('Device witness layout exceeds its supported observation scope')
        if any(type(value) is not int or value <= 0 for value in (self.max_tokens, self.max_reqs, self.width)):
            raise ValueError('Actual native request, token and hidden dimensions are required')
        self.stream = runner.main_stream
        self.hidden_dtype = (self.torch.float32 if runner.model_config.hf_text_config.fp32_residual_connection
                             else runner.model_config.dtype)
        self.embed_dtype = runner.model_config.dtype
        self.chunks = (self.max_tokens+limits.scan_rows-1)//limits.scan_rows
        tables = runner.block_tables
        if (tables.cp_size != 1 or tables.cp_rank != 0 or
                len(tables.input_block_tables) != len(self.groups) or
                len(tables.block_sizes) != len(self.groups)):
            raise ValueError('Actual native block tables do not match the approved non-CP KV groups')
        self.block_sizes, self.table_layouts = list(tables.block_sizes), []
        for table, block_size in zip(tables.input_block_tables, self.block_sizes):
            if (not isinstance(table, sites.tensor_type) or table.device != runner.device or
                    table.dtype != self.torch.int32 or len(table.shape) != 2 or
                    table.shape[0] != self.max_reqs or table.shape[1] <= 0 or
                    type(block_size) is not int or block_size <= 0):
                raise ValueError('Actual resident block-table dimensions/dtype are required')
            self.table_layouts.append(Metadata(sites, self.cpu_limits).value(table))
        if (tuple(tables.num_blocks.gpu.shape) != (len(self.groups), self.max_reqs) or
                tables.num_blocks.gpu.dtype != self.torch.int32 or
                tables.num_blocks.gpu.device != runner.device):
            raise ValueError('Actual native persistent-index block counts have an unsupported layout')
        self.layout = dict(
            input_ids=((self.max_tokens,), self.torch.int32),
            positions=((self.max_tokens,), self.torch.int64),
            is_padding=((self.max_tokens,), self.torch.bool),
            query_start_loc=((self.max_reqs+1,), self.torch.int32),
            seq_lens=((self.max_reqs,), self.torch.int32),
            idx_mapping=((self.max_reqs,), self.torch.int64),
            request_state=((self.max_reqs,), self.torch.int64),
            final_input_ids=((self.max_tokens,), self.torch.int32),
            final_positions=((self.max_tokens,), self.torch.int64),
            inputs_embeds=((self.max_reqs, self.width), self.embed_dtype),
            hidden_states=((self.max_reqs, self.width), self.hidden_dtype),
            finite_counts=((self.chunks,), self.torch.int64),
            block_counts=((len(self.groups), self.max_reqs), self.torch.int32))
        for index in range(len(self.groups)):
            self.layout['slot_mapping_'+str(index)] = ((self.max_tokens,), self.torch.int64)
            self.layout['block_table_'+str(index)] = (tuple(self.table_layouts[index]['shape']), self.torch.int32)
        from role_state_observation import StateObservation, dtype_bytes
        def size(dtype):
            return dtype_bytes(self.torch, dtype)
        self.state = StateObservation(runner, sites, limits, metadata_limits,
            lambda value: Metadata(sites, metadata_limits).value(value), limits.scan_rows*self.width)
        self.state.check_resident_readiness(deadline)
        self.layout.update(self.state.step_layout)
        self.host_bytes = limits.steps*sum(
            self.elements(shape)*size(dtype) for name, (shape, dtype) in self.layout.items()
            if name not in self.state.step_layout)
        self.host_bytes += limits.steps*self.state.byte_count(self.state.step_layout)+self.state.byte_count(self.state.once_layout)
        self.device_bytes = limits.scan_rows*self.width*(size(self.hidden_dtype)+size(self.torch.bool)) + limits.steps*self.chunks*size(self.torch.int64)
        self.device_bytes += self.state.byte_count({name: layout for name, layout in self.state.device_layout.items()
            if name != 'state_absolute' or self.hidden_dtype != self.torch.float32})
        self.owned_bytes = self.host_bytes+self.device_bytes
        if self.owned_bytes > min(limits.owned_bytes, rank_remaining):
            raise ValueError('Owned diagnostic destinations exceed the remaining rank allowance')
        if reserve is not None:
            reserve(self.owned_bytes)
        self.destinations = [{name: self.torch.empty(shape, dtype=dtype, device='cpu', pin_memory=True)
                              for name, (shape, dtype) in self.layout.items()} for _ in range(limits.steps)]
        self.absolute = self.torch.empty((limits.scan_rows, self.width), dtype=self.hidden_dtype,
                                         device=runner.device)
        self.finite = self.torch.empty((limits.scan_rows, self.width), dtype=self.torch.bool,
                                       device=runner.device)
        self.counts = self.torch.empty((limits.steps, self.chunks), dtype=self.torch.int64,
                                       device=runner.device)
        self.state.allocate(self.absolute, self.finite)
        with self.torch.inference_mode():
            self.check_stream()
            self.state.submit_resident(deadline)
        self.events = [self.torch.cuda.Event(enable_timing=False) for _ in range(limits.steps)]
        for event in self.events:
            event.record(self.stream)
        self.check_stream()

    @staticmethod
    def elements(shape):
        result = 1
        for dim in shape:
            result *= dim
        return result

    def error(self, kind, failure):
        self.stopped = True
        if not self.errors:
            self.errors.append(dict(kind=kind, error_type=class_name(failure), error=str(failure)))

    def check_stream(self):
        actual = self.torch.cuda.current_stream(self.runner.device)
        if (actual.cuda_stream != self.stream.cuda_stream or actual.device != self.stream.device or
                self.torch.cuda.is_current_stream_capturing()):
            raise ValueError('Device witnesses require the original execution stream outside capture')

    def registrations(self, scheduler):
        if self.stopped:
            return
        try:
            for request in scheduler.scheduled_new_reqs:
                internal = request.req_id
                if internal in self.registered:
                    raise ValueError('Duplicate original worker request registration')
                prompt = list(request.prefill_token_ids)
                expected = [row for row in self.requests if row['prompt_token_ids'] == prompt]
                if len(expected) != 1 or request.prompt_len != len(prompt) or request.num_computed_tokens != 0:
                    raise ValueError('Worker registration does not identify an approved fresh text request')
                index = self.runner.req_states.req_id_to_index[internal]
                if (self.runner.req_states.index_to_req_id[index] != internal or
                        int(self.runner.req_states.prompt_len.np[index]) != len(prompt) or
                        int(self.runner.req_states.prefill_len.np[index]) != len(prompt)):
                    raise ValueError('Original worker registration and persistent request state disagree')
                resolved = {name: copy.deepcopy(getattr(request.sampling_params, name)) for name in self.sampling}
                if resolved != self.sampling:
                    raise ValueError('Worker resolved sampling differs from original frontend submission')
                self.registered[internal] = dict(internal_request_id=internal, name=expected[0]['name'],
                    prompt_token_ids=prompt, resolved_sampling_fields=resolved, persistent_index=index,
                    native_request_object_id=id(request), registration_owner_id=id(self.runner.req_states))
        except BaseException as failure:
            self.error('worker_request_registration', failure)

    def copy_tensor(self, entry, name, tensor, rows):
        if not isinstance(tensor, self.sites.tensor_type) or tensor.device != self.runner.device:
            raise ValueError('Device witness is not a tensor on the actual worker device: '+name)
        destination = self.destinations[entry['index']][name]
        shape = (rows, *destination.shape[1:])
        if tuple(tensor.shape) != shape or tensor.dtype != destination.dtype:
            raise ValueError('Actual device witness shape/dtype differs from its prepared destination: '+name)
        metadata = Metadata(self.sites, self.cpu_limits).value(tensor)
        destination[:rows].copy_(tensor, non_blocking=True)
        entry['tensors'][name] = dict(rows=rows, source=metadata)

    def inputs(self, values, batch, invocation, input_id, *, descriptor=None):
        if self.stopped:
            return None
        try:
            self.check_stream()
            if len(self.entries) >= self.limits.steps:
                raise ValueError('Device witness step capacity exhausted; collection cannot wait or reuse a slot')
            frontiers = request_frontiers(batch)
            for row in frontiers:
                internal, index = row['req_id'], row['idx_mapping_np']
                registered = self.registered.get(internal)
                if (registered is None or registered['persistent_index'] != index or
                        self.runner.req_states.req_id_to_index.get(internal) != index or
                        self.runner.req_states.index_to_req_id.get(index) != internal):
                    raise ValueError('Final native batch has no complete original request registration')
            n, r = batch['num_tokens_after_padding'], batch['num_reqs_after_padding']
            if (not 0 < batch['num_tokens'] <= n <= self.max_tokens or
                    not 0 < batch['num_reqs'] <= r <= self.max_reqs):
                raise ValueError('Actual valid/padded batch dimensions exceed prepared destinations')
            actual = values['input_batch']
            entry = dict(index=len(self.entries), invocation=invocation, input_observation=input_id,
                native_input_batch_id=id(actual), status='SUBMITTED_NOT_COMPLETED',
                descriptor=copy.deepcopy(descriptor), padding_mask_used=self.padding_mask_used,
                slot_groups=copy.deepcopy(self.groups),
                prepared_cpu_batch=copy.deepcopy(batch), tensors={},
                stream=dict(cuda_stream=self.stream.cuda_stream, device=str(self.stream.device)),
                device_operations='diagnostic_copies_and_finite_reductions_not_clean_timing')
            self.entries.append(entry)
            for name, rows in (('input_ids', n), ('positions', n), ('is_padding', n),
                               ('query_start_loc', r+1), ('seq_lens', r), ('idx_mapping', batch['num_reqs'])):
                self.copy_tensor(entry, name, getattr(actual, name), rows)
            self.copy_tensor(entry, 'request_state', self.runner.model_state.request_state[:r], r)
            model_inputs = values['model_inputs']
            if model_inputs['intermediate_tensors'] is not None:
                raise ValueError('The approved PP1 witness cannot contain intermediate tensors')
            self.copy_tensor(entry, 'final_positions', model_inputs['positions'], n)
            entry['model_input_ids_none'] = model_inputs['input_ids'] is None
            entry['model_inputs_embeds_none'] = model_inputs['inputs_embeds'] is None
            if model_inputs['input_ids'] is not None:
                self.copy_tensor(entry, 'final_input_ids', model_inputs['input_ids'], n)
            if model_inputs['inputs_embeds'] is not None:
                embed = model_inputs['inputs_embeds']
                if tuple(embed.shape) != (n, self.width):
                    raise ValueError('Final model embeddings do not cover the actual padded batch')
                self.copy_rows(entry, 'inputs_embeds', embed, frontiers)
            if entry['model_input_ids_none'] and entry['model_inputs_embeds_none']:
                raise ValueError('Finalized text model has neither tokens nor embeddings')
            slots = values['slot_mappings_by_layer']
            if type(slots) is not dict or set(slots) != set(self.layers):
                raise ValueError('Actual per-layer slot bindings differ from native KV groups')
            for index, layers in enumerate(self.groups):
                first = slots[layers[0]]
                descriptor = Metadata(self.sites, self.cpu_limits).value(first)
                if any(Metadata(self.sites, self.cpu_limits).value(slots[name]) != descriptor for name in layers):
                    raise ValueError('Native group layer slot bindings do not share the same device view')
                self.copy_tensor(entry, 'slot_mapping_'+str(index), first, n)
            tables = values['block_tables']
            if type(tables) is not tuple or len(tables) != len(self.groups):
                raise ValueError('Finalized native block tables do not cover every KV group')
            for index, table in enumerate(tables):
                layout = Metadata(self.sites, self.cpu_limits).value(table)
                expected = dict(self.table_layouts[index], shape=[r, self.table_layouts[index]['shape'][1]])
                if layout != expected:
                    raise ValueError('Finalized block-table view differs from the prepared resident owner')
                self.copy_tensor(entry, 'block_table_'+str(index), table, r)
            self.copy_tensor(entry, 'block_counts', self.runner.block_tables.num_blocks.gpu, len(self.groups))
            entry['block_sizes'] = list(self.block_sizes)
            entry['block_table_semantics'] = 'BATCH_ROWS; ACTIVE_COUNT_BY_PERSISTENT_INDEX; SUFFIX_NOT_OWNERSHIP'
            entry['request_frontiers'] = frontiers
            self.state.select(entry)
            return entry['index']
        except BaseException as failure:
            self.error('finalized_device_inputs', failure)
            return None

    def copy_rows(self, entry, name, tensor, frontiers):
        destination = self.destinations[entry['index']][name]
        if tensor.device != self.runner.device or tensor.dtype != destination.dtype:
            raise ValueError('Actual witness row device/dtype differs from its prepared destination: '+name)
        source = Metadata(self.sites, self.cpu_limits).value(tensor)
        positions = [row['query_end']-1 for row in frontiers]
        if any(position < 0 or position >= entry['prepared_cpu_batch']['num_tokens'] for position in positions):
            raise ValueError('Witness rows must belong to valid native request queries')
        for index, position in enumerate(positions):
            destination[index:index+1].copy_(tensor[position:position+1], non_blocking=True)
        entry['tensors'][name] = dict(rows=len(positions), source=source, valid_query_rows=positions)

    def output(self, index, native_error, observer_errors, *, events=()):
        entry = self.entries[index]
        if native_error is not None or observer_errors or self.stopped:
            entry['status'] = 'UNVERIFIED_NATIVE_OR_OBSERVER_ERROR'
            self.stopped = True
            return
        with self.torch.inference_mode():
            self.check_stream()
            state = self.runner.execute_model_state
            if state is None or id(state.input_batch) != entry['native_input_batch_id']:
                raise ValueError('Original post-execution state no longer owns the prepared input batch')
            hidden = state.hidden_states
            n = entry['prepared_cpu_batch']['num_tokens_after_padding']
            if not isinstance(hidden, self.sites.tensor_type) or tuple(hidden.shape) != (n, self.width):
                raise ValueError('Actual hidden output does not cover the prepared native batch')
            self.copy_rows(entry, 'hidden_states', hidden, entry['request_frontiers'])
            valid = entry['prepared_cpu_batch']['num_tokens']
            chunks = (valid+self.limits.scan_rows-1)//self.limits.scan_rows
            from role_state_observation import finite_count
            for chunk in range(chunks):
                start = chunk*self.limits.scan_rows
                rows = min(self.limits.scan_rows, valid-start)
                finite_count(self.torch, hidden[start:start+rows], self.absolute, self.finite,
                             self.counts[index:index+1, chunk:chunk+1].view(()))
            self.destinations[index]['finite_counts'][:chunks].copy_(self.counts[index, :chunks], non_blocking=True)
            entry['tensors']['finite_counts'] = dict(rows=chunks)
            entry['finite_elements_expected'] = valid*self.width
            self.state.scan(entry, self.destinations[index])
            self.state.attach_resident(entry)
            self.events[index].record(self.stream)
        entry['status'] = 'COMPLETION_EVENT_RECORDED_NOT_QUERIED'
        entry['native_async_output_consumed_by_observer'] = False
        entry['native_replay_events'] = copy.deepcopy([row for row in events if row['kind'] in
            ('replay_attempt', 'replay_outcome', 'graph_lifecycle_clear')])

    def drain(self, record):
        """Query readiness once after generate; no event/device synchronization or polling."""
        if self.stopped:
            raise ValueError('Device witness collection is unverified')
        pending = self.entries[self.drained:]
        if not pending:
            raise ValueError('Post-generate device drain has no new execution witnesses')
        rows, compact = [], []
        for entry in pending:
            index = entry['index']
            if entry['status'] != 'COMPLETION_EVENT_RECORDED_NOT_QUERIED' or not self.events[index].query():
                raise ValueError('Diagnostic completion event is not ready at the post-generate RPC')
            row = copy.deepcopy(entry)
            values = {}
            for name, descriptor in row['tensors'].items():
                tensor = self.destinations[index][name][:descriptor['rows']]
                if name in ('hidden_states', 'inputs_embeds'):
                    raw = tensor.view(self.torch.uint8).numpy().tobytes()
                    values[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                                        raw_hex=raw.hex(), sha256=hashlib.sha256(raw).hexdigest())
                else:
                    values[name] = tensor.numpy().tolist()
            row['values'] = values
            row['state_content'] = self.state.drain(row, self.destinations[index], values)
            row['state_content_status'] = row['state_content']['status']
            row['finite_elements_observed'] = sum(values['finite_counts'])
            row['status'] = ('OBSERVED_DIAGNOSTIC_DEVICE_WITNESS_NOT_G2' if
                row['finite_elements_observed'] == row['finite_elements_expected'] else
                'UNVERIFIED_NONFINITE_VALID_HIDDEN_OUTPUT')
            row['completion'] = 'OWNED_COPY_EVENT_QUERY_READY_AFTER_NATIVE_GENERATE'
            row['padding_finite'] = 'NOT_SCANNED_NOT_VALID_TOKEN_WORK'
            rows.append(row)
            small = copy.deepcopy(row)
            for name in ('hidden_states', 'inputs_embeds'):
                if name in small['values']:
                    del small['values'][name]['raw_hex']
            small['state_content'] = self.state.compact(row['state_content'])
            compact.append(small)
        previous = self.entries[:self.drained]
        summary = self.summary(steps=previous+compact, drained=len(self.entries))
        if len((json.dumps(summary, allow_nan=False, separators=(',', ':'))+'\n').encode('utf-8')) > self.limits.rpc_bytes:
            raise ValueError('Device witness summary exceeds the configured RPC byte cap')
        record(dict(scope='diagnostic_device_witness_not_G2', rows=rows,
                    registered_requests=list(self.registered.values()),
                    native_async_output_consumed_by_observer=False))
        for entry, small in zip(pending, compact):
            entry.clear()
            entry.update(small)
        self.drained = len(self.entries)
        return summary

    def summary(self, *, steps=None, drained=None):
        rows = self.entries if steps is None else steps
        drained = self.drained if drained is None else drained
        return dict(scope='diagnostic_device_witness_not_G2',
            status='UNVERIFIED' if self.stopped or self.errors or any(
                row['status'].startswith('UNVERIFIED') for row in rows) else
                'OBSERVED_DIAGNOSTIC_DEVICE_WITNESSES_NOT_G2' if drained else 'PREPARED_NOT_EXECUTED',
            owned_bytes=self.owned_bytes, host_bytes=self.host_bytes, device_bytes=self.device_bytes,
            state_inventory=self.state.inventory(),
            limits=vars(self.limits), collected_steps=len(self.entries), drained_steps=drained,
            collection_stopped=self.stopped, errors=copy.deepcopy(self.errors),
            registered_requests=copy.deepcopy(list(self.registered.values())),
            steps=copy.deepcopy(rows[:drained]),
            source_status='LOADED_CODE_MATCHES_PINNED_SOURCE',
            native_async_output_consumed_by_observer=False, clean_performance='UNVERIFIED', G2='UNVERIFIED')


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
    last_summary: dict | None = field(default_factory=dict)
    identities: dict = field(default_factory=dict)
    identity_serial: int = 0
    stopped: bool = False
    device: object | None = None
    census_inventory_record: dict | None = None

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
        return dict(status='UNVERIFIED' if self.stopped or self.failures else
                    'CPU_METADATA_ONLY_NATIVE_COMPATIBILITY_UNVERIFIED',
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
        self.device_step = None
        self.pending, self.local_hooks, self.errors = {}, {}, []
        self.active, self.stopped, self.installed = False, False, False
        self.opcodes, self.native_calls, self.excluded_capture_calls = {}, {}, {}
        self.capture_binding_event, self.capture_window = None, False
        self.capture_node_state = None
        self.annotations = {}
        self.call_tensors, self.tensor_keys = [], {}
        self.profile_hook, self.trace_hook = self.profile, self.trace
        self.previous_profile, self.previous_trace = sys.getprofile(), sys.gettrace()

    def error(self, kind, failure):
        self.state.failures += 1
        self.stopped = self.state.stopped = True
        if not self.errors:
            self.errors.append(dict(kind=kind, error_type=class_name(failure), error=str(failure)))

    def emit(self, kind, *, frontiers=True, **row):
        if self.stopped:
            return None
        event = dict(event=len(self.events)+1, kind=kind, lifecycle=self.kind,
                     input_observation=self.input_id, request_frontiers=self.frontiers if frontiers else None,
                     cpu_frontier_semantics='optimistic_upper_bounds_not_device_progress',
                     **row)
        payload = json.dumps(event, allow_nan=False, separators=(',', ':'))
        if (len(self.events) >= self.state.limits.scope_events or
                self.used+len(payload.encode()) > self.state.limits.scope_bytes):
            raise ValueError('Model observation scope exceeds the CPU evidence limit')
        self.events.append(event)
        self.used += len(payload.encode())
        return event

    def start_annotation(self, key, row, kind):
        """Use only the current original native profiler and existing event identity."""
        observed = getattr(self.runner, '_step5_profiler_observation', None)
        if observed is None or row is None:
            return
        if key in self.annotations:
            raise ValueError('A native annotation is already active at this boundary')
        from role_profiler_observation import annotation_name
        sequence = self.state.captures if self.kind == 'capture' else self.state.invocations
        name = annotation_name(self.kind, sequence, row['event'], kind)
        context = observed.annotate(name)
        if context is None:
            return
        context.__enter__()
        self.annotations[key] = context
        self.update(row, trace_annotation=name)

    def finish_annotation(self, key):
        context = self.annotations.pop(key, None)
        if context is not None:
            context.__exit__(None, None, None)

    def capture_nodes(self, boundary, owner=None):
        observed = getattr(self.runner, '_step5_profiler_observation', None)
        if observed is None or self.kind != 'capture' or not self.capture_window:
            return
        self.capture_node_state, values = observed.capture_nodes(self.capture_node_state)
        if values['nodes'] or boundary in ('capture_begin', 'capture_end'):
            self.emit('capture_node_interval', frontiers=False, boundary=boundary,
                capture_binding_event=self.capture_binding_event,
                call_event=owner['event'] if owner is not None else None, **values)

    def metadata(self):
        return Metadata(self.sites, self.state.limits)

    def tensor_reference(self, descriptor):
        """Reuse equal CPU descriptors within this record; content/lifetimes stay unknown."""
        key = tuple((name, tuple(value) if type(value) is list else value)
                    for name, value in sorted(descriptor.items()))
        index = self.tensor_keys.get(key)
        if index is None:
            index = len(self.call_tensors)+1
            row = dict(tensor_id=index, **descriptor)
            size = len(json.dumps(row, allow_nan=False, separators=(',', ':')).encode())
            if self.used+size > self.state.limits.scope_bytes:
                raise ValueError('Native call tensor metadata exceeds the CPU evidence limit')
            self.call_tensors.append(row)
            self.tensor_keys[key] = index
            self.used += size
        return dict(kind='tensor_metadata_reference', tensor_id=index)

    def update(self, row, **values):
        """Charge call-return growth before the complete final record check."""
        previous = len(json.dumps(row, allow_nan=False, separators=(',', ':')).encode())
        candidate = dict(row, **values)
        current = len(json.dumps(candidate, allow_nan=False, separators=(',', ':')).encode())
        if self.used+current-previous > self.state.limits.scope_bytes:
            raise ValueError('Model call return exceeds the CPU evidence limit')
        row.update(values)
        self.used += current-previous

    def returned(self, frame):
        code = frame.f_code
        if code not in self.opcodes:
            self.opcodes[code] = {instruction.offset: instruction.opname
                                  for instruction in dis.get_instructions(code)}
        return self.opcodes[code].get(frame.f_lasti) in ('RETURN_VALUE', 'RETURN_CONST')

    def record_fits(self, record):
        payload = (json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n').encode('utf-8')
        return len(payload) <= self.state.limits.scope_bytes

    def bounded_record(self, record):
        """Check the complete encoded record; an omitted record earns no evidence credit."""
        if self.record_fits(record):
            return record
        self.error('scope_record_bytes', ValueError('Complete model observation record exceeds the CPU evidence limit'))
        rejected = dict(status='UNVERIFIED', lifecycle=self.kind, events=[],
                        native_outcome=record['native_outcome'],
                        native_error_type=record['native_error_type'],
                        observer_errors=[dict(kind='scope_record_bytes')])
        return rejected if self.record_fits(rejected) else None

    def parent_call(self, frame):
        parent = frame.f_back
        while parent is not None:
            row = self.native_calls.get(id(parent))
            if row is not None:
                return row
            parent = parent.f_back
        return None

    def lifecycle_metadata(self, name, frame, *, returned=False):
        values, row = frame.f_locals, {}
        if name in ('request_finish', 'request_add', 'request_update'):
            scheduler = values['scheduler_output']
            row['scheduler'] = dict(total_num_scheduled_tokens=scheduler.total_num_scheduled_tokens,
                finished_req_ids=sorted(scheduler.finished_req_ids),
                preempted_req_ids=sorted(scheduler.preempted_req_ids))
            if name == 'request_add' and not returned:
                row['scheduler']['new_requests'] = [dict(req_id=req.req_id, prompt_len=req.prompt_len,
                    num_computed_tokens=req.num_computed_tokens, block_ids=req.block_ids)
                    for req in scheduler.scheduled_new_reqs]
            if name == 'request_update' and not returned:
                reqs = scheduler.scheduled_cached_reqs
                if not len(reqs.req_ids) == len(reqs.num_computed_tokens) == len(reqs.new_block_ids):
                    raise ValueError('Cached request/block assignments have incomplete CPU associations')
                row['scheduler']['cached_requests'] = [dict(req_id=req, num_computed_tokens=computed,
                    new_block_ids=blocks) for req, computed, blocks in
                    zip(reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids)]
                for attr in ('new_block_ids_to_zero', 'new_block_ids_to_zero_by_group',
                             'kv_cache_block_copies', 'kv_cache_block_copies_by_group'):
                    row['scheduler'][attr] = getattr(scheduler, attr)
        if name in CENSUS_REQUEST_SITES:
            req_states = self.runner.req_states
            row['request_indices'] = [dict(req_id=req, persistent_index=index)
                for req, index in sorted(req_states.req_id_to_index.items())]
            row['block_counts_cpu'] = self.runner.block_tables.num_blocks.np.copy().tolist()
            row['ownership_semantics'] = 'OBSERVED_CPU_MAPPING_ONLY; NO_ALLOCATION_EPOCH_OR_WRITE_PROOF'
        return self.metadata().value(row)

    def native_call(self, frame, event, arg):
        name = self.sites.census_names.get(id(frame.f_code))
        if name is None or frame.f_code is not self.sites.codes['census_'+name]:
            return
        if event == 'call':
            if self.kind == 'capture' and not self.capture_window:
                self.excluded_capture_calls[name] = self.excluded_capture_calls.get(name, 0)+1
                return
            parent = self.parent_call(frame)
            self.capture_nodes('call_enter', parent)
            owner = frame.f_locals.get('self')
            binding = self.sites.census_owners.get(id(owner))
            owner_row = binding[1] if binding is not None and binding[0]() is owner else None
            metadata = Metadata(self.sites, self.state.limits, tensor_record=self.tensor_reference)
            arguments = {}
            for key in CENSUS_CALLS[name][2]:
                value = frame.f_locals[key]
                if value is not None and key in ('compute_type', 'out_dtype'):
                    value = str(value)
                arguments[key] = metadata.value(value)
            if name in CENSUS_REQUEST_SITES:
                arguments['lifecycle'] = self.lifecycle_metadata(name, frame)
            caller = frame.f_back.f_code if frame.f_back is not None else None
            gemm = next((key.upper() for key in ('gemm1', 'gemm2')
                if caller is self.sites.codes.get('census_'+key)), 'UNVERIFIED_NATIVE_CALLER') if name == 'triton_gemm' else None
            inherited = parent is not None and owner is None
            row = self.emit('native_call', frontiers=False, site=name, gemm=gemm,
                capture_binding_event=self.capture_binding_event if self.kind == 'capture' else None,
                owner_binding='RESIDENT_BOUND_OWNER' if owner_row else
                    'NATIVE_FUNCTION_ANCESTOR' if inherited else 'UNVERIFIED_OWNER_NOT_RESIDENT',
                owner_id=id(owner) if owner_row else parent.get('owner_id') if inherited else None,
                layer_id=owner_row['layer_id'] if owner_row else parent.get('layer_id') if inherited else None,
                parent_call=parent['event'] if parent else None, arguments=arguments,
                metadata_complete=metadata.complete, source_line=frame.f_lineno,
                outcome='CALL_ENTERED_NOT_COMPLETED', gpu_completion='UNMEASURED')
            if row is not None:
                self.native_calls[id(frame)] = row
                self.start_annotation(('call', id(frame)), row, 'call')
        elif event == 'return':
            self.finish_annotation(('call', id(frame)))
            row = self.native_calls.pop(id(frame), None)
            if row is None:
                return
            self.capture_nodes('call_return', row)
            returned = self.returned(frame)
            metadata = Metadata(self.sites, self.state.limits, tensor_record=self.tensor_reference)
            self.update(row, outcome='RETURNED_CPU_CALL_NOT_GPU_COMPLETION' if returned else
                        'UNVERIFIED_NATIVE_UNWIND', return_line=frame.f_lineno,
                        result=metadata.value(arg) if returned else None,
                        return_metadata_complete=metadata.complete)
            if returned and name in CENSUS_REQUEST_SITES:
                self.update(row, lifecycle_after=self.lifecycle_metadata(name, frame, returned=True))
                if self.state.device is not None and name in ('request_state_remove', 'block_append'):
                    self.state.device.state.assignment(row)

    def call_summary(self):
        calls = [row for row in self.events if row['kind'] == 'native_call']
        returned = [row for row in calls if row['site'] == 'layer' and
                    row['outcome'] == 'RETURNED_CPU_CALL_NOT_GPU_COMPLETION' and
                    row['owner_binding'] == 'RESIDENT_BOUND_OWNER']
        layers = sorted({row['layer_id'] for row in returned if row['layer_id'] is not None})
        expected = sorted({row['layer_id'] for row in self.sites.census_inventory.get('layers', [])})
        modes = [row['descriptor']['cg_mode'] for row in self.events if row['kind'] == 'finalized_model_inputs']
        return dict(status='CPU_NATIVE_CALLS_ONLY_NOT_PHYSICAL_OR_CONTENT_QUALIFICATION',
            inventory_record=copy.deepcopy(self.state.census_inventory_record),
            source_gaps=copy.deepcopy(self.sites.census_inventory.get('gaps', [])),
            calls=len(calls), pending_calls=len(self.native_calls), returned_layer_ids=layers,
            incomplete_calls=sum(row['outcome'] != 'RETURNED_CPU_CALL_NOT_GPU_COMPLETION' for row in calls),
            capture_policy='OBSERVE_ACTUAL_FULL_CAPTURE_ONLY; EXCLUDE_WARMUP_AND_PIECEWISE',
            excluded_capture_calls=dict(self.excluded_capture_calls),
            declared_layer_ids=expected, missing_python_layer_returns=sorted(set(expected)-set(layers)),
            layer_coverage='FULL_REPLAY_BYPASSES_PYTHON_LAYER_FRAMES' if modes and modes[-1] == 'FULL' else
                'OBSERVED_CPU_LAYER_RETURNS' if expected and layers == expected else 'UNVERIFIED_INCOMPLETE_PYTHON_LAYER_RETURNS',
            physical_kernel_census='UNMEASURED', transient_scale_content='UNMEASURED', G2='UNVERIFIED')

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
        from role_capacity_observation import known_state_inventory
        self.emit('known_state_storage', frontiers=False,
                  **known_state_inventory(self.runner, self.sites.torch, self.state.limits.tracked_entries))
        self.input_id = row['event']
        row['input_observation'] = self.input_id
        if self.state.device is not None and not values['dummy_run']:
            self.device_step = self.state.device.inputs(values, batch, self.state.invocations, self.input_id,
                                                        descriptor=row['descriptor'])

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
        observed = getattr(self.runner, '_step5_profiler_observation', None)
        if row is not None and observed is not None:
            self.update(row, **observed.graph_identity(graph))
        if row is not None:
            self.pending[key] = dict(row=row, exception=None)
            self.start_annotation(key, row, 'replay')

    def finish_replay(self, key, outcome, error=None):
        self.finish_annotation(key)
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
        event = self.emit(kind, **row)
        if self.kind == 'capture' and kind == 'capture_forward':
            self.capture_window = row['native_capture_region']
            self.capture_binding_event = event['event'] if event is not None and self.capture_window else None
            if self.capture_window:
                self.start_annotation(('capture', id(frame)), event, 'capture')
                self.capture_nodes('capture_begin')

    def profile(self, frame, event, arg):
        if self.previous_profile is not None:
            self.previous_profile(frame, event, arg)
        if not self.active:
            return
        if self.stopped:
            if event == 'return':
                if self.kind == 'capture' and frame.f_code is self.sites.codes['capture_forward']:
                    self.capture_node_state = None
                    self.capture_window = False
                keys = (('call', id(frame)), ('capture', id(frame)), ('python', id(frame)))
            elif event in ('c_return', 'c_exception'):
                keys = (('c', id(frame), id(getattr(arg, '__self__', None))),)
            else:
                keys = ()
            for key in keys:
                try:
                    self.finish_annotation(key)
                except BaseException as failure:
                    self.error('trace_annotation_unwind', failure)
            return
        try:
            codes = self.sites.codes
            self.native_call(frame, event, arg)
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
                if self.kind == 'capture' and frame.f_code is codes['capture_forward']:
                    try:
                        self.capture_nodes('capture_end')
                    finally:
                        self.capture_node_state = None
                    self.finish_annotation(('capture', id(frame)))
                    self.capture_window = False
                key = ('python', id(frame))
                registration = (self.state.device is not None and frame.f_code is codes['add_requests'] and
                                frame.f_locals.get('self') is self.runner)
                if registration or key in self.pending:
                    returned = self.returned(frame)
                if registration and returned:
                    self.state.device.registrations(frame.f_locals['scheduler_output'])
                if self.kind == 'capture' and frame.f_code is codes['create_capture']:
                    if frame.f_locals['self'] is self.runner.cudagraph_manager and arg is not None:
                        self.capture_inputs(frame, 'capture_inputs_prepared')
                if key in self.pending:
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
            selected = ('manager_capture', 'pw', 'break_capture', 'segments', 'census_backbone')
            relevant = (any(frame.f_code is self.sites.codes.get(name) for name in selected)
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
        if (frame.f_code is codes.get('census_backbone') and
                frame.f_lineno == lines.get('census_layer') and
                (self.kind != 'capture' or self.capture_window)):
            owner = values['layer']
            binding = self.sites.census_owners.get(id(owner))
            row = binding[1] if binding is not None and binding[0]() is owner else None
            self.emit('native_layer_dispatch', frontiers=False, layer_index=values['i'],
                      owner_id=id(owner), layer_id=row['layer_id'] if row else None,
                      source_line=frame.f_lineno, outcome='ORIGINAL_LAYER_CALL_ATTEMPT_ONLY',
                      module_path=row['module_path'] if row else None, gpu_completion='UNMEASURED')
        elif (frame.f_code is codes['manager_capture'] and frame.f_lineno == lines['full_capture_store']
                and values['self'] is self.runner.cudagraph_manager):
            self.capture_stored(values['self'], values['graph'], values['graph'], values['desc'], 'FULL', frame)
        elif frame.f_code is codes['pw']:
            if frame.f_lineno == lines['pw_capture']:
                self.wrapper_capture(frame, 'PIECEWISE', values['cudagraph'])
            elif frame.f_lineno == lines['pw_capture_store']:
                self.finish_annotation(('capture', id(frame)))
                self.capture_stored(values['self'], values['entry'], values['entry'].cudagraph,
                                    values['entry'].batch_descriptor, 'PIECEWISE', frame)
        elif frame.f_code is codes['break_capture']:
            if frame.f_lineno == lines['break_capture']:
                self.wrapper_capture(frame, 'BREAKABLE_PIECEWISE', values['capture'])
            elif frame.f_lineno == lines['break_capture_store']:
                self.finish_annotation(('capture', id(frame)))
                self.capture_stored(values['self'], values['entry'], values['entry'].capture,
                                    values['entry'].batch_descriptor, 'BREAKABLE_PIECEWISE', frame)
        elif frame.f_code is codes['segments'] and frame.f_lineno == lines['break_segment']:
            segment = values['r']
            receiver = getattr(segment, '__self__', None)
            row = self.emit('breakable_segment_attempt', capture_id=id(values['self']),
                      segment_id=id(segment),
                      segment_kind='GRAPH_REPLAY' if isinstance(receiver, self.sites.graph_type) else 'EAGER_BREAK',
                      graph_id=id(receiver) if isinstance(receiver, self.sites.graph_type) else None,
                      source_line=frame.f_lineno, outcome='ATTEMPT_ONLY', gpu_completion='UNMEASURED')
            observed = getattr(self.runner, '_step5_profiler_observation', None)
            if row is not None and observed is not None and row['segment_kind'] == 'GRAPH_REPLAY':
                self.update(row, **observed.graph_identity(receiver))

    def capture_stored(self, owner, entry, graph, descriptor, path, frame):
        metadata = self.metadata()
        descriptor = metadata.descriptor(descriptor)
        key = (self.breakable_key(frame, entry, metadata) if path == 'BREAKABLE_PIECEWISE'
               else descriptor)
        row = self.emit('capture_entry_stored', path=path, owner_id=id(owner), entry_id=id(entry),
                  capture_binding_event=self.capture_binding_event if path == 'FULL' else None,
                  graph_id=id(graph), descriptor=descriptor, selection_key=key, source_line=frame.f_lineno,
                  continuity=self.state.continuity(owner, entry, graph, key, path, captured=True),
                  device_operations_added=False, gpu_completion='UNMEASURED')
        observed = getattr(self.runner, '_step5_profiler_observation', None)
        if row is not None and observed is not None:
            self.update(row, **observed.graph_identity(graph))

    def wrapper_capture(self, frame, path, graph):
        values, metadata = frame.f_locals, self.metadata()
        descriptor = metadata.descriptor(values['entry'].batch_descriptor)
        key = (self.breakable_key(frame, values['entry'], metadata) if path == 'BREAKABLE_PIECEWISE'
               else descriptor)
        row = self.emit('wrapper_capture_forward', path=path, owner_id=id(values['self']),
                  entry_id=id(values['entry']), graph_id=id(graph),
                  descriptor=descriptor, selection_key=key,
                  args=metadata.value(values['args']), kwargs=metadata.value(values['kwargs']),
                  metadata_complete=metadata.complete, device_operations_added=False,
                  source_line=frame.f_lineno, gpu_completion='UNMEASURED')
        self.start_annotation(('capture', id(frame)), row, 'capture')

    def __enter__(self):
        if self.state.stopped:
            self.stopped = True
            return self
        if any(hook is not None and not callable(hook) for hook in (self.previous_profile, self.previous_trace)):
            self.error('hook_compatibility', TypeError('Prior non-callable hook cannot be composed or restored'))
            return self
        try:
            self.sites.bind_census(self.runner, self.state.limits)
            if self.state.census_inventory_record is None:
                row = self.emit('resident_model_bindings', frontiers=False,
                                **self.sites.census_inventory)
                if row is not None:
                    self.state.census_inventory_record = dict(lifecycle=self.kind,
                        sequence=self.state.captures if self.kind == 'capture' else self.state.invocations,
                        event=row['event'])
            state_rows, state_gaps = self.sites.refresh_state_bindings(self.runner, self.state.limits)
            if state_rows or state_gaps:
                self.emit('resident_state_bindings', frontiers=False, owners=state_rows, gaps=state_gaps,
                          device_completion='UNMEASURED', lifetime='OBSERVED_BINDING_BOUNDARIES_ONLY')
            self.start_annotation(('scope',), dict(event=0), 'scope')
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
        self.capture_node_state = None
        try:
            if self.installed:
                try:
                    if sys.getprofile() is not self.profile_hook or sys.gettrace() is not self.trace_hook:
                        self.error('hook_delivery', RuntimeError('Native execution replaced scoped hooks'))
                finally:
                    self.restore()
        except BaseException as failure:
            self.error('hook_restoration', failure)
        for key in reversed(list(self.annotations)):
            try:
                self.finish_annotation(key)
            except BaseException as failure:
                self.error('trace_annotation_exit', failure)
        if self.state.device is not None and self.device_step is not None:
            try:
                self.state.device.output(self.device_step, exc_type, self.errors, events=self.events)
            except BaseException as failure:
                self.state.device.error('post_execution_device_witness', failure)
                self.error('post_execution_device_witness', failure)
        for pending in self.pending.values():
            pending['row']['outcome'] = 'UNVERIFIED_MISSING_REPLAY_RETURN'
        if self.kind == 'execute' and not self.stopped:
            try:
                from role_capacity_observation import known_state_inventory
                self.emit('known_state_storage_after', frontiers=False,
                    **known_state_inventory(self.runner, self.sites.torch, self.state.limits.tracked_entries))
            except BaseException as failure:
                self.error('known_state_storage_after', failure)
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
                      sequence=self.state.captures if self.kind == 'capture' else self.state.invocations,
                      status=status, observation_status=observation_status,
                      events=self.events, call_tensor_metadata=self.call_tensors,
                      observer_errors=copy.deepcopy(self.errors), call_census=self.call_summary(),
                      native_outcome='raised' if exc_type is not None else 'returned',
                      native_error_type=exc_type.__module__+'.'+exc_type.__qualname__ if exc_type else None,
                      native_compatibility='UNVERIFIED', evidence_status=dict(UNMEASURED),
                      source_status='LOADED_CODE_MATCHES_PINNED_SOURCE',
                      device_step=self.device_step,
                      sources=self.sites.sources, hook_delivery='UNVERIFIED' if self.errors else
                      'OBSERVED_CALLBACKS' if self.events else 'NOT_OBSERVED')
        record = self.bounded_record(record)
        if self.kind == 'capture' and record is not None:
            try:
                self.state.capture_record(record)
            except BaseException as failure:
                self.error('capture_record', failure)
                record = self.bounded_record(dict(record, status='UNVERIFIED',
                    observer_errors=copy.deepcopy(self.errors)))
        self.state.last_record = record
        if record is None or 'observation_status' not in record:
            summary = None if record is None else dict(status='UNVERIFIED', lifecycle=self.kind,
                events=0, native_outcome=record['native_outcome'], observer_errors=copy.deepcopy(record['observer_errors']))
        else:
            summary = dict(status=record['status'], lifecycle=self.kind, events=len(self.events),
                observation_status=dict(observation_status), final_inputs_observed=self.input_id is not None,
                replay_attempts=sum(row['kind'] == 'replay_attempt' for row in self.events),
                native_outcome=record['native_outcome'], observer_errors=copy.deepcopy(record['observer_errors']))
        self.state.last_summary = summary if summary is not None and self.record_fits(summary) else None
        if self.stopped:
            self.state.stopped = True
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
