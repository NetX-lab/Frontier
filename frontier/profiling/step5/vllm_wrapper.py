"""Complete native Step attention forwards, with TP collectives excluded."""

from __future__ import annotations

import copy
import json
import os
from contextlib import ExitStack
from math import ceil
from pathlib import Path
from unittest.mock import patch

from frontier.attention.step5 import describe_step5_layer
from frontier.profiling.common.runtime_identity import step5_measurement_columns
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.step5.inputs import Step5ProfileInput
from frontier.profiling.step5.rows import OPERATOR_NAMES, Step5ProfileResult, build_row, summarize_replays
from frontier.profiling.step5.scopes import NativeScopes, trace_signature
from frontier.profiling.step5.replay import capture_native_replay
from frontier.profiling.step5.timing import ScopedReplay, collect_replay_timings, collect_scoped_timings


class VllmStep5AttentionWrapper:
    """Reuse the image's layer, cache views and backend metadata builder.

    One physical GPU measures one TP-local rank. Only the distributed shape
    descriptor is replaced; native projections, preparation, attention, selector,
    gating and cache updates execute unchanged. No simulator module imports vLLM.
    """

    def __init__(self, *, config, model_path, device, tensor_parallel_size,
                  layer_id, block_size, max_model_len, max_batch_size, trace_dir, seed, graph_mode,
                  cudagraph_capture_sizes=(1, 2), enable_chunked_prefill=True,
                  max_num_batched_tokens=None):
        # Native eager-break decorators bind at model import, before EngineArgs
        # can otherwise auto-enable the Step architecture's default.
        os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
        import torch
        import vllm.distributed.parallel_state as parallel
        from vllm.config import set_current_vllm_config
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.model_loader.utils import configure_quant_config
        from vllm.utils.network_utils import get_open_port
        from vllm.v1.worker.workspace import init_workspace_manager

        if not torch.cuda.is_available() or torch.version.hip:
            raise RuntimeError("The Step producer requires an NVIDIA CUDA device")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Use one process/GPU; this producer profiles one isolated TP rank")
        self.torch, self.config, self.device = torch, config, device
        self.tp, self.layer_id = tensor_parallel_size, layer_id
        self.block_size, self.max_model_len = block_size, max_model_len
        self.max_batch_size = max_batch_size
        self.seed = seed
        self.graph_mode = graph_mode
        self.cudagraph_capture_sizes = tuple(cudagraph_capture_sizes)
        self.profile_index = 0
        self.dummy_batches = {}
        if block_size not in (16, 32, 64, 128):
            raise ValueError("Native Step paged prefill requires block_size in 16, 32, 64, 128")
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.description = describe_step5_layer(config, layer_id, attention_tp_size=self.tp,
                                               moe_tp_size=1, expert_parallel_size=1)
        self._set_current = set_current_vllm_config
        self._cleanup = ExitStack()
        try:
            torch.cuda.set_device(0)
            torch.manual_seed(seed)
            outer = EngineArgs(
                model=model_path, load_format="dummy", skip_tokenizer_init=True,
                tensor_parallel_size=1, block_size=block_size,
                enable_prefix_caching=False, enable_chunked_prefill=enable_chunked_prefill,
                enforce_eager=False, max_num_seqs=max_batch_size,
                compilation_config={"mode": 0, "cudagraph_mode": "FULL_AND_PIECEWISE",
                                    "cudagraph_capture_sizes": list(self.cudagraph_capture_sizes),
                                    "max_cudagraph_capture_size": max(self.cudagraph_capture_sizes)},
                max_num_batched_tokens=(max_model_len * max_batch_size
                                        if max_num_batched_tokens is None else max_num_batched_tokens),
                max_model_len=max_model_len,
            ).create_engine_config()
            self.native = outer.with_hf_config(outer.model_config.hf_text_config)
            self._cleanup.enter_context(set_current_vllm_config(self.native))
            from vllm.models.step5.nvidia.model import Step5ForCausalLM

            configure_quant_config(self.native.quant_config, Step5ForCausalLM)
            if torch.distributed.is_initialized():
                raise RuntimeError("The isolated Step producer must own its process groups")
            parallel.init_distributed_environment(
                world_size=1, rank=0, local_rank=0, backend="nccl",
                distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}")
            self._cleanup.callback(parallel.destroy_distributed_environment)
            parallel.initialize_model_parallel(1, 1)
            self._cleanup.callback(parallel.destroy_model_parallel)
            import vllm.models.step5.nvidia.model as step_model
            import vllm.model_executor.layers.linear as linear

            # These are the native constructors' imported shape getters. Keep
            # the real process groups/configuration at world size one. No
            # collective implementation or kernel is replaced.
            for owner in (step_model, linear):
                self._cleanup.enter_context(patch.object(
                    owner, "get_tensor_model_parallel_world_size", lambda: self.tp))
                self._cleanup.enter_context(patch.object(
                    owner, "get_tensor_model_parallel_rank", lambda: 0))
            init_workspace_manager(torch.device("cuda"))
            original_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                with torch.device("cuda"), torch.no_grad():
                    self.layer = self._construct_layer()
                    self.layer.o_proj.reduce_results = False
                    self._initialize_weights()
                    self._bind_cache()
            finally:
                torch.set_default_dtype(original_dtype)
            self.builder = self.layer.attn.attn_backend.get_builder_cls()(
                kv_cache_spec=self.cache_spec, layer_names=[self.layer.attn.layer_name],
                vllm_config=self.native, device=torch.device("cuda"))
            self.layer.eval().requires_grad_(False)
            self._check_shapes()
            manifest = os.environ.get("FRONTIER_STEP5_PROFILE_MANIFEST")
            self.native_facts = None
            if manifest:
                from frontier.profiling.common.runtime_identity import file_sha256, verify_step5_profile_runtime
                self.native_facts = verify_step5_profile_runtime(manifest, callables=(type(self.layer), self.layer.forward))
                if file_sha256(Path(model_path) / "config.json") != self.native_facts["shared"]["model_config_sha256"]:
                    raise ValueError("native attention effective config differs from sealed local configuration")
            self.scopes = NativeScopes(torch)
            identity = step5_measurement_columns(self.native_facts)
            self.runtime_signature = identity["runtime_stack_signature"]
            self.evidence_kind = identity["evidence_kind"]
        except BaseException:
            self._cleanup.close()
            raise

    def _construct_layer(self):
        from vllm.models.step5.nvidia.config import get_step5_sparse_config
        from vllm.models.step5.nvidia.layernorm import get_step5_norm_dtype
        from vllm.models.step5.nvidia.model import Step5Attention, _get_step5_layer_types

        text = self.native.model_config.hf_config
        cache = copy.copy(self.native.cache_config)
        cache.sliding_window = None
        sparse = self.description.attention_spec.variant_id == "sparse_gqa"
        return Step5Attention(
            hidden_size=text.hidden_size, num_heads=text.num_attention_heads,
            num_kv_heads=text.num_attention_groups, max_position=text.max_position_embeddings,
            head_dim=text.head_dim, rms_norm_eps=text.rms_norm_eps,
            qkv_bias=getattr(text, "attention_bias", False), rope_theta=text.rope_theta,
            cache_config=cache, quant_config=self.native.quant_config,
            rope_scaling=getattr(text, "rope_scaling", None),
            prefix=f"model.layers.{self.layer_id}.self_attn", sliding_window=text.sliding_window,
            use_head_wise_attn_gate=text.use_head_wise_attn_gate,
            layer_types=_get_step5_layer_types(text), use_rope_layers=text.use_rope_layers,
            yarn_only_types=text.yarn_only_types, swa_num_attention_heads=text.swa_num_attention_heads,
            partial_rotary_factor=text.partial_rotary_factors[self.layer_id],
            zero_centered=text.zero_centered, vllm_config=self.native,
            sparse_config=get_step5_sparse_config(text) if sparse else None,
            norm_dtype=get_step5_norm_dtype(text))

    def _initialize_weights(self):
        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
        from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        record_metadata_for_reloading(self.layer)
        DummyModelLoader(self.native.load_config).load_weights(self.layer, self.native.model_config)
        process_weights_after_loading(self.layer, self.native.model_config, self.torch.device("cuda"))

    def _bind_cache(self):
        from vllm.v1.kv_cache_interface import KVCacheTensor, create_kv_cache_views

        torch, layer = self.torch, self.layer.attn
        self.cache_spec = layer.get_kv_cache_spec(self.native)
        self.block_size = self.cache_spec.block_size
        pages = 1 + self.max_batch_size * ceil(self.max_model_len / self.block_size)
        raw = torch.zeros(pages * self.cache_spec.page_size_bytes, dtype=torch.int8, device="cuda")
        self.raw_cache = raw
        layer.kv_cache = create_kv_cache_views(
            raw, self.cache_spec, pages, layer.attn_backend.supported_kv_cache_layouts()[0],
            KVCacheTensor(size=raw.numel(), layers=[layer.layer_name], layer_stride=raw.numel(),
                          block_stride=self.cache_spec.page_size_bytes))[0]
        self.state = [layer.kv_cache]
        if self.layer.sparse_summary_cache is not None:
            self.layer.sparse_summary_cache.bind_kv_cache_side_storage(
                self.native.compilation_config.static_forward_context)
            cache = self.layer.sparse_summary_cache._cache
            self.state += [cache._mean, cache.numerator, cache.denominator, cache.maximum, cache.last_slot]
            if cache.request_journal_width:
                self.state += [cache.request_journal_values, cache.request_journal_last, cache.request_journal_tag]
        self.empty_state = self._snapshot()

    def _check_shapes(self):
        expected = {p.module_prefix.rsplit(".", 1)[-1]: p.rank_weight_shape
                    for p in self.description.projections if p.operator_name in OPERATOR_NAMES}
        for name, shape in expected.items():
            projection = getattr(self.layer, name)
            if tuple(projection.weight.shape) != shape or projection.weight.dtype != self.torch.bfloat16:
                raise RuntimeError(f"Native projection {name} shape/dtype differs from Step contract: "
                                   f"{tuple(projection.weight.shape)} != {shape}")
        if self.layer.o_proj.reduce_results:
            raise RuntimeError("The attention compute scope must exclude the TP all-reduce")

    def _snapshot(self):
        return tuple(t.clone() for t in self.state)

    def _restore(self, state):
        for tensor, value in zip(self.state, state):
            tensor.copy_(value)

    def _metadata(self, cell, *, request_offset=0):
        if cell.dummy_descriptor is not None:
            return self._dummy_metadata(cell)
        if not cell.has_attention_metadata:
            return self.torch.zeros(1, dtype=self.torch.int64, device="cuda"), None, None, "none"
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        torch = self.torch
        width = (cell.block_table_columns if cell.block_table_columns is not None
                 else ceil(max(cell.sequence_lens) / self.block_size))
        # Reserve block zero and give each live request disjoint physical pages.
        stride = ceil(self.max_model_len / self.block_size)
        table = torch.arange(1 + request_offset * stride,
                             1 + (request_offset + cell.batch_size) * stride, dtype=torch.int32,
                             device="cuda").reshape(cell.batch_size, stride)[:, :width]
        positions = torch.cat([torch.arange(c, c + q, device="cuda")
                               for q, c in zip(cell.query_lens, cell.context_lens)])
        slots = torch.cat([table[i, p // self.block_size].long() * self.block_size + p % self.block_size
                           for i, p in enumerate(positions.split(cell.query_lens))])
        # MRv2 gives FULL padded requests empty queries, zero lengths and zero
        # table rows, and gives every padded token row position 0 and PAD_SLOT_ID.
        pad_requests = cell.physical_requests - cell.batch_size
        pad_rows = cell.physical_query_rows - cell.num_tokens
        table = torch.cat((table, table.new_zeros(pad_requests, width)))
        positions = torch.cat((positions, positions.new_zeros(pad_rows)))
        slots = torch.cat((slots, slots.new_full((pad_rows,), PAD_SLOT_ID)))
        starts = torch.tensor([0, *cell.query_lens, *(0,) * pad_requests],
                              dtype=torch.int32).cumsum(0, dtype=torch.int32)
        lengths = torch.tensor((*cell.sequence_lens, *(0,) * pad_requests), dtype=torch.int32)
        # FULL metadata spans the padded rows; PIECEWISE and eager use the valid counts.
        common = CommonAttentionMetadata(
            query_start_loc=starts.cuda(), query_start_loc_cpu=starts, seq_lens=lengths.cuda(),
            num_reqs=cell.physical_requests,
            num_actual_tokens=cell.physical_query_rows if cell.graph_mode == "full" else cell.num_tokens,
            max_query_len=max(cell.query_lens), max_seq_len=max(cell.sequence_lens),
            block_table_tensor=table, slot_mapping=slots, positions=positions,
            seq_lens_cpu_upper_bound=lengths,
            is_prefilling=torch.tensor([query > 1 for query in cell.query_lens]))
        with self._set_current(self.native), torch.device("cpu"):
            metadata = self.builder.build(0, common, fast_build=False)
        return positions, slots, metadata, cell.graph_mode

    def _dummy_metadata(self, cell):
        """Use the same MRv2 helpers as execute_model after DP synchronization."""
        from vllm.config import CUDAGraphMode
        from vllm.models.step5.nvidia.model_state import Step5ModelState
        from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
        from vllm.v1.worker.gpu.block_table import BlockTables
        from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
        from vllm.v1.worker.utils import AttentionGroup

        torch, width = self.torch, cell.block_table_columns
        if width not in self.dummy_batches:
            device = torch.device("cuda")
            budget = self.native.scheduler_config.max_num_batched_tokens
            buffers = InputBuffers(self.max_batch_size, budget, device)
            tables = BlockTables([self.block_size], self.max_batch_size, budget, [width], device)
            name = self.layer.attn.layer_name
            group = AttentionGroup(self.layer.attn.attn_backend, [name], self.cache_spec, 0,
                                   metadata_builders=[self.builder])
            # The native layer/cache is already bound. This config identifies
            # its one isolated metadata group; it allocates no replacement KV.
            cache = KVCacheConfig(self.raw_cache.numel() // self.cache_spec.page_size_bytes,
                                  [], [KVCacheGroupSpec([name], self.cache_spec)])
            state = Step5ModelState(self.native, self.layer, None, device)
            self.dummy_batches[width] = buffers, tables, state, [[group]], cache
        buffers, tables, state, groups, cache = self.dummy_batches[width]
        descriptor = cell.dummy_descriptor
        batch = InputBatch.make_dummy(cell.batch_size, descriptor.num_tokens, buffers,
                                     max_query_len=descriptor.max_query_len, is_padding=True)
        # These helpers zero persistent block tables and fill the entire slots
        # buffer with PAD_SLOT_ID. A nonzero caller-owned history is never used.
        blocks = tables.get_dummy_block_tables(batch.num_reqs)
        slots = tables.get_dummy_slot_mappings(batch.num_tokens)
        mode = CUDAGraphMode[descriptor.graph_mode.upper()]
        with self._set_current(self.native):
            metadata = state.prepare_attn(batch, mode, blocks, slots, groups, cache,
                                          for_capture=descriptor.graph_mode == "full")
        return batch.positions, slots[0], metadata[self.layer.attn.layer_name], cell.graph_mode, batch.is_padding

    def _forward(self, hidden, prepared, *, runtime_mode=None, layer=None):
        from vllm.forward_context import BatchDescriptor, set_forward_context
        from vllm.config import CUDAGraphMode

        positions, slots, metadata, graph_mode, *padding = prepared
        graph_mode = runtime_mode if runtime_mode is not None else graph_mode
        name = self.layer.attn.layer_name
        with self._set_current(self.native), set_forward_context(
                {name: metadata} if metadata is not None else None,
                self.native, num_tokens=len(hidden), slot_mapping={name: slots} if slots is not None else None,
                is_padding=padding[0] if padding else None,
                cudagraph_runtime_mode=CUDAGraphMode[graph_mode.upper()],
                batch_descriptor=BatchDescriptor(num_tokens=len(hidden))):
            return (self.layer if layer is None else layer)(positions, hidden)

    def _bindings(self):
        import vllm._custom_ops as ops
        import vllm.models.step5.nvidia.model as model
        import vllm.models.step5.nvidia.ops.indexer_norm_rope as indexer
        import vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope as joint

        layer = self.layer
        result = [(layer.o_proj, "forward", "step5_output_projection"),
                  (ops, "fused_qknorm_rope_cache", "step5_attention_prepare"),
                  (layer.q_norm, "forward", "step5_attention_prepare"),
                  (layer.k_norm, "forward", "step5_attention_prepare"),
                  (layer.rotary_emb, "forward", "step5_attention_prepare")]
        if layer.use_dsa_backend:
            result += [(layer.qkv_indexer_proj, "forward", "step5_qkv_projection"),
                       (layer.sparse_indexer_w, "forward", "step5_indexer_weights"),
                       (indexer, "indexer_norm_rope", "step5_indexer_prepare"),
                       (joint, "qkv_indexer_norm_rope", "step5_attention_prepare"),
                       (model, "step5_dsa_attention_with_output", "step5_sparse_attention")]
        else:
            result += [(getattr(layer, "qkvg_proj" if layer.fuse_qkv_gate else "qkv_proj"),
                        "forward", "step5_qkv_projection"),
                       (layer.attn, "forward", "step5_swa_attention"),
                       (model, "headwise_sigmoid_gate", "step5_headwise_gate")]
            if not layer.fuse_qkv_gate:
                result.append((layer.g_proj, "forward", "step5_gate_projection"))
        return result

    def _live_state(self, state):
        """Padded and dummy rows may update the CSA scratch pages; only live pages are results."""
        if self.layer.sparse_summary_cache is None:
            return tuple(state)
        pages = self.layer.sparse_summary_cache._cache.pages
        return (state[0], *(tensor[:pages] for tensor in state[1:6]), *state[6:])

    def _validate_output_state(self, actual, expected, expected_state):
        # Rows past the valid tokens are graph padding, not request output.
        actual = actual[:len(expected)]
        if not self.torch.isfinite(actual).all():
            raise AssertionError("Native Step attention produced nonfinite output")
        self.torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        for current, reference in zip(self._live_state(self.state), self._live_state(expected_state)):
            self.torch.testing.assert_close(current, reference, rtol=2e-2, atol=2e-2)

    def _scoped_replay(self, hidden, prepared, initial, expected, expected_state):
        torch = self.torch
        recorded = {}

        def forward(**kwargs):
            start = torch.cuda.Event(enable_timing=True, external=self.scopes.graph_capture)
            end = torch.cuda.Event(enable_timing=True, external=self.scopes.graph_capture)
            start.record()
            output = self._forward(hidden, prepared, **kwargs)
            end.record()
            recorded["layer"] = start, end
            # Breakable replay re-enters Python but retains its captured events.
            if self.graph_mode == "none" or "operators" not in recorded:
                recorded["operators"] = self.scopes.events
            return output

        if self.graph_mode == "none":
            def run():
                self.scopes.events = {}
                with self.scopes.observe(self._bindings()):
                    return forward(runtime_mode="none")
        else:
            self._restore(initial)
            self.scopes.events = {}
            self.scopes.graph_capture = True
            try:
                with self.scopes.observe(self._bindings()):
                    run = capture_native_replay(torch, self.native, self.layer, forward, self.graph_mode)
            finally:
                self.scopes.graph_capture = False
            self._restore(initial)
            actual = run()
            torch.cuda.synchronize()
            self._validate_output_state(actual, expected, expected_state)

        def times():
            result = {name: sum(start.elapsed_time(end) for start, end in pairs)
                      for name, pairs in recorded["operators"].items()}
            result["step5_layer_scoped"] = recorded["layer"][0].elapsed_time(recorded["layer"][1])
            return result

        return ScopedReplay(run=run, times=times)

    def _gpu_activity(self, hidden, prepared, initial, active, stem, *, sample_count, min_window_ms):
        from frontier.profiling.step5.device_activity import parse_device_activity
        from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer

        torch = self.torch

        def capture(replays, suffix):
            trace = self.trace_dir / f"{stem}_activity_{suffix}.json"
            tracer = RecordFunctionTracer(output_path=str(self.trace_dir))
            tracer.trace_path = str(trace)
            intervals = []
            with tracer, self.scopes.observe(self._bindings(), record_events=False):
                for window in range(sample_count):
                    self._restore(initial)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for replay in range(replays):
                        with torch.profiler.record_function(f"step5_iteration/{window}/{replay}"):
                            output = self._forward(hidden, prepared, runtime_mode="none")
                    end.record()
                    end.synchronize()
                    intervals.append(start.elapsed_time(end))
            return parse_device_activity(trace, active), trace, intervals, output

        pilot, _, _, _ = capture(1, "pilot")
        replays = max(1, ceil(min_window_ms / min(
            item["step5_layer_e2e"] for item in pilot["per_iteration"].values())))
        attempt = 0
        while True:
            attempt += 1
            activity, trace, intervals, output = capture(replays, str(attempt))
            window_times = {name: [] for name in (*OPERATOR_NAMES, "step5_layer_e2e")}
            raw_times = {name: [] for name in window_times}
            for window in range(sample_count):
                entries = [activity["per_iteration"][f"step5_iteration/{window}/{replay}"]
                           for replay in range(replays)]
                for name in window_times:
                    values = [entry[name] if name in entry else 0. for entry in entries]
                    raw_times[name].extend(values)
                    window_times[name].append(sum(values) / replays)
            durations = [value * replays for value in window_times["step5_layer_e2e"]]
            shortest = min(durations)
            if shortest >= min_window_ms:
                break
            replays = max(replays + 1, ceil(replays * min_window_ms / shortest))
        evidence = {key: value for key, value in activity.items()
                    if key not in ("per_iteration", "per_iteration_kernel_counts")}
        evidence["per_replay_kernel_counts"] = activity["per_iteration_kernel_counts"]
        evidence["kernel_names"] = json.dumps(evidence["kernel_names"])
        evidence["memory_copy_names"] = json.dumps(evidence["memory_copy_names"])
        evidence.update({"trace_path": str(trace), "timing_basis": "correlated_kernel_plus_memcpy_duration_sum",
                         "kernel_only_includes_memory_copies": True,
                         "window_replays": json.dumps([replays] * sample_count),
                         "window_durations_ms": json.dumps(durations),
                         "trace_window_event_durations_ms": json.dumps(intervals),
                         "operator_ratio_definition": "exclusive_gpu_activity_sum / same_iteration_gpu_activity_sum",
                         "operator_sum_to_layer_ratio": sum(sum(window_times[name]) for name in active)
                                                        / sum(window_times["step5_layer_e2e"]),
                         "calibration_candidate": True})
        return window_times, raw_times, evidence, output

    def profile(self, cell, *, warmup_iterations, profile_iterations, max_time_cv, min_timing_window_ms=0.5):
        if cell.global_layer_id != self.layer_id or cell.graph_mode != self.graph_mode:
            raise ValueError("Profile cell must match the constructed layer and graph policy")
        if cell.physical_requests > self.max_batch_size:
            raise ValueError("Profile batch exceeds the configured request capacity")
        if cell.dummy_descriptor is not None and cell.num_tokens > self.native.scheduler_config.max_num_batched_tokens:
            raise ValueError("MRv2 synchronized descriptor exceeds the native token budget")
        cell.workload(self.config, block_size=self.block_size, max_model_len=self.max_model_len)
        if self.layer.use_dsa_backend and any(
                context % self.config.sparse_config["region_block_size"] for context in cell.context_lens):
            raise ValueError("Repeated Step DSA timing requires region-aligned contexts; partial summary tails are stateful")
        torch = self.torch
        self.profile_index += 1
        torch.manual_seed(self.seed)
        with torch.no_grad(), self._set_current(self.native):
            self._restore(self.empty_state)
            prefix = max(cell.context_lens)
            # Keep the same physical page assignments for prefix and continuation.
            if prefix:
                if 0 in cell.context_lens:
                    raise ValueError("Mixed zero/nonzero prefixes require independent request-page preparation")
                prefix_cell = Step5ProfileInput(cell.global_layer_id, cell.context_lens,
                    (0,) * cell.batch_size, "decode" if prefix == 1 else "prefill",
                    block_table_columns=cell.block_table_columns)
                prefix_hidden = torch.randn((prefix_cell.num_tokens, self.config.embedding_dim),
                                            dtype=torch.bfloat16, device="cuda")
                if prefix_cell.num_tokens <= self.native.scheduler_config.max_num_batched_tokens:
                    self._forward(prefix_hidden, self._metadata(prefix_cell))
                else:
                    # Preparation may span more history than one serving step.
                    # Preserve each request's pages without inflating its budget.
                    for index, (length, values) in enumerate(zip(
                            cell.context_lens, prefix_hidden.split(cell.context_lens))):
                        part = Step5ProfileInput(cell.global_layer_id, (length,), (0,),
                            "decode" if length == 1 else "prefill", block_table_columns=cell.block_table_columns)
                        self._forward(values, self._metadata(part, request_offset=index))
            prepared = self._metadata(cell)
            initial = self._snapshot()
            hidden = torch.randn((cell.physical_query_rows, self.config.embedding_dim),
                                 dtype=torch.bfloat16, device="cuda")
            for _ in range(warmup_iterations):
                self._restore(initial)
                self._forward(hidden, prepared)
            torch.cuda.synchronize()
            self._restore(initial)
            expected = self._forward(hidden, prepared)[:cell.num_tokens].clone()
            expected_state = self._snapshot()
            if not cell.has_attention_metadata:
                for actual_state, reference in zip(expected_state, initial):
                    torch.testing.assert_close(actual_state, reference, rtol=0, atol=0)
            if cell.dummy_descriptor is not None:
                # The KV cache and live CSA summaries must remain unchanged.
                for actual_state, reference in zip(self._live_state(expected_state)[:6],
                                                   self._live_state(initial)[:6]):
                    torch.testing.assert_close(actual_state, reference, rtol=0, atol=0)
            if prefix:
                full_lens = cell.sequence_lens
                full_cell = Step5ProfileInput(cell.global_layer_id, full_lens,
                    (0,) * cell.batch_size, "prefill", block_table_columns=cell.block_table_columns)
                full_hidden = torch.cat([torch.cat((p, h)) for p, h in zip(
                    prefix_hidden.split(cell.context_lens), hidden[:cell.num_tokens].split(cell.query_lens))])
                self._restore(self.empty_state)
                if full_cell.num_tokens <= self.native.scheduler_config.max_num_batched_tokens:
                    full_output = self._forward(full_hidden, self._metadata(full_cell))
                    full_tail = torch.cat([part[context:] for part, context in
                                           zip(full_output.split(full_lens), cell.context_lens)])
                else:
                    tails = []
                    for index, (length, context, values) in enumerate(zip(
                            full_lens, cell.context_lens, full_hidden.split(full_lens))):
                        part = Step5ProfileInput(cell.global_layer_id, (length,), (0,), "prefill",
                                                block_table_columns=cell.block_table_columns)
                        output = self._forward(values, self._metadata(part, request_offset=index))
                        tails.append(output[context:].clone())
                    full_tail = torch.cat(tails)
                torch.testing.assert_close(expected, full_tail, rtol=2e-2, atol=2e-2)
                for actual_state, reference in zip(self._live_state(self.state),
                                                   self._live_state(expected_state)):
                    torch.testing.assert_close(actual_state, reference, rtol=2e-2, atol=2e-2)
                # FA3 FULL metadata owns persistent scheduler buffers; a full
                # reference build overwrites them. Rebuild the measured decode
                # metadata only after the reference has finished using them.
                prepared = self._metadata(cell)
            self._restore(initial)
            self.scopes.events = {}
            with self.scopes.observe(self._bindings()):
                actual = self._forward(hidden, prepared)
            torch.cuda.synchronize()
            if not torch.isfinite(expected).all():
                raise AssertionError("Native Step attention produced nonfinite output")
            self._validate_output_state(actual, expected, expected_state)
            active = tuple(self.scopes.times())
            expected_active = tuple(name for name, _ in self.description.attention_scopes(
                cell.workload(self.config, block_size=self.block_size, max_model_len=self.max_model_len)))
            if active != expected_active:
                raise RuntimeError(f"Native Step scope coverage differs from contract: {active} != {expected_active}")
            self._restore(initial)
            clean_run = capture_native_replay(torch, self.native, self.layer,
                lambda **kwargs: self._forward(hidden, prepared, **kwargs), cell.graph_mode)
            self._restore(initial)
            self._validate_output_state(clean_run(), expected, expected_state)
            clean = collect_replay_timings(torch, clean_run, lambda: self._restore(initial),
                sample_count=profile_iterations, min_window_ms=min_timing_window_ms)
            self._validate_output_state(clean.last_output, expected, expected_state)
            scoped = collect_scoped_timings(torch,
                lambda: self._scoped_replay(hidden, prepared, initial, expected, expected_state),
                lambda: self._restore(initial), window_replays=clean.window_replays,
                operator_names=active)
            self._validate_output_state(scoped.last_output, expected, expected_state)
            samples = {name: list(scoped.amortized_times[name]) if name in active
                       else [0.] * profile_iterations for name in OPERATOR_NAMES}
            samples["step5_layer_e2e"] = clean.amortized_times
            raw = {name: scoped.per_replay_times[name] if name in active
                   else [0.] * profile_iterations for name in OPERATOR_NAMES}
            stats, quality = summarize_replays(samples, raw, active, max_time_cv=max_time_cv,
                layer_reference_times=clean.reference_times)
            scoped_stats = TimerStatsStore.get_stats_from_times(scoped.amortized_times)
            outer_stats = TimerStatsStore.get_stats_from_times({"layer": [duration / count
                for duration, count in zip(scoped.window_durations_ms, clean.window_replays)]})["layer"]
            stem = (f"layer{self.layer_id}_{cell.phase}_{cell.graph_mode}"
                f"_batch{cell.batch_size}_q{max(cell.query_lens)}_c{prefix}"
                f"_attempt{self.profile_index}")
            trace = self.trace_dir / f"{stem}.json"
            self._restore(initial)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                clean_run()
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(trace))
            signature = trace_signature(trace)
            if not signature["kernel_count"]:
                raise RuntimeError("Native Step kernel observation was empty")
            stats["step5_layer_e2e"]["kernel_count"] = signature["kernel_count"]
            common = {
                    "evidence_kind": self.evidence_kind,
                    "attention_backend": self.layer.attn.attn_backend.get_name(),
                    "native_impl": type(self.layer.attn.impl).__name__,
                    "rank_aggregation": "isolated_rank_zero_no_collectives",
                    "physical_gpus": 1, "output_state_validation": "PASS",
                    "seed": self.seed,
                    "warmup_iterations": warmup_iterations, "profile_iterations": profile_iterations,
                    "native_cudagraph_policy": str(self.native.compilation_config.cudagraph_mode),
                    "native_cudagraph_capture_sizes": json.dumps(self.native.compilation_config.cudagraph_capture_sizes),
                    "native_max_num_seqs": self.native.scheduler_config.max_num_seqs,
                    "native_max_num_batched_tokens": self.native.scheduler_config.max_num_batched_tokens,
                    "native_breakable_cudagraph": os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"],
                    "graph_dispatch_basis": ("explicit_mrv2_post_dp_sync_descriptor" if cell.dummy_descriptor else
                                             "independent_eager_profile" if cell.graph_mode == "none"
                                             else "native_dispatch"),
                    "prefix_continuation_validation": "PASS" if prefix else "not_applicable",
                    "active_operators": json.dumps(sorted(active)),
                    "min_timing_window_ms": min_timing_window_ms}
            if self.native_facts is not None:
                from frontier.profiling.common.runtime_identity import verify_step5_profile_runtime
                facts = verify_step5_profile_runtime(os.environ["FRONTIER_STEP5_PROFILE_MANIFEST"], callables=(type(self.layer), self.layer.forward))
                common["native_facts"] = json.dumps(facts, sort_keys=True)
            row_args = dict(config=self.config, cell=cell, tp=self.tp, block_size=self.block_size,
                max_model_len=self.max_model_len, runtime_signature=self.runtime_signature, device=self.device)
            event_row = build_row(**row_args, time_stats=stats, evidence={**common, **signature, **quality,
                "trace_path": str(trace), "timing_basis": "cuda_event_native_runtime_latency",
                "calibration_candidate": cell.graph_mode != "none",
                "window_replays": json.dumps(clean.window_replays),
                "window_durations_ms": json.dumps(clean.window_durations_ms),
                "layer_reference_timing_basis": "independent_outer_only_normalized_clean_replay_windows",
                "layer_reference_times_ms": json.dumps(clean.reference_times),
                "reference_window_replays": json.dumps(clean.reference_window_replays),
                "reference_window_durations_ms": json.dumps(clean.reference_window_durations_ms),
                "boundary_single_replay_role": "instrumentation_diagnostics_only",
                "boundary_single_replay_times_ms": json.dumps(clean.single_replay_times),
                "boundary_single_replay_time_stats": json.dumps(TimerStatsStore.get_stats_from_times(
                    {"layer": clean.single_replay_times})["layer"]),
                "scoped_layer_time_stats": json.dumps(scoped_stats["step5_layer_scoped"]),
                "scoped_outer_time_stats": json.dumps(outer_stats),
                "operator_ratio_definition": "exclusive_scope_window_sum / matching_instrumented_layer_window_sum",
                "operator_sum_to_layer_ratio": sum(stats[name]["mean"] for name in active)
                    / scoped_stats["step5_layer_scoped"]["mean"],
                "instrumentation_overhead_fraction": float(outer_stats["mean"]
                    / stats["step5_layer_e2e"]["mean"] - 1)})
            gpu_active_row = None
            if cell.graph_mode == "none":
                activity, raw_activity, evidence, actual = self._gpu_activity(
                    hidden, prepared, initial, active, stem, sample_count=profile_iterations,
                    min_window_ms=min_timing_window_ms)
                self._validate_output_state(actual, expected, expected_state)
                gpu_stats, gpu_quality = summarize_replays(activity, raw_activity, active, max_time_cv=max_time_cv)
                counts = evidence.pop("per_replay_kernel_counts")
                root_counts = [entry["step5_layer_e2e"] for entry in counts.values()]
                if len(set(root_counts)) != 1:
                    raise ValueError("A fixed profile cell has varying measured per-replay root kernel counts")
                gpu_stats["step5_layer_e2e"]["kernel_count"] = root_counts[0]
                gpu_active_row = build_row(**row_args, time_stats=gpu_stats,
                    measurement_type="KERNEL_ONLY", evidence={**common, **evidence, **gpu_quality})
            return Step5ProfileResult(event_row=event_row, gpu_active_row=gpu_active_row)

    def close(self):
        self._cleanup.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
