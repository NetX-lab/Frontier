"""Truth-backed Qwen3.5 GDN profiler for the pinned vLLM runtime."""

from __future__ import annotations

import importlib.metadata
import os
from contextlib import nullcontext
from typing import Any

from frontier.gdn import GATED_DELTA_NET_FAMILY
from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.gdn.inputs import GDNProfileInput
from frontier.profiling.utils import profile_method_to_measurement_type


class VllmQwen35GDNWrapper:
    """Profile the actual vLLM Qwen GDN module with synthetic BF16 weights.

    Multi-rank profiles use torchrun and aggregate the slowest rank for every
    timing sample. The module is constructed with ``reduce_results=False`` so
    Frontier can model TP collectives through its communication family.
    """

    def __init__(
        self,
        *,
        frontier_model_config: Any,
        model_path: str,
        device_name: str,
        profile_method: str = "cuda_event",
        max_model_len: int = 4096,
        max_batch_size: int = 128,
        tensor_parallel_size: int = 1,
    ) -> None:
        tensor_parallel_size = int(tensor_parallel_size)
        if tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if frontier_model_config.get_gdn_config() is None:
            raise ValueError("GDN profiler requires a model with GDN dimensions")
        if frontier_model_config.get_num_gdn_layers() <= 0:
            raise ValueError("GDN profiler requires at least one GDN layer")

        import torch
        import triton
        import vllm
        from vllm.config import (
            CacheConfig,
            CompilationConfig,
            CompilationMode,
            ModelConfig as VllmModelConfig,
            ParallelConfig,
            VllmConfig,
            set_current_vllm_config,
        )
        from vllm.distributed import (
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
            model_parallel_is_initialized,
        )
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            GDN_AITER_TRITON_AVAILABLE,
            QwenGatedDeltaNetAttention,
            _encode_layer_name,
        )
        from vllm.utils.network_utils import get_open_port
        from vllm.utils.torch_utils import set_default_torch_dtype

        if not torch.cuda.is_available():
            raise RuntimeError("GDN profiling requires a CUDA/HIP device")
        if getattr(torch.version, "hip", None) is None:
            raise RuntimeError("This GDN profiler slice targets ROCm/gfx950")
        self.torch = torch
        self.frontier_model_config = frontier_model_config
        self.model_path = str(model_path)
        self.device_name = str(device_name)
        self.profile_method = str(profile_method)
        self.max_batch_size = int(max_batch_size)
        self.tensor_parallel_size = tensor_parallel_size
        distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if distributed_world_size != self.tensor_parallel_size:
            raise RuntimeError(
                "GDN profiling world size must equal tensor parallel size: "
                f"WORLD_SIZE={distributed_world_size}, "
                f"tensor_parallel_size={self.tensor_parallel_size}. Use "
                "torchrun --nproc-per-node=<tensor_parallel_size>."
            )
        self.prefix = "model.layers.0.linear_attn"
        self.encoded_layer_name = _encode_layer_name(self.prefix)
        self.use_aiter_dispatch = bool(GDN_AITER_TRITON_AVAILABLE)
        self._owns_distributed = False
        self._destroy_model_parallel = destroy_model_parallel

        torch.cuda.set_device(self.local_rank)
        vllm_model_config = VllmModelConfig(
            model=self.model_path,
            tokenizer=self.model_path,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=int(max_model_len),
            enforce_eager=True,
            # The checkpoint excludes every GDN projection from MXFP4. Avoid
            # constructing unrelated Quark quantizers for this isolated layer.
            hf_overrides={"quantization_config": None},
        )
        self.vllm_config = VllmConfig(
            model_config=vllm_model_config,
            cache_config=CacheConfig(
                block_size=16,
                enable_prefix_caching=False,
                mamba_cache_dtype="auto",
                mamba_ssm_cache_dtype="auto",
            ),
            parallel_config=ParallelConfig(
                tensor_parallel_size=self.tensor_parallel_size
            ),
            compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        )
        self._set_current_vllm_config = set_current_vllm_config

        with set_current_vllm_config(self.vllm_config):
            if not torch.distributed.is_initialized():
                distributed_init_method = (
                    "env://"
                    if distributed_world_size > 1
                    else f"tcp://127.0.0.1:{get_open_port()}"
                )
                init_distributed_environment(
                    world_size=distributed_world_size,
                    rank=self.rank,
                    local_rank=self.local_rank,
                    distributed_init_method=distributed_init_method,
                )
                self._owns_distributed = True
            if not model_parallel_is_initialized():
                initialize_model_parallel(
                    tensor_model_parallel_size=self.tensor_parallel_size,
                    pipeline_model_parallel_size=1,
                )
            with set_default_torch_dtype(torch.bfloat16):
                self.layer = QwenGatedDeltaNetAttention(
                    vllm_model_config.hf_config,
                    self.vllm_config,
                    prefix=self.prefix,
                    gqa_interleaved_layout=False,
                    reduce_results=False,
                ).to(device="cuda")

        self._initialize_synthetic_weights()
        self.layer.requires_grad_(False)
        self.layer.eval()
        self.kv_cache_spec = self.layer.get_kv_cache_spec(self.vllm_config)
        if self.kv_cache_spec is None:
            raise RuntimeError("vLLM Qwen GDN layer did not expose a state-cache spec")
        self.raw_state_pages = torch.zeros(
            (self.max_batch_size + 1, 1, 1, self.kv_cache_spec.page_size_bytes),
            dtype=torch.int8,
            device="cuda",
        )
        self.layer.bind_kv_cache(self.raw_state_pages)

        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

        self.metadata_builder = GDNAttentionMetadataBuilder(
            self.kv_cache_spec,
            [self.prefix],
            self.vllm_config,
            torch.device("cuda"),
        )
        self.timer_stats_store = TimerStatsStore(profile_method=self.profile_method)
        self.runtime_stack_signature = ";".join(
            (
                f"vllm={vllm.__version__}",
                f"torch={torch.__version__}",
                f"hip={torch.version.hip}",
                f"triton={triton.__version__}",
                f"aiter={self._package_version('amd-aiter')}",
            )
        )

        actual_shapes = tuple(tuple(shape) for shape in self.layer.get_state_shape())
        actual_dtypes = tuple(self.layer.get_state_dtype())
        expected_layout = frontier_model_config.get_gdn_config().get_state_layout(
            tensor_parallel_size=self.tensor_parallel_size,
            conv_bytes_per_element=actual_dtypes[0].itemsize,
            recurrent_bytes_per_element=actual_dtypes[1].itemsize,
        )
        expected_shapes = (
            expected_layout.conv_state_shape,
            expected_layout.recurrent_state_shape,
        )
        if actual_shapes != expected_shapes:
            raise RuntimeError(
                f"Frontier/vLLM GDN state-shape mismatch: {expected_shapes} != "
                f"{actual_shapes}"
            )
        if int(self.kv_cache_spec.page_size_bytes) != expected_layout.total_bytes:
            raise RuntimeError(
                "Frontier/vLLM GDN state-byte mismatch: "
                f"{expected_layout.total_bytes} != {self.kv_cache_spec.page_size_bytes}"
            )

    @staticmethod
    def _package_version(package_name: str) -> str:
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    def _initialize_synthetic_weights(self) -> None:
        torch = self.torch
        with torch.no_grad():
            for name, parameter in self.layer.named_parameters():
                if name.endswith("A_log") or name.endswith("dt_bias"):
                    parameter.zero_()
                elif name.endswith("norm.weight"):
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(mean=0.0, std=0.01)

    def _build_metadata(self, profile_input: GDNProfileInput):
        if profile_input.batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size={profile_input.batch_size} exceeds configured "
                f"max_batch_size={self.max_batch_size}"
            )
        torch = self.torch
        from vllm.v1.attention.backend import CommonAttentionMetadata

        query_start_loc_cpu = torch.zeros(
            profile_input.batch_size + 1,
            dtype=torch.int32,
        )
        torch.cumsum(
            torch.tensor(profile_input.query_lens, dtype=torch.int32),
            dim=0,
            out=query_start_loc_cpu[1:],
        )
        seq_lens_cpu = torch.tensor(
            [
                query_len + context_len
                for query_len, context_len in zip(
                    profile_input.query_lens,
                    profile_input.context_lens,
                )
            ],
            dtype=torch.int32,
        )
        common_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc_cpu.cuda(),
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens_cpu.cuda(),
            num_reqs=profile_input.batch_size,
            num_actual_tokens=profile_input.num_tokens,
            max_query_len=profile_input.max_query_len,
            max_seq_len=int(seq_lens_cpu.max().item()),
            # vLLM reserves block zero as NULL_BLOCK_ID. The extra state page
            # allocated by the wrapper is the null page; requests begin at 1.
            block_table_tensor=torch.tensor(
                profile_input.state_block_ids,
                dtype=torch.int32,
                device="cuda",
            ).view(-1, 1),
            slot_mapping=torch.arange(
                profile_input.num_tokens,
                dtype=torch.int64,
                device="cuda",
            ),
            _num_computed_tokens_cpu=torch.tensor(
                profile_input.context_lens,
                dtype=torch.int32,
            ),
        )
        return self.metadata_builder.build(0, common_metadata)

    def _forward_context(self, metadata, num_tokens: int):
        from vllm.forward_context import set_forward_context

        return set_forward_context(
            {self.prefix: metadata},
            self.vllm_config,
            num_tokens=num_tokens,
        )

    def _run_e2e(self, hidden_states, metadata, *, timed: bool):
        timer = CudaTimer("gdn_layer_e2e") if timed else nullcontext()
        with self._set_current_vllm_config(self.vllm_config):
            with self._forward_context(metadata, hidden_states.shape[0]):
                with timer:
                    if self.use_aiter_dispatch:
                        return self.layer.forward_hip(hidden_states)
                    return self.layer.forward_cuda(hidden_states)

    def _run_decomposed(
        self,
        hidden_states,
        metadata,
        phase: str,
        *,
        timed: bool,
    ):
        torch = self.torch
        projection_timer = (
            CudaTimer("gdn_input_projections") if timed else nullcontext()
        )
        core_timer = CudaTimer(f"gdn_core_{phase}") if timed else nullcontext()
        output_timer = (
            CudaTimer("gdn_output_projection") if timed else nullcontext()
        )
        with self._set_current_vllm_config(self.vllm_config):
            with self._forward_context(metadata, hidden_states.shape[0]):
                with projection_timer:
                    projected_qkvz, _ = self.layer.in_proj_qkvz(hidden_states)
                    projected_ba, _ = self.layer.in_proj_ba(hidden_states)
                    projected_qkvz = projected_qkvz.view(hidden_states.shape[0], -1)
                    projected_ba = projected_ba.view(hidden_states.shape[0], -1)
                # Match vLLM's generic ROCm path exactly. Some metadata rows do
                # not overwrite every output lane, so an uninitialized buffer
                # would make the decomposition both incorrect and unstable.
                core_attn_out = torch.zeros(
                    (
                        hidden_states.shape[0],
                        self.layer.num_v_heads // self.layer.tp_size,
                        self.layer.head_v_dim,
                    ),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                z = torch.empty_like(core_attn_out)
                with core_timer:
                    if self.use_aiter_dispatch:
                        torch.ops.vllm.qwen_gdn_attention_core(
                            projected_qkvz,
                            projected_ba,
                            z,
                            core_attn_out,
                            layer_name=self.encoded_layer_name,
                            use_aiter=True,
                        )
                    else:
                        qkv_size = (
                            self.layer.key_dim * 2 + self.layer.value_dim
                        ) // self.layer.tp_size
                        z_size = self.layer.value_dim // self.layer.tp_size
                        mixed_qkv, z_flat = projected_qkvz.split(
                            [qkv_size, z_size], dim=-1
                        )
                        z.copy_(
                            z_flat.reshape(
                                hidden_states.shape[0],
                                -1,
                                self.layer.head_v_dim,
                            )
                        )
                        b, a = projected_ba.chunk(2, dim=-1)
                        torch.ops.vllm.qwen_gdn_attention_core(
                            mixed_qkv,
                            b.contiguous(),
                            a.contiguous(),
                            core_attn_out,
                            layer_name=self.encoded_layer_name,
                            use_aiter=False,
                        )
                with output_timer:
                    return self.layer._output_projection(core_attn_out, z)

    def _reset_state(self) -> None:
        self.raw_state_pages.zero_()

    def _aggregate_time_samples(self, time_samples):
        """Take the per-iteration slowest rank for TP critical-path timing."""

        if self.tensor_parallel_size == 1:
            return time_samples
        torch = self.torch
        operator_names = sorted(time_samples)
        sample_counts = {len(time_samples[name]) for name in operator_names}
        if len(sample_counts) != 1:
            raise RuntimeError(
                f"GDN operators have inconsistent sample counts: {sample_counts}"
            )
        sample_matrix = torch.tensor(
            [time_samples[name] for name in operator_names],
            dtype=torch.float64,
            device="cuda",
        )
        torch.distributed.all_reduce(
            sample_matrix,
            op=torch.distributed.ReduceOp.MAX,
        )
        aggregated = sample_matrix.cpu().tolist()
        return dict(zip(operator_names, aggregated))

    def _validate_decomposition(
        self,
        hidden_states,
        profile_input: GDNProfileInput,
        metadata,
    ) -> None:
        torch = self.torch
        self._reset_state()
        expected = self._run_e2e(
            hidden_states,
            metadata,
            timed=False,
        )
        # Materialize the result before resetting the recurrent state. The
        # underlying Triton kernels are asynchronous and may reuse scratch
        # storage on a later invocation.
        torch.cuda.synchronize()
        expected = expected.clone()
        expected_state = tuple(state.clone() for state in self.layer.kv_cache)
        torch.cuda.synchronize()
        self._reset_state()
        actual = self._run_decomposed(
            hidden_states,
            metadata,
            profile_input.phase,
            timed=False,
        )
        torch.cuda.synchronize()
        actual = actual.clone()
        actual_state = tuple(state.clone() for state in self.layer.kv_cache)
        torch.cuda.synchronize()
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise AssertionError("GDN decomposition produced non-finite output")
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        for actual_tensor, expected_tensor in zip(actual_state, expected_state):
            if not torch.isfinite(expected_tensor).all() or not torch.isfinite(
                actual_tensor
            ).all():
                raise AssertionError("GDN decomposition produced non-finite state")
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                rtol=2e-2,
                atol=2e-2,
            )

    @staticmethod
    def _zero_stats(count: int) -> dict[str, float | int]:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "count": int(count),
        }

    def profile(
        self,
        profile_input: GDNProfileInput,
        *,
        warmup_iterations: int = 3,
        profile_iterations: int = 10,
    ) -> dict[str, Any]:
        if warmup_iterations < 0 or profile_iterations <= 0:
            raise ValueError("warmup_iterations must be >= 0 and profile_iterations > 0")
        torch = self.torch
        hidden_states = torch.randn(
            (profile_input.num_tokens, self.frontier_model_config.embedding_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        # Retain the metadata object until every asynchronous kernel using its
        # device-side pointer tables has completed. This also keeps the timed
        # region focused on layer execution instead of metadata construction.
        metadata = self._build_metadata(profile_input)
        # The first invocation may compile/autotune shape-specific Triton/FLA
        # kernels. Warm both paths before the numerical decomposition check so
        # compilation work and temporary autotuning buffers cannot masquerade
        # as a correctness mismatch.
        for _ in range(max(1, warmup_iterations)):
            self._reset_state()
            self._run_e2e(
                hidden_states,
                metadata,
                timed=False,
            )
            self._reset_state()
            self._run_decomposed(
                hidden_states,
                metadata,
                profile_input.phase,
                timed=False,
            )
        torch.cuda.synchronize()
        self._validate_decomposition(hidden_states, profile_input, metadata)
        torch.cuda.synchronize()
        self.timer_stats_store.clear_stats()

        for _ in range(profile_iterations):
            self._reset_state()
            self._run_e2e(
                hidden_states,
                metadata,
                timed=True,
            )
            self._reset_state()
            self._run_decomposed(
                hidden_states,
                metadata,
                profile_input.phase,
                timed=True,
            )
        torch.cuda.synchronize()
        time_samples = self.timer_stats_store.get_times()
        time_samples = self._aggregate_time_samples(time_samples)
        time_stats = self.timer_stats_store.get_stats_from_times(time_samples)
        for operator in GATED_DELTA_NET_FAMILY.operators:
            time_stats.setdefault(
                operator.profiling_name(),
                self._zero_stats(profile_iterations),
            )

        state_dtypes = self.layer.get_state_dtype()
        gdn_config = self.frontier_model_config.get_gdn_config()
        return {
            "measurement_type": profile_method_to_measurement_type(
                self.profile_method
            ).value,
            "model_architecture_profile": (
                self.frontier_model_config.get_model_architecture_profile().profile_id
            ),
            "quant_signature": self.frontier_model_config.get_quant_signature(),
            "device": self.device_name,
            "runtime_stack_signature": self.runtime_stack_signature,
            "gdn_runtime_backend": (
                "vllm_rocm_aiter_triton_dispatch"
                if self.use_aiter_dispatch
                else "vllm_rocm_standard_dispatch"
            ),
            "gdn_rank_aggregation": (
                "single_rank"
                if self.tensor_parallel_size == 1
                else "per_sample_rank_max"
            ),
            "gdn_prefill_backend": str(self.layer.gdn_prefill_backend),
            "gdn_decode_backend": (
                "packed_recurrent_triton"
                if self.layer.enable_packed_recurrent_decode
                and self.layer.gdn_decode_kernel == "triton"
                else str(self.layer.gdn_decode_kernel)
            ),
            "gqa_interleaved_layout": bool(self.layer.gqa_interleaved_layout),
            "packed_recurrent_decode": bool(
                self.layer.enable_packed_recurrent_decode
            ),
            "model_dtype": str(self.frontier_model_config.dtype),
            "conv_state_dtype": str(state_dtypes[0]).removeprefix("torch."),
            "recurrent_state_dtype": str(state_dtypes[1]).removeprefix("torch."),
            "weight_source": "synthetic_bf16",
            "num_tensor_parallel_workers": self.tensor_parallel_size,
            "hidden_size": self.frontier_model_config.embedding_dim,
            "conv_kernel_size": gdn_config.conv_kernel_size,
            "key_head_dim": gdn_config.key_head_dim,
            "value_head_dim": gdn_config.value_head_dim,
            "num_key_heads": gdn_config.num_key_heads,
            "num_value_heads": gdn_config.num_value_heads,
            "batch_size": profile_input.batch_size,
            "batch_num_tokens": profile_input.num_tokens,
            "batch_num_prefill_tokens": profile_input.num_prefill_tokens,
            "batch_num_decode_tokens": profile_input.num_decode_tokens,
            "max_query_len": profile_input.max_query_len,
            "has_initial_state": profile_input.has_initial_state,
            "query_lens": list(profile_input.query_lens),
            "context_lens": list(profile_input.context_lens),
            "time_stats": time_stats,
        }

    def close(self) -> None:
        if self._owns_distributed:
            self._destroy_model_parallel()
            if self.torch.distributed.is_initialized():
                self.torch.distributed.destroy_process_group()
            self._owns_distributed = False

    def __enter__(self) -> "VllmQwen35GDNWrapper":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
