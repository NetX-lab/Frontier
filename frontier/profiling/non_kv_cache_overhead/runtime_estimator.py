"""Runtime estimator for non-KV cache overhead bytes on one representative rank."""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from frontier.config import ReplicaConfig
from frontier.logger import init_logger
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_utils.parallel_state import (
    destroy_model_parallel,
    initialize_model_parallel,
)
from frontier.profiling.common.parallel_utils.tensor_parallel_utils import (
    get_padded_vocab_size,
)
from frontier.profiling.common.timer_stats_store import TimerStatsStore
from frontier.profiling.common.utils import (
    configure_quantization_manager_for_model_name,
    initialize_dummy_weights,
)
from frontier.profiling.common.layers.layernorm import RMSNorm
from frontier.profiling.common.layers.rotary_embedding import clear_rope_cache
from frontier.profiling.linear_op.linear_op_impl import (
    GPTBlock,
    GPTModel,
    VocabParallelEmbedding,
)
from frontier.profiling.non_kv_cache_overhead.memory_accounting import MemorySnapshot
from frontier.profiling.non_kv_cache_overhead.nccl_buffer_estimator import (
    NCCLBufferEstimationConfig,
    estimate_vllm_worker_non_torch_bytes,
    get_effective_nccl_buffer_config,
)
from frontier.profiling.non_kv_cache_overhead.runner import (
    SingleRankProfileInput,
    run_single_rank_profile,
)
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter

logger = init_logger(__name__)

_ALLOWED_WEIGHTS_MEMORY_SOURCES = {
    "param_counter",
    "runtime_model_load",
}
_MiB = 1024 * 1024

# Empirical torch peak padding to mimic vLLM profile_run runtime workspace peak.
_DEFAULT_TORCH_PEAK_PADDING_BYTES = 96 * _MiB
_QWEN_TORCH_PEAK_PADDING_BYTES = 112 * _MiB

_RUNTIME_NON_KV_CACHE_OVERHEAD_CACHE: Dict[Tuple[object, ...], "RuntimeNonKVProfileResult"] = {}
_RUNTIME_NON_KV_CACHE_OVERHEAD_CACHE_LOCK = threading.Lock()

_CUDART_HANDLE = None
_CUDART_HANDLE_LOCK = threading.Lock()


@dataclass(frozen=True)
class RuntimeNonKVProfileResult:
    """Runtime non-KV profiling result for one representative rank."""

    input_weights_memory_bytes: int
    measured_weights_memory_bytes: int
    overhead_bytes: int
    non_kv_cache_memory_bytes: int
    torch_peak_increase_bytes: int
    non_torch_increase_bytes: int
    total_memory_bytes: Optional[int] = None


@dataclass(frozen=True)
class _PipelineStageSlice:
    """Pipeline-stage-local layer placement for runtime profiling."""

    stage_idx: int
    num_pipeline_stages: int
    start_layer_idx: int
    end_layer_idx: int
    include_embed_tokens: bool
    include_final_norm: bool
    include_lm_head: bool


def _resolve_pipeline_stage_slice(
    *,
    num_layers: int,
    num_pipeline_stages: int,
    pipeline_stage_idx: int,
) -> _PipelineStageSlice:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got={num_layers!r}")
    if num_pipeline_stages <= 0:
        raise ValueError(
            "num_pipeline_stages must be > 0, "
            f"got={num_pipeline_stages!r}"
        )
    if pipeline_stage_idx < 0 or pipeline_stage_idx >= num_pipeline_stages:
        raise ValueError(
            "pipeline_stage_idx must satisfy "
            f"0 <= stage_idx < num_pipeline_stages, got stage_idx={pipeline_stage_idx}, "
            f"num_pipeline_stages={num_pipeline_stages}"
        )
    if num_layers % num_pipeline_stages != 0:
        raise ValueError(
            "num_layers must be divisible by num_pipeline_stages, "
            f"got num_layers={num_layers}, num_pipeline_stages={num_pipeline_stages}"
        )

    layers_per_stage = num_layers // num_pipeline_stages
    start_layer_idx = pipeline_stage_idx * layers_per_stage
    end_layer_idx = start_layer_idx + layers_per_stage
    return _PipelineStageSlice(
        stage_idx=int(pipeline_stage_idx),
        num_pipeline_stages=int(num_pipeline_stages),
        start_layer_idx=int(start_layer_idx),
        end_layer_idx=int(end_layer_idx),
        include_embed_tokens=(pipeline_stage_idx == 0),
        include_final_norm=(pipeline_stage_idx == num_pipeline_stages - 1),
        include_lm_head=(pipeline_stage_idx == num_pipeline_stages - 1),
    )


class _MoEExpertWeights(torch.nn.Module):
    """Weight-only module for MoE routed experts and router gate.

    Creates the correct number of expert weight parameters for memory accounting.
    Does not implement a functional forward pass — only holds weights.

    Expert structure (gated MLP per expert):
      up_proj:   Linear(embedding_dim, 2 * mlp_hidden_dim / moe_tp_size, bias=False)
      down_proj:  Linear(mlp_hidden_dim / moe_tp_size, embedding_dim, bias=False)

    Router gate:
      gate: Linear(embedding_dim, num_experts, bias=False)
    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_hidden_dim: int,
        num_experts: int,
        num_experts_per_device: int,
        moe_tp_size: int = 1,
        use_gated_mlp: bool = True,
    ):
        super().__init__()
        self.num_experts_per_device = num_experts_per_device

        # Router gate: embedding_dim → num_experts (not TP-sharded)
        self.gate = torch.nn.Linear(embedding_dim, num_experts, bias=False)

        # Expert weights: each expert has up_proj + down_proj
        sharded_mlp_dim = mlp_hidden_dim // moe_tp_size
        up_out_dim = 2 * sharded_mlp_dim if use_gated_mlp else sharded_mlp_dim
        experts = []
        for _ in range(num_experts_per_device):
            expert = torch.nn.ModuleDict({
                "up_proj": torch.nn.Linear(embedding_dim, up_out_dim, bias=False),
                "down_proj": torch.nn.Linear(sharded_mlp_dim, embedding_dim, bias=False),
            })
            experts.append(expert)
        self.experts = torch.nn.ModuleList(experts)

    def forward(self, hidden_states):
        # Weight-only module; forward is a no-op for profiling
        return hidden_states


class _MoESharedExpertWeights(torch.nn.Module):
    """Weight-only module for sparse-MoE shared-expert parameters."""

    def __init__(
        self,
        embedding_dim: int,
        share_expert_dim: int,
        tensor_parallel_size: int,
        use_gated_mlp: bool = True,
    ):
        super().__init__()
        if share_expert_dim <= 0:
            raise ValueError(
                f"share_expert_dim must be > 0, got={share_expert_dim!r}"
            )
        if tensor_parallel_size <= 0:
            raise ValueError(
                "tensor_parallel_size must be > 0 for shared expert weights, "
                f"got={tensor_parallel_size!r}"
            )
        if share_expert_dim % tensor_parallel_size != 0:
            raise ValueError(
                "share_expert_dim must be divisible by tensor_parallel_size, "
                f"share_expert_dim={share_expert_dim}, "
                f"tensor_parallel_size={tensor_parallel_size}"
            )

        sharded_share_expert_dim = int(share_expert_dim) // int(tensor_parallel_size)
        up_out_dim = (
            2 * sharded_share_expert_dim if use_gated_mlp else sharded_share_expert_dim
        )
        self.up_proj = torch.nn.Linear(embedding_dim, up_out_dim, bias=False)
        self.down_proj = torch.nn.Linear(
            sharded_share_expert_dim,
            embedding_dim,
            bias=False,
        )

    def forward(self, hidden_states):
        # Weight-only module; forward is a no-op for profiling
        return hidden_states


def _get_moe_layer_id_set(config: ModelConfig) -> set[int]:
    """Return the set of sparse-MoE layer IDs for a model config."""
    if not getattr(config, "is_moe", False):
        return set()
    if hasattr(config, "get_moe_layer_ids"):
        return {int(layer_id) for layer_id in config.get_moe_layer_ids()}
    return set(range(int(config.num_layers)))


def _supports_share_expert_weights(config: ModelConfig) -> bool:
    """Return whether sparse-MoE layers include shared-expert weights."""
    if not getattr(config, "is_moe", False):
        return False
    if hasattr(config, "supports_share_expert"):
        return bool(config.supports_share_expert())
    return int(getattr(config, "share_expert_dim", 0) or 0) > 0


def _build_sparse_moe_block_profiling_plan(
    profiling_plan: Optional[dict],
) -> dict:
    """Build a GPTBlock profiling plan for sparse-MoE weight accounting."""
    block_plan = dict(profiling_plan or {})
    block_plan.setdefault("ffn_enabled", True)
    block_plan["ffn_sharded_enabled"] = False
    return block_plan


class _FullStructureGPTModel(torch.nn.Module):
    """Layer-complete profiling model to mirror one-rank full model load scope.

    Mirrors vLLM's LlamaForCausalLM structure:
      model.embed_tokens  (VocabParallelEmbedding)
      model.layers[0..N]  (GPTBlock × num_layers)
      model.norm           (RMSNorm — final layer norm)
      lm_head              (VocabParallelEmbedding — only when tie_word_embeddings=False)

    For sparse-MoE models, additionally creates per-layer MoE weights without
    also constructing dense GPTBlock FFN weights:
      model.moe_expert_weights[i]         (_MoEExpertWeights)
      model.moe_shared_expert_weights[i]  (_MoESharedExpertWeights, if present)
    """

    def __init__(
        self,
        config: ModelConfig,
        world_size: int,
        pad_vocab_size: bool = False,
        profiling_plan: Optional[dict] = None,
        ep_size: int = 1,
        moe_tp_size: int = 1,
        num_pipeline_stages: int = 1,
        pipeline_stage_idx: int = 0,
        mtp_parameter_count: int = 0,
    ):
        super().__init__()

        num_layers = int(config.num_layers)
        if num_layers <= 0:
            raise ValueError(
                "config.num_layers must be > 0 for full-structure model, "
                f"got={config.num_layers!r}"
            )
        stage_slice = _resolve_pipeline_stage_slice(
            num_layers=num_layers,
            num_pipeline_stages=int(num_pipeline_stages),
            pipeline_stage_idx=int(pipeline_stage_idx),
        )
        self._stage_slice = stage_slice
        self._embedding_dim = int(config.embedding_dim)
        self._hidden_states_dtype = config.dtype

        enabled_ops = (
            set(profiling_plan.get("enabled_ops", [])) if profiling_plan else None
        )
        self._profile_emb = (
            stage_slice.include_embed_tokens
            and (True if enabled_ops is None else "emb" in enabled_ops)
        )

        if stage_slice.include_embed_tokens:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.embedding_dim,
                linear_metric_name="emb" if self._profile_emb else None,
                reduce_results=False,
                world_size=world_size,
                rank=0,
                pad_vocab_size=pad_vocab_size,
            )
        else:
            self.embed_tokens = None

        moe_layer_ids = _get_moe_layer_id_set(config)
        sparse_moe_block_plan = (
            _build_sparse_moe_block_profiling_plan(profiling_plan)
            if moe_layer_ids
            else None
        )
        layers = []
        for layer_id in range(
            int(stage_slice.start_layer_idx),
            int(stage_slice.end_layer_idx),
        ):
            block_profiling_plan = (
                sparse_moe_block_plan
                if int(layer_id) in moe_layer_ids
                else profiling_plan
            )
            layers.append(
                GPTBlock(
                    config,
                    world_size=world_size,
                    profiling_plan=block_profiling_plan,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

        # Final layer norm (mirrors model.norm in vLLM)
        if stage_slice.include_final_norm:
            self.final_norm = RMSNorm(
                config.embedding_dim, eps=config.rms_norm_eps
            )
        else:
            self.final_norm = None

        # LM head: separate weight when tie_word_embeddings is False
        tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        if stage_slice.include_lm_head and not tie_word_embeddings:
            self.lm_head = VocabParallelEmbedding(
                config.vocab_size,
                config.embedding_dim,
                linear_metric_name=None,
                reduce_results=False,
                world_size=world_size,
                rank=0,
                pad_vocab_size=pad_vocab_size,
            )
        else:
            self.lm_head = None

        if stage_slice.include_lm_head and int(mtp_parameter_count) > 0:
            self.mtp_parameter_reservoir = torch.nn.Parameter(
                torch.empty(int(mtp_parameter_count), dtype=torch.get_default_dtype()),
                requires_grad=False,
            )
        else:
            self.register_parameter("mtp_parameter_reservoir", None)

        # MoE routed expert weights (for MoE layers only)
        self.moe_expert_weights = self._build_moe_expert_weights(
            config,
            ep_size=ep_size,
            moe_tp_size=moe_tp_size,
            start_layer_idx=int(stage_slice.start_layer_idx),
            end_layer_idx=int(stage_slice.end_layer_idx),
        )
        self.moe_shared_expert_weights = self._build_moe_shared_expert_weights(
            config,
            tensor_parallel_size=world_size,
            start_layer_idx=int(stage_slice.start_layer_idx),
            end_layer_idx=int(stage_slice.end_layer_idx),
        )

    @staticmethod
    def _build_moe_expert_weights(
        config: ModelConfig,
        ep_size: int,
        moe_tp_size: int,
        start_layer_idx: int,
        end_layer_idx: int,
    ) -> Optional[torch.nn.ModuleList]:
        """Build routed expert weight modules for MoE layers.

        Returns None for dense models. For MoE models, returns a ModuleList
        with one _MoEExpertWeights per MoE layer.
        """
        if not getattr(config, "is_moe", False):
            return None

        num_experts = int(getattr(config, "num_experts", 0))
        if num_experts <= 0:
            return None

        if ep_size <= 0:
            raise ValueError(f"ep_size must be > 0, got={ep_size!r}")
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by "
                f"ep_size ({ep_size})"
            )
        num_experts_per_device = num_experts // ep_size

        moe_layer_ids = _get_moe_layer_id_set(config)
        if not moe_layer_ids:
            return None

        local_moe_layer_ids = [
            int(layer_id)
            for layer_id in moe_layer_ids
            if int(start_layer_idx) <= int(layer_id) < int(end_layer_idx)
        ]
        if not local_moe_layer_ids:
            return None

        embedding_dim = int(config.embedding_dim)
        mlp_hidden_dim = int(config.mlp_hidden_dim)
        use_gated_mlp = bool(getattr(config, "use_gated_mlp", True))

        expert_modules = []
        for _ in local_moe_layer_ids:
            expert_modules.append(
                _MoEExpertWeights(
                    embedding_dim=embedding_dim,
                    mlp_hidden_dim=mlp_hidden_dim,
                    num_experts=num_experts,
                    num_experts_per_device=num_experts_per_device,
                    moe_tp_size=moe_tp_size,
                    use_gated_mlp=use_gated_mlp,
                )
            )
        return torch.nn.ModuleList(expert_modules)

    @staticmethod
    def _build_moe_shared_expert_weights(
        config: ModelConfig,
        tensor_parallel_size: int,
        start_layer_idx: int,
        end_layer_idx: int,
    ) -> Optional[torch.nn.ModuleList]:
        """Build shared-expert weight modules for sparse-MoE layers."""
        if not _supports_share_expert_weights(config):
            return None

        moe_layer_ids = _get_moe_layer_id_set(config)
        if not moe_layer_ids:
            return None

        local_moe_layer_ids = [
            int(layer_id)
            for layer_id in moe_layer_ids
            if int(start_layer_idx) <= int(layer_id) < int(end_layer_idx)
        ]
        if not local_moe_layer_ids:
            return None

        embedding_dim = int(config.embedding_dim)
        share_expert_dim = int(getattr(config, "share_expert_dim", 0) or 0)
        use_gated_mlp = bool(getattr(config, "use_gated_mlp", True))

        shared_expert_modules = []
        for _ in local_moe_layer_ids:
            shared_expert_modules.append(
                _MoESharedExpertWeights(
                    embedding_dim=embedding_dim,
                    share_expert_dim=share_expert_dim,
                    tensor_parallel_size=int(tensor_parallel_size),
                    use_gated_mlp=use_gated_mlp,
                )
            )
        return torch.nn.ModuleList(shared_expert_modules)

    def forward(self, input_ids, positions):
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = torch.zeros(
                (int(input_ids.shape[0]), int(self._embedding_dim)),
                device=input_ids.device,
                dtype=self._hidden_states_dtype,
            )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        if self.lm_head is not None:
            hidden_states = self.lm_head(input_ids)
        return hidden_states


class _CudaNonTorchAllocation:
    """Explicit CUDA allocation not tracked by torch allocator."""

    def __init__(self, size_bytes: int):
        self._size_bytes = int(size_bytes)
        self._ptr = ctypes.c_void_p()

    def allocate(self) -> None:
        if self._size_bytes <= 0:
            return

        cudart = _get_cudart_handle()
        ret = cudart.cudaMalloc(ctypes.byref(self._ptr), ctypes.c_size_t(self._size_bytes))
        if ret != 0 or self._ptr.value is None:
            raise RuntimeError(
                "cudaMalloc failed for runtime auxiliary non-torch allocation, "
                f"size_bytes={self._size_bytes}, cuda_error={ret}"
            )
        torch.cuda.synchronize()

    def free(self) -> None:
        if self._ptr.value is None:
            return

        cudart = _get_cudart_handle()
        ret = cudart.cudaFree(self._ptr)
        if ret != 0:
            raise RuntimeError(
                "cudaFree failed for runtime auxiliary non-torch allocation, "
                f"size_bytes={self._size_bytes}, cuda_error={ret}"
            )
        self._ptr = ctypes.c_void_p()
        torch.cuda.synchronize()


def _validate_weights_memory_source(weights_memory_source: str) -> str:
    normalized = str(weights_memory_source)
    if normalized not in _ALLOWED_WEIGHTS_MEMORY_SOURCES:
        raise ValueError(
            "weights_memory_source must be one of "
            f"{sorted(_ALLOWED_WEIGHTS_MEMORY_SOURCES)}, got={weights_memory_source!r}"
        )
    return normalized


def _build_runtime_profile_model(
    *,
    profiling_model_config: ModelConfig,
    tp_size: int,
    pad_vocab_size: bool,
    weights_memory_source: str,
    ep_size: int = 1,
    moe_tp_size: int = 1,
    num_pipeline_stages: int = 1,
    pipeline_stage_idx: int = 0,
    mtp_parameter_count: int = 0,
) -> torch.nn.Module:
    """Build profiling model according to the selected weights-memory semantics."""
    normalized_source = _validate_weights_memory_source(weights_memory_source)
    if normalized_source == "runtime_model_load":
        return _FullStructureGPTModel(
            profiling_model_config,
            world_size=tp_size,
            pad_vocab_size=pad_vocab_size,
            ep_size=ep_size,
            moe_tp_size=moe_tp_size,
            num_pipeline_stages=num_pipeline_stages,
            pipeline_stage_idx=pipeline_stage_idx,
            mtp_parameter_count=mtp_parameter_count,
        )
    return GPTModel(
        profiling_model_config,
        world_size=tp_size,
        num_repeat_steps=1,
        pad_vocab_size=pad_vocab_size,
    )


def _get_vllm_style_current_memory_usage_bytes() -> int:
    """Match vLLM DeviceMemoryProfiler CUDA semantics for current memory usage."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.max_memory_allocated())


def _get_cudart_handle():
    global _CUDART_HANDLE

    with _CUDART_HANDLE_LOCK:
        if _CUDART_HANDLE is not None:
            return _CUDART_HANDLE

        library_path = ctypes.util.find_library("cudart")
        if not library_path:
            library_path = "libcudart.so"

        try:
            handle = ctypes.CDLL(library_path)
        except OSError as exc:
            raise RuntimeError(
                "Failed to load libcudart for runtime non-torch auxiliary allocation"
            ) from exc

        if not hasattr(handle, "cudaMalloc") or not hasattr(handle, "cudaFree"):
            raise RuntimeError(
                "Loaded libcudart does not expose cudaMalloc/cudaFree; cannot "
                "simulate TP auxiliary non-torch allocation"
            )

        handle.cudaMalloc.restype = ctypes.c_int
        handle.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        handle.cudaFree.restype = ctypes.c_int
        handle.cudaFree.argtypes = [ctypes.c_void_p]

        _CUDART_HANDLE = handle
        return _CUDART_HANDLE



def _estimate_torch_peak_padding_bytes(model_name: str) -> int:
    normalized = str(model_name).lower()
    if "qwen" in normalized:
        return int(_QWEN_TORCH_PEAK_PADDING_BYTES)
    return int(_DEFAULT_TORCH_PEAK_PADDING_BYTES)


def _measure_runtime_model_load_bytes(
    *,
    model_loader: Callable[[], torch.nn.Module],
) -> Tuple[torch.nn.Module, int, int, int]:
    """Measure model-load memory delta with vLLM-style current-memory snapshots."""
    before_model_load_allocated_bytes = _get_vllm_style_current_memory_usage_bytes()
    model = model_loader()
    after_model_load_allocated_bytes = _get_vllm_style_current_memory_usage_bytes()

    profile_weights_memory_bytes = (
        after_model_load_allocated_bytes - before_model_load_allocated_bytes
    )
    if profile_weights_memory_bytes <= 0:
        raise RuntimeError(
            "Runtime model-load weights memory measurement must be > 0, "
            f"got={profile_weights_memory_bytes}"
        )

    return (
        model,
        int(profile_weights_memory_bytes),
        int(before_model_load_allocated_bytes),
        int(after_model_load_allocated_bytes),
    )


def _get_current_cuda_total_memory_cache_key() -> object:
    """Return CUDA total memory for runtime profile cache partitioning."""
    try:
        if not torch.cuda.is_available():
            return "cuda_unavailable"
        current_device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(current_device)
        return int(properties.total_memory)
    except Exception:
        return "cuda_total_memory_unknown"


def _build_cache_key(
    *,
    replica_config: ReplicaConfig,
    cluster_type: ClusterType,
    max_num_batched_tokens: int,
    weights_memory_bytes: int,
    weights_memory_source: str,
    nccl_buffer_config: Optional["NCCLBufferEstimationConfig"] = None,
) -> Tuple[object, ...]:
    model_config = replica_config.model_config
    model_name = model_config.get_name()

    base_key = (
        cluster_type.name,
        model_name,
        int(model_config.num_layers),
        int(model_config.num_q_heads),
        int(model_config.num_kv_heads),
        int(model_config.embedding_dim),
        int(model_config.mlp_hidden_dim),
        int(model_config.get_head_dim()),
        int(replica_config.attn_tensor_parallel_size),
        int(getattr(replica_config, "attn_data_parallel_size", 1)),
        int(getattr(replica_config, "moe_tensor_parallel_size", 1)),
        int(getattr(replica_config, "moe_expert_parallel_size", 1)),
        int(replica_config.num_pipeline_stages),
        int(max_num_batched_tokens),
        int(weights_memory_bytes),
        str(weights_memory_source),
        str(getattr(model_config, "torch_dtype", getattr(model_config, "dtype", "float16"))),
        str(getattr(getattr(replica_config, "speculative_decoding_config", None), "method", "")),
        str(getattr(getattr(replica_config, "speculative_decoding_config", None), "spec_model_name", "")),
        int(getattr(getattr(replica_config, "speculative_decoding_config", None), "mtp_n_predict", 0)),
        int(getattr(getattr(replica_config, "speculative_decoding_config", None), "mtp_num_layers", 0)),
        _get_current_cuda_total_memory_cache_key(),
    )

    effective_nccl_config = get_effective_nccl_buffer_config(nccl_buffer_config)
    return base_key + effective_nccl_config.cache_fingerprint()


def _profile_non_kv_cache_overhead_bytes_uncached(
    *,
    replica_config: ReplicaConfig,
    cluster_type: ClusterType,
    max_num_batched_tokens: int,
    weights_memory_bytes: int,
    weights_memory_source: str,
    nccl_buffer_config: Optional[NCCLBufferEstimationConfig] = None,
) -> RuntimeNonKVProfileResult:
    if cluster_type == ClusterType.DECODE_FFN:
        raise ValueError(
            "Runtime non-KV cache overhead profiling is not supported for "
            "DECODE_FFN cluster because it does not allocate KV cache blocks"
        )

    if max_num_batched_tokens <= 0:
        raise ValueError(
            "max_num_batched_tokens must be > 0, "
            f"got={max_num_batched_tokens!r}"
        )

    if weights_memory_bytes < 0:
        raise ValueError(
            "weights_memory_bytes must be >= 0, "
            f"got={weights_memory_bytes!r}"
        )

    weights_memory_source = _validate_weights_memory_source(weights_memory_source)

    tp_size = int(replica_config.attn_tensor_parallel_size)
    if tp_size <= 0:
        raise ValueError(
            "replica_config.attn_tensor_parallel_size must be > 0, "
            f"got={replica_config.attn_tensor_parallel_size!r}"
        )

    ep_size = int(getattr(replica_config, "moe_expert_parallel_size", 1))
    if ep_size <= 0:
        ep_size = 1
    dp_size = int(getattr(replica_config, "attn_data_parallel_size", 1))
    if dp_size <= 0:
        dp_size = 1
    moe_tp_size = int(getattr(replica_config, "moe_tensor_parallel_size", 1))
    if moe_tp_size <= 0:
        moe_tp_size = 1

    num_pipeline_stages = int(getattr(replica_config, "num_pipeline_stages", 1))
    if num_pipeline_stages <= 0:
        raise ValueError(
            "replica_config.num_pipeline_stages must be > 0, "
            f"got={getattr(replica_config, 'num_pipeline_stages', None)!r}"
        )

    model_name = replica_config.model_config.get_name()
    profiling_model_config = ModelConfig.from_model_name(model_name)
    mtp_parameter_count = int(
        ParamCounter(replica_config=replica_config, cluster_type=cluster_type)
        .get_num_mtp_parameters_per_device()
    )

    if int(profiling_model_config.num_layers) % int(num_pipeline_stages) != 0:
        raise ValueError(
            "profiling_model_config.num_layers must be divisible by num_pipeline_stages, "
            f"got num_layers={profiling_model_config.num_layers}, "
            f"num_pipeline_stages={num_pipeline_stages}"
        )

    def _profile_single_pipeline_stage(
        *, pipeline_stage_idx: int
    ) -> RuntimeNonKVProfileResult:
        baseline_snapshot = MemorySnapshot()
        model = None
        input_ids = None
        positions = None
        non_torch_aux_allocation = None
        previous_fp8_gemm_surrogate = os.environ.get("FRONTIER_FP8_GEMM_SURROGATE")

        try:
            clear_rope_cache()
            initialize_model_parallel(tensor_model_parallel_size=tp_size)
            configure_quantization_manager_for_model_name(profiling_model_config.name)
            if profiling_model_config.quantization_config is not None:
                os.environ["FRONTIER_FP8_GEMM_SURROGATE"] = "1"
                logger.info(
                    "[FRONTIER_RUNTIME_FP8_GEMM_SURROGATE] model=%s, tp=%s, pp=%s, "
                    "stage_idx=%s, reason=runtime_non_kv_memory_probe",
                    replica_config.model_config.get_name(),
                    tp_size,
                    num_pipeline_stages,
                    pipeline_stage_idx,
                )

            # Ensure CudaTimer can reuse a singleton TimerStatsStore instance
            # without requiring per-layer constructor arguments.
            timer_store = TimerStatsStore(profile_method="cuda_event", disabled=True)
            timer_store.clear_stats()

            pad_vocab_size = profiling_model_config.vocab_size % tp_size != 0

            def _load_profile_model() -> torch.nn.Module:
                loaded_model = _build_runtime_profile_model(
                    profiling_model_config=profiling_model_config,
                    tp_size=tp_size,
                    pad_vocab_size=pad_vocab_size,
                    weights_memory_source=weights_memory_source,
                    ep_size=ep_size,
                    moe_tp_size=moe_tp_size,
                    num_pipeline_stages=num_pipeline_stages,
                    pipeline_stage_idx=int(pipeline_stage_idx),
                    mtp_parameter_count=mtp_parameter_count,
                )
                initialize_dummy_weights(loaded_model)
                loaded_model = loaded_model.to(dtype=profiling_model_config.dtype).cuda().eval()
                return loaded_model

            profile_weights_memory_bytes = int(weights_memory_bytes)
            if weights_memory_source == "runtime_model_load":
                (
                    model,
                    profile_weights_memory_bytes,
                    before_model_load_allocated_bytes,
                    after_model_load_allocated_bytes,
                ) = _measure_runtime_model_load_bytes(model_loader=_load_profile_model)
                logger.info(
                    "[FRONTIER_RUNTIME_MODEL_LOAD] model=%s, tp=%s, pp=%s, stage_idx=%s, "
                    "before_model_load_allocated_bytes=%s, "
                    "after_model_load_allocated_bytes=%s, measured_weights_memory_bytes=%s",
                    replica_config.model_config.get_name(),
                    tp_size,
                    num_pipeline_stages,
                    pipeline_stage_idx,
                    before_model_load_allocated_bytes,
                    after_model_load_allocated_bytes,
                    profile_weights_memory_bytes,
                )
            else:
                model = _load_profile_model()

            nccl_estimate = estimate_vllm_worker_non_torch_bytes(
                tp_size=tp_size,
                pp_size=num_pipeline_stages,
                dp_size=dp_size,
                ep_size=ep_size,
                pipeline_stage_idx=int(pipeline_stage_idx),
                is_moe=bool(getattr(replica_config.model_config, "is_moe", False)),
                config=nccl_buffer_config,
            )
            aux_non_torch_bytes = nccl_estimate.total_bytes
            if aux_non_torch_bytes > 0:
                non_torch_aux_allocation = _CudaNonTorchAllocation(aux_non_torch_bytes)
                non_torch_aux_allocation.allocate()
                logger.info(
                    "[FRONTIER_RUNTIME_AUX_NON_TORCH_ALLOC] model=%s, tp=%s, pp=%s, dp=%s, ep=%s, stage_idx=%s, "
                    "nccl_channel_bytes=%s, nccl_comm_overhead_bytes=%s, "
                    "custom_ar_bytes=%s, tp_nccl_bytes=%s, "
                    "vllm_worker_base_extra_bytes=%s, pp_final_stage_extra_bytes=%s, "
                    "dp_communicator_extra_bytes=%s, ep_all2all_extra_bytes=%s, "
                    "total_aux_non_torch_bytes=%s",
                    replica_config.model_config.get_name(),
                    tp_size,
                    num_pipeline_stages,
                    dp_size,
                    ep_size,
                    pipeline_stage_idx,
                    nccl_estimate.nccl_channel_bytes,
                    nccl_estimate.nccl_comm_overhead_bytes,
                    nccl_estimate.custom_ar_bytes,
                    nccl_estimate.tp_nccl_bytes,
                    nccl_estimate.vllm_worker_base_extra_bytes,
                    nccl_estimate.pp_final_stage_extra_bytes,
                    nccl_estimate.dp_communicator_extra_bytes,
                    nccl_estimate.ep_all2all_extra_bytes,
                    aux_non_torch_bytes,
                )

            torch_peak_padding_bytes = _estimate_torch_peak_padding_bytes(model_name)

            num_profile_tokens = min(
                int(max_num_batched_tokens),
                int(profiling_model_config.max_position_embeddings),
            )
            if num_profile_tokens <= 0:
                raise ValueError(
                    "Derived num_profile_tokens must be > 0, "
                    f"got={num_profile_tokens!r}"
                )

            padded_vocab_size = (
                get_padded_vocab_size(profiling_model_config.vocab_size, tp_size)
                if pad_vocab_size
                else profiling_model_config.vocab_size
            )
            vocab_range = padded_vocab_size // tp_size
            if vocab_range <= 0:
                raise ValueError(
                    "Derived vocab_range must be > 0, "
                    f"got={vocab_range!r}"
                )

            input_ids = torch.randint(
                low=0,
                high=vocab_range,
                size=(num_profile_tokens,),
                device="cuda",
                dtype=torch.long,
            )
            positions = torch.arange(
                num_profile_tokens,
                device="cuda",
                dtype=torch.long,
            )

            @torch.inference_mode()
            def _profile_run_callback() -> None:
                torch_peak_padding = None
                previous_fp8_surrogate = os.environ.get("FRONTIER_FP8_GEMM_SURROGATE")
                os.environ["FRONTIER_FP8_GEMM_SURROGATE"] = "1"
                try:
                    if torch_peak_padding_bytes > 0:
                        torch_peak_padding = torch.empty(
                            (torch_peak_padding_bytes,),
                            dtype=torch.uint8,
                            device="cuda",
                        )
                        torch_peak_padding.fill_(1)

                    model(input_ids, positions)

                    if torch_peak_padding is not None:
                        _ = int(torch_peak_padding[0].item())
                        del torch_peak_padding
                finally:
                    if previous_fp8_surrogate is None:
                        os.environ.pop("FRONTIER_FP8_GEMM_SURROGATE", None)
                    else:
                        os.environ["FRONTIER_FP8_GEMM_SURROGATE"] = previous_fp8_surrogate

                torch.cuda.synchronize()

            profile_output = run_single_rank_profile(
                SingleRankProfileInput(
                    profile_run_callback=_profile_run_callback,
                    weights_memory_bytes=profile_weights_memory_bytes,
                    baseline_snapshot=baseline_snapshot,
                )
            )

            breakdown = profile_output.breakdown
            non_kv_cache_memory_bytes = int(breakdown.non_kv_cache_memory_bytes)
            overhead_bytes = non_kv_cache_memory_bytes - int(profile_weights_memory_bytes)
            if overhead_bytes < 0:
                raise RuntimeError(
                    "Profiled non-KV overhead became negative after removing weights, "
                    f"non_kv_cache_memory_bytes={non_kv_cache_memory_bytes}, "
                    f"profile_weights_memory_bytes={profile_weights_memory_bytes}"
                )

            logger.info(
                "[FRONTIER_RUNTIME_NON_KV_BREAKDOWN] cluster_type=%s, model=%s, tp=%s, pp=%s, "
                "stage_idx=%s, max_num_batched_tokens=%s, weights_memory_source=%s, "
                "input_weights_memory_bytes=%s, non_kv_cache_memory_bytes=%s, "
                "weights_memory_bytes=%s, torch_peak_increase_bytes=%s, "
                "non_torch_increase_bytes=%s, overhead_bytes=%s",
                cluster_type.name,
                replica_config.model_config.get_name(),
                tp_size,
                num_pipeline_stages,
                pipeline_stage_idx,
                max_num_batched_tokens,
                weights_memory_source,
                int(weights_memory_bytes),
                non_kv_cache_memory_bytes,
                int(breakdown.weights_memory_bytes),
                int(breakdown.torch_peak_increase_bytes),
                int(breakdown.non_torch_increase_bytes),
                overhead_bytes,
            )

            expected_non_kv = (
                int(profile_weights_memory_bytes)
                + int(breakdown.torch_peak_increase_bytes)
                + int(breakdown.non_torch_increase_bytes)
            )
            if non_kv_cache_memory_bytes != expected_non_kv:
                raise RuntimeError(
                    "non_kv invariant violated: "
                    f"non_kv={non_kv_cache_memory_bytes} != "
                    f"weights({profile_weights_memory_bytes}) + "
                    f"torch_peak({breakdown.torch_peak_increase_bytes}) + "
                    f"non_torch({breakdown.non_torch_increase_bytes})"
                )

            return RuntimeNonKVProfileResult(
                input_weights_memory_bytes=int(weights_memory_bytes),
                measured_weights_memory_bytes=int(profile_weights_memory_bytes),
                overhead_bytes=int(overhead_bytes),
                non_kv_cache_memory_bytes=int(non_kv_cache_memory_bytes),
                torch_peak_increase_bytes=int(breakdown.torch_peak_increase_bytes),
                non_torch_increase_bytes=int(breakdown.non_torch_increase_bytes),
                total_memory_bytes=int(baseline_snapshot.total_memory),
            )
        finally:
            if non_torch_aux_allocation is not None:
                non_torch_aux_allocation.free()

            del input_ids
            del positions
            del model
            gc.collect()
            if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            clear_rope_cache()
            destroy_model_parallel()
            if previous_fp8_gemm_surrogate is None:
                os.environ.pop("FRONTIER_FP8_GEMM_SURROGATE", None)
            else:
                os.environ["FRONTIER_FP8_GEMM_SURROGATE"] = previous_fp8_gemm_surrogate

    if num_pipeline_stages == 1:
        return _profile_single_pipeline_stage(pipeline_stage_idx=0)

    stage_results: List[Tuple[int, RuntimeNonKVProfileResult]] = []
    for stage_idx in range(num_pipeline_stages):
        stage_result = _profile_single_pipeline_stage(pipeline_stage_idx=stage_idx)
        stage_results.append((int(stage_idx), stage_result))

    selected_stage_idx, selected_result = max(
        stage_results,
        key=lambda item: (
            int(item[1].non_kv_cache_memory_bytes),
            int(item[1].measured_weights_memory_bytes),
            -int(item[0]),
        ),
    )
    logger.info(
        "[FRONTIER_RUNTIME_PP_STAGE_SELECTED] cluster_type=%s, model=%s, tp=%s, pp=%s, "
        "selected_stage_idx=%s, selected_non_kv_cache_memory_bytes=%s, "
        "selected_measured_weights_memory_bytes=%s, per_stage_non_kv_cache_memory_bytes=%s",
        cluster_type.name,
        replica_config.model_config.get_name(),
        tp_size,
        num_pipeline_stages,
        selected_stage_idx,
        int(selected_result.non_kv_cache_memory_bytes),
        int(selected_result.measured_weights_memory_bytes),
        {int(idx): int(result.non_kv_cache_memory_bytes) for idx, result in stage_results},
    )
    return selected_result


def _normalize_profile_result(
    *,
    profiled,
    input_weights_memory_bytes: int,
) -> RuntimeNonKVProfileResult:
    if isinstance(profiled, RuntimeNonKVProfileResult):
        return profiled

    if isinstance(profiled, int):
        overhead_bytes = int(profiled)
        if overhead_bytes < 0:
            raise ValueError(
                "profiled overhead must be >= 0 when using integer compatibility mode, "
                f"got={overhead_bytes!r}"
            )
        return RuntimeNonKVProfileResult(
            input_weights_memory_bytes=int(input_weights_memory_bytes),
            measured_weights_memory_bytes=int(input_weights_memory_bytes),
            overhead_bytes=overhead_bytes,
            non_kv_cache_memory_bytes=int(input_weights_memory_bytes) + overhead_bytes,
            torch_peak_increase_bytes=0,
            non_torch_increase_bytes=0,
            total_memory_bytes=None,
        )

    raise TypeError(
        "profiled result must be RuntimeNonKVProfileResult or int, "
        f"got={type(profiled)}"
    )


def estimate_non_kv_cache_profile(
    *,
    replica_config: ReplicaConfig,
    cluster_type: ClusterType,
    max_num_batched_tokens: int,
    weights_memory_bytes: int,
    weights_memory_source: str = "param_counter",
    nccl_buffer_config: Optional[NCCLBufferEstimationConfig] = None,
) -> RuntimeNonKVProfileResult:
    """Estimate runtime non-KV profile with per-config caching."""
    if max_num_batched_tokens <= 0:
        raise ValueError(
            "max_num_batched_tokens must be > 0, "
            f"got={max_num_batched_tokens!r}"
        )

    if weights_memory_bytes < 0:
        raise ValueError(
            "weights_memory_bytes must be >= 0, "
            f"got={weights_memory_bytes!r}"
        )

    weights_memory_source = _validate_weights_memory_source(weights_memory_source)

    cache_key = _build_cache_key(
        replica_config=replica_config,
        cluster_type=cluster_type,
        max_num_batched_tokens=max_num_batched_tokens,
        weights_memory_bytes=weights_memory_bytes,
        weights_memory_source=weights_memory_source,
        nccl_buffer_config=nccl_buffer_config,
    )

    with _RUNTIME_NON_KV_CACHE_OVERHEAD_CACHE_LOCK:
        cached = _RUNTIME_NON_KV_CACHE_OVERHEAD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    profiled = _profile_non_kv_cache_overhead_bytes_uncached(
        replica_config=replica_config,
        cluster_type=cluster_type,
        max_num_batched_tokens=max_num_batched_tokens,
        weights_memory_bytes=weights_memory_bytes,
        weights_memory_source=weights_memory_source,
        nccl_buffer_config=nccl_buffer_config,
    )

    normalized_profiled = _normalize_profile_result(
        profiled=profiled,
        input_weights_memory_bytes=weights_memory_bytes,
    )

    with _RUNTIME_NON_KV_CACHE_OVERHEAD_CACHE_LOCK:
        existing = _RUNTIME_NON_KV_CACHE_OVERHEAD_CACHE.get(cache_key)
        if existing is not None:
            return existing
        _RUNTIME_NON_KV_CACHE_OVERHEAD_CACHE[cache_key] = normalized_profiled

    logger.info(
        "[RUNTIME_NON_KV_PROFILE] cluster_type=%s, model=%s, tp=%s, "
        "max_num_batched_tokens=%s, weights_memory_source=%s, "
        "input_weights_memory_bytes=%s, measured_weights_memory_bytes=%s, "
        "overhead_bytes=%s",
        cluster_type.name,
        replica_config.model_config.get_name(),
        replica_config.attn_tensor_parallel_size,
        max_num_batched_tokens,
        weights_memory_source,
        weights_memory_bytes,
        int(normalized_profiled.measured_weights_memory_bytes),
        int(normalized_profiled.overhead_bytes),
    )

    return normalized_profiled


def estimate_non_kv_cache_overhead_bytes(
    *,
    replica_config: ReplicaConfig,
    cluster_type: ClusterType,
    max_num_batched_tokens: int,
    weights_memory_bytes: int,
    weights_memory_source: str = "param_counter",
    nccl_buffer_config: Optional[NCCLBufferEstimationConfig] = None,
) -> int:
    """Estimate non-KV cache overhead bytes with per-config result caching."""
    profile_result = estimate_non_kv_cache_profile(
        replica_config=replica_config,
        cluster_type=cluster_type,
        max_num_batched_tokens=max_num_batched_tokens,
        weights_memory_bytes=weights_memory_bytes,
        weights_memory_source=weights_memory_source,
        nccl_buffer_config=nccl_buffer_config,
    )
    return int(profile_result.overhead_bytes)


def clear_runtime_non_kv_cache_overhead_cache() -> None:
    """Clear runtime overhead cache, primarily for tests."""
    with _RUNTIME_NON_KV_CACHE_OVERHEAD_CACHE_LOCK:
        _RUNTIME_NON_KV_CACHE_OVERHEAD_CACHE.clear()
