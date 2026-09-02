from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontier.profiling.common.layers.activation import SiluAndMul
from frontier.profiling.common.layers.layernorm import GemmaRMSNorm, RMSNorm
from frontier.profiling.common.layers.rotary_embedding import get_rope
from frontier.profiling.common.parallel_utils.tensor_parallel_layers import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from frontier.profiling.common.parallel_utils.tensor_parallel_utils import (
    get_padded_vocab_size,
)

from frontier.profiling.common.cuda_timer import CudaTimer
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.utils import raise_if_fp8_requested
from frontier.profiling.linear_op.profiling_plan import memory_operator_enabled
from frontier.model_architectures import (
    LinearAttentionImplementation,
    LayerKind,
    ResolvedLayerContract,
    get_model_architecture_profile,
)
from frontier.operators.typed_contracts import typed_metadata_value_matches

REUSE_MEMORY = True


def _resolve_num_kv_heads_per_worker(num_kv_heads: int, world_size: int) -> int:
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")

    if num_kv_heads >= world_size:
        if num_kv_heads % world_size != 0:
            raise ValueError(
                "KV heads must be divisible by world_size when KV heads are partitioned. "
                f"num_kv_heads={num_kv_heads}, world_size={world_size}"
            )
        return num_kv_heads // world_size

    if world_size % num_kv_heads != 0:
        raise ValueError(
            "KV head replication requires world_size to be divisible by num_kv_heads. "
            f"num_kv_heads={num_kv_heads}, world_size={world_size}"
        )
    return 1


def _supports_share_expert(config: ModelConfig) -> bool:
    if hasattr(config, "supports_share_expert"):
        return bool(config.supports_share_expert())
    return bool(
        _get_architecture_profile(config).supports_share_expert(config)
    )


def _get_architecture_profile(config: ModelConfig):
    """Resolve the architecture profile through the config-owned API."""

    profile_getter = getattr(config, "get_model_architecture_profile", None)
    if callable(profile_getter):
        return profile_getter()
    return get_model_architecture_profile(config)


def _has_active_layer_kind(config: ModelConfig, layer_kind: LayerKind) -> bool:
    """Return whether the profile materializes a typed layer domain."""

    profile = _get_architecture_profile(config)
    return any(
        contract.layer_kind is layer_kind
        for contract in profile.iter_active_layer_contracts(config)
    )


def _typed_operator_is_selected(
    profiling_plan: Optional[dict], operator_name: str, world_size: int
) -> bool:
    """Check a plan's typed selection without weakening legacy plans."""

    if profiling_plan is None or "typed_operator_contracts" not in profiling_plan:
        return True
    typed_contracts = profiling_plan["typed_operator_contracts"]
    if not isinstance(typed_contracts, dict):
        raise TypeError(
            "profiling_plan['typed_operator_contracts'] must be a dict, "
            f"got {type(typed_contracts).__name__}"
        )
    metadata = typed_contracts.get(operator_name)
    if metadata is None:
        return False
    if not isinstance(metadata, dict):
        raise TypeError(
            "typed operator contract metadata must be a dict, "
            f"got {type(metadata).__name__} for {operator_name!r}"
        )
    selected_size = metadata.get("selected_tensor_parallel_size")
    if selected_size is None:
        return False
    if selected_size != world_size:
        raise ValueError(
            f"Typed operator {operator_name!r} selected TP={selected_size}, "
            f"but GPTBlock was built with world_size={world_size}"
        )
    return True


def _resolve_typed_linear_contract(
    config: ModelConfig,
    *,
    operator_name: str,
    world_size: int,
    layer_id: int | None = None,
    profiling_plan: Optional[dict] = None,
) -> tuple[ResolvedLayerContract, int]:
    """Resolve one profile-owned linear contract and its selected padded width."""

    profile = _get_architecture_profile(config)
    contract = profile.resolve_layer_contract(
        config,
        layer_id=layer_id,
        operator_name=operator_name,
        tensor_parallel_size=world_size,
    )
    width = contract.effective_ffn_width

    if profiling_plan is None or "typed_operator_contracts" not in profiling_plan:
        return contract, width

    typed_contracts = profiling_plan["typed_operator_contracts"]
    metadata = typed_contracts.get(operator_name)
    if metadata is None:
        raise ValueError(
            f"Profiling plan is missing typed metadata for active operator "
            f"{operator_name!r}"
        )
    if not isinstance(metadata, dict):
        raise TypeError(
            "typed operator contract metadata must be a dict, "
            f"got {type(metadata).__name__} for {operator_name!r}"
        )
    padded_width = _validate_typed_linear_metadata(
        metadata,
        contract,
        operator_name=operator_name,
        world_size=world_size,
    )
    return contract, padded_width


def _validate_typed_linear_metadata(
    metadata: dict,
    contract: ResolvedLayerContract,
    *,
    operator_name: str,
    world_size: int,
) -> int:
    """Validate serialized producer metadata against the resolved contract."""

    expected_fields = contract.typed_metadata_identity()
    missing_fields = [
        field_name
        for field_name in expected_fields
        if field_name not in metadata
    ]
    if missing_fields:
        raise ValueError(
            f"Typed operator {operator_name!r} metadata is missing required "
            f"contract fields: {', '.join(missing_fields)}"
        )

    for field_name, expected_value in expected_fields.items():
        actual_value = metadata[field_name]
        if not typed_metadata_value_matches(actual_value, expected_value):
            raise ValueError(
                f"Typed operator {operator_name!r} {field_name} disagrees with "
                "its resolved profile contract: "
                f"plan={actual_value!r}, resolved={expected_value!r}"
            )

    selected_tp = metadata["selected_tensor_parallel_size"]
    if type(selected_tp) is not int or selected_tp <= 0:
        raise ValueError(
            f"Typed operator {operator_name!r} selected TP must be a positive "
            f"int, got {selected_tp!r}"
        )
    if selected_tp != world_size:
        raise ValueError(
            f"Typed operator {operator_name!r} selected TP={selected_tp} does "
            f"not match runtime world_size={world_size}"
        )

    padded_width = metadata.get("selected_padded_ffn_width")
    if type(padded_width) is not int or padded_width < contract.effective_ffn_width:
        raise ValueError(
            f"Typed operator {operator_name!r} has invalid selected padded width "
            f"{padded_width!r} for resolved width "
            f"{contract.effective_ffn_width}"
        )
    if padded_width % selected_tp != 0:
        raise ValueError(
            f"Typed operator {operator_name!r} selected padded width="
            f"{padded_width} must be divisible by selected TP={selected_tp}"
        )
    return padded_width


def _uses_gemma_rms_norm(config: ModelConfig) -> bool:
    return (
        getattr(config, "norm", None) == "rms_norm"
        and getattr(config, "model_type", None) == "qwen3_next"
    )


def _build_untimed_norm(config: ModelConfig, hidden_dim: int) -> torch.nn.Module:
    if config.norm == "layer_norm":
        return torch.nn.LayerNorm(hidden_dim)
    if config.norm == "rms_norm":
        norm_cls = GemmaRMSNorm if _uses_gemma_rms_norm(config) else RMSNorm
        return norm_cls(
            hidden_dim,
            eps=getattr(config, "rms_norm_eps", 1e-6),
            norm_name=None,
        )
    raise ValueError(f"Unknown norm: {config.norm}")


class DummyAttention(nn.Module):
    """No-op attention used when attention ops are disabled in profiling."""

    def forward(self, hidden_states, positions):
        return hidden_states


class DummyMLP(nn.Module):
    """No-op MLP used when FFN ops are disabled in profiling."""

    def forward(self, hidden_states):
        return hidden_states


class QKNorm(nn.Module):
    """QK-Norm layer without internal timing.
    
    This wraps the vLLM RMSNorm kernel to match runtime behavior. It does not
    have its own timer because it's meant to be timed together with the QKV
    projection as part of attn_pre_proj.
    """
    
    def __init__(
        self,
        head_dim: int,
        eps: float = 1e-6,
        use_gemma_rms_norm: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        norm_cls = GemmaRMSNorm if use_gemma_rms_norm else RMSNorm
        self.norm = norm_cls(head_dim, eps=eps, norm_name=None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply vLLM RMSNorm to the input tensor."""
        return self.norm(x)


class Step2MiniCausalSelfAttention(torch.nn.Module):
    """CausalSelfAttention for Step2Mini architecture.

    Step2Mini has a unique Q projection pipeline:
    QKV split → inter_norm (RMSNorm on Q) → wq (ColumnParallelLinear on Q) → RoPE

    This is different from standard attention where Q goes directly to RoPE after split.
    The inter_norm and wq operations are timed separately for accurate profiling.
    """

    def __init__(self, config: ModelConfig, world_size: int):
        super().__init__()
        assert config.embedding_dim % config.num_q_heads == 0
        assert config.embedding_dim % world_size == 0
        assert config.num_q_heads % world_size == 0

        fp8_block_size = None
        if config.quantization_config is not None:
            fp8_block_size = config.quantization_config.weight_block_size

        # Get head_dim from config if available, otherwise compute from embedding_dim
        self.head_dim = config.get_head_dim()
        self.num_q_heads_per_worker = config.num_q_heads // world_size
        self.num_kv_heads_per_worker = _resolve_num_kv_heads_per_worker(
            config.num_kv_heads, world_size
        )

        # Step2Mini uses share_q_dim for Q dimension if specified
        # q_size is the dimension of Q after QKV split, before inter_norm and wq
        if config.share_q_dim is not None:
            self.q_size = config.share_q_dim
        else:
            self.q_size = self.num_q_heads_per_worker * self.head_dim

        self.kv_size = self.num_kv_heads_per_worker * self.head_dim
        self.scaling = self.head_dim**-0.5

        # QKV projection (Q output is share_q_dim, not num_heads * head_dim)
        qkv_output_size = self.q_size + 2 * self.kv_size
        self.qkv_proj = ColumnParallelLinear(
            config.embedding_dim,
            qkv_output_size * world_size,  # Will be split across workers
            bias=config.use_bias or config.use_qkv_bias,
            gather_output=False,
            linear_metric_name="attn_pre_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )

        # Step2Mini-specific: inter_norm (RMSNorm on Q after split)
        rms_norm_eps = getattr(config, 'rms_norm_eps', 1e-6)
        self.inter_norm = RMSNorm(self.q_size, eps=rms_norm_eps, norm_name=None)
        self._attn_inter_norm_timer = CudaTimer("attn_inter_norm")
        raise_if_fp8_requested(
            "attn_inter_norm",
            "FP8 RMSNorm kernel is unavailable for attn_inter_norm profiling.",
        )

        # Step2Mini-specific: wq (ColumnParallelLinear on Q after inter_norm)
        # Transforms Q from share_q_dim to num_heads * head_dim
        self.wq = ColumnParallelLinear(
            self.q_size,
            self.head_dim * config.num_q_heads,
            bias=False,
            gather_output=False,
            linear_metric_name="attn_wq_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )

        # Output projection
        self.o_proj = RowParallelLinear(
            config.num_q_heads * self.head_dim,
            config.embedding_dim,
            bias=config.use_bias,
            input_is_parallel=True,
            reduce_results=False,
            linear_metric_name="attn_post_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )

        # Rotary embedding
        self.rotary_emb = None
        if isinstance(config.rope_theta, int) or isinstance(config.rope_theta, float):
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=config.is_neox_style,
                rope_scaling=config.rope_scaling,
            )
        self._attn_rope_timer = CudaTimer("attn_rope")
        if self.rotary_emb is not None:
            raise_if_fp8_requested(
                "attn_rope",
                "FP8 RoPE kernel is unavailable for attn_rope profiling.",
            )

    def forward(self, hidden_states, positions):
        # QKV projection
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Step2Mini-specific: inter_norm (RMSNorm on Q)
        with self._attn_inter_norm_timer:
            q = self.inter_norm(q.contiguous())

        # Step2Mini-specific: wq projection (timing handled internally by ColumnParallelLinear)
        q, _ = self.wq(q)

        # RoPE
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)

        # Simulate attention output (same shape as q after wq)
        attn_output = torch.randn_like(q)
        output, _ = self.o_proj(attn_output)
        return output


class Step3TextCausalSelfAttention(torch.nn.Module):
    """CausalSelfAttention for Step3Text architecture.

    Step3Text uses a unique Q pipeline:
    QKV split → inter_norm (RMSNorm on Q) → wq → RoPE.

    Sub-scopes are recorded under attn_pre_proj for profiling alignment.
    """

    def __init__(self, config: ModelConfig, world_size: int):
        super().__init__()
        assert config.embedding_dim % config.num_q_heads == 0
        assert config.embedding_dim % world_size == 0
        assert config.num_q_heads % world_size == 0

        fp8_block_size = None
        if config.quantization_config is not None:
            fp8_block_size = config.quantization_config.weight_block_size

        self.head_dim = config.get_head_dim()
        self.num_q_heads_per_worker = config.num_q_heads // world_size
        self.num_kv_heads_per_worker = _resolve_num_kv_heads_per_worker(
            config.num_kv_heads, world_size
        )

        if config.share_q_dim is not None:
            self.q_size = config.share_q_dim
        else:
            self.q_size = self.num_q_heads_per_worker * self.head_dim

        self.kv_size = self.num_kv_heads_per_worker * self.head_dim
        self.scaling = self.head_dim**-0.5

        qkv_output_size = self.q_size + 2 * self.kv_size
        self.qkv_proj = ReplicatedLinear(
            config.embedding_dim,
            qkv_output_size,
            bias=config.use_bias or config.use_qkv_bias,
            linear_metric_name="attn_pre_proj_qkv",
            precision_op_name="attn_pre_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.inter_norm = RMSNorm(self.q_size, eps=rms_norm_eps, norm_name=None)
        self._attn_pre_proj_timer = CudaTimer("attn_pre_proj")
        self._attn_pre_proj_q_norm_timer = CudaTimer("attn_pre_proj_q_norm")

        self.wq = ColumnParallelLinear(
            self.q_size,
            self.head_dim * config.num_q_heads,
            bias=False,
            gather_output=False,
            linear_metric_name="attn_pre_proj_wq",
            precision_op_name="attn_pre_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )

        self.o_proj = RowParallelLinear(
            config.num_q_heads * self.head_dim,
            config.embedding_dim,
            bias=config.use_bias,
            input_is_parallel=True,
            reduce_results=False,
            linear_metric_name="attn_post_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )

        self.rotary_emb = None
        if isinstance(config.rope_theta, int) or isinstance(config.rope_theta, float):
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=config.is_neox_style,
                rope_scaling=config.rope_scaling,
            )
        self._attn_rope_timer = CudaTimer("attn_rope")
        if self.rotary_emb is not None:
            raise_if_fp8_requested(
                "attn_rope",
                "FP8 RoPE kernel is unavailable for attn_rope profiling.",
            )

    def forward(self, hidden_states, positions):
        with self._attn_pre_proj_timer:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            with self._attn_pre_proj_q_norm_timer:
                q = self.inter_norm(q.contiguous())

            q, _ = self.wq(q)

        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = torch.randn_like(q)
        output, _ = self.o_proj(attn_output)
        return output


class Step3TextReplicatedPreProj(torch.nn.Module):
    """Replicated-only Step3 pre-projection profiling path.

    This path profiles only Step3's replicated QKV projection and Q norm
    (`attn_pre_proj_qkv`, `attn_pre_proj_q_norm`) at TP=1.
    """

    def __init__(self, config: ModelConfig, world_size: int, enabled_ops: set[str]):
        super().__init__()

        fp8_block_size = None
        if config.quantization_config is not None:
            fp8_block_size = config.quantization_config.weight_block_size

        self.head_dim = config.get_head_dim()
        self.q_size = config.share_q_dim if config.share_q_dim is not None else self.head_dim
        self.kv_size = config.num_kv_heads * self.head_dim

        self._profile_qkv = "attn_pre_proj_qkv" in enabled_ops
        self._profile_q_norm = "attn_pre_proj_q_norm" in enabled_ops
        # Keep only sub-op metrics for this replicated-only path.
        self._attn_pre_proj_timer = CudaTimer(None)
        self._attn_pre_proj_q_norm_timer = CudaTimer(
            "attn_pre_proj_q_norm" if self._profile_q_norm else None
        )

        qkv_output_size = self.q_size + 2 * self.kv_size
        self.qkv_proj = ReplicatedLinear(
            config.embedding_dim,
            qkv_output_size,
            bias=config.use_bias or config.use_qkv_bias,
            linear_metric_name="attn_pre_proj_qkv" if self._profile_qkv else None,
            precision_op_name="attn_pre_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.inter_norm = RMSNorm(self.q_size, eps=rms_norm_eps, norm_name=None)

    def forward(self, hidden_states, positions):
        del positions
        with self._attn_pre_proj_timer:
            qkv, _ = self.qkv_proj(hidden_states)
            q, _, _ = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            with self._attn_pre_proj_q_norm_timer:
                if self._profile_q_norm:
                    _ = self.inter_norm(q.contiguous())
        return hidden_states


class CausalSelfAttention(torch.nn.Module):

    def __init__(self, config: ModelConfig, world_size: int):
        super().__init__()
        assert config.embedding_dim % config.num_q_heads == 0
        assert config.embedding_dim % world_size == 0
        assert config.num_q_heads % world_size == 0

        # Use config.get_head_dim() to prioritize explicit head_dim from JSON config
        self.head_dim = config.get_head_dim()
        self.num_q_heads_per_worker = config.num_q_heads // world_size
        self.num_kv_heads_per_worker = _resolve_num_kv_heads_per_worker(
            config.num_kv_heads, world_size
        )

        self.q_size = self.num_q_heads_per_worker * self.head_dim
        self.kv_size = self.num_kv_heads_per_worker * self.head_dim
        self.attn_output_gate = bool(getattr(config, "attn_output_gate", False))
        self.scaling = self.head_dim**-0.5

        fp8_block_size = None
        if config.quantization_config is not None:
            fp8_block_size = config.quantization_config.weight_block_size

        # QK-norm support (for Qwen3, Gemma3, OLMo2, etc.)
        # Use getattr for backward compatibility with configs that don't have use_qk_norm
        self.use_qk_norm = getattr(config, 'use_qk_norm', False)
        self._attn_pre_proj_timer = CudaTimer("attn_pre_proj")
        
        if self.use_qk_norm:
            # When QK-norm is enabled, we use an external timer to wrap
            # QKV projection + QK-norm together (matching vLLM's attn_pre_proj scope)
            q_with_gate_size = self.q_size * (2 if self.attn_output_gate else 1)
            qkv_output_size = (q_with_gate_size + 2 * self.kv_size) * world_size
            self.qkv_proj = ColumnParallelLinear(
                config.embedding_dim,
                qkv_output_size,
                bias=config.use_bias or config.use_qkv_bias,
                gather_output=False,
                linear_metric_name=None,  # Disable internal timing
                precision_op_name="attn_pre_proj",
                fp8_weight_block_size=fp8_block_size,
                world_size=world_size,
            )
            # QK-norm layers (without internal timing)
            rms_norm_eps = getattr(config, 'rms_norm_eps', 1e-6)
            use_gemma_rms_norm = _uses_gemma_rms_norm(config)
            self.q_norm = QKNorm(
                self.head_dim,
                eps=rms_norm_eps,
                use_gemma_rms_norm=use_gemma_rms_norm,
            )
            self.k_norm = QKNorm(
                self.head_dim,
                eps=rms_norm_eps,
                use_gemma_rms_norm=use_gemma_rms_norm,
            )
        else:
            # Keep attn_pre_proj boundary at the attention scope so the timed
            # region can include qkv split, matching vLLM's source contract.
            q_with_gate_size = self.q_size * (2 if self.attn_output_gate else 1)
            qkv_output_size = (q_with_gate_size + 2 * self.kv_size) * world_size
            self.qkv_proj = ColumnParallelLinear(
                config.embedding_dim,
                qkv_output_size,
                bias=config.use_bias or config.use_qkv_bias,
                gather_output=False,
                linear_metric_name=None,
                precision_op_name="attn_pre_proj",
                fp8_weight_block_size=fp8_block_size,
                world_size=world_size,
            )

        self.o_proj = RowParallelLinear(
            config.num_q_heads * self.head_dim,
            config.embedding_dim,
            bias=config.use_bias,
            input_is_parallel=True,
            reduce_results=False,
            linear_metric_name="attn_post_proj",
            fp8_weight_block_size=fp8_block_size,
            world_size=world_size,
        )
        self.rotary_emb = None
        if isinstance(config.rope_theta, int) or isinstance(config.rope_theta, float):
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=config.is_neox_style,
                rope_scaling=config.rope_scaling,
            )
        self._attn_rope_timer = CudaTimer("attn_rope")
        if self.rotary_emb is not None:
            raise_if_fp8_requested(
                "attn_rope",
                "FP8 RoPE kernel is unavailable for attn_rope profiling.",
            )

    def forward(self, hidden_states, positions):
        if self.use_qk_norm:
            # QK-norm enabled: wrap QKV projection + QK-norm in single timer
            with self._attn_pre_proj_timer:
                qkv, _ = self.qkv_proj(hidden_states)
                q_and_gate, k, v = qkv.split(
                    [
                        self.q_size * (2 if self.attn_output_gate else 1),
                        self.kv_size,
                        self.kv_size,
                    ],
                    dim=-1,
                )
                if self.attn_output_gate:
                    q, _ = torch.chunk(q_and_gate, 2, dim=-1)
                else:
                    q = q_and_gate

                # Apply QK-norm (inside attn_pre_proj scope to match vLLM)
                # Reshape to [batch, num_heads, head_dim] for per-head normalization
                q_by_head = q.view(*q.shape[:-1], -1, self.head_dim)
                q_by_head = self.q_norm(q_by_head)
                q = q_by_head.view(q.shape)
                
                k_by_head = k.view(*k.shape[:-1], -1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
                k = k_by_head.view(k.shape)
        else:
            with self._attn_pre_proj_timer:
                qkv, _ = self.qkv_proj(hidden_states)
                q_and_gate, k, v = qkv.split(
                    [
                        self.q_size * (2 if self.attn_output_gate else 1),
                        self.kv_size,
                        self.kv_size,
                    ],
                    dim=-1,
                )
                if self.attn_output_gate:
                    q, _ = torch.chunk(q_and_gate, 2, dim=-1)
                else:
                    q = q_and_gate
        
        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)
        # output from attn has the same shape as q
        attn_output = torch.randn_like(q)
        output, _ = self.o_proj(attn_output)
        return output


class MLP(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        world_size: int,
        embedding_dim: Optional[int] = None,
        mlp_hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        self.embedding_dim = (
            embedding_dim if embedding_dim is not None else config.embedding_dim
        )
        self.layer_contract, profile_width = _resolve_typed_linear_contract(
            config,
            operator_name="mlp_up_proj",
            world_size=world_size,
        )
        # Keep the explicit constructor override for legacy callers that pass a
        # deliberately padded width. GPTBlock supplies this value only after it
        # has resolved the profile-owned contract above.
        self.mlp_hidden_dim = (
            mlp_hidden_dim if mlp_hidden_dim is not None else profile_width
        )
        if type(self.mlp_hidden_dim) is not int or self.mlp_hidden_dim <= 0:
            raise ValueError(
                "MLP mlp_hidden_dim must be a positive int, "
                f"got {self.mlp_hidden_dim!r}"
            )
        if mlp_hidden_dim is not None and self.mlp_hidden_dim < profile_width:
            raise ValueError(
                "MLP explicit mlp_hidden_dim is below the profile-owned width: "
                f"override={self.mlp_hidden_dim}, profile_width={profile_width}"
            )
        if self.mlp_hidden_dim % world_size != 0:
            raise ValueError(
                "MLP mlp_hidden_dim must be divisible by world_size, "
                f"got width={self.mlp_hidden_dim}, world_size={world_size}"
            )

        assert self.embedding_dim % world_size == 0

        fp8_block_size = None
        if config.quantization_config is not None:
            fp8_block_size = config.quantization_config.weight_block_size

        if config.use_gated_mlp:
            self.up_proj = ColumnParallelLinear(
                self.embedding_dim,
                2 * self.mlp_hidden_dim,
                bias=config.use_bias,
                gather_output=False,
                world_size=world_size,
                linear_metric_name="mlp_up_proj",
                fp8_weight_block_size=fp8_block_size,
            )
            self.act = SiluAndMul()
        else:
            self.up_proj = ColumnParallelLinear(
                self.embedding_dim,
                self.mlp_hidden_dim,
                bias=config.use_bias,
                gather_output=False,
                world_size=world_size,
                linear_metric_name="mlp_up_proj",
                fp8_weight_block_size=fp8_block_size,
            )
            self.act = torch.nn.GELU()

        self.down_proj = RowParallelLinear(
            self.mlp_hidden_dim,
            self.embedding_dim,
            bias=config.use_bias,
            input_is_parallel=True,
            world_size=world_size,
            reduce_results=False,
            linear_metric_name="mlp_down_proj",
            fp8_weight_block_size=fp8_block_size,
        )

        self.mlp_act_timer = CudaTimer("mlp_act")
        raise_if_fp8_requested(
            "mlp_act",
            "FP8 activation kernel is unavailable for mlp_act profiling.",
        )

    def forward(self, hidden_states):
        hidden_states, _ = self.up_proj(hidden_states)
        with self.mlp_act_timer:
            hidden_states = self.act(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class ShareExpertMLP(torch.nn.Module):
    """Shared-expert MLP for MoE architectures with a share_expert branch.

    The shared expert has a different intermediate dimension (share_expert_dim)
    from the regular routed MLP (mlp_hidden_dim).

    This class uses separate timer names (share_expert_*) to distinguish
    shared-expert profiling data from the regular routed FFN path.
    """

    def __init__(
        self,
        config: ModelConfig,
        world_size: int,
        embedding_dim: Optional[int] = None,
        mlp_hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        if not _supports_share_expert(config):
            raise ValueError(
                "ShareExpertMLP requires a model config that supports share_expert"
            )

        self.layer_contract, profile_width = _resolve_typed_linear_contract(
            config,
            operator_name="share_expert_up_proj",
            world_size=world_size,
        )
        # Match MLP's compatibility behavior while keeping the profile-owned
        # shared width as the default source of truth.
        self.mlp_hidden_dim = (
            mlp_hidden_dim if mlp_hidden_dim is not None else profile_width
        )
        if type(self.mlp_hidden_dim) is not int or self.mlp_hidden_dim <= 0:
            raise ValueError(
                "ShareExpertMLP mlp_hidden_dim must be a positive int, "
                f"got {self.mlp_hidden_dim!r}"
            )
        if mlp_hidden_dim is not None and self.mlp_hidden_dim < profile_width:
            raise ValueError(
                "ShareExpertMLP explicit mlp_hidden_dim is below the "
                "profile-owned width: "
                f"override={self.mlp_hidden_dim}, profile_width={profile_width}"
            )
        if self.mlp_hidden_dim % world_size != 0:
            raise ValueError(
                "ShareExpertMLP mlp_hidden_dim must be divisible by world_size, "
                f"got width={self.mlp_hidden_dim}, world_size={world_size}"
            )

        self.embedding_dim = (
            embedding_dim if embedding_dim is not None else config.embedding_dim
        )

        assert self.embedding_dim % world_size == 0

        fp8_block_size = None
        if config.quantization_config is not None:
            fp8_block_size = config.quantization_config.weight_block_size

        # Reuse the gated-MLP structure used by the shared-expert branch.
        self.up_proj = ColumnParallelLinear(
            self.embedding_dim,
            2 * self.mlp_hidden_dim,
            bias=False,  # Step2Mini uses no bias
            gather_output=False,
            world_size=world_size,
            linear_metric_name="share_expert_up_proj",  # Distinct timer name
            fp8_weight_block_size=fp8_block_size,
        )
        self.act = SiluAndMul()

        self.down_proj = RowParallelLinear(
            self.mlp_hidden_dim,
            self.embedding_dim,
            bias=False,  # Step2Mini uses no bias
            input_is_parallel=True,
            world_size=world_size,
            reduce_results=False,
            linear_metric_name="share_expert_down_proj",  # Distinct timer name
            fp8_weight_block_size=fp8_block_size,
        )

        self.share_expert_act_timer = CudaTimer("share_expert_act")
        raise_if_fp8_requested(
            "share_expert_act",
            "FP8 activation kernel is unavailable for share_expert_act profiling.",
        )

    def forward(self, hidden_states):
        hidden_states, _ = self.up_proj(hidden_states)
        with self.share_expert_act_timer:
            hidden_states = self.act(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


def build_linear_op_attention_module(
    config: ModelConfig,
    world_size: int,
    enabled_ops: set[str] | None,
    attn_sharded_enabled: bool,
) -> torch.nn.Module:
    """Build the linear-op profiling attention module from architecture profile."""

    linear_attention = _get_architecture_profile(config).linear_attention

    if attn_sharded_enabled:
        if linear_attention.sharded_impl is LinearAttentionImplementation.STEP3_TEXT:
            return Step3TextCausalSelfAttention(config, world_size)
        if linear_attention.sharded_impl is LinearAttentionImplementation.STEP2_MINI:
            return Step2MiniCausalSelfAttention(config, world_size)
        if linear_attention.sharded_impl is LinearAttentionImplementation.GENERIC:
            return CausalSelfAttention(config, world_size)
        raise ValueError(
            "Unknown linear attention implementation: "
            f"{linear_attention.sharded_impl}"
        )

    if (
        linear_attention.sharded_impl is LinearAttentionImplementation.STEP3_TEXT
        and linear_attention.has_replicated_pre_projection(enabled_ops)
    ):
        return Step3TextReplicatedPreProj(config, world_size, enabled_ops or set())

    return DummyAttention()


class GPTBlock(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        world_size: int,
        profiling_plan: Optional[dict] = None,
        layer_id: int | None = None,
    ):
        super().__init__()

        self._profiling_plan = profiling_plan
        self.layer_id = layer_id
        layer_kind = None
        if layer_id is not None:
            # Full-structure runtime loading supplies the global layer identity
            # so mixed models materialize only the domain owned by that layer.
            layer_kind = _get_architecture_profile(config).resolve_layer_contract(
                config,
                layer_id=layer_id,
            ).layer_kind
        self._attn_enabled = (
            profiling_plan.get("attn_enabled", True) if profiling_plan else True
        )
        self._ffn_enabled = (
            profiling_plan.get("ffn_enabled", True) if profiling_plan else True
        )
        self._attn_sharded_enabled = (
            profiling_plan.get("attn_sharded_enabled", self._attn_enabled)
            if profiling_plan
            else self._attn_enabled
        )
        self._ffn_sharded_enabled = (
            profiling_plan.get("ffn_sharded_enabled", self._ffn_enabled)
            if profiling_plan
            else self._ffn_enabled
        )
        enabled_ops = (
            set(profiling_plan.get("enabled_ops", [])) if profiling_plan else None
        )
        self._profile_input_layernorm = (
            memory_operator_enabled(enabled_ops, "input_layernorm")
        )
        self._profile_post_attention_layernorm = (
            memory_operator_enabled(enabled_ops, "post_attention_layernorm")
        )
        self._profile_add = (
            memory_operator_enabled(enabled_ops, "add_attn_residual")
            or memory_operator_enabled(enabled_ops, "add_ffn_residual")
        )
        # RMSNorm uses fused_add_rms_norm kernel — add is already included in layernorm time
        if config.uses_fused_add_norm:
            self._profile_add = False
        self._padded_n_embd = (
            profiling_plan.get("padded_n_embd", config.embedding_dim)
            if profiling_plan
            else config.embedding_dim
        )
        self._padded_n_expanded_embd = (
            profiling_plan.get("padded_n_expanded_embd", config.mlp_hidden_dim)
            if profiling_plan
            else config.mlp_hidden_dim
        )

        self._use_inner_input_layernorm_timer = False
        self._use_inner_post_attention_layernorm_timer = False

        if config.norm == "layer_norm":
            self.input_layernorm = torch.nn.LayerNorm(config.embedding_dim)
        elif config.norm == "rms_norm":
            norm_cls = GemmaRMSNorm if _uses_gemma_rms_norm(config) else RMSNorm
            self.input_layernorm = norm_cls(
                config.embedding_dim,
                norm_name=(
                    "input_layernorm" if self._profile_input_layernorm else None
                ),
                eps=getattr(config, "rms_norm_eps", 1e-6),
            )
            self._use_inner_input_layernorm_timer = self._profile_input_layernorm
        else:
            raise ValueError(f"Unknown norm: {config.norm} for input_layernorm")

        self._post_attn_norm = config.post_attn_norm
        if config.post_attn_norm:
            post_attn_dim = (
                self._padded_n_embd if self._ffn_enabled else config.embedding_dim
            )
            if config.norm == "rms_norm":
                norm_cls = GemmaRMSNorm if _uses_gemma_rms_norm(config) else RMSNorm
                self.post_attention_layernorm = norm_cls(
                    post_attn_dim,
                    norm_name=(
                        "post_attention_layernorm"
                        if self._profile_post_attention_layernorm
                        else None
                    ),
                    eps=getattr(config, "rms_norm_eps", 1e-6),
                )
                self._use_inner_post_attention_layernorm_timer = (
                    self._profile_post_attention_layernorm
                )
            else:
                raise ValueError(
                    f"Unknown norm: {config.norm} for post_attention_layernorm"
                )

        self.attn = build_linear_op_attention_module(
            config=config,
            world_size=world_size,
            enabled_ops=enabled_ops,
            attn_sharded_enabled=self._attn_sharded_enabled,
        )

        self.mlp_layer_contract = None
        dense_contract_active = (
            layer_kind is LayerKind.DENSE
            if layer_kind is not None
            else _has_active_layer_kind(config, LayerKind.DENSE)
        )
        if self._ffn_sharded_enabled and dense_contract_active:
            if not _typed_operator_is_selected(
                profiling_plan, "mlp_up_proj", world_size
            ):
                raise ValueError(
                    "GPTBlock requested a sharded dense FFN without a selected "
                    "mlp_up_proj typed contract"
                )
            self.mlp_layer_contract, dense_width = _resolve_typed_linear_contract(
                config,
                operator_name="mlp_up_proj",
                world_size=world_size,
                layer_id=layer_id,
                profiling_plan=profiling_plan,
            )
            self.mlp = MLP(
                config,
                world_size,
                embedding_dim=self._padded_n_embd,
                mlp_hidden_dim=dense_width,
            )
        else:
            # Pure routed-MoE layers are materialized by the dedicated MoE
            # producer/weight path and must not allocate a dense FFN here.
            self.mlp = DummyMLP()

        self.share_expert_layer_contract = None
        # A layer-aware full-structure block leaves routed/shared weights to
        # the dedicated MoE containers. Legacy representative blocks without
        # layer identity retain the existing profile-level shared branch.
        shared_contract_active = (
            False
            if layer_kind is not None
            else _has_active_layer_kind(config, LayerKind.SHARED)
        )
        if self._ffn_sharded_enabled and shared_contract_active:
            if not _typed_operator_is_selected(
                profiling_plan, "share_expert_up_proj", world_size
            ):
                raise ValueError(
                    "GPTBlock requested a sharded shared expert without a selected "
                    "share_expert_up_proj typed contract"
                )
            (
                self.share_expert_layer_contract,
                shared_width,
            ) = _resolve_typed_linear_contract(
                config,
                operator_name="share_expert_up_proj",
                world_size=world_size,
                layer_id=layer_id,
                profiling_plan=profiling_plan,
            )
            self.share_expert = ShareExpertMLP(
                config,
                world_size,
                embedding_dim=self._padded_n_embd,
                mlp_hidden_dim=shared_width,
            )
        else:
            self.share_expert = None

        self.input_layernorm_timer = CudaTimer(
            (
                None
                if self._use_inner_input_layernorm_timer
                else "input_layernorm" if self._profile_input_layernorm else None
            )
        )
        self.post_attention_layernorm_timer = CudaTimer(
            (
                None
                if self._use_inner_post_attention_layernorm_timer
                else "post_attention_layernorm"
                if self._profile_post_attention_layernorm
                else None
            )
        )
        self.add_timer = CudaTimer("add" if self._profile_add else None)
        if self._attn_enabled and self._profile_input_layernorm:
            raise_if_fp8_requested(
                "input_layernorm",
                "FP8 norm kernel is unavailable for input_layernorm profiling.",
            )
        if (
            self._post_attn_norm
            and self._ffn_enabled
            and self._profile_post_attention_layernorm
        ):
            raise_if_fp8_requested(
                "post_attention_layernorm",
                "FP8 norm kernel is unavailable for post_attention_layernorm profiling.",
            )
        if (self._attn_enabled or self._ffn_enabled) and self._profile_add:
            raise_if_fp8_requested(
                "add",
                "FP8 add kernel is unavailable for add profiling.",
            )

    def forward(self, positions, hidden_states, residual):
        if self._post_attn_norm:
            return self._forward_with_post_attn_norm(positions, hidden_states, residual)
        else:
            return self._forward_without_post_attn_norm(positions, hidden_states, residual)

    def _maybe_pad_tensor(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self._ffn_enabled:
            return tensor
        if tensor is None:
            return None
        current_dim = tensor.shape[-1]
        if current_dim == self._padded_n_embd:
            return tensor
        if current_dim > self._padded_n_embd:
            raise ValueError(
                f"Cannot pad tensor with dim={current_dim} to smaller dim={self._padded_n_embd}"
            )
        pad_size = self._padded_n_embd - current_dim
        return F.pad(tensor, (0, pad_size))

    def _forward_with_post_attn_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        # Self Attention
        if self._attn_enabled:
            with self.input_layernorm_timer:
                if residual is None:
                    residual = hidden_states
                    hidden_states = self.input_layernorm(hidden_states)
                else:
                    with self.add_timer if self._profile_add else nullcontext():
                        hidden_states, residual = self.input_layernorm(
                            hidden_states, residual
                        )
            hidden_states = self.attn(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            if residual is None:
                residual = hidden_states

        if not self._ffn_enabled:
            return hidden_states, residual

        hidden_states = self._maybe_pad_tensor(hidden_states)
        residual = self._maybe_pad_tensor(residual)
        # Fully Connected
        with self.post_attention_layernorm_timer:
            with self.add_timer if self._profile_add else nullcontext():
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
        mlp_output = self.mlp(hidden_states)
        # Profile the shared-expert branch alongside the routed FFN path.
        if self.share_expert is not None:
            share_expert_output = self.share_expert(hidden_states)
            hidden_states = mlp_output + share_expert_output
        else:
            hidden_states = mlp_output
        return hidden_states, residual

    def _forward_without_post_attn_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        if self._attn_enabled:
            with self.input_layernorm_timer:
                if residual is None:
                    residual = hidden_states
                    hidden_states = self.input_layernorm(hidden_states)
                else:
                    with self.add_timer if self._profile_add else nullcontext():
                        hidden_states, residual = self.input_layernorm(
                            hidden_states, residual
                        )
            attn_outputs = self.attn(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            if residual is None:
                residual = hidden_states
            attn_outputs = hidden_states

        # An attention-only role has no FFN contribution.  DummyMLP is an
        # identity module for constructor compatibility, so invoking it here
        # would incorrectly add the input a second time on the residual path.
        if self._attn_enabled and not self._ffn_enabled:
            with self.add_timer if self._profile_add else nullcontext():
                hidden_states = attn_outputs + residual
            return hidden_states, residual

        if self._ffn_enabled:
            hidden_states = self._maybe_pad_tensor(hidden_states)
            residual = self._maybe_pad_tensor(residual)
            attn_outputs = self._maybe_pad_tensor(attn_outputs)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # Profile the shared-expert branch alongside the routed FFN path.
        if self.share_expert is not None:
            share_expert_output = self.share_expert(hidden_states)
            feed_forward_hidden_states = feed_forward_hidden_states + share_expert_output
        if self._attn_enabled or self._ffn_enabled:
            with self.add_timer if self._profile_add else nullcontext():
                hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states, residual


class GPTModel(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        world_size: int,
        num_repeat_steps: int = 1,
        pad_vocab_size: bool = False,
        profiling_plan: Optional[dict] = None,
    ):
        super().__init__()

        self.num_repeat_steps = num_repeat_steps
        enabled_ops = (
            set(profiling_plan.get("enabled_ops", [])) if profiling_plan else None
        )
        self._profile_emb = memory_operator_enabled(enabled_ops, "emb")
        self._profile_mtp_fusion_proj = bool(
            enabled_ops is not None and "mtp_fusion_proj" in enabled_ops
        )
        self._profile_mtp_lm_head = bool(
            enabled_ops is not None and "lm_head_linear" in enabled_ops
        )
        self._profile_target_embedded_mtp = bool(
            self._profile_mtp_fusion_proj or self._profile_mtp_lm_head
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.embedding_dim,
            linear_metric_name="emb" if self._profile_emb else None,
            reduce_results=False,
            world_size=world_size,
            rank=0,
            pad_vocab_size=pad_vocab_size,
        )

        self.block = GPTBlock(
            config,
            world_size=world_size,
            profiling_plan=profiling_plan,
        )

        if self._profile_target_embedded_mtp:
            fp8_block_size = None
            if getattr(config, "quantization_config", None) is not None:
                fp8_block_size = config.quantization_config.weight_block_size

            padded_vocab_size = (
                get_padded_vocab_size(config.vocab_size, world_size)
                if pad_vocab_size
                else config.vocab_size
            )
            self.mtp_embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.embedding_dim,
                linear_metric_name=None,
                reduce_results=False,
                world_size=world_size,
                rank=0,
                pad_vocab_size=pad_vocab_size,
            )
            self.mtp_embedding_reused_norm = _build_untimed_norm(
                config,
                config.embedding_dim,
            )
            self.mtp_hidden_reused_norm = _build_untimed_norm(
                config,
                config.embedding_dim,
            )
            self.mtp_fusion_proj_timer = CudaTimer(
                "mtp_fusion_proj" if self._profile_mtp_fusion_proj else None
            )
            self.mtp_fusion_proj = ColumnParallelLinear(
                2 * config.embedding_dim,
                config.embedding_dim,
                bias=bool(getattr(config, "use_bias", False)),
                gather_output=False,
                linear_metric_name=None,
                precision_op_name="mtp_fusion_proj",
                fp8_weight_block_size=fp8_block_size,
                world_size=world_size,
            )
            self.mtp_output_reused_norm = _build_untimed_norm(
                config,
                config.embedding_dim,
            )
            self.mtp_lm_head = ColumnParallelLinear(
                config.embedding_dim,
                padded_vocab_size,
                bias=False,
                gather_output=False,
                linear_metric_name="lm_head_linear",
                precision_op_name="lm_head_linear",
                fp8_weight_block_size=fp8_block_size,
                world_size=world_size,
            )

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for _ in range(self.num_repeat_steps):
            hidden_states = self.embed_tokens(input_ids)
            residual = hidden_states
            hidden_states, residual = self.block(
                positions,
                hidden_states,
                residual,
            )

        if self._profile_target_embedded_mtp:
            if self._profile_mtp_fusion_proj:
                mtp_embedding = self.mtp_embed_tokens(input_ids)
                mtp_embedding = self.mtp_embedding_reused_norm(
                    mtp_embedding.contiguous()
                )
                mtp_hidden = self.mtp_hidden_reused_norm(hidden_states.contiguous())
                # MTP fusion timing must cover the front-end concat path, not
                # only the local GEMM, while the reused norms stay separate.
                with self.mtp_fusion_proj_timer:
                    mtp_hidden = torch.cat([mtp_embedding, mtp_hidden], dim=-1)
                    self.mtp_fusion_proj(mtp_hidden)

            # lm_head consumes full hidden states after decoder output norm.
            # The all-gather between fusion and decoder is modeled by CC backend
            # in predictor/runtime, not inside linear-op profiling.
            if self._profile_mtp_lm_head:
                mtp_hidden = self.mtp_output_reused_norm(hidden_states.contiguous())
                self.mtp_lm_head(mtp_hidden)

        return hidden_states
