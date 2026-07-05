"""
Time component dataclasses for modular execution time composition.

This module defines fine-grained time components that can be composed
to form complete ExecutionTime objects for different cluster types.
"""

from dataclasses import dataclass, field
from typing import Mapping

from frontier.attention.families import iter_execution_enabled_families
from frontier.operators.families import (
    COMM_FAMILY,
    FFN_FAMILY,
    MEMORY_FAMILY,
    MOE_FAMILY,
    SHARE_EXPERT_FAMILY,
)


def _attention_operator_execution_time_attrs() -> dict[str, str]:
    return {
        operator.name: operator.execution_time_attr
        for family in iter_execution_enabled_families()
        for operator in family.e2e_trace_ops()
        if operator.execution_time_attr is not None
    }


def _mlp_operator_execution_time_attrs() -> dict[str, str]:
    return {
        operator.name: operator.execution_time_attr
        for family in (MEMORY_FAMILY, FFN_FAMILY)
        for operator in family.e2e_trace_ops()
        if operator.execution_time_attr
        in {
            "mlp_norm_time",
            "mlp_layer_up_proj_execution_time",
            "mlp_layer_act_execution_time",
            "mlp_layer_down_proj_execution_time",
        }
    }


def _moe_operator_execution_time_attrs() -> dict[str, str]:
    return {
        operator.name: operator.execution_time_attr
        for family in (MEMORY_FAMILY, MOE_FAMILY, SHARE_EXPERT_FAMILY)
        for operator in family.e2e_trace_ops()
        if operator.execution_time_attr
        in {
            "mlp_norm_time",
            "moe_gating_linear_time",
            "moe_gating_routing_topk_time",
            "moe_shuffling_time",
            "moe_grouped_gemm_time",
            "share_expert_up_proj_time",
            "share_expert_act_time",
            "share_expert_down_proj_time",
        }
    }


def _communication_operator_execution_time_attrs() -> dict[str, str]:
    return {
        operator.name: operator.execution_time_attr
        for operator in COMM_FAMILY.e2e_trace_ops()
        if operator.execution_time_attr is not None
    }


def canonical_operator_execution_time_attrs() -> dict[str, str]:
    """Return the canonical op_id -> ExecutionTime attribute mapping."""

    operator_attrs: dict[str, str] = {}
    for family in (
        *tuple(iter_execution_enabled_families()),
        MEMORY_FAMILY,
        FFN_FAMILY,
        MOE_FAMILY,
        SHARE_EXPERT_FAMILY,
        COMM_FAMILY,
    ):
        for operator in family.e2e_trace_ops():
            if operator.execution_time_attr is None:
                continue
            if operator.name in operator_attrs:
                raise ValueError(
                    f"Duplicate ExecutionTime operator timing key: {operator.name}"
                )
            operator_attrs[operator.name] = operator.execution_time_attr
    return operator_attrs


def normalize_execution_op_times(op_times: Mapping[str, float]) -> dict[str, float]:
    """Validate and return op timings in canonical operator order."""

    operator_attrs = canonical_operator_execution_time_attrs()
    normalized_by_name: dict[str, float] = {}
    for op_name, time_ms in op_times.items():
        normalized_op_name = str(op_name)
        if not normalized_op_name:
            raise ValueError("Execution operator name must be non-empty")
        if normalized_op_name not in operator_attrs:
            raise ValueError(
                f"Unsupported ExecutionTime operator timing: {normalized_op_name}"
            )
        normalized_time_ms = float(time_ms)
        if normalized_time_ms < 0.0:
            raise ValueError(
                "Negative ExecutionTime operator timing is invalid: "
                f"{normalized_op_name}={normalized_time_ms}"
            )
        normalized_by_name[normalized_op_name] = normalized_time_ms
    return {
        op_name: normalized_by_name[op_name]
        for op_name in operator_attrs
        if op_name in normalized_by_name
    }


def execution_op_time_values_by_attr(op_times: Mapping[str, float]) -> dict[str, float]:
    """Aggregate canonical op timings by the legacy ExecutionTime attribute they replace."""

    operator_attrs = canonical_operator_execution_time_attrs()
    attr_values: dict[str, float] = {}
    for op_name, time_ms in normalize_execution_op_times(op_times).items():
        attr_name = operator_attrs[op_name]
        attr_values[attr_name] = attr_values.get(attr_name, 0.0) + float(time_ms)
    return attr_values


def _build_operator_times_subset(
    op_times: Mapping[str, float],
    allowed_attrs: Mapping[str, str],
) -> dict[str, float]:
    normalized_op_times = normalize_execution_op_times(op_times)
    return {
        op_name: normalized_op_times[op_name]
        for op_name in normalized_op_times
        if op_name in allowed_attrs
    }


def build_attention_operator_times_from_op_times(
    op_times: Mapping[str, float],
) -> "AttentionOperatorTimes | None":
    subset = _build_operator_times_subset(
        op_times,
        _attention_operator_execution_time_attrs(),
    )
    return AttentionOperatorTimes(subset) if subset else None


def build_mlp_operator_times_from_op_times(
    op_times: Mapping[str, float],
) -> "MLPOperatorTimes | None":
    subset = _build_operator_times_subset(
        op_times,
        _mlp_operator_execution_time_attrs(),
    )
    return MLPOperatorTimes(subset) if subset else None


def build_moe_operator_times_from_op_times(
    op_times: Mapping[str, float],
) -> "MoEOperatorTimes | None":
    subset = _build_operator_times_subset(
        op_times,
        _moe_operator_execution_time_attrs(),
    )
    return MoEOperatorTimes(subset) if subset else None


def build_communication_operator_times_from_op_times(
    op_times: Mapping[str, float],
) -> "CommunicationOperatorTimes | None":
    subset = _build_operator_times_subset(
        op_times,
        _communication_operator_execution_time_attrs(),
    )
    return CommunicationOperatorTimes(subset) if subset else None


@dataclass
class CommunicationOperatorTimes:
    """Single-layer timings keyed by physical communication operator name."""

    op_times: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_op_names = _communication_operator_execution_time_attrs()
        normalized_op_times: dict[str, float] = {}
        for op_name, time_ms in self.op_times.items():
            if not op_name:
                raise ValueError("Communication operator name must be non-empty")
            normalized_op_name = str(op_name)
            if normalized_op_name not in allowed_op_names:
                raise ValueError(
                    f"Unsupported communication operator timing: {normalized_op_name}"
                )
            normalized_time_ms = float(time_ms)
            if normalized_time_ms < 0.0:
                raise ValueError(
                    "Negative communication operator timing is invalid: "
                    f"{normalized_op_name}={normalized_time_ms}"
                )
            normalized_op_times[normalized_op_name] = normalized_time_ms
        self.op_times = normalized_op_times

    def get_required_time(self, op_name: str) -> float:
        try:
            return float(self.op_times[op_name])
        except KeyError as exc:
            raise ValueError(
                f"ExecutionTime is missing structured communication operator "
                f"timing for {op_name}"
            ) from exc

    def total_time(self) -> float:
        return sum(float(time_ms) for time_ms in self.op_times.values())

    def covers_attr(self, attr_name: str) -> bool:
        communication_operator_attrs = _communication_operator_execution_time_attrs()
        return any(
            communication_operator_attrs[op_name] == attr_name
            for op_name in self.op_times
        )

    def legacy_covered_time(self, communication_time: "CommunicationTime") -> float:
        covered_time_ms = 0.0
        communication_operator_attrs = _communication_operator_execution_time_attrs()
        covered_attrs: set[str] = set()
        for op_name in self.op_times:
            attr_name = communication_operator_attrs[op_name]
            if attr_name in covered_attrs:
                continue
            covered_time_ms += float(getattr(communication_time, attr_name))
            covered_attrs.add(attr_name)
        return covered_time_ms


@dataclass
class AttentionOperatorTimes:
    """Single-layer timings keyed by physical attention operator name."""

    op_times: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_op_times: dict[str, float] = {}
        for op_name, time_ms in self.op_times.items():
            if not op_name:
                raise ValueError("Attention operator name must be non-empty")
            normalized_time_ms = float(time_ms)
            if normalized_time_ms < 0.0:
                raise ValueError(
                    "Negative attention operator timing is invalid: "
                    f"{op_name}={normalized_time_ms}"
                )
            normalized_op_times[str(op_name)] = normalized_time_ms
        self.op_times = normalized_op_times

    def get_required_time(self, op_name: str) -> float:
        try:
            return float(self.op_times[op_name])
        except KeyError as exc:
            raise ValueError(
                f"ExecutionTime is missing structured attention operator "
                f"timing for {op_name}"
            ) from exc

    def total_time(self) -> float:
        return sum(float(time_ms) for time_ms in self.op_times.values())

    def legacy_covered_time(self, attention_time: "AttentionTime") -> float:
        covered_time_ms = 0.0
        attention_operator_attrs = _attention_operator_execution_time_attrs()
        for op_name in self.op_times:
            attr_name = attention_operator_attrs.get(op_name)
            if attr_name is not None:
                covered_time_ms += float(getattr(attention_time, attr_name))
        return covered_time_ms


@dataclass
class MLPOperatorTimes:
    """Single-layer timings keyed by physical dense-MLP operator name."""

    op_times: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_op_names = _mlp_operator_execution_time_attrs()
        normalized_op_times: dict[str, float] = {}
        for op_name, time_ms in self.op_times.items():
            if not op_name:
                raise ValueError("MLP operator name must be non-empty")
            normalized_op_name = str(op_name)
            if normalized_op_name not in allowed_op_names:
                raise ValueError(
                    f"Unsupported MLP operator timing: {normalized_op_name}"
                )
            normalized_time_ms = float(time_ms)
            if normalized_time_ms < 0.0:
                raise ValueError(
                    "Negative MLP operator timing is invalid: "
                    f"{normalized_op_name}={normalized_time_ms}"
                )
            normalized_op_times[normalized_op_name] = normalized_time_ms
        self.op_times = normalized_op_times

    def get_required_time(self, op_name: str) -> float:
        try:
            return float(self.op_times[op_name])
        except KeyError as exc:
            raise ValueError(
                f"ExecutionTime is missing structured MLP operator "
                f"timing for {op_name}"
            ) from exc

    def total_time(self) -> float:
        return sum(float(time_ms) for time_ms in self.op_times.values())

    def legacy_covered_time(self, mlp_time: "MLPTime") -> float:
        covered_time_ms = 0.0
        mlp_operator_attrs = _mlp_operator_execution_time_attrs()
        for op_name in self.op_times:
            attr_name = mlp_operator_attrs[op_name]
            covered_time_ms += float(getattr(mlp_time, attr_name))
        return covered_time_ms


@dataclass
class MoEOperatorTimes:
    """Single-layer timings keyed by physical MoE operator name."""

    op_times: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed_op_names = _moe_operator_execution_time_attrs()
        normalized_op_times: dict[str, float] = {}
        for op_name, time_ms in self.op_times.items():
            if not op_name:
                raise ValueError("MoE operator name must be non-empty")
            normalized_op_name = str(op_name)
            if normalized_op_name not in allowed_op_names:
                raise ValueError(
                    f"Unsupported MoE operator timing: {normalized_op_name}"
                )
            normalized_time_ms = float(time_ms)
            if normalized_time_ms < 0.0:
                raise ValueError(
                    "Negative MoE operator timing is invalid: "
                    f"{normalized_op_name}={normalized_time_ms}"
                )
            normalized_op_times[normalized_op_name] = normalized_time_ms
        self.op_times = normalized_op_times

    def get_required_time(self, op_name: str) -> float:
        try:
            return float(self.op_times[op_name])
        except KeyError as exc:
            raise ValueError(
                f"ExecutionTime is missing structured MoE operator "
                f"timing for {op_name}"
            ) from exc

    def total_time(self) -> float:
        return sum(float(time_ms) for time_ms in self.op_times.values())

    def legacy_covered_time(self, moe_time: "MoETime") -> float:
        covered_time_ms = 0.0
        moe_operator_attrs = _moe_operator_execution_time_attrs()
        for op_name in self.op_times:
            attr_name = moe_operator_attrs[op_name]
            covered_time_ms += float(getattr(moe_time, attr_name))
        return covered_time_ms


@dataclass
class AttentionTime:
    """
    Execution time for attention operations in a single transformer layer.

    Includes all attention-related computations and memory operations.
    All times are in milliseconds.

    Step2Mini-specific fields (attn_inter_norm_time, attn_wq_proj_time):
    - Part of forward_1 in AFD mode (after Q split, before RoPE)
    - attn_inter_norm: RMSNorm on Q after split from QKV
    - attn_wq_proj: ColumnParallelLinear on Q after inter_norm
    """
    # Attention computation
    attention_prefill_execution_time: float = 0.0  # Prefill attention (QK^T + softmax + V)
    attention_decode_execution_time: float = 0.0   # Decode attention (single token)

    # Attention projections
    attention_layer_pre_proj_execution_time: float = 0.0   # QKV projection
    attention_layer_post_proj_execution_time: float = 0.0  # Output projection

    # Attention auxiliary operations
    attention_rope_execution_time: float = 0.0              # RoPE (Rotary Position Embedding)
    attention_kv_cache_save_execution_time: float = 0.0     # KV cache write

    # MLA physical attention operations from the vLLM V1 latent-attention path.
    attn_mla_kv_cache_save_time: float = 0.0
    attn_mla_prefill_kv_up_proj_time: float = 0.0
    attn_mla_prefill_time: float = 0.0
    attn_mla_decode_q_latent_proj_time: float = 0.0
    attn_mla_decode_time: float = 0.0
    attn_mla_v_up_proj_time: float = 0.0

    # Normalization
    attn_norm_time: float = 0.0  # Layer norm before attention

    # Step2Mini-specific operations (forward_1: inter_norm + wq after Q split)
    # These are 0.0 for non-Step2Mini models
    attn_inter_norm_time: float = 0.0  # RMSNorm on Q after split from QKV
    attn_wq_proj_time: float = 0.0     # ColumnParallelLinear on Q after inter_norm
    operator_times: AttentionOperatorTimes | None = None

    def total_time(self) -> float:
        """Calculate total attention time for this layer."""
        legacy_total_time = (
            self.attention_prefill_execution_time
            + self.attention_decode_execution_time
            + self.attention_layer_pre_proj_execution_time
            + self.attention_layer_post_proj_execution_time
            + self.attention_rope_execution_time
            + self.attention_kv_cache_save_execution_time
            + self.attn_mla_kv_cache_save_time
            + self.attn_mla_prefill_kv_up_proj_time
            + self.attn_mla_prefill_time
            + self.attn_mla_decode_q_latent_proj_time
            + self.attn_mla_decode_time
            + self.attn_mla_v_up_proj_time
            + self.attn_norm_time
            # Step2Mini-specific operations (0.0 for non-Step2Mini models)
            + self.attn_inter_norm_time
            + self.attn_wq_proj_time
        )
        if self.operator_times is None:
            return legacy_total_time
        return (
            legacy_total_time
            - self.operator_times.legacy_covered_time(self)
            + self.operator_times.total_time()
        )


@dataclass
class MLPTime:
    """
    Execution time for dense MLP operations in a single transformer layer.
    
    Used for non-MoE models. Mutually exclusive with MoETime.
    All times are in milliseconds.
    """
    mlp_layer_up_proj_execution_time: float = 0.0    # Up projection (hidden -> intermediate)
    mlp_layer_down_proj_execution_time: float = 0.0  # Down projection (intermediate -> hidden)
    mlp_layer_act_execution_time: float = 0.0        # Activation function (e.g., SwiGLU, GELU)
    mlp_norm_time: float = 0.0                       # Layer norm before MLP
    operator_times: MLPOperatorTimes | None = None
    
    def total_time(self) -> float:
        """Calculate total MLP time for this layer."""
        legacy_total_time = (
            self.mlp_layer_up_proj_execution_time
            + self.mlp_layer_down_proj_execution_time
            + self.mlp_layer_act_execution_time
            + self.mlp_norm_time
        )
        if self.operator_times is None:
            return legacy_total_time
        return (
            legacy_total_time
            - self.operator_times.legacy_covered_time(self)
            + self.operator_times.total_time()
        )


@dataclass
class MoETime:
    """
    Execution time for Mixture-of-Experts operations in a single transformer layer.

    Used for MoE models. Mutually exclusive with MLPTime.
    All times are in milliseconds.

    Note: expert_parallel_communication_time is NOT included here.
    Communication time should be obtained separately via _get_expert_parallel_communication_time().
    This maintains clear separation between compute and communication times.

    Step2Mini/Step3-specific fields (share_expert_*):
    - Part of forward_3 in AFD mode (shared expert alongside routed experts)
    - share_expert_up_proj: Up projection for shared expert
    - share_expert_down_proj: Down projection for shared expert
    - share_expert_act: Activation function for shared expert
    """
    moe_grouped_gemm_time: float = 0.0                    # Grouped GEMM for expert computation
    moe_gating_linear_time: float = 0.0                   # Gating linear layer (hidden_dim -> num_experts)
    moe_gating_routing_topk_time: float = 0.0             # TopK selection + Softmax normalization
    moe_shuffling_time: float = 0.0                       # Token shuffling/dispatch overhead
    mlp_norm_time: float = 0.0                            # Layer norm before MoE (post_attention_layernorm)

    # Step2Mini/Step3 share_expert operations (forward_3: shared expert alongside routed experts)
    # These are 0.0 for models without share_expert
    share_expert_up_proj_time: float = 0.0    # Shared expert up projection
    share_expert_down_proj_time: float = 0.0  # Shared expert down projection
    share_expert_act_time: float = 0.0        # Shared expert activation
    operator_times: MoEOperatorTimes | None = None

    @property
    def moe_gating_time(self) -> float:
        """Total gating time (linear + routing_topk) for backward compatibility."""
        return self.moe_gating_linear_time + self.moe_gating_routing_topk_time

    @property
    def share_expert_time(self) -> float:
        """Total shared expert time (Step2Mini-specific)."""
        return self.share_expert_up_proj_time + self.share_expert_down_proj_time + self.share_expert_act_time

    def total_time(self) -> float:
        """Calculate total MoE time for this layer (computation only, excludes communication)."""
        legacy_total_time = (
            self.moe_grouped_gemm_time
            + self.moe_gating_linear_time
            + self.moe_gating_routing_topk_time
            + self.moe_shuffling_time
            + self.mlp_norm_time
            # Step2Mini-specific operations (0.0 for non-Step2Mini models)
            + self.share_expert_up_proj_time
            + self.share_expert_down_proj_time
            + self.share_expert_act_time
        )
        if self.operator_times is None:
            return legacy_total_time
        return (
            legacy_total_time
            - self.operator_times.legacy_covered_time(self)
            + self.operator_times.total_time()
        )


@dataclass
class CommunicationTime:
    """
    Execution time for collective communication operations.
    
    Includes tensor parallelism, pipeline parallelism, and expert parallelism.
    All times are in milliseconds.
    """
    # Tensor parallelism (TP)
    attn_tensor_parallel_allreduce_time: float = 0.0  # TP all-reduce for attention output
    moe_tensor_parallel_allreduce_time: float = 0.0  # TP all-reduce for MLP/MoE output
    # Legacy fallback: used when attn/moe fields are not populated
    tensor_parallel_allreduce_time: float = 0.0  # TP all-reduce for attention/MLP output
    tensor_parallel_allgather_time: float = 0.0  # TP all-gather for FFN input (Step3)
    share_expert_tensor_parallel_allreduce_time: float = 0.0  # TP all-reduce for share_expert output

    # Data parallelism (DP)
    # Stepfun-vllm MoE prefill path uses two DP allreduce collectives:
    # 1) (hidden + router_logits) multicast via allreduce
    # 2) final hidden allreduce
    dp_input_allreduce_time: float = 0.0
    dp_output_allreduce_time: float = 0.0
    
    # Pipeline parallelism (PP)
    pipeline_parallel_send_recv_time: float = 0.0  # PP send/recv between stages
    
    # Expert parallelism (EP) - for MoE models
    # Note: EP communication is also tracked in MoETime.expert_parallel_communication_time
    # This field is for additional EP-specific collectives (e.g., all-gather)
    expert_parallel_allgather_time: float = 0.0  # EP all-gather for result aggregation
    expert_parallel_alltoall_time: float = 0.0   # EP all-to-all for token dispatch/return
    operator_times: CommunicationOperatorTimes | None = None
    
    def total_time(self) -> float:
        """Calculate total communication time."""
        operator_times_cover_split_tp = (
            self.operator_times is not None
            and (
                self.operator_times.covers_attr("attn_tensor_parallel_allreduce_time")
                or self.operator_times.covers_attr("moe_tensor_parallel_allreduce_time")
            )
        )
        if (
            operator_times_cover_split_tp
            or self.attn_tensor_parallel_allreduce_time > 0
            or self.moe_tensor_parallel_allreduce_time > 0
        ):
            tp_allreduce_time = (
                self.attn_tensor_parallel_allreduce_time
                + self.moe_tensor_parallel_allreduce_time
            )
        else:
            tp_allreduce_time = self.tensor_parallel_allreduce_time
        legacy_total_time = (
            tp_allreduce_time
            + self.tensor_parallel_allgather_time
            + self.share_expert_tensor_parallel_allreduce_time
            + self.dp_input_allreduce_time
            + self.dp_output_allreduce_time
            + self.pipeline_parallel_send_recv_time
            + self.expert_parallel_allgather_time
            + self.expert_parallel_alltoall_time
        )
        if self.operator_times is None:
            return legacy_total_time
        return (
            legacy_total_time
            - self.operator_times.legacy_covered_time(self)
            + self.operator_times.total_time()
        )


@dataclass
class OverheadTime:
    """
    Execution time for CPU scheduling and framework overhead.
    
    These are non-GPU computation times that contribute to end-to-end latency.
    All times are in milliseconds.
    """
    schedule_time: float = 0.0                    # Batch scheduling overhead
    sampler_e2e_time: float = 0.0                 # Token sampling (argmax/top-k/top-p)
    prepare_inputs_e2e_time: float = 0.0          # Input preparation (tokenization, padding)
    process_model_outputs_time: float = 0.0       # Output processing (detokenization)
    ray_comm_time: float = 0.0                    # Ray framework communication overhead
    pp_producer_send_path_runtime_time: float = 0.0  # Active PP producer send-path runtime overhead
    pp_receiver_head_runtime_time: float = 0.0    # Active PP receiver-head runtime overhead
    pp_prefill_consumer_active_runtime_time: float = 0.0  # Active PP prefill consumer runtime overhead
    pp_stage_boundary_residual_runtime_time: float = 0.0  # Active shared-domain PP boundary residual on consumer stage
    pp_stage_boundary_handoff_time: float = 0.0   # Diagnostic-only PP boundary overhead beyond wire cost

    def simulated_total_time(self) -> float:
        """Calculate overhead time that actively contributes to simulated stage occupancy."""
        return (
            self.schedule_time
            + self.sampler_e2e_time
            + self.prepare_inputs_e2e_time
            + self.process_model_outputs_time
            + self.ray_comm_time
            + self.pp_producer_send_path_runtime_time
            + self.pp_receiver_head_runtime_time
            + self.pp_prefill_consumer_active_runtime_time
            + self.pp_stage_boundary_residual_runtime_time
        )

    def diagnostic_only_total_time(self) -> float:
        """Calculate diagnostic-only overhead that is exported but not simulated."""
        return self.pp_stage_boundary_handoff_time

    def diagnostic_total_time(self) -> float:
        """Calculate the full diagnostic overhead sum, including inactive components."""
        return self.simulated_total_time() + self.diagnostic_only_total_time()

    def total_time(self) -> float:
        """Calculate overhead time used by the simulator runtime."""
        return self.simulated_total_time()


@dataclass
class ResidualTime:
    """
    Execution time for residual connection operations.

    All times are in milliseconds.

    vLLM has two separate residual add operations:
    - add_attn_residual: Attention output + residual (after attention block)
    - add_ffn_residual: FFN/MoE output + residual (after FFN/MoE block)
    """
    add_attn_residual_time: float = 0.0  # Residual addition after attention
    add_ffn_residual_time: float = 0.0   # Residual addition after FFN/MoE

    @property
    def add_time(self) -> float:
        """Total residual time (attn + ffn) for backward compatibility."""
        return self.add_attn_residual_time + self.add_ffn_residual_time

    def total_time(self) -> float:
        """Calculate total residual time."""
        return self.add_attn_residual_time + self.add_ffn_residual_time
