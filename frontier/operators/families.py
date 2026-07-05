from __future__ import annotations

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
)
from frontier.operators.registry import OperatorRegistry
from frontier.operators.spec import (
    CommOperatorSpec,
    CommPayloadContext,
    OperatorFamilySpec,
    OperatorPhase,
    OperatorRole,
    OperatorSpec,
    ProjectionOwnership,
    ResourceClass,
    TensorParallelMode,
    TraceKind,
)
from frontier.types import ClusterType


_ALL_PHASES = (OperatorPhase.PREFILL, OperatorPhase.DECODE, OperatorPhase.MIXED)


FFN_FAMILY = OperatorFamilySpec(
    family_id="ffn",
    display_name="Dense FFN",
    supported_variants=("dense",),
    resource_class=ResourceClass.COMP,
    profiling_order=("mlp_up_proj", "mlp_down_proj", "mlp_act"),
    operators=(
        OperatorSpec(
            name="mlp_up_proj",
            role=OperatorRole.PROJECTION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="mlp_layer_up_proj_execution_time",
            resource_class=ResourceClass.COMP,
            tp_mode=TensorParallelMode.FFN_TP,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
            calibration_key="mlp_up_proj",
        ),
        OperatorSpec(
            name="mlp_act",
            role=OperatorRole.ACTIVATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="mlp_layer_act_execution_time",
            resource_class=ResourceClass.COMP,
            tp_mode=TensorParallelMode.FFN_TP,
        ),
        OperatorSpec(
            name="mlp_down_proj",
            role=OperatorRole.PROJECTION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="mlp_layer_down_proj_execution_time",
            resource_class=ResourceClass.COMP,
            tp_mode=TensorParallelMode.FFN_TP,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
            calibration_key="mlp_down_proj",
        ),
    ),
)


MOE_FAMILY = OperatorFamilySpec(
    family_id="moe",
    display_name="Mixture-of-Experts",
    supported_variants=("routed",),
    resource_class=ResourceClass.COMP,
    profiling_order=(
        "moe_gating_linear",
        "moe_gating_routing_topk",
        "moe_shuffling",
        "moe_grouped_gemm",
    ),
    operators=(
        OperatorSpec(
            name="moe_gating_linear",
            role=OperatorRole.PROJECTION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="moe_gating_linear_time",
            resource_class=ResourceClass.COMP,
            tp_mode=TensorParallelMode.MOE_TP,
            ep_agnostic=True,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
            precision_op="moe_gating",
        ),
        OperatorSpec(
            name="moe_gating_routing_topk",
            role=OperatorRole.RESHAPE,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="moe_gating_routing_topk_time",
            resource_class=ResourceClass.MEMORY,
            tp_mode=TensorParallelMode.MOE_TP,
            ep_agnostic=True,
            precision_op="moe_gating",
        ),
        OperatorSpec(
            name="moe_shuffling",
            role=OperatorRole.RESHAPE,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="moe_shuffling_time",
            resource_class=ResourceClass.MEMORY,
            tp_mode=TensorParallelMode.MOE_TP,
            ep_agnostic=True,
            calibration_key="moe_shuffling",
        ),
        OperatorSpec(
            name="moe_grouped_gemm",
            role=OperatorRole.PROJECTION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="moe_grouped_gemm_time",
            resource_class=ResourceClass.COMP,
            tp_mode=TensorParallelMode.MOE_TP,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
            calibration_key="moe_grouped_gemm",
        ),
    ),
)


_DEFERRED_DISAGGREGATED_CLUSTER_TYPES = frozenset(
    (
        ClusterType.PREFILL,
        ClusterType.DECODE,
        ClusterType.DECODE_ATTN,
        ClusterType.DECODE_FFN,
    )
)

_DEFERRED_DISAGGREGATED_LEGACY_MOE_TP_OPS = frozenset(
    (
        "moe_gating_linear",
        "moe_gating_routing_topk",
        "moe_shuffling",
    )
)


def get_moe_family_operator_by_profiling_name(
    op_name: str,
    family: OperatorFamilySpec | None = None,
) -> OperatorSpec:
    family = family or MOE_FAMILY
    moe_ops = {
        operator.profiling_name(): operator
        for operator in family.profiling_ops()
    }
    if op_name not in moe_ops:
        raise ValueError(f"Unsupported MoE op: {op_name}")
    return moe_ops[op_name]


def resolve_moe_operator_tp_key(
    op_name: str,
    moe_tp_size: int,
    cluster_type: ClusterType | None = None,
    family: OperatorFamilySpec | None = None,
) -> int:
    if moe_tp_size <= 0:
        raise ValueError(f"Invalid MoE TP size: {moe_tp_size}")

    operator = get_moe_family_operator_by_profiling_name(op_name, family=family)
    if (
        cluster_type in _DEFERRED_DISAGGREGATED_CLUSTER_TYPES
        and op_name in _DEFERRED_DISAGGREGATED_LEGACY_MOE_TP_OPS
    ):
        # Decision #13 keeps PDD/disaggregated predictor internals as
        # no-regression evidence in this PR. Preserve the legacy replicated
        # profiling-key contract for auxiliary MoE routing ops on deferred
        # disaggregated clusters while monolithic/co-location uses MOE_TP.
        return 1
    if operator.tp_mode is TensorParallelMode.REPLICATED:
        return 1
    if operator.tp_mode is TensorParallelMode.MOE_TP:
        return moe_tp_size
    raise ValueError(
        f"Unsupported MoE TP mode for {op_name}: {operator.tp_mode}"
    )


def is_moe_operator_ep_agnostic(
    op_name: str,
    family: OperatorFamilySpec | None = None,
) -> bool:
    return get_moe_family_operator_by_profiling_name(
        op_name,
        family=family,
    ).ep_agnostic


SHARE_EXPERT_FAMILY = OperatorFamilySpec(
    family_id="share_expert",
    display_name="Shared Expert",
    supported_variants=("shared_dense",),
    resource_class=ResourceClass.COMP,
    profiling_order=(
        "share_expert_up_proj",
        "share_expert_down_proj",
        "share_expert_act",
    ),
    operators=(
        OperatorSpec(
            name="share_expert_up_proj",
            role=OperatorRole.PROJECTION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="share_expert_up_proj_time",
            resource_class=ResourceClass.COMP,
            tp_mode=TensorParallelMode.FFN_TP,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
            calibration_key="share_expert_up_proj",
        ),
        OperatorSpec(
            name="share_expert_act",
            role=OperatorRole.ACTIVATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="share_expert_act_time",
            resource_class=ResourceClass.COMP,
            tp_mode=TensorParallelMode.FFN_TP,
        ),
        OperatorSpec(
            name="share_expert_down_proj",
            role=OperatorRole.PROJECTION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="share_expert_down_proj_time",
            resource_class=ResourceClass.COMP,
            tp_mode=TensorParallelMode.FFN_TP,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
            calibration_key="share_expert_down_proj",
        ),
    ),
)


MEMORY_FAMILY = OperatorFamilySpec(
    family_id="memory",
    display_name="Replicated Memory Ops",
    supported_variants=("replicated",),
    resource_class=ResourceClass.MEMORY,
    operators=(
        OperatorSpec(
            name="input_layernorm",
            role=OperatorRole.NORMALIZATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="attn_norm_time",
            resource_class=ResourceClass.MEMORY,
            tp_mode=TensorParallelMode.REPLICATED,
        ),
        OperatorSpec(
            name="post_attention_layernorm",
            role=OperatorRole.NORMALIZATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            execution_time_attr="mlp_norm_time",
            resource_class=ResourceClass.MEMORY,
            tp_mode=TensorParallelMode.REPLICATED,
        ),
        OperatorSpec(
            name="add_attn_residual",
            role=OperatorRole.RESIDUAL,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            profiling_key="add",
            precision_op="add",
            execution_time_attr="add_attn_residual_time",
            resource_class=ResourceClass.MEMORY,
            tp_mode=TensorParallelMode.REPLICATED,
        ),
        OperatorSpec(
            name="add_ffn_residual",
            role=OperatorRole.RESIDUAL,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            profiling_key="add",
            precision_op="add",
            execution_time_attr="add_ffn_residual_time",
            resource_class=ResourceClass.MEMORY,
            tp_mode=TensorParallelMode.REPLICATED,
        ),
        OperatorSpec(
            name="emb",
            role=OperatorRole.EMBEDDING,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMPUTE,
            predictor_target=False,
            e2e_trace_target=False,
            execution_time_attr=None,
            resource_class=ResourceClass.MEMORY,
            tp_mode=TensorParallelMode.REPLICATED,
        ),
    ),
)


def _effective_tokens(ctx: CommPayloadContext) -> int:
    return int(ctx.batch.get_effective_total_tokens_rounded(ctx.cluster_type))


def _hidden_state_bytes(ctx: CommPayloadContext, collective: str) -> int:
    data_size_bytes = int(ctx.model_config.embedding_dim) * 2 * _effective_tokens(ctx)
    return int(
        ctx.quantization_manager.adjust_tensor_size(
            collective,
            data_size_bytes,
            ctx.cluster_type,
        )
    )


def _tp_allreduce_payload_bytes(ctx: CommPayloadContext) -> int:
    return _hidden_state_bytes(ctx, "allreduce")


def _pp_send_recv_payload_bytes(ctx: CommPayloadContext) -> int:
    return _hidden_state_bytes(ctx, "send_recv")


def _moe_tp_allgather_payload_bytes(ctx: CommPayloadContext) -> int:
    data_size_bytes = int(ctx.model_config.embedding_dim) * 2 * _effective_tokens(ctx)
    moe_tp_size = int(ctx.replica_config.moe_tensor_parallel_size)
    if data_size_bytes % moe_tp_size != 0:
        raise ValueError(
            "MoE TP allgather requires per-device tensor bytes to be divisible by "
            f"moe_tp_size, got data_size_bytes={data_size_bytes}, "
            f"moe_tp_size={moe_tp_size}"
        )
    return int(
        ctx.quantization_manager.adjust_tensor_size(
            "allgather",
            data_size_bytes // moe_tp_size,
            ctx.cluster_type,
        )
    )


def _expert_parallel_payload_bytes(ctx: CommPayloadContext) -> int:
    per_expert_tokens = getattr(ctx.batch, "per_expert_tokens", None)
    if per_expert_tokens:
        routed_tokens = sum(int(token_count) for token_count in per_expert_tokens.values())
    else:
        router_topk = int(getattr(ctx.replica_config, "router_topk", 0) or 0)
        if router_topk <= 0:
            router_topk = int(getattr(ctx.model_config, "num_experts_per_tok", 0) or 0)
        if router_topk <= 0:
            raise ValueError("router_topk must be set for expert-parallel communication")
        routed_tokens = _effective_tokens(ctx) * router_topk
    data_size_bytes = int(ctx.model_config.embedding_dim) * 2 * int(routed_tokens)
    return int(
        ctx.quantization_manager.adjust_tensor_size(
            "expert_parallel_communication",
            data_size_bytes,
            ctx.cluster_type,
        )
    )


def _attn_tp_devices(ctx: CommPayloadContext) -> int:
    return int(ctx.replica_config.attn_tensor_parallel_size)


def _moe_tp_devices(ctx: CommPayloadContext) -> int:
    return int(ctx.replica_config.moe_tensor_parallel_size)


def _moe_ep_devices(ctx: CommPayloadContext) -> int:
    return int(ctx.replica_config.moe_expert_parallel_size)


def _pp_devices(ctx: CommPayloadContext) -> int:
    return 2


COMM_FAMILY = OperatorFamilySpec(
    family_id="comm",
    display_name="Collective Communication",
    supported_variants=("collective",),
    resource_class=ResourceClass.COMM,
    operators=(
        CommOperatorSpec(
            name="attn_tensor_parallel_allreduce",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="attn_tensor_parallel_allreduce_time",
            precision_op="allreduce",
            collective_alias="allreduce",
            comm_group="attn_tp",
            comm_domain="ATTN_TP",
            payload_builder=_tp_allreduce_payload_bytes,
            num_devices_builder=_attn_tp_devices,
            apply_allreduce_launch_overhead_strip=True,
        ),
        CommOperatorSpec(
            name="mlp_tensor_parallel_allreduce",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="moe_tensor_parallel_allreduce_time",
            precision_op="allreduce",
            collective_alias="allreduce",
            comm_group="attn_tp",
            comm_domain="ATTN_TP",
            payload_builder=_tp_allreduce_payload_bytes,
            num_devices_builder=_attn_tp_devices,
            apply_allreduce_launch_overhead_strip=True,
        ),
        CommOperatorSpec(
            name="moe_tensor_parallel_allreduce",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="moe_tensor_parallel_allreduce_time",
            precision_op="allreduce",
            collective_alias="allreduce",
            comm_group="moe_tp",
            comm_domain="MOE_TP",
            payload_builder=_tp_allreduce_payload_bytes,
            num_devices_builder=_moe_tp_devices,
            apply_allreduce_launch_overhead_strip=True,
        ),
        CommOperatorSpec(
            name="moe_tensor_parallel_allgather",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="tensor_parallel_allgather_time",
            precision_op="allgather",
            collective_alias="allgather",
            comm_group="moe_tp",
            comm_domain="MOE_TP",
            payload_builder=_moe_tp_allgather_payload_bytes,
            num_devices_builder=_moe_tp_devices,
        ),
        CommOperatorSpec(
            name="share_expert_tensor_parallel_allreduce",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="share_expert_tensor_parallel_allreduce_time",
            precision_op="allreduce",
            collective_alias="allreduce",
            comm_group="moe_tp",
            comm_domain="MOE_TP",
            payload_builder=_tp_allreduce_payload_bytes,
            num_devices_builder=_moe_tp_devices,
            apply_allreduce_launch_overhead_strip=True,
        ),
        CommOperatorSpec(
            name="expert_parallel_allreduce",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="expert_parallel_alltoall_time",
            precision_op="allreduce",
            collective_alias="allreduce",
            comm_group="moe_ep",
            comm_domain="EP",
            payload_builder=_tp_allreduce_payload_bytes,
            num_devices_builder=_moe_ep_devices,
            apply_allreduce_launch_overhead_strip=True,
        ),
        CommOperatorSpec(
            name="expert_parallel_alltoall",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="expert_parallel_alltoall_time",
            precision_op="expert_parallel_communication",
            collective_alias="alltoall",
            comm_group="moe_ep",
            comm_domain="EP",
            payload_builder=_expert_parallel_payload_bytes,
            num_devices_builder=_moe_ep_devices,
        ),
        CommOperatorSpec(
            name="expert_parallel_alltoall_dispatch",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="expert_parallel_alltoall_time",
            precision_op="expert_parallel_communication",
            collective_alias="alltoall",
            comm_group="moe_ep",
            comm_domain="EP",
            payload_builder=_expert_parallel_payload_bytes,
            num_devices_builder=_moe_ep_devices,
        ),
        CommOperatorSpec(
            name="expert_parallel_alltoall_combine",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="expert_parallel_alltoall_time",
            precision_op="expert_parallel_communication",
            collective_alias="alltoall",
            comm_group="moe_ep",
            comm_domain="EP",
            payload_builder=_expert_parallel_payload_bytes,
            num_devices_builder=_moe_ep_devices,
        ),
        CommOperatorSpec(
            name="pipeline_parallel_send_recv",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            resource_class=ResourceClass.COMM,
            execution_time_attr="pipeline_parallel_send_recv_time",
            precision_op="send_recv",
            collective_alias="send_recv",
            comm_group="pp",
            comm_domain="PP",
            payload_builder=_pp_send_recv_payload_bytes,
            num_devices_builder=_pp_devices,
        ),
    ),
)


KV_TRANSFER_FAMILY = OperatorFamilySpec(
    family_id="kv_transfer",
    display_name="KV Cache Transfer",
    supported_variants=("request_level",),
    resource_class=ResourceClass.COMM,
    execution_enabled=False,
    disabled_reason=(
        "KV cache transfer is emitted as a request-level trace event outside "
        "batch ExecutionTime prediction"
    ),
    operators=(
        OperatorSpec(
            name="kv_cache_transfer",
            role=OperatorRole.COMMUNICATION,
            phases=_ALL_PHASES,
            trace_kind=TraceKind.COMM,
            predictor_target=False,
            profiling_target=False,
            e2e_trace_target=False,
            execution_time_attr=None,
            precision_op="kv_cache_transfer",
            resource_class=ResourceClass.COMM,
            tp_mode=TensorParallelMode.REPLICATED,
        ),
    ),
)


def get_comm_operator(op_name: str) -> CommOperatorSpec:
    for operator in COMM_FAMILY.operators:
        if operator.name == op_name:
            if not isinstance(operator, CommOperatorSpec):
                raise TypeError(f"COMM operator {op_name} is not a CommOperatorSpec")
            return operator
    raise ValueError(f"Unsupported COMM op: {op_name}")


OPERATOR_REGISTRY = OperatorRegistry()
for _family in (
    DENSE_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    MEMORY_FAMILY,
    FFN_FAMILY,
    MOE_FAMILY,
    SHARE_EXPERT_FAMILY,
    COMM_FAMILY,
    KV_TRANSFER_FAMILY,
):
    OPERATOR_REGISTRY.register(_family)


def get_operator_family(family_id: str) -> OperatorFamilySpec:
    return OPERATOR_REGISTRY.get_family(family_id)


def iter_operator_families() -> tuple[OperatorFamilySpec, ...]:
    return OPERATOR_REGISTRY.iter_families()


def iter_execution_enabled_operator_families() -> tuple[OperatorFamilySpec, ...]:
    return OPERATOR_REGISTRY.iter_execution_enabled_families()


def get_family_profiling_names(family: OperatorFamilySpec) -> tuple[str, ...]:
    return tuple(dict.fromkeys(operator.profiling_name() for operator in family.profiling_ops()))


def get_family_profiling_name_set(family: OperatorFamilySpec) -> frozenset[str]:
    return frozenset(get_family_profiling_names(family))
