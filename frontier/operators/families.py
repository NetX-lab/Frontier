from __future__ import annotations

from functools import cache

from frontier.attention.families import (
    DEEPSEEK_V4_ATTENTION_FAMILY,
    DENSE_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
)
from frontier.config.parallel_semantics import (
    resolve_shared_expert_tensor_parallel_mode,
    resolve_shared_expert_tensor_parallel_size,
)
from frontier.config.precision_type import PrecisionType
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
    ZeroPayloadPolicy,
)
from frontier.moe_ep_workload import resolve_ep_lane_workload
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


DEEPSEEK_V4_MHC_FAMILY = OperatorFamilySpec(
    family_id="deepseek_v4_mhc",
    display_name="DeepSeek-V4 mHC and Model Endpoints",
    supported_variants=("first", "middle", "last", "first_last"),
    execution_enabled=False,
    disabled_reason="DeepSeek-V4 mHC and endpoint scopes are source inventory only",
    # Fused pre/post scopes own the layer norms and residual mixing. Generic
    # MEMORY_FAMILY norm/add scopes would count that work a second time.
    operators=tuple(
        OperatorSpec(
            name=name,
            role=role,
            phases=_ALL_PHASES,
            predictor_target=False,
            profiling_target=False,
            e2e_trace_target=False,
            resource_class=resource,
            tp_mode=tp_mode,
            projection_ownership=(
                ProjectionOwnership.OUTSIDE_ATTENTION
                if role is OperatorRole.PROJECTION
                else ProjectionOwnership.NOT_PROJECTION
            ),
        )
        for name, role, resource, tp_mode in (
            ("dsv4_embedding", OperatorRole.EMBEDDING, ResourceClass.MEMORY,
             TensorParallelMode.ATTENTION_TP),
            ("dsv4_mhc_first_pre", OperatorRole.NORMALIZATION, ResourceClass.COMP,
             TensorParallelMode.REPLICATED),
            ("dsv4_mhc_ffn_post_attn_pre", OperatorRole.NORMALIZATION, ResourceClass.COMP,
             TensorParallelMode.REPLICATED),
            ("dsv4_mhc_attn_post_ffn_pre", OperatorRole.NORMALIZATION, ResourceClass.COMP,
             TensorParallelMode.REPLICATED),
            ("dsv4_mhc_final_post", OperatorRole.RESIDUAL, ResourceClass.COMP,
             TensorParallelMode.REPLICATED),
            ("dsv4_mhc_head", OperatorRole.PROJECTION, ResourceClass.COMP,
             TensorParallelMode.REPLICATED),
            ("dsv4_final_norm", OperatorRole.NORMALIZATION, ResourceClass.MEMORY,
             TensorParallelMode.REPLICATED),
            ("dsv4_lm_head", OperatorRole.PROJECTION, ResourceClass.COMP,
             TensorParallelMode.ATTENTION_TP),
        )
    ),
)


def _effective_tokens(ctx: CommPayloadContext) -> int:
    return int(ctx.batch.get_effective_total_tokens_rounded(ctx.cluster_type))


def _resolve_lane_workload(ctx: CommPayloadContext):
    """Resolve an explicit physical EP lane for lane-scoped payloads."""

    if ctx.lane_workload is not None:
        return resolve_ep_lane_workload(ctx.lane_workload, required=True)
    return resolve_ep_lane_workload(ctx.batch, required=False)


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


def _moe_tp_allreduce_payload_bytes(ctx: CommPayloadContext) -> int:
    # MoE-TP all-reduce operates on the shared hidden-state domain before
    # expert routing.  A physical lane descriptor is therefore irrelevant to
    # this payload; only the source batch's compute-effective width applies.
    data_size_bytes = int(ctx.model_config.embedding_dim) * 2 * _effective_tokens(ctx)
    return int(
        ctx.quantization_manager.adjust_tensor_size(
            "allreduce",
            data_size_bytes,
            ctx.cluster_type,
        )
    )


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


def _dsv4_sp_local_hidden_bytes(ctx: CommPayloadContext) -> int:
    """Source BF16 shard width from sequence_parallel.sp_shard padding."""

    from frontier.entities.batch import EPBatchGroup

    if (
        ctx.model_config.get_model_architecture_profile().profile_id != "deepseek_v4"
        or ctx.model_config.get_default_precision() is not PrecisionType.BF16
    ):
        raise ValueError("DeepSeek-V4 SP source payload requires the V4 BF16 contract")
    if ctx.cluster_type is not ClusterType.MONOLITHIC:
        raise ValueError("DeepSeek-V4 SP source payload requires co-location")
    replica = ctx.replica_config
    for name in (
        "num_pipeline_stages", "attn_tensor_parallel_size", "attn_dp",
        "moe_tensor_parallel_size", "moe_expert_parallel_size",
    ):
        value = getattr(replica, name)
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"DeepSeek-V4 SP source payload requires positive integer {name}"
            )
    tp_size = replica.attn_tensor_parallel_size
    if (
        replica.num_pipeline_stages != 1 or tp_size <= 1 or replica.attn_dp <= 1
        or replica.moe_tensor_parallel_size != 1
        or replica.moe_expert_parallel_size != tp_size * replica.attn_dp
    ):
        raise ValueError(
            "DeepSeek-V4 SP source payload requires PP1, attention TP>1/DP>1, "
            "MoE TP1 and EP=attention TP*DP"
        )
    if (
        isinstance(ctx.batch, EPBatchGroup)
        or ctx.lane_workload is not None
        or resolve_ep_lane_workload(ctx.batch, required=False) is not None
    ):
        raise ValueError(
            "DeepSeek-V4 SP payload requires a source DP attention batch, "
            "not a materialized EP destination or aggregate group"
        )
    tokens = ctx.batch.get_effective_total_tokens_rounded(ctx.cluster_type)
    hidden = ctx.model_config.embedding_dim
    if type(tokens) is not int or tokens < 0:
        raise ValueError("DeepSeek-V4 SP source rows must be a non-negative integer")
    if type(hidden) is not int or hidden <= 0:
        raise ValueError("DeepSeek-V4 SP hidden width must be a positive integer")
    local_rows = (tokens + tp_size - 1) // tp_size
    # These collectives carry BF16 activations irrespective of weight dtype.
    return local_rows * hidden * int(PrecisionType.BF16.bytes_per_element)


def _dsv4_sp_attention_reduce_scatter_bytes(ctx: CommPayloadContext) -> int:
    # CC reduce_scatter consumes the full padded input on each TP rank.
    return _dsv4_sp_local_hidden_bytes(ctx) * ctx.replica_config.attn_tensor_parallel_size


def _dsv4_sp_final_mhc_allgather_bytes(ctx: CommPayloadContext) -> int:
    local_hidden_bytes = _dsv4_sp_local_hidden_bytes(ctx)
    streams = ctx.model_config.hc_mult
    if type(streams) is not int or streams <= 0:
        raise ValueError("DeepSeek-V4 SP mHC stream count must be a positive integer")
    # model.forward gathers [local_rows, hc_mult, hidden] before head collapse.
    return local_hidden_bytes * streams


def _expert_parallel_payload_bytes(ctx: CommPayloadContext) -> int:
    lane_workload = _resolve_lane_workload(ctx)
    if lane_workload is None:
        raise ValueError(
            "expert-parallel all-to-all payload requires an EPLaneWorkload "
            "descriptor"
        )
    routed_tokens = lane_workload.routed_token_count
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


def _shared_expert_tp_devices(ctx: CommPayloadContext) -> int:
    return resolve_shared_expert_tensor_parallel_size(
        cluster_type=ctx.cluster_type,
        replica_config=ctx.replica_config,
        model_config=ctx.model_config,
    )


def _shared_expert_tp_group(ctx: CommPayloadContext) -> str:
    if ctx.cluster_type is ClusterType.DECODE_FFN:
        return "moe_tp"

    tp_mode = resolve_shared_expert_tensor_parallel_mode(
        model_config=ctx.model_config
    )
    if tp_mode is TensorParallelMode.ATTENTION_TP:
        return "attn_tp"
    if tp_mode in {TensorParallelMode.FFN_TP, TensorParallelMode.MOE_TP}:
        return "moe_tp"
    raise ValueError(
        f"Shared-expert all-reduce does not support TP mode {tp_mode!r}"
    )


def _shared_expert_tp_domain(ctx: CommPayloadContext) -> str:
    return "MOE_TP" if _shared_expert_tp_group(ctx) == "moe_tp" else "ATTN_TP"


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
            payload_builder=_moe_tp_allreduce_payload_bytes,
            num_devices_builder=_moe_tp_devices,
            apply_allreduce_launch_overhead_strip=True,
            zero_payload_policy=ZeroPayloadPolicy.EXACT_NOOP,
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
            comm_group="attn_tp",
            comm_group_builder=_shared_expert_tp_group,
            comm_domain="ATTN_TP",
            comm_domain_builder=_shared_expert_tp_domain,
            payload_builder=_tp_allreduce_payload_bytes,
            num_devices_builder=_shared_expert_tp_devices,
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
        *(
            CommOperatorSpec(
                name=name,
                role=OperatorRole.COMMUNICATION,
                phases=_ALL_PHASES,
                trace_kind=TraceKind.COMM,
                resource_class=ResourceClass.COMM,
                predictor_target=False,
                profiling_target=False,
                e2e_trace_target=False,
                precision_op=alias,
                tp_mode=TensorParallelMode.ATTENTION_TP,
                collective_alias=alias,
                comm_group="attn_tp",
                comm_domain="ATTN_TP",
                payload_builder=payload_builder,
                num_devices_builder=_attn_tp_devices,
            )
            for name, alias, payload_builder in (
                ("dsv4_sp_attention_allgather", "allgather", _dsv4_sp_local_hidden_bytes),
                (
                    "dsv4_sp_attention_reduce_scatter", "reduce_scatter",
                    _dsv4_sp_attention_reduce_scatter_bytes,
                ),
                (
                    "dsv4_sp_final_mhc_allgather", "allgather",
                    _dsv4_sp_final_mhc_allgather_bytes,
                ),
            )
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
    DEEPSEEK_V4_ATTENTION_FAMILY,
    DEEPSEEK_V4_MHC_FAMILY,
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


@cache
def _get_registered_family_profiling_name_set(
    family_id: str,
) -> frozenset[str]:
    return frozenset(get_family_profiling_names(get_operator_family(family_id)))


def get_family_profiling_name_set(family: OperatorFamilySpec) -> frozenset[str]:
    try:
        registered_family = get_operator_family(family.family_id)
    except ValueError:
        registered_family = None
    if registered_family is not family:
        return frozenset(get_family_profiling_names(family))
    return _get_registered_family_profiling_name_set(family.family_id)
