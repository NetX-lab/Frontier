"""Declarative operator family for gated-delta-network sequence mixing."""

from __future__ import annotations

from dataclasses import dataclass

from frontier.operators.spec import (
    OperatorFamilySpec,
    OperatorPhase,
    OperatorRole,
    OperatorSpec,
    ProjectionOwnership,
    ResourceClass,
)


@dataclass(frozen=True)
class GatedDeltaNetFamilySpec(OperatorFamilySpec):
    """GDN operator catalog plus its profiling feature contract."""

    required_profiling_feature_columns: tuple[str, ...] = ()


_ALL_PHASES = (
    OperatorPhase.PREFILL,
    OperatorPhase.DECODE,
    OperatorPhase.MIXED,
)


GATED_DELTA_NET_FAMILY = GatedDeltaNetFamilySpec(
    family_id="gated_delta_net",
    display_name="Gated Delta Network",
    supported_variants=("qwen3_5",),
    operators=(
        OperatorSpec(
            name="gdn_input_projections",
            role=OperatorRole.PROJECTION,
            phases=_ALL_PHASES,
            execution_time_attr="attention_layer_pre_proj_execution_time",
            resource_class=ResourceClass.COMP,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
        ),
        OperatorSpec(
            name="gdn_core_prefill",
            role=OperatorRole.PREFILL_KERNEL,
            phases=(OperatorPhase.PREFILL,),
            execution_time_attr="attention_prefill_execution_time",
            resource_class=ResourceClass.COMP,
        ),
        OperatorSpec(
            name="gdn_core_decode",
            role=OperatorRole.DECODE_KERNEL,
            phases=(OperatorPhase.DECODE,),
            execution_time_attr="attention_decode_execution_time",
            resource_class=ResourceClass.COMP,
        ),
        OperatorSpec(
            name="gdn_core_mixed",
            role=OperatorRole.MIXED_KERNEL,
            phases=(OperatorPhase.MIXED,),
            # The structured operator map owns the physical identity.  The
            # legacy prefill field is used only as a compatibility carrier.
            execution_time_attr="attention_prefill_execution_time",
            resource_class=ResourceClass.COMP,
        ),
        OperatorSpec(
            name="gdn_output_projection",
            role=OperatorRole.PROJECTION,
            phases=_ALL_PHASES,
            execution_time_attr="attention_layer_post_proj_execution_time",
            resource_class=ResourceClass.COMP,
            projection_ownership=ProjectionOwnership.OUTSIDE_ATTENTION,
        ),
    ),
    resource_class=ResourceClass.COMP,
    profiling_order=(
        "gdn_input_projections",
        "gdn_core_prefill",
        "gdn_core_decode",
        "gdn_core_mixed",
        "gdn_output_projection",
    ),
    required_profiling_feature_columns=(
        "measurement_type",
        "model_architecture_profile",
        "quant_signature",
        "device",
        "runtime_stack_signature",
        "gdn_runtime_backend",
        "gdn_rank_aggregation",
        "gdn_prefill_backend",
        "gdn_decode_backend",
        "gqa_interleaved_layout",
        "packed_recurrent_decode",
        "model_dtype",
        "conv_state_dtype",
        "recurrent_state_dtype",
        "num_tensor_parallel_workers",
        "hidden_size",
        "conv_kernel_size",
        "key_head_dim",
        "value_head_dim",
        "num_key_heads",
        "num_value_heads",
        "batch_size",
        "batch_num_tokens",
        "batch_num_prefill_tokens",
        "batch_num_decode_tokens",
        "max_query_len",
        "has_initial_state",
    ),
)
