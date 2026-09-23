"""Module-level helpers shared by the MoE execution-time predictor parts."""

import math
import pandas as pd

from frontier.entities.time_components import MoEOperatorTimes
from frontier.moe_gating_runtime import get_moe_gating_base_model_name
from frontier.operators.families import MOE_FAMILY, get_family_profiling_names
from typing import Mapping


def _normalize_routing_details_for_trace(
    routing_details: Mapping[int, Mapping[int, Mapping[int, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return a strict JSON-safe copy of runtime routing details.

    The matrix checker compares this emitted object with an independently
    materialized sidecar.  The trace must therefore contain the actual
    predictor-owned map, not a derived token allocation or a digest.
    """

    if not isinstance(routing_details, Mapping) or not routing_details:
        raise ValueError("routing_details trace payload must be a non-empty mapping")
    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for replica_id, per_layer in routing_details.items():
        if type(replica_id) is not int or replica_id < 0:
            raise ValueError(
                "routing_details trace replica IDs must be non-negative integers"
            )
        if not isinstance(per_layer, Mapping) or not per_layer:
            raise ValueError(
                f"routing_details trace replica {replica_id} has no layer map"
            )
        normalized_layers: dict[str, dict[str, float]] = {}
        for layer_id, per_expert in per_layer.items():
            if type(layer_id) is not int or layer_id < 0:
                raise ValueError(
                    "routing_details trace layer IDs must be non-negative integers"
                )
            if not isinstance(per_expert, Mapping) or not per_expert:
                raise ValueError(
                    f"routing_details trace layer {layer_id} has no expert map"
                )
            normalized_experts: dict[str, float] = {}
            for expert_id, ratio in per_expert.items():
                if type(expert_id) is not int or expert_id < 0:
                    raise ValueError(
                        "routing_details trace expert IDs must be non-negative integers"
                    )
                value = float(ratio)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "routing_details trace ratios must be finite and non-negative"
                    )
                normalized_experts[str(expert_id)] = value
            ratio_sum = sum(normalized_experts.values())
            if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "routing_details trace ratios must sum to one "
                    f"for replica={replica_id} layer={layer_id}, got {ratio_sum}"
                )
            normalized_layers[str(layer_id)] = normalized_experts
        normalized[str(replica_id)] = normalized_layers
    return normalized


def _get_moe_family_model_names() -> list[str]:
    return list(get_family_profiling_names(MOE_FAMILY))


def _get_moe_family_operator_by_model_name(model_name: str):
    moe_ops = {
        operator.profiling_name(): operator
        for operator in MOE_FAMILY.profiling_ops()
    }
    if model_name not in moe_ops:
        raise ValueError(f"Unsupported MoE op: {model_name}")
    return moe_ops[model_name]


def _get_moe_gating_family_model_names() -> list[str]:
    return [
        operator.profiling_name()
        for operator in MOE_FAMILY.profiling_ops()
        if operator.precision_name() == "moe_gating"
    ]


_MOE_GATING_OPERATOR_NAMES = frozenset(
    operator.name
    for operator in MOE_FAMILY.profiling_ops()
    if operator.precision_name() == "moe_gating"
)


def _get_prefill_hot_moe_gating_model_names() -> list[str]:
    return [
        f"{model_name}__prefill_hot"
        for model_name in _get_moe_gating_family_model_names()
    ]


def _is_moe_gating_family_model_name(model_name: str) -> bool:
    base_model_name = get_moe_gating_base_model_name(model_name)
    return _get_moe_family_operator_by_model_name(
        base_model_name
    ).precision_name() == "moe_gating"


def _build_moe_operator_times(
    *,
    mlp_norm_time: float,
    moe_gating_linear_time: float,
    moe_gating_routing_topk_time: float,
    moe_shuffling_time: float,
    moe_grouped_gemm_time: float,
    share_expert_up_proj_time: float = 0.0,
    share_expert_act_time: float = 0.0,
    share_expert_down_proj_time: float = 0.0,
    include_share_expert: bool = False,
) -> MoEOperatorTimes:
    op_times = {
        "post_attention_layernorm": mlp_norm_time,
        "moe_gating_linear": moe_gating_linear_time,
        "moe_gating_routing_topk": moe_gating_routing_topk_time,
        "moe_shuffling": moe_shuffling_time,
        "moe_grouped_gemm": moe_grouped_gemm_time,
    }
    if include_share_expert:
        op_times.update(
            {
                "share_expert_up_proj": share_expert_up_proj_time,
                "share_expert_act": share_expert_act_time,
                "share_expert_down_proj": share_expert_down_proj_time,
            }
        )
    return MoEOperatorTimes(op_times=op_times)


def _validate_moe_columns(moe_df: pd.DataFrame) -> None:
    """
    Validate that MoE DataFrame contains required split gating columns.

    This function enforces fail-fast behavior by rejecting legacy moe_gating
    column format and requiring the split columns (moe_gating_linear and
    moe_gating_routing_topk).

    Args:
        moe_df: DataFrame containing MoE profiling data

    Raises:
        ValueError: If required split columns are missing or if legacy
                   moe_gating column is present without split columns
    """
    required_columns = [
        f"time_stats.{operator_name}.median"
        for operator_name in get_family_profiling_names(MOE_FAMILY)
    ]

    missing_columns = [col for col in required_columns if col not in moe_df.columns]

    if missing_columns:
        # Check if legacy moe_gating column exists (for better error message)
        legacy_col = "time_stats.moe_gating.median"
        if legacy_col in moe_df.columns:
            raise ValueError(
                f"Missing required MoE columns: {missing_columns}. "
                f"Found legacy '{legacy_col}' column which is no longer supported. "
                f"Re-run MoE profiling with split gating scopes enabled to generate "
                f"'moe_gating_linear' and 'moe_gating_routing_topk' columns."
            )
        else:
            raise ValueError(
                f"Missing required MoE columns: {missing_columns}. "
                f"Re-run MoE profiling with split gating scopes enabled."
            )
