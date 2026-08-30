import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import AttentionOperatorRole
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_metric_name_by_role,
)
from frontier.entities import Batch, EPBatchGroup, ExecutionTime
from frontier.entities.time_components import (
    AttentionTime,
    CommunicationOperatorTimes,
    MLPOperatorTimes,
    MoEOperatorTimes,
    MoETime,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.logger import init_logger
from frontier.model_architectures import (
    LayerKind,
    ResolvedLayerContract,
    ResidualAddPolicy,
)
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT,
    filter_moe_gating_rows_by_runtime_context,
    get_moe_gating_base_model_name,
    get_moe_gating_prediction_model_context,
    get_moe_gating_prediction_model_name,
    has_prefill_warmed_moe_gating_rows,
    should_enable_prefill_warmed_moe_gating_contract,
    should_use_prefill_warmed_moe_gating_context,
)
from frontier.moe_routing_runtime import (
    filter_moe_gating_routing_topk_rows,
    resolve_moe_gating_routing_runtime_path,
)
from frontier.moe_ep_workload import (
    EPLaneWorkload,
    LayerEPWorkload,
    build_contiguous_expert_ownership,
    materialize_layer_ep_workload,
    resolve_ep_lane_workload,
    resolve_routing_details,
)
from frontier.operators.families import (
    MOE_FAMILY,
    get_family_profiling_names,
    get_comm_operator,
    is_moe_operator_ep_agnostic,
    resolve_moe_operator_tp_key,
)

if TYPE_CHECKING:
    from frontier.entities import EPBatchGroup
    from frontier.cc_backend import BaseCCBackend
from frontier.config import (
    BaseExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
    BaseReplicaSchedulerConfig,
    get_quantization_manager,
)
from frontier.types import ClusterType
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)

logger = init_logger(__name__)


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


def _get_prefill_warmed_moe_gating_model_names() -> list[str]:
    return [
        get_moe_gating_prediction_model_name(
            model_name,
            requested_context=PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT,
        )
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


class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):
    @staticmethod
    def _emit_routing_details_snapshot(
        cluster_type: ClusterType,
        routing_details: Mapping[int, Mapping[int, Mapping[int, float]]],
    ) -> None:
        """Emit the exact predictor-owned routing map for external validation."""

        normalized = _normalize_routing_details_for_trace(routing_details)
        payload = {
            "schema_version": 1,
            "cluster": cluster_type.name,
            "routing_details": normalized,
        }
        logger.info(
            "[ROUTING-SNAPSHOT] %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def _get_requested_moe_gating_routing_runtime_path(self) -> str:
        return resolve_moe_gating_routing_runtime_path(
            getattr(self, "_moe_routing_distribution_type", "balanced")
        )

    @staticmethod
    def _get_ep_lane_routed_token_count(
        batch: Batch,
        lane_workload: Optional[EPLaneWorkload] = None,
    ) -> Optional[int]:
        """Return an EP lane's routed-token count, or ``None`` for full batches.

        Dummy mode still models the same five-phase EP contract as the
        profiling-backed path.  The lane-local routed compute therefore has
        to depend on the materialized expert map even when the other dummy
        components use fixed structural timings.
        """

        if lane_workload is None:
            lane_workload = resolve_ep_lane_workload(batch, required=False)
        if lane_workload is None:
            return None
        return lane_workload.routed_token_count

    def _admit_routed_ep_aggregate(
        self,
        batch: Batch,
        *,
        routed_moe: bool,
        ep_size: Optional[int] = None,
        router_topk: Optional[int] = None,
        lane_workload: Optional[EPLaneWorkload] = None,
        conservation_context: str = "routed MoE admission",
    ) -> Optional[EPLaneWorkload]:
        """Admit a concrete routed MoE call at the public predictor boundary.

        Concrete predictors own the semantic classification of a call.  Once
        that classification is routed MoE, an EP>1 call must identify one
        physical lane before any mode-specific timing or lookup work begins.
        The helper also validates the descriptor against the active predictor
        topology and the source/lane token ledger. Workload construction
        remains owned by the scheduler/materializer path.
        """
        if type(routed_moe) is not bool:
            raise ValueError("routed_moe must be a bool")
        if not routed_moe:
            return None

        if ep_size is None:
            configured_ep_size = getattr(self, "_moe_ep_size", None)
            if configured_ep_size is None:
                replica_config = getattr(self, "_replica_config", None)
                configured_ep_size = getattr(
                    replica_config,
                    "moe_expert_parallel_size",
                    None,
                )
        else:
            configured_ep_size = ep_size
        if type(configured_ep_size) is not int or configured_ep_size < 1:
            raise ValueError(
                "routed MoE admission requires a positive integer EP size, got "
                f"{configured_ep_size!r}"
            )

        batch_lane_workload = resolve_ep_lane_workload(batch, required=False)
        explicit_lane_workload = (
            resolve_ep_lane_workload(lane_workload, required=True)
            if lane_workload is not None
            else None
        )
        if (
            batch_lane_workload is not None
            and explicit_lane_workload is not None
            and batch_lane_workload != explicit_lane_workload
        ):
            raise ValueError(
                "batch and lane_workload must refer to the same "
                "EPLaneWorkload descriptor"
            )
        resolved_lane_workload = explicit_lane_workload or batch_lane_workload
        if configured_ep_size > 1 and resolved_lane_workload is None:
            raise ValueError(
                "Routed MoE prediction with EP>1 requires an EPLaneWorkload "
                "descriptor"
            )
        if resolved_lane_workload is None:
            return None

        configured_router_topk = (
            getattr(self, "_router_topk", None)
            if router_topk is None
            else router_topk
        )
        if configured_router_topk is not None:
            if (
                type(configured_router_topk) is not int
                or configured_router_topk < 1
            ):
                raise ValueError(
                    "routed MoE admission requires a positive integer router top-k, "
                    f"got {configured_router_topk!r}"
                )
            if resolved_lane_workload.router_topk != configured_router_topk:
                raise ValueError(
                    "lane_workload router_topk does not match predictor topology: "
                    f"descriptor={resolved_lane_workload.router_topk}, "
                    f"predictor={configured_router_topk}"
                )
        else:
            configured_router_topk = resolved_lane_workload.router_topk

        if (
            resolved_lane_workload.moe_expert_parallel_size
            != configured_ep_size
        ):
            raise ValueError(
                "lane_workload EP size does not match predictor topology: "
                f"descriptor={resolved_lane_workload.moe_expert_parallel_size}, "
                f"predictor={configured_ep_size}"
            )

        # A descriptor attached to an EP lane entity already contains routed
        # assignments, so its count must match that entity's physical width.
        # An explicit descriptor paired with an ordinary source batch represents
        # one assignment subset of the aggregate.  The aggregate materializer,
        # rather than this predictor boundary, owns its conservation ledger.
        source_total_num_tokens = getattr(batch, "total_num_tokens", None)
        if source_total_num_tokens is not None:
            if (
                type(source_total_num_tokens) is not int
                or source_total_num_tokens < 0
            ):
                raise ValueError(
                    "routed MoE admission requires batch.total_num_tokens to be "
                    "a non-negative integer, got "
                    f"{source_total_num_tokens!r}"
                )
            if batch_lane_workload is not None:
                expected_routed_token_count = source_total_num_tokens
                if (
                    resolved_lane_workload.routed_token_count
                    != expected_routed_token_count
                ):
                    raise ValueError(
                        f"Token conservation violated in {conservation_context}: "
                        f"allocated {resolved_lane_workload.routed_token_count}, "
                        f"expected {expected_routed_token_count}"
                    )

        return resolved_lane_workload

    @staticmethod
    def _resolve_moe_layer_classification(
        model_config: Any,
        *,
        layer_id: int,
        num_layers: int,
        include_moe: Optional[bool],
        include_ffn: bool,
    ) -> bool:
        """Resolve the routed/dense selector owned by a public predictor call.

        A call with an explicit ``include_moe`` selector already carries its
        classification.  Identity-free multi-layer aggregates use the model
        level ``is_moe`` capability.  A concrete single-layer call needs the
        model-owned ``is_moe_layer`` predicate so mixed-layer models cannot be
        silently treated as routed or dense based on a broad model flag.
        """
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError(
                "num_layers must be a positive integer, "
                f"got {num_layers!r}"
            )
        if type(include_ffn) is not bool:
            raise ValueError("include_ffn must be a bool")
        if include_moe is not None and type(include_moe) is not bool:
            raise ValueError("include_moe must be a bool or None")
        if not include_ffn:
            return False
        if include_moe is not None:
            return include_moe

        model_is_moe = bool(getattr(model_config, "is_moe", False))
        if not model_is_moe:
            return False
        if num_layers != 1:
            return True

        layer_predicate = getattr(model_config, "is_moe_layer", None)
        if not callable(layer_predicate):
            raise ValueError(
                "Concrete MoE layer prediction requires callable "
                "model_config.is_moe_layer(layer_id)"
            )
        return bool(layer_predicate(layer_id))

    def _get_dummy_execution_time(
        self,
        batch: Batch,
        pipeline_stage: int,
        *,
        num_layers: Optional[int] = None,
        layer_id: int = 0,
        layer_ids: Optional[List[int] | tuple[int, ...]] = None,
        include_attention: bool = True,
        include_ffn: bool = True,
        include_moe: Optional[bool] = None,
        lane_workload: Optional[EPLaneWorkload] = None,
    ) -> ExecutionTime:
        """Return fixed dummy ExecutionTime object with MoE-aware fields."""
        configured_num_layers = getattr(self, "_num_layers_per_pipeline_stage", 1)
        effective_num_layers = (
            configured_num_layers if num_layers is None else num_layers
        )
        normalized_layer_ids = self._normalize_stage_layer_ids(
            num_layers=effective_num_layers,
            layer_id=layer_id,
            layer_ids=layer_ids,
        )
        if normalized_layer_ids is not None:
            layer_id = normalized_layer_ids[0]
        if type(include_attention) is not bool:
            raise ValueError("include_attention must be a bool")
        if type(include_ffn) is not bool:
            raise ValueError("include_ffn must be a bool")
        if include_moe is not None and type(include_moe) is not bool:
            raise ValueError("include_moe must be a bool or None")
        if not include_ffn and include_moe is True:
            raise ValueError("include_moe cannot be true when include_ffn is false")
        base_time = self._dummy_execution_time
        model_is_moe = bool(getattr(self._model_config, "is_moe", False))
        is_moe = include_ffn and (
            model_is_moe if include_moe is None else include_moe
        )
        routed_token_count = (
            self._get_ep_lane_routed_token_count(
                batch,
                lane_workload=lane_workload,
            )
            if is_moe
            else None
        )
        zero_routed_ep_lane = is_moe and routed_token_count == 0
        architecture_profile = self._get_model_architecture_profile()
        share_expert_enabled = bool(
            include_ffn and is_moe and self._model_config.supports_share_expert()
        )

        attn_tp_size = self._replica_config.attn_tensor_parallel_size
        moe_tp_size = self._replica_config.moe_tensor_parallel_size
        moe_ep_size = self._replica_config.moe_expert_parallel_size

        # COMM_SKIP: TP all-reduce not needed when tp_size <= 1 (no tensor sharding)
        attn_tp_allreduce_time = (
            base_time if include_attention and attn_tp_size > 1 else 0.0
        )
        # MoE TP all-reduce covers the shared pre-routing hidden-state domain.
        # A zero-routed physical lane still participates in that collective.
        # Dense MONOLITHIC FFN uses the existing attention-TP owner for the
        # same legacy ``moe_tensor_parallel_allreduce_time`` field.
        ffn_tp_allreduce_time = base_time if include_ffn and (
            (moe_tp_size > 1) if is_moe else (attn_tp_size > 1)
        ) else 0.0
        moe_grouped_gemm_time = (
            0.0 if zero_routed_ep_lane else base_time
        ) if is_moe else 0.0
        # EP=1 retains the named protocol phases with zero collective cost.
        expert_parallel_phase_time = (
            base_time if is_moe and moe_ep_size > 1 else 0.0
        )
        expert_parallel_comm_time = expert_parallel_phase_time * 2

        # Attention-DP MoE gather/scatter is retired.  MoE communication is
        # represented by the Replica-local EP wave instead.
        attn_dp_size = int(
            getattr(self._replica_config, "attn_dp", 1)
        )
        if attn_dp_size != 1:
            raise ValueError(
                "MoE attention-DP communication is retired; expected "
                f"attn_dp=1, got {attn_dp_size}"
            )
        dp_input_allreduce_time = 0.0
        dp_output_allreduce_time = 0.0

        ffn_tp_allgather_time = 0.0
        share_expert_tp_allreduce_time = 0.0
        if is_moe and architecture_profile.moe_tensor_parallel_allgather_op and moe_tp_size > 1:
            ffn_tp_allgather_time = base_time
            if (
                share_expert_enabled
                and architecture_profile.share_expert_tensor_parallel_allreduce_op
            ):
                share_expert_tp_allreduce_time = base_time

        add_time = base_time if include_ffn else 0.0
        add_attn_residual_time = 0.0
        add_ffn_residual_time = 0.0
        if include_ffn and architecture_profile.residual_add_policy is ResidualAddPolicy.FFN_RESIDUAL_ONLY:
            add_attn_residual_time = 0.0
            add_ffn_residual_time = base_time
            add_time = 0.0

        share_expert_time = base_time if share_expert_enabled else 0.0
        pp_stage_boundary_handoff_time = (
            base_time
            if pipeline_stage < self._replica_config.num_pipeline_stages - 1
            else 0.0
        )

        if is_moe:
            communication_operator_times = CommunicationOperatorTimes(
                {
                    "expert_parallel_alltoall_dispatch": expert_parallel_phase_time,
                    "expert_parallel_alltoall_combine": expert_parallel_phase_time,
                }
            )
            mlp_operator_times = None
            moe_operator_times = _build_moe_operator_times(
                mlp_norm_time=base_time,
                moe_gating_linear_time=base_time * 0.5,
                moe_gating_routing_topk_time=base_time * 0.5,
                moe_shuffling_time=0.0 if zero_routed_ep_lane else base_time,
                moe_grouped_gemm_time=moe_grouped_gemm_time,
                share_expert_up_proj_time=share_expert_time,
                share_expert_act_time=share_expert_time,
                share_expert_down_proj_time=share_expert_time,
                include_share_expert=share_expert_enabled,
            )
        elif include_ffn:
            dense_communication_operator_times: dict[str, float] = {}
            if include_attention and attn_tp_allreduce_time > 0.0:
                dense_communication_operator_times[
                    "attn_tensor_parallel_allreduce"
                ] = attn_tp_allreduce_time
            if ffn_tp_allreduce_time > 0.0:
                dense_communication_operator_times[
                    "mlp_tensor_parallel_allreduce"
                ] = ffn_tp_allreduce_time
            communication_operator_times = CommunicationOperatorTimes(
                dense_communication_operator_times
            )
            mlp_operator_times = MLPOperatorTimes(
                {
                    "post_attention_layernorm": base_time,
                    "mlp_up_proj": base_time,
                    "mlp_act": base_time,
                    "mlp_down_proj": base_time,
                }
            )
            moe_operator_times = None
        else:
            attention_only_communication_operator_times: dict[str, float] = {}
            if include_attention and attn_tp_allreduce_time > 0.0:
                attention_only_communication_operator_times[
                    "attn_tensor_parallel_allreduce"
                ] = attn_tp_allreduce_time
            communication_operator_times = CommunicationOperatorTimes(
                attention_only_communication_operator_times
            )
            mlp_operator_times = None
            moe_operator_times = None

        return ExecutionTime(
            num_layers_per_pipeline_stage=effective_num_layers,
            attention_rope_execution_time=(base_time if include_attention else 0.0),
            attention_kv_cache_save_execution_time=(
                base_time if include_attention else 0.0
            ),
            attention_decode_execution_time=(base_time if include_attention else 0.0),
            attention_prefill_execution_time=(
                base_time if include_attention else 0.0
            ),
            attention_layer_pre_proj_execution_time=(
                base_time if include_attention else 0.0
            ),
            attention_layer_post_proj_execution_time=(
                base_time if include_attention else 0.0
            ),
            attn_norm_time=base_time if include_attention else 0.0,
            mlp_norm_time=base_time if include_ffn else 0.0,
            add_time=add_time,
            add_attn_residual_time=add_attn_residual_time,
            add_ffn_residual_time=add_ffn_residual_time,
            tensor_parallel_communication_time=attn_tp_allreduce_time,
            attn_tensor_parallel_allreduce_time=attn_tp_allreduce_time,
            moe_tensor_parallel_allreduce_time=ffn_tp_allreduce_time,
            pipeline_parallel_communication_time=base_time,
            expert_parallel_communication_time=expert_parallel_comm_time,
            moe_gating_time=base_time if is_moe else 0.0,
            moe_shuffling_time=(
                0.0 if zero_routed_ep_lane else base_time
            ) if is_moe else 0.0,
            schedule_time=base_time,
            sampler_e2e_time=base_time,
            prepare_inputs_e2e_time=base_time,
            process_model_outputs_time=base_time,
            ray_comm_time=base_time,
            pp_stage_boundary_handoff_time=pp_stage_boundary_handoff_time,
            is_moe=is_moe,
            mlp_layer_up_proj_execution_time=(
                base_time if include_ffn and not is_moe else 0.0
            ),
            mlp_layer_down_proj_execution_time=(
                base_time if include_ffn and not is_moe else 0.0
            ),
            mlp_layer_act_execution_time=(
                base_time if include_ffn and not is_moe else 0.0
            ),
            moe_grouped_gemm_time=moe_grouped_gemm_time,
            share_expert_up_proj_time=share_expert_time,
            share_expert_down_proj_time=share_expert_time,
            share_expert_act_time=share_expert_time,
            tensor_parallel_allgather_time=ffn_tp_allgather_time,
            share_expert_tensor_parallel_allreduce_time=share_expert_tp_allreduce_time,
            dp_input_allreduce_time=dp_input_allreduce_time,
            dp_output_allreduce_time=dp_output_allreduce_time,
            communication_operator_times=communication_operator_times,
            mlp_operator_times=mlp_operator_times,
            moe_operator_times=moe_operator_times,
            layer_ids=normalized_layer_ids,
        )

    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
        model_manager: ExecutionTimePredictionModelManager = None,
        cluster_type: ClusterType = None,
        training_file_paths: Dict[str, str] = None,
        cc_backend: Optional["BaseCCBackend"] = None,
    ) -> None:
        self._is_moe = True
        self._router_topk = replica_config.router_topk
        self._moe_tp_size = replica_config.moe_tensor_parallel_size
        self._moe_ep_size = replica_config.moe_expert_parallel_size

        # Initialize the canonical distribution selector before parent init so
        # profiling paths choose matching gating-runtime metadata.
        self._moe_routing_distribution_type = str(
            getattr(replica_config, "moe_routing_distribution_type", "balanced")
        ).strip().lower()
        valid_distribution_types = {"balanced", "random", "skewed", "zipf"}
        if self._moe_routing_distribution_type not in valid_distribution_types:
            raise ValueError(
                "moe_routing_distribution_type must be one of "
                f"{sorted(valid_distribution_types)}, got "
                f"{self._moe_routing_distribution_type!r}"
            )
        self._moe_routing_seed = getattr(replica_config, "moe_routing_seed", 42)
        if type(self._moe_routing_seed) is not int or self._moe_routing_seed < 0:
            raise ValueError(
                "moe_routing_seed must be an exact non-negative int, "
                f"got {self._moe_routing_seed}."
            )
        self._moe_gating_routing_runtime_path = (
            resolve_moe_gating_routing_runtime_path(
                self._moe_routing_distribution_type
            )
        )

        super().__init__(
            predictor_config,
            replica_config,
            replica_scheduler_config,
            metrics_config,
            model_manager,
            cluster_type,
            training_file_paths,
            cc_backend,
        )

        # Pre-compute one global routing source. EP ownership is applied later
        # by the shared per-layer materializer.
        self._global_routing_allocations: Optional[Dict[int, Dict[int, float]]] = None
        self._monolithic_routing_details = None
        self._global_routing_allocations = self._init_global_routing_allocations()
        if self._cluster_type == ClusterType.MONOLITHIC and getattr(
            self._model_config, "is_moe", True
        ) is not False:
            self._monolithic_routing_details = self._build_shared_routing_details()
            self._emit_routing_details_snapshot(
                ClusterType.MONOLITHIC,
                self._monolithic_routing_details,
            )
        logger.info(
            "[MoE Routing] Initialized global routing allocations: "
            "distribution=%s, seed=%s, num_layers=%s",
            self._moe_routing_distribution_type,
            self._moe_routing_seed,
            len(self._global_routing_allocations),
        )

    def _init_routing_allocations(self) -> Dict[int, Dict[int, float]]:
        """
        Return the canonical global routing source.

        This private name is retained for existing internal callers, but it
        delegates to the single generator so local and global maps cannot
        diverge.
        """
        return self._init_global_routing_allocations()

    def _init_global_routing_allocations(self) -> Dict[int, Dict[int, float]]:
        """Pre-compute global expert allocation ratios for shared-domain EP sync.

        Monolithic decode with EP enabled needs a global view across all experts to
        derive per-lane post-MoE arrival skew before the shared-domain all-reduce.
        """
        total_experts = self._replica_config.total_expert_num
        cluster_type = getattr(self, "_cluster_type", None)
        if cluster_type == ClusterType.DECODE_ATTN or getattr(
            self._model_config, "is_moe", None
        ) is False:
            return {}
        num_layers = self._model_config.num_layers

        if type(total_experts) is not int or total_experts <= 0:
            raise ValueError(
                "total_expert_num must be an exact positive int for routing; "
                f"got {total_experts!r}"
            )
        if type(self._moe_ep_size) is not int or self._moe_ep_size <= 0:
            raise ValueError(
                "moe_expert_parallel_size must be an exact positive int for routing; "
                f"got {self._moe_ep_size!r}"
            )
        if total_experts % self._moe_ep_size != 0:
            raise ValueError(
                "total_expert_num must be divisible by moe_expert_parallel_size; "
                f"got total_expert_num={total_experts}, "
                f"moe_expert_parallel_size={self._moe_ep_size}"
            )

        distribution_type = self._moe_routing_distribution_type
        allocations: Dict[int, Dict[int, float]] = {}
        for layer_id in range(num_layers):
            layer_seed = self._moe_routing_seed + layer_id
            rng = np.random.default_rng(layer_seed)
            if distribution_type == "balanced":
                weights = np.ones(total_experts, dtype=float)
            elif distribution_type == "random":
                weights = rng.uniform(0.1, 1.0, total_experts)
            elif distribution_type == "skewed":
                ranks = np.arange(1, total_experts + 1, dtype=float)
                weights = 1.0 / np.power(ranks, 0.35)
            elif distribution_type == "zipf":
                ranks = np.arange(1, total_experts + 1, dtype=float)
                weights = 1.0 / ranks
            else:
                raise ValueError(
                    "Unsupported moe_routing_distribution_type="
                    f"{distribution_type!r}"
                )
            total_weight = float(np.sum(weights))
            if not np.isfinite(total_weight) or total_weight <= 0.0:
                raise ValueError(
                    "MoE routing distribution produced an invalid weight sum: "
                    f"distribution={distribution_type!r}, layer_id={layer_id}, "
                    f"sum={total_weight!r}"
                )
            expert_ratios = weights / total_weight
            allocations[layer_id] = {
                expert_id: float(expert_ratios[expert_id])
                for expert_id in range(total_experts)
            }

        return allocations

    def _build_shared_routing_details(
        self,
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        """Expose one immutable-shape routing source for monolithic schedulers.

        ``_global_routing_allocations`` is generated once per model layer and is
        intentionally replica-independent.  The scheduler, however, performs an
        exact ``(replica_id, global_layer_id)`` lookup.  Materialize that lookup
        shape here without generating a second distribution or assigning tokens
        to requests.  Integer token accounting and EP ownership splitting remain
        the responsibility of the shared per-layer materializer.

        The current monolithic predictor is constructed from ``ReplicaConfig``
        rather than ``ClusterConfig``.  The canonical cluster capacity is
        injected as ``_cluster_num_replicas`` (or the explicit
        ``ReplicaConfig.cluster_num_replicas`` field) before this method is
        called.  A missing capacity is an invalid topology, not a condition to
        infer from an attention-DP field.
        """
        allocations = getattr(self, "_global_routing_allocations", None)
        if type(allocations) is not dict:
            raise ValueError(
                "_global_routing_allocations must be an exact dict before "
                "building shared routing details"
            )

        replica_config = getattr(self, "_replica_config", None)
        replica_count = getattr(self, "_cluster_num_replicas", None)
        if replica_count is None:
            replica_count = getattr(replica_config, "cluster_num_replicas", None)
        if type(replica_count) is not int or replica_count <= 0:
            raise ValueError(
                "A positive cluster replica count is required to build shared "
                f"routing details; got {replica_count!r}"
            )

        shared_details: Dict[int, Dict[int, Dict[int, float]]] = {}
        for replica_id in range(replica_count):
            per_layer: Dict[int, Dict[int, float]] = {}
            for layer_id, expert_ratios in allocations.items():
                if type(layer_id) is not int or layer_id < 0:
                    raise ValueError(
                        "Global routing layer IDs must be exact non-negative ints"
                    )
                if type(expert_ratios) is not dict:
                    raise ValueError(
                        "Global routing expert ratios must be exact dicts"
                    )
                per_layer[layer_id] = dict(expert_ratios)
            shared_details[replica_id] = per_layer
        return shared_details


    def _get_routing_details_for_cluster(self, cluster_type: ClusterType):
        """Return the exact pre-generated routing map for one serving role."""
        attribute_by_cluster = {
            ClusterType.MONOLITHIC: "_monolithic_routing_details",
            ClusterType.PREFILL: "_prefill_routing_details",
            ClusterType.DECODE: "_decode_routing_details",
            ClusterType.DECODE_FFN: "_decode_ffn_routing_details",
        }
        attribute_name = attribute_by_cluster.get(cluster_type)
        if attribute_name is None:
            raise ValueError(
                f"MoE routing materialization does not support cluster_type={cluster_type}"
            )
        routing_details = getattr(self, attribute_name, None)
        if routing_details is None:
            raise ValueError(
                f"Missing pre-generated routing details for cluster_type={cluster_type}"
            )
        return routing_details

    def _get_moe_replica_config_for_cluster(self, cluster_type: ClusterType):
        if cluster_type == ClusterType.MONOLITHIC:
            return self._replica_config
        cluster_getter = getattr(self, "_get_cluster_replica_config", None)
        if callable(cluster_getter):
            return cluster_getter(cluster_type)
        return self._replica_config

    def _materialize_layer_ep_workload(
        self, batch: Batch, cluster_type: ClusterType, layer_id: int
    ) -> LayerEPWorkload:
        """Materialize one exact Replica-local EP workload for a MoE layer."""
        cluster_replica_config = self._get_moe_replica_config_for_cluster(cluster_type)
        routing_details = self._get_routing_details_for_cluster(cluster_type)
        target_replica_id = int(batch.replica_id)
        global_layer_id = int(layer_id)
        total_expert_num = int(cluster_replica_config.total_expert_num)
        moe_ep_size = int(cluster_replica_config.moe_expert_parallel_size)
        workload = materialize_layer_ep_workload(
            routing_ratios=resolve_routing_details(
                routing_details,
                target_replica_id=target_replica_id,
                global_layer_id=global_layer_id,
            ),
            target_replica_id=target_replica_id,
            global_layer_id=global_layer_id,
            routing_token_count=int(batch.total_num_tokens),
            router_topk=int(cluster_replica_config.router_topk),
            total_expert_num=total_expert_num,
            moe_expert_parallel_size=moe_ep_size,
            expert_to_ep=build_contiguous_expert_ownership(
                total_expert_num,
                moe_ep_size,
            ),
        )
        return workload

    def _resolve_layer_lane_workload(
        self,
        batch: Batch,
        *,
        cluster_type: ClusterType,
        layer_id: int,
    ) -> EPLaneWorkload:
        """Resolve one physical lane descriptor at a predictor boundary.

        Scheduler-created lane entities already carry the descriptor.  A
        regular batch may be materialized into one lane only for EP=1; an EP>1
        aggregate must be expanded by the scheduler's lane wave first so the
        predictor never guesses which local expert domain a global map denotes.
        """

        lane_workload = resolve_ep_lane_workload(batch, required=False)
        if lane_workload is not None:
            return lane_workload

        layer_workload = self._materialize_layer_ep_workload(
            batch=batch,
            cluster_type=cluster_type,
            layer_id=layer_id,
        )
        participant_ep_ids = tuple(layer_workload.participant_ep_ids)
        if len(participant_ep_ids) != int(self._moe_ep_size):
            raise ValueError(
                "materialized EP lane count does not match predictor topology: "
                f"descriptors={len(participant_ep_ids)}, predictor={self._moe_ep_size}"
            )
        if len(participant_ep_ids) != 1:
            raise ValueError(
                "regular aggregate MoE prediction requires a physical EP lane "
                "for EP>1; scheduler lane materialization is required"
            )
        return layer_workload.lane(participant_ep_ids[0])

    def _resolve_shared_domain_lane_workloads(
        self,
        batch: Batch,
        *,
        cluster_type: ClusterType,
        layer_id: int,
    ) -> tuple[EPLaneWorkload, ...]:
        """Resolve every physical lane for a shared-domain MoE timing probe."""

        lane_count = int(self._moe_ep_size)
        if lane_count <= 0:
            raise ValueError(
                "shared-domain MoE lane resolution requires a positive EP size, "
                f"got {lane_count}"
            )

        lane_workload = resolve_ep_lane_workload(batch, required=False)
        if lane_workload is not None:
            if lane_workload.moe_expert_parallel_size != lane_count:
                raise ValueError(
                    "batch lane workload EP size does not match predictor: "
                    f"descriptor={lane_workload.moe_expert_parallel_size}, "
                    f"predictor={lane_count}"
                )
            lane_workloads = (lane_workload,)
        else:
            workload = self._materialize_layer_ep_workload(
                batch=batch,
                cluster_type=cluster_type,
                layer_id=layer_id,
            )
            lane_workloads = tuple(
                workload.lane(ep_id) for ep_id in workload.participant_ep_ids
            )

        if len(lane_workloads) != lane_count:
            raise ValueError(
                "materialized EP lane count does not match predictor topology: "
                f"descriptors={len(lane_workloads)}, predictor={lane_count}"
            )
        return lane_workloads

    @staticmethod
    def _get_dummy_shared_domain_moe_scope_time(
        execution_time: ExecutionTime,
    ) -> float:
        """Return the fixed per-operator MoE scope used by shared-domain decode.

        The generic dummy ``ExecutionTime`` keeps the deprecated aggregate
        ``moe_gating_time`` contract by splitting that baseline across the two
        structured gating fields.  The shared-domain decode contract models
        each named gating operator as one fixed structural slot, matching its
        historical five-operator scope.  Resolve that compatibility at this
        boundary from the MoE family registry; all other operators retain the
        descriptor-aware structured timing, including zero-lane routed work.
        """

        moe_time = execution_time.moe_or_mlp_time_component
        if not isinstance(moe_time, MoETime):
            raise ValueError(
                "shared-domain dummy timing requires a MoE execution component"
            )
        operator_times = moe_time.operator_times
        if operator_times is None:
            raise ValueError(
                "shared-domain dummy timing requires structured MoE operator times"
            )

        scope_time = 0.0
        for operator_name, operator_time in operator_times.op_times.items():
            if operator_name in _MOE_GATING_OPERATOR_NAMES:
                # ``moe_gating_time`` is the one fixed dummy baseline for each
                # named gating operator; the structured fields store its
                # compatibility split as 0.5 * baseline each.
                scope_time += float(moe_time.moe_gating_time)
            else:
                scope_time += float(operator_time)
        return scope_time

    def predict_monolithic_decode_shared_domain_lane_moe_times_ms(
        self,
        batch: Batch,
        layer_id: int,
    ) -> Dict[int, float]:
        """Estimate per-EP-lane pre-collective MoE time for monolithic pure decode.

        Returns per-lane post-attention MoE compute in milliseconds. The result is
        used by the MONOLITHIC decode sync path to model shared-domain readiness skew
        before `expert_parallel_allreduce`.
        """
        if self._enable_dummy_mode:
            lane_workloads = self._resolve_shared_domain_lane_workloads(
                batch,
                cluster_type=ClusterType.MONOLITHIC,
                layer_id=layer_id,
            )
            lane_times_ms: Dict[int, float] = {}
            for lane_workload in lane_workloads:
                # This helper returns only the per-layer MoE scope.  The dummy
                # execution seam needs a stage value for its complete object,
                # but no stage-boundary term is included in the component total.
                execution_time = self._get_dummy_execution_time(
                    batch,
                    pipeline_stage=0,
                    include_attention=False,
                    lane_workload=lane_workload,
                )
                lane_times_ms[lane_workload.ep_id] = (
                    self._get_dummy_shared_domain_moe_scope_time(execution_time)
                )
            return lane_times_ms

        lane_workloads = self._resolve_shared_domain_lane_workloads(
            batch,
            cluster_type=ClusterType.MONOLITHIC,
            layer_id=layer_id,
        )

        post_attention_layernorm_time = self._get_mlp_norm_layer_act_execution_time(batch)
        gating_linear_time = self._get_gating_linear_time(batch)
        gating_routing_topk_time = self._get_gating_routing_topk_time(batch)
        share_expert_total_time = 0.0
        if self._model_config.supports_share_expert():
            share_expert_total_time = (
                self._get_share_expert_up_proj_execution_time(batch)
                + self._get_share_expert_down_proj_execution_time(batch)
                + self._get_share_expert_act_execution_time(batch)
            )

        lane_times_ms: Dict[int, float] = {}
        for lane_workload in lane_workloads:
            lane_id = lane_workload.ep_id
            shuffling_time = self._get_moe_shuffling_time(
                batch,
                moe_tokens_input=lane_workload,
            )
            grouped_gemm_time = self._get_grouped_gemm_time(
                lane_workload,
                batch=batch,
            )

            lane_times_ms[lane_id] = (
                post_attention_layernorm_time
                + gating_linear_time
                + gating_routing_topk_time
                + shuffling_time
                + grouped_gemm_time
                + share_expert_total_time
            )

        return lane_times_ms

    # Load imbalance feature columns used for MoE training (aligned with SharedPredictionModelManager)
    # Reference: frontier/training/moe_trainer.py lines 224-239 (authoritative source)
    MOE_LOAD_IMBALANCE_FEATURES = [
        # Config features (6) - describe model configuration
        "total_routed_tokens",  # Total tokens after routing (num_tokens * router_topk)
        "num_experts_per_device",  # Number of experts per device after EP sharding
        "hidden_dim",  # Model hidden dimension
        "expert_hidden_dim",  # Expert FFN hidden dimension
        "router_topk",  # Number of experts each token is routed to
        "model_expansion_ratio",  # expert_hidden_dim / hidden_dim
        # Derived features (2) - derived from config and routing
        "tokens_per_expert_avg",  # Average tokens per expert
        "tokens_to_experts_ratio",  # tokens / num_experts ratio
        # Load features (6) - describe load distribution characteristics
        "expert_utilization",  # Proportion of experts with non-zero load
        "min_load_ratio",  # Min load / average load
        "load_imbalance_cv",  # Coefficient of Variation: std/mean, key imbalance metric
        "max_load_ratio",  # Max load / average load
        "load_entropy",  # Entropy of load distribution (higher = more uniform)
        "load_gini_coefficient",  # Gini coefficient: 0=equality, 1=inequality
    ]

    @staticmethod
    def _get_moe_op_tp_key(
        op_name: str,
        moe_tp_size: int,
        cluster_type: ClusterType | None = None,
    ) -> int:
        try:
            return resolve_moe_operator_tp_key(
                op_name,
                moe_tp_size=moe_tp_size,
                cluster_type=cluster_type,
                family=MOE_FAMILY,
            )
        except ValueError as exc:
            if str(exc).startswith("Unsupported MoE op:"):
                raise ValueError(
                    f"Unsupported MoE op for TP mapping: {op_name}"
                ) from exc
            raise

    @staticmethod
    def _is_moe_op_ep_agnostic(op_name: str) -> bool:
        try:
            return is_moe_operator_ep_agnostic(op_name, family=MOE_FAMILY)
        except ValueError as exc:
            if str(exc).startswith("Unsupported MoE op:"):
                raise ValueError(
                    f"Unsupported MoE op for EP mapping: {op_name}"
                ) from exc
            raise

    def _resolve_routed_moe_layer_contract(
        self,
        *,
        moe_tp_size: Optional[int] = None,
        moe_ep_size: Optional[int] = None,
    ) -> Optional[ResolvedLayerContract]:
        """Resolve the profile-owned routed contract used by MoE consumers.

        Older lightweight predictor fixtures do not expose an architecture
        profile and retain the legacy ``mlp_hidden_dim`` behavior. Production
        model configs expose the profile, so routed width and parallel domains
        are resolved from the same typed contract used by the shared manager.
        """

        model_config = getattr(self, "_model_config", None)
        profile_getter = getattr(model_config, "get_model_architecture_profile", None)
        if not callable(profile_getter):
            return None

        replica_config = getattr(self, "_replica_config", None)
        if moe_tp_size is None:
            moe_tp_size = getattr(
                replica_config,
                "moe_tensor_parallel_size",
                getattr(self, "_moe_tp_size", 1),
            )
        if moe_ep_size is None:
            moe_ep_size = getattr(
                replica_config,
                "moe_expert_parallel_size",
                getattr(self, "_moe_ep_size", 1),
            )
        if type(moe_tp_size) is not int or moe_tp_size <= 0:
            raise ValueError(
                f"routed MoE contract requires a positive TP size, got {moe_tp_size!r}"
            )
        if type(moe_ep_size) is not int or moe_ep_size <= 0:
            raise ValueError(
                f"routed MoE contract requires a positive EP size, got {moe_ep_size!r}"
            )

        profile = self._get_model_architecture_profile()
        contract = profile.resolve_layer_contract(
            model_config,
            operator_name="moe_grouped_gemm",
            moe_tp_size=moe_tp_size,
            ffn_tp_size=moe_tp_size,
            expert_parallel_size=moe_ep_size,
        )
        if contract.layer_kind is not LayerKind.ROUTED:
            raise ValueError(
                "moe_grouped_gemm must resolve to a routed layer contract, got "
                f"{contract.layer_kind.value}"
            )
        return contract

    def _validate_moe_dataset_contract(
        self,
        moe_df: pd.DataFrame,
        moe_input_file: str,
        model_names: List[str],
        moe_tp_size: int,
        moe_ep_size: int,
    ) -> pd.DataFrame:
        """Validate op-level MoE key coverage and return model-filtered dataframe."""
        _validate_moe_columns(moe_df)
        required_columns = [
            "num_experts",
            "router_topk",
            "hidden_dim",
            "expert_hidden_dim",
            "num_tensor_parallel_workers",
            "expert_parallel_size",
        ]
        missing_columns = [col for col in required_columns if col not in moe_df.columns]
        if missing_columns:
            raise ValueError(
                f"MoE dataset contract validation failed for {moe_input_file}: "
                f"missing required columns {missing_columns}."
            )

        model_config = self._model_config
        routed_contract = self._resolve_routed_moe_layer_contract(
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )
        expected_expert_hidden_dim = (
            routed_contract.effective_ffn_width
            if routed_contract is not None
            else model_config.mlp_hidden_dim
        )
        base_df = moe_df[
            (moe_df["num_experts"] == model_config.num_experts)
            & (moe_df["router_topk"] == model_config.num_experts_per_tok)
            & (moe_df["hidden_dim"] == model_config.embedding_dim)
            & (moe_df["expert_hidden_dim"] == expected_expert_hidden_dim)
        ].copy()

        if len(base_df) == 0:
            raise ValueError(
                "MoE dataset contract validation failed: no rows match model configuration in "
                f"{moe_input_file}. Required: num_experts={model_config.num_experts}, "
                f"router_topk={model_config.num_experts_per_tok}, hidden_dim={model_config.embedding_dim}, "
                f"expert_hidden_dim={expected_expert_hidden_dim}."
            )

        available_pairs = sorted(
            {
                (int(tp), int(ep))
                for tp, ep in base_df[
                    ["num_tensor_parallel_workers", "expert_parallel_size"]
                ].drop_duplicates().itertuples(index=False, name=None)
            }
        )
        requested_routing_runtime_path = (
            self._get_requested_moe_gating_routing_runtime_path()
        )

        missing_requirements: List[str] = []
        for model_name in model_names:
            base_model_name = get_moe_gating_base_model_name(model_name)
            tp_key = self._get_moe_op_tp_key(
                base_model_name,
                moe_tp_size,
                cluster_type=getattr(self, "_cluster_type", None),
            )
            requirement_parts = [f"TP={tp_key}"]
            if self._is_moe_op_ep_agnostic(base_model_name):
                op_df = base_df[base_df["num_tensor_parallel_workers"] == tp_key]
                requirement_parts.append("EP=ANY")
            else:
                op_df = base_df[
                    (base_df["num_tensor_parallel_workers"] == tp_key)
                    & (base_df["expert_parallel_size"] == moe_ep_size)
                ]
                requirement_parts.append(f"EP={moe_ep_size}")
            if base_model_name == "moe_gating_routing_topk":
                op_df = filter_moe_gating_routing_topk_rows(
                    op_df,
                    requested_runtime_path=requested_routing_runtime_path,
                    source_name=moe_input_file,
                )
                requirement_parts.append(
                    f"routing_runtime_path={requested_routing_runtime_path}"
                )
            if _is_moe_gating_family_model_name(base_model_name):
                op_df = filter_moe_gating_rows_by_runtime_context(
                    op_df,
                    requested_context=DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
                    source_name=moe_input_file,
                )
                requirement_parts.append(
                    "gating_runtime_context="
                    f"{DEFAULT_MOE_GATING_RUNTIME_CONTEXT}"
                )
            requirement = ", ".join(requirement_parts)
            if len(op_df) == 0:
                missing_requirements.append(f"{model_name} requires {requirement}")
                continue
            target_col = f"time_stats.{base_model_name}.median"
            if op_df[target_col].dropna().empty:
                missing_requirements.append(
                    f"{model_name} requires {requirement}, target={target_col} "
                    "to contain at least one non-NaN row"
                )

        if missing_requirements:
            requirement_text = "\n  - ".join(missing_requirements)
            raise ValueError(
                "MoE dataset contract validation failed before training.\n"
                f"File: {moe_input_file}\n"
                "Missing op-level key coverage:\n"
                f"  - {requirement_text}\n"
                f"Available (TP, EP) pairs for matched model rows: {available_pairs}"
            )

        return base_df

    def _train_moe_models(self) -> Dict[str, BaseEstimator]:
        """Train MoE-specific models (gating, shuffling, grouped_gemm) for independent training mode.

        For moe_grouped_gemm, uses 14 load-imbalance features if available in the profiling data.
        This enables simulation mode with per-expert token allocation.
        Other MoE models (gating_linear, gating_routing_topk, shuffling) use only num_tokens.
        """
        models = {}
        moe_input_file = getattr(self, "_moe_input_file", "/synthetic/moe.csv")

        if not os.path.exists(moe_input_file):
            logger.warning(f"MoE input file does not exist: {moe_input_file}")
            return models

        try:
            moe_df = pd.read_csv(moe_input_file)
        except Exception as e:
            logger.warning(f"Failed to load MoE data from {moe_input_file}: {e}")
            return models

        metadata = self._get_profiling_metadata(moe_df, moe_input_file)
        self._validate_active_measurement_type(metadata, moe_input_file)

        tp_col = "num_tensor_parallel_workers"
        ep_col = "expert_parallel_size"
        moe_tp_size = self._replica_config.moe_tensor_parallel_size
        moe_ep_size = self._replica_config.moe_expert_parallel_size

        if tp_col not in moe_df.columns:
            raise ValueError(
                f"Required column '{tp_col}' is missing in {moe_input_file}. "
                "Re-run MoE profiling with TP metadata enabled."
            )
        if ep_col not in moe_df.columns:
            raise ValueError(
                f"Required column '{ep_col}' is missing in {moe_input_file}. "
                "Re-run MoE profiling with EP metadata enabled."
            )

        base_model_names = _get_moe_family_model_names()
        model_names = list(base_model_names)
        model_filtered_df = self._validate_moe_dataset_contract(
            moe_df,
            moe_input_file,
            base_model_names,
            moe_tp_size,
            moe_ep_size,
        )
        if should_enable_prefill_warmed_moe_gating_contract(
            model_config=self._model_config,
        ):
            if has_prefill_warmed_moe_gating_rows(model_filtered_df):
                model_names.extend(_get_prefill_warmed_moe_gating_model_names())
            else:
                logger.warning(
                    "Prefill-warmed gating contract is enabled for model=%s, but "
                    "dataset %s has no usable prefill_warmed rows; skipping "
                    "warmed-context pseudo-model training.",
                    self._replica_config.model_name,
                    moe_input_file,
                )

        self._register_profiling_metadata_for_ops(
            model_names, metadata, moe_input_file
        )

        requested_routing_runtime_path = (
            self._get_requested_moe_gating_routing_runtime_path()
        )
        moe_df_cache: Dict[
            tuple[int, Optional[int], Optional[str], Optional[str]], pd.DataFrame
        ] = {}

        def _get_moe_df_for_op(
            model_name: str,
        ) -> tuple[pd.DataFrame, int, Optional[int]]:
            base_model_name = get_moe_gating_base_model_name(model_name)
            tp_key = self._get_moe_op_tp_key(
                base_model_name,
                moe_tp_size,
                cluster_type=getattr(self, "_cluster_type", None),
            )
            ep_key: Optional[int]
            if self._is_moe_op_ep_agnostic(base_model_name):
                ep_key = None
            else:
                ep_key = moe_ep_size
            runtime_path_key: Optional[str] = None
            if base_model_name == "moe_gating_routing_topk":
                runtime_path_key = requested_routing_runtime_path
            gating_context_key: Optional[str] = None
            if _is_moe_gating_family_model_name(base_model_name):
                gating_context_key = get_moe_gating_prediction_model_context(
                    model_name
                )
            cache_key = (tp_key, ep_key, runtime_path_key, gating_context_key)
            if cache_key not in moe_df_cache:
                filtered_df = model_filtered_df[
                    model_filtered_df[tp_col] == tp_key
                ].copy()
                if ep_key is not None:
                    filtered_df = filtered_df[
                        filtered_df[ep_col] == ep_key
                    ].copy()
                if runtime_path_key is not None:
                    filtered_df = filter_moe_gating_routing_topk_rows(
                        filtered_df,
                        requested_runtime_path=runtime_path_key,
                        source_name=moe_input_file,
                    )
                if gating_context_key is not None:
                    filtered_df = filter_moe_gating_rows_by_runtime_context(
                        filtered_df,
                        requested_context=gating_context_key,
                        source_name=moe_input_file,
                    )
                if len(filtered_df) == 0:
                    ep_desc = "ANY" if ep_key is None else str(ep_key)
                    raise ValueError(
                        f"No MoE data after filtering for TP={tp_key}, EP={ep_desc}. "
                        f"Requested by op-level TP mapping in {moe_input_file}."
                    )
                filtered_df["num_tokens_rounded"] = filtered_df["num_tokens"].apply(
                    lambda x: max(1, round(x / 8) * 8)
                )
                moe_df_cache[cache_key] = filtered_df
            return moe_df_cache[cache_key], tp_key, ep_key

        for model_name in model_names:
            try:
                op_df, moe_tp_key, moe_ep_key = _get_moe_df_for_op(model_name)
            except ValueError as e:
                if get_moe_gating_base_model_name(model_name) != model_name:
                    logger.warning(
                        "Skipping %s because prefill-warmed gating rows are unavailable "
                        "for the requested TP/EP slice (%s).",
                        model_name,
                        e,
                    )
                    continue
                raise
            target_op_name = get_moe_gating_base_model_name(model_name)
            target_col = f"time_stats.{target_op_name}.median"
            if target_col not in op_df.columns:
                ep_desc = "ANY" if moe_ep_key is None else str(moe_ep_key)
                raise ValueError(
                    f"Column '{target_col}' not found in MoE dataframe for TP={moe_tp_key}, EP={ep_desc}. "
                    "Re-run MoE profiling with split gating columns."
                )

            # Per-operation feature selection (aligned with SharedPredictionModelManager).
            if model_name == "moe_grouped_gemm":
                available_load_features = [
                    f for f in self.MOE_LOAD_IMBALANCE_FEATURES if f in op_df.columns
                ]
                has_load_imbalance_features = len(available_load_features) == len(
                    self.MOE_LOAD_IMBALANCE_FEATURES
                )
                if 0 < len(available_load_features) < len(self.MOE_LOAD_IMBALANCE_FEATURES):
                    missing_features = [
                        f
                        for f in self.MOE_LOAD_IMBALANCE_FEATURES
                        if f not in op_df.columns
                    ]
                    raise ValueError(
                        f"Partial load imbalance features found ({len(available_load_features)}/"
                        f"{len(self.MOE_LOAD_IMBALANCE_FEATURES)}) for TP={moe_tp_key}. "
                        f"Missing: {missing_features}."
                    )
                if has_load_imbalance_features:
                    feature_cols = available_load_features
                    logger.info(
                        f"  {model_name}: Using load imbalance features ({len(feature_cols)} features, TP={moe_tp_key})"
                    )
                else:
                    feature_cols = ["num_tokens"]
                    logger.info(
                        f"  {model_name}: Load imbalance features not found; using num_tokens only (TP={moe_tp_key})"
                    )
            elif model_name == "moe_shuffling":
                available_load_features = [
                    f for f in self.MOE_LOAD_IMBALANCE_FEATURES if f in op_df.columns
                ]
                if len(available_load_features) == len(self.MOE_LOAD_IMBALANCE_FEATURES):
                    feature_cols = available_load_features
                    logger.info(
                        f"  {model_name}: Using load imbalance features ({len(feature_cols)} features, TP={moe_tp_key})"
                    )
                else:
                    feature_cols = ["num_tokens"]
                    logger.info(
                        f"  {model_name}: Full load imbalance features unavailable; using num_tokens only (TP={moe_tp_key})"
                    )
            else:
                feature_cols = ["num_tokens"]
                logger.info(f"  {model_name}: Using num_tokens only (1 feature, TP={moe_tp_key})")

            models[model_name] = self._train_model(
                model_name=model_name,
                df=op_df,
                feature_cols=feature_cols,
                target_col=target_col,
            )
            logger.info(f"Trained MoE model: {model_name}")

        return models

    def _register_additional_profiling_metadata_from_files(self) -> None:
        moe_input_file = self._moe_input_file
        model_names = _get_moe_family_model_names()
        if should_enable_prefill_warmed_moe_gating_contract(
            model_config=self._model_config,
        ):
            moe_df = pd.read_csv(moe_input_file)
            include_prefill_warmed_models = has_prefill_warmed_moe_gating_rows(
                moe_df
            )
            if include_prefill_warmed_models:
                model_names.extend(_get_prefill_warmed_moe_gating_model_names())
        self._register_profiling_metadata_from_file(moe_input_file, model_names)

    def _train_models(self) -> Dict[str, BaseEstimator]:
        """Override to include MoE model training for independent training mode."""
        models = super()._train_models()

        if self._model_manager is None:
            moe_models = self._train_moe_models()
            models.update(moe_models)
            logger.info(f"Trained MoE models independently: {list(moe_models.keys())}")
        else:
            logger.info("MoE models loaded from ExecutionTimePredictionModelManager.")

        return models

    def _predict_for_compute_models(self) -> Dict[str, Any]:
        predictions = super()._predict_for_compute_models()
        extra_model_names = _get_prefill_warmed_moe_gating_model_names()
        num_token_range = np.arange(1, self._max_tokens + 1)
        X = pd.DataFrame({"num_tokens": num_token_range})
        for model_name in extra_model_names:
            if model_name not in self._models:
                continue
            model = self._models[model_name]
            predictions[model_name] = self._get_model_prediction(
                model_name, model, X
            )
        return predictions

    def _select_moe_gating_prediction_model_name(
        self,
        base_model_name: str,
        batch: Batch,
    ) -> str:
        requested_context = DEFAULT_MOE_GATING_RUNTIME_CONTEXT
        if should_use_prefill_warmed_moe_gating_context(
            model_config=self._model_config,
            batch=batch,
        ):
            requested_context = PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT
        candidate_model_name = get_moe_gating_prediction_model_name(
            base_model_name,
            requested_context=requested_context,
        )
        if candidate_model_name in self._predictions:
            return candidate_model_name
        return base_model_name

    def _use_expert_parallel_alltoall_path(self, batch: Batch) -> bool:
        moe_ep_size = int(getattr(self, "_moe_ep_size", 1))
        if moe_ep_size <= 1:
            return False
        # EP is replica-local and is independent of the retired attention-DP
        # lane concept.  A full batch on any MoE serving role therefore uses
        # the EP communication/accounting path whenever EP>1.
        return True

    def _predict_expert_parallel_phase_operator_times(
        self,
        batch: Batch,
        *,
        lane_workload: Optional[EPLaneWorkload] = None,
    ) -> dict[str, float]:
        """Predict exact dispatch and combine collectives for one MoE layer."""

        if self._moe_ep_size <= 1:
            return {
                "expert_parallel_alltoall_dispatch": 0.0,
                "expert_parallel_alltoall_combine": 0.0,
            }
        if not self._use_expert_parallel_alltoall_path(batch):
            raise ValueError(
                "Canonical MoE EP execution requires named all-to-all dispatch "
                "and combine phases"
            )
        return {
            op_name: self._predict_comm_operator(
                get_comm_operator(op_name),
                batch,
                lane_workload=lane_workload,
            )
            for op_name in (
                "expert_parallel_alltoall_dispatch",
                "expert_parallel_alltoall_combine",
            )
        }

    def _get_effective_moe_total_tokens(self, batch: Batch) -> int:
        effective_tokens = int(
            batch.get_effective_total_tokens_rounded(self._cluster_type)
        )
        if effective_tokens < 0:
            raise ValueError(
                f"effective MoE tokens must be non-negative, got {effective_tokens}"
            )
        return effective_tokens

    def _get_moe_pre_routing_token_count(self, batch: Optional[Batch]) -> int:
        """Return the source-batch width used by pre-routing MoE models.

        A physical EP lane carries only an assignment subset, so its routed
        count cannot identify the source width.  Callers that need the
        one-feature profiling domain must provide the source batch explicitly.
        """

        if batch is None:
            raise ValueError(
                "MoE pre-routing token lookup requires the source batch; "
                "an EPLaneWorkload cannot supply that width"
            )
        return self._get_effective_moe_total_tokens(batch)

    def _get_local_ep_routed_tokens(
        self,
        batch: Batch,
        *,
        lane_workload: Optional[EPLaneWorkload] = None,
    ) -> int:
        source = batch if lane_workload is None else lane_workload
        resolved_lane_workload = resolve_ep_lane_workload(source, required=True)
        assert resolved_lane_workload is not None
        return resolved_lane_workload.routed_token_count

    def _get_moe_tokens_input(
        self, batch: Batch, layer_id: int = 0
    ) -> EPLaneWorkload | int:
        """
        Unified entry point to get MoE tokens input for grouped GEMM prediction.

        EP lane batches carry the canonical physical descriptor.  A regular
        non-lane batch may use the scalar pre-routing token path for legacy
        one-feature models; load-aware models require an explicit descriptor.

        Args:
            batch: The batch being processed
            layer_id: The layer ID for which to get token allocation (default 0)

        Returns:
            - In load-imbalance mode: ``EPLaneWorkload``
            - In single-token-count profiling mode: pre-routing token count

        Raises:
            ValueError: If the selected routing mode is not supported by the active predictor
        """
        lane_workload = resolve_ep_lane_workload(batch, required=False)
        if lane_workload is not None:
            if lane_workload.router_topk != int(self._router_topk):
                raise ValueError(
                    "EPLaneWorkload router_topk does not match predictor topology: "
                    f"descriptor={lane_workload.router_topk}, predictor={self._router_topk}"
                )
            return lane_workload

        load_aware = any(
            isinstance(prediction, dict)
            and prediction.get("_on_demand_prediction", False)
            for prediction in (
                getattr(self, "_predictions", {}).get("moe_shuffling"),
                getattr(self, "_predictions", {}).get("moe_grouped_gemm"),
            )
        )
        if load_aware:
            cluster_type = getattr(self, "_cluster_type", None)
            if not isinstance(cluster_type, ClusterType):
                raise ValueError(
                    "load-aware MoE prediction requires an initialized cluster_type"
                )
            workload = self._materialize_layer_ep_workload(
                batch=batch,
                cluster_type=cluster_type,
                layer_id=layer_id,
            )
            if len(workload.participant_ep_ids) != int(self._moe_ep_size):
                raise ValueError(
                    "materialized EP lane count does not match predictor topology: "
                    f"descriptors={len(workload.participant_ep_ids)}, "
                    f"predictor={self._moe_ep_size}"
                )
            if int(self._moe_ep_size) != 1:
                raise ValueError(
                    "load-aware regular-batch prediction requires an explicit "
                    "physical EP lane for EP>1"
                )
            return workload.lane(0)
        return self._get_effective_moe_total_tokens(batch)

    def _get_gating_time(self, batch: Batch) -> float:
        """
        Get total MoE gating network execution time (linear + routing_topk).

        The gating network determines which experts each token should be routed to.
        Prediction is based on num_tokens feature from profiling data.

        Returns:
            Total gating time (sum of linear and routing_topk times)
        """
        return self._get_gating_linear_time(batch) + self._get_gating_routing_topk_time(
            batch
        )

    def _get_gating_linear_time(self, batch: Batch) -> float:
        """
        Get MoE gating linear layer execution time.

        The gating linear layer computes logits from hidden states (hidden_dim -> num_experts).
        """
        if not self._supports_operation("moe_gating_linear"):
            raise NotImplementedError(
                "MoE gating linear is not supported for cluster type"
            )
        model_name = self._select_moe_gating_prediction_model_name(
            "moe_gating_linear",
            batch,
        )
        if model_name not in self._predictions:
            raise NotImplementedError(
                "MoE gating linear is not supported for cluster type"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._get_prediction_for_features(
            model_name,
            {"num_tokens": effective_tokens},
            feature_names=("num_tokens",),
        )

    def _get_gating_routing_topk_time(self, batch: Batch) -> float:
        """
        Get MoE gating routing topk execution time.

        The routing topk operation selects top-K experts and applies softmax normalization.
        """
        if not self._supports_operation("moe_gating_routing_topk"):
            raise NotImplementedError(
                "MoE gating routing topk is not supported for cluster type"
            )
        model_name = self._select_moe_gating_prediction_model_name(
            "moe_gating_routing_topk",
            batch,
        )
        if model_name not in self._predictions:
            raise NotImplementedError(
                "MoE gating routing topk is not supported for cluster type"
            )
        effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
        return self._get_prediction_for_features(
            model_name,
            {"num_tokens": effective_tokens},
            feature_names=("num_tokens",),
        )

    def _resolve_shuffling_per_expert_tokens(
        self,
        batch: Batch,
        moe_tokens_input: Optional[EPLaneWorkload] = None,
    ) -> EPLaneWorkload:
        source = batch if moe_tokens_input is None else moe_tokens_input
        lane_workload = resolve_ep_lane_workload(source, required=True)
        assert lane_workload is not None
        return lane_workload

    def _build_moe_load_imbalance_features(
        self,
        lane_workload: EPLaneWorkload,
        *,
        batch: Optional[Batch] = None,
    ) -> Dict[str, float]:
        lane_workload = resolve_ep_lane_workload(lane_workload, required=True)
        assert lane_workload is not None

        from frontier.moe_load_imbalance import MoELoadImbalanceInput

        expert_token_counts = [int(v) for v in lane_workload.local_token_counts]

        total_routed_tokens = int(sum(expert_token_counts))
        if lane_workload.router_topk <= 0:
            raise ValueError(f"Invalid router_topk={lane_workload.router_topk}")

        source_num_tokens = self._get_moe_pre_routing_token_count(batch)

        routed_contract = self._resolve_routed_moe_layer_contract()
        expert_hidden_dim = (
            routed_contract.effective_ffn_width
            if routed_contract is not None
            else int(self._model_config.mlp_hidden_dim)
        )
        load_input = MoELoadImbalanceInput(
            num_tokens=source_num_tokens,
            num_experts_per_device=lane_workload.local_expert_width,
            hidden_dim=int(self._model_config.embedding_dim),
            expert_hidden_dim=expert_hidden_dim,
            router_topk=int(lane_workload.router_topk),
            expert_token_counts=expert_token_counts,
            load_distribution="runtime",
        )
        features = load_input.to_features_dict()
        features.pop("load_distribution", None)
        features.pop("seed", None)
        missing_features = [
            name
            for name in self.MOE_LOAD_IMBALANCE_FEATURES
            if name not in features
        ]
        if missing_features:
            raise ValueError(
                "MoE load-imbalance feature construction is missing canonical "
                f"features: {missing_features}"
            )
        unexpected_features = sorted(
            set(features) - set(self.MOE_LOAD_IMBALANCE_FEATURES)
        )
        if unexpected_features:
            raise ValueError(
                "MoE load-imbalance feature construction produced unexpected "
                f"features: {unexpected_features}"
            )
        return {
            name: features[name]
            for name in self.MOE_LOAD_IMBALANCE_FEATURES
        }

    def _get_moe_shuffling_time(
        self,
        batch: Batch,
        moe_tokens_input: Optional[EPLaneWorkload] = None,
    ) -> float:
        """
        Get MoE token shuffling execution time using trained prediction model.

        Shuffling involves dispatching tokens to assigned experts. When the model is
        trained with load-imbalance features, use on-demand prediction driven by
        per-expert allocation; otherwise use the legacy num_tokens lookup table.
        """
        if not self._supports_operation("moe_shuffling"):
            raise NotImplementedError("MoE shuffling is not supported for cluster type")
        if "moe_shuffling" not in self._predictions:
            raise NotImplementedError("MoE shuffling is not supported for cluster type")
        if moe_tokens_input is not None and not isinstance(
            moe_tokens_input, EPLaneWorkload
        ):
            raise TypeError(
                "MoE shuffling requires an EPLaneWorkload descriptor when an "
                "explicit workload is supplied"
            )

        prediction_cache = self._predictions["moe_shuffling"]
        if isinstance(prediction_cache, dict) and prediction_cache.get(
            "_on_demand_prediction", False
        ):
            lane_workload = self._resolve_shuffling_per_expert_tokens(
                batch,
                moe_tokens_input=moe_tokens_input,
            )
            if lane_workload.routed_token_count == 0:
                raw_time = 0.0
            else:
                features = self._build_moe_load_imbalance_features(
                    lane_workload,
                    batch=batch,
                )
                raw_time = self._get_on_demand_prediction(
                    "moe_shuffling", features
                )
        else:
            lane_workload = resolve_ep_lane_workload(batch, required=False)
            if moe_tokens_input is not None:
                lane_workload = resolve_ep_lane_workload(
                    moe_tokens_input,
                    required=True,
                )
            if lane_workload is not None:
                if lane_workload.routed_token_count == 0:
                    return 0.0
                effective_tokens = self._get_moe_pre_routing_token_count(batch)
            else:
                effective_tokens = batch.get_effective_total_tokens_rounded(
                    self._cluster_type
                )
            raw_time = self._get_prediction_for_features(
                "moe_shuffling",
                {"num_tokens": effective_tokens},
                feature_names=("num_tokens",),
            )

        return raw_time

    def _get_expert_parallel_communication_time(
        self,
        batch: Batch,
        *,
        lane_workload: Optional[EPLaneWorkload] = None,
    ) -> float:
        """
        Get expert parallel communication time.

        Shared-domain MoE execution (monolithic / prefill / decode) uses
        expert-parallel all-reduce when EP is enabled without all-to-all routing.
        Post-routing EP batches (e.g. DECODE_FFN) and flattened multi-DP MoE
        paths keep the all-to-all communication model.
        """
        if self._moe_ep_size <= 1:
            return 0.0

        uses_alltoall = self._use_expert_parallel_alltoall_path(batch)
        resolved_lane_workload = None
        if uses_alltoall:
            resolved_lane_workload = resolve_ep_lane_workload(
                batch if lane_workload is None else lane_workload,
                required=True,
            )
            assert resolved_lane_workload is not None

        if self._cc_backend is not None:
            quant_manager = get_quantization_manager()

            if uses_alltoall:
                routed_tokens = self._get_local_ep_routed_tokens(
                    batch,
                    lane_workload=resolved_lane_workload,
                )
                data_size_bytes = self._model_config.embedding_dim * 2 * routed_tokens
                data_size_bytes = quant_manager.adjust_tensor_size(
                    "expert_parallel_communication", data_size_bytes, self._cluster_type
                )
                result = self._cc_backend.predict_all_to_all(
                    data_size_bytes=data_size_bytes,
                    num_devices=self._moe_ep_size,
                    cluster_type=self._cluster_type,
                    comm_domain="EP",
                )
                logger.debug(
                    f"_get_expert_parallel_communication_time: using EP all-to-all, "
                    f"data_size={data_size_bytes}, num_devices={self._moe_ep_size}, "
                    f"result={result:.6f} ms"
                )
                return result

            effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
            data_size_bytes = self._model_config.embedding_dim * 2 * effective_tokens
            data_size_bytes = quant_manager.adjust_tensor_size(
                "allreduce", data_size_bytes, self._cluster_type
            )
            result = self._cc_backend.predict_allreduce(
                data_size_bytes=data_size_bytes,
                num_devices=self._moe_ep_size,
                cluster_type=self._cluster_type,
                comm_domain="EP",
            )
            result = self._strip_collective_sim_allreduce_launch_overhead_if_needed(
                batch=batch,
                predicted_ms=result,
                num_devices=self._moe_ep_size,
                comm_domain="EP",
            )
            logger.debug(
                f"_get_expert_parallel_communication_time: using EP all-reduce, "
                f"data_size={data_size_bytes}, num_devices={self._moe_ep_size}, "
                f"result={result:.6f} ms"
            )
            return result

        if self._enable_dummy_mode:
            logger.debug(
                f"_get_expert_parallel_communication_time: CC Backend not available, "
                f"using dummy mode value={self._dummy_execution_time} ms"
            )
            return self._dummy_execution_time

        raise RuntimeError(
            f"CC Backend is required for expert parallel communication prediction "
            f"but was not provided. Either:\n"
            f"  1. Configure a CC Backend (e.g., --cc_backend vidur or --cc_backend analytical)\n"
            f"  2. Enable dummy mode explicitly (--enable_dummy_mode)\n"
            f"Current state: cc_backend=None, enable_dummy_mode={self._enable_dummy_mode}"
        )

    def _is_grouped_gemm_on_demand_mode(self) -> bool:
        """
        Check if moe_grouped_gemm predictor is in on-demand (load-imbalance) mode.

        Returns:
            True if the model was trained with 14 load-imbalance features (requires Dict input),
            False if trained with 1 feature (num_tokens only, accepts int input).
        """
        if "moe_grouped_gemm" not in self._predictions:
            return False
        prediction_cache = self._predictions["moe_grouped_gemm"]
        return isinstance(prediction_cache, dict) and prediction_cache.get(
            "_on_demand_prediction", False
        )

    def _get_grouped_gemm_time(
        self,
        num_tokens_or_allocation,
        batch: Optional[Batch] = None,
    ) -> float:
        """
        Calculate grouped GEMM time using trained prediction model.

        Args:
            num_tokens_or_allocation: An ``EPLaneWorkload`` for EP-aware
                                    prediction, or an integer for the legacy
                                    one-feature non-lane path.

        Returns:
            Total grouped GEMM execution time
        """
        if not self._supports_operation("moe_grouped_gemm"):
            raise NotImplementedError(
                "MoE grouped_gemm is not supported for cluster type"
            )

        if "moe_grouped_gemm" not in self._predictions:
            raise NotImplementedError(
                "MoE grouped_gemm is not supported for cluster type"
            )

        prediction_cache = self._predictions["moe_grouped_gemm"]

        if isinstance(num_tokens_or_allocation, Mapping):
            raise TypeError(
                "MoE grouped_gemm requires an EPLaneWorkload descriptor; raw "
                "expert-token maps are not a predictor workload contract"
            )
        lane_workload = (
            resolve_ep_lane_workload(num_tokens_or_allocation, required=True)
            if isinstance(num_tokens_or_allocation, EPLaneWorkload)
            else None
        )

        # Check if this model uses on-demand prediction (trained with load imbalance features)
        if isinstance(prediction_cache, dict) and prediction_cache.get(
            "_on_demand_prediction"
        ):
            # On-demand prediction mode: model was trained with load imbalance features.
            # We must provide the full feature set computed from per-expert token distribution.
            if lane_workload is None:
                raise ValueError(
                    "moe_grouped_gemm is in load-imbalance (on-demand) mode and "
                    "requires an EPLaneWorkload descriptor"
                )

            if lane_workload.routed_token_count == 0:
                return 0.0

            features = self._build_moe_load_imbalance_features(
                lane_workload,
                batch=batch,
            )
            return self._get_on_demand_prediction("moe_grouped_gemm", features)

        # Standard cache lookup mode (trained with num_tokens only)
        if lane_workload is not None:
            if lane_workload.routed_token_count == 0:
                return 0.0
            source_num_tokens = self._get_moe_pre_routing_token_count(batch)
            raw_time = self._get_prediction_for_features(
                "moe_grouped_gemm",
                {"num_tokens": source_num_tokens},
                feature_names=("num_tokens",),
            )
            return raw_time

        # Backward compatibility: single number of tokens
        num_tokens = num_tokens_or_allocation
        if isinstance(num_tokens, bool) or not isinstance(num_tokens, (int, float)):
            raise TypeError(
                "MoE grouped_gemm requires an EPLaneWorkload descriptor or a "
                "numeric token count"
            )
        if num_tokens <= 0:
            return 0.0
        raw_time = self._get_prediction_for_features(
            "moe_grouped_gemm",
            {"num_tokens": num_tokens},
            feature_names=("num_tokens",),
        )
        return raw_time

    @staticmethod
    def _resolve_moe_execution_inputs(
        *,
        moe_tokens_input: object,
        lane_workload: Optional[EPLaneWorkload],
        include_moe: bool,
    ) -> tuple[object, Optional[EPLaneWorkload]]:
        """Resolve one canonical MoE input and its optional physical lane.

        ``moe_tokens_input`` is retained for the legacy scalar one-feature
        lookup, while ``lane_workload`` carries the physical routed domain.
        A physical call must use one descriptor for both roles; allowing a
        scalar or a second descriptor alongside it would let communication and
        routed compute describe different workloads.
        """

        if isinstance(moe_tokens_input, Mapping):
            raise TypeError(
                "moe_tokens_input cannot be a raw expert-token map; provide an "
                "EPLaneWorkload descriptor"
            )

        explicit_lane = (
            resolve_ep_lane_workload(lane_workload, required=True)
            if lane_workload is not None
            else None
        )
        input_lane = (
            resolve_ep_lane_workload(moe_tokens_input, required=True)
            if isinstance(moe_tokens_input, EPLaneWorkload)
            else None
        )

        if explicit_lane is not None:
            if input_lane is not None:
                if input_lane != explicit_lane:
                    raise ValueError(
                        "moe_tokens_input and lane_workload must refer to the "
                        "same EPLaneWorkload descriptor"
                    )
                return explicit_lane, explicit_lane
            if moe_tokens_input is not None:
                raise TypeError(
                    "cannot combine a scalar moe_tokens_input with an "
                    "explicit lane_workload"
                )
            return explicit_lane, explicit_lane

        if input_lane is not None:
            return input_lane, input_lane

        if include_moe and moe_tokens_input is None:
            raise ValueError(
                "moe_tokens_input is required when include_moe=True. "
                "Provide a scalar token count or an EPLaneWorkload descriptor."
            )
        return moe_tokens_input, None

    # This is now a private method used internally for MoE-specific logic
    def _get_execution_time_internal(
        self,
        batch: Batch,
        pipeline_stage: int,
        moe_tokens_input: "EPLaneWorkload | int | None" = None,
        lane_workload: Optional[EPLaneWorkload] = None,
        include_moe: bool = True,
        include_ffn: bool = True,
        include_attention: bool = True,
        layer_id: int = 0,
        num_layers: Optional[int] = None,
        layer_ids: Optional[List[int] | tuple[int, ...]] = None,
    ) -> "ExecutionTime":
        """
        Calculate execution time for a pipeline stage.

        Args:
            batch: The batch being processed
            pipeline_stage: Pipeline stage index
            moe_tokens_input: Typed EP lane workload for EP-aware prediction,
                or a scalar pre-routing token count for the legacy one-feature
                path. ``None`` is valid for attention-only and dense-layer calls.
            include_moe: Whether to include MoE-specific calculations
            include_ffn: Whether to include the post-attention FFN block.  When
                false, only attention and stage-level communication/overhead are
                constructed; no MLP/MoE profiling lookup is allowed.
            include_attention: Whether to include attention operators.  When
                false, the caller is supplying a post-attention EP lane and
                attention profiling rows must not be queried.
            layer_id: Global transformer layer identity used by layer-aware
                attention and terminal-MTP prediction. The default preserves
                the legacy layer-zero behavior for internal post-attention
                callers that do not carry a layer identity.
            num_layers: Optional requested aggregate layer count. When omitted,
                the predictor's configured pipeline-stage count is used.
            layer_ids: Optional complete global layer identity tuple for the
                requested aggregate.

        Returns:
            ExecutionTime with all component times

        Raises:
            ValueError: If include_moe=True but moe_tokens_input is None (fail-fast)
        """
        if type(include_ffn) is not bool:
            raise ValueError("include_ffn must be a bool")
        if type(include_attention) is not bool:
            raise ValueError("include_attention must be a bool")
        configured_num_layers = getattr(self, "_num_layers_per_pipeline_stage", 1)
        effective_num_layers = (
            configured_num_layers if num_layers is None else num_layers
        )
        normalized_layer_ids = self._normalize_stage_layer_ids(
            num_layers=effective_num_layers,
            layer_id=layer_id,
            layer_ids=layer_ids,
        )
        if normalized_layer_ids is not None:
            layer_id = normalized_layer_ids[0]
        if not include_ffn and include_moe:
            raise ValueError("include_moe cannot be true when include_ffn is false")
        if not include_attention and not include_ffn:
            raise ValueError(
                "include_attention=False requires an FFN/MoE post-attention probe"
            )
        moe_tokens_input, lane_workload = self._resolve_moe_execution_inputs(
            moe_tokens_input=moe_tokens_input,
            lane_workload=lane_workload,
            include_moe=include_moe,
        )

        attention_time = (
            self.predict_attention_layer_time(
                batch=batch,
                layer_id=layer_id,
                cluster_type=self._cluster_type,
            )
            if include_attention
            else AttentionTime()
        )

        communication_operator_times: dict[str, float] = {}

        if pipeline_stage == self._replica_config.num_pipeline_stages - 1:
            pipeline_parallel_communication_time = 0
        else:
            pipeline_parallel_communication_time = (
                self._predict_comm_operator(
                    get_comm_operator("pipeline_parallel_send_recv"),
                    batch,
                )
            )
            communication_operator_times["pipeline_parallel_send_recv"] = (
                pipeline_parallel_communication_time
            )

        # For MoE models, attention still uses Tensor Parallelism (AllReduce).
        if (
            not include_attention
            or self._replica_config.attn_tensor_parallel_size == 1
        ):
            attn_tp_allreduce_time = 0
        else:
            attn_tp_allreduce_time = self._predict_comm_operator(
                get_comm_operator("attn_tensor_parallel_allreduce"),
                batch,
            )
            communication_operator_times["attn_tensor_parallel_allreduce"] = (
                attn_tp_allreduce_time
            )

        # Dense-FFN (non-MoE layer) path still uses FFN TP allreduce semantics.
        # Keep it aligned with dense predictor behavior for mixed-layer models.
        moe_tp_allreduce_time = 0.0
        if include_ffn and include_moe and self._replica_config.moe_tensor_parallel_size > 1:
            moe_tp_allreduce_time = self._predict_comm_operator(
                get_comm_operator("moe_tensor_parallel_allreduce"),
                batch,
                lane_workload=lane_workload,
            )
            communication_operator_times["moe_tensor_parallel_allreduce"] = (
                moe_tp_allreduce_time
            )
        elif include_ffn and self._replica_config.attn_tensor_parallel_size > 1:
            moe_tp_allreduce_time = attn_tp_allreduce_time
            communication_operator_times["mlp_tensor_parallel_allreduce"] = (
                moe_tp_allreduce_time
            )

        share_expert_up_proj_time = 0.0
        share_expert_down_proj_time = 0.0
        share_expert_act_time = 0.0
        if include_ffn and include_moe and self._model_config.supports_share_expert():
            share_expert_up_proj_time = self._get_share_expert_up_proj_execution_time(batch)
            share_expert_down_proj_time = self._get_share_expert_down_proj_execution_time(batch)
            share_expert_act_time = self._get_share_expert_act_execution_time(batch)

        mlp_up_proj_time = 0.0
        mlp_down_proj_time = 0.0
        mlp_act_time = 0.0

        if include_ffn and include_moe:
            expert_parallel_operator_times = (
                self._predict_expert_parallel_phase_operator_times(
                    batch,
                    lane_workload=lane_workload,
                )
            )
            communication_operator_times.update(expert_parallel_operator_times)
            expert_parallel_communication_time = sum(
                expert_parallel_operator_times.values()
            )
            moe_gating_linear_time = self._get_gating_linear_time(batch)
            moe_gating_routing_topk_time = self._get_gating_routing_topk_time(batch)
            moe_gating_time = moe_gating_linear_time + moe_gating_routing_topk_time
            moe_shuffling_time = self._get_moe_shuffling_time(
                batch,
                moe_tokens_input=moe_tokens_input,
            )
            moe_grouped_gemm_time = self._get_grouped_gemm_time(
                moe_tokens_input,
                batch=batch,
            )
        elif include_ffn:
            # Dense FFN branch for mixed-layer MoE models.
            expert_parallel_communication_time = 0.0
            moe_gating_time = 0.0
            moe_gating_linear_time = 0.0
            moe_gating_routing_topk_time = 0.0
            moe_shuffling_time = 0.0
            moe_grouped_gemm_time = 0.0
            if self._model_config.supports_share_expert():
                # Step2Mini/Step3 dense layers are the shared-expert FFN.  Map
                # those profiled operations into the dense MLP component
                # fields so the layer remains a FULL_STAGE_WORLD operation and
                # does not acquire MoE routing or EP collective semantics.
                mlp_up_proj_time = self._get_share_expert_up_proj_execution_time(batch)
                mlp_down_proj_time = self._get_share_expert_down_proj_execution_time(batch)
                mlp_act_time = self._get_share_expert_act_execution_time(batch)
            else:
                mlp_up_proj_time = self._get_mlp_layer_up_proj_execution_time(batch)
                mlp_down_proj_time = self._get_mlp_layer_down_proj_execution_time(batch)
                mlp_act_time = self._get_mlp_layer_act_execution_time(batch)
        else:
            # Attention-only probe: no FFN/MoE operation or profiling lookup.
            expert_parallel_communication_time = 0.0
            moe_gating_time = 0.0
            moe_gating_linear_time = 0.0
            moe_gating_routing_topk_time = 0.0
            moe_shuffling_time = 0.0
            moe_grouped_gemm_time = 0.0

        add_time = self._get_add_layer_act_execution_time(batch) if include_ffn else 0.0
        add_attn_residual_time = 0.0
        add_ffn_residual_time = 0.0
        architecture_profile = self._get_model_architecture_profile()
        if architecture_profile.residual_add_policy is ResidualAddPolicy.FFN_RESIDUAL_ONLY:
            add_attn_residual_time = 0.0
            add_ffn_residual_time = add_time
            add_time = 0.0

        ffn_tp_allgather_time = 0.0
        share_expert_tp_allreduce_time = 0.0
        moe_tp_allgather_op = architecture_profile.moe_tensor_parallel_allgather_op
        if include_ffn and include_moe and moe_tp_allgather_op:
            moe_tp_size = self._replica_config.moe_tensor_parallel_size
            if moe_tp_size > 1:
                ffn_tp_allgather_time = self._predict_comm_operator(
                    get_comm_operator(moe_tp_allgather_op),
                    batch,
                )
                communication_operator_times[moe_tp_allgather_op] = ffn_tp_allgather_time
                share_expert_tp_allreduce_op = (
                    architecture_profile.share_expert_tensor_parallel_allreduce_op
                )
                if (
                    share_expert_tp_allreduce_op
                    and share_expert_up_proj_time + share_expert_down_proj_time + share_expert_act_time > 0
                ):
                    raw_share_expert_tp_allreduce_time = self._predict_comm_operator(
                        get_comm_operator(share_expert_tp_allreduce_op),
                        batch,
                    )
                    share_expert_tp_allreduce_time = raw_share_expert_tp_allreduce_time
                    communication_operator_times[
                        share_expert_tp_allreduce_op
                    ] = share_expert_tp_allreduce_time

        dp_input_allreduce_time = 0.0
        dp_output_allreduce_time = 0.0
        if include_ffn and include_moe and self._cluster_type is not None:
            dp_input_allreduce_time, dp_output_allreduce_time = (
                self.predict_dp_moe_allreduce_times(batch, self._cluster_type)
            )
        pp_producer_send_path_runtime_time = self._get_pp_producer_send_path_runtime_time(
            batch, pipeline_stage
        )
        pp_receiver_head_runtime_time = self._get_pp_receiver_head_runtime_time(
            batch, pipeline_stage
        )
        pp_prefill_consumer_active_runtime_time = (
            self._get_pp_prefill_consumer_active_runtime_time(batch, pipeline_stage)
        )
        decode_draft_proposer_time = 0.0
        spec_metadata = getattr(batch, "spec_decode_metadata", None)
        if self._should_include_spec_decode_proposer_overhead(batch):
            decode_draft_proposer_time = self._validate_prediction_value(
                self._get_spec_decode_proposer_overhead_time(
                    batch,
                    method_name=str(spec_metadata.method),
                ),
                "decode_draft_proposer",
                batch,
                f"stage={pipeline_stage}",
            )
        mtp_terminal_overshoot_time = self._validate_prediction_value(
            self._get_mtp_terminal_overshoot_time(
                batch,
                stage_id=pipeline_stage,
                cluster_type=self._cluster_type,
                num_layers=self._num_layers_per_pipeline_stage,
                layer_id=layer_id,
            ),
            "mtp_terminal_overshoot",
            batch,
            f"stage={pipeline_stage}",
        )

        mlp_norm_time = (
            self._get_mlp_norm_layer_act_execution_time(batch) if include_ffn else 0.0
        )

        return ExecutionTime(
            num_layers_per_pipeline_stage=effective_num_layers,
            attention_rope_execution_time=attention_time.attention_rope_execution_time,
            attention_kv_cache_save_execution_time=attention_time.attention_kv_cache_save_execution_time,
            attention_decode_execution_time=attention_time.attention_decode_execution_time,
            attention_prefill_execution_time=attention_time.attention_prefill_execution_time,
            attention_layer_pre_proj_execution_time=attention_time.attention_layer_pre_proj_execution_time,
            attention_layer_post_proj_execution_time=attention_time.attention_layer_post_proj_execution_time,
            attn_norm_time=attention_time.attn_norm_time,
            attention_operator_times=attention_time.operator_times,
            mlp_norm_time=mlp_norm_time,
            add_time=add_time,
            add_attn_residual_time=add_attn_residual_time,
            add_ffn_residual_time=add_ffn_residual_time,
            tensor_parallel_communication_time=attn_tp_allreduce_time,
            attn_tensor_parallel_allreduce_time=attn_tp_allreduce_time,
            moe_tensor_parallel_allreduce_time=moe_tp_allreduce_time,
            pipeline_parallel_communication_time=pipeline_parallel_communication_time,
            expert_parallel_communication_time=expert_parallel_communication_time,
            moe_gating_time=moe_gating_time,
            moe_gating_linear_time=moe_gating_linear_time,
            moe_gating_routing_topk_time=moe_gating_routing_topk_time,
            moe_shuffling_time=moe_shuffling_time,
            schedule_time=self._get_schedule_time(batch),
            sampler_e2e_time=self._get_sampler_e2e_time(batch),
            prepare_inputs_e2e_time=self._get_prepare_inputs_e2e_time(batch),
            process_model_outputs_time=self._get_process_model_outputs_time(batch),
            ray_comm_time=self._get_ray_comm_time(batch),
            pp_producer_send_path_runtime_time=pp_producer_send_path_runtime_time,
            pp_receiver_head_runtime_time=pp_receiver_head_runtime_time,
            pp_prefill_consumer_active_runtime_time=(
                pp_prefill_consumer_active_runtime_time
            ),
            pp_stage_boundary_handoff_time=self._get_pp_stage_boundary_handoff_time(
                batch, pipeline_stage
            ),
            is_moe=bool(include_ffn and include_moe),
            mlp_layer_up_proj_execution_time=mlp_up_proj_time,
            mlp_layer_down_proj_execution_time=mlp_down_proj_time,
            mlp_layer_act_execution_time=mlp_act_time,
            moe_grouped_gemm_time=moe_grouped_gemm_time,
            share_expert_up_proj_time=share_expert_up_proj_time,
            share_expert_down_proj_time=share_expert_down_proj_time,
            share_expert_act_time=share_expert_act_time,
            tensor_parallel_allgather_time=ffn_tp_allgather_time,
            share_expert_tensor_parallel_allreduce_time=share_expert_tp_allreduce_time,
            dp_input_allreduce_time=dp_input_allreduce_time,
            dp_output_allreduce_time=dp_output_allreduce_time,
            decode_draft_proposer_time=decode_draft_proposer_time,
            mtp_terminal_overshoot_time=mtp_terminal_overshoot_time,
            communication_operator_times=CommunicationOperatorTimes(
                communication_operator_times
            ),
            moe_operator_times=(
                _build_moe_operator_times(
                    mlp_norm_time=mlp_norm_time,
                    moe_gating_linear_time=moe_gating_linear_time,
                    moe_gating_routing_topk_time=moe_gating_routing_topk_time,
                    moe_shuffling_time=moe_shuffling_time,
                    moe_grouped_gemm_time=moe_grouped_gemm_time,
                    share_expert_up_proj_time=share_expert_up_proj_time,
                    share_expert_act_time=share_expert_act_time,
                    share_expert_down_proj_time=share_expert_down_proj_time,
                    include_share_expert=self._model_config.supports_share_expert(),
                )
                if include_ffn and include_moe
                else None
            ),
            layer_ids=normalized_layer_ids,
        )

    def _simulate_routing_per_layer(
        self, batches: List[Batch], stage_id: int
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        """
        Simulate routing for each layer in the stage.
        Returns: {layer_id: {replica_id: {moe_component: time_value}}}
        """
        del stage_id
        cluster_type = getattr(self, "_cluster_type", None)
        if not isinstance(cluster_type, ClusterType):
            raise ValueError(
                "layer routing prediction requires an initialized cluster_type"
            )

        # Routing materialization is stage-local and follows the canonical
        # aggregate-to-lane seam.  Predictor consumers receive only physical
        # lane descriptors, even when this legacy helper returns one result per
        # source replica.
        num_layers = self._num_layers_per_pipeline_stage
        layer_routing_results = {}

        for layer_id in range(num_layers):
            layer_routing_results[layer_id] = {}

            for batch in batches:
                replica_id = int(batch.replica_id)
                layer_workload = self._materialize_layer_ep_workload(
                    batch=batch,
                    cluster_type=cluster_type,
                    layer_id=layer_id,
                )
                lane_workloads = tuple(
                    layer_workload.lane(ep_id)
                    for ep_id in layer_workload.participant_ep_ids
                )
                if not lane_workloads:
                    raise ValueError(
                        "layer routing materialization produced no EP lanes: "
                        f"replica_id={replica_id}, layer_id={layer_id}"
                    )
                grouped_gemm_time = max(
                    self._get_grouped_gemm_time(lane_workload, batch=batch)
                    for lane_workload in lane_workloads
                )
                shuffling_time = max(
                    self._get_moe_shuffling_time(
                        batch,
                        moe_tokens_input=lane_workload,
                    )
                    for lane_workload in lane_workloads
                )
                communication_time = max(
                    self._get_expert_parallel_communication_time(
                        batch,
                        lane_workload=lane_workload,
                    )
                    for lane_workload in lane_workloads
                )
                layer_routing_results[layer_id][replica_id] = {
                    "moe_grouped_gemm_time": grouped_gemm_time,
                    "expert_parallel_communication_time": communication_time,
                    "moe_gating_time": self._get_gating_time(batch),
                    "moe_shuffling_time": shuffling_time,
                }

        return layer_routing_results

    # Phase 2.5: Removed deprecated get_moe_stage_execution_details() method
    # MoE models now use predict_moe_layer_time() and other fine-grained APIs

    # ========================================================================
    # New unified API implementation (Phase 0) - MoE extensions
    # ========================================================================

    def predict_moe_lane_phase_times(
        self,
        *,
        batch: Batch,
        lane_workload: EPLaneWorkload,
        pipeline_stage: int,
        cluster_type: ClusterType,
    ) -> tuple[float, float, float, float, float]:
        """Return the five physical MoE EP phase times for one typed lane.

        This seam keeps MTP structural replay on the predictor's normal
        feature/model path while carrying the physical lane explicitly.  It
        does not create scheduler entities or infer a lane from a global map.
        """

        lane_workload = self._admit_routed_ep_aggregate(
            batch,
            routed_moe=True,
            lane_workload=lane_workload,
            conservation_context="predict_moe_lane_phase_times",
        )
        if lane_workload is None:
            raise ValueError("MTP MoE phase prediction requires an EP lane descriptor")
        if cluster_type != self._cluster_type:
            raise ValueError(
                "MTP MoE phase prediction cluster_type does not match predictor: "
                f"requested={cluster_type}, configured={self._cluster_type}"
            )

        if self._enable_dummy_mode:
            execution_time = self._get_dummy_execution_time(
                batch,
                pipeline_stage,
                include_attention=False,
                lane_workload=lane_workload,
            )
        else:
            execution_time = self._get_execution_time_internal(
                batch=batch,
                pipeline_stage=pipeline_stage,
                moe_tokens_input=lane_workload,
                lane_workload=lane_workload,
                include_moe=True,
                include_ffn=True,
                include_attention=False,
            )
        phase_times = (
            float(execution_time.get_single_layer_moe_pre_dispatch_time()),
            float(execution_time.get_single_layer_moe_dispatch_time()),
            float(execution_time.get_single_layer_moe_post_dispatch_compute_time()),
            float(execution_time.get_single_layer_moe_combine_time()),
            float(execution_time.get_single_layer_moe_post_combine_time()),
        )
        if any(not math.isfinite(value) or value < 0 for value in phase_times):
            raise ValueError(
                "MTP MoE lane phase times must be finite and non-negative: "
                f"ep_id={lane_workload.ep_id}, values={phase_times}"
            )
        post_attention_time = float(
            execution_time.get_single_layer_post_attention_time()
        )
        if not math.isfinite(post_attention_time) or post_attention_time < 0:
            raise ValueError(
                "MTP MoE lane post-attention time must be finite and non-negative: "
                f"ep_id={lane_workload.ep_id}, value={post_attention_time}"
            )
        if not math.isclose(
            sum(phase_times),
            post_attention_time,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "MTP MoE lane phase decomposition does not match post-attention "
                f"time: ep_id={lane_workload.ep_id}, phase_sum_ms={sum(phase_times)}, "
                f"post_attention_ms={post_attention_time}"
            )
        return phase_times

    def predict_moe_layer_time(
        self,
        batch_or_group: "Batch | EPBatchGroup",
        layer_id: int,
        cluster_type: ClusterType,
        lane_workload: Optional[EPLaneWorkload] = None,
        ep_size: Optional[int] = None,
        router_topk: Optional[int] = None,
    ) -> MoETime:
        """
        Predict MoE execution time for a single transformer layer.

        The optional ``lane_workload`` is the canonical physical EP-lane
        descriptor.  When the batch entity already carries the descriptor, the
        explicit argument may be omitted.  Raw expert-token mappings are not a
        predictor input because they do not identify a physical topology.

        Args:
            batch_or_group: Batch or EPBatchGroup to predict for
            layer_id: Layer index (0-based)
            cluster_type: Type of cluster (PREFILL, DECODE_FFN, etc.)
            lane_workload: Optional immutable physical EP-lane descriptor.  When
                           omitted, it is resolved from ``batch_or_group``.
            ep_size: Optional active role EP size for topology admission. When
                     omitted, the predictor's configured EP size is used.
            router_topk: Optional active role router top-k for topology admission.
                         When omitted, the predictor's configured top-k is used.

        Returns:
            MoETime component with all MoE-related times

        Raises:
            ValueError: If token conservation is violated
            NotImplementedError: If MoE operations not supported for cluster type
        """
        # This public MoE boundary has already established the routed-MoE
        # operation family.  Enforce the physical lane contract before either
        # dummy timing or operation/model lookup; an explicit descriptor is
        # authoritative when the caller supplies it separately from the batch.
        lane_workload = self._admit_routed_ep_aggregate(
            batch_or_group,
            routed_moe=True,
            ep_size=ep_size,
            router_topk=router_topk,
            lane_workload=lane_workload,
            conservation_context="predict_moe_layer_time",
        ) or lane_workload

        if self._enable_dummy_mode:
            base_time = self._dummy_execution_time
            routed_token_count = self._get_ep_lane_routed_token_count(
                batch_or_group,
                lane_workload=lane_workload,
            )
            zero_routed_ep_lane = routed_token_count == 0
            moe_grouped_gemm_time = 0.0 if zero_routed_ep_lane else base_time
            moe_shuffling_time = 0.0 if zero_routed_ep_lane else base_time
            share_expert_time = (
                base_time if self._model_config.supports_share_expert() else 0.0
            )
            return MoETime(
                moe_grouped_gemm_time=moe_grouped_gemm_time,
                moe_gating_linear_time=base_time * 0.5,
                moe_gating_routing_topk_time=base_time * 0.5,
                moe_shuffling_time=moe_shuffling_time,
                mlp_norm_time=base_time,
                share_expert_up_proj_time=share_expert_time,
                share_expert_down_proj_time=share_expert_time,
                share_expert_act_time=share_expert_time,
                operator_times=_build_moe_operator_times(
                    mlp_norm_time=base_time,
                    moe_gating_linear_time=base_time * 0.5,
                    moe_gating_routing_topk_time=base_time * 0.5,
                    moe_shuffling_time=moe_shuffling_time,
                    moe_grouped_gemm_time=moe_grouped_gemm_time,
                    share_expert_up_proj_time=share_expert_time,
                    share_expert_act_time=share_expert_time,
                    share_expert_down_proj_time=share_expert_time,
                    include_share_expert=self._model_config.supports_share_expert(),
                ),
            )

        if not self._supports_operation("moe_grouped_gemm"):
            raise NotImplementedError(
                f"MoE operations not supported for cluster type {cluster_type}"
            )

        # Extract detailed batch information for logging
        batch_input_lens = (
            [req.num_prefill_tokens for req in batch_or_group.requests]
            if hasattr(batch_or_group, "requests")
            else []
        )
        batch_request_ids = (
            [req.id for req in batch_or_group.requests]
            if hasattr(batch_or_group, "requests")
            else []
        )

        logger.debug(
            f"Predicting MoE layer time for layer_id={layer_id}, cluster_type={cluster_type.name}, "
            f"batch_id={batch_or_group.id if hasattr(batch_or_group, 'id') else 'N/A'}, "
            f"num_tokens={batch_or_group.total_num_tokens if hasattr(batch_or_group, 'total_num_tokens') else 'N/A'}, "
            f"batch_size={len(batch_or_group.requests) if hasattr(batch_or_group, 'requests') else 'N/A'}, "
            f"batch_input_lens={batch_input_lens}, "
            f"batch_request_ids={batch_request_ids}"
        )

        batch = batch_or_group
        if lane_workload is None:
            # EP=1 ordinary batches retain the standard one-feature lookup.
            # Load-aware predictors return the single physical lane here; no
            # synthetic lane is created for the scalar compatibility path.
            moe_tokens_input = self._get_moe_tokens_input(
                batch,
                layer_id=layer_id,
            )
            if isinstance(moe_tokens_input, EPLaneWorkload):
                lane_workload = self._admit_routed_ep_aggregate(
                    batch,
                    routed_moe=True,
                    ep_size=ep_size,
                    router_topk=router_topk,
                    lane_workload=moe_tokens_input,
                )
        else:
            moe_tokens_input = lane_workload
        if lane_workload is not None:
            logger.debug(
                "Using typed EP lane workload: ep_id=%s, local_width=%s, "
                "routed_tokens=%s",
                lane_workload.ep_id,
                lane_workload.local_expert_width,
                lane_workload.routed_token_count,
            )
        grouped_gemm_time = self._get_grouped_gemm_time(
            moe_tokens_input,
            batch=batch,
        )

        # Get individual MoE operation times (compute only, communication is separate)
        gating_linear_time = self._get_gating_linear_time(batch)
        gating_routing_topk_time = self._get_gating_routing_topk_time(batch)
        gating_time = gating_linear_time + gating_routing_topk_time
        shuffling_time = self._get_moe_shuffling_time(
            batch,
            moe_tokens_input=lane_workload,
        )
        # Get post_attention_layernorm time (mlp_norm_time) for MoE models
        # This is the normalization layer before the MoE block
        mlp_norm_time = 0.0
        if self._model_config.post_attn_norm and self._supports_operation(
            "post_attention_layernorm"
        ):
            mlp_norm_time = self._get_mlp_norm_layer_act_execution_time(batch)
        # Note: expert_parallel_communication_time is NOT included in MoETime.
        # It should be obtained separately via _get_expert_parallel_communication_time()
        # to maintain clear separation between compute and communication times.

        # Step2Mini/Step3 share_expert operations (forward_3: shared expert alongside routed experts)
        # These are 0.0 for models without share_expert
        share_expert_up_proj_time = 0.0
        share_expert_down_proj_time = 0.0
        share_expert_act_time = 0.0
        if self._model_config.supports_share_expert():
            share_expert_up_proj_time = self._get_share_expert_up_proj_execution_time(batch)
            share_expert_down_proj_time = self._get_share_expert_down_proj_execution_time(batch)
            share_expert_act_time = self._get_share_expert_act_execution_time(batch)

        # Operation-level tracing for GPU execution (MoE operations)
        # This enables comparison with real vLLM operation-level GPU execution traces
        # Uses cluster_type.name for dynamic cluster identification (supports all cluster types
        # including MONOLITHIC, PREFILL, DECODE, DECODE_FFN, etc.)
        share_expert_total_time = share_expert_up_proj_time + share_expert_down_proj_time + share_expert_act_time
        cluster_name = cluster_type.name

        logger.info(
            f"[OP-TRACE][{cluster_name}][MOE] batch_id={batch.id}, layer_id={layer_id}, "
            f"num_tokens={batch.total_num_tokens}, batch_size={len(batch.requests)}, "
            f"router_topk={self._router_topk}, moe_ep_size={self._moe_ep_size}, moe_tp_size={self._moe_tp_size}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][MOE][post_attention_layernorm] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={mlp_norm_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][MOE][moe_gating] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={gating_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][MOE][moe_shuffling] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={shuffling_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][MOE][moe_grouped_gemm] batch_id={batch.id}, layer_id={layer_id}, "
            f"predicted_time_ms={grouped_gemm_time:.6f}"
        )
        # Step2Mini/Step3 share_expert operation tracing
        if self._model_config.supports_share_expert():
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][share_expert_up_proj] batch_id={batch.id}, layer_id={layer_id}, "
                f"predicted_time_ms={share_expert_up_proj_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][share_expert_act] batch_id={batch.id}, layer_id={layer_id}, "
                f"predicted_time_ms={share_expert_act_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][share_expert_down_proj] batch_id={batch.id}, layer_id={layer_id}, "
                f"predicted_time_ms={share_expert_down_proj_time:.6f}"
            )
        total_moe_time = (
            mlp_norm_time + gating_time + shuffling_time + grouped_gemm_time
            + share_expert_total_time  # Step2Mini-specific (0.0 for non-Step2Mini)
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][MOE][TOTAL] batch_id={batch.id}, layer_id={layer_id}, "
            f"total_moe_time_ms={total_moe_time:.6f}"
        )

        return MoETime(
            moe_grouped_gemm_time=grouped_gemm_time,
            moe_gating_linear_time=gating_linear_time,
            moe_gating_routing_topk_time=gating_routing_topk_time,
            moe_shuffling_time=shuffling_time,
            mlp_norm_time=mlp_norm_time,
            # Step2Mini-specific operations (0.0 for non-Step2Mini models)
            share_expert_up_proj_time=share_expert_up_proj_time,
            share_expert_down_proj_time=share_expert_down_proj_time,
            share_expert_act_time=share_expert_act_time,
            operator_times=_build_moe_operator_times(
                mlp_norm_time=mlp_norm_time,
                moe_gating_linear_time=gating_linear_time,
                moe_gating_routing_topk_time=gating_routing_topk_time,
                moe_shuffling_time=shuffling_time,
                moe_grouped_gemm_time=grouped_gemm_time,
                share_expert_up_proj_time=share_expert_up_proj_time,
                share_expert_act_time=share_expert_act_time,
                share_expert_down_proj_time=share_expert_down_proj_time,
                include_share_expert=self._model_config.supports_share_expert(),
            ),
        )

    def predict_allgather_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict expert parallel all-gather communication time.

        Delegates to CC Backend if available, otherwise falls back to dummy mode.

        Used for aggregating MoE results across EP replicas in DECODE_FFN cluster.

        Args:
            data_size_bytes: Size of data per device in bytes
            num_devices: Number of participating devices
            cluster_type: Type of cluster for context-aware prediction

        Returns:
            Predicted execution time in milliseconds
        """
        # Use CC Backend if available for communication predictions
        if self._cc_backend is not None:
            result = self._cc_backend.predict_allgather(
                data_size_bytes=data_size_bytes,
                num_devices=num_devices,
                cluster_type=cluster_type,
                comm_domain=comm_domain,
            )
            logger.debug(
                f"predict_allgather_time (MoE): using CC Backend, "
                f"data_size={data_size_bytes}, num_devices={num_devices}, result={result:.6f} ms"
            )
            return result

        raise NotImplementedError("MoE all-gather prediction not implemented")
        # return self._dummy_execution_time

    def predict_alltoall_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict expert parallel all-to-all communication time.

        Delegates to CC Backend if available, otherwise falls back to dummy mode.

        Used for MoE token dispatch/return in DECODE_FFN cluster.

        Args:
            data_size_bytes: Total size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Type of cluster for context-aware prediction

        Returns:
            Predicted execution time in milliseconds
        """
        # Use CC Backend if available for communication predictions
        if self._cc_backend is not None:
            result = self._cc_backend.predict_all_to_all(
                data_size_bytes=data_size_bytes,
                num_devices=num_devices,
                cluster_type=cluster_type,
                comm_domain=comm_domain,
            )
            logger.debug(
                f"predict_alltoall_time (MoE): using CC Backend, "
                f"data_size={data_size_bytes}, num_devices={num_devices}, result={result:.6f} ms"
            )
            return result

        raise NotImplementedError("MoE all-to-all prediction not implemented")
        # return self._dummy_execution_time

    def _predict_mtp_moe_lane_phase_aggregate(
        self,
        *,
        predictor,
        batch: Batch,
        pipeline_stage: int,
        cluster_type: ClusterType,
        layer_id: int,
        num_layers: int,
    ) -> tuple[ExecutionTime, tuple[float, float, float, float, float]]:
        """Return one shared attention result and the five lane barriers.

        ``predictor`` is explicit because structural MTP may run against a
        secondary predictor owned by this parent.  The attention probe is kept
        at one layer: pipeline and CPU overhead are batch-level terms, while
        the returned physical phase barriers are the only values scaled by
        ``num_layers`` at the caller.
        """

        if type(num_layers) is not int or num_layers < 1:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}")

        attention_execution_time = predictor.predict_stage_execution_time(
            batch=batch,
            stage_id=pipeline_stage,
            cluster_type=cluster_type,
            num_layers=1,
            layer_id=layer_id,
            include_ffn=False,
        )
        attention_time_ms = float(attention_execution_time.model_time_ms)
        if not math.isfinite(attention_time_ms) or attention_time_ms < 0:
            raise ValueError(
                "MTP structural attention time must be finite and non-negative, "
                f"got {attention_time_ms}"
            )

        workload = predictor._materialize_layer_ep_workload(
            batch=batch,
            cluster_type=cluster_type,
            layer_id=layer_id,
        )
        participant_ep_ids = tuple(workload.participant_ep_ids)
        if not participant_ep_ids:
            raise ValueError("MTP MoE replay produced no EP participants")

        effective_tokens = int(
            batch.get_effective_total_tokens_for_compute(cluster_type)
        )
        if effective_tokens <= 0:
            raise ValueError(
                "MTP MoE replay requires positive pre-routing effective tokens, "
                f"got {effective_tokens}"
            )

        phase_values: list[list[float]] = []
        for ep_id in participant_ep_ids:
            lane_workload = workload.lane(int(ep_id))
            lane_phases = predictor.predict_moe_lane_phase_times(
                batch=batch,
                lane_workload=lane_workload,
                pipeline_stage=pipeline_stage,
                cluster_type=cluster_type,
            )
            if len(lane_phases) != 5:
                raise ValueError(
                    "MTP MoE lane phase API must return five values, "
                    f"got ep_id={ep_id}, values={lane_phases!r}"
                )
            normalized_phases = [float(value) for value in lane_phases]
            if any(
                not math.isfinite(value) or value < 0
                for value in normalized_phases
            ):
                raise ValueError(
                    "MTP MoE lane phase times must be finite and non-negative, "
                    f"got ep_id={ep_id}, values={normalized_phases}"
                )
            phase_values.append(normalized_phases)

        phase_maxima = tuple(
            max(values[index] for values in phase_values) for index in range(5)
        )
        return attention_execution_time, phase_maxima

    def _predict_mtp_terminal_row_time_ms(
        self,
        *,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int,
        layer_id: int,
    ) -> float:
        """Predict a terminal MTP row with physical EP barriers when required."""

        model_config = getattr(self, "_model_config", None)
        if model_config is None or not bool(getattr(model_config, "is_moe", False)):
            return super()._predict_mtp_terminal_row_time_ms(
                batch=batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
            )
        is_moe_layer = getattr(model_config, "is_moe_layer", None)
        if not callable(is_moe_layer):
            raise ValueError(
                "MTP terminal MoE prediction requires model_config.is_moe_layer"
            )
        if not bool(is_moe_layer(layer_id)):
            return super()._predict_mtp_terminal_row_time_ms(
                batch=batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
            )
        if cluster_type not in (ClusterType.MONOLITHIC, ClusterType.DECODE):
            return super()._predict_mtp_terminal_row_time_ms(
                batch=batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
            )
        if int(getattr(self, "_moe_ep_size", 1)) <= 1:
            return super()._predict_mtp_terminal_row_time_ms(
                batch=batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
            )

        attention_execution_time, phase_maxima = (
            self._predict_mtp_moe_lane_phase_aggregate(
                predictor=self,
                batch=batch,
                pipeline_stage=stage_id,
                cluster_type=cluster_type,
                layer_id=layer_id,
                num_layers=num_layers,
            )
        )
        attention_time_ms = float(attention_execution_time.total_time * 1e3)
        if not math.isfinite(attention_time_ms) or attention_time_ms < 0:
            raise ValueError(
                "MTP terminal attention time must be finite and non-negative, "
                f"got {attention_time_ms}"
            )
        return attention_time_ms + sum(phase_maxima) * int(num_layers)

    def _predict_mtp_decoder_layer_time_ms(
        self,
        *,
        predictor,
        batch: Batch,
    ) -> float:
        layer_id = 0
        model_config = getattr(predictor, "_model_config", None)
        if model_config is None:
            raise ValueError(
                "MTP structural decoder prediction requires model_config"
            )
        if not bool(getattr(model_config, "is_moe", False)):
            return super()._predict_mtp_decoder_layer_time_ms(
                predictor=predictor,
                batch=batch,
            )

        is_moe_layer = getattr(model_config, "is_moe_layer", None)
        if not callable(is_moe_layer):
            raise ValueError(
                "MTP structural MoE decoder prediction requires "
                "model_config.is_moe_layer"
            )
        if not bool(is_moe_layer(layer_id)):
            return super()._predict_mtp_decoder_layer_time_ms(
                predictor=predictor,
                batch=batch,
            )

        cluster_type = getattr(predictor, "_cluster_type", None)
        if not isinstance(cluster_type, ClusterType):
            raise ValueError(
                "MTP structural MoE decoder prediction requires a valid cluster_type"
            )

        attention_execution_time, phase_maxima = (
            self._predict_mtp_moe_lane_phase_aggregate(
                predictor=predictor,
                batch=batch,
                pipeline_stage=0,
                cluster_type=cluster_type,
                layer_id=layer_id,
                num_layers=1,
            )
        )
        attention_time_ms = float(attention_execution_time.model_time_ms)
        return attention_time_ms + sum(phase_maxima)

    def predict_stage_execution_time(
        self,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int = 1,
        layer_id: int = 0,
        layer_ids: Optional[List[int] | tuple[int, ...]] = None,
        include_moe: bool | None = None,
        include_ffn: bool = True,
        include_attention: bool = True,
    ) -> ExecutionTime:
        """
        Predict execution time for MoE models using per-layer component semantics.

        Predictor components are represented as single-layer times (milliseconds), while
        ExecutionTime aggregates across ``num_layers_per_pipeline_stage``.
        Therefore, changing ``num_layers`` must update only the layer count, not rescale
        per-layer components.
        """
        normalized_layer_ids = self._normalize_stage_layer_ids(
            num_layers=num_layers,
            layer_id=layer_id,
            layer_ids=layer_ids,
        )
        if normalized_layer_ids is not None:
            layer_id = normalized_layer_ids[0]
        if type(include_ffn) is not bool:
            raise ValueError("include_ffn must be a bool")
        if type(include_attention) is not bool:
            raise ValueError("include_attention must be a bool")
        if not include_attention and (
            cluster_type not in (ClusterType.MONOLITHIC, ClusterType.DECODE)
            or not include_ffn
        ):
            raise ValueError(
                "Post-attention-only prediction requires MONOLITHIC or unified "
                "DECODE with FFN enabled"
            )
        if not include_ffn and include_moe is not None:
            raise ValueError(
                "include_moe must be None for an attention-only stage probe"
            )

        if include_moe is not None and type(include_moe) is not bool:
            raise ValueError("include_moe must be a bool or None")
        if not include_attention and include_moe is False:
            raise ValueError(
                "Post-attention-only prediction requires a MoE layer; "
                "include_moe=False selects a dense FFN branch"
            )

        # Resolve the existing concrete layer/aggregate classification before
        # either dummy timing or profiling-backed measurement work.  An
        # identity-free aggregate uses the model-level MoE capability, while a
        # concrete layer uses the model-owned layer predicate.
        include_moe_for_layer = self._resolve_moe_layer_classification(
            self._model_config,
            layer_id=layer_id,
            num_layers=num_layers,
            include_moe=include_moe,
            include_ffn=include_ffn,
        )

        if not include_attention and not include_moe_for_layer:
            raise ValueError(
                "Post-attention-only prediction requires a MoE layer; "
                f"layer_id={layer_id} is dense"
            )

        # Every routed public call crosses the typed-lane admission boundary
        # before mode-specific timing or lookup work.
        self._admit_routed_ep_aggregate(
            batch,
            routed_moe=include_moe_for_layer,
        )

        if self._enable_dummy_mode:
            return self._get_dummy_execution_time(
                batch,
                stage_id,
                num_layers=num_layers,
                layer_id=layer_id,
                layer_ids=normalized_layer_ids,
                include_attention=include_attention,
                include_ffn=include_ffn,
                include_moe=include_moe_for_layer,
            )

        logger.debug(
            "[EXEC_TIME_PREDICT_MOE] stage_id=%s, cluster_type=%s, num_layers=%s, "
            "layer_id=%s, batch_id=%s, batch_size=%s, num_tokens=%s",
            stage_id,
            cluster_type,
            num_layers,
            layer_id,
            batch.id,
            batch.size,
            batch.num_tokens,
        )

        measurement_type = self._select_measurement_type_for_batch(batch)
        self._require_predictions_for_measurement_type(measurement_type, batch)
        self._activate_measurement_type(measurement_type)
        self._emit_cuda_graph_activation_records(
            batch,
            measurement_type,
            cluster_type,
        )

        # Validate cluster_type consistency
        if self._cluster_type is not None and cluster_type != self._cluster_type:
            logger.error(
                f"Cluster type mismatch: predictor initialized with {self._cluster_type}, "
                f"but predict_stage_execution_time called with {cluster_type}"
            )

        moe_tokens_input = None
        if include_moe_for_layer:
            # Use the canonical distribution source for per-layer MoE input
            # selection. EP lane batches carry their materialized map directly.
            moe_tokens_input = self._get_moe_tokens_input(batch, layer_id=layer_id)

            if isinstance(moe_tokens_input, EPLaneWorkload):
                logger.debug(
                    "[EXEC_TIME_PREDICT_MOE] Using typed EP lane: ep_id=%s, "
                    "local_width=%s, routed_tokens=%s, layer_id=%s",
                    moe_tokens_input.ep_id,
                    moe_tokens_input.local_expert_width,
                    moe_tokens_input.routed_token_count,
                    layer_id,
                )
            else:
                logger.debug(
                    "[EXEC_TIME_PREDICT_MOE] Using %s mode with post_routing_batch_tokens=%s, "
                    "layer_id=%s",
                    self._moe_routing_distribution_type,
                    moe_tokens_input,
                    layer_id,
                )
        else:
            logger.debug(
                "[EXEC_TIME_PREDICT_MOE] layer_id=%s is dense-only by moe_layers_enum; "
                "skip MoE compute/comm components",
                layer_id,
            )

        base_execution_time = self._get_execution_time_internal(
            batch,
            stage_id,
            moe_tokens_input=moe_tokens_input,
            include_moe=include_moe_for_layer,
            include_ffn=include_ffn,
            include_attention=include_attention,
            layer_id=layer_id,
            num_layers=num_layers,
            layer_ids=normalized_layer_ids,
        )

        # Communication OP-TRACE: log per-layer allreduce times for op-level comparison
        cluster_name = cluster_type.name
        logger.info(
            f"[OP-TRACE][{cluster_name}][COMM][attn_tp_allreduce] batch_id={batch.id}, "
            f"layer_id={layer_id}, predicted_time_ms="
            f"{base_execution_time._attn_tensor_parallel_allreduce_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][COMM][moe_tp_allreduce] batch_id={batch.id}, "
            f"layer_id={layer_id}, predicted_time_ms="
            f"{base_execution_time._moe_tensor_parallel_allreduce_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][COMM][share_expert_tp_allreduce] batch_id={batch.id}, "
            f"layer_id={layer_id}, predicted_time_ms="
            f"{base_execution_time._share_expert_tensor_parallel_allreduce_time:.6f}"
        )

        # Attention OP-TRACE: log per-layer attention op times
        et = base_execution_time
        prefill_op_name = get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.PREFILL_KERNEL,
        )
        cache_write_op_name = get_enabled_predictor_metric_name_by_role(
            DENSE_ATTENTION_FAMILY,
            AttentionOperatorRole.CACHE_WRITE,
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][input_layernorm] batch_id={batch.id}, "
            f"layer_id={layer_id}, predicted_time_ms={et._attn_norm_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][attn_pre_proj] batch_id={batch.id}, "
            f"layer_id={layer_id}, predicted_time_ms={et._attention_layer_pre_proj_execution_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][attn_rope] batch_id={batch.id}, "
            f"layer_id={layer_id}, predicted_time_ms={et._attention_rope_execution_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][{prefill_op_name}] batch_id={batch.id}, "
            f"layer_id={layer_id}, predicted_time_ms={et._attention_prefill_execution_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][{cache_write_op_name}] batch_id={batch.id}, "
            f"layer_id={layer_id}, predicted_time_ms={et._attention_kv_cache_save_execution_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][ATTENTION][attn_post_proj] batch_id={batch.id}, "
            f"layer_id={layer_id}, predicted_time_ms={et._attention_layer_post_proj_execution_time:.6f}"
        )

        if include_ffn and include_moe_for_layer:
            # MOE OP-TRACE: log only the routed-expert protocol for an actual
            # MoE layer.  Dense layers in a mixed model must not be counted as
            # EP work merely because the model has an MoE-capable topology.
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][post_attention_layernorm] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._mlp_norm_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][moe_gating] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._moe_gating_routing_topk_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][moe_gating_linear] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._moe_gating_linear_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][moe_gating_routing_topk] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._moe_gating_routing_topk_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][moe_shuffling] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._moe_shuffling_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][moe_grouped_gemm] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._moe_grouped_gemm_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][add] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._add_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][add_attn_residual] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._add_attn_residual_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][MOE][add_ffn_residual] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._add_ffn_residual_time:.6f}"
            )
        elif include_ffn:
            # Dense OP-TRACE: this is a FULL_STAGE_WORLD FFN operation.  The
            # component fields may be populated from shared-expert profile
            # rows for Step2Mini/Step3 models, but the protocol remains dense.
            logger.info(
                f"[OP-TRACE][{cluster_name}][DENSE_FFN][post_attention_layernorm] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._mlp_norm_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][DENSE_FFN][mlp_up_proj] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._mlp_layer_up_proj_execution_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][DENSE_FFN][mlp_down_proj] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._mlp_layer_down_proj_execution_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][DENSE_FFN][mlp_act] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._mlp_layer_act_execution_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][DENSE_FFN][add] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._add_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][DENSE_FFN][add_attn_residual] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._add_attn_residual_time:.6f}"
            )
            logger.info(
                f"[OP-TRACE][{cluster_name}][DENSE_FFN][add_ffn_residual] batch_id={batch.id}, "
                f"layer_id={layer_id}, predicted_time_ms={et._add_ffn_residual_time:.6f}"
            )
        logger.info(
            f"[OP-TRACE][{cluster_name}][SPEC_DECODE][decode_draft_proposer] "
            f"batch_id={batch.id}, layer_id={layer_id}, predicted_time_ms="
            f"{et._decode_draft_proposer_time:.6f}"
        )
        logger.info(
            f"[OP-TRACE][{cluster_name}][SPEC_DECODE][mtp_terminal_overshoot] "
            f"batch_id={batch.id}, layer_id={layer_id}, predicted_time_ms="
            f"{et._mtp_terminal_overshoot_time:.6f}"
        )

        # Fast path: requested layer count matches base predictor stage layer count.
        if num_layers == self._num_layers_per_pipeline_stage:
            return base_execution_time

        logger.debug(
            "[EXEC_TIME_PREDICT_MOE] Create ExecutionTime view with num_layers=%s "
            "from per-layer components (base_num_layers=%s)",
            num_layers,
            self._num_layers_per_pipeline_stage,
        )

        # Keep all per-layer components unchanged; only update the aggregation layer count.
        return ExecutionTime(
            num_layers_per_pipeline_stage=num_layers,
            attention_rope_execution_time=base_execution_time._attention_rope_execution_time,
            attention_kv_cache_save_execution_time=base_execution_time._attention_kv_cache_save_execution_time,
            attention_decode_execution_time=base_execution_time._attention_decode_execution_time,
            attention_prefill_execution_time=base_execution_time._attention_prefill_execution_time,
            attention_layer_pre_proj_execution_time=base_execution_time._attention_layer_pre_proj_execution_time,
            attention_layer_post_proj_execution_time=base_execution_time._attention_layer_post_proj_execution_time,
            attn_norm_time=base_execution_time._attn_norm_time,
            mlp_norm_time=base_execution_time._mlp_norm_time,
            add_time=base_execution_time._add_time,
            add_attn_residual_time=base_execution_time._add_attn_residual_time,
            add_ffn_residual_time=base_execution_time._add_ffn_residual_time,
            tensor_parallel_communication_time=base_execution_time._tensor_parallel_communication_time,
            attn_tensor_parallel_allreduce_time=(
                base_execution_time._attn_tensor_parallel_allreduce_time
                if base_execution_time._has_attn_tensor_parallel_allreduce_time
                else None
            ),
            moe_tensor_parallel_allreduce_time=(
                base_execution_time._moe_tensor_parallel_allreduce_time
                if base_execution_time._has_moe_tensor_parallel_allreduce_time
                else None
            ),
            tensor_parallel_allgather_time=base_execution_time._tensor_parallel_allgather_time,
            share_expert_tensor_parallel_allreduce_time=base_execution_time._share_expert_tensor_parallel_allreduce_time,
            dp_input_allreduce_time=base_execution_time._dp_input_allreduce_time,
            dp_output_allreduce_time=base_execution_time._dp_output_allreduce_time,
            pipeline_parallel_communication_time=base_execution_time._pipeline_parallel_communication_time,
            expert_parallel_communication_time=base_execution_time._expert_parallel_communication_time,
            moe_gating_time=base_execution_time._moe_gating_time,
            moe_gating_linear_time=base_execution_time._moe_gating_linear_time,
            moe_gating_routing_topk_time=base_execution_time._moe_gating_routing_topk_time,
            moe_shuffling_time=base_execution_time._moe_shuffling_time,
            schedule_time=base_execution_time._schedule_time,
            sampler_e2e_time=base_execution_time._sampler_e2e_time,
            prepare_inputs_e2e_time=base_execution_time._prepare_inputs_e2e_time,
            process_model_outputs_time=base_execution_time._process_model_outputs_time,
            ray_comm_time=base_execution_time._ray_comm_time,
            pp_producer_send_path_runtime_time=base_execution_time._pp_producer_send_path_runtime_time,
            pp_receiver_head_runtime_time=base_execution_time._pp_receiver_head_runtime_time,
            pp_prefill_consumer_active_runtime_time=base_execution_time._pp_prefill_consumer_active_runtime_time,
            pp_stage_boundary_handoff_time=base_execution_time._pp_stage_boundary_handoff_time,
            is_moe=base_execution_time._is_moe,
            mlp_layer_up_proj_execution_time=base_execution_time._mlp_layer_up_proj_execution_time,
            mlp_layer_down_proj_execution_time=base_execution_time._mlp_layer_down_proj_execution_time,
            mlp_layer_act_execution_time=base_execution_time._mlp_layer_act_execution_time,
            moe_grouped_gemm_time=base_execution_time._moe_grouped_gemm_time,
            share_expert_up_proj_time=base_execution_time._share_expert_up_proj_time,
            share_expert_down_proj_time=base_execution_time._share_expert_down_proj_time,
            share_expert_act_time=base_execution_time._share_expert_act_time,
            decode_draft_proposer_time=base_execution_time._decode_draft_proposer_time,
            mtp_terminal_overshoot_time=base_execution_time._mtp_terminal_overshoot_time,
            attention_operator_times=base_execution_time.attention_operator_times,
            communication_operator_times=base_execution_time.communication_operator_times,
            moe_operator_times=base_execution_time.moe_operator_times,
            layer_ids=normalized_layer_ids,
        )
