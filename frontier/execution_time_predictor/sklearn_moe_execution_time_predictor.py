import json
import math
import os
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import AttentionOperatorRole
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_metric_name_by_role,
)
from frontier.entities import Batch, EPBatchGroup, ExecutionTime, StageExecutionTime
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
from frontier.model_architectures import ResidualAddPolicy
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
    PrefillHotRowsUnavailableError,
    PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT,
    filter_moe_gating_rows_by_runtime_context,
    get_moe_gating_base_model_name,
    get_moe_gating_prediction_model_name,
    has_prefill_hot_moe_gating_rows,
    should_enable_prefill_hot_moe_gating_contract,
    should_use_prefill_hot_moe_gating_context,
)
from frontier.moe_routing_runtime import (
    filter_moe_gating_routing_topk_rows,
    resolve_moe_gating_routing_runtime_path,
)
from frontier.moe_ep_workload import (
    EPLaneWorkload,
    LayerEPWorkload,
    build_contiguous_expert_ownership,
    generate_moe_routing_ratios,
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
from frontier.operators.typed_contracts import (
    TYPED_OPERATOR_CONTRACTS_COLUMN,
    validate_typed_operator_contracts,
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
from frontier.config.parallel_semantics import (
    resolve_shared_expert_tensor_parallel_size,
)
from frontier.types import ClusterType
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)

logger = init_logger(__name__)


from frontier.execution_time_predictor.moe_dataset_training import (
    MoeDatasetTraining,
)
from frontier.execution_time_predictor.moe_mtp_replay import MoeMtpReplay
from frontier.execution_time_predictor.moe_operator_times import MoeOperatorTimes
from frontier.execution_time_predictor.moe_predictor_helpers import (
    _build_moe_operator_times,
    _get_moe_family_model_names,
    _get_moe_family_operator_by_model_name,
    _get_moe_gating_family_model_names,
    _get_prefill_hot_moe_gating_model_names,
    _is_moe_gating_family_model_name,
    _normalize_routing_details_for_trace,
    _validate_moe_columns,
)
from frontier.execution_time_predictor.moe_routing_workload import (
    MoeRoutingWorkload,
)


class SklearnMoEExecutionTimePredictor(
    MoeOperatorTimes,
    MoeRoutingWorkload,
    MoeDatasetTraining,
    MoeMtpReplay,
    SklearnExecutionTimePredictor,
):
    # Layer routing is deterministic for a predictor and the resulting
    # workload is immutable. Keep a bounded cache because the same
    # replica/layer/token shape is revisited across EP waves, while a
    # long-lived predictor must not retain an unbounded request history.
    _LAYER_WORKLOAD_CACHE_CAPACITY = 256


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
        include_attention: bool = True,
        include_ffn: bool = True,
        include_moe: Optional[bool] = None,
        lane_workload: Optional[EPLaneWorkload] = None,
        include_stage_owned: bool = True,
    ) -> ExecutionTime:
        """Return fixed dummy ExecutionTime object with MoE-aware fields."""
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
        dp_input_allreduce_time = 0.0
        dp_output_allreduce_time = 0.0

        ffn_tp_allgather_time = 0.0
        share_expert_tp_allreduce_time = 0.0
        if (
            is_moe
            and architecture_profile.moe_tensor_parallel_allgather_op
            and moe_tp_size > 1
        ):
            ffn_tp_allgather_time = base_time
        if (
            share_expert_enabled
            and architecture_profile.share_expert_tensor_parallel_allreduce_op
            and resolve_shared_expert_tensor_parallel_size(
                cluster_type=self._cluster_type,
                replica_config=self._replica_config,
                model_config=self._model_config,
            )
            > 1
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
            if include_stage_owned and pipeline_stage < self._replica_config.num_pipeline_stages - 1
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
            num_layers_per_pipeline_stage=1,
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
            pipeline_parallel_communication_time=base_time if include_stage_owned else 0.0,
            expert_parallel_communication_time=expert_parallel_comm_time,
            moe_gating_time=base_time if is_moe else 0.0,
            moe_shuffling_time=(
                0.0 if zero_routed_ep_lane else base_time
            ) if is_moe else 0.0,
            schedule_time=base_time if include_stage_owned else 0.0,
            sampler_e2e_time=base_time if include_stage_owned else 0.0,
            prepare_inputs_e2e_time=base_time if include_stage_owned else 0.0,
            process_model_outputs_time=base_time if include_stage_owned else 0.0,
            ray_comm_time=base_time if include_stage_owned else 0.0,
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
        actual_replica_ids: Optional[list] = None,
    ) -> None:
        self._is_moe = True
        self._router_topk = replica_config.router_topk
        self._moe_tp_size = replica_config.moe_tensor_parallel_size
        self._moe_ep_size = replica_config.moe_expert_parallel_size
        self._actual_replica_ids = actual_replica_ids
        self._attention_query_cache_hits = 0
        self._attention_query_cache_misses = 0
        self._layer_workload_cache_capacity = self._LAYER_WORKLOAD_CACHE_CAPACITY
        self._layer_workload_cache = OrderedDict()

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
        self._monolithic_routing_details = None
        self._global_routing_allocations = self._init_global_routing_allocations()
        if self._cluster_type == ClusterType.MONOLITHIC and self._model_config.is_moe:
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

    def _predict_attention_layer_time_with_query_cache(
        self,
        *,
        batch: Batch,
        layer_id: int,
        cluster_type: ClusterType,
        cache: dict[tuple[str, str], AttentionTime] | None = None,
    ) -> AttentionTime:
        """Reuse numerics only within one synchronous stage prediction.

        The stage owns one batch, cluster, predictor configuration and loaded
        artifact set. Phase, context, padding, state initialization, physical
        shape and TP/quantization therefore cannot change during this lifetime.
        Only the normalized attention family/variant varies between layers.
        Global layer identity and MoE routing remain outside this numeric cache.
        Direct layer calls do not retain numerical results across requests.
        """
        if cache is None:
            return self.predict_attention_layer_time(
                batch=batch, layer_id=layer_id, cluster_type=cluster_type,
            )
        spec = self._model_config.get_layer_attention_spec(layer_id)
        key = (spec.family_id, spec.variant_id)
        cached = cache.get(key)
        if cached is not None:
            self._attention_query_cache_hits += 1
            return self._clone_attention_time(cached)
        self._attention_query_cache_misses += 1
        result = self.predict_attention_layer_time(
            batch=batch, layer_id=layer_id, cluster_type=cluster_type,
        )
        cache[key] = self._clone_attention_time(result)
        return result

    @staticmethod

    def _clone_attention_time(value: AttentionTime) -> AttentionTime:
        """Clone scalar attention timings without recursive object copying."""

        if not isinstance(value, AttentionTime):
            raise TypeError(
                "attention query cache requires an AttentionTime result, "
                f"got {type(value).__name__}"
            )
        operator_times = value.operator_times
        if operator_times is not None:
            operator_times = type(operator_times)(dict(operator_times.op_times))
        return replace(value, operator_times=operator_times)

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
        attention_query_cache: dict[tuple[str, str], AttentionTime] | None = None,
        include_stage_owned: bool = True,
        stage_num_layers: int = 1,
    ) -> "ExecutionTime":
        """
        Calculate one physical layer and, for its owner, stage-level work.

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
                attention and terminal-MTP prediction.
            include_stage_owned: Whether this physical layer owns the stage's
                CPU, PP, proposer, and terminal-MTP values.
            stage_num_layers: Actual public stage range for terminal-MTP replay.

        Returns:
            ExecutionTime with all component times

        Raises:
            ValueError: If include_moe=True but moe_tokens_input is None (fail-fast)
        """
        if type(include_ffn) is not bool:
            raise ValueError("include_ffn must be a bool")
        if type(include_attention) is not bool:
            raise ValueError("include_attention must be a bool")
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
            self._predict_attention_layer_time_with_query_cache(
                batch=batch,
                layer_id=layer_id,
                cluster_type=self._cluster_type,
                cache=attention_query_cache,
            )
            if include_attention
            else AttentionTime()
        )

        communication_operator_times: dict[str, float] = {}

        if not include_stage_owned or pipeline_stage == self._replica_config.num_pipeline_stages - 1:
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
        if include_ffn and include_moe:
            if (
                moe_tp_allgather_op
                and self._replica_config.moe_tensor_parallel_size > 1
            ):
                ffn_tp_allgather_time = self._predict_comm_operator(
                    get_comm_operator(moe_tp_allgather_op),
                    batch,
                )
                communication_operator_times[moe_tp_allgather_op] = ffn_tp_allgather_time
            share_expert_tp_allreduce_op = (
                architecture_profile.share_expert_tensor_parallel_allreduce_op
            )
            shared_expert_compute_time = (
                share_expert_up_proj_time
                + share_expert_down_proj_time
                + share_expert_act_time
            )
            if (
                share_expert_tp_allreduce_op
                and shared_expert_compute_time > 0
                and resolve_shared_expert_tensor_parallel_size(
                    cluster_type=self._cluster_type,
                    replica_config=self._replica_config,
                    model_config=self._model_config,
                )
                > 1
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
        pp_producer_send_path_runtime_time = (
            self._get_pp_producer_send_path_runtime_time(batch, pipeline_stage)
            if include_stage_owned else 0.0
        )
        pp_receiver_head_runtime_time = (
            self._get_pp_receiver_head_runtime_time(batch, pipeline_stage)
            if include_stage_owned else 0.0
        )
        pp_prefill_consumer_active_runtime_time = (
            self._get_pp_prefill_consumer_active_runtime_time(batch, pipeline_stage)
            if include_stage_owned else 0.0
        )
        decode_draft_proposer_time = 0.0
        spec_metadata = getattr(batch, "spec_decode_metadata", None)
        if include_stage_owned and self._should_include_spec_decode_proposer_overhead(batch):
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
                num_layers=stage_num_layers,
                layer_id=layer_id,
            ),
            "mtp_terminal_overshoot",
            batch,
            f"stage={pipeline_stage}",
        ) if include_stage_owned else 0.0

        mlp_norm_time = (
            self._get_mlp_norm_layer_act_execution_time(batch) if include_ffn else 0.0
        )


        return ExecutionTime(
            num_layers_per_pipeline_stage=1,
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
            schedule_time=self._get_schedule_time(batch) if include_stage_owned else 0.0,
            sampler_e2e_time=self._get_sampler_e2e_time(batch) if include_stage_owned else 0.0,
            prepare_inputs_e2e_time=self._get_prepare_inputs_e2e_time(batch) if include_stage_owned else 0.0,
            process_model_outputs_time=self._get_process_model_outputs_time(batch) if include_stage_owned else 0.0,
            ray_comm_time=self._get_ray_comm_time(batch) if include_stage_owned else 0.0,
            pp_producer_send_path_runtime_time=pp_producer_send_path_runtime_time,
            pp_receiver_head_runtime_time=pp_receiver_head_runtime_time,
            pp_prefill_consumer_active_runtime_time=(
                pp_prefill_consumer_active_runtime_time
            ),
            pp_stage_boundary_handoff_time=self._get_pp_stage_boundary_handoff_time(
                batch, pipeline_stage
            ) if include_stage_owned else 0.0,
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
        )

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

    def predict_stage_execution_time(
        self,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int = 1,
        layer_id: int = 0,
        include_moe: bool | None = None,
        include_ffn: bool = True,
        include_attention: bool = True,
    ) -> StageExecutionTime:
        """Predict layer-specific routing with a shared stage-local attention cache."""
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive int")
        cache: dict[tuple[str, str], AttentionTime] = {}
        layers = [
            self._predict_moe_layer_execution_time(
                batch, stage_id, cluster_type, layer_id + offset,
                include_moe, include_ffn, include_attention, cache,
                include_stage_owned=offset == 0,
                stage_num_layers=num_layers,
            )
            for offset in range(num_layers)
        ]
        return self._assemble_stage(layers, first_layer_id=layer_id)

    def _predict_moe_layer_execution_time(
        self,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        layer_id: int = 0,
        include_moe: bool | None = None,
        include_ffn: bool = True,
        include_attention: bool = True,
        _attention_query_cache: dict[tuple[str, str], AttentionTime] | None = None,
        *,
        include_stage_owned: bool = True,
        stage_num_layers: int = 1,
    ) -> ExecutionTime:
        """Predict one physical layer, retaining its independent MoE routing."""
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

        # Classify the actual physical layer before dummy or profiled work.
        include_moe_for_layer = self._resolve_moe_layer_classification(
            self._model_config,
            layer_id=layer_id,
            num_layers=1,
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
            dummy_execution_time = self._get_dummy_execution_time(
                batch,
                stage_id,
                include_attention=include_attention,
                include_ffn=include_ffn,
                include_moe=include_moe_for_layer,
                include_stage_owned=include_stage_owned,
            )
            return dummy_execution_time

        logger.debug(
            "[EXEC_TIME_PREDICT_MOE] stage_id=%s, cluster_type=%s, num_layers=%s, "
            "layer_id=%s, batch_id=%s, batch_size=%s, num_tokens=%s",
            stage_id,
            cluster_type,
            1,
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
            attention_query_cache=_attention_query_cache,
            include_stage_owned=include_stage_owned,
            stage_num_layers=stage_num_layers,
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

        return base_execution_time
