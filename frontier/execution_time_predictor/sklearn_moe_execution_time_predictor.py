from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import AttentionOperatorRole
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_metric_name_by_role,
)
from frontier.entities import Batch, ExecutionTime
from frontier.entities.time_components import (
    CommunicationOperatorTimes,
    MoEOperatorTimes,
    MoETime,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.logger import init_logger
from frontier.model_architectures import ModelArchitectureProfile, ResidualAddPolicy
from frontier.moe_gating_runtime import (
    DEFAULT_MOE_GATING_RUNTIME_CONTEXT,
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


class SklearnMoEExecutionTimePredictor(SklearnExecutionTimePredictor):
    def _get_requested_moe_gating_routing_runtime_path(self) -> str:
        return resolve_moe_gating_routing_runtime_path(
            getattr(self, "_moe_routing_mode", "simulation")
        )

    def _get_dummy_execution_time(self, batch: Batch, pipeline_stage: int) -> ExecutionTime:
        """Return fixed dummy ExecutionTime object with MoE-aware fields."""
        base_time = self._dummy_execution_time
        architecture_profile = self._get_model_architecture_profile()
        share_expert_enabled = self._model_config.supports_share_expert()

        attn_tp_size = self._replica_config.attn_tensor_parallel_size
        moe_tp_size = self._replica_config.moe_tensor_parallel_size
        moe_ep_size = self._replica_config.moe_expert_parallel_size

        # COMM_SKIP: TP all-reduce not needed when tp_size <= 1 (no tensor sharding)
        attn_tp_allreduce_time = base_time if attn_tp_size > 1 else 0.0
        moe_tp_allreduce_time = base_time if moe_tp_size > 1 else 0.0
        # COMM_SKIP: EP all-to-all not needed when ep_size <= 1 (experts co-located)
        expert_parallel_comm_time = base_time if moe_ep_size > 1 else 0.0

        # COMM_SKIP: DP allreduce not needed when dp_size <= 1 (no cross-DP communication)
        dp_size = getattr(self._replica_config, "data_parallel_size", 1)
        cluster_type = getattr(self, "_cluster_type", None)
        dp_input_allreduce_time = 0.0
        dp_output_allreduce_time = 0.0
        if (
            self._model_config.is_moe
            and cluster_type in (ClusterType.PREFILL, ClusterType.MONOLITHIC)
            and dp_size > 1
        ):
            dp_input_allreduce_time = base_time
            dp_output_allreduce_time = base_time

        ffn_tp_allgather_time = 0.0
        share_expert_tp_allreduce_time = 0.0
        if architecture_profile.moe_tensor_parallel_allgather_op and moe_tp_size > 1:
            ffn_tp_allgather_time = base_time
            if (
                share_expert_enabled
                and architecture_profile.share_expert_tensor_parallel_allreduce_op
            ):
                share_expert_tp_allreduce_time = base_time

        add_time = base_time
        add_attn_residual_time = 0.0
        add_ffn_residual_time = 0.0
        if architecture_profile.residual_add_policy is ResidualAddPolicy.FFN_RESIDUAL_ONLY:
            add_attn_residual_time = 0.0
            add_ffn_residual_time = base_time
            add_time = 0.0

        share_expert_time = base_time if share_expert_enabled else 0.0
        pp_stage_boundary_handoff_time = (
            base_time
            if pipeline_stage < self._replica_config.num_pipeline_stages - 1
            else 0.0
        )

        return ExecutionTime(
            num_layers_per_pipeline_stage=self._num_layers_per_pipeline_stage,
            attention_rope_execution_time=base_time,
            attention_kv_cache_save_execution_time=base_time,
            attention_decode_execution_time=base_time,
            attention_prefill_execution_time=base_time,
            attention_layer_pre_proj_execution_time=base_time,
            attention_layer_post_proj_execution_time=base_time,
            attn_norm_time=base_time,
            mlp_norm_time=base_time,
            add_time=add_time,
            add_attn_residual_time=add_attn_residual_time,
            add_ffn_residual_time=add_ffn_residual_time,
            tensor_parallel_communication_time=attn_tp_allreduce_time,
            attn_tensor_parallel_allreduce_time=attn_tp_allreduce_time,
            moe_tensor_parallel_allreduce_time=moe_tp_allreduce_time,
            pipeline_parallel_communication_time=base_time,
            expert_parallel_communication_time=expert_parallel_comm_time,
            moe_gating_time=base_time,
            moe_shuffling_time=base_time,
            schedule_time=base_time,
            sampler_e2e_time=base_time,
            prepare_inputs_e2e_time=base_time,
            process_model_outputs_time=base_time,
            ray_comm_time=base_time,
            pp_stage_boundary_handoff_time=pp_stage_boundary_handoff_time,
            is_moe=True,
            mlp_layer_up_proj_execution_time=0.0,
            mlp_layer_down_proj_execution_time=0.0,
            mlp_layer_act_execution_time=0.0,
            moe_grouped_gemm_time=base_time,
            share_expert_up_proj_time=share_expert_time,
            share_expert_down_proj_time=share_expert_time,
            share_expert_act_time=share_expert_time,
            tensor_parallel_allgather_time=ffn_tp_allgather_time,
            share_expert_tensor_parallel_allreduce_time=share_expert_tp_allreduce_time,
            dp_input_allreduce_time=dp_input_allreduce_time,
            dp_output_allreduce_time=dp_output_allreduce_time,
            moe_operator_times=_build_moe_operator_times(
                mlp_norm_time=base_time,
                moe_gating_linear_time=base_time * 0.5,
                moe_gating_routing_topk_time=base_time * 0.5,
                moe_shuffling_time=base_time,
                moe_grouped_gemm_time=base_time,
                share_expert_up_proj_time=share_expert_time,
                share_expert_act_time=share_expert_time,
                share_expert_down_proj_time=share_expert_time,
                include_share_expert=share_expert_enabled,
            ),
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

        # Initialize routing mode before parent init so independent training paths
        # select the correct moe_gating_routing_topk profiling rows.
        self._moe_routing_mode = getattr(replica_config, "moe_routing_mode", "simulation")
        self._moe_routing_seed = getattr(replica_config, "moe_routing_seed", 42)
        if self._moe_routing_mode not in (
            "simulation",
            "uniform_legacy",
            "uniform_random",
        ):
            raise ValueError(
                f"Invalid moe_routing_mode: '{self._moe_routing_mode}'. "
                f"Must be 'simulation', 'uniform_legacy', or 'uniform_random'."
            )
        if (
            self._moe_routing_mode in ("simulation", "uniform_random")
            and self._moe_routing_seed < 0
        ):
            raise ValueError(
                "moe_routing_seed must be non-negative when "
                "moe_routing_mode is 'simulation' or 'uniform_random', "
                f"got {self._moe_routing_seed}."
            )
        self._moe_gating_routing_runtime_path = (
            resolve_moe_gating_routing_runtime_path(self._moe_routing_mode)
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

        # Pre-compute routing details for simulation mode
        # Structure: {layer_id: {expert_id: allocation_ratio}}
        # This is computed once at init and reused for all batches.
        self._routing_allocations: Optional[Dict[int, Dict[int, float]]] = None
        self._global_routing_allocations: Optional[Dict[int, Dict[int, float]]] = None
        if self._moe_routing_mode == "simulation":
            self._routing_allocations = self._init_routing_allocations()
            self._global_routing_allocations = self._init_global_routing_allocations()
            logger.info(
                f"[MoE Routing] Initialized routing allocations for simulation mode: "
                f"seed={self._moe_routing_seed}, num_layers={len(self._routing_allocations)}"
            )

        self._share_expert_tp_allreduce_visibility_scale = float(
            getattr(
                self._config,
                "share_expert_tp_allreduce_visibility_scale",
                2.0 / 3.0,
            )
        )
        if self._share_expert_tp_allreduce_visibility_scale <= 0.0:
            raise ValueError(
                "share_expert_tp_allreduce_visibility_scale must be > 0, "
                f"got={self._share_expert_tp_allreduce_visibility_scale}"
            )

    def _init_routing_allocations(self) -> Dict[int, Dict[int, float]]:
        """
        Pre-compute expert allocation ratios for all layers.

        Uses deterministic seed to ensure reproducible results across runs.
        The allocations are computed once at init time and cached.

        Returns:
            Dictionary mapping layer_id -> {expert_id: allocation_ratio}
            Allocation ratios sum to 1.0 across all experts for each layer.
        """
        num_layers = self._model_config.num_layers
        total_experts = self._replica_config.total_expert_num

        # Get number of experts per device (after EP partitioning)
        if self._moe_ep_size > 0:
            num_experts_per_device = total_experts // self._moe_ep_size
        else:
            num_experts_per_device = total_experts

        if num_experts_per_device <= 0:
            logger.warning(
                f"num_experts_per_device={num_experts_per_device}, cannot init routing."
            )
            return {}

        allocations: Dict[int, Dict[int, float]] = {}
        for layer_id in range(num_layers):
            # Use deterministic seed derived from config seed + layer_id
            layer_seed = self._moe_routing_seed + layer_id
            np.random.seed(layer_seed)

            # Generate random weights and normalize to get allocation ratios
            # This simulates realistic load imbalance across experts
            random_weights = np.random.uniform(0.1, 1.0, num_experts_per_device)
            total_weight = np.sum(random_weights)
            expert_ratios = random_weights / total_weight

            allocations[layer_id] = {
                expert_id: float(expert_ratios[expert_id])
                for expert_id in range(num_experts_per_device)
            }

        return allocations

    def _init_global_routing_allocations(self) -> Dict[int, Dict[int, float]]:
        """Pre-compute global expert allocation ratios for shared-domain EP sync.

        Monolithic decode with EP enabled needs a global view across all experts to
        derive per-lane post-MoE arrival skew before the shared-domain all-reduce.
        """
        num_layers = self._model_config.num_layers
        total_experts = self._replica_config.total_expert_num

        if total_experts <= 0:
            logger.warning(
                "total_experts=%s, cannot initialize global routing allocations",
                total_experts,
            )
            return {}

        allocations: Dict[int, Dict[int, float]] = {}
        for layer_id in range(num_layers):
            layer_seed = self._moe_routing_seed + layer_id
            np.random.seed(layer_seed)
            random_weights = np.random.uniform(0.1, 1.0, total_experts)
            total_weight = np.sum(random_weights)
            expert_ratios = random_weights / total_weight
            allocations[layer_id] = {
                expert_id: float(expert_ratios[expert_id])
                for expert_id in range(total_experts)
            }

        return allocations

    def _get_global_per_expert_tokens(
        self,
        total_routed_tokens: int,
        layer_id: int,
    ) -> Dict[int, int]:
        """Discretize routed tokens across global experts for the given layer."""
        if total_routed_tokens <= 0:
            return {}

        total_experts = int(self._replica_config.total_expert_num)
        if total_experts <= 0:
            raise ValueError(f"Invalid total_expert_num={total_experts}")

        if self._moe_routing_mode == "uniform_legacy":
            per_expert = total_routed_tokens // total_experts
            remainder = total_routed_tokens % total_experts
            return {
                expert_id: per_expert + (1 if expert_id < remainder else 0)
                for expert_id in range(total_experts)
            }
        if self._moe_routing_mode == "uniform_random":
            return self._build_uniform_random_per_expert_tokens(
                total_routed_tokens=total_routed_tokens,
                num_experts=total_experts,
                layer_id=layer_id,
            )

        if self._global_routing_allocations is None:
            raise ValueError(
                "Global routing allocations not initialized for simulation mode. "
                "Ensure _init_global_routing_allocations() was called in __init__."
            )

        effective_layer_id = (
            layer_id if layer_id in self._global_routing_allocations else 0
        )
        if effective_layer_id not in self._global_routing_allocations:
            raise ValueError(
                f"No global routing allocations for layer {layer_id} or fallback layer 0. "
                f"Available layers: {list(self._global_routing_allocations.keys())}"
            )

        allocation_ratios = self._global_routing_allocations[effective_layer_id]
        return self._build_proportional_per_expert_tokens(
            total_routed_tokens=total_routed_tokens,
            allocation_ratios=allocation_ratios,
        )

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
        lane_count = int(getattr(self, "_moe_ep_size", 1))

        if self._enable_dummy_mode:
            # In dummy mode, this monolithic decode shared-domain helper should never
            # depend on profiling-backed prediction caches.
            # Components mirrored from the non-dummy path:
            # - post_attention_layernorm
            # - moe_gating_linear
            # - moe_gating_routing_topk
            # - moe_shuffling
            # - moe_grouped_gemm
            op_count = 5
            if self._model_config.supports_share_expert():
                op_count += 3  # share_expert_up_proj/act/down_proj

            lane_dummy_time = float(self._dummy_execution_time) * float(op_count)
            if lane_count <= 1:
                return {0: lane_dummy_time}
            return {lane_id: lane_dummy_time for lane_id in range(lane_count)}

        if lane_count <= 1:
            moe_tokens_input = self._get_moe_tokens_input(batch, layer_id=layer_id)
            shuffling_time = self._get_moe_shuffling_time(
                batch,
                moe_tokens_input=moe_tokens_input,
            )
            grouped_gemm_time = self._get_grouped_gemm_time(
                moe_tokens_input,
                batch=batch,
            )
            return {
                0: (
                    self._get_mlp_norm_layer_act_execution_time(batch)
                    + self._get_gating_linear_time(batch)
                    + self._get_gating_routing_topk_time(batch)
                    + shuffling_time
                    + grouped_gemm_time
                )
            }

        total_experts = int(self._replica_config.total_expert_num)
        if total_experts <= 0 or total_experts % lane_count != 0:
            raise ValueError(
                "Monolithic decode shared-domain sync requires total_expert_num to be "
                f"positive and divisible by moe_ep_size; got total_expert_num={total_experts}, "
                f"moe_ep_size={lane_count}"
            )

        total_routed_tokens = int(
            self._get_effective_moe_total_tokens(batch) * self._router_topk
        )
        if total_routed_tokens <= 0:
            return {lane_id: 0.0 for lane_id in range(lane_count)}

        experts_per_lane = total_experts // lane_count
        global_per_expert_tokens = self._get_global_per_expert_tokens(
            total_routed_tokens=total_routed_tokens,
            layer_id=layer_id,
        )

        lane_per_expert_tokens: Dict[int, Dict[int, int]] = {
            lane_id: {} for lane_id in range(lane_count)
        }
        for global_expert_id, token_count in global_per_expert_tokens.items():
            lane_id = min(lane_count - 1, global_expert_id // experts_per_lane)
            local_expert_id = global_expert_id % experts_per_lane
            lane_per_expert_tokens[lane_id][local_expert_id] = int(token_count)

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
        for lane_id in range(lane_count):
            per_expert_tokens = lane_per_expert_tokens[lane_id]
            shuffling_time = 0.0
            grouped_gemm_time = 0.0
            if per_expert_tokens:
                shuffling_time = self._get_moe_shuffling_time(
                    batch,
                    moe_tokens_input=per_expert_tokens,
                )
                grouped_gemm_time = self._get_grouped_gemm_time(
                    per_expert_tokens,
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
        base_df = moe_df[
            (moe_df["num_experts"] == model_config.num_experts)
            & (moe_df["router_topk"] == model_config.num_experts_per_tok)
            & (moe_df["hidden_dim"] == model_config.embedding_dim)
            & (moe_df["expert_hidden_dim"] == model_config.mlp_hidden_dim)
        ].copy()

        if len(base_df) == 0:
            raise ValueError(
                "MoE dataset contract validation failed: no rows match model configuration in "
                f"{moe_input_file}. Required: num_experts={model_config.num_experts}, "
                f"router_topk={model_config.num_experts_per_tok}, hidden_dim={model_config.embedding_dim}, "
                f"expert_hidden_dim={model_config.mlp_hidden_dim}."
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
        if should_enable_prefill_hot_moe_gating_contract(
            model_config=self._model_config,
        ):
            if has_prefill_hot_moe_gating_rows(model_filtered_df):
                model_names.extend(_get_prefill_hot_moe_gating_model_names())
            else:
                logger.warning(
                    "Prefill-hot gating contract is enabled for model=%s, but "
                    "dataset %s has no usable prefill_hot rows; skipping "
                    "__prefill_hot pseudo-model training.",
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
                gating_context_key = DEFAULT_MOE_GATING_RUNTIME_CONTEXT
                if model_name.endswith("__prefill_hot"):
                    gating_context_key = PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT
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
                if model_name.endswith("__prefill_hot"):
                    logger.warning(
                        "Skipping %s because prefill-hot gating rows are unavailable "
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
        if should_enable_prefill_hot_moe_gating_contract(
            model_config=self._model_config,
        ):
            include_prefill_hot_models = False
            try:
                moe_df = pd.read_csv(moe_input_file)
                include_prefill_hot_models = has_prefill_hot_moe_gating_rows(moe_df)
            except Exception:
                include_prefill_hot_models = False
            if include_prefill_hot_models:
                model_names.extend(_get_prefill_hot_moe_gating_model_names())
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
        extra_model_names = _get_prefill_hot_moe_gating_model_names()
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
        if should_use_prefill_hot_moe_gating_context(
            model_config=self._model_config,
            batch=batch,
        ):
            requested_context = PREFILL_HOT_MOE_GATING_RUNTIME_CONTEXT
        candidate_model_name = get_moe_gating_prediction_model_name(
            base_model_name,
            requested_context=requested_context,
        )
        if candidate_model_name in self._predictions:
            return candidate_model_name
        return base_model_name

    def _use_expert_parallel_alltoall_path(self, batch: Batch) -> bool:
        from frontier.entities import EPBatchGroup

        moe_ep_size = int(getattr(self, "_moe_ep_size", 1))
        if moe_ep_size <= 1:
            return False
        if isinstance(batch, EPBatchGroup):
            return True
        if self._cluster_type == ClusterType.DECODE_FFN:
            return True
        replica_config = getattr(self, "_replica_config", None)
        return int(getattr(replica_config, "attn_data_parallel_size", 1)) > 1

    def _get_effective_moe_total_tokens(self, batch: Batch) -> int:
        effective_tokens = int(
            batch.get_effective_total_tokens_rounded(self._cluster_type)
        )
        if effective_tokens < 0:
            raise ValueError(
                f"effective MoE tokens must be non-negative, got {effective_tokens}"
            )
        return effective_tokens

    def _get_local_ep_routed_tokens(self, batch: Batch) -> int:
        total_routed_tokens = int(
            self._get_effective_moe_total_tokens(batch) * self._router_topk
        )
        if total_routed_tokens <= 0:
            return 0

        from frontier.entities import EPBatchGroup

        if isinstance(batch, EPBatchGroup):
            per_expert_tokens = getattr(batch, "per_expert_tokens", None)
            if per_expert_tokens:
                return int(
                    sum(int(token_count) for token_count in per_expert_tokens.values())
                )
            return int(batch.total_num_tokens)

        moe_ep_size = int(getattr(self, "_moe_ep_size", 1))
        if moe_ep_size <= 1:
            return total_routed_tokens
        if not bool(getattr(batch, "is_pure_decode_batch", False)):
            return total_routed_tokens
        if self._use_expert_parallel_alltoall_path(batch):
            return total_routed_tokens
        return max(1, (total_routed_tokens + moe_ep_size - 1) // moe_ep_size)

    def _get_moe_tokens_input(
        self, batch: Batch, layer_id: int = 0
    ) -> Union[int, Dict[int, int]]:
        """
        Unified entry point to get MoE tokens input for grouped GEMM prediction.

        This method supports three comparison-time routing modes:
        - 'uniform_legacy': evenly split routed tokens across experts
        - 'uniform_random': deterministically sample experts uniformly at random
        - 'simulation': use pre-computed proportional allocations

        Args:
            batch: The batch being processed
            layer_id: The layer ID for which to get token allocation (default 0)

        Returns:
            - In 'uniform_legacy' mode: int (post_routing_batch_tokens = num_tokens * topk)
            - In 'uniform_random' mode with on-demand model: Dict[int, int] mapping expert_id to token count
            - In 'simulation' mode with on-demand model: Dict[int, int] mapping expert_id to token count
            - In 'uniform_legacy' mode without on-demand model: int (post_routing_batch_tokens)

        Raises:
            ValueError: If the selected routing mode is not supported by the active predictor
        """
        num_tokens = self._get_effective_moe_total_tokens(batch)
        local_routed_tokens = self._get_local_ep_routed_tokens(batch)

        if self._moe_routing_mode == "uniform_legacy":
            post_routing_batch_tokens = num_tokens * self._router_topk

            if self._is_grouped_gemm_on_demand_mode():
                return self._build_uniform_per_expert_tokens(local_routed_tokens)

            return post_routing_batch_tokens

        if self._moe_routing_mode == "uniform_random":
            if not self._is_grouped_gemm_on_demand_mode():
                raise ValueError(
                    "moe_routing_mode='uniform_random' requires moe_grouped_gemm model trained with "
                    "load-imbalance features (14 features). The current model appears to be trained "
                    "with legacy 1D features only."
                )
            return self._build_uniform_random_per_expert_tokens(
                total_routed_tokens=local_routed_tokens,
                num_experts=self._get_num_experts_per_device(),
                layer_id=layer_id,
            )

        if not self._is_grouped_gemm_on_demand_mode():
            raise ValueError(
                "moe_routing_mode='simulation' requires moe_grouped_gemm model trained with "
                "load-imbalance features (14 features). The current model appears to be trained "
                "with legacy 1D features only. Either:\n"
                "  1. Use a model trained with load-imbalance features, or\n"
                "  2. Set moe_routing_mode='uniform_legacy' to use legacy 1D mode."
            )

        if self._routing_allocations is None:
            raise ValueError(
                "Routing allocations not initialized. "
                "Ensure _init_routing_allocations() was called in __init__."
            )

        effective_layer_id = layer_id if layer_id in self._routing_allocations else 0
        if effective_layer_id not in self._routing_allocations:
            raise ValueError(
                f"No routing allocations for layer {layer_id} or fallback layer 0. "
                f"Available layers: {list(self._routing_allocations.keys())}"
            )

        allocation_ratios = self._routing_allocations[effective_layer_id]
        per_expert_tokens = self._build_proportional_per_expert_tokens(
            total_routed_tokens=local_routed_tokens,
            allocation_ratios=allocation_ratios,
        )

        actual_total = sum(per_expert_tokens.values())
        if actual_total != local_routed_tokens:
            raise ValueError(
                "MoE routing token conservation failed after largest-remainder discretization: "
                f"expected={local_routed_tokens}, got={actual_total}"
            )

        logger.debug(
            f"[_get_moe_tokens_input] layer={effective_layer_id}, num_tokens={num_tokens}, "
            f"local_routed_tokens={local_routed_tokens}, topk={self._router_topk}, "
            f"experts={len(per_expert_tokens)}"
        )

        return per_expert_tokens

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
        return self._predictions[model_name][(effective_tokens,)]

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
        return self._predictions[model_name][(effective_tokens,)]

    def _get_num_experts_per_device(self) -> int:
        total_experts = int(self._replica_config.total_expert_num)
        if total_experts <= 0:
            raise ValueError(f"Invalid total_expert_num={total_experts}")
        if self._moe_ep_size <= 0:
            return total_experts
        if total_experts % self._moe_ep_size != 0:
            raise ValueError(
                "total_expert_num must be divisible by moe_expert_parallel_size. "
                f"total_expert_num={total_experts}, moe_ep_size={self._moe_ep_size}"
            )
        return total_experts // self._moe_ep_size

    def _build_uniform_per_expert_tokens(
        self,
        total_routed_tokens: int,
    ) -> Dict[int, int]:
        if total_routed_tokens < 0:
            raise ValueError(
                f"total_routed_tokens must be non-negative, got {total_routed_tokens}"
            )
        if total_routed_tokens == 0:
            return {}

        num_experts_per_device = self._get_num_experts_per_device()
        per_expert = total_routed_tokens // num_experts_per_device
        remainder = total_routed_tokens % num_experts_per_device

        per_expert_tokens: Dict[int, int] = {}
        for expert_id in range(num_experts_per_device):
            token_count = per_expert + (1 if expert_id < remainder else 0)
            per_expert_tokens[expert_id] = token_count
        return per_expert_tokens

    def _build_uniform_random_per_expert_tokens(
        self,
        total_routed_tokens: int,
        num_experts: int,
        layer_id: int,
    ) -> Dict[int, int]:
        if total_routed_tokens < 0:
            raise ValueError(
                f"total_routed_tokens must be non-negative, got {total_routed_tokens}"
            )
        if total_routed_tokens == 0:
            return {}
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        rng = np.random.default_rng(int(self._moe_routing_seed) + int(layer_id))
        sampled_expert_ids = rng.integers(
            low=0,
            high=num_experts,
            size=total_routed_tokens,
        )
        expert_counts = np.bincount(sampled_expert_ids, minlength=num_experts)
        return {
            expert_id: int(expert_counts[expert_id])
            for expert_id in range(num_experts)
        }


    def _build_proportional_per_expert_tokens(
        self,
        total_routed_tokens: int,
        allocation_ratios: Dict[int, float],
    ) -> Dict[int, int]:
        """Discretize routing ratios with largest-remainder token conservation."""
        if total_routed_tokens < 0:
            raise ValueError(
                f"total_routed_tokens must be non-negative, got {total_routed_tokens}"
            )
        if total_routed_tokens == 0:
            return {}
        if len(allocation_ratios) == 0:
            raise ValueError("allocation_ratios must be non-empty")

        ratio_sum = float(sum(float(ratio) for ratio in allocation_ratios.values()))
        if ratio_sum <= 0.0:
            raise ValueError(
                f"allocation_ratios must sum to a positive value, got {ratio_sum}"
            )

        expert_base_allocation: Dict[int, int] = {}
        expert_fractional_parts: Dict[int, float] = {}
        normalized_ratios: Dict[int, float] = {}
        total_base_allocated = 0

        for expert_id in sorted(allocation_ratios):
            normalized_ratio = float(allocation_ratios[expert_id]) / ratio_sum
            exact_allocation = total_routed_tokens * normalized_ratio
            base_allocation = int(exact_allocation)
            fractional_part = exact_allocation - base_allocation

            expert_base_allocation[expert_id] = base_allocation
            expert_fractional_parts[expert_id] = fractional_part
            normalized_ratios[expert_id] = normalized_ratio
            total_base_allocated += base_allocation

        remaining_tokens = total_routed_tokens - total_base_allocated
        if remaining_tokens > 0:
            sorted_experts = sorted(
                expert_fractional_parts.keys(),
                key=lambda expert_id: (
                    -expert_fractional_parts[expert_id],
                    -normalized_ratios[expert_id],
                    expert_id,
                ),
            )
            for index in range(remaining_tokens):
                expert_id = sorted_experts[index % len(sorted_experts)]
                expert_base_allocation[expert_id] += 1

        total_allocated = sum(expert_base_allocation.values())
        if total_allocated != total_routed_tokens:
            raise ValueError(
                "Largest-remainder MoE routing allocation failed token conservation: "
                f"allocated={total_allocated}, expected={total_routed_tokens}"
            )

        return expert_base_allocation

    def _resolve_shuffling_per_expert_tokens(
        self,
        batch: Batch,
        moe_tokens_input: Optional[Union[int, Dict[int, int]]] = None,
    ) -> Dict[int, int]:
        if isinstance(moe_tokens_input, dict):
            normalized = {
                int(expert_id): int(token_count)
                for expert_id, token_count in moe_tokens_input.items()
            }
            if any(token_count < 0 for token_count in normalized.values()):
                raise ValueError(
                    f"Negative token count in moe_tokens_input={moe_tokens_input}"
                )
            return normalized

        if isinstance(moe_tokens_input, int):
            total_routed_tokens = moe_tokens_input
        else:
            total_routed_tokens = int(
                self._get_effective_moe_total_tokens(batch) * self._router_topk
            )

        return self._build_uniform_per_expert_tokens(total_routed_tokens)

    def _build_moe_load_imbalance_features(
        self,
        per_expert_tokens: Dict[int, int],
    ) -> Dict[str, float]:
        if len(per_expert_tokens) == 0:
            raise ValueError(
                "per_expert_tokens must be non-empty for load-imbalance feature construction"
            )

        from frontier.profiling.moe.moe_input import MoELoadImbalanceInput

        expert_token_counts = [int(v) for v in per_expert_tokens.values()]
        if any(v < 0 for v in expert_token_counts):
            raise ValueError(
                f"Negative token count in per_expert_tokens={per_expert_tokens}"
            )

        total_routed_tokens = int(sum(expert_token_counts))
        if self._router_topk <= 0:
            raise ValueError(f"Invalid router_topk={self._router_topk}")

        approx_num_tokens = max(
            1, int(round(total_routed_tokens / float(self._router_topk)))
        )

        load_input = MoELoadImbalanceInput(
            num_tokens=approx_num_tokens,
            num_experts_per_device=len(expert_token_counts),
            hidden_dim=int(self._model_config.embedding_dim),
            expert_hidden_dim=int(self._model_config.mlp_hidden_dim),
            router_topk=int(self._router_topk),
            expert_token_counts=expert_token_counts,
            load_distribution="runtime",
        )
        features = load_input.to_features_dict()
        features.pop("load_distribution", None)
        return features

    def _get_moe_compute_calibration_scale(
        self,
        batch: Optional[Batch],
        decode_phase_attr_name: str,
        decode_phase_field_name: str,
        global_attr_name: str,
        global_field_name: str,
    ) -> float:
        if batch is None:
            return self._get_calibration_scale(global_attr_name, global_field_name)

        decode_phase_scale = self._get_decode_phase_only_calibration_scale(
            batch,
            decode_phase_attr_name,
            decode_phase_field_name,
        )
        if decode_phase_scale is not None:
            scale = decode_phase_scale
        else:
            scale = self._get_calibration_scale(global_attr_name, global_field_name)

        request_length_scale = self._get_decode_request_length_calibration_scale(batch)
        if request_length_scale is not None:
            scale *= request_length_scale
        return scale

    def _get_moe_shuffling_time(
        self,
        batch: Batch,
        moe_tokens_input: Optional[Union[int, Dict[int, int]]] = None,
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

        prediction_cache = self._predictions["moe_shuffling"]
        if isinstance(prediction_cache, dict) and prediction_cache.get(
            "_on_demand_prediction", False
        ):
            per_expert_tokens = self._resolve_shuffling_per_expert_tokens(
                batch,
                moe_tokens_input=moe_tokens_input,
            )
            if len(per_expert_tokens) == 0:
                raw_time = 0.0
            else:
                runtime_cache = getattr(
                    self, "_runtime_moe_shuffling_on_demand_prediction_cache", None
                )
                if runtime_cache is None:
                    runtime_cache = {}
                    self._runtime_moe_shuffling_on_demand_prediction_cache = (
                        runtime_cache
                    )
                family_name = self._measurement_family_name(
                    self._active_measurement_type
                )
                # The on-demand MoE feature vector only contains aggregate load
                # statistics, so expert IDs and insertion order are not prediction
                # inputs. Canonicalize by token-count distribution to maximize
                # result-equivalent cache hits without changing predicted values.
                distribution_key = tuple(
                    sorted(
                        int(token_count)
                        for token_count in per_expert_tokens.values()
                    )
                )
                cache_key = (
                    family_name,
                    distribution_key,
                    int(self._model_config.embedding_dim),
                    int(self._model_config.mlp_hidden_dim),
                    int(self._router_topk),
                )
                raw_time = runtime_cache.get(cache_key)
                if raw_time is None:
                    features = self._build_moe_load_imbalance_features(
                        per_expert_tokens
                    )
                    raw_time = self._get_on_demand_prediction(
                        "moe_shuffling", features
                    )
                    runtime_cache[cache_key] = raw_time
        else:
            effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
            raw_time = self._predictions["moe_shuffling"][(effective_tokens,)]

        scale = self._get_moe_compute_calibration_scale(
            batch,
            "_decode_phase_moe_shuffling_calibration_scale",
            "decode_phase_moe_shuffling_calibration_scale",
            "_moe_shuffling_calibration_scale",
            "moe_shuffling_calibration_scale",
        )
        return raw_time * scale

    def _apply_share_expert_tp_allreduce_overlap(self, raw_time_ms: float) -> float:
        """Apply profile-declared overlap scaling for share_expert TP allreduce.

        vLLM records ``share_expert_tp_allreduce`` around an NCCL call that can overlap
        with subsequent MoE kernels on separate streams. Architecture profiles opt in
        by declaring the calibrated visibility scale.
        """
        if raw_time_ms <= 0.0:
            return 0.0

        model_config = getattr(self, "_model_config", None)
        if model_config is None:
            replica_config = getattr(self, "_replica_config", None)
            model_config = getattr(replica_config, "model_config", None)

        if model_config is None:
            return raw_time_ms
        architecture_getter = getattr(model_config, "get_model_architecture_profile", None)
        if not callable(architecture_getter):
            raise TypeError(
                "MoE share-expert TP allreduce overlap requires "
                "model_config.get_model_architecture_profile()"
            )
        architecture_profile = architecture_getter()
        if not isinstance(architecture_profile, ModelArchitectureProfile):
            raise TypeError(
                "model_config.get_model_architecture_profile() must return "
                "ModelArchitectureProfile"
            )
        overlap_visibility_scale = (
            architecture_profile.share_expert_tp_allreduce_visibility_scale
        )
        if overlap_visibility_scale is None:
            return raw_time_ms
        configured_visibility_scale = getattr(
            self,
            "_share_expert_tp_allreduce_visibility_scale",
            None,
        )
        if configured_visibility_scale is not None:
            overlap_visibility_scale = float(configured_visibility_scale)
        return raw_time_ms * float(overlap_visibility_scale)


    def _apply_moe_grouped_gemm_decode_visibility(
        self,
        raw_time_ms: float,
        batch: Batch,
    ) -> float:
        """Return raw grouped-GEMM time for runtime-only CUDA Graph modeling."""
        if raw_time_ms <= 0.0:
            return 0.0
        return raw_time_ms


    def _get_moe_tensor_parallel_allreduce_time(self, batch: Batch) -> float:
        """
        Get MoE/FFN tensor-parallel all-reduce time using moe_tensor_parallel_size.

        This is separate from attention TP all-reduce and is required when
        attn_tp != moe_tp in unified clusters (MONOLITHIC/DECODE/PREFILL).
        """
        moe_tp_size = self._replica_config.moe_tensor_parallel_size
        # COMM_SKIP: TP all-reduce not needed when moe_tp_size <= 1 (no tensor sharding)
        if moe_tp_size <= 1:
            return 0.0

        if self._cc_backend is not None:
            effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
            data_size_bytes = self._model_config.embedding_dim * 2 * effective_tokens
            quant_manager = get_quantization_manager()
            data_size_bytes = quant_manager.adjust_tensor_size(
                "allreduce", data_size_bytes, self._cluster_type
            )
            result = self._cc_backend.predict_allreduce(
                data_size_bytes=data_size_bytes,
                num_devices=moe_tp_size,
                cluster_type=self._cluster_type,
                comm_domain="MOE_TP",
            )
            result = self._strip_collective_sim_allreduce_launch_overhead_if_needed(
                batch=batch,
                predicted_ms=result,
                num_devices=moe_tp_size,
                comm_domain="MOE_TP",
            )
            logger.debug(
                f"_get_moe_tensor_parallel_allreduce_time: using CC Backend, "
                f"data_size={data_size_bytes}, num_devices={moe_tp_size}, result={result:.6f} ms"
            )
            return result

        if self._enable_dummy_mode:
            logger.debug(
                f"_get_moe_tensor_parallel_allreduce_time: CC Backend not available, "
                f"using dummy mode value={self._dummy_execution_time} ms"
            )
            return self._dummy_execution_time

        raise RuntimeError(
            f"CC Backend is required for MoE tensor-parallel allreduce prediction "
            f"but was not provided. Either:\n"
            f"  1. Configure a CC Backend (e.g., --cc_backend vidur or --cc_backend analytical)\n"
            f"  2. Enable dummy mode explicitly (--enable_dummy_mode)\n"
            f"Current state: cc_backend=None, enable_dummy_mode={self._enable_dummy_mode}"
        )

    def _get_expert_parallel_communication_calibration_scale(self, batch: Batch) -> float:
        late_decode_scale = self._get_late_decode_only_calibration_scale(
            batch,
            "_late_decode_expert_parallel_communication_calibration_scale",
            "late_decode_expert_parallel_communication_calibration_scale",
        )
        if late_decode_scale is not None:
            return late_decode_scale

        decode_phase_scale = self._get_decode_phase_only_calibration_scale(
            batch,
            "_decode_phase_expert_parallel_communication_calibration_scale",
            "decode_phase_expert_parallel_communication_calibration_scale",
        )
        if decode_phase_scale is not None:
            return decode_phase_scale
        return self._get_calibration_scale(
            "_expert_parallel_communication_calibration_scale",
            "expert_parallel_communication_calibration_scale",
        )

    def _get_expert_parallel_communication_time(self, batch: Batch) -> float:
        """
        Get expert parallel communication time.

        Shared-domain MoE execution (monolithic / prefill / decode) uses
        expert-parallel all-reduce when EP is enabled without all-to-all routing.
        Post-routing EP batches (e.g. DECODE_FFN) and flattened multi-DP MoE
        paths keep the all-to-all communication model.
        """
        if self._moe_ep_size <= 1:
            return 0.0

        if self._cc_backend is not None:
            effective_tokens = batch.get_effective_total_tokens_rounded(self._cluster_type)
            quant_manager = get_quantization_manager()

            if self._use_expert_parallel_alltoall_path(batch):
                from frontier.entities import EPBatchGroup

                if isinstance(batch, EPBatchGroup) and getattr(batch, "per_expert_tokens", None):
                    routed_tokens = int(
                        sum(int(token_count) for token_count in batch.per_expert_tokens.values())
                    )
                else:
                    routed_tokens = int(effective_tokens * self._router_topk)
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
                return result * self._get_expert_parallel_communication_calibration_scale(
                    batch
                )

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
            return result * self._get_expert_parallel_communication_calibration_scale(
                batch
            )

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

    def _round_to_valid_key(self, num_tokens: int) -> int:
        """
        Round num_tokens to the nearest valid key in prediction cache.

        This handles cases where the exact token count is not in the cache.
        """
        if num_tokens <= 0:
            return 1  # Minimum valid token count
        if num_tokens > self._max_tokens:
            return self._max_tokens
        return num_tokens

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
            num_tokens_or_allocation: Either an integer (num_tokens) for backward compatibility,
                                    or a dict {expert_id: num_tokens} for detailed allocation

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

        # Check if this model uses on-demand prediction (trained with load imbalance features)
        if isinstance(prediction_cache, dict) and prediction_cache.get(
            "_on_demand_prediction"
        ):
            # On-demand prediction mode: model was trained with load imbalance features.
            # We must provide the full feature set computed from per-expert token distribution.
            if not isinstance(num_tokens_or_allocation, dict):
                raise ValueError(
                    "moe_grouped_gemm is in load-imbalance (on-demand) mode, but per-expert token allocation "
                    f"was not provided (got type={type(num_tokens_or_allocation).__name__})."
                )

            per_expert_tokens: Dict[int, int] = num_tokens_or_allocation
            if len(per_expert_tokens) == 0:
                return 0.0

            expert_token_counts = [int(v) for v in per_expert_tokens.values()]
            if any(v < 0 for v in expert_token_counts):
                raise ValueError(
                    f"Negative token count in per_expert_tokens: {per_expert_tokens}"
                )

            total_routed_tokens = int(sum(expert_token_counts))
            if total_routed_tokens == 0:
                return 0.0

            # The grouped_gemm predictor expects pre-routing num_tokens metadata, while runtime
            # allocation is post-routing (already expanded by router_topk). Recover the
            # approximate pre-routing token count to keep feature semantics aligned with
            # the profiling dataset contract.
            if self._router_topk <= 0:
                raise ValueError(f"Invalid router_topk={self._router_topk}")
            approx_num_tokens = max(
                1, int(round(total_routed_tokens / float(self._router_topk)))
            )

            runtime_cache = getattr(
                self, "_runtime_grouped_gemm_on_demand_prediction_cache", None
            )
            if runtime_cache is None:
                runtime_cache = {}
                self._runtime_grouped_gemm_on_demand_prediction_cache = runtime_cache
            family_name = self._measurement_family_name(
                self._active_measurement_type
            )
            # The load-imbalance predictor consumes aggregate distribution
            # statistics, not expert identity/order. Sort token counts so
            # equivalent allocations share one raw-prediction cache entry.
            distribution_key = tuple(sorted(expert_token_counts))
            cache_key = (
                family_name,
                distribution_key,
                approx_num_tokens,
                int(self._model_config.embedding_dim),
                int(self._model_config.mlp_hidden_dim),
                int(self._router_topk),
            )
            raw_time = runtime_cache.get(cache_key)
            if raw_time is None:
                features = self._build_moe_load_imbalance_features(per_expert_tokens)

                # Use the shared on-demand prediction path (with runtime caching and strict feature checks).
                raw_time = self._get_on_demand_prediction("moe_grouped_gemm", features)
                runtime_cache[cache_key] = raw_time
            scale = self._get_moe_compute_calibration_scale(
                batch,
                "_decode_phase_moe_grouped_gemm_calibration_scale",
                "decode_phase_moe_grouped_gemm_calibration_scale",
                "_moe_grouped_gemm_calibration_scale",
                "moe_grouped_gemm_calibration_scale",
            )
            return raw_time * scale

        def _get_cached_grouped_gemm_prediction(rounded_tokens: int) -> float:
            cache_key = (rounded_tokens,)
            if cache_key not in prediction_cache:
                raise KeyError(
                    "Missing moe_grouped_gemm cached prediction for "
                    f"tokens={rounded_tokens}. This indicates a profiling coverage gap. "
                    "Please regenerate profiling data or enable on-demand prediction."
                )
            value = float(prediction_cache[cache_key])
            if value < 0:
                raise ValueError(
                    f"Invalid moe_grouped_gemm cached prediction {value} for tokens={rounded_tokens}"
                )
            return value

        # Standard cache lookup mode (trained with num_tokens only)
        if isinstance(num_tokens_or_allocation, dict):
            # Cached grouped-gemm profiling rows represent the full fused grouped-GEMM
            # iteration for one MoE layer, keyed by pre-routing num_tokens. They are not
            # per-expert unit costs, so allocation input must be collapsed back to the
            # corresponding pre-routing token count instead of summing per-expert lookups.
            total_routed_tokens = int(sum(num_tokens_or_allocation.values()))
            if total_routed_tokens == 0:
                return 0.0
            if self._router_topk <= 0:
                raise ValueError(f"Invalid router_topk={self._router_topk}")
            approx_num_tokens = max(
                1, int(round(total_routed_tokens / float(self._router_topk)))
            )
            rounded_tokens = self._round_to_valid_key(approx_num_tokens)
            raw_time = _get_cached_grouped_gemm_prediction(rounded_tokens)
            scale = self._get_moe_compute_calibration_scale(
                batch,
                "_decode_phase_moe_grouped_gemm_calibration_scale",
                "decode_phase_moe_grouped_gemm_calibration_scale",
                "_moe_grouped_gemm_calibration_scale",
                "moe_grouped_gemm_calibration_scale",
            )
            return raw_time * scale

        # Backward compatibility: single number of tokens
        num_tokens = num_tokens_or_allocation
        if num_tokens <= 0:
            return 0.0
        rounded_tokens = self._round_to_valid_key(num_tokens)
        raw_time = _get_cached_grouped_gemm_prediction(rounded_tokens)
        scale = self._get_moe_compute_calibration_scale(
            batch,
            "_decode_phase_moe_grouped_gemm_calibration_scale",
            "decode_phase_moe_grouped_gemm_calibration_scale",
            "_moe_grouped_gemm_calibration_scale",
            "moe_grouped_gemm_calibration_scale",
        )
        return raw_time * scale

    # This is now a private method used internally for MoE-specific logic
    def _get_execution_time_internal(
        self,
        batch: Batch,
        pipeline_stage: int,
        moe_tokens_input: "int | Dict[int, int] | None" = None,
        include_moe: bool = True,
    ) -> "ExecutionTime":
        """
        Calculate execution time for a pipeline stage.

        Args:
            batch: The batch being processed
            pipeline_stage: Pipeline stage index
            moe_tokens_input: Input for MoE grouped GEMM time calculation.
                - For standard mode (1 feature): int (post_routing_batch_tokens)
                - For on-demand mode (14 features): Dict[int, int] (per_expert_tokens)
                - None is only valid when include_moe=False
            include_moe: Whether to include MoE-specific calculations

        Returns:
            ExecutionTime with all component times

        Raises:
            ValueError: If include_moe=True but moe_tokens_input is None (fail-fast)
        """
        attention_time = self.predict_attention_layer_time(
            batch=batch,
            layer_id=0,
            cluster_type=self._cluster_type,
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
        if self._replica_config.attn_tensor_parallel_size == 1:
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
        if include_moe and self._replica_config.moe_tensor_parallel_size > 1:
            moe_tp_allreduce_time = self._predict_comm_operator(
                get_comm_operator("moe_tensor_parallel_allreduce"),
                batch,
            )
            communication_operator_times["moe_tensor_parallel_allreduce"] = (
                moe_tp_allreduce_time
            )
        elif self._replica_config.attn_tensor_parallel_size > 1:
            moe_tp_allreduce_time = attn_tp_allreduce_time
            communication_operator_times["mlp_tensor_parallel_allreduce"] = (
                moe_tp_allreduce_time
            )

        share_expert_up_proj_time = 0.0
        share_expert_down_proj_time = 0.0
        share_expert_act_time = 0.0
        if include_moe and self._model_config.supports_share_expert():
            share_expert_up_proj_time = self._get_share_expert_up_proj_execution_time(batch)
            share_expert_down_proj_time = self._get_share_expert_down_proj_execution_time(batch)
            share_expert_act_time = self._get_share_expert_act_execution_time(batch)

        mlp_up_proj_time = 0.0
        mlp_down_proj_time = 0.0
        mlp_act_time = 0.0

        if include_moe:
            expert_parallel_communication_time = 0.0
            if self._moe_ep_size > 1:
                expert_parallel_operator_name = (
                    "expert_parallel_alltoall"
                    if self._use_expert_parallel_alltoall_path(batch)
                    else "expert_parallel_allreduce"
                )
                expert_parallel_communication_time = (
                    self._predict_comm_operator(
                        get_comm_operator(expert_parallel_operator_name),
                        batch,
                    )
                    * self._get_expert_parallel_communication_calibration_scale(batch)
                )
                communication_operator_times[expert_parallel_operator_name] = (
                    expert_parallel_communication_time
                )
            moe_gating_linear_time = self._get_gating_linear_time(batch)
            moe_gating_routing_topk_time = self._get_gating_routing_topk_time(batch)
            moe_gating_time = moe_gating_linear_time + moe_gating_routing_topk_time
            moe_shuffling_time = self._get_moe_shuffling_time(
                batch,
                moe_tokens_input=moe_tokens_input,
            )

            # Fail-fast: moe_tokens_input is required when include_moe=True
            if moe_tokens_input is None:
                raise ValueError(
                    "moe_tokens_input is required when include_moe=True. "
                    "For standard mode, provide post_routing_batch_tokens (int). "
                    "For on-demand mode, provide per_expert_tokens (Dict[int, int])."
                )
            moe_grouped_gemm_time = self._get_grouped_gemm_time(
                moe_tokens_input,
                batch=batch,
            )
            moe_grouped_gemm_time = self._apply_moe_grouped_gemm_decode_visibility(
                moe_grouped_gemm_time,
                batch,
            )
        else:
            # Dense FFN branch for mixed-layer MoE models.
            expert_parallel_communication_time = 0.0
            moe_gating_time = 0.0
            moe_gating_linear_time = 0.0
            moe_gating_routing_topk_time = 0.0
            moe_shuffling_time = 0.0
            moe_grouped_gemm_time = 0.0
            mlp_up_proj_time = self._get_mlp_layer_up_proj_execution_time(batch)
            mlp_down_proj_time = self._get_mlp_layer_down_proj_execution_time(batch)
            mlp_act_time = self._get_mlp_layer_act_execution_time(batch)

        add_time = self._get_add_layer_act_execution_time(batch)
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
        if include_moe and moe_tp_allgather_op:
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
                    share_expert_tp_allreduce_time = self._apply_share_expert_tp_allreduce_overlap(
                        raw_share_expert_tp_allreduce_time
                    )
                    communication_operator_times[
                        share_expert_tp_allreduce_op
                    ] = share_expert_tp_allreduce_time

        dp_input_allreduce_time = 0.0
        dp_output_allreduce_time = 0.0
        if include_moe and self._cluster_type is not None:
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
                layer_id=pipeline_stage,
            ),
            "mtp_terminal_overshoot",
            batch,
            f"stage={pipeline_stage}",
        )

        mlp_norm_time = self._get_mlp_norm_layer_act_execution_time(batch)

        return ExecutionTime(
            num_layers_per_pipeline_stage=self._num_layers_per_pipeline_stage,
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
            is_moe=include_moe,
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
                if include_moe
                else None
            ),
        )

    def _simulate_routing_per_layer(
        self, batches: List[Batch], stage_id: int
    ) -> Dict[int, Dict[str, Dict[int, float]]]:
        """
        Simulate routing for each layer in the stage.
        Returns: {layer_id: {replica_id: {moe_component: time_value}}}
        """
        # Routing simulation is stage-local and follows the current pipeline-stage scope.
        num_layers = self._num_layers_per_pipeline_stage
        layer_routing_results = {}

        for layer_id in range(num_layers):
            # For each layer, simulate independent routing
            layer_routing_results[layer_id] = {}

            # Simulate routing for this specific layer
            post_routing_workloads = self._simulate_routing(batches)

            for replica_id, num_tokens_after_routing in post_routing_workloads.items():
                batch = next(b for b in batches if b.replica_id == replica_id)

                layer_routing_results[layer_id][replica_id] = {
                    "moe_grouped_gemm_time": self._get_grouped_gemm_time(
                        num_tokens_after_routing,
                        batch=batch,
                    ),
                    "expert_parallel_communication_time": self._get_expert_parallel_communication_time(
                        batch
                    ),
                    "moe_gating_time": self._get_gating_time(batch),
                    "moe_shuffling_time": self._get_moe_shuffling_time(
                        batch,
                        moe_tokens_input=num_tokens_after_routing,
                    ),
                }

        return layer_routing_results

    # Phase 2.5: Removed deprecated get_moe_stage_execution_details() method
    # MoE models now use predict_moe_layer_time() and other fine-grained APIs

    # ========================================================================
    # New unified API implementation (Phase 0) - MoE extensions
    # ========================================================================

    def predict_moe_layer_time(
        self,
        batch_or_group: "Batch | EPBatchGroup",
        layer_id: int,
        cluster_type: ClusterType,
        per_expert_tokens: Optional[Dict[int, int]] = None,
    ) -> MoETime:
        """
        Predict MoE execution time for a single transformer layer.

        Phase 3 Enhancement: Now accepts per_expert_tokens parameter for direct expert allocation.

        Args:
            batch_or_group: Batch or EPBatchGroup to predict for
            layer_id: Layer index (0-based)
            cluster_type: Type of cluster (PREFILL, DECODE_FFN, etc.)
            per_expert_tokens: Optional dict mapping expert_id -> token_count.
                              When provided (from EPBatchGroup), uses actual expert allocation.
                              When None, falls back to routing simulation.

        Returns:
            MoETime component with all MoE-related times

        Raises:
            ValueError: If token conservation is violated
            NotImplementedError: If MoE operations not supported for cluster type
        """
        if self._enable_dummy_mode:
            base_time = self._dummy_execution_time
            share_expert_time = (
                base_time if self._model_config.supports_share_expert() else 0.0
            )
            return MoETime(
                moe_grouped_gemm_time=base_time,
                moe_gating_linear_time=base_time * 0.5,
                moe_gating_routing_topk_time=base_time * 0.5,
                moe_shuffling_time=base_time,
                mlp_norm_time=base_time,
                share_expert_up_proj_time=share_expert_time,
                share_expert_down_proj_time=share_expert_time,
                share_expert_act_time=share_expert_time,
                operator_times=_build_moe_operator_times(
                    mlp_norm_time=base_time,
                    moe_gating_linear_time=base_time * 0.5,
                    moe_gating_routing_topk_time=base_time * 0.5,
                    moe_shuffling_time=base_time,
                    moe_grouped_gemm_time=base_time,
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

        # Determine batch to use for communication/gating predictions
        if hasattr(batch_or_group, "per_expert_tokens"):
            # EPBatchGroup case
            batch = batch_or_group  # EPBatchGroup provides the Batch-compatible fields used below
            # Phase 3: If per_expert_tokens not explicitly provided, extract from EPBatchGroup
            if per_expert_tokens is None:
                per_expert_tokens = batch_or_group.per_expert_tokens
                logger.debug(
                    f"Extracted per_expert_tokens from EPBatchGroup: {len(per_expert_tokens)} experts"
                )
        else:
            # Regular Batch case
            batch = batch_or_group

        # Phase 3: When per_expert_tokens is provided, use it directly (actual MoE routing data)
        if per_expert_tokens is not None:
            # Validate token conservation
            total_allocated_tokens = sum(per_expert_tokens.values())
            # EPBatchGroup.per_expert_tokens is already post-routing allocation and
            # Batch.total_num_tokens already includes router_topk expansion.
            is_ep_batch_group = hasattr(batch_or_group, "source_batch_ids") and hasattr(
                batch_or_group, "ep_id"
            )
            expected_tokens = (
                batch.total_num_tokens
                if is_ep_batch_group
                else self._get_effective_moe_total_tokens(batch) * self._router_topk
            )

            if (
                abs(total_allocated_tokens - expected_tokens) > 1
            ):  # Allow small rounding errors
                raise ValueError(
                    f"Token conservation violated in predict_moe_layer_time: "
                    f"allocated {total_allocated_tokens} tokens, "
                    f"expected {expected_tokens} (batch_tokens={batch.total_num_tokens}, "
                    f"effective_batch_tokens={self._get_effective_moe_total_tokens(batch)}, "
                    f"router_topk={self._router_topk})"
                )

            logger.debug(
                f"Using provided per_expert_tokens: {total_allocated_tokens} tokens allocated across {len(per_expert_tokens)} experts"
            )
            grouped_gemm_time = self._get_grouped_gemm_time(
                per_expert_tokens,
                batch=batch,
            )
        else:
            # Fail fast: per_expert_tokens must be provided by the caller.
            # For disaggregation mode, the caller (SklearnDisaggregationExecutionTimePredictor)
            # should use _calculate_expert_token_allocation() to compute per_expert_tokens
            # from pre-initialized routing_details.
            #
            # This ensures:
            # 1. Token distribution is pre-calculated during initialization (not runtime)
            # 2. Consistent behavior across all clusters using the same routing_details
            # 3. Clear separation of concerns: routing initialization vs. execution time prediction
            raise ValueError(
                f"per_expert_tokens must be provided for MoE layer time prediction. "
                f"For disaggregation mode, use _calculate_expert_token_allocation() to compute "
                f"per_expert_tokens from pre-initialized routing_details. "
                f"(layer_id={layer_id}, cluster_type={cluster_type.name}, "
                f"batch_id={batch.id if hasattr(batch, 'id') else 'N/A'})"
            )

        # Get individual MoE operation times (compute only, communication is separate)
        gating_linear_time = self._get_gating_linear_time(batch)
        gating_routing_topk_time = self._get_gating_routing_topk_time(batch)
        gating_time = gating_linear_time + gating_routing_topk_time
        shuffling_time = self._get_moe_shuffling_time(
            batch,
            moe_tokens_input=per_expert_tokens,
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

    def predict_stage_execution_time(
        self,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int = 1,
        layer_id: int = 0,
    ) -> ExecutionTime:
        """
        Predict execution time for MoE models using per-layer component semantics.

        Predictor components are represented as single-layer times (milliseconds), while
        ExecutionTime aggregates across ``num_layers_per_pipeline_stage``.
        Therefore, changing ``num_layers`` must update only the layer count, not rescale
        per-layer components.
        """
        if self._enable_dummy_mode:
            return self._get_dummy_execution_time(batch, stage_id)

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

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        measurement_type = self._select_measurement_type_for_batch(batch)
        self._require_predictions_for_measurement_type(measurement_type, batch)
        self._activate_measurement_type(measurement_type)

        # Validate cluster_type consistency
        if self._cluster_type is not None and cluster_type != self._cluster_type:
            logger.error(
                f"Cluster type mismatch: predictor initialized with {self._cluster_type}, "
                f"but predict_stage_execution_time called with {cluster_type}"
            )

        # Mixed-layer MoE models (e.g., Step3) have dense FFN layers where MoE
        # ops must be disabled. The runtime scheduler provides explicit layer_id
        # for single-layer prediction calls (num_layers == 1).
        include_moe = True
        if num_layers == 1:
            include_moe = self._model_config.is_moe_layer(layer_id)

        moe_tokens_input = None
        if include_moe:
            # Use unified _get_moe_tokens_input() for mode-aware input selection:
            # - 'uniform_legacy' mode: returns int (post_routing_batch_tokens)
            # - 'simulation' mode with on-demand: returns Dict[int, int] (per_expert_tokens)
            # - 'simulation' mode without on-demand: returns int (post_routing_batch_tokens)
            moe_tokens_input = self._get_moe_tokens_input(batch, layer_id=layer_id)

            if isinstance(moe_tokens_input, dict):
                logger.debug(
                    "[EXEC_TIME_PREDICT_MOE] Using simulation mode with per_expert_tokens: "
                    "%s experts, total_tokens=%s, layer_id=%s",
                    len(moe_tokens_input),
                    sum(moe_tokens_input.values()),
                    layer_id,
                )
            else:
                logger.debug(
                    "[EXEC_TIME_PREDICT_MOE] Using %s mode with post_routing_batch_tokens=%s, "
                    "layer_id=%s",
                    self._moe_routing_mode,
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
            include_moe=include_moe,
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

        # MOE OP-TRACE: log per-layer MoE op times
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
        )
