from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass
import numpy as np
from frontier.logger import init_logger
from frontier.model_architectures import (
    ModelArchitectureProfile,
    ResidualAddPolicy,
    get_model_architecture_profile,
)

from frontier.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
    ClusterConfig,
    get_quantization_manager,
)
from frontier.config import global_vars
from frontier.entities import Batch, EPBatchGroup, ExecutionTime
from frontier.entities.time_components import (
    AttentionTime,
    CommunicationOperatorTimes,
    OverheadTime,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.operators.families import get_comm_operator
from frontier.operators.spec import CommPayloadContext
from frontier.types import ClusterType
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.moe_ep_workload import EPLaneWorkload, resolve_ep_lane_workload

if TYPE_CHECKING:
    from frontier.cc_backend import BaseCCBackend

logger = init_logger(__name__)


@dataclass
class CommunicationTime:
    """Container for communication times."""

    tensor_parallel_time: float = 0.0
    pipeline_parallel_time: float = 0.0


class WorkloadDistributionType(Enum):
    """Enum for different workload distribution types."""

    BALANCED = "balanced"
    RANDOM = "random"
    SKEWED = "skewed"
    ZIPF = "zipf"


class SklearnDisaggregationExecutionTimePredictor(SklearnMoEExecutionTimePredictor):
    # The disaggregated ClusterConfig owns role availability through these
    # attributes. Keep this tuple as the single source of truth for routed
    # aggregate materialization.
    _ROUTED_ROLE_CONFIG_ATTRIBUTES = (
        (ClusterType.PREFILL, "prefill_replica_config"),
        (ClusterType.DECODE_FFN, "decode_ffn_replica_config"),
        (ClusterType.DECODE, "decode_replica_config"),
    )

    @staticmethod
    def _resolve_workload_distribution_type(
        distribution_type: str,
    ) -> WorkloadDistributionType:
        normalized_type = str(distribution_type).strip().lower()
        try:
            return WorkloadDistributionType(normalized_type)
        except ValueError as exc:
            valid_values = [item.value for item in WorkloadDistributionType]
            raise ValueError(
                "moe_routing_distribution_type must be one of "
                f"{valid_values}, got {distribution_type!r}"
            ) from exc

    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,  # This is a representative config
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
        cluster_config: ClusterConfig = None,
        model_manager: ExecutionTimePredictionModelManager = None,
        cluster_type: ClusterType = None,
        training_file_paths: Dict[str, str] = None,
        actual_replica_ids: Optional[list] = None,
        cc_backend: Optional["BaseCCBackend"] = None,
    ) -> None:
        # We still call super() with one of the configs to set up the basic models.
        # The prefill config is a good representative as it's a full model.
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

        assert (
            cluster_config is not None
        ), "cluster_config cannot be None for SklearnDisaggregationExecutionTimePredictor"
        self._cluster_config = cluster_config

        # Store actual replica ids if provided (to align routing_details keys with cluster replica IDs)
        self._actual_replica_ids = actual_replica_ids

        # Override MoE parameters with cluster-specific values
        # The parent class uses the representative replica_config, but we need cluster-specific configs
        self._cluster_type = cluster_type
        cluster_replica_config = replica_config
        if cluster_type:
            cluster_replica_config = self._get_cluster_replica_config(cluster_type)
            # Override MoE parameters for this specific cluster
            self._moe_ep_size = cluster_replica_config.moe_expert_parallel_size
            self._moe_tp_size = cluster_replica_config.moe_tensor_parallel_size
            self._router_topk = cluster_replica_config.router_topk

        self._workload_distribution_type = self._resolve_workload_distribution_type(
            getattr(
                cluster_replica_config,
                "moe_routing_distribution_type",
                "balanced",
            )
        )
        # Use moe_routing_seed from config for deterministic routing simulation.
        self._distribution_seed = getattr(cluster_replica_config, "moe_routing_seed", 42)

        if (
            not hasattr(self._cluster_config, "prefill_replica_config")
            or self._cluster_config.prefill_replica_config is None
        ):

            if (
                hasattr(self._cluster_config, "replica_config")
                and self._cluster_config.replica_config is not None
            ):
                self._cluster_config.prefill_replica_config = (
                    self._cluster_config.replica_config
                )
                self._cluster_config.decode_ffn_replica_config = (
                    self._cluster_config.replica_config
                )
                if not hasattr(self._cluster_config, "prefill_cluster_num_replicas"):
                    self._cluster_config.prefill_cluster_num_replicas = getattr(
                        self._cluster_config, "num_replicas", 1
                    )
                if not hasattr(self._cluster_config, "decode_ffn_cluster_num_replicas"):
                    self._cluster_config.decode_ffn_cluster_num_replicas = getattr(
                        self._cluster_config, "num_replicas", 1
                    )
            else:
                raise ValueError(
                    "Neither prefill_replica_config nor replica_config is available in cluster_config"
                )

        # Pre-calculate routing details only for relevant clusters to avoid unnecessary computation
        # Each predictor only calculates routing for clusters it will actually serve

        self._prefill_routing_details = None
        self._decode_ffn_routing_details = None
        self._decode_routing_details = None  # For unified DECODE cluster in PD-disaggregation mode

        # For an aggregate predictor, materialize only routed roles whose
        # ClusterConfig entry exists. An explicit role remains on the existing
        # path so an unavailable requested role still fails at its owner.
        routed_role_config_attributes = self._ROUTED_ROLE_CONFIG_ATTRIBUTES
        routed_roles = {role for role, _ in routed_role_config_attributes}
        if cluster_type is None:
            current_cluster_types = tuple(
                role
                for role, config_attribute in routed_role_config_attributes
                if getattr(self._cluster_config, config_attribute, None) is not None
            )
        elif cluster_type in routed_roles:
            current_cluster_types = (cluster_type,)
        else:
            current_cluster_types = ()

        # Calculate routing details for each relevant cluster type
        for target_cluster_type in current_cluster_types:
            routing_details: Dict[int, Dict[int, Dict[int, float]]] = (
                self._simulate_and_store_routing(target_cluster_type)
            )
            target_replica_config = self._get_cluster_replica_config(
                target_cluster_type
            )
            if (
                getattr(target_replica_config.model_config, "is_moe", None)
                is False
            ):
                if routing_details:
                    raise ValueError(
                        f"Dense {target_cluster_type.name} predictor produced "
                        "unexpected MoE routing details"
                    )
            else:
                self._emit_routing_details_snapshot(
                    target_cluster_type,
                    routing_details,
                )

            if target_cluster_type == ClusterType.PREFILL:
                self._prefill_routing_details = routing_details
            elif target_cluster_type == ClusterType.DECODE_FFN:
                self._decode_ffn_routing_details = routing_details
            elif target_cluster_type == ClusterType.DECODE:
                self._decode_routing_details = routing_details

        # Initialize empty routing details for clusters that don't need MoE routing
        if cluster_type == ClusterType.DECODE_ATTN:
            logger.debug(
                "DECODE_ATTN predictor skipping MoE routing calculation (not needed)"
            )

    def _get_cluster_replica_config(self, cluster_type: ClusterType) -> ReplicaConfig:
        """Get the replica config for a specific cluster type."""
        if cluster_type == ClusterType.PREFILL:
            return getattr(
                self._cluster_config, "prefill_replica_config", self._replica_config
            )
        elif cluster_type == ClusterType.DECODE_ATTN:
            return getattr(
                self._cluster_config, "decode_attn_replica_config", self._replica_config
            )
        elif cluster_type == ClusterType.DECODE_FFN:
            return getattr(
                self._cluster_config, "decode_ffn_replica_config", self._replica_config
            )
        elif cluster_type == ClusterType.DECODE:
            # Unified DECODE cluster in PD-disaggregation mode
            return getattr(
                self._cluster_config, "decode_replica_config", self._replica_config
            )
        else:
            return self._replica_config

    @staticmethod
    def _resolve_model_architecture_profile_for_config(
        model_config: Any,
    ) -> ModelArchitectureProfile:
        if model_config is None:
            raise ValueError("PDD predictor requires cluster replica model_config")
        getter = getattr(model_config, "get_model_architecture_profile", None)
        profile = getter() if callable(getter) else get_model_architecture_profile(model_config)
        if not isinstance(profile, ModelArchitectureProfile):
            raise TypeError(
                "model_config architecture profile must be ModelArchitectureProfile"
            )
        return profile

    def _get_cluster_model_architecture_profile(
        self, cluster_type: ClusterType
    ) -> ModelArchitectureProfile:
        cluster_replica_config = self._get_cluster_replica_config(cluster_type)
        return self._resolve_model_architecture_profile_for_config(
            getattr(cluster_replica_config, "model_config", None)
        )

    def _get_tensor_parallel_size_for_comm(self) -> int:
        cluster_type = self._cluster_type
        if cluster_type is None:
            return super()._get_tensor_parallel_size_for_comm()
        cluster_replica_config = self._get_cluster_replica_config(cluster_type)
        if cluster_type == ClusterType.DECODE_FFN:
            return cluster_replica_config.moe_tensor_parallel_size
        return cluster_replica_config.attn_tensor_parallel_size

    def _simulate_and_store_routing(
        self, cluster_type: ClusterType
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        """
        Pre-calculates the allocation ratio for each replica, layer, and expert in a MoE cluster.
        Returns: {replica_id: {layer_id: {global_expert_id: allocation_ratio}}}

        Args:
            cluster_type: Type of cluster (PREFILL, DECODE_FFN, or DECODE)

        Returns:
            Nested dictionary containing allocation ratios for each replica, layer, and expert
        """

        if cluster_type == ClusterType.PREFILL:
            cluster_replica_config = self._cluster_config.prefill_replica_config
            num_replicas = self._cluster_config.prefill_cluster_num_replicas
        elif cluster_type == ClusterType.DECODE_FFN:
            cluster_replica_config = self._cluster_config.decode_ffn_replica_config
            num_replicas = self._cluster_config.decode_ffn_cluster_num_replicas
        elif cluster_type == ClusterType.DECODE:
            # Unified DECODE cluster in PD-disaggregation mode
            cluster_replica_config = getattr(
                self._cluster_config, "decode_replica_config", self._replica_config
            )
            num_replicas = getattr(
                self._cluster_config, "decode_cluster_num_replicas",
                getattr(self._cluster_config, "num_replicas", 1)
            )
        else:
            raise NotImplementedError(f"Unsupported cluster_type: {cluster_type}")

        if getattr(cluster_replica_config.model_config, "is_moe", None) is False:
            return {}

        # In dummy mode, generate a valid uniform routing map instead of returning an empty dict
        if self._enable_dummy_mode:
            logger.debug(
                f"Generating uniform MoE routing for {cluster_type.name} in dummy mode"
            )
            # Determine actual replica IDs within the global ID space
            prefill_num = getattr(
                self._cluster_config, "prefill_cluster_num_replicas", None
            )
            decode_attn_num = getattr(
                self._cluster_config, "decode_attn_cluster_num_replicas", None
            )
            decode_num = getattr(
                self._cluster_config, "decode_cluster_num_replicas", None
            )
            if self._actual_replica_ids:
                replica_ids = list(self._actual_replica_ids)
            elif cluster_type == ClusterType.PREFILL:
                start_id = 0
                replica_ids = list(
                    range(start_id, start_id + (prefill_num or num_replicas))
                )
            elif cluster_type == ClusterType.DECODE_FFN:
                start_id = (prefill_num or 0) + (decode_attn_num or 0)
                decode_ffn_num = (
                    getattr(
                        self._cluster_config, "decode_ffn_cluster_num_replicas", None
                    )
                    or num_replicas
                )
                replica_ids = list(range(start_id, start_id + decode_ffn_num))
            elif cluster_type == ClusterType.DECODE:
                # Unified DECODE cluster in PD-disaggregation mode
                # DECODE cluster starts after PREFILL cluster
                start_id = prefill_num or 0
                replica_ids = list(range(start_id, start_id + (decode_num or num_replicas)))
            else:
                start_id = 0
                replica_ids = list(range(num_replicas))

            num_layers = cluster_replica_config.model_config.num_layers
            total_expert_num = max(1, cluster_replica_config.total_expert_num)
            uniform_ratio = 1.0 / float(total_expert_num)

            routing_details: Dict[int, Dict[int, Dict[int, float]]] = {}
            for rid in replica_ids:
                routing_details[rid] = {}
                for layer_id in range(num_layers):
                    routing_details[rid][layer_id] = {
                        eid: uniform_ratio for eid in range(total_expert_num)
                    }
            logger.info(
                f"[ROUTING-DUMMY] Built uniform routing for {cluster_type.name}: replica_ids={sorted(list(routing_details.keys()))}, layers={num_layers}, experts={total_expert_num}"
            )
            return routing_details

        # Allow ep=1 for testing purposes (all experts on same device)
        # For production with real EP distribution, ep > 1 is recommended
        if cluster_replica_config.total_expert_num > 1:
            assert (
                cluster_replica_config.moe_expert_parallel_size >= 1
            ), f"Expert parallel size must be >= 1 for disaggregated mode with {cluster_replica_config.total_expert_num} experts"
            if cluster_replica_config.moe_expert_parallel_size == 1:
                logger.warning(
                    f"[ROUTING] EP=1 with {cluster_replica_config.total_expert_num} experts: "
                    f"all experts on same device (no expert parallelism). "
                    f"This is valid for testing but not recommended for production."
                )
        else:
            # For non-MoE models, ep=1 is acceptable
            assert (
                cluster_replica_config.moe_expert_parallel_size >= 1
            ), f"Expert parallel size must be >= 1"

        logger.debug(
            f"Simulating routing for {cluster_type.name} cluster: "
            f"{num_replicas} replicas, {cluster_replica_config.total_expert_num} experts, "
            f"EP{cluster_replica_config.moe_expert_parallel_size}"
        )

        # # Allow expert_parallel_size = 1 for cases without expert parallelism
        # # Only require > 1 when we actually have multiple experts to distribute
        # if cluster_replica_config.total_expert_num > 1:
        #     assert cluster_replica_config.moe_expert_parallel_size >= 1, \
        #         f"Expert parallel size must be >= 1 for disaggregated mode with {cluster_replica_config.total_expert_num} experts"
        #     logger.debug(f"✅ MoE configuration valid for {cluster_type.name} cluster")
        # else:
        #     # For models without MoE (total_expert_num = 1), expert_parallel_size can be 1
        #     assert cluster_replica_config.moe_expert_parallel_size >= 1, \
        #         "Expert parallel size must be >= 1"
        #     logger.debug(f"✅ Non-MoE configuration valid for {cluster_type.name} cluster")

        # TODO: we should confirm that are the following variables well defined? We should use per-stage info or global info?
        # I think we should pre-assign here for all layers, and for process of each stage, they should know which layers to process by
        # transform local layer index to global layer index
        # num_layers = (
        #     cluster_replica_config.model_config.num_layers
        #     // cluster_replica_config.num_pipeline_stages
        # )
        num_layers = cluster_replica_config.model_config.num_layers
        total_expert_num = cluster_replica_config.total_expert_num
        expert_parallel_size = cluster_replica_config.moe_expert_parallel_size
        assert (
            total_expert_num % expert_parallel_size == 0
        ), f"Total expert num {total_expert_num} must be divisible by expert parallel size {expert_parallel_size}"

        # Initialize the routing details structure
        # Preserve replica_id key structure; enforce homogeneous allocation across replicas within the same cluster
        routing_details = {}
        # Cache per-layer expert allocations to reuse across all replicas in this cluster
        _shared_layer_allocations: Dict[int, List[float]] = {}

        # Generate allocation ratios for each replica (homogeneous across replicas)
        # Use actual global replica IDs when possible to match scheduler expectations
        prefill_num = getattr(
            self._cluster_config, "prefill_cluster_num_replicas", None
        )
        decode_attn_num = getattr(
            self._cluster_config, "decode_attn_cluster_num_replicas", None
        )
        decode_num = getattr(
            self._cluster_config, "decode_cluster_num_replicas", None
        )
        if self._actual_replica_ids:
            actual_replica_ids = list(self._actual_replica_ids)
        elif cluster_type == ClusterType.PREFILL:
            start_id = 0
            actual_replica_ids = list(
                range(start_id, start_id + (prefill_num or num_replicas))
            )
        elif cluster_type == ClusterType.DECODE_FFN:
            start_id = (prefill_num or 0) + (decode_attn_num or 0)
            actual_replica_ids = list(range(start_id, start_id + num_replicas))
        elif cluster_type == ClusterType.DECODE:
            # Unified DECODE cluster in PD-disaggregation mode
            # DECODE cluster starts after PREFILL cluster
            start_id = prefill_num or 0
            actual_replica_ids = list(range(start_id, start_id + (decode_num or num_replicas)))
        else:
            actual_replica_ids = list(range(num_replicas))

        for replica_id in actual_replica_ids:
            routing_details[replica_id] = {}

            # Generate allocation ratios for each layer
            for layer_id in range(num_layers):
                routing_details[replica_id][layer_id] = {}

                # Generate allocation ratios for each global expert; reuse per-layer allocation across replicas
                if layer_id not in _shared_layer_allocations:
                    # Use a fixed replica_id (0) to ensure identical distribution across replicas
                    _shared_layer_allocations[layer_id] = (
                        self._generate_expert_allocations(
                            total_expert_num, expert_parallel_size, 0, layer_id
                        )
                    )
                expert_allocations = _shared_layer_allocations[layer_id]

                for global_expert_id in range(total_expert_num):
                    routing_details[replica_id][layer_id][global_expert_id] = (
                        expert_allocations[global_expert_id]
                    )

        logger.info(
            f"[ROUTING] Built routing for {cluster_type.name}: replica_ids={sorted(list(routing_details.keys()))}, layers={num_layers}, experts={total_expert_num}"
        )

        return routing_details

    def _generate_expert_allocations(
        self,
        total_expert_num: int,
        expert_parallel_size: int,
        replica_id: int,
        layer_id: int,
    ) -> List[float]:
        """
        Generate allocation ratios for all experts based on the configured distribution type.

        Args:
            total_expert_num: Total number of experts in the model
            expert_parallel_size: Number of experts handled in parallel
            replica_id: ID of the current replica
            layer_id: ID of the current layer

        Returns:
            List of allocation ratios for each expert (sum should be 1.0)
        """
        # Replica IDs select the shared routing lookup only; they must not
        # create different per-layer distributions across architectures.
        rng = np.random.default_rng(self._distribution_seed + layer_id)

        if self._workload_distribution_type == WorkloadDistributionType.BALANCED:
            # Balanced distribution: each expert gets equal allocation
            allocation_ratios = [1.0 / total_expert_num] * total_expert_num

        elif self._workload_distribution_type == WorkloadDistributionType.RANDOM:
            # Random distribution: generate random weights and normalize
            random_weights = rng.uniform(0.1, 1.0, total_expert_num)
            total_weight = np.sum(random_weights)
            allocation_ratios = (random_weights / total_weight).tolist()

        elif self._workload_distribution_type == WorkloadDistributionType.SKEWED:
            # Moderate deterministic power-law skew for realistic hot-expert stress.
            ranks = np.arange(1, total_expert_num + 1)
            skew_weights = 1.0 / np.power(ranks, 0.35)
            total_weight = np.sum(skew_weights)
            allocation_ratios = (skew_weights / total_weight).tolist()

        elif self._workload_distribution_type == WorkloadDistributionType.ZIPF:
            # Zipf distribution: some experts get more load than others
            ranks = np.arange(1, total_expert_num + 1)
            zipf_weights = 1.0 / ranks  # Zipf-like distribution
            total_weight = np.sum(zipf_weights)
            allocation_ratios = (zipf_weights / total_weight).tolist()

        else:
            raise ValueError(
                f"Unsupported workload distribution type: {self._workload_distribution_type}"
            )

        # Ensure the allocation ratios sum to 1.0 (handle floating point precision)
        total_allocation = sum(allocation_ratios)
        allocation_ratios = [ratio / total_allocation for ratio in allocation_ratios]

        return allocation_ratios

    def _get_replica_expert_workload_ratio(
        self,
        routing_details: Dict[int, Dict[int, Dict[int, float]]],
        replica_id: int,
        layer_id: int,
    ) -> float:
        """
        Calculate the total workload ratio for a replica at a specific layer.
        This aggregates the allocation ratios across all experts for the replica.

        Args:
            routing_details: The routing details dictionary
            replica_id: ID of the replica
            layer_id: ID of the layer

        Returns:
            Total workload ratio for the replica at the specified layer
        """
        if (
            replica_id not in routing_details
            or layer_id not in routing_details[replica_id]
        ):
            return 0.0

        # Sum up allocation ratios across all experts for this replica and layer
        total_ratio = sum(routing_details[replica_id][layer_id].values())
        return total_ratio

    def _get_grouped_gemm_time(
        self,
        num_tokens_or_allocation,
        batch: Optional[Batch] = None,
    ) -> float:
        """Delegate grouped GEMM prediction to the MoE base predictor implementation.

        This disaggregation predictor must share exactly the same grouped GEMM
        modeling semantics as monolithic and pd-disaggregation MoE predictors.

        Args:
            num_tokens_or_allocation: Either an integer token count for the
                scalar compatibility path or an ``EPLaneWorkload`` descriptor
                for physical EP-lane prediction.
            batch: Optional batch context for decode-phase-only calibration.

        Returns:
            Predicted grouped GEMM execution time in milliseconds.
        """
        return super()._get_grouped_gemm_time(
            num_tokens_or_allocation,
            batch=batch,
        )

    def _get_dummy_execution_time_for_cluster(
        self,
        batch: Batch,
        pipeline_stage: int,
        cluster_type: ClusterType = None,
        include_attention: bool = True,
        include_moe: Optional[bool] = None,
        include_ffn: bool = True,
        layer_ids: Optional[List[int] | tuple[int, ...]] = None,
    ) -> ExecutionTime:
        """Return cluster-specific dummy ExecutionTime object."""
        if cluster_type is None:
            raise ValueError(
                "cluster_type cannot be None for cluster-specific dummy execution time"
            )
        if include_moe is not None and type(include_moe) is not bool:
            raise ValueError("include_moe must be a bool or None")
        if type(include_ffn) is not bool:
            raise ValueError("include_ffn must be a bool")
        if not include_ffn and include_moe is not None:
            raise ValueError(
                "include_moe must be None for an attention-only stage probe"
            )
        if not include_ffn and cluster_type == ClusterType.DECODE_FFN:
            raise ValueError(
                "Attention-only prediction is invalid for the DECODE_FFN cluster"
            )

        base_time = self._dummy_execution_time
        routed_token_count = self._get_ep_lane_routed_token_count(batch)

        cluster_replica_config = self._get_cluster_replica_config(cluster_type)
        # A direct helper caller has no layer identity, so preserve the historic
        # model-level default. Public single-layer calls pass the resolved
        # concrete classification explicitly.
        model_config = cluster_replica_config.model_config
        model_is_moe = bool(
            model_config is not None and getattr(model_config, "is_moe", False)
        )
        # Shared-domain dummy stages use this selector to suppress their FFN
        # portion.  DECODE_ATTN and DECODE_FFN retain their role-owned legacy
        # composition; the public stage boundary validates their selectors.
        ffn_enabled = include_ffn or cluster_type not in (
            ClusterType.PREFILL,
            ClusterType.DECODE,
        )
        is_moe_model = (
            ffn_enabled and (model_is_moe if include_moe is None else include_moe)
        )
        zero_routed_ep_lane = is_moe_model and routed_token_count == 0
        moe_ep_size = cluster_replica_config.moe_expert_parallel_size
        # DECODE_ATTN is intentionally attention-only; its cluster replica
        # config sets MoE parallelism fields to zero even when the model itself
        # is MoE. Validate EP topology only for clusters that execute MoE/FFN.
        if (
            is_moe_model
            and cluster_type != ClusterType.DECODE_ATTN
            and (type(moe_ep_size) is not int or moe_ep_size <= 0)
        ):
            raise ValueError(
                "Dummy MoE prediction requires a positive integer "
                f"moe_expert_parallel_size, got {moe_ep_size!r}"
            )
        ep_phase_time = (
            base_time if is_moe_model and moe_ep_size > 1 else 0.0
        )
        ep_communication_time = ep_phase_time * 2
        ep_operator_times = (
            CommunicationOperatorTimes(
                {
                    "expert_parallel_alltoall_dispatch": ep_phase_time,
                    "expert_parallel_alltoall_combine": ep_phase_time,
                }
            )
            if is_moe_model
            else None
        )
        architecture_profile = self._get_cluster_model_architecture_profile(cluster_type)
        share_expert_enabled = (
            is_moe_model
            and cluster_replica_config.model_config is not None
            and cluster_replica_config.model_config.supports_share_expert()
        )
        share_expert_time = base_time if share_expert_enabled else 0.0
        attn_tp_size = int(cluster_replica_config.attn_tensor_parallel_size)
        moe_tp_size = int(cluster_replica_config.moe_tensor_parallel_size)
        if cluster_type == ClusterType.DECODE_FFN:
            tp_size = moe_tp_size
        else:
            tp_size = attn_tp_size
        pp_stage_boundary_handoff_time = (
            base_time
            if pipeline_stage < cluster_replica_config.num_pipeline_stages - 1
            else 0.0
        )
        # COMM_SKIP: TP all-reduce not needed when tp_size <= 1 (no tensor sharding)
        attn_tp_comm_time = base_time if attn_tp_size > 1 else 0.0
        # DECODE_FFN owns the FFN tensor-parallel domain for every FFN layer,
        # including dense boundary layers in a mixed model.  The legacy
        # ``moe_tensor_parallel_allreduce_time`` field is the shared
        # MLP/MoE-output slot used by ExecutionTime for that role-specific
        # collective; PREFILL and unified DECODE retain their model-level MoE
        # selection here.
        moe_tp_comm_time = (
            base_time
            if moe_tp_size > 1
            and (is_moe_model or cluster_type == ClusterType.DECODE_FFN)
            else 0.0
        )
        tp_comm_time = (
            moe_tp_comm_time if cluster_type == ClusterType.DECODE_FFN else attn_tp_comm_time
        )
        attention_tp_comm_time = attn_tp_comm_time if include_attention else 0.0
        # MoE TP all-reduce covers the shared pre-routing hidden-state domain.
        # A zero-routed physical lane still participates in that collective.
        moe_tp_allreduce_time = moe_tp_comm_time
        routed_grouped_gemm_time = (
            base_time if is_moe_model and not zero_routed_ep_lane else 0.0
        )
        routed_moe_shuffling_time = (
            base_time if is_moe_model and not zero_routed_ep_lane else 0.0
        )
        ffn_tp_comm_enabled = (
            cluster_type == ClusterType.DECODE_FFN
            and is_moe_model
            and architecture_profile.moe_tensor_parallel_allgather_op is not None
            and moe_tp_size > 1
        )
        ffn_tp_allgather_time = base_time if ffn_tp_comm_enabled else 0.0
        share_expert_tp_allreduce_time = (
            base_time
            if (
                ffn_tp_comm_enabled
                and share_expert_enabled
                and architecture_profile.share_expert_tensor_parallel_allreduce_op is not None
            )
            else 0.0
        )
        ffn_time = base_time if ffn_enabled else 0.0

        if cluster_type == ClusterType.PREFILL:
            # PREFILL cluster handles full model layers
            return ExecutionTime(
                num_layers_per_pipeline_stage=self._num_layers_per_pipeline_stage,
                attention_rope_execution_time=(
                    base_time if include_attention else 0.0
                ),
                attention_kv_cache_save_execution_time=(
                    base_time if include_attention else 0.0
                ),
                attention_decode_execution_time=0.0,  # No decode in prefill
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
                mlp_norm_time=ffn_time,
                add_time=ffn_time,
                tensor_parallel_communication_time=attention_tp_comm_time,
                attn_tensor_parallel_allreduce_time=attention_tp_comm_time,
                moe_tensor_parallel_allreduce_time=moe_tp_allreduce_time,
                pipeline_parallel_communication_time=base_time,
                expert_parallel_communication_time=ep_communication_time,
                moe_gating_time=base_time if is_moe_model else 0.0,
                moe_shuffling_time=routed_moe_shuffling_time,
                schedule_time=base_time,
                sampler_e2e_time=base_time,
                prepare_inputs_e2e_time=base_time,
                process_model_outputs_time=base_time,
                ray_comm_time=base_time,
                pp_stage_boundary_handoff_time=pp_stage_boundary_handoff_time,
                is_moe=is_moe_model,  # Determined by cluster replica config
                mlp_layer_up_proj_execution_time=ffn_time,
                mlp_layer_down_proj_execution_time=ffn_time,
                mlp_layer_act_execution_time=ffn_time,
                moe_grouped_gemm_time=routed_grouped_gemm_time,
                share_expert_up_proj_time=share_expert_time if ffn_enabled else 0.0,
                share_expert_down_proj_time=share_expert_time if ffn_enabled else 0.0,
                share_expert_act_time=share_expert_time if ffn_enabled else 0.0,
                communication_operator_times=ep_operator_times,
                layer_ids=layer_ids,
            )
        elif cluster_type == ClusterType.DECODE:
            # Unified DECODE cluster (PD-disaggregation mode): attention + (MLP/MoE)
            return ExecutionTime(
                num_layers_per_pipeline_stage=self._num_layers_per_pipeline_stage,
                attention_rope_execution_time=(
                    base_time if include_attention else 0.0
                ),
                attention_kv_cache_save_execution_time=(
                    base_time if include_attention else 0.0
                ),
                attention_decode_execution_time=(
                    base_time if include_attention else 0.0
                ),
                attention_prefill_execution_time=0.0,  # No prefill in decode
                attention_layer_pre_proj_execution_time=(
                    base_time if include_attention else 0.0
                ),
                attention_layer_post_proj_execution_time=(
                    base_time if include_attention else 0.0
                ),
                attn_norm_time=base_time if include_attention else 0.0,
                mlp_norm_time=ffn_time,
                add_time=ffn_time,
                tensor_parallel_communication_time=attention_tp_comm_time,
                attn_tensor_parallel_allreduce_time=attention_tp_comm_time,
                moe_tensor_parallel_allreduce_time=moe_tp_allreduce_time,
                pipeline_parallel_communication_time=base_time,
                expert_parallel_communication_time=ep_communication_time,
                moe_gating_time=base_time if is_moe_model else 0.0,
                moe_shuffling_time=routed_moe_shuffling_time,
                schedule_time=base_time,
                sampler_e2e_time=base_time,
                prepare_inputs_e2e_time=base_time,
                process_model_outputs_time=base_time,
                ray_comm_time=base_time,
                pp_stage_boundary_handoff_time=pp_stage_boundary_handoff_time,
                is_moe=is_moe_model,  # Determined by cluster replica config
                mlp_layer_up_proj_execution_time=ffn_time,
                mlp_layer_down_proj_execution_time=ffn_time,
                mlp_layer_act_execution_time=ffn_time,
                moe_grouped_gemm_time=routed_grouped_gemm_time,
                share_expert_up_proj_time=share_expert_time if ffn_enabled else 0.0,
                share_expert_down_proj_time=share_expert_time if ffn_enabled else 0.0,
                share_expert_act_time=share_expert_time if ffn_enabled else 0.0,
                communication_operator_times=ep_operator_times,
                layer_ids=layer_ids,
            )
        elif cluster_type == ClusterType.DECODE_ATTN:
            # DECODE_ATTN cluster only handles attention operations
            add_attn_residual_time = (
                0.0 if architecture_profile.skip_decode_attn_residual else base_time
            )
            return ExecutionTime(
                num_layers_per_pipeline_stage=1,
                attention_rope_execution_time=base_time,
                attention_kv_cache_save_execution_time=base_time,
                attention_decode_execution_time=base_time,
                attention_prefill_execution_time=0.0,  # No prefill in decode
                attention_layer_pre_proj_execution_time=base_time,
                attention_layer_post_proj_execution_time=base_time,
                attn_norm_time=base_time,
                mlp_norm_time=base_time,
                add_time=0.0,
                add_attn_residual_time=add_attn_residual_time,
                add_ffn_residual_time=0.0,
                tensor_parallel_communication_time=tp_comm_time,
                pipeline_parallel_communication_time=0.0,
                expert_parallel_communication_time=0.0,
                moe_gating_time=0.0,
                moe_shuffling_time=0.0,
                schedule_time=base_time,
                sampler_e2e_time=base_time,
                prepare_inputs_e2e_time=base_time,
                process_model_outputs_time=base_time,
                ray_comm_time=base_time,
                pp_stage_boundary_handoff_time=pp_stage_boundary_handoff_time,
                is_moe=False,  # DECODE_ATTN cluster doesn't handle MoE
                mlp_layer_up_proj_execution_time=0.0,  # No MLP in attention cluster
                mlp_layer_down_proj_execution_time=0.0,
                mlp_layer_act_execution_time=0.0,
                moe_grouped_gemm_time=0.0,  # No MoE in attention cluster
                layer_ids=layer_ids,
            )
        elif cluster_type == ClusterType.DECODE_FFN:
            # DECODE_FFN cluster only handles FFN/MoE operations
            routed_grouped_gemm_time = (
                base_time * 0.5
                if is_moe_model and not zero_routed_ep_lane
                else 0.0
            )
            return ExecutionTime(
                num_layers_per_pipeline_stage=1,
                attention_rope_execution_time=0.0,  # No attention in FFN cluster
                attention_kv_cache_save_execution_time=0.0,
                attention_decode_execution_time=0.0,
                attention_prefill_execution_time=0.0,
                attention_layer_pre_proj_execution_time=0.0,
                attention_layer_post_proj_execution_time=0.0,
                attn_norm_time=0.0,
                mlp_norm_time=base_time,
                add_time=base_time,
                tensor_parallel_communication_time=tp_comm_time,
                moe_tensor_parallel_allreduce_time=moe_tp_allreduce_time,
                pipeline_parallel_communication_time=0.0,
                expert_parallel_communication_time=ep_communication_time,
                # In dummy mode, keep the per-layer MoE compute (gating + grouped_gemm)
                # roughly equal to base_time to avoid artificial Te >> Ta imbalance.
                moe_gating_time=base_time * 0.5 if is_moe_model else 0.0,
                moe_shuffling_time=routed_moe_shuffling_time,
                schedule_time=base_time,
                sampler_e2e_time=base_time,
                prepare_inputs_e2e_time=base_time,
                process_model_outputs_time=base_time,
                ray_comm_time=base_time,
                pp_stage_boundary_handoff_time=pp_stage_boundary_handoff_time,
                is_moe=is_moe_model,  # Determined by cluster replica config
                mlp_layer_up_proj_execution_time=base_time,
                mlp_layer_down_proj_execution_time=base_time,
                mlp_layer_act_execution_time=base_time,
                moe_grouped_gemm_time=routed_grouped_gemm_time,
                share_expert_up_proj_time=share_expert_time,
                share_expert_down_proj_time=share_expert_time,
                share_expert_act_time=share_expert_time,
                tensor_parallel_allgather_time=ffn_tp_allgather_time,
                share_expert_tensor_parallel_allreduce_time=share_expert_tp_allreduce_time,
                communication_operator_times=ep_operator_times,
                layer_ids=layer_ids,
            )

        raise ValueError(
            f"Unsupported cluster_type for dummy execution time: {cluster_type}"
        )

    # Phase 2.5: Removed deprecated get_execution_time() method
    # All active code paths now use predict_stage_execution_time() instead

    def _get_zero_moe_mlp_params(self) -> Dict[str, Any]:
        return {
            "mlp_layer_up_proj_execution_time": 0.0,
            "mlp_layer_down_proj_execution_time": 0.0,
            "mlp_layer_act_execution_time": 0.0,
            "mlp_norm_time": 0.0,
            "moe_grouped_gemm_time": 0.0,
            "expert_parallel_communication_time": 0.0,
            "moe_gating_time": 0.0,
            "moe_shuffling_time": 0.0,
            "is_moe": False,
        }

    def _get_zero_moe_params(self) -> Dict[str, Any]:
        """Return zero values for MoE-specific parameters (for dense models)."""
        return {
            "moe_grouped_gemm_time": 0.0,
            "expert_parallel_communication_time": 0.0,
            "moe_gating_time": 0.0,
            "moe_shuffling_time": 0.0,
            "is_moe": False,
        }

    def _get_zero_attn_params(self) -> Dict[str, Any]:
        """Return zero values for attention-specific parameters (for FFN/MoE cluster)."""
        return {
            "attention_rope_execution_time": 0.0,
            "attention_kv_cache_save_execution_time": 0.0,
            "attention_decode_execution_time": 0.0,
            "attention_prefill_execution_time": 0.0,
            "attention_layer_pre_proj_execution_time": 0.0,
            "attention_layer_post_proj_execution_time": 0.0,
            "attn_norm_time": 0.0,
        }

    @staticmethod
    def _is_zero_token_decode_ffn_ep_barrier(
        batch: Batch,
        cluster_type: ClusterType,
    ) -> bool:
        """Return True for explicit zero-token DECODE_FFN EP barrier batches."""
        if cluster_type != ClusterType.DECODE_FFN:
            return False
        lane_workload = resolve_ep_lane_workload(batch, required=False)
        if lane_workload is None:
            return False
        return batch.total_num_tokens == 0 and lane_workload.routed_token_count == 0

    @staticmethod
    def _get_zero_decode_ffn_ep_barrier_execution_time(
        num_layers: int,
        layer_ids: Optional[List[int] | tuple[int, ...]] = None,
    ) -> ExecutionTime:
        """Build a zero-cost execution-time object for DECODE_FFN EP barriers."""
        return ExecutionTime(
            num_layers_per_pipeline_stage=num_layers,
            attention_rope_execution_time=0.0,
            attention_kv_cache_save_execution_time=0.0,
            attention_decode_execution_time=0.0,
            attention_prefill_execution_time=0.0,
            attention_layer_pre_proj_execution_time=0.0,
            attention_layer_post_proj_execution_time=0.0,
            attn_norm_time=0.0,
            mlp_norm_time=0.0,
            add_time=0.0,
            tensor_parallel_communication_time=0.0,
            pipeline_parallel_communication_time=0.0,
            expert_parallel_communication_time=0.0,
            moe_gating_time=0.0,
            moe_shuffling_time=0.0,
            schedule_time=0.0,
            sampler_e2e_time=0.0,
            prepare_inputs_e2e_time=0.0,
            process_model_outputs_time=0.0,
            ray_comm_time=0.0,
            is_moe=True,
            moe_grouped_gemm_time=0.0,
            moe_gating_linear_time=0.0,
            moe_gating_routing_topk_time=0.0,
            add_attn_residual_time=0.0,
            add_ffn_residual_time=0.0,
            share_expert_up_proj_time=0.0,
            share_expert_down_proj_time=0.0,
            share_expert_act_time=0.0,
            tensor_parallel_allgather_time=0.0,
            share_expert_tensor_parallel_allreduce_time=0.0,
            dp_input_allreduce_time=0.0,
            dp_output_allreduce_time=0.0,
            attn_tensor_parallel_allreduce_time=0.0,
            moe_tensor_parallel_allreduce_time=0.0,
            pp_stage_boundary_handoff_time=0.0,
            communication_operator_times=CommunicationOperatorTimes(
                {
                    "expert_parallel_alltoall_dispatch": 0.0,
                    "expert_parallel_alltoall_combine": 0.0,
                }
            ),
            layer_ids=layer_ids,
        )

    # Phase 2.5: Removed deprecated get_moe_stage_execution_details() method
    # MoE models now use predict_moe_layer_time() and other fine-grained APIs

    # ========================================================================
    # New unified API implementation (Phase 0) - Disaggregation extensions
    # ========================================================================

    def _get_communication_time(
        self,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        *,
        include_attention: bool = True,
    ) -> CommunicationTime:
        """
        Get communication times for a batch at a given stage.

        This includes:
        - Tensor parallel all-reduce (if TP > 1)
        - Pipeline parallel send/recv (if PP > 1)

        Args:
            batch: The batch being processed
            stage_id: Pipeline stage ID
            cluster_type: Type of cluster

        Returns:
            CommunicationTime object with tensor_parallel_time and pipeline_parallel_time
        """
        tensor_parallel_time = 0.0
        pipeline_parallel_time = 0.0

        # Tensor parallel communication (all-reduce)
        if (
            include_attention
            and self._supports_operation("tensor_parallel_communication")
        ):
            tensor_parallel_time = self._get_tensor_parallel_communication_time(batch)

        # Pipeline parallel communication (send/recv)
        if self._supports_operation("pipeline_parallel_communication"):
            pipeline_parallel_time = self._get_pipeline_parallel_communication_time(
                batch
            )

        return CommunicationTime(
            tensor_parallel_time=tensor_parallel_time,
            pipeline_parallel_time=pipeline_parallel_time,
        )

    def _get_overhead_time(
        self, batch: Batch, cluster_type: ClusterType, stage_id: int
    ) -> OverheadTime:
        """
        Get CPU overhead times for a batch.

        This includes:
        - Schedule time
        - Sampler time
        - Prepare inputs time
        - Process outputs time
        - Ray communication time
        - Active PP producer send-path runtime overhead
        - Active PP receiver-head runtime overhead

        Args:
            batch: The batch being processed
            cluster_type: Type of cluster
            stage_id: Pipeline stage ID

        Returns:
            OverheadTime object with all CPU overhead times
        """
        pp_receiver_head_runtime_time = self._get_pp_receiver_head_runtime_time(
            batch, stage_id
        )
        pp_prefill_consumer_active_runtime_time = (
            self._get_pp_prefill_consumer_active_runtime_time(batch, stage_id)
        )
        pp_stage_boundary_residual_runtime_time = (
            self._get_pp_stage_boundary_residual_runtime_time(
                batch=batch,
                cluster_type=cluster_type,
                stage_id=stage_id,
                pp_receiver_head_runtime_time=pp_receiver_head_runtime_time,
                pp_prefill_consumer_active_runtime_time=(
                    pp_prefill_consumer_active_runtime_time
                ),
            )
        )
        return OverheadTime(
            schedule_time=self._get_schedule_time(batch),
            sampler_e2e_time=self._get_sampler_e2e_time(batch),
            prepare_inputs_e2e_time=self._get_prepare_inputs_e2e_time(batch),
            process_model_outputs_time=self._get_process_model_outputs_time(batch),
            ray_comm_time=self._get_ray_comm_time(batch),
            pp_producer_send_path_runtime_time=(
                self._get_pp_producer_send_path_runtime_time(batch, stage_id)
            ),
            pp_receiver_head_runtime_time=pp_receiver_head_runtime_time,
            pp_prefill_consumer_active_runtime_time=(
                pp_prefill_consumer_active_runtime_time
            ),
            pp_stage_boundary_residual_runtime_time=(
                pp_stage_boundary_residual_runtime_time
            ),
            pp_stage_boundary_handoff_time=(
                self._get_pp_stage_boundary_handoff_time(batch, stage_id)
            ),
        )

    def _get_pp_stage_boundary_residual_runtime_time(
        self,
        *,
        batch: Batch,
        cluster_type: ClusterType,
        stage_id: int,
        pp_receiver_head_runtime_time: float,
        pp_prefill_consumer_active_runtime_time: float,
    ) -> float:
        """Return active shared-domain PP boundary residual for consumer stages."""
        if stage_id <= 0:
            return 0.0

        num_prefill_tokens = int(getattr(batch, "num_prefill_tokens", 0))
        num_decode_tokens = int(getattr(batch, "num_decode_tokens", 0))
        boundary_lookup_stage_id = stage_id - 1
        boundary_runtime_ms = self._get_pp_stage_boundary_handoff_time(
            batch, boundary_lookup_stage_id
        )
        if boundary_runtime_ms <= 0.0:
            return 0.0

        if cluster_type == ClusterType.DECODE:
            if num_prefill_tokens != 0 or num_decode_tokens <= 0:
                return 0.0
            covered_runtime_ms = pp_receiver_head_runtime_time
            return max(0.0, boundary_runtime_ms - covered_runtime_ms)

        if cluster_type == ClusterType.PREFILL:
            if num_prefill_tokens <= 0 or num_decode_tokens != 0:
                return 0.0
            covered_runtime_ms = (
                self._get_pp_producer_send_path_runtime_time(
                    batch, boundary_lookup_stage_id
                )
                + pp_prefill_consumer_active_runtime_time
            )
            return max(0.0, boundary_runtime_ms - covered_runtime_ms)

        return 0.0

    def _predict_one_op_time(
        self,
        op_name: str,
        op_time_ms: float,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int,
    ) -> float:
        """Validate and return single-layer op/comm/residual time in milliseconds."""
        if num_layers < 1:
            raise ValueError(
                f"[LAYER_SCALING_ERROR] num_layers must be >= 1, got {num_layers} "
                f"(op={op_name}, cluster={cluster_type}, stage={stage_id})"
            )

        if op_time_ms is None:
            raise ValueError(
                f"[LAYER_SCALING_ERROR] Predicted time is None for op={op_name} "
                f"(cluster={cluster_type}, stage={stage_id}, num_layers={num_layers})"
            )

        return self._validate_prediction_value(
            op_time_ms,
            operation_name=op_name,
            batch=batch,
            context=f"cluster={cluster_type}, stage={stage_id}, num_layers={num_layers}",
        )

    def _predict_named_ep_phase_operator_times(
        self,
        *,
        batch: Batch,
        lane_workload: EPLaneWorkload | None = None,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int,
    ) -> dict[str, float]:
        phase_times = self._predict_expert_parallel_phase_operator_times(
            batch,
            lane_workload=lane_workload,
        )
        return {
            op_name: self._predict_one_op_time(
                op_name,
                phase_time_ms,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            )
            for op_name, phase_time_ms in phase_times.items()
        }

    def _predict_attention_only_stage_execution_time(
        self,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int,
        layer_id: int = 0,
        layer_ids: Optional[List[int] | tuple[int, ...]] = None,
    ) -> ExecutionTime:
        """Predict only attention for a shared-domain layer probe.

        The shared PREFILL/DECODE schedulers use this result to schedule the
        attention prefix before the layer-local MoE wave.  It must not inspect
        either dense FFN or routed-expert profiling rows; the subsequent
        dense/EP operation performs the only FFN lookup for the layer.

        ``layer_id`` is the global transformer-layer identity. The default is
        retained for legacy direct helper callers that do not carry it.
        """

        attention_time = self.predict_attention_layer_time(
            batch, layer_id=layer_id, cluster_type=cluster_type
        )
        communication_time = self._get_communication_time(
            batch, stage_id, cluster_type
        )
        overhead_time = self._get_overhead_time(batch, cluster_type, stage_id)

        return ExecutionTime(
            num_layers_per_pipeline_stage=num_layers,
            attention_rope_execution_time=self._predict_one_op_time(
                "attention_rope_execution_time",
                attention_time.attention_rope_execution_time,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            ),
            attention_kv_cache_save_execution_time=self._predict_one_op_time(
                "attention_kv_cache_save_execution_time",
                attention_time.attention_kv_cache_save_execution_time,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            ),
            attention_decode_execution_time=self._predict_one_op_time(
                "attention_decode_execution_time",
                attention_time.attention_decode_execution_time,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            ),
            attention_prefill_execution_time=self._predict_one_op_time(
                "attention_prefill_execution_time",
                attention_time.attention_prefill_execution_time,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            ),
            attention_layer_pre_proj_execution_time=self._predict_one_op_time(
                "attention_layer_pre_proj_execution_time",
                attention_time.attention_layer_pre_proj_execution_time,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            ),
            attention_layer_post_proj_execution_time=self._predict_one_op_time(
                "attention_layer_post_proj_execution_time",
                attention_time.attention_layer_post_proj_execution_time,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            ),
            attn_norm_time=self._predict_one_op_time(
                "attn_norm_time",
                attention_time.attn_norm_time,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            ),
            mlp_norm_time=0.0,
            add_time=0.0,
            add_attn_residual_time=0.0,
            add_ffn_residual_time=0.0,
            tensor_parallel_communication_time=self._predict_one_op_time(
                "tensor_parallel_communication_time",
                communication_time.tensor_parallel_time,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            ),
            attn_tensor_parallel_allreduce_time=self._predict_one_op_time(
                "attn_tensor_parallel_allreduce_time",
                communication_time.tensor_parallel_time,
                batch,
                stage_id,
                cluster_type,
                num_layers,
            ),
            moe_tensor_parallel_allreduce_time=0.0,
            tensor_parallel_allgather_time=0.0,
            share_expert_tensor_parallel_allreduce_time=0.0,
            pipeline_parallel_communication_time=communication_time.pipeline_parallel_time,
            schedule_time=overhead_time.schedule_time,
            sampler_e2e_time=overhead_time.sampler_e2e_time,
            prepare_inputs_e2e_time=overhead_time.prepare_inputs_e2e_time,
            process_model_outputs_time=overhead_time.process_model_outputs_time,
            ray_comm_time=overhead_time.ray_comm_time,
            pp_producer_send_path_runtime_time=(
                overhead_time.pp_producer_send_path_runtime_time
            ),
            pp_receiver_head_runtime_time=overhead_time.pp_receiver_head_runtime_time,
            pp_prefill_consumer_active_runtime_time=(
                overhead_time.pp_prefill_consumer_active_runtime_time
            ),
            pp_stage_boundary_residual_runtime_time=(
                overhead_time.pp_stage_boundary_residual_runtime_time
            ),
            pp_stage_boundary_handoff_time=overhead_time.pp_stage_boundary_handoff_time,
            mlp_layer_up_proj_execution_time=0.0,
            mlp_layer_down_proj_execution_time=0.0,
            mlp_layer_act_execution_time=0.0,
            attention_operator_times=attention_time.operator_times,
            layer_ids=layer_ids,
            **self._get_zero_moe_params(),
        )

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
        Predict aggregated execution time for one or more transformer layers (disaggregated architecture).

        Overrides parent implementation to handle cluster-specific operation filtering:
        - PREFILL: Full model (attention + MLP/MoE)
        - DECODE_ATTN: Attention only
        - DECODE_FFN: MLP/MoE only

        Layer aggregation contract:
        - This predictor emits single-layer op/comm/residual components.
        - ExecutionTime applies num_layers_per_pipeline_stage aggregation.
        """
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
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
            cluster_type not in (ClusterType.PREFILL, ClusterType.DECODE)
            or not include_ffn
        ):
            raise ValueError(
                "Post-attention-only prediction requires shared-domain PREFILL "
                "or unified DECODE with FFN enabled"
            )
        if include_moe is not None and type(include_moe) is not bool:
            raise ValueError("include_moe must be a bool or None")
        if not include_attention and include_moe is False:
            raise ValueError(
                "Post-attention-only prediction requires a MoE layer; "
                "include_moe=False selects a dense FFN branch"
            )
        if not include_ffn and include_moe is not None:
            raise ValueError(
                "include_moe must be None for an attention-only stage probe"
            )
        if not include_ffn and cluster_type == ClusterType.DECODE_FFN:
            raise ValueError(
                "Attention-only prediction is invalid for the DECODE_FFN cluster"
            )

        # Resolve the existing cluster/layer classification before dummy timing,
        # measurement activation, or any downstream lookup.  DECODE_ATTN is an
        # attention-only role even when its model is MoE; the other FFN-capable
        # roles use the shared model/layer capability resolver.
        admission_replica_config = None
        routed_moe_for_admission = False
        admission_ep_size = None
        admission_router_topk = None
        execution_include_moe: bool | None = (
            False
            if not include_ffn or cluster_type == ClusterType.DECODE_ATTN
            else None
        )
        if include_ffn and cluster_type in (
            ClusterType.PREFILL,
            ClusterType.DECODE,
            ClusterType.DECODE_FFN,
        ):
            admission_replica_config = self._get_cluster_replica_config(cluster_type)
            admission_model_config = getattr(
                admission_replica_config, "model_config", None
            )
            execution_include_moe = self._resolve_moe_layer_classification(
                admission_model_config,
                layer_id=layer_id,
                num_layers=num_layers,
                include_moe=include_moe,
                include_ffn=True,
            )
            routed_moe_for_admission = execution_include_moe
            admission_ep_size = getattr(
                admission_replica_config,
                "moe_expert_parallel_size",
                None,
            )
            admission_router_topk = getattr(
                admission_replica_config,
                "router_topk",
                None,
            )

        if not include_attention and not execution_include_moe:
            raise ValueError(
                "Post-attention-only prediction requires a MoE layer; "
                f"layer_id={layer_id} is dense"
            )

        self._admit_routed_ep_aggregate(
            batch,
            routed_moe=routed_moe_for_admission,
            ep_size=admission_ep_size,
            router_topk=admission_router_topk,
        )

        if self._enable_dummy_mode:
            if cluster_type in (
                ClusterType.PREFILL,
                ClusterType.DECODE_ATTN,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            ):
                self._log_architecture_attention_shape(batch)
            # Phase 1 Fix: Use cluster-specific dummy execution time
            dummy_exec_time = self._get_dummy_execution_time_for_cluster(
                batch,
                stage_id,
                cluster_type,
                include_attention=include_attention,
                include_moe=(execution_include_moe if include_ffn else None),
                include_ffn=include_ffn,
                layer_ids=(
                    normalized_layer_ids
                    if normalized_layer_ids is not None
                    and len(normalized_layer_ids)
                    == (
                        self._num_layers_per_pipeline_stage
                        if cluster_type
                        in (ClusterType.PREFILL, ClusterType.DECODE)
                        else 1
                    )
                    else None
                ),
            )

            # If num_layers matches, return as-is
            if num_layers == dummy_exec_time.num_layers:
                return dummy_exec_time

            # Otherwise, scale to requested num_layers
            if num_layers != 1:
                logger.warning(
                    f"Dummy-mode layer scaling: requested num_layers={num_layers}; "
                    "scaling dummy single-layer components and preserving ExecutionTime aggregation contract."
                )

            scale_factor = num_layers / dummy_exec_time.num_layers
            scaled_communication_operator_times = None
            if dummy_exec_time._is_moe:
                source_operator_times = dummy_exec_time.communication_operator_times
                if source_operator_times is None:
                    raise ValueError(
                        "Dummy MoE ExecutionTime is missing named EP phase times"
                    )
                scaled_communication_operator_times = CommunicationOperatorTimes(
                    {
                        op_name: time_ms * scale_factor
                        for op_name, time_ms in source_operator_times.op_times.items()
                    }
                )

            # Create scaled ExecutionTime
            return ExecutionTime(
                num_layers_per_pipeline_stage=num_layers,
                attention_rope_execution_time=dummy_exec_time._attention_rope_execution_time
                * scale_factor,
                attention_kv_cache_save_execution_time=dummy_exec_time._attention_kv_cache_save_execution_time
                * scale_factor,
                attention_decode_execution_time=dummy_exec_time._attention_decode_execution_time
                * scale_factor,
                attention_prefill_execution_time=dummy_exec_time._attention_prefill_execution_time
                * scale_factor,
                attention_layer_pre_proj_execution_time=dummy_exec_time._attention_layer_pre_proj_execution_time
                * scale_factor,
                attention_layer_post_proj_execution_time=dummy_exec_time._attention_layer_post_proj_execution_time
                * scale_factor,
                attn_norm_time=dummy_exec_time._attn_norm_time * scale_factor,
                mlp_norm_time=dummy_exec_time._mlp_norm_time * scale_factor,
                add_time=dummy_exec_time._add_time * scale_factor,
                add_attn_residual_time=dummy_exec_time._add_attn_residual_time
                * scale_factor,
                add_ffn_residual_time=dummy_exec_time._add_ffn_residual_time
                * scale_factor,
                tensor_parallel_communication_time=dummy_exec_time._tensor_parallel_communication_time
                * scale_factor,
                attn_tensor_parallel_allreduce_time=(
                    dummy_exec_time._attn_tensor_parallel_allreduce_time
                    * scale_factor
                    if dummy_exec_time._has_attn_tensor_parallel_allreduce_time
                    else None
                ),
                moe_tensor_parallel_allreduce_time=(
                    dummy_exec_time._moe_tensor_parallel_allreduce_time
                    * scale_factor
                    if dummy_exec_time._has_moe_tensor_parallel_allreduce_time
                    else None
                ),
                tensor_parallel_allgather_time=dummy_exec_time._tensor_parallel_allgather_time
                * scale_factor,
                share_expert_tensor_parallel_allreduce_time=dummy_exec_time._share_expert_tensor_parallel_allreduce_time
                * scale_factor,
                pipeline_parallel_communication_time=dummy_exec_time._pipeline_parallel_communication_time,  # No scaling
                expert_parallel_communication_time=dummy_exec_time._expert_parallel_communication_time
                * scale_factor,
                moe_gating_time=dummy_exec_time._moe_gating_time * scale_factor,
                moe_shuffling_time=dummy_exec_time._moe_shuffling_time * scale_factor,
                schedule_time=dummy_exec_time._schedule_time,  # No scaling
                sampler_e2e_time=dummy_exec_time._sampler_e2e_time,  # No scaling
                prepare_inputs_e2e_time=dummy_exec_time._prepare_inputs_e2e_time,  # No scaling
                process_model_outputs_time=dummy_exec_time._process_model_outputs_time,  # No scaling
                ray_comm_time=dummy_exec_time._ray_comm_time,  # No scaling
                pp_stage_boundary_handoff_time=dummy_exec_time._pp_stage_boundary_handoff_time,  # No scaling
                is_moe=dummy_exec_time._is_moe,
                mlp_layer_up_proj_execution_time=dummy_exec_time._mlp_layer_up_proj_execution_time
                * scale_factor,
                mlp_layer_down_proj_execution_time=dummy_exec_time._mlp_layer_down_proj_execution_time
                * scale_factor,
                mlp_layer_act_execution_time=dummy_exec_time._mlp_layer_act_execution_time
                * scale_factor,
                moe_grouped_gemm_time=dummy_exec_time._moe_grouped_gemm_time
                * scale_factor,
                share_expert_up_proj_time=dummy_exec_time._share_expert_up_proj_time
                * scale_factor,
                share_expert_down_proj_time=dummy_exec_time._share_expert_down_proj_time
                * scale_factor,
                share_expert_act_time=dummy_exec_time._share_expert_act_time
                * scale_factor,
                communication_operator_times=scaled_communication_operator_times,
                layer_ids=normalized_layer_ids,
            )

        logger.debug(
            f"Predicting disaggregated stage execution time: stage_id={stage_id}, "
            f"cluster_type={cluster_type}, num_layers={num_layers}"
        )

        if self._is_zero_token_decode_ffn_ep_barrier(batch, cluster_type):
            logger.debug(
                "[DECODE_FFN] Zero-token EP barrier batch_id=%s returns zero "
                "execution time without predictor lookup.",
                getattr(batch, "id", "N/A"),
            )
            return self._get_zero_decode_ffn_ep_barrier_execution_time(
                num_layers,
                layer_ids=normalized_layer_ids,
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
            logger.warning(
                f"Cluster type mismatch: predictor initialized with {self._cluster_type}, "
                f"but predict_stage_execution_time called with {cluster_type}"
            )

        if not include_ffn:
            return self._predict_attention_only_stage_execution_time(
                batch=batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
                layer_ids=normalized_layer_ids,
            )

        # Phase 2.5: Refactored to use new unified APIs instead of deprecated get_execution_time()
        # Build execution time using cluster-specific operations

        # num_layers is an aggregation factor consumed by ExecutionTime.
        # Predictor must keep op/comm/residual components at single-layer granularity.
        if num_layers != 1:
            logger.debug(
                f"Building disaggregated stage execution time with layer aggregation factor num_layers={num_layers}."
            )

        # Use new unified APIs to build execution time components
        communication_time = self._get_communication_time(
            batch,
            stage_id,
            cluster_type,
            include_attention=include_attention,
        )
        overhead_time = self._get_overhead_time(batch, cluster_type, stage_id)
        overhead_time.pp_stage_boundary_handoff_time = (
            self._get_pp_stage_boundary_handoff_time(batch, stage_id)
        )

        # Build cluster-specific execution time
        if cluster_type == ClusterType.DECODE_ATTN:
            # Attention-only cluster - predict attention time
            cluster_replica_config = self._get_cluster_replica_config(cluster_type)
            architecture_profile = self._get_cluster_model_architecture_profile(cluster_type)
            attention_time = self.predict_attention_layer_time(
                batch, layer_id=layer_id, cluster_type=cluster_type
            )
            # Post-attention layernorm runs on attention cluster in Step3
            mlp_norm_time = self._get_mlp_norm_layer_act_execution_time(batch)
            # Get residual add time (first residual connection after attention)
            add_attn_residual_time = self._get_add_layer_act_execution_time(batch)
            if architecture_profile.skip_decode_attn_residual:
                add_attn_residual_time = 0.0
            # Attention-only cluster
            return ExecutionTime(
                num_layers_per_pipeline_stage=num_layers,
                attention_rope_execution_time=self._predict_one_op_time(
                    "attention_rope_execution_time",
                    attention_time.attention_rope_execution_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),
                attention_kv_cache_save_execution_time=self._predict_one_op_time(
                    "attention_kv_cache_save_execution_time",
                    attention_time.attention_kv_cache_save_execution_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),
                attention_decode_execution_time=self._predict_one_op_time(
                    "attention_decode_execution_time",
                    attention_time.attention_decode_execution_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),
                attention_prefill_execution_time=self._predict_one_op_time(
                    "attention_prefill_execution_time",
                    attention_time.attention_prefill_execution_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),
                attention_layer_pre_proj_execution_time=self._predict_one_op_time(
                    "attention_layer_pre_proj_execution_time",
                    attention_time.attention_layer_pre_proj_execution_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),
                attention_layer_post_proj_execution_time=self._predict_one_op_time(
                    "attention_layer_post_proj_execution_time",
                    attention_time.attention_layer_post_proj_execution_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),
                attn_norm_time=self._predict_one_op_time(
                    "attn_norm_time",
                    attention_time.attn_norm_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),
                mlp_norm_time=self._predict_one_op_time(
                    "mlp_norm_time",
                    mlp_norm_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),
                add_time=0.0,
                add_attn_residual_time=self._predict_one_op_time(
                    "add_attn_residual_time",
                    add_attn_residual_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),  # First residual connection: x + attention(x)
                add_ffn_residual_time=0.0,
                tensor_parallel_communication_time=self._predict_one_op_time(
                    "tensor_parallel_communication_time",
                    communication_time.tensor_parallel_time,
                    batch,
                    stage_id,
                    cluster_type,
                    num_layers,
                ),
                pipeline_parallel_communication_time=communication_time.pipeline_parallel_time,
                schedule_time=overhead_time.schedule_time,
                sampler_e2e_time=overhead_time.sampler_e2e_time,
                prepare_inputs_e2e_time=overhead_time.prepare_inputs_e2e_time,
                process_model_outputs_time=overhead_time.process_model_outputs_time,
                ray_comm_time=overhead_time.ray_comm_time,
                pp_producer_send_path_runtime_time=(
                    overhead_time.pp_producer_send_path_runtime_time
                ),
                pp_receiver_head_runtime_time=(
                    overhead_time.pp_receiver_head_runtime_time
                ),
                pp_prefill_consumer_active_runtime_time=(
                    overhead_time.pp_prefill_consumer_active_runtime_time
                ),
                pp_stage_boundary_residual_runtime_time=(
                    overhead_time.pp_stage_boundary_residual_runtime_time
                ),
                pp_stage_boundary_handoff_time=overhead_time.pp_stage_boundary_handoff_time,
                mlp_layer_up_proj_execution_time=0.0,
                mlp_layer_down_proj_execution_time=0.0,
                mlp_layer_act_execution_time=0.0,
                moe_grouped_gemm_time=0.0,
                expert_parallel_communication_time=0.0,
                moe_gating_time=0.0,
                moe_shuffling_time=0.0,
                is_moe=False,
                layer_ids=normalized_layer_ids,
            )

        elif cluster_type == ClusterType.DECODE_FFN:
            # FFN-only cluster (can be MoE or MLP depending on the current layer)
            cluster_replica_config = (
                admission_replica_config
                or self._get_cluster_replica_config(cluster_type)
            )
            model_config = cluster_replica_config.model_config
            architecture_profile = self._get_cluster_model_architecture_profile(cluster_type)
            is_moe_layer = execution_include_moe

            if is_moe_layer:
                # MoE layer: use MoE operations
                logger.debug(
                    f"[DECODE_FFN] Processing MoE layer {layer_id}: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                lane_workload = self._resolve_layer_lane_workload(
                    batch,
                    cluster_type=cluster_type,
                    layer_id=layer_id,
                )
                logger.debug(
                    "[DECODE_FFN] resolved typed EP lane: ep_id=%s, local_width=%s, "
                    "routed_tokens=%s",
                    lane_workload.ep_id,
                    lane_workload.local_expert_width,
                    lane_workload.routed_token_count,
                )

                moe_time = self.predict_moe_layer_time(
                    batch,
                    layer_id=layer_id,
                    cluster_type=cluster_type,
                    lane_workload=lane_workload,
                    ep_size=admission_ep_size,
                    router_topk=admission_router_topk,
                )
                # In PD-AF flow, post-attention layernorm is accounted on DECODE_ATTN.
                # Keep DECODE_FFN free of this op to avoid double-counting per layer.
                mlp_norm_time = 0.0
                # Get residual add time (second residual connection after MoE)
                add_time = self._get_add_layer_act_execution_time(batch)
                add_attn_residual_time = 0.0
                add_ffn_residual_time = 0.0
                if architecture_profile.residual_add_policy is ResidualAddPolicy.FFN_RESIDUAL_ONLY:
                    add_attn_residual_time = 0.0
                    add_ffn_residual_time = add_time
                    add_time = 0.0
                ep_operator_times = self._predict_named_ep_phase_operator_times(
                    batch=batch,
                    lane_workload=lane_workload,
                    stage_id=stage_id,
                    cluster_type=cluster_type,
                    num_layers=num_layers,
                )
                ep_comm_time = sum(ep_operator_times.values())
                ffn_tp_allgather_time = 0.0
                share_expert_tp_allreduce_time = 0.0
                moe_tp_size = int(cluster_replica_config.moe_tensor_parallel_size)
                moe_tp_allreduce_time = (
                    self._predict_comm_operator_with_context(
                        get_comm_operator("moe_tensor_parallel_allreduce"),
                        CommPayloadContext(
                            batch=batch,
                            model_config=model_config,
                            replica_config=cluster_replica_config,
                            cluster_type=cluster_type,
                            quantization_manager=get_quantization_manager(),
                            lane_workload=lane_workload,
                        ),
                    )
                    if moe_tp_size > 1
                    else 0.0
                )
                if architecture_profile.moe_tensor_parallel_allgather_op and moe_tp_size > 1:
                    # Allgather and shared-expert collectives use the source
                    # batch's pre-routing hidden-state payload. The MoE-TP
                    # all-reduce above uses the same shared domain; only the
                    # expert-parallel all-to-all phases use lane-local
                    # assignments.
                    effective_tokens = batch.get_effective_total_tokens_rounded(
                        cluster_type
                    )
                    data_size_bytes = model_config.embedding_dim * 2 * effective_tokens
                    if data_size_bytes % moe_tp_size != 0:
                        raise ValueError(
                            "Profile-declared FFN TP allgather requires per-device tensor bytes to be "
                            f"divisible by moe_tp_size, got data_size_bytes={data_size_bytes}, "
                            f"moe_tp_size={moe_tp_size}"
                        )
                    per_device_data_size_bytes = data_size_bytes // moe_tp_size
                    quant_manager = get_quantization_manager()
                    allgather_bytes = quant_manager.adjust_tensor_size(
                        "allgather", per_device_data_size_bytes, cluster_type
                    )
                    ffn_tp_allgather_time = self.predict_allgather_time(
                        data_size_bytes=allgather_bytes,
                        num_devices=moe_tp_size,
                        cluster_type=cluster_type,
                        comm_domain="MOE_TP",
                    )
                    if moe_time.share_expert_time > 0:
                        allreduce_bytes = quant_manager.adjust_tensor_size(
                            "allreduce", data_size_bytes, cluster_type
                        )
                        raw_share_expert_tp_allreduce_time = self.predict_allreduce_time(
                            data_size_bytes=allreduce_bytes,
                            num_devices=moe_tp_size,
                            cluster_type=cluster_type,
                            comm_domain="MOE_TP",
                        )
                        share_expert_tp_allreduce_time = raw_share_expert_tp_allreduce_time
                return ExecutionTime(
                    num_layers_per_pipeline_stage=num_layers,
                    mlp_norm_time=self._predict_one_op_time(
                        "mlp_norm_time",
                        mlp_norm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # post_attention_layernorm before MoE
                    add_time=self._predict_one_op_time(
                        "add_time",
                        add_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # Second residual connection: x + moe(x)
                    add_attn_residual_time=self._predict_one_op_time(
                        "add_attn_residual_time",
                        add_attn_residual_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    add_ffn_residual_time=self._predict_one_op_time(
                        "add_ffn_residual_time",
                        add_ffn_residual_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    moe_grouped_gemm_time=self._predict_one_op_time(
                        "moe_grouped_gemm_time",
                        moe_time.moe_grouped_gemm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    expert_parallel_communication_time=self._predict_one_op_time(
                        "expert_parallel_communication_time",
                        ep_comm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    communication_operator_times=CommunicationOperatorTimes(
                        ep_operator_times
                    ),
                    moe_gating_time=self._predict_one_op_time(
                        "moe_gating_time",
                        moe_time.moe_gating_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    moe_shuffling_time=self._predict_one_op_time(
                        "moe_shuffling_time",
                        moe_time.moe_shuffling_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_up_proj_time=self._predict_one_op_time(
                        "share_expert_up_proj_time",
                        moe_time.share_expert_up_proj_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_down_proj_time=self._predict_one_op_time(
                        "share_expert_down_proj_time",
                        moe_time.share_expert_down_proj_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_act_time=self._predict_one_op_time(
                        "share_expert_act_time",
                        moe_time.share_expert_act_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    tensor_parallel_allgather_time=self._predict_one_op_time(
                        "tensor_parallel_allgather_time",
                        ffn_tp_allgather_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_tensor_parallel_allreduce_time=self._predict_one_op_time(
                        "share_expert_tensor_parallel_allreduce_time",
                        share_expert_tp_allreduce_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    tensor_parallel_communication_time=self._predict_one_op_time(
                        "tensor_parallel_communication_time",
                        communication_time.tensor_parallel_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    moe_tensor_parallel_allreduce_time=self._predict_one_op_time(
                        "moe_tensor_parallel_allreduce_time",
                        moe_tp_allreduce_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    pipeline_parallel_communication_time=communication_time.pipeline_parallel_time,
                    schedule_time=overhead_time.schedule_time,
                    sampler_e2e_time=overhead_time.sampler_e2e_time,
                    prepare_inputs_e2e_time=overhead_time.prepare_inputs_e2e_time,
                    process_model_outputs_time=overhead_time.process_model_outputs_time,
                    ray_comm_time=overhead_time.ray_comm_time,
                    pp_producer_send_path_runtime_time=(
                        overhead_time.pp_producer_send_path_runtime_time
                    ),
                    pp_receiver_head_runtime_time=(
                        overhead_time.pp_receiver_head_runtime_time
                    ),
                    pp_prefill_consumer_active_runtime_time=(
                        overhead_time.pp_prefill_consumer_active_runtime_time
                    ),
                    pp_stage_boundary_residual_runtime_time=(
                        overhead_time.pp_stage_boundary_residual_runtime_time
                    ),
                    pp_stage_boundary_handoff_time=overhead_time.pp_stage_boundary_handoff_time,
                    is_moe=True,
                    **self._get_zero_attn_params(),
                    layer_ids=normalized_layer_ids,
                )
            else:
                # Dense layer: use MLP operations
                logger.debug(
                    f"[DECODE_FFN] Processing dense layer {layer_id}: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                mlp_time = self.predict_mlp_layer_time(
                    batch, layer_id=layer_id, cluster_type=cluster_type
                )
                # In PD-AF flow, post-attention layernorm is accounted on DECODE_ATTN.
                # Keep DECODE_FFN free of this op to avoid double-counting per layer.
                mlp_norm_time = 0.0
                # Get residual add time (second residual connection after MLP)
                add_time = self._get_add_layer_act_execution_time(batch)
                add_attn_residual_time = 0.0
                add_ffn_residual_time = 0.0
                if architecture_profile.residual_add_policy is ResidualAddPolicy.FFN_RESIDUAL_ONLY:
                    add_attn_residual_time = 0.0
                    add_ffn_residual_time = add_time
                    add_time = 0.0
                return ExecutionTime(
                    num_layers_per_pipeline_stage=num_layers,
                    mlp_norm_time=self._predict_one_op_time(
                        "mlp_norm_time",
                        mlp_norm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # post_attention_layernorm before MLP
                    add_time=self._predict_one_op_time(
                        "add_time",
                        add_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # Second residual connection: x + mlp(x)
                    add_attn_residual_time=self._predict_one_op_time(
                        "add_attn_residual_time",
                        add_attn_residual_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    add_ffn_residual_time=self._predict_one_op_time(
                        "add_ffn_residual_time",
                        add_ffn_residual_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    mlp_layer_up_proj_execution_time=self._predict_one_op_time(
                        "mlp_layer_up_proj_execution_time",
                        mlp_time.mlp_layer_up_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    mlp_layer_down_proj_execution_time=self._predict_one_op_time(
                        "mlp_layer_down_proj_execution_time",
                        mlp_time.mlp_layer_down_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    mlp_layer_act_execution_time=self._predict_one_op_time(
                        "mlp_layer_act_execution_time",
                        mlp_time.mlp_layer_act_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    tensor_parallel_communication_time=self._predict_one_op_time(
                        "tensor_parallel_communication_time",
                        communication_time.tensor_parallel_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    pipeline_parallel_communication_time=communication_time.pipeline_parallel_time,
                    schedule_time=overhead_time.schedule_time,
                    sampler_e2e_time=overhead_time.sampler_e2e_time,
                    prepare_inputs_e2e_time=overhead_time.prepare_inputs_e2e_time,
                    process_model_outputs_time=overhead_time.process_model_outputs_time,
                    ray_comm_time=overhead_time.ray_comm_time,
                    pp_producer_send_path_runtime_time=(
                        overhead_time.pp_producer_send_path_runtime_time
                    ),
                    pp_receiver_head_runtime_time=(
                        overhead_time.pp_receiver_head_runtime_time
                    ),
                    pp_prefill_consumer_active_runtime_time=(
                        overhead_time.pp_prefill_consumer_active_runtime_time
                    ),
                    pp_stage_boundary_residual_runtime_time=(
                        overhead_time.pp_stage_boundary_residual_runtime_time
                    ),
                    pp_stage_boundary_handoff_time=overhead_time.pp_stage_boundary_handoff_time,
                    **self._get_zero_attn_params(),
                    **self._get_zero_moe_params(),
                    layer_ids=normalized_layer_ids,
                )

        elif cluster_type == ClusterType.DECODE:
            # Unified DECODE cluster (PD-disaggregation mode)
            # Handles both dense models (MLP) and MoE models
            # For dense models: attention + MLP
            # For MoE models: attention + MoE
            attention_execution_params = self._get_zero_attn_params()
            if include_attention:
                attention_time = self.predict_attention_layer_time(
                    batch, layer_id=layer_id, cluster_type=cluster_type
                )
                attention_execution_params = {
                    "attention_rope_execution_time": self._predict_one_op_time(
                        "attention_rope_execution_time",
                        attention_time.attention_rope_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    "attention_kv_cache_save_execution_time": self._predict_one_op_time(
                        "attention_kv_cache_save_execution_time",
                        attention_time.attention_kv_cache_save_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    "attention_decode_execution_time": self._predict_one_op_time(
                        "attention_decode_execution_time",
                        attention_time.attention_decode_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    "attention_prefill_execution_time": self._predict_one_op_time(
                        "attention_prefill_execution_time",
                        attention_time.attention_prefill_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    "attention_layer_pre_proj_execution_time": self._predict_one_op_time(
                        "attention_layer_pre_proj_execution_time",
                        attention_time.attention_layer_pre_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    "attention_layer_post_proj_execution_time": self._predict_one_op_time(
                        "attention_layer_post_proj_execution_time",
                        attention_time.attention_layer_post_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    "attn_norm_time": self._predict_one_op_time(
                        "attn_norm_time",
                        attention_time.attn_norm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                }

            # Check if this is a MoE model or dense model
            # Use model_config.is_moe for MoE detection - NOT parallelism settings
            cluster_replica_config = (
                admission_replica_config
                or self._get_cluster_replica_config(cluster_type)
            )
            is_moe_model = execution_include_moe
            if not include_attention and not is_moe_model:
                raise ValueError(
                    "Post-attention-only unified DECODE prediction requires a MoE layer"
                )

            if is_moe_model:
                # MoE model: use MoE operations
                logger.debug(
                    f"[DECODE] Processing MoE model: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                lane_workload = self._resolve_layer_lane_workload(
                    batch,
                    cluster_type=cluster_type,
                    layer_id=layer_id,
                )

                moe_time = self.predict_moe_layer_time(
                    batch,
                    layer_id=layer_id,
                    cluster_type=cluster_type,
                    lane_workload=lane_workload,
                    ep_size=admission_ep_size,
                    router_topk=admission_router_topk,
                )
                # Get post_attention_layernorm time (runs before MoE)
                mlp_norm_time = self._get_mlp_norm_layer_act_execution_time(batch)
                # Get residual add time (both residual connections)
                add_time = self._get_add_layer_act_execution_time(batch)
                ep_operator_times = self._predict_named_ep_phase_operator_times(
                    batch=batch,
                    lane_workload=lane_workload,
                    stage_id=stage_id,
                    cluster_type=cluster_type,
                    num_layers=num_layers,
                )
                ep_comm_time = sum(ep_operator_times.values())

                # Calculate MoE TP allreduce time using moe_tensor_parallel_size
                # (communication_time.tensor_parallel_time uses attn_tensor_parallel_size,
                #  so we need a separate calculation for MoE TP allreduce)
                moe_tp_size = int(cluster_replica_config.moe_tensor_parallel_size)
                moe_tp_allreduce_time = (
                    self._predict_comm_operator_with_context(
                        get_comm_operator("moe_tensor_parallel_allreduce"),
                        CommPayloadContext(
                            batch=batch,
                            model_config=cluster_replica_config.model_config,
                            replica_config=cluster_replica_config,
                            cluster_type=cluster_type,
                            quantization_manager=get_quantization_manager(),
                            lane_workload=lane_workload,
                        ),
                    )
                    if moe_tp_size > 1
                    else 0.0
                )

                return ExecutionTime(
                    num_layers_per_pipeline_stage=num_layers,
                    **attention_execution_params,
                    mlp_norm_time=self._predict_one_op_time(
                        "mlp_norm_time",
                        mlp_norm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # post_attention_layernorm before MoE
                    add_time=self._predict_one_op_time(
                        "add_time",
                        add_time * 2,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # Both residual connections: x + attention(x) + x + moe(x)
                    moe_grouped_gemm_time=self._predict_one_op_time(
                        "moe_grouped_gemm_time",
                        moe_time.moe_grouped_gemm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    expert_parallel_communication_time=self._predict_one_op_time(
                        "expert_parallel_communication_time",
                        ep_comm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    communication_operator_times=CommunicationOperatorTimes(
                        ep_operator_times
                    ),
                    moe_gating_time=self._predict_one_op_time(
                        "moe_gating_time",
                        moe_time.moe_gating_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    moe_shuffling_time=self._predict_one_op_time(
                        "moe_shuffling_time",
                        moe_time.moe_shuffling_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_up_proj_time=self._predict_one_op_time(
                        "share_expert_up_proj_time",
                        moe_time.share_expert_up_proj_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_down_proj_time=self._predict_one_op_time(
                        "share_expert_down_proj_time",
                        moe_time.share_expert_down_proj_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_act_time=self._predict_one_op_time(
                        "share_expert_act_time",
                        moe_time.share_expert_act_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    tensor_parallel_communication_time=self._predict_one_op_time(
                        "tensor_parallel_communication_time",
                        (
                            communication_time.tensor_parallel_time
                            if include_attention
                            else 0.0
                        ),
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attn_tensor_parallel_allreduce_time=self._predict_one_op_time(
                        "attn_tensor_parallel_allreduce_time",
                        (
                            communication_time.tensor_parallel_time
                            if include_attention
                            else 0.0
                        ),
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    moe_tensor_parallel_allreduce_time=self._predict_one_op_time(
                        "moe_tensor_parallel_allreduce_time",
                        moe_tp_allreduce_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                pipeline_parallel_communication_time=communication_time.pipeline_parallel_time,  # No scaling
                    schedule_time=overhead_time.schedule_time,
                    sampler_e2e_time=overhead_time.sampler_e2e_time,
                    prepare_inputs_e2e_time=overhead_time.prepare_inputs_e2e_time,
                    process_model_outputs_time=overhead_time.process_model_outputs_time,
                    ray_comm_time=overhead_time.ray_comm_time,
                    pp_producer_send_path_runtime_time=(
                        overhead_time.pp_producer_send_path_runtime_time
                    ),
                    pp_receiver_head_runtime_time=(
                        overhead_time.pp_receiver_head_runtime_time
                    ),
                    pp_prefill_consumer_active_runtime_time=(
                        overhead_time.pp_prefill_consumer_active_runtime_time
                    ),
                    pp_stage_boundary_residual_runtime_time=(
                        overhead_time.pp_stage_boundary_residual_runtime_time
                    ),
                    pp_stage_boundary_handoff_time=overhead_time.pp_stage_boundary_handoff_time,
                    is_moe=True,
                    layer_ids=normalized_layer_ids,
                )
            else:
                # Dense model: use MLP operations
                logger.debug(
                    f"[DECODE] Processing dense model: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                mlp_time = self.predict_mlp_layer_time(
                    batch, layer_id=layer_id, cluster_type=cluster_type
                )
                # Get residual add time (both residual connections)
                add_time = self._get_add_layer_act_execution_time(batch)
                return ExecutionTime(
                    num_layers_per_pipeline_stage=num_layers,
                    **attention_execution_params,
                    mlp_norm_time=self._predict_one_op_time(
                        "mlp_norm_time",
                        mlp_time.mlp_norm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    add_time=self._predict_one_op_time(
                        "add_time",
                        add_time * 2,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # Both residual connections: x + attention(x) + x + mlp(x)
                    mlp_layer_up_proj_execution_time=self._predict_one_op_time(
                        "mlp_layer_up_proj_execution_time",
                        mlp_time.mlp_layer_up_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    mlp_layer_down_proj_execution_time=self._predict_one_op_time(
                        "mlp_layer_down_proj_execution_time",
                        mlp_time.mlp_layer_down_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    mlp_layer_act_execution_time=self._predict_one_op_time(
                        "mlp_layer_act_execution_time",
                        mlp_time.mlp_layer_act_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    tensor_parallel_communication_time=self._predict_one_op_time(
                        "tensor_parallel_communication_time",
                        communication_time.tensor_parallel_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    pipeline_parallel_communication_time=communication_time.pipeline_parallel_time,  # No scaling
                    schedule_time=overhead_time.schedule_time,
                    sampler_e2e_time=overhead_time.sampler_e2e_time,
                    prepare_inputs_e2e_time=overhead_time.prepare_inputs_e2e_time,
                    process_model_outputs_time=overhead_time.process_model_outputs_time,
                    ray_comm_time=overhead_time.ray_comm_time,
                    pp_producer_send_path_runtime_time=(
                        overhead_time.pp_producer_send_path_runtime_time
                    ),
                    pp_receiver_head_runtime_time=(
                        overhead_time.pp_receiver_head_runtime_time
                    ),
                    pp_prefill_consumer_active_runtime_time=(
                        overhead_time.pp_prefill_consumer_active_runtime_time
                    ),
                    pp_stage_boundary_residual_runtime_time=(
                        overhead_time.pp_stage_boundary_residual_runtime_time
                    ),
                    pp_stage_boundary_handoff_time=overhead_time.pp_stage_boundary_handoff_time,
                    **self._get_zero_moe_params(),
                    layer_ids=normalized_layer_ids,
                )

        elif cluster_type == ClusterType.PREFILL:
            # Full model (attention + FFN) - predict both attention and FFN time
            # FFN can be MoE (for MoE models) or MLP (for dense models)
            attention_time = (
                self.predict_attention_layer_time(
                    batch,
                    layer_id=layer_id,
                    cluster_type=cluster_type,
                )
                if include_attention
                else AttentionTime()
            )

            # Check whether the requested layer is MoE or dense.  Step3 models
            # are mixed-layer: the model-level ``is_moe`` flag is true even for
            # dense boundary layers (for example layers 0-3 and 60).  The
            # layer-specific predicate must therefore control the PREFILL FFN
            # path just as it does for DECODE_FFN.
            cluster_replica_config = (
                admission_replica_config
                or self._get_cluster_replica_config(cluster_type)
            )
            model_config = cluster_replica_config.model_config
            is_moe_model = execution_include_moe

            if is_moe_model:
                # MoE model: use MoE operations for FFN
                logger.debug(
                    f"[PREFILL] Processing MoE model: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                lane_workload = self._resolve_layer_lane_workload(
                    batch,
                    cluster_type=cluster_type,
                    layer_id=layer_id,
                )

                moe_time = self.predict_moe_layer_time(
                    batch,
                    layer_id=layer_id,
                    cluster_type=cluster_type,
                    lane_workload=lane_workload,
                    ep_size=admission_ep_size,
                    router_topk=admission_router_topk,
                )

                # Get post_attention_layernorm time (runs before MoE)
                mlp_norm_time = self._get_mlp_norm_layer_act_execution_time(batch)
                # Get residual add time (both residual connections in full model)
                add_time = self._get_add_layer_act_execution_time(batch)
                ep_operator_times = self._predict_named_ep_phase_operator_times(
                    batch=batch,
                    lane_workload=lane_workload,
                    stage_id=stage_id,
                    cluster_type=cluster_type,
                    num_layers=num_layers,
                )
                ep_comm_time = sum(ep_operator_times.values())
                (
                    dp_input_allreduce_time,
                    dp_output_allreduce_time,
                ) = self.predict_dp_moe_allreduce_times(batch, cluster_type)

                # Keep PREFILL MoE TP communication composition aligned with the
                # shared pre-routing hidden-state contract. The source batch's
                # effective token count drives MoE-TP all-reduce, allgather, and
                # shared-expert collectives; only EP all-to-all uses lane-local
                # assignments.
                moe_tp_size = int(cluster_replica_config.moe_tensor_parallel_size)
                moe_tp_allreduce_time = (
                    self._predict_comm_operator_with_context(
                        get_comm_operator("moe_tensor_parallel_allreduce"),
                        CommPayloadContext(
                            batch=batch,
                            model_config=model_config,
                            replica_config=cluster_replica_config,
                            cluster_type=cluster_type,
                            quantization_manager=get_quantization_manager(),
                            lane_workload=lane_workload,
                        ),
                    )
                    if moe_tp_size > 1
                    else 0.0
                )
                ffn_tp_allgather_time = 0.0
                share_expert_tp_allreduce_time = 0.0
                if moe_tp_size > 1:
                    # Use compute-effective tokens. AFD paths already include CUDA Graph
                    # padding in metadata; non-CUDA-Graph paths keep exact token counts.
                    effective_tokens = batch.get_effective_total_tokens_rounded(cluster_type)
                    data_size_bytes = (
                        cluster_replica_config.model_config.embedding_dim
                        * 2
                        * effective_tokens
                    )
                    if data_size_bytes % moe_tp_size != 0:
                        raise ValueError(
                            "Profile-declared FFN TP allgather requires per-device tensor bytes to be "
                            f"divisible by moe_tp_size, got data_size_bytes={data_size_bytes}, "
                            f"moe_tp_size={moe_tp_size}"
                        )
                    per_device_data_size_bytes = data_size_bytes // moe_tp_size
                    quant_manager = get_quantization_manager()
                    architecture_profile = self._resolve_model_architecture_profile_for_config(
                        cluster_replica_config.model_config
                    )
                    if architecture_profile.moe_tensor_parallel_allgather_op:
                        allgather_bytes = quant_manager.adjust_tensor_size(
                            "allgather", per_device_data_size_bytes, cluster_type
                        )
                        ffn_tp_allgather_time = self.predict_allgather_time(
                            data_size_bytes=allgather_bytes,
                            num_devices=moe_tp_size,
                            cluster_type=cluster_type,
                            comm_domain="MOE_TP",
                        )
                        if (
                            moe_time.share_expert_up_proj_time
                            + moe_time.share_expert_down_proj_time
                            + moe_time.share_expert_act_time
                            > 0
                        ):
                            moe_tp_allreduce_bytes = quant_manager.adjust_tensor_size(
                                "allreduce", data_size_bytes, cluster_type
                            )
                            raw_share_expert_tp_allreduce_time = self.predict_allreduce_time(
                                data_size_bytes=moe_tp_allreduce_bytes,
                                num_devices=moe_tp_size,
                                cluster_type=cluster_type,
                                comm_domain="MOE_TP",
                            )
                            share_expert_tp_allreduce_time = raw_share_expert_tp_allreduce_time

                # Build ExecutionTime object for MoE model
                exec_time = ExecutionTime(
                    num_layers_per_pipeline_stage=num_layers,
                    attention_rope_execution_time=self._predict_one_op_time(
                        "attention_rope_execution_time",
                        attention_time.attention_rope_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_kv_cache_save_execution_time=self._predict_one_op_time(
                        "attention_kv_cache_save_execution_time",
                        attention_time.attention_kv_cache_save_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_decode_execution_time=self._predict_one_op_time(
                        "attention_decode_execution_time",
                        attention_time.attention_decode_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_prefill_execution_time=self._predict_one_op_time(
                        "attention_prefill_execution_time",
                        attention_time.attention_prefill_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_layer_pre_proj_execution_time=self._predict_one_op_time(
                        "attention_layer_pre_proj_execution_time",
                        attention_time.attention_layer_pre_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_layer_post_proj_execution_time=self._predict_one_op_time(
                        "attention_layer_post_proj_execution_time",
                        attention_time.attention_layer_post_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attn_norm_time=self._predict_one_op_time(
                        "attn_norm_time",
                        attention_time.attn_norm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    mlp_norm_time=self._predict_one_op_time(
                        "mlp_norm_time",
                        mlp_norm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # post_attention_layernorm before MoE
                    add_time=self._predict_one_op_time(
                        "add_time",
                        add_time * 2,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # Both residual connections: x + attention(x) + x + moe(x)
                    moe_grouped_gemm_time=self._predict_one_op_time(
                        "moe_grouped_gemm_time",
                        moe_time.moe_grouped_gemm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    expert_parallel_communication_time=self._predict_one_op_time(
                        "expert_parallel_communication_time",
                        ep_comm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    communication_operator_times=CommunicationOperatorTimes(
                        ep_operator_times
                    ),
                    moe_gating_time=self._predict_one_op_time(
                        "moe_gating_time",
                        moe_time.moe_gating_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    moe_shuffling_time=self._predict_one_op_time(
                        "moe_shuffling_time",
                        moe_time.moe_shuffling_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_up_proj_time=self._predict_one_op_time(
                        "share_expert_up_proj_time",
                        moe_time.share_expert_up_proj_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_down_proj_time=self._predict_one_op_time(
                        "share_expert_down_proj_time",
                        moe_time.share_expert_down_proj_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_act_time=self._predict_one_op_time(
                        "share_expert_act_time",
                        moe_time.share_expert_act_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    tensor_parallel_communication_time=self._predict_one_op_time(
                        "tensor_parallel_communication_time",
                        communication_time.tensor_parallel_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attn_tensor_parallel_allreduce_time=self._predict_one_op_time(
                        "attn_tensor_parallel_allreduce_time",
                        communication_time.tensor_parallel_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    moe_tensor_parallel_allreduce_time=self._predict_one_op_time(
                        "moe_tensor_parallel_allreduce_time",
                        moe_tp_allreduce_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    tensor_parallel_allgather_time=self._predict_one_op_time(
                        "tensor_parallel_allgather_time",
                        ffn_tp_allgather_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    share_expert_tensor_parallel_allreduce_time=self._predict_one_op_time(
                        "share_expert_tensor_parallel_allreduce_time",
                        share_expert_tp_allreduce_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    dp_input_allreduce_time=self._predict_one_op_time(
                        "dp_input_allreduce_time",
                        dp_input_allreduce_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    dp_output_allreduce_time=self._predict_one_op_time(
                        "dp_output_allreduce_time",
                        dp_output_allreduce_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    pipeline_parallel_communication_time=communication_time.pipeline_parallel_time,  # No scaling
                    schedule_time=overhead_time.schedule_time,
                    sampler_e2e_time=overhead_time.sampler_e2e_time,
                    prepare_inputs_e2e_time=overhead_time.prepare_inputs_e2e_time,
                    process_model_outputs_time=overhead_time.process_model_outputs_time,
                    ray_comm_time=overhead_time.ray_comm_time,
                    pp_producer_send_path_runtime_time=(
                        overhead_time.pp_producer_send_path_runtime_time
                    ),
                    pp_receiver_head_runtime_time=(
                        overhead_time.pp_receiver_head_runtime_time
                    ),
                    pp_prefill_consumer_active_runtime_time=(
                        overhead_time.pp_prefill_consumer_active_runtime_time
                    ),
                    pp_stage_boundary_residual_runtime_time=(
                        overhead_time.pp_stage_boundary_residual_runtime_time
                    ),
                    pp_stage_boundary_handoff_time=overhead_time.pp_stage_boundary_handoff_time,
                    is_moe=True,
                    layer_ids=normalized_layer_ids,
                )
            else:
                # Dense model: use MLP operations for FFN
                logger.debug(
                    f"[PREFILL] Processing dense model: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                mlp_time = self.predict_mlp_layer_time(
                    batch, layer_id=layer_id, cluster_type=cluster_type
                )
                # Get residual add time (both residual connections)
                add_time = self._get_add_layer_act_execution_time(batch)

                # Build ExecutionTime object for dense model
                exec_time = ExecutionTime(
                    num_layers_per_pipeline_stage=num_layers,
                    attention_rope_execution_time=self._predict_one_op_time(
                        "attention_rope_execution_time",
                        attention_time.attention_rope_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_kv_cache_save_execution_time=self._predict_one_op_time(
                        "attention_kv_cache_save_execution_time",
                        attention_time.attention_kv_cache_save_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_decode_execution_time=self._predict_one_op_time(
                        "attention_decode_execution_time",
                        attention_time.attention_decode_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_prefill_execution_time=self._predict_one_op_time(
                        "attention_prefill_execution_time",
                        attention_time.attention_prefill_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_layer_pre_proj_execution_time=self._predict_one_op_time(
                        "attention_layer_pre_proj_execution_time",
                        attention_time.attention_layer_pre_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attention_layer_post_proj_execution_time=self._predict_one_op_time(
                        "attention_layer_post_proj_execution_time",
                        attention_time.attention_layer_post_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    attn_norm_time=self._predict_one_op_time(
                        "attn_norm_time",
                        attention_time.attn_norm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    mlp_norm_time=self._predict_one_op_time(
                        "mlp_norm_time",
                        mlp_time.mlp_norm_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # post_attention_layernorm before MLP
                    add_time=self._predict_one_op_time(
                        "add_time",
                        add_time * 2,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),  # Both residual connections: x + attention(x) + x + mlp(x)
                    mlp_layer_up_proj_execution_time=self._predict_one_op_time(
                        "mlp_layer_up_proj_execution_time",
                        mlp_time.mlp_layer_up_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    mlp_layer_down_proj_execution_time=self._predict_one_op_time(
                        "mlp_layer_down_proj_execution_time",
                        mlp_time.mlp_layer_down_proj_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    mlp_layer_act_execution_time=self._predict_one_op_time(
                        "mlp_layer_act_execution_time",
                        mlp_time.mlp_layer_act_execution_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    tensor_parallel_communication_time=self._predict_one_op_time(
                        "tensor_parallel_communication_time",
                        communication_time.tensor_parallel_time,
                        batch,
                        stage_id,
                        cluster_type,
                        num_layers,
                    ),
                    pipeline_parallel_communication_time=communication_time.pipeline_parallel_time,  # No scaling
                    schedule_time=overhead_time.schedule_time,
                    sampler_e2e_time=overhead_time.sampler_e2e_time,
                    prepare_inputs_e2e_time=overhead_time.prepare_inputs_e2e_time,
                    process_model_outputs_time=overhead_time.process_model_outputs_time,
                    ray_comm_time=overhead_time.ray_comm_time,
                    pp_producer_send_path_runtime_time=(
                        overhead_time.pp_producer_send_path_runtime_time
                    ),
                    pp_receiver_head_runtime_time=(
                        overhead_time.pp_receiver_head_runtime_time
                    ),
                    pp_prefill_consumer_active_runtime_time=(
                        overhead_time.pp_prefill_consumer_active_runtime_time
                    ),
                    pp_stage_boundary_residual_runtime_time=(
                        overhead_time.pp_stage_boundary_residual_runtime_time
                    ),
                    pp_stage_boundary_handoff_time=overhead_time.pp_stage_boundary_handoff_time,
                    **self._get_zero_moe_params(),
                    layer_ids=normalized_layer_ids,
                )

            # High-level batch execution summary for PREFILL cluster
            logger.info(
                f"[OP-TRACE][PREFILL][SUMMARY] batch_id={batch.id}, stage_id={stage_id}, "
                f"num_layers={num_layers}, num_tokens={batch.total_num_tokens}, batch_size={len(batch.requests)}, "
                f"is_moe_model={is_moe_model}"
            )
            if is_moe_model:
                logger.info(
                    f"[OP-TRACE][PREFILL][SUMMARY][TIMES] batch_id={batch.id}, "
                    f"total_time_ms={exec_time.total_time * 1000:.6f}, "
                    f"model_time_ms={exec_time.model_time * 1000:.6f}, "
                    f"attention_time_ms={attention_time.total_time() * num_layers:.6f}, "
                    f"moe_time_ms={moe_time.total_time() * num_layers:.6f}, "
                    f"tp_comm_time_ms={communication_time.tensor_parallel_time * num_layers:.6f}, "
                    f"pp_comm_time_ms={communication_time.pipeline_parallel_time:.6f}"
                )
            else:
                logger.info(
                    f"[OP-TRACE][PREFILL][SUMMARY][TIMES] batch_id={batch.id}, "
                    f"total_time_ms={exec_time.total_time * 1000:.6f}, "
                    f"model_time_ms={exec_time.model_time * 1000:.6f}, "
                    f"attention_time_ms={attention_time.total_time() * num_layers:.6f}, "
                    f"mlp_time_ms={mlp_time.total_time() * num_layers:.6f}, "
                    f"tp_comm_time_ms={communication_time.tensor_parallel_time * num_layers:.6f}, "
                    f"pp_comm_time_ms={communication_time.pipeline_parallel_time:.6f}"
                )
            logger.info(
                f"[OP-TRACE][PREFILL][SUMMARY][OVERHEAD] batch_id={batch.id}, "
                f"schedule_time_ms={overhead_time.schedule_time:.6f}, "
                f"sampler_time_ms={overhead_time.sampler_e2e_time:.6f}, "
                f"prepare_inputs_time_ms={overhead_time.prepare_inputs_e2e_time:.6f}, "
                f"process_outputs_time_ms={overhead_time.process_model_outputs_time:.6f}, "
                f"ray_comm_time_ms={overhead_time.ray_comm_time:.6f}"
            )

            return exec_time

        raise ValueError(f"Unsupported cluster_type: {cluster_type}")
