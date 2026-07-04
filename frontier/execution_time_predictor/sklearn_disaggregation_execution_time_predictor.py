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
from frontier.entities.time_components import OverheadTime
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.types import ClusterType
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)

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

        # Define cluster types that require MoE routing details
        # DECODE is included for PD-disaggregation mode where DECODE handles both attention + MoE
        moe_cluster_types = {ClusterType.PREFILL, ClusterType.DECODE_FFN, ClusterType.DECODE}
        current_cluster_types = {cluster_type} if cluster_type else moe_cluster_types

        # Calculate routing details for each relevant cluster type
        for target_cluster_type in current_cluster_types.intersection(
            moe_cluster_types
        ):
            routing_details: Dict[int, Dict[int, Dict[int, float]]] = (
                self._simulate_and_store_routing(target_cluster_type)
            )

            if target_cluster_type == ClusterType.PREFILL:
                self._prefill_routing_details = routing_details
                del self._decode_ffn_routing_details
                del self._decode_routing_details
            elif target_cluster_type == ClusterType.DECODE_FFN:
                self._decode_ffn_routing_details = routing_details
                del self._prefill_routing_details
                del self._decode_routing_details
            elif target_cluster_type == ClusterType.DECODE:
                self._decode_routing_details = routing_details
                del self._prefill_routing_details
                del self._decode_ffn_routing_details

        # Initialize empty routing details for clusters that don't need MoE routing
        if cluster_type == ClusterType.DECODE_ATTN:
            logger.debug(
                "DECODE_ATTN predictor skipping MoE routing calculation (not needed)"
            )
            del self._prefill_routing_details
            del self._decode_ffn_routing_details
            del self._decode_routing_details
            # self._prefill_routing_details = {}
            # self._decode_ffn_routing_details = {}

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
        np.random.seed(self._distribution_seed + replica_id * 1000 + layer_id)

        if self._workload_distribution_type == WorkloadDistributionType.BALANCED:
            # Balanced distribution: each expert gets equal allocation
            allocation_ratios = [1.0 / total_expert_num] * total_expert_num

        elif self._workload_distribution_type == WorkloadDistributionType.RANDOM:
            # Random distribution: generate random weights and normalize
            random_weights = np.random.uniform(0.1, 1.0, total_expert_num)
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
            num_tokens_or_allocation: Either an integer token count or
                a dict mapping expert_id -> token_count.
            batch: Optional batch context for decode-phase-only calibration.

        Returns:
            Predicted grouped GEMM execution time in milliseconds.
        """
        if batch is None:
            return super()._get_grouped_gemm_time(num_tokens_or_allocation)
        return super()._get_grouped_gemm_time(
            num_tokens_or_allocation,
            batch=batch,
        )

    def _calculate_expert_token_allocation(
        self, batch: Batch, cluster_type: ClusterType, layer_id: int
    ) -> Dict[int, int]:
        """
        Calculate actual token allocation for each expert based on batch and routing details.

        Args:
            batch: The batch being processed
            cluster_type: Type of cluster (PREFILL, DECODE_FFN, or DECODE)
            layer_id: Layer ID within the pipeline stage

        Returns:
            Dictionary mapping global_expert_id to number of tokens
        """
        # Get routing details for the appropriate cluster
        if cluster_type == ClusterType.PREFILL:
            routing_details = self._prefill_routing_details
            cluster_replica_config = self._cluster_config.prefill_replica_config
        elif cluster_type == ClusterType.DECODE_FFN:
            routing_details = self._decode_ffn_routing_details
            cluster_replica_config = self._cluster_config.decode_ffn_replica_config
        elif cluster_type == ClusterType.DECODE:
            # Unified DECODE cluster in PD-disaggregation mode
            routing_details = self._decode_routing_details
            cluster_replica_config = getattr(
                self._cluster_config, "decode_replica_config", self._replica_config
            )
        else:
            raise ValueError(
                f"Unsupported cluster_type for MoE calculation: {cluster_type}"
            )

        # Check if routing details are available
        assert (
            routing_details is not None
        ), f"Routing details not available for {cluster_type}"

        replica_id = batch.replica_id

        # Get total tokens in batch and multiply by top_k to get total expert tokens
        total_batch_tokens = batch.total_num_tokens
        router_topk = cluster_replica_config.router_topk
        total_expert_tokens = total_batch_tokens * router_topk

        # Get allocation ratios for this replica and layer
        if replica_id not in routing_details:
            logger.error(
                f"Replica {replica_id} not found in routing details for {cluster_type}"
            )
            raise KeyError(
                f"Replica {replica_id} not found in routing details for {cluster_type}"
            )

        if layer_id not in routing_details[replica_id]:
            logger.error(
                f"Layer {layer_id} not found in routing details for replica {replica_id} in {cluster_type}"
            )
            raise KeyError(
                f"Layer {layer_id} not found in routing details for replica {replica_id} in {cluster_type}"
            )

        expert_ratios = routing_details[replica_id][layer_id]

        # Calculate actual token allocation for each expert using proportional allocation
        # with remainder distribution to ensure token conservation.
        # 
        # Problem: Simple int(total_expert_tokens * ratio) causes token loss due to truncation.
        # Example: 4 tokens, 8 experts with uniform ratio 0.125 each
        #   - int(4 * 0.125) = 0 for all experts → total = 0, expected = 4
        # 
        # Solution: Use largest remainder method (Hare quota) for fair distribution:
        # 1. Calculate base allocation (floor) for each expert
        # 2. Distribute remaining tokens to experts with largest fractional parts
        
        # Step 1: Calculate base allocation and fractional parts
        expert_base_allocation = {}
        expert_fractional_parts = {}
        total_base_allocated = 0
        
        for global_expert_id, allocation_ratio in expert_ratios.items():
            exact_allocation = total_expert_tokens * allocation_ratio
            base_allocation = int(exact_allocation)
            fractional_part = exact_allocation - base_allocation
            
            expert_base_allocation[global_expert_id] = base_allocation
            expert_fractional_parts[global_expert_id] = fractional_part
            total_base_allocated += base_allocation
        
        # Step 2: Distribute remaining tokens to experts with largest fractional parts
        remaining_tokens = total_expert_tokens - total_base_allocated
        
        if remaining_tokens > 0:
            # Sort experts by fractional part (descending) to distribute remaining tokens fairly
            sorted_experts = sorted(
                expert_fractional_parts.keys(),
                key=lambda eid: expert_fractional_parts[eid],
                reverse=True
            )
            
            # Distribute remaining tokens one by one to experts with largest fractional parts
            for i in range(remaining_tokens):
                expert_id = sorted_experts[i % len(sorted_experts)]
                expert_base_allocation[expert_id] += 1
        
        # Verify token conservation
        total_allocated = sum(expert_base_allocation.values())
        if total_allocated != total_expert_tokens:
            logger.warning(
                f"Token allocation mismatch after distribution: allocated={total_allocated}, "
                f"expected={total_expert_tokens}. This should not happen."
            )
        
        return expert_base_allocation

    def _get_dummy_execution_time_for_cluster(
        self, batch: Batch, pipeline_stage: int, cluster_type: ClusterType = None
    ) -> ExecutionTime:
        """Return cluster-specific dummy ExecutionTime object."""
        if cluster_type is None:
            raise ValueError(
                "cluster_type cannot be None for cluster-specific dummy execution time"
            )

        base_time = self._dummy_execution_time
        # PD+AF dummy-mode calibration: DECODE_FFN can otherwise appear far slower than
        # DECODE_ATTN because its modeled MoE path hits additional scaling factors.
        # Keep dummy-mode Te within the same order of magnitude as Ta for validation.
        if cluster_type == ClusterType.DECODE_FFN:
            base_time *= 0.02

        cluster_replica_config = self._get_cluster_replica_config(cluster_type)
        # Use model_config.is_moe for MoE detection - NOT parallelism settings
        # A MoE model remains MoE regardless of moe_expert_parallel_size
        model_config = cluster_replica_config.model_config
        is_moe_model = model_config is not None and model_config.is_moe
        architecture_profile = self._get_cluster_model_architecture_profile(cluster_type)
        share_expert_enabled = (
            is_moe_model
            and cluster_replica_config.model_config is not None
            and cluster_replica_config.model_config.supports_share_expert()
        )
        share_expert_time = base_time if share_expert_enabled else 0.0
        if cluster_type == ClusterType.DECODE_FFN:
            tp_size = cluster_replica_config.moe_tensor_parallel_size
        else:
            tp_size = cluster_replica_config.attn_tensor_parallel_size
        pp_stage_boundary_handoff_time = (
            base_time
            if pipeline_stage < cluster_replica_config.num_pipeline_stages - 1
            else 0.0
        )
        # COMM_SKIP: TP all-reduce not needed when tp_size <= 1 (no tensor sharding)
        tp_comm_time = base_time if tp_size > 1 else 0.0
        ffn_tp_comm_enabled = (
            cluster_type == ClusterType.DECODE_FFN
            and architecture_profile.moe_tensor_parallel_allgather_op is not None
            and tp_size > 1
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

        if cluster_type == ClusterType.PREFILL:
            # PREFILL cluster handles full model layers
            return ExecutionTime(
                num_layers_per_pipeline_stage=self._num_layers_per_pipeline_stage,
                attention_rope_execution_time=base_time,
                attention_kv_cache_save_execution_time=base_time,
                attention_decode_execution_time=0.0,  # No decode in prefill
                attention_prefill_execution_time=base_time,
                attention_layer_pre_proj_execution_time=base_time,
                attention_layer_post_proj_execution_time=base_time,
                attn_norm_time=base_time,
                mlp_norm_time=base_time,
                add_time=base_time,
                tensor_parallel_communication_time=tp_comm_time,
                pipeline_parallel_communication_time=base_time,
                expert_parallel_communication_time=base_time,
                moe_gating_time=base_time,
                moe_shuffling_time=base_time,
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
                moe_grouped_gemm_time=base_time,
                share_expert_up_proj_time=share_expert_time,
                share_expert_down_proj_time=share_expert_time,
                share_expert_act_time=share_expert_time,
            )
        elif cluster_type == ClusterType.DECODE:
            # Unified DECODE cluster (PD-disaggregation mode): attention + (MLP/MoE)
            return ExecutionTime(
                num_layers_per_pipeline_stage=self._num_layers_per_pipeline_stage,
                attention_rope_execution_time=base_time,
                attention_kv_cache_save_execution_time=base_time,
                attention_decode_execution_time=base_time,
                attention_prefill_execution_time=0.0,  # No prefill in decode
                attention_layer_pre_proj_execution_time=base_time,
                attention_layer_post_proj_execution_time=base_time,
                attn_norm_time=base_time,
                mlp_norm_time=base_time,
                add_time=base_time,
                tensor_parallel_communication_time=tp_comm_time,
                pipeline_parallel_communication_time=base_time,
                expert_parallel_communication_time=base_time,
                moe_gating_time=base_time,
                moe_shuffling_time=base_time,
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
                moe_grouped_gemm_time=base_time,
                share_expert_up_proj_time=share_expert_time,
                share_expert_down_proj_time=share_expert_time,
                share_expert_act_time=share_expert_time,
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
            )
        elif cluster_type == ClusterType.DECODE_FFN:
            # DECODE_FFN cluster only handles FFN/MoE operations
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
                pipeline_parallel_communication_time=0.0,
                expert_parallel_communication_time=base_time,
                # In dummy mode, keep the per-layer MoE compute (gating + grouped_gemm)
                # roughly equal to base_time to avoid artificial Te >> Ta imbalance.
                moe_gating_time=base_time * 0.5,
                moe_shuffling_time=base_time,
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
                moe_grouped_gemm_time=base_time * 0.5,
                share_expert_up_proj_time=share_expert_time,
                share_expert_down_proj_time=share_expert_time,
                share_expert_act_time=share_expert_time,
                tensor_parallel_allgather_time=ffn_tp_allgather_time,
                share_expert_tensor_parallel_allreduce_time=share_expert_tp_allreduce_time,
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
            "is_moe": True,  # For FFN/MoE cluster
        }

    @staticmethod
    def _is_zero_token_decode_ffn_ep_barrier(
        batch: Batch,
        cluster_type: ClusterType,
    ) -> bool:
        """Return True for explicit zero-token DECODE_FFN EP barrier batches."""
        if cluster_type != ClusterType.DECODE_FFN:
            return False
        if not isinstance(batch, EPBatchGroup):
            return False

        per_expert_tokens = getattr(batch, "per_expert_tokens", None)
        if per_expert_tokens is None:
            return False

        expert_token_counts = [int(value) for value in per_expert_tokens.values()]
        if any(value < 0 for value in expert_token_counts):
            raise ValueError(
                "DECODE_FFN EP barrier per_expert_tokens must be non-negative, "
                f"got {per_expert_tokens}"
            )

        return batch.total_num_tokens == 0 and sum(expert_token_counts) == 0

    @staticmethod
    def _get_zero_decode_ffn_ep_barrier_execution_time(
        num_layers: int,
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
        )

    # Phase 2.5: Removed deprecated get_moe_stage_execution_details() method
    # MoE models now use predict_moe_layer_time() and other fine-grained APIs

    # ========================================================================
    # New unified API implementation (Phase 0) - Disaggregation extensions
    # ========================================================================

    def _get_communication_time(
        self, batch: Batch, stage_id: int, cluster_type: ClusterType
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
        if self._supports_operation("tensor_parallel_communication"):
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

    def predict_stage_execution_time(
        self,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int = 1,
        layer_id: int = 0,
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
        if self._enable_dummy_mode:
            if cluster_type in (
                ClusterType.PREFILL,
                ClusterType.DECODE_ATTN,
                ClusterType.DECODE,
                ClusterType.MONOLITHIC,
            ):
                self._log_step3_attention_shape(batch)
            # Phase 1 Fix: Use cluster-specific dummy execution time
            dummy_exec_time = self._get_dummy_execution_time_for_cluster(
                batch, stage_id, cluster_type
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
            )

        logger.debug(
            f"Predicting disaggregated stage execution time: stage_id={stage_id}, "
            f"cluster_type={cluster_type}, num_layers={num_layers}"
        )

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        if self._is_zero_token_decode_ffn_ep_barrier(batch, cluster_type):
            logger.debug(
                "[DECODE_FFN] Zero-token EP barrier batch_id=%s returns zero "
                "execution time without predictor lookup.",
                getattr(batch, "id", "N/A"),
            )
            return self._get_zero_decode_ffn_ep_barrier_execution_time(num_layers)

        measurement_type = self._select_measurement_type_for_batch(batch)
        self._require_predictions_for_measurement_type(measurement_type, batch)
        self._activate_measurement_type(measurement_type)

        # Validate cluster_type consistency
        if self._cluster_type is not None and cluster_type != self._cluster_type:
            logger.warning(
                f"Cluster type mismatch: predictor initialized with {self._cluster_type}, "
                f"but predict_stage_execution_time called with {cluster_type}"
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
        communication_time = self._get_communication_time(batch, stage_id, cluster_type)
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
            )

        elif cluster_type == ClusterType.DECODE_FFN:
            # FFN-only cluster (can be MoE or MLP depending on model type)
            # Check if this is a MoE model or dense model
            # Use model_config.is_moe for MoE detection - NOT parallelism settings
            cluster_replica_config = self._get_cluster_replica_config(cluster_type)
            model_config = cluster_replica_config.model_config
            architecture_profile = self._get_cluster_model_architecture_profile(cluster_type)
            is_moe_model = (
                model_config is not None and model_config.is_moe
            )

            if is_moe_model:
                # MoE model: use MoE operations
                logger.debug(
                    f"[DECODE_FFN] Processing MoE model: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                # Extract per_expert_tokens from EPBatchGroup if available
                per_expert_tokens = None
                logger.info(
                    f"[DECODE_FFN] Processing batch: type={type(batch).__name__}, id={batch.id}, "
                    f"hasattr(per_expert_tokens)={hasattr(batch, 'per_expert_tokens')}"
                )
                if hasattr(batch, "per_expert_tokens"):
                    per_expert_tokens = batch.per_expert_tokens
                    logger.info(
                        f"[DECODE_FFN] per_expert_tokens extracted: {per_expert_tokens}"
                    )
                    if per_expert_tokens:
                        logger.info(
                            f"Extracted per_expert_tokens from EPBatchGroup for DECODE_FFN: {len(per_expert_tokens)} experts"
                        )
                    else:
                        logger.warning(
                            f"[DECODE_FFN] per_expert_tokens is empty or None for batch {batch.id}"
                        )
                else:
                    logger.warning(
                        f"[DECODE_FFN] Batch {batch.id} does not have per_expert_tokens attribute (type={type(batch).__name__})"
                    )

                moe_time = self.predict_moe_layer_time(
                    batch,
                    layer_id=layer_id,
                    cluster_type=cluster_type,
                    per_expert_tokens=per_expert_tokens,  # Pass actual expert allocation
                )
                # Get post_attention_layernorm time (runs before MoE)
                mlp_norm_time = self._get_mlp_norm_layer_act_execution_time(batch)
                # Get residual add time (second residual connection after MoE)
                add_time = self._get_add_layer_act_execution_time(batch)
                add_attn_residual_time = 0.0
                add_ffn_residual_time = 0.0
                if architecture_profile.skip_decode_ffn_attn_norm_residual:
                    mlp_norm_time = 0.0
                if architecture_profile.residual_add_policy is ResidualAddPolicy.FFN_RESIDUAL_ONLY:
                    add_attn_residual_time = 0.0
                    add_ffn_residual_time = add_time
                    add_time = 0.0
                # Get expert parallel communication time separately (not from MoETime)
                ep_comm_time = self._get_expert_parallel_communication_time(batch)
                ffn_tp_allgather_time = 0.0
                share_expert_tp_allreduce_time = 0.0
                moe_tp_size = cluster_replica_config.moe_tensor_parallel_size
                if architecture_profile.moe_tensor_parallel_allgather_op and moe_tp_size > 1:
                    # Use compute-effective tokens. AFD paths already include CUDA Graph
                    # padding in metadata; non-CUDA-Graph paths keep exact token counts.
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
                        share_expert_tp_allreduce_time = self._apply_share_expert_tp_allreduce_overlap(
                            raw_share_expert_tp_allreduce_time
                        )
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
                )
            else:
                # Dense model: use MLP operations
                logger.debug(
                    f"[DECODE_FFN] Processing dense model: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                mlp_time = self.predict_mlp_layer_time(
                    batch, layer_id=layer_id, cluster_type=cluster_type
                )
                mlp_norm_time = mlp_time.mlp_norm_time
                # Get residual add time (second residual connection after MLP)
                add_time = self._get_add_layer_act_execution_time(batch)
                add_attn_residual_time = 0.0
                add_ffn_residual_time = 0.0
                if architecture_profile.skip_decode_ffn_attn_norm_residual:
                    mlp_norm_time = 0.0
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
                )

        elif cluster_type == ClusterType.DECODE:
            # Unified DECODE cluster (PD-disaggregation mode)
            # Handles both dense models (MLP) and MoE models
            # For dense models: attention + MLP
            # For MoE models: attention + MoE
            attention_time = self.predict_attention_layer_time(
                batch, layer_id=layer_id, cluster_type=cluster_type
            )

            # Check if this is a MoE model or dense model
            # Use model_config.is_moe for MoE detection - NOT parallelism settings
            cluster_replica_config = self._get_cluster_replica_config(cluster_type)
            is_moe_model = (
                cluster_replica_config.model_config is not None
                and cluster_replica_config.model_config.is_moe
            )

            if is_moe_model:
                # MoE model: use MoE operations
                logger.debug(
                    f"[DECODE] Processing MoE model: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                # Calculate per_expert_tokens from pre-initialized routing_details
                # This uses the routing distribution that was computed during predictor initialization
                # via _simulate_and_store_routing() method.
                per_expert_tokens = None
                if hasattr(batch, "per_expert_tokens") and batch.per_expert_tokens:
                    # EPBatchGroup case: use actual expert allocation from batch
                    per_expert_tokens = batch.per_expert_tokens
                    logger.debug(
                        f"Extracted per_expert_tokens from EPBatchGroup for DECODE: {len(per_expert_tokens)} experts"
                    )
                else:
                    # Regular Batch case: calculate from pre-initialized routing_details
                    # Use caller-provided layer_id so per-layer routing distributions are preserved
                    # In multi-layer scenarios, this should be called per-layer
                    per_expert_tokens = self._calculate_expert_token_allocation(
                        batch=batch,
                        cluster_type=cluster_type,
                        layer_id=layer_id,
                    )
                    logger.debug(
                        f"Calculated per_expert_tokens from routing_details for DECODE: "
                        f"{len(per_expert_tokens)} experts, total_tokens={sum(per_expert_tokens.values())}"
                    )

                moe_time = self.predict_moe_layer_time(
                    batch,
                    layer_id=layer_id,
                    cluster_type=cluster_type,
                    per_expert_tokens=per_expert_tokens,  # Pass calculated expert allocation
                )
                # Get post_attention_layernorm time (runs before MoE)
                mlp_norm_time = self._get_mlp_norm_layer_act_execution_time(batch)
                # Get residual add time (both residual connections)
                add_time = self._get_add_layer_act_execution_time(batch)
                # Get expert parallel communication time separately (not from MoETime)
                ep_comm_time = self._get_expert_parallel_communication_time(batch)

                # Calculate MoE TP allreduce time using moe_tensor_parallel_size
                # (communication_time.tensor_parallel_time uses attn_tensor_parallel_size,
                #  so we need a separate calculation for MoE TP allreduce)
                moe_tp_size = cluster_replica_config.moe_tensor_parallel_size
                moe_tp_allreduce_time = 0.0
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
                    moe_tp_allreduce_bytes = quant_manager.adjust_tensor_size(
                        "allreduce", data_size_bytes, cluster_type
                    )
                    moe_tp_allreduce_time = self.predict_allreduce_time(
                        data_size_bytes=moe_tp_allreduce_bytes,
                        num_devices=moe_tp_size,
                        cluster_type=cluster_type,
                        comm_domain="MOE_TP",
                    )

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
                )

        elif cluster_type == ClusterType.PREFILL:
            # Full model (attention + FFN) - predict both attention and FFN time
            # FFN can be MoE (for MoE models) or MLP (for dense models)
            attention_time = self.predict_attention_layer_time(
                batch, layer_id=layer_id, cluster_type=cluster_type
            )

            # Check if this is a MoE model or dense model
            # Use model_config.is_moe for MoE detection - NOT parallelism settings
            cluster_replica_config = self._get_cluster_replica_config(cluster_type)
            is_moe_model = (
                cluster_replica_config.model_config is not None
                and cluster_replica_config.model_config.is_moe
            )

            if is_moe_model:
                # MoE model: use MoE operations for FFN
                logger.debug(
                    f"[PREFILL] Processing MoE model: total_expert_num={self._replica_config.total_expert_num}, "
                    f"moe_expert_parallel_size={self._replica_config.moe_expert_parallel_size}"
                )

                # Calculate per_expert_tokens from pre-initialized routing_details
                # This uses the routing distribution that was computed during predictor initialization
                # via _simulate_and_store_routing() method.
                per_expert_tokens = None
                if hasattr(batch, "per_expert_tokens") and batch.per_expert_tokens:
                    # EPBatchGroup case: use actual expert allocation from batch
                    per_expert_tokens = batch.per_expert_tokens
                    logger.debug(
                        f"Extracted per_expert_tokens from EPBatchGroup for PREFILL: {len(per_expert_tokens)} experts"
                    )
                else:
                    # Regular Batch case: calculate from pre-initialized routing_details
                    # Use caller-provided layer_id so per-layer routing distributions are preserved
                    # In multi-layer scenarios, this should be called per-layer
                    per_expert_tokens = self._calculate_expert_token_allocation(
                        batch=batch,
                        cluster_type=cluster_type,
                        layer_id=layer_id,
                    )
                    logger.debug(
                        f"Calculated per_expert_tokens from routing_details for PREFILL: "
                        f"{len(per_expert_tokens)} experts, total_tokens={sum(per_expert_tokens.values())}"
                    )

                moe_time = self.predict_moe_layer_time(
                    batch,
                    layer_id=layer_id,
                    cluster_type=cluster_type,
                    per_expert_tokens=per_expert_tokens,  # Pass calculated expert allocation
                )

                # Get post_attention_layernorm time (runs before MoE)
                mlp_norm_time = self._get_mlp_norm_layer_act_execution_time(batch)
                # Get residual add time (both residual connections in full model)
                add_time = self._get_add_layer_act_execution_time(batch)
                # Get expert parallel communication time separately (not from MoETime)
                ep_comm_time = self._get_expert_parallel_communication_time(batch)
                (
                    dp_input_allreduce_time,
                    dp_output_allreduce_time,
                ) = self.predict_dp_moe_allreduce_times(batch, cluster_type)

                # Keep PREFILL MoE TP communication composition aligned with monolithic MoE path.
                moe_tp_size = cluster_replica_config.moe_tensor_parallel_size
                moe_tp_allreduce_time = 0.0
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
                    moe_tp_allreduce_bytes = quant_manager.adjust_tensor_size(
                        "allreduce", data_size_bytes, cluster_type
                    )
                    moe_tp_allreduce_time = self.predict_allreduce_time(
                        data_size_bytes=moe_tp_allreduce_bytes,
                        num_devices=moe_tp_size,
                        cluster_type=cluster_type,
                        comm_domain="MOE_TP",
                    )

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
                            raw_share_expert_tp_allreduce_time = self.predict_allreduce_time(
                                data_size_bytes=moe_tp_allreduce_bytes,
                                num_devices=moe_tp_size,
                                cluster_type=cluster_type,
                                comm_domain="MOE_TP",
                            )
                            share_expert_tp_allreduce_time = self._apply_share_expert_tp_allreduce_overlap(
                                raw_share_expert_tp_allreduce_time
                            )

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
