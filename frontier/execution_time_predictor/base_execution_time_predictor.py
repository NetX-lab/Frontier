from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TYPE_CHECKING

from frontier.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from frontier.entities import Batch, ExecutionTime
from frontier.types import ClusterType
from frontier.logger import init_logger

if TYPE_CHECKING:
    from frontier.entities.time_components import AttentionTime, MLPTime, MoETime
    from frontier.entities import EPBatchGroup

logger = init_logger(__name__)


class BaseExecutionTimePredictor(ABC):
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        self._config = predictor_config
        self._replica_config = replica_config
        self._model_config = replica_config.model_config

        # get configs
        self._replica_scheduler_provider = str(replica_scheduler_config.get_type())
        self._block_size = replica_scheduler_config.block_size
        self._cache_dir = metrics_config.cache_dir
        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )

        # Dummy mode configuration
        self._enable_dummy_mode = predictor_config.enable_dummy_mode
        self._dummy_execution_time = predictor_config.dummy_execution_time_ms

        if self._enable_dummy_mode:
            self._validate_dummy_mode_config()
            logger.info("ExecutionTimePredictor running in DUMMY mode")
            logger.info(f"Using fixed execution time: {self._dummy_execution_time}ms")
            logger.warning("DUMMY mode active - predictions are not realistic!")
            self._initialize_dummy_mode()
        else:
            self._initialize_normal_mode()

    def _validate_dummy_mode_config(self):
        """Validate dummy mode configuration."""
        if self._dummy_execution_time <= 0:
            raise ValueError(f"dummy_execution_time_ms must be positive, got {self._dummy_execution_time}")

        if self._dummy_execution_time > 1000:
            logger.warning(f"Large dummy execution time: {self._dummy_execution_time}ms - this may affect simulation realism")

    def _initialize_dummy_mode(self):
        """Initialize dummy mode with minimal overhead."""
        pass

    def _initialize_normal_mode(self):
        """Initialize normal mode - to be implemented by subclasses."""
        pass

    def _get_dummy_execution_time(self, batch: Batch, pipeline_stage: int) -> ExecutionTime:
        """Return fixed dummy ExecutionTime object."""
        base_time = self._dummy_execution_time

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
            add_time=base_time,
            tensor_parallel_communication_time=base_time,
            pipeline_parallel_communication_time=base_time,
            expert_parallel_communication_time=base_time,
            moe_gating_time=base_time,
            moe_shuffling_time=base_time,
            schedule_time=base_time,
            sampler_e2e_time=base_time,
            prepare_inputs_e2e_time=base_time,
            process_model_outputs_time=base_time,
            ray_comm_time=base_time,
            pp_producer_send_path_runtime_time=base_time,
            pp_prefill_consumer_active_runtime_time=base_time,
            pp_stage_boundary_handoff_time=base_time,
            is_moe=batch.is_moe if hasattr(batch, 'is_moe') else False,
            mlp_layer_up_proj_execution_time=base_time,
            mlp_layer_down_proj_execution_time=base_time,
            mlp_layer_act_execution_time=base_time,
            moe_grouped_gemm_time=base_time,
        )

    @abstractmethod
    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_schedule_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_ray_comm_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    # ========================================================================
    # New unified API for single-layer granularity and communication primitives
    # ========================================================================

    @abstractmethod
    def predict_attention_layer_time(
        self,
        batch: Batch,
        layer_id: int,
        cluster_type: ClusterType
    ) -> "AttentionTime":
        """
        Predict attention execution time for a single transformer layer.

        Args:
            batch: The batch being processed
            layer_id: Layer index (0-based, 0 to num_layers-1)
            cluster_type: Type of cluster (PREFILL, DECODE_ATTN, DECODE_FFN, MONOLITHIC)

        Returns:
            AttentionTime component with all attention-related times

        Raises:
            NotImplementedError: If not supported for this cluster type
            ValueError: If layer_id is out of range or batch is invalid
        """
        pass

    @abstractmethod
    def predict_mlp_layer_time(
        self,
        batch: Batch,
        layer_id: int,
        cluster_type: ClusterType
    ) -> "MLPTime":
        """
        Predict dense MLP execution time for a single transformer layer.

        Only applicable for non-MoE models. Mutually exclusive with predict_moe_layer_time.

        Args:
            batch: The batch being processed
            layer_id: Layer index (0-based, can be None for uniform layers)
            cluster_type: Type of cluster

        Returns:
            MLPTime component with all MLP-related times

        Raises:
            NotImplementedError: If model is MoE or cluster doesn't support MLP
            ValueError: If batch is invalid
        """
        pass

    @abstractmethod
    def predict_moe_layer_time(
        self,
        batch_or_group: "Batch | EPBatchGroup",
        layer_id: int,
        cluster_type: ClusterType,
        per_expert_tokens: Optional[Dict[int, int]] = None
    ) -> "MoETime":
        """
        Predict MoE execution time for a single transformer layer.

        Phase 3 Enhancement: Parameter renamed from expert_allocation to per_expert_tokens.

        Only applicable for MoE models. Mutually exclusive with predict_mlp_layer_time.

        Args:
            batch_or_group: Batch or EPBatchGroup being processed
            layer_id: Layer index (0-based, required for routing lookup)
            cluster_type: Type of cluster (PREFILL or DECODE_FFN)
            per_expert_tokens: Optional dict {expert_id: num_tokens} for grouped GEMM.
                              When provided (from EPBatchGroup), uses actual expert allocation.
                              When None, falls back to routing simulation.
                              Validates token conservation when provided.

        Returns:
            MoETime component with all MoE-related times

        Raises:
            NotImplementedError: If model is not MoE or cluster doesn't support MoE
            ValueError: If token conservation fails or layer_id invalid
        """
        pass

    @abstractmethod
    def predict_allreduce_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict tensor parallel all-reduce communication time.

        Used for aggregating attention/MLP outputs across TP replicas.

        Args:
            data_size_bytes: Size of data to reduce (in bytes)
            num_devices: Number of devices in the TP group
            cluster_type: Type of cluster

        Returns:
            Communication time in milliseconds

        Raises:
            NotImplementedError: If profiling data is missing
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    def predict_allgather_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict expert parallel all-gather communication time.

        Used for aggregating MoE results across EP replicas in DECODE_FFN cluster.

        Args:
            data_size_bytes: Size of data to gather (in bytes)
            num_devices: Number of devices in the EP group
            cluster_type: Type of cluster (should be DECODE_FFN)

        Returns:
            Communication time in milliseconds

        Raises:
            NotImplementedError: If profiling data is missing
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    def predict_alltoall_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict expert parallel all-to-all communication time.

        Used for MoE token dispatch/return in DECODE_FFN cluster.

        Args:
            data_size_bytes: Size of data to exchange (in bytes)
            num_devices: Number of devices in the EP group
            cluster_type: Type of cluster (should be DECODE_FFN)

        Returns:
            Communication time in milliseconds

        Raises:
            NotImplementedError: If profiling data is missing
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    def predict_p2p_time(
        self,
        data_size_bytes: int,
        cluster_type: ClusterType,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict pipeline parallel point-to-point communication time.

        Used for send/recv between pipeline stages.

        Args:
            data_size_bytes: Size of data to send/recv (in bytes)
            cluster_type: Type of cluster

        Returns:
            Communication time in milliseconds

        Raises:
            NotImplementedError: If profiling data is missing
            ValueError: If parameters are invalid
        """
        pass

    def predict_dp_moe_allreduce_times(
        self,
        batch: Batch,
        cluster_type: ClusterType,
    ) -> tuple[float, float]:
        """Return zero for the retired attention-DP MoE communication scope.

        MoE routing imbalance is modeled by Replica-local EP lanes.  The old
        attention-DP gather/scatter domain is not a valid execution scope;
        shared-domain roles require one attention world per serving Replica.
        Keep this structural seam so callers receive an explicit zero instead
        of deriving a second communication domain.
        """
        del batch, cluster_type
        attn_dp_size = int(getattr(self._replica_config, "attn_dp", 1))
        if attn_dp_size != 1:
            raise ValueError(
                "MoE attention-DP communication is retired; expected "
                f"attn_dp=1, got {attn_dp_size}"
            )
        return 0.0, 0.0

    @abstractmethod
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
    ) -> ExecutionTime:
        """
        Predict aggregated execution time for one or more transformer layers.

        This is the main entry point for execution time prediction. It composes
        attention, MLP/MoE, communication, overhead, and residual times.

        Args:
            batch: The batch being processed
            stage_id: Pipeline stage ID
            cluster_type: Type of cluster
            num_layers: Number of layers to aggregate (default=1 for single-layer)
                       For PD+AF disaggregation: always 1
                       For monolithic: may be > 1
            layer_id: Layer index used by per-layer MoE routing simulation.
                     Callers that do not model per-layer routing may keep default 0.
            include_moe: Optional explicit protocol selector. ``False`` is used by
                         mixed-layer calls to select the dense FFN branch.
            include_ffn: Whether to include the FFN/MoE block.  Set to ``False``
                         for an attention-only probe; such a probe must not read
                         dense FFN or routed-expert profiling rows.
            include_attention: Whether to include the attention block. Concrete
                               predictors must define the supported scope
                               explicitly.

        Returns:
            ExecutionTime object with aggregated times

        Raises:
            NotImplementedError: If not supported for this configuration
            ValueError: If parameters are invalid
        """
        pass
