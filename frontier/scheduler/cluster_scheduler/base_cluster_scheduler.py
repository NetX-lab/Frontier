from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

from frontier.config import ClusterConfig, MetricsConfig, BaseRequestGeneratorConfig
from frontier.entities import Batch, EPBatchGroup, ExecutionTime, Replica, Request, Cluster
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
# Phase 2.5: Removed deprecated MoECollectiveScheduleEvent import
from frontier.execution_time_predictor import (
    BaseExecutionTimePredictor,
    ExecutionTimePredictorRegistry,
)
from frontier.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from frontier.types import ClusterType, ClusterSchedulerType, ReplicaSchedulerType

class BaseClusterScheduler(ABC):
    def _validate_prefix_cache_cluster_config(self, replica_scheduler_config) -> None:
        prefix_enabled = bool(
            getattr(replica_scheduler_config, "enable_prefix_caching", False)
        )
        if not prefix_enabled:
            return

        scheduler_type = replica_scheduler_config.get_type()
        if scheduler_type not in {
            ReplicaSchedulerType.VLLM_V1,
            ReplicaSchedulerType.SGLANG,
            ReplicaSchedulerType.SJ2Q_FASTSERVE_LITE,
            ReplicaSchedulerType.SJ2Q_PENALTY_ONLY,
            ReplicaSchedulerType.SJ2Q_BOUNDED_CARRYOVER,
        }:
            raise ValueError(
                "Prefix caching only supports vllm_v1, sj2q_fastserve_lite, sj2q_penalty_only, sj2q_bounded_carryover, or sglang replica schedulers. "
                f"Got {scheduler_type}."
            )

        if self._cluster_type not in (ClusterType.MONOLITHIC, ClusterType.PREFILL):
            return

        cluster_scheduler_type = self._config.cluster_scheduler_config.get_type()
        if self._num_replicas > 1 and cluster_scheduler_type not in {
            ClusterSchedulerType.STICKY_ROUND_ROBIN,
            ClusterSchedulerType.STICKY_LOR,
        }:
            raise ValueError(
                "Multi-replica prefix caching requires a sticky cluster scheduler. "
                f"Got {cluster_scheduler_type}."
            )

    def _get_cluster_specific_replica_scheduler_config(self, config: ClusterConfig, cluster_type: ClusterType):
        """
        Get cluster-specific replica scheduler configuration.
        Priority: cluster-specific config -> global replica_scheduler_config -> default
        
        For scheduler type override:
        1. If cluster-specific type is specified (e.g., prefill_replica_scheduler_config_type),
           create a new config instance of that type and copy compatible parameters.
        2. Otherwise, use the global replica_scheduler_config.
        
        Args:
            config: ClusterConfig object
            cluster_type: Type of the cluster
            
        Returns:
            BaseReplicaSchedulerConfig: Configuration for the replica scheduler
        """
        from frontier.config import BaseReplicaSchedulerConfig
        from frontier.types import ReplicaSchedulerType
        
        # Get the base configuration
        base_config = config.replica_scheduler_config
        
        # Map cluster type to prefix
        prefix_map = {
            ClusterType.PREFILL: "prefill",
            ClusterType.DECODE: "decode", 
            ClusterType.DECODE_ATTN: "decode_attn",
            ClusterType.DECODE_FFN: "decode_ffn",
        }
        
        prefix = prefix_map.get(cluster_type)
        if not prefix:
            # If cluster type not in map, use global config
            import copy
            return copy.deepcopy(base_config)
        
        # Check for cluster-specific scheduler type override
        type_field_name = f"{prefix}_replica_scheduler_config_type"
        override_type_str = getattr(config, type_field_name, None) if hasattr(config, type_field_name) else None
        
        if override_type_str is not None:
            # Map string type to ReplicaSchedulerType enum
            type_mapping = {
                "vllm": ReplicaSchedulerType.VLLM,
                "vllm_v1": ReplicaSchedulerType.VLLM_V1,
                "sj2q_fastserve_lite": ReplicaSchedulerType.SJ2Q_FASTSERVE_LITE,
                "sj2q_penalty_only": ReplicaSchedulerType.SJ2Q_PENALTY_ONLY,
                "sj2q_bounded_carryover": ReplicaSchedulerType.SJ2Q_BOUNDED_CARRYOVER,
                "sglang": ReplicaSchedulerType.SGLANG,
                "orca": ReplicaSchedulerType.ORCA,
                "sarathi": ReplicaSchedulerType.SARATHI,
                "lightllm": ReplicaSchedulerType.LIGHTLLM,
                "faster_transformer": ReplicaSchedulerType.FASTER_TRANSFORMER,
            }
            override_type = type_mapping.get(override_type_str.lower())
            if override_type is None:
                raise ValueError(
                    f"Invalid scheduler type '{override_type_str}' for {cluster_type.name}. "
                    f"Valid options: {list(type_mapping.keys())}"
                )
            
            # Create new config instance of the overridden type
            cluster_config = BaseReplicaSchedulerConfig.create_from_type(override_type)

            # Copy all overlapping dataclass fields from the base config.
            # This keeps new scheduler fields (e.g., runtime profiling gates)
            # in sync without requiring manual updates here.
            from dataclasses import fields, is_dataclass

            if is_dataclass(base_config) and is_dataclass(cluster_config):
                base_field_names = {field.name for field in fields(base_config)}
                cluster_field_names = {field.name for field in fields(cluster_config)}
                for field_name in sorted(base_field_names & cluster_field_names):
                    setattr(cluster_config, field_name, getattr(base_config, field_name))
            else:
                # Fail fast: this path should always be dataclass-based.
                raise TypeError(
                    "Replica scheduler configs must be dataclasses for field-copy behavior"
                )
        else:
            # No type override, use a copy of the base config
            import copy
            cluster_config = copy.deepcopy(base_config)
        
        # Override individual parameters if specified (cluster-specific values take precedence)
        param_fields = [
            "batch_size_cap",
            "max_tokens_in_batch", 
            "num_blocks",
            "block_size",
            "watermark_blocks_fraction",
        ]
        
        for param in param_fields:
            field_name = f"{prefix}_replica_scheduler_config_{param}"
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                if value is not None and hasattr(cluster_config, param):
                    setattr(cluster_config, param, value)
        
        return cluster_config

    def __init__(
        self,
        config: ClusterConfig,
        cluster: Cluster,
        request_generator_config: BaseRequestGeneratorConfig,
        predictor: BaseExecutionTimePredictor = None,
        available_clusters: Optional[set] = None,
    ):
        self._config = config
        self._cluster = cluster
        self._cluster_type = cluster.cluster_type
        if self._cluster_type != ClusterType.MONOLITHIC:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        self._num_replicas = len(self._cluster.replicas)
        self._predictor = predictor
        self._replica_dp_size = self._config.replica_config.data_parallel_size
        self._available_clusters = available_clusters or set()
        self._request_generator_config = request_generator_config
        self._decode_shared_domain_related_wait_ms_by_batch: Dict[
            Tuple[int, int, int], float
        ] = defaultdict(float)

        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Validate and fix data_parallel_size if needed
        if self._replica_dp_size is None or self._replica_dp_size <= 0:
            logger.error(f"Invalid data_parallel_size: {self._replica_dp_size}, defaulting to 1")
            raise ValueError(f"Invalid data_parallel_size: {self._replica_dp_size}")

        # Initialize replica schedulers based on cluster type
        # DECODE_FFN: Use EP (Expert Parallel) concept instead of DP
        # Other clusters: Use DP (Data Parallel) concept
        self._dp_replica_schedulers = {}

        # Get cluster-specific replica scheduler configuration
        cluster_specific_config = self._get_cluster_specific_replica_scheduler_config(
            self._config, self._cluster_type
        )
        self._validate_prefix_cache_cluster_config(cluster_specific_config)

        # Validate scheduler type for DECODE_FFN cluster
        # DECODE_FFN requires "orca" scheduler for EP-based workload grouping
        if self._cluster_type == ClusterType.DECODE_FFN:
            scheduler_type = cluster_specific_config.get_type()
            if scheduler_type != ReplicaSchedulerType.ORCA:
                raise ValueError(
                    f"DECODE_FFN cluster requires 'orca' scheduler, got '{scheduler_type}'. "
                    f"Reason: DECODE_FFN uses EP-based workload grouping which is only implemented in OrcaReplicaScheduler."
                )

        if self._cluster_type == ClusterType.DECODE_FFN:
            # For DECODE_FFN cluster: use EP (Expert Parallel) instead of DP
            # Each replica has ep_size EP replicas for expert parallelism
            self._replica_ep_size = self._config.replica_config.moe_expert_parallel_size
            for replica_id, replica in self._cluster.replicas.items():
                for ep_id in range(self._replica_ep_size):
                    scheduler_key = (replica_id, ep_id)
                    self._dp_replica_schedulers[scheduler_key] = ReplicaSchedulerRegistry.get(
                        cluster_specific_config.get_type(),
                        replica_config=self._config.replica_config,
                        replica_scheduler_config=cluster_specific_config,
                        request_generator_config=request_generator_config,
                        replica=replica,
                        predictor=self._predictor,
                        cluster_type=self._cluster_type,
                        dp_id=ep_id,  # Use ep_id as dp_id for compatibility
                        af_pipeline_num_micro_batch=getattr(self._config, 'af_pipeline_num_micro_batch', -1),
                        cluster_scheduler=self,
                    )
        else:
            # For other clusters: use traditional DP concept
            for replica_id, replica in self._cluster.replicas.items():
                for dp_id in range(self._replica_dp_size):
                    scheduler_key = (replica_id, dp_id)
                    self._dp_replica_schedulers[scheduler_key] = ReplicaSchedulerRegistry.get(
                        cluster_specific_config.get_type(),
                        replica_config=self._config.replica_config,
                        replica_scheduler_config=cluster_specific_config,
                        request_generator_config=request_generator_config,
                        replica=replica,
                        predictor=self._predictor,
                        cluster_type=self._cluster_type,
                        dp_id=dp_id,
                        af_pipeline_num_micro_batch=getattr(self._config, 'af_pipeline_num_micro_batch', -1),
                        cluster_scheduler=self,
                    )
        self._request_queue = []

        # Initialize specialized queues for PD+AF disaggregation
        if self._cluster_type == ClusterType.DECODE_ATTN:
            # Queue for receiving requests from decode-ffn cluster (A→F communication)
            self._af_batch_queue = []
        elif self._cluster_type == ClusterType.DECODE_FFN:
            allow_multi_decode_ffn = bool(
                getattr(
                    self._config,
                    "allow_experiment_multi_decode_ffn_replicas",
                    False,
                )
            )
            # Fail fast at runtime as well: DECODE_FFN is modeled as a single
            # replica by default to avoid expert-set duplication across replicas.
            if self._num_replicas != 1 and not allow_multi_decode_ffn:
                raise ValueError(
                    "DECODE_FFN cluster requires exactly one replica for EP-based modeling; "
                    f"got {self._num_replicas} replicas. Set "
                    "allow_experiment_multi_decode_ffn_replicas=True only for "
                    "explicit experiment-only studies."
                )

            # Per-key waiting rooms for grouping distinct lanes (attn→ffn arrivals)
            # key=(target_ffn_replica_id, layer_id, afd_stage_idx)
            # -> {per_lane_queues, lanes_rr_order, rr_cursor}
            self._m2n_waiting_by_layer: Dict[tuple[int, int, int], dict] = {}
            self._m2n_ready_groups = deque()  # Deque[List[(batch, transfer_info)]]
            # Determine grouping lanes for decode-ffn based on DECODE_ATTN configuration
            # Prefer cross-cluster value propagated via config; fallback to local if absent
            attn_dp_lanes = getattr(self._config, 'decode_attn_dp_lanes_for_ffn', None)
            assert attn_dp_lanes is not None, "decode_attn_dp_lanes_for_ffn must be set for DECODE_FFN cluster"
            dp_lanes = int(attn_dp_lanes)
            dp_lanes = max(1, dp_lanes)
            self._ffn_group_micro_batches = dp_lanes
            attn_num_replicas = getattr(self._config, "decode_attn_cluster_num_replicas", None)
            attn_dp_size = getattr(
                self._config, "decode_attn_replica_config_attn_data_parallel_size", None
            )
            if attn_num_replicas is None or attn_dp_size is None:
                raise ValueError(
                    "decode_attn_cluster_num_replicas and decode_attn_replica_config_attn_data_parallel_size "
                    "must be set for DECODE_FFN lane barrier"
                )
            if int(attn_num_replicas) <= 0 or int(attn_dp_size) <= 0:
                raise ValueError(
                    f"Invalid decode-attn lane config: replicas={attn_num_replicas}, dp_size={attn_dp_size}"
                )
            attn_replica_id_start = getattr(
                self._config, "decode_attn_replica_id_start_for_ffn", None
            )
            if attn_replica_id_start is None:
                raise ValueError(
                    "decode_attn_replica_id_start_for_ffn must be set for "
                    "DECODE_FFN lane barrier"
                )
            attn_replica_id_start = int(attn_replica_id_start)
            self._ffn_expected_lanes = [
                (attn_replica_id_start + replica_ordinal, dp_id)
                for replica_ordinal in range(int(attn_num_replicas))
                for dp_id in range(int(attn_dp_size))
            ]
            if len(self._ffn_expected_lanes) != dp_lanes:
                raise ValueError(
                    "decode_attn_dp_lanes_for_ffn mismatch with expected lane topology: "
                    f"expected={len(self._ffn_expected_lanes)} configured={dp_lanes}"
                )
            self._ffn_replica_ids = sorted(self._cluster.replicas.keys())
            if not self._ffn_replica_ids:
                raise ValueError("DECODE_FFN cluster must have at least one replica")
            if len(self._ffn_replica_ids) != self._num_replicas:
                raise ValueError(
                    "DECODE_FFN replica ID inventory mismatch: "
                    f"ids={self._ffn_replica_ids}, num_replicas={self._num_replicas}"
                )
            self._ffn_expected_lanes_by_target: Dict[int, List[Tuple[int, int]]] = {
                replica_id: [] for replica_id in self._ffn_replica_ids
            }
            self._ffn_lane_to_target_replica: Dict[Tuple[int, int], int] = {}
            for lane_ordinal, lane in enumerate(self._ffn_expected_lanes):
                target_replica_id = self._ffn_replica_ids[
                    lane_ordinal % len(self._ffn_replica_ids)
                ]
                self._ffn_lane_to_target_replica[lane] = target_replica_id
                self._ffn_expected_lanes_by_target[target_replica_id].append(lane)
            expected_group_sizes = {
                replica_id: len(lanes)
                for replica_id, lanes in self._ffn_expected_lanes_by_target.items()
            }
            if any(group_size <= 0 for group_size in expected_group_sizes.values()):
                raise ValueError(
                    "DECODE_FFN experiment lane assignment must give every target "
                    f"replica at least one decode-attn lane, got {expected_group_sizes}"
                )
            self._ffn_group_micro_batches = max(expected_group_sizes.values())
            self._ffn_idle_lanes = set()
            total_requests = getattr(self._request_generator_config, "num_requests", None)
            if total_requests is not None:
                total_requests = int(total_requests)
                if total_requests < len(self._ffn_expected_lanes):
                    self._ffn_idle_lanes = set(self._ffn_expected_lanes[total_requests:])
                    if self._ffn_idle_lanes:
                        logger.info(
                            f"[FFN-GROUPING] Precomputed idle lanes for barrier: "
                            f"idle_lanes={sorted(self._ffn_idle_lanes)} total_requests={total_requests}"
                        )
            self._ffn_outstanding_group_credit_per_lane = 0
            logger.info(
                f"[FFN-GROUPING] Initialized with {dp_lanes} lanes for strict (layer_id, afd_stage_idx) grouping"
            )

            # EP waiting room for combine synchronization in decode-ffn cluster
            # Structure: replica_id -> stage_id -> batch_global_id -> {batches: {ep_id: batch}, arrival_times: {ep_id: time}}
            self._ep_allgather_waiting_room = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
                )
            )
        elif self._cluster_type in [ClusterType.PREFILL, ClusterType.MONOLITHIC]:
            # Prefill sync waiting room: replica_id -> stage_id -> batch_global_id -> layer_id -> sync_stage -> {dp_id: {batch, time}}
            # Used by disaggregated PREFILL and monolithic MoE prefill layer-by-layer paths.
            # MONOLITHIC MoE decode now also reuses the decode sync waiting-room path.
            model_is_moe = (
                self._config.replica_config.model_config is not None
                and self._config.replica_config.model_config.is_moe
            )
            if model_is_moe:
                self._prefill_sync_waiting_room = defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
                            )
                        )
                    )
                )
                if self._cluster_type == ClusterType.MONOLITHIC:
                    self._decode_sync_waiting_room = defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: defaultdict(
                                    lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
                                )
                            )
                        )
                    )
                else:
                    self._decode_sync_waiting_room = None
            else:
                # Dense model: no sync waiting room needed
                self._prefill_sync_waiting_room = None
                self._decode_sync_waiting_room = None
        elif self._cluster_type == ClusterType.DECODE:
            # Decode sync waiting room: replica_id -> stage_id -> batch_global_id -> layer_id -> sync_stage -> {dp_id: {batch, time}}
            # Similar to PREFILL, used for DP synchronization in unified DECODE cluster with MoE
            # Only initialize for MoE models (dense models don't need sync)
            # Use model_config.is_moe for MoE detection - NOT parallelism settings
            self._prefill_sync_waiting_room = None
            model_is_moe = (
                self._config.replica_config.model_config is not None
                and self._config.replica_config.model_config.is_moe
            )
            if model_is_moe:
                self._decode_sync_waiting_room = defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
                            )
                        )
                    )
                )
            else:
                # Dense model: no sync waiting room needed
                self._decode_sync_waiting_room = None

        # Phase 2.5: Removed deprecated _moe_waiting_room (old MoE synchronization)
        # Current architecture uses EP-based synchronization instead

        # Store raw batches by id for O(1) retrieval during F→A return path
        self._raw_batch_waiting_for_m2n_back = {}

        # Initialize periodic scheduling if enabled for this cluster type
        self._is_periodic_scheduling_enabled = self._cluster_type in config.periodic_scheduling_clusters
        self._periodic_scheduling_interval_ms = config.periodic_scheduling_interval_ms

        # Validate periodic scheduling configuration
        if self._is_periodic_scheduling_enabled:
            if self._cluster_type not in [ClusterType.DECODE_ATTN]:
                raise NotImplementedError(
                    f"Periodic scheduling is not implemented for cluster type {self._cluster_type.name}. "
                    f"Currently only DECODE_ATTN is supported."
                )

            # from frontier.logger import get_cluster_logger
            # logger = get_cluster_logger(__name__, self._cluster_type.name)
            logger.info(f"Periodic scheduling enabled for {self._cluster_type.name} cluster "
                       f"with interval {self._periodic_scheduling_interval_ms}ms")

        self._batch_group_creation_counter = 0


    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda request: request._arrived_at)

    def _schedule_batch_mode(self) -> List[Tuple[int, int, Request]]:
        """
        Default batch processing logic for clusters.
        This is a placeholder that should be overridden by specific schedulers.
        """
        return []

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def initialize_periodic_scheduling(self, start_time: float = 0.0) -> List:
        """
        Initialize periodic scheduling for this cluster if enabled.

        Args:
            start_time: Time to start the first periodic scheduling event

        Returns:
            List containing the initial PeriodicScheduleEvent if periodic scheduling is enabled
        """
        if not self._is_periodic_scheduling_enabled:
            return []

        from frontier.events.periodic_schedule_event import PeriodicScheduleEvent
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)
        first_schedule_time = start_time + self._periodic_scheduling_interval_ms / 1000.0

        logger.info(f"Initializing periodic scheduling for {self._cluster_type.name} cluster: "
                   f"first event at {first_schedule_time:.3f}s, interval={self._periodic_scheduling_interval_ms}ms")

        return [PeriodicScheduleEvent(first_schedule_time, self._cluster_type, self._periodic_scheduling_interval_ms)]

    def get_replica(self, replica_id: int) -> Replica:
        return self._cluster.replicas[replica_id]

    def get_dp_replica_scheduler(self, replica_id: int, dp_id: int):
        return self._dp_replica_schedulers[(replica_id, dp_id)]

    def get_dp_replica_stage_scheduler(self, replica_id: int, dp_id: int, stage_id: int):
        return self._dp_replica_schedulers[(replica_id, dp_id)].get_replica_stage_scheduler(
            stage_id
        )

    def make_decode_sync_global_id(
        self,
        replica_id: int,
        dp_id: int,
        lane_decode_sync_counter: int,
    ) -> int:
        """Encode a MONOLITHIC MoE decode-sync id with lane scope."""
        del replica_id

        ep_participant_count = getattr(self, "_replica_ep_size", None)
        if ep_participant_count is None and hasattr(self, "_config"):
            ep_participant_count = getattr(
                self._config.replica_config,
                "moe_expert_parallel_size",
                1,
            )
        ep_participant_count = max(1, int(ep_participant_count or 1))

        lane_id = int(dp_id or 0)
        if lane_id < 0:
            raise ValueError(f"dp_id must be non-negative, got {dp_id!r}")
        if lane_id >= ep_participant_count:
            raise ValueError(
                "MONOLITHIC decode sync lane id must be within the EP participant "
                f"domain, got dp_id={lane_id}, ep_participant_count={ep_participant_count}"
            )

        lane_counter = int(lane_decode_sync_counter or 0)
        if lane_counter < 0:
            raise ValueError(
                "lane_decode_sync_counter must be non-negative, "
                f"got {lane_decode_sync_counter!r}"
            )
        return lane_counter * ep_participant_count + lane_id

    def _get_decode_target_cluster(self) -> ClusterType:
        """
        Determine the target decode cluster based on system architecture.

        This method is called by PREFILL cluster to determine where to send
        KV cache after prefill completion.

        Returns:
            ClusterType.DECODE for PD-disaggregation mode
            ClusterType.DECODE_ATTN for PD+AF-disaggregation mode
        """
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Check if DECODE cluster exists (PD-disaggregation mode)
        if ClusterType.DECODE in self._available_clusters:
            logger.debug(f"[ROUTE] PREFILL → DECODE (PD-disaggregation mode)")
            return ClusterType.DECODE

        # Default to DECODE_ATTN for PD+AF-disaggregation mode
        logger.debug(f"[ROUTE] PREFILL → DECODE_ATTN (PD+AF-disaggregation mode)")
        return ClusterType.DECODE_ATTN

    @staticmethod
    def _debug_request_id(request: Request) -> int:
        if not hasattr(request, "id"):
            raise TypeError(f"Expected Request-like object with id, got {type(request)}")
        return int(request.id)

    @classmethod
    def _debug_request_collection_state(cls, requests: Any) -> Dict[str, Any]:
        if requests is None:
            return {"status": "not_applicable"}
        request_values = list(requests.values()) if isinstance(requests, dict) else list(requests)
        return {
            "count": len(request_values),
            "request_ids": [
                cls._debug_request_id(request) for request in request_values
            ],
            "requests": [
                {
                    "id": cls._debug_request_id(request),
                    "arrived_at": getattr(request, "arrived_at", None),
                    "num_prefill_tokens": getattr(
                        request, "num_prefill_tokens", None
                    ),
                    "num_decode_tokens": getattr(request, "num_decode_tokens", None),
                    "num_processed_tokens": getattr(
                        request, "num_processed_tokens", None
                    ),
                    "current_decode_token_index": getattr(
                        request, "current_decode_token_index", None
                    ),
                    "completed_layer_count": getattr(
                        request, "completed_layer_count", None
                    ),
                    "af_roundtrip_inflight": getattr(
                        request, "af_roundtrip_inflight", None
                    ),
                    "completed": getattr(request, "completed", None),
                }
                for request in request_values
            ],
        }

    @staticmethod
    def _debug_batch_id(batch: Batch) -> int:
        if not hasattr(batch, "id"):
            raise TypeError(f"Expected Batch-like object with id, got {type(batch)}")
        return int(batch.id)

    @classmethod
    def _debug_batch_collection_state(cls, batches: Any) -> Dict[str, Any]:
        if batches is None:
            return {"status": "not_applicable"}
        batch_values = list(batches)
        return {
            "count": len(batch_values),
            "batch_ids": [cls._debug_batch_id(batch) for batch in batch_values],
            "batch_global_ids": [
                getattr(batch, "global_id", None) for batch in batch_values
            ],
            "request_ids": [
                list(getattr(batch, "request_ids", [])) for batch in batch_values
            ],
            "batches": [
                {
                    "id": cls._debug_batch_id(batch),
                    "global_id": getattr(batch, "global_id", None),
                    "replica_id": getattr(batch, "replica_id", None),
                    "afd_stage_idx": getattr(batch, "afd_stage_idx", None),
                    "target_ffn_replica_id": getattr(
                        batch, "target_ffn_replica_id", None
                    ),
                    "total_num_tokens": getattr(batch, "total_num_tokens", None),
                    "request_ids": list(getattr(batch, "request_ids", [])),
                    "is_idle": getattr(batch, "is_idle", None),
                }
                for batch in batch_values
            ],
        }

    @staticmethod
    def _debug_lane_tuple(lane: Any) -> List[Any]:
        if not isinstance(lane, tuple) or len(lane) != 2:
            raise TypeError(f"Expected lane tuple(replica_id, dp_id), got {lane!r}")
        return [lane[0], lane[1]]

    @classmethod
    def _debug_batch_transfer_pairs_state(cls, pairs: Any) -> Dict[str, Any]:
        pair_values = list(pairs)
        batch_values = []
        pair_details = []
        for pair in pair_values:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"Expected (batch, transfer_info) pair, got {pair!r}")
            batch, transfer_info = pair
            batch_values.append(batch)
            pair_details.append(
                {
                    "batch_id": cls._debug_batch_id(batch),
                    "batch_global_id": getattr(batch, "global_id", None),
                    "request_ids": list(getattr(batch, "request_ids", [])),
                    "source_lane": [
                        getattr(transfer_info, "source_replica_id", None),
                        getattr(transfer_info, "source_dp_id", None),
                    ],
                    "target_ffn_replica_id": getattr(
                        transfer_info, "target_ffn_replica_id", None
                    ),
                    "layer_id": getattr(transfer_info, "layer_id", None),
                    "afd_stage_idx": getattr(transfer_info, "afd_stage_idx", None),
                    "activation_size_bytes": getattr(
                        transfer_info, "activation_size_bytes", None
                    ),
                }
            )
        return {
            "count": len(pair_values),
            "batch_ids": [cls._debug_batch_id(batch) for batch in batch_values],
            "request_ids": [
                list(getattr(batch, "request_ids", [])) for batch in batch_values
            ],
            "pairs": pair_details,
        }

    @classmethod
    def _debug_m2n_waiting_groups_state(
        cls,
        waiting_by_layer: Dict[Tuple[int, int, int], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        groups = []
        for group_key, room in sorted(
            waiting_by_layer.items(), key=lambda item: str(item[0])
        ):
            if not isinstance(group_key, tuple) or len(group_key) != 3:
                raise TypeError(
                    f"Expected DECODE_FFN waiting key(target, layer, stage), got {group_key!r}"
                )
            target_ffn_replica_id, layer_id, afd_stage_idx = group_key
            if "per_lane_queues" not in room or "lanes_rr_order" not in room:
                raise RuntimeError(
                    f"M2N waiting room {group_key} missing per_lane_queues or lanes_rr_order"
                )
            lane_queues = []
            for lane, lane_queue in sorted(
                room["per_lane_queues"].items(), key=lambda item: str(item[0])
            ):
                lane_queues.append(
                    {
                        "lane": cls._debug_lane_tuple(lane),
                        "queue": cls._debug_batch_transfer_pairs_state(lane_queue),
                    }
                )
            groups.append(
                {
                    "key": {
                        "target_ffn_replica_id": target_ffn_replica_id,
                        "layer_id": layer_id,
                        "afd_stage_idx": afd_stage_idx,
                    },
                    "lanes_rr_order": [
                        cls._debug_lane_tuple(lane)
                        for lane in list(room["lanes_rr_order"])
                    ],
                    "rr_cursor": room.get("rr_cursor"),
                    "lane_queues": lane_queues,
                }
            )
        return groups

    @classmethod
    def _debug_m2n_ready_groups_state(cls, ready_groups: Any) -> List[Dict[str, Any]]:
        return [
            cls._debug_batch_transfer_pairs_state(group)
            for group in list(ready_groups)
        ]

    @classmethod
    def _debug_raw_batch_waiting_map_state(
        cls, raw_batch_waiting_map: Dict[Any, Batch]
    ) -> Dict[str, Any]:
        if raw_batch_waiting_map is None:
            raise RuntimeError(
                "_raw_batch_waiting_for_m2n_back is required for cluster diagnostics"
            )
        keys = sorted(raw_batch_waiting_map.keys())
        batches = [raw_batch_waiting_map[key] for key in keys]
        return {
            "count": len(raw_batch_waiting_map),
            "keys": [int(key) for key in keys],
            "batch_ids": [cls._debug_batch_id(batch) for batch in batches],
            "request_ids": [
                list(getattr(batch, "request_ids", [])) for batch in batches
            ],
        }

    def get_debug_state(self) -> Dict[str, Any]:
        """Return fail-fast diagnostic state for this cluster scheduler."""
        required_attrs = [
            "_cluster_type",
            "_request_queue",
            "_dp_replica_schedulers",
            "_raw_batch_waiting_for_m2n_back",
        ]
        for attr_name in required_attrs:
            if not hasattr(self, attr_name):
                raise RuntimeError(
                    f"Cluster scheduler missing required debug field {attr_name}"
                )

        if self._cluster_type == ClusterType.DECODE_ATTN:
            if not hasattr(self, "_af_batch_queue"):
                raise RuntimeError("DECODE_ATTN scheduler missing _af_batch_queue")
            af_queue = self._debug_batch_collection_state(self._af_batch_queue)
        else:
            af_queue = {"status": "not_applicable"}

        if self._cluster_type == ClusterType.DECODE_FFN:
            if not hasattr(self, "_m2n_waiting_by_layer"):
                raise RuntimeError("DECODE_FFN scheduler missing _m2n_waiting_by_layer")
            if not hasattr(self, "_m2n_ready_groups"):
                raise RuntimeError("DECODE_FFN scheduler missing _m2n_ready_groups")
            m2n_waiting_groups = self._debug_m2n_waiting_groups_state(
                self._m2n_waiting_by_layer
            )
            m2n_ready_groups = self._debug_m2n_ready_groups_state(
                self._m2n_ready_groups
            )
        else:
            m2n_waiting_groups = {"status": "not_applicable"}
            m2n_ready_groups = {"status": "not_applicable"}

        replica_states = {}
        for scheduler_key, replica_scheduler in sorted(
            self._dp_replica_schedulers.items(), key=lambda item: str(item[0])
        ):
            if not hasattr(replica_scheduler, "get_debug_state"):
                raise RuntimeError(
                    f"Replica scheduler {scheduler_key} missing get_debug_state()"
                )
            replica_states[str(scheduler_key)] = replica_scheduler.get_debug_state()

        return {
            "scheduler_class": self.__class__.__name__,
            "cluster_type": self._cluster_type.name,
            "request_queue": self._debug_request_collection_state(
                self._request_queue
            ),
            "af_queue": af_queue,
            "m2n_waiting_groups": m2n_waiting_groups,
            "m2n_ready_groups": m2n_ready_groups,
            "raw_batch_waiting_map": self._debug_raw_batch_waiting_map_state(
                self._raw_batch_waiting_for_m2n_back
            ),
            "replica_schedulers": replica_states,
        }

    def is_empty(self) -> bool:
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        rq_len = len(self._request_queue)
        # Optional AF queue (only exists for decode-attn)
        af_q_len = len(self._af_batch_queue) if hasattr(self, '_af_batch_queue') else 0

        replica_states = []
        all_empty = True
        for key, replica_scheduler in self._dp_replica_schedulers.items():
            rs_empty = replica_scheduler.is_empty()
            replica_states.append((key, rs_empty))
            all_empty = all_empty and rs_empty

        logger.info(f"[IDLE-CHECK][{self._cluster_type.name}] request_queue={rq_len}, af_batch_queue={af_q_len}, replica_empty={[(str(k), v) for k, v in replica_states]}")

        # Return True only if request queue, AF queue (if exists), and all replicas are empty
        return rq_len == 0 and af_q_len == 0 and all_empty

    @staticmethod
    def _conserve_tokens_allocation(total_tokens: int, expert_ids: List[int], ratios: Dict[int, float]) -> Dict[int, int]:
        """Allocate integer tokens per expert with strict conservation using the
        Largest Remainder Method (Hamilton).

        Args:
            total_tokens: Target total tokens to distribute (non-negative integer)
            expert_ids: Ordered list of expert global IDs to allocate to
            ratios: Mapping expert_id -> routing ratio (non-negative floats). Ratios
                    are allowed to sum to a value different from 1.0; they will be
                    normalized internally when total_tokens > 0.

        Returns:
            Dict[expert_id, tokens] with sum(values) == total_tokens.

        Raises:
            ValueError: If total_tokens < 0, or ratios contain negative values,
                        or ratios sum to 0 while total_tokens > 0
        """
        if total_tokens < 0:
            raise ValueError("total_tokens must be non-negative")
        if any(r < 0.0 for r in ratios.values()):
            raise ValueError("Routing ratios must be non-negative")

        # Edge case: no tokens to distribute => all zeros
        if total_tokens == 0:
            return {eid: 0 for eid in expert_ids}

        # Normalize ratios if necessary
        sum_ratios = sum(ratios.get(eid, 0.0) for eid in expert_ids)
        if sum_ratios <= 0.0:
            # No routing mass but need to place tokens => fail fast
            raise ValueError("Sum of routing ratios must be > 0 when total_tokens > 0")

        normalized = {eid: (ratios.get(eid, 0.0) / sum_ratios) for eid in expert_ids}

        # Initial allocation by floor
        ideal = {eid: total_tokens * normalized[eid] for eid in expert_ids}
        base = {eid: int(math.floor(ideal[eid])) for eid in expert_ids}
        allocated = sum(base.values())
        remainder = total_tokens - allocated

        # Distribute remaining tokens to experts with largest fractional parts
        # Deterministic tie-breaker: lower expert_id first
        if remainder > 0:
            ranked = sorted(
                ((eid, ideal[eid] - base[eid]) for eid in expert_ids),
                key=lambda x: (-x[1], x[0])
            )
            for i in range(remainder):
                eid = ranked[i % len(ranked)][0]
                base[eid] += 1

        # Validation: non-negative and exact conservation
        assert all(v >= 0 for v in base.values()), "Negative allocation detected"
        assert sum(base.values()) == total_tokens, (
            f"Token conservation violated: allocated={sum(base.values())}, target={total_tokens}"
        )

        return base

    @staticmethod
    def _get_ep_subset_routed_token_total(
        total_routed_tokens: int,
        expert_ids: List[int],
        ratios: Dict[int, float],
    ) -> int:
        """Return the routed-token total that belongs to one EP expert subset.

        ``ratios`` describes the global routing distribution across all experts.
        A single EP lane owns only ``expert_ids``, so it must receive only the
        routing mass covered by that subset. Allocating the full global token
        count to every EP lane would multiply MoE work by ``ep_size`` and create
        pathological O(total_tokens) remainder loops for large prefill batches.
        """
        allocation = BaseClusterScheduler._get_ep_subset_routed_token_allocation(
            total_routed_tokens,
            expert_ids,
            ratios,
        )
        return sum(allocation.values())

    def _get_cached_ep_subset_routed_token_allocation(
        self,
        total_routed_tokens: int,
        expert_ids: List[int],
        ratios: Dict[int, float],
    ) -> Dict[int, int]:
        """Allocate routed tokens for an EP subset using an exact global cache.

        DECODE_FFN creates one EPBatchGroup per EP lane for the same ready group.
        The Hamilton allocation across the global expert set is identical for all
        EP lanes in that group, so recomputing it once per lane is pure overhead.
        Cache the exact global allocation and then return the requested subset.
        """
        global_expert_ids = tuple(sorted(set(ratios.keys()) | set(expert_ids)))
        cache_key = (id(ratios), int(total_routed_tokens), global_expert_ids)
        cache = getattr(self, "_ep_routed_token_allocation_cache", None)
        if cache is None:
            cache = {}
            self._ep_routed_token_allocation_cache = cache

        global_allocation = cache.get(cache_key)
        if global_allocation is None:
            global_allocation = self._conserve_tokens_allocation(
                total_routed_tokens,
                list(global_expert_ids),
                ratios,
            )
            cache[cache_key] = global_allocation

        return {eid: global_allocation.get(eid, 0) for eid in expert_ids}

    @staticmethod
    def _get_ep_subset_routed_token_allocation(
        total_routed_tokens: int,
        expert_ids: List[int],
        ratios: Dict[int, float],
    ) -> Dict[int, int]:
        """Allocate routed tokens to one EP subset from a global expert split.

        The global expert allocation is computed once with strict conservation,
        then restricted to the experts owned by the EP lane. This avoids
        independent per-subset rounding, whose accumulated error can create or
        drop routed tokens across the full EP group.
        """
        global_expert_ids = sorted(set(ratios.keys()) | set(expert_ids))
        if not global_expert_ids:
            if total_routed_tokens == 0:
                return {}
            raise ValueError(
                "At least one expert ID is required when total_routed_tokens > 0"
            )
        global_allocation = BaseClusterScheduler._conserve_tokens_allocation(
            total_routed_tokens,
            global_expert_ids,
            ratios,
        )
        return {eid: global_allocation.get(eid, 0) for eid in expert_ids}

    def _validate_token_conservation(self, input_tokens: int, per_expert_tokens: Dict[int, int], context: str):
        """
        Phase 3 Task 2: Validate that tokens are conserved in MoE routing.

        Args:
            input_tokens: Total number of tokens entering MoE layer
            per_expert_tokens: Dict mapping expert_id -> token_count
            context: Description of where validation is happening (for error messages)

        Raises:
            ValueError: If token conservation is violated
        """
        total_expert_tokens = sum(per_expert_tokens.values())
        if total_expert_tokens != input_tokens:
            raise ValueError(
                f"Token conservation violated in {context}: "
                f"Input tokens={input_tokens}, Expert tokens={total_expert_tokens}, "
                f"Difference={input_tokens - total_expert_tokens}, "
                f"Per-expert allocation={per_expert_tokens}"
            )

    def _distribute_tokens_within_ep_replica(self, group: List[Batch], replica_id, ep_id, expert_global_ids, layer_global_id, routing_details) -> EPBatchGroup:
        # Diagnostic logging to validate input structure during PREFILL EP sync
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        logger.info(
            f"[_distribute_tokens_within_ep_replica][ENTER] cluster={self._cluster_type.name}, "
            f"replica={replica_id}, ep_id={ep_id}, layer={layer_global_id}, group_type={type(group).__name__}, "
            f"group_len={len(group) if hasattr(group, '__len__') else 'NA'}"
        )

        if not isinstance(group, list):
            raise ValueError("group must be a list")
        if len(group) == 0:
            raise ValueError("group must be non-empty")

        afd_stage_idx_values = {getattr(batch, "afd_stage_idx", None) for (batch, _) in group}
        if None in afd_stage_idx_values:
            raise ValueError("afd_stage_idx missing in DECODE_FFN group batches")
        if len(afd_stage_idx_values) != 1:
            raise ValueError(
                f"afd_stage_idx mismatch in group: {sorted(afd_stage_idx_values)}"
            )
        afd_stage_idx = afd_stage_idx_values.pop()

        # ISSUE-007 FIX: Validate that source batches have required decode_attn_original_* attributes
        # This validation helps identify where the attributes are lost in the A→F transfer path
        if self._cluster_type == ClusterType.DECODE_FFN:
            for (batch, _) in group:
                orig_replica_id = getattr(batch, 'decode_attn_original_replica_id', None)
                orig_dp_id = getattr(batch, 'decode_attn_original_dp_id', None)
                logger.info(
                    f"[ISSUE-007][VALIDATE] batch_id={batch.id} entering DECODE_FFN: "
                    f"decode_attn_original_replica_id={orig_replica_id}, "
                    f"decode_attn_original_dp_id={orig_dp_id}"
                )
                if orig_replica_id is None or orig_dp_id is None:
                    raise ValueError(
                        f"[ISSUE-007] Batch {batch.id} entering DECODE_FFN without decode_attn_original_* attributes. "
                        f"decode_attn_original_replica_id={orig_replica_id}, "
                        f"decode_attn_original_dp_id={orig_dp_id}. "
                        f"This indicates a bug in the A→F transfer path - the original batch from DECODE_ATTN "
                        f"should have these attributes set when created by _create_batch()."
                    )

        # We aim to get experts_tokens_mapping result
        experts_tokens_mapping = {}

        # 1) Aggregate total tokens and register raw batches for F→A return path
        ep_batch_group_total_num_token = 0
        source_batch_ids = []
        for (batch, _) in group:
            source_batch_ids.append(batch.id)
            ep_batch_group_total_num_token += batch.total_num_tokens
            assert batch.num_routing_tokens == batch.total_num_tokens, "num_tokens mismatch"
            if self._cluster_type == ClusterType.DECODE_FFN:
                self._raw_batch_waiting_for_m2n_back[batch.id] = batch

        router_topk = self._config.replica_config.router_topk
        ep_batch_group_total_num_token *= router_topk

        # Use a stable time for the group (avoid relying on a loop variable)
        group_time = max((b.time or 0.0) for (b, _) in group)

        # 2) Calculate experts_tokens_mapping based on routing_details and total tokens
        # logic_requets is designed for compatibility with existing logic (enrich ep_batch_group with per-request token info)
        # We assume one expert has only one request here
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        rd_replicas = list(getattr(routing_details, 'keys', lambda: [])())
        sub = routing_details.get(replica_id) if isinstance(routing_details, dict) else None
        rd_layers = list(sub.keys()) if isinstance(sub, dict) else []
        logger.info(f"[FFN-DIST][DEBUG] replica_id={replica_id}, layer_global_id={layer_global_id}, ep_id={ep_id}, expert_ids={expert_global_ids}, rd_replicas={rd_replicas}, rd_layers_for_replica={rd_layers}")

        global_routing_ratios = routing_details[replica_id][layer_global_id]
        experts_tokens_mapping = self._get_cached_ep_subset_routed_token_allocation(
            ep_batch_group_total_num_token,
            expert_global_ids,
            global_routing_ratios,
        )
        ep_batch_group_total_num_token = sum(experts_tokens_mapping.values())

        # Create logic requests with corrected token counts
        logic_requets = []
        logic_num_tokens = []
        for expert_global_id in expert_global_ids:
            num_tokens = experts_tokens_mapping[expert_global_id]
            logic_req = Request(0.0, 0, num_tokens)
            logic_num_tokens.append(num_tokens)
            logic_requets.append(logic_req)

        # Phase 3 Task 2: Validate token conservation in MoE routing
        self._validate_token_conservation(
            input_tokens=ep_batch_group_total_num_token,
            per_expert_tokens=experts_tokens_mapping,
            context=f"_distribute_tokens_within_ep_replica (cluster={self._cluster_type.name}, "
                   f"replica={replica_id}, ep_id={ep_id}, layer={layer_global_id})"
        )

        # 3) Create EPBatchGroup (global_id will be assigned by caller to keep all EP sub-batches consistent)
        ep_batch_group = self._create_batch_group(
            logic_requets, logic_num_tokens, replica_id, ep_id, group_time, source_batch_ids, experts_tokens_mapping
        )
        ep_batch_group.afd_stage_idx = afd_stage_idx
        ep_batch_group.source_batches = [batch for (batch, _) in group]

        return ep_batch_group

    # Phase 2.5: Removed deprecated on_moe_ready() method
    # Old MoE synchronization architecture is no longer supported
    # Current architecture uses EP-based synchronization (EPAllToAllCombineReadyEvent/EPAllToAllCombineCollectiveEvent)

    def on_ep_alltoall_combine_ready(self, time: float, replica_id: int, stage_id: int, batch, ep_id: int):
        """
        Handle EP AllToAll combine readiness in decode-ffn cluster.

        This method is called when an EP replica completes its expert computation
        and is ready for AllToAll combine synchronization to aggregate results.

        Args:
            time: Current simulation time
            replica_id: ID of the replica
            stage_id: Pipeline stage ID
            batch: The batch that completed expert computation
            ep_id: Expert parallel replica ID
        """
        from frontier.events.ep_alltoall_combine_collective_event import (
            EPAllToAllCombineCollectiveEvent,
        )
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # DIAGNOSTIC: Log EP combine ready with global_id
        batch_global_id = batch.global_id  # Use batch.global_id for EP batch synchronization
        logger.info(
            f"[EP-WAIT-ROOM][ENTER] time={time:.3f}s, batch_id={batch.id}, global_id={batch_global_id}, "
            f"replica={replica_id}, stage={stage_id}, ep_id={ep_id}"
        )

        ep_wait_room = self._ep_allgather_waiting_room[replica_id][stage_id][batch_global_id]
        ep_wait_room["batches"][ep_id] = batch
        ep_wait_room["arrival_times"][ep_id] = time

        # Get the replica to check ep_size
        replica = self.get_replica(replica_id)
        expected_ep_size = getattr(replica, 'ep_size', self._config.replica_config.moe_expert_parallel_size)

        # DIAGNOSTIC: Log wait room status with all waiting batches
        arrived_ep_ids = list(ep_wait_room['batches'].keys())
        arrived_batch_ids = [ep_wait_room['batches'][eid].id for eid in arrived_ep_ids]
        logger.info(
            f"[EP-WAIT-ROOM][STATUS] global_id={batch_global_id}, "
            f"arrived={len(ep_wait_room['batches'])}/{expected_ep_size}, "
            f"ep_ids={arrived_ep_ids}, batch_ids={arrived_batch_ids}"
        )

        # Check if all EP replicas in this replica have arrived
        if len(ep_wait_room["batches"]) == expected_ep_size:
            # Synchronize to the maximum time across all EP replicas
            logger.info(
                "[DEBUG] All EP replicas arrived! Creating EPAllToAllCombineCollectiveEvent"
            )

            # Phase 1: Migrated to new unified API
            # Calculate data_size_bytes for EP combine based on batch information
            # Use the first batch as representative (all EP batches should have similar size)
            representative_batch = list(ep_wait_room["batches"].values())[0]
            total_tokens = representative_batch.total_num_tokens

            # Get model embedding dimension from replica config
            model_config = self._config.replica_config.model_config
            hidden_size = model_config.embedding_dim

            # Calculate data size: tokens × hidden_size × 2 bytes (float16)
            data_size_bytes = total_tokens * hidden_size * 2

            use_step3_alltoall = (
                model_config is not None
                and model_config.model_type == "step3_text"
                and expected_ep_size > 1
            )
            if use_step3_alltoall:
                # EP alltoall combine phase
                ep_collective_exec_time_ms = self._predictor.predict_alltoall_time(
                    data_size_bytes=data_size_bytes,
                    num_devices=expected_ep_size,
                    cluster_type=self._cluster_type,
                    comm_domain="EP",
                )
            else:
                ep_collective_exec_time_ms = self._predictor.predict_allgather_time(
                    data_size_bytes=data_size_bytes,
                    num_devices=expected_ep_size,
                    cluster_type=self._cluster_type,
                    comm_domain="EP",
                )

            ep_collective_sync_time = max(ep_wait_room["arrival_times"].values())
            ep_collective_exec_time = ep_collective_exec_time_ms * 1e-3
            collective_event_time = ep_collective_sync_time + ep_collective_exec_time

            logger.info(
                f"[DEBUG] Creating EPAllToAllCombineCollectiveEvent at time={collective_event_time:.3f}s, "
                f"sync_time={ep_collective_sync_time:.3f}s, exec_time={ep_collective_exec_time_ms:.3f}ms, "
                f"data_size={data_size_bytes} bytes ({total_tokens} tokens × {hidden_size} hidden_size)"
            )

            return [
                EPAllToAllCombineCollectiveEvent(
                    collective_event_time, replica_id, stage_id, batch_global_id
                )
            ]
        else:
            logger.info(f"[DEBUG] Waiting for more EP replicas: {len(ep_wait_room['batches'])}/{expected_ep_size}")

        return []

    def on_ep_alltoall_combine_collective_schedule(
        self, time: float, replica_id: int, stage_id: int, batch_global_id: int, metrics_store
    ):
        """
        Handle EP AllToAll combine collective synchronization in decode-ffn cluster.

        This method aggregates results from all EP replicas and creates M2N transfer
        events to send the aggregated batch back to decode-attn cluster.

        Args:
            time: Synchronized time when all EP replicas have reached this point
            replica_id: ID of the replica
            stage_id: Pipeline stage ID
            batch_global_id: Global ID of the batch
            metrics_store: Metrics store for recording performance data
        """
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        logger.info(
            f"[DEBUG] on_ep_alltoall_combine_collective_schedule called: time={time:.3f}s, "
            f"replica_id={replica_id}, stage_id={stage_id}, batch_global_id={batch_global_id}"
        )

        # Get the synchronized batches and clean up waiting room
        ep_wait_room = self._ep_allgather_waiting_room[replica_id][stage_id].pop(batch_global_id)
        ep_batches = ep_wait_room["batches"]

        logger.info(f"[DEBUG] Retrieved {len(ep_batches)} EP batches from waiting room: "
                   f"ep_ids={list(ep_batches.keys())}")

        # Phase 3 Task 2: Validate token conservation across EP batches
        # In EP parallelism, each EP replica processes a SUBSET of experts for the SAME tokens
        # The total_num_tokens in EPBatchGroup already includes the router_topk effect
        # (calculated in _distribute_tokens_within_ep_replica as: original_tokens * router_topk * ratio)
        # So we validate that per_expert_tokens sums to total_num_tokens (no additional multiplication)
        for ep_id, ep_batch in ep_batches.items():
            if hasattr(ep_batch, 'per_expert_tokens') and ep_batch.per_expert_tokens:
                # Each EP batch should conserve tokens independently
                # NOTE: Do NOT multiply by router_topk here - total_num_tokens already accounts for it
                expected_tokens = ep_batch.total_num_tokens
                self._validate_token_conservation(
                    input_tokens=expected_tokens,
                    per_expert_tokens=ep_batch.per_expert_tokens,
                    context=(
                        f"EP AllToAll combine collective - EP batch (cluster={self._cluster_type.name}, "
                        f"replica={replica_id}, stage={stage_id}, ep_id={ep_id}, batch_global_id={batch_global_id})"
                    ),
                )
                logger.info(f"[TOKEN_CONSERVATION] Validated EP batch {ep_id}: {expected_tokens} tokens across {len(ep_batch.per_expert_tokens)} experts")

        # CRITICAL FIX: Release stage scheduler busy state for all EP replicas
        # This is essential because EP workflow bypasses BatchStageEndEvent, so we must
        # manually call on_stage_end() to allow subsequent batches to be processed
        logger.info(f"[CRITICAL_FIX] Releasing stage scheduler busy state for all EP replicas")
        for ep_id in ep_batches.keys():
            stage_scheduler = self.get_dp_replica_stage_scheduler(replica_id, ep_id, stage_id)
            stage_scheduler.on_stage_end()
            logger.info(f"[CRITICAL_FIX] Released busy state for replica {replica_id}, ep_id {ep_id}, stage {stage_id}")

        # CRITICAL FIX: Decrement _num_running_batches for each EP replica scheduler
        # EP workflow bypasses ClusterBatchEndEvent, so we must manually decrement here
        # Each EP replica scheduler incremented its counter when scheduling the EPBatchGroup
        logger.info(f"[CRITICAL_FIX] Decrementing _num_running_batches for all EP replica schedulers")
        for ep_id in ep_batches.keys():
            replica_scheduler = self.get_dp_replica_scheduler(replica_id, ep_id)
            replica_scheduler.decrement_num_running_batches()
            logger.info(f"[CRITICAL_FIX] Decremented _num_running_batches for replica {replica_id}, ep_id {ep_id}, "
                       f"new count={replica_scheduler.num_running_batches}")

        # Release activation memory accounting for DECODE_FFN EP batches
        for ep_id, ep_batch in ep_batches.items():
            activation_bytes = getattr(ep_batch, "activation_bytes", 0)
            if activation_bytes:
                replica_scheduler = self.get_dp_replica_scheduler(replica_id, ep_id)
                replica_scheduler.release_activation_memory_bytes(int(activation_bytes))
                # Record memory usage after release to capture baseline recovery.
                metrics_store.on_replica_schedule(
                    time,
                    replica_id,
                    replica_scheduler.memory_usage_percent,
                    self._cluster_type,
                    dp_id=ep_id,
                )

        # Instead of aggregating the batch, pick raw batches from
        # _raw_batch_waiting_for_m2n_back using a canonical EP lane.
        if not ep_batches:
            raise ValueError(
                "EP all-to-all collective reached with empty ep_batches"
            )
        canonical_ep_id = min(ep_batches.keys())
        raw_batch_ids = ep_batches[canonical_ep_id].source_batch_ids
        for ep_id, ep_batch in ep_batches.items():
            if ep_batch.source_batch_ids != raw_batch_ids:
                raise ValueError(
                    f"source_batch_ids mismatch: ep_id={ep_id} has "
                    f"{ep_batch.source_batch_ids}, expected {raw_batch_ids}"
                )

        # Calculate the actual FFN execution time from EPBatchGroup
        # Each EP batch stores stage execution_time (seconds) set by ReplicaStageScheduleEvent
        # We use the maximum execution time across all EP batches (they should be similar)
        ep_execution_times = []
        for ep_id, ep_batch in ep_batches.items():
            if hasattr(ep_batch, 'execution_time') and ep_batch.execution_time > 0:
                ep_execution_times.append(ep_batch.execution_time)
                logger.info(f"[FFN-EXEC-TIME] EP batch {ep_id} execution_time={ep_batch.execution_time:.6f}s")

        # Use the maximum EP execution time; fail fast if missing to avoid silent metric corruption.
        if ep_execution_times:
            ffn_execution_time = max(ep_execution_times)
            logger.info(f"[FFN-EXEC-TIME] Using EP execution time: {ffn_execution_time:.6f}s (max of {len(ep_execution_times)} EP batches)")
        else:
            raise ValueError(f"Missing ep_execution_times")

        for ep_id, ep_batch in ep_batches.items():
            metrics_store.flush_frontier_stage_batch_ledger_row(
                time=time,
                batch_id=ep_batch.id,
                replica_id=replica_id,
                stage_id=stage_id,
                cluster_type=self._cluster_type,
                dp_id=ep_id,
                completion_source="ep_alltoall_combine_collective",
            )

        # Record batch-level DECODE_FFN metrics exactly once per raw batch.
        # EP lanes are synchronized here, so we emit metrics from the canonical lane.
        metrics_lane_id = canonical_ep_id
        memory_usage_percent = max(
            self.get_dp_replica_scheduler(replica_id, ep_id).memory_usage_percent
            for ep_id in ep_batches.keys()
        )

        m2n_events = []
        for bid in raw_batch_ids:
            raw = self._raw_batch_waiting_for_m2n_back.pop(bid, None)
            if raw is None:
                raise ValueError(f"Missing raw batch for id={bid} in _raw_batch_waiting_for_m2n_back")

            # ISSUE-007 DIAGNOSTIC: Log batch attributes before F→A transfer
            logger.info(
                f"[ISSUE-007][F2A][CREATE] batch_id={raw.id}, "
                f"decode_attn_original_replica_id={getattr(raw, 'decode_attn_original_replica_id', 'MISSING')}, "
                f"decode_attn_original_dp_id={getattr(raw, 'decode_attn_original_dp_id', 'MISSING')}"
            )

            # Record DECODE_FFN execution time for each request using the synchronized
            # stage execution time from EP batches.
            for request, runtime_epoch in zip(
                raw.requests,
                raw.request_runtime_epochs,
            ):
                if int(getattr(request, "runtime_epoch", 0)) != int(runtime_epoch):
                    continue
                request.on_batch_stage_end(
                    time, ffn_execution_time, ffn_execution_time, self._cluster_type
                )
            logger.info(
                f"[FFN-EXEC-TIME] Recorded execution time for batch {bid}: "
                f"execution_time={ffn_execution_time:.6f}s, num_requests={len(raw.requests)}"
            )

            metrics_store.on_batch_end(
                time,
                raw,
                replica_id,
                memory_usage_percent,
                self._cluster_type,
                metrics_lane_id,
            )

            # update time
            raw.time = time
            # Create M2N transfer events to send aggregated batch back to decode-attn
            m2n_events.extend(self._create_m2n_transfer_events_for_aggregated_batch(raw, raw.time))
        logger.info(f"[DEBUG] Created {len(m2n_events)} M2N transfer events: "
                    f"{[event.event_type.name if event and hasattr(event, 'event_type') and event.event_type else 'Unknown' for event in m2n_events]}")

        # Immediately trigger scheduling on all EP (dp) lanes for continued progress
        from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
        schedule_events = [
            ReplicaStageScheduleEvent(time, replica_id, stage_id, self._cluster_type, ep_id)
            for ep_id in ep_batches.keys()
        ]

        return m2n_events + schedule_events

    def _create_m2n_transfer_events_for_aggregated_batch(self, batch, current_time):
        """Disaggregated M2N transfer events are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _get_current_layer_id_from_batch(self, batch: Batch) -> int:
        if not batch.requests:
            raise ValueError(
                "_get_current_layer_id_from_batch: batch.requests is empty"
            )
        for request in batch.requests:
            if not request.completed:
                return request.completed_layer_count
        return batch.requests[0].completed_layer_count

    # Phase 2.5: Removed deprecated on_moe_collective_schedule() method
    # Old MoE synchronization architecture is no longer supported
    # Current architecture uses EP-based synchronization (EPAllToAllCombineReadyEvent/EPAllToAllCombineCollectiveEvent)

    """
    Layer 0: attn (include tp allreduce) → sync → moe_comm → moe_comp → sync → moe_comm
    Layer 1: attn → sync → moe_comm → moe_comp → sync → moe_comm
    ...
    Layer N-1: attn → sync → moe_comm → moe_comp → sync → moe_comm
    Pipeline: pipeline_time
    """
    def on_prefill_sync(self, time: float, replica_id: int, stage_id: int, batch: Batch,
                       dp_id: int, sync_stage: str, layer_id: int, stage_execution_time: float):
        """
        Handle prefill cluster synchronization points.

        Args:
            time: Current simulation time
            replica_id: ID of the replica
            stage_id: Pipeline stage ID
            batch: The batch being processed
            dp_id: Data parallel replica ID within the replica
            sync_stage: "pre_moe" or "post_moe"
            layer_id: Current layer being processed
            stage_execution_time: Execution time for this stage
        """
        # Guard: This method should only be called for MoE models
        if self._prefill_sync_waiting_room is None:
            raise ValueError(
                f"on_prefill_sync called for non-MoE model in PREFILL cluster. "
                f"Dense models should not use sync events."
            )

        from frontier.events.prefill_sync_collective_event import PrefillSyncCollectiveEvent
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        batch_global_id = batch.global_id
        # DP waiting room for prefill cluster
        sync_wait_room = self._prefill_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id][sync_stage]

        # CRITICAL FIX: If this is an idle batch and there's already a real batch
        # in the waiting room for this dp_id, skip the idle batch.
        if batch.is_idle and dp_id in sync_wait_room["batches"]:
            existing_batch = sync_wait_room["batches"][dp_id]
            if not existing_batch.is_idle:
                logger.info(
                    f"[PREFILL_SYNC][IDLE_SKIP] Skipping idle batch {batch.id} for dp_id={dp_id} "
                    f"because real batch {existing_batch.id} already exists in waiting room"
                )
                return []

        sync_wait_room["batches"][dp_id] = batch
        sync_wait_room["arrival_times"][dp_id] = time

        # Get the replica to check dp_size
        replica = self.get_replica(replica_id)
        dp_size = replica.dp_size

        arrived = len(sync_wait_room["batches"])
        logger.info(
            f"[PREFILL_SYNC][ARRIVAL] stage={stage_id}, layer={layer_id}, sync_stage={sync_stage}, "
            f"replica={replica_id}, dp_id={dp_id}, arrived={arrived}/{dp_size}, is_idle={batch.is_idle}, t={time:.6f}s"
        )

        # Check if all DP replicas in this replica have arrived
        if arrived == dp_size:
            # Synchronize to the maximum time across all DP replicas
            sync_time = max(sync_wait_room["arrival_times"].values())
            logger.info(
                f"[PREFILL_SYNC][COLLECTIVE_READY] Scheduling PrefillSyncCollectiveEvent at t={sync_time:.6f}s "
                f"for batch_global_id={batch_global_id}, stage={stage_id}, layer={layer_id}, sync_stage={sync_stage}"
            )
            return [
                PrefillSyncCollectiveEvent(
                    sync_time,
                    replica_id,
                    stage_id,
                    batch_global_id,
                    sync_stage,
                    layer_id,
                    cluster_type=self._cluster_type,
                )
            ]
        else:
            # Not all DP replicas have arrived yet.
            # For pre_moe sync, create idle batches for missing DP lanes so the
            # collective can complete when num_requests < dp_size.
            if sync_stage == "pre_moe" and not batch.is_idle:
                arrived_dp_ids = set(sync_wait_room["batches"].keys())
                all_dp_ids = set(range(dp_size))
                missing_dp_ids = all_dp_ids - arrived_dp_ids

                if missing_dp_ids:
                    logger.info(
                        f"[PREFILL_SYNC][IDLE_CREATE] Creating idle batches for missing DP replicas: "
                        f"missing_dp_ids={sorted(missing_dp_ids)}, arrived_dp_ids={sorted(arrived_dp_ids)}"
                    )

                    from frontier.events.prefill_sync_event import PrefillSyncEvent

                    idle_batch_events = []
                    for missing_dp_id in sorted(missing_dp_ids):
                        if missing_dp_id in sync_wait_room["batches"]:
                            logger.info(
                                f"[PREFILL_SYNC][IDLE_SKIP] Skipping idle batch creation for dp_id={missing_dp_id} "
                                f"(already exists in waiting room)"
                            )
                            continue

                        idle_batch = Batch(
                            replica_id=replica_id,
                            requests=[],
                            num_tokens=[],
                            is_idle=True,
                            is_moe=batch.is_moe,
                        )
                        idle_batch.set_global_id(batch_global_id)

                        logger.info(
                            f"[PREFILL_SYNC][IDLE_CREATE] Created idle batch {idle_batch.id} for "
                            f"replica={replica_id}, dp_id={missing_dp_id}, "
                            f"batch_global_id={batch_global_id}, layer={layer_id}"
                        )

                        idle_batch_events.append(
                            PrefillSyncEvent(
                                time=time,
                                replica_id=replica_id,
                                stage_id=stage_id,
                                batch=idle_batch,
                                dp_id=missing_dp_id,
                                sync_stage=sync_stage,
                                layer_id=layer_id,
                                stage_execution_time=0.0,
                                cluster_type=self._cluster_type,
                            )
                        )

                    return idle_batch_events

            logger.info(
                f"[PREFILL_SYNC][WAITING] Not all DP replicas arrived yet: "
                f"arrived={arrived}/{dp_size}, batch_global_id={batch_global_id}, "
                f"layer={layer_id}, sync_stage={sync_stage}"
            )

        return []

    def on_prefill_sync_collective(self, time: float, replica_id: int, stage_id: int,
                                  batch_global_id: int, sync_stage: str, layer_id: int, metrics_store):
        """
        Handle collective synchronization completion in prefill cluster.

        This method implements the exact layer-by-layer processing flow:
        - pre_moe sync: execute get_moe_comm_time() + get_moe_comp_time(), then schedule post_moe sync
        - post_moe sync: execute get_moe_comm_time(), then continue to next layer or finish

        Args:
            time: Synchronized time when all DP replicas have reached this point
            replica_id: ID of the replica
            stage_id: Pipeline stage ID
            batch_global_id: Global ID of the batch
            sync_stage: "pre_moe" or "post_moe"
            layer_id: Current layer being processed
            metrics_store: Metrics store for recording performance data
        """
        from frontier.events.batch_stage_end_event import BatchStageEndEvent
        from frontier.events.prefill_sync_event import PrefillSyncEvent
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Check if this sync_stage has already been processed by another replica
        # This can happen when multiple replicas reach the same sync point and each creates a PrefillSyncCollectiveEvent
        if sync_stage not in self._prefill_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id]:
            logger.debug(
                f"[PREFILL_SYNC][COLLECTIVE_SKIP] sync_stage={sync_stage} already processed for "
                f"replica={replica_id}, stage={stage_id}, batch_global_id={batch_global_id}, layer={layer_id}"
            )
            return []

        # Get the synchronized batches and clean up waiting room
        sync_wait_room = self._prefill_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id].pop(sync_stage)
        dp_batches = sync_wait_room["batches"]

        try:
            dp_keys = list(dp_batches.keys())
        except Exception:
            dp_keys = []
        logger.info(
            f"[PREFILL_SYNC][COLLECTIVE] ENTER: t={time:.6f}s, replica={replica_id}, stage={stage_id}, "
            f"layer={layer_id}, sync_stage={sync_stage}, batch_global_id={batch_global_id}, dp_keys={dp_keys}, "
            f"dp_batches_type={type(dp_batches).__name__}"
        )

        events = []

        if sync_stage == "pre_moe":
            # After pre_moe sync, execute: DP Gather + MoE computation + DP Scatter
            # Then schedule post_moe sync

            # Get DP size and aggregate tokens for observability.
            # Timing composition comes from canonical single-layer ExecutionTime.
            replica = self.get_replica(replica_id)
            dp_size = replica.dp_size
            total_global_tokens = 0
            for dp_id, batch in dp_batches.items():
                if not batch.is_idle:
                    total_global_tokens += batch.total_num_tokens

            logger.info(
                f"[GLOBAL_TOKEN_AGGREGATION] total_global_tokens={total_global_tokens}, "
                f"dp_size={dp_size}, batches={[(dp_id, batch.total_num_tokens, batch.is_idle) for dp_id, batch in dp_batches.items()]}"
            )

            # Use one non-idle batch to derive shared layer timings for all DP lanes.
            sample_batch = next((b for b in dp_batches.values() if not b.is_idle), None)
            if sample_batch is None:
                raise ValueError(
                    f"pre_moe collective has no non-idle batch for replica={replica_id}, "
                    f"stage={stage_id}, batch_global_id={batch_global_id}, layer={layer_id}"
                )

            stage_scheduler = self.get_dp_replica_stage_scheduler(replica_id, 0, stage_id)
            execution_time_predictor = stage_scheduler._execution_time_predictor

            # Canonical post-attention already includes MoE computation + DP comm semantics.
            # Predictor returns milliseconds for single-layer components.
            execution_time = execution_time_predictor.predict_stage_execution_time(
                sample_batch,
                stage_id,
                cluster_type=self._cluster_type,
                num_layers=1,
                layer_id=layer_id,
            )
            moe_stage_time = execution_time.get_single_layer_post_attention_time() * 1e-3
            dp_input_allreduce_time = (
                execution_time.get_single_layer_dp_input_allreduce_time() * 1e-3
            )
            dp_output_allreduce_time = (
                execution_time.get_single_layer_dp_output_allreduce_time() * 1e-3
            )

            logger.info(
                f"[MoE_TIME_BREAKDOWN] post_attention={moe_stage_time:.6f}s, "
                f"dp_input_allreduce={dp_input_allreduce_time:.6f}s, "
                f"dp_output_allreduce={dp_output_allreduce_time:.6f}s"
            )

            for dp_id, batch in dp_batches.items():
                # Schedule post_moe sync for this layer
                events.append(
                    PrefillSyncEvent(
                        time + moe_stage_time,
                        replica_id,
                        stage_id,
                        batch,
                        dp_id,
                        "post_moe",
                        layer_id,
                        moe_stage_time,
                        cluster_type=self._cluster_type,
                    )
                )

            # for dp_id, batch in batches.items():
            #     execution_time = self._predictor.get_execution_time(batch, stage_id, self._cluster_type)

            #     # Calculate MoE stage time: ONLY pre_moe_comm + moe_comp
            #     moe_comm_time = execution_time.get_single_layer_moe_comm_time()
            #     moe_comp_time = execution_time.get_single_layer_moe_comp_time()
            #     moe_stage_time = moe_comm_time + moe_comp_time

            #     # Schedule post_moe sync for this layer
            #     events.append(PrefillSyncEvent(
            #         time + moe_stage_time, replica_id, stage_id, batch,
            #         dp_id, "post_moe", layer_id, moe_stage_time
            #     ))

        elif sync_stage == "post_moe":
            # post_moe is a synchronization boundary. Model execution for this layer has
            # already been accounted in pre_moe; only layer transition / pipeline handoff
            # remains after this collective.
            sample_batch = next((b for b in dp_batches.values() if not b.is_idle), None)
            if sample_batch is None:
                logger.warning(
                    f"[PREFILL_SYNC][COLLECTIVE] post_moe has no non-idle batch for "
                    f"replica={replica_id}, stage={stage_id}, batch_global_id={batch_global_id}, layer={layer_id}"
                )
                return events

            # Use one non-idle batch to derive shared layer timings for all DP lanes.
            execution_time = self._predictor.predict_stage_execution_time(
                sample_batch,
                stage_id,
                cluster_type=self._cluster_type,
                num_layers=1,  # Single-layer granularity for prefill sync
                layer_id=layer_id,
            )

            # IMPORTANT: execution_time here is a single-layer prediction (num_layers=1)
            # for component extraction, so it cannot be used as the stage layer count.
            num_layers = self._predictor._num_layers_per_pipeline_stage
            if num_layers < 1:
                raise ValueError(
                    f"Invalid prefill stage layer count: num_layers={num_layers} "
                    f"(replica={replica_id}, stage={stage_id})"
                )

            if layer_id < num_layers - 1:
                # Not the last layer, continue to next layer by paying next-layer attention.
                next_layer_id = layer_id + 1
                next_layer_execution_time = self._predictor.predict_stage_execution_time(
                    sample_batch,
                    stage_id,
                    cluster_type=self._cluster_type,
                    num_layers=1,
                    layer_id=next_layer_id,
                )
                attention_time = (
                    next_layer_execution_time.get_single_layer_attention_time() * 1e-3
                )
                total_time_to_next_sync = attention_time

                for dp_id, batch in dp_batches.items():
                    if batch.is_idle:
                        logger.info(
                            f"[PREFILL_SYNC][IDLE_SKIP] Skip next-layer pre_moe scheduling for idle batch {batch.id} "
                            f"(replica={replica_id}, dp_id={dp_id}, layer={layer_id})"
                        )
                        continue
                    events.append(
                        PrefillSyncEvent(
                            time + total_time_to_next_sync,
                            replica_id,
                            stage_id,
                            batch,
                            dp_id,
                            "pre_moe",
                            next_layer_id,
                            total_time_to_next_sync,
                            cluster_type=self._cluster_type,
                        )
                    )
            else:
                # Last layer completed, proceed to pipeline communication.
                # Idle batches are synthetic synchronization placeholders and should not
                # create stage-end / kv-transfer events in PREFILL.
                full_stage_execution_time = self._predictor.predict_stage_execution_time(
                    sample_batch,
                    stage_id,
                    cluster_type=self._cluster_type,
                    num_layers=num_layers,
                )
                for dp_id, batch in dp_batches.items():
                    if batch.is_idle:
                        logger.info(
                            f"[PREFILL_SYNC][IDLE_SKIP] Skip final stage-end for idle batch {batch.id} "
                            f"(replica={replica_id}, dp_id={dp_id}, layer={layer_id})"
                        )
                        continue

                    stage_scheduler = self.get_dp_replica_stage_scheduler(replica_id, dp_id, stage_id)
                    is_last_stage = stage_scheduler.is_last_stage
                    pipeline_time = full_stage_execution_time.pipeline_time * 1e-3

                    # Final transition must preserve both pipeline handoff and any
                    # active CPU overhead already modeled in the stage payload.
                    cpu_overhead_time = max(
                        full_stage_execution_time.total_time
                        - full_stage_execution_time.model_time,
                        0.0,
                    )
                    total_final_time = pipeline_time + cpu_overhead_time

                    # Create batch stage for metrics
                    batch_stage, _ = stage_scheduler.predict_and_create_stage(batch, skip_get_execution_time=True)

                    # Use original start time for metrics
                    original_start_time = getattr(batch, '_prefill_stage_start_time', time - execution_time.total_time)

                    # Schedule the batch stage with the original start time
                    batch_stage.on_schedule(original_start_time)

                    # Calculate actual execution time including synchronization overhead
                    actual_execution_time = time + total_final_time - original_start_time

                    # Override with correct values:
                    # - execution_time: actual wall-clock time including sync overhead
                    # - model_execution_time: pure model computation time (no CPU overhead)
                    batch_stage.override_execution_time(actual_execution_time)
                    batch_stage.override_model_execution_time(full_stage_execution_time.model_time)

                    # TODO: CHECK OVERIDE LOGIC AND METRIC LOGIC HERE
                    # Create a corrected ExecutionTime object for metrics recording.
                    # For mixed-layer MoE models, augment trace-only dense MLP components
                    # from a representative dense layer so op-level traces include both
                    # dense and MoE FFN scopes.
                    corrected_execution_time = (
                        self._create_prefill_corrected_execution_time_for_metrics(
                            sample_batch,
                            stage_id,
                            full_stage_execution_time,
                            actual_execution_time,
                            original_start_time,
                        )
                    )

                    # Record metrics with correct start time and corrected execution time
                    metrics_store.on_replica_stage_schedule(
                        original_start_time, replica_id, stage_id, batch_stage, corrected_execution_time,
                        self._cluster_type, dp_id
                    )

                    # Schedule batch stage end
                    events.append(BatchStageEndEvent(
                        time + total_final_time, replica_id, stage_id, is_last_stage,
                        batch, batch_stage, self._cluster_type, dp_id
                    ))

                    # Check if KV cache transfer should be triggered
                    if self._should_trigger_kv_transfer(batch):
                        kv_transfer_events = self._create_kv_transfer_events(
                            time + total_final_time, batch, replica_id, dp_id
                        )
                        events.extend(kv_transfer_events)

                    # Note: _prefill_stage_start_time cleanup moved to BatchStageEndEvent
                    # to ensure proper detection of completed prefill sync batches

        return events

    def _create_prefill_corrected_execution_time_for_metrics(
        self,
        sample_batch: Batch,
        stage_id: int,
        original_execution_time,
        actual_execution_time_ms,
        original_start_time,
    ):
        """Build corrected prefill metrics payload and attach mixed-layer trace hints."""
        corrected_execution_time = self._create_corrected_execution_time_for_metrics(
            original_execution_time,
            actual_execution_time_ms,
            original_start_time,
        )

        dense_reference_execution_time = self._get_prefill_dense_reference_execution_time(
            sample_batch,
            stage_id,
        )
        if dense_reference_execution_time is None:
            return corrected_execution_time

        corrected_execution_time._trace_dense_mlp_layer_up_proj_execution_time = (
            dense_reference_execution_time._mlp_layer_up_proj_execution_time
        )
        corrected_execution_time._trace_dense_mlp_layer_act_execution_time = (
            dense_reference_execution_time._mlp_layer_act_execution_time
        )
        corrected_execution_time._trace_dense_mlp_layer_down_proj_execution_time = (
            dense_reference_execution_time._mlp_layer_down_proj_execution_time
        )
        corrected_execution_time._trace_dense_layer_id = (
            self._get_first_dense_layer_id_for_mixed_moe()
        )
        return corrected_execution_time

    def _get_first_dense_layer_id_for_mixed_moe(self) -> Optional[int]:
        """Return first dense FFN layer id for mixed-layer MoE models, else None."""
        config = getattr(self, "_config", None)
        replica_config = getattr(config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        if model_config is None:
            return None

        if not getattr(model_config, "is_moe", False):
            return None

        if not hasattr(model_config, "get_moe_layer_ids") or not hasattr(
            model_config, "num_layers"
        ):
            return None

        moe_layer_ids = set(model_config.get_moe_layer_ids())
        if len(moe_layer_ids) == 0:
            return None

        num_layers = int(model_config.num_layers)
        if len(moe_layer_ids) >= num_layers:
            return None

        for layer_id in range(num_layers):
            if layer_id not in moe_layer_ids:
                return layer_id

        return None

    def _get_prefill_dense_reference_execution_time(
        self,
        sample_batch: Batch,
        stage_id: int,
    ) -> Optional[ExecutionTime]:
        """Predict one dense layer execution for mixed-layer MoE trace completion."""
        dense_layer_id = self._get_first_dense_layer_id_for_mixed_moe()
        if dense_layer_id is None:
            return None

        dense_execution_time = self._predictor.predict_stage_execution_time(
            sample_batch,
            stage_id,
            cluster_type=self._cluster_type,
            num_layers=1,
            layer_id=dense_layer_id,
        )
        if dense_execution_time._is_moe:
            raise ValueError(
                f"Expected dense execution for layer_id={dense_layer_id}, "
                f"but predictor returned is_moe=True"
            )

        if (
            dense_execution_time._mlp_layer_up_proj_execution_time <= 0.0
            or dense_execution_time._mlp_layer_act_execution_time <= 0.0
            or dense_execution_time._mlp_layer_down_proj_execution_time <= 0.0
        ):
            raise ValueError(
                "Dense reference execution_time must provide positive mlp_up_proj/mlp_act/"
                "mlp_down_proj components"
            )

        return dense_execution_time

    def _create_corrected_execution_time_for_metrics(
        self,
        original_execution_time,
        actual_execution_time_ms,
        original_start_time,
    ):
        """Create corrected ExecutionTime payload used by metrics/trace emission."""
        from frontier.entities.execution_time import ExecutionTime

        corrected_execution_time = ExecutionTime(
            num_layers_per_pipeline_stage=1,  # Avoid double-counting in sync path.
            attention_rope_execution_time=original_execution_time._attention_rope_execution_time,
            attention_kv_cache_save_execution_time=original_execution_time._attention_kv_cache_save_execution_time,
            attention_decode_execution_time=original_execution_time._attention_decode_execution_time,
            attention_prefill_execution_time=original_execution_time._attention_prefill_execution_time,
            attention_layer_pre_proj_execution_time=original_execution_time._attention_layer_pre_proj_execution_time,
            attention_layer_post_proj_execution_time=original_execution_time._attention_layer_post_proj_execution_time,
            attn_norm_time=original_execution_time._attn_norm_time,
            mlp_norm_time=original_execution_time._mlp_norm_time,
            add_time=original_execution_time._add_time,
            add_attn_residual_time=original_execution_time._add_attn_residual_time,
            add_ffn_residual_time=original_execution_time._add_ffn_residual_time,
            tensor_parallel_communication_time=original_execution_time._tensor_parallel_communication_time,
            attn_tensor_parallel_allreduce_time=original_execution_time._attn_tensor_parallel_allreduce_time,
            moe_tensor_parallel_allreduce_time=original_execution_time._moe_tensor_parallel_allreduce_time,
            tensor_parallel_allgather_time=original_execution_time._tensor_parallel_allgather_time,
            share_expert_tensor_parallel_allreduce_time=original_execution_time._share_expert_tensor_parallel_allreduce_time,
            dp_input_allreduce_time=original_execution_time._dp_input_allreduce_time,
            dp_output_allreduce_time=original_execution_time._dp_output_allreduce_time,
            pipeline_parallel_communication_time=original_execution_time._pipeline_parallel_communication_time,
            expert_parallel_communication_time=original_execution_time._expert_parallel_communication_time,
            moe_gating_time=original_execution_time._moe_gating_time,
            moe_gating_linear_time=original_execution_time._moe_gating_linear_time,
            moe_gating_routing_topk_time=original_execution_time._moe_gating_routing_topk_time,
            moe_shuffling_time=original_execution_time._moe_shuffling_time,
            schedule_time=original_execution_time._schedule_time,
            sampler_e2e_time=original_execution_time._sampler_e2e_time,
            prepare_inputs_e2e_time=original_execution_time._prepare_inputs_e2e_time,
            pp_producer_send_path_runtime_time=original_execution_time._pp_producer_send_path_runtime_time,
            pp_receiver_head_runtime_time=original_execution_time._pp_receiver_head_runtime_time,
            pp_prefill_consumer_active_runtime_time=original_execution_time._pp_prefill_consumer_active_runtime_time,
            process_model_outputs_time=original_execution_time._process_model_outputs_time,
            ray_comm_time=original_execution_time._ray_comm_time,
            is_moe=original_execution_time._is_moe,
            mlp_layer_up_proj_execution_time=original_execution_time._mlp_layer_up_proj_execution_time,
            mlp_layer_down_proj_execution_time=original_execution_time._mlp_layer_down_proj_execution_time,
            mlp_layer_act_execution_time=original_execution_time._mlp_layer_act_execution_time,
            moe_grouped_gemm_time=original_execution_time._moe_grouped_gemm_time,
            share_expert_up_proj_time=original_execution_time._share_expert_up_proj_time,
            share_expert_down_proj_time=original_execution_time._share_expert_down_proj_time,
            share_expert_act_time=original_execution_time._share_expert_act_time,
            decode_draft_proposer_time=original_execution_time._decode_draft_proposer_time,
            mtp_terminal_overshoot_time=(
                original_execution_time._mtp_terminal_overshoot_time
            ),
        )

        return corrected_execution_time

    def _record_mtp_terminal_completion_delay(
        self,
        batch: Batch,
        terminal_delay_s: float,
    ) -> None:
        """Record terminal MTP tail work as post-first-token batch service."""
        delay_value = float(terminal_delay_s)
        if delay_value < 0.0:
            raise ValueError(
                f"terminal MTP completion delay must be >= 0, got={delay_value}"
            )
        if delay_value == 0.0:
            return

        metadata = getattr(batch, "spec_decode_metadata", None)
        if metadata is None:
            raise ValueError(
                "terminal MTP completion delay requires spec_decode_metadata"
            )
        terminal_rows = getattr(
            metadata,
            "terminal_overshoot_verify_tokens_per_request",
            None,
        )
        if terminal_rows is None:
            raise ValueError(
                "terminal MTP completion delay requires terminal overshoot rows"
            )
        if len(terminal_rows) != len(batch.requests):
            raise ValueError(
                "terminal overshoot row count mismatch: "
                f"rows={len(terminal_rows)}, requests={len(batch.requests)}"
            )

        has_terminal_rows = any(len(rows) > 0 for rows in terminal_rows)
        if not has_terminal_rows:
            raise ValueError(
                "positive terminal MTP completion delay has no active request rows"
            )
        request_ids_with_terminal_rows = [
            int(request.id)
            for request, rows in zip(batch.requests, terminal_rows)
            if len(rows) > 0
        ]
        if not request_ids_with_terminal_rows:
            raise ValueError(
                "positive terminal MTP completion delay has no request-local "
                "terminal rows"
            )

        # Terminal overshoot rows are generated only for requests that have
        # logically completed but still appear in the target-embedded MTP trace.
        # Do not smear that post-response trace work onto unrelated active
        # batchmates; vLLM clean request metrics do not extend those active
        # requests' response latency with another request's terminal rows.
        batch.add_spec_terminal_completion_delay(
            request_ids_with_terminal_rows,
            delay_value,
        )

    def _should_trigger_kv_transfer(self, batch: Batch) -> bool:
        """KV cache transfer is not available in the co-location-only release."""
        return False

    def _create_kv_transfer_events(
        self,
        time: float,
        batch: Batch,
        replica_id: int,
        dp_id: int
    ) -> List:
        """Disaggregated KV cache transfer events are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _create_virtual_global_batch(
        self,
        sample_batch: Batch,
        total_global_tokens: int,
        total_global_prefill_tokens: int,
    ) -> Batch:
        """
        Create a virtual global batch for MoE execution time prediction.

        In DP-based MoE processing, all DP replicas gather their tokens into a global buffer
        before MoE computation. This method creates a virtual batch that represents this
        global token set for accurate execution time prediction.

        Args:
            sample_batch: A sample batch from one DP replica (used for metadata)
            total_global_tokens: Total number of tokens across all DP replicas
            total_global_prefill_tokens: Aggregated prefill tokens across all
                non-idle DP participants. The virtual batch preserves this split
                so decode-only runtime paths keep their decode semantics.

        Returns:
            A virtual Batch object with total_global_tokens for execution time prediction

        Note:
            - The virtual batch reuses the sample_batch's replica_id and other metadata
            - Only the token count is modified to reflect the global aggregation
            - This batch should ONLY be used for execution time prediction, not actual processing
        """
        import copy
        from dataclasses import replace
        from frontier.entities.batch import DecodeCudaGraphMetadata

        # Create a shallow copy of the sample batch
        virtual_batch = copy.copy(sample_batch)

        # Override token count to reflect global aggregation
        # Use a single-element list with total tokens
        virtual_batch._num_tokens = [total_global_tokens]
        virtual_batch._total_num_tokens = total_global_tokens

        if total_global_prefill_tokens < 0 or total_global_prefill_tokens > total_global_tokens:
            raise ValueError(
                "Virtual global batch requires prefill tokens to be within the "
                f"aggregated token range, got total_global_prefill_tokens="
                f"{total_global_prefill_tokens}, total_global_tokens={total_global_tokens}"
            )

        # Preserve the aggregated prefill/decode split.
        # This is critical for decode CUDA graph semantics because pure-decode
        # batches must keep num_decode_tokens > 0 for launch-overhead stripping.
        virtual_batch._num_prefill_tokens = total_global_prefill_tokens

        # Decode CUDA Graph metadata is attached per scheduler-visible local lane.
        # A virtual global sync batch represents the DP-gathered token domain, so
        # it must not reuse one lane's local token count for MoE/communication
        # prediction. Preserve the runtime mode but rebase token-count fields to
        # the aggregated batch domain.
        metadata = getattr(virtual_batch, "decode_cuda_graph_metadata", None)
        if metadata is not None and total_global_tokens != sample_batch.total_num_tokens:
            if not isinstance(metadata, DecodeCudaGraphMetadata):
                raise TypeError(
                    "decode_cuda_graph_metadata must be DecodeCudaGraphMetadata "
                    f"when rebasing virtual global batch tokens, got {type(metadata).__name__}"
                )
            global_decode_tokens = total_global_tokens - total_global_prefill_tokens
            virtual_batch.decode_cuda_graph_metadata = replace(
                metadata,
                original_total_tokens=total_global_tokens,
                padded_total_tokens=total_global_tokens,
                original_decode_batch_size=global_decode_tokens,
                padded_decode_batch_size=global_decode_tokens,
            )

        # Keep other attributes unchanged (replica_id, requests, etc.)
        # These are used for metadata but not for token-based predictions

        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)
        logger.debug(
            f"[VIRTUAL-BATCH] Created virtual global batch: "
            f"sample_batch_id={sample_batch.id}, total_global_tokens={total_global_tokens}, "
            f"num_prefill_tokens={total_global_prefill_tokens}, "
            f"num_decode_tokens={total_global_tokens - total_global_prefill_tokens}"
        )

        return virtual_batch


    def _is_monolithic_decode_shared_domain_sync(self, batch: Batch) -> bool:
        """Return whether decode sync should use shared-domain EP lanes."""
        model_config = getattr(getattr(self, "_config", None), "replica_config", None)
        model_config = getattr(model_config, "model_config", None)
        model_is_moe = model_config is not None and model_config.is_moe
        if not model_is_moe:
            return False
        if self._cluster_type != ClusterType.MONOLITHIC:
            return False

        ep_size = getattr(self, "_replica_ep_size", None)
        if ep_size is None and hasattr(self, "_config"):
            ep_size = getattr(self._config.replica_config, "moe_expert_parallel_size", 1)
        ep_size = int(ep_size or 1)
        if ep_size <= 1:
            return False

        if batch.is_idle:
            return True
        return batch.num_prefill_tokens == 0 and batch.num_decode_tokens > 0

    def _get_decode_sync_participant_count(self, replica: Replica, batch: Batch) -> int:
        """Return the synchronization cardinality for the current decode sync event."""
        if self._is_monolithic_decode_shared_domain_sync(batch):
            participant_count = getattr(replica, "ep_size", None)
            if participant_count is None:
                participant_count = getattr(self, "_replica_ep_size", None)
            if participant_count is None and hasattr(self, "_config"):
                participant_count = getattr(
                    self._config.replica_config,
                    "moe_expert_parallel_size",
                    1,
                )
            participant_count = int(participant_count)
            if participant_count <= 0:
                raise ValueError(
                    f"Invalid shared-domain decode sync participant_count={participant_count}"
                )
            return participant_count

        participant_count = int(replica.dp_size)
        if participant_count <= 0:
            raise ValueError(
                f"Invalid decode sync dp_size={participant_count} for replica={replica.id}"
            )
        return participant_count

    def _get_decode_sync_wait_key(self, batch: Batch) -> int:
        if self._is_monolithic_decode_shared_domain_sync(batch):
            if batch.is_idle:
                return int(batch.global_id)
            if not hasattr(batch, "decode_sync_global_id"):
                raise ValueError(
                    "MONOLITHIC MoE decode shared-domain batch is missing "
                    "decode_sync_global_id; real decode batches must be created "
                    "through BaseReplicaScheduler._create_batch so lane-scoped "
                    "decode sync ids are assigned."
                )
            return int(batch.decode_sync_global_id)
        return int(batch.global_id)

    def _build_monolithic_decode_shared_domain_trace_execution_time(
        self,
        base_execution_time,
        related_wait_ms: float,
    ):
        """Clone trace payload and attach shared-domain wait as a separate trace component."""
        import copy

        trace_execution_time = copy.deepcopy(base_execution_time)
        merged_wait_ms = max(0.0, float(related_wait_ms))
        num_layers = max(
            1,
            int(getattr(base_execution_time, "_num_layers_per_pipeline_stage", 1)),
        )
        per_layer_wait_ms = merged_wait_ms / num_layers
        trace_execution_time._shared_domain_wait_merged_ms = merged_wait_ms
        trace_execution_time._trace_related_collective_waits = []
        if merged_wait_ms > 0.0:
            trace_execution_time._trace_related_collective_waits = [
                {
                    "op_name": "expert_parallel_allreduce",
                    "related_wait_ms": merged_wait_ms,
                    "per_layer_related_wait_ms": per_layer_wait_ms,
                    "collective_domain": "EP_SHARED_DOMAIN",
                    "scope_alignment_mode": "wait_inclusive",
                    "reason": "monolithic_decode_shared_domain_sync_wait",
                }
            ]
        return trace_execution_time

    def _accumulate_monolithic_decode_shared_domain_related_wait_ms(
        self,
        *,
        replica_id: int,
        stage_id: int,
        batch_global_id: int,
        sync_stage: str,
        sync_wait_room: dict,
    ) -> float:
        """Accumulate decode shared-domain wait from sync arrival skew."""
        if sync_stage != "post_moe":
            return 0.0

        arrival_times = sync_wait_room.get("arrival_times")
        if not isinstance(arrival_times, dict) or not arrival_times:
            return 0.0

        sync_time = max(float(arrival_time) for arrival_time in arrival_times.values())
        related_wait_ms = 0.0
        for arrival_time in arrival_times.values():
            wait_s = max(0.0, sync_time - float(arrival_time))
            related_wait_ms += wait_s * 1e3

        key = (int(replica_id), int(stage_id), int(batch_global_id))
        self._decode_shared_domain_related_wait_ms_by_batch[key] += related_wait_ms
        return related_wait_ms

    def _pop_monolithic_decode_shared_domain_related_wait_ms(
        self,
        *,
        replica_id: int,
        stage_id: int,
        batch_global_id: int,
    ) -> float:
        """Pop accumulated decode shared-domain wait for one batch lifecycle."""
        key = (int(replica_id), int(stage_id), int(batch_global_id))
        related_wait_ms = float(
            self._decode_shared_domain_related_wait_ms_by_batch.pop(key, 0.0) or 0.0
        )
        return max(0.0, related_wait_ms)

    def on_decode_sync(self, time: float, replica_id: int, stage_id: int, batch: Batch,
                      dp_id: int, sync_stage: str, layer_id: int, stage_execution_time: float):
        """
        Handle DECODE cluster synchronization points.

        Similar to on_prefill_sync(), this method handles DP synchronization in the
        unified DECODE cluster when MoE is enabled.

        Args:
            time: Current simulation time
            replica_id: ID of the replica
            stage_id: Pipeline stage ID
            batch: The batch being processed
            dp_id: Data parallel replica ID within the replica
            sync_stage: "pre_moe" or "post_moe"
            layer_id: Current layer being processed
            stage_execution_time: Execution time for this stage
        """
        if self._decode_sync_waiting_room is None:
            raise ValueError(
                f"on_decode_sync called for non-MoE model in DECODE cluster. "
                f"Dense models should not use sync events."
            )

        from frontier.events.decode_sync_collective_event import DecodeSyncCollectiveEvent
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        batch_global_id = self._get_decode_sync_wait_key(batch)
        sync_wait_room = self._decode_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id][sync_stage]

        if batch.is_idle and dp_id in sync_wait_room["batches"]:
            existing_batch = sync_wait_room["batches"][dp_id]
            if not existing_batch.is_idle:
                logger.info(
                    f"[IDLE_BATCH][SKIP] Skipping idle batch {batch.id} for dp_id={dp_id} "
                    f"because real batch {existing_batch.id} already exists in waiting room"
                )
                return []

        sync_wait_room["batches"][dp_id] = batch
        sync_wait_room["arrival_times"][dp_id] = time

        replica = self.get_replica(replica_id)
        participant_count = self._get_decode_sync_participant_count(replica, batch)
        participant_domain = (
            "EP_SHARED_DOMAIN"
            if self._is_monolithic_decode_shared_domain_sync(batch)
            else "DP"
        )

        arrived = len(sync_wait_room["batches"])

        request_ids = [req.id for req in batch.requests] if batch.requests else []
        waiting_room_key = f"[{replica_id}][{stage_id}][{batch_global_id}][{layer_id}][{sync_stage}]"
        participant_ids_in_room = list(sync_wait_room["batches"].keys())
        debug_msg = (
            f"[DECODE_SYNC][ARRIVAL] batch_id={batch.id}, global_id={batch_global_id}, "
            f"requests={request_ids}, replica={replica_id}, dp_id={dp_id}, "
            f"stage={stage_id}, layer={layer_id}, sync_stage={sync_stage}, "
            f"arrived={arrived}/{participant_count}, participant_domain={participant_domain}, "
            f"participant_ids_in_room={participant_ids_in_room}, "
            f"waiting_room_key={waiting_room_key}, is_idle={batch.is_idle}, t={time:.6f}s"
        )
        logger.info(debug_msg)

        if arrived == participant_count:
            sync_time = max(sync_wait_room["arrival_times"].values())
            logger.info(
                f"[DECODE_SYNC][COLLECTIVE_READY] Scheduling DecodeSyncCollectiveEvent at t={sync_time:.6f}s "
                f"for batch_global_id={batch_global_id}, stage={stage_id}, layer={layer_id}, "
                f"sync_stage={sync_stage}, participant_domain={participant_domain}"
            )
            return [
                DecodeSyncCollectiveEvent(
                    sync_time,
                    replica_id,
                    stage_id,
                    batch_global_id,
                    sync_stage,
                    layer_id,
                    cluster_type=self._cluster_type,
                )
            ]

        if (
            sync_stage == "pre_moe"
            and not batch.is_idle
            and self._is_monolithic_decode_shared_domain_sync(batch)
        ):
            arrived_participant_ids = set(sync_wait_room["batches"].keys())
            all_participant_ids = set(range(participant_count))
            missing_participant_ids = all_participant_ids - arrived_participant_ids

            if missing_participant_ids:
                logger.info(
                    f"[DECODE_SYNC][IDLE_COMPACT] Compacting missing shared-domain lanes into waiting room: "
                    f"missing_participant_ids={sorted(missing_participant_ids)}, "
                    f"arrived_participant_ids={sorted(arrived_participant_ids)}, "
                    f"participant_domain={participant_domain}"
                )

                for missing_participant_id in sorted(missing_participant_ids):
                    if missing_participant_id in sync_wait_room["batches"]:
                        logger.info(
                            f"[DECODE_SYNC][IDLE_COMPACT][SKIP] Lane {missing_participant_id} already exists in waiting room"
                        )
                        continue

                    idle_batch = Batch(
                        replica_id=replica_id,
                        requests=[],
                        num_tokens=[],
                        is_idle=True,
                        is_moe=self._config.replica_config.model_config.is_moe,
                    )
                    idle_batch.set_global_id(batch_global_id)
                    sync_wait_room["batches"][missing_participant_id] = idle_batch
                    sync_wait_room["arrival_times"][missing_participant_id] = time

                    logger.info(
                        f"[DECODE_SYNC][IDLE_COMPACT] Inserted idle batch {idle_batch.id} for "
                        f"replica={replica_id}, dp_id={missing_participant_id}, "
                        f"batch_global_id={batch_global_id}, layer={layer_id}, "
                        f"participant_domain={participant_domain}"
                    )

                compact_arrived = len(sync_wait_room["batches"])
                if compact_arrived != participant_count:
                    raise ValueError(
                        f"Shared-domain decode idle compaction produced arrived={compact_arrived} "
                        f"but expected participant_count={participant_count} "
                        f"for replica={replica_id}, stage={stage_id}, batch_global_id={batch_global_id}, "
                        f"layer={layer_id}, sync_stage={sync_stage}"
                    )

                sync_time = max(sync_wait_room["arrival_times"].values())
                logger.info(
                    f"[DECODE_SYNC][COLLECTIVE_READY][IDLE_COMPACT] Scheduling DecodeSyncCollectiveEvent at "
                    f"t={sync_time:.6f}s for batch_global_id={batch_global_id}, stage={stage_id}, "
                    f"layer={layer_id}, sync_stage={sync_stage}, participant_domain={participant_domain}"
                )
                return [
                    DecodeSyncCollectiveEvent(
                        sync_time,
                        replica_id,
                        stage_id,
                        batch_global_id,
                        sync_stage,
                        layer_id,
                        cluster_type=self._cluster_type,
                    )
                ]

        if sync_stage == "pre_moe" and not batch.is_idle:
            arrived_participant_ids = set(sync_wait_room["batches"].keys())
            all_participant_ids = set(range(participant_count))
            missing_participant_ids = all_participant_ids - arrived_participant_ids

            if missing_participant_ids:
                logger.info(
                    f"[IDLE_BATCH] Creating idle batches for missing decode sync participants: "
                    f"missing_participant_ids={sorted(missing_participant_ids)}, "
                    f"arrived_participant_ids={sorted(arrived_participant_ids)}, "
                    f"participant_domain={participant_domain}"
                )

                from frontier.events.decode_sync_event import DecodeSyncEvent
                idle_batch_events = []

                for missing_participant_id in sorted(missing_participant_ids):
                    if missing_participant_id not in sync_wait_room["batches"]:
                        idle_batch = Batch(
                            replica_id=replica_id,
                            requests=[],
                            num_tokens=[],
                            is_idle=True,
                            is_moe=self._config.replica_config.model_config.is_moe,
                        )
                        idle_batch.set_global_id(batch_global_id)

                        logger.info(
                            f"[IDLE_BATCH] Created idle batch {idle_batch.id} for "
                            f"replica={replica_id}, dp_id={missing_participant_id}, "
                            f"batch_global_id={batch_global_id}, layer={layer_id}, "
                            f"participant_domain={participant_domain}"
                        )

                        idle_sync_event = DecodeSyncEvent(
                            time=time,
                            replica_id=replica_id,
                            stage_id=stage_id,
                            batch=idle_batch,
                            dp_id=missing_participant_id,
                            sync_stage=sync_stage,
                            layer_id=layer_id,
                            stage_execution_time=0.0,
                            cluster_type=self._cluster_type,
                        )
                        idle_batch_events.append(idle_sync_event)
                    else:
                        logger.info(
                            f"[IDLE_BATCH] Skipping idle batch creation for dp_id={missing_participant_id} "
                            f"(already exists in waiting room)"
                        )

                return idle_batch_events

        return []

    def on_decode_sync_collective(self, time: float, replica_id: int, stage_id: int,
                                  batch_global_id: int, sync_stage: str, layer_id: int, metrics_store):
        """
        Handle collective synchronization completion in DECODE cluster.

        Similar to on_prefill_sync_collective(), this method implements the layer-by-layer
        processing flow for the unified DECODE cluster with MoE:
        - pre_moe sync: execute pre-collective MoE work, then schedule post_moe sync
        - post_moe sync: execute post-MoE communication, then continue to next layer or finish

        Args:
            time: Current simulation time
            replica_id: ID of the replica
            stage_id: Pipeline stage ID
            batch_global_id: Global ID of the batch
            sync_stage: "pre_moe" or "post_moe"
            layer_id: Current layer being processed
            metrics_store: Metrics store for recording
        """
        from frontier.events.decode_sync_event import DecodeSyncEvent
        from frontier.events.batch_stage_end_event import BatchStageEndEvent
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        if sync_stage not in self._decode_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id]:
            logger.debug(
                f"[DECODE_SYNC][COLLECTIVE_SKIP] sync_stage={sync_stage} already processed for "
                f"replica={replica_id}, stage={stage_id}, batch_global_id={batch_global_id}, layer={layer_id}"
            )
            return []

        sync_wait_room = self._decode_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id].pop(sync_stage)
        dp_batches = sync_wait_room["batches"]

        try:
            dp_keys = list(dp_batches.keys())
        except Exception:
            dp_keys = []
        logger.info(
            f"[DECODE_SYNC][COLLECTIVE] ENTER: t={time:.6f}s, replica={replica_id}, stage={stage_id}, "
            f"layer={layer_id}, sync_stage={sync_stage}, batch_global_id={batch_global_id}, dp_keys={dp_keys}"
        )

        events = []
        non_idle_batches = [batch for batch in dp_batches.values() if not batch.is_idle]
        sample_batch = non_idle_batches[0] if non_idle_batches else next(iter(dp_batches.values()))
        total_global_tokens = sum(batch.total_num_tokens for batch in non_idle_batches)
        total_global_prefill_tokens = sum(
            batch.num_prefill_tokens for batch in non_idle_batches
        )
        if total_global_tokens <= 0:
            total_global_tokens = sample_batch.total_num_tokens
            total_global_prefill_tokens = sample_batch.num_prefill_tokens

        stage_scheduler = self.get_dp_replica_stage_scheduler(replica_id, 0, stage_id)
        execution_time_predictor = stage_scheduler._execution_time_predictor
        replica = self.get_replica(replica_id)
        participant_count = self._get_decode_sync_participant_count(replica, sample_batch)
        shared_domain_sync = self._is_monolithic_decode_shared_domain_sync(sample_batch)
        global_batch = self._create_virtual_global_batch(
            sample_batch,
            total_global_tokens,
            total_global_prefill_tokens,
        )
        if shared_domain_sync:
            related_wait_ms = (
                self._accumulate_monolithic_decode_shared_domain_related_wait_ms(
                    replica_id=replica_id,
                    stage_id=stage_id,
                    batch_global_id=batch_global_id,
                    sync_stage=sync_stage,
                    sync_wait_room=sync_wait_room,
                )
            )
            if related_wait_ms > 0.0:
                logger.info(
                    f"[MONOLITHIC_DECODE_SYNC][WAIT] batch_global_id={batch_global_id}, "
                    f"stage={stage_id}, layer={layer_id}, sync_stage={sync_stage}, "
                    f"related_wait_ms={related_wait_ms:.6f}"
                )

        if sync_stage == "pre_moe":
            if shared_domain_sync:
                if not hasattr(
                    execution_time_predictor,
                    "predict_monolithic_decode_shared_domain_lane_moe_times_ms",
                ):
                    raise AttributeError(
                        f"{type(execution_time_predictor).__name__} does not expose "
                        "predict_monolithic_decode_shared_domain_lane_moe_times_ms required for "
                        "monolithic decode shared-domain sync"
                    )

                lane_times_ms = (
                    execution_time_predictor
                    .predict_monolithic_decode_shared_domain_lane_moe_times_ms(
                        global_batch,
                        layer_id,
                    )
                )
                expected_lane_ids = set(dp_batches.keys())
                missing_lane_ids = expected_lane_ids - set(lane_times_ms.keys())
                if missing_lane_ids:
                    raise ValueError(
                        "predict_monolithic_decode_shared_domain_lane_moe_times_ms did not return "
                        f"timings for lane_ids={sorted(missing_lane_ids)}"
                    )

                logger.info(
                    f"[MONOLITHIC_DECODE_SYNC][PRE_MOE] total_global_tokens={total_global_tokens}, "
                    f"participant_count={participant_count}, lane_times_ms={lane_times_ms}"
                )

                for participant_id, batch in dp_batches.items():
                    lane_time_ms = float(lane_times_ms[participant_id])
                    if lane_time_ms < 0.0:
                        raise ValueError(
                            f"Negative shared-domain lane_time_ms={lane_time_ms} for lane={participant_id}"
                        )
                    lane_time_s = lane_time_ms * 1e-3
                    events.append(DecodeSyncEvent(
                        time + lane_time_s,
                        replica_id,
                        stage_id,
                        batch,
                        participant_id,
                        "post_moe",
                        layer_id,
                        lane_time_s,
                        cluster_type=self._cluster_type,
                    ))

                logger.info(
                    f"[DECODE_SYNC][COLLECTIVE] pre_moe shared-domain completed, "
                    f"scheduled post_moe sync arrivals using lane-specific times"
                )
                return events

            dp_gather_time = execution_time_predictor.predict_dp_gather_time(
                total_global_tokens,
                participant_count,
            )

            per_expert_tokens = None
            if hasattr(global_batch, "per_expert_tokens") and global_batch.per_expert_tokens:
                per_expert_tokens = global_batch.per_expert_tokens
            elif hasattr(execution_time_predictor, "_calculate_expert_token_allocation"):
                per_expert_tokens = execution_time_predictor._calculate_expert_token_allocation(
                    batch=global_batch,
                    cluster_type=self._cluster_type,
                    layer_id=layer_id,
                )
            elif hasattr(execution_time_predictor, "_get_moe_tokens_input"):
                moe_tokens_input = execution_time_predictor._get_moe_tokens_input(
                    global_batch,
                    layer_id=layer_id,
                )
                if isinstance(moe_tokens_input, dict):
                    per_expert_tokens = moe_tokens_input
                elif hasattr(execution_time_predictor, "_build_uniform_per_expert_tokens"):
                    per_expert_tokens = execution_time_predictor._build_uniform_per_expert_tokens(
                        int(moe_tokens_input)
                    )
                else:
                    raise ValueError(
                        "predictor returned scalar moe_tokens_input but does not expose "
                        "_build_uniform_per_expert_tokens for decode sync collective"
                    )
            else:
                raise AttributeError(
                    f"{type(execution_time_predictor).__name__} does not expose a supported "
                    "MoE token-allocation API for decode sync collective"
                )

            moe_time = execution_time_predictor.predict_moe_layer_time(
                global_batch,
                layer_id,
                self._cluster_type,
                per_expert_tokens=per_expert_tokens,
            )
            dp_scatter_time = execution_time_predictor.predict_dp_scatter_time(
                total_global_tokens,
                participant_count,
            )
            moe_compute_time = moe_time.total_time() * 1e-3
            moe_stage_time = dp_gather_time + moe_compute_time + dp_scatter_time

            logger.info(
                f"[MoE_TIME_BREAKDOWN] dp_gather={dp_gather_time:.6f}s, moe_comp={moe_compute_time:.6f}s, "
                f"dp_scatter={dp_scatter_time:.6f}s, total={moe_stage_time:.6f}s"
            )

            for participant_id, batch in dp_batches.items():
                events.append(DecodeSyncEvent(
                    time + moe_stage_time,
                    replica_id,
                    stage_id,
                    batch,
                    participant_id,
                    "post_moe",
                    layer_id,
                    moe_stage_time,
                    cluster_type=self._cluster_type,
                ))

            logger.info(
                f"[DECODE_SYNC][COLLECTIVE] pre_moe completed, scheduled post_moe sync at t={time + moe_stage_time:.6f}s"
            )
            return events

        single_layer_execution_time = execution_time_predictor.predict_stage_execution_time(
            sample_batch,
            stage_id,
            self._cluster_type,
            num_layers=1,
            layer_id=layer_id,
        )
        if hasattr(execution_time_predictor, "_get_expert_parallel_communication_time"):
            post_moe_comm_time = (
                execution_time_predictor._get_expert_parallel_communication_time(global_batch)
                * 1e-3
            )
        else:
            post_moe_comm_time = (
                single_layer_execution_time.expert_parallel_communication_time * 1e-3
            )

        num_layers = execution_time_predictor._num_layers_per_pipeline_stage
        next_layer_id = layer_id + 1

        if next_layer_id < num_layers:
            next_layer_execution_time = execution_time_predictor.predict_stage_execution_time(
                sample_batch,
                stage_id,
                self._cluster_type,
                num_layers=1,
                layer_id=next_layer_id,
            )
            attention_time = next_layer_execution_time.get_single_layer_attention_time() * 1e-3

            incremented_requests = set()
            for participant_id, batch in dp_batches.items():
                if batch.is_idle:
                    logger.info(
                        f"[DECODE_SYNC][IDLE_SKIP] Skip next-layer pre_moe scheduling for idle batch {batch.id} "
                        f"(replica={replica_id}, lane={participant_id}, layer={layer_id})"
                    )
                    continue

                for request in batch.requests:
                    if request.id not in incremented_requests:
                        request.mb_on_step_layer_count_increment(num_layers_completed=1)
                        incremented_requests.add(request.id)

                total_time_to_next_sync = post_moe_comm_time + attention_time
                events.append(DecodeSyncEvent(
                    time + total_time_to_next_sync,
                    replica_id,
                    stage_id,
                    batch,
                    participant_id,
                    "pre_moe",
                    next_layer_id,
                    total_time_to_next_sync,
                    cluster_type=self._cluster_type,
                ))

            logger.info(
                f"[DECODE_SYNC][COLLECTIVE] post_moe completed, incremented layer count for {len(incremented_requests)} unique requests, "
                f"scheduled next layer pre_moe sync at t={time + post_moe_comm_time + attention_time:.6f}s"
            )
            return events

        full_stage_execution_time = execution_time_predictor.predict_stage_execution_time(
            sample_batch,
            stage_id,
            self._cluster_type,
            num_layers=num_layers,
        )
        is_last_stage = stage_scheduler.is_last_stage
        pipeline_time = full_stage_execution_time.pipeline_time * 1e-3
        cpu_overhead_time = max(
            full_stage_execution_time.total_time
            - full_stage_execution_time.model_time,
            0.0,
        )
        decode_draft_proposer_time = (
            full_stage_execution_time.decode_draft_proposer_time * 1e-3
        )
        mtp_terminal_overshoot_time = (
            float(
                getattr(
                    full_stage_execution_time,
                    "mtp_terminal_overshoot_time",
                    0.0,
                )
            )
            * 1e-3
        )
        total_final_time = (
            post_moe_comm_time
            + pipeline_time
            + cpu_overhead_time
            + decode_draft_proposer_time
        )
        shared_domain_related_wait_ms = 0.0
        if shared_domain_sync:
            shared_domain_related_wait_ms = (
                self._pop_monolithic_decode_shared_domain_related_wait_ms(
                    replica_id=replica_id,
                    stage_id=stage_id,
                    batch_global_id=batch_global_id,
                )
            )

        for participant_id, batch in dp_batches.items():
            if batch.is_idle:
                logger.info(
                    f"[DECODE_SYNC][IDLE_SKIP] Skip final stage-end for idle batch {batch.id} "
                    f"(replica={replica_id}, lane={participant_id}, layer={layer_id})"
                )
                continue
            self._record_mtp_terminal_completion_delay(
                batch,
                mtp_terminal_overshoot_time,
            )

            dp_stage_scheduler = self.get_dp_replica_stage_scheduler(replica_id, participant_id, stage_id)
            batch_stage, _ = dp_stage_scheduler.predict_and_create_stage(batch, skip_get_execution_time=True)

            original_start_time = getattr(
                batch,
                '_decode_stage_start_time',
                time - full_stage_execution_time.total_time,
            )
            batch_stage.on_schedule(original_start_time)

            actual_execution_time = time + total_final_time - original_start_time

            batch_stage.override_execution_time(actual_execution_time)
            batch_stage.override_model_execution_time(full_stage_execution_time.model_time)

            corrected_execution_time = self._create_corrected_execution_time_for_metrics(
                full_stage_execution_time,
                actual_execution_time,
                original_start_time,
            )
            trace_execution_time = full_stage_execution_time
            if shared_domain_sync:
                trace_execution_time = (
                    self._build_monolithic_decode_shared_domain_trace_execution_time(
                        full_stage_execution_time,
                        related_wait_ms=shared_domain_related_wait_ms,
                    )
                )
            corrected_execution_time._trace_execution_time_override = trace_execution_time

            metrics_store.on_replica_stage_schedule(
                original_start_time,
                replica_id,
                stage_id,
                batch_stage,
                corrected_execution_time,
                self._cluster_type,
                participant_id,
            )

            events.append(BatchStageEndEvent(
                time + total_final_time,
                replica_id,
                stage_id,
                is_last_stage,
                batch,
                batch_stage,
                self._cluster_type,
                participant_id,
            ))

        logger.info(
            f"[DECODE_SYNC][COLLECTIVE] Last layer completed, scheduled batch stage end at "
            f"t={time + total_final_time:.6f}s"
        )
        return events

    def on_kv_cache_arrival(
        self,
        time: float,
        batch: Batch,
        transfer_info,
    ) -> List:
        """Disaggregated KV cache arrivals are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _handle_decode_attn_arrival(
        self,
        time: float,
        batch: Batch,
        transfer_info,
        logger,
    ) -> List:
        """Disaggregated decode-attn arrivals are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _handle_decode_arrival(
        self,
        time: float,
        batch: Batch,
        transfer_info,
        logger,
    ) -> List:
        """Disaggregated decode arrivals are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)


    def on_m2n_arrival(
        self,
        time: float,
        batch: Batch,
        transfer_info,
    ) -> List:
        """Disaggregated M2N arrivals are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _handle_m2n_arrival_decode_ffn(
        self,
        time: float,
        batch: Batch,
        transfer_info,
        logger,
    ) -> List:
        """Disaggregated decode-ffn M2N arrivals are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _promote_incomplete_m2n_groups_with_idle_lanes(self, logger) -> int:
        """Disaggregated terminal M2N group promotion is not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _apply_dp_padding_on_promotion(
        self,
        picked: List[tuple],
        logger,
    ) -> None:
        """Apply DP padding (Layer 2 of three-layer padding) to promoted batches.

        StepFun-vLLM three-layer padding order (gpu_model_runner.py):
          Layer 1: Stage count padding — dummy stages with 1 token
          Layer 2: DP padding — per-stage max across DP ranks  ← THIS METHOD
          Layer 3: CUDA Graph padding — nearest capture size

        DP padding must happen at the cluster scheduler level because it
        requires cross-DP-lane visibility: the replica scheduler only sees
        its own lane's token distribution.  Once all DP lanes arrive at the
        (layer_id, afd_stage_idx) barrier, this method computes the per-stage
        max and updates each batch's AFDStageMetadata accordingly.

        Reference: StepFun-vLLM gpu_model_runner.py:1240-1244
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
            num_stage_tokens_across_dp = DPMetadata.num_stage_tokens_across_dp(
                afd_tokens_lens, dp_size, dp_rank)
            afd_tokens_lens = torch.max(num_stage_tokens_across_dp, dim=1)[0]

        Args:
            picked: List of (batch, transfer_info) tuples from all DP lanes
            logger: Logger instance
        """
        from frontier.entities.batch import AFDStageMetadata

        # Filter to non-idle batches that carry stage metadata
        batches_with_meta = [
            (b, t) for (b, t) in picked
            if not b.is_idle and b.afd_stage_metadata is not None
        ]

        if len(batches_with_meta) <= 1:
            # Single DP lane or no metadata — DP padding is a no-op
            return

        num_stages = batches_with_meta[0][0].afd_stage_metadata.num_stages

        # Recompute per-stage token lens from each DP lane's batch.
        # Layer 1 (stage count padding) is applied inline so that
        # all lanes have exactly num_stages entries before the max.
        all_stage_lens = []
        for b, _ in batches_with_meta:
            meta = b.afd_stage_metadata
            if meta.num_stages != num_stages:
                raise ValueError(
                    f"Inconsistent num_stages across DP lanes: "
                    f"expected {num_stages}, got {meta.num_stages}"
                )
            stage_lens = AFDStageMetadata.compute_stage_token_lens(
                num_reqs=len(b.requests),
                num_tokens_per_req=list(b.num_tokens),
                num_stages=num_stages,
            )
            # Layer 1: stage count padding (dummy stages with 1 token)
            while len(stage_lens) < num_stages:
                stage_lens.append(1)
            all_stage_lens.append(stage_lens)

        # Layer 2: per-stage max across DP ranks
        dp_stage_max_tokens = [
            max(lane_lens[s] for lane_lens in all_stage_lens)
            for s in range(num_stages)
        ]

        # Update each non-idle batch's metadata with DP-padded values
        for b, _ in batches_with_meta:
            b.afd_stage_metadata = b.afd_stage_metadata.with_dp_padding(
                dp_stage_max_tokens=dp_stage_max_tokens,
            )

        logger.info(
            f"[FFN-DP-PADDING] Applied DP padding across {len(batches_with_meta)} lanes: "
            f"dp_stage_max_tokens={dp_stage_max_tokens} "
            f"padded_total={sum(dp_stage_max_tokens)}"
        )

    def _handle_m2n_arrival_decode_attn(
        self,
        time: float,
        micro_batch: Batch,
        transfer_info,
        logger,
    ) -> List:
        """Disaggregated decode-attn M2N arrivals are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)



        # # Determine target micro-batch size per (replica, dp)
        # if not hasattr(self._config, 'af_pipeline_num_micro_batch'):
        #     raise ValueError("Missing required config attribute: af_pipeline_num_micro_batch")
        # if not hasattr(self._config, 'batch_size'):
        #     raise ValueError("Missing required config attribute: batch_size")
        # if not hasattr(self._config, 'decode_attn_micro_batch_size'):
        #     raise ValueError("Missing required config attribute: decode_attn_micro_batch_size")

        # dp_size = self._replica_dp_size
        # af_num = self._config.af_pipeline_num_micro_batch
        # batch_size = self._config.batch_size
        # micro_batch_size = self._config.decode_attn_micro_batch_size

        # TODO: check process shoudle be done in config init
        # if micro_batch_size is None and batch_size is not None and af_num and af_num > 0 and dp_size and dp_size > 0:
        #     if batch_size % (af_num * dp_size) != 0:
        #         from frontier.logger import get_cluster_logger
        #         get_cluster_logger(__name__, self._cluster_type.name).error(
        #             f"decode-attn batch_size {batch_size} not divisible by af({af_num})*dp({dp_size}); using carry-over size"
        #         )
        #         micro_batch_size = None
        #         raise ValueError(f"decode-attn batch_size {batch_size} not divisible by af({af_num})*dp({dp_size})")
        #     else:
        #         micro_batch_size = batch_size // (af_num * dp_size)

        # target_size = micro_batch_size
        # need = max(0, target_size - len(ongoing_requests))

        # # Build candidate pool from cluster request queue (FIFO order)
        # # State consistency: match completed_layer_count with base ongoing request
        # base_req = ongoing_requests[0]
        # candidates_pool = []
        # for req in list(self._request_queue):  # copy for safe removal later
        #     if not req.is_prefill_complete or req.completed:
        #         logger.error(f"[AF-ARRIVAL] error situation: request {req.id} for dynamic top-up: not prefill complete or already completed")
        #         raise ValueError(f"Request {req.id} for dynamic top-up: not prefill complete or already completed")
        #         # continue

        #     if req.completed_layer_count == base_req.completed_layer_count:
        #         candidates_pool.append(req)

        # # Prepare memory-aware FIFO selector
        # # TODO: we check can_allocate by var _max_blocks_per_sequence; any other ideas?
        # per_req_blocks = getattr(rs, '_max_blocks_per_sequence', None)
        # if per_req_blocks is None:
        #     per_req_blocks = 0
        # def can_allocate_one() -> bool:
        #     return rs.can_allocate(per_req_blocks) if per_req_blocks > 0 else True
        # selector = FIFOCandidateRequestSelector(can_allocate_fn=can_allocate_one)
        # selected = selector.select_candidates(base_req, need, candidates_pool)

        # # Remove selected from cluster queue and allocate KV blocks
        # selected_ids = set(r.id for r in selected)
        # if selected_ids:
        #     self._request_queue = [r for r in self._request_queue if r.id not in selected_ids]
        #     for r in selected:
        #         if per_req_blocks > 0 and rs.can_allocate(per_req_blocks):
        #             rs.allocate(r.id, per_req_blocks)
        #         elif per_req_blocks > 0:
        #             logger.info(f"[AF-ARRIVAL] Insufficient KV blocks during allocation for req {r.id}; skipping")
        #             # If cannot allocate now, skip adding this request
        #             selected_ids.discard(r.id)
        #     # Filter out any not actually allocated
        #     selected = [r for r in selected if r.id in selected_ids]

        # # Form next-hop MicroBatch on same (replica, dp)
        # next_requests = ongoing_requests + selected
        # num_tokens = [1 for _ in next_requests]
        # next_mb = MicroBatch(
        #     replica_id=replica_id,
        #     requests=next_requests,
        #     num_tokens=num_tokens,
        #     parent_batch_id=getattr(micro_batch, 'id', None),
        #     micro_batch_index=None,
        #     num_micro_batches=None,
        #     parent_batch_global_id=getattr(micro_batch, 'global_id', None),
        # )
        # next_mb.decode_attn_original_replica_id = replica_id
        # next_mb.decode_attn_original_dp_id = dp_id

        # # Push to immediate queue and schedule this (replica, dp)
        # rs.add_batch_to_immediate_queue(next_mb)
        # logger.info(
        #     f"[AF-ARRIVAL] Created next-hop MicroBatch {next_mb.id} on (replica={replica_id}, dp={dp_id}) "
        #     f"with size={len(next_requests)} (carry={len(ongoing_requests)}, topup={len(selected)})"
        # )
        # # Emit a lightweight stdout marker for integration tests to detect dynamic top-up without relying on logger sinks
        # try:
        #     print(
        #         f"TOPUP: carry={len(ongoing_requests)} topup={len(selected)} size={len(next_requests)} "
        #         f"replica={replica_id} dp={dp_id} next_mb={getattr(next_mb, 'id', 'N/A')}"
        #     )
        # except Exception:
        #     pass

        # return next_events + [ReplicaScheduleEvent(time, replica_id, self._cluster_type, dp_id)]

        # (Periodic scheduling path is intentionally bypassed for returned AF batches to keep priority)

    def get_af_queue_size(self) -> int:
        """Get the size of the A→F request queue."""
        if hasattr(self, '_af_batch_queue'):
            return len(self._af_batch_queue)
        return 0

    def clear_af_queue(self) -> List:
        """Clear and return all batches from A→F request queue."""
        if hasattr(self, '_af_batch_queue'):
            batches = self._af_batch_queue[:]
            self._af_batch_queue.clear()
            return batches
        return []

    def _create_batch_group(self, requests: List[Request], num_tokens: List[int], replica_id: int, ep_id: int, time: float,
                            source_batch_ids: List[int], per_expert_tokens: Dict[int, int]) -> EPBatchGroup:
        batch_group = EPBatchGroup(
            requests,
            num_tokens,
            replica_id,
            ep_id,
            time,
            source_batch_ids,
            per_expert_tokens,
            self._cluster_type,
            is_moe=self._config.replica_config.model_config.is_moe,
        )

        return batch_group

    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass
