from abc import ABC, abstractmethod
from collections import defaultdict, deque
from copy import deepcopy
import csv
import math
from pathlib import Path
from numbers import Real

from typing import Any, Dict, List, NamedTuple, Tuple, Optional, TYPE_CHECKING

from frontier.config import ClusterConfig, BaseRequestGeneratorConfig
from frontier.entities import Batch, EPBatchGroup, ExecutionTime, Replica, Request, Cluster
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR
# Phase 2.5: Removed deprecated MoECollectiveScheduleEvent import
from frontier.execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from frontier.model_architectures import (
    ExpertParallelCollective,
    ModelArchitectureProfile,
)
from frontier.moe_ep_workload import (
    LayerEPWorkload,
    build_contiguous_expert_ownership,
    materialize_layer_ep_workload,
    resolve_routing_details,
)
from frontier.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    EP_WAVE,
    FULL_STAGE_WORLD,
    StageExecutionContext,
)
from frontier.types import (
    ClusterType,
    ClusterSchedulerType,
    ReplicaSchedulerType,
    RequestGeneratorType,
)

if TYPE_CHECKING:
    from frontier.kv_cache_transfer import BaseKVCacheTransferPredictor
    from frontier.m2n_transfer import BaseM2NTransferPredictor


def resolve_ep_collective_kind(
    model_config: Any,
    cluster_type: ClusterType,
    expected_ep_size: int,
) -> ExpertParallelCollective:
    """Resolve EP policy from the runtime config's profile snapshot."""

    if model_config is None:
        raise ValueError(
            "EP collective resolution requires replica_config.model_config"
        )
    profile_getter = getattr(model_config, "get_model_architecture_profile", None)
    if not callable(profile_getter):
        raise TypeError(
            "EP collective resolution requires "
            "model_config.get_model_architecture_profile()"
        )
    profile = profile_getter()
    if not isinstance(profile, ModelArchitectureProfile):
        raise TypeError(
            "model_config.get_model_architecture_profile() must return "
            "ModelArchitectureProfile"
        )
    if profile.uses_expert_parallel_alltoall(cluster_type, expected_ep_size):
        return ExpertParallelCollective.ALLTOALL
    return ExpertParallelCollective.ALLGATHER


class EPBatchGroupPlan(NamedTuple):
    """Immutable, side-effect-free inputs for one DECODE_FFN EP batch."""

    replica_id: int
    ep_id: int
    layer_global_id: int
    afd_stage_idx: int
    group_time: float
    pre_routing_effective_total_tokens: int
    source_batches: Tuple[Batch, ...]
    source_batch_ids: Tuple[int, ...]
    per_expert_tokens: Tuple[Tuple[int, int], ...]


class BaseClusterScheduler(ABC):
    @staticmethod
    def _map_source_attn_replica_to_ffn_replica(
        source_replica_ordinal: int,
        target_ffn_replica_ids: List[int] | Tuple[int, ...],
    ) -> int:
        """Map one Attention-Replica ordinal to a sticky FFN Replica.

        The mapping is intentionally based on the source Replica ordinal, not
        on a retired attention-DP lane.  A target list may be larger than the
        source population; those extra FFN Replicas remain valid and idle.
        """
        if type(source_replica_ordinal) is not int or source_replica_ordinal < 0:
            raise ValueError(
                "source_replica_ordinal must be an exact non-negative int, "
                f"got {source_replica_ordinal!r}"
            )
        if type(target_ffn_replica_ids) not in {list, tuple}:
            raise ValueError(
                "target_ffn_replica_ids must be an exact list or tuple, "
                f"got {target_ffn_replica_ids!r}"
            )
        if not target_ffn_replica_ids:
            raise ValueError("target_ffn_replica_ids must not be empty")
        if any(
            type(replica_id) is not int or replica_id < 0
            for replica_id in target_ffn_replica_ids
        ):
            raise ValueError(
                "target_ffn_replica_ids must contain exact non-negative ints"
            )
        if len(set(target_ffn_replica_ids)) != len(target_ffn_replica_ids):
            raise ValueError("target_ffn_replica_ids must not contain duplicates")
        return target_ffn_replica_ids[
            source_replica_ordinal % len(target_ffn_replica_ids)
        ]

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

        request_generator_config = getattr(self, "_request_generator_config", None)
        if request_generator_config is None:
            return

        request_generator_type = request_generator_config.get_type()
        if request_generator_type != RequestGeneratorType.TRACE_REPLAY:
            raise ValueError(
                "Prefix caching requires a trace request source with session_id "
                "and block_hash_ids metadata before scheduling. "
                f"Got {request_generator_type}."
            )

        trace_file = Path(request_generator_config.trace_file)
        if not trace_file.exists():
            raise ValueError(
                "Prefix caching trace request source requires an existing trace file "
                f"with session_id and block_hash_ids columns. Got {trace_file}."
            )

        with trace_file.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            header = reader.fieldnames
            required_columns = {"session_id", "block_hash_ids"}
            missing_columns = sorted(required_columns - set(header or []))
            if missing_columns:
                raise ValueError(
                    "Prefix caching trace request source requires session_id and "
                    "block_hash_ids columns before scheduling. "
                    f"Missing columns: {missing_columns}."
                )

            for row_number, row in enumerate(reader, start=2):
                missing_values = sorted(
                    column
                    for column in required_columns
                    if row.get(column) is None or not row[column].strip()
                )
                if missing_values:
                    raise ValueError(
                        "Prefix caching trace request source requires non-empty "
                        "session_id and block_hash_ids values before scheduling. "
                        f"Trace file: {trace_file}; row {row_number}; "
                        f"missing values: {missing_values}."
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
        kv_cache_transfer_predictor: Optional["BaseKVCacheTransferPredictor"] = None,
        m2n_transfer_predictor: Optional["BaseM2NTransferPredictor"] = None,
        available_clusters: Optional[set] = None,
    ):
        self._config = config
        self._cluster = cluster
        self._cluster_type = cluster.cluster_type
        self._num_replicas = len(self._cluster.replicas)
        self._predictor = predictor
        self._kv_cache_transfer_predictor = kv_cache_transfer_predictor
        self._m2n_transfer_predictor = m2n_transfer_predictor
        # Attention-DP lanes are retired.  Cluster capacity is represented by
        # ``len(cluster.replicas)``; every physical Replica owns one shared
        # attention world, while MoE-only work creates replica-local EP lanes.
        if self._cluster_type == ClusterType.DECODE_FFN:
            self._replica_ep_size = int(
                self._config.replica_config.moe_expert_parallel_size
            )
            self._replica_scheduler_count = self._replica_ep_size
        else:
            attn_dp = getattr(self._config.replica_config, "attn_data_parallel_size", None)
            if type(attn_dp) is not int or attn_dp != 1:
                raise ValueError(
                    "Replica-local attention DP lanes are retired; "
                    f"{self._cluster_type.name} requires attn_data_parallel_size=1, got {attn_dp!r}"
                )
            self._replica_ep_size = None
            self._replica_scheduler_count = 1
        self._available_clusters = available_clusters or set()
        self._request_generator_config = request_generator_config
        self._stage_execution_contexts = self._build_stage_execution_contexts()

        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Validate the canonical replica-local lane count.
        if (
            type(self._replica_scheduler_count) is not int
            or self._replica_scheduler_count <= 0
        ):
            logger.error(
                "Invalid replica-local scheduler count: %s",
                self._replica_scheduler_count,
            )
            raise ValueError(
                "Invalid replica-local scheduler count: "
                f"{self._replica_scheduler_count}"
            )

        # Initialize replica schedulers based on cluster type
        # DECODE_FFN: Use EP (Expert Parallel) concept instead of DP
        # Other clusters: one full-model child scheduler per Replica.
        self._replica_schedulers = {}
        self._full_stage_replica_schedulers = {}

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
            # MoE routed-expert work owns explicit local EP children.
            self._replica_ep_size = self._config.replica_config.moe_expert_parallel_size
            for replica_id, replica in self._cluster.replicas.items():
                for ep_id in range(self._replica_ep_size):
                    scheduler_key = (replica_id, ep_id)
                    self._replica_schedulers[scheduler_key] = ReplicaSchedulerRegistry.get(
                        cluster_specific_config.get_type(),
                        replica_config=self._config.replica_config,
                        replica_scheduler_config=cluster_specific_config,
                        request_generator_config=request_generator_config,
                        replica=replica,
                        predictor=self._predictor,
                        cluster_type=self._cluster_type,
                        replica_local_id=ep_id,
                        af_pipeline_num_micro_batch=getattr(self._config, 'af_pipeline_num_micro_batch', -1),
                        cluster_scheduler=self,
                    )
                # Dense FFN work owns the complete stage world and therefore
                # uses a separate child scheduler keyed by no EP lane.
                self._full_stage_replica_schedulers[replica_id] = (
                    ReplicaSchedulerRegistry.get(
                        cluster_specific_config.get_type(),
                        replica_config=self._config.replica_config,
                        replica_scheduler_config=cluster_specific_config,
                        request_generator_config=request_generator_config,
                        replica=replica,
                        predictor=self._predictor,
                        cluster_type=self._cluster_type,
                        replica_local_id=None,
                        af_pipeline_num_micro_batch=getattr(
                            self._config, "af_pipeline_num_micro_batch", -1
                        ),
                        cluster_scheduler=self,
                    )
                )
        else:
            # Every non-FFN Replica owns one complete full-stage scheduler.
            # Attention-DP lanes are retired; ``None`` is the explicit absence
            # of a replica-local EP identity.
            for replica_id, replica in self._cluster.replicas.items():
                full_stage_scheduler = ReplicaSchedulerRegistry.get(
                    cluster_specific_config.get_type(),
                    replica_config=self._config.replica_config,
                    replica_scheduler_config=cluster_specific_config,
                    request_generator_config=request_generator_config,
                    replica=replica,
                    predictor=self._predictor,
                    cluster_type=self._cluster_type,
                    replica_local_id=None,
                    af_pipeline_num_micro_batch=getattr(
                        self._config, "af_pipeline_num_micro_batch", -1
                    ),
                    cluster_scheduler=self,
                )
                self._full_stage_replica_schedulers[replica_id] = (
                    full_stage_scheduler
                )
                self._replica_schedulers[(replica_id, None)] = full_stage_scheduler
        self._request_queue = []

        # Initialize specialized queues for PD+AF disaggregation
        if self._cluster_type == ClusterType.DECODE_ATTN:
            # Queue for receiving requests from decode-ffn cluster (A→F communication)
            self._af_batch_queue = []
            # A→F waiting room is scoped to one concrete decode-attn wave.
            # key=(wire_layer_id, afd_stage_idx) -> {per_lane_queues}
            self._a2f_waiting_by_layer: Dict[tuple[int, int], dict] = {}
            self._a2f_expected_lanes = [
                (replica_id, None)
                for replica_id in list(self._cluster.replicas.keys())
            ]
            self._a2f_group_micro_batches = len(self._a2f_expected_lanes)
            # F→A waiting room keeps per-lane FIFO semantics scoped by next_layer
            self._f2a_waiting_by_round: Dict[tuple, dict] = {}
            self._f2a_expected_lanes = list(self._a2f_expected_lanes)
            self._f2a_group_micro_batches = len(self._f2a_expected_lanes)
            self._decode_attn_barrier_round_counter = 0
        elif self._cluster_type == ClusterType.DECODE_FFN:
            # Per-key waiting rooms for grouping distinct lanes (attn→ffn arrivals)
            # key=(layer_id, afd_stage_idx[, barrier_round_id])
            # -> {per_lane_queues, lanes_rr_order, rr_cursor,
            #     expected_lane_contract}
            self._m2n_waiting_by_layer: Dict[
                tuple[int, int] | tuple[int, int, int], dict
            ] = {}
            self._m2n_ready_groups = deque()  # Deque[List[(batch, transfer_info)]]
            attn_num_replicas = getattr(self._config, "decode_attn_cluster_num_replicas", None)
            if attn_num_replicas is None:
                raise ValueError(
                    "decode_attn_cluster_num_replicas must be set for DECODE_FFN grouping"
                )
            if int(attn_num_replicas) <= 0:
                raise ValueError(
                    "DECODE_ATTN cluster capacity must be a positive cluster-level "
                    f"num_replicas, got {attn_num_replicas}"
                )
            # AFD source identity is one full-stage tuple per attention Replica;
            # no attention-DP lane dimension participates in this grouping.
            source_replica_count = int(attn_num_replicas)
            self._ffn_group_micro_batches = source_replica_count
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
                (attn_replica_id_start + replica_ordinal, None)
                for replica_ordinal in range(source_replica_count)
            ]
            if len(self._ffn_expected_lanes) != source_replica_count:
                raise ValueError(
                    "DECODE_ATTN Replica grouping mismatch with expected source topology: "
                    f"expected={len(self._ffn_expected_lanes)} configured={source_replica_count}"
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
                target_replica_id = self._map_source_attn_replica_to_ffn_replica(
                    lane_ordinal,
                    self._ffn_replica_ids,
                )
                self._ffn_lane_to_target_replica[lane] = target_replica_id
                self._ffn_expected_lanes_by_target[target_replica_id].append(lane)
            expected_group_sizes = {
                replica_id: len(lanes)
                for replica_id, lanes in self._ffn_expected_lanes_by_target.items()
            }
            self._ffn_group_micro_batches = max(
                expected_group_sizes.values(),
                default=1,
            )
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
                f"[FFN-GROUPING] Initialized with {source_replica_count} full-stage source Replicas for strict (layer_id, afd_stage_idx) grouping"
            )

            # EP waiting room for combine synchronization in decode-ffn cluster
            # Structure: replica_id -> stage_id -> batch_global_id -> {batches: {ep_id: batch}, arrival_times: {ep_id: time}}
            self._ep_allgather_waiting_room = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
                )
            )

            # EP waiting room for dispatch synchronization in decode-ffn cluster.
            # Structure: replica_id -> stage_id -> batch_global_id -> {batches: {ep_id: batch}, arrival_times: {ep_id: time}}
            self._ep_alltoall_dispatch_waiting_room = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
                )
            )
        elif self._cluster_type in [ClusterType.PREFILL, ClusterType.MONOLITHIC]:
            # Prefill sync waiting room: replica_id -> stage_id -> batch_global_id -> layer_id -> sync_stage -> {replica_local_id: {batch, time}}
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
            # Decode sync waiting room: replica_id -> stage_id -> batch_global_id -> layer_id -> sync_stage -> {replica_local_id: {batch, time}}
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

    def _build_stage_execution_contexts(self) -> dict[tuple[int, int], StageExecutionContext]:
        """Create one parent admission context for every physical Replica stage."""

        replica_config = getattr(
            getattr(self, "_config", None),
            "replica_config",
            None,
        )
        model_config = getattr(replica_config, "model_config", None)
        if replica_config is None or model_config is None:
            raise ValueError(
                "Stage execution contexts require replica_config.model_config"
            )
        model_is_moe = bool(getattr(model_config, "is_moe", False))
        # Shared full-model roles (co-location, PDD PREFILL/DECODE) execute
        # routed-expert work inline, while PD-AF DECODE_FFN uses explicit EP
        # children.  All of those MoE operation paths must validate the same
        # Replica-local participant domain.  DECODE_ATTN has no routed-expert
        # operation and therefore keeps the dense/full-stage singleton scope.
        has_local_ep_domain = model_is_moe and self._cluster_type in (
            ClusterType.MONOLITHIC,
            ClusterType.PREFILL,
            ClusterType.DECODE,
            ClusterType.DECODE_FFN,
        )
        configured_ep_size = getattr(replica_config, "moe_expert_parallel_size", None)
        if model_is_moe and has_local_ep_domain:
            if type(configured_ep_size) is not int or configured_ep_size <= 0:
                raise ValueError(
                    "MoE stage execution contexts require an exact positive "
                    "moe_expert_parallel_size"
                )
            ep_size = configured_ep_size
        else:
            ep_size = 1

        contexts: dict[tuple[int, int], StageExecutionContext] = {}
        for replica_id, replica in self._cluster.replicas.items():
            if type(replica_id) is not int or replica_id < 0:
                raise ValueError(
                    "Cluster Replica IDs must be exact non-negative ints"
                )
            num_stages = getattr(replica, "num_pipeline_stages", None)
            if num_stages is None:
                num_stages = getattr(replica_config, "num_pipeline_stages", None)
            if type(num_stages) is not int or num_stages <= 0:
                raise ValueError(
                    "Replica num_pipeline_stages must be an exact positive int"
                )
            for stage_id in range(num_stages):
                contexts[(replica_id, stage_id)] = StageExecutionContext(
                    replica_id=replica_id,
                    stage_id=stage_id,
                    ep_size=ep_size,
                )
        return contexts


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

    def get_full_stage_replica_scheduler(self, replica_id: int):
        try:
            return self._full_stage_replica_schedulers[replica_id]
        except KeyError as exc:
            raise ValueError(
                "Full-stage Replica scheduler is unavailable: "
                f"replica_id={replica_id}"
            ) from exc

    def get_replica_scheduler(self, replica_id: int, replica_local_id: int | None):
        if replica_local_id is None:
            return self.get_full_stage_replica_scheduler(replica_id)
        return self._replica_schedulers[(replica_id, replica_local_id)]

    def get_replica_stage_scheduler(
        self, replica_id: int, replica_local_id: int | None, stage_id: int
    ):
        if replica_local_id is None:
            return self.get_full_stage_replica_scheduler(
                replica_id
            ).get_replica_stage_scheduler(stage_id)
        return self._replica_schedulers[
            (replica_id, replica_local_id)
        ].get_replica_stage_scheduler(stage_id)

    def get_stage_execution_context(
        self, replica_id: int, stage_id: int
    ) -> StageExecutionContext:
        """Return the parent admission owner for one physical Replica stage."""

        if type(replica_id) is not int or replica_id < 0:
            raise ValueError("replica_id must be an exact non-negative int")
        if type(stage_id) is not int or stage_id < 0:
            raise ValueError("stage_id must be an exact non-negative int")
        contexts = getattr(self, "_stage_execution_contexts", None)
        if type(contexts) is not dict:
            raise RuntimeError("Stage execution contexts were not initialized")
        try:
            return contexts[(replica_id, stage_id)]
        except KeyError as exc:
            raise ValueError(
                "Unknown Replica/stage admission context: "
                f"replica_id={replica_id}, stage_id={stage_id}"
            ) from exc

    def release_stage_admission_for_batch(
        self,
        batch: Batch,
        *,
        stage_id: int | None = None,
    ) -> None:
        """Release a batch's parent stage owner at its true completion boundary."""

        ticket = getattr(batch, "_stage_admission_ticket", None)
        if ticket is None:
            return
        if stage_id is None:
            stage_id = getattr(batch, "afd_stage_idx", None)
        if type(stage_id) is not int or stage_id < 0:
            raise ValueError(
                "stage_id is required to release a stage admission ticket"
            )
        context = self.get_stage_execution_context(
            int(ticket.replica_id),
            stage_id,
        )
        context.release(ticket)
        batch.__dict__.pop("_stage_admission_ticket", None)

    def transition_stage_admission_for_layer(
        self,
        batch: Batch,
        *,
        stage_id: int,
        layer_id: int,
        scope: str,
        participant_ep_ids: tuple[int, ...] = (),
    ) -> None:
        """Switch a shared batch's active parent scope at a layer boundary.

        Shared co-location/PDD execution keeps the outer batch admitted to one
        pipeline stage, but its layer operation alternates between full-stage
        attention/dense work and a complete local EP wave.  The active ticket is
        replaced atomically so the stage never becomes idle between those
        dependent operations.

        Direct unit probes may call the pure layer helpers without constructing a
        cluster scheduler.  Such probes have neither a context registry nor a
        ticket; production schedulers always have both, and missing state there
        is an explicit error rather than a fallback.
        """

        if type(stage_id) is not int or stage_id < 0:
            raise ValueError("stage_id must be an exact non-negative int")
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError("layer_id must be an exact non-negative int")
        ticket = getattr(batch, "_stage_admission_ticket", None)
        contexts = getattr(self, "_stage_execution_contexts", None)
        if ticket is None and contexts is None:
            # Standalone materializer tests intentionally omit the outer DES
            # scheduler.  They do not claim stage ownership and therefore have
            # no scope to transition.
            return
        if ticket is None:
            raise ValueError(
                "shared layer operation is missing its stage admission ticket"
            )
        context = self.get_stage_execution_context(ticket.replica_id, stage_id)
        operation_id = (
            "shared_layer",
            int(batch.id),
            int(batch.schedule_epoch),
            int(stage_id),
            int(layer_id),
            scope,
        )
        next_ticket = context.transition_active_scope(
            ticket,
            operation_id=operation_id,
            scope=scope,
            participant_ep_ids=participant_ep_ids,
        )
        batch._stage_admission_ticket = next_ticket
        history = getattr(batch, "_stage_admission_scope_history", None)
        if history is None:
            history = []
            batch._stage_admission_scope_history = history
        history.append(
            {
                "stage_id": int(stage_id),
                "layer_id": int(layer_id),
                "scope": scope,
                "admission_seq": int(next_ticket.admission_seq),
                "participant_ep_ids": tuple(next_ticket.participant_ep_ids),
            }
        )

    def make_decode_sync_global_id(
        self,
        replica_id: int,
        replica_local_id: int,
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

        lane_id = int(replica_local_id or 0)
        if lane_id < 0:
            raise ValueError(
                "replica_local_id must be non-negative, "
                f"got {replica_local_id!r}"
            )
        if lane_id >= ep_participant_count:
            raise ValueError(
                "MONOLITHIC decode sync lane id must be within the EP participant "
                f"domain, got replica_local_id={lane_id}, "
                f"ep_participant_count={ep_participant_count}"
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
            raise TypeError(
                "Expected lane tuple(replica_id, replica_local_id), "
                f"got {lane!r}"
            )
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
                        getattr(transfer_info, "source_replica_local_id", None),
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
        waiting_by_layer: Dict[
            tuple[int, int] | tuple[int, int, int], Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        groups = []
        for group_key, room in sorted(
            waiting_by_layer.items(), key=lambda item: str(item[0])
        ):
            if not isinstance(group_key, tuple) or len(group_key) not in (2, 3):
                raise TypeError(
                    "Expected DECODE_FFN waiting key(layer, stage[, round]), "
                    f"got {group_key!r}"
                )
            layer_id, afd_stage_idx = group_key[:2]
            key_state = {
                "layer_id": layer_id,
                "afd_stage_idx": afd_stage_idx,
            }
            if len(group_key) == 3:
                key_state["barrier_round_id"] = group_key[2]
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
                    "key": key_state,
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
            "_replica_schedulers",
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
        scheduler_items = list(self._replica_schedulers.items())
        scheduler_items.extend(
            ((replica_id, None), replica_scheduler)
            for replica_id, replica_scheduler in getattr(
                self, "_full_stage_replica_schedulers", {}
            ).items()
        )
        for scheduler_key, replica_scheduler in sorted(
            scheduler_items, key=lambda item: str(item[0])
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
        scheduler_items = list(self._replica_schedulers.items())
        scheduler_items.extend(
            ((replica_id, None), replica_scheduler)
            for replica_id, replica_scheduler in getattr(
                self, "_full_stage_replica_schedulers", {}
            ).items()
        )
        for key, replica_scheduler in scheduler_items:
            rs_empty = replica_scheduler.is_empty()
            replica_states.append((key, rs_empty))
            all_empty = all_empty and rs_empty

        logger.info(f"[IDLE-CHECK][{self._cluster_type.name}] request_queue={rq_len}, af_batch_queue={af_q_len}, replica_empty={[(str(k), v) for k, v in replica_states]}")

        # Return True only if request queue, AF queue (if exists), and all replicas are empty
        return rq_len == 0 and af_q_len == 0 and all_empty

    @staticmethod
    def _validate_token_conservation(
        input_tokens: int,
        per_expert_tokens: Dict[int, int],
        context: str,
    ) -> None:
        """Validate exact conservation for one already-materialized EP subset.

        The shared MoE materializer owns integerization.  The scheduler keeps
        this small fail-fast check at the entity-construction boundary so a
        malformed lane workload cannot enter the event queue.  It intentionally
        does not allocate or rebalance tokens.
        """
        total_expert_tokens = sum(per_expert_tokens.values())
        if total_expert_tokens != input_tokens:
            raise ValueError(
                f"Token conservation violated in {context}: "
                f"Input tokens={input_tokens}, Expert tokens={total_expert_tokens}, "
                f"Difference={input_tokens - total_expert_tokens}, "
                f"Per-expert allocation={per_expert_tokens}"
            )


    def _materialize_ep_wave_workload(
        self,
        group: List[Tuple[Batch, Any]],
        replica_id: int,
        layer_global_id: int,
        routing_details,
    ) -> LayerEPWorkload:
        """Materialize one aggregate workload shared by every EP lane in a wave."""
        if type(group) is not list or not group:
            raise ValueError("DECODE_FFN EP wave group must be a non-empty list")
        routing_token_count = 0
        for entry in group:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError(
                    "DECODE_FFN EP wave group entries must be (batch, transfer_info) tuples"
                )
            batch = entry[0]
            batch_tokens = getattr(batch, "total_num_tokens", None)
            if type(batch_tokens) is not int or batch_tokens < 0:
                raise ValueError(
                    "DECODE_FFN EP wave source batch total_num_tokens must be a "
                    f"non-negative int, got {batch_tokens!r}"
                )
            routing_token_count += batch_tokens

        replica_config = getattr(self._config, "replica_config", None)
        if replica_config is None:
            raise ValueError("DECODE_FFN requires replica_config for EP materialization")
        total_expert_num = getattr(replica_config, "total_expert_num", None)
        moe_expert_parallel_size = getattr(
            replica_config,
            "moe_expert_parallel_size",
            None,
        )
        router_topk = getattr(replica_config, "router_topk", None)
        if type(total_expert_num) is not int or total_expert_num <= 0:
            raise ValueError(
                "DECODE_FFN total_expert_num must be an exact positive int for EP materialization"
            )
        if type(moe_expert_parallel_size) is not int or moe_expert_parallel_size <= 0:
            raise ValueError(
                "DECODE_FFN moe_expert_parallel_size must be an exact positive int for EP materialization"
            )
        if type(router_topk) is not int or router_topk <= 0:
            raise ValueError(
                "DECODE_FFN router_topk must be an exact positive int for EP materialization"
            )
        expert_to_ep = build_contiguous_expert_ownership(
            total_expert_num,
            moe_expert_parallel_size,
        )
        return materialize_layer_ep_workload(
            routing_ratios=resolve_routing_details(
                routing_details,
                target_replica_id=replica_id,
                global_layer_id=layer_global_id,
            ),
            target_replica_id=replica_id,
            global_layer_id=layer_global_id,
            routing_token_count=routing_token_count,
            router_topk=router_topk,
            total_expert_num=total_expert_num,
            moe_expert_parallel_size=moe_expert_parallel_size,
            expert_to_ep=expert_to_ep,
        )

    def _prepare_ep_batch_group_plan(
        self,
        group: List[Batch],
        replica_id,
        ep_id,
        expert_global_ids,
        layer_global_id,
        routing_details,
        layer_workload: Optional[LayerEPWorkload] = None,
    ) -> EPBatchGroupPlan:
        """Prepare one EP batch without constructing entities or mutating caches."""

        if type(group) is not list:
            raise ValueError("group must be a list")
        if len(group) == 0:
            raise ValueError("group must be non-empty")

        if type(replica_id) is not int or replica_id < 0:
            raise ValueError(
                "DECODE_FFN EP replica_id must be an exact non-negative int, "
                f"got {replica_id!r}"
            )
        if type(ep_id) is not int or ep_id < 0:
            raise ValueError(
                "DECODE_FFN ep_id must be an exact non-negative int, "
                f"got {ep_id!r}"
            )
        if type(layer_global_id) is not int or layer_global_id < 0:
            raise ValueError(
                "DECODE_FFN layer_global_id must be an exact non-negative int, "
                f"got {layer_global_id!r}"
            )
        if type(expert_global_ids) is not list or any(
            type(expert_id) is not int or expert_id < 0
            for expert_id in expert_global_ids
        ):
            raise ValueError(
                "DECODE_FFN expert_global_ids must be an exact list of "
                "non-negative ints"
            )
        if type(routing_details) is not dict:
            raise ValueError("DECODE_FFN routing_details must be an exact dict")

        afd_stage_idx_values = {getattr(batch, "afd_stage_idx", None) for (batch, _) in group}
        if None in afd_stage_idx_values:
            raise ValueError("afd_stage_idx missing in DECODE_FFN group batches")
        if len(afd_stage_idx_values) != 1:
            raise ValueError(
                f"afd_stage_idx mismatch in group: {sorted(afd_stage_idx_values)}"
            )
        afd_stage_idx = afd_stage_idx_values.pop()
        if type(afd_stage_idx) is not int or afd_stage_idx < 0:
            raise ValueError(
                "DECODE_FFN afd_stage_idx must be an exact non-negative int, "
                f"got {afd_stage_idx!r}"
            )

        # ISSUE-007 FIX: Validate that source batches have required decode_attn_original_* attributes
        # This validation helps identify where the attributes are lost in the A→F transfer path
        if self._cluster_type == ClusterType.DECODE_FFN:
            for (batch, _) in group:
                orig_replica_id = getattr(batch, 'decode_attn_original_replica_id', None)
                orig_replica_local_id = getattr(batch, 'decode_attn_original_replica_local_id', None)
                if orig_replica_id is None:
                    raise ValueError(
                        f"[ISSUE-007] Batch {batch.id} entering DECODE_FFN without "
                        "decode_attn_original_replica_id. "
                        f"decode_attn_original_replica_id={orig_replica_id}, "
                        f"decode_attn_original_replica_local_id={orig_replica_local_id}. "
                        "The A→F source identity is the attention serving Replica; "
                        "its full-stage local identity is None."
                    )
                if type(orig_replica_id) is not int or orig_replica_id < 0:
                    raise ValueError(
                        "DECODE_FFN source decode_attn_original_replica_id must be "
                        f"an exact non-negative int, got {orig_replica_id!r}"
                    )
                if orig_replica_local_id is not None and (
                    type(orig_replica_local_id) is not int or orig_replica_local_id < 0
                ):
                    raise ValueError(
                        "DECODE_FFN source decode_attn_original_replica_local_id must be None "
                        f"or an exact non-negative int, got {orig_replica_local_id!r}"
                    )

        source_batches = []
        source_batch_ids = []
        ep_batch_group_total_num_token = 0
        pre_routing_effective_total_tokens = 0
        for (batch, _) in group:
            batch_id = batch.id
            if type(batch_id) is not int or batch_id < 0:
                raise ValueError(
                    "DECODE_FFN source batch id must be an exact non-negative int, "
                    f"got {batch_id!r}"
                )
            batch_num_tokens = batch.num_tokens
            if type(batch_num_tokens) is not list or any(
                type(num_tokens) is not int or num_tokens < 0
                for num_tokens in batch_num_tokens
            ):
                raise ValueError(
                    "DECODE_FFN source batch num_tokens must be an exact list of "
                    "non-negative ints"
                )
            batch_total_num_tokens = batch.total_num_tokens
            if (
                type(batch_total_num_tokens) is not int
                or batch_total_num_tokens < 0
                or sum(batch_num_tokens) != batch_total_num_tokens
            ):
                raise ValueError(
                    "DECODE_FFN source batch total_num_tokens must exactly equal "
                    "the sum of num_tokens"
                )
            source_batches.append(batch)
            source_batch_ids.append(batch.id)
            ep_batch_group_total_num_token += batch_total_num_tokens
            pre_routing_effective_total_tokens += int(
                batch.get_effective_total_tokens_for_compute(ClusterType.DECODE_FFN)
            )

        if pre_routing_effective_total_tokens <= 0:
            raise ValueError(
                "DECODE_FFN EP group requires positive pre-routing effective "
                f"tokens, got {pre_routing_effective_total_tokens}"
            )

        router_topk = self._config.replica_config.router_topk
        if type(router_topk) is not int or router_topk <= 0:
            raise ValueError(
                "DECODE_FFN router_topk must be an exact positive int, "
                f"got {router_topk!r}"
            )
        group_time = max((b.time or 0.0) for (b, _) in group)

        replica_config = getattr(self._config, "replica_config", None)
        if replica_config is None:
            raise ValueError("DECODE_FFN requires replica_config for EP materialization")
        total_expert_num = getattr(replica_config, "total_expert_num", None)
        moe_expert_parallel_size = getattr(
            replica_config,
            "moe_expert_parallel_size",
            None,
        )
        if type(total_expert_num) is not int or total_expert_num <= 0:
            raise ValueError(
                "DECODE_FFN total_expert_num must be an exact positive int for "
                "EP materialization"
            )
        if type(moe_expert_parallel_size) is not int or moe_expert_parallel_size <= 0:
            raise ValueError(
                "DECODE_FFN moe_expert_parallel_size must be an exact positive int "
                "for EP materialization"
            )
        expert_to_ep = build_contiguous_expert_ownership(
            total_expert_num,
            moe_expert_parallel_size,
        )
        expected_expert_ids = [
            expert_id
            for expert_id in range(total_expert_num)
            if expert_to_ep[expert_id] == ep_id
        ]
        if sorted(expert_global_ids) != expected_expert_ids:
            raise ValueError(
                "DECODE_FFN expert_global_ids do not match contiguous ownership "
                f"for ep_id={ep_id}: expected={expected_expert_ids}, "
                f"got={expert_global_ids}"
            )
        if layer_workload is None:
            layer_workload = self._materialize_ep_wave_workload(
                group,
                replica_id,
                layer_global_id,
                routing_details,
            )
        if not isinstance(layer_workload, LayerEPWorkload):
            raise ValueError(
                "DECODE_FFN shared EP workload must be a LayerEPWorkload instance"
            )
        if layer_workload.target_replica_id != replica_id:
            raise ValueError(
                "DECODE_FFN shared EP workload target replica mismatch: "
                f"expected={replica_id}, got={layer_workload.target_replica_id}"
            )
        if layer_workload.global_layer_id != layer_global_id:
            raise ValueError(
                "DECODE_FFN shared EP workload layer mismatch: "
                f"expected={layer_global_id}, got={layer_workload.global_layer_id}"
            )
        if layer_workload.routing_token_count != ep_batch_group_total_num_token:
            raise ValueError(
                "DECODE_FFN shared EP workload routing-token mismatch: "
                f"expected={ep_batch_group_total_num_token}, "
                f"got={layer_workload.routing_token_count}"
            )
        experts_tokens_mapping = dict(
            layer_workload.per_ep_per_expert_tokens[ep_id]
        )
        ep_batch_group_total_num_token = layer_workload.per_ep_routed_tokens[ep_id]

        self._validate_token_conservation(
            input_tokens=ep_batch_group_total_num_token,
            per_expert_tokens=experts_tokens_mapping,
            context=f"_prepare_ep_batch_group_plan (cluster={self._cluster_type.name}, "
                   f"replica={replica_id}, ep_id={ep_id}, layer={layer_global_id})"
        )

        return EPBatchGroupPlan(
            replica_id=replica_id,
            ep_id=ep_id,
            layer_global_id=layer_global_id,
            afd_stage_idx=afd_stage_idx,
            group_time=group_time,
            pre_routing_effective_total_tokens=pre_routing_effective_total_tokens,
            source_batches=tuple(source_batches),
            source_batch_ids=tuple(source_batch_ids),
            per_expert_tokens=tuple(
                (expert_id, experts_tokens_mapping[expert_id])
                for expert_id in expert_global_ids
            ),
        )

    def _materialize_ep_batch_group(
        self,
        plan: EPBatchGroupPlan,
    ) -> EPBatchGroup:
        """Materialize one already validated DECODE_FFN EP batch plan."""

        experts_tokens_mapping = dict(plan.per_expert_tokens)
        logic_num_tokens = list(experts_tokens_mapping.values())
        logic_requests = [
            Request(0.0, 0, num_tokens) for num_tokens in logic_num_tokens
        ]
        ep_batch_group = self._create_batch_group(
            logic_requests,
            logic_num_tokens,
            plan.replica_id,
            plan.ep_id,
            plan.group_time,
            list(plan.source_batch_ids),
            experts_tokens_mapping,
        )
        ep_batch_group.afd_stage_idx = plan.afd_stage_idx
        ep_batch_group.decode_ffn_layer_id = plan.layer_global_id
        ep_batch_group.moe_pre_routing_effective_total_tokens = (
            plan.pre_routing_effective_total_tokens
        )
        ep_batch_group.source_batches = list(plan.source_batches)

        return ep_batch_group

    def _distribute_tokens_within_ep_replica(
        self,
        group: List[Batch],
        replica_id,
        ep_id,
        expert_global_ids,
        layer_global_id,
        routing_details,
        layer_workload: Optional[LayerEPWorkload] = None,
    ) -> EPBatchGroup:
        """Build one EP batch without committing scheduler-owned state."""

        plan = self._prepare_ep_batch_group_plan(
            group,
            replica_id,
            ep_id,
            expert_global_ids,
            layer_global_id,
            routing_details,
            layer_workload=layer_workload,
        )
        return self._materialize_ep_batch_group(plan)

    # Phase 2.5: Removed deprecated on_moe_ready() method
    # Old MoE synchronization architecture is no longer supported.
    # Current architecture uses explicit EP dispatch and combine synchronization.

    def _get_step3_ep_alltoall_payload_bytes(self, ep_batches):
        """Return per-lane payload summary for trace-driven Step3 EP all-to-all.

        Collective-sim currently accepts a single ``tensor_bytes`` scalar per
        collective. Under trace-driven MoE routing, Step3 EP lanes can carry
        different post-routing token counts, so the collective completion time
        must be driven by the largest local payload rather than the first lane
        that happened to arrive at the barrier.
        """
        if not ep_batches:
            raise ValueError("Step3 EP all-to-all payload requested with no EP batches")

        hidden_size = int(self._config.replica_config.model_config.embedding_dim)
        local_tokens_by_ep_id = {}
        for lane_ep_id, ep_batch in ep_batches.items():
            local_tokens = int(getattr(ep_batch, "total_num_tokens", 0))
            if local_tokens < 0:
                raise ValueError(
                    "EP batch has negative local token count for Step3 all-to-all: "
                    f"ep_id={lane_ep_id}, total_num_tokens={local_tokens}"
                )
            local_tokens_by_ep_id[int(lane_ep_id)] = local_tokens

        max_local_tokens = max(local_tokens_by_ep_id.values(), default=0)
        data_size_bytes = max_local_tokens * hidden_size * 2
        return data_size_bytes, local_tokens_by_ep_id, max_local_tokens, hidden_size

    def _validate_ep_barrier_arrival(
        self,
        *,
        phase: str,
        waiting_rooms,
        replica_id: int,
        stage_id: int,
        batch,
        ep_id: int,
    ) -> tuple[int, Optional[dict], frozenset[int], bool]:
        """Validate one EP barrier participant without mutating waiting-room state."""

        if type(replica_id) is not int or replica_id < 0:
            raise ValueError(
                f"EP {phase} replica_id must be an exact non-negative int, "
                f"got {replica_id!r}"
            )
        if type(stage_id) is not int or stage_id < 0:
            raise ValueError(
                f"EP {phase} stage_id must be an exact non-negative int, "
                f"got {stage_id!r}"
            )

        batch_global_id = batch.global_id
        if type(batch_global_id) is not int or batch_global_id < 0:
            raise ValueError(
                f"EP {phase} batch global_id must be an exact non-negative int, "
                f"got {batch_global_id!r}"
            )

        batch_replica_id = getattr(batch, "replica_id", None)
        if type(batch_replica_id) is not int or batch_replica_id < 0:
            raise ValueError(
                f"EP {phase} batch replica_id must be an exact non-negative int, "
                f"got {batch_replica_id!r}"
            )
        if batch_replica_id != replica_id:
            raise ValueError(
                f"EP {phase} event/batch replica_id mismatch: "
                f"event={replica_id!r}, batch={batch_replica_id!r}"
            )

        replica = self.get_replica(replica_id)
        expected_ep_size = getattr(
            replica,
            "ep_size",
            self._config.replica_config.moe_expert_parallel_size,
        )
        if type(expected_ep_size) is not int or expected_ep_size <= 0:
            raise ValueError(
                f"EP {phase} expected_ep_size must be an exact positive int, "
                f"got {expected_ep_size!r}"
            )
        expected_ep_ids = frozenset(range(expected_ep_size))

        if type(ep_id) is not int or ep_id not in expected_ep_ids:
            raise ValueError(
                f"EP {phase} ep_id must be an exact int in "
                f"{sorted(expected_ep_ids)}, got {ep_id!r}"
            )

        batch_ep_id = getattr(batch, "ep_id", None)
        if type(batch_ep_id) is not int or batch_ep_id != ep_id:
            raise ValueError(
                f"EP {phase} event/batch ep_id mismatch: "
                f"event={ep_id!r}, batch={batch_ep_id!r}"
            )

        replica_rooms = waiting_rooms.get(replica_id)
        stage_rooms = (
            replica_rooms.get(stage_id) if replica_rooms is not None else None
        )
        room = (
            stage_rooms.get(batch_global_id) if stage_rooms is not None else None
        )
        if room is None:
            existing_ep_ids = set()
        else:
            batch_ep_ids = set(room["batches"])
            arrival_ep_ids = set(room["arrival_times"])
            if batch_ep_ids != arrival_ep_ids:
                raise ValueError(
                    f"EP {phase} waiting-room batch/arrival key mismatch: "
                    f"batches={sorted(batch_ep_ids, key=repr)}, "
                    f"arrival_times={sorted(arrival_ep_ids, key=repr)}"
                )
            if any(type(existing_ep_id) is not int for existing_ep_id in batch_ep_ids):
                raise ValueError(
                    f"EP {phase} waiting room contains a non-exact ep_id: "
                    f"{sorted(batch_ep_ids, key=repr)}"
                )
            if not batch_ep_ids.issubset(expected_ep_ids):
                raise ValueError(
                    f"EP {phase} waiting-room lane set is outside the expected "
                    f"ep_id domain: lanes={sorted(batch_ep_ids)}, "
                    f"expected={sorted(expected_ep_ids)}"
                )
            for lane_ep_id, stored_batch in room["batches"].items():
                stored_global_id = getattr(stored_batch, "global_id", None)
                if (
                    type(stored_global_id) is not int
                    or stored_global_id != batch_global_id
                ):
                    raise ValueError(
                        f"EP {phase} waiting-room batch global_id mismatch: "
                        f"room={batch_global_id!r}, stored={stored_global_id!r}, "
                        f"lane={lane_ep_id!r}"
                    )
                stored_ep_id = getattr(stored_batch, "ep_id", None)
                if type(stored_ep_id) is not int or stored_ep_id != lane_ep_id:
                    raise ValueError(
                        f"EP {phase} waiting-room batch ep_id mismatch: "
                        f"lane={lane_ep_id!r}, stored={stored_ep_id!r}"
                    )
                stored_replica_id = getattr(stored_batch, "replica_id", None)
                if (
                    type(stored_replica_id) is not int
                    or stored_replica_id != replica_id
                ):
                    raise ValueError(
                        f"EP {phase} waiting-room batch replica_id mismatch: "
                        f"event={replica_id!r}, stored={stored_replica_id!r}, "
                        f"lane={lane_ep_id!r}"
                    )
            existing_ep_ids = batch_ep_ids

        if ep_id in existing_ep_ids:
            raise ValueError(
                f"EP {phase} duplicate ep_id arrival: ep_id={ep_id}, "
                f"global_id={batch_global_id}"
            )

        arrived_ep_ids = existing_ep_ids | {ep_id}
        return (
            batch_global_id,
            room,
            expected_ep_ids,
            arrived_ep_ids == expected_ep_ids,
        )

    @staticmethod
    def _validate_ep_collective_exec_time(
        *,
        phase: str,
        exec_time_ms,
        sync_time,
    ) -> tuple[float, float]:
        """Validate a collective latency and derive its event time.

        The predictor result is part of the DES state transition.  It must be
        validated before the final lane is committed so that predictor errors
        cannot leave a complete waiting room without a corresponding event.
        """

        if not isinstance(exec_time_ms, Real) or isinstance(exec_time_ms, bool):
            raise ValueError(
                f"EP {phase} collective latency must be an exact int or float, "
                f"got {exec_time_ms!r}"
            )
        if not math.isfinite(float(exec_time_ms)):
            raise ValueError(
                f"EP {phase} collective latency must be finite, "
                f"got {exec_time_ms!r}"
            )
        if exec_time_ms < 0:
            raise ValueError(
                f"EP {phase} collective latency must be non-negative, "
                f"got {exec_time_ms!r}"
            )
        if (
            not isinstance(sync_time, Real)
            or isinstance(sync_time, bool)
            or not math.isfinite(float(sync_time))
        ):
            raise ValueError(
                f"EP {phase} collective sync time must be finite, "
                f"got {sync_time!r}"
            )

        exec_time_value = float(exec_time_ms)
        collective_event_time = float(sync_time) + exec_time_value / 1000.0
        if not math.isfinite(collective_event_time):
            raise ValueError(
                f"EP {phase} collective event time must be finite, "
                f"got {collective_event_time!r}"
            )
        if collective_event_time < float(sync_time):
            raise ValueError(
                f"EP {phase} collective event time cannot precede its sync time: "
                f"sync={sync_time!r}, event={collective_event_time!r}"
            )
        return exec_time_value, collective_event_time

    def on_ep_alltoall_dispatch_ready(
        self, time: float, replica_id: int, stage_id: int, batch, ep_id: int
    ):
        """Handle EP dispatch readiness before expert compute begins."""
        from frontier.events.ep_alltoall_dispatch_collective_event import (
            EPAllToAllDispatchCollectiveEvent,
        )
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        if (
            not isinstance(time, Real)
            or isinstance(time, bool)
            or not math.isfinite(float(time))
        ):
            raise ValueError(
                f"EP dispatch arrival time must be a finite int or float, got {time!r}"
            )
        time = float(time)

        (
            batch_global_id,
            dispatch_wait_room,
            expected_ep_ids,
            is_complete,
        ) = self._validate_ep_barrier_arrival(
            phase="dispatch",
            waiting_rooms=self._ep_alltoall_dispatch_waiting_room,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            ep_id=ep_id,
        )

        existing_batches = (
            {} if dispatch_wait_room is None else dispatch_wait_room["batches"]
        )
        existing_arrival_times = (
            {}
            if dispatch_wait_room is None
            else dispatch_wait_room["arrival_times"]
        )
        prospective_batches = dict(existing_batches)
        prospective_arrival_times = dict(existing_arrival_times)
        prospective_batches[ep_id] = batch
        prospective_arrival_times[ep_id] = time

        expected_ep_size = len(expected_ep_ids)
        if not is_complete:
            if dispatch_wait_room is None:
                dispatch_wait_room = self._ep_alltoall_dispatch_waiting_room[
                    replica_id
                ][stage_id][batch_global_id]
            dispatch_wait_room["batches"][ep_id] = batch
            dispatch_wait_room["arrival_times"][ep_id] = time
            return []

        prospective_room = {
            "batches": prospective_batches,
            "arrival_times": prospective_arrival_times,
        }

        (
            data_size_bytes,
            local_tokens_by_ep_id,
            max_local_tokens,
            hidden_size,
        ) = self._get_step3_ep_alltoall_payload_bytes(prospective_batches)

        ep_collective_exec_time_ms = self._predictor.predict_alltoall_time(
            data_size_bytes=data_size_bytes,
            num_devices=expected_ep_size,
            cluster_type=self._cluster_type,
            comm_domain="EP",
        )
        ep_collective_sync_time = max(prospective_arrival_times.values())
        (
            ep_collective_exec_time_ms,
            collective_event_time,
        ) = self._validate_ep_collective_exec_time(
            phase="dispatch",
            exec_time_ms=ep_collective_exec_time_ms,
            sync_time=ep_collective_sync_time,
        )

        if dispatch_wait_room is None:
            dispatch_wait_room = self._ep_alltoall_dispatch_waiting_room[
                replica_id
            ][stage_id][batch_global_id]
        dispatch_wait_room["batches"][ep_id] = batch
        dispatch_wait_room["arrival_times"][ep_id] = time
        logger.info(
            f"[EP-DISPATCH][COLLECTIVE] global_id={batch_global_id}, "
            f"sync_time={ep_collective_sync_time:.6f}s, "
            f"exec_time={ep_collective_exec_time_ms:.6f}ms, "
            f"collective_end={collective_event_time:.6f}s, "
            f"max_local_tokens={max_local_tokens}, hidden_size={hidden_size}, "
            f"local_tokens_by_ep_id={local_tokens_by_ep_id}"
        )

        return [
            EPAllToAllDispatchCollectiveEvent(
                collective_event_time, replica_id, stage_id, batch_global_id
            )
        ]

    def on_ep_alltoall_dispatch_collective_schedule(
        self, time: float, replica_id: int, stage_id: int, batch_global_id: int
    ):
        """Advance EP batches into expert compute after dispatch collective finishes."""
        from frontier.events.ep_alltoall_combine_ready_event import (
            EPAllToAllCombineReadyEvent,
        )

        dispatch_wait_room = self._ep_alltoall_dispatch_waiting_room[replica_id][
            stage_id
        ][batch_global_id]
        ep_batches = dispatch_wait_room["batches"]
        if not ep_batches:
            raise ValueError(
                f"EP dispatch collective reached with empty ep_batches: "
                f"global_id={batch_global_id}"
            )

        prepared_lanes = []
        for lane_ep_id, ep_batch in ep_batches.items():
            expert_compute_time = getattr(ep_batch, "expert_compute_time", None)
            if expert_compute_time is None:
                raise ValueError(
                    f"Missing expert_compute_time for EP batch {ep_batch.id} "
                    f"(global_id={batch_global_id}, ep_id={lane_ep_id})"
                )
            ready_time = time + expert_compute_time
            prepared_lanes.append(
                (
                    ep_batch,
                    ready_time,
                    EPAllToAllCombineReadyEvent(
                        ready_time, replica_id, stage_id, ep_batch, lane_ep_id
                    ),
                )
            )

        self._ep_alltoall_dispatch_waiting_room[replica_id][stage_id].pop(
            batch_global_id
        )
        for ep_batch, ready_time, _ in prepared_lanes:
            ep_batch.time = ready_time

        return [event for _, _, event in prepared_lanes]

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

        if (
            not isinstance(time, Real)
            or isinstance(time, bool)
            or not math.isfinite(float(time))
        ):
            raise ValueError(
                f"EP combine arrival time must be a finite int or float, got {time!r}"
            )
        time = float(time)

        (
            batch_global_id,
            ep_wait_room,
            expected_ep_ids,
            is_complete,
        ) = self._validate_ep_barrier_arrival(
            phase="combine",
            waiting_rooms=self._ep_allgather_waiting_room,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            ep_id=ep_id,
        )

        existing_batches = {} if ep_wait_room is None else ep_wait_room["batches"]
        existing_arrival_times = (
            {} if ep_wait_room is None else ep_wait_room["arrival_times"]
        )
        prospective_batches = dict(existing_batches)
        prospective_arrival_times = dict(existing_arrival_times)
        prospective_batches[ep_id] = batch
        prospective_arrival_times[ep_id] = time

        expected_ep_size = len(expected_ep_ids)

        if not is_complete:
            if ep_wait_room is None:
                ep_wait_room = self._ep_allgather_waiting_room[replica_id][stage_id][
                    batch_global_id
                ]
            ep_wait_room["batches"][ep_id] = batch
            ep_wait_room["arrival_times"][ep_id] = time

        # DIAGNOSTIC: Log EP combine ready with global_id after validation has
        # established that this arrival can be committed safely.
        logger.info(
            f"[EP-WAIT-ROOM][ENTER] time={time:.3f}s, batch_id={batch.id}, global_id={batch_global_id}, "
            f"replica={replica_id}, stage={stage_id}, ep_id={ep_id}"
        )

        # DIAGNOSTIC: Log wait room status with all waiting batches
        arrived_ep_ids = list(prospective_batches.keys())
        arrived_batch_ids = [prospective_batches[eid].id for eid in arrived_ep_ids]
        logger.info(
            f"[EP-WAIT-ROOM][STATUS] global_id={batch_global_id}, "
            f"arrived={len(prospective_batches)}/{expected_ep_size}, "
            f"ep_ids={arrived_ep_ids}, batch_ids={arrived_batch_ids}"
        )

        # Check if all EP replicas in this replica have arrived
        if is_complete:
            # Synchronize to the maximum time across all EP replicas
            logger.info(
                "[DEBUG] All EP replicas arrived! Creating EPAllToAllCombineCollectiveEvent"
            )

            # Phase 1: Migrated to new unified API
            # Calculate data_size_bytes for EP combine based on batch information
            # Use the first batch as representative (all EP batches should have similar size)
            representative_batch = list(prospective_batches.values())[0]
            total_tokens = representative_batch.total_num_tokens

            # Get model embedding dimension from replica config
            model_config = self._config.replica_config.model_config
            hidden_size = model_config.embedding_dim

            # Calculate data size: tokens × hidden_size × 2 bytes (float16)
            data_size_bytes = total_tokens * hidden_size * 2

            ep_collective_kind = resolve_ep_collective_kind(
                model_config,
                self._cluster_type,
                expected_ep_size,
            )
            if ep_collective_kind is ExpertParallelCollective.ALLTOALL:
                (
                    data_size_bytes,
                    local_tokens_by_ep_id,
                    max_local_tokens,
                    hidden_size,
                ) = self._get_step3_ep_alltoall_payload_bytes(
                    prospective_batches
                )
                payload_description = (
                    f"max_local_tokens={max_local_tokens}, "
                    f"hidden_size={hidden_size}, "
                    f"local_tokens_by_ep_id={local_tokens_by_ep_id}"
                )
                # EP alltoall combine phase
                ep_collective_exec_time_ms = self._predictor.predict_alltoall_time(
                    data_size_bytes=data_size_bytes,
                    num_devices=expected_ep_size,
                    cluster_type=self._cluster_type,
                    comm_domain="EP",
                )
            else:
                payload_description = (
                    f"{total_tokens} tokens × {hidden_size} hidden_size"
                )
                ep_collective_exec_time_ms = self._predictor.predict_allgather_time(
                    data_size_bytes=data_size_bytes,
                    num_devices=expected_ep_size,
                    cluster_type=self._cluster_type,
                    comm_domain="EP",
                )

            ep_collective_sync_time = max(prospective_arrival_times.values())
            (
                ep_collective_exec_time_ms,
                collective_event_time,
            ) = self._validate_ep_collective_exec_time(
                phase="combine",
                exec_time_ms=ep_collective_exec_time_ms,
                sync_time=ep_collective_sync_time,
            )

            if ep_wait_room is None:
                ep_wait_room = self._ep_allgather_waiting_room[replica_id][stage_id][
                    batch_global_id
                ]
            ep_wait_room["batches"][ep_id] = batch
            ep_wait_room["arrival_times"][ep_id] = time

            logger.info(
                f"[DEBUG] Creating EPAllToAllCombineCollectiveEvent at time={collective_event_time:.3f}s, "
                f"sync_time={ep_collective_sync_time:.3f}s, exec_time={ep_collective_exec_time_ms:.3f}ms, "
                f"data_size={data_size_bytes} bytes ({payload_description})"
            )

            return [
                EPAllToAllCombineCollectiveEvent(
                    collective_event_time, replica_id, stage_id, batch_global_id
                )
            ]
        else:
            logger.info(f"[DEBUG] Waiting for more EP replicas: {len(ep_wait_room['batches'])}/{expected_ep_size}")

        return []

    @staticmethod
    def _resolve_ep_execution_time(ep_batches: Dict[int, EPBatchGroup]) -> float:
        """Resolve synchronized FFN time while preserving zero-work lane semantics.

        An EP lane with no routed tokens has no local FFN work and therefore a
        predictor result of exactly zero is valid.  It must not contribute a
        fabricated duration to the synchronized request metric.  A zero result
        for a lane that does carry routed tokens is invalid and fails fast.
        """
        positive_execution_times: list[float] = []
        for ep_id, ep_batch in ep_batches.items():
            execution_time = getattr(ep_batch, "execution_time", None)
            if (
                not isinstance(execution_time, Real)
                or isinstance(execution_time, bool)
                or not math.isfinite(float(execution_time))
                or float(execution_time) < 0.0
            ):
                raise ValueError(
                    f"Invalid execution_time for EP batch: ep_id={ep_id}, "
                    f"execution_time={execution_time!r}"
                )

            per_expert_tokens = getattr(ep_batch, "per_expert_tokens", None)
            if not isinstance(per_expert_tokens, dict):
                raise ValueError(
                    "EP batch must expose per_expert_tokens when resolving "
                    f"execution_time: ep_id={ep_id}"
                )
            routed_tokens = 0
            for expert_id, token_count in per_expert_tokens.items():
                if (
                    not isinstance(token_count, int)
                    or isinstance(token_count, bool)
                    or token_count < 0
                ):
                    raise ValueError(
                        "EP batch per_expert_tokens must contain exact "
                        f"non-negative ints: ep_id={ep_id}, expert_id={expert_id}, "
                        f"token_count={token_count!r}"
                    )
                routed_tokens += token_count

            execution_time_value = float(execution_time)
            if execution_time_value == 0.0:
                if routed_tokens != 0:
                    raise ValueError(
                        "EP batch has zero execution_time with routed tokens: "
                        f"ep_id={ep_id}, routed_tokens={routed_tokens}"
                    )
                continue

            positive_execution_times.append(execution_time_value)

        if not positive_execution_times:
            raise ValueError(
                "EP combine has no positive execution_time lane with routed tokens"
            )
        return max(positive_execution_times)

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

        ep_wait_room = self._ep_allgather_waiting_room[replica_id][stage_id][
            batch_global_id
        ]
        ep_batches = ep_wait_room["batches"]

        if not ep_batches:
            raise ValueError(
                "EP all-to-all collective reached with empty ep_batches"
            )

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

        # Instead of aggregating the batch, pick raw batches from
        # _raw_batch_waiting_for_m2n_back using a canonical EP lane.
        canonical_ep_id = min(ep_batches.keys())
        raw_batch_ids = list(ep_batches[canonical_ep_id].source_batch_ids)
        for ep_id, ep_batch in ep_batches.items():
            lane_raw_batch_ids = list(ep_batch.source_batch_ids)
            if not lane_raw_batch_ids:
                raise ValueError(
                    f"EP combine has empty source_batch_ids for ep_id={ep_id}"
                )
            if len(set(lane_raw_batch_ids)) != len(lane_raw_batch_ids):
                raise ValueError(
                    "EP combine has duplicate source_batch_ids for "
                    f"ep_id={ep_id}: {lane_raw_batch_ids}"
                )
            if lane_raw_batch_ids != raw_batch_ids:
                raise ValueError(
                    f"source_batch_ids mismatch: ep_id={ep_id} has "
                    f"{lane_raw_batch_ids}, expected {raw_batch_ids}"
                )

        ffn_execution_time = self._resolve_ep_execution_time(ep_batches)
        logger.info(
            f"[FFN-EXEC-TIME] Using EP execution time: "
            f"{ffn_execution_time:.6f}s"
        )

        raw_batches = []
        for batch_id in raw_batch_ids:
            raw_batch = self._raw_batch_waiting_for_m2n_back.get(batch_id)
            if raw_batch is None:
                raise ValueError(
                    f"Missing raw batch for id={batch_id} in "
                    "_raw_batch_waiting_for_m2n_back"
                )
            raw_batches.append((batch_id, raw_batch))

        stage_schedulers = {
            ep_id: self.get_replica_stage_scheduler(
                replica_id, ep_id, stage_id
            )
            for ep_id in ep_batches.keys()
        }
        replica_schedulers = {
            ep_id: self.get_replica_scheduler(replica_id, ep_id)
            for ep_id in ep_batches.keys()
        }
        activation_bytes_by_ep_id = {}
        for ep_id, ep_batch in ep_batches.items():
            activation_bytes = getattr(ep_batch, "activation_bytes", 0)
            activation_bytes_by_ep_id[ep_id] = (
                int(activation_bytes) if activation_bytes else 0
            )

        prepared_raw_commits = []
        m2n_events = []
        for batch_id, raw_batch in raw_batches:
            active_requests = []
            for request, runtime_epoch in zip(
                raw_batch.requests,
                raw_batch.request_runtime_epochs,
            ):
                if int(getattr(request, "runtime_epoch", 0)) != int(
                    runtime_epoch
                ):
                    continue
                active_requests.append(request)

            prepared_events = (
                self._create_m2n_transfer_events_for_aggregated_batch(
                    raw_batch, time
                )
            )
            m2n_events.extend(prepared_events)
            prepared_raw_commits.append(
                (batch_id, raw_batch, active_requests)
            )

        from frontier.events.replica_stage_schedule_event import (
            ReplicaStageScheduleEvent,
        )

        schedule_events = [
            ReplicaStageScheduleEvent(
                time,
                replica_id,
                stage_id,
                self._cluster_type,
                ep_id,
            )
            for ep_id, stage_scheduler in stage_schedulers.items()
            if not callable(getattr(stage_scheduler, "is_empty", None))
            or not bool(stage_scheduler.is_empty())
        ]

        self._ep_allgather_waiting_room[replica_id][stage_id].pop(
            batch_global_id
        )

        # EP execution bypasses BatchStageEndEvent, so release every stage lane
        # only after the complete cohort and all return events are prepared.
        logger.info(
            "[CRITICAL_FIX] Releasing stage scheduler busy state for all EP replicas"
        )
        for ep_id, stage_scheduler in stage_schedulers.items():
            stage_scheduler.on_stage_end()
            logger.info(
                f"[CRITICAL_FIX] Released busy state for replica {replica_id}, "
                f"ep_id {ep_id}, stage {stage_id}"
            )

        logger.info(
            "[CRITICAL_FIX] Decrementing _num_running_batches for all EP "
            "replica schedulers"
        )
        for ep_id, replica_scheduler in replica_schedulers.items():
            replica_scheduler.decrement_num_running_batches()
            logger.info(
                f"[CRITICAL_FIX] Decremented _num_running_batches for replica "
                f"{replica_id}, ep_id {ep_id}, "
                f"new count={replica_scheduler.num_running_batches}"
            )

        for ep_id, replica_scheduler in replica_schedulers.items():
            activation_bytes = activation_bytes_by_ep_id[ep_id]
            if activation_bytes:
                replica_scheduler.release_activation_memory_bytes(
                    activation_bytes
                )
                metrics_store.on_replica_schedule(
                    time,
                    replica_id,
                    replica_scheduler.memory_usage_percent,
                    self._cluster_type,
                    replica_local_id=ep_id,
                )

        for ep_id, ep_batch in ep_batches.items():
            metrics_store.flush_frontier_stage_batch_ledger_row(
                time=time,
                batch_id=ep_batch.id,
                replica_id=replica_id,
                stage_id=stage_id,
                cluster_type=self._cluster_type,
                replica_local_id=ep_id,
                completion_source="ep_alltoall_combine_collective",
            )

        # Record batch-level DECODE_FFN metrics exactly once per raw batch.
        # EP lanes are synchronized here, so we emit metrics from the canonical lane.
        metrics_lane_id = canonical_ep_id
        memory_usage_percent = max(
            replica_scheduler.memory_usage_percent
            for replica_scheduler in replica_schedulers.values()
        )

        for batch_id, raw_batch, active_requests in prepared_raw_commits:
            self._raw_batch_waiting_for_m2n_back.pop(batch_id)

            # ISSUE-007 DIAGNOSTIC: Log batch attributes before F→A transfer
            logger.info(
                f"[ISSUE-007][F2A][CREATE] batch_id={raw_batch.id}, "
                f"decode_attn_original_replica_id={getattr(raw_batch, 'decode_attn_original_replica_id', 'MISSING')}, "
                f"decode_attn_original_replica_local_id={getattr(raw_batch, 'decode_attn_original_replica_local_id', 'MISSING')}"
            )

            # Record DECODE_FFN execution time for each request using the synchronized
            # stage execution time from EP batches.
            for request in active_requests:
                request.on_batch_stage_end(
                    time, ffn_execution_time, ffn_execution_time, self._cluster_type
                )
            logger.info(
                f"[FFN-EXEC-TIME] Recorded execution time for batch {batch_id}: "
                f"execution_time={ffn_execution_time:.6f}s, "
                f"num_requests={len(raw_batch.requests)}"
            )

            metrics_store.on_batch_end(
                time,
                raw_batch,
                replica_id,
                memory_usage_percent,
                self._cluster_type,
                metrics_lane_id,
            )

            raw_batch.time = time

        # Parent ownership is released only after combine, transfer creation,
        # request accounting, and activation cleanup have all completed.
        canonical_ep_batch = ep_batches[canonical_ep_id]
        self.release_stage_admission_for_batch(
            canonical_ep_batch,
            stage_id=stage_id,
        )
        logger.info(f"[DEBUG] Created {len(m2n_events)} M2N transfer events: "
                    f"{[event.event_type.name if event and hasattr(event, 'event_type') and event.event_type else 'Unknown' for event in m2n_events]}")

        return m2n_events + schedule_events

    def _create_m2n_transfer_events_for_aggregated_batch(self, batch, current_time):
        """Create M2N transfer events for aggregated batch to return to decode-attn cluster."""
        from frontier.events.m2n_transfer_start_event import M2NTransferStartEvent
        from frontier.types import ClusterType
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        logger.info(f"[DEBUG] _create_m2n_transfer_events_for_aggregated_batch called: "
                   f"batch_id={batch.id}, time={current_time:.3f}s, "
                   f"num_requests={len(batch.requests)}")

        activation_size, transfer_time = self._m2n_transfer_predictor.get_transfer_info(
            source_cluster_type=ClusterType.DECODE_FFN,
            target_cluster_type=ClusterType.DECODE_ATTN,
            batch=batch,
            replica_config=self._config.replica_config
        )

        layer_id = self._get_current_layer_id_from_batch(batch)

        m2n_event = M2NTransferStartEvent(
            time=current_time,
            source_replica_id=batch.decode_attn_original_replica_id,
            source_replica_local_id=batch.decode_attn_original_replica_local_id,
            source_cluster_type=ClusterType.DECODE_FFN,
            target_cluster_type=ClusterType.DECODE_ATTN,
            batch=batch,
            activation_size_bytes=activation_size,
            transfer_time_ms=transfer_time,
            layer_id=layer_id,
            afd_stage_idx=batch.afd_stage_idx,
        )

        try:
            req_ids = [r.id for r in batch.requests]
            logger.info(
                f"[M2N][F2A][CREATE] batch_id={batch.id} reqs={req_ids} "
                f"batch_global_id={getattr(batch, 'global_id', '?')} "
                f"decode_attn_orig=(replica={getattr(batch, 'decode_attn_original_replica_id', '?')},dp={getattr(batch, 'decode_attn_original_replica_local_id', '?')}) "
                f"target={ClusterType.DECODE_ATTN.name} size={activation_size}B t_ms={transfer_time:.3f}"
            )
        except Exception:
            logger.info(f"[M2N][F2A][CREATE] batch_id={batch.id} (details unavailable)")

        return [m2n_event]

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
    def _materialize_layer_ep_workload_for_batch(
        self,
        *,
        batch: Batch,
        target_replica_id: int,
        global_layer_id: int,
    ):
        """Materialize one canonical per-layer workload for a full-model MoE batch."""

        replica_config = getattr(self._config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        if replica_config is None or model_config is None:
            raise ValueError(
                "Per-layer EP materialization requires replica_config.model_config"
            )
        if not model_config.is_moe:
            raise ValueError(
                "Per-layer EP materialization is invalid for a dense model"
            )
        if type(target_replica_id) is not int or target_replica_id < 0:
            raise ValueError("target_replica_id must be an exact non-negative int")
        if type(global_layer_id) is not int or global_layer_id < 0:
            raise ValueError("global_layer_id must be an exact non-negative int")
        routing_attr_by_cluster = {
            ClusterType.PREFILL: "_prefill_routing_details",
            ClusterType.DECODE: "_decode_routing_details",
            ClusterType.DECODE_FFN: "_decode_ffn_routing_details",
            ClusterType.MONOLITHIC: "_monolithic_routing_details",
        }
        routing_attr = routing_attr_by_cluster.get(self._cluster_type)
        if routing_attr is None:
            raise ValueError(
                "Per-layer EP materialization is unsupported for cluster "
                f"{self._cluster_type!r}"
            )
        routing_details = getattr(self._predictor, routing_attr, None)
        if routing_details is None:
            raise ValueError(
                f"Missing {routing_attr} for {self._cluster_type.name} EP materialization"
            )
        total_expert_num = getattr(replica_config, "total_expert_num", None)
        moe_ep_size = getattr(replica_config, "moe_expert_parallel_size", None)
        router_topk = getattr(replica_config, "router_topk", None)
        if type(total_expert_num) is not int or total_expert_num <= 0:
            raise ValueError("total_expert_num must be an exact positive int")
        if type(moe_ep_size) is not int or moe_ep_size <= 0:
            raise ValueError("moe_expert_parallel_size must be an exact positive int")
        if type(router_topk) is not int or router_topk <= 0:
            raise ValueError("router_topk must be an exact positive int")
        routing_token_count = getattr(batch, "total_num_tokens", None)
        if type(routing_token_count) is not int or routing_token_count < 0:
            raise ValueError(
                "batch.total_num_tokens must be an exact non-negative int for routing"
            )
        expert_to_ep = build_contiguous_expert_ownership(
            total_expert_num,
            moe_ep_size,
        )
        routing_ratios = resolve_routing_details(
            routing_details,
            target_replica_id,
            global_layer_id,
        )
        return materialize_layer_ep_workload(
            routing_ratios=routing_ratios,
            target_replica_id=target_replica_id,
            global_layer_id=global_layer_id,
            routing_token_count=routing_token_count,
            router_topk=router_topk,
            total_expert_num=total_expert_num,
            moe_expert_parallel_size=moe_ep_size,
            expert_to_ep=expert_to_ep,
        )

    def _build_prefill_ep_lane_batch(
        self,
        *,
        source_batch: Batch,
        layer_id: int,
        ep_id: int,
        layer_workload,
    ) -> EPBatchGroup:
        """Build an EP lane batch for predictor evaluation without request mutation."""

        per_expert_tokens = dict(layer_workload.per_ep_per_expert_tokens[ep_id])
        logic_num_tokens = list(per_expert_tokens.values())
        logic_requests = [
            Request(0.0, 0, num_tokens) for num_tokens in logic_num_tokens
        ]
        lane_batch = self._create_batch_group(
            logic_requests,
            logic_num_tokens,
            source_batch.replica_id,
            ep_id,
            getattr(source_batch, "time", 0.0) or 0.0,
            [source_batch.id],
            per_expert_tokens,
        )
        lane_batch.set_global_id(source_batch.global_id)
        lane_batch.source_batches = [source_batch]
        lane_batch.decode_ffn_layer_id = layer_id
        lane_batch.afd_stage_idx = getattr(source_batch, "afd_stage_idx", None)
        effective_tokens_getter = getattr(
            source_batch,
            "get_effective_total_tokens_for_compute",
            None,
        )
        effective_tokens = (
            int(effective_tokens_getter(self._cluster_type))
            if callable(effective_tokens_getter)
            else int(source_batch.total_num_tokens)
        )
        if effective_tokens <= 0:
            raise ValueError(
                "Prefill EP lane requires positive pre-routing effective tokens"
            )
        lane_batch.moe_pre_routing_effective_total_tokens = effective_tokens
        return lane_batch

    def _on_prefill_ep_wave_ready(
        self,
        *,
        time: float,
        replica_id: int,
        stage_id: int,
        batch: Batch,
        layer_id: int,
        replica_local_id: int | None = None,
    ) -> List:
        """Run one layer's FFN wave and schedule its slowest-lane barrier."""

        from frontier.events.prefill_sync_collective_event import (
            PrefillSyncCollectiveEvent,
        )

        if type(time) not in (int, float) or not math.isfinite(float(time)):
            raise ValueError("prefill EP wave time must be finite")
        model_config = self._config.replica_config.model_config
        predictor = self._predictor
        layer_workload = None
        lane_times_ms: list[float] = []
        if model_config.is_moe_layer(layer_id):
            layer_workload = self._materialize_layer_ep_workload_for_batch(
                batch=batch,
                target_replica_id=replica_id,
                global_layer_id=layer_id,
            )
            for ep_id in layer_workload.participant_ep_ids:
                lane_batch = self._build_prefill_ep_lane_batch(
                    source_batch=batch,
                    layer_id=layer_id,
                    ep_id=ep_id,
                    layer_workload=layer_workload,
                )
                execution_time = predictor.predict_stage_execution_time(
                    lane_batch,
                    stage_id,
                    cluster_type=self._cluster_type,
                    num_layers=1,
                    layer_id=layer_id,
                )
                post_attention_getter = getattr(
                    execution_time,
                    "get_single_layer_post_attention_time",
                    None,
                )
                if not callable(post_attention_getter):
                    raise ValueError(
                        "Prefill EP predictor result is missing post-attention timing"
                    )
                lane_time_ms = float(post_attention_getter())
                if not math.isfinite(lane_time_ms) or lane_time_ms < 0:
                    raise ValueError(
                        "Prefill EP lane post-attention time must be finite and non-negative"
                    )
                lane_times_ms.append(lane_time_ms)
            self.transition_stage_admission_for_layer(
                batch,
                stage_id=stage_id,
                layer_id=layer_id,
                scope=EP_WAVE,
                participant_ep_ids=tuple(layer_workload.participant_ep_ids),
            )
        else:
            execution_time = predictor.predict_stage_execution_time(
                batch,
                stage_id,
                cluster_type=self._cluster_type,
                num_layers=1,
                layer_id=layer_id,
            )
            post_attention_getter = getattr(
                execution_time,
                "get_single_layer_post_attention_time",
                None,
            )
            if not callable(post_attention_getter):
                raise ValueError(
                    "Prefill dense predictor result is missing post-attention timing"
                )
            dense_time_ms = float(post_attention_getter())
            if not math.isfinite(dense_time_ms) or dense_time_ms < 0:
                raise ValueError(
                    "Prefill dense post-attention time must be finite and non-negative"
                )
            self.transition_stage_admission_for_layer(
                batch,
                stage_id=stage_id,
                layer_id=layer_id,
                scope=FULL_STAGE_WORLD,
            )
            component_ledger = getattr(
                batch,
                "_prefill_model_execution_components_ms_by_stage",
                None,
            )
            if (
                not isinstance(component_ledger, dict)
                or stage_id not in component_ledger
                or not isinstance(component_ledger[stage_id], list)
            ):
                raise ValueError(
                    "missing PREFILL model-execution component ledger for dense layer: "
                    f"replica={replica_id}, stage={stage_id}, layer={layer_id}, batch={batch.id}"
                )
            component_ledger[stage_id].append(dense_time_ms)
            from frontier.events.dense_layer_complete_event import (
                DenseLayerCompleteEvent,
            )

            return [
                DenseLayerCompleteEvent(
                    time + dense_time_ms * 1e-3,
                    replica_id,
                    stage_id,
                    batch,
                    layer_id,
                    "prefill",
                    self._cluster_type,
                )
            ]

        if not lane_times_ms:
            raise ValueError("Prefill layer wave produced no participant timing")
        barrier_time_ms = max(lane_times_ms)
        component_ledger = getattr(
            batch,
            "_prefill_model_execution_components_ms_by_stage",
            None,
        )
        if (
            not isinstance(component_ledger, dict)
            or stage_id not in component_ledger
            or not isinstance(component_ledger[stage_id], list)
        ):
            raise ValueError(
                "missing PREFILL model-execution component ledger for EP wave: "
                f"replica={replica_id}, stage={stage_id}, layer={layer_id}, batch={batch.id}"
            )
        component_ledger[stage_id].append(barrier_time_ms)
        batch._prefill_ep_wave_lane_times_ms = tuple(lane_times_ms)
        batch._prefill_ep_wave_workload = layer_workload

        sync_room = self._prefill_sync_waiting_room[replica_id][stage_id][
            batch.global_id
        ][layer_id]["post_moe"]
        if sync_room["batches"]:
            raise ValueError(
                "PREFILL EP wave post_moe room already contains a batch: "
                f"replica={replica_id}, stage={stage_id}, layer={layer_id}, "
                f"batch_global_id={batch.global_id}"
            )
        sync_room["batches"][0] = batch
        sync_room["arrival_times"][0] = time + barrier_time_ms * 1e-3
        return [
            PrefillSyncCollectiveEvent(
                time + barrier_time_ms * 1e-3,
                replica_id,
                stage_id,
                batch.global_id,
                "post_moe",
                layer_id,
                cluster_type=self._cluster_type,
            )
        ]

    def _uses_shared_prefill_ep_wave(self, batch: Batch, layer_id: int) -> bool:
        """Return whether the canonical shared-domain PREFILL path is active."""

        if self._cluster_type not in (ClusterType.PREFILL, ClusterType.MONOLITHIC):
            return False
        replica_config = getattr(self._config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        if model_config is None or not getattr(model_config, "is_moe", False):
            return False
        if getattr(replica_config, "attn_data_parallel_size", None) != 1:
            raise ValueError(
                "Shared-domain MoE PREFILL requires attn_data_parallel_size=1"
            )
        if not isinstance(layer_id, int) or layer_id < 0:
            raise ValueError("PREFILL layer_id must be an exact non-negative int")
        if not model_config.is_moe_layer(layer_id):
            return False
        routing_attr = (
            "_prefill_routing_details"
            if self._cluster_type == ClusterType.PREFILL
            else "_monolithic_routing_details"
        )
        routing_details = getattr(self._predictor, routing_attr, None)
        if routing_details is None:
            raise ValueError(f"Missing {routing_attr} for MoE PREFILL")
        return True

    def _uses_shared_prefill_layer_protocol(self, batch: Batch, layer_id: int) -> bool:
        """Return whether a shared-domain MoE model needs layer stepping.

        Mixed models use the same per-layer event loop for both protocols.  The
        dense branch is still a full-stage operation and bypasses routing and EP
        materialization inside ``_on_prefill_ep_wave_ready``; this predicate is
        deliberately broader than ``_uses_shared_prefill_ep_wave`` so dense
        layers cannot fall into the legacy aggregate MoE path.
        """
        if self._cluster_type not in (ClusterType.PREFILL, ClusterType.MONOLITHIC):
            return False
        replica_config = getattr(self._config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        if model_config is None or not getattr(model_config, "is_moe", False):
            return False
        if getattr(replica_config, "attn_data_parallel_size", None) != 1:
            raise ValueError(
                "Shared-domain MoE PREFILL requires attn_data_parallel_size=1"
            )
        if not isinstance(layer_id, int) or layer_id < 0:
            raise ValueError("PREFILL layer_id must be an exact non-negative int")
        if model_config.is_moe_layer(layer_id):
            routing_attr = (
                "_prefill_routing_details"
                if self._cluster_type == ClusterType.PREFILL
                else "_monolithic_routing_details"
            )
            if getattr(self._predictor, routing_attr, None) is None:
                raise ValueError(f"Missing {routing_attr} for MoE PREFILL")
        return True

    def _on_decode_ep_wave_ready(
        self,
        *,
        time: float,
        replica_id: int,
        stage_id: int,
        batch: Batch,
        layer_id: int,
        replica_local_id: int | None = None,
    ) -> List:
        """Run one unified-DECODE layer's local EP wave and barrier."""

        from frontier.events.decode_sync_collective_event import (
            DecodeSyncCollectiveEvent,
        )

        if type(time) not in (int, float) or not math.isfinite(float(time)):
            raise ValueError("decode EP wave time must be finite")
        model_config = self._config.replica_config.model_config
        predictor = self._predictor
        layer_workload = None
        lane_times_ms: list[float] = []
        lane_comm_times_ms: list[float] = []
        if model_config.is_moe_layer(layer_id):
            layer_workload = self._materialize_layer_ep_workload_for_batch(
                batch=batch,
                target_replica_id=replica_id,
                global_layer_id=layer_id,
            )
            for ep_id in layer_workload.participant_ep_ids:
                lane_batch = self._build_prefill_ep_lane_batch(
                    source_batch=batch,
                    layer_id=layer_id,
                    ep_id=ep_id,
                    layer_workload=layer_workload,
                )
                execution_time = predictor.predict_stage_execution_time(
                    lane_batch,
                    stage_id,
                    cluster_type=self._cluster_type,
                    num_layers=1,
                    layer_id=layer_id,
                )
                post_attention_getter = getattr(
                    execution_time,
                    "get_single_layer_post_attention_time",
                    None,
                )
                if not callable(post_attention_getter):
                    raise ValueError(
                        "Decode EP predictor result is missing post-attention timing"
                    )
                lane_time_ms = float(post_attention_getter())
                if not math.isfinite(lane_time_ms) or lane_time_ms < 0:
                    raise ValueError(
                        "Decode EP lane post-attention time must be finite and non-negative"
                    )
                comm_getter = getattr(
                    predictor,
                    "_get_expert_parallel_communication_time",
                    None,
                )
                if callable(comm_getter):
                    lane_comm_ms = float(comm_getter(lane_batch))
                else:
                    lane_comm_ms = float(
                        getattr(execution_time, "expert_parallel_communication_time", 0.0)
                    )
                if not math.isfinite(lane_comm_ms) or lane_comm_ms < 0:
                    raise ValueError(
                        "Decode EP lane communication time must be finite and non-negative"
                    )
                # ``get_single_layer_post_attention_time()`` is the complete
                # post-attention block and therefore already contains the EP
                # communication component.  Keep lane prediction as compute
                # only; the explicit post-MoE transition below accounts for
                # the collective exactly once.
                lane_compute_ms = lane_time_ms - lane_comm_ms
                if not math.isfinite(lane_compute_ms) or lane_compute_ms < 0:
                    raise ValueError(
                        "Decode EP lane compute time must remain non-negative "
                        "after removing the explicit EP collective"
                    )
                lane_times_ms.append(lane_compute_ms)
                lane_comm_times_ms.append(lane_comm_ms)
            self.transition_stage_admission_for_layer(
                batch,
                stage_id=stage_id,
                layer_id=layer_id,
                scope=EP_WAVE,
                participant_ep_ids=tuple(layer_workload.participant_ep_ids),
            )
        else:
            execution_time = predictor.predict_stage_execution_time(
                batch,
                stage_id,
                cluster_type=self._cluster_type,
                num_layers=1,
                layer_id=layer_id,
            )
            post_attention_getter = getattr(
                execution_time,
                "get_single_layer_post_attention_time",
                None,
            )
            if not callable(post_attention_getter):
                raise ValueError(
                    "Decode dense predictor result is missing post-attention timing"
                )
            dense_time_ms = float(post_attention_getter())
            if not math.isfinite(dense_time_ms) or dense_time_ms < 0:
                raise ValueError(
                    "Decode dense post-attention time must be finite and non-negative"
                )
            self.transition_stage_admission_for_layer(
                batch,
                stage_id=stage_id,
                layer_id=layer_id,
                scope=FULL_STAGE_WORLD,
            )
            from frontier.events.dense_layer_complete_event import (
                DenseLayerCompleteEvent,
            )

            return [
                DenseLayerCompleteEvent(
                    time + dense_time_ms * 1e-3,
                    replica_id,
                    stage_id,
                    batch,
                    layer_id,
                    "decode",
                    self._cluster_type,
                )
            ]

        if not lane_times_ms:
            raise ValueError("Decode layer wave produced no participant timing")
        barrier_time_ms = max(lane_times_ms)
        batch._decode_ep_wave_lane_times_ms = tuple(lane_times_ms)
        batch._decode_ep_wave_post_moe_comm_time_s = max(lane_comm_times_ms) * 1e-3

        batch_global_id = self._get_decode_sync_wait_key(batch)
        sync_room = self._decode_sync_waiting_room[replica_id][stage_id][
            batch_global_id
        ][layer_id]["post_moe"]
        if sync_room["batches"]:
            raise ValueError(
                "DECODE EP wave post_moe room already contains a batch: "
                f"replica={replica_id}, stage={stage_id}, layer={layer_id}, "
                f"batch_global_id={batch_global_id}"
            )
        sync_room["batches"][0] = batch
        sync_room["arrival_times"][0] = time + barrier_time_ms * 1e-3
        return [
            DecodeSyncCollectiveEvent(
                time + barrier_time_ms * 1e-3,
                replica_id,
                stage_id,
                batch_global_id,
                "post_moe",
                layer_id,
                cluster_type=self._cluster_type,
            )
        ]

    def _uses_shared_decode_ep_wave(self, batch: Batch, layer_id: int) -> bool:
        """Return whether canonical PDD unified-DECODE EP waves are active."""

        if self._cluster_type not in (ClusterType.DECODE, ClusterType.MONOLITHIC):
            return False
        replica_config = getattr(self._config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        if model_config is None or not getattr(model_config, "is_moe", False):
            return False
        if getattr(replica_config, "attn_data_parallel_size", None) != 1:
            raise ValueError(
                "Shared-domain MoE DECODE requires attn_data_parallel_size=1"
            )
        if not isinstance(layer_id, int) or layer_id < 0:
            raise ValueError("DECODE layer_id must be an exact non-negative int")
        if not model_config.is_moe_layer(layer_id):
            return False
        routing_attr = (
            "_decode_routing_details"
            if self._cluster_type == ClusterType.DECODE
            else "_monolithic_routing_details"
        )
        routing_details = getattr(self._predictor, routing_attr, None)
        if routing_details is None:
            raise ValueError(f"Missing {routing_attr} for MoE DECODE")
        return True

    def _uses_shared_decode_layer_protocol(self, batch: Batch, layer_id: int) -> bool:
        """Return whether a shared-domain DECODE model needs layer stepping."""
        if self._cluster_type not in (ClusterType.DECODE, ClusterType.MONOLITHIC):
            return False
        replica_config = getattr(self._config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        if model_config is None or not getattr(model_config, "is_moe", False):
            return False
        if getattr(replica_config, "attn_data_parallel_size", None) != 1:
            raise ValueError(
                "Shared-domain MoE DECODE requires attn_data_parallel_size=1"
            )
        if not isinstance(layer_id, int) or layer_id < 0:
            raise ValueError("DECODE layer_id must be an exact non-negative int")
        if model_config.is_moe_layer(layer_id):
            routing_attr = (
                "_decode_routing_details"
                if self._cluster_type == ClusterType.DECODE
                else "_monolithic_routing_details"
            )
            if getattr(self._predictor, routing_attr, None) is None:
                raise ValueError(f"Missing {routing_attr} for MoE DECODE")
        return True

    def on_prefill_sync(self, time: float, replica_id: int, stage_id: int, batch: Batch,
                       replica_local_id: int | None, sync_stage: str, layer_id: int, stage_execution_time: float):
        del stage_execution_time, replica_local_id
        if self._prefill_sync_waiting_room is None:
            raise ValueError(
                "PREFILL synchronization is unavailable for a dense model; "
                "dense execution must use the full-stage protocol"
            )
        if sync_stage != "pre_moe":
            raise ValueError(
                "PREFILL synchronization entry must start at pre_moe; "
                "post_moe completion is handled by PrefillSyncCollectiveEvent"
            )
        if not self._uses_shared_prefill_layer_protocol(batch, layer_id):
            raise RuntimeError(
                "Legacy PREFILL DP synchronization is removed; "
                "the current layer must use the canonical per-layer protocol"
            )
        return self._on_prefill_ep_wave_ready(
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            layer_id=layer_id,
        )

    def on_dense_layer_complete(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch: Batch,
        layer_id: int,
        phase: str,
        metrics_store,
    ) -> List:
        """Advance a dense layer without emitting or waiting on an EP collective.

        The existing layer-transition code is shared with the post-MoE path so
        request counters, per-layer attention scheduling, and final stage
        accounting stay identical.  A single local entry is used only as an
        internal handoff to that transition helper; no collective event is
        created and no EP participant is admitted.
        """
        if phase == "prefill":
            batch_global_id = int(batch.global_id)
            return self.on_prefill_sync_collective(
                time,
                replica_id,
                stage_id,
                batch_global_id,
                "post_moe",
                layer_id,
                metrics_store,
                direct_batch=batch,
            )

        if phase == "decode":
            batch_global_id = self._get_decode_sync_wait_key(batch)
            return self.on_decode_sync_collective(
                time,
                replica_id,
                stage_id,
                batch_global_id,
                "post_moe",
                layer_id,
                metrics_store,
                direct_batch=batch,
            )

        raise ValueError(f"Unsupported dense layer completion phase: {phase!r}")

    def on_prefill_sync_collective(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch_global_id: int,
        sync_stage: str,
        layer_id: int,
        metrics_store,
        *,
        direct_batch: Optional[Batch] = None,
    ):
        """
        Handle collective synchronization completion in prefill cluster.

        This method handles completion of a canonical layer-local PREFILL wave.
        The entry side is materialized and predicted by
        ``_on_prefill_ep_wave_ready``; this method receives only the resulting
        ``post_moe`` completion and advances to the next layer or handoff.

        Args:
            time: Synchronized time when the complete EP wave has reached this point
            replica_id: ID of the replica
            stage_id: Pipeline stage ID
            batch_global_id: Global ID of the batch
            sync_stage: only "post_moe" is valid here
            layer_id: Current layer being processed
            metrics_store: Metrics store for recording performance data
        """
        from frontier.events.batch_stage_end_event import BatchStageEndEvent
        from frontier.events.prefill_sync_event import PrefillSyncEvent
        from frontier.logger import get_cluster_logger
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        # Check if this sync_stage has already been processed by another replica
        # This can happen when multiple replicas reach the same sync point and each creates a PrefillSyncCollectiveEvent
        if direct_batch is not None:
            if sync_stage != "post_moe":
                raise ValueError(
                    "Direct dense PREFILL completion is valid only for post_moe transition"
                )
            sync_wait_room = {
                # Dense completion is a full-stage operation.  Keep the
                # internal handoff lane-free; using EP0 here would make the
                # subsequent stage lookup accidentally depend on an EP child.
                "batches": {None: direct_batch},
                "arrival_times": {None: time},
            }
            participant_batches = sync_wait_room["batches"]
        else:
            if sync_stage not in self._prefill_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id]:
                logger.debug(
                    f"[PREFILL_SYNC][COLLECTIVE_SKIP] sync_stage={sync_stage} already processed for "
                    f"replica={replica_id}, stage={stage_id}, batch_global_id={batch_global_id}, layer={layer_id}"
                )
                return []

            # Get the synchronized batches and clean up waiting room
            sync_wait_room = self._prefill_sync_waiting_room[replica_id][stage_id][batch_global_id][layer_id].pop(sync_stage)
            participant_batches = sync_wait_room["batches"]

        try:
            participant_keys = list(participant_batches.keys())
        except Exception:
            participant_keys = []
        logger.info(
            f"[PREFILL_SYNC][COLLECTIVE] ENTER: t={time:.6f}s, replica={replica_id}, stage={stage_id}, "
            f"layer={layer_id}, sync_stage={sync_stage}, batch_global_id={batch_global_id}, "
            f"participant_keys={participant_keys}, "
            f"participant_batches_type={type(participant_batches).__name__}"
        )

        if sync_stage != "post_moe":
            raise ValueError(
                "PREFILL collective completion accepts only post_moe for the "
                "canonical per-layer EP protocol"
            )

        # Shared-domain PREFILL/MONOLITHIC MoE uses the canonical per-layer
        # protocol.  The EP key (currently 0 in the waiting room) identifies
        # the materialized wave, not a scheduler lane for the layer
        # transition after combine.  All downstream stage/metrics/events must
        # therefore use the full-stage identity ``None``.
        events = []

        if sync_stage == "post_moe":
            # post_moe is a synchronization boundary. Model execution for this layer has
            # already been accounted in pre_moe; only layer transition / pipeline handoff
            # remains after this collective.
            sample_batch = next(
                (b for b in participant_batches.values() if not b.is_idle), None
            )
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
                include_moe=False,
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
                self.transition_stage_admission_for_layer(
                    sample_batch,
                    stage_id=stage_id,
                    layer_id=next_layer_id,
                    scope=FULL_STAGE_WORLD,
                )
                next_layer_execution_time = self._predictor.predict_stage_execution_time(
                    sample_batch,
                    stage_id,
                    cluster_type=self._cluster_type,
                    num_layers=1,
                    layer_id=next_layer_id,
                    include_moe=False,
                )
                attention_time_ms = (
                    next_layer_execution_time.get_single_layer_attention_time()
                )
                attention_time = attention_time_ms * 1e-3
                total_time_to_next_sync = attention_time

                for replica_local_id, batch in participant_batches.items():
                    if batch.is_idle:
                        logger.info(
                            f"[PREFILL_SYNC][IDLE_SKIP] Skip next-layer pre_moe scheduling for idle batch {batch.id} "
                            f"(replica={replica_id}, replica_local_id={replica_local_id}, "
                            f"layer={layer_id})"
                        )
                        continue
                    component_ledger = getattr(
                        batch,
                        "_prefill_model_execution_components_ms_by_stage",
                        None,
                    )
                    if (
                        not isinstance(component_ledger, dict)
                        or stage_id not in component_ledger
                        or not isinstance(component_ledger[stage_id], list)
                    ):
                        raise ValueError(
                            "missing PREFILL model-execution component ledger: "
                            f"replica={replica_id}, replica_local_id={replica_local_id}, "
                            f"stage={stage_id}, layer={layer_id}, "
                            f"batch_global_id={batch_global_id}, batch_id={batch.id}"
                        )
                    component_ledger[stage_id].append(attention_time_ms)
                    events.append(
                        PrefillSyncEvent(
                            time + total_time_to_next_sync,
                            replica_id,
                            stage_id,
                            batch,
                            None,
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
                for replica_local_id, batch in participant_batches.items():
                    if batch.is_idle:
                        logger.info(
                            f"[PREFILL_SYNC][IDLE_SKIP] Skip final stage-end for idle batch {batch.id} "
                            f"(replica={replica_id}, replica_local_id={replica_local_id}, "
                            f"layer={layer_id})"
                        )
                        continue

                    stage_identity = None
                    stage_scheduler = self.get_replica_stage_scheduler(
                        replica_id, stage_identity, stage_id
                    )
                    is_last_stage = stage_scheduler.is_last_stage
                    pipeline_time = execution_time.pipeline_time * 1e-3
                    if not hasattr(batch, "_prefill_stage_start_time"):
                        raise ValueError(
                            "missing PREFILL stage start time: "
                            f"replica={replica_id}, replica_local_id={replica_local_id}, "
                            f"stage={stage_id}, layer={layer_id}, "
                            f"batch_global_id={batch_global_id}, batch_id={batch.id}"
                        )
                    original_start_time = batch._prefill_stage_start_time
                    elapsed_stage_wall_time = time - original_start_time
                    if elapsed_stage_wall_time < 0:
                        raise ValueError(
                            "Prefill sync completion time is earlier than the recorded "
                            "stage start time: "
                            f"replica={replica_id}, replica_local_id={replica_local_id}, "
                            f"stage={stage_id}, "
                            f"layer={layer_id}, batch_global_id={batch_global_id}, "
                            f"time={time}, original_start_time={original_start_time}, "
                            f"elapsed_stage_wall_time={elapsed_stage_wall_time}"
                        )

                    component_ledger = getattr(
                        batch,
                        "_prefill_model_execution_components_ms_by_stage",
                        None,
                    )
                    if (
                        not isinstance(component_ledger, dict)
                        or stage_id not in component_ledger
                        or not isinstance(component_ledger[stage_id], list)
                        or not component_ledger[stage_id]
                    ):
                        raise ValueError(
                            "missing PREFILL model-execution component ledger: "
                            f"replica={replica_id}, replica_local_id={replica_local_id}, "
                            f"stage={stage_id}, layer={layer_id}, "
                            f"batch_global_id={batch_global_id}, batch_id={batch.id}"
                        )
                    explicit_model_execution_time = (
                        math.fsum(component_ledger[stage_id]) * 1e-3
                    )

                    stage_cpu_overhead = execution_time.total_time - execution_time.model_time
                    if stage_cpu_overhead < 0:
                        raise ValueError(
                            "Prefill stage CPU overhead cannot be negative: "
                            f"replica={replica_id}, replica_local_id={replica_local_id}, "
                            f"stage={stage_id}, "
                            f"layer={layer_id}, batch_global_id={batch_global_id}, "
                            f"total_time={execution_time.total_time}, "
                            f"model_time={execution_time.model_time}, "
                            f"stage_cpu_overhead={stage_cpu_overhead}"
                        )

                    actual_model_execution_time = (
                        explicit_model_execution_time + pipeline_time
                    )
                    total_final_time = pipeline_time + stage_cpu_overhead
                    completion_time = time + total_final_time
                    actual_execution_time = completion_time - original_start_time

                    # Create batch stage for metrics
                    batch_stage, _ = stage_scheduler.predict_and_create_stage(batch, skip_get_execution_time=True)

                    # Schedule the batch stage with the original start time
                    batch_stage.on_schedule(original_start_time)

                    # Override with correct values:
                    # - execution_time: actual wall-clock time including sync overhead
                    # - model_execution_time: pure model computation time (no CPU overhead)
                    batch_stage.override_execution_time(actual_execution_time)
                    batch_stage.override_model_execution_time(
                        actual_model_execution_time
                    )

                    # TODO: CHECK OVERIDE LOGIC AND METRIC LOGIC HERE
                    # Create a corrected ExecutionTime object for metrics recording.
                    # For mixed-layer MoE models, augment trace-only dense MLP components
                    # from a representative dense layer so op-level traces include both
                    # dense and MoE FFN scopes.
                    corrected_execution_time = (
                        self._create_prefill_corrected_execution_time_for_metrics(
                            sample_batch,
                            stage_id,
                            execution_time,
                            actual_execution_time,
                            original_start_time,
                        )
                    )

                    # Record metrics with correct start time and corrected execution time
                    metrics_store.on_replica_stage_schedule(
                        original_start_time, replica_id, stage_id, batch_stage, corrected_execution_time,
                        self._cluster_type, stage_identity
                    )

                    # Schedule batch stage end
                    events.append(BatchStageEndEvent(
                        completion_time, replica_id, stage_id, is_last_stage,
                        batch, batch_stage, self._cluster_type, stage_identity
                    ))

                    # Check if KV cache transfer should be triggered
                    if self._should_trigger_kv_transfer(batch):
                        kv_transfer_events = self._create_kv_transfer_events(
                            completion_time, batch, replica_id, stage_identity
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
            attn_tensor_parallel_allreduce_time=(
                original_execution_time._attn_tensor_parallel_allreduce_time
                if original_execution_time._has_attn_tensor_parallel_allreduce_time
                else None
            ),
            moe_tensor_parallel_allreduce_time=(
                original_execution_time._moe_tensor_parallel_allreduce_time
                if original_execution_time._has_moe_tensor_parallel_allreduce_time
                else None
            ),
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
        replica_local_id: int | None,
    ) -> List:
        """Disaggregated KV cache transfer events are not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _get_decode_sync_wait_key(self, batch: Batch) -> int:
        decode_sync_global_id = getattr(batch, "decode_sync_global_id", None)
        if decode_sync_global_id is not None:
            return int(decode_sync_global_id)
        return int(batch.global_id)

    def on_decode_sync(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch: Batch,
        replica_local_id: int | None,
        sync_stage: str,
        layer_id: int,
        stage_execution_time: float,
    ):
        """Enter the canonical layer-local DECODE MoE protocol.

        A DECODE sync entry is valid only before the current layer's EP wave.
        The old DP waiting-room and aggregate scalar path is retired; the
        complete replica-local EP participant set is materialized by
        ``_on_decode_ep_wave_ready`` and its collective completion event owns
        the layer transition.
        """
        del stage_execution_time, replica_local_id
        if self._decode_sync_waiting_room is None:
            raise ValueError(
                "DECODE synchronization is unavailable for a dense model; "
                "dense execution must use the full-stage protocol"
            )
        if sync_stage != "pre_moe":
            raise ValueError(
                "DECODE synchronization entry must start at pre_moe; "
                "post_moe completion is handled by DecodeSyncCollectiveEvent"
            )
        if not self._uses_shared_decode_layer_protocol(batch, layer_id):
            raise RuntimeError(
                "Legacy DECODE DP synchronization is removed; "
                "the current layer must use the canonical per-layer protocol"
            )
        return self._on_decode_ep_wave_ready(
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            layer_id=layer_id,
        )

    def on_decode_sync_collective(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch_global_id: int,
        sync_stage: str,
        layer_id: int,
        metrics_store,
        *,
        direct_batch: Optional[Batch] = None,
    ):
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

        if direct_batch is not None:
            if sync_stage != "post_moe":
                raise ValueError(
                    "Direct dense DECODE completion is valid only for post_moe transition"
                )
            sync_wait_room = {
                # Dense completion is a full-stage handoff; EP0 is not a
                # valid scheduler identity for the next layer.
                "batches": {None: direct_batch},
                "arrival_times": {None: time},
            }
            dp_batches = sync_wait_room["batches"]
        else:
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
        canonical_ep_wave = hasattr(sample_batch, "_decode_ep_wave_lane_times_ms")
        if direct_batch is None and not canonical_ep_wave:
            raise RuntimeError(
                "Legacy DECODE aggregate synchronization is removed; "
                "collective completion requires a canonical EP_WAVE or dense full-stage handoff"
            )
        stage_identity = None
        stage_scheduler = self.get_replica_stage_scheduler(
            replica_id, stage_identity, stage_id
        )
        execution_time_predictor = stage_scheduler._execution_time_predictor

        if sync_stage == "pre_moe":
            raise ValueError(
                "DECODE collective completion cannot start at pre_moe; "
                "the canonical EP_WAVE enters this method at post_moe"
            )

        total_layers = self._config.replica_config.model_config.num_layers
        active_unique_requests = []
        active_request_ids = set()
        for batch in dp_batches.values():
            if batch.is_idle:
                continue
            for request in batch.requests:
                if request.completed or request.id in active_request_ids:
                    continue
                active_request_ids.add(request.id)
                active_unique_requests.append(request)

        for request in active_unique_requests:
            if request.completed_layer_count >= total_layers:
                raise ValueError(
                    "Decode post_moe layer counter cannot advance: "
                    f"request_id={request.id}, "
                    f"completed_layer_count={request.completed_layer_count}, "
                    f"total_layers={total_layers}, "
                    "current_decode_token_index="
                    f"{request.current_decode_token_index}, "
                    "spec_last_committed_tokens="
                    f"{getattr(request, '_spec_last_committed_tokens', None)}; "
                    "possible missing prior decode-step reset"
                )

        for request in active_unique_requests:
            request.mb_on_step_layer_count_increment(num_layers_completed=1)

        if direct_batch is not None:
            post_moe_comm_time = 0.0
        else:
            post_moe_comm_time = float(
                getattr(sample_batch, "_decode_ep_wave_post_moe_comm_time_s", 0.0)
            )
            if not math.isfinite(post_moe_comm_time) or post_moe_comm_time < 0:
                raise ValueError(
                    "Canonical DECODE EP wave post_moe communication time must be "
                    "finite and non-negative"
                )

        num_layers = execution_time_predictor._num_layers_per_pipeline_stage
        next_layer_id = layer_id + 1

        if next_layer_id < num_layers:
            self.transition_stage_admission_for_layer(
                sample_batch,
                stage_id=stage_id,
                layer_id=next_layer_id,
                scope=FULL_STAGE_WORLD,
            )
            next_layer_execution_time = execution_time_predictor.predict_stage_execution_time(
                sample_batch,
                stage_id,
                self._cluster_type,
                num_layers=1,
                layer_id=next_layer_id,
                include_moe=False,
            )
            attention_time = next_layer_execution_time.get_single_layer_attention_time() * 1e-3

            for participant_id, batch in dp_batches.items():
                if batch.is_idle:
                    logger.info(
                        f"[DECODE_SYNC][IDLE_SKIP] Skip next-layer pre_moe scheduling for idle batch {batch.id} "
                        f"(replica={replica_id}, lane={participant_id}, layer={layer_id})"
                    )
                    continue

                total_time_to_next_sync = post_moe_comm_time + attention_time
                transition_identity = (
                    None if stage_identity is None else participant_id
                )
                events.append(DecodeSyncEvent(
                    time + total_time_to_next_sync,
                    replica_id,
                    stage_id,
                    batch,
                    transition_identity,
                    "pre_moe",
                    next_layer_id,
                    total_time_to_next_sync,
                    cluster_type=self._cluster_type,
                ))

            logger.info(
                f"[DECODE_SYNC][COLLECTIVE] post_moe completed, incremented layer count for {len(active_unique_requests)} unique requests, "
                f"scheduled next layer pre_moe sync at t={time + post_moe_comm_time + attention_time:.6f}s"
            )
            return events

        full_stage_execution_time = execution_time_predictor.predict_stage_execution_time(
            sample_batch,
            stage_id,
            self._cluster_type,
            num_layers=num_layers,
            include_moe=False,
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

            transition_identity = None if stage_identity is None else participant_id
            dp_stage_scheduler = self.get_replica_stage_scheduler(
                replica_id, transition_identity, stage_id
            )
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
            corrected_execution_time._trace_execution_time_override = trace_execution_time

            metrics_store.on_replica_stage_schedule(
                original_start_time,
                replica_id,
                stage_id,
                batch_stage,
                corrected_execution_time,
                self._cluster_type,
                transition_identity,
            )

            events.append(BatchStageEndEvent(
                time + total_final_time,
                replica_id,
                stage_id,
                is_last_stage,
                batch,
                batch_stage,
                self._cluster_type,
                transition_identity,
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
        """Handle KV cache arrival at a decode-side cluster."""
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._cluster_type.name)

        if self._cluster_type == ClusterType.DECODE_ATTN:
            return self._handle_decode_attn_arrival(time, batch, transfer_info, logger)
        if self._cluster_type == ClusterType.DECODE:
            return self._handle_decode_arrival(time, batch, transfer_info, logger)
        raise ValueError(
            f"Unexpected cluster type for KV cache arrival: {self._cluster_type}"
        )

    def _handle_decode_attn_arrival(
        self,
        time: float,
        batch: Batch,
        transfer_info,
        logger,
    ) -> List:
        """Handle KV cache arrival at a decode-attention cluster."""
        request_ids = [req.id for req in batch.requests]
        logger.info(
            "Decode-attn cluster received KV cache at %.3fs: requests %s, "
            "batch_id=%s, transfer_size=%s bytes, source_cluster=%s",
            time,
            request_ids,
            batch.id,
            transfer_info.kv_cache_size_bytes,
            transfer_info.source_cluster_type.name,
        )

        queue_was_empty = len(self._request_queue) == 0
        for request in batch.requests:
            request.on_disaggregated_decode_handoff(
                time,
                self._cluster_type,
            )
            request.on_arrival(time, self._cluster_type)
            self.add_request(request)
            logger.info(
                "Request %s added to decode-attn cluster queue, prefill_tokens=%s, "
                "decode_tokens=%s, num_processed_tokens=%s, total_tokens=%s, "
                "is_prefill_complete=%s, current_decode_token_index=%s, "
                "completed_layer_count=%s.",
                request.id,
                request.num_prefill_tokens,
                request.num_decode_tokens,
                request.num_processed_tokens,
                request.total_tokens,
                request.is_prefill_complete,
                request.current_decode_token_index,
                request.completed_layer_count,
            )

        if self._is_periodic_scheduling_enabled:
            logger.info(
                "Requests cached for periodic scheduling (interval=%sms), current queue size: %s",
                self._periodic_scheduling_interval_ms,
                len(self._request_queue),
            )
            return []

        from frontier.config.global_vars import get_simulation_mode
        from frontier.events.cluster_schedule_event import ClusterScheduleEvent

        simulation_mode = get_simulation_mode()
        if not queue_was_empty:
            logger.info(
                "Decode-attn queue already has pending requests; skip redundant schedule trigger in %s mode",
                simulation_mode,
            )
            return []

        logger.info(
            "KV-cache arrival triggers immediate decode-attn scheduling in %s mode; queue size=%d",
            simulation_mode,
            len(self._request_queue),
        )
        return [ClusterScheduleEvent(time, self._cluster_type)]

    def _handle_decode_arrival(
        self,
        time: float,
        batch: Batch,
        transfer_info,
        logger,
    ) -> List:
        """Handle KV cache arrival at a unified decode cluster."""
        request_ids = [req.id for req in batch.requests]
        logger.info(
            "Decode cluster received KV cache at %.3fs: requests %s, "
            "batch_id=%s, transfer_size=%s bytes, source_cluster=%s",
            time,
            request_ids,
            batch.id,
            transfer_info.kv_cache_size_bytes,
            transfer_info.source_cluster_type.name,
        )

        for request in batch.requests:
            request.on_arrival(time, self._cluster_type)
            self.add_request(request)
            logger.info(
                "Request %s added to decode cluster queue, prefill_tokens=%s, "
                "decode_tokens=%s, num_processed_tokens=%s, total_tokens=%s, "
                "is_prefill_complete=%s, current_decode_token_index=%s, "
                "completed_layer_count=%s.",
                request.id,
                request.num_prefill_tokens,
                request.num_decode_tokens,
                request.num_processed_tokens,
                request.total_tokens,
                request.is_prefill_complete,
                request.current_decode_token_index,
                request.completed_layer_count,
            )

        if self._is_periodic_scheduling_enabled:
            logger.info(
                "Requests cached for periodic scheduling (interval=%sms), current queue size: %s",
                self._periodic_scheduling_interval_ms,
                len(self._request_queue),
            )
            return []

        from frontier.config.global_vars import get_simulation_mode
        from frontier.events.cluster_schedule_event import ClusterScheduleEvent

        simulation_mode = get_simulation_mode()
        if simulation_mode == "offline":
            expected_num_requests = getattr(
                self._request_generator_config, "num_decode_bound_requests", None
            )
            if expected_num_requests is None:
                raise ValueError(
                    "Offline DECODE scheduling requires "
                    "request_generator_config.num_decode_bound_requests to be set "
                    "by request generation."
                )

            current_num_requests = len(self._request_queue)
            if current_num_requests > expected_num_requests:
                raise ValueError(
                    "Offline DECODE received more decode-bound requests than "
                    f"expected: current={current_num_requests}, "
                    f"expected={expected_num_requests}"
                )
            if current_num_requests < expected_num_requests:
                logger.info(
                    "Offline mode: buffering decode-bound requests (%s/%s), "
                    "deferring scheduling until all decode-bound requests arrive",
                    current_num_requests,
                    expected_num_requests,
                )
                return []
            logger.info(
                "Offline mode: all %s decode-bound requests arrived, "
                "triggering batch scheduling",
                expected_num_requests,
            )
            return [ClusterScheduleEvent(time, self._cluster_type)]

        logger.info(
            "Online mode: triggering immediate cluster scheduling for %s requests",
            len(batch.requests),
        )
        return [ClusterScheduleEvent(time, self._cluster_type)]


    def on_m2n_arrival(
        self,
        time: float,
        batch: Batch,
        transfer_info,
    ) -> List:
        """Route M2N transfer arrival to the appropriate cluster handler."""
        from frontier.logger import get_cluster_logger

        if self._cluster_type is ClusterType.DECODE_ATTN:
            self._validate_decode_attn_m2n_receipt(
                batch,
                transfer_info,
                expected_roundtrip_inflight=False,
            )
        else:
            self.preflight_m2n_arrival(batch, transfer_info)
        logger = get_cluster_logger(__name__, self._cluster_type.name)

        request_ids = [req.id for req in batch.requests]
        pipeline_stage = "attn→ffn" if transfer_info.is_attn_to_ffn else "ffn→attn"
        logger.info(f"{self._cluster_type.name} cluster received M2N data at {time:.3f}s: "
                   f"requests {request_ids} from {pipeline_stage} transfer, "
                   f"batch_id={batch.id}, transfer_size={transfer_info.activation_size_bytes} bytes, "
                   f"source_cluster={transfer_info.source_cluster_type.name}")

        if self._cluster_type == ClusterType.DECODE_FFN:
            return self._handle_m2n_arrival_decode_ffn(time, batch, transfer_info, logger)
        if self._cluster_type == ClusterType.DECODE_ATTN:
            return self._handle_m2n_arrival_decode_attn(time, batch, transfer_info, logger)
        raise RuntimeError(
            f"Validated M2N arrival has no handler for cluster {self._cluster_type.name}"
        )

    def validate_m2n_arrival_target(self, transfer_info: "M2NTransferInfo") -> None:
        """Validate that this scheduler is the declared M2N transfer target."""

        if type(self._cluster_type) is not ClusterType:
            raise ValueError(
                "M2N scheduler cluster_type must be an exact ClusterType, "
                f"got {self._cluster_type!r}"
            )
        transfer_info.validate_direction()
        if self._cluster_type not in {
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
        }:
            raise ValueError(
                f"M2N arrival is unsupported for cluster {self._cluster_type.name}"
            )
        if transfer_info.target_cluster_type != self._cluster_type:
            raise ValueError(
                "M2N target scheduler mismatch: "
                f"declared_target={transfer_info.target_cluster_type.name}, "
                f"scheduler_cluster={self._cluster_type.name}"
            )

    def preflight_m2n_arrival(
        self,
        batch: Batch,
        transfer_info: "M2NTransferInfo",
    ) -> None:
        """Validate one M2N arrival without mutating scheduler or request state."""

        if self._cluster_type == ClusterType.DECODE_FFN:
            self._validate_decode_ffn_m2n_receipt(batch, transfer_info)
            return
        if self._cluster_type == ClusterType.DECODE_ATTN:
            self._validate_decode_attn_m2n_receipt(
                batch,
                transfer_info,
                expected_roundtrip_inflight=True,
            )
            return
        self.validate_m2n_arrival_target(transfer_info)

    @staticmethod
    def _normalize_m2n_lane_contract(
        raw_lanes,
        *,
        field_name: str,
        require_nonempty: bool,
    ) -> List[tuple[int, int | None]]:
        """Validate and normalize one exact M2N lane contract."""

        if type(raw_lanes) not in {list, tuple}:
            raise ValueError(
                f"{field_name} must be an exact list or tuple, got {raw_lanes!r}"
            )

        normalized_lanes: List[tuple[int, int | None]] = []
        seen_lanes = set()
        for raw_lane in raw_lanes:
            if type(raw_lane) is not tuple or len(raw_lane) != 2:
                raise ValueError(
                    f"{field_name} must contain exact 2-tuples, got {raw_lane!r}"
                )
            lane_replica_id, lane_replica_local_id = raw_lane
            if type(lane_replica_id) is not int or lane_replica_id < 0:
                raise ValueError(
                    f"{field_name} replica_id must be an exact non-negative int, "
                    f"got {lane_replica_id!r}"
                )
            # AFD transport lane contracts identify serving Replicas.  Their
            # second coordinate is intentionally absent on both sides of the
            # A→F/F→A path; only the DECODE_FFN local EP queues use integer
            # lane IDs, and they do not pass through this helper.
            allow_full_stage_identity = field_name.startswith(
                ("DECODE_ATTN", "DECODE_FFN")
            )
            if lane_replica_local_id is not None and (
                type(lane_replica_local_id) is not int or lane_replica_local_id < 0
            ):
                raise ValueError(
                    f"{field_name} replica_local_id must be an exact "
                    f"non-negative int, got {lane_replica_local_id!r}"
                )
            if lane_replica_local_id is None and not allow_full_stage_identity:
                raise ValueError(
                    f"{field_name} replica_local_id cannot be None outside a full-stage "
                    "DECODE_ATTN identity"
                )
            lane = (lane_replica_id, lane_replica_local_id)
            if lane in seen_lanes:
                raise ValueError(f"{field_name} contains duplicate lane {lane!r}")
            seen_lanes.add(lane)
            normalized_lanes.append(lane)

        if require_nonempty and not normalized_lanes:
            raise ValueError(f"{field_name} must not be empty")
        return normalized_lanes

    def _validate_decode_ffn_waiting_room(
        self,
        *,
        group_key: tuple[int, int] | tuple[int, int, int],
        room: dict,
        expected_lane_contract: Optional[tuple[tuple[int, int], ...]] = None,
        incoming_batch: Optional[Batch] = None,
    ) -> tuple[tuple[int, int], ...]:
        """Validate one DECODE_FFN waiting room without mutating runtime state."""

        from frontier.entities.m2n_transfer_info import M2NTransferInfo

        if type(group_key) is not tuple or len(group_key) not in (2, 3):
            raise RuntimeError(
                "DECODE_FFN waiting-room key must be an exact "
                f"(layer, stage[, round]) tuple, got {group_key!r}"
            )
        for field_name, value in zip(
            ("layer_id", "afd_stage_idx", "barrier_round_id"),
            group_key,
        ):
            if type(value) is not int or value < 0:
                raise RuntimeError(
                    f"DECODE_FFN waiting-room {field_name} must be an exact "
                    f"non-negative int, got {value!r}"
                )
        layer_id, afd_stage_idx = group_key[:2]
        barrier_round_id = group_key[2] if len(group_key) == 3 else None

        if type(room) is not dict:
            raise RuntimeError(
                "DECODE_FFN waiting room must be an exact dict, "
                f"got {type(room).__name__}"
            )
        expected_room_fields = {
            "per_lane_queues",
            "lanes_rr_order",
            "rr_cursor",
            "expected_lane_contract",
        }
        if set(room) != expected_room_fields:
            missing_room_fields = expected_room_fields - set(room)
            if missing_room_fields:
                missing_field_labels = ", ".join(
                    field_name.replace("_", " ")
                    for field_name in sorted(missing_room_fields)
                )
                raise RuntimeError(
                    "DECODE_FFN waiting room missing required fields: "
                    f"{missing_field_labels}"
                )
            raise RuntimeError(
                "DECODE_FFN waiting-room schema mismatch: "
                f"expected={sorted(expected_room_fields)}, actual={sorted(room)}"
            )

        per_lane_queues = room["per_lane_queues"]
        if (
            type(per_lane_queues) is not defaultdict
            or per_lane_queues.default_factory is not deque
        ):
            raise RuntimeError(
                "DECODE_FFN waiting-room per_lane_queues must be an exact "
                "defaultdict(deque)"
            )
        lanes_rr_order = room["lanes_rr_order"]
        if type(lanes_rr_order) is not deque:
            raise RuntimeError(
                "DECODE_FFN waiting-room lanes_rr_order must be an exact deque"
            )
        rr_cursor = room["rr_cursor"]
        if type(rr_cursor) is not int or rr_cursor < 0:
            raise RuntimeError(
                "DECODE_FFN waiting-room rr_cursor must be an exact "
                f"non-negative int, got {rr_cursor!r}"
            )

        raw_room_contract = room["expected_lane_contract"]
        if type(raw_room_contract) is not tuple:
            raise RuntimeError(
                "DECODE_FFN waiting-room expected lane contract must be an "
                f"exact tuple, got {raw_room_contract!r}"
            )
        room_contract = tuple(
            self._normalize_m2n_lane_contract(
                raw_room_contract,
                field_name="DECODE_FFN waiting-room expected lane contract",
                require_nonempty=True,
            )
        )
        canonical_room_contract = tuple(sorted(room_contract))
        if room_contract != canonical_room_contract:
            raise RuntimeError(
                "DECODE_FFN waiting-room expected lane contract must be "
                f"canonical, got {room_contract!r}"
            )
        if expected_lane_contract is not None:
            normalized_expected_contract = tuple(
                sorted(
                    self._normalize_m2n_lane_contract(
                        expected_lane_contract,
                        field_name="DECODE_FFN receipt expected lane contract",
                        require_nonempty=True,
                    )
                )
            )
            if canonical_room_contract != normalized_expected_contract:
                raise ValueError(
                    "Inconsistent DECODE_FFN expected lane contract for waiting "
                    f"room: group_key={group_key}, "
                    f"existing={canonical_room_contract}, "
                    f"received={normalized_expected_contract}"
                )

        queue_lanes = tuple(
            self._normalize_m2n_lane_contract(
                tuple(per_lane_queues),
                field_name="DECODE_FFN waiting-room queue lanes",
                require_nonempty=False,
            )
        )
        rr_lanes = tuple(
            self._normalize_m2n_lane_contract(
                tuple(lanes_rr_order),
                field_name="DECODE_FFN waiting-room round-robin lanes",
                require_nonempty=False,
            )
        )
        contract_lane_set = set(canonical_room_contract)
        if not set(queue_lanes).issubset(contract_lane_set):
            raise RuntimeError(
                "DECODE_FFN waiting-room queue lane is outside the expected "
                f"contract: queues={queue_lanes}, contract={canonical_room_contract}"
            )
        if not set(rr_lanes).issubset(contract_lane_set):
            raise RuntimeError(
                "DECODE_FFN waiting-room round-robin lane is outside the expected "
                f"contract: order={rr_lanes}, contract={canonical_room_contract}"
            )

        nonempty_queue_lanes = set()
        seen_batch_identities = set()
        for queue_lane, lane_queue in per_lane_queues.items():
            if type(lane_queue) is not deque:
                raise RuntimeError(
                    "DECODE_FFN waiting-room lane queue must be an exact deque: "
                    f"lane={queue_lane}, got={type(lane_queue).__name__}"
                )
            if lane_queue:
                nonempty_queue_lanes.add(queue_lane)
            for queue_index, queued_entry in enumerate(lane_queue):
                if type(queued_entry) is not tuple or len(queued_entry) != 2:
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued entry must be an exact "
                        f"(batch, transfer_info) tuple: lane={queue_lane}, "
                        f"index={queue_index}, value={queued_entry!r}"
                    )
                queued_batch, queued_transfer_info = queued_entry
                if type(queued_transfer_info) is not M2NTransferInfo:
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued transfer must be an exact "
                        f"M2NTransferInfo: lane={queue_lane}, index={queue_index}"
                    )
                if queued_transfer_info.batch is not queued_batch:
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued batch identity does not "
                        "match transfer_info.batch"
                    )
                queued_is_idle = getattr(queued_batch, "is_idle", None)
                if type(queued_is_idle) is not bool:
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued batch is_idle must be an "
                        f"exact bool, got {queued_is_idle!r}"
                    )
                if incoming_batch is not None and queued_batch is incoming_batch:
                    raise ValueError(
                        "DECODE_FFN waiting room already contains the incoming "
                        "batch object"
                    )
                queued_batch_identity = id(queued_batch)
                if queued_batch_identity in seen_batch_identities:
                    raise RuntimeError(
                        "DECODE_FFN waiting room contains a duplicate queued "
                        "batch object"
                    )
                seen_batch_identities.add(queued_batch_identity)

                if (
                    queued_transfer_info.source_cluster_type
                    is not ClusterType.DECODE_ATTN
                    or queued_transfer_info.target_cluster_type
                    is not ClusterType.DECODE_FFN
                ):
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued transfer must be an exact "
                        "DECODE_ATTN -> DECODE_FFN transfer"
                    )
                queued_source_replica_id = queued_transfer_info.source_replica_id
                queued_source_replica_local_id = queued_transfer_info.source_replica_local_id
                if (
                    type(queued_source_replica_id) is not int
                    or queued_source_replica_id < 0
                    or (
                        queued_source_replica_local_id is not None
                        and (
                            type(queued_source_replica_local_id) is not int
                            or queued_source_replica_local_id < 0
                        )
                    )
                    or (queued_source_replica_id, queued_source_replica_local_id) != queue_lane
                ):
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued transfer lane mismatch: "
                        f"queue={queue_lane}, transfer="
                        f"{(queued_source_replica_id, queued_source_replica_local_id)}"
                    )
                queued_layer_id = queued_transfer_info.layer_id
                if type(queued_layer_id) is not int or queued_layer_id != layer_id:
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued transfer layer mismatch: "
                        f"room={layer_id!r}, transfer={queued_layer_id!r}"
                    )
                queued_stage_idx = queued_transfer_info.afd_stage_idx
                if (
                    type(queued_stage_idx) is not int
                    or queued_stage_idx != afd_stage_idx
                ):
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued transfer stage mismatch: "
                        f"room={afd_stage_idx!r}, transfer={queued_stage_idx!r}"
                    )

                queued_round_id = getattr(
                    queued_batch,
                    "decode_attn_barrier_round_id",
                    None,
                )
                if barrier_round_id is None:
                    if queued_round_id is not None:
                        raise RuntimeError(
                            "DECODE_FFN waiting-room queued batch round mismatch: "
                            f"room=None, batch={queued_round_id!r}"
                        )
                elif (
                    type(queued_round_id) is not int
                    or queued_round_id != barrier_round_id
                ):
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued batch round mismatch: "
                        f"room={barrier_round_id!r}, batch={queued_round_id!r}"
                    )

                queued_expected_lanes = getattr(
                    queued_batch,
                    "decode_attn_barrier_expected_lanes",
                    (),
                )
                if queued_expected_lanes is None:
                    queued_expected_lanes = ()
                normalized_queued_expected_lanes = tuple(
                    sorted(
                        self._normalize_m2n_lane_contract(
                            queued_expected_lanes,
                            field_name=(
                                "DECODE_FFN waiting-room queued batch expected "
                                "lane metadata"
                            ),
                            require_nonempty=False,
                        )
                    )
                )
                if (
                    normalized_queued_expected_lanes
                    and normalized_queued_expected_lanes
                    != canonical_room_contract
                ):
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued batch lane contract "
                        f"mismatch: room={canonical_room_contract}, "
                        f"batch={normalized_queued_expected_lanes}"
                    )

                queued_batch_stage_idx = getattr(
                    queued_batch,
                    "afd_stage_idx",
                    None,
                )
                if queued_batch_stage_idx is not None and (
                    type(queued_batch_stage_idx) is not int
                    or queued_batch_stage_idx != afd_stage_idx
                ):
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued batch stage mismatch: "
                        f"room={afd_stage_idx!r}, batch={queued_batch_stage_idx!r}"
                    )
                queued_batch_layer_id = getattr(
                    queued_batch,
                    "decode_ffn_layer_id",
                    None,
                )
                if queued_batch_layer_id is not None and (
                    type(queued_batch_layer_id) is not int
                    or queued_batch_layer_id != layer_id
                ):
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued batch layer mismatch: "
                        f"room={layer_id!r}, batch={queued_batch_layer_id!r}"
                    )
                queued_original_replica_id = getattr(
                    queued_batch,
                    "decode_attn_original_replica_id",
                    None,
                )
                queued_original_replica_local_id = getattr(
                    queued_batch,
                    "decode_attn_original_replica_local_id",
                    None,
                )
                if queued_original_replica_id is not None and (
                    type(queued_original_replica_id) is not int
                    or queued_original_replica_id != queue_lane[0]
                ):
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued batch replica lane mismatch: "
                        f"queue={queue_lane[0]!r}, "
                        f"batch={queued_original_replica_id!r}"
                    )
                if queued_original_replica_local_id is not None and (
                    type(queued_original_replica_local_id) is not int
                    or queued_original_replica_local_id != queue_lane[1]
                ):
                    raise RuntimeError(
                        "DECODE_FFN waiting-room queued batch local identity mismatch: "
                        f"queue={queue_lane[1]!r}, batch={queued_original_replica_local_id!r}"
                    )

        if set(rr_lanes) != nonempty_queue_lanes:
            raise RuntimeError(
                "DECODE_FFN waiting-room round-robin lanes must exactly match "
                "non-empty queue lanes: "
                f"order={rr_lanes}, nonempty={sorted(nonempty_queue_lanes)}"
            )
        return canonical_room_contract

    def _validate_decode_ffn_m2n_receipt(
        self,
        batch: Batch,
        transfer_info: "M2NTransferInfo",
    ) -> tuple[
        int,
        int,
        Optional[int],
        tuple[int, int],
        List[tuple[int, int]],
        int,
        tuple[int, int] | tuple[int, int, int],
        tuple[tuple[int, int], ...],
    ]:
        """Validate one A-to-F receipt without mutating scheduler or batch state."""

        self.validate_m2n_arrival_target(transfer_info)
        if self._cluster_type is not ClusterType.DECODE_FFN:
            raise ValueError(
                "DECODE_FFN receipt validation requires a DECODE_FFN scheduler, "
                f"got {self._cluster_type.name}"
            )
        if batch is not transfer_info.batch:
            raise ValueError(
                "DECODE_FFN M2N batch identity mismatch: batch is not "
                "transfer_info.batch"
            )

        layer_id = getattr(transfer_info, "layer_id", None)
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError(
                "DECODE_FFN receipt layer_id must be an exact non-negative int, "
                f"got {layer_id!r}"
            )

        afd_stage_idx = getattr(transfer_info, "afd_stage_idx", None)
        if type(afd_stage_idx) is not int or afd_stage_idx < 0:
            raise ValueError(
                "DECODE_FFN receipt afd_stage_idx must be an exact non-negative int, "
                f"got {afd_stage_idx!r}"
            )

        barrier_round_id = getattr(batch, "decode_attn_barrier_round_id", None)
        if barrier_round_id is not None and (
            type(barrier_round_id) is not int or barrier_round_id < 0
        ):
            raise ValueError(
                "DECODE_FFN receipt barrier_round_id must be None or an exact "
                f"non-negative int, got {barrier_round_id!r}"
            )

        source_replica_id = getattr(transfer_info, "source_replica_id", None)
        if type(source_replica_id) is not int or source_replica_id < 0:
            raise ValueError(
                "DECODE_FFN receipt source_replica_id must be an exact "
                f"non-negative int, got {source_replica_id!r}"
            )
        source_replica_local_id = getattr(transfer_info, "source_replica_local_id", None)
        if source_replica_local_id is not None and (
            type(source_replica_local_id) is not int or source_replica_local_id < 0
        ):
            raise ValueError(
                "DECODE_FFN receipt source_replica_local_id must be None or an exact "
                "non-negative int, "
                f"got {source_replica_local_id!r}"
            )
        lane = (source_replica_id, source_replica_local_id)

        raw_expected_lanes = getattr(
            batch,
            "decode_attn_barrier_expected_lanes",
            (),
        )
        if raw_expected_lanes is None:
            raw_expected_lanes = ()
        barrier_expected_lanes = self._normalize_m2n_lane_contract(
            raw_expected_lanes,
            field_name="DECODE_FFN receipt expected lane metadata",
            require_nonempty=False,
        )

        if barrier_expected_lanes:
            if lane not in barrier_expected_lanes:
                raise ValueError(
                    "Unexpected lane observed in DECODE_FFN round-scoped waiting "
                    f"room: lane={lane}, expected_lanes={barrier_expected_lanes}"
                )
            expected_lanes = len(barrier_expected_lanes)
            expected_lane_contract = tuple(sorted(barrier_expected_lanes))
        else:
            scheduler_expected_lanes = self._normalize_m2n_lane_contract(
                getattr(self, "_ffn_expected_lanes", None),
                field_name="DECODE_FFN scheduler lane topology",
                require_nonempty=True,
            )
            if lane not in scheduler_expected_lanes:
                raise ValueError(
                    "Unexpected lane observed in DECODE_FFN scheduler lane topology: "
                    f"lane={lane}, expected_lanes={scheduler_expected_lanes}"
                )
            expected_lanes = getattr(self, "_ffn_group_micro_batches", None)
            if type(expected_lanes) is not int or expected_lanes <= 0:
                raise ValueError(
                    "DECODE_FFN _ffn_group_micro_batches must be an exact positive "
                    f"int when expected lane metadata is empty, got {expected_lanes!r}"
                )
            expected_lane_contract = tuple(sorted(scheduler_expected_lanes))

        if barrier_round_id is None:
            group_key = (layer_id, afd_stage_idx)
        else:
            group_key = (layer_id, afd_stage_idx, barrier_round_id)

        if not hasattr(self, "_m2n_waiting_by_layer"):
            raise RuntimeError(
                "DECODE_FFN scheduler missing _m2n_waiting_by_layer during receipt "
                "preflight"
            )
        if type(self._m2n_waiting_by_layer) is not dict:
            raise RuntimeError(
                "DECODE_FFN _m2n_waiting_by_layer must be an exact dict"
            )
        room = self._m2n_waiting_by_layer.get(group_key)
        if room is not None:
            self._validate_decode_ffn_waiting_room(
                group_key=group_key,
                room=room,
                expected_lane_contract=expected_lane_contract,
                incoming_batch=batch,
            )

        return (
            layer_id,
            afd_stage_idx,
            barrier_round_id,
            lane,
            barrier_expected_lanes,
            expected_lanes,
            group_key,
            expected_lane_contract,
        )

    def _validate_decode_attn_f2a_cohort_binding(
        self,
        batch: Batch,
        *,
        lane: tuple[int, int],
        afd_stage_idx: int,
        requests: List[Request],
        active_requests: List[Request],
        context: str,
    ) -> None:
        """Validate one batch against its lane-local active cohort."""

        def require_non_negative_int(value, field_name: str) -> int:
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{field_name} must be an exact non-negative int, got {value!r}"
                )
            return value

        cohort_id = require_non_negative_int(
            getattr(batch, "decode_attn_cohort_id", None),
            f"DECODE_ATTN {context} decode_attn_cohort_id",
        )
        replica_schedulers = getattr(self, "_replica_schedulers", None)
        if type(replica_schedulers) is not dict:
            raise RuntimeError(
                "DECODE_ATTN replica scheduler topology must be an exact dict"
            )
        if lane not in replica_schedulers:
            raise ValueError(
                f"DECODE_ATTN {context} lane is absent from the replica scheduler "
                f"topology: lane={lane}"
            )
        replica_scheduler = replica_schedulers[lane]
        cohort_states = getattr(
            replica_scheduler,
            "_decode_attn_active_cohort_states",
            None,
        )
        if type(cohort_states) is not dict:
            raise RuntimeError(
                "DECODE_ATTN active cohort states must be an exact dict"
            )
        if cohort_id not in cohort_states:
            raise ValueError(
                f"DECODE_ATTN {context} references an inactive or unknown cohort: "
                f"cohort_id={cohort_id}, lane={lane}"
            )
        cohort_state = cohort_states[cohort_id]
        if type(cohort_state) is not dict:
            raise RuntimeError(
                "DECODE_ATTN active cohort state must be an exact dict: "
                f"cohort_id={cohort_id}, lane={lane}"
            )

        def require_cohort_id_set(
            field_name: str,
            *,
            require_nonempty: bool,
        ) -> set[int]:
            request_ids = cohort_state.get(field_name)
            if type(request_ids) is not set:
                raise RuntimeError(
                    f"DECODE_ATTN cohort {field_name} must be an exact set, "
                    f"got {request_ids!r}"
                )
            if require_nonempty and not request_ids:
                raise RuntimeError(
                    f"DECODE_ATTN cohort {field_name} must not be empty"
                )
            for request_id in request_ids:
                if type(request_id) is not int or request_id < 0:
                    raise RuntimeError(
                        f"DECODE_ATTN cohort {field_name} must contain exact "
                        f"non-negative ints, got {request_id!r}"
                    )
            return request_ids

        all_request_ids = require_cohort_id_set(
            "all_request_ids",
            require_nonempty=True,
        )
        pending_request_ids = require_cohort_id_set(
            "pending_request_ids",
            require_nonempty=False,
        )
        if not pending_request_ids.issubset(all_request_ids):
            raise RuntimeError(
                "DECODE_ATTN cohort pending_request_ids must be a subset of "
                "all_request_ids"
            )

        batch_cohort_request_ids = getattr(
            batch,
            "decode_attn_cohort_request_ids",
            None,
        )
        if type(batch_cohort_request_ids) is not tuple:
            raise ValueError(
                f"DECODE_ATTN {context} decode_attn_cohort_request_ids must be an "
                f"exact tuple, got {batch_cohort_request_ids!r}"
            )
        normalized_batch_cohort_request_ids = [
            require_non_negative_int(
                request_id,
                f"DECODE_ATTN {context} cohort request ID",
            )
            for request_id in batch_cohort_request_ids
        ]
        if len(set(normalized_batch_cohort_request_ids)) != len(
            normalized_batch_cohort_request_ids
        ):
            raise ValueError(
                f"DECODE_ATTN {context} cohort request IDs must not contain "
                "duplicates"
            )
        if set(normalized_batch_cohort_request_ids) != all_request_ids:
            raise ValueError(
                f"DECODE_ATTN {context} cohort request IDs do not match active "
                "cohort all_request_ids: "
                f"batch={normalized_batch_cohort_request_ids}, "
                f"active={sorted(all_request_ids)}"
            )

        batch_request_ids = [
            require_non_negative_int(
                getattr(request, "id", None),
                f"DECODE_ATTN {context} request ID",
            )
            for request in requests
        ]
        if len(set(batch_request_ids)) != len(batch_request_ids):
            raise ValueError(
                f"DECODE_ATTN {context} request IDs must not contain duplicates"
            )
        requests_outside_cohort = sorted(
            set(batch_request_ids) - all_request_ids
        )
        if requests_outside_cohort:
            raise ValueError(
                f"DECODE_ATTN {context} contains requests outside the active "
                f"cohort: request_ids={requests_outside_cohort}, "
                f"cohort_id={cohort_id}"
            )

        active_request_ids = {
            require_non_negative_int(
                getattr(request, "id", None),
                f"DECODE_ATTN {context} active request ID",
            )
            for request in active_requests
        }
        requests_outside_pending = sorted(
            active_request_ids - pending_request_ids
        )
        if requests_outside_pending:
            raise ValueError(
                f"DECODE_ATTN {context} requests are not pending in the active "
                f"cohort: request_ids={requests_outside_pending}, "
                f"cohort_id={cohort_id}"
            )

        active_stage_indices = cohort_state.get("active_stage_indices")
        if type(active_stage_indices) is not set:
            raise RuntimeError(
                "DECODE_ATTN cohort active_stage_indices must be an exact set, "
                f"got {active_stage_indices!r}"
            )
        for active_stage_idx in active_stage_indices:
            if type(active_stage_idx) is not int or active_stage_idx < 0:
                raise RuntimeError(
                    "DECODE_ATTN cohort active_stage_indices must contain exact "
                    f"non-negative ints, got {active_stage_idx!r}"
                )
        if afd_stage_idx not in active_stage_indices:
            raise ValueError(
                f"DECODE_ATTN {context} afd_stage_idx is not active in the "
                f"cohort: afd_stage_idx={afd_stage_idx}, "
                f"active_stage_indices={sorted(active_stage_indices)}"
            )

    def _validate_decode_attn_m2n_receipt(
        self,
        batch: Batch,
        transfer_info: "M2NTransferInfo",
        *,
        expected_roundtrip_inflight: bool,
    ) -> Dict[str, Any]:
        """Validate one F-to-A receipt without mutating runtime state."""

        def require_non_negative_int(value, field_name: str) -> int:
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{field_name} must be an exact non-negative int, got {value!r}"
                )
            return value

        if type(expected_roundtrip_inflight) is not bool:
            raise ValueError(
                "DECODE_ATTN expected roundtrip state must be an exact bool, "
                f"got {expected_roundtrip_inflight!r}"
            )

        self.validate_m2n_arrival_target(transfer_info)
        if self._cluster_type is not ClusterType.DECODE_ATTN:
            raise ValueError(
                "DECODE_ATTN receipt validation requires a DECODE_ATTN scheduler, "
                f"got {self._cluster_type.name}"
            )
        if (
            transfer_info.source_cluster_type is not ClusterType.DECODE_FFN
            or transfer_info.target_cluster_type is not ClusterType.DECODE_ATTN
        ):
            raise ValueError(
                "DECODE_ATTN receipt validation requires an exact "
                "DECODE_FFN -> DECODE_ATTN transfer"
            )
        if batch is not transfer_info.batch:
            raise ValueError(
                "DECODE_ATTN M2N batch identity mismatch: batch is not "
                "transfer_info.batch"
            )

        source_replica_id = require_non_negative_int(
            getattr(transfer_info, "source_replica_id", None),
            "DECODE_ATTN receipt source_replica_id",
        )
        source_replica_local_id = getattr(transfer_info, "source_replica_local_id", None)
        if source_replica_local_id is not None:
            source_replica_local_id = require_non_negative_int(
                source_replica_local_id,
                "DECODE_ATTN receipt source_replica_local_id",
            )
        transfer_layer_id = require_non_negative_int(
            getattr(transfer_info, "layer_id", None),
            "DECODE_ATTN receipt layer_id",
        )
        transfer_stage_idx = require_non_negative_int(
            getattr(transfer_info, "afd_stage_idx", None),
            "DECODE_ATTN receipt afd_stage_idx",
        )
        replica_id = require_non_negative_int(
            getattr(batch, "decode_attn_original_replica_id", None),
            "DECODE_ATTN receipt decode_attn_original_replica_id",
        )
        replica_local_id = getattr(batch, "decode_attn_original_replica_local_id", None)
        if replica_local_id is not None:
            replica_local_id = require_non_negative_int(
                replica_local_id,
                "DECODE_ATTN receipt decode_attn_original_replica_local_id",
            )
        batch_global_id = require_non_negative_int(
            getattr(batch, "global_id", None),
            "DECODE_ATTN receipt batch.global_id",
        )
        afd_stage_idx = require_non_negative_int(
            getattr(batch, "afd_stage_idx", None),
            "DECODE_ATTN receipt batch.afd_stage_idx",
        )
        lane = (replica_id, replica_local_id)
        source_lane = (source_replica_id, source_replica_local_id)
        if source_lane != lane:
            raise ValueError(
                "DECODE_ATTN receipt source lane does not match the original ATTN "
                f"lane: source={source_lane}, original={lane}"
            )
        if transfer_stage_idx != afd_stage_idx:
            raise ValueError(
                "DECODE_ATTN receipt afd_stage_idx does not match the batch stage: "
                f"transfer={transfer_stage_idx}, batch={afd_stage_idx}"
            )

        requests = getattr(batch, "requests", None)
        if type(requests) is not list or not requests:
            raise ValueError("DECODE_ATTN F-to-A receipt requires a non-empty request list")
        active_requests = []
        for request in requests:
            if type(request) is not Request:
                raise ValueError(
                    "DECODE_ATTN F-to-A incoming receipt contains a request "
                    "that is not an exact Request: "
                    f"value={request!r}"
                )
            completed = getattr(request, "completed", None)
            if type(completed) is not bool:
                raise ValueError(
                    "DECODE_ATTN receipt request.completed must be an exact bool, "
                    f"got {completed!r} for request {getattr(request, 'id', '?')}"
                )
            roundtrip_inflight = getattr(request, "af_roundtrip_inflight", None)
            if type(roundtrip_inflight) is not bool:
                raise ValueError(
                    "DECODE_ATTN receipt request.af_roundtrip_inflight must be an "
                    f"exact bool, got {roundtrip_inflight!r} for request "
                    f"{getattr(request, 'id', '?')}"
                )
            if roundtrip_inflight is not expected_roundtrip_inflight:
                raise ValueError(
                    "DECODE_ATTN receipt request roundtrip state does not match the "
                    f"admission phase: expected={expected_roundtrip_inflight}, "
                    f"actual={roundtrip_inflight}, request="
                    f"{getattr(request, 'id', '?')}"
                )
            if not completed:
                active_requests.append(request)
        if not active_requests:
            raise ValueError(
                "DECODE_ATTN F-to-A receipt requires at least one active request"
            )

        active_layer_ids = [
            require_non_negative_int(
                getattr(request, "completed_layer_count", None),
                "DECODE_ATTN receipt active request completed_layer_count",
            )
            for request in active_requests
        ]
        if len(set(active_layer_ids)) != 1:
            raise ValueError(
                "DECODE_ATTN receipt active requests must have a consistent layer: "
                f"layers={active_layer_ids}"
            )
        current_layer_id = active_layer_ids[0]
        if transfer_layer_id != current_layer_id:
            raise ValueError(
                "DECODE_ATTN receipt layer_id does not match the active request layer: "
                f"transfer={transfer_layer_id}, active={current_layer_id}"
            )

        total_layers = require_non_negative_int(
            getattr(self._config.replica_config.model_config, "num_layers", None),
            "DECODE_ATTN model num_layers",
        )
        if total_layers == 0:
            raise ValueError("DECODE_ATTN model num_layers must be positive")
        if current_layer_id >= total_layers:
            raise ValueError(
                "DECODE_ATTN receipt active request layer must be below num_layers: "
                f"layer={current_layer_id}, num_layers={total_layers}"
            )
        next_layer_id = current_layer_id + 1
        is_last_layer = next_layer_id == total_layers

        active_decode_token_indices = [
            require_non_negative_int(
                getattr(request, "current_decode_token_index", None),
                "DECODE_ATTN receipt active request decode_token_index",
            )
            for request in active_requests
        ]
        replay_decode_token_index = getattr(
            batch,
            "replay_decode_token_index",
            None,
        )
        if replay_decode_token_index is None:
            if len(set(active_decode_token_indices)) != 1:
                raise ValueError(
                    "DECODE_ATTN F-to-A receipt requires a batch-level replay decode "
                    "token index for mixed active request positions; got "
                    f"{active_decode_token_indices}"
                )
            decode_token_index = active_decode_token_indices[0]
        else:
            decode_token_index = require_non_negative_int(
                replay_decode_token_index,
                "DECODE_ATTN receipt replay_decode_token_index",
            )
            if decode_token_index != active_decode_token_indices[0]:
                raise ValueError(
                    "DECODE_ATTN receipt replay_decode_token_index does not match the "
                    "active batch head: "
                    f"replay={decode_token_index}, head={active_decode_token_indices[0]}"
                )

        scheduler_expected_lanes = self._normalize_m2n_lane_contract(
            self._get_decode_attn_f2a_expected_lanes(
                replica_id,
                afd_stage_idx=afd_stage_idx,
            ),
            field_name="DECODE_ATTN F-to-A scheduler lane topology",
            require_nonempty=True,
        )
        if lane not in scheduler_expected_lanes:
            raise ValueError(
                "Unexpected lane observed in DECODE_ATTN F-to-A scheduler topology: "
                f"lane={lane}, expected_lanes={scheduler_expected_lanes}"
            )
        scheduler_expected_lane_set = set(scheduler_expected_lanes)

        self._validate_decode_attn_f2a_cohort_binding(
            batch,
            lane=lane,
            afd_stage_idx=afd_stage_idx,
            requests=requests,
            active_requests=active_requests,
            context="receipt",
        )

        barrier_round_id = getattr(batch, "decode_attn_barrier_round_id", None)
        if barrier_round_id is not None:
            barrier_round_id = require_non_negative_int(
                barrier_round_id,
                "DECODE_ATTN receipt barrier_round_id",
            )

        raw_expected_lanes = getattr(
            batch,
            "decode_attn_barrier_expected_lanes",
            (),
        )
        if raw_expected_lanes is None:
            raw_expected_lanes = ()
        barrier_expected_lanes = self._normalize_m2n_lane_contract(
            raw_expected_lanes,
            field_name="DECODE_ATTN receipt expected lane metadata",
            require_nonempty=False,
        )
        if barrier_expected_lanes and lane not in barrier_expected_lanes:
            raise ValueError(
                "Unexpected lane observed in DECODE_ATTN receipt expected lane "
                f"metadata: lane={lane}, expected_lanes={barrier_expected_lanes}"
            )
        scheduler_lane_sets_by_replica = {
            replica_id: scheduler_expected_lane_set,
        }
        metadata_lanes_outside_topology = []
        for expected_lane in barrier_expected_lanes:
            expected_replica_id = expected_lane[0]
            expected_replica_lane_set = scheduler_lane_sets_by_replica.get(
                expected_replica_id
            )
            if expected_replica_lane_set is None:
                expected_replica_lanes = self._normalize_m2n_lane_contract(
                    self._get_decode_attn_f2a_expected_lanes(
                        expected_replica_id,
                        afd_stage_idx=afd_stage_idx,
                    ),
                    field_name=(
                        "DECODE_ATTN F-to-A scheduler lane topology for "
                        f"replica {expected_replica_id}"
                    ),
                    require_nonempty=True,
                )
                expected_replica_lane_set = set(expected_replica_lanes)
                scheduler_lane_sets_by_replica[expected_replica_id] = (
                    expected_replica_lane_set
                )
            if expected_lane not in expected_replica_lane_set:
                metadata_lanes_outside_topology.append(expected_lane)
        if metadata_lanes_outside_topology:
            raise ValueError(
                "DECODE_ATTN receipt expected lanes are outside the scheduler "
                f"topology: outside={metadata_lanes_outside_topology}, "
                f"topology_by_replica={scheduler_lane_sets_by_replica}"
            )
        filtered_expected_lanes = tuple(
            expected_lane
            for expected_lane in barrier_expected_lanes
            if expected_lane[0] == replica_id
        )

        if barrier_round_id is None:
            round_key = (replica_id, next_layer_id, afd_stage_idx)
        else:
            round_key = (
                replica_id,
                next_layer_id,
                afd_stage_idx,
                barrier_round_id,
            )

        waiting_rooms = getattr(self, "_f2a_waiting_by_round", None)
        if type(waiting_rooms) is not dict:
            raise RuntimeError(
                "DECODE_ATTN scheduler missing exact _f2a_waiting_by_round mapping"
            )

        room = waiting_rooms.get(round_key)
        if is_last_layer and room is not None:
            raise RuntimeError(
                "DECODE_ATTN final F-to-A receipt must not have an existing "
                f"waiting room: round_key={round_key}"
            )
        existing_expected_lanes: tuple[tuple[int, int], ...] = ()
        if room is not None:
            if type(room) is not dict:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A waiting room must be an exact dict: "
                    f"round_key={round_key}"
                )
            if "expected_lanes" not in room:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A waiting room missing expected lanes: "
                    f"round_key={round_key}"
                )
            raw_room_expected_lanes = room["expected_lanes"]
            if raw_room_expected_lanes is not None:
                existing_expected_lanes = tuple(
                    self._normalize_m2n_lane_contract(
                        raw_room_expected_lanes,
                        field_name="DECODE_ATTN F-to-A waiting room expected lanes",
                        require_nonempty=True,
                    )
                )
                if any(
                    room_replica_id != replica_id
                    for room_replica_id, _ in existing_expected_lanes
                ):
                    raise RuntimeError(
                        "DECODE_ATTN F-to-A waiting room expected lanes contain a "
                        f"different replica: round_key={round_key}, "
                        f"expected_lanes={existing_expected_lanes}"
                    )
                room_lanes_outside_topology = [
                    expected_lane
                    for expected_lane in existing_expected_lanes
                    if expected_lane not in scheduler_expected_lane_set
                ]
                if room_lanes_outside_topology:
                    raise RuntimeError(
                        "DECODE_ATTN F-to-A waiting room expected lanes are outside "
                        f"the scheduler topology: outside={room_lanes_outside_topology}, "
                        f"expected_lanes={scheduler_expected_lanes}"
                    )
            if (
                filtered_expected_lanes
                and existing_expected_lanes
                and existing_expected_lanes != filtered_expected_lanes
            ):
                raise ValueError(
                    "Mismatched DECODE_ATTN F-to-A expected lanes contract for the "
                    f"same round: round_key={round_key}, "
                    f"existing={existing_expected_lanes}, "
                    f"new={filtered_expected_lanes}"
                )

        stored_expected_lanes = (
            filtered_expected_lanes
            or existing_expected_lanes
            or None
        )
        expected_lanes = list(
            stored_expected_lanes
            if stored_expected_lanes is not None
            else tuple(scheduler_expected_lanes)
        )
        if lane not in expected_lanes:
            raise ValueError(
                "Unexpected lane observed in DECODE_ATTN F-to-A waiting room: "
                f"round_key={round_key}, lane={lane}, "
                f"expected_lanes={expected_lanes}"
            )

        if room is not None:
            per_lane_queues = room.get("per_lane_queues")
            if (
                type(per_lane_queues) is not defaultdict
                or per_lane_queues.default_factory is not deque
            ):
                raise RuntimeError(
                    "DECODE_ATTN F-to-A waiting room per_lane_queues must be a "
                    f"defaultdict(deque): round_key={round_key}"
                )
            room_lanes = self._normalize_m2n_lane_contract(
                list(per_lane_queues.keys()),
                field_name="DECODE_ATTN F-to-A waiting room queue lanes",
                require_nonempty=False,
            )
            queued_identities_by_lane = {}
            for room_lane in room_lanes:
                if room_lane not in expected_lanes:
                    raise RuntimeError(
                        "DECODE_ATTN F-to-A waiting room contains a queue outside "
                        f"its expected lanes: lane={room_lane}, "
                        f"expected_lanes={expected_lanes}"
                    )
                lane_queue = per_lane_queues.get(room_lane)
                if type(lane_queue) is not deque:
                    raise RuntimeError(
                        "DECODE_ATTN F-to-A waiting room lane queue must be a deque: "
                        f"lane={room_lane}, queue={lane_queue!r}"
                    )
                queued_identities_by_lane[room_lane] = [
                    self._validate_decode_attn_f2a_queued_batch(
                        queued_batch,
                        queue_lane=room_lane,
                        round_key=round_key,
                        expected_lanes=expected_lanes,
                        current_batch=batch,
                    )
                    for queued_batch in lane_queue
                ]

            current_identity = (batch_global_id, decode_token_index)
            if barrier_round_id is not None:
                for queued_identities in queued_identities_by_lane.values():
                    for queued_identity in queued_identities:
                        if queued_identity != current_identity:
                            raise RuntimeError(
                                "DECODE_ATTN F-to-A explicit round mixes batch/token "
                                f"identities: queued={queued_identity}, "
                                f"current={current_identity}, round_key={round_key}"
                            )
            else:
                max_queue_depth = max(
                    (
                        len(queued_identities)
                        for queued_identities in queued_identities_by_lane.values()
                    ),
                    default=0,
                )
                for queue_position in range(max_queue_depth):
                    position_identities = {
                        queued_identities[queue_position]
                        for queued_identities in queued_identities_by_lane.values()
                        if queue_position < len(queued_identities)
                    }
                    if len(position_identities) > 1:
                        raise RuntimeError(
                            "DECODE_ATTN F-to-A legacy FIFO position mixes "
                            f"batch/token identities: position={queue_position}, "
                            f"identities={sorted(position_identities)}, "
                            f"round_key={round_key}"
                        )

                current_lane_depth = len(
                    queued_identities_by_lane.get(lane, ())
                )
                matching_position_identities = {
                    queued_identities[current_lane_depth]
                    for room_lane, queued_identities in queued_identities_by_lane.items()
                    if room_lane != lane
                    and current_lane_depth < len(queued_identities)
                }
                if (
                    matching_position_identities
                    and matching_position_identities != {current_identity}
                ):
                    raise RuntimeError(
                        "DECODE_ATTN F-to-A legacy FIFO arrival identity does not "
                        f"match position={current_lane_depth}: "
                        f"queued={sorted(matching_position_identities)}, "
                        f"current={current_identity}, round_key={round_key}"
                    )

        return {
            "current_layer_id": current_layer_id,
            "next_layer_id": next_layer_id,
            "replica_id": replica_id,
            "replica_local_id": replica_local_id,
            "lane": lane,
            "batch_global_id": batch_global_id,
            "decode_token_index": decode_token_index,
            "afd_stage_idx": afd_stage_idx,
            "barrier_round_id": barrier_round_id,
            "round_key": round_key,
            "stored_expected_lanes": stored_expected_lanes,
            "expected_lanes": expected_lanes,
            "room": room,
            "is_last_layer": is_last_layer,
        }

    def _validate_decode_attn_f2a_queued_batch(
        self,
        queued_batch: Batch,
        *,
        queue_lane: tuple[int, int],
        round_key: tuple,
        expected_lanes: List[tuple[int, int]],
        current_batch: Batch,
    ) -> tuple[int, int]:
        """Validate an existing F-to-A queue entry without mutating it."""

        def require_non_negative_int(value, field_name: str) -> int:
            if type(value) is not int or value < 0:
                raise RuntimeError(
                    f"{field_name} must be an exact non-negative int, got {value!r}"
                )
            return value

        if type(queued_batch) is not Batch:
            raise RuntimeError(
                "DECODE_ATTN F-to-A waiting room contains a queued object that is "
                f"not an exact Batch: lane={queue_lane}, value={queued_batch!r}"
            )
        if queued_batch is current_batch:
            raise ValueError(
                "Duplicate DECODE_ATTN F-to-A receipt for the same batch object: "
                f"round_key={round_key}, lane={queue_lane}"
            )

        replica_id, next_layer_id, afd_stage_idx = round_key[:3]
        barrier_round_id = round_key[3] if len(round_key) == 4 else None
        queued_lane = (
            require_non_negative_int(
                getattr(queued_batch, "decode_attn_original_replica_id", None),
                "DECODE_ATTN F-to-A queued batch original replica_id",
            ),
            getattr(queued_batch, "decode_attn_original_replica_local_id", None),
        )
        if queued_lane[1] is not None:
            queued_lane = (
                queued_lane[0],
                require_non_negative_int(
                    queued_lane[1],
                    "DECODE_ATTN F-to-A queued batch original replica_local_id",
                ),
            )
        if queued_lane != queue_lane or queued_lane not in expected_lanes:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued batch lane does not match its waiting "
                f"room: queued={queued_lane}, room={queue_lane}, "
                f"expected_lanes={expected_lanes}"
            )
        if queued_lane[0] != replica_id:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued batch belongs to a different replica: "
                f"round_key={round_key}, queued_lane={queued_lane}"
            )

        queued_global_id = require_non_negative_int(
            getattr(queued_batch, "global_id", None),
            "DECODE_ATTN F-to-A queued batch global_id",
        )
        queued_stage_idx = require_non_negative_int(
            getattr(queued_batch, "afd_stage_idx", None),
            "DECODE_ATTN F-to-A queued batch afd_stage_idx",
        )
        if queued_stage_idx != afd_stage_idx:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued batch stage does not match its waiting "
                f"room: queued={queued_stage_idx}, expected={afd_stage_idx}"
            )

        queued_round_id = getattr(
            queued_batch,
            "decode_attn_barrier_round_id",
            None,
        )
        if queued_round_id is not None:
            queued_round_id = require_non_negative_int(
                queued_round_id,
                "DECODE_ATTN F-to-A queued batch barrier_round_id",
            )
        if queued_round_id != barrier_round_id:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued batch round does not match its waiting "
                f"room: queued={queued_round_id}, expected={barrier_round_id}"
            )

        raw_queued_expected_lanes = getattr(
            queued_batch,
            "decode_attn_barrier_expected_lanes",
            (),
        )
        if raw_queued_expected_lanes is None:
            raw_queued_expected_lanes = ()
        queued_expected_lanes = self._normalize_m2n_lane_contract(
            raw_queued_expected_lanes,
            field_name="DECODE_ATTN F-to-A queued batch expected lanes",
            require_nonempty=False,
        )
        if queued_expected_lanes:
            queued_replica_lanes = tuple(
                lane for lane in queued_expected_lanes if lane[0] == replica_id
            )
            if queued_replica_lanes != tuple(expected_lanes):
                raise RuntimeError(
                    "DECODE_ATTN F-to-A queued batch expected lanes do not match the "
                    f"waiting room: queued={queued_replica_lanes}, "
                    f"room={expected_lanes}"
                )

        queued_requests = getattr(queued_batch, "requests", None)
        if type(queued_requests) is not list or not queued_requests:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued Batch requires a non-empty request list"
            )
        active_requests = []
        for queued_request in queued_requests:
            if type(queued_request) is not Request:
                raise ValueError(
                    "DECODE_ATTN F-to-A queued Batch contains a queued request "
                    "that is not an exact Request: "
                    f"value={queued_request!r}"
                )
            completed = getattr(queued_request, "completed", None)
            if type(completed) is not bool:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A queued request.completed must be an exact "
                    f"bool, got {completed!r}"
                )
            roundtrip_inflight = getattr(
                queued_request,
                "af_roundtrip_inflight",
                None,
            )
            if type(roundtrip_inflight) is not bool:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A queued request.af_roundtrip_inflight must "
                    f"be an exact bool, got {roundtrip_inflight!r}"
                )
            if roundtrip_inflight is not False:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A queued request roundtrip must already be "
                    f"complete, got {roundtrip_inflight!r}"
                )
            if not completed:
                active_requests.append(queued_request)
        if not active_requests:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued Batch requires an active request"
            )

        self._validate_decode_attn_f2a_cohort_binding(
            queued_batch,
            lane=queue_lane,
            afd_stage_idx=queued_stage_idx,
            requests=queued_requests,
            active_requests=active_requests,
            context="queued batch",
        )

        queued_layers = [
            require_non_negative_int(
                getattr(queued_request, "completed_layer_count", None),
                "DECODE_ATTN F-to-A queued request completed_layer_count",
            )
            for queued_request in active_requests
        ]
        if set(queued_layers) != {next_layer_id}:
            raise RuntimeError(
                "DECODE_ATTN F-to-A queued requests do not match the waiting-room "
                f"layer: queued={queued_layers}, expected={next_layer_id}"
            )

        queued_request_token_indices = [
            require_non_negative_int(
                getattr(queued_request, "current_decode_token_index", None),
                "DECODE_ATTN F-to-A queued request decode_token_index",
            )
            for queued_request in active_requests
        ]
        queued_replay_token_index = getattr(
            queued_batch,
            "replay_decode_token_index",
            None,
        )
        if queued_replay_token_index is None:
            if len(set(queued_request_token_indices)) != 1:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A queued Batch has mixed decode token indices "
                    "without replay identity"
                )
            queued_token_index = queued_request_token_indices[0]
        else:
            queued_token_index = require_non_negative_int(
                queued_replay_token_index,
                "DECODE_ATTN F-to-A queued batch replay_decode_token_index",
            )
            if queued_token_index != queued_request_token_indices[0]:
                raise RuntimeError(
                    "DECODE_ATTN F-to-A queued batch replay decode token does not "
                    "match its active request head"
                )
        return queued_global_id, queued_token_index

    def _handle_m2n_arrival_decode_ffn(
        self,
        time: float,
        batch: Batch,
        transfer_info: "M2NTransferInfo",
        logger
    ) -> List:
        """Handle M2N transfer arrival at decode-ffn cluster (EP=1 path).

        When activation data arrives from decode-attn cluster:
        1. Record per-request arrival time at DECODE_FFN
        2. Add to per-(wire-layer, stage-slot) grouping barrier
        3. When all expected lanes arrive, promote group and trigger scheduling
        """
        from frontier.events.cluster_schedule_event import ClusterScheduleEvent

        (
            layer_id,
            afd_stage_idx,
            barrier_round_id,
            lane,
            barrier_expected_lanes,
            expected_lanes,
            group_key,
            expected_lane_contract,
        ) = self._validate_decode_ffn_m2n_receipt(batch, transfer_info)

        for request in batch.requests:
            request.on_arrival(time, self._cluster_type)

        batch.decode_ffn_m2n_arrival_time = time

        room = self._m2n_waiting_by_layer.get(group_key)
        if room is None:
            room = {
                'per_lane_queues': defaultdict(deque),
                'lanes_rr_order': deque(),
                'rr_cursor': 0,
                'expected_lane_contract': expected_lane_contract,
            }
            self._m2n_waiting_by_layer[group_key] = room

        if hasattr(self, '_ffn_lane_to_target_replica'):
            transfer_info.target_ffn_replica_id = self._ffn_lane_to_target_replica.get(lane)
        if lane not in room['per_lane_queues']:
            room['per_lane_queues'][lane] = deque()
        was_empty = (len(room['per_lane_queues'][lane]) == 0)
        room['per_lane_queues'][lane].append((batch, transfer_info))
        if was_empty:
            room['lanes_rr_order'].append(lane)

        logger.info(
            f"[FFN-M2N-ARRIVAL] wire_layer={layer_id} afd_stage_idx={afd_stage_idx} "
            f"barrier_round_id={barrier_round_id} lane={lane} "
            f"enqueued; ready_lanes={len(room['lanes_rr_order'])}/{expected_lanes}"
        )

        promoted = self._try_promote_decode_ffn_group(
            time,
            group_key,
            room,
            logger,
            allow_idle_injection=(not batch.is_idle) and not bool(barrier_expected_lanes),
            expected_lanes=expected_lanes,
            expected_lane_ids=barrier_expected_lanes or None,
        )

        if self._is_periodic_scheduling_enabled:
            return []
        if promoted:
            return [ClusterScheduleEvent(time, self._cluster_type)]
        return []

    def _try_promote_decode_ffn_group(
        self,
        time: float,
        group_key,
        room: dict,
        logger,
        *,
        allow_idle_injection: bool,
        expected_lanes: int | None = None,
        expected_lane_ids: Optional[List[tuple[int, int]]] = None,
    ) -> bool:
        if type(allow_idle_injection) is not bool:
            raise ValueError(
                "DECODE_FFN allow_idle_injection must be an exact bool, "
                f"got {allow_idle_injection!r}"
            )
        if expected_lanes is None:
            expected_lanes = getattr(self, "_ffn_group_micro_batches", None)
        if type(expected_lanes) is not int or expected_lanes <= 0:
            raise ValueError(
                "DECODE_FFN expected_lanes must be an exact positive int, "
                f"got {expected_lanes!r}"
            )
        if type(getattr(self, "_m2n_waiting_by_layer", None)) is not dict:
            raise RuntimeError(
                "DECODE_FFN _m2n_waiting_by_layer must be an exact dict"
            )
        if self._m2n_waiting_by_layer.get(group_key) is not room:
            raise RuntimeError(
                "DECODE_FFN promotion room is not the registered waiting-room "
                f"owner for group_key={group_key!r}"
            )
        if type(getattr(self, "_m2n_ready_groups", None)) is not deque:
            raise RuntimeError(
                "DECODE_FFN _m2n_ready_groups must be an exact deque"
            )

        room_lane_contract = self._validate_decode_ffn_waiting_room(
            group_key=group_key,
            room=room,
        )
        normalized_expected_lane_ids = None
        if expected_lane_ids is not None:
            normalized_expected_lane_ids = tuple(
                sorted(
                    self._normalize_m2n_lane_contract(
                        expected_lane_ids,
                        field_name="DECODE_FFN promotion expected lane IDs",
                        require_nonempty=True,
                    )
                )
            )
            if normalized_expected_lane_ids != room_lane_contract:
                raise ValueError(
                    "DECODE_FFN promotion expected lane IDs do not match the "
                    f"waiting-room contract: expected={normalized_expected_lane_ids}, "
                    f"room={room_lane_contract}"
                )
        if expected_lanes > len(room_lane_contract):
            raise ValueError(
                "DECODE_FFN expected lane count exceeds the waiting-room lane "
                f"contract: expected={expected_lanes}, "
                f"contract={room_lane_contract}"
            )

        lanes = list(room["lanes_rr_order"])
        if len(lanes) > expected_lanes:
            raise ValueError(
                f"DECODE_FFN grouping lanes exceed expected count: "
                f"lanes={len(lanes)} expected={expected_lanes}"
            )

        idle_lanes_to_inject: List[tuple[int, int]] = []
        if len(lanes) < expected_lanes and allow_idle_injection:
            raw_idle_lanes = getattr(self, "_ffn_idle_lanes", None)
            if type(raw_idle_lanes) is not set:
                raise RuntimeError(
                    "DECODE_FFN _ffn_idle_lanes must be an exact set when idle "
                    "injection is enabled"
                )
            normalized_idle_lanes = set(
                self._normalize_m2n_lane_contract(
                    tuple(raw_idle_lanes),
                    field_name="DECODE_FFN idle lane inventory",
                    require_nonempty=False,
                )
            )
            if not normalized_idle_lanes.issubset(set(room_lane_contract)):
                raise RuntimeError(
                    "DECODE_FFN idle lane inventory is outside the waiting-room "
                    f"contract: idle={sorted(normalized_idle_lanes)}, "
                    f"contract={room_lane_contract}"
                )
            candidate_lane_order = (
                normalized_expected_lane_ids
                if normalized_expected_lane_ids is not None
                else room_lane_contract
            )
            required_idle_lanes = expected_lanes - len(lanes)
            idle_lanes_to_inject = [
                lane
                for lane in candidate_lane_order
                if lane in normalized_idle_lanes
                and not room["per_lane_queues"].get(lane)
            ][:required_idle_lanes]

        prospective_lanes = lanes + idle_lanes_to_inject
        if len(prospective_lanes) < expected_lanes:
            return False
        if len(prospective_lanes) > expected_lanes:
            raise RuntimeError(
                "DECODE_FFN prospective promotion lanes exceed the expected "
                f"count: lanes={prospective_lanes}, expected={expected_lanes}"
            )

        picked_before_idle_injection = [
            room["per_lane_queues"][lane][0] for lane in lanes
        ]
        padding_plan, padding_summary = self._prepare_dp_padding_on_promotion(
            picked_before_idle_injection
        )

        if idle_lanes_to_inject:
            injected_lanes = self._inject_ffn_idle_lanes_for_barrier(
                time,
                group_key,
                room,
                logger,
                expected_lane_ids=idle_lanes_to_inject,
            )
            if injected_lanes != idle_lanes_to_inject:
                raise RuntimeError(
                    "DECODE_FFN idle lane injection did not match its prepared "
                    f"plan: prepared={idle_lanes_to_inject}, actual={injected_lanes}"
                )

        lanes = list(room["lanes_rr_order"])
        picked = [room["per_lane_queues"][lane][0] for lane in lanes]
        if len(picked) != expected_lanes:
            raise RuntimeError(
                "DECODE_FFN promotion head count changed after preparation: "
                f"picked={len(picked)}, expected={expected_lanes}"
            )

        for batch, padded_metadata in padding_plan:
            batch.afd_stage_metadata = padded_metadata
        for lane in lanes:
            room["per_lane_queues"][lane].popleft()

        ready = [(batch, info) for batch, info in picked if not batch.is_idle]
        if ready:
            self._m2n_ready_groups.append(ready)

        room["lanes_rr_order"] = deque(
            [ln for ln in room["lanes_rr_order"] if room["per_lane_queues"][ln]]
        )
        if not room["lanes_rr_order"]:
            self._m2n_waiting_by_layer.pop(group_key, None)

        if padding_summary is not None:
            padded_lane_count, dp_stage_max_tokens = padding_summary
            logger.info(
                f"[FFN-DP-PADDING] Applied DP padding across {padded_lane_count} "
                f"lanes: dp_stage_max_tokens={dp_stage_max_tokens} "
                f"padded_total={sum(dp_stage_max_tokens)}"
            )
        return True

    def _inject_ffn_idle_lanes_for_barrier(
        self,
        time: float,
        group_key,
        room: dict,
        logger,
        *,
        expected_lane_ids: Optional[List[tuple[int, int]]] = None,
    ) -> List[tuple[int, int]]:
        """Inject idle sentinel batches for missing FFN lanes to unblock the barrier."""
        if self._cluster_type is not ClusterType.DECODE_FFN:
            raise ValueError(
                "FFN idle lane injection requires a DECODE_FFN scheduler"
            )

        if (
            not isinstance(time, Real)
            or isinstance(time, bool)
            or not math.isfinite(float(time))
        ):
            raise ValueError(
                "DECODE_FFN idle injection time must be a finite int or float, "
                f"got {time!r}"
            )
        time = float(time)
        room_lane_contract = self._validate_decode_ffn_waiting_room(
            group_key=group_key,
            room=room,
        )

        ffn_idle_lanes = getattr(self, "_ffn_idle_lanes", None)
        if type(ffn_idle_lanes) is not set:
            raise RuntimeError("DECODE_FFN _ffn_idle_lanes must be an exact set")
        if not ffn_idle_lanes:
            return []

        normalized_idle_lanes = set(
            self._normalize_m2n_lane_contract(
                tuple(ffn_idle_lanes),
                field_name="DECODE_FFN idle lane inventory",
                require_nonempty=False,
            )
        )
        if not normalized_idle_lanes.issubset(set(room_lane_contract)):
            raise RuntimeError(
                "DECODE_FFN idle lane inventory is outside the waiting-room "
                f"contract: idle={sorted(normalized_idle_lanes)}, "
                f"contract={room_lane_contract}"
            )

        if expected_lane_ids is not None:
            normalized_expected_lane_ids = self._normalize_m2n_lane_contract(
                expected_lane_ids,
                field_name="DECODE_FFN idle injection candidate lanes",
                require_nonempty=False,
            )
            if not set(normalized_expected_lane_ids).issubset(
                set(room_lane_contract)
            ):
                raise ValueError(
                    "DECODE_FFN idle injection candidate lane is outside the "
                    f"waiting-room contract: candidates={normalized_expected_lane_ids}, "
                    f"contract={room_lane_contract}"
                )
            candidate_lanes = [
                lane
                for lane in normalized_expected_lane_ids
                if lane in normalized_idle_lanes
            ]
        else:
            candidate_lanes = [
                lane for lane in room_lane_contract if lane in normalized_idle_lanes
            ]

        idle_created: List[tuple[int, int]] = []
        prepared_idle_entries = []
        afd_stage_idx = group_key[1]
        barrier_round_id = group_key[2] if len(group_key) >= 3 else None
        wire_layer_id = group_key[0]

        for missing_lane in candidate_lanes:
            if room["per_lane_queues"].get(missing_lane):
                continue

            replica_config = getattr(self._config, "replica_config", None)
            if replica_config is None:
                raise RuntimeError(
                    "DECODE_FFN idle injection requires replica_config"
                )
            model_config = getattr(replica_config, "model_config", None)
            if model_config is None:
                raise RuntimeError(
                    "DECODE_FFN idle injection requires model_config"
                )
            is_moe = getattr(model_config, "is_moe", None)
            if type(is_moe) is not bool:
                raise RuntimeError(
                    "DECODE_FFN idle injection model_config.is_moe must be an "
                    f"exact bool, got {is_moe!r}"
                )

            idle_batch = Batch(
                replica_id=missing_lane[0],
                requests=[],
                num_tokens=[],
                is_idle=True,
                is_moe=is_moe,
            )
            idle_batch.afd_stage_idx = afd_stage_idx
            idle_batch.decode_attn_original_replica_id = missing_lane[0]
            idle_batch.decode_attn_original_replica_local_id = missing_lane[1]
            idle_batch.decode_attn_barrier_round_id = barrier_round_id
            idle_batch.decode_attn_barrier_expected_lanes = room_lane_contract
            idle_batch.decode_ffn_layer_id = wire_layer_id
            idle_batch.time = time

            from frontier.entities.m2n_transfer_info import M2NTransferInfo
            idle_transfer = M2NTransferInfo(
                batch=idle_batch,
                source_cluster_type=ClusterType.DECODE_ATTN,
                target_cluster_type=ClusterType.DECODE_FFN,
                source_replica_id=missing_lane[0],
                source_replica_local_id=missing_lane[1],
                activation_size_bytes=0,
                transfer_time_ms=0.0,
                transfer_start_time=time,
                layer_id=wire_layer_id,
                afd_stage_idx=afd_stage_idx,
            )
            prepared_idle_entries.append(
                (missing_lane, (idle_batch, idle_transfer))
            )

        if not prepared_idle_entries:
            return []

        prospective_room = {
            "per_lane_queues": defaultdict(
                deque,
                {
                    lane: deque(lane_queue)
                    for lane, lane_queue in room["per_lane_queues"].items()
                },
            ),
            "lanes_rr_order": deque(room["lanes_rr_order"]),
            "rr_cursor": room["rr_cursor"],
            "expected_lane_contract": room_lane_contract,
        }
        for missing_lane, idle_entry in prepared_idle_entries:
            prospective_room["per_lane_queues"][missing_lane].append(idle_entry)
            prospective_room["lanes_rr_order"].append(missing_lane)
        self._validate_decode_ffn_waiting_room(
            group_key=group_key,
            room=prospective_room,
        )

        for missing_lane, idle_entry in prepared_idle_entries:
            room["per_lane_queues"][missing_lane].append(idle_entry)
            room["lanes_rr_order"].append(missing_lane)
            idle_created.append(missing_lane)

        if idle_created:
            logger.info(
                f"[FFN-M2N-IDLE] Injected idle lanes for barrier: "
                f"afd_stage_idx={afd_stage_idx} wire_layer={wire_layer_id} "
                f"barrier_round_id={barrier_round_id} "
                f"missing={sorted(idle_created)}"
            )
        return idle_created

    def _promote_incomplete_m2n_groups_with_idle_lanes(self, logger) -> int:
        """Promote any incomplete FFN grouping barriers by injecting idle lanes."""
        from frontier.events.cluster_schedule_event import ClusterScheduleEvent
        import time as _time_mod

        promoted_count = 0
        for group_key, room in list(self._m2n_waiting_by_layer.items()):
            if self._try_promote_decode_ffn_group(
                0.0,
                group_key,
                room,
                logger,
                allow_idle_injection=True,
            ):
                promoted_count += 1
        return promoted_count

    def _prepare_dp_padding_on_promotion(
        self,
        picked: List[tuple],
    ) -> tuple[List[tuple[Any, Any]], Optional[tuple[int, List[int]]]]:
        """Build DP-padding replacements without mutating queued batches."""

        from frontier.entities.batch import AFDStageMetadata
        from frontier.entities.m2n_transfer_info import M2NTransferInfo

        batches_with_meta = []
        for picked_index, picked_entry in enumerate(picked):
            if type(picked_entry) is not tuple or len(picked_entry) != 2:
                raise RuntimeError(
                    "DECODE_FFN promotion entry must be an exact "
                    f"(batch, transfer_info) tuple, got {picked_entry!r} at "
                    f"index {picked_index}"
                )
            batch, transfer_info = picked_entry
            if type(transfer_info) is not M2NTransferInfo:
                raise RuntimeError(
                    "DECODE_FFN promotion transfer must be an exact "
                    f"M2NTransferInfo at index {picked_index}"
                )
            if transfer_info.batch is not batch:
                raise RuntimeError(
                    "DECODE_FFN promotion batch identity does not match "
                    f"transfer_info.batch at index {picked_index}"
                )
            is_idle = getattr(batch, "is_idle", None)
            if type(is_idle) is not bool:
                raise RuntimeError(
                    "DECODE_FFN promotion batch is_idle must be an exact bool, "
                    f"got {is_idle!r}"
                )
            metadata = getattr(batch, "afd_stage_metadata", None)
            if is_idle or metadata is None:
                continue
            if type(metadata) is not AFDStageMetadata:
                raise RuntimeError(
                    "DECODE_FFN promotion afd_stage_metadata must be an exact "
                    f"AFDStageMetadata, got {type(metadata).__name__}"
                )
            requests = getattr(batch, "requests", None)
            num_tokens = getattr(batch, "num_tokens", None)
            if type(requests) is not list or type(num_tokens) is not list:
                raise RuntimeError(
                    "DECODE_FFN promotion batch requests and num_tokens must be "
                    "exact lists before DP padding"
                )
            if len(requests) != len(num_tokens):
                raise RuntimeError(
                    "DECODE_FFN promotion batch request/token lengths mismatch: "
                    f"requests={len(requests)}, num_tokens={len(num_tokens)}"
                )
            for token_count in num_tokens:
                if type(token_count) is not int or token_count < 0:
                    raise RuntimeError(
                        "DECODE_FFN promotion num_tokens must contain exact "
                        f"non-negative ints, got {token_count!r}"
                    )
            batches_with_meta.append((batch, metadata, num_tokens))

        if len(batches_with_meta) <= 1:
            return [], None

        num_stages = batches_with_meta[0][1].num_stages
        if type(num_stages) is not int or num_stages <= 0:
            raise RuntimeError(
                "DECODE_FFN promotion metadata num_stages must be an exact "
                f"positive int, got {num_stages!r}"
            )

        all_stage_lens = []
        for batch, metadata, num_tokens in batches_with_meta:
            if type(metadata.num_stages) is not int or metadata.num_stages <= 0:
                raise RuntimeError(
                    "DECODE_FFN promotion metadata num_stages must be an exact "
                    f"positive int, got {metadata.num_stages!r}"
                )
            if metadata.num_stages != num_stages:
                raise ValueError(
                    "Inconsistent num_stages across DP lanes: "
                    f"expected {num_stages}, got {metadata.num_stages}"
                )
            stage_lens = AFDStageMetadata.compute_stage_token_lens(
                num_reqs=len(batch.requests),
                num_tokens_per_req=list(num_tokens),
                num_stages=num_stages,
            )
            while len(stage_lens) < num_stages:
                stage_lens.append(1)
            if len(stage_lens) != num_stages:
                raise RuntimeError(
                    "DECODE_FFN promotion stage-token plan does not match "
                    f"num_stages: planned={len(stage_lens)}, "
                    f"num_stages={num_stages}"
                )
            all_stage_lens.append(stage_lens)

        dp_stage_max_tokens = [
            max(lane_lens[stage_index] for lane_lens in all_stage_lens)
            for stage_index in range(num_stages)
        ]
        padding_plan = [
            (
                batch,
                metadata.with_dp_padding(
                    dp_stage_max_tokens=dp_stage_max_tokens,
                ),
            )
            for batch, metadata, _ in batches_with_meta
        ]
        return padding_plan, (len(batches_with_meta), dp_stage_max_tokens)

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
        padding_plan, padding_summary = self._prepare_dp_padding_on_promotion(
            picked
        )
        if padding_summary is None:
            return
        for batch, padded_metadata in padding_plan:
            batch.afd_stage_metadata = padded_metadata

        padded_lane_count, dp_stage_max_tokens = padding_summary
        logger.info(
            f"[FFN-DP-PADDING] Applied DP padding across {padded_lane_count} lanes: "
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
        """Handle M2N transfer arrival at decode-attn cluster (return from decode-ffn).

        When results return from decode-ffn cluster:
        1. Increment layer count on micro-batch
        2. If last layer: emit GlobalBatchEndEvent
        3. If not last layer: enqueue for next attention round
        """
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent
        from frontier.events.global_batch_end_event import GlobalBatchEndEvent
        from frontier.events.cluster_schedule_event import ClusterScheduleEvent

        receipt = self._validate_decode_attn_m2n_receipt(
            micro_batch,
            transfer_info,
            expected_roundtrip_inflight=False,
        )
        logger.info(f"[AF-ARRIVAL] M2N returned micro batch {micro_batch.id} at decode-attn; advancing request states")

        next_events = []

        model_config = self._config.replica_config.model_config
        total_layers = model_config.num_layers

        try:
            req_stats = [
                f"req={r.id}|tok_idx={getattr(r,'current_decode_token_index',None)}|completed_layers={getattr(r,'completed_layer_count',None)}"
                for r in micro_batch.requests
            ]
            logger.info(
                f"[AF-ARRIVAL][BEFORE] mb={micro_batch.id} inflight_layers={getattr(micro_batch,'af_inflight_layer_count',None)} "
                f"/ total_layers={total_layers}; {', '.join(req_stats)}"
            )
        except Exception as e:
            logger.info(f"[AF-ARRIVAL][BEFORE] debug failed: {e}")

        micro_batch.mb_on_step_layer_count_increment()

        is_mb_last_layer = receipt["is_last_layer"]

        logger.info(
            f"[AF-ARRIVAL][AFTER] mb={micro_batch.id} inflight_layers={getattr(micro_batch,'af_inflight_layer_count',None)} "
            f"/ total_layers={total_layers}; is_mb_last_layer={is_mb_last_layer}"
        )

        replica_id = receipt["replica_id"]
        replica_local_id = receipt["replica_local_id"]

        ready_for_reschedule = False
        if not is_mb_last_layer:
            ready_for_reschedule = self._enqueue_decode_attn_return_round(
                micro_batch,
                receipt=receipt,
                logger=logger,
            )
        else:
            global_end_time = self.resolve_decode_attn_boundary_first_mixed_global_end_time(
                time,
                micro_batch,
            )
            logger.info(
                f"[AF-ARRIVAL][FINAL-LAYER] mb={micro_batch.id} emitting GlobalBatchEndEvent; "
                f"replica={replica_id}, replica_local_id={replica_local_id}, "
                f"global_end_time={global_end_time}"
            )
            current_exec_sigs = [
                Batch._get_request_execution_signature(r) for r in micro_batch.requests
            ]
            current_mut_sigs = [
                Batch._get_request_mutation_signature(r) for r in micro_batch.requests
            ]
            next_events.append(
                GlobalBatchEndEvent(
                    global_end_time,
                    replica_id,
                    replica_local_id,
                    micro_batch,
                    self._cluster_type,
                    request_execution_signatures=current_exec_sigs,
                    request_mutation_signatures=current_mut_sigs,
                )
            )

        if self._is_periodic_scheduling_enabled:
            logger.info(
                f"[AF-ARRIVAL] Will process in next periodic cycle (interval={self._periodic_scheduling_interval_ms}ms); "
                f"af_queue_size={len(self._af_batch_queue)}, next_events={len(next_events)}"
            )
            return next_events
        else:
            logger.info(
                f"[AF-ARRIVAL] Trigger immediate cluster scheduling; af_queue_size={len(self._af_batch_queue)}, next_events={len(next_events)}"
            )
            if next_events or ready_for_reschedule:
                return next_events + [ClusterScheduleEvent(time, self._cluster_type)]
            return next_events

    def _is_dense_decode_ffn_workflow(self) -> bool:
        """Return whether PD-AF decode-FFN has no MoE/EP lane barrier semantics."""
        replica_config = getattr(self._config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        return not bool(getattr(model_config, "is_moe", False))

    def _partition_decode_attn_lanes_for_dense_ffn(
        self,
        lanes: tuple[tuple[int, int], ...],
    ) -> List[tuple[tuple[int, int], ...]]:
        """Split a dense A→F barrier cohort into FFN-replica-sized subgroups."""
        if not lanes:
            return []
        ffn_replicas = int(getattr(self._config, "decode_ffn_cluster_num_replicas", 1) or 1)
        partition_count = max(1, min(ffn_replicas, len(lanes)))
        chunk_size = (len(lanes) + partition_count - 1) // partition_count
        return [
            tuple(lanes[start : start + chunk_size])
            for start in range(0, len(lanes), chunk_size)
        ]

    def resolve_decode_attn_boundary_first_mixed_global_end_time(
        self,
        time: float,
        batch: "Batch",
    ) -> float:
        """Return the global end time for a completed decode-attn batch.

        For mixed-layer models this would account for the first dense→MoE boundary;
        for the current release (single-model, uniform layer type per iteration),
        the batch ends at the current event time.
        """
        return time

    @staticmethod
    def _validate_decode_attn_a2f_topology_value(
        value: Any,
        *,
        field_name: str,
    ) -> int:
        """Validate one exact non-negative A-to-F topology coordinate."""

        if type(value) is not int or value < 0:
            raise ValueError(
                f"DECODE_ATTN A-to-F {field_name} must be an exact "
                f"non-negative int, got {value!r}"
            )
        return value

    def _validate_decode_attn_a2f_batch_entry(
        self,
        *,
        batch: Batch,
        lane: tuple[int, int],
        layer_id: int,
        afd_stage_idx: int,
        model_is_moe: bool,
        context: str,
        allow_idle: bool,
    ) -> None:
        """Validate one A-to-F Batch before any scheduler state is touched."""

        if type(batch) is not Batch:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} must be an exact Batch, "
                f"got {type(batch).__name__}"
            )
        if type(model_is_moe) is not bool:
            raise RuntimeError(
                "DECODE_ATTN A-to-F model_config.is_moe must be an exact bool, "
                f"got {model_is_moe!r}"
            )
        normalized_lane = self._normalize_m2n_lane_contract(
            [lane],
            field_name=f"DECODE_ATTN A-to-F {context} lane",
            require_nonempty=True,
        )[0]
        layer_id = self._validate_decode_attn_a2f_topology_value(
            layer_id,
            field_name=f"{context} layer_id",
        )
        afd_stage_idx = self._validate_decode_attn_a2f_topology_value(
            afd_stage_idx,
            field_name=f"{context} afd_stage_idx",
        )

        raw_is_idle = getattr(batch, "is_idle", None)
        raw_is_moe = getattr(batch, "is_moe", None)
        if type(raw_is_idle) is not bool:
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} is_idle must be an exact bool, "
                f"got {raw_is_idle!r}"
            )
        if type(raw_is_moe) is not bool:
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} is_moe must be an exact bool, "
                f"got {raw_is_moe!r}"
            )
        if raw_is_moe is not model_is_moe:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} is_moe does not match model "
                f"configuration: batch={raw_is_moe}, model={model_is_moe}"
            )
        if raw_is_idle and not allow_idle:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} idle Batch is not valid for an "
                "incoming lane"
            )

        batch_stage_idx = getattr(batch, "afd_stage_idx", None)
        if type(batch_stage_idx) is not int or batch_stage_idx != afd_stage_idx:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} afd_stage_idx mismatch: "
                f"expected={afd_stage_idx}, got={batch_stage_idx!r}"
            )
        batch_lane = (
            getattr(batch, "decode_attn_original_replica_id", None),
            getattr(batch, "decode_attn_original_replica_local_id", None),
        )
        if (
            type(batch_lane[0]) is not int
            or batch_lane[0] < 0
            or (
                batch_lane[1] is not None
                and (type(batch_lane[1]) is not int or batch_lane[1] < 0)
            )
        ):
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} original lane must contain a "
                f"Replica ID and optional full-stage identity, got {batch_lane!r}"
            )
        if batch_lane != normalized_lane:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} original lane mismatch: "
                f"expected={normalized_lane}, got={batch_lane}"
            )

        requests = getattr(batch, "requests", None)
        num_tokens = getattr(batch, "num_tokens", None)
        if type(requests) is not list:
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} requests must be an exact list, "
                f"got {type(requests).__name__}"
            )
        if type(num_tokens) is not list or len(num_tokens) != len(requests):
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} num_tokens must be an exact list "
                "matching requests"
            )
        for token_count in num_tokens:
            if type(token_count) is not int or token_count < 0:
                raise RuntimeError(
                    f"DECODE_ATTN A-to-F {context} num_tokens must contain exact "
                    f"non-negative ints, got {token_count!r}"
                )
        if raw_is_idle:
            if requests or num_tokens:
                raise ValueError(
                    f"DECODE_ATTN A-to-F {context} idle Batch must not contain "
                    "requests or token counts"
                )
            active_requests: List[Request] = []
        else:
            if not requests:
                raise ValueError(
                    f"DECODE_ATTN A-to-F {context} non-idle Batch must contain "
                    "requests"
                )
            active_requests = []
            for request in requests:
                if type(request) is not Request:
                    raise ValueError(
                        f"DECODE_ATTN A-to-F {context} contains a request that "
                        f"is not an exact Request: {request!r}"
                    )
                request_id = getattr(request, "id", None)
                if type(request_id) is not int or request_id < 0:
                    raise ValueError(
                        f"DECODE_ATTN A-to-F {context} request ID must be an exact "
                        f"non-negative int, got {request_id!r}"
                    )
                completed = getattr(request, "completed", None)
                if type(completed) is not bool:
                    raise RuntimeError(
                        f"DECODE_ATTN A-to-F {context} request.completed must be "
                        f"an exact bool, got {completed!r}"
                    )
                request_layer_id = getattr(request, "completed_layer_count", None)
                if type(request_layer_id) is not int or request_layer_id < 0:
                    raise RuntimeError(
                        f"DECODE_ATTN A-to-F {context} request layer must be an "
                        f"exact non-negative int, got {request_layer_id!r}"
                    )
                if not completed:
                    active_requests.append(request)
                    if request_layer_id != layer_id:
                        raise ValueError(
                            f"DECODE_ATTN A-to-F {context} request layer mismatch: "
                            f"expected={layer_id}, got={request_layer_id}, "
                            f"request_id={request_id}"
                        )
            if not active_requests:
                raise ValueError(
                    f"DECODE_ATTN A-to-F {context} non-idle Batch has no active "
                    "requests"
                )

        decode_ffn_layer_id = getattr(batch, "decode_ffn_layer_id", None)
        if decode_ffn_layer_id is not None:
            if type(decode_ffn_layer_id) is not int or decode_ffn_layer_id < 0:
                raise ValueError(
                    f"DECODE_ATTN A-to-F {context} decode_ffn_layer_id must be "
                    f"None or an exact non-negative int, got {decode_ffn_layer_id!r}"
                )
            if decode_ffn_layer_id != layer_id:
                raise ValueError(
                    f"DECODE_ATTN A-to-F {context} decode_ffn_layer_id mismatch: "
                    f"expected={layer_id}, got={decode_ffn_layer_id}"
                )

        cohort_id = getattr(batch, "decode_attn_cohort_id", None)
        cohort_request_ids = getattr(batch, "decode_attn_cohort_request_ids", None)
        if cohort_id is None:
            if cohort_request_ids is not None:
                raise ValueError(
                    f"DECODE_ATTN A-to-F {context} has cohort request IDs without "
                    "a cohort ID"
                )
            return
        if type(cohort_id) is not int or cohort_id < 0:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} cohort ID must be an exact "
                f"non-negative int, got {cohort_id!r}"
            )
        if type(cohort_request_ids) is not tuple:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} cohort request IDs must be an "
                f"exact tuple, got {cohort_request_ids!r}"
            )
        self._validate_decode_attn_f2a_cohort_binding(
            batch,
            lane=normalized_lane,
            afd_stage_idx=afd_stage_idx,
            requests=requests,
            active_requests=active_requests,
            context=f"A-to-F {context}",
        )
        self._validate_decode_attn_a2f_cohort_phase(
            batch,
            layer_id=layer_id,
            afd_stage_idx=afd_stage_idx,
            context=context,
        )

    @staticmethod
    def _validate_decode_attn_cohort_stage_maps(
        cohort_state: dict[str, Any],
        *,
        context: str,
    ) -> tuple[set[int], dict[int, str], dict[int, int]]:
        """Validate the complete stage-local state for one DECODE_ATTN cohort."""

        if type(cohort_state) is not dict:
            raise RuntimeError(
                f"DECODE_ATTN {context} active cohort state must be an exact dict"
            )

        active_stage_indices = cohort_state.get("active_stage_indices")
        if type(active_stage_indices) is not set or not active_stage_indices:
            raise RuntimeError(
                f"DECODE_ATTN {context} cohort active_stage_indices must be a "
                "non-empty exact set"
            )
        for stage_idx in active_stage_indices:
            if type(stage_idx) is not int or stage_idx < 0:
                raise RuntimeError(
                    f"DECODE_ATTN {context} cohort active stage indices must "
                    f"contain exact non-negative ints, got {stage_idx!r}"
                )

        stage_phases = cohort_state.get("stage_phases")
        if type(stage_phases) is not dict:
            raise RuntimeError(
                f"DECODE_ATTN {context} cohort stage phases must be an exact dict"
            )
        for stage_idx, stage_phase in stage_phases.items():
            if type(stage_idx) is not int or stage_idx < 0:
                raise RuntimeError(
                    f"DECODE_ATTN {context} cohort stage phase indices must "
                    f"contain exact non-negative ints, got {stage_idx!r}"
                )
            if type(stage_phase) is not str or stage_phase not in {
                "local_attn",
                "ffn_inflight",
            }:
                raise RuntimeError(
                    f"DECODE_ATTN {context} cohort stage phase must be "
                    "local_attn or ffn_inflight, "
                    f"got {stage_phase!r}"
                )
        if set(stage_phases) != active_stage_indices:
            raise RuntimeError(
                f"DECODE_ATTN {context} cohort stage phase key set must exactly "
                "match active stages: "
                f"phase_keys={sorted(stage_phases)}, "
                f"active={sorted(active_stage_indices)}"
            )

        stage_layers = cohort_state.get("stage_current_layer_ids")
        if type(stage_layers) is not dict:
            raise RuntimeError(
                f"DECODE_ATTN {context} cohort stage layers must be an exact dict"
            )
        for stage_idx, stage_layer in stage_layers.items():
            if type(stage_idx) is not int or stage_idx < 0:
                raise RuntimeError(
                    f"DECODE_ATTN {context} cohort stage layer indices must "
                    f"contain exact non-negative ints, got {stage_idx!r}"
                )
            if type(stage_layer) is not int or stage_layer < 0:
                raise RuntimeError(
                    f"DECODE_ATTN {context} cohort stage layer must be an exact "
                    f"non-negative int, got {stage_layer!r}"
                )
        if set(stage_layers) != active_stage_indices:
            raise RuntimeError(
                f"DECODE_ATTN {context} cohort stage layer key set must exactly "
                "match active stages: "
                f"layer_keys={sorted(stage_layers)}, "
                f"active={sorted(active_stage_indices)}"
            )

        return active_stage_indices, stage_phases, stage_layers

    def _validate_decode_attn_a2f_cohort_phase(
        self,
        batch: Batch,
        *,
        layer_id: int,
        afd_stage_idx: int,
        context: str,
    ) -> None:
        """Validate the stage-local phase/layer of an A-to-F cohort."""

        cohort_id = getattr(batch, "decode_attn_cohort_id", None)
        lane = (
            getattr(batch, "decode_attn_original_replica_id", None),
            getattr(batch, "decode_attn_original_replica_local_id", None),
        )
        replica_schedulers = getattr(self, "_replica_schedulers", None)
        if type(replica_schedulers) is not dict or lane not in replica_schedulers:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} cohort lane is absent from the "
                f"replica scheduler topology: lane={lane}"
            )
        cohort_states = getattr(
            replica_schedulers[lane],
            "_decode_attn_active_cohort_states",
            None,
        )
        if type(cohort_states) is not dict:
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} active cohort states must be an "
                "exact dict"
            )
        cohort_state = cohort_states.get(cohort_id)
        if type(cohort_state) is not dict:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} references an inactive or unknown "
                f"cohort: cohort_id={cohort_id}, lane={lane}"
            )
        active_stage_indices, stage_phases, stage_layers = (
            self._validate_decode_attn_cohort_stage_maps(
                cohort_state,
                context=f"A-to-F {context}",
            )
        )
        if afd_stage_idx not in active_stage_indices:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} stage is not active in the cohort: "
                f"stage={afd_stage_idx}, active={sorted(active_stage_indices)}"
            )

        aggregate_phase = cohort_state.get("af_phase")
        if type(aggregate_phase) is not str:
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} cohort af_phase must be an exact "
                f"str, got {aggregate_phase!r}"
            )
        aggregate_layer = cohort_state.get("current_layer_id")
        if type(aggregate_layer) is not int or aggregate_layer < 0:
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} cohort current_layer_id must be an "
                f"exact non-negative int, got {aggregate_layer!r}"
            )
        stage_phase = stage_phases[afd_stage_idx]
        stage_layer = stage_layers[afd_stage_idx]
        if stage_phase != "local_attn":
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} cohort stage is not in local_attn "
                f"phase: stage={afd_stage_idx}, phase={stage_phase!r}"
            )
        if stage_layer != layer_id:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} cohort layer mismatch: "
                f"expected={layer_id}, got={stage_layer}"
            )

    def _validate_decode_attn_a2f_waiting_room(
        self,
        *,
        group_key: tuple[int, int],
        room: dict,
        expected_lane_contract: tuple[tuple[int, int], ...],
        incoming_batch: Optional[Batch] = None,
    ) -> tuple[tuple[int, int], ...]:
        """Validate one A-to-F waiting room without mutating runtime state."""

        if type(group_key) is not tuple or len(group_key) != 2:
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting-room key must be an exact "
                f"(layer_id, afd_stage_idx) tuple, got {group_key!r}"
            )
        layer_id = self._validate_decode_attn_a2f_topology_value(
            group_key[0],
            field_name="waiting-room layer_id",
        )
        afd_stage_idx = self._validate_decode_attn_a2f_topology_value(
            group_key[1],
            field_name="waiting-room afd_stage_idx",
        )
        replica_config = getattr(self._config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        model_is_moe = getattr(model_config, "is_moe", None)
        if type(model_is_moe) is not bool:
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting-room model_config.is_moe must be "
                f"an exact bool, got {model_is_moe!r}"
            )

        normalized_expected_contract = tuple(
            sorted(
                self._normalize_m2n_lane_contract(
                    expected_lane_contract,
                    field_name="DECODE_ATTN A-to-F expected lane topology",
                    require_nonempty=True,
                )
            )
        )
        if expected_lane_contract != normalized_expected_contract:
            raise RuntimeError(
                "DECODE_ATTN A-to-F expected lane topology must be an exact "
                f"canonical tuple, got {expected_lane_contract!r}"
            )

        if type(room) is not dict:
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting room must be an exact dict, "
                f"got {type(room).__name__}"
            )
        expected_room_fields = {
            "per_lane_queues",
            "expected_lane_contract",
        }
        if set(room) != expected_room_fields:
            missing_fields = expected_room_fields - set(room)
            if "expected_lane_contract" in missing_fields:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting room is missing the expected "
                    "lane contract"
                )
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting-room schema mismatch: "
                f"expected={sorted(expected_room_fields)}, actual={sorted(room)}"
            )

        raw_room_contract = room["expected_lane_contract"]
        if type(raw_room_contract) is not tuple:
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting-room expected lane contract must "
                f"be an exact tuple, got {raw_room_contract!r}"
            )
        room_contract = tuple(
            sorted(
                self._normalize_m2n_lane_contract(
                    raw_room_contract,
                    field_name=(
                        "DECODE_ATTN A-to-F waiting-room expected lane contract"
                    ),
                    require_nonempty=True,
                )
            )
        )
        if raw_room_contract != room_contract:
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting-room expected lane contract must "
                f"be canonical, got {raw_room_contract!r}"
            )
        if room_contract != normalized_expected_contract:
            raise ValueError(
                "DECODE_ATTN A-to-F waiting-room lane contract mismatch: "
                f"room={room_contract}, expected={normalized_expected_contract}"
            )

        per_lane_queues = room["per_lane_queues"]
        if (
            type(per_lane_queues) is not defaultdict
            or per_lane_queues.default_factory is not deque
        ):
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting-room per_lane_queues must be an "
                "exact defaultdict(deque)"
            )
        queue_lanes = self._normalize_m2n_lane_contract(
            tuple(per_lane_queues),
            field_name="DECODE_ATTN A-to-F waiting-room queue lanes",
            require_nonempty=False,
        )
        if not set(queue_lanes).issubset(set(room_contract)):
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting-room queue lane is outside the "
                f"expected contract: queues={queue_lanes}, contract={room_contract}"
            )

        seen_batch_identities = set()
        for queue_lane, lane_queue in per_lane_queues.items():
            if type(lane_queue) is not deque:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F waiting-room lane queue must be an "
                    f"exact deque: lane={queue_lane}, "
                    f"got={type(lane_queue).__name__}"
                )
            for queue_index, queued_entry in enumerate(lane_queue):
                if type(queued_entry) is not tuple or len(queued_entry) != 2:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F waiting-room queued entry must be "
                        f"an exact (layer_id, Batch) tuple: lane={queue_lane}, "
                        f"index={queue_index}, value={queued_entry!r}"
                    )
                queued_layer_id, queued_batch = queued_entry
                if type(queued_layer_id) is not int or queued_layer_id < 0:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F waiting-room queued layer_id must "
                        f"be an exact non-negative int, got {queued_layer_id!r}"
                    )
                if queued_layer_id != layer_id:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F waiting-room queued layer mismatch: "
                        f"room={layer_id}, queued={queued_layer_id}"
                    )
                if type(queued_batch) is not Batch:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F waiting-room queued batch must be "
                        f"an exact Batch, got {type(queued_batch).__name__}"
                    )
                if incoming_batch is not None and queued_batch is incoming_batch:
                    raise ValueError(
                        "DECODE_ATTN A-to-F waiting room already contains the "
                        "incoming batch object"
                    )
                queued_batch_identity = id(queued_batch)
                if queued_batch_identity in seen_batch_identities:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F waiting room contains a duplicate "
                        "queued batch object"
                    )
                seen_batch_identities.add(queued_batch_identity)

                queued_is_idle = getattr(queued_batch, "is_idle", None)
                if type(queued_is_idle) is not bool:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F waiting-room queued batch is_idle "
                        f"must be an exact bool, got {queued_is_idle!r}"
                    )
                queued_stage_idx = getattr(queued_batch, "afd_stage_idx", None)
                if (
                    type(queued_stage_idx) is not int
                    or queued_stage_idx != afd_stage_idx
                ):
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F waiting-room queued batch stage "
                        f"mismatch: room={afd_stage_idx}, batch={queued_stage_idx!r}"
                    )
                queued_replica_id = getattr(
                    queued_batch,
                    "decode_attn_original_replica_id",
                    None,
                )
                queued_replica_local_id = getattr(
                    queued_batch,
                    "decode_attn_original_replica_local_id",
                    None,
                )
                if (
                    type(queued_replica_id) is not int
                    or (
                        queued_replica_local_id is not None
                        and type(queued_replica_local_id) is not int
                    )
                    or (queued_replica_id, queued_replica_local_id) != queue_lane
                ):
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F waiting-room queued batch lane "
                        f"mismatch: queue={queue_lane}, batch="
                        f"{(queued_replica_id, queued_replica_local_id)}"
                    )

                self._validate_decode_attn_a2f_batch_entry(
                    batch=queued_batch,
                    lane=queue_lane,
                    layer_id=layer_id,
                    afd_stage_idx=afd_stage_idx,
                    model_is_moe=model_is_moe,
                    context="waiting-room queued batch",
                    allow_idle=True,
                )

                cohort_id = getattr(queued_batch, "decode_attn_cohort_id", None)
                if cohort_id is not None and (
                    type(cohort_id) is not int or cohort_id < 0
                ):
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F queued batch cohort ID must be None "
                        f"or an exact non-negative int, got {cohort_id!r}"
                    )

                requests = getattr(queued_batch, "requests", None)
                if type(requests) is not list:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F queued batch requests must be an "
                        f"exact list, got {type(requests).__name__}"
                    )
                if queued_is_idle:
                    if requests:
                        raise RuntimeError(
                            "DECODE_ATTN A-to-F queued idle batch must not "
                            "contain requests"
                        )
                    continue

                active_requests = []
                for request in requests:
                    completed = getattr(request, "completed", None)
                    if type(completed) is not bool:
                        raise RuntimeError(
                            "DECODE_ATTN A-to-F queued request completed state "
                            f"must be an exact bool, got {completed!r}"
                        )
                    request_layer_id = getattr(
                        request,
                        "completed_layer_count",
                        None,
                    )
                    if type(request_layer_id) is not int or request_layer_id < 0:
                        raise RuntimeError(
                            "DECODE_ATTN A-to-F queued request layer must be an "
                            f"exact non-negative int, got {request_layer_id!r}"
                        )
                    if not completed:
                        active_requests.append((request, request_layer_id))
                if not active_requests:
                    raise RuntimeError(
                        "DECODE_ATTN A-to-F queued non-idle batch has no active "
                        "requests"
                    )
                for request, request_layer_id in active_requests:
                    if request_layer_id != layer_id:
                        raise RuntimeError(
                            "DECODE_ATTN A-to-F waiting-room queued request "
                            f"layer mismatch: room={layer_id}, request="
                            f"{request_layer_id}, request_id={request.id}"
                        )

        return room_contract

    @staticmethod
    def _validate_decode_attn_a2f_predictor_result(
        predictor_result: Any,
    ) -> tuple[int, int | float]:
        """Validate one A-to-F predictor result without coercing its values."""

        if type(predictor_result) is not tuple or len(predictor_result) != 2:
            raise RuntimeError(
                "DECODE_ATTN A-to-F predictor transfer result must be an exact "
                f"(activation_size, transfer_time) tuple, got {predictor_result!r}"
            )
        activation_size, transfer_time = predictor_result
        if type(activation_size) is not int or activation_size < 0:
            raise ValueError(
                "DECODE_ATTN A-to-F predictor activation_size must be an exact "
                f"non-negative int, got {activation_size!r}"
            )
        if not isinstance(transfer_time, Real) or isinstance(transfer_time, bool):
            raise ValueError(
                "DECODE_ATTN A-to-F predictor transfer_time must be an exact int "
                f"or float, got {transfer_time!r}"
            )
        transfer_time = float(transfer_time)
        if not math.isfinite(transfer_time) or transfer_time < 0:
            raise ValueError(
                "DECODE_ATTN A-to-F predictor transfer_time must be finite and "
                f"non-negative, got {transfer_time!r}"
            )
        return activation_size, transfer_time

    def _prepare_decode_attn_idle_lanes_for_barrier(
        self,
        *,
        time: float,
        group_key: tuple[int, int],
        idle_lanes: List[tuple[int, int]],
        is_moe: bool,
    ) -> List[tuple[tuple[int, int], tuple[int, Batch]]]:
        """Build A-to-F idle entries without mutating the waiting room."""

        normalized_idle_lanes = self._normalize_m2n_lane_contract(
            idle_lanes,
            field_name="DECODE_ATTN A-to-F prepared idle lanes",
            require_nonempty=False,
        )
        if type(is_moe) is not bool:
            raise RuntimeError(
                "DECODE_ATTN A-to-F idle batch is_moe must be an exact bool, "
                f"got {is_moe!r}"
            )

        layer_id, afd_stage_idx = group_key
        prepared_entries = []
        for missing_lane in normalized_idle_lanes:
            idle_batch = Batch(
                replica_id=missing_lane[0],
                requests=[],
                num_tokens=[],
                is_idle=True,
                is_moe=is_moe,
            )
            idle_batch.afd_stage_idx = afd_stage_idx
            idle_batch.decode_attn_original_replica_id = missing_lane[0]
            idle_batch.decode_attn_original_replica_local_id = missing_lane[1]
            idle_batch.time = time
            prepared_entries.append(
                (missing_lane, (layer_id, idle_batch))
            )
        return prepared_entries

    def _release_dense_decode_ffn_a2f_without_lane_barrier(
        self,
        time: float,
        batch: Batch,
        *,
        replica_id: int,
        replica_local_id: int | None,
        layer_id: int,
        logger,
    ) -> List:
        """Stream dense A→F traffic without the MoE all-lane barrier."""
        from frontier.events.m2n_transfer_start_event import M2NTransferStartEvent
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent

        lane = (replica_id, replica_local_id)
        barrier_round_id = self._peek_decode_attn_barrier_round_id()
        activation_size, transfer_time = (
            self._validate_decode_attn_a2f_predictor_result(
                self._m2n_transfer_predictor.get_transfer_info(
                    source_cluster_type=ClusterType.DECODE_ATTN,
                    target_cluster_type=ClusterType.DECODE_FFN,
                    batch=batch,
                    replica_config=self._config.replica_config,
                )
            )
        )

        transfer_event = M2NTransferStartEvent(
            time=time,
            source_replica_id=replica_id,
            source_replica_local_id=replica_local_id,
            source_cluster_type=ClusterType.DECODE_ATTN,
            target_cluster_type=ClusterType.DECODE_FFN,
            batch=batch,
            activation_size_bytes=activation_size,
            transfer_time_ms=transfer_time,
            layer_id=layer_id,
            afd_stage_idx=batch.afd_stage_idx,
        )
        schedule_event = ReplicaScheduleEvent(
            time,
            replica_id,
            self._cluster_type,
            replica_local_id,
        )
        prepared_cohort_update = self._set_decode_attn_batch_cohort_phase(
            batch,
            phase="ffn_inflight",
            replica_id=replica_id,
            replica_local_id=replica_local_id,
            layer_id=layer_id,
            prepare_only=True,
        )

        self._commit_decode_attn_batch_cohort_phases([prepared_cohort_update])
        batch.decode_attn_barrier_round_id = barrier_round_id
        batch.decode_attn_barrier_expected_lanes = (lane,)
        self._decode_attn_barrier_round_counter = barrier_round_id + 1
        logger.info(
            f"[A2F-DENSE-STREAM] layer={layer_id} afd_stage_idx={batch.afd_stage_idx} "
            f"lane={lane} batch_id={batch.id} round={batch.decode_attn_barrier_round_id}"
        )
        return [transfer_event, schedule_event]

    def on_decode_attn_a2f_ready(
        self,
        time: float,
        batch: Batch,
        *,
        replica_id: int,
        replica_local_id: int | None,
        layer_id: int,
        logger,
    ) -> List:
        """Called after DECODE_ATTN batch completion to initiate A→F M2N transfer."""
        from frontier.events.m2n_transfer_start_event import M2NTransferStartEvent
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent

        if self._cluster_type != ClusterType.DECODE_ATTN:
            raise ValueError(
                "on_decode_attn_a2f_ready is only valid for DECODE_ATTN cluster"
            )
        if type(batch) is not Batch:
            raise ValueError(
                "DECODE_ATTN A-to-F admission requires an exact Batch, "
                f"got {type(batch).__name__}"
            )
        layer_id = self._validate_decode_attn_a2f_topology_value(
            layer_id,
            field_name="layer_id",
        )
        afd_stage_idx = self._validate_decode_attn_a2f_topology_value(
            getattr(batch, "afd_stage_idx", None),
            field_name="afd_stage_idx",
        )
        replica_id = self._validate_decode_attn_a2f_topology_value(
            replica_id,
            field_name="replica_id",
        )
        # DECODE_ATTN has one full-stage scheduler per serving Replica.  The
        # second tuple coordinate is intentionally absent; it is not an
        # attention-DP lane and must remain ``None`` on A→F transport.
        if replica_local_id is not None:
            raise ValueError(
                "DECODE_ATTN A-to-F requires full-stage identity with "
                f"replica_local_id=None, got {replica_local_id!r}"
            )
        if (
            not isinstance(time, Real)
            or isinstance(time, bool)
            or not math.isfinite(time)
            or time < 0
        ):
            raise ValueError(
                "DECODE_ATTN A-to-F event time must be a finite non-negative "
                f"int or float, got {time!r}"
            )
        # Predictors commonly return numpy scalar real values. Normalize the
        # validated timestamp before constructing events so downstream event
        # contracts receive a built-in numeric type.
        time = float(time)
        if self._m2n_transfer_predictor is None:
            raise ValueError("M2N transfer predictor not found in decode-attn cluster scheduler")

        # Bind the source Attention Replica at the A→F boundary.  The
        # DECODE_ATTN scheduler is full-stage, so there is no local DP value to
        # carry; ``None`` is the only valid second coordinate.
        original_replica_id = getattr(
            batch, "decode_attn_original_replica_id", None
        )
        if original_replica_id is not None:
            original_replica_id = self._validate_decode_attn_a2f_topology_value(
                original_replica_id,
                field_name="batch original replica_id",
            )
            if original_replica_id != replica_id:
                raise ValueError(
                    "DECODE_ATTN A-to-F batch source Replica mismatch: "
                    f"batch={original_replica_id!r}, event={replica_id!r}"
                )
        batch.decode_attn_original_replica_id = replica_id
        original_replica_local_id = getattr(
            batch, "decode_attn_original_replica_local_id", None
        )
        if original_replica_local_id is not None:
            raise ValueError(
                "DECODE_ATTN A-to-F batch must use full-stage identity with "
                "decode_attn_original_replica_local_id=None, got "
                f"{original_replica_local_id!r}"
            )
        batch.decode_attn_original_replica_local_id = None

        replica_config = getattr(self._config, "replica_config", None)
        if replica_config is None:
            raise RuntimeError(
                "DECODE_ATTN A-to-F admission requires replica_config"
            )
        model_config = getattr(replica_config, "model_config", None)
        if model_config is None:
            raise RuntimeError(
                "DECODE_ATTN A-to-F admission requires model_config"
            )
        model_is_moe = getattr(model_config, "is_moe", None)
        if type(model_is_moe) is not bool:
            raise RuntimeError(
                "DECODE_ATTN A-to-F model_config.is_moe must be an exact bool, "
                f"got {model_is_moe!r}"
            )

        self._validate_decode_attn_a2f_batch_entry(
            batch=batch,
            lane=(replica_id, replica_local_id),
            layer_id=layer_id,
            afd_stage_idx=afd_stage_idx,
            model_is_moe=model_is_moe,
            context="incoming batch",
            allow_idle=False,
        )

        cohort_id = getattr(batch, "decode_attn_cohort_id", None)
        cohort_request_ids = getattr(batch, "decode_attn_cohort_request_ids", None)
        if cohort_id is not None and cohort_request_ids is not None:
            active_local_attn_lanes = self._get_decode_attn_a2f_active_local_attn_lanes(
                cohort_id=cohort_id,
                request_ids=cohort_request_ids,
                afd_stage_idx=afd_stage_idx,
                layer_id=layer_id,
            )
            expected_lane_contract = tuple(
                sorted(
                    self._normalize_m2n_lane_contract(
                        active_local_attn_lanes,
                        field_name=(
                            "DECODE_ATTN A-to-F active cohort local_attn topology"
                        ),
                        require_nonempty=True,
                    )
                )
            )
        else:
            expected_lane_contract = tuple(
                sorted(
                    self._normalize_m2n_lane_contract(
                        self._get_decode_attn_a2f_expected_lanes(
                            afd_stage_idx,
                            layer_id=layer_id,
                        ),
                        field_name="DECODE_ATTN A-to-F expected lane topology",
                        require_nonempty=True,
                    )
                )
            )

        group_key = (layer_id, afd_stage_idx)
        lane = (replica_id, replica_local_id)
        if lane not in expected_lane_contract:
            raise ValueError(
                "Unexpected lane observed in DECODE_ATTN A→F waiting room: "
                f"group_key={group_key}, lane={lane}, "
                f"expected_lanes={expected_lane_contract}"
            )

        idle_expected_lanes = getattr(self, "_decode_attn_idle_expected_lanes", None)
        if idle_expected_lanes is not None:
            if type(idle_expected_lanes) is not set:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F idle lane inventory must be an exact set"
                )
            normalized_idle_expected_lanes = set(
                self._normalize_m2n_lane_contract(
                    tuple(idle_expected_lanes),
                    field_name="DECODE_ATTN A-to-F idle lane topology",
                    require_nonempty=False,
                )
            )
        else:
            normalized_idle_expected_lanes = set()

        self._peek_decode_attn_barrier_round_id()

        if not model_is_moe:
            events = self._release_dense_decode_ffn_a2f_without_lane_barrier(
                time,
                batch,
                replica_id=replica_id,
                replica_local_id=replica_local_id,
                layer_id=layer_id,
                logger=logger,
            )
            if idle_expected_lanes is not None:
                idle_expected_lanes.discard(lane)
            return events

        waiting_rooms = getattr(self, "_a2f_waiting_by_layer", None)
        if type(waiting_rooms) is not dict:
            raise RuntimeError(
                "DECODE_ATTN A-to-F waiting-room inventory must be an exact dict"
            )
        room_exists = group_key in waiting_rooms
        room = waiting_rooms[group_key] if room_exists else None

        if room_exists:
            self._validate_decode_attn_a2f_waiting_room(
                group_key=group_key,
                room=room,
                expected_lane_contract=expected_lane_contract,
                incoming_batch=batch,
            )
            prospective_per_lane_queues = defaultdict(
                deque,
                {
                    room_lane: deque(lane_queue)
                    for room_lane, lane_queue in room["per_lane_queues"].items()
                },
            )
        else:
            prospective_per_lane_queues = defaultdict(deque)

        prospective_room = {
            "per_lane_queues": prospective_per_lane_queues,
            "expected_lane_contract": expected_lane_contract,
        }
        prospective_per_lane_queues[lane].append((layer_id, batch))
        self._validate_decode_attn_a2f_waiting_room(
            group_key=group_key,
            room=prospective_room,
            expected_lane_contract=expected_lane_contract,
        )

        prepared_idle_lanes = [
            expected_lane
            for expected_lane in expected_lane_contract
            if not prospective_per_lane_queues.get(expected_lane)
            and expected_lane in normalized_idle_expected_lanes
        ]
        barrier_is_ready = all(
            prospective_per_lane_queues.get(expected_lane)
            or expected_lane in prepared_idle_lanes
            for expected_lane in expected_lane_contract
        )
        ready_lanes = sum(
            1
            for expected_lane in expected_lane_contract
            if prospective_per_lane_queues.get(expected_lane)
            or expected_lane in prepared_idle_lanes
        )

        prepared_transfers = []
        if barrier_is_ready:
            for ready_lane in expected_lane_contract:
                lane_queue = prospective_per_lane_queues.get(ready_lane)
                if not lane_queue:
                    continue
                ready_layer_id, ready_batch = lane_queue[0]
                if ready_batch.is_idle:
                    continue
                activation_size, transfer_time = (
                    self._validate_decode_attn_a2f_predictor_result(
                        self._m2n_transfer_predictor.get_transfer_info(
                            source_cluster_type=ClusterType.DECODE_ATTN,
                            target_cluster_type=ClusterType.DECODE_FFN,
                            batch=ready_batch,
                            replica_config=replica_config,
                        )
                    )
                )
                prepared_transfers.append(
                    (
                        ready_lane,
                        ready_layer_id,
                        ready_batch,
                        activation_size,
                        transfer_time,
                    )
                )

        prepared_idle_entries = self._prepare_decode_attn_idle_lanes_for_barrier(
            time=time,
            group_key=group_key,
            idle_lanes=prepared_idle_lanes,
            is_moe=model_is_moe,
        )
        for idle_lane, idle_entry in prepared_idle_entries:
            prospective_per_lane_queues[idle_lane].append(idle_entry)
        self._validate_decode_attn_a2f_waiting_room(
            group_key=group_key,
            room=prospective_room,
            expected_lane_contract=expected_lane_contract,
        )

        picked: List[tuple[tuple[int, int], int, Batch]] = []
        prospective_after_release = defaultdict(
            deque,
            {
                ready_lane: deque(lane_queue)
                for ready_lane, lane_queue in prospective_per_lane_queues.items()
            },
        )
        if barrier_is_ready:
            for ready_lane in expected_lane_contract:
                ready_layer_id, ready_batch = prospective_after_release[
                    ready_lane
                ].popleft()
                picked.append((ready_lane, ready_layer_id, ready_batch))

        non_idle_expected_lanes = tuple(
            ready_lane
            for ready_lane, _, ready_batch in picked
            if not ready_batch.is_idle
        )
        barrier_round_id = (
            self._peek_decode_attn_barrier_round_id() if barrier_is_ready else None
        )
        transfer_plan_by_batch = {
            id(ready_batch): (activation_size, transfer_time)
            for (
                _,
                _,
                ready_batch,
                activation_size,
                transfer_time,
            ) in prepared_transfers
        }

        events = []
        prepared_cohort_updates = []
        if barrier_is_ready:
            for (source_replica_id, source_replica_local_id), ready_layer_id, ready_batch in picked:
                if ready_batch.is_idle:
                    continue
                activation_size, transfer_time = transfer_plan_by_batch[
                    id(ready_batch)
                ]
                events.append(
                    M2NTransferStartEvent(
                        time=time,
                        source_replica_id=source_replica_id,
                        source_replica_local_id=source_replica_local_id,
                        source_cluster_type=ClusterType.DECODE_ATTN,
                        target_cluster_type=ClusterType.DECODE_FFN,
                        batch=ready_batch,
                        activation_size_bytes=activation_size,
                        transfer_time_ms=transfer_time,
                        layer_id=ready_layer_id,
                        afd_stage_idx=ready_batch.afd_stage_idx,
                    )
                )
                events.append(
                    ReplicaScheduleEvent(
                        time,
                        source_replica_id,
                        self._cluster_type,
                        source_replica_local_id,
                    )
                )
                prepared_cohort_updates.append(
                    self._set_decode_attn_batch_cohort_phase(
                        ready_batch,
                        phase="ffn_inflight",
                        replica_id=source_replica_id,
                        replica_local_id=source_replica_local_id,
                        layer_id=ready_layer_id,
                        prepare_only=True,
                    )
                )

        self._commit_decode_attn_batch_cohort_phases(prepared_cohort_updates)

        if room_exists:
            committed_room = room
            committed_per_lane_queues = committed_room["per_lane_queues"]
            for queue_lane in tuple(committed_per_lane_queues):
                if queue_lane not in prospective_after_release:
                    committed_per_lane_queues[queue_lane].clear()
        else:
            committed_per_lane_queues = defaultdict(deque)
            committed_room = {
                "per_lane_queues": committed_per_lane_queues,
                "expected_lane_contract": expected_lane_contract,
            }

        for queue_lane, prepared_queue in prospective_after_release.items():
            committed_queue = committed_per_lane_queues[queue_lane]
            committed_queue.clear()
            committed_queue.extend(prepared_queue)

        if any(committed_per_lane_queues.values()):
            waiting_rooms[group_key] = committed_room
        else:
            waiting_rooms.pop(group_key, None)
        if idle_expected_lanes is not None:
            idle_expected_lanes.discard(lane)

        if barrier_is_ready:
            for ready_lane, ready_layer_id, ready_batch in picked:
                if ready_batch.is_idle:
                    logger.info(
                        f"[A2F-GROUP-RELEASE-IDLE] layer={ready_layer_id} "
                        f"afd_stage_idx={ready_batch.afd_stage_idx} slot={ready_batch.afd_stage_idx} "
                        f"lane={ready_lane}"
                    )
                    continue
                ready_batch.decode_attn_barrier_round_id = barrier_round_id
                ready_batch.decode_attn_barrier_expected_lanes = (
                    non_idle_expected_lanes
                )
            self._decode_attn_barrier_round_counter = barrier_round_id + 1

        logger.info(
            f"[A2F-GROUP-READY] layer={layer_id} afd_stage_idx={afd_stage_idx} "
            f"slot={afd_stage_idx} lane={lane} "
            f"depth={len(prospective_per_lane_queues[lane])} "
            f"ready_lanes={ready_lanes}/{len(expected_lane_contract)}"
        )
        if prepared_idle_entries:
            logger.info(
                f"[A2F-GROUP-IDLE] layer_id={layer_id} "
                f"afd_stage_idx={afd_stage_idx} "
                f"missing={sorted(prepared_idle_lanes)} "
                f"layer_hint={layer_id}"
            )
        for ready_lane, ready_layer_id, ready_batch in picked:
            if not ready_batch.is_idle:
                logger.info(
                    f"[A2F-GROUP-RELEASE] layer={ready_layer_id} afd_stage_idx={ready_batch.afd_stage_idx} "
                    f"slot={ready_batch.afd_stage_idx} lane={ready_lane} batch_id={ready_batch.id}"
                )
        return events

    def _prepare_decode_attn_batch_cohort_phase(
        self,
        batch: Batch,
        *,
        phase: str,
        replica_id: int,
        replica_local_id: int | None,
        layer_id: int | None = None,
    ) -> Optional[dict[str, Any]]:
        """Prepare a cohort phase update without mutating cohort state."""

        if self._cluster_type != ClusterType.DECODE_ATTN:
            return None

        cohort_id = getattr(batch, "decode_attn_cohort_id", None)
        if cohort_id is None:
            return None
        if type(cohort_id) is not int or cohort_id < 0:
            raise ValueError(
                "DECODE_ATTN cohort ID must be an exact non-negative int, "
                f"got {cohort_id!r}"
            )
        if type(phase) is not str or phase not in {"local_attn", "ffn_inflight"}:
            raise ValueError(f"Unsupported DECODE_ATTN cohort phase: {phase!r}")

        batch_replica_id = getattr(batch, "decode_attn_original_replica_id", None)
        batch_replica_local_id = getattr(
            batch, "decode_attn_original_replica_local_id", None
        )
        if batch_replica_id is None:
            batch_replica_id = replica_id
        if replica_local_id is not None:
            raise ValueError(
                "DECODE_ATTN cohort phase requires full-stage identity with "
                f"replica_local_id=None, got {replica_local_id!r}"
            )
        if batch_replica_local_id is not None:
            raise ValueError(
                "DECODE_ATTN cohort batch requires full-stage identity with "
                f"replica_local_id=None, got {batch_replica_local_id!r}"
            )
        batch_replica_local_id = None
        if type(batch_replica_id) is not int or batch_replica_id < 0:
            raise ValueError(
                "DECODE_ATTN cohort lane replica_id must be an exact "
                f"non-negative int, got {batch_replica_id!r}"
            )
        if type(layer_id) is not int and layer_id is not None:
            raise ValueError(
                "DECODE_ATTN cohort layer_id must be None or an exact int, "
                f"got {layer_id!r}"
            )
        if layer_id is not None and layer_id < 0:
            raise ValueError(
                "DECODE_ATTN cohort layer_id must be non-negative, "
                f"got {layer_id!r}"
            )

        replica_schedulers = getattr(self, "_replica_schedulers", None)
        if type(replica_schedulers) is not dict:
            raise RuntimeError(
                "DECODE_ATTN replica scheduler topology must be an exact dict"
            )
        lane = (batch_replica_id, batch_replica_local_id)
        if lane not in replica_schedulers:
            raise ValueError(
                "DECODE_ATTN cohort lane is absent from the replica scheduler "
                f"topology: lane={lane}"
            )
        replica_scheduler = replica_schedulers[lane]
        cohort_states = getattr(
            replica_scheduler,
            "_decode_attn_active_cohort_states",
            {},
        )
        if type(cohort_states) is not dict:
            raise RuntimeError(
                "DECODE_ATTN active cohort states must be an exact dict"
            )
        cohort_state = cohort_states.get(cohort_id)
        if cohort_state is not None and type(cohort_state) is not dict:
            raise RuntimeError(
                "DECODE_ATTN active cohort state must be an exact dict"
            )

        afd_stage_idx = getattr(batch, "afd_stage_idx", None)
        if afd_stage_idx is not None and (
            type(afd_stage_idx) is not int or afd_stage_idx < 0
        ):
            raise ValueError(
                "DECODE_ATTN cohort afd_stage_idx must be None or an exact "
                f"non-negative int, got {afd_stage_idx!r}"
            )
        if cohort_state is not None and afd_stage_idx is not None:
            active_stage_indices, _, _ = (
                self._validate_decode_attn_cohort_stage_maps(
                    cohort_state,
                    context="cohort phase update",
                )
            )
            if afd_stage_idx not in active_stage_indices:
                raise ValueError(
                    "DECODE_ATTN cohort stage is not active: "
                    f"stage={afd_stage_idx}, "
                    f"active={sorted(active_stage_indices)}"
                )

        return {
            "batch": batch,
            "cohort_state": cohort_state,
            "cohort_id": cohort_id,
            "phase": phase,
            "layer_id": layer_id,
            "afd_stage_idx": afd_stage_idx,
        }

    @staticmethod
    def _apply_decode_attn_batch_cohort_phase(
        prepared_update: Optional[dict[str, Any]],
    ) -> None:
        """Commit a previously validated cohort phase update."""

        if prepared_update is None:
            return
        cohort_state = prepared_update["cohort_state"]
        if cohort_state is None:
            return

        phase = prepared_update["phase"]
        layer_id = prepared_update["layer_id"]
        afd_stage_idx = prepared_update["afd_stage_idx"]
        if afd_stage_idx is None:
            cohort_state["af_phase"] = phase
            if layer_id is not None:
                cohort_state["current_layer_id"] = layer_id
            return

        stage_phases = cohort_state["stage_phases"]
        stage_phases[afd_stage_idx] = phase
        if layer_id is not None:
            stage_layers = cohort_state["stage_current_layer_ids"]
            stage_layers[afd_stage_idx] = layer_id
            cohort_state["current_layer_id"] = layer_id

        phases = set(stage_phases.values())
        cohort_state["af_phase"] = phases.pop() if len(phases) == 1 else "mixed"

    def _commit_decode_attn_batch_cohort_phases(
        self,
        prepared_updates: List[Optional[dict[str, Any]]],
    ) -> None:
        """Apply prepared cohort updates atomically after all preflight checks."""

        if type(prepared_updates) is not list:
            raise RuntimeError(
                "DECODE_ATTN prepared cohort updates must be an exact list"
            )

        prospective_states: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
        for prepared_update in prepared_updates:
            if prepared_update is None:
                continue
            if type(prepared_update) is not dict:
                raise RuntimeError(
                    "DECODE_ATTN prepared cohort update must be an exact dict"
                )
            cohort_state = prepared_update.get("cohort_state")
            if cohort_state is None:
                continue
            if type(cohort_state) is not dict:
                raise RuntimeError(
                    "DECODE_ATTN prepared cohort state must be an exact dict"
                )

            state_key = id(cohort_state)
            state_pair = prospective_states.get(state_key)
            if state_pair is None:
                prospective_state = deepcopy(cohort_state)
                state_pair = (cohort_state, prospective_state)
                prospective_states[state_key] = state_pair
            else:
                prospective_state = state_pair[1]

            prospective_update = dict(prepared_update)
            prospective_update["cohort_state"] = prospective_state
            self._apply_decode_attn_batch_cohort_phase(prospective_update)

        for cohort_state, prospective_state in prospective_states.values():
            cohort_state.clear()
            cohort_state.update(prospective_state)

    def _set_decode_attn_batch_cohort_phase(
        self,
        batch: Batch,
        *,
        phase: str,
        replica_id: int,
        replica_local_id: int | None,
        layer_id: int | None = None,
        prepare_only: bool = False,
    ) -> Optional[dict[str, Any]]:
        prepared_update = self._prepare_decode_attn_batch_cohort_phase(
            batch,
            phase=phase,
            replica_id=replica_id,
            replica_local_id=replica_local_id,
            layer_id=layer_id,
        )
        if prepare_only:
            return prepared_update
        self._apply_decode_attn_batch_cohort_phase(prepared_update)

    def _peek_decode_attn_barrier_round_id(self) -> int:
        """Return the next A-to-F barrier round without mutating its counter."""

        next_round_id = getattr(self, "_decode_attn_barrier_round_counter", 0)
        if type(next_round_id) is not int or next_round_id < 0:
            raise RuntimeError(
                "DECODE_ATTN A-to-F barrier round counter must be an exact "
                f"non-negative int, got {next_round_id!r}"
            )
        return next_round_id

    def _next_decode_attn_barrier_round_id(self) -> int:
        next_round_id = self._peek_decode_attn_barrier_round_id()
        self._decode_attn_barrier_round_counter = next_round_id + 1
        return next_round_id

    def _get_decode_attn_a2f_active_local_attn_lanes(
        self,
        *,
        cohort_id: int,
        request_ids: tuple[int, ...],
        afd_stage_idx: int,
        layer_id: int,
    ) -> List[tuple[int, int]]:
        """Read active local-attention lanes without invoking lazy getters."""

        if type(cohort_id) is not int or cohort_id < 0:
            raise ValueError(
                "DECODE_ATTN A-to-F cohort_id must be an exact non-negative int"
            )
        if type(request_ids) is not tuple:
            raise ValueError(
                "DECODE_ATTN A-to-F cohort request IDs must be an exact tuple"
            )
        for request_id in request_ids:
            if type(request_id) is not int or request_id < 0:
                raise ValueError(
                    "DECODE_ATTN A-to-F cohort request IDs must contain exact "
                    f"non-negative ints, got {request_id!r}"
                )
        afd_stage_idx = self._validate_decode_attn_a2f_topology_value(
            afd_stage_idx,
            field_name="active local-attn afd_stage_idx",
        )
        layer_id = self._validate_decode_attn_a2f_topology_value(
            layer_id,
            field_name="active local-attn layer_id",
        )

        replica_schedulers = getattr(self, "_replica_schedulers", None)
        if type(replica_schedulers) is not dict:
            raise RuntimeError(
                "DECODE_ATTN A-to-F replica scheduler topology must be an exact dict"
            )
        scheduler_lanes = self._normalize_m2n_lane_contract(
            list(replica_schedulers),
            field_name="DECODE_ATTN A-to-F active lane topology",
            require_nonempty=False,
        )
        active_lanes: List[tuple[int, int]] = []
        requested_ids = set(request_ids)
        for lane in scheduler_lanes:
            replica_scheduler = replica_schedulers[lane]
            cohort_states = getattr(
                replica_scheduler,
                "_decode_attn_active_cohort_states",
                None,
            )
            if type(cohort_states) is not dict:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F active cohort states must be an exact dict"
                )
            cohort_state = cohort_states.get(cohort_id)
            if cohort_state is None:
                continue
            if type(cohort_state) is not dict:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F active cohort state must be an exact dict"
                )
            all_request_ids = cohort_state.get("all_request_ids")
            pending_request_ids = cohort_state.get("pending_request_ids")
            if type(all_request_ids) is not set or type(pending_request_ids) is not set:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F cohort request registries must be exact sets"
                )
            for registry_name, registry in (
                ("all_request_ids", all_request_ids),
                ("pending_request_ids", pending_request_ids),
            ):
                for request_id in registry:
                    if type(request_id) is not int or request_id < 0:
                        raise RuntimeError(
                            f"DECODE_ATTN A-to-F cohort {registry_name} must contain "
                            f"exact non-negative ints, got {request_id!r}"
                        )
            if not pending_request_ids or not requested_ids.issubset(all_request_ids):
                continue
            active_stage_indices, stage_phases, stage_layers = (
                self._validate_decode_attn_cohort_stage_maps(
                    cohort_state,
                    context="A-to-F active local-attn lane",
                )
            )
            if afd_stage_idx not in active_stage_indices:
                continue

            aggregate_phase = cohort_state.get("af_phase")
            aggregate_layer = cohort_state.get("current_layer_id")
            if type(aggregate_phase) is not str:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F cohort af_phase must be an exact str"
                )
            if type(aggregate_layer) is not int or aggregate_layer < 0:
                raise RuntimeError(
                    "DECODE_ATTN A-to-F cohort current_layer_id must be an exact "
                    "non-negative int"
                )
            stage_phase = stage_phases[afd_stage_idx]
            stage_layer = stage_layers[afd_stage_idx]
            if stage_phase == "local_attn" and stage_layer == layer_id:
                active_lanes.append(lane)

        return active_lanes

    def _get_decode_attn_a2f_expected_lanes(
        self,
        afd_stage_idx: int | None = None,
        *,
        layer_id: int | None = None,
    ) -> List[tuple[int, int]]:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            raise ValueError(
                "_get_decode_attn_a2f_expected_lanes is only valid for DECODE_ATTN cluster"
            )

        if afd_stage_idx is not None:
            afd_stage_idx = self._validate_decode_attn_a2f_topology_value(
                afd_stage_idx,
                field_name="expected-lane afd_stage_idx",
            )
        if layer_id is not None:
            layer_id = self._validate_decode_attn_a2f_topology_value(
                layer_id,
                field_name="expected-lane layer_id",
            )

        if afd_stage_idx is not None:
            stage_slot_lanes = self._get_decode_attn_stage_slot_active_lanes(
                afd_stage_idx,
                phase="local_attn",
                layer_id=layer_id,
            )
            if stage_slot_lanes:
                return self._normalize_m2n_lane_contract(
                    stage_slot_lanes,
                    field_name="DECODE_ATTN A-to-F active stage lane topology",
                    require_nonempty=True,
                )

        active_wave_request_ids_by_lane = getattr(
            self,
            "_decode_attn_active_serving_wave_request_ids_by_lane",
            {},
        )
        active_wave_expected_lanes = getattr(
            self,
            "_decode_attn_active_serving_wave_expected_lanes",
            (),
        )
        normalized_active_wave_lanes = self._normalize_m2n_lane_contract(
            active_wave_expected_lanes,
            field_name="DECODE_ATTN A-to-F active wave lane topology",
            require_nonempty=False,
        )
        if type(active_wave_request_ids_by_lane) is not dict:
            raise RuntimeError(
                "DECODE_ATTN A-to-F active wave request topology must be an "
                "exact dict"
            )
        normalized_active_wave_request_lanes = self._normalize_m2n_lane_contract(
            list(active_wave_request_ids_by_lane),
            field_name="DECODE_ATTN A-to-F active wave request lane topology",
            require_nonempty=False,
        )
        if normalized_active_wave_lanes:
            return normalized_active_wave_lanes
        if normalized_active_wave_request_lanes:
            return sorted(normalized_active_wave_request_lanes)

        configured_lanes = getattr(self, "_a2f_expected_lanes", None)
        if configured_lanes is not None:
            return self._normalize_m2n_lane_contract(
                configured_lanes,
                field_name="DECODE_ATTN A-to-F scheduler lane topology",
                require_nonempty=False,
            )

        return []

    def _get_decode_attn_stage_slot_active_lanes(
        self,
        afd_stage_idx: int,
        *,
        replica_id: int | None = None,
        phase: str | None = None,
        layer_id: int | None = None,
    ) -> List[tuple[int, int]]:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            raise ValueError(
                "_get_decode_attn_stage_slot_active_lanes is only valid for DECODE_ATTN cluster"
            )
        if type(afd_stage_idx) is not int or afd_stage_idx < 0:
            raise ValueError(
                "DECODE_ATTN active stage slot must be an exact non-negative int, "
                f"got {afd_stage_idx!r}"
            )
        if replica_id is not None and (
            type(replica_id) is not int or replica_id < 0
        ):
            raise ValueError(
                "DECODE_ATTN active stage replica_id must be an exact non-negative "
                f"int, got {replica_id!r}"
            )

        active_lanes: List[tuple[int, int]] = []
        replica_schedulers = getattr(self, "_replica_schedulers", {})
        if type(replica_schedulers) is not dict:
            raise RuntimeError(
                "DECODE_ATTN replica scheduler topology must be an exact dict"
            )
        scheduler_lanes = self._normalize_m2n_lane_contract(
            list(replica_schedulers.keys()),
            field_name="DECODE_ATTN replica scheduler lane topology",
            require_nonempty=False,
        )
        for lane_replica_id, lane_replica_local_id in scheduler_lanes:
            if replica_id is not None and lane_replica_id != replica_id:
                continue
            replica_scheduler = replica_schedulers[
                (lane_replica_id, lane_replica_local_id)
            ]

            get_active_stage_slots = getattr(
                replica_scheduler,
                "get_decode_attn_active_stage_slots",
                None,
            )
            if callable(get_active_stage_slots):
                raw_active_stage_slots = get_active_stage_slots(
                    phase=phase,
                    layer_id=layer_id,
                )
            else:
                cohort_states = getattr(
                    replica_scheduler,
                    "_decode_attn_active_cohort_states",
                    {},
                )
                if type(cohort_states) is not dict:
                    raise RuntimeError(
                        "DECODE_ATTN active cohort states must be an exact dict"
                    )
                raw_active_stage_slots = []
                for state in cohort_states.values():
                    if type(state) is not dict:
                        raise RuntimeError(
                            "DECODE_ATTN active cohort state must be an exact dict"
                        )
                    if "afd_stage_idx" not in state:
                        continue
                    if phase is not None and state.get("af_phase") != phase:
                        continue
                    state_layer_id = state.get("current_layer_id")
                    if layer_id is not None and state_layer_id is not None:
                        if type(state_layer_id) is not int or state_layer_id < 0:
                            raise RuntimeError(
                                "DECODE_ATTN cohort current_layer_id must be an exact "
                                f"non-negative int, got {state_layer_id!r}"
                            )
                        if state_layer_id != layer_id:
                            continue
                    raw_active_stage_slots.append(state["afd_stage_idx"])

            if type(raw_active_stage_slots) not in {list, tuple, set}:
                raise RuntimeError(
                    "DECODE_ATTN active stage slots must be an exact list, tuple, "
                    f"or set, got {raw_active_stage_slots!r}"
                )
            active_stage_slots = set()
            for active_stage_idx in raw_active_stage_slots:
                if type(active_stage_idx) is not int or active_stage_idx < 0:
                    raise RuntimeError(
                        "DECODE_ATTN active stage slot must be an exact non-negative "
                        f"int, got {active_stage_idx!r}"
                    )
                active_stage_slots.add(active_stage_idx)

            if afd_stage_idx in active_stage_slots:
                active_lanes.append((lane_replica_id, lane_replica_local_id))

        return active_lanes

    def _get_decode_attn_f2a_expected_lanes(
        self,
        replica_id: int,
        *,
        afd_stage_idx: int | None = None,
    ) -> List[tuple[int, int]]:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            raise ValueError(
                "_get_decode_attn_f2a_expected_lanes is only valid for DECODE_ATTN cluster"
            )
        if type(replica_id) is not int or replica_id < 0:
            raise ValueError(
                "DECODE_ATTN F-to-A replica_id must be an exact non-negative int, "
                f"got {replica_id!r}"
            )
        if afd_stage_idx is not None and (
            type(afd_stage_idx) is not int or afd_stage_idx < 0
        ):
            raise ValueError(
                "DECODE_ATTN F-to-A afd_stage_idx must be an exact non-negative int, "
                f"got {afd_stage_idx!r}"
            )

        cluster = getattr(self, "_cluster", None)
        cluster_replicas = getattr(cluster, "replicas", None)
        if type(cluster_replicas) is not dict:
            raise RuntimeError(
                "DECODE_ATTN replica inventory must be an exact dict"
            )
        for inventory_replica_id in cluster_replicas:
            if type(inventory_replica_id) is not int or inventory_replica_id < 0:
                raise RuntimeError(
                    "DECODE_ATTN replica inventory IDs must be exact non-negative "
                    f"ints, got {inventory_replica_id!r}"
                )
        if replica_id not in cluster_replicas:
            raise ValueError(
                "DECODE_ATTN F-to-A replica is outside the cluster replica "
                f"inventory: replica_id={replica_id}, "
                f"replica_ids={list(cluster_replicas.keys())}"
            )

        raw_idle_expected_lanes = getattr(
            self,
            "_decode_attn_idle_expected_lanes",
            set(),
        )
        if type(raw_idle_expected_lanes) not in {list, tuple, set}:
            raise RuntimeError(
                "DECODE_ATTN idle lane topology must be an exact list, tuple, or set"
            )
        idle_expected_lanes = set(
            self._normalize_m2n_lane_contract(
                list(raw_idle_expected_lanes),
                field_name="DECODE_ATTN idle lane topology",
                require_nonempty=False,
            )
        )
        if afd_stage_idx is not None:
            stage_slot_lanes = self._get_decode_attn_stage_slot_active_lanes(
                afd_stage_idx,
                replica_id=replica_id,
                phase="ffn_inflight",
            )
            filtered_stage_slot_lanes = [
                lane
                for lane in stage_slot_lanes
                if lane not in idle_expected_lanes
            ]
            if filtered_stage_slot_lanes:
                return filtered_stage_slot_lanes

        active_wave_request_ids_by_lane = getattr(
            self,
            "_decode_attn_active_serving_wave_request_ids_by_lane",
            {},
        )
        active_wave_expected_lanes = getattr(
            self,
            "_decode_attn_active_serving_wave_expected_lanes",
            (),
        )
        normalized_active_wave_lanes = self._normalize_m2n_lane_contract(
            active_wave_expected_lanes,
            field_name="DECODE_ATTN active wave lane topology",
            require_nonempty=False,
        )
        if type(active_wave_request_ids_by_lane) is not dict:
            raise RuntimeError(
                "DECODE_ATTN active wave request topology must be an exact dict"
            )
        normalized_active_wave_request_lanes = self._normalize_m2n_lane_contract(
            list(active_wave_request_ids_by_lane.keys()),
            field_name="DECODE_ATTN active wave request lane topology",
            require_nonempty=False,
        )
        if normalized_active_wave_lanes:
            return [
                lane
                for lane in normalized_active_wave_lanes
                if lane[0] == replica_id and lane not in idle_expected_lanes
            ]
        if normalized_active_wave_request_lanes:
            return [
                lane
                for lane in sorted(normalized_active_wave_request_lanes)
                if lane[0] == replica_id and lane not in idle_expected_lanes
            ]

        configured_lanes = getattr(self, "_f2a_expected_lanes", None)
        if configured_lanes is not None:
            normalized_configured_lanes = self._normalize_m2n_lane_contract(
                configured_lanes,
                field_name="DECODE_ATTN F-to-A scheduler lane topology",
                require_nonempty=False,
            )
            if normalized_configured_lanes:
                return [
                    lane
                    for lane in normalized_configured_lanes
                    if lane[0] == replica_id and lane not in idle_expected_lanes
                ]

        if (
            type(self._replica_scheduler_count) is not int
            or self._replica_scheduler_count <= 0
        ):
            raise RuntimeError(
                "DECODE_ATTN replica scheduler count must be an exact positive int, "
                f"got {self._replica_scheduler_count!r}"
            )
        full_stage_lane = (replica_id, None)
        return [] if full_stage_lane in idle_expected_lanes else [full_stage_lane]

    def _release_decode_attn_ready_return_round(
        self,
        round_key: tuple,
        expected_lanes: List[tuple[int, int]],
        logger,
    ) -> List[Batch]:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            raise ValueError(
                "_release_decode_attn_ready_return_round is only valid for DECODE_ATTN cluster"
            )

        room = self._f2a_waiting_by_round.get(round_key)
        if room is None:
            return []

        replica_id, next_layer_id, afd_stage_idx = round_key[:3]
        per_lane_batches = room["per_lane_queues"]
        if not all(per_lane_batches.get(lane) for lane in expected_lanes):
            return []

        released_batches = [
            per_lane_batches[lane].popleft() for lane in expected_lanes
        ]
        if all(not lane_queue for lane_queue in per_lane_batches.values()):
            self._f2a_waiting_by_round.pop(round_key, None)

        logger.info(
            f"[F2A-GROUP-RELEASE] replica={replica_id} next_layer={next_layer_id} "
            f"afd_stage_idx={afd_stage_idx} lanes={len(expected_lanes)}"
        )
        return released_batches

    def _enqueue_decode_attn_return_round(
        self,
        micro_batch: Batch,
        *,
        receipt: Dict[str, Any],
        logger,
    ) -> bool:
        if self._cluster_type != ClusterType.DECODE_ATTN:
            raise ValueError(
                "_enqueue_decode_attn_return_round is only valid for DECODE_ATTN cluster"
            )

        replica_id = receipt["replica_id"]
        lane = receipt["lane"]
        batch_global_id = receipt["batch_global_id"]
        decode_token_index = receipt["decode_token_index"]
        next_layer_id = receipt["next_layer_id"]
        afd_stage_idx = receipt["afd_stage_idx"]
        round_key = receipt["round_key"]
        stored_expected_lanes = receipt["stored_expected_lanes"]
        expected_lanes = receipt["expected_lanes"]
        room = receipt["room"]
        if room is None:
            room = {
                "per_lane_queues": defaultdict(deque),
                "expected_lanes": stored_expected_lanes,
            }
            self._f2a_waiting_by_round[round_key] = room
        elif room["expected_lanes"] is None and stored_expected_lanes is not None:
            room["expected_lanes"] = stored_expected_lanes

        room["per_lane_queues"][lane].append(micro_batch)
        ready_lanes = sum(
            1 for expected_lane in expected_lanes
            if room["per_lane_queues"].get(expected_lane)
        )

        logger.info(
            f"[F2A-GROUP-READY] replica={replica_id} global_id={batch_global_id} "
            f"token_idx={decode_token_index} next_layer={next_layer_id} "
            f"afd_stage_idx={afd_stage_idx} lane={lane} "
            f"depth={len(room['per_lane_queues'][lane])} "
            f"ready_lanes={ready_lanes}/{len(expected_lanes)}"
        )

        released_batches = self._release_decode_attn_ready_return_round(
            round_key,
            expected_lanes,
            logger,
        )
        enqueued_batches = 0
        for ready_batch in released_batches:
            self._set_decode_attn_batch_cohort_phase(
                ready_batch,
                phase="local_attn",
                replica_id=int(ready_batch.decode_attn_original_replica_id),
                replica_local_id=ready_batch.decode_attn_original_replica_local_id,
                layer_id=int(ready_batch.af_inflight_layer_count),
            )
            if getattr(
                ready_batch,
                "trace_replay_initial_hydration_moe_head_consumed",
                False,
            ):
                self.get_replica_scheduler(
                    int(ready_batch.decode_attn_original_replica_id),
                    ready_batch.decode_attn_original_replica_local_id,
                ).on_batch_end(ready_batch)
                logger.info(
                    "[AF-ARRIVAL][DROP] mb=%s global_id=%s dropped after synthetic "
                    "trace-replay hydration head completed its first MoE consume",
                    ready_batch.id,
                    ready_batch.global_id,
                )
                continue
            self._af_batch_queue.append(ready_batch)
            enqueued_batches += 1
            logger.info(
                f"[AF-ARRIVAL][ENQUEUE] mb={ready_batch.id} global_id={ready_batch.global_id} "
                f"re-enqueued to AF priority queue after F→A round barrier; "
                f"af_queue_size={len(self._af_batch_queue)}"
            )

        return enqueued_batches > 0

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
