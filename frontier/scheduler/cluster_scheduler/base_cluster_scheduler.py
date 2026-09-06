from abc import ABC, abstractmethod
from collections import defaultdict, deque
from copy import deepcopy
import logging
import math
from numbers import Real

from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING

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
    EPLaneWorkload,
    LayerEPWorkload,
    build_contiguous_expert_ownership,
    materialize_layer_ep_workload,
    resolve_ep_lane_workload,
    resolve_routing_details,
)
from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
from frontier.scheduler.utils import ep_trace
from frontier.scheduler.utils.expert_parallel import (
    EPBatchGroupPlan,
    materialize_batch_group,
    materialize_wave_workload,
    prepare_batch_group_plan,
    validate_token_conservation,
    validate_collective_exec_time,
    validate_barrier_arrival,
    summarize_alltoall_payload,
    prepare_combine_timing,
    resolve_ep_execution_time,
    validate_completion_time,
    resolve_source_batch_ids,
    get_ep_phase_times_ms,
)
from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs
from frontier.scheduler.utils.ep_wave import prepare_moe_wave_from_inputs
from frontier.scheduler.utils.scheduler_diagnostics import (
    SchedulerDiagnostics,
    format_ep_trace_identity,
)
from frontier.scheduler.utils.pdaf_transfer import (
    LaneIdentityScope,
    normalize_lanes,
    validate_decode_ffn_receipt,
    validate_decode_ffn_waiting_room,
    validate_decode_attn_wave_binding,
    validate_decode_attn_receipt,
    validate_decode_attn_queued_batch,
    prepare_dp_padding,
    prepare_decode_attn_idle_lanes,
    validate_decode_attn_a2f_batch_entry,
    validate_a2f_predictor_result,
)
from frontier.scheduler.utils.pdaf_validation import (
    validate_decode_attn_a2f_waiting_room,
)
from frontier.scheduler.utils.pdaf_entries import build_decode_ffn_idle_entries
from frontier.scheduler.utils.pdaf_a2f import prepare_a2f_admission
from frontier.scheduler.utils.pdaf_a2f_ready import schedule_decode_attn_a2f_ready
from frontier.scheduler.utils.pdaf_dense_a2f import release_dense_a2f
from frontier.scheduler.utils.ep_combine import prepare_ep_combine_completion
from frontier.scheduler.utils.ep_dispatch import handle_dispatch_ready, prepare_dispatch_advance
from frontier.scheduler.utils.ep_combine_schedule import schedule_combine_completion
from frontier.scheduler.utils.m2n_grouping import prepare_ffn_group_promotion
from frontier.scheduler.utils.m2n_state import M2NTransferState
from frontier.scheduler.utils.attention_transfer_state import AttentionTransferState
from frontier.scheduler.utils.kv_arrival import (
    handle_decode_arrival,
    handle_decode_attn_arrival,
)
from frontier.scheduler.utils.m2n_arrival import (
    route_m2n_arrival,
    handle_decode_attn_arrival,
)
from frontier.scheduler.utils.sync_entry import enter_decode_sync, enter_prefill_sync
from frontier.scheduler.utils.pdaf_phase import (
    prepare_decode_attn_batch_phase,
    apply_decode_attn_batch_phase,
    commit_decode_attn_batch_phases,
    set_decode_attn_batch_phase,
)
from frontier.scheduler.utils.pdaf_attention import (
    get_a2f_active_local_attn_lanes,
    get_stage_slot_active_lanes,
    get_a2f_expected_lanes,
    get_f2a_expected_lanes,
)
from frontier.scheduler.utils.collective_timing import (
    attention_delay_seconds,
    prepare_decode_final_timing,
    prepare_prefill_final_timing,
    select_active_batch,
    validate_decode_layer_advance,
)
from frontier.scheduler.utils.prefill_collective import handle_prefill_sync_collective
from frontier.scheduler.utils.decode_collective import handle_decode_sync_collective
from frontier.scheduler.utils.execution_time_metrics import (
    build_single_layer_metrics_execution_time,
)
from frontier.scheduler.utils.afd_metadata import aggregate_afd_metadata
from frontier.scheduler.utils.request_selection import collect_active_requests
from frontier.scheduler.utils.replica_schedulers import build_replica_scheduler_maps
from frontier.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from frontier.scheduler.utils.layer_path import uses_shared_layer_path
from frontier.scheduler.utils.replica_config import resolve_replica_scheduler_config
from frontier.scheduler.utils.stage_contexts import build_stage_execution_contexts
from frontier.scheduler.utils.prefix_cache import validate_prefix_cache_config
from frontier.scheduler.utils.dense_metrics import first_dense_layer_id, predict_dense_reference
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    EP_WAVE,
    FULL_STAGE_WORLD,
    StageAdmissionTicket,
    StageExecutionContext,
)
from frontier.types import (
    ClusterType,
    ReplicaSchedulerType,
)


logger = logging.getLogger(__name__)

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
    if not profile.uses_expert_parallel_alltoall(cluster_type, expected_ep_size):
        raise ValueError(
            f"Model architecture profile {profile.profile_id} does not support "
            f"EP collectives for {cluster_type.name}"
        )
    return profile.expert_parallel_collective


M2NLaneIdentityScope = LaneIdentityScope


class BaseClusterScheduler(ABC):
    def _get_attention_transfer_state(self) -> AttentionTransferState:
        state = getattr(self, "_attention_transfer_state", None)
        if state is None:
            state = AttentionTransferState()
            self._attention_transfer_state = state
        return state

    @property
    def _a2f_waiting_by_layer(self):
        return self._get_attention_transfer_state().a2f_waiting_by_layer

    @_a2f_waiting_by_layer.setter
    def _a2f_waiting_by_layer(self, value):
        self._get_attention_transfer_state().a2f_waiting_by_layer = value

    @property
    def _f2a_waiting_by_round(self):
        return self._get_attention_transfer_state().f2a_waiting_by_round

    @_f2a_waiting_by_round.setter
    def _f2a_waiting_by_round(self, value):
        self._get_attention_transfer_state().f2a_waiting_by_round = value

    @property
    def _decode_attn_idle_expected_lanes(self):
        return self._get_attention_transfer_state().idle_expected_lanes

    @_decode_attn_idle_expected_lanes.setter
    def _decode_attn_idle_expected_lanes(self, value):
        self._get_attention_transfer_state().idle_expected_lanes = value

    @property
    def _decode_attn_barrier_round_counter(self):
        return self._get_attention_transfer_state().barrier_round_counter

    @_decode_attn_barrier_round_counter.setter
    def _decode_attn_barrier_round_counter(self, value):
        self._get_attention_transfer_state().barrier_round_counter = value

    def _get_m2n_state(self) -> M2NTransferState:
        state = getattr(self, "_m2n_state", None)
        if state is None:
            state = M2NTransferState()
            self._m2n_state = state
        return state

    @property
    def _m2n_waiting_by_layer(self):
        return self._get_m2n_state().waiting_by_layer

    @_m2n_waiting_by_layer.setter
    def _m2n_waiting_by_layer(self, value):
        self._get_m2n_state().waiting_by_layer = value

    @property
    def _m2n_ready_groups(self):
        return self._get_m2n_state().ready_groups

    @_m2n_ready_groups.setter
    def _m2n_ready_groups(self, value):
        self._get_m2n_state().ready_groups = value

    @property
    def _raw_batch_waiting_for_m2n_back(self):
        return self._get_m2n_state().raw_batches

    @_raw_batch_waiting_for_m2n_back.setter
    def _raw_batch_waiting_for_m2n_back(self, value):
        self._get_m2n_state().raw_batches = value

    @staticmethod
    def get_pipeline_stage_layer_bounds(
        stage_id: int,
        num_layers_per_pipeline_stage: int,
    ) -> tuple[int, int]:
        """Return the global half-open layer range owned by one PP stage."""

        if type(stage_id) is not int or stage_id < 0:
            raise ValueError(
                "pipeline stage_id must be an exact non-negative int, "
                f"got {stage_id!r}"
            )
        if (
            type(num_layers_per_pipeline_stage) is not int
            or num_layers_per_pipeline_stage <= 0
        ):
            raise ValueError(
                "num_layers_per_pipeline_stage must be an exact positive int, "
                f"got {num_layers_per_pipeline_stage!r}"
            )
        first_layer_id = stage_id * num_layers_per_pipeline_stage
        return first_layer_id, first_layer_id + num_layers_per_pipeline_stage

    @staticmethod
    def _aggregate_decode_ffn_afd_metadata(
        source_batches: List[Batch] | Tuple[Batch, ...],
    ) -> tuple[Any | None, bool]:
        """Aggregate AFD metadata through the shared pure helper."""
        return aggregate_afd_metadata(source_batches)

    @staticmethod
    def _resolve_ep_trace_identity(
        ep_batches: Dict[int, Any],
        batch_global_id: int,
    ) -> tuple[int, int]:
        """Resolve one logical batch/layer identity for a completed EP wave."""

        return ep_trace.resolve_trace_identity(ep_batches, batch_global_id)

    @staticmethod
    def _build_ep_trace_identity(
        *,
        batch: Any,
        replica_id: int,
        stage_id: int,
        operation_id: int,
        operation_kind: str,
        afd_stage_idx: int | None = None,
    ) -> dict[str, Any]:
        """Build the structured identity attached to every EP trace record."""

        return ep_trace.build_trace_identity(
            batch=batch,
            replica_id=replica_id,
            stage_id=stage_id,
            operation_id=operation_id,
            operation_kind=operation_kind,
            afd_stage_idx=afd_stage_idx,
        )

    @staticmethod
    def _format_ep_trace_identity(identity: Dict[str, Any]) -> str:
        """Serialize an already validated identity in a stable log order."""

        return format_ep_trace_identity(identity)

    @staticmethod
    def _log_ep_workload_trace(
        *,
        cluster_type: ClusterType,
        batch_id: int,
        layer_id: int,
        lane_workload: EPLaneWorkload,
        lane_compute_ms: float,
        routed_compute_ms: float,
        lane_comm_ms: float,
        pre_dispatch_ms: float,
        dispatch_ms: float,
        combine_ms: float,
        post_combine_ms: float,
        trace_identity: Dict[str, Any],
    ) -> None:
        """Emit one source-level record for a materialized EP participant."""

        ep_trace.log_workload_trace(
            cluster_type=cluster_type,
            batch_id=batch_id,
            layer_id=layer_id,
            lane_workload=lane_workload,
            lane_compute_ms=lane_compute_ms,
            routed_compute_ms=routed_compute_ms,
            lane_comm_ms=lane_comm_ms,
            pre_dispatch_ms=pre_dispatch_ms,
            dispatch_ms=dispatch_ms,
            combine_ms=combine_ms,
            post_combine_ms=post_combine_ms,
            trace_identity=trace_identity,
            format_identity=BaseClusterScheduler._format_ep_trace_identity,
        )

    @staticmethod
    def _log_ep_wave_end_trace(
        *,
        cluster_type: ClusterType,
        batch_id: int,
        layer_id: int,
        wave_start_time_s: float,
        combine_barrier_end_time_s: float,
        post_combine_time_ms: float,
        wave_end_time_s: float,
        trace_identity: Dict[str, Any],
    ) -> None:
        """Emit the final post-combine end of one EP wave."""
        ep_trace.log_wave_end_trace(
            cluster_type=cluster_type,
            batch_id=batch_id,
            layer_id=layer_id,
            wave_start_time_s=wave_start_time_s,
            combine_barrier_end_time_s=combine_barrier_end_time_s,
            post_combine_time_ms=post_combine_time_ms,
            wave_end_time_s=wave_end_time_s,
            trace_identity=trace_identity,
            format_identity=BaseClusterScheduler._format_ep_trace_identity,
        )

    @staticmethod
    def _log_ep_barrier_trace(
        *,
        cluster_type: ClusterType,
        batch_id: int,
        layer_id: int,
        phase: str,
        expected_ep_ids: Tuple[int, ...],
        arrived_ep_ids: Tuple[int, ...],
        max_lane_time_ms: float,
        collective_time_ms: float,
        barrier_time_ms: float,
        barrier_start_time_s: float,
        barrier_end_time_s: float,
        trace_identity: Dict[str, Any],
    ) -> None:
        """Emit the completed per-layer EP barrier without changing timing."""
        ep_trace.log_barrier_trace(
            cluster_type=cluster_type,
            batch_id=batch_id,
            layer_id=layer_id,
            phase=phase,
            expected_ep_ids=expected_ep_ids,
            arrived_ep_ids=arrived_ep_ids,
            max_lane_time_ms=max_lane_time_ms,
            collective_time_ms=collective_time_ms,
            barrier_time_ms=barrier_time_ms,
            barrier_start_time_s=barrier_start_time_s,
            barrier_end_time_s=barrier_end_time_s,
            trace_identity=trace_identity,
            format_identity=BaseClusterScheduler._format_ep_trace_identity,
        )

    @staticmethod
    def _log_ep_conservation_trace(
        *,
        cluster_type: ClusterType,
        batch_id: int,
        layer_id: int,
        routing_token_count: int,
        router_topk: int,
        total_routed_assignments: int,
        per_ep_routed_tokens: Dict[int, int],
        trace_identity: Dict[str, Any],
    ) -> None:
        """Emit exact routing-to-EP token conservation for one layer wave."""
        ep_trace.log_conservation_trace(
            cluster_type=cluster_type,
            batch_id=batch_id,
            layer_id=layer_id,
            routing_token_count=routing_token_count,
            router_topk=router_topk,
            total_routed_assignments=total_routed_assignments,
            per_ep_routed_tokens=per_ep_routed_tokens,
            trace_identity=trace_identity,
            format_identity=BaseClusterScheduler._format_ep_trace_identity,
        )

    @staticmethod
    def _get_shared_ep_phase_times_ms(
        execution_time,
        *,
        cluster_type: ClusterType,
        batch_id: int,
        layer_id: int,
        ep_id: int,
    ) -> tuple[float, float, float, float, float]:
        """Read shared EP phase timings through the EP utility."""
        return get_ep_phase_times_ms(
            execution_time,
            cluster_type=cluster_type,
            batch_id=batch_id,
            layer_id=layer_id,
            ep_id=ep_id,
        )

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
        """Compatibility wrapper for prefix-cache configuration validation."""

        validate_prefix_cache_config(
            replica_scheduler_config=replica_scheduler_config,
            cluster_type=self._cluster_type,
            num_replicas=self._num_replicas,
            cluster_scheduler_config=self._config.cluster_scheduler_config,
            request_generator_config=getattr(self, "_request_generator_config", None),
        )

    def _get_cluster_specific_replica_scheduler_config(self, config: ClusterConfig, cluster_type: ClusterType):
        """Compatibility wrapper for cluster-local scheduler config resolution."""

        return resolve_replica_scheduler_config(config, cluster_type)

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
        # Non-FFN Replicas own one logical scheduler per attention-DP lane.
        # TP ranks remain abstracted inside each logical execution unit.
        if self._cluster_type == ClusterType.DECODE_FFN:
            self._replica_ep_size = int(
                self._config.replica_config.moe_expert_parallel_size
            )
            self._replica_scheduler_count = self._replica_ep_size
        elif self._cluster_type == ClusterType.DECODE_ATTN:
            attn_dp = getattr(self._config.replica_config, "attn_dp", None)
            if type(attn_dp) is not int or attn_dp != 1:
                raise ValueError(
                    "DECODE_ATTN requires attn_dp=1, "
                    f"got {attn_dp!r}"
                )
            self._replica_dp_size = 1
            self._replica_ep_size = None
            self._replica_scheduler_count = 1
        else:
            attn_dp = getattr(self._config.replica_config, "attn_dp", None)
            if type(attn_dp) is not int or attn_dp <= 0:
                raise ValueError(f"{self._cluster_type.name} requires positive attn_dp, got {attn_dp!r}")
            self._replica_dp_size = attn_dp
            self._replica_ep_size = None
            self._replica_scheduler_count = attn_dp
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
        self._replica_scheduler_type = cluster_specific_config.get_type()
        if type(self._replica_scheduler_type) is not ReplicaSchedulerType:
            raise TypeError(
                "Cluster replica scheduler type must be an exact "
                f"ReplicaSchedulerType, got {self._replica_scheduler_type!r}"
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
            self._replica_ep_size = self._config.replica_config.moe_expert_parallel_size

        self._replica_schedulers, self._full_stage_replica_schedulers = (
            build_replica_scheduler_maps(
                cluster=self._cluster,
                cluster_type=self._cluster_type,
                scheduler_type=cluster_specific_config.get_type(),
                replica_config=self._config.replica_config,
                scheduler_config=cluster_specific_config,
                request_generator_config=request_generator_config,
                predictor=self._predictor,
                af_pipeline_num_micro_batch=getattr(
                    self._config, "af_pipeline_num_micro_batch", -1
                ),
                cluster_scheduler=self,
                dp_size=getattr(self, "_replica_dp_size", None),
                ep_size=getattr(self, "_replica_ep_size", None),
                registry=ReplicaSchedulerRegistry,
            )
        )
        self._request_queue = []
        # Sync completion is tracked per concrete batch event.  A cohort ID is
        # a reusable lane-local hint, so it cannot by itself identify a
        # duplicate after an idle-placeholder wave has closed.
        self._forward_sync_state = ForwardSyncState()
        self._bind_forward_sync_state_views()

        # Initialize specialized queues for PD+AF disaggregation
        if self._cluster_type == ClusterType.DECODE_ATTN:
            self._attention_transfer_state = AttentionTransferState()
            # Queue for receiving requests from decode-ffn cluster (A→F communication)
            self._af_batch_queue = []
            # A→F waiting room is scoped to one concrete decode-attn wave.
            # key=(wire_layer_id, afd_stage_idx) -> {per_lane_queues}
            self._a2f_expected_lanes = [
                (replica_id, None)
                for replica_id in list(self._cluster.replicas.keys())
            ]
            self._a2f_group_micro_batches = len(self._a2f_expected_lanes)
            # F→A waiting room keeps per-lane FIFO semantics scoped by next_layer
            self._f2a_expected_lanes = list(self._a2f_expected_lanes)
            self._f2a_group_micro_batches = len(self._f2a_expected_lanes)
        elif self._cluster_type == ClusterType.DECODE_FFN:
            from frontier.scheduler.utils.ffn_state import initialize_decode_ffn_state

            initialize_decode_ffn_state(self, logger)
        elif self._cluster_type in (
            ClusterType.PREFILL,
            ClusterType.MONOLITHIC,
            ClusterType.DECODE,
        ):
            from frontier.scheduler.utils.sync_state import initialize_sync_waiting_rooms

            initialize_sync_waiting_rooms(self)

        # Phase 2.5: Removed deprecated _moe_waiting_room (old MoE synchronization)
        # Current architecture uses EP-based synchronization instead

        # Store raw batches by id for O(1) retrieval during F→A return path
        self._get_m2n_state().raw_batches = {}

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
        """Compatibility wrapper for stage admission context construction."""

        replica_config = getattr(self._config, "replica_config", None)
        return build_stage_execution_contexts(
            cluster=self._cluster,
            cluster_type=self._cluster_type,
            replica_config=replica_config,
            replica_dp_size=getattr(
                self,
                "_replica_dp_size",
                getattr(replica_config, "attn_dp", 1) or 1,
            ),
        )


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

    def get_waiting_replica_stage_schedule_events(
        self,
        *,
        time: float,
        replica_id: int,
        stage_id: int,
        exclude_replica_local_id: int | None,
    ) -> list:
        """Wake queued sibling lanes after a shared stage owner releases.

        Attention-DP child schedulers intentionally share one Replica/stage
        admission context in the current DES model. A schedule event that
        observes the context busy cannot retry after returning, so the release
        boundary explicitly wakes every sibling lane that still owns queued
        work. EP child lanes share one pre-attached wave ticket and remain
        eligible for the same wakeup; already-empty lanes are skipped.
        """

        if not isinstance(time, Real) or not math.isfinite(float(time)):
            raise ValueError(f"stage wakeup time must be finite, got {time!r}")
        if type(replica_id) is not int or replica_id < 0:
            raise ValueError("replica_id must be an exact non-negative int")
        if type(stage_id) is not int or stage_id < 0:
            raise ValueError("stage_id must be an exact non-negative int")

        from frontier.events.replica_stage_schedule_event import (
            ReplicaStageScheduleEvent,
        )

        events = []
        replica_schedulers = getattr(self, "_replica_schedulers", {})
        if not isinstance(replica_schedulers, dict):
            raise RuntimeError("Replica scheduler registry was not initialized")
        for (candidate_replica_id, candidate_local_id), replica_scheduler in sorted(
            replica_schedulers.items(), key=lambda item: str(item[0])
        ):
            if candidate_replica_id != replica_id:
                continue
            if candidate_local_id == exclude_replica_local_id:
                continue
            stage_scheduler = replica_scheduler.get_replica_stage_scheduler(stage_id)
            if stage_scheduler.is_busy or stage_scheduler.is_empty():
                continue
            events.append(
                ReplicaStageScheduleEvent(
                    float(time),
                    replica_id,
                    stage_id,
                    self._cluster_type,
                    candidate_local_id,
                )
            )

        return events

    def make_attention_dp_batch_global_id(
        self,
        replica_id: int,
        replica_local_id: int | None,
        lane_batch_counter: int,
    ) -> int:
        """Return a batch ID unique within one Replica's attention-DP lanes.

        Each child scheduler owns a local counter. Shared MoE waiting rooms are
        scoped by physical Replica, so the lane counter is packed with the
        attention-DP lane ID to prevent two lanes from claiming the same room.
        The default single-lane path keeps its historical counter unchanged.
        """

        if type(replica_id) is not int or replica_id < 0:
            raise ValueError("replica_id must be an exact non-negative int")
        if type(lane_batch_counter) is not int or lane_batch_counter < 0:
            raise ValueError(
                "lane_batch_counter must be an exact non-negative int, "
                f"got {lane_batch_counter!r}"
            )
        lane_count = int(getattr(self, "_replica_dp_size", 1) or 1)
        if lane_count <= 0:
            raise ValueError(f"attention-DP lane count must be positive, got {lane_count}")
        if replica_local_id is None:
            lane_id = 0
        elif type(replica_local_id) is int and 0 <= replica_local_id < lane_count:
            lane_id = replica_local_id
        else:
            raise ValueError(
                "replica_local_id must be None or an exact lane ID in the "
                f"attention-DP domain [0, {lane_count}), got {replica_local_id!r}"
            )
        return lane_batch_counter * lane_count + lane_id

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

    def discard_stage_admission_ticket(
        self,
        ticket: StageAdmissionTicket,
        *,
        stage_id: int,
    ) -> None:
        """Clean up one ticket captured by a stale stage event."""

        if type(stage_id) is not int or stage_id < 0:
            raise ValueError(
                "stage_id is required to discard a stage admission ticket"
            )
        context = self.get_stage_execution_context(
            int(ticket.replica_id),
            stage_id,
        )
        if context.is_active(ticket):
            context.release(ticket)
        elif context.is_queued(ticket):
            context.cancel(ticket)
        elif not context.is_cancelled(ticket):
            raise ValueError(
                "stale stage event carries an unknown stage admission ticket: "
                f"{ticket.operation_id!r}"
            )

    def transition_stage_admission_for_layer(
        self,
        batch: Batch,
        *,
        stage_id: int,
        layer_id: int,
        operation_kind: str,
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
        if operation_kind not in ("attention", "ffn"):
            raise ValueError(
                "operation_kind must be 'attention' or 'ffn', "
                f"got {operation_kind!r}"
            )
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
            operation_kind,
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

        lane_count = getattr(self, "_replica_dp_size", None)
        if lane_count is None and hasattr(self, "_config"):
            lane_count = getattr(self._config.replica_config, "attn_dp", 1)
        lane_count = max(1, int(lane_count or 1))

        lane_id = int(replica_local_id or 0)
        if lane_id < 0:
            raise ValueError(
                "replica_local_id must be non-negative, "
                f"got {replica_local_id!r}"
            )
        if lane_id >= lane_count:
            raise ValueError(
                "MONOLITHIC decode sync lane id must be within the attention-DP "
                f"domain, got replica_local_id={lane_id}, "
                f"lane_count={lane_count}"
            )

        lane_counter = int(lane_decode_sync_counter or 0)
        if lane_counter < 0:
            raise ValueError(
                "lane_decode_sync_counter must be non-negative, "
                f"got {lane_decode_sync_counter!r}"
            )
        return lane_counter * lane_count + lane_id

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
        return SchedulerDiagnostics.request_id(request)

    @classmethod
    def _debug_request_collection_state(cls, requests: Any) -> Dict[str, Any]:
        return SchedulerDiagnostics.request_collection(requests)

    @staticmethod
    def _debug_batch_id(batch: Batch) -> int:
        return SchedulerDiagnostics.batch_id(batch)

    @classmethod
    def _debug_batch_collection_state(cls, batches: Any) -> Dict[str, Any]:
        return SchedulerDiagnostics.batch_collection(batches)

    @classmethod
    def _debug_m2n_waiting_groups_state(
        cls,
        waiting_by_layer: Dict[
            tuple[int, int] | tuple[int, int, int], Dict[str, Any]
        ],
    ) -> List[Dict[str, Any]]:
        return SchedulerDiagnostics.waiting_groups(waiting_by_layer)

    def get_debug_state(self) -> Dict[str, Any]:
        return SchedulerDiagnostics.collect(self)

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
        lane_workload: EPLaneWorkload,
        context: str,
    ) -> None:
        """Compatibility wrapper for EP lane token validation."""

        validate_token_conservation(input_tokens, lane_workload, context)


    def _materialize_ep_wave_workload(
        self,
        group: List[Tuple[Batch, Any]],
        replica_id: int,
        layer_global_id: int,
        routing_details,
    ) -> LayerEPWorkload:
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
        return materialize_wave_workload(
            group,
            replica_id,
            layer_global_id,
            routing_details,
            total_expert_num=total_expert_num,
            moe_expert_parallel_size=moe_expert_parallel_size,
            router_topk=router_topk,
            # Keep the module-level aliases so existing tests can monkeypatch
            # the scheduler's routing materializer.
            routing_resolver=resolve_routing_details,
            workload_materializer=materialize_layer_ep_workload,
            ownership_builder=build_contiguous_expert_ownership,
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

        replica_config = getattr(self._config, "replica_config", None)
        if replica_config is None:
            raise ValueError("DECODE_FFN requires replica_config for EP materialization")
        return prepare_batch_group_plan(
            group,
            replica_id,
            ep_id,
            expert_global_ids,
            layer_global_id,
            routing_details,
            cluster_type=self._cluster_type,
            router_topk=getattr(replica_config, "router_topk", None),
            total_expert_num=getattr(replica_config, "total_expert_num", None),
            moe_expert_parallel_size=getattr(
                replica_config, "moe_expert_parallel_size", None
            ),
            layer_workload=layer_workload,
            wave_materializer=self._materialize_ep_wave_workload,
        )

    def _materialize_ep_batch_group(
        self,
        plan: EPBatchGroupPlan,
    ) -> EPBatchGroup:
        """Materialize one validated EP plan through scheduler callbacks."""

        return materialize_batch_group(
            plan,
            create_batch_group=self._create_batch_group,
            aggregate_metadata=self._aggregate_decode_ffn_afd_metadata,
        )

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
        """Return the max-lane payload summary for one EP collective."""

        hidden_size = int(self._config.replica_config.model_config.embedding_dim)
        return summarize_alltoall_payload(ep_batches, hidden_size)

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
        """Validate one EP barrier participant without mutating room state."""

        return validate_barrier_arrival(
            phase=phase,
            waiting_rooms=waiting_rooms,
            get_replica=self.get_replica,
            default_ep_size=self._config.replica_config.moe_expert_parallel_size,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            ep_id=ep_id,
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

        return validate_collective_exec_time(
            phase=phase,
            exec_time_ms=exec_time_ms,
            sync_time=sync_time,
        )

    def on_ep_alltoall_dispatch_ready(
        self, time: float, replica_id: int, stage_id: int, batch, ep_id: int
    ):
        """Handle EP dispatch readiness before expert compute begins."""
        return handle_dispatch_ready(self, time, replica_id, stage_id, batch, ep_id)

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
        prepared_lanes = prepare_dispatch_advance(ep_batches=ep_batches, time=time)
        events = [
            EPAllToAllCombineReadyEvent(
                lane.ready_time, replica_id, stage_id, lane.batch, lane.ep_id
            )
            for lane in prepared_lanes
        ]

        self._ep_alltoall_dispatch_waiting_room[replica_id][stage_id].pop(
            batch_global_id
        )
        for lane in prepared_lanes:
            lane.batch._ep_dispatch_collective_end_time_s = float(time)
            lane.batch.time = lane.ready_time

        return events

    def on_ep_alltoall_combine_ready(self, time: float, replica_id: int, stage_id: int, batch, ep_id: int):
        """Route EP combine readiness through the EP collective handler utility."""
        from frontier.scheduler.utils.ep_combine_ready import handle_combine_ready

        return handle_combine_ready(
            self,
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            ep_id=ep_id,
        )

    def _handle_ep_alltoall_combine_ready(
        self, time: float, replica_id: int, stage_id: int, batch, ep_id: int
    ):
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

            model_config = self._config.replica_config.model_config
            # Validate every lane descriptor before resolving the model profile.
            self._get_step3_ep_alltoall_payload_bytes(prospective_batches)
            ep_collective_kind = resolve_ep_collective_kind(
                model_config,
                self._cluster_type,
                expected_ep_size,
            )
            timing = prepare_combine_timing(
                prospective_batches=prospective_batches,
                prospective_arrival_times=prospective_arrival_times,
                expected_ep_size=expected_ep_size,
                collective_kind=ep_collective_kind,
                cluster_type=self._cluster_type,
                hidden_size=int(model_config.embedding_dim),
                predict_alltoall=self._predictor.predict_alltoall_time,
                predict_allgather=self._predictor.predict_allgather_time,
                collective_time_validator=self._validate_ep_collective_exec_time,
            )
            data_size_bytes = timing.data_size_bytes
            payload_description = timing.payload_description
            ep_collective_sync_time = timing.sync_time
            ep_collective_exec_time_ms = timing.exec_time_ms
            combine_end_time = timing.combine_end_time
            post_combine_time_s = timing.post_combine_time_s
            final_event_time = timing.final_event_time

            if ep_wait_room is None:
                ep_wait_room = self._ep_allgather_waiting_room[replica_id][stage_id][
                    batch_global_id
                ]
            ep_wait_room["batches"][ep_id] = batch
            ep_wait_room["arrival_times"][ep_id] = time

            logger.info(
                f"[DEBUG] Creating EPAllToAllCombineCollectiveEvent at time={final_event_time:.3f}s, "
                f"combine_end_time={combine_end_time:.3f}s, "
                f"sync_time={ep_collective_sync_time:.3f}s, exec_time={ep_collective_exec_time_ms:.3f}ms, "
                f"post_combine_time={post_combine_time_s:.6f}s, "
                f"data_size={data_size_bytes} bytes ({payload_description})"
            )

            return [
                EPAllToAllCombineCollectiveEvent(
                    final_event_time,
                    replica_id,
                    stage_id,
                    batch_global_id,
                    combine_end_time=combine_end_time,
                )
            ]
        else:
            logger.info(f"[DEBUG] Waiting for more EP replicas: {len(ep_wait_room['batches'])}/{expected_ep_size}")

        return []

    @staticmethod
    def _resolve_ep_execution_time(ep_batches: Dict[int, EPBatchGroup]) -> float:
        return resolve_ep_execution_time(ep_batches)

    def on_ep_alltoall_combine_collective_schedule(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch_global_id: int,
        metrics_store,
        combine_end_time: float,
    ):
        """Route EP combine completion through the scheduling utility."""
        return schedule_combine_completion(
            self,
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            batch_global_id=batch_global_id,
            metrics_store=metrics_store,
            combine_end_time=combine_end_time,
        )

    def _create_m2n_transfer_events_for_aggregated_batch(
        self,
        batch,
        current_time,
        *,
        source_replica_id: int,
        source_replica_local_id: int | None,
    ):
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
            source_execution_replica_id=source_replica_id,
            source_execution_replica_local_id=source_replica_local_id,
            target_execution_replica_id=batch.decode_attn_original_replica_id,
            target_execution_replica_local_id=(
                batch.decode_attn_original_replica_local_id
            ),
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

        lane_workload = layer_workload.lane(ep_id)
        logic_num_tokens = list(lane_workload.local_token_counts)
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
            lane_workload,
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

    def _create_virtual_global_batch(
        self,
        sample_batch: Batch,
        total_global_tokens: int,
        total_global_prefill_tokens: int,
    ) -> Batch:
        """Create a predictor-only batch for one cross-DP token domain."""

        import copy
        from dataclasses import replace
        from frontier.entities.batch import DecodeCudaGraphMetadata

        if type(total_global_tokens) is not int or total_global_tokens < 0:
            raise ValueError("total_global_tokens must be a non-negative int")
        if (
            type(total_global_prefill_tokens) is not int
            or total_global_prefill_tokens < 0
            or total_global_prefill_tokens > total_global_tokens
        ):
            raise ValueError(
                "total_global_prefill_tokens must be within the aggregate token range"
            )
        virtual_batch = copy.copy(sample_batch)
        virtual_batch._num_tokens = [total_global_tokens]
        virtual_batch._total_num_tokens = total_global_tokens
        virtual_batch._num_prefill_tokens = total_global_prefill_tokens
        metadata = getattr(virtual_batch, "decode_cuda_graph_metadata", None)
        if metadata is not None and total_global_tokens != sample_batch.total_num_tokens:
            if not isinstance(metadata, DecodeCudaGraphMetadata):
                raise TypeError(
                    "decode_cuda_graph_metadata must be DecodeCudaGraphMetadata"
                )
            total_decode_tokens = total_global_tokens - total_global_prefill_tokens
            virtual_batch.decode_cuda_graph_metadata = replace(
                metadata,
                original_total_tokens=total_global_tokens,
                padded_total_tokens=total_global_tokens,
                original_decode_batch_size=total_decode_tokens,
                padded_decode_batch_size=total_decode_tokens,
            )
        return virtual_batch

    @staticmethod
    def _get_forward_step_id(batch: Batch) -> int:
        """Return the shared forward-step identity for one scheduler batch."""

        return ForwardSyncState.get_step_id(batch)

    def _get_forward_sync_state(self) -> ForwardSyncState:
        state = getattr(self, "_forward_sync_state", None)
        if state is None:
            state = ForwardSyncState()
            self._forward_sync_state = state
            self._bind_forward_sync_state_views()
        return state

    def _bind_forward_sync_state_views(self) -> None:
        state = self._forward_sync_state
        self._prefill_sync_completed_keys = state.completed_keys("prefill")
        self._decode_sync_completed_keys = state.completed_keys("decode")
        self._prefill_sync_open_steps = state.open_steps("prefill")
        self._decode_sync_open_steps = state.open_steps("decode")
        self._prefill_sync_closed_steps = state.closed_steps("prefill")
        self._decode_sync_closed_steps = state.closed_steps("decode")
        self._next_forward_step_id_by_replica = state._next_step_id_by_replica
        self._forward_step_used_ids_by_scope = state._used_ids_by_scope

    def _resolve_forward_step(
        self,
        *,
        sync_kind: str,
        waiting_room,
        replica_id: int,
        stage_id: int,
        batch: Batch,
        lane_id: int,
        layer_id: int,
        sync_stage: str,
    ) -> tuple[int, bool]:
        """Resolve one lane through the forward-sync state owner."""

        state = self._get_forward_sync_state()

        def lookup(step_id: int):
            replica_rooms = waiting_room.get(replica_id)
            stage_rooms = replica_rooms.get(stage_id) if replica_rooms else None
            step_rooms = stage_rooms.get(step_id) if stage_rooms else None
            layer_rooms = step_rooms.get(layer_id) if step_rooms else None
            return layer_rooms.get(sync_stage) if layer_rooms else None

        return state.resolve_step(
            sync_kind=sync_kind,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            lane_id=lane_id,
            layer_id=layer_id,
            sync_stage=sync_stage,
            room_lookup=lookup,
        )

    def _close_forward_step(
        self,
        *,
        sync_kind: str,
        replica_id: int,
        stage_id: int,
        layer_id: int,
        sync_stage: str,
        provisional_id: int,
        cohort_id: int,
        cohort_batches: dict[int, Batch],
    ) -> None:
        """Close one room through the forward-sync state owner."""

        self._get_forward_sync_state().close_step(
            sync_kind=sync_kind,
            replica_id=replica_id,
            stage_id=stage_id,
            layer_id=layer_id,
            sync_stage=sync_stage,
            provisional_id=provisional_id,
            step_id=cohort_id,
            source_batches=cohort_batches,
        )

    @staticmethod
    def _forward_step_source_batches(cohort_batches: dict[int, Batch] | None, batch: Batch) -> dict[int, Batch]:
        """Normalize a direct wave call and preserve lane identity."""

        if cohort_batches is None:
            lane_id = getattr(batch, "_stage_owner_replica_local_id", None)
            if lane_id is None:
                lane_id = 0
            return {int(lane_id): batch}
        if not isinstance(cohort_batches, dict) or not cohort_batches:
            raise ValueError("cohort_batches must be a non-empty lane mapping")
        normalized: dict[int, Batch] = {}
        for lane_id, source_batch in cohort_batches.items():
            if type(lane_id) is not int or lane_id < 0:
                raise ValueError(f"cohort lane ID must be non-negative int, got {lane_id!r}")
            if not isinstance(source_batch, Batch):
                raise TypeError(
                    "cohort_batches values must be Batch instances, "
                    f"got {type(source_batch).__name__}"
                )
            normalized[lane_id] = source_batch
        return normalized


    def _promote_forward_step_to_ep_wave(
        self,
        *,
        source_batches: dict[int, Batch],
        replica_id: int,
        stage_id: int,
        layer_id: int,
        cohort_id: int,
        participant_ep_ids: tuple[int, ...],
    ) -> None:
        """Atomically replace active lane owners with one Replica-local EP wave."""

        live_batches = [batch for batch in source_batches.values() if not batch.is_idle]
        tickets = []
        for source_batch in live_batches:
            ticket = getattr(source_batch, "_stage_admission_ticket", None)
            if ticket is None:
                if getattr(self, "_stage_execution_contexts", None) is None:
                    # Standalone layer probes intentionally omit the outer DES
                    # admission registry and cannot claim stage ownership.
                    return
                raise ValueError(
                    "cohort EP promotion requires a stage admission ticket for "
                    "every live batch"
                )
            if ticket not in tickets:
                tickets.append(ticket)
        if not tickets:
            return
        context = self.get_stage_execution_context(replica_id, stage_id)
        if len(tickets) == 1 and tickets[0].scope == EP_WAVE:
            wave_ticket = tickets[0]
        elif len(tickets) == 1:
            owner_batch = next(
                source_batch
                for source_batch in live_batches
                if getattr(source_batch, "_stage_admission_ticket", None) == tickets[0]
            )
            self.transition_stage_admission_for_layer(
                owner_batch,
                stage_id=stage_id,
                layer_id=layer_id,
                operation_kind="ffn",
                scope=EP_WAVE,
                participant_ep_ids=participant_ep_ids,
            )
            wave_ticket = owner_batch._stage_admission_ticket
        else:
            if any(ticket.scope != FULL_STAGE_WORLD for ticket in tickets):
                raise ValueError("cohort EP promotion requires full-stage owner tickets")
            wave_ticket = context.replace_full_stage_owners_with_ep_wave(
                tickets,
                operation_id=(
                    "shared_ep_wave",
                    int(replica_id),
                    int(stage_id),
                    int(cohort_id),
                    int(layer_id),
                ),
                participant_ep_ids=participant_ep_ids,
            )
        for source_batch in live_batches:
            source_batch._stage_admission_ticket = wave_ticket
            history = getattr(source_batch, "_stage_admission_scope_history", None)
            if history is None:
                history = []
                source_batch._stage_admission_scope_history = history
            history.append(
                {
                    "stage_id": int(stage_id),
                    "layer_id": int(layer_id),
                    "scope": EP_WAVE,
                    "admission_seq": int(wave_ticket.admission_seq),
                    "participant_ep_ids": tuple(wave_ticket.participant_ep_ids),
                }
            )

    def _restore_forward_step_full_stage_owners(
        self,
        *,
        source_batches: dict[int, Batch],
        replica_id: int,
        stage_id: int,
        layer_id: int,
        cohort_id: int,
        operation_kind: str,
    ) -> bool:
        """Restore one active full-stage ticket per request-owner lane."""

        live_batches = [batch for batch in source_batches.values() if not batch.is_idle]
        if not live_batches:
            return False
        tickets = []
        for batch in live_batches:
            ticket = getattr(batch, "_stage_admission_ticket", None)
            if ticket is None:
                if getattr(self, "_stage_execution_contexts", None) is None:
                    # Standalone layer probes intentionally omit the outer DES
                    # admission registry and cannot restore stage ownership.
                    return False
                raise ValueError(
                    "cohort full-stage restoration requires a stage admission "
                    "ticket for every live batch"
                )
            tickets.append(ticket)
        wave_tickets = []
        for ticket in tickets:
            if ticket not in wave_tickets:
                wave_tickets.append(ticket)
        if len(wave_tickets) != 1 or wave_tickets[0].scope != EP_WAVE:
            return False
        context = self.get_stage_execution_context(replica_id, stage_id)
        operation_ids = [
            (
                "shared_layer",
                int(batch.id),
                int(batch.schedule_epoch),
                int(stage_id),
                int(layer_id),
                operation_kind,
                FULL_STAGE_WORLD,
            )
            for batch in live_batches
        ]
        owners = context.replace_ep_wave_with_full_stage_owners(
            wave_tickets[0],
            operation_ids=operation_ids,
        )
        for source_batch, owner in zip(live_batches, owners):
            source_batch._stage_admission_ticket = owner
        return True
    def _prepare_moe_ep_wave_plan(
        self,
        *,
        wave_inputs,
        time: float,
        replica_id: int,
        stage_id: int,
        layer_id: int,
    ):
        """Prepare shared MoE workload and timing through the utility layer."""

        return prepare_moe_wave_from_inputs(
            wave_inputs=wave_inputs,
            time=time,
            materialize_workload=self._materialize_layer_ep_workload_for_batch,
            trace_identity_builder=self._build_ep_trace_identity,
            conservation_logger=self._log_ep_conservation_trace,
            predictor=self._predictor,
            lane_builder=self._build_prefill_ep_lane_batch,
            phase_getter=self._get_shared_ep_phase_times_ms,
            workload_logger=self._log_ep_workload_trace,
            barrier_logger=self._log_ep_barrier_trace,
            wave_logger=self._log_ep_wave_end_trace,
            cluster_type=self._cluster_type,
            replica_id=replica_id,
            stage_id=stage_id,
            layer_id=layer_id,
        )

    def _on_prefill_ep_wave_ready(
        self,
        *,
        time: float,
        replica_id: int,
        stage_id: int,
        batch: Batch,
        layer_id: int,
        replica_local_id: int | None = None,
        cohort_batches: dict[int, Batch] | None = None,
    ) -> List:
        """Run one layer's FFN wave and schedule its slowest-lane barrier."""

        from frontier.events.prefill_sync_collective_event import (
            PrefillSyncCollectiveEvent,
        )

        if not isinstance(time, Real) or not math.isfinite(float(time)):
            raise ValueError("prefill EP wave time must be finite")
        time = float(time)
        wave_inputs = prepare_ep_wave_inputs(
            source_batches=self._forward_step_source_batches(cohort_batches, batch),
            batch=batch,
            step_id_getter=self._get_forward_step_id,
            aggregate_batch_builder=self._create_virtual_global_batch,
        )
        source_batches = wave_inputs.source_batches
        cohort_id = wave_inputs.step_id
        for lane_id, source_batch in source_batches.items():
            if not hasattr(source_batch, "_stage_owner_replica_local_id"):
                source_batch._stage_owner_replica_local_id = (
                    replica_local_id if replica_local_id is not None else lane_id
                )
        model_config = self._config.replica_config.model_config
        predictor = self._predictor
        non_idle_source_batches = list(wave_inputs.non_idle_batches)
        layer_workload = None
        lane_compute_times_ms: list[float] = []
        if model_config.is_moe_layer(layer_id):
            plan = self._prepare_moe_ep_wave_plan(
                wave_inputs=wave_inputs,
                time=time,
                replica_id=replica_id,
                stage_id=stage_id,
                layer_id=layer_id,
            )
            layer_workload = plan.layer_workload
            phase_times = plan.phase_times
            trace_identity = plan.trace_identity
            lane_compute_times_ms = list(phase_times.lane_compute_times_ms)
            self._promote_forward_step_to_ep_wave(
                source_batches=source_batches,
                replica_id=replica_id,
                stage_id=stage_id,
                layer_id=layer_id,
                cohort_id=cohort_id,
                participant_ep_ids=tuple(layer_workload.participant_ep_ids),
            )
        else:
            from frontier.events.dense_layer_complete_event import (
                DenseLayerCompleteEvent,
            )

            dense_events = []
            for source_batch in non_idle_source_batches:
                execution_time = predictor.predict_stage_execution_time(
                    source_batch,
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
                    source_batch,
                    stage_id=stage_id,
                    layer_id=layer_id,
                    operation_kind="ffn",
                    scope=FULL_STAGE_WORLD,
                )
                component_ledger = getattr(
                    source_batch,
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
                        f"replica={replica_id}, stage={stage_id}, layer={layer_id}, batch={source_batch.id}"
                    )
                component_ledger[stage_id].append(dense_time_ms)
                dense_events.append(
                    DenseLayerCompleteEvent(
                        time + dense_time_ms * 1e-3,
                        replica_id,
                        stage_id,
                        source_batch,
                        layer_id,
                        "prefill",
                        self._cluster_type,
                    )
                )
            return dense_events

        if not lane_compute_times_ms:
            raise ValueError("Prefill layer wave produced no participant timing")
        timing = plan.timing
        barrier_end_time_s = timing.wave_end_time_s
        wave_time_ms = timing.dispatch_barrier_time_ms + timing.combine_barrier_time_ms + timing.post_combine_barrier_time_ms
        for source_batch in non_idle_source_batches:
            component_ledger = getattr(
                source_batch,
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
                    f"replica={replica_id}, stage={stage_id}, layer={layer_id}, "
                    f"batch={source_batch.id}"
                )
            component_ledger[stage_id].append(wave_time_ms)
            source_batch._prefill_ep_wave_lane_times_ms = tuple(lane_compute_times_ms)
            source_batch._prefill_ep_wave_workload = layer_workload

        sync_room = self._prefill_sync_waiting_room[replica_id][stage_id][
            cohort_id
        ][layer_id]["post_moe"]
        if sync_room["batches"]:
            raise ValueError(
                "PREFILL EP wave post_moe room already contains a batch: "
                f"replica={replica_id}, stage={stage_id}, layer={layer_id}, "
                f"forward_cohort_id={cohort_id}"
            )
        sync_room["batches"].update(source_batches)
        sync_room["arrival_times"].update(
            {lane_id: barrier_end_time_s for lane_id in source_batches}
        )
        return [
            PrefillSyncCollectiveEvent(
                barrier_end_time_s,
                replica_id,
                stage_id,
                cohort_id,
                "post_moe",
                layer_id,
                cluster_type=self._cluster_type,
            )
        ]

    def _uses_shared_prefill_ep_wave(self, batch: Batch, layer_id: int) -> bool:
        """Return whether the canonical shared-domain PREFILL EP path is active."""
        model_config = getattr(getattr(self._config, "replica_config", None), "model_config", None)
        routing_attr = (
            "_prefill_routing_details"
            if self._cluster_type == ClusterType.PREFILL
            else "_monolithic_routing_details"
        )
        return uses_shared_layer_path(
            cluster_type=self._cluster_type,
            allowed_clusters=(ClusterType.PREFILL, ClusterType.MONOLITHIC),
            model_config=model_config,
            predictor=self._predictor,
            layer_id=layer_id,
            routing_attribute=routing_attr,
            require_moe_layer=True,
        )

    def _uses_shared_prefill_layer_path(self, batch: Batch, layer_id: int) -> bool:
        """Return whether a shared-domain PREFILL model needs layer stepping."""
        model_config = getattr(getattr(self._config, "replica_config", None), "model_config", None)
        routing_attr = (
            "_prefill_routing_details"
            if self._cluster_type == ClusterType.PREFILL
            else "_monolithic_routing_details"
        )
        return uses_shared_layer_path(
            cluster_type=self._cluster_type,
            allowed_clusters=(ClusterType.PREFILL, ClusterType.MONOLITHIC),
            model_config=model_config,
            predictor=self._predictor,
            layer_id=layer_id,
            routing_attribute=routing_attr,
            require_moe_layer=False,
        )

    def _uses_shared_prefill_layer_protocol(self, batch: Batch, layer_id: int) -> bool:
        """Compatibility alias for the pre-refactor execution-path name."""

        return self._uses_shared_prefill_layer_path(batch, layer_id)

    def _on_decode_ep_wave_ready(
        self,
        *,
        time: float,
        replica_id: int,
        stage_id: int,
        batch: Batch,
        layer_id: int,
        replica_local_id: int | None = None,
        cohort_batches: dict[int, Batch] | None = None,
    ) -> List:
        """Run one unified-DECODE layer's local EP wave and barrier."""

        from frontier.events.decode_sync_collective_event import (
            DecodeSyncCollectiveEvent,
        )

        if not isinstance(time, Real) or not math.isfinite(float(time)):
            raise ValueError("decode EP wave time must be finite")
        time = float(time)
        wave_inputs = prepare_ep_wave_inputs(
            source_batches=self._forward_step_source_batches(cohort_batches, batch),
            batch=batch,
            step_id_getter=self._get_forward_step_id,
            aggregate_batch_builder=self._create_virtual_global_batch,
        )
        source_batches = wave_inputs.source_batches
        cohort_id = wave_inputs.step_id
        for lane_id, source_batch in source_batches.items():
            if not hasattr(source_batch, "_stage_owner_replica_local_id"):
                source_batch._stage_owner_replica_local_id = (
                    replica_local_id if replica_local_id is not None else lane_id
                )
        model_config = self._config.replica_config.model_config
        predictor = self._predictor
        non_idle_source_batches = list(wave_inputs.non_idle_batches)
        layer_workload = None
        lane_compute_times_ms: list[float] = []
        if model_config.is_moe_layer(layer_id):
            plan = self._prepare_moe_ep_wave_plan(
                wave_inputs=wave_inputs,
                time=time,
                replica_id=replica_id,
                stage_id=stage_id,
                layer_id=layer_id,
            )
            layer_workload = plan.layer_workload
            phase_times = plan.phase_times
            trace_identity = plan.trace_identity
            lane_compute_times_ms = list(phase_times.lane_compute_times_ms)
            self._promote_forward_step_to_ep_wave(
                source_batches=source_batches,
                replica_id=replica_id,
                stage_id=stage_id,
                layer_id=layer_id,
                cohort_id=cohort_id,
                participant_ep_ids=tuple(layer_workload.participant_ep_ids),
            )
        else:
            from frontier.events.dense_layer_complete_event import (
                DenseLayerCompleteEvent,
            )

            dense_events = []
            for source_batch in non_idle_source_batches:
                execution_time = predictor.predict_stage_execution_time(
                    source_batch,
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
                    source_batch,
                    stage_id=stage_id,
                    layer_id=layer_id,
                    operation_kind="ffn",
                    scope=FULL_STAGE_WORLD,
                )
                dense_events.append(
                    DenseLayerCompleteEvent(
                        time + dense_time_ms * 1e-3,
                        replica_id,
                        stage_id,
                        source_batch,
                        layer_id,
                        "decode",
                        self._cluster_type,
                    )
                )
            return dense_events

        if not lane_compute_times_ms:
            raise ValueError("Decode layer wave produced no participant timing")
        timing = plan.timing
        barrier_end_time_s = timing.wave_end_time_s
        for source_batch in non_idle_source_batches:
            source_batch._decode_ep_wave_lane_times_ms = tuple(lane_compute_times_ms)

        batch_global_id = cohort_id
        sync_room = self._decode_sync_waiting_room[replica_id][stage_id][
            batch_global_id
        ][layer_id]["post_moe"]
        if sync_room["batches"]:
            raise ValueError(
                "DECODE EP wave post_moe room already contains a batch: "
                f"replica={replica_id}, stage={stage_id}, layer={layer_id}, "
                f"forward_cohort_id={batch_global_id}"
            )
        sync_room["batches"].update(source_batches)
        sync_room["arrival_times"].update(
            {lane_id: barrier_end_time_s for lane_id in source_batches}
        )
        return [
            DecodeSyncCollectiveEvent(
                barrier_end_time_s,
                replica_id,
                stage_id,
                batch_global_id,
                "post_moe",
                layer_id,
                cluster_type=self._cluster_type,
            )
        ]

    def _uses_shared_decode_ep_wave(self, batch: Batch, layer_id: int) -> bool:
        """Return whether the canonical unified-DECODE EP path is active."""
        model_config = getattr(getattr(self._config, "replica_config", None), "model_config", None)
        routing_attr = (
            "_decode_routing_details"
            if self._cluster_type == ClusterType.DECODE
            else "_monolithic_routing_details"
        )
        return uses_shared_layer_path(
            cluster_type=self._cluster_type,
            allowed_clusters=(ClusterType.DECODE, ClusterType.MONOLITHIC),
            model_config=model_config,
            predictor=self._predictor,
            layer_id=layer_id,
            routing_attribute=routing_attr,
            require_moe_layer=True,
        )

    def _uses_shared_decode_layer_path(self, batch: Batch, layer_id: int) -> bool:
        """Return whether a shared-domain DECODE model needs layer stepping."""
        model_config = getattr(getattr(self._config, "replica_config", None), "model_config", None)
        routing_attr = (
            "_decode_routing_details"
            if self._cluster_type == ClusterType.DECODE
            else "_monolithic_routing_details"
        )
        return uses_shared_layer_path(
            cluster_type=self._cluster_type,
            allowed_clusters=(ClusterType.DECODE, ClusterType.MONOLITHIC),
            model_config=model_config,
            predictor=self._predictor,
            layer_id=layer_id,
            routing_attribute=routing_attr,
            require_moe_layer=False,
        )

    def _uses_shared_decode_layer_protocol(self, batch: Batch, layer_id: int) -> bool:
        """Compatibility alias for the pre-refactor execution-path name."""

        return self._uses_shared_decode_layer_path(batch, layer_id)

    def on_prefill_sync(self, time: float, replica_id: int, stage_id: int, batch: Batch,
                       replica_local_id: int | None, sync_stage: str, layer_id: int, stage_execution_time: float):
        return enter_prefill_sync(
            self, time, replica_id, stage_id, batch, replica_local_id,
            sync_stage, layer_id, stage_execution_time,
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
        """Delegate PREFILL collective completion to the utility handler."""
        return handle_prefill_sync_collective(
            self,
            time,
            replica_id,
            stage_id,
            batch_global_id,
            sync_stage,
            layer_id,
            metrics_store,
            direct_batch=direct_batch,
        )


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
        model_config = getattr(getattr(self._config, "replica_config", None), "model_config", None)
        return first_dense_layer_id(model_config)

    def _get_prefill_dense_reference_execution_time(
        self,
        sample_batch: Batch,
        stage_id: int,
    ) -> Optional[ExecutionTime]:
        """Predict one dense layer execution for mixed-layer MoE trace completion."""
        model_config = getattr(getattr(self._config, "replica_config", None), "model_config", None)
        return predict_dense_reference(
            predictor=self._predictor,
            batch=sample_batch,
            stage_id=stage_id,
            cluster_type=self._cluster_type,
            model_config=model_config,
        )

    def _create_corrected_execution_time_for_metrics(
        self,
        original_execution_time,
        actual_execution_time_ms,
        original_start_time,
    ):
        """Create corrected ExecutionTime payload used by metrics/trace emission."""
        del actual_execution_time_ms, original_start_time
        return build_single_layer_metrics_execution_time(original_execution_time)

    def _record_mtp_terminal_completion_delay(
        self,
        batch: Batch,
        terminal_delay_s: float,
    ) -> None:
        from frontier.scheduler.utils.mtp_metrics import record_terminal_completion_delay

        record_terminal_completion_delay(batch, terminal_delay_s)

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
        return self._get_forward_step_id(batch)

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
        """Enter the canonical layer-local DECODE synchronization path."""
        from frontier.scheduler.utils.sync_entry import enter_decode_sync

        return enter_decode_sync(
            self,
            time,
            replica_id,
            stage_id,
            batch,
            replica_local_id,
            sync_stage,
            layer_id,
            stage_execution_time,
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
        """Delegate DECODE collective completion to the utility handler."""

        return handle_decode_sync_collective(
            self,
            time,
            replica_id,
            stage_id,
            batch_global_id,
            sync_stage,
            layer_id,
            metrics_store,
            direct_batch=direct_batch,
        )

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
        return handle_decode_attn_arrival(self, time, batch, transfer_info, logger)

    def _handle_decode_arrival(
        self,
        time: float,
        batch: Batch,
        transfer_info,
        logger,
    ) -> List:
        """Handle KV cache arrival at a unified decode cluster."""
        return handle_decode_arrival(self, time, batch, transfer_info, logger)


    def on_m2n_arrival(
        self,
        time: float,
        batch: Batch,
        transfer_info,
        *,
        expected_roundtrip_inflight: bool = False,
        request_end_deferred: bool = False,
    ) -> List:
        """Route M2N transfer arrival to the appropriate cluster handler.

        ``expected_roundtrip_inflight`` describes the request lifecycle state
        that the handler must validate against.  The M2N end event uses
        ``False`` while the request's F→A end hook is still deferred until the
        target admission succeeds.  Direct scheduler callers retain the
        historical default, where the end hook has already been applied.
        """
        return route_m2n_arrival(
            self,
            time,
            batch,
            transfer_info,
            expected_roundtrip_inflight=expected_roundtrip_inflight,
            request_end_deferred=request_end_deferred,
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
    def _normalize_m2n_lanes(
        raw_lanes,
        *,
        identity_scope: M2NLaneIdentityScope,
        field_name: str,
        require_nonempty: bool,
    ) -> List[tuple[int, int | None]]:
        """Compatibility wrapper for M2N lane normalization."""

        return normalize_lanes(
            raw_lanes,
            identity_scope=identity_scope,
            field_name=field_name,
            require_nonempty=require_nonempty,
        )

    @staticmethod
    def _normalize_m2n_lane_contract(
        raw_lanes,
        *,
        identity_scope: M2NLaneIdentityScope,
        field_name: str,
        require_nonempty: bool,
    ) -> List[tuple[int, int | None]]:
        """Compatibility alias for the pre-refactor lane normalizer name."""

        return BaseClusterScheduler._normalize_m2n_lanes(
            raw_lanes,
            identity_scope=identity_scope,
            field_name=field_name,
            require_nonempty=require_nonempty,
        )

    def _validate_decode_ffn_waiting_room(
        self,
        *,
        group_key: tuple[int, int] | tuple[int, int, int],
        room: dict,
        expected_lane_contract: Optional[tuple[tuple[int, int], ...]] = None,
        incoming_batch: Optional[Batch] = None,
    ) -> tuple[tuple[int, int], ...]:
        """Validate one DECODE_FFN waiting room through the transfer utility."""

        return validate_decode_ffn_waiting_room(
            group_key=group_key,
            room=room,
            expected_lane_contract=expected_lane_contract,
            incoming_batch=incoming_batch,
        )

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
        int,
    ]:
        """Validate one A-to-F receipt through the transfer utility."""

        return validate_decode_ffn_receipt(self, batch, transfer_info)

    def _validate_decode_attn_wave_binding(
        self,
        batch: Batch,
        *,
        lane: tuple[int, int],
        afd_stage_idx: int,
        requests: List[Request],
        active_requests: List[Request],
        context: str,
    ) -> None:
        """Validate one DECODE_ATTN wave through the transfer utility."""

        return validate_decode_attn_wave_binding(
            self,
            batch,
            lane=lane,
            afd_stage_idx=afd_stage_idx,
            requests=requests,
            active_requests=active_requests,
            context=context,
        )

    def _validate_decode_attn_m2n_receipt(
        self,
        batch: Batch,
        transfer_info: "M2NTransferInfo",
        *,
        expected_roundtrip_inflight: bool,
        request_end_deferred: bool = False,
    ) -> Dict[str, Any]:
        """Validate one F-to-A receipt through the transfer utility."""

        return validate_decode_attn_receipt(
            self,
            batch,
            transfer_info,
            expected_roundtrip_inflight=expected_roundtrip_inflight,
            request_end_deferred=request_end_deferred,
        )


    def _validate_decode_attn_f2a_queued_batch(
        self,
        queued_batch: Batch,
        *,
        queue_lane: tuple[int, int],
        round_key: tuple,
        expected_lanes: List[tuple[int, int]],
        current_batch: Batch,
    ) -> tuple[int, int]:
        """Validate a queued F-to-A batch through the transfer utility."""

        return validate_decode_attn_queued_batch(
            self,
            queued_batch,
            queue_lane=queue_lane,
            round_key=round_key,
            expected_lanes=expected_lanes,
            current_batch=current_batch,
        )



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
            target_replica_id,
        ) = self._validate_decode_ffn_m2n_receipt(batch, transfer_info)

        transfer_info.target_ffn_replica_id = target_replica_id
        transfer_info.target_execution_replica_id = target_replica_id
        transfer_info.target_execution_replica_local_id = None

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
        from frontier.scheduler.utils.m2n_promotion import promote_decode_ffn_group

        return promote_decode_ffn_group(
            self,
            time,
            group_key,
            room,
            logger,
            allow_idle_injection=allow_idle_injection,
            expected_lanes=expected_lanes,
            expected_lane_ids=expected_lane_ids,
        )

    def _inject_ffn_idle_lanes_for_barrier(
        self,
        time: float,
        group_key,
        room: dict,
        logger,
        *,
        expected_lane_ids: Optional[List[tuple[int, int]]] = None,
    ) -> List[tuple[int, int]]:
        """Inject idle FFN lanes through the dedicated utility."""

        from frontier.scheduler.utils.m2n_idle import inject_ffn_idle_lanes

        return inject_ffn_idle_lanes(
            self,
            time,
            group_key,
            room,
            logger,
            expected_lane_ids=expected_lane_ids,
        )

    def _promote_incomplete_m2n_groups_with_idle_lanes(self, logger) -> int:
        """Promote any incomplete FFN grouping barriers by injecting idle lanes."""
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
        """Build DP-padding replacements through the transfer utility."""
        return prepare_dp_padding(picked)

    def _handle_m2n_arrival_decode_attn(
        self,
        time: float,
        micro_batch: Batch,
        transfer_info,
        logger,
        *,
        expected_roundtrip_inflight: bool = False,
        request_end_deferred: bool = False,
    ) -> List:
        """Route DECODE_ATTN M2N completion through the arrival utility."""
        return handle_decode_attn_arrival(
            self,
            time,
            micro_batch,
            transfer_info,
            logger,
            expected_roundtrip_inflight=expected_roundtrip_inflight,
            request_end_deferred=request_end_deferred,
        )

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
        """Validate one A-to-F batch through the transfer utility."""
        return validate_decode_attn_a2f_batch_entry(
            self,
            batch=batch,
            lane=lane,
            layer_id=layer_id,
            afd_stage_idx=afd_stage_idx,
            model_is_moe=model_is_moe,
            context=context,
            allow_idle=allow_idle,
        )

    @staticmethod
    def _validate_decode_attn_wave_stages(
        wave_state: dict[str, Any],
        *,
        context: str,
    ) -> tuple[set[int], dict[int, str], dict[int, int]]:
        """Validate the complete stage-local state for one DECODE_ATTN cohort."""

        if type(wave_state) is not dict:
            raise RuntimeError(
                f"DECODE_ATTN {context} active cohort state must be an exact dict"
            )

        active_stage_indices = wave_state.get("active_stage_indices")
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

        stage_phases = wave_state.get("stage_phases")
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

        stage_layers = wave_state.get("stage_current_layer_ids")
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

    def _validate_decode_attn_a2f_wave_phase(
        self,
        batch: Batch,
        *,
        layer_id: int,
        afd_stage_idx: int,
        context: str,
    ) -> None:
        """Validate the stage-local phase/layer of an A-to-F cohort."""

        wave_id = getattr(batch, "decode_attn_cohort_id", None)
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
        wave_states = getattr(
            replica_schedulers[lane],
            "_decode_attn_active_cohort_states",
            None,
        )
        if type(wave_states) is not dict:
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} active cohort states must be an "
                "exact dict"
            )
        wave_state = wave_states.get(wave_id)
        if type(wave_state) is not dict:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} references an inactive or unknown "
                f"cohort: cohort_id={wave_id}, lane={lane}"
            )
        active_stage_indices, stage_phases, stage_layers = (
            self._validate_decode_attn_wave_stages(
                wave_state,
                context=f"A-to-F {context}",
            )
        )
        if afd_stage_idx not in active_stage_indices:
            raise ValueError(
                f"DECODE_ATTN A-to-F {context} stage is not active in the cohort: "
                f"stage={afd_stage_idx}, active={sorted(active_stage_indices)}"
            )

        aggregate_phase = wave_state.get("af_phase")
        if type(aggregate_phase) is not str:
            raise RuntimeError(
                f"DECODE_ATTN A-to-F {context} cohort af_phase must be an exact "
                f"str, got {aggregate_phase!r}"
            )
        aggregate_layer = wave_state.get("current_layer_id")
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
        replica_config = getattr(self._config, "replica_config", None)
        model_config = getattr(replica_config, "model_config", None)
        model_is_moe = getattr(model_config, "is_moe", None)
        return validate_decode_attn_a2f_waiting_room(
            group_key=group_key,
            room=room,
            expected_lane_contract=expected_lane_contract,
            incoming_batch=incoming_batch,
            topology_validator=self._validate_decode_attn_a2f_topology_value,
            lane_normalizer=self._normalize_m2n_lanes,
            batch_validator=self._validate_decode_attn_a2f_batch_entry,
            model_is_moe=model_is_moe,
        )

    @staticmethod
    def _validate_decode_attn_a2f_predictor_result(
        predictor_result: Any,
    ) -> tuple[int, int | float]:
        """Validate an A-to-F predictor result through the transfer utility."""
        return validate_a2f_predictor_result(predictor_result)

    def _prepare_decode_attn_idle_lanes_for_barrier(
        self,
        *,
        time: float,
        group_key: tuple[int, int],
        idle_lanes: List[tuple[int, int]],
        is_moe: bool,
    ) -> List[tuple[tuple[int, int], tuple[int, Batch]]]:
        """Build A-to-F idle entries through the transfer utility."""
        return prepare_decode_attn_idle_lanes(
            time=time,
            group_key=group_key,
            idle_lanes=idle_lanes,
            is_moe=is_moe,
        )

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
        return release_dense_a2f(
            self,
            time,
            batch,
            replica_id=replica_id,
            replica_local_id=replica_local_id,
            layer_id=layer_id,
            logger=logger,
        )

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
        """Admit a completed DECODE_ATTN batch into A-to-F transfer."""
        return schedule_decode_attn_a2f_ready(
            self,
            time,
            batch,
            replica_id=replica_id,
            replica_local_id=replica_local_id,
            layer_id=layer_id,
            logger=logger,
        )

    def _prepare_decode_attn_batch_phase(
        self,
        batch: Batch,
        *,
        phase: str,
        replica_id: int,
        replica_local_id: int | None,
        layer_id: int | None = None,
    ) -> Optional[dict[str, Any]]:
        """Prepare a DECODE_ATTN phase update through the utility."""
        return prepare_decode_attn_batch_phase(
            self,
            batch,
            phase=phase,
            replica_id=replica_id,
            replica_local_id=replica_local_id,
            layer_id=layer_id,
        )

    @staticmethod
    def _apply_decode_attn_batch_phase(
        prepared_update: Optional[dict[str, Any]],
    ) -> None:
        """Apply a previously prepared DECODE_ATTN phase update."""
        apply_decode_attn_batch_phase(prepared_update)

    def _commit_decode_attn_batch_phases(
        self,
        prepared_updates: List[Optional[dict[str, Any]]],
    ) -> None:
        """Commit DECODE_ATTN phase updates atomically."""
        commit_decode_attn_batch_phases(
            self,
            prepared_updates,
            apply_fn=self._apply_decode_attn_batch_cohort_phase,
        )

    def _set_decode_attn_batch_phase(
        self,
        batch: Batch,
        *,
        phase: str,
        replica_id: int,
        replica_local_id: int | None,
        layer_id: int | None = None,
        prepare_only: bool = False,
    ) -> Optional[dict[str, Any]]:
        """Prepare or apply one DECODE_ATTN phase update."""
        return set_decode_attn_batch_phase(
            self,
            batch,
            phase=phase,
            replica_id=replica_id,
            replica_local_id=replica_local_id,
            layer_id=layer_id,
            prepare_only=prepare_only,
            prepare_fn=lambda _scheduler, phase_batch, **kwargs: self._prepare_decode_attn_batch_cohort_phase(
                phase_batch,
                **kwargs,
            ),
            apply_fn=self._apply_decode_attn_batch_cohort_phase,
        )

    def _peek_decode_attn_barrier_round_id(self) -> int:
        """Return the next A-to-F barrier round without mutating its counter."""

        next_round_id = getattr(self, "_decode_attn_barrier_round_counter", 0)
        if type(next_round_id) is not int or next_round_id < 0:
            raise RuntimeError(
                "DECODE_ATTN A-to-F barrier round counter must be an exact "
                f"non-negative int, got {next_round_id!r}"
            )
        return next_round_id

    def _get_decode_attn_a2f_active_local_attn_lanes(
        self, *, cohort_id: int, request_ids: tuple[int, ...],
        afd_stage_idx: int, layer_id: int,
    ) -> List[tuple[int, int]]:
        return get_a2f_active_local_attn_lanes(
            self, cohort_id=cohort_id, request_ids=request_ids,
            afd_stage_idx=afd_stage_idx, layer_id=layer_id,
        )

    def _get_decode_attn_stage_slot_active_lanes(
        self, afd_stage_idx: int, *, replica_id: int | None = None,
        phase: str | None = None, layer_id: int | None = None,
    ) -> List[tuple[int, int]]:
        return get_stage_slot_active_lanes(
            self, afd_stage_idx, replica_id=replica_id,
            phase=phase, layer_id=layer_id,
        )

    def _get_decode_attn_a2f_expected_lanes(
        self, afd_stage_idx: int | None = None, *, layer_id: int | None = None,
    ) -> List[tuple[int, int]]:
        return get_a2f_expected_lanes(self, afd_stage_idx, layer_id=layer_id)

    def _get_decode_attn_f2a_expected_lanes(
        self, replica_id: int, *, afd_stage_idx: int | None = None,
    ) -> List[tuple[int, int]]:
        return get_f2a_expected_lanes(self, replica_id, afd_stage_idx=afd_stage_idx)

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

    def _create_batch_group(
        self,
        requests: List[Request],
        num_tokens: List[int],
        replica_id: int,
        ep_id: int,
        time: float,
        source_batch_ids: List[int],
        lane_workload: EPLaneWorkload,
    ) -> EPBatchGroup:
        batch_group = EPBatchGroup(
            requests,
            num_tokens,
            replica_id,
            ep_id,
            time,
            source_batch_ids,
            lane_workload,
            self._cluster_type,
            is_moe=self._config.replica_config.model_config.is_moe,
        )

        return batch_group

    # Compatibility aliases for private callers that still use the old
    # cohort terminology. Internal scheduler paths use the forward-step names.
    _get_forward_cohort_id = _get_forward_step_id
    _resolve_forward_sync_cohort = _resolve_forward_step
    _close_forward_sync_cohort = _close_forward_step
    _cohort_source_batches = _forward_step_source_batches
    _promote_cohort_to_ep_wave = _promote_forward_step_to_ep_wave
    _restore_cohort_full_stage_owners = _restore_forward_step_full_stage_owners
    _validate_decode_attn_f2a_cohort_binding = _validate_decode_attn_wave_binding
    _validate_decode_attn_cohort_stage_maps = _validate_decode_attn_wave_stages
    _validate_decode_attn_a2f_cohort_phase = _validate_decode_attn_a2f_wave_phase
    _prepare_decode_attn_batch_cohort_phase = _prepare_decode_attn_batch_phase
    _apply_decode_attn_batch_cohort_phase = _apply_decode_attn_batch_phase
    _commit_decode_attn_batch_cohort_phases = _commit_decode_attn_batch_phases
    _set_decode_attn_batch_cohort_phase = _set_decode_attn_batch_phase

    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass
