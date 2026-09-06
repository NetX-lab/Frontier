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
from frontier.scheduler.utils.ep_wave_schedule import schedule_layer_wave
from frontier.scheduler.utils.ep_wave import prepare_moe_wave_from_inputs
from frontier.scheduler.utils.layer_workload import materialize_layer_workload
from frontier.scheduler.utils.layer_admission import transition_layer_admission
from frontier.scheduler.utils.m2n_events import build_aggregated_batch_transfer_events
from frontier.scheduler.utils.pdaf_return import (
    release_ready_return_round,
    enqueue_return_round,
)
from frontier.scheduler.utils.batch_builders import (
    build_ep_lane_batch,
    build_virtual_global_batch,
    create_ep_batch_group,
)
from frontier.scheduler.utils.pdaf_wave_validation import validate_wave_stages, validate_a2f_wave_phase
from frontier.scheduler.utils.forward_step_admission import promote_to_ep_wave, restore_full_stage_owners
from frontier.scheduler.utils.stage_wakeup import build_stage_wakeup_events
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
from frontier.scheduler.utils.attention_transfer_state import (
    AttentionTransferState,
    initialize_attention_transfer_state,
)
from frontier.scheduler.utils.kv_arrival import (
    handle_decode_arrival,
    handle_decode_attn_arrival,
)
from frontier.scheduler.utils.m2n_arrival import (
    route_m2n_arrival,
    handle_decode_attn_arrival,
    handle_decode_ffn_arrival,
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
from frontier.scheduler.utils.ffn_state import map_source_replica_to_target
from frontier.scheduler.utils.stage_contexts import build_stage_execution_contexts
from frontier.scheduler.utils.batch_ids import attention_batch_id, decode_sync_id
from frontier.scheduler.utils.prefix_cache import validate_prefix_cache_config
from frontier.scheduler.utils.dense_metrics import (
    complete_dense_layer,
    build_prefill_metrics_execution_time,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
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
        return format_ep_trace_identity(identity)

    @staticmethod
    def _log_ep_workload_trace(**kwargs) -> None:
        kwargs["format_identity"] = format_ep_trace_identity
        ep_trace.log_workload_trace(**kwargs)

    @staticmethod
    def _log_ep_wave_end_trace(**kwargs) -> None:
        kwargs["format_identity"] = format_ep_trace_identity
        ep_trace.log_wave_end_trace(**kwargs)

    @staticmethod
    def _log_ep_barrier_trace(**kwargs) -> None:
        kwargs["format_identity"] = format_ep_trace_identity
        ep_trace.log_barrier_trace(**kwargs)

    @staticmethod
    def _log_ep_conservation_trace(**kwargs) -> None:
        kwargs["format_identity"] = format_ep_trace_identity
        ep_trace.log_conservation_trace(**kwargs)

    @staticmethod
    def _get_shared_ep_phase_times_ms(
        execution_time,
        *,
        cluster_type: ClusterType,
        batch_id: int,
        layer_id: int,
        ep_id: int,
    ) -> tuple[float, float, float, float, float]:
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
        return map_source_replica_to_target(source_replica_ordinal, target_ffn_replica_ids)

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
        replica_config = getattr(self._config, "replica_config", None)
        self._stage_execution_contexts = build_stage_execution_contexts(
            cluster=self._cluster,
            cluster_type=self._cluster_type,
            replica_config=replica_config,
            replica_dp_size=getattr(
                self,
                "_replica_dp_size",
                getattr(replica_config, "attn_dp", 1) or 1,
            ),
        )

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

        from frontier.scheduler.utils.replica_state import initialize_replica_schedulers

        initialize_replica_schedulers(self, request_generator_config, logger)
        self._request_queue = []
        # Sync completion is tracked per concrete batch event.  A cohort ID is
        # a reusable lane-local hint, so it cannot by itself identify a
        # duplicate after an idle-placeholder wave has closed.
        self._forward_sync_state = ForwardSyncState()
        self._bind_forward_sync_state_views()

        # Initialize specialized queues for PD+AF disaggregation
        if self._cluster_type == ClusterType.DECODE_ATTN:
            initialize_attention_transfer_state(self)
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
        """Wake queued sibling lanes after a shared stage owner releases."""

        return build_stage_wakeup_events(
            self,
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            exclude_replica_local_id=exclude_replica_local_id,
        )

    def make_attention_dp_batch_global_id(
        self,
        replica_id: int,
        replica_local_id: int | None,
        lane_batch_counter: int,
    ) -> int:
        lane_count = int(getattr(self, "_replica_dp_size", 1) or 1)
        return attention_batch_id(replica_id, replica_local_id, lane_batch_counter, lane_count)

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
        """Switch a shared batch's active parent scope at a layer boundary."""

        return transition_layer_admission(
            self,
            batch,
            stage_id=stage_id,
            layer_id=layer_id,
            operation_kind=operation_kind,
            scope=scope,
            participant_ep_ids=participant_ep_ids,
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

        return decode_sync_id(replica_local_id, lane_decode_sync_counter, lane_count)

    def _get_decode_target_cluster(self) -> ClusterType:
        return (
            ClusterType.DECODE
            if ClusterType.DECODE in self._available_clusters
            else ClusterType.DECODE_ATTN
        )

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
            resolve_collective_kind=resolve_ep_collective_kind,
            prepare_timing=prepare_combine_timing,
        )

    def _handle_ep_alltoall_combine_ready(
        self, time: float, replica_id: int, stage_id: int, batch, ep_id: int
    ):
        """Compatibility adapter for direct private handler callers."""
        from frontier.scheduler.utils.ep_combine_ready import handle_combine_ready

        return handle_combine_ready(
            self,
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            ep_id=ep_id,
            resolve_collective_kind=resolve_ep_collective_kind,
            prepare_timing=prepare_combine_timing,
        )

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
        """Create M2N transfer events for an aggregated return batch."""

        return build_aggregated_batch_transfer_events(
            self,
            batch,
            current_time,
            source_replica_id=source_replica_id,
            source_replica_local_id=source_replica_local_id,
        )

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

        return materialize_layer_workload(
            scheduler=self,
            batch=batch,
            target_replica_id=target_replica_id,
            global_layer_id=global_layer_id,
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

        return build_ep_lane_batch(
            source_batch=source_batch,
            layer_id=layer_id,
            ep_id=ep_id,
            layer_workload=layer_workload,
            create_batch_group=self._create_batch_group,
            cluster_type=self._cluster_type,
        )

    def _create_virtual_global_batch(
        self,
        sample_batch: Batch,
        total_global_tokens: int,
        total_global_prefill_tokens: int,
    ) -> Batch:
        """Create a predictor-only batch for one cross-DP token domain."""

        return build_virtual_global_batch(
            sample_batch,
            total_global_tokens,
            total_global_prefill_tokens,
        )

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

        return promote_to_ep_wave(
            self,
            source_batches=source_batches,
            replica_id=replica_id,
            stage_id=stage_id,
            layer_id=layer_id,
            step_id=cohort_id,
            participant_ep_ids=participant_ep_ids,
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

        return restore_full_stage_owners(
            self,
            source_batches=source_batches,
            replica_id=replica_id,
            stage_id=stage_id,
            layer_id=layer_id,
            operation_kind=operation_kind,
        )
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
        """Schedule one PREFILL layer wave through the shared utility."""

        return schedule_layer_wave(
            self,
            mode="prefill",
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            layer_id=layer_id,
            replica_local_id=replica_local_id,
            cohort_batches=cohort_batches,
        )

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
        """Schedule one DECODE layer wave through the shared utility."""

        return schedule_layer_wave(
            self,
            mode="decode",
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            layer_id=layer_id,
            replica_local_id=replica_local_id,
            cohort_batches=cohort_batches,
        )

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
        """Advance a dense layer through the existing phase-specific handler."""
        return complete_dense_layer(
            self,
            time=time,
            replica_id=replica_id,
            stage_id=stage_id,
            batch=batch,
            layer_id=layer_id,
            phase=phase,
            metrics_store=metrics_store,
        )

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
        model_config = getattr(getattr(self._config, "replica_config", None), "model_config", None)
        return build_prefill_metrics_execution_time(
            original_execution_time=original_execution_time,
            sample_batch=sample_batch,
            predictor=self._predictor,
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
        """Queue one DECODE_ATTN-to-FFN transfer and trigger promotion."""

        return handle_decode_ffn_arrival(self, time, batch, transfer_info, logger)

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
        """Validate the complete stage-local state for one DECODE_ATTN wave."""

        return validate_wave_stages(wave_state, context=context)

    def _validate_decode_attn_a2f_wave_phase(
        self,
        batch: Batch,
        *,
        layer_id: int,
        afd_stage_idx: int,
        context: str,
    ) -> None:
        """Validate the stage-local phase/layer of an A-to-F wave."""

        return validate_a2f_wave_phase(
            self,
            batch,
            layer_id=layer_id,
            afd_stage_idx=afd_stage_idx,
            context=context,
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
        return release_ready_return_round(self, round_key, expected_lanes, logger)

    def _enqueue_decode_attn_return_round(
        self,
        micro_batch: Batch,
        *,
        receipt: Dict[str, Any],
        logger,
    ) -> bool:
        return enqueue_return_round(self, micro_batch, receipt=receipt, logger=logger)

    def get_af_queue_size(self) -> int:
        """Get the size of the A→F request queue."""
        return len(getattr(self, "_af_batch_queue", ()))

    def clear_af_queue(self) -> List:
        """Clear and return all batches from A→F request queue."""
        queue = getattr(self, "_af_batch_queue", None)
        if queue is None:
            return []
        batches = list(queue)
        queue.clear()
        return batches

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
        return create_ep_batch_group(
            requests=requests,
            num_tokens=num_tokens,
            replica_id=replica_id,
            ep_id=ep_id,
            time=time,
            source_batch_ids=source_batch_ids,
            lane_workload=lane_workload,
            cluster_type=self._cluster_type,
            is_moe=self._config.replica_config.model_config.is_moe,
        )

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
