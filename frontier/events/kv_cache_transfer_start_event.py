from typing import TYPE_CHECKING, List, Optional

from frontier.events.base_event import BaseEvent
from frontier.types import ClusterType, EventType

if TYPE_CHECKING:
    from frontier.entities import Batch
    from frontier.metrics import MetricsStore
    from frontier.scheduler import BaseGlobalScheduler


class KVCacheTransferStartEvent(BaseEvent):
    """Event emitted when a KV cache transfer starts."""

    def __init__(
        self,
        time: float,
        source_replica_id: int,
        source_replica_local_id: Optional[int],
        target_cluster_type: ClusterType,
        batch: "Batch",
        kv_cache_size_bytes: int | None,
        transfer_time_ms: float | None,
        source_cluster_type: ClusterType = ClusterType.PREFILL,
        source_batch_stage_id: int | None = None,
        kv_connector: str = "none",
    ) -> None:
        super().__init__(time, EventType.KV_CACHE_TRANSFER_START)
        if (
            type(source_batch_stage_id) is not int
            or source_batch_stage_id < 0
        ):
            raise ValueError(
                "source_batch_stage_id must be a non-negative int, "
                f"got {source_batch_stage_id!r}"
            )
        self._source_replica_id = source_replica_id
        self._source_replica_local_id = source_replica_local_id
        self._source_cluster_type = source_cluster_type
        self._target_cluster_type = target_cluster_type
        self._batch = batch
        self._kv_cache_size_bytes = kv_cache_size_bytes
        self._transfer_time_ms = transfer_time_ms
        self._source_batch_stage_id = source_batch_stage_id
        self._kv_connector = kv_connector
        self._target_replica_id: int | None = None
        self._target_replica_local_id: int | None = None
        self._source_prompt_common_blocks: int | None = None
        self._target_reserved_common_blocks: int | None = None
        self._source_claimed = False
        self._started = False
        if kv_connector == "nixl":
            if len(batch.requests) != 1:
                raise ValueError("NIXL structural transfer requires one request")
        elif kv_cache_size_bytes is None or transfer_time_ms is None:
            raise ValueError("Legacy KV transfer requires size and duration")

    def get_request_id(self) -> int:
        if len(self._batch.requests) != 1:
            raise RuntimeError("Request identity requires a single-request transfer")
        return int(self._batch.requests[0].id)

    def try_reserve_target(self, target_replica_scheduler) -> bool:
        """Commit one real D reservation without making the request runnable."""
        if self._kv_connector != "nixl":
            raise RuntimeError("Only structural NIXL transfers reserve before start")
        if self._target_reserved_common_blocks is not None:
            return True
        request = self._batch.requests[0]
        reserved_blocks = target_replica_scheduler.reserve_kv_transfer_target(
            request
        )
        if reserved_blocks is None:
            return False
        self._target_reserved_common_blocks = int(reserved_blocks)
        return True

    def set_capacity_retry_time(self, time: float) -> None:
        """Move a newly reserved deferred start to its capacity-release time."""
        if self._started or self._target_reserved_common_blocks is None:
            raise RuntimeError("Only a reserved deferred NIXL start can be retried")
        self._time = time
        self._priority_number = self._get_priority_number()

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        from frontier.entities.kv_cache_transfer_info import KVCacheTransferInfo
        from frontier.events.kv_cache_transfer_end_event import KVCacheTransferEndEvent

        if self._kv_connector == "nixl":
            if self._started:
                raise RuntimeError(
                    f"NIXL transfer for request {self.get_request_id()} already started"
                )
            source_cluster_scheduler = scheduler.get_cluster_scheduler(
                self._source_cluster_type
            )
            source_config = source_cluster_scheduler._config
            predictor_config = source_config.execution_time_predictor_config
            if not bool(getattr(predictor_config, "enable_dummy_mode", False)):
                raise NotImplementedError(
                    "NIXL currently supports only dummy GLM structural PD"
                )
            replica_config = source_config.replica_config
            if int(replica_config.num_pipeline_stages) != 1:
                raise NotImplementedError("NIXL structural GLM PD supports PP1 only")
            source_replica_scheduler = source_cluster_scheduler.get_replica_scheduler(
                self._source_replica_id,
                self._source_replica_local_id,
            )
            if source_replica_scheduler._is_prefix_caching_enabled():
                raise NotImplementedError(
                    "NIXL structural GLM PD excludes source prefix caching"
                )
            request = self._batch.requests[0]
            self._source_prompt_common_blocks = (
                source_replica_scheduler.get_kv_transfer_source_prompt_common_blocks(
                    request
                )
            )
            source_capacity = source_replica_scheduler._kv_cache_capacity
            if (
                source_capacity is None
                or source_capacity.family_id != "dsa_attention"
                or source_capacity.is_grouped
                or source_capacity.block_size != 64
            ):
                raise NotImplementedError(
                    "NIXL currently supports only dummy GLM DSA block64 structural PD"
                )
            predictor = source_cluster_scheduler._kv_cache_transfer_predictor
            if not hasattr(
                predictor, "get_structural_nixl_transfer_info_for_request"
            ):
                raise NotImplementedError(
                    "NIXL structural PD requires component-aware analytical sizing"
                )

            target_cluster_scheduler = scheduler.get_cluster_scheduler(
                self._target_cluster_type
            )
            target_cluster_scheduler.validate_kv_transfer_target_scope(
                source_capacity, request
            )
            if not self._source_claimed:
                source_replica_scheduler.claim_kv_transfer_start(request)
                self._source_claimed = True

            if self._target_replica_id is None:
                (
                    self._target_replica_id,
                    self._target_replica_local_id,
                ) = target_cluster_scheduler.select_kv_transfer_target()
            target_replica_scheduler = target_cluster_scheduler.get_replica_scheduler(
                self._target_replica_id,
                self._target_replica_local_id,
            )
            has_older_pending = (
                self._target_reserved_common_blocks is None
                and target_replica_scheduler.has_pending_kv_transfer_starts()
            )
            if has_older_pending or not self.try_reserve_target(target_replica_scheduler):
                target_replica_scheduler.defer_kv_transfer_start(self)
                from frontier.events.replica_schedule_event import ReplicaScheduleEvent

                return [
                    ReplicaScheduleEvent(
                        self.time,
                        self._target_replica_id,
                        self._target_cluster_type,
                        self._target_replica_local_id,
                    )
                ]

            try:
                (
                    self._kv_cache_size_bytes,
                    self._transfer_time_ms,
                ) = predictor.get_structural_nixl_transfer_info_for_request(
                    request=request,
                    replica_config=replica_config,
                    cache_capacity=source_capacity,
                    source_prompt_common_blocks=self._source_prompt_common_blocks,
                    target_reserved_common_blocks=(
                        self._target_reserved_common_blocks
                    ),
                )
            except Exception:
                target_replica_scheduler.abort_kv_transfer_target_reservation(
                    request
                )
                source_replica_scheduler.abandon_kv_transfer_start_claim(request)
                self._source_claimed = False
                self._target_reserved_common_blocks = None
                raise

        transfer_info = KVCacheTransferInfo(
            batch=self._batch,
            source_cluster_type=self._source_cluster_type,
            target_cluster_type=self._target_cluster_type,
            source_replica_id=self._source_replica_id,
            source_replica_local_id=self._source_replica_local_id,
            source_batch_stage_id=self._source_batch_stage_id,
            kv_cache_size_bytes=self._kv_cache_size_bytes,
            transfer_time_ms=self._transfer_time_ms,
            transfer_start_time=self.time,
            target_replica_id=self._target_replica_id,
            target_replica_local_id=self._target_replica_local_id,
            kv_connector=self._kv_connector,
            payload_kind=(
                "structural_whole_prompt_pages_per_worker_lane"
                if self._kv_connector == "nixl"
                else None
            ),
            source_prompt_common_blocks=self._source_prompt_common_blocks,
            target_reserved_common_blocks=self._target_reserved_common_blocks,
        )
        if self._kv_connector == "nixl":
            self._started = True

        metrics_store.on_kv_cache_transfer_start(
            self.time,
            self._source_replica_id,
            self._source_replica_local_id,
            self._target_cluster_type,
            self._kv_cache_size_bytes,
            transfer_info,
        )

        for request in self._batch.requests:
            request.on_kv_cache_transfer_start(self.time)

        transfer_end_time = self.time + self._transfer_time_ms * 1e-3
        return [KVCacheTransferEndEvent(transfer_end_time, transfer_info)]

    def get_target_cluster(self) -> ClusterType:
        return self._target_cluster_type

    def to_dict(self) -> dict:
        payload = {
            "time": self.time,
            "event_type": self.event_type.name,
            "source_replica_id": self._source_replica_id,
            "source_cluster_type": self._source_cluster_type.name,
            "target_cluster_type": self._target_cluster_type.name,
            "batch_id": self._batch.id,
            "batch_global_id": self._batch.global_id,
            "source_batch_stage_id": self._source_batch_stage_id,
            "kv_cache_size_bytes": self._kv_cache_size_bytes,
            "transfer_time_ms": self._transfer_time_ms,
        }
        if self._kv_connector == "nixl":
            payload.update(
                {
                    "kv_connector": self._kv_connector,
                    "target_replica_id": self._target_replica_id,
                    "target_replica_local_id": self._target_replica_local_id,
                    "payload_kind": "structural_whole_prompt_pages_per_worker_lane",
                    "source_prompt_common_blocks": self._source_prompt_common_blocks,
                    "target_reserved_common_blocks": self._target_reserved_common_blocks,
                }
            )
        return payload
