from typing import TYPE_CHECKING, List

from frontier.events.base_event import BaseEvent
from frontier.types import ClusterType, EventType

if TYPE_CHECKING:
    from frontier.entities import KVCacheTransferInfo
    from frontier.metrics import MetricsStore
    from frontier.scheduler import BaseGlobalScheduler


class KVCacheTransferEndEvent(BaseEvent):
    """Event emitted when a KV cache transfer completes."""

    def __init__(
        self,
        time: float,
        transfer_info: "KVCacheTransferInfo",
    ) -> None:
        super().__init__(time, EventType.KV_CACHE_TRANSFER_END)
        self._transfer_info = transfer_info
        self._transfer_info.transfer_end_time = time

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        transfer_duration_s = self.time - self._transfer_info.transfer_start_time
        transfer_duration_ms = transfer_duration_s * 1e3
        metrics_store.on_kv_cache_transfer_end(
            self.time,
            transfer_duration_ms,
            self._transfer_info.kv_cache_size_bytes,
            self._transfer_info.target_cluster_type,
            self._transfer_info,
        )

        batch = self._transfer_info.batch
        for request in batch.requests:
            request.on_kv_cache_transfer_complete(self.time, transfer_duration_s)

        target_cluster_scheduler = scheduler.get_cluster_scheduler(
            self._transfer_info.target_cluster_type
        )
        arrival_events = target_cluster_scheduler.on_kv_cache_arrival(
            self.time,
            self._transfer_info.batch,
            self._transfer_info,
        )

        if self._transfer_info.source_cluster_type == ClusterType.PREFILL:
            from frontier.events.replica_schedule_event import ReplicaScheduleEvent

            source_cluster_scheduler = scheduler.get_cluster_scheduler(
                self._transfer_info.source_cluster_type
            )
            source_replica_scheduler = source_cluster_scheduler.get_dp_replica_scheduler(
                self._transfer_info.source_replica_id,
                self._transfer_info.source_dp_id,
            )
            source_replica_scheduler.complete_kv_transfer_for_requests(batch.requests)

            memory_usage_percent = source_replica_scheduler.memory_usage_percent
            metrics_store.on_replica_schedule(
                self.time,
                self._transfer_info.source_replica_id,
                memory_usage_percent,
                self._transfer_info.source_cluster_type,
                dp_id=self._transfer_info.source_dp_id,
            )

            if source_replica_scheduler.should_schedule_after_kv_transfer_completion():
                source_cluster_logical_time = scheduler.get_cluster_logical_time(
                    self._transfer_info.source_cluster_type
                )
                source_reschedule_time = max(self.time, source_cluster_logical_time)
                arrival_events.append(
                    ReplicaScheduleEvent(
                        source_reschedule_time,
                        self._transfer_info.source_replica_id,
                        self._transfer_info.source_cluster_type,
                        self._transfer_info.source_dp_id,
                    )
                )

        return arrival_events

    def get_target_cluster(self) -> ClusterType:
        return self._transfer_info.target_cluster_type

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type.name,
            "batch_id": self._transfer_info.batch.id,
            "batch_global_id": self._transfer_info.batch.global_id,
            "source_cluster_type": self._transfer_info.source_cluster_type.name,
            "target_cluster_type": self._transfer_info.target_cluster_type.name,
            "source_replica_id": self._transfer_info.source_replica_id,
            "kv_cache_size_bytes": self._transfer_info.kv_cache_size_bytes,
            "transfer_time_ms": self._transfer_info.transfer_time_ms,
            "transfer_start_time": self._transfer_info.transfer_start_time,
            "transfer_end_time": self._transfer_info.transfer_end_time,
        }
