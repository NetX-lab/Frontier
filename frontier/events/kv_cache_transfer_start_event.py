from typing import TYPE_CHECKING, List

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
        source_dp_id: int,
        target_cluster_type: ClusterType,
        batch: "Batch",
        kv_cache_size_bytes: int,
        transfer_time_ms: float,
        source_cluster_type: ClusterType = ClusterType.PREFILL,
    ) -> None:
        super().__init__(time, EventType.KV_CACHE_TRANSFER_START)
        self._source_replica_id = source_replica_id
        self._source_dp_id = source_dp_id
        self._source_cluster_type = source_cluster_type
        self._target_cluster_type = target_cluster_type
        self._batch = batch
        self._kv_cache_size_bytes = kv_cache_size_bytes
        self._transfer_time_ms = transfer_time_ms

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        from frontier.entities.kv_cache_transfer_info import KVCacheTransferInfo
        from frontier.events.kv_cache_transfer_end_event import KVCacheTransferEndEvent

        transfer_info = KVCacheTransferInfo(
            batch=self._batch,
            source_cluster_type=self._source_cluster_type,
            target_cluster_type=self._target_cluster_type,
            source_replica_id=self._source_replica_id,
            source_dp_id=self._source_dp_id,
            kv_cache_size_bytes=self._kv_cache_size_bytes,
            transfer_time_ms=self._transfer_time_ms,
            transfer_start_time=self.time,
        )

        metrics_store.on_kv_cache_transfer_start(
            self.time,
            self._source_replica_id,
            self._source_dp_id,
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
        return {
            "time": self.time,
            "event_type": self.event_type.name,
            "source_replica_id": self._source_replica_id,
            "source_cluster_type": self._source_cluster_type.name,
            "target_cluster_type": self._target_cluster_type.name,
            "batch_id": self._batch.id,
            "batch_global_id": self._batch.global_id,
            "kv_cache_size_bytes": self._kv_cache_size_bytes,
            "transfer_time_ms": self._transfer_time_ms,
        }
