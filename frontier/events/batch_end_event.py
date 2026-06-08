from typing import List

from frontier.entities import Batch
from frontier.events import BaseEvent
from frontier.logger import init_logger
from frontier.metrics import MetricsStore
from frontier.scheduler import BaseGlobalScheduler
from frontier.types import EventType, ClusterType
from frontier.config.config import DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR

logger = init_logger(__name__)


class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch, cluster_type: ClusterType, dp_id: int):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch
        self._cluster_type = cluster_type
        self._dp_id = dp_id

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from frontier.events.replica_schedule_event import ReplicaScheduleEvent

        if self._cluster_type != ClusterType.MONOLITHIC:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

        # Get the appropriate cluster scheduler for this cluster-internal event
        cluster_scheduler = scheduler.get_cluster_scheduler(self._cluster_type)
        replica_scheduler = cluster_scheduler.get_dp_replica_scheduler(self._replica_id, self._dp_id)

        self._batch.on_batch_end(self.time)
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time,
            self._batch,
            self._replica_id,
            memory_usage_percent,
            self._cluster_type,
            self._dp_id,
        )

        return [ReplicaScheduleEvent(self.time, self._replica_id, self._cluster_type, self._dp_id)]

    def _handle_decode_attn_completion(self, scheduler, cluster_scheduler, replica_scheduler):
        """Disaggregated decode-attn completion is not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _handle_decode_ffn_completion(self, scheduler, cluster_scheduler, replica_scheduler):
        """Disaggregated decode-ffn completion is not included in this release."""
        raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _get_current_layer_id_from_batch(self, batch: "Batch") -> int:
        """
        Get the current layer ID from the batch's requests.

        In PD+AF disaggregation, all requests in a batch should be at the same layer.
        We use the completed_layer_count from the first request as the representative.

        Args:
            batch: The batch to get layer ID from

        Returns:
            Current layer ID (0-based)
        """
        if not batch.requests:
            raise ValueError("_get_current_layer_id_from_batch: batch.requests is empty")

        # All requests in the batch should be at the same layer in PD+AF
        # Use the first request as representative
        # Completed_layer_count starts from 0
        first_request = batch.requests[0]
        return first_request.completed_layer_count

    def _advance_batch_layer_completion(self, batch: "Batch") -> None:
        """
        Advance layer completion for all requests in the batch.

        This method is called when a batch completes processing in decode-ffn cluster
        and is about to transfer back to decode-attn cluster for the next layer.

        Args:
            batch: The batch to advance layer completion for
        """
        for request in batch.requests:
            # Advance layer completion by 1 (attention + FFN = 1 complete layer)
            request.advance_decode_layer(num_layers_completed=1)

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
            "cluster_type": self._cluster_type.name,
            "replica_id": self._replica_id,
            "dp_id": self._dp_id,
        }
