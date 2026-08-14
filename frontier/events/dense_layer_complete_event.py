from typing import List, TYPE_CHECKING

from frontier.events.base_event import BaseEvent
from frontier.metrics import MetricsStore
from frontier.types import ClusterType, EventType

if TYPE_CHECKING:
    from frontier.scheduler import BaseGlobalScheduler


class DenseLayerCompleteEvent(BaseEvent):
    """Complete one dense FFN layer without entering an EP collective."""

    def __init__(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch,
        dp_id: int,
        layer_id: int,
        phase: str,
        cluster_type: ClusterType,
    ) -> None:
        if phase not in ("prefill", "decode"):
            raise ValueError(
                "DenseLayerCompleteEvent phase must be 'prefill' or 'decode'"
            )
        if cluster_type not in (
            ClusterType.MONOLITHIC,
            ClusterType.PREFILL,
            ClusterType.DECODE,
        ):
            raise ValueError(
                "DenseLayerCompleteEvent is only valid for shared full-model clusters"
            )
        super().__init__(time, EventType.DENSE_LAYER_COMPLETE)
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch = batch
        self._dp_id = dp_id
        self._layer_id = layer_id
        self._phase = phase
        self._cluster_type = cluster_type

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: MetricsStore,
    ) -> List[BaseEvent]:
        cluster_scheduler = scheduler.get_cluster_scheduler(self._cluster_type)
        return cluster_scheduler.on_dense_layer_complete(
            self.time,
            self._replica_id,
            self._stage_id,
            self._batch,
            self._dp_id,
            self._layer_id,
            self._phase,
            metrics_store,
        )

    def get_target_cluster(self) -> ClusterType:
        return self._cluster_type

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "cluster_type": self._cluster_type.name,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id,
            "dp_id": self._dp_id,
            "layer_id": self._layer_id,
            "phase": self._phase,
            "protocol": "FULL_STAGE_WORLD",
        }
