from typing import List, Optional, TYPE_CHECKING

from frontier.events.base_event import BaseEvent
from frontier.types import EventType, ClusterType

if TYPE_CHECKING:
    from frontier.entities import Batch, M2NTransferInfo
    from frontier.metrics import MetricsStore
    from frontier.scheduler import BaseGlobalScheduler


class M2NTransferStartEvent(BaseEvent):

    def __init__(
        self,
        time: float,
        source_replica_id: int,
        source_replica_local_id: Optional[int],
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        batch: "Batch",
        activation_size_bytes: int,
        transfer_time_ms: float,
        layer_id: int = None,
        afd_stage_idx: Optional[int] = None,
        source_execution_replica_id: Optional[int] = None,
        source_execution_replica_local_id: Optional[int] = None,
        target_execution_replica_id: Optional[int] = None,
        target_execution_replica_local_id: Optional[int] = None,
    ):
        super().__init__(time, EventType.M2N_TRANSFER_START)

        self._source_replica_id = source_replica_id
        self._source_replica_local_id = source_replica_local_id
        self._source_cluster_type = source_cluster_type
        self._target_cluster_type = target_cluster_type
        self._batch = batch
        self._activation_size_bytes = activation_size_bytes
        self._transfer_time_ms = transfer_time_ms
        self._layer_id = layer_id
        self._source_execution_replica_id = source_execution_replica_id
        self._source_execution_replica_local_id = source_execution_replica_local_id
        self._target_execution_replica_id = target_execution_replica_id
        self._target_execution_replica_local_id = target_execution_replica_local_id
        if afd_stage_idx is None:
            afd_stage_idx = getattr(batch, "afd_stage_idx", None)
        if afd_stage_idx is None:
            raise ValueError("afd_stage_idx must be set for M2N transfer")
        self._afd_stage_idx = afd_stage_idx

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        from frontier.events.m2n_transfer_end_event import M2NTransferEndEvent
        from frontier.entities.m2n_transfer_info import M2NTransferInfo
        from frontier.entities.request import (
            validate_inter_cluster_transfer_request_cohort,
        )
        from frontier.logger import get_cluster_logger

        logger = get_cluster_logger(__name__, self._source_cluster_type.name)

        validate_inter_cluster_transfer_request_cohort(self._batch.requests)

        transfer_info = M2NTransferInfo(
            batch=self._batch,
            source_cluster_type=self._source_cluster_type,
            target_cluster_type=self._target_cluster_type,
            source_replica_id=self._source_replica_id,
            source_replica_local_id=self._source_replica_local_id,
            activation_size_bytes=self._activation_size_bytes,
            transfer_time_ms=self._transfer_time_ms,
            transfer_start_time=self.time,
            layer_id=self._layer_id,
            afd_stage_idx=self._afd_stage_idx,
            source_execution_replica_id=self._source_execution_replica_id,
            source_execution_replica_local_id=self._source_execution_replica_local_id,
            target_execution_replica_id=self._target_execution_replica_id,
            target_execution_replica_local_id=self._target_execution_replica_local_id,
        )

        for request in self._batch.requests:
            request.validate_inter_cluster_transfer_start(
                time=self.time,
                source_cluster=self._source_cluster_type,
                target_cluster=self._target_cluster_type,
                activation_size_bytes=self._activation_size_bytes,
            )

        request_ids = [req.id for req in self._batch.requests]
        pipeline_stage = "attn→ffn" if self._source_cluster_type == ClusterType.DECODE_ATTN else "ffn→attn"
        transfer_time_s = self._transfer_time_ms * 1e-3
        logger.info(
            f"M2N transfer started at {self.time:.3f}s: "
            f"requests {request_ids} {pipeline_stage} transfer, "
            f"size={self._activation_size_bytes} bytes, expected_duration={self._transfer_time_ms:.2f}ms, "
            f"will_arrive_at={self.time + transfer_time_s:.3f}s"
            f"{f', layer={self._layer_id}' if self._layer_id is not None else ''}"
            f"{f', afd_stage_idx={self._afd_stage_idx}' if self._afd_stage_idx is not None else ''}"
        )

        metrics_store.on_m2n_transfer_start(
            self.time,
            self._source_replica_id,
            self._source_cluster_type,
            self._target_cluster_type,
            self._activation_size_bytes,
            transfer_info,
        )

        for req in self._batch.requests:
            req.on_inter_cluster_transfer_start(
                time=self.time,
                source_cluster=self._source_cluster_type,
                target_cluster=self._target_cluster_type,
                activation_size_bytes=self._activation_size_bytes,
            )

        transfer_end_time = self.time + transfer_time_s
        transfer_end_event = M2NTransferEndEvent(
            time=transfer_end_time,
            transfer_info=transfer_info,
        )

        return [transfer_end_event]

    def get_target_cluster(self) -> ClusterType:
        return self._source_cluster_type

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type.name,
            "batch_id": self._batch.id,
            "batch_global_id": self._batch.global_id,
            "source_cluster_type": self._source_cluster_type.name,
            "target_cluster_type": self._target_cluster_type.name,
            "source_replica_id": self._source_replica_id,
            "activation_size_bytes": self._activation_size_bytes,
            "transfer_time_ms": self._transfer_time_ms,
            "layer_id": self._layer_id,
            "afd_stage_idx": self._afd_stage_idx,
            "source_execution_replica_id": self._source_execution_replica_id,
            "source_execution_replica_local_id": self._source_execution_replica_local_id,
            "target_execution_replica_id": self._target_execution_replica_id,
            "target_execution_replica_local_id": self._target_execution_replica_local_id,
        }
