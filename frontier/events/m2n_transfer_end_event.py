from typing import List, TYPE_CHECKING

from frontier.events.base_event import BaseEvent
from frontier.types import EventType, ClusterType

if TYPE_CHECKING:
    from frontier.entities import M2NTransferInfo
    from frontier.metrics import MetricsStore
    from frontier.scheduler import BaseGlobalScheduler


class M2NTransferEndEvent(BaseEvent):

    def __init__(
        self,
        time: float,
        transfer_info: "M2NTransferInfo",
    ):
        super().__init__(time, EventType.M2N_TRANSFER_END)
        self._transfer_info = transfer_info

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        from frontier.entities.request import (
            validate_inter_cluster_transfer_request_cohort,
        )
        from frontier.logger import get_cluster_logger

        self._transfer_info.validate_direction()
        batch = self._transfer_info.batch
        validate_inter_cluster_transfer_request_cohort(batch.requests)
        target_cluster_scheduler = scheduler.get_cluster_scheduler(
            self._transfer_info.target_cluster_type
        )
        target_cluster_scheduler.preflight_m2n_arrival(
            self._transfer_info.batch,
            self._transfer_info,
        )
        for request in batch.requests:
            request.validate_inter_cluster_transfer_end(
                time=self.time,
                source_cluster=self._transfer_info.source_cluster_type,
                target_cluster=self._transfer_info.target_cluster_type,
                activation_size_bytes=self._transfer_info.activation_size_bytes,
            )
        self._transfer_info.transfer_end_time = self.time

        logger = get_cluster_logger(__name__, self._transfer_info.target_cluster_type.name)

        transfer_duration_s = self.time - self._transfer_info.transfer_start_time
        transfer_duration_ms = transfer_duration_s * 1e3
        metrics_store.on_m2n_transfer_end(
            self.time,
            transfer_duration_ms,
            self._transfer_info.activation_size_bytes,
            self._transfer_info.source_cluster_type,
            self._transfer_info.target_cluster_type,
            self._transfer_info,
        )

        is_attn_to_ffn = self._transfer_info.is_attn_to_ffn
        for request in batch.requests:
            request.on_m2n_transfer_complete(transfer_duration_s, is_attn_to_ffn)

        request_ids = [req.id for req in batch.requests]
        pipeline_stage = "attn→ffn" if is_attn_to_ffn else "ffn→attn"
        logger.info(
            f"M2N transfer completed at {self.time:.3f}s: "
            f"requests {request_ids} {pipeline_stage} → {self._transfer_info.target_cluster_type.name} cluster, "
            f"transfer_time={transfer_duration_ms:.2f}ms, size={self._transfer_info.activation_size_bytes} bytes"
            f"{f', layer={self._transfer_info.layer_id}' if self._transfer_info.layer_id is not None else ''}"
        )

        for req in self._transfer_info.batch.requests:
            req.on_inter_cluster_transfer_end(
                time=self.time,
                source_cluster=self._transfer_info.source_cluster_type,
                target_cluster=self._transfer_info.target_cluster_type,
                activation_size_bytes=self._transfer_info.activation_size_bytes,
            )

        try:
            if (not self._transfer_info.is_attn_to_ffn) and self._transfer_info.target_cluster_type == ClusterType.DECODE_ATTN:
                b = self._transfer_info.batch
                req_ids = [r.id for r in b.requests]
                logger.info(
                    f"[M2N][F2A][ARRIVE][PRE] batch_id={b.id} reqs={req_ids} "
                    f"batch_global_id={getattr(b, 'global_id', '?')} "
                    f"decode_attn_orig=(replica={getattr(b, 'decode_attn_original_replica_id', '?')},"
                    f"dp={getattr(b, 'decode_attn_original_replica_local_id', '?')})"
                )
        except Exception as _e:
            logger.debug(f"[M2N][F2A][ARRIVE] pre-log error: {_e}")

        arrival_events = target_cluster_scheduler.on_m2n_arrival(
            self.time,
            self._transfer_info.batch,
            self._transfer_info,
        )

        try:
            if self._transfer_info.target_cluster_type == ClusterType.DECODE_ATTN:
                qsize = len(getattr(target_cluster_scheduler, "_af_batch_queue", []))
                logger.info(f"[M2N][F2A][ARRIVE][POST] af_queue_size={qsize}")
        except Exception as _e:
            logger.debug(f"[M2N][F2A][ARRIVE] post-log error: {_e}")

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
            "activation_size_bytes": self._transfer_info.activation_size_bytes,
            "transfer_time_ms": self._transfer_info.transfer_time_ms,
            "transfer_start_time": self._transfer_info.transfer_start_time,
            "transfer_end_time": self._transfer_info.transfer_end_time,
            "layer_id": self._transfer_info.layer_id,
            "pipeline_stage": self._transfer_info.pipeline_stage,
        }
