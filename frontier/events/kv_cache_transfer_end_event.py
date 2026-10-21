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

    def _validate_nixl_read(
        self, request, source_cluster_scheduler, source_replica_scheduler,
        target_cluster_scheduler, target_replica_scheduler,
    ) -> None:
        target_replica_scheduler.validate_kv_transfer_target_completion(
            request, self._transfer_info.target_reserved_common_blocks,
        )
        source_blocks = source_replica_scheduler.get_kv_transfer_source_prompt_common_blocks(request)
        if source_blocks != self._transfer_info.source_prompt_common_blocks:
            raise RuntimeError(
                f"Request {request.id} PREFILL ownership changed from "
                f"{self._transfer_info.source_prompt_common_blocks} to "
                f"{source_blocks} prompt blocks"
            )
        source_capacity = source_replica_scheduler._kv_cache_capacity
        predictor = source_cluster_scheduler._kv_cache_transfer_predictor
        transfer_args = dict(
            request=request,
            replica_config=source_cluster_scheduler._config.replica_config,
            cache_capacity=source_capacity,
            source_prompt_common_blocks=source_blocks,
            target_reserved_common_blocks=self._transfer_info.target_reserved_common_blocks,
        )
        if self._transfer_info.nixl_coverage is None:
            expected_payload, _ = predictor.get_structural_nixl_transfer_info_for_request(**transfer_args)
        else:
            expected_coverage = predictor.get_measured_nixl_coverage(
                request=request,
                source_replica_config=source_cluster_scheduler._config.replica_config,
                target_replica_config=target_cluster_scheduler._config.replica_config,
                cache_capacity=source_capacity,
                cost_provider=source_cluster_scheduler._predictor,
                target_cost_provider=target_cluster_scheduler._predictor,
            )
            if expected_coverage != self._transfer_info.nixl_coverage:
                raise RuntimeError("Measured NIXL coverage changed during READ")
            expected_payload, expected_time_ms = predictor.get_measured_nixl_transfer_info_for_request(
                **transfer_args, coverage=expected_coverage,
                cost_provider=source_cluster_scheduler._predictor,
            )
            if (
                self._transfer_info.transfer_time_ms != expected_time_ms
                or self.time != self._transfer_info.transfer_start_time + expected_time_ms * 1e-3
            ):
                raise RuntimeError("Measured NIXL READ-to-ready duration changed")
        if (
            self._transfer_info.transfer_payload != expected_payload
            or self._transfer_info.kv_cache_size_bytes != expected_payload.transfer_bytes_per_rank
        ):
            raise RuntimeError(
                f"Request {request.id} NIXL payload changed from "
                f"{expected_payload.transfer_bytes_per_rank} to "
                f"{self._transfer_info.kv_cache_size_bytes} bytes"
            )
        source_replica_scheduler.validate_kv_transfer_source_release(request)

    def handle_event(
        self,
        scheduler: "BaseGlobalScheduler",
        metrics_store: "MetricsStore",
    ) -> List[BaseEvent]:
        batch = self._transfer_info.batch
        target_cluster_scheduler = scheduler.get_cluster_scheduler(
            self._transfer_info.target_cluster_type
        )
        source_replica_scheduler = None
        target_replica_scheduler = None
        is_structural_nixl = (
            getattr(self._transfer_info, "kv_connector", "none") == "nixl"
        )
        if is_structural_nixl:
            if len(batch.requests) != 1:
                raise RuntimeError("NIXL transfer end requires one request")
            request = batch.requests[0]
            target_replica_scheduler = target_cluster_scheduler.get_replica_scheduler(
                self._transfer_info.target_replica_id,
                self._transfer_info.target_replica_local_id,
            )
            source_cluster_scheduler = scheduler.get_cluster_scheduler(
                self._transfer_info.source_cluster_type
            )
            source_replica_scheduler = source_cluster_scheduler.get_replica_scheduler(
                self._transfer_info.source_replica_id,
                self._transfer_info.source_replica_local_id,
            )
            try:
                self._validate_nixl_read(
                    request, source_cluster_scheduler, source_replica_scheduler,
                    target_cluster_scheduler, target_replica_scheduler,
                )
            except Exception:
                if self._transfer_info.nixl_coverage is not None:
                    target_replica_scheduler.abort_kv_transfer_target_reservation(request)
                    target_cluster_scheduler.abandon_kv_transfer_target_selection(request)
                    source_replica_scheduler.abandon_kv_transfer_start_claim(request)
                raise

        transfer_duration_s = self.time - self._transfer_info.transfer_start_time
        transfer_duration_ms = transfer_duration_s * 1e3
        metrics_store.on_kv_cache_transfer_end(
            self.time,
            transfer_duration_ms,
            self._transfer_info.kv_cache_size_bytes,
            self._transfer_info.target_cluster_type,
            self._transfer_info,
        )

        for request in batch.requests:
            if not is_structural_nixl or self._transfer_info.nixl_coverage is None:
                request.on_kv_cache_transfer_complete(self.time, transfer_duration_s)
            else:
                request.on_kv_cache_transfer_complete(
                    self.time, transfer_duration_s,
                    num_computed_tokens=self._transfer_info.nixl_coverage.num_computed_tokens,
                )

        if is_structural_nixl:
            from frontier.logger import get_cluster_logger
            from frontier.scheduler.utils.kv_arrival import (
                handle_reserved_decode_arrival,
            )

            arrival_events = handle_reserved_decode_arrival(
                target_replica_scheduler,
                self.time,
                batch,
                self._transfer_info,
                get_cluster_logger(__name__, ClusterType.DECODE.name),
            )
        else:
            arrival_events = target_cluster_scheduler.on_kv_cache_arrival(
                self.time,
                batch,
                self._transfer_info,
            )

        if self._transfer_info.source_cluster_type == ClusterType.PREFILL:
            from frontier.events.replica_schedule_event import ReplicaScheduleEvent

            if source_replica_scheduler is None:
                source_cluster_scheduler = scheduler.get_cluster_scheduler(
                    self._transfer_info.source_cluster_type
                )
                source_replica_scheduler = (
                    source_cluster_scheduler.get_replica_scheduler(
                        self._transfer_info.source_replica_id,
                        self._transfer_info.source_replica_local_id,
                    )
                )
            source_replica_scheduler.complete_kv_transfer_for_requests(batch.requests)
            if is_structural_nixl:
                self._transfer_info.source_release_time = self.time

            memory_usage_percent = source_replica_scheduler.memory_usage_percent
            metrics_store.on_replica_schedule(
                self.time,
                self._transfer_info.source_replica_id,
                memory_usage_percent,
                self._transfer_info.source_cluster_type,
                replica_local_id=self._transfer_info.source_replica_local_id,
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
                        self._transfer_info.source_replica_local_id,
                    )
                )

        return arrival_events

    def get_target_cluster(self) -> ClusterType:
        return self._transfer_info.target_cluster_type

    def to_dict(self) -> dict:
        payload = {
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
        if getattr(self._transfer_info, "kv_connector", "none") == "nixl":
            payload.update(
                {
                    "kv_connector": self._transfer_info.kv_connector,
                    "target_replica_id": self._transfer_info.target_replica_id,
                    "target_replica_local_id": self._transfer_info.target_replica_local_id,
                    "payload_kind": self._transfer_info.payload_kind,
                    "source_prompt_common_blocks": self._transfer_info.source_prompt_common_blocks,
                    "target_reserved_common_blocks": self._transfer_info.target_reserved_common_blocks,
                    "target_ready_time": self._transfer_info.target_ready_time,
                    "source_release_time": self._transfer_info.source_release_time,
                }
            )
        return payload
