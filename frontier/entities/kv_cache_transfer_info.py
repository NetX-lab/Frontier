from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.entities import Batch


@dataclass
class KVCacheTransferInfo:
    """Information about a KV cache transfer operation."""

    batch: "Batch"
    source_cluster_type: ClusterType
    target_cluster_type: ClusterType
    source_replica_id: int
    source_replica_local_id: Optional[int]
    source_batch_stage_id: int
    kv_cache_size_bytes: int
    transfer_time_ms: float
    transfer_start_time: float
    transfer_end_time: Optional[float] = None
    enable_compression: bool = False
    compression_ratio: float = 1.0
    enable_latency_hiding: bool = False
    transfer_protocol: str = "rdma"
    transfer_requests: bool = False
    target_replica_id: Optional[int] = None
    target_replica_local_id: Optional[int] = None
    target_batch_stage_id: Optional[int] = None
    kv_connector: str = "none"
    payload_kind: Optional[str] = None
    source_prompt_common_blocks: Optional[int] = None
    target_reserved_common_blocks: Optional[int] = None
    target_ready_time: Optional[float] = None
    source_release_time: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            type(self.source_batch_stage_id) is not int
            or self.source_batch_stage_id < 0
        ):
            raise ValueError(
                "source_batch_stage_id must be a non-negative int, "
                f"got {self.source_batch_stage_id!r}"
            )
        if self.target_batch_stage_id is not None and (
            type(self.target_batch_stage_id) is not int
            or self.target_batch_stage_id < 0
        ):
            raise ValueError(
                "target_batch_stage_id must be a non-negative int or None, "
                f"got {self.target_batch_stage_id!r}"
            )
        if self.transfer_end_time is None:
            self.transfer_end_time = self.transfer_start_time + (self.transfer_time_ms * 1e-3)
        if self.kv_connector == "nixl":
            if self.payload_kind != "structural_whole_prompt_pages_per_worker_lane":
                raise ValueError("NIXL transfer requires the structural payload label")
            if (
                type(self.source_prompt_common_blocks) is not int
                or self.source_prompt_common_blocks <= 0
            ):
                raise ValueError("NIXL transfer requires source prompt common blocks")
            if (
                type(self.target_reserved_common_blocks) is not int
                or self.target_reserved_common_blocks
                < self.source_prompt_common_blocks
            ):
                raise ValueError("NIXL target reservation does not cover its payload")
            if self.target_replica_id is None:
                raise ValueError("NIXL transfer requires a selected target replica")

    @property
    def is_completed(self) -> bool:
        return self.transfer_end_time is not None

    @property
    def effective_data_size_bytes(self) -> int:
        if self.enable_compression:
            return int(self.kv_cache_size_bytes / self.compression_ratio)
        return self.kv_cache_size_bytes

    def to_dict(self) -> dict:
        payload = {
            "batch_id": self.batch.id,
            "batch_global_id": self.batch.global_id,
            "source_cluster_type": self.source_cluster_type.name,
            "target_cluster_type": self.target_cluster_type.name,
            "source_replica_id": self.source_replica_id,
            "source_replica_local_id": self.source_replica_local_id,
            "source_batch_stage_id": self.source_batch_stage_id,
            "target_replica_id": self.target_replica_id,
            "target_replica_local_id": self.target_replica_local_id,
            "target_batch_stage_id": self.target_batch_stage_id,
            "kv_cache_size_bytes": self.kv_cache_size_bytes,
            "effective_data_size_bytes": self.effective_data_size_bytes,
            "transfer_time_ms": self.transfer_time_ms,
            "transfer_start_time": self.transfer_start_time,
            "transfer_end_time": self.transfer_end_time,
            "enable_compression": self.enable_compression,
            "compression_ratio": self.compression_ratio,
            "enable_latency_hiding": self.enable_latency_hiding,
            "transfer_protocol": self.transfer_protocol,
            "transfer_requests": self.transfer_requests,
        }
        if self.kv_connector == "nixl":
            payload.update(
                {
                    "kv_connector": self.kv_connector,
                    "payload_kind": self.payload_kind,
                    "source_prompt_common_blocks": self.source_prompt_common_blocks,
                    "target_reserved_common_blocks": self.target_reserved_common_blocks,
                    "target_ready_time": self.target_ready_time,
                    "source_release_time": self.source_release_time,
                }
            )
        return payload
