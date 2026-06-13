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
    source_dp_id: int
    kv_cache_size_bytes: int
    transfer_time_ms: float
    transfer_start_time: float
    transfer_end_time: Optional[float] = None
    enable_compression: bool = False
    compression_ratio: float = 1.0
    enable_latency_hiding: bool = False
    transfer_protocol: str = "rdma"
    transfer_requests: bool = False

    def __post_init__(self) -> None:
        if self.transfer_end_time is None:
            self.transfer_end_time = self.transfer_start_time + (self.transfer_time_ms * 1e-3)

    @property
    def is_completed(self) -> bool:
        return self.transfer_end_time is not None

    @property
    def effective_data_size_bytes(self) -> int:
        if self.enable_compression:
            return int(self.kv_cache_size_bytes / self.compression_ratio)
        return self.kv_cache_size_bytes

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch.id,
            "batch_global_id": self.batch.global_id,
            "source_cluster_type": self.source_cluster_type.name,
            "target_cluster_type": self.target_cluster_type.name,
            "source_replica_id": self.source_replica_id,
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
