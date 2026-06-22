from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.entities import Batch


@dataclass
class M2NTransferInfo:
    """Information about a Memory-to-Memory transfer operation."""

    batch: "Batch"
    source_cluster_type: ClusterType
    target_cluster_type: ClusterType
    source_replica_id: int
    source_dp_id: int
    activation_size_bytes: int
    transfer_time_ms: float
    transfer_start_time: float
    transfer_end_time: Optional[float] = None
    enable_p2p_optimization: bool = True
    p2p_protocol: str = "nvlink"
    enable_compression: bool = False
    compression_ratio: float = 1.0
    enable_latency_hiding: bool = False
    layer_id: Optional[int] = None
    afd_stage_idx: Optional[int] = None
    pipeline_stage: Optional[str] = None
    target_ffn_replica_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.transfer_end_time is None:
            self.transfer_end_time = self.transfer_start_time + (self.transfer_time_ms * 1e-3)

        valid_transfers = [
            (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN),
            (ClusterType.DECODE_FFN, ClusterType.DECODE_ATTN),
        ]
        if (self.source_cluster_type, self.target_cluster_type) not in valid_transfers:
            raise ValueError(
                f"Invalid M2N transfer: {self.source_cluster_type.name} -> {self.target_cluster_type.name}. "
                "M2N transfers only support DECODE_ATTN <-> DECODE_FFN communication."
            )

        if self.pipeline_stage is None:
            if self.source_cluster_type == ClusterType.DECODE_ATTN:
                self.pipeline_stage = "attn_to_ffn"
            else:
                self.pipeline_stage = "ffn_to_attn"

    @property
    def is_completed(self) -> bool:
        return self.transfer_end_time is not None

    @property
    def effective_data_size_bytes(self) -> int:
        if self.enable_compression:
            return int(self.activation_size_bytes / self.compression_ratio)
        return self.activation_size_bytes

    @property
    def is_attn_to_ffn(self) -> bool:
        return self.source_cluster_type == ClusterType.DECODE_ATTN

    @property
    def is_ffn_to_attn(self) -> bool:
        return self.source_cluster_type == ClusterType.DECODE_FFN

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch.id,
            "batch_global_id": self.batch.global_id,
            "source_cluster_type": self.source_cluster_type.name,
            "target_cluster_type": self.target_cluster_type.name,
            "source_replica_id": self.source_replica_id,
            "source_dp_id": self.source_dp_id,
            "activation_size_bytes": self.activation_size_bytes,
            "effective_data_size_bytes": self.effective_data_size_bytes,
            "transfer_time_ms": self.transfer_time_ms,
            "transfer_start_time": self.transfer_start_time,
            "transfer_end_time": self.transfer_end_time,
            "enable_p2p_optimization": self.enable_p2p_optimization,
            "p2p_protocol": self.p2p_protocol,
            "enable_compression": self.enable_compression,
            "compression_ratio": self.compression_ratio,
            "enable_latency_hiding": self.enable_latency_hiding,
            "layer_id": self.layer_id,
            "afd_stage_idx": self.afd_stage_idx,
            "pipeline_stage": self.pipeline_stage,
            "target_ffn_replica_id": self.target_ffn_replica_id,
        }
