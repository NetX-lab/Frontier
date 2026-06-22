from dataclasses import dataclass, field
from typing import Optional

from frontier.config.base_poly_config import BasePolyConfig
from frontier.types import KVCacheTransferType


def _require_positive(value: float | int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0, got {value}")


def _require_non_negative(value: float | int, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")


def _require_optional_positive(value: Optional[int], field_name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{field_name} must be > 0 when set, got {value}")


@dataclass
class BaseKVCacheTransferConfig(BasePolyConfig):
    """Base configuration for KV Cache transfer predictors."""
    
    # Network configuration
    network_bandwidth_gbps: float = field(
        default=100.0,
        metadata={"help": "Network bandwidth in Gbps for KV cache transfer."},
    )
    network_latency_ms: float = field(
        default=0.1,
        metadata={"help": "Network latency in milliseconds for KV cache transfer."},
    )
    
    # Transfer optimization configuration
    enable_compression: bool = field(
        default=False,
        metadata={"help": "Whether to enable compression for KV cache transfer."},
    )
    compression_ratio: float = field(
        default=1.0,
        metadata={"help": "Compression ratio for KV cache transfer (1.0 = no compression)."},
    )
    
    # Future extension configuration
    enable_latency_hiding: bool = field(
        default=False,
        metadata={"help": "Whether to enable latency hiding with layer-by-layer transfer."},
    )
    enable_request_transfer: bool = field(
        default=False,
        metadata={"help": "Whether to transfer request information along with KV cache."},
    )
    transfer_protocol: str = field(
        default="rdma",
        metadata={"help": "Transfer protocol: rdma, tcp, infiniband."},
    )

    def __post_init__(self) -> None:
        _require_positive(self.network_bandwidth_gbps, "network_bandwidth_gbps")
        _require_non_negative(self.network_latency_ms, "network_latency_ms")
        _require_positive(self.compression_ratio, "compression_ratio")


@dataclass
class AnalyticalKVCacheTransferConfig(BaseKVCacheTransferConfig):
    """Configuration for analytical KV cache transfer predictor.

    Deprecation timeline:
        - kv_cache_dtype_size_bytes is deprecated as of 2026-01.
        - Planned removal no earlier than 2026-07-01.
        - Prefer op-quantization config entries for kv_cache_transfer precision.
    """
    
    # Data type configuration for KV cache size calculation
    kv_cache_dtype_size_bytes: int = field(
        default=2,
        metadata={
            "help": "Size of KV cache data type in bytes (e.g., 2 for fp16, 4 for fp32). Deprecated as of 2026-01; planned removal no earlier than 2026-07-01. Use op quantization config."
        },
    )
    
    # Model-specific overrides (optional)
    override_num_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Override number of layers for KV cache size calculation."},
    )
    override_num_heads: Optional[int] = field(
        default=None,
        metadata={"help": "Override number of KV heads (num_kv_heads) for KV cache size calculation. This should be the total number of key-value heads, not query heads."},
    )
    override_head_dim: Optional[int] = field(
        default=None,
        metadata={"help": "Override head dimension for KV cache size calculation."},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_positive(self.kv_cache_dtype_size_bytes, "kv_cache_dtype_size_bytes")
        _require_optional_positive(self.override_num_layers, "override_num_layers")
        _require_optional_positive(self.override_num_heads, "override_num_heads")
        _require_optional_positive(self.override_head_dim, "override_head_dim")

    @classmethod
    def get_type(cls) -> KVCacheTransferType:
        return KVCacheTransferType.ANALYTICAL
    
    @classmethod
    def get_name(cls) -> str:
        return "analytical"
