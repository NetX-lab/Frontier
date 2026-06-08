from dataclasses import dataclass, field
from typing import Optional

from frontier.config.base_poly_config import BasePolyConfig


@dataclass
class BaseM2NTransferConfig(BasePolyConfig):
    """Base configuration for Memory-to-Memory (M2N) transfer predictors."""
    
    # Network configuration for P2P communication
    memory_bandwidth_gbps: float = field(
        default=200.0,
        metadata={"help": "Memory bandwidth in Gbps for M2N transfer between decode clusters."},
    )
    network_latency_ms: float = field(
        default=0.05,
        metadata={"help": "Network latency in milliseconds for M2N transfer (typically lower than KV cache transfer)."},
    )
    
    # Transfer optimization configuration
    enable_compression: bool = field(
        default=False,
        metadata={"help": "Whether to enable compression for M2N transfer."},
    )
    compression_ratio: float = field(
        default=1.0,
        metadata={"help": "Compression ratio for M2N transfer (1.0 = no compression)."},
    )
    
    # P2P communication configuration
    enable_p2p_optimization: bool = field(
        default=True,
        metadata={"help": "Whether to enable Point-to-Point communication optimizations."},
    )
    p2p_protocol: str = field(
        default="nvlink",
        metadata={"help": "P2P transfer protocol: nvlink, pcie, rdma."},
    )
    
    # Future extension configuration
    enable_latency_hiding: bool = field(
        default=False,
        metadata={"help": "Whether to enable latency hiding with pipelined transfer."},
    )


@dataclass
class AnalyticalM2NTransferConfig(BaseM2NTransferConfig):
    """Configuration for analytical M2N transfer predictor.

    Deprecation timeline:
        - activation_dtype_size_bytes is deprecated as of 2026-01.
        - Planned removal no earlier than 2026-07-01.
        - Prefer op-quantization config entries for m2n_transfer precision.
    """
    
    # Data type configuration for attention output/FFN input size calculation
    activation_dtype_size_bytes: int = field(
        default=2,
        metadata={
            "help": "Size of activation data type in bytes (e.g., 2 for fp16, 4 for fp32). Deprecated as of 2026-01; planned removal no earlier than 2026-07-01. Use op quantization config."
        },
    )
    
    # Model-specific overrides (optional)
    override_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Override hidden size for M2N transfer size calculation."},
    )
    override_intermediate_size: Optional[int] = field(
        default=None,
        metadata={"help": "Override intermediate size for FFN M2N transfer calculation."},
    )
    
    @classmethod
    def get_type(cls) -> str:
        return "analytical"
    
    @classmethod
    def get_name(cls) -> str:
        return "analytical"
