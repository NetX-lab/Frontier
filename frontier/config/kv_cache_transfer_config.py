from dataclasses import dataclass, field
from typing import Optional

from frontier.config.base_poly_config import BasePolyConfig
from frontier.types import KVCacheTransferType

# vLLM KV connectors of a PDD prefill instance whose device work Frontier prices.
KV_CONNECTORS = ("none", "p2p_nccl", "nixl")
# When a NIXL READ completes for the DECODE scheduler: at the end of its wire
# time, or at the output of the DECODE engine step that finds it complete.
NIXL_READ_COMPLETIONS = ("transfer_end", "engine_step")


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
    kv_connector: str = field(
        default="none",
        metadata={
            "help": "vLLM KV connector of the PDD prefill instance: none, p2p_nccl "
            "(P2pNcclConnector), or nixl. NIXL is currently admitted only by the "
            "dummy structural PD path of attention families with a transfer "
            "payload (GLM DSA and Step)."
        },
    )
    kv_send_slowdown_per_layer_ms: float = field(
        default=0.0,
        metadata={
            "help": "Measured correction, not a model of the send: device time (ms) added to each layer "
            "of a PDD prefill step that completes a prompt, for the slowdown of the layer's later "
            "operators while the kv_connector sends its KV. Requires a kv_connector; 0 disables it."
        },
    )
    nixl_read_completion: str = field(
        default="transfer_end",
        metadata={
            "help": "When a NIXL READ completes for the DECODE scheduler: transfer_end (the request "
            "is ready when the wire time ends) or engine_step (the DECODE engine posts the READ at "
            "its next step, a later step finds it complete, and that step's output reports it, as "
            "vLLM's start_load_kv / get_finished / finished_recving do). engine_step requires "
            "kv_connector=nixl."
        },
    )

    def __post_init__(self) -> None:
        _require_positive(self.network_bandwidth_gbps, "network_bandwidth_gbps")
        _require_non_negative(self.network_latency_ms, "network_latency_ms")
        _require_positive(self.compression_ratio, "compression_ratio")
        _require_non_negative(self.kv_send_slowdown_per_layer_ms, "kv_send_slowdown_per_layer_ms")
        if self.kv_connector not in KV_CONNECTORS:
            raise ValueError(f"kv_connector must be one of {list(KV_CONNECTORS)}, got {self.kv_connector!r}")
        if self.kv_send_slowdown_per_layer_ms > 0 and self.kv_connector == "none":
            raise ValueError("kv_send_slowdown_per_layer_ms corrects the KV send of a kv_connector; kv_connector is 'none'.")
        if self.kv_connector == "nixl" and self.kv_send_slowdown_per_layer_ms > 0:
            raise ValueError(
                "kv_send_slowdown_per_layer_ms is a P2P per-layer correction and "
                "must be zero for the NIXL structural path."
            )
        if self.kv_connector == "nixl" and self.enable_latency_hiding:
            raise ValueError(
                "NIXL structural PD models one whole-request handoff and does not "
                "support latency hiding."
            )
        if self.nixl_read_completion not in NIXL_READ_COMPLETIONS:
            raise ValueError(
                f"nixl_read_completion must be one of {list(NIXL_READ_COMPLETIONS)}, "
                f"got {self.nixl_read_completion!r}"
            )
        if self.nixl_read_completion == "engine_step" and self.kv_connector != "nixl":
            raise ValueError(
                "nixl_read_completion='engine_step' models NIXL READs; "
                f"kv_connector is {self.kv_connector!r}."
            )


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
