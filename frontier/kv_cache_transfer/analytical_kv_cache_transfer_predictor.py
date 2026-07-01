from typing import TYPE_CHECKING

from frontier.attention.memory import get_attention_runtime_kv_layout
from frontier.attention.model_binding import bind_attention_family
from frontier.config import get_quantization_manager
from frontier.kv_cache_transfer.base_kv_cache_transfer_predictor import BaseKVCacheTransferPredictor
from frontier.logger import init_logger
from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.config import ReplicaConfig
    from frontier.config.config import AnalyticalKVCacheTransferConfig
    from frontier.entities import Batch, Request


class AnalyticalKVCacheTransferPredictor(BaseKVCacheTransferPredictor):
    """Analytical KV cache transfer predictor using bandwidth and latency."""

    def __init__(self, config: "AnalyticalKVCacheTransferConfig") -> None:
        super().__init__(config)
        self._config: "AnalyticalKVCacheTransferConfig" = config
        self._logger = init_logger(__name__)

    def get_transfer_time(
        self,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        batch: "Batch",
        kv_cache_size_bytes: int,
    ) -> float:
        effective_size_bytes = kv_cache_size_bytes
        if self._config.enable_compression:
            effective_size_bytes = kv_cache_size_bytes / self._config.compression_ratio

        bandwidth_bytes_per_ms = (self._config.network_bandwidth_gbps * 1e9) / (8 * 1000)
        return self._config.network_latency_ms + (effective_size_bytes / bandwidth_bytes_per_ms)

    def get_kv_cache_size(self, batch: "Batch", replica_config: "ReplicaConfig") -> int:
        total_tokens = sum(req.num_prefill_tokens for req in batch.requests)
        return self._calculate_kv_cache_size_for_tokens(total_tokens, replica_config)

    def get_kv_cache_size_for_request(
        self, request: "Request", replica_config: "ReplicaConfig"
    ) -> int:
        return self._calculate_kv_cache_size_for_tokens(
            request.num_prefill_tokens, replica_config
        )

    def _calculate_kv_cache_size_for_tokens(
        self, num_tokens: int, replica_config: "ReplicaConfig"
    ) -> int:
        model_config = replica_config.model_config
        family = bind_attention_family(model_config).family
        num_layers = (
            self._config.override_num_layers
            if self._config.override_num_layers is not None
            else model_config.num_layers
        )
        # Runtime KV layout is family-aware: dense uses (num_kv_heads, head_dim, kv_factor=2);
        # latent MLA collapses to (1, kv_lora_rank + qk_rope_head_dim, kv_factor=1). Overrides,
        # when set, replace the head count / head size but the family still owns kv_factor.
        num_heads = (
            self._config.override_num_heads
            if self._config.override_num_heads is not None
            else model_config.get_runtime_num_kv_heads()
        )
        head_dim = (
            self._config.override_head_dim
            if self._config.override_head_dim is not None
            else model_config.get_runtime_head_size()
        )
        layout = get_attention_runtime_kv_layout(
            family,
            runtime_num_kv_heads_per_worker=num_heads,
            runtime_head_size=head_dim,
        )
        dtype_size = self._get_kv_cache_dtype_size_bytes()
        return int(
            num_tokens
            * num_layers
            * layout.runtime_num_kv_heads_per_worker
            * layout.runtime_head_size
            * layout.kv_factor
            * dtype_size
        )

    def _get_kv_cache_dtype_size_bytes(self) -> float:
        quant_manager = get_quantization_manager()
        has_explicit_quant = quant_manager.has_explicit_precision("kv_cache_transfer")
        quant_precision = quant_manager.get_precision("kv_cache_transfer")
        quant_dtype_size = quant_precision.bytes_per_element

        if has_explicit_quant:
            if (
                self._config.kv_cache_dtype_size_bytes is not None
                and self._config.kv_cache_dtype_size_bytes != quant_dtype_size
            ):
                raise ValueError(
                    "kv_cache_dtype_size_bytes is deprecated and conflicts with quantization "
                    f"config for kv_cache_transfer (config={self._config.kv_cache_dtype_size_bytes}, "
                    f"quantization={quant_dtype_size})."
                )
            return quant_dtype_size

        if self._config.kv_cache_dtype_size_bytes is not None:
            return self._config.kv_cache_dtype_size_bytes

        return quant_dtype_size

    def supports_latency_hiding(self) -> bool:
        return self._config.enable_latency_hiding
