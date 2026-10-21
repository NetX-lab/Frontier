from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from frontier.attention.memory import (
    get_attention_kv_transfer_payload,
    get_attention_runtime_kv_layout,
)
from frontier.attention.gdn.guards import validate_gdn_runtime_support
from frontier.attention.model_binding import bind_attention_family
from frontier.config import get_quantization_manager
from frontier.kv_cache_transfer.base_kv_cache_transfer_predictor import BaseKVCacheTransferPredictor
from frontier.logger import init_logger
from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.attention.memory import (
        AttentionKVCacheCapacity,
        AttentionKVTransferPayload,
    )
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
        if self._config.kv_connector == "nixl":
            raise NotImplementedError(
                "NIXL size requires the reservation-backed structural path"
            )
        total_tokens = sum(req.num_prefill_tokens for req in batch.requests)
        return self._calculate_kv_cache_size_for_tokens(total_tokens, replica_config)

    def get_kv_cache_size_for_request(
        self, request: "Request", replica_config: "ReplicaConfig"
    ) -> int:
        if self._config.kv_connector == "nixl":
            raise NotImplementedError(
                "NIXL size requires the reservation-backed structural path"
            )
        return self._calculate_kv_cache_size_for_tokens(
            request.num_prefill_tokens, replica_config
        )

    def get_structural_nixl_transfer_info_for_request(
        self,
        *,
        request: "Request",
        replica_config: "ReplicaConfig",
        cache_capacity: "AttentionKVCacheCapacity",
        source_prompt_common_blocks: int,
        target_reserved_common_blocks: int,
    ) -> tuple["AttentionKVTransferPayload", float]:
        """Price the source-backed prompt pages after a real D reservation.

        The family resolves the pages each source TP rank sends; ranks of a
        lane send in parallel, so one rank's bytes set the duration. These are
        modeled page bytes, not a native NIXL wire byte count. Empty
        lookahead capacity reserved by D is deliberately excluded.
        """
        payload = self._get_nixl_transfer_payload(
            request=request, replica_config=replica_config,
            cache_capacity=cache_capacity,
            source_prompt_common_blocks=source_prompt_common_blocks,
            target_reserved_common_blocks=target_reserved_common_blocks,
            prompt_tokens=int(request.num_prefill_tokens),
        )
        transfer_time_ms = self.get_transfer_time(
            ClusterType.PREFILL, ClusterType.DECODE, None,
            payload.transfer_bytes_per_rank,
        )
        return payload, transfer_time_ms

    def get_measured_nixl_coverage(
        self, *, request, source_replica_config, target_replica_config,
        cache_capacity, cost_provider, target_cost_provider,
    ):
        from frontier.entities.kv_cache_transfer_info import NixlTransferCoverage

        getter = getattr(cost_provider, "get_nixl_transfer_coverage", None)
        if not callable(getter) or not callable(
            getattr(cost_provider, "get_nixl_transfer_time_ms", None)
        ):
            raise ValueError(
                "Missing measured NIXL coverage: the cost provider must supply "
                "admitted geometry and READ-to-GPU-ready costs"
            )
        coverage = getter(
            request=request, source_replica_config=source_replica_config,
            target_replica_config=target_replica_config,
            cache_capacity=cache_capacity, target_cost_provider=target_cost_provider,
        )
        if not isinstance(coverage, NixlTransferCoverage):
            raise ValueError("Missing measured NIXL coverage for this request")
        expected = (
            int(request.num_prefill_tokens), cache_capacity,
            int(source_replica_config.attn_tensor_parallel_size),
            int(target_replica_config.attn_tensor_parallel_size),
            source_replica_config.device, target_replica_config.device,
        )
        observed = (
            coverage.num_prefill_tokens, coverage.cache_capacity,
            coverage.source_attention_tp, coverage.target_attention_tp,
            coverage.source_device, coverage.target_device,
        )
        if observed != expected:
            raise ValueError("Measured NIXL coverage does not match request geometry")
        return coverage

    def get_measured_nixl_transfer_info_for_request(
        self, *, request, replica_config, cache_capacity,
        source_prompt_common_blocks, target_reserved_common_blocks,
        coverage, cost_provider,
    ):
        payload = self._get_nixl_transfer_payload(
            request=request, replica_config=replica_config,
            cache_capacity=cache_capacity,
            source_prompt_common_blocks=source_prompt_common_blocks,
            target_reserved_common_blocks=target_reserved_common_blocks,
            prompt_tokens=coverage.num_computed_tokens,
        )
        duration = cost_provider.get_nixl_transfer_time_ms(
            coverage=coverage, payload=payload,
        )
        if isinstance(duration, bool) or not isinstance(duration, Real) or not isfinite(duration) or duration <= 0:
            raise ValueError("Measured NIXL coverage requires a positive finite READ cost")
        return payload, float(duration)

    def _get_nixl_transfer_payload(
        self, *, request, replica_config, cache_capacity,
        source_prompt_common_blocks, target_reserved_common_blocks, prompt_tokens,
    ):
        if self._config.kv_connector != "nixl":
            raise ValueError("Structural NIXL sizing requires kv_connector='nixl'")
        if (
            type(source_prompt_common_blocks) is not int
            or source_prompt_common_blocks <= 0
        ):
            raise ValueError("source_prompt_common_blocks must be a positive int")
        if (
            type(target_reserved_common_blocks) is not int
            or target_reserved_common_blocks < source_prompt_common_blocks
        ):
            raise ValueError(
                "D reservation does not cover every source-backed prompt common block"
            )
        payload = get_attention_kv_transfer_payload(
            replica_config.model_config,
            cache_capacity,
            prompt_tokens=prompt_tokens,
            tensor_parallel_size=int(replica_config.attn_tensor_parallel_size),
            target_reserved_blocks=target_reserved_common_blocks,
        )
        if any(
            component.page_count > source_prompt_common_blocks
            for component in payload.components
        ):
            raise ValueError("NIXL payload exceeds the source-backed prompt pages")
        return payload

    def _calculate_kv_cache_size_for_tokens(
        self, num_tokens: int, replica_config: "ReplicaConfig"
    ) -> int:
        model_config = replica_config.model_config
        validate_gdn_runtime_support(model_config, pd_enabled=True)
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
