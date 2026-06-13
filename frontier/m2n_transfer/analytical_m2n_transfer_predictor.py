from typing import TYPE_CHECKING

from frontier.config import get_quantization_manager
from frontier.logger import init_logger
from frontier.m2n_transfer.base_m2n_transfer_predictor import BaseM2NTransferPredictor
from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.config import ReplicaConfig
    from frontier.config.m2n_transfer_config import AnalyticalM2NTransferConfig
    from frontier.entities import Batch, Request


class AnalyticalM2NTransferPredictor(BaseM2NTransferPredictor):
    """Analytical M2N transfer predictor using bandwidth and latency."""

    def __init__(self, config: "AnalyticalM2NTransferConfig") -> None:
        super().__init__(config)
        self._config: "AnalyticalM2NTransferConfig" = config
        self._logger = init_logger(__name__)

    def get_transfer_time(
        self,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        batch: "Batch",
        activation_size_bytes: int,
    ) -> float:
        valid_transfers = [
            (ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN),
            (ClusterType.DECODE_FFN, ClusterType.DECODE_ATTN),
        ]
        if (source_cluster_type, target_cluster_type) not in valid_transfers:
            raise ValueError(
                f"Invalid M2N transfer: {source_cluster_type.name} -> {target_cluster_type.name}. "
                "M2N transfers only support DECODE_ATTN <-> DECODE_FFN communication."
            )

        effective_size_bytes = activation_size_bytes
        if self._config.enable_compression:
            effective_size_bytes = int(activation_size_bytes / self._config.compression_ratio)

        bandwidth_bytes_per_ms = self._config.memory_bandwidth_gbps * 125_000
        transfer_time_ms = self._config.network_latency_ms + (
            effective_size_bytes / bandwidth_bytes_per_ms
        )
        if self._config.enable_p2p_optimization:
            transfer_time_ms = transfer_time_ms / 1.2
        return transfer_time_ms

    def get_activation_size(
        self,
        batch: "Batch",
        replica_config: "ReplicaConfig",
        source_cluster_type: ClusterType,
    ) -> int:
        hidden_size = self._config.override_hidden_size or replica_config.model_config.embedding_dim
        total_tokens = batch.get_effective_total_tokens_for_transfer(source_cluster_type)
        dtype_size = self._get_activation_dtype_size_bytes(source_cluster_type)

        if source_cluster_type in {ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN}:
            return int(total_tokens * hidden_size * dtype_size)
        raise ValueError(
            f"Invalid source cluster type for M2N transfer: {source_cluster_type.name}"
        )

    def get_activation_size_for_request(
        self,
        request: "Request",
        replica_config: "ReplicaConfig",
        source_cluster_type: ClusterType,
    ) -> int:
        hidden_size = self._config.override_hidden_size or replica_config.model_config.embedding_dim
        dtype_size = self._get_activation_dtype_size_bytes(source_cluster_type)

        if source_cluster_type in {ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN}:
            return int(1 * hidden_size * dtype_size)
        raise ValueError(
            f"Invalid source cluster type for M2N transfer: {source_cluster_type.name}"
        )

    def _get_activation_dtype_size_bytes(self, source_cluster_type: ClusterType) -> float:
        quant_manager = get_quantization_manager()
        has_explicit_quant = quant_manager.has_explicit_precision(
            "m2n_transfer", source_cluster_type
        )
        quant_precision = quant_manager.get_precision("m2n_transfer", source_cluster_type)
        quant_dtype_size = quant_precision.bytes_per_element

        if has_explicit_quant:
            if (
                self._config.activation_dtype_size_bytes is not None
                and self._config.activation_dtype_size_bytes != quant_dtype_size
            ):
                raise ValueError(
                    "activation_dtype_size_bytes is deprecated and conflicts with quantization "
                    f"config for m2n_transfer (config={self._config.activation_dtype_size_bytes}, "
                    f"quantization={quant_dtype_size})."
                )
            return quant_dtype_size

        if self._config.activation_dtype_size_bytes is not None:
            return self._config.activation_dtype_size_bytes

        return quant_dtype_size
