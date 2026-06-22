from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.config import ReplicaConfig
    from frontier.config.kv_cache_transfer_config import BaseKVCacheTransferConfig
    from frontier.entities import Batch, Request


class BaseKVCacheTransferPredictor(ABC):
    """Abstract base class for KV cache transfer predictors."""

    def __init__(self, config: "BaseKVCacheTransferConfig") -> None:
        self._config = config

    @abstractmethod
    def get_transfer_time(
        self,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        batch: "Batch",
        kv_cache_size_bytes: int,
    ) -> float:
        pass

    @abstractmethod
    def get_kv_cache_size(self, batch: "Batch", replica_config: "ReplicaConfig") -> int:
        pass

    @abstractmethod
    def get_kv_cache_size_for_request(
        self, request: "Request", replica_config: "ReplicaConfig"
    ) -> int:
        pass

    @abstractmethod
    def supports_latency_hiding(self) -> bool:
        pass

    def get_transfer_info(
        self,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        batch: "Batch",
        replica_config: "ReplicaConfig",
    ) -> tuple[int, float]:
        kv_cache_size = self.get_kv_cache_size(batch, replica_config)
        transfer_time = self.get_transfer_time(
            source_cluster_type, target_cluster_type, batch, kv_cache_size
        )
        return kv_cache_size, transfer_time

    def get_transfer_info_for_request(
        self,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        request: "Request",
        replica_config: "ReplicaConfig",
    ) -> tuple[int, float]:
        kv_cache_size = self.get_kv_cache_size_for_request(request, replica_config)
        transfer_time = self.get_transfer_time(
            source_cluster_type, target_cluster_type, None, kv_cache_size
        )
        return kv_cache_size, transfer_time
