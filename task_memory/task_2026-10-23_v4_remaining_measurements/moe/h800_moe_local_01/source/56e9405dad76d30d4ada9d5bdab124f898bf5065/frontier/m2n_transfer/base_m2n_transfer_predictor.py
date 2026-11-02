from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.config import ReplicaConfig
    from frontier.config.m2n_transfer_config import BaseM2NTransferConfig
    from frontier.entities import Batch, Request


class BaseM2NTransferPredictor(ABC):
    """Abstract base class for Memory-to-Memory transfer predictors."""

    def __init__(self, config: "BaseM2NTransferConfig") -> None:
        self._config = config

    @abstractmethod
    def get_transfer_time(
        self,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        batch: "Batch",
        activation_size_bytes: int,
    ) -> float:
        pass

    @abstractmethod
    def get_activation_size(
        self,
        batch: "Batch",
        replica_config: "ReplicaConfig",
        source_cluster_type: ClusterType,
    ) -> int:
        pass

    @abstractmethod
    def get_activation_size_for_request(
        self,
        request: "Request",
        replica_config: "ReplicaConfig",
        source_cluster_type: ClusterType,
    ) -> int:
        pass

    def get_transfer_info(
        self,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        batch: "Batch",
        replica_config: "ReplicaConfig",
    ) -> tuple[int, float]:
        activation_size = self.get_activation_size(
            batch, replica_config, source_cluster_type
        )
        transfer_time = self.get_transfer_time(
            source_cluster_type, target_cluster_type, batch, activation_size
        )
        return activation_size, transfer_time

    def get_transfer_info_for_request(
        self,
        source_cluster_type: ClusterType,
        target_cluster_type: ClusterType,
        request: "Request",
        replica_config: "ReplicaConfig",
    ) -> tuple[int, float]:
        activation_size = self.get_activation_size_for_request(
            request, replica_config, source_cluster_type
        )
        from frontier.entities import Batch

        single_request_batch = Batch(
            replica_id=0,
            requests=[request],
            num_tokens=[1],
            is_moe=replica_config.model_config.is_moe,
        )
        transfer_time = self.get_transfer_time(
            source_cluster_type,
            target_cluster_type,
            single_request_batch,
            activation_size,
        )
        return activation_size, transfer_time
