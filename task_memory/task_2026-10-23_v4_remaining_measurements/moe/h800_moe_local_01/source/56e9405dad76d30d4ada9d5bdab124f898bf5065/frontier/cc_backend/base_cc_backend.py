"""
Abstract base class for Collective Communication backends.

This module defines the interface that all CC backend implementations must follow.
"""

from abc import ABC, abstractmethod
import math
from typing import Optional, TYPE_CHECKING

from frontier.logger import init_logger
from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.cc_backend.cc_backend_config import BaseCCBackendConfig

logger = init_logger(__name__)


class BaseCCBackend(ABC):
    """
    Abstract base class for collective communication backends.

    All communication prediction backends must inherit from this class
    and implement the abstract methods for each collective operation.

    Attributes:
        _config: Backend configuration
        _cluster_type: Type of cluster this backend serves
        _device_type: Device type (e.g., "a100", "h100")
        _network_device: Network device identifier (e.g., "a100_pairwise_nvlink")
        _num_devices: Number of devices in the cluster
    """

    # Supported collective operations for dispatch
    SUPPORTED_OPERATIONS = frozenset(
        [
            "allreduce",
            "allgather",
            "broadcast",
            "send_recv",
            "reduce_scatter",
            "all_to_all",
        ]
    )

    def __init__(
        self,
        config: "BaseCCBackendConfig",
        cluster_type: ClusterType,
        device_type: str,
        network_device: str,
        num_devices: int,
    ) -> None:
        """
        Initialize the CC backend.

        Args:
            config: Backend configuration
            cluster_type: Type of cluster (MONOLITHIC, PREFILL, DECODE, etc.)
            device_type: Device type (e.g., "a100", "h100")
            network_device: Network device identifier (e.g., "a100_pairwise_nvlink")
            num_devices: Number of devices in the cluster

        Raises:
            ValueError: If num_devices < 1
        """
        if num_devices < 1:
            raise ValueError(f"num_devices must be >= 1, got {num_devices}")

        self._config = config
        self._cluster_type = cluster_type
        self._device_type = device_type
        self._network_device = network_device
        self._num_devices = num_devices

        logger.info(
            f"Initialized {self.__class__.__name__} for cluster_type={cluster_type}, "
            f"device_type={device_type}, network_device={network_device}, "
            f"num_devices={num_devices}"
        )

    @property
    def config(self) -> "BaseCCBackendConfig":
        """Get the backend configuration."""
        return self._config

    @property
    def cluster_type(self) -> ClusterType:
        """Get the cluster type."""
        return self._cluster_type

    @property
    def device_type(self) -> str:
        """Get the device type."""
        return self._device_type

    @property
    def network_device(self) -> str:
        """Get the network device identifier."""
        return self._network_device

    @property
    def num_devices(self) -> int:
        """Get the number of devices."""
        return self._num_devices

    def predict_comm_cost(self, comm_op: str, **kwargs) -> float:
        """
        Unified interface for predicting communication cost.

        This method dispatches to the appropriate specific operation method
        based on the comm_op parameter.

        Args:
            comm_op: Operation type ("allreduce", "allgather", "broadcast",
                     "send_recv", "reduce_scatter", "all_to_all")
            **kwargs: Operation-specific parameters:
                - data_size_bytes (int): Size of data in bytes (required for all ops)
                - num_devices (int): Number of participating devices (required for collective ops)
                - cluster_type (ClusterType, optional): Context-aware cluster type
                - comm_domain (str, optional): Communication domain label (TP/DP/EP/PP)

        Notes:
            Callers are responsible for providing data_size_bytes that already
            accounts for any precision or quantization adjustments.

        Returns:
            Predicted execution time in milliseconds

        Raises:
            NotImplementedError: If operation is not supported
            ValueError: If required parameters are missing or invalid
        """
        dispatch_map = {
            "allreduce": self.predict_allreduce,
            "allgather": self.predict_allgather,
            "broadcast": self.predict_broadcast,
            "send_recv": self.predict_send_recv,
            "reduce_scatter": self.predict_reduce_scatter,
            "all_to_all": self.predict_all_to_all,
        }

        if comm_op not in dispatch_map:
            raise NotImplementedError(
                f"Unsupported communication operation: {comm_op}. "
                f"Supported operations: {list(dispatch_map.keys())}"
            )

        logger.debug(f"Dispatching {comm_op} with kwargs: {kwargs}")
        return dispatch_map[comm_op](**kwargs)

    def _validate_data_size(self, data_size_bytes: int) -> None:
        """
        Validate data_size_bytes parameter.

        Args:
            data_size_bytes: Size of data in bytes

        Raises:
            ValueError: If data_size_bytes is negative
        """
        if data_size_bytes < 0:
            raise ValueError(
                f"data_size_bytes must be non-negative, got {data_size_bytes}"
            )

    def _validate_num_devices(self, num_devices: int, operation: str) -> None:
        """
        Validate num_devices parameter for collective operations.

        Args:
            num_devices: Number of participating devices
            operation: Name of the operation (for error message)

        Raises:
            ValueError: If num_devices < 1
        """
        if num_devices < 1:
            raise ValueError(
                f"num_devices must be >= 1 for {operation}, got {num_devices}"
            )

    @abstractmethod
    def predict_allreduce(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict all-reduce communication time.

        All-reduce combines values from all processes and distributes the result
        back to all processes.

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    def predict_allgather(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict all-gather communication time.

        All-gather gathers data from all processes and distributes the combined
        data to all processes.

        Args:
            data_size_bytes: Size of data per device in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    def predict_broadcast(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict broadcast communication time.

        Broadcast sends data from one process to all other processes.

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    def predict_send_recv(
        self,
        data_size_bytes: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict point-to-point send/recv communication time.

        Send/recv is a point-to-point communication between two processes.

        Args:
            data_size_bytes: Size of data in bytes
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    def predict_reduce_scatter(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict reduce-scatter communication time.

        Reduce-scatter reduces data from all processes and scatters the result
        across all processes.

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    def predict_all_to_all(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict all-to-all communication time.

        All-to-all exchanges data between all pairs of processes.

        Args:
            data_size_bytes: Total size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    # ========================================================================
    # High-Level Communication APIs (Concrete Methods)
    # ========================================================================
    # These methods provide high-level abstractions for common communication
    # patterns, encapsulating data size calculation and delegating to the
    # appropriate low-level prediction methods.

    def predict_expert_parallel_communication(
        self,
        num_tokens: int,
        embedding_dim: int,
        router_topk: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        precision: Optional[str] = None,
    ) -> float:
        """
        Predict expert parallel communication time for MoE token dispatch/return.

        This high-level API encapsulates the data size calculation for MoE
        expert parallel communication (All-to-All operation) and delegates
        to the underlying predict_all_to_all() method.

        In MoE models, expert parallel communication involves:
        1. Dispatch: Sending tokens to their assigned experts across devices
        2. Combine: Gathering results back from experts

        Data size calculation:
            data_size_bytes = embedding_dim * bytes_per_element * num_tokens * router_topk

        Args:
            num_tokens: Number of tokens in the batch
            embedding_dim: Model embedding dimension (hidden size)
            router_topk: Number of experts each token is routed to
            num_devices: Number of devices participating in expert parallelism
            cluster_type: Optional cluster type for context-aware prediction
            precision: Optional precision label for data size calculation (default: FP16)

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid (negative values)

        Example:
            >>> cc_backend.predict_expert_parallel_communication(
            ...     num_tokens=1024,
            ...     embedding_dim=4096,
            ...     router_topk=2,
            ...     num_devices=8,
            ...     cluster_type=ClusterType.DECODE_FFN
            ... )
            0.125  # milliseconds
        """
        # Validate parameters
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if router_topk <= 0:
            raise ValueError(f"router_topk must be positive, got {router_topk}")
        if num_devices < 1:
            raise ValueError(f"num_devices must be >= 1, got {num_devices}")

        # Calculate data size for all-to-all communication
        # Formula: embedding_dim * bytes_per_element * num_tokens * router_topk
        bytes_per_element = 2
        if precision is not None:
            from frontier.config.precision_type import PrecisionType

            bytes_per_element = PrecisionType.from_string(precision).bytes_per_element
        data_size_bytes = int(
            math.ceil(embedding_dim * bytes_per_element * num_tokens * router_topk)
        )

        logger.debug(
            f"predict_expert_parallel_communication: num_tokens={num_tokens}, "
            f"embedding_dim={embedding_dim}, router_topk={router_topk}, "
            f"num_devices={num_devices}, data_size_bytes={data_size_bytes}"
        )

        # Delegate to the low-level all-to-all prediction
        return self.predict_all_to_all(
            data_size_bytes=data_size_bytes,
            num_devices=num_devices,
            cluster_type=cluster_type,
            comm_domain="EP",
        )
