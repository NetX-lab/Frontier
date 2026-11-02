"""
Analytical Collective Communication Backend.

This module provides an analytical model for predicting collective communication
latencies using simple bandwidth/latency formulas.
"""

import math
from typing import Optional

from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.logger import init_logger
from frontier.types import CCBackendType, ClusterType

logger = init_logger(__name__)


class AnalyticalCCBackend(BaseCCBackend):
    """
    Analytical collective communication backend using bandwidth/latency formulas.

    This backend uses simple analytical models for communication prediction:
        time = latency + (data_size / bandwidth)

    The analytical models are based on standard collective communication algorithms:
        - All-reduce: Ring algorithm with 2*(n-1)/n data volume factor
        - All-gather: Ring algorithm with (n-1)/n data volume factor
        - Broadcast: Tree algorithm with log2(n) steps
        - Send/recv: Direct point-to-point transfer
        - Reduce-scatter: Ring algorithm with (n-1)/n data volume factor
        - All-to-all: Full exchange with (n-1) transfers per device

    Attributes:
        _bandwidth_gbps: Network bandwidth in Gbps
        _latency_us: Network latency in microseconds
        _intra_node_bandwidth_gbps: Intra-node bandwidth in Gbps (NVLink)
    """

    def __init__(
        self,
        config: AnalyticalCCBackendConfig,
        cluster_type: ClusterType,
        device_type: str,
        network_device: str,
        num_devices: int,
    ) -> None:
        """
        Initialize the analytical CC backend.

        Args:
            config: Analytical backend configuration
            cluster_type: Type of cluster (MONOLITHIC, PREFILL, DECODE, etc.)
            device_type: Device type (e.g., "a100", "h100")
            network_device: Network device identifier (e.g., "a100_pairwise_nvlink")
            num_devices: Number of devices in the cluster
        """
        super().__init__(config, cluster_type, device_type, network_device, num_devices)

        self._bandwidth_gbps = config.network_bandwidth_gbps
        self._latency_us = config.network_latency_us
        self._intra_node_bandwidth_gbps = config.intra_node_bandwidth_gbps

        logger.info(
            f"AnalyticalCCBackend initialized with bandwidth={self._bandwidth_gbps} Gbps, "
            f"latency={self._latency_us} us, intra_node_bandwidth={self._intra_node_bandwidth_gbps} Gbps"
        )

    def _calculate_time(
        self,
        data_size_bytes: int,
        num_devices: int,
        is_intra_node: bool = True,
    ) -> float:
        """
        Calculate communication time using analytical formula.

        The formula used is:
            time_ms = latency_ms + (data_size_bytes / bandwidth_bytes_per_ms)

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            is_intra_node: Whether communication is intra-node (uses higher bandwidth)

        Returns:
            Communication time in milliseconds (non-negative)
        """
        # Select bandwidth based on communication type
        bandwidth_gbps = (
            self._intra_node_bandwidth_gbps if is_intra_node else self._bandwidth_gbps
        )

        # Convert Gbps to bytes/ms
        # 1 Gbps = 1e9 bits/s = 1e9/8 bytes/s = 1e9/8/1000 bytes/ms
        bandwidth_bytes_per_ms = (bandwidth_gbps * 1e9) / (8 * 1000)

        # Convert latency from us to ms
        latency_ms = self._latency_us / 1000

        # time = latency + (data_size / bandwidth)
        time_ms = latency_ms + (data_size_bytes / bandwidth_bytes_per_ms)

        logger.debug(
            f"_calculate_time: data_size={data_size_bytes} bytes, "
            f"bandwidth={bandwidth_gbps} Gbps, latency={self._latency_us} us, "
            f"result={time_ms:.6f} ms"
        )

        return max(0.0, time_ms)

    def predict_allreduce(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict all-reduce time using ring algorithm model.

        Ring all-reduce algorithm:
            - Data is split into n chunks
            - Each device sends (n-1) chunks and receives (n-1) chunks
            - Total data transferred per device: 2 * (n-1) / n * data_size

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allreduce")

        if num_devices <= 1:
            return 0.0

        # Ring all-reduce data volume: 2 * (n-1) / n * data_size
        effective_data_size = 2 * (num_devices - 1) / num_devices * data_size_bytes

        result = self._calculate_time(effective_data_size, num_devices)

        logger.debug(
            f"predict_allreduce: data_size={data_size_bytes}, num_devices={num_devices}, "
            f"effective_data_size={effective_data_size:.0f}, result={result:.6f} ms"
        )

        return result

    def predict_allgather(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict all-gather time using ring algorithm model.

        Ring all-gather algorithm:
            - Each device has data_size_bytes of data
            - Total data after gather: n * data_size_bytes
            - Each device sends (n-1) chunks
            - Effective data transferred: (n-1) / n * total_data_size

        Args:
            data_size_bytes: Size of data per device in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allgather")

        if num_devices <= 1:
            return 0.0

        # Total data size after gather
        total_data_size = data_size_bytes * num_devices
        # Ring all-gather data volume: (n-1) / n * total_data_size
        effective_data_size = (num_devices - 1) / num_devices * total_data_size

        result = self._calculate_time(effective_data_size, num_devices)

        logger.debug(
            f"predict_allgather: data_size={data_size_bytes}, num_devices={num_devices}, "
            f"effective_data_size={effective_data_size:.0f}, result={result:.6f} ms"
        )

        return result

    def predict_broadcast(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict broadcast time using tree algorithm model.

        Tree broadcast algorithm:
            - Uses a binary tree structure
            - Number of steps: ceil(log2(n))
            - Each step transfers data_size_bytes

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "broadcast")

        if num_devices <= 1:
            return 0.0

        # Tree broadcast: log2(n) steps
        num_steps = math.ceil(math.log2(num_devices))

        # Each step is a point-to-point transfer
        single_step_time = self._calculate_time(data_size_bytes, 2)
        result = num_steps * single_step_time

        logger.debug(
            f"predict_broadcast: data_size={data_size_bytes}, num_devices={num_devices}, "
            f"num_steps={num_steps}, result={result:.6f} ms"
        )

        return result

    def predict_send_recv(
        self,
        data_size_bytes: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict point-to-point send/recv communication time.

        Simple point-to-point transfer between two devices.

        Args:
            data_size_bytes: Size of data in bytes
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)

        result = self._calculate_time(data_size_bytes, 2)

        logger.debug(
            f"predict_send_recv: data_size={data_size_bytes}, result={result:.6f} ms"
        )

        return result

    def predict_reduce_scatter(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict reduce-scatter time using ring algorithm model.

        Ring reduce-scatter algorithm:
            - Reduces data from all devices and scatters result
            - Each device ends up with 1/n of the reduced data
            - Effective data transferred: (n-1) / n * data_size

        Args:
            data_size_bytes: Size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "reduce_scatter")

        if num_devices <= 1:
            return 0.0

        # Ring reduce-scatter data volume: (n-1) / n * data_size
        effective_data_size = (num_devices - 1) / num_devices * data_size_bytes

        result = self._calculate_time(effective_data_size, num_devices)

        logger.debug(
            f"predict_reduce_scatter: data_size={data_size_bytes}, num_devices={num_devices}, "
            f"effective_data_size={effective_data_size:.0f}, result={result:.6f} ms"
        )

        return result

    def predict_all_to_all(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        """
        Predict all-to-all communication time.

        All-to-all exchange algorithm:
            - Each device sends different data to every other device
            - Total data per device: data_size_bytes (split into n chunks)
            - Each device sends (n-1) chunks of size data_size_bytes/n
            - Effective data transferred: (n-1) * data_size_bytes / n

        Args:
            data_size_bytes: Total size of data in bytes
            num_devices: Number of participating devices
            cluster_type: Optional cluster type for context-aware prediction

        Returns:
            Predicted execution time in milliseconds

        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "all_to_all")

        if num_devices <= 1:
            return 0.0

        # All-to-all: each device sends (n-1) chunks
        data_per_device = data_size_bytes // num_devices
        effective_data_size = (num_devices - 1) * data_per_device

        result = self._calculate_time(effective_data_size, num_devices)

        logger.debug(
            f"predict_all_to_all: data_size={data_size_bytes}, num_devices={num_devices}, "
            f"data_per_device={data_per_device}, effective_data_size={effective_data_size}, "
            f"result={result:.6f} ms"
        )

        return result


# Register the backend with the factory
CCBackendFactory.register(CCBackendType.ANALYTICAL, AnalyticalCCBackend)
