"""
CC Backend Module - Collective Communication Backend for Frontier LLM Inference Simulator.

This module provides a unified abstraction layer for predicting collective communication
operation latencies with pluggable backend support.

Public API:
    - BaseCCBackend: Abstract base class for all CC backends
    - CCBackendFactory: Factory for creating CC backend instances
    - BaseCCBackendConfig: Base configuration for CC backends
    - VidurCCBackendConfig: Configuration for Vidur ML-based backend
    - AnalyticalCCBackendConfig: Configuration for analytical backend
    - CollectiveSimCCBackendConfig: Configuration for collective-sim backend
    - AstraSimAnalyticalCCBackendConfig: Configuration for ASTRA-Sim analytical backend
    - VidurCCBackend: ML-based predictor implementation
    - AnalyticalCCBackend: Analytical model implementation
    - CollectiveSimCCBackend: Topology-aware simulator implementation
    - AstraSimAnalyticalCCBackend: Lightweight ASTRA-Sim analytical implementation

Usage:
    >>> from frontier.cc_backend import CCBackendFactory, AnalyticalCCBackendConfig
    >>> from frontier.types import CCBackendType, ClusterType
    >>>
    >>> config = AnalyticalCCBackendConfig()
    >>> backend = CCBackendFactory.create(
    ...     backend_type=CCBackendType.ANALYTICAL,
    ...     config=config,
    ...     cluster_type=ClusterType.MONOLITHIC,
    ...     device_type="a100",
    ...     network_device="a100_pairwise_nvlink",
    ...     num_devices=8,
    ... )
    >>>
    >>> # Predict all-reduce time
    >>> time_ms = backend.predict_allreduce(data_size_bytes=1024*1024, num_devices=8)
    >>>
    >>> # Or use unified API
    >>> time_ms = backend.predict_comm_cost("allreduce", data_size_bytes=1024*1024, num_devices=8)

Note: Imports are structured to avoid circular dependencies with frontier.config.config.
The config classes are imported first, then the backend implementations.
"""

# Import config classes first (they only depend on base_poly_config and types)
# Note: These imports must come before backend imports to avoid circular dependencies
from frontier.cc_backend.cc_backend_config import (
    BaseCCBackendConfig,
    VidurCCBackendConfig,
    AnalyticalCCBackendConfig,
    CollectiveSimCCBackendConfig,
    AstraSimAnalyticalCCBackendConfig,
)

# Import base class (no dependencies on config.config)
from frontier.cc_backend.base_cc_backend import BaseCCBackend

# Import factory (depends on base class and config)
from frontier.cc_backend.cc_backend_factory import CCBackendFactory

# Import backend implementations (depend on base class and config)
from frontier.cc_backend.backends.analytical_cc_backend import AnalyticalCCBackend
from frontier.cc_backend.backends.vidur_cc_backend import VidurCCBackend
from frontier.cc_backend.backends.collective_sim_cc_backend import CollectiveSimCCBackend
from frontier.cc_backend.backends.astra_sim_analytical_cc_backend import (
    AstraSimAnalyticalCCBackend,
)

__all__ = [
    "BaseCCBackend",
    "CCBackendFactory",
    "BaseCCBackendConfig",
    "VidurCCBackendConfig",
    "AnalyticalCCBackendConfig",
    "CollectiveSimCCBackendConfig",
    "AstraSimAnalyticalCCBackendConfig",
    "VidurCCBackend",
    "AnalyticalCCBackend",
    "CollectiveSimCCBackend",
    "AstraSimAnalyticalCCBackend",
]
