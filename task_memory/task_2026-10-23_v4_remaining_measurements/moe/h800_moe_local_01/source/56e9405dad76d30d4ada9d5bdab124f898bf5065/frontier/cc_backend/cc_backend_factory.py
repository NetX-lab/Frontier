"""
Factory for creating CC Backend instances.

This module provides a factory class that uses the registry pattern
to create appropriate CC backend instances based on configuration.
"""

from typing import TYPE_CHECKING, Dict, List, Type

from frontier.logger import init_logger
from frontier.types import CCBackendType
from frontier.utils.base_registry import BaseRegistry
from frontier.config.config import AICONFIGURATOR_BACKEND_RELEASE_ERROR

if TYPE_CHECKING:
    from frontier.cc_backend.base_cc_backend import BaseCCBackend
    from frontier.cc_backend.cc_backend_config import BaseCCBackendConfig
    from frontier.types import ClusterType

logger = init_logger(__name__)


class CCBackendFactory(BaseRegistry):
    """
    Factory for creating CC backend instances.

    Uses the registry pattern to allow easy addition of new backend types.
    Backend implementations are registered with their corresponding CCBackendType
    and can be instantiated via the create() method.

    Example:
        >>> from frontier.cc_backend import CCBackendFactory
        >>> from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
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
    """

    _builtin_backends_loaded: bool = False

    @classmethod
    def _ensure_builtin_backends_registered(cls) -> None:
        """Ensure built-in backend modules are imported and registered."""
        if cls._builtin_backends_loaded:
            return

        # Import side-effects register backends into cls._registry.
        import frontier.cc_backend.backends  # noqa: F401

        cls._builtin_backends_loaded = True

    @classmethod
    def get_key_from_str(cls, key_str: str) -> CCBackendType:
        """
        Convert string key to CCBackendType enum.

        Args:
            key_str: String representation of backend type (e.g., "vidur", "analytical")

        Returns:
            Corresponding CCBackendType enum value

        Raises:
            ValueError: If key_str is not a valid backend type
        """
        return CCBackendType.from_str(key_str)

    @classmethod
    def create(
        cls,
        backend_type: CCBackendType,
        config: "BaseCCBackendConfig",
        cluster_type: "ClusterType",
        device_type: str,
        network_device: str,
        num_devices: int,
    ) -> "BaseCCBackend":
        """
        Create a CC backend instance.

        Args:
            backend_type: Type of backend (VIDUR, ANALYTICAL)
            config: Backend configuration
            cluster_type: Type of cluster (MONOLITHIC, PREFILL, DECODE, etc.)
            device_type: Device type (e.g., "a100", "h100")
            network_device: Network device identifier (e.g., "a100_pairwise_nvlink")
            num_devices: Number of devices in the cluster

        Returns:
            CC backend instance

        Raises:
            ValueError: If backend_type is not registered
        """
        if backend_type == CCBackendType.AICONFIGURATOR:
            raise ValueError(AICONFIGURATOR_BACKEND_RELEASE_ERROR)

        cls._ensure_builtin_backends_registered()
        if backend_type not in cls._registry:
            available = cls.get_available_backends()
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                f"Available types: {available}"
            )

        backend_class = cls._registry[backend_type]
        logger.info(
            f"Creating {backend_class.__name__} for cluster_type={cluster_type}, "
            f"device_type={device_type}, network_device={network_device}, "
            f"num_devices={num_devices}"
        )

        return backend_class(
            config=config,
            cluster_type=cluster_type,
            device_type=device_type,
            network_device=network_device,
            num_devices=num_devices,
        )

    @classmethod
    def create_from_str(
        cls,
        backend_type_str: str,
        config: "BaseCCBackendConfig",
        cluster_type: "ClusterType",
        device_type: str,
        network_device: str,
        num_devices: int,
    ) -> "BaseCCBackend":
        """
        Create a CC backend instance from string type.

        Args:
            backend_type_str: String representation of backend type (e.g., "vidur", "analytical")
            config: Backend configuration
            cluster_type: Type of cluster (MONOLITHIC, PREFILL, DECODE, etc.)
            device_type: Device type (e.g., "a100", "h100")
            network_device: Network device identifier (e.g., "a100_pairwise_nvlink")
            num_devices: Number of devices in the cluster

        Returns:
            CC backend instance

        Raises:
            ValueError: If backend_type_str is not a valid backend type
        """
        backend_type = cls.get_key_from_str(backend_type_str)
        return cls.create(
            backend_type=backend_type,
            config=config,
            cluster_type=cluster_type,
            device_type=device_type,
            network_device=network_device,
            num_devices=num_devices,
        )

    @classmethod
    def get_available_backends(cls) -> List[str]:
        """
        Return list of available backend type names.

        Returns:
            List of registered backend type names
        """
        cls._ensure_builtin_backends_registered()
        return [str(backend_type) for backend_type in cls._registry.keys()]

    @classmethod
    def is_registered(cls, backend_type: CCBackendType) -> bool:
        """
        Check if a backend type is registered.

        Args:
            backend_type: Backend type to check

        Returns:
            True if registered, False otherwise
        """
        cls._ensure_builtin_backends_registered()
        return backend_type in cls._registry


# Note: Backend implementations will be registered after they are created
# in the backends/ subdirectory. Registration happens at module import time.
#
# Example registration (done in backends/__init__.py or individual backend files):
# CCBackendFactory.register(CCBackendType.VIDUR, VidurCCBackend)
# CCBackendFactory.register(CCBackendType.ANALYTICAL, AnalyticalCCBackend)
