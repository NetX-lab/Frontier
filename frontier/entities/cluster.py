import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from frontier.config import BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig
from frontier.config.parallel_semantics import (
    FrontierParallelismMapping,
    build_collective_sim_layout,
    resolve_collective_sim_physical_topology,
)
from frontier.entities.base_entity import BaseEntity
from frontier.entities.replica import Replica
from frontier.logger import get_cluster_logger
from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.cc_backend import BaseCCBackend


class Cluster(BaseEntity):
    def __init__(
        self,
        cluster_config: ClusterConfig,
        metrics_config: MetricsConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Cluster.generate_id()
        self._config = cluster_config
        self._cluster_type = self._config.cluster_type
        self.metrics_config = metrics_config

        # get metrics config
        self._output_dir = metrics_config.output_dir

        # CC Backend instance (lazy initialized)
        self._cc_backend: Optional["BaseCCBackend"] = None

        # Init replica object handles
        self._replicas = {}
        num_replicas = self._config.num_replicas
        replica_config = self._config.replica_config

        # Use cluster-tagged logger instead of direct prints
        _logger = get_cluster_logger(__name__, self._cluster_type.name)
        _logger.info(
            f"Cluster initialized: type={self._cluster_type.name}, num_replicas={num_replicas}, "
            f"device={replica_config.device if hasattr(replica_config, 'device') else 'unknown'}"
        )

        assert (
            replica_config is not None
        ), f"Replica config not found for cluster type {self.cluster_type}"

        for _ in range(num_replicas):
            replica = Replica(replica_config, generator_config, self.cluster_type)
            self._replicas[replica.id] = replica

        if metrics_config.write_json_trace:
            self._write_cluster_info_to_file()

    @property
    def replicas(self):
        return self._replicas

    @property
    def cluster_type(self) -> ClusterType:
        return self._cluster_type

    @property
    def cc_backend(self) -> "BaseCCBackend":
        """
        Get the CC (Collective Communication) backend for this cluster.

        The backend is lazily initialized on first access to avoid unnecessary
        initialization overhead if communication prediction is not needed.

        Returns:
            BaseCCBackend: The CC backend instance for this cluster.

        Raises:
            ValueError: If the CC backend configuration is invalid.
        """
        if self._cc_backend is None:
            self._cc_backend = self._initialize_cc_backend()
        return self._cc_backend

    def _initialize_cc_backend(self) -> "BaseCCBackend":
        """
        Initialize the CC backend for this cluster.

        Creates a CC backend instance using the factory pattern, passing
        cluster-specific configuration parameters.

        Returns:
            BaseCCBackend: The initialized CC backend instance.
        """
        from frontier.cc_backend import CCBackendFactory
        from frontier.cc_backend.cc_backend_config import (
            VidurCCBackendConfig,
            AnalyticalCCBackendConfig,
            CollectiveSimCCBackendConfig,
            AiconfiguratorCCBackendConfig,
            AstraSimAnalyticalCCBackendConfig,
        )
        from frontier.types import CCBackendType

        # Get CC backend config from cluster config
        cc_config = self._config.cc_backend_config

        # Determine backend type from config
        if isinstance(cc_config, VidurCCBackendConfig):
            backend_type = CCBackendType.VIDUR
        elif isinstance(cc_config, AnalyticalCCBackendConfig):
            backend_type = CCBackendType.ANALYTICAL
        elif isinstance(cc_config, CollectiveSimCCBackendConfig):
            backend_type = CCBackendType.COLLECTIVE_SIM
        elif isinstance(cc_config, AiconfiguratorCCBackendConfig):
            backend_type = CCBackendType.AICONFIGURATOR
        elif isinstance(cc_config, AstraSimAnalyticalCCBackendConfig):
            backend_type = CCBackendType.ASTRA_SIM_ANALYTICAL
        else:
            raise ValueError(
                "Unknown CC backend config type: "
                f"{type(cc_config).__name__}. "
                "Supported: VidurCCBackendConfig, AnalyticalCCBackendConfig, "
                "CollectiveSimCCBackendConfig, AiconfiguratorCCBackendConfig, "
                "AstraSimAnalyticalCCBackendConfig"
            )

        # Get replica config for device information
        replica_config = self._config.replica_config

        # Calculate total number of devices in this cluster
        # This is the world_size of a single replica
        num_devices = replica_config.world_size if replica_config else 1

        if isinstance(cc_config, CollectiveSimCCBackendConfig) and replica_config is not None:
            cc_config = self._materialize_collective_sim_cc_config(
                cc_config=cc_config,
                replica_config=replica_config,
                num_devices=num_devices,
            )
        elif isinstance(cc_config, AiconfiguratorCCBackendConfig) and replica_config is not None:
            cc_config = self._materialize_aiconfigurator_cc_config(
                cc_config=cc_config,
                replica_config=replica_config,
            )
        elif (
            isinstance(cc_config, AstraSimAnalyticalCCBackendConfig)
            and replica_config is not None
        ):
            cc_config = self._materialize_astra_sim_analytical_cc_config(
                cc_config=cc_config,
                replica_config=replica_config,
                num_devices=num_devices,
            )

        # Create and return the backend
        return CCBackendFactory.create(
            backend_type=backend_type,
            config=cc_config,
            cluster_type=self._cluster_type,
            device_type=replica_config.device if replica_config else "a100",
            network_device=(
                replica_config.network_device
                if replica_config
                else "a100_pairwise_nvlink"
            ),
            num_devices=num_devices,
        )

    def _materialize_collective_sim_cc_config(
        self,
        cc_config,
        replica_config,
        num_devices: int,
    ):
        """Materialize cluster-specific runtime dims for collective-sim config."""
        num_replicas = max(1, int(self._config.num_replicas or 1))
        mapping = FrontierParallelismMapping(
            cluster_num_replicas=num_replicas,
            attn_tensor_parallel_size=int(replica_config.attn_tensor_parallel_size),
            attn_data_parallel_size=int(replica_config.attn_data_parallel_size),
            moe_tensor_parallel_size=int(replica_config.moe_tensor_parallel_size),
            moe_expert_parallel_size=int(replica_config.moe_expert_parallel_size),
        )
        if self._cluster_type == ClusterType.DECODE_FFN:
            layout = build_collective_sim_layout(
                mapping=mapping,
                num_pipeline_stages=int(replica_config.num_pipeline_stages),
                domain="moe",
            )
        else:
            layout = build_collective_sim_layout(
                mapping=mapping,
                num_pipeline_stages=int(replica_config.num_pipeline_stages),
                domain="attention",
            )
        physical_topology = resolve_collective_sim_physical_topology(
            cluster_total_devices=num_replicas * int(num_devices),
            num_devices_per_node=int(replica_config.node_config.num_devices_per_node),
            scenario_profile=getattr(cc_config, "scenario_profile", None),
        )

        if cc_config.runner_out_dir:
            out_root = Path(cc_config.runner_out_dir)
        else:
            out_root = Path(cc_config.cache_dir) / "collective-sim"
        runner_out_dir = str(out_root / self._cluster_type.name.lower())

        return replace(
            cc_config,
            cluster_servers=int(physical_topology.servers),
            cluster_gpus_per_server=int(physical_topology.gpus_per_server),
            parallel_tp=max(1, int(layout.tp)),
            parallel_cp=max(1, int(layout.cp)),
            parallel_dp=max(1, int(layout.dp)),
            parallel_ep=max(1, int(layout.ep)),
            runtime_num_replicas=max(1, num_replicas),
            runtime_num_pipeline_stages=max(1, int(replica_config.num_pipeline_stages)),
            runtime_attn_tensor_parallel_size=max(1, int(replica_config.attn_tensor_parallel_size)),
            runtime_attn_data_parallel_size=max(1, int(replica_config.attn_data_parallel_size)),
            runtime_moe_tensor_parallel_size=max(1, int(replica_config.moe_tensor_parallel_size)),
            runtime_moe_expert_parallel_size=max(1, int(replica_config.moe_expert_parallel_size)),
            runner_out_dir=runner_out_dir,
        )

    def _materialize_aiconfigurator_cc_config(
        self,
        cc_config,
        replica_config,
    ):
        """Materialize cluster-specific defaults for aiconfigurator config."""
        device_to_system = {
            "a100": "a100_sxm",
            "h100": "h100_sxm",
            "h200": "h200_sxm",
            "b200": "b200_sxm",
            "gb200": "gb200_sxm",
            "l40s": "l40s",
        }
        configured_system = str(getattr(cc_config, "system", "") or "").strip()
        if configured_system:
            return replace(cc_config, system=configured_system)

        normalized_device = str(replica_config.device or "").strip().lower()
        if normalized_device not in device_to_system:
            raise ValueError(
                "Unable to infer aiconfigurator system from replica_config.device="
                f"{replica_config.device!r}. Supported devices: {sorted(device_to_system)}"
            )
        return replace(cc_config, system=device_to_system[normalized_device])

    def _materialize_astra_sim_analytical_cc_config(
        self,
        cc_config,
        replica_config,
        num_devices: int,
    ):
        """Materialize cluster-specific runtime dims for astra-sim analytical config."""
        num_replicas = max(1, int(self._config.num_replicas or 1))
        physical_topology = resolve_collective_sim_physical_topology(
            cluster_total_devices=num_replicas * int(num_devices),
            num_devices_per_node=int(replica_config.node_config.num_devices_per_node),
            scenario_profile=None,
        )

        return replace(
            cc_config,
            cluster_servers=int(physical_topology.servers),
            cluster_gpus_per_server=int(physical_topology.gpus_per_server),
            runtime_num_replicas=max(1, num_replicas),
            runtime_num_pipeline_stages=max(1, int(replica_config.num_pipeline_stages)),
            runtime_attn_tensor_parallel_size=max(
                1, int(replica_config.attn_tensor_parallel_size)
            ),
            runtime_attn_data_parallel_size=max(
                1, int(replica_config.attn_data_parallel_size)
            ),
            runtime_moe_tensor_parallel_size=max(
                1, int(replica_config.moe_tensor_parallel_size)
            ),
            runtime_moe_expert_parallel_size=max(
                1, int(replica_config.moe_expert_parallel_size)
            ),
        )

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "cluster_type": str(self.cluster_type),
            "num_replicas": len(self._replicas),
        }

    def _write_cluster_info_to_file(self) -> None:
        replica_dicts = [replica.to_dict() for replica in self._replicas.values()]
        cluster_info = {"replicas": replica_dicts}

        cluster_file = f"{self._output_dir}/cluster.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_info, f)
