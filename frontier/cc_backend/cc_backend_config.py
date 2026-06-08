"""
Configuration dataclasses for CC Backend module.

This module defines configuration classes for different CC backend types,
following the BasePolyConfig pattern used throughout Frontier.
"""

from dataclasses import dataclass, field
from typing import Optional

# Use direct import to avoid circular import through frontier.config.__init__
from frontier.config.base_poly_config import BasePolyConfig
from frontier.types import CCBackendType


@dataclass
class BaseCCBackendConfig(BasePolyConfig):
    """
    Base configuration for CC backends.

    This class defines common configuration fields shared by all CC backend
    implementations. Specific backends extend this class with their own
    configuration parameters.

    Attributes:
        profiling_data_dir: Directory containing network profiling data
        cache_dir: Directory for caching trained models
        no_cache: Whether to disable model caching
    """

    profiling_data_dir: str = field(
        default="data/profiling/network",
        metadata={"help": "Directory containing network profiling data."},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Directory for caching trained models."},
    )
    no_cache: bool = field(
        default=False,
        metadata={"help": "Disable model caching."},
    )


@dataclass
class VidurCCBackendConfig(BaseCCBackendConfig):
    """
    Configuration for Vidur ML-based CC backend.

    This backend uses sklearn models trained on profiling data to predict
    collective communication latencies.

    Attributes:
        all_reduce_input_file: Path template to all-reduce profiling data
        send_recv_input_file: Path template to send-recv profiling data
        k_fold_cv_splits: Number of k-fold cross validation splits
        num_training_job_threads: Number of training job threads (-1 for auto)
        network_bandwidth_gbps: Inter-node fallback bandwidth in Gbps
        network_latency_us: Fallback latency in microseconds
        intra_node_bandwidth_gbps: Intra-node fallback bandwidth in Gbps
    """

    all_reduce_input_file: str = field(
        default="{profiling_data_dir}/{NETWORK_DEVICE}/all_reduce.csv",
        metadata={"help": "Path template to all-reduce profiling data."},
    )
    send_recv_input_file: str = field(
        default="{profiling_data_dir}/{NETWORK_DEVICE}/send_recv.csv",
        metadata={"help": "Path template to send-recv profiling data."},
    )
    k_fold_cv_splits: int = field(
        default=10,
        metadata={"help": "Number of k-fold cross validation splits."},
    )
    num_training_job_threads: int = field(
        default=-1,
        metadata={"help": "Number of training job threads (-1 for auto)."},
    )
    network_bandwidth_gbps: float = field(
        default=100.0,
        metadata={"help": "Fallback inter-node bandwidth in Gbps for analytical fallback paths."},
    )
    network_latency_us: float = field(
        default=1.0,
        metadata={"help": "Fallback network latency in microseconds for analytical fallback paths."},
    )
    intra_node_bandwidth_gbps: float = field(
        default=600.0,
        metadata={"help": "Fallback intra-node bandwidth in Gbps for analytical fallback paths."},
    )

    @staticmethod
    def get_type() -> CCBackendType:
        """Return the backend type for this configuration."""
        return CCBackendType.VIDUR

    @staticmethod
    def get_name() -> str:
        """Return the backend name for this configuration."""
        return "vidur"


@dataclass
class AnalyticalCCBackendConfig(BaseCCBackendConfig):
    """
    Configuration for analytical CC backend.

    This backend uses simple bandwidth/latency formulas for prediction:
    time = latency + (data_size / bandwidth)

    Attributes:
        network_bandwidth_gbps: Network bandwidth in Gbps
        network_latency_us: Network latency in microseconds
        intra_node_bandwidth_gbps: Intra-node bandwidth in Gbps (NVLink)
    """

    network_bandwidth_gbps: float = field(
        default=100.0,
        metadata={"help": "Network bandwidth in Gbps."},
    )
    network_latency_us: float = field(
        default=1.0,
        metadata={"help": "Network latency in microseconds."},
    )
    intra_node_bandwidth_gbps: float = field(
        default=600.0,
        metadata={"help": "Intra-node bandwidth in Gbps (NVLink)."},
    )

    @staticmethod
    def get_type() -> CCBackendType:
        """Return the backend type for this configuration."""
        return CCBackendType.ANALYTICAL

    @staticmethod
    def get_name() -> str:
        """Return the backend name for this configuration."""
        return "analytical"


@dataclass
class CollectiveSimCCBackendConfig(BaseCCBackendConfig):
    """
    Configuration for collective-sim CC backend.

    This backend runs topology-aware communication simulation through the
    collective-sim submodule and returns predicted collective latencies.
    """

    scenario_profile: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional collective-sim scenario profile (e.g., h100_rail).",
        },
    )

    # Cluster and parallelism dimensions
    cluster_servers: int = field(
        default=1,
        metadata={"help": "Number of servers in the simulated cluster."},
    )
    cluster_gpus_per_server: int = field(
        default=8,
        metadata={"help": "GPUs per server in the simulated cluster."},
    )
    parallel_tp: int = field(
        default=1,
        metadata={"help": "Tensor parallel dimension for collective-sim."},
    )
    parallel_cp: int = field(
        default=1,
        metadata={"help": "Context/pipeline proxy dimension for collective-sim."},
    )
    parallel_dp: int = field(
        default=1,
        metadata={"help": "Data parallel dimension for collective-sim."},
    )
    parallel_ep: int = field(
        default=1,
        metadata={"help": "Expert parallel dimension for collective-sim."},
    )
    runtime_num_replicas: Optional[int] = field(
        default=None,
        metadata={"help": "Frontier runtime cluster replica count for per-call layout mapping."},
    )
    runtime_num_pipeline_stages: Optional[int] = field(
        default=None,
        metadata={"help": "Frontier runtime pipeline stage count for per-call layout mapping."},
    )
    runtime_attn_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Frontier runtime attention TP size for per-call layout mapping."},
    )
    runtime_attn_data_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Frontier runtime attention DP size for per-call layout mapping."},
    )
    runtime_moe_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Frontier runtime MoE TP size for per-call layout mapping."},
    )
    runtime_moe_expert_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Frontier runtime MoE EP size for per-call layout mapping."},
    )
    placement_order: str = field(
        default="TP,CP,DP,EP",
        metadata={"help": "Placement order passed to collective-sim."},
    )

    # Topology
    topology_kind: str = field(
        default="rail",
        metadata={"help": "Topology kind: rail/rail3/fattree/fattree3."},
    )
    topology_spines: int = field(
        default=8,
        metadata={"help": "Spine count for rail/fattree topology."},
    )
    topology_servers_per_tor: int = field(
        default=16,
        metadata={"help": "Servers per ToR switch."},
    )
    topology_paths: int = field(
        default=8,
        metadata={"help": "Equal-cost paths used by the topology model."},
    )
    topology_podsize: int = field(
        default=0,
        metadata={"help": "Pod size for fattree variants."},
    )
    topology_tor_down: int = field(
        default=16,
        metadata={"help": "Downlinks per ToR for fattree variants."},
    )
    topology_tor_up: int = field(
        default=0,
        metadata={"help": "Uplinks per ToR for fattree variants."},
    )
    topology_fattree_oversub: int = field(
        default=1,
        metadata={"help": "Fat-tree oversubscription ratio."},
    )
    topology_latency_ns: int = field(
        default=800,
        metadata={"help": "Link latency in nanoseconds."},
    )
    topology_switch_latency_ns: int = field(
        default=800,
        metadata={"help": "Switch latency in nanoseconds."},
    )

    # Network
    network_linkspeed_mbps: int = field(
        default=400000,
        metadata={"help": "Per-link bandwidth in Mbps."},
    )
    network_mtu: int = field(
        default=9216,
        metadata={"help": "MTU in bytes."},
    )
    network_q: int = field(
        default=256,
        metadata={"help": "Queue depth in packets."},
    )
    network_cwnd: int = field(
        default=64,
        metadata={"help": "Congestion window in packets."},
    )

    # Intra-server model
    intra_server_model: str = field(
        default="legacy_fabric",
        metadata={"help": "Intra-server model: legacy_fabric|ignore|nvlink_analytic."},
    )
    nvlink_one_way_bw_GBps: float = field(
        default=450.0,
        metadata={"help": "One-way NVLink bandwidth in GB/s."},
    )
    nvlink_latency_us: float = field(
        default=0.5,
        metadata={"help": "NVLink latency in microseconds."},
    )
    nvlink_efficiency: float = field(
        default=0.8,
        metadata={"help": "NVLink efficiency in (0, 1]."},
    )
    nvlink_allreduce_launch_overhead_us: float = field(
        default=50.0,
        metadata={
            "help": (
                "Per-step intra-server launch overhead for allreduce in microseconds. "
                "Used only when intra_server_model=nvlink_analytic."
            )
        },
    )

    # Collective algorithm knobs
    collective_use_triggers: bool = field(
        default=True,
        metadata={"help": "Whether to use trigger-based sequencing in htsim runner."},
    )
    collective_exclude_intra_server: bool = field(
        default=True,
        metadata={"help": "Whether to exclude intra-server traffic from network model."},
    )
    allreduce_model: str = field(
        default="ring_steps",
        metadata={"help": "All-reduce model type."},
    )
    allgather_model: str = field(
        default="ring_steps",
        metadata={"help": "All-gather model type."},
    )
    reducescatter_model: str = field(
        default="ring_steps",
        metadata={"help": "Reduce-scatter model type."},
    )
    alltoall_model: str = field(
        default="pairwise_steps",
        metadata={"help": "All-to-all model type."},
    )
    alltoall_channels: int = field(
        default=8,
        metadata={"help": "All-to-all channels."},
    )
    alltoall_chunk_bytes: int = field(
        default=8_000_000,
        metadata={"help": "All-to-all chunk size in bytes."},
    )
    alltoall_chunk_inflight_per_peer: int = field(
        default=2,
        metadata={"help": "Max in-flight all-to-all chunks per peer."},
    )
    nchannels: int = field(
        default=8,
        metadata={"help": "Parallel ring channels for nccl_ring-style collectives."},
    )
    p2p_src_index: int = field(
        default=0,
        metadata={"help": "P2P source index within a domain group."},
    )
    p2p_dst_index: int = field(
        default=1,
        metadata={"help": "P2P destination index within a domain group."},
    )
    p2p_direction: str = field(
        default="0->1",
        metadata={"help": "P2P direction: 0->1|1->0|bidir."},
    )

    # Runner and backend internals
    runner_end_us: int = field(
        default=20_000_000,
        metadata={"help": "Simulation end time in microseconds."},
    )
    runner_build: bool = field(
        default=False,
        metadata={"help": "Whether to trigger htsim build automatically."},
    )
    runner_progress: bool = field(
        default=False,
        metadata={"help": "Enable runner progress output."},
    )
    runner_print_args: bool = field(
        default=False,
        metadata={"help": "Enable runner argument debug output."},
    )
    runner_stop_on_finished: bool = field(
        default=False,
        metadata={"help": "Stop simulation once all flows complete."},
    )
    runner_heartbeat_s: int = field(
        default=5,
        metadata={"help": "Runner heartbeat interval in seconds."},
    )
    runner_out_dir: str = field(
        default="",
        metadata={
            "help": "Runner output directory. Empty means backend-managed path under cache_dir.",
        },
    )
    prediction_cache_size: int = field(
        default=4096,
        metadata={"help": "In-memory LRU cache size for prediction results."},
    )

    @staticmethod
    def get_type() -> CCBackendType:
        """Return the backend type for this configuration."""
        return CCBackendType.COLLECTIVE_SIM

    @staticmethod
    def get_name() -> str:
        """Return the backend name for this configuration."""
        return "collective_sim"


@dataclass
class AiconfiguratorCCBackendConfig(BaseCCBackendConfig):
    __include_in_cli__ = False
    """
    Internal-only configuration for the aiconfigurator CC backend.

    This backend queries the local aiconfigurator source-tree perf database
    for communication predictions.
    """

    repo_root: str = field(
        default="sota-infer-engine/aiconfigurator",
        metadata={"help": "Path to the local aiconfigurator repository root."},
    )
    system: str = field(
        default="",
        metadata={
            "help": "Aiconfigurator system name (for example h100_sxm). "
            "Empty means infer from Frontier device type at cluster materialization time."
        },
    )
    source_backend: str = field(
        default="vllm",
        metadata={"help": "Aiconfigurator source backend: vllm|sglang|trtllm."},
    )
    source_version: str = field(
        default="",
        metadata={"help": "Aiconfigurator source backend version."},
    )
    database_mode: str = field(
        default="silicon",
        metadata={"help": "Aiconfigurator database mode: silicon|hybrid|empirical|sol|sol_full."},
    )
    tp_allreduce_impl: str = field(
        default="custom_allreduce",
        metadata={"help": "TP allreduce implementation: custom_allreduce|nccl_all_reduce."},
    )
    custom_allreduce_variant: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional runtime label for custom allreduce tables when the raw file contains multiple variants.",
        },
    )

    @staticmethod
    def get_type() -> CCBackendType:
        """Return the backend type for this configuration."""
        return CCBackendType.AICONFIGURATOR

    @staticmethod
    def get_name() -> str:
        """Return the backend name for this configuration."""
        return "aiconfigurator"


@dataclass
class AstraSimAnalyticalCCBackendConfig(BaseCCBackendConfig):
    """
    Configuration for the lightweight ASTRA-Sim analytical CC backend.

    This backend implements Frontier-local analytical communication prediction
    using ASTRA-Sim-inspired topology and ring-collective logic.
    """

    prediction_cache_size: int = field(
        default=4096,
        metadata={"help": "Maximum number of cached analytical predictions."},
    )
    placement_order: str = field(
        default="TP,CP,DP,EP",
        metadata={"help": "Rank placement order for runtime domain mapping."},
    )
    cluster_servers: int = field(
        default=1,
        metadata={
            "help": "Number of servers in the materialized cluster topology.",
            "include_in_cli": False,
        },
    )
    cluster_gpus_per_server: int = field(
        default=8,
        metadata={
            "help": "Number of GPUs per server in the materialized cluster topology.",
            "include_in_cli": False,
        },
    )
    runtime_num_replicas: Optional[int] = field(
        default=None,
        metadata={
            "help": "Frontier runtime cluster replica count for per-call layout mapping.",
            "include_in_cli": False,
        },
    )
    runtime_num_pipeline_stages: Optional[int] = field(
        default=None,
        metadata={
            "help": "Frontier runtime pipeline stage count for per-call layout mapping.",
            "include_in_cli": False,
        },
    )
    runtime_attn_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Frontier runtime attention TP size for per-call layout mapping.",
            "include_in_cli": False,
        },
    )
    runtime_attn_data_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Frontier runtime attention DP size for per-call layout mapping.",
            "include_in_cli": False,
        },
    )
    runtime_moe_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Frontier runtime MoE TP size for per-call layout mapping.",
            "include_in_cli": False,
        },
    )
    runtime_moe_expert_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Frontier runtime MoE EP size for per-call layout mapping.",
            "include_in_cli": False,
        },
    )
    intra_server_topology: str = field(
        default="FullyConnected",
        metadata={"help": "Intra-server topology primitive: FullyConnected|Switch|Ring."},
    )
    inter_server_topology: str = field(
        default="FullyConnected",
        metadata={"help": "Inter-server topology primitive: FullyConnected|Switch|Ring."},
    )
    intra_server_bandwidth_gbps: float = field(
        default=600.0,
        metadata={"help": "Intra-server link bandwidth in Gbps."},
    )
    intra_server_latency_us: float = field(
        default=1.0,
        metadata={"help": "Intra-server link latency in microseconds."},
    )
    inter_server_bandwidth_gbps: float = field(
        default=100.0,
        metadata={"help": "Inter-server link bandwidth in Gbps."},
    )
    inter_server_latency_us: float = field(
        default=1.0,
        metadata={"help": "Inter-server link latency in microseconds."},
    )
    ring_bidirectional: bool = field(
        default=True,
        metadata={"help": "Whether ring topologies use the shortest bidirectional path."},
    )
    p2p_src_index: int = field(
        default=0,
        metadata={"help": "Source participant index for point-to-point predictions."},
    )
    p2p_dst_index: int = field(
        default=1,
        metadata={"help": "Destination participant index for point-to-point predictions."},
    )

    @staticmethod
    def get_type() -> CCBackendType:
        """Return the backend type for this configuration."""
        return CCBackendType.ASTRA_SIM_ANALYTICAL

    @staticmethod
    def get_name() -> str:
        """Return the backend name for this configuration."""
        return "astra_sim_analytical"
