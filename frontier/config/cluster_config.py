"""The cluster topology configuration and its flat per-role field surface.

`ClusterConfig` declares one flat field for every role-specific override the
CLI exposes, validates their combinations, and sets up either the single
monolithic cluster or the disaggregated role clusters.  Construction of the
per-role configurations lives in `cluster_role_config`, topology reporting in
`cluster_topology_summary`.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Dict, List, Optional

from frontier.config.cluster_role_config import (
    ClusterRoleConfigBuilder,
    _get_cc_backend_configs,
)
from frontier.config.cluster_scheduler_config import (
    BaseClusterSchedulerConfig,
    RoundRobinClusterSchedulerConfig,
)
from frontier.config.cluster_topology_summary import ClusterTopologySummary
from frontier.config.execution_time_predictor_config import (
    BaseExecutionTimePredictorConfig,
    RandomForrestExecutionTimePredictorConfig,
)
from frontier.config.parallel_semantics import (
    FrontierParallelismMapping,
    validate_frontier_shared_parallel_domains,
)
from frontier.config.release_guards import (
    AICONFIGURATOR_BACKEND_RELEASE_ERROR,
    DISAGGREGATED_CLUSTER_FIELD_NAMES,
    DISAGGREGATED_CLUSTER_FIELD_PREFIXES,
)
from frontier.config.replica_config import ReplicaConfig
from frontier.config.replica_scheduler_config import (
    BaseReplicaSchedulerConfig,
    SarathiSchedulerConfig,
)
from frontier.logger import init_logger
from frontier.types import ClusterSchedulerType, ClusterType

logger = init_logger(__name__)


@dataclass
class ClusterConfig(ClusterRoleConfigBuilder, ClusterTopologySummary):
    # === Common fields for all modes ===
    cluster_scheduler_config: BaseClusterSchedulerConfig = field(
        default_factory=RoundRobinClusterSchedulerConfig,
        metadata={
            "help": "Cluster scheduler config.",
        },
    )
    replica_scheduler_config: BaseReplicaSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig,
        metadata={"help": "Replica scheduler config."},
    )
    cluster_type: ClusterType = field(
        default=None,
        metadata={
            "help": "Type of the cluster: monolithic, prefill, decode-attn, or decode-ffn."
        },
    )
    execution_time_predictor_config: BaseExecutionTimePredictorConfig = field(
        default_factory=RandomForrestExecutionTimePredictorConfig,
        metadata={"help": "Execution time predictor config."},
    )
    cc_backend_config: BaseCCBackendConfig = field(
        default_factory=lambda: _get_cc_backend_configs()[5](),  # AstraSimAnalyticalCCBackendConfig
        metadata={
            "help": "CC (Collective Communication) backend config for communication latency prediction."
        },
    )

    # === co-location/Monolithic mode fields ===
    num_replicas: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of replicas",
        },
    )
    replica_config: Optional[ReplicaConfig] = field(
        default_factory=lambda: ReplicaConfig(model_name="meta-llama/Llama-2-7b-hf"),
        metadata={
            "help": "Replica configuration",
        },
    )

    # === Disaggregated mode fields ===
    prefill_cluster_num_replicas: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of replicas for prefill cluster. Used only in pd-af-disaggregation mode.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cluster_num_replicas: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of replicas for decode attention cluster. Used only in pd-af-disaggregation mode.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cluster_num_replicas: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of replicas for decode FFN cluster. "
            "Each replica is an independent FFN serving copy; MoE EP lanes are "
            "scoped inside the selected replica. Used only in pd-af-disaggregation mode.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_cluster_num_replicas: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of replicas for unified decode cluster. Used only in pd-disaggregation mode.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    prefill_replica_config_memory_margin_fraction: Optional[float] = field(
        default=None,
        metadata={
            "help": "Memory margin fraction for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_num_pipeline_stages: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of pipeline stages for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_attn_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Attention tensor parallel size for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_moe_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE tensor parallel size for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_moe_expert_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE expert parallel size for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_total_expert_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "Total expert number for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_local_expert_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "Local expert number for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_router_load_balancing_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "MOE router load balancing type for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_router_topk: Optional[int] = field(
        default=None,
        metadata={
            "help": "Router topk for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "Device for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    prefill_replica_config_network_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "Network device for prefill cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_config_memory_margin_fraction: Optional[float] = field(
        default=None,
        metadata={
            "help": "Memory margin fraction for decode attention cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_config_num_pipeline_stages: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of pipeline stages for decode attention cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_config_attn_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Attention tensor parallel size for decode attention cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_config_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "Device for decode attention cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_config_network_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "Network device for decode attention cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_memory_margin_fraction: Optional[float] = field(
        default=None,
        metadata={
            "help": "Memory margin fraction for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_num_pipeline_stages: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of pipeline stages for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_moe_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE tensor parallel size for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_moe_expert_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE expert parallel size for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_total_expert_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "Total expert number for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_local_expert_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "Local expert number for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_router_load_balancing_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "MOE router load balancing type for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_router_topk: Optional[int] = field(
        default=None,
        metadata={
            "help": "Router topk for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "Device for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_config_network_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "Network device for decode FFN cluster.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )

    # === PD-Disaggregation Mode: Unified DECODE Cluster Configuration ===
    decode_replica_config_memory_margin_fraction: Optional[float] = field(
        default=None,
        metadata={
            "help": "Memory margin fraction for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_num_pipeline_stages: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of pipeline stages for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_attn_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Attention tensor parallel size for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_moe_tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE tensor parallel size for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_moe_expert_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE expert parallel size for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_total_expert_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "Total expert number for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_local_expert_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "Local expert number for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_router_load_balancing_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "MOE router load balancing type for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_router_topk: Optional[int] = field(
        default=None,
        metadata={
            "help": "Router topk for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "Device for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_config_network_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "Network device for unified decode cluster.",
            "mode_dependency": "pd-disaggregation",
        },
    )

    # === AF Pipeline Configuration ===
    # This field is for internal use by the created cluster-specific configs
    af_pipeline_num_micro_batch: int = field(
        default=-1,
        metadata={
            "help": "Internal field for the number of micro-batches. Should be set via cluster-specific parameters below.",
        },
    )

    # User-facing parameters for setting micro-batch number in decode clusters
    decode_attn_af_pipeline_num_micro_batch: Optional[int] = field(
        default=None,
        metadata={"help": "Number of micro-batches for the decode_attn cluster."},
    )
    decode_ffn_af_pipeline_num_micro_batch: Optional[int] = field(
        default=None,
        metadata={"help": "Number of micro-batches for the decode_ffn cluster."},
    )

    # User-facing parameter for setting micro-batch SIZE specifically for decode-attn
    decode_attn_micro_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Target micro-batch SIZE for decode-attn cluster (per (replica, dp)).",
        },
    )

    # User-facing parameter for setting request allocation threshold for decode-attn
    decode_attn_request_allocation_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": "Request accumulation threshold for decode-attn cluster. "
            "Only trigger allocation when accumulated requests reach this threshold. "
            "Default: None (equals total number of requests in offline mode).",
        },
    )

    # === AFD CUDA Graph Configuration ===
    # Aligned with StepFun-vLLM's cudagraph_batch_sizes for AFD attention server
    decode_attn_use_cuda_graph: bool = field(
        default=False,
        metadata={
            "help": "Deprecated. Use SimulationConfig.use_cuda_graph instead. "
            "CUDA Graph is now a global setting for pd-af-disaggregation.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cudagraph_capture_sizes: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Deprecated. Use SimulationConfig.cudagraph_capture_sizes instead. "
            "CUDA Graph capture sizes are now shared across decode-attn and decode-ffn.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )

    decode_attn_replica_id_start_for_ffn: Optional[int] = field(
        default=None,
        metadata={
            "help": "Derived first global replica id for DECODE_ATTN lanes; used by DECODE_FFN grouping.",
        },
    )

    # === Per-Cluster Replica Scheduler Configuration ===
    # These fields allow per-cluster-type customization of replica scheduler parameters
    # If not set, they fall back to the base replica_scheduler_config values

    # PREFILL cluster scheduler configuration
    prefill_replica_scheduler_config_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Replica scheduler type for prefill cluster. Overrides base replica_scheduler_config_type.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_replica_scheduler_config_batch_size_cap: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size cap (max_num_seqs) for prefill cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_replica_scheduler_config_max_tokens_in_batch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max tokens in batch (max_num_batched_tokens) for prefill cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_replica_scheduler_config_enable_chunked_prefill: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Enable Chunked Prefill for the prefill cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_replica_scheduler_config_long_prefill_token_threshold: Optional[int] = (
        field(
            default=None,
            metadata={
                "help": "Long-prefill token threshold for the prefill cluster replica scheduler.",
                "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
            },
        )
    )
    prefill_replica_scheduler_config_num_blocks: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of blocks for prefill cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_replica_scheduler_config_block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Block size for prefill cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_replica_scheduler_config_watermark_blocks_fraction: Optional[float] = field(
        default=None,
        metadata={
            "help": "Watermark blocks fraction for prefill cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )

    # DECODE cluster scheduler configuration (for unified decode in pd-disaggregation mode)
    decode_replica_scheduler_config_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Replica scheduler type for decode cluster. Overrides base replica_scheduler_config_type.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_scheduler_config_batch_size_cap: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size cap (max_num_seqs) for decode cluster replica scheduler.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_scheduler_config_max_tokens_in_batch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max tokens in batch (max_num_batched_tokens) for decode cluster replica scheduler.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_scheduler_config_num_blocks: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of blocks for decode cluster replica scheduler.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_scheduler_config_block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Block size for decode cluster replica scheduler.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_replica_scheduler_config_watermark_blocks_fraction: Optional[float] = field(
        default=None,
        metadata={
            "help": "Watermark blocks fraction for decode cluster replica scheduler.",
            "mode_dependency": "pd-disaggregation",
        },
    )

    # DECODE_ATTN cluster scheduler configuration (for pd-af-disaggregation mode)
    decode_attn_replica_scheduler_config_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Replica scheduler type for decode attention cluster. Overrides base replica_scheduler_config_type.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_scheduler_config_batch_size_cap: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size cap (max_num_seqs) for decode attention cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_scheduler_config_max_tokens_in_batch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max tokens in batch (max_num_batched_tokens) for decode attention cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_scheduler_config_num_blocks: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of blocks for decode attention cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_scheduler_config_block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Block size for decode attention cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_replica_scheduler_config_watermark_blocks_fraction: Optional[float] = (
        field(
            default=None,
            metadata={
                "help": "Watermark blocks fraction for decode attention cluster replica scheduler.",
                "mode_dependency": "pd-af-disaggregation",
            },
        )
    )

    # DECODE_FFN cluster scheduler configuration (for pd-af-disaggregation mode)
    decode_ffn_replica_scheduler_config_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Replica scheduler type for decode FFN cluster. Overrides base replica_scheduler_config_type.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_scheduler_config_batch_size_cap: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size cap (max_num_seqs) for decode FFN cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_scheduler_config_max_tokens_in_batch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max tokens in batch (max_num_batched_tokens) for decode FFN cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_scheduler_config_num_blocks: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of blocks for decode FFN cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_scheduler_config_block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Block size for decode FFN cluster replica scheduler.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_replica_scheduler_config_watermark_blocks_fraction: Optional[float] = (
        field(
            default=None,
            metadata={
                "help": "Watermark blocks fraction for decode FFN cluster replica scheduler.",
                "mode_dependency": "pd-af-disaggregation",
            },
        )
    )

    # === Per-Cluster CC Backend Configuration ===
    # These fields allow per-cluster-type customization of CC backend parameters
    # If not set, they fall back to the base cc_backend_config values

    # PREFILL cluster CC backend configuration
    prefill_cc_backend_config_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "CC backend type for prefill cluster. Options: 'vidur', 'analytical', 'collective_sim', 'astra_sim_analytical'. Overrides base cc_backend_config type.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_network_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Network bandwidth in Gbps for prefill cluster CC backend (analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_network_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Network latency in microseconds for prefill cluster CC backend (analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_intra_node_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-node bandwidth in Gbps for prefill cluster CC backend (analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_repo_root: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication backend repo root for prefill cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_system: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication backend system for prefill cluster CC backend (internal-only mode). Empty means infer from device.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_source_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication source backend for prefill cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_source_version: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication source version for prefill cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_database_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication database mode for prefill cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_tp_allreduce_impl: Optional[str] = field(
        default=None,
        metadata={
            "help": "TP allreduce implementation for prefill cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_custom_allreduce_variant: Optional[str] = field(
        default=None,
        metadata={
            "help": "Custom allreduce runtime label for prefill cluster CC backend when internal communication backend raw data has multiple variants.",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_prediction_cache_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Prediction cache size for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_placement_order: Optional[str] = field(
        default=None,
        metadata={
            "help": "Rank placement order for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_intra_server_topology: Optional[str] = field(
        default=None,
        metadata={
            "help": "Intra-server topology for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_inter_server_topology: Optional[str] = field(
        default=None,
        metadata={
            "help": "Inter-server topology for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_intra_server_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-server bandwidth in Gbps for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_intra_server_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-server latency in microseconds for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_inter_server_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Inter-server bandwidth in Gbps for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_inter_server_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Inter-server latency in microseconds for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_p2p_src_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "P2P source participant index for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_p2p_dst_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "P2P destination participant index for prefill cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_cc_backend_config_nvlink_allreduce_launch_overhead_us: Optional[float] = (
        field(
            default=None,
            metadata={
                "help": (
                    "Per-step intra-server allreduce launch overhead in microseconds "
                    "for prefill cluster collective-sim backend."
                ),
                "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
            },
        )
    )
    prefill_execution_time_predictor_config_mlp_up_proj_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override mlp_up_proj calibration scale for the prefill cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_execution_time_predictor_config_attn_pre_proj_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override attn_pre_proj calibration scale for the prefill cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_execution_time_predictor_config_attn_post_proj_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override attn_post_proj calibration scale for the prefill cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_execution_time_predictor_config_attn_decode_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override attn_decode calibration scale for the prefill cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_execution_time_predictor_config_attn_kv_cache_save_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override attn_kv_cache_save calibration scale for the prefill cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_execution_time_predictor_config_mlp_down_proj_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override mlp_down_proj calibration scale for the prefill cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )

    # DECODE cluster CC backend configuration (for unified decode in pd-disaggregation mode)
    decode_cc_backend_config_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "CC backend type for decode cluster. Options: 'vidur', 'analytical', 'collective_sim', 'astra_sim_analytical'. Overrides base cc_backend_config type.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_network_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Network bandwidth in Gbps for decode cluster CC backend (analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_network_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Network latency in microseconds for decode cluster CC backend (analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_intra_node_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-node bandwidth in Gbps for decode cluster CC backend (analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_repo_root: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication backend repo root for decode cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_system: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication backend system for decode cluster CC backend (internal-only mode). Empty means infer from device.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_source_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication source backend for decode cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_source_version: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication source version for decode cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_database_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication database mode for decode cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_tp_allreduce_impl: Optional[str] = field(
        default=None,
        metadata={
            "help": "TP allreduce implementation for decode cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_custom_allreduce_variant: Optional[str] = field(
        default=None,
        metadata={
            "help": "Custom allreduce runtime label for decode cluster CC backend when internal communication backend raw data has multiple variants.",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_prediction_cache_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Prediction cache size for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_placement_order: Optional[str] = field(
        default=None,
        metadata={
            "help": "Rank placement order for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_intra_server_topology: Optional[str] = field(
        default=None,
        metadata={
            "help": "Intra-server topology for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_inter_server_topology: Optional[str] = field(
        default=None,
        metadata={
            "help": "Inter-server topology for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_intra_server_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-server bandwidth in Gbps for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_intra_server_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-server latency in microseconds for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_inter_server_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Inter-server bandwidth in Gbps for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_inter_server_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Inter-server latency in microseconds for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_p2p_src_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "P2P source participant index for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_p2p_dst_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "P2P destination participant index for decode cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_cc_backend_config_nvlink_allreduce_launch_overhead_us: Optional[float] = (
        field(
            default=None,
            metadata={
                "help": (
                    "Per-step intra-server allreduce launch overhead in microseconds "
                    "for decode cluster collective-sim backend."
                ),
                "mode_dependency": "pd-disaggregation",
            },
        )
    )
    decode_execution_time_predictor_config_mlp_up_proj_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override mlp_up_proj calibration scale for the decode cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_execution_time_predictor_config_attn_pre_proj_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override attn_pre_proj calibration scale for the decode cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_execution_time_predictor_config_attn_post_proj_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override attn_post_proj calibration scale for the decode cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_execution_time_predictor_config_attn_decode_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override attn_decode calibration scale for the decode cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_execution_time_predictor_config_attn_kv_cache_save_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override attn_kv_cache_save calibration scale for the decode cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_execution_time_predictor_config_mlp_down_proj_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override mlp_down_proj calibration scale for the decode cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_execution_time_predictor_config_decode_phase_mlp_down_proj_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override decode-phase-only mlp_down_proj calibration scale for the "
                "decode cluster execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-disaggregation",
        },
    )

    # DECODE_ATTN cluster CC backend configuration (for pd-af-disaggregation mode)
    decode_attn_cc_backend_config_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "CC backend type for decode attention cluster. Options: 'vidur', 'analytical', 'collective_sim', 'astra_sim_analytical'. Overrides base cc_backend_config type.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_network_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Network bandwidth in Gbps for decode attention cluster CC backend (analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_network_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Network latency in microseconds for decode attention cluster CC backend (analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_intra_node_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-node bandwidth in Gbps for decode attention cluster CC backend (analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_repo_root: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication backend repo root for decode attention cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_system: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication backend system for decode attention cluster CC backend (internal-only mode). Empty means infer from device.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_source_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication source backend for decode attention cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_source_version: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication source version for decode attention cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_database_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication database mode for decode attention cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_tp_allreduce_impl: Optional[str] = field(
        default=None,
        metadata={
            "help": "TP allreduce implementation for decode attention cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_custom_allreduce_variant: Optional[str] = field(
        default=None,
        metadata={
            "help": "Custom allreduce runtime label for decode attention cluster CC backend when internal communication backend raw data has multiple variants.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_prediction_cache_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Prediction cache size for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_placement_order: Optional[str] = field(
        default=None,
        metadata={
            "help": "Rank placement order for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_intra_server_topology: Optional[str] = field(
        default=None,
        metadata={
            "help": "Intra-server topology for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_inter_server_topology: Optional[str] = field(
        default=None,
        metadata={
            "help": "Inter-server topology for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_intra_server_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-server bandwidth in Gbps for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_intra_server_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-server latency in microseconds for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_inter_server_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Inter-server bandwidth in Gbps for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_inter_server_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Inter-server latency in microseconds for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_p2p_src_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "P2P source participant index for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_attn_cc_backend_config_p2p_dst_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "P2P destination participant index for decode attention cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )

    # DECODE_FFN cluster CC backend configuration (for pd-af-disaggregation mode)
    decode_ffn_cc_backend_config_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "CC backend type for decode FFN cluster. Options: 'vidur', 'analytical', 'collective_sim', 'astra_sim_analytical'. Overrides base cc_backend_config type.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_network_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Network bandwidth in Gbps for decode FFN cluster CC backend (analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_network_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Network latency in microseconds for decode FFN cluster CC backend (analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_intra_node_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-node bandwidth in Gbps for decode FFN cluster CC backend (analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_repo_root: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication backend repo root for decode FFN cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_system: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication backend system for decode FFN cluster CC backend (internal-only mode). Empty means infer from device.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_source_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication source backend for decode FFN cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_source_version: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication source version for decode FFN cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_database_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "Internal-only communication database mode for decode FFN cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_tp_allreduce_impl: Optional[str] = field(
        default=None,
        metadata={
            "help": "TP allreduce implementation for decode FFN cluster CC backend (internal-only mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_custom_allreduce_variant: Optional[str] = field(
        default=None,
        metadata={
            "help": "Custom allreduce runtime label for decode FFN cluster CC backend when internal communication backend raw data has multiple variants.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_prediction_cache_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Prediction cache size for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_placement_order: Optional[str] = field(
        default=None,
        metadata={
            "help": "Rank placement order for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_intra_server_topology: Optional[str] = field(
        default=None,
        metadata={
            "help": "Intra-server topology for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_inter_server_topology: Optional[str] = field(
        default=None,
        metadata={
            "help": "Inter-server topology for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_intra_server_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-server bandwidth in Gbps for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_intra_server_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Intra-server latency in microseconds for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_inter_server_bandwidth_gbps: Optional[float] = field(
        default=None,
        metadata={
            "help": "Inter-server bandwidth in Gbps for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_inter_server_latency_us: Optional[float] = field(
        default=None,
        metadata={
            "help": "Inter-server latency in microseconds for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_p2p_src_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "P2P source participant index for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    decode_ffn_cc_backend_config_p2p_dst_index: Optional[int] = field(
        default=None,
        metadata={
            "help": "P2P destination participant index for decode FFN cluster CC backend (astra_sim_analytical mode).",
            "mode_dependency": "pd-af-disaggregation",
        },
    )

    def __post_init__(self):
        self._validate_open_source_release_cc_backend_guard()

        # check and set args only in first init (not in Cluster())
        if self.cluster_type is None:
            # Early validation based on mode
            self._validate_mode_consistency()

            # Basic validation for micro-batch size if provided
            if self.decode_attn_micro_batch_size is not None:
                assert (
                    self.decode_attn_micro_batch_size >= 1
                ), f"decode_attn_micro_batch_size must be >=1, got {self.decode_attn_micro_batch_size}"

            # Ensure micro_batch_size equals batch_size_cap for DECODE_ATTN
            # In DECODE_ATTN, micro-batch and batch are semantically equivalent
            if (
                self.decode_attn_micro_batch_size is not None
                and self.decode_attn_replica_scheduler_config_batch_size_cap is not None
            ):
                # Both specified: enforce equality
                if (
                    self.decode_attn_micro_batch_size
                    != self.decode_attn_replica_scheduler_config_batch_size_cap
                ):
                    raise ValueError(
                        f"DECODE_ATTN micro_batch_size ({self.decode_attn_micro_batch_size}) "
                        f"must equal batch_size_cap ({self.decode_attn_replica_scheduler_config_batch_size_cap}). "
                        f"Reason: In DECODE_ATTN, micro-batch and batch are semantically equivalent."
                    )
            elif self.decode_attn_micro_batch_size is not None:
                # Only micro_batch_size specified: propagate to batch_size_cap
                self.decode_attn_replica_scheduler_config_batch_size_cap = (
                    self.decode_attn_micro_batch_size
                )
            elif self.decode_attn_replica_scheduler_config_batch_size_cap is not None:
                # Only batch_size_cap specified: propagate to micro_batch_size
                self.decode_attn_micro_batch_size = (
                    self.decode_attn_replica_scheduler_config_batch_size_cap
                )
            # If neither is specified, keep both as None (use defaults later)

            if self._has_disaggregation_params_set():
                self._setup_disaggregated_configs()

                # Add a check to ensure af_pipeline_num_micro_batch is consistent (only for PD+AF mode)
                is_pd_af_mode = (
                    self.decode_attn_cluster_num_replicas is not None
                    and self.decode_ffn_cluster_num_replicas is not None
                )
                if is_pd_af_mode:
                    attn_mb = self.decode_attn_af_pipeline_num_micro_batch
                    ffn_mb = self.decode_ffn_af_pipeline_num_micro_batch

                    assert (
                        attn_mb is not None and ffn_mb is not None
                    ), "In PD+AF disaggregated mode, both decode_attn_af_pipeline_num_micro_batch and decode_ffn_af_pipeline_num_micro_batch must be set."

                    assert (
                        attn_mb == ffn_mb
                    ), "The af_pipeline_num_micro_batch must be the same for both decode_attn and decode_ffn clusters."

                    # AFD Divisibility Validation (Fail Fast Strategy)
                    # Aligned with StepFun-vLLM's requirement that batch sizes be divisible by num_stages
                    # Unlike StepFun which silently rounds down, we fail fast to help users identify
                    # configuration issues early.
                    num_stages = attn_mb
                    if num_stages > 1:
                        self._validate_afd_divisibility(num_stages)
            else:
                self._setup_monolithic_config()

            self._validate_prefix_cache_spec_decode_compatibility()

    def _validate_prefix_cache_spec_decode_compatibility(self) -> None:
        from frontier.spec_decode.runtime import (
            method_requires_prefix_matching_disabled,
        )

        prefix_enabled = bool(
            getattr(self.replica_scheduler_config, "enable_prefix_caching", False)
        )
        if not prefix_enabled:
            return

        replica_configs = [
            ("replica_config", getattr(self, "replica_config", None)),
            ("prefill_replica_config", getattr(self, "prefill_replica_config", None)),
            ("decode_replica_config", getattr(self, "decode_replica_config", None)),
            (
                "decode_attn_replica_config",
                getattr(self, "decode_attn_replica_config", None),
            ),
            (
                "decode_ffn_replica_config",
                getattr(self, "decode_ffn_replica_config", None),
            ),
        ]
        for replica_config_name, replica_config in replica_configs:
            if replica_config is None:
                continue
            spec_decode_config = getattr(
                replica_config, "speculative_decoding_config", None
            )
            if spec_decode_config is None or not spec_decode_config.enabled:
                continue
            method = str(getattr(spec_decode_config, "method", "")).strip()
            if method and method_requires_prefix_matching_disabled(method):
                raise ValueError(
                    "Speculative decoding method "
                    f"{method!r} requires prefix caching to be disabled, "
                    f"but replica_scheduler_config.enable_prefix_caching=True "
                    f"for {replica_config_name}."
                )

    def _validate_afd_divisibility(self, num_stages: int):
        """Validate that key batch size parameters are divisible by num_stages.

        Currently relaxed — no divisibility enforcement.
        """
        return None

    def _validate_mode_consistency(self):
        """Validate that configuration is consistent with the intended mode."""

        has_disaggregated_fields = (
            self.prefill_cluster_num_replicas is not None
            or self.decode_attn_cluster_num_replicas is not None
            or self.decode_ffn_cluster_num_replicas is not None
            or self.decode_cluster_num_replicas is not None
        )

        # The `num_replicas` field is exclusively for monolithic mode.
        # `replica_config` can be used as a template in disaggregated mode, so its presence is not a conflict.
        has_monolithic_exclusive_field = self.num_replicas is not None

        if has_disaggregated_fields and has_monolithic_exclusive_field:
            logger.warning(
                "Both disaggregated and monolithic configuration fields are set. "
                "The 'num_replicas' field (for monolithic mode) was provided but will be ignored in disaggregated mode. "
                "Please use cluster-specific replica counts like 'prefill_cluster_num_replicas'."
            )

    def _validate_open_source_release_cc_backend_guard(self) -> None:
        from frontier.cc_backend.cc_backend_config import AiconfiguratorCCBackendConfig

        if isinstance(self.cc_backend_config, AiconfiguratorCCBackendConfig):
            raise ValueError(AICONFIGURATOR_BACKEND_RELEASE_ERROR)

    def _setup_monolithic_config(self):
        """Setup configuration for monolithic (co-location) mode."""
        # Ensure required fields are set for monolithic mode
        assert self.num_replicas != None, "Num replicas must be set"
        assert self.replica_config != None, "Replica config must be set"

        # Set cluster type for monolithic mode
        self.cluster_type = ClusterType.MONOLITHIC

        # Predictor routing details are indexed by serving Replica identity.
        # Keep that capacity dimension explicit instead of deriving it from
        # attention-DP lanes.
        self.replica_config.cluster_num_replicas = int(self.num_replicas)

        # Reuse the same parallel-domain validation used by disaggregated clusters so
        # monolithic MoE layouts fail fast when attention and MoE domains disagree.
        self._validate_replica_config(self.replica_config, "monolithic")
        self.world_size = self.replica_config.world_size * self.num_replicas

        # Clear disaggregated fields to avoid confusion
        self.prefill_replica_config = None
        self.decode_attn_replica_config = None
        self.decode_ffn_replica_config = None
        self.prefill_cluster_num_replicas = None
        self.decode_attn_cluster_num_replicas = None
        self.decode_ffn_cluster_num_replicas = None

        # else:
        #     assert self.replica_config.expert_parallel_size == self.replica_config.tensor_parallel_size, "For local MoE, expert_parallel_size must be equal to tensor_parallel_size"

    def _setup_disaggregated_configs(self):
        """Setup configuration for disaggregated mode (PD or PD+AF)."""
        # Clear monolithic fields since they're not used
        # self.replica_config = None
        self.num_replicas = None

        # Determine disaggregation mode
        is_pd_af_mode = (
            self.decode_attn_cluster_num_replicas is not None
            and self.decode_ffn_cluster_num_replicas is not None
        )
        is_pd_mode = self.decode_cluster_num_replicas is not None

        # Ensure required disaggregated fields are set
        assert (
            self.prefill_cluster_num_replicas != None
        ), "Prefill cluster num replicas must be set"

        # Requirement 10.4: Validate that replica counts are positive
        if (
            self.prefill_cluster_num_replicas is not None
            and self.prefill_cluster_num_replicas <= 0
        ):
            raise ValueError(
                f"prefill_cluster_num_replicas must be positive, got {self.prefill_cluster_num_replicas}"
            )

        if is_pd_af_mode:
            # PD+AF disaggregation mode
            assert (
                self.decode_attn_cluster_num_replicas != None
            ), "Decode attention cluster num replicas must be set"
            assert (
                self.decode_ffn_cluster_num_replicas != None
            ), "Decode FFN cluster num replicas must be set"
            assert (
                not is_pd_mode
            ), "Cannot set both PD and PD+AF disaggregation parameters"

            # Requirement 10.4: Validate that replica counts are positive (PD+AF mode)
            if self.decode_attn_cluster_num_replicas <= 0:
                raise ValueError(
                    f"decode_attn_cluster_num_replicas must be positive, got {self.decode_attn_cluster_num_replicas}"
                )
            if self.decode_ffn_cluster_num_replicas <= 0:
                raise ValueError(
                    f"decode_ffn_cluster_num_replicas must be positive, got {self.decode_ffn_cluster_num_replicas}"
                )

            # DECODE_FFN grouping semantics are implemented only in the
            # RoundRobinClusterScheduler path.
            cluster_scheduler_type = self.cluster_scheduler_config.get_type()
            if cluster_scheduler_type != ClusterSchedulerType.ROUND_ROBIN:
                raise ValueError(
                    "PD+AF mode requires RoundRobin cluster scheduler when DECODE_FFN is enabled. "
                    f"Got cluster_scheduler_config_type={cluster_scheduler_type}."
                )

            for field_name, decode_role in (
                (
                    "decode_attn_replica_config_num_pipeline_stages",
                    "decode_attn",
                ),
                (
                    "decode_ffn_replica_config_num_pipeline_stages",
                    "decode_ffn",
                ),
            ):
                pipeline_stages = getattr(self, field_name)
                if pipeline_stages not in (None, 1):
                    raise ValueError(
                        f"{field_name} must be 1 for {decode_role}, "
                        f"got {pipeline_stages}."
                    )

            # Create ReplicaConfig objects from flattened fields
            self.prefill_replica_config = self._create_replica_config_from_fields(
                "prefill"
            )
            self.decode_attn_replica_config = self._create_replica_config_from_fields(
                "decode_attn"
            )
            self.decode_ffn_replica_config = self._create_replica_config_from_fields(
                "decode_ffn"
            )

            # Enforce disaggregation constraints
            assert (
                self.decode_attn_replica_config.num_pipeline_stages == 1
            ), "Decode attention cluster must have 1 pipeline stage"
            assert (
                self.decode_ffn_replica_config.num_pipeline_stages == 1
            ), "Decode FFN cluster must have 1 pipeline stage"

            # Validate each cluster
            self._validate_replica_config(self.prefill_replica_config, "prefill")
            self._validate_replica_config(
                self.decode_attn_replica_config, "decode_attn"
            )
            self._validate_replica_config(self.decode_ffn_replica_config, "decode_ffn")

            # Calculate world sizes
            self.prefill_world_size = (
                self.prefill_cluster_num_replicas
                * self.prefill_replica_config.world_size
            )
            self.decode_attn_world_size = (
                self.decode_attn_cluster_num_replicas
                * self.decode_attn_replica_config.world_size
            )
            self.decode_ffn_world_size = (
                self.decode_ffn_cluster_num_replicas
                * self.decode_ffn_replica_config.world_size
            )
            self.world_size = (
                self.prefill_world_size
                + self.decode_attn_world_size
                + self.decode_ffn_world_size
            )

        elif is_pd_mode:
            # PD disaggregation mode
            assert (
                self.decode_cluster_num_replicas != None
            ), "Decode cluster num replicas must be set"
            assert (
                not is_pd_af_mode
            ), "Cannot set both PD and PD+AF disaggregation parameters"

            # Requirement 10.4: Validate that replica counts are positive (PD mode)
            if self.decode_cluster_num_replicas <= 0:
                raise ValueError(
                    f"decode_cluster_num_replicas must be positive, got {self.decode_cluster_num_replicas}"
                )

            # Create ReplicaConfig objects from flattened fields
            self.prefill_replica_config = self._create_replica_config_from_fields(
                "prefill"
            )
            self.decode_replica_config = self._create_replica_config_from_fields(
                "decode"
            )

            # Validate each cluster
            self._validate_replica_config(self.prefill_replica_config, "prefill")
            self._validate_replica_config(self.decode_replica_config, "decode")

            # Calculate world sizes
            self.prefill_world_size = (
                self.prefill_cluster_num_replicas
                * self.prefill_replica_config.world_size
            )
            self.decode_world_size = (
                self.decode_cluster_num_replicas * self.decode_replica_config.world_size
            )
            self.world_size = self.prefill_world_size + self.decode_world_size

        else:
            raise ValueError(
                "Invalid disaggregation configuration: must set either PD or PD+AF parameters"
            )

        # Ensure consistent dummy mode configuration across all clusters
        self._ensure_consistent_dummy_mode()

        print(f"Total world size: {self.world_size}")

    def _ensure_consistent_dummy_mode(self):
        """Ensure all clusters use the same dummy mode configuration."""
        # Get the main execution_time_predictor_config dummy mode settings
        main_config = self.execution_time_predictor_config
        main_dummy_mode = main_config.enable_dummy_mode
        main_dummy_time = main_config.dummy_execution_time_ms

        if main_dummy_mode:
            print(f"Applying dummy mode (time={main_dummy_time}ms) to all clusters")

            # Apply dummy mode settings to all cluster configs
            self.execution_time_predictor_config.enable_dummy_mode = True
            self.execution_time_predictor_config.dummy_execution_time_ms = (
                main_dummy_time
            )

            # Note: In the current architecture, all clusters share the same execution_time_predictor_config
            # This ensures consistency across all clusters

    def _field_is_set_to_non_default(self, field_def) -> bool:
        value = getattr(self, field_def.name)
        if field_def.default is not MISSING:
            return value != field_def.default
        if field_def.default_factory is not MISSING:
            return value != field_def.default_factory()
        return value is not None

    def _has_disaggregation_params_set(self) -> bool:
        """Check if any disaggregation-specific cluster fields have been set."""
        for field_def in self.__dataclass_fields__.values():
            if (
                not field_def.name.startswith(DISAGGREGATED_CLUSTER_FIELD_PREFIXES)
                and field_def.name not in DISAGGREGATED_CLUSTER_FIELD_NAMES
            ):
                continue
            if self._field_is_set_to_non_default(field_def):
                return True
        return False

    def _validate_replica_config(
        self, replica_config: ReplicaConfig, cluster_name: str
    ):
        """Validate replica configuration for specific cluster."""
        # Validate pipeline parallelism configuration (double-check, should already be validated in __post_init__)
        if (
            replica_config.model_config.num_layers % replica_config.num_pipeline_stages
            != 0
        ):
            raise ValueError(
                f"Pipeline parallelism configuration error in {cluster_name} cluster: "
                f"num_layers ({replica_config.model_config.num_layers}) must be evenly divisible by "
                f"num_pipeline_stages ({replica_config.num_pipeline_stages}). "
                f"Current configuration would result in uneven layer distribution across pipeline stages."
            )

        # Validate dense model configuration in disaggregated modes.
        is_dense_model = not replica_config.model_config.is_moe
        if is_dense_model and cluster_name in ["prefill", "decode"]:
            # For dense models in PD-disaggregation mode, enforce attn_dp = 1
            if replica_config.attn_dp != 1:
                raise ValueError(
                    f"Dense models in PD-disaggregation mode require attn_dp=1 "
                    f"in {cluster_name} cluster, got {replica_config.attn_dp}. "
                    f"Dense models do not support attn data parallelism in disaggregated mode."
                )
            # Ensure MoE parallelism is disabled for dense models
            if replica_config.moe_expert_parallel_size != 1:
                raise ValueError(
                    f"Dense models require moe_expert_parallel_size=1 in {cluster_name} cluster, "
                    f"got {replica_config.moe_expert_parallel_size}. "
                    f"Dense models do not have expert parallelism."
                )

        if is_dense_model and cluster_name == "decode_attn":
            if replica_config.attn_dp != 1:
                raise ValueError(
                    "Dense models require attn_dp=1 in "
                    f"{cluster_name} cluster, got "
                    f"{replica_config.attn_dp}."
                )

        if is_dense_model and cluster_name == "decode_ffn":
            if replica_config.moe_expert_parallel_size != 1:
                raise ValueError(
                    "Dense models require moe_expert_parallel_size=1 in "
                    f"{cluster_name} cluster, got "
                    f"{replica_config.moe_expert_parallel_size}."
                )
            if replica_config.router_topk != 1:
                raise ValueError(
                    f"Dense models require router_topk=1 in {cluster_name} "
                    f"cluster, got {replica_config.router_topk}."
                )

        normalized_cluster_name = str(cluster_name).strip().lower()
        if normalized_cluster_name in {"prefill", "decode", "monolithic"} and replica_config.model_config.is_moe:
            validate_frontier_shared_parallel_domains(
                FrontierParallelismMapping(
                    cluster_num_replicas=1,
                    attn_tensor_parallel_size=replica_config.attn_tensor_parallel_size,
                    attn_dp=replica_config.attn_dp,
                    moe_tensor_parallel_size=replica_config.moe_tensor_parallel_size,
                    moe_expert_parallel_size=replica_config.moe_expert_parallel_size,
                )
            )

        if cluster_name != "decode_attn":
            pass
            # else:
            #     assert replica_config.moe_expert_parallel_size == replica_config.moe_tensor_parallel_size, f"For local MoE in {cluster_name} cluster, moe_expert_parallel_size must be equal to moe_tensor_parallel_size"
        else:
            assert (
                replica_config.moe_expert_parallel_size == 0
                and replica_config.local_expert_num == 0
            ), "For decode attention cluster, moe_expert_parallel_size and local_expert_num must be 0"
