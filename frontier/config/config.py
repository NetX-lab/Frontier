"""Top-level simulation configuration and the cluster topology it owns.

The workload, scheduler, metrics, speculative-decoding, replica and predictor
configuration families live in sibling modules and are re-exported here so that
the historical import surface is unchanged.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import MISSING, dataclass, field, replace
from datetime import datetime
import json
import os
from typing import Any, Callable, List, Optional, Dict, Tuple, TYPE_CHECKING

from frontier.config.base_poly_config import BasePolyConfig
from frontier.config.device_sku_config import BaseDeviceSKUConfig
from frontier.config.flat_dataclass import create_flat_dataclass
from frontier.config.kv_cache_transfer_config import (
    BaseKVCacheTransferConfig,
    AnalyticalKVCacheTransferConfig,
)
from frontier.config.m2n_transfer_config import (
    BaseM2NTransferConfig,
    AnalyticalM2NTransferConfig,
)

# Use TYPE_CHECKING to avoid circular imports at runtime
# The actual imports are done lazily when needed
if TYPE_CHECKING:
    pass

# NOTE: CC backend configs are imported lazily via _get_cc_backend_configs() to
# avoid a circular import. Do NOT add direct imports here. The lazy import is
# used by the ClusterConfig cc_backend_config default factory and by the
# per-backend creators below.

from frontier.config.model_config import BaseModelConfig
from frontier.config.node_sku_config import BaseNodeSKUConfig
from frontier.config.parallel_semantics import (
    FrontierParallelismMapping,
    resolve_collective_sim_physical_topology,
    validate_frontier_shared_parallel_domains,
)
from frontier.spec_decode.proposer_profile import (
    load_decode_draft_proposer_latency_profile,
)
from frontier.config.utils import dataclass_to_dict
from frontier.logger import init_logger
from frontier.types import (
    ClusterType,
    ExecutionTimePredictorType,
    ClusterSchedulerType,
    ReplicaSchedulerType,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
    CCBackendType,
)
from frontier.utils.output_paths import (
    build_metrics_run_output_dir,
    validate_output_filename,
    validate_run_id,
)

# The configuration families below were split out of this module.  They are
# imported here both because ClusterConfig and SimulationConfig reference them
# and because `frontier/config/__init__.py` re-exports this module with a star
# import, so every historical `from frontier.config import X` and
# `from frontier.config.config import X` keeps resolving.
from frontier.config.release_guards import (
    AICONFIGURATOR_BACKEND_RELEASE_ERROR,
    DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR,
    DISAGGREGATED_CLUSTER_FIELD_NAMES,
    DISAGGREGATED_CLUSTER_FIELD_PREFIXES,
    PD_AF_DISAGGREGATION_PARALLEL_CLUSTER_RELEASE_ERROR,
    PD_AF_PREFIX_CACHING_RELEASE_ERROR,
    PD_AF_TRACE_REPLAY_DEFERRED_ERROR,
    PD_DISAGGREGATION_PARALLEL_CLUSTER_RELEASE_ERROR,
)
from frontier.config.request_generator_config import (
    BaseRequestGeneratorConfig,
    BaseRequestIntervalGeneratorConfig,
    BaseRequestLengthGeneratorConfig,
    FixedRequestLengthGeneratorConfig,
    GammaRequestIntervalGeneratorConfig,
    PoissonRequestIntervalGeneratorConfig,
    StaticRequestIntervalGeneratorConfig,
    SyntheticRequestGeneratorConfig,
    TraceRequestGeneratorConfig,
    TraceRequestIntervalGeneratorConfig,
    TraceRequestLengthGeneratorConfig,
    UniformRequestLengthGeneratorConfig,
    ZipfRequestLengthGeneratorConfig,
)
from frontier.config.replica_scheduler_config import (
    BaseReplicaSchedulerConfig,
    FasterTransformerSchedulerConfig,
    LightllmSchedulerConfig,
    OrcaSchedulerConfig,
    SarathiSchedulerConfig,
    SglangSchedulerConfig,
    Sj2QBoundedCarryoverSchedulerConfig,
    Sj2QFastServeLiteSchedulerConfig,
    Sj2QPenaltyOnlySchedulerConfig,
    Sj2qBoundedCarryoverSchedulerConfig,
    Sj2qFastserveLiteSchedulerConfig,
    Sj2qPenaltyOnlySchedulerConfig,
    VllmSchedulerConfig,
    VllmV1SchedulerConfig,
)
from frontier.config.metrics_config import MetricsConfig
from frontier.config.speculative_decoding_config import SpeculativeDecodingConfig
from frontier.config.replica_config import ReplicaConfig
from frontier.config.cluster_scheduler_config import (
    BaseClusterSchedulerConfig,
    LORClusterSchedulerConfig,
    RandomClusterSchedulerConfig,
    RoundRobinClusterSchedulerConfig,
    StickyLORClusterSchedulerConfig,
    StickyRoundRobinClusterSchedulerConfig,
    VllmLoadBalancingClusterSchedulerConfig,
)
from frontier.config.execution_time_predictor_config import (
    BaseExecutionTimePredictorConfig,
    LinearRegressionExecutionTimePredictorConfig,
    RandomForrestExecutionTimePredictorConfig,
)
from frontier.config.cluster_config import ClusterConfig


logger = init_logger(__name__)


@dataclass
class SimulationConfig(ABC):
    simulation_mode: str = field(
        default="offline",
        metadata={
            "help": "Simulation mode, can be 'online' or 'offline'.",
            "choices": ["online", "offline"],
        },
    )
    offline_use_generated_request_arrivals: bool = field(
        default=False,
        metadata={
            "help": (
                "In offline simulations, replay generated request arrival timestamps "
                "instead of forcing every request to arrive at t=0. Default false "
                "preserves legacy offline batch behavior."
            ),
        },
    )
    sys_arch: str = field(
        default="co-location",
        metadata={
            "help": "System architecture type. 'co-location' for baseline, 'pd-disaggregation' for prefill/decode disaggregation, 'pd-af-disaggregation' for prefill/decode/attention-FFN disaggregation.",
            "choices": ["pd-af-disaggregation", "pd-disaggregation", "co-location"],
        },
    )
    use_cuda_graph: bool = field(
        default=False,
        metadata={
            "help": "Enable CUDA Graph simulation for pd-af-disaggregation.",
        },
    )
    decode_cuda_graph_mode: str = field(
        default="none",
        metadata={
            "help": "Decode CUDA Graph mode for co-location and pd-disaggregation. "
            "Use full_decode_only to pad pure decode batches only, and piecewise "
            "to model segmented CUDA graphs with eager attention.",
            "choices": ["none", "full_decode_only", "piecewise"],
        },
    )
    allow_spec_decode_cuda_graph_diagnostic: bool = field(
        default=False,
        metadata={
            "help": "Comparison-only opt-in for speculative decode CUDA graph "
            "diagnostics. Keeps the default speculative baseline eager-only.",
        },
    )
    cudagraph_capture_sizes: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "CUDA Graph capture sizes shared by decode-attn and decode-ffn. "
            "If not set, defaults follow StepFun-vLLM's capture size generation.",
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Logging level."},
    )
    time_limit: int = field(
        default=0,  # in seconds, 0 is no limit
        metadata={"help": "Time limit for simulation in seconds. 0 means no limit."},
    )
    enable_thinking_mode: bool = field(
        default=False,
        metadata={"help": "Enable multi-round Thinking Mode request execution."},
    )
    thinking_depth: int = field(
        default=1,
        metadata={
            "help": "Number of request rounds when Thinking Mode is enabled. "
            "Depth 1 preserves the original single-round workflow."
        },
    )
    tool_call_latency: float = field(
        default=0.001,
        metadata={
            "help": "Tool call latency in seconds inserted between non-final Thinking "
            "Mode rounds. Default is 1ms."
        },
    )
    thinking_round_prefill_tokens: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Optional explicit hidden-round prefill-token plan. Length must "
            "equal thinking_depth - 1 when provided."
        },
    )
    thinking_round_decode_tokens: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Optional explicit hidden-round decode-token plan. Length must "
            "equal thinking_depth - 1 when provided."
        },
    )
    cluster_config: ClusterConfig = field(
        default_factory=ClusterConfig,
        metadata={"help": "Cluster config."},
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "Request generator config."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Metrics config."},
    )
    kv_cache_transfer_config: BaseKVCacheTransferConfig = field(
        default_factory=AnalyticalKVCacheTransferConfig,
        metadata={"help": "KV cache transfer predictor config."},
    )
    m2n_transfer_config: BaseM2NTransferConfig = field(
        default_factory=AnalyticalM2NTransferConfig,
        metadata={
            "help": "M2N transfer predictor config for decode cluster communication."
        },
    )
    op_quantization_config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Deprecated. Operation-level quantization config is no longer supported; use model config only.",
        },
    )

    # Parallel cluster processing configuration
    enable_parallel_clusters: bool = field(
        default=True,
        metadata={
            "help": (
                "Enable parallel cluster processing for internal correctness tests. "
                "pre-release-v0.3 public pd-disaggregation and "
                "pd-af-disaggregation runs require "
                "--no-enable_parallel_clusters."
            )
        },
    )
    cluster_sync_interval_ms: float = field(
        default=1.0,
        metadata={"help": "Synchronization interval between clusters in milliseconds."},
    )
    max_inter_cluster_queue_size: int = field(
        default=1000,
        metadata={"help": "Maximum size of inter-cluster communication queue."},
    )

    # Cluster event logging configuration
    enable_cluster_event_logging: bool = field(
        default=False,
        metadata={
            "help": "Enable detailed event logging for each cluster. Default: False."
        },
    )
    cluster_event_log_dir: str = field(
        default="logs/cluster_events",
        metadata={
            "help": "Directory path for cluster event log files. Default: logs/cluster_events."
        },
    )
    cluster_event_log_level: str = field(
        default="INFO",
        metadata={
            "help": "Log level for cluster event logging. Options: DEBUG, INFO, WARNING, ERROR. Default: INFO.",
            "choices": ["DEBUG", "INFO", "WARNING", "ERROR"],
        },
    )

    # Performance profiling configuration
    enable_performance_profiling: bool = field(
        default=False,
        metadata={
            "help": "Enable performance profiling to identify bottlenecks. Default: False."
        },
    )
    performance_profiling_output_file: str = field(
        default="performance_profile.json",
        metadata={
            "help": "Output file for performance profiling results. Default: performance_profile.json."
        },
    )

    # Cluster log filtering configuration
    cluster_log_filter: Optional[str] = field(
        default=None,
        metadata={
            "help": "Filter logs by cluster type(s). Options: 'PREFILL', 'DECODE', 'DECODE_ATTN', 'DECODE_FFN', "
            "'PREFILL,DECODE', 'PREFILL,DECODE_ATTN,DECODE_FFN', etc. If not specified, suppresses all cluster-level logs. Default: None (suppress).",
        },
    )
    enable_cluster_log_prefix: bool = field(
        default=True,
        metadata={
            "help": "Add cluster type prefix to log entries (e.g., '[PREFILL] INFO 19:11:39'). Default: True.",
        },
    )
    use_short_timestamp: bool = field(
        default=True,
        metadata={
            "help": "Use short timestamp format (HH:MM:SS) instead of full date format. Default: True.",
        },
    )
    enable_sequential_checkpoint_observer: bool = field(
        default=False,
        metadata={
            "help": "Enable a sequential-mode checkpoint observer that exports a raw "
            "snapshot and terminates once the expected survivor set is reached.",
        },
    )
    sequential_checkpoint_expected_survivor_count: int = field(
        default=0,
        metadata={
            "help": "Expected number of unfinished requests required before the "
            "sequential checkpoint observer exports a snapshot.",
        },
    )
    sequential_checkpoint_expected_session_ids_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a JSON or text file containing the exact survivor "
            "session_id set expected by the sequential checkpoint observer.",
        },
    )
    sequential_checkpoint_raw_snapshot_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path where the sequential checkpoint observer writes the raw "
            "JSONL request snapshot.",
        },
    )

    def __post_init__(self):
        self._validate_open_source_release_architecture_guard()
        self._validate_pdaf_trace_replay_contract()
        self.performance_profiling_output_file = validate_output_filename(
            self.performance_profiling_output_file,
            "performance_profiling_output_file",
        )

        # Set global variables from configuration
        from frontier.config import global_vars

        if self.op_quantization_config_file is not None:
            raise NotImplementedError(
                "Operation-level quantization config is deprecated. "
                "Use model config (torch_dtype + quantization_config) only."
            )

        self._validate_simulation_mode_arch_compatibility()
        self._validate_thinking_mode_config()
        self._validate_sequential_checkpoint_observer_config()
        global_vars.set_global_vars(self.simulation_mode, self.sys_arch)
        self._validate_cuda_graph_config()
        self._validate_gdn_runtime_guards()
        global_vars.set_cuda_graph_config(
            self.use_cuda_graph,
            self.cudagraph_capture_sizes,
            self.decode_cuda_graph_mode,
            self.allow_spec_decode_cuda_graph_diagnostic,
        )

        # Initialize global IS_MOE flag from model configuration
        # This must be done early, before any code checks is_moe
        # The IS_MOE flag is determined SOLELY by model architecture (model_config.is_moe),
        # NOT by parallelism settings like moe_expert_parallel_size
        model_config_for_is_moe = self._get_model_config_for_is_moe()
        if model_config_for_is_moe is not None:
            global_vars.set_is_moe(model_config_for_is_moe.is_moe)

        # PD+AF disaggregation: dense models are allowed (via DenseFFNBatchGroup).
        # No MoE requirement enforced.

        self._maybe_set_default_decode_attn_request_allocation_threshold()

        # Configure logging level and cluster-aware logging
        from frontier.logger import set_log_level, configure_cluster_logging

        set_log_level(self.log_level)
        configure_cluster_logging(
            cluster_filter=self.cluster_log_filter,
            enable_cluster_prefix=self.enable_cluster_log_prefix,
            use_short_timestamp=self.use_short_timestamp,
            cluster_event_log_level=self.cluster_event_log_level,
        )

        # Print cluster statistics after configuration is complete
        self.cluster_config.print_cluster_statistics(
            self.simulation_mode, self.sys_arch
        )
        self._normalize_metrics_output_dir()
        self.write_config_to_file()

    def _validate_gdn_runtime_guards(self) -> None:
        """Reject unsupported GDN features after all cluster replicas exist."""

        from frontier.attention.gdn.guards import validate_gdn_runtime_support

        prefix_cache_enabled = bool(
            getattr(
                self.cluster_config.replica_scheduler_config,
                "enable_prefix_caching",
                False,
            )
        )
        pd_enabled = self.sys_arch in {"pd-disaggregation", "pd-af-disaggregation"}
        replica_configs = []
        for attr_name in (
            "replica_config",
            "prefill_replica_config",
            "decode_replica_config",
            "decode_attn_replica_config",
            "decode_ffn_replica_config",
        ):
            replica_config = getattr(self.cluster_config, attr_name, None)
            if replica_config is not None and all(
                replica_config is not existing for existing in replica_configs
            ):
                replica_configs.append(replica_config)
        for replica_config in replica_configs:
            validate_gdn_runtime_support(
                replica_config.model_config,
                prefix_cache_enabled=prefix_cache_enabled,
                pd_enabled=pd_enabled,
                speculative_enabled=bool(
                    replica_config.speculative_decoding_config.enabled
                ),
                num_pipeline_stages=replica_config.num_pipeline_stages,
                moe_expert_parallel_size=replica_config.moe_expert_parallel_size,
                attn_dp=replica_config.attn_dp,
            )

    def _validate_pdaf_trace_replay_contract(self) -> None:
        """Reject deferred StepFun trace replay inputs at the PD-AF boundary."""
        if self.sys_arch != "pd-af-disaggregation":
            return

        replica_configs = []
        for attr_name in (
            "replica_config",
            "prefill_replica_config",
            "decode_replica_config",
            "decode_attn_replica_config",
            "decode_ffn_replica_config",
        ):
            replica_config = getattr(self.cluster_config, attr_name, None)
            if replica_config is not None and all(
                replica_config is not existing for existing in replica_configs
            ):
                replica_configs.append(replica_config)

        configured_fields = set()
        trace_driven = False
        deferred_trace_fields = (
            "moe_routing_trace_path",
            "decode_attn_initial_lane_trace_path",
            "decode_attn_steady_state_snapshot_path",
            "decode_attn_steady_state_measurement_report_path",
        )
        for replica_config in replica_configs:
            for field_name in deferred_trace_fields:
                value = getattr(replica_config, field_name, "")
                if value is not None and str(value).strip():
                    configured_fields.add(field_name)
            if str(getattr(replica_config, "moe_routing_trace_path", "")).strip():
                trace_driven = True

        if trace_driven:
            configured_fields.add("moe_routing_trace_path")
        if not configured_fields:
            return

        fields_text = ", ".join(sorted(configured_fields))
        raise ValueError(
            f"{PD_AF_TRACE_REPLAY_DEFERRED_ERROR} Configured fields: {fields_text}."
        )

    def _maybe_set_default_decode_attn_request_allocation_threshold(self) -> None:
        """Keep offline PD+AF decode-attn staged admission opt-in only.

        The immutable Reference runtime leaves this threshold unset unless the
        PairRunSpec explicitly requests staged admission. Applying an implicit
        request-count threshold changes decode-attn wave formation and therefore
        invalidates numerical parity even when the CLI arguments match.
        """
        if (
            self.simulation_mode == "offline"
            and self.sys_arch == "pd-af-disaggregation"
            and self.cluster_config.decode_attn_request_allocation_threshold is None
        ):
            logger.info(
                "Leaving decode_attn_request_allocation_threshold unset in offline "
                "pd-af-disaggregation; staged decode-attn admission is disabled by "
                "default. Set the threshold explicitly to re-enable the legacy "
                "wave-based barrier."
            )

    def _validate_open_source_release_architecture_guard(self) -> None:
        if (
            self.sys_arch == "pd-af-disaggregation"
            and bool(
                getattr(
                    self.cluster_config.replica_scheduler_config,
                    "enable_prefix_caching",
                    False,
                )
            )
        ):
            raise ValueError(PD_AF_PREFIX_CACHING_RELEASE_ERROR)
        if self.sys_arch == "pd-af-disaggregation" and self.enable_parallel_clusters:
            raise ValueError(PD_AF_DISAGGREGATION_PARALLEL_CLUSTER_RELEASE_ERROR)
        if self.sys_arch == "pd-disaggregation" and self.enable_parallel_clusters:
            raise ValueError(PD_DISAGGREGATION_PARALLEL_CLUSTER_RELEASE_ERROR)

    def _validate_simulation_mode_arch_compatibility(self) -> None:
        """
        Validate supported simulation-mode/system-architecture combinations.
        """
        pass

    def _validate_thinking_mode_config(self) -> None:
        if self.thinking_depth < 1:
            raise ValueError(
                f"thinking_depth must be >= 1, got {self.thinking_depth}"
            )
        if self.tool_call_latency < 0:
            raise ValueError(
                f"tool_call_latency must be >= 0, got {self.tool_call_latency}"
            )

        has_explicit_hidden_rounds = (
            self.thinking_round_prefill_tokens is not None
            or self.thinking_round_decode_tokens is not None
        )

        if not self.enable_thinking_mode:
            if self.thinking_depth > 1 or has_explicit_hidden_rounds:
                raise ValueError(
                    "Multi-round thinking requests require enable_thinking_mode=True."
                )
            return

        if self.sys_arch == "pd-af-disaggregation":
            raise ValueError(
                "Thinking Mode v1 supports only 'co-location' and "
                "'pd-disaggregation'."
            )

        if has_explicit_hidden_rounds:
            if (
                self.thinking_round_prefill_tokens is None
                or self.thinking_round_decode_tokens is None
            ):
                raise ValueError(
                    "thinking_round_prefill_tokens and thinking_round_decode_tokens "
                    "must be provided together."
                )
            expected_hidden_rounds = self.thinking_depth - 1
            if len(self.thinking_round_prefill_tokens) != expected_hidden_rounds:
                raise ValueError(
                    "thinking_round_prefill_tokens length must equal "
                    f"thinking_depth - 1 ({expected_hidden_rounds})."
                )
            if len(self.thinking_round_decode_tokens) != expected_hidden_rounds:
                raise ValueError(
                    "thinking_round_decode_tokens length must equal "
                    f"thinking_depth - 1 ({expected_hidden_rounds})."
                )

    def _validate_cuda_graph_config(self) -> None:
        valid_decode_cuda_graph_modes = {"none", "full_decode_only", "piecewise"}
        if self.decode_cuda_graph_mode not in valid_decode_cuda_graph_modes:
            raise ValueError(
                "decode_cuda_graph_mode must be one of "
                f"{sorted(valid_decode_cuda_graph_modes)}, got={self.decode_cuda_graph_mode!r}"
            )

        speculative_replica_configs = [
            getattr(self.cluster_config, "replica_config", None),
            getattr(self.cluster_config, "prefill_replica_config", None),
            getattr(self.cluster_config, "decode_replica_config", None),
            getattr(self.cluster_config, "decode_attn_replica_config", None),
            getattr(self.cluster_config, "decode_ffn_replica_config", None),
        ]
        spec_decode_enabled = any(
            bool(
                getattr(
                    getattr(replica_config, "speculative_decoding_config", None),
                    "enabled",
                    False,
                )
            )
            for replica_config in speculative_replica_configs
            if replica_config is not None
        )
        if (
            spec_decode_enabled
            and self.decode_cuda_graph_mode != "none"
            and not self.allow_spec_decode_cuda_graph_diagnostic
        ):
            raise ValueError(
                "Speculative decoding currently requires decode_cuda_graph_mode='none'. "
                "Frontier MTP/speculative CUDA graph support is deferred as future work."
            )

        if self.use_cuda_graph and self.decode_cuda_graph_mode != "none":
            raise ValueError(
                "decode_cuda_graph_mode cannot be combined with use_cuda_graph=True. "
                "Use pd-af use_cuda_graph for AFD CUDA Graph simulation, or "
                "decode_cuda_graph_mode for VLLM V1 co-location / PD decode modeling."
            )

        if self.use_cuda_graph and self.sys_arch != "pd-af-disaggregation":
            raise ValueError(
                "CUDA Graph simulation is only supported in 'pd-af-disaggregation' mode. "
                f"Got sys_arch='{self.sys_arch}' with use_cuda_graph=True."
            )

        if (
            self.decode_cuda_graph_mode != "none"
            and self.sys_arch not in {"co-location", "pd-disaggregation"}
        ):
            raise ValueError(
                "decode_cuda_graph_mode is supported only in 'co-location' and "
                "'pd-disaggregation' modes. "
                f"Got sys_arch='{self.sys_arch}' with "
                f"decode_cuda_graph_mode={self.decode_cuda_graph_mode!r}."
            )

        if getattr(self.cluster_config, "decode_attn_use_cuda_graph", False):
            raise ValueError(
                "decode_attn_use_cuda_graph is deprecated. Use the global "
                "'use_cuda_graph' setting in SimulationConfig instead."
            )

        if (
            getattr(self.cluster_config, "decode_attn_cudagraph_capture_sizes", None)
            is not None
        ):
            raise ValueError(
                "decode_attn_cudagraph_capture_sizes is deprecated. Use the global "
                "'cudagraph_capture_sizes' setting in SimulationConfig instead."
            )

        if (
            self.cudagraph_capture_sizes is not None
            and not self.use_cuda_graph
            and self.decode_cuda_graph_mode == "none"
        ):
            raise ValueError(
                "cudagraph_capture_sizes requires either use_cuda_graph=True or "
                "decode_cuda_graph_mode != 'none'."
            )

    def _validate_sequential_checkpoint_observer_config(self) -> None:
        if not self.enable_sequential_checkpoint_observer:
            return

        if self.enable_parallel_clusters and self.is_disaggregated_mode():
            raise ValueError(
                "Sequential checkpoint observer cannot run with parallel cluster mode enabled."
            )
        if self.sequential_checkpoint_expected_survivor_count <= 0:
            raise ValueError(
                "sequential_checkpoint_expected_survivor_count must be > 0 when "
                "the sequential checkpoint observer is enabled."
            )
        if not self.sequential_checkpoint_expected_session_ids_file:
            raise ValueError(
                "sequential_checkpoint_expected_session_ids_file is required when "
                "the sequential checkpoint observer is enabled."
            )
        if not self.sequential_checkpoint_raw_snapshot_path:
            raise ValueError(
                "sequential_checkpoint_raw_snapshot_path is required when the "
                "sequential checkpoint observer is enabled."
            )

    def get_clusters(self) -> Dict[ClusterType, ClusterConfig]:
        """Get all cluster configurations."""
        if self.sys_arch in ["pd-disaggregation", "pd-af-disaggregation"]:
            return self.cluster_config.get_cluster_configs_for_disaggregation()
        else:
            return {ClusterType.MONOLITHIC: self.cluster_config}

    def is_disaggregated_mode(self) -> bool:
        """Check if this is disaggregated mode (either PD or PD+AF)."""
        return self.sys_arch in ["pd-disaggregation", "pd-af-disaggregation"]

    def _get_model_config_for_is_moe(self):
        """
        Get the model configuration to determine IS_MOE flag.

        This method searches through all possible replica configurations to find
        a valid model_config. The IS_MOE flag is determined by model architecture
        (model_config.is_moe), NOT by parallelism settings.

        Returns:
            BaseModelConfig or None: The model configuration if found, None otherwise.
        """
        # Try cluster-specific replica configs first (for disaggregated modes)
        for attr_name in [
            "prefill_replica_config",
            "decode_replica_config",
            "decode_attn_replica_config",
            "decode_ffn_replica_config",
            "replica_config",  # For monolithic mode
        ]:
            rc = getattr(self.cluster_config, attr_name, None)
            if (
                rc is not None
                and hasattr(rc, "model_config")
                and rc.model_config is not None
            ):
                return rc.model_config

        return None

    def _normalize_metrics_output_dir(self) -> None:
        model_config = self._get_model_config_for_is_moe()
        if model_config is None:
            raise ValueError("metrics output taxonomy requires a model_config")
        workload_type = (
            "offline_batch"
            if self.simulation_mode == "offline"
            else "online_serving"
        )
        self.metrics_config.output_dir = build_metrics_run_output_dir(
            output_root=self.metrics_config.output_dir,
            model_type=model_config.get_name(),
            workload_type=workload_type,
            run_id=self.metrics_config.run_id,
        )
        os.makedirs(self.metrics_config.output_dir, exist_ok=True)

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)
        with open(f"{self.metrics_config.output_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
