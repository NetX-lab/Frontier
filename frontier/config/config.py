from __future__ import annotations

from abc import ABC
from dataclasses import MISSING, dataclass, field, replace
from datetime import datetime
import json
import os
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

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

# NOTE: CC backend configs are imported lazily via _get_cc_backend_configs() function
# to avoid circular imports. Do NOT add direct imports here.
# The lazy import is used in:
# - cc_backend_config field default_factory
# - _create_cc_backend_config_for_cluster()
# - _create_analytical_cc_backend_config()
# - _create_vidur_cc_backend_config()
# - _create_collective_sim_cc_backend_config()
# - _create_aiconfigurator_cc_backend_config()
# - _create_astra_sim_analytical_cc_backend_config()

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

logger = init_logger(__name__)


DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR = (
    "Error: Disaggregated architecture support is currently being optimized and is not included in this release. "
    "It will be available in an upcoming version. Please use the co-located architecture for current usage and testing."
)

AICONFIGURATOR_BACKEND_RELEASE_ERROR = (
    "Error: The aiconfigurator communication backend is not included in this release. "
    "Please use collective_sim, astra_sim_analytical, analytical, or vidur for current usage and testing."
)

DISAGGREGATED_CLUSTER_FIELD_PREFIXES = (
    "prefill_",
    "decode_",
    "decode_attn_",
    "decode_ffn_",
)

DISAGGREGATED_CLUSTER_FIELD_NAMES = frozenset(
    {
        "af_pipeline_num_micro_batch",
    }
)


# Lazy import helper for cc_backend_config to avoid circular imports
def _get_cc_backend_configs():
    """Lazily import CC backend config classes to avoid circular imports."""
    from frontier.cc_backend.cc_backend_config import (
        BaseCCBackendConfig,
        VidurCCBackendConfig,
        AnalyticalCCBackendConfig,
        CollectiveSimCCBackendConfig,
        AiconfiguratorCCBackendConfig,
        AstraSimAnalyticalCCBackendConfig,
    )

    return (
        BaseCCBackendConfig,
        VidurCCBackendConfig,
        AnalyticalCCBackendConfig,
        CollectiveSimCCBackendConfig,
        AiconfiguratorCCBackendConfig,
        AstraSimAnalyticalCCBackendConfig,
    )


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens."},
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace request interval generator file."},
    )
    start_time: str = field(
        default="1970-01-04 12:00:00",
        metadata={"help": "Start time of the trace request interval generator."},
    )
    end_time: str = field(
        default="1970-01-04 15:00:00",
        metadata={"help": "End time of the trace request interval generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={
            "help": "Time scale factor for the trace request interval generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.5,
        metadata={"help": "Queries per second for Poisson Request Interval Generator."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.2,
        metadata={"help": "Queries per second for Gamma Request Interval Generator."},
    )
    cv: float = field(
        default=0.5,
        metadata={
            "help": "Coefficient of variation for Gamma Request Interval Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace request length generator file."},
    )
    prefill_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Prefill scale factor for the trace request length generator."
        },
    )
    decode_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Decode scale factor for the trace request length generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6,
        metadata={"help": "Theta for Zipf Request Length Generator."},
    )
    scramble: bool = field(
        default=False,
        metadata={"help": "Scramble for Zipf Request Length Generator."},
    )
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Zipf Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={"help": "Prefill to decode ratio for Zipf Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Uniform Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={
            "help": "Prefill to decode ratio for Uniform Request Length Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=2048,
        metadata={"help": "Prefill tokens for Fixed Request Length Generator."},
    )
    decode_tokens: int = field(
        default=512,
        metadata={"help": "Decode tokens for Fixed Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED

    def __post_init__(self):
        if self.decode_tokens < 1:
            raise ValueError(f"decode_tokens must be >= 1, got {self.decode_tokens}")
        if self.prefill_tokens < 2:
            raise ValueError(f"prefill_tokens must be >1, got {self.prefill_tokens}")


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig,
        metadata={"help": "Length generator config for Synthetic Request Generator."},
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig,
        metadata={"help": "Interval generator config for Synthetic Request Generator."},
    )
    num_requests: Optional[int] = field(
        default=128,
        metadata={"help": "Number of requests for Synthetic Request Generator."},
    )
    duration: Optional[float] = field(
        default=None,
        metadata={"help": "Duration of the synthetic request generator."},
    )
    default_priority: int = field(
        default=0,
        metadata={
            "help": "Default priority for all generated requests. "
            "Lower value = higher priority (0 = highest). "
            "Matches vLLM v1 semantics."
        },
    )

    def __post_init__(self):
        self.max_tokens = self.length_generator_config.max_tokens

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/splitwise_conv.csv",
        metadata={"help": "Path to the trace request generator file."},
    )
    prefill_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Prefill scale factor for the trace request generator."},
    )
    decode_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Decode scale factor for the trace request generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Time scale factor for the trace request generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens for the trace request generator."},
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE_REPLAY


@dataclass
class BaseReplicaSchedulerConfig(BasePolyConfig):
    batch_size_cap: int = field(
        default=128,
        metadata={"help": "Maximum batch size cap (max_num_seqs in vLLM)"},
    )
    block_size: int = field(
        default=16,
        metadata={"help": "Block size."},
    )
    watermark_blocks_fraction: float = field(
        default=0.01,
        metadata={"help": "Watermark blocks fraction."},
    )
    num_blocks: Optional[int] = field(
        default=106596,
        metadata={"help": "Number of blocks."},
    )


@dataclass
class VllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens (max_num_batched_tokens) in batch for vLLM."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.VLLM


@dataclass
class LightllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens in batch for LightLLM."},
    )
    max_waiting_iters: int = field(
        default=10,
        metadata={"help": "Maximum waiting iterations for LightLLM."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.LIGHTLLM


@dataclass
class OrcaSchedulerConfig(BaseReplicaSchedulerConfig):
    @staticmethod
    def get_type():
        return ReplicaSchedulerType.ORCA


@dataclass
class FasterTransformerSchedulerConfig(BaseReplicaSchedulerConfig):
    @staticmethod
    def get_type():
        return ReplicaSchedulerType.FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for Sarathi."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SARATHI


@dataclass
class VllmV1SchedulerConfig(BaseReplicaSchedulerConfig):
    """
    Configuration for the vLLM v1 engine replica scheduler.

    This scheduler simulates the admission control behavior of vLLM v1 engine,
    including two-phase scheduling, token budget management, and preemption.

    Note: Class name uses 'VllmV1' (not 'VLLMv1') to generate clean CLI parameter
    names: --vllm_v1_scheduler_config_* instead of --v_l_l_mv1_scheduler_config_*
    """

    max_tokens_in_batch: int = field(
        default=16384,
        metadata={
            "help": "Maximum tokens per scheduling iteration (max_num_batched_tokens in vLLM v1)."
        },
    )
    scheduling_policy: str = field(
        default="fcfs",
        metadata={
            "help": "Scheduling policy: 'fcfs' (First-Come-First-Served) or 'priority'."
        },
    )
    enable_preemption: bool = field(
        default=True,
        metadata={
            "help": "Enable preemption when memory is insufficient for running requests."
        },
    )
    enable_chunked_prefill: bool = field(
        default=False,
        metadata={
            "help": "Enable chunked prefill admission when waiting prefill requests exceed current token budget."
        },
    )
    enable_phase_aware_thinking_profile: bool = field(
        default=False,
        metadata={
            "help": "Enable an iteration-scoped hidden-round/final-round scheduler profile override for Thinking Mode home queues."
        },
    )
    hidden_phase_max_tokens_in_batch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional hidden-round override for max_tokens_in_batch when enable_phase_aware_thinking_profile=True."
        },
    )
    hidden_phase_enable_chunked_prefill: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Optional hidden-round override for enable_chunked_prefill when enable_phase_aware_thinking_profile=True."
        },
    )
    hidden_phase_batch_size_cap: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional hidden-round override for batch_size_cap when enable_phase_aware_thinking_profile=True."
        },
    )
    final_phase_max_tokens_in_batch: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional final-round override for max_tokens_in_batch when enable_phase_aware_thinking_profile=True."
        },
    )
    final_phase_enable_chunked_prefill: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Optional final-round override for enable_chunked_prefill when enable_phase_aware_thinking_profile=True."
        },
    )
    final_phase_batch_size_cap: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional final-round override for batch_size_cap when enable_phase_aware_thinking_profile=True."
        },
    )
    final_prefill_reserved_slots: int = field(
        default=0,
        metadata={
            "help": "Per-iteration PREFILL admission slots reserved for final-round prefill requests. Hidden requests may borrow idle reserved slots."
        },
    )
    final_prefill_reserved_tokens: int = field(
        default=0,
        metadata={
            "help": "Per-iteration PREFILL token budget reserved for final-round prefill requests. Hidden requests may borrow idle reserved tokens."
        },
    )
    final_decode_reserved_slots: int = field(
        default=0,
        metadata={
            "help": "Per-iteration DECODE running/admission slots reserved for final-round decode requests. Hidden requests may borrow idle reserved slots."
        },
    )
    enable_final_running_request_reclaim: bool = field(
        default=False,
        metadata={
            "help": "When final backlog appears, reclaim hidden requests that have borrowed final reserved running slots so the final slice becomes active running capacity."
        },
    )
    enable_final_round_priority_boost: bool = field(
        default=False,
        metadata={
            "help": "Promote re-entered final-round Thinking Mode requests into a higher-priority band under priority scheduling."
        },
    )
    final_round_priority_value: int = field(
        default=-1,
        metadata={
            "help": "Priority value assigned to promoted final-round requests. Lower values mean higher priority."
        },
    )
    enable_prefix_caching: bool = field(
        default=False,
        metadata={
            "help": "Enable block-hash-based prefix matching and KV cache reuse."
        },
    )
    prefix_caching_hash_algo: str = field(
        default="builtin",
        metadata={
            "help": "Hash algorithm label for explicit prefix block hashes. Supported: 'builtin', 'sha256'."
        },
    )
    num_preallocate_tokens: int = field(
        default=0,
        metadata={
            "help": "Number of tokens worth of KV cache blocks to preallocate for each request."
        },
    )
    long_prefill_token_threshold: int = field(
        default=0,
        metadata={
            "help": "Optional upper bound on per-iteration prefill tokens for each request. 0 disables threshold."
        },
    )
    num_blocks: Optional[int] = field(
        default=0,
        metadata={
            "help": "Number of KV cache blocks. Use 0 to auto-derive from the memory planner in planner modes."
        },
    )
    num_blocks_mode: str = field(
        default="memory_planner_profiled",
        metadata={
            "help": "How to initialize num_blocks: 'memory_planner' (auto-derive with parameter-only estimate), 'memory_planner_profiled' (auto-derive with calibrated non-KV overhead), or 'explicit' (require num_blocks>0)."
        },
    )
    gpu_memory_utilization: Optional[float] = field(
        default=None,
        metadata={
            "help": "vLLM-style GPU memory utilization ratio used by memory_planner mode. If unset, fallback to 1 - replica memory_margin_fraction."
        },
    )
    non_kv_cache_overhead_bytes: int = field(
        default=0,
        metadata={
            "help": "Calibrated non-KV memory overhead in bytes for memory_planner_profiled mode."
        },
    )
    runtime_weights_memory_source: str = field(
        default="param_counter",
        metadata={
            "help": "Weights memory source for runtime non-KV profiling: 'param_counter' (estimated bytes) or 'runtime_model_load' (measure loaded model parameter bytes)."
        },
    )
    enable_runtime_non_kv_cache_overhead_profiling: bool = field(
        default=False,
        metadata={
            "help": "Enable runtime single-rank profiling to auto-estimate non_kv_cache_overhead_bytes during scheduler initialization. Requires num_blocks_mode=memory_planner_profiled."
        },
    )
    nccl_buffer_comm_base_overhead_bytes: int = field(
        default=100 * 1024 * 1024,
        metadata={
            "help": "Per-communicator fixed NCCL overhead in bytes (proxy, queues). "
                    "Default 100 MiB, calibrated for A800."
        },
    )
    nccl_buffer_per_peer_overhead_bytes: int = field(
        default=15 * 1024 * 1024,
        metadata={
            "help": "Per-peer NCCL transport buffer overhead in bytes. "
                    "Default 15 MiB, calibrated for A800 intra-node."
        },
    )
    nccl_buffer_custom_ar_enabled: bool = field(
        default=False,
        metadata={
            "help": "Enable CustomAllreduce buffer estimation. "
                    "False for A800 (compute 8.0), True for H100 (9.0+)."
        },
    )
    nccl_buffer_vllm_worker_base_extra_bytes: int = field(
        default=0,
        metadata={
            "help": "Domain-aware vLLM worker-process non-torch addend in bytes "
                    "for runtime non-KV profiling. Default 0; pass validated "
                    "case-local values explicitly."
        },
    )
    nccl_buffer_pp_final_stage_extra_bytes: int = field(
        default=0,
        metadata={
            "help": "Additional final pipeline-stage vLLM worker non-torch addend "
                    "in bytes for runtime non-KV profiling. Default 0."
        },
    )
    nccl_buffer_dp_communicator_extra_bytes: int = field(
        default=0,
        metadata={
            "help": "Additional data-parallel communicator non-torch addend in "
                    "bytes for runtime non-KV profiling. Default 0."
        },
    )
    nccl_buffer_ep_all2all_extra_bytes: int = field(
        default=0,
        metadata={
            "help": "Additional MoE expert-parallel all-to-all non-torch addend "
                    "in bytes for runtime non-KV profiling. Default 0."
        },
    )
    use_analytical_param_memory: bool = field(
        default=False,
        metadata={
            "help": "When runtime non-KV profiling is enabled in memory_planner_profiled mode, keep analytical ParamCounter param memory for planner calculation. Default False uses runtime-profiled param memory."
        },
    )

    def __post_init__(self) -> None:
        allowed_modes = {"memory_planner", "memory_planner_profiled", "explicit"}
        if self.num_blocks_mode not in allowed_modes:
            raise ValueError(
                "VllmV1SchedulerConfig.num_blocks_mode must be one of "
                f"{sorted(allowed_modes)}, got={self.num_blocks_mode!r}"
            )

        if self.gpu_memory_utilization is not None:
            if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1.0:
                raise ValueError(
                    "VllmV1SchedulerConfig.gpu_memory_utilization must be in (0, 1], got="
                    f"{self.gpu_memory_utilization!r}"
                )

        if self.non_kv_cache_overhead_bytes < 0:
            raise ValueError(
                "VllmV1SchedulerConfig.non_kv_cache_overhead_bytes must be >= 0, got="
                f"{self.non_kv_cache_overhead_bytes!r}"
            )

        allowed_hash_algorithms = {"builtin", "sha256"}
        if self.prefix_caching_hash_algo not in allowed_hash_algorithms:
            raise ValueError(
                "VllmV1SchedulerConfig.prefix_caching_hash_algo must be one of "
                f"{sorted(allowed_hash_algorithms)}, got={self.prefix_caching_hash_algo!r}"
            )

        if self.num_preallocate_tokens < 0:
            raise ValueError(
                "VllmV1SchedulerConfig.num_preallocate_tokens must be >= 0, got="
                f"{self.num_preallocate_tokens!r}"
            )

        if self.long_prefill_token_threshold < 0:
            raise ValueError(
                "VllmV1SchedulerConfig.long_prefill_token_threshold must be >= 0, got="
                f"{self.long_prefill_token_threshold!r}"
            )
        if (
            self.long_prefill_token_threshold > 0
            and not self.enable_chunked_prefill
        ):
            raise ValueError(
                "VllmV1SchedulerConfig.long_prefill_token_threshold > 0 "
                "requires enable_chunked_prefill=True"
            )

        phase_override_values = (
            self.hidden_phase_max_tokens_in_batch,
            self.hidden_phase_enable_chunked_prefill,
            self.hidden_phase_batch_size_cap,
            self.final_phase_max_tokens_in_batch,
            self.final_phase_enable_chunked_prefill,
            self.final_phase_batch_size_cap,
        )
        if not self.enable_phase_aware_thinking_profile and any(
            value is not None for value in phase_override_values
        ):
            raise ValueError(
                "VllmV1SchedulerConfig phase-aware override fields require "
                "enable_phase_aware_thinking_profile=True"
            )
        if self.enable_phase_aware_thinking_profile and all(
            value is None for value in phase_override_values
        ):
            raise ValueError(
                "VllmV1SchedulerConfig.enable_phase_aware_thinking_profile=True "
                "requires at least one hidden/final override field"
            )

        for field_name in (
            "hidden_phase_max_tokens_in_batch",
            "hidden_phase_batch_size_cap",
            "final_phase_max_tokens_in_batch",
            "final_phase_batch_size_cap",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None and field_value <= 0:
                raise ValueError(
                    f"VllmV1SchedulerConfig.{field_name} must be > 0 when set, "
                    f"got={field_value!r}"
                )

        for field_name in (
            "final_prefill_reserved_slots",
            "final_prefill_reserved_tokens",
            "final_decode_reserved_slots",
        ):
            field_value = getattr(self, field_name)
            if field_value < 0:
                raise ValueError(
                    f"VllmV1SchedulerConfig.{field_name} must be >= 0, "
                    f"got={field_value!r}"
                )

        if (
            self.long_prefill_token_threshold > 0
            and self.enable_phase_aware_thinking_profile
        ):
            if self.hidden_phase_enable_chunked_prefill is False:
                raise ValueError(
                    "VllmV1SchedulerConfig.hidden_phase_enable_chunked_prefill=False "
                    "is incompatible with long_prefill_token_threshold > 0"
                )
            if self.final_phase_enable_chunked_prefill is False:
                raise ValueError(
                    "VllmV1SchedulerConfig.final_phase_enable_chunked_prefill=False "
                    "is incompatible with long_prefill_token_threshold > 0"
                )

        if self.nccl_buffer_comm_base_overhead_bytes < 0:
            raise ValueError(
                "VllmV1SchedulerConfig.nccl_buffer_comm_base_overhead_bytes must be >= 0, got="
                f"{self.nccl_buffer_comm_base_overhead_bytes!r}"
            )

        if self.nccl_buffer_per_peer_overhead_bytes < 0:
            raise ValueError(
                "VllmV1SchedulerConfig.nccl_buffer_per_peer_overhead_bytes must be >= 0, got="
                f"{self.nccl_buffer_per_peer_overhead_bytes!r}"
            )

        for field_name in (
            "nccl_buffer_vllm_worker_base_extra_bytes",
            "nccl_buffer_pp_final_stage_extra_bytes",
            "nccl_buffer_dp_communicator_extra_bytes",
            "nccl_buffer_ep_all2all_extra_bytes",
        ):
            field_value = getattr(self, field_name)
            if field_value < 0:
                raise ValueError(
                    f"VllmV1SchedulerConfig.{field_name} must be >= 0, "
                    f"got={field_value!r}"
                )

        allowed_weights_sources = {"param_counter", "runtime_model_load"}
        if self.runtime_weights_memory_source not in allowed_weights_sources:
            raise ValueError(
                "VllmV1SchedulerConfig.runtime_weights_memory_source must be one of "
                f"{sorted(allowed_weights_sources)}, got={self.runtime_weights_memory_source!r}"
            )

        if (
            self.enable_runtime_non_kv_cache_overhead_profiling
            and self.num_blocks_mode != "memory_planner_profiled"
        ):
            raise ValueError(
                "VllmV1SchedulerConfig.enable_runtime_non_kv_cache_overhead_profiling "
                "requires num_blocks_mode=memory_planner_profiled, got="
                f"{self.num_blocks_mode!r}"
            )

        if (
            self.use_analytical_param_memory
            and not self.enable_runtime_non_kv_cache_overhead_profiling
        ):
            raise ValueError(
                "VllmV1SchedulerConfig.use_analytical_param_memory "
                "requires enable_runtime_non_kv_cache_overhead_profiling=True"
            )

    enable_thinking_round_priority: bool = field(
        default=False,
        metadata={
            "help": "When enabled, final-round thinking requests are prioritized "
            "over non-final-round requests in the waiting queue.",
        },
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.VLLM_V1


@dataclass
class Sj2qFastserveLiteSchedulerConfig(VllmV1SchedulerConfig):
    """
    Configuration for the SJ-2Q / FastServe-lite scheduler.

    Note: Class name uses 'Sj2qFastserve' (not 'Sj2QFastServe') to generate clean
    CLI parameter names: --sj2q_fastserve_lite_scheduler_config_* instead of
    --sj2_q_fast_serve_lite_scheduler_config_*.
    """

    long_round_new_prompt_threshold: int = field(
        default=2048,
        metadata={
            "help": "Rounds whose new prompt tokens exceed this threshold enter QL and mark long_history."
        },
    )
    short_round_boost_threshold: int = field(
        default=512,
        metadata={
            "help": "Tiny-prefill threshold used for QH prioritization and the prefill-release-only boost when long_history is already true."
        },
    )
    boost_credit_token_budget: int = field(
        default=2048,
        metadata={
            "help": "Deprecated compatibility field retained for CLI stability; current prefill-release-only boost demotes on prefill completion instead of token-budget exhaustion."
        },
    )
    enable_aging: bool = field(
        default=False,
        metadata={
            "help": "Enable optional aging-based QL promotion back into QH. The UC3 v2 enhancement lane keeps this disabled."
        },
    )
    aging_wait_threshold_ms: float = field(
        default=7.5,
        metadata={
            "help": "QL waiting-time threshold in milliseconds for a temporary aging-based QH boost."
        },
    )
    aging_boost_token_budget: int = field(
        default=512,
        metadata={
            "help": "Token budget granted when an aged QL session is temporarily promoted into QH."
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.enable_phase_aware_thinking_profile:
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig does not allow phase-aware oracle scheduling."
            )
        if self.enable_thinking_round_priority:
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig does not allow final-round priority override."
            )
        if (
            self.final_prefill_reserved_slots != 0
            or self.final_prefill_reserved_tokens != 0
            or self.final_decode_reserved_slots != 0
        ):
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig requires all final reserved slot/token settings to remain 0."
            )
        if self.enable_final_running_request_reclaim:
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig does not allow final running-request reclaim."
            )
        if self.enable_final_round_priority_boost:
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig does not allow final-round priority boost."
            )

        if self.long_round_new_prompt_threshold <= 0:
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig.long_round_new_prompt_threshold must be > 0."
            )
        if self.short_round_boost_threshold <= 0:
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig.short_round_boost_threshold must be > 0."
            )
        if (
            self.short_round_boost_threshold
            > self.long_round_new_prompt_threshold
        ):
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig.short_round_boost_threshold must be <= long_round_new_prompt_threshold."
            )
        if self.boost_credit_token_budget <= 0:
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig.boost_credit_token_budget must be > 0."
            )
        if self.enable_aging and self.aging_wait_threshold_ms <= 0:
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig.aging_wait_threshold_ms must be > 0 when aging is enabled."
            )
        if self.aging_boost_token_budget <= 0:
            raise ValueError(
                "Sj2QFastserveLiteSchedulerConfig.aging_boost_token_budget must be > 0."
            )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SJ2Q_FASTSERVE_LITE


Sj2QFastServeLiteSchedulerConfig = Sj2qFastserveLiteSchedulerConfig


@dataclass
class Sj2qPenaltyOnlySchedulerConfig(VllmV1SchedulerConfig):
    """
    Configuration for the penalty-only SJ-2Q scheduler.

    Note: Class name uses 'Sj2q' to generate clean CLI parameter names like
    --sj2q_penalty_only_scheduler_config_*.
    """

    long_round_new_prompt_threshold: int = field(
        default=4096,
        metadata={
            "help": "Rounds whose new prompt tokens exceed this threshold immediately enter Qlong and mark long_history."
        },
    )
    service_cap_tokens: int = field(
        default=8192,
        metadata={
            "help": "Session-level cumulative new-token service cap after which the session stays in Qlong."
        },
    )
    long_liveness_quota: int = field(
        default=32,
        metadata={
            "help": "Maximum consecutive Qshort slices allowed before forcing one Qlong slice when Qlong is non-empty."
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.enable_phase_aware_thinking_profile:
            raise ValueError(
                "Sj2qPenaltyOnlySchedulerConfig does not allow phase-aware oracle scheduling."
            )
        if self.enable_thinking_round_priority:
            raise ValueError(
                "Sj2qPenaltyOnlySchedulerConfig does not allow final-round priority override."
            )
        if (
            self.final_prefill_reserved_slots != 0
            or self.final_prefill_reserved_tokens != 0
            or self.final_decode_reserved_slots != 0
        ):
            raise ValueError(
                "Sj2qPenaltyOnlySchedulerConfig requires all final reserved slot/token settings to remain 0."
            )
        if self.enable_final_running_request_reclaim:
            raise ValueError(
                "Sj2qPenaltyOnlySchedulerConfig does not allow final running-request reclaim."
            )
        if self.enable_final_round_priority_boost:
            raise ValueError(
                "Sj2qPenaltyOnlySchedulerConfig does not allow final-round priority boost."
            )
        if self.long_round_new_prompt_threshold <= 0:
            raise ValueError(
                "Sj2qPenaltyOnlySchedulerConfig.long_round_new_prompt_threshold must be > 0."
            )
        if self.service_cap_tokens <= 0:
            raise ValueError(
                "Sj2qPenaltyOnlySchedulerConfig.service_cap_tokens must be > 0."
            )
        if self.service_cap_tokens < self.long_round_new_prompt_threshold:
            raise ValueError(
                "Sj2qPenaltyOnlySchedulerConfig.service_cap_tokens must be >= long_round_new_prompt_threshold."
            )
        if self.long_liveness_quota <= 0:
            raise ValueError(
                "Sj2qPenaltyOnlySchedulerConfig.long_liveness_quota must be > 0."
            )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SJ2Q_PENALTY_ONLY


Sj2QPenaltyOnlySchedulerConfig = Sj2qPenaltyOnlySchedulerConfig


@dataclass
class Sj2qBoundedCarryoverSchedulerConfig(Sj2qPenaltyOnlySchedulerConfig):
    """
    Configuration for the bounded-carryover SJ-2Q scheduler.

    Note: Class name uses 'Sj2q' to generate clean CLI parameter names like
    --sj2q_bounded_carryover_scheduler_config_*.
    """

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SJ2Q_BOUNDED_CARRYOVER


Sj2QBoundedCarryoverSchedulerConfig = Sj2qBoundedCarryoverSchedulerConfig


@dataclass
class SglangSchedulerConfig(VllmV1SchedulerConfig):
    """
    Thin config wrapper for the Frontier SGLang-style replica scheduler.

    This intentionally reuses the vLLM v1 scheduler fields and only changes
    the scheduler type to keep the integration surface minimal.
    """

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SGLANG


@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to write metrics."},
    )
    write_json_trace: bool = field(
        default=False,
        metadata={"help": "Whether to write json trace."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases group name."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name."},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases sweep id."},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run id."},
    )
    enable_chrome_trace: bool = field(
        default=True,
        metadata={"help": "Enable Chrome tracing."},
    )

    # Op-Level Tracing
    enable_op_level_tracing: bool = field(
        default=False,
        metadata={"help": "Enable detailed op-level tracing (output to JSONL)."},
    )
    trace_output_file: str = field(
        default="op_traces.jsonl",
        metadata={"help": "Output filename for op-level traces."},
    )
    enable_metrics_ground_truth_trace: bool = field(
        default=False,
        metadata={
            "help": "Enable explicit request-level metrics ground-truth JSONL output."
        },
    )
    metrics_ground_truth_trace_file: str = field(
        default="metrics_ground_truth.jsonl",
        metadata={"help": "Output filename for metrics ground-truth request traces."},
    )
    enable_per_layer_expansion: bool = field(
        default=False,
        metadata={
            "help": "Enable per-layer trace expansion. When enabled, traces show "
            "individual layer operations instead of aggregated spans."
        },
    )
    num_requests_to_trace_per_layer: int = field(
        default=5,
        metadata={
            "help": "Number of requests to capture with per-layer expansion. "
            "Only applies when enable_per_layer_expansion is True."
        },
    )

    save_table_to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to save table to wandb."},
    )
    store_plots: bool = field(
        default=True,
        metadata={"help": "Whether to store plots."},
    )
    enable_memory_time_series: bool = field(
        default=False,
        metadata={
            "help": "Enable memory usage time series output. "
            "Only valid when log_level is 'debug'."
        },
    )
    store_operation_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store operation metrics."},
    )
    store_token_completion_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store token completion metrics."},
    )
    store_request_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store request metrics."},
    )
    store_batch_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store batch metrics."},
    )
    store_utilization_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store utilization metrics."},
    )
    keep_individual_batch_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to keep individual batch metrics."},
    )
    store_frontier_stage_batch_ledger: bool = field(
        default=True,
        metadata={"help": "Whether to write the full Frontier stage-batch ledger."},
    )
    store_frontier_stage_batch_ledger_summary: bool = field(
        default=False,
        metadata={
            "help": "Whether to write a bounded Frontier stage-batch ledger summary."
        },
    )
    subsamples: Optional[int] = field(
        default=None,
        metadata={"help": "Subsamples."},
    )
    min_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum batch index."},
    )
    max_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum batch index."},
    )
    output_dir: str = field(
        default="outputs/metrics",
        metadata={"help": "Metrics output root directory."},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Cache directory."},
    )
    run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Metrics run id used under outputs/metrics/<model>/<workload>/<run_id>."
        },
    )

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        self.run_id = validate_run_id(self.run_id)
        self.trace_output_file = validate_output_filename(
            self.trace_output_file, "trace_output_file"
        )
        self.metrics_ground_truth_trace_file = validate_output_filename(
            self.metrics_ground_truth_trace_file, "metrics_ground_truth_trace_file"
        )
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class SpeculativeDecodingConfig:
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable speculative decoding simulation."},
    )
    method: str = field(
        default="eagle",
        metadata={
            "help": "Speculative decoding method. Must match vLLM method names."
        },
    )
    spec_model_name: str = field(
        default="",
        metadata={
            "help": "Optional draft/spec model name for methods whose proposer "
            "decoder comes from a separate draft model (for example draft-model MTP)."
        },
    )
    num_speculative_tokens: int = field(
        default=4,
        metadata={"help": "Number of draft tokens planned per speculative iteration."},
    )
    committed_tokens_per_iteration: int = field(
        default=2,
        metadata={
            "help": "Deterministic committed token count per speculative iteration "
            "(includes 1 target token + accepted drafts)."
        },
    )
    acceptance_trace_file: str = field(
        default="",
        metadata={
            "help": "Optional deterministic acceptance trace JSON file. Supported "
            "formats: list[int] or {'committed_tokens_per_iteration': list[int], "
            "'scheduled_draft_tokens_per_iteration': optional list[int], "
            "'per_request_committed_tokens_per_iteration': optional dict[str, list[int]], "
            "'per_request_scheduled_draft_tokens_per_iteration': optional dict[str, list[int]]}. "
            "When set, trace overrides committed_tokens_per_iteration and can "
            "optionally override planned draft widths per iteration."
        },
    )
    proposer_overhead_ms_by_method: Dict[str, float] = field(
        default_factory=dict,
        metadata={
            "help": "Method-aware proposer overhead in milliseconds per speculative "
            "verify request (method -> overhead_ms >= 0)."
        },
    )
    decode_draft_proposer_latency_profile_file: str = field(
        default="",
        metadata={
            "help": "Optional structured latency profile JSON for decode draft "
            "proposer overhead. Expected workload key: "
            "(method, model_name, attn_tp_size, num_speculative_tokens, "
            "spec_verify_request_count)."
        },
    )
    mtp_n_predict: int = field(
        default=0,
        metadata={
            "help": "Optional MTP capability metadata. Number of tokens predicted "
            "per MTP block. Only valid for MTP methods."
        },
    )
    mtp_num_layers: int = field(
        default=0,
        metadata={
            "help": "Optional MTP capability metadata. Number of MTP layers. "
            "Only valid for MTP methods."
        },
    )
    trace_calibration_file: str = field(
        default="",
        metadata={
            "help": "Optional calibration JSON file. Supported keys: "
            "proposer_overhead_ms_by_method and metadata."
        },
    )

    @staticmethod
    def _validate_method_float_map(
        *,
        map_name: str,
        raw_map: Optional[Dict[str, float]],
        supported_methods: set[str],
        min_value: float,
        inclusive_min: bool,
    ) -> Dict[str, float]:
        if raw_map is None:
            return {}
        if not isinstance(raw_map, dict):
            raise ValueError(
                f"SpeculativeDecodingConfig.{map_name} must be a dict, "
                f"got={type(raw_map).__name__}"
            )

        validated: Dict[str, float] = {}
        for method_name, value in raw_map.items():
            if method_name not in supported_methods:
                raise ValueError(
                    f"SpeculativeDecodingConfig.{map_name} contains unsupported method "
                    f"{method_name!r}; supported={sorted(supported_methods)}"
                )
            numeric_value = float(value)
            if inclusive_min:
                if numeric_value < min_value:
                    raise ValueError(
                        f"SpeculativeDecodingConfig.{map_name}[{method_name!r}] "
                        f"must be >= {min_value}, got={numeric_value!r}"
                    )
            elif numeric_value <= min_value:
                raise ValueError(
                    f"SpeculativeDecodingConfig.{map_name}[{method_name!r}] "
                    f"must be > {min_value}, got={numeric_value!r}"
                )
            validated[method_name] = numeric_value
        return validated

    @staticmethod
    def _load_trace_calibration_payload(
        trace_calibration_file: str,
    ) -> Dict[str, Dict[str, float]]:
        if not trace_calibration_file:
            return {}
        if not os.path.isfile(trace_calibration_file):
            raise ValueError(
                "SpeculativeDecodingConfig.trace_calibration_file does not exist: "
                f"{trace_calibration_file!r}"
            )
        try:
            with open(trace_calibration_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "SpeculativeDecodingConfig.trace_calibration_file must be valid JSON: "
                f"{trace_calibration_file!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "SpeculativeDecodingConfig.trace_calibration_file must contain a JSON "
                f"object, got={type(payload).__name__}"
            )
        return payload

    @staticmethod
    def _load_acceptance_trace_payload(
        *,
        acceptance_trace_file: str,
    ):
        if not acceptance_trace_file:
            return None
        if not os.path.isfile(acceptance_trace_file):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file does not exist: "
                f"{acceptance_trace_file!r}"
            )
        try:
            with open(acceptance_trace_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file must be valid JSON: "
                f"{acceptance_trace_file!r}"
            ) from exc

        if not isinstance(payload, (list, dict)):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file must be list or dict, "
                f"got={type(payload).__name__}"
            )
        return payload

    @staticmethod
    def _load_committed_tokens_trace(
        *,
        acceptance_trace_payload,
        max_committed_tokens: int,
    ) -> Optional[List[int]]:
        if acceptance_trace_payload is None:
            return None

        if isinstance(acceptance_trace_payload, list):
            committed_tokens_trace_raw = acceptance_trace_payload
        else:
            if "committed_tokens_per_iteration" not in acceptance_trace_payload:
                return None
            committed_tokens_trace_raw = acceptance_trace_payload[
                "committed_tokens_per_iteration"
            ]

        if not isinstance(committed_tokens_trace_raw, list):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file committed token trace "
                f"must be a list, got={type(committed_tokens_trace_raw).__name__}"
            )
        if len(committed_tokens_trace_raw) == 0:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file committed token trace "
                "must be non-empty."
            )

        committed_tokens_trace: List[int] = []
        for idx, value in enumerate(committed_tokens_trace_raw):
            committed = int(value)
            if committed < 0:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file values must be >= 0, "
                    f"got index={idx}, value={value!r}"
                )
            if committed > max_committed_tokens:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file values must be <= "
                    f"1 + num_speculative_tokens ({max_committed_tokens}), "
                    f"got index={idx}, value={value!r}"
                )
            committed_tokens_trace.append(committed)
        return committed_tokens_trace

    @staticmethod
    def _load_per_request_committed_tokens_trace(
        *,
        acceptance_trace_payload,
        max_committed_tokens: int,
    ) -> Optional[Dict[str, List[int]]]:
        if acceptance_trace_payload is None or not isinstance(
            acceptance_trace_payload, dict
        ):
            return None
        if "per_request_committed_tokens_per_iteration" not in acceptance_trace_payload:
            return None

        raw_trace_map = acceptance_trace_payload[
            "per_request_committed_tokens_per_iteration"
        ]
        if not isinstance(raw_trace_map, dict):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_committed_tokens_per_iteration must be a dict, "
                f"got={type(raw_trace_map).__name__}"
            )
        if len(raw_trace_map) == 0:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_committed_tokens_per_iteration must be non-empty."
            )

        per_request_trace: Dict[str, List[int]] = {}
        for raw_request_id, raw_trace in raw_trace_map.items():
            request_id = str(raw_request_id)
            if not request_id:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                    "trace keys must be non-empty strings."
                )
            if request_id in per_request_trace:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file contains "
                    f"duplicate request_id={request_id!r} after normalization."
                )
            if not isinstance(raw_trace, list):
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file "
                    "per-request committed token trace must be a list, "
                    f"got request_id={request_id!r}, type={type(raw_trace).__name__}"
                )
            if len(raw_trace) == 0:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                    f"committed token trace must be non-empty, request_id={request_id!r}"
                )

            validated_trace: List[int] = []
            for idx, value in enumerate(raw_trace):
                committed = int(value)
                if committed < 0:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                        "committed token values must be >= 0, "
                        f"got request_id={request_id!r}, index={idx}, value={value!r}"
                    )
                if committed > max_committed_tokens:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                        "committed token values must be <= 1 + num_speculative_tokens "
                        f"({max_committed_tokens}), got request_id={request_id!r}, "
                        f"index={idx}, value={value!r}"
                    )
                validated_trace.append(committed)
            per_request_trace[request_id] = validated_trace
        return per_request_trace

    @staticmethod
    def _load_scheduled_draft_tokens_trace(
        *,
        acceptance_trace_payload,
        max_scheduled_draft_tokens: int,
        committed_trace_length: int,
    ) -> Optional[List[int]]:
        if acceptance_trace_payload is None or not isinstance(acceptance_trace_payload, dict):
            return None
        if "scheduled_draft_tokens_per_iteration" not in acceptance_trace_payload:
            return None

        scheduled_draft_tokens_trace_raw = acceptance_trace_payload[
            "scheduled_draft_tokens_per_iteration"
        ]
        if not isinstance(scheduled_draft_tokens_trace_raw, list):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file scheduled draft token "
                f"trace must be a list, got={type(scheduled_draft_tokens_trace_raw).__name__}"
            )
        if len(scheduled_draft_tokens_trace_raw) != committed_trace_length:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "scheduled_draft_tokens_per_iteration length must match "
                "committed_tokens_per_iteration length."
            )

        scheduled_draft_tokens_trace: List[int] = []
        for idx, value in enumerate(scheduled_draft_tokens_trace_raw):
            scheduled_drafts = int(value)
            if scheduled_drafts < 0:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file scheduled draft "
                    "trace values must be >= 0, "
                    f"got index={idx}, value={value!r}"
                )
            if scheduled_drafts > max_scheduled_draft_tokens:
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file scheduled draft "
                    "trace values must be <= num_speculative_tokens "
                    f"({max_scheduled_draft_tokens}), got index={idx}, value={value!r}"
                )
            scheduled_draft_tokens_trace.append(scheduled_drafts)
        return scheduled_draft_tokens_trace

    @staticmethod
    def _load_per_request_scheduled_draft_tokens_trace(
        *,
        acceptance_trace_payload,
        max_scheduled_draft_tokens: int,
        per_request_committed_trace: Optional[Dict[str, List[int]]],
    ) -> Optional[Dict[str, List[int]]]:
        if acceptance_trace_payload is None or not isinstance(
            acceptance_trace_payload, dict
        ):
            return None
        if (
            "per_request_scheduled_draft_tokens_per_iteration"
            not in acceptance_trace_payload
        ):
            return None
        if per_request_committed_trace is None:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_scheduled_draft_tokens_per_iteration requires "
                "per_request_committed_tokens_per_iteration."
            )

        raw_trace_map = acceptance_trace_payload[
            "per_request_scheduled_draft_tokens_per_iteration"
        ]
        if not isinstance(raw_trace_map, dict):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_scheduled_draft_tokens_per_iteration must be a dict, "
                f"got={type(raw_trace_map).__name__}"
            )

        normalized_keys = {str(request_id) for request_id in raw_trace_map.keys()}
        committed_keys = set(per_request_committed_trace.keys())
        if normalized_keys != committed_keys:
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file "
                "per_request_scheduled_draft_tokens_per_iteration keys must match "
                "per_request_committed_tokens_per_iteration keys."
            )

        per_request_trace: Dict[str, List[int]] = {}
        for request_id, committed_trace in per_request_committed_trace.items():
            raw_trace = raw_trace_map[request_id]
            if not isinstance(raw_trace, list):
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                    "scheduled draft token trace must be a list, "
                    f"got request_id={request_id!r}, type={type(raw_trace).__name__}"
                )
            if len(raw_trace) != len(committed_trace):
                raise ValueError(
                    "SpeculativeDecodingConfig.acceptance_trace_file "
                    "per_request_scheduled_draft_tokens_per_iteration length must "
                    "match per_request_committed_tokens_per_iteration length, "
                    f"request_id={request_id!r}"
                )

            validated_trace: List[int] = []
            for idx, value in enumerate(raw_trace):
                scheduled_drafts = int(value)
                if scheduled_drafts < 0:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                        "scheduled draft token values must be >= 0, "
                        f"got request_id={request_id!r}, index={idx}, value={value!r}"
                    )
                if scheduled_drafts > max_scheduled_draft_tokens:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                        "scheduled draft token values must be <= num_speculative_tokens "
                        f"({max_scheduled_draft_tokens}), got request_id={request_id!r}, "
                        f"index={idx}, value={value!r}"
                    )
                validated_trace.append(scheduled_drafts)
            per_request_trace[request_id] = validated_trace
        return per_request_trace

    def __post_init__(self) -> None:
        supported_methods = {
            "ngram",
            "medusa",
            "eagle",
            "eagle3",
            "deepseek_mtp",
            "ernie_mtp",
            "qwen3_moe_mtp",
            "qwen3_next_mtp",
        }
        mtp_methods = {
            "deepseek_mtp",
            "ernie_mtp",
            "qwen3_moe_mtp",
            "qwen3_next_mtp",
        }
        if self.enabled and self.method not in supported_methods:
            raise ValueError(
                "SpeculativeDecodingConfig.method must match vLLM method names, "
                f"got={self.method!r}, supported={sorted(supported_methods)}"
            )
        if self.enabled and self.method in mtp_methods and self.mtp_n_predict <= 0:
            raise ValueError(
                "MTP methods require mtp_n_predict > 0 when enabled=True, "
                f"got method={self.method!r}, mtp_n_predict={self.mtp_n_predict!r}"
            )
        if self.enabled and self.method in mtp_methods and self.mtp_num_layers <= 0:
            raise ValueError(
                "MTP methods require mtp_num_layers > 0 when enabled=True, "
                f"got method={self.method!r}, mtp_num_layers={self.mtp_num_layers!r}"
            )
        if self.mtp_n_predict < 0:
            raise ValueError(
                "SpeculativeDecodingConfig.mtp_n_predict must be >= 0, "
                f"got={self.mtp_n_predict!r}"
            )
        if self.mtp_num_layers < 0:
            raise ValueError(
                "SpeculativeDecodingConfig.mtp_num_layers must be >= 0, "
                f"got={self.mtp_num_layers!r}"
            )
        if self.mtp_n_predict > 0 and self.method not in mtp_methods:
            raise ValueError(
                "SpeculativeDecodingConfig.mtp_n_predict is only valid for MTP "
                f"methods, got method={self.method!r}"
            )
        if self.mtp_num_layers > 0 and self.method not in mtp_methods:
            raise ValueError(
                "SpeculativeDecodingConfig.mtp_num_layers is only valid for MTP "
                f"methods, got method={self.method!r}"
            )
        if self.enabled and self.num_speculative_tokens <= 0:
            raise ValueError(
                "SpeculativeDecodingConfig.num_speculative_tokens must be > 0 when "
                f"enabled=True, got={self.num_speculative_tokens}"
            )
        if (
            self.method in mtp_methods
            and self.mtp_n_predict > 0
            and self.num_speculative_tokens % self.mtp_n_predict != 0
        ):
            raise ValueError(
                "SpeculativeDecodingConfig.num_speculative_tokens must be divisible "
                "by mtp_n_predict when mtp_n_predict > 0 for MTP methods, "
                f"got num_speculative_tokens={self.num_speculative_tokens}, "
                f"mtp_n_predict={self.mtp_n_predict}"
            )
        max_committed_tokens = int(self.num_speculative_tokens) + 1
        if self.committed_tokens_per_iteration < 1:
            raise ValueError(
                "SpeculativeDecodingConfig.committed_tokens_per_iteration must be >= 1, "
                f"got={self.committed_tokens_per_iteration!r}"
            )
        if self.committed_tokens_per_iteration > max_committed_tokens:
            raise ValueError(
                "SpeculativeDecodingConfig.committed_tokens_per_iteration must be <= "
                f"1 + num_speculative_tokens ({max_committed_tokens}), "
                f"got={self.committed_tokens_per_iteration!r}"
            )
        acceptance_trace_payload = self._load_acceptance_trace_payload(
            acceptance_trace_file=self.acceptance_trace_file,
        )
        self._committed_tokens_trace = self._load_committed_tokens_trace(
            acceptance_trace_payload=acceptance_trace_payload,
            max_committed_tokens=max_committed_tokens,
        )
        self._per_request_committed_tokens_trace = (
            self._load_per_request_committed_tokens_trace(
                acceptance_trace_payload=acceptance_trace_payload,
                max_committed_tokens=max_committed_tokens,
            )
        )
        if (
            acceptance_trace_payload is not None
            and self._committed_tokens_trace is None
            and self._per_request_committed_tokens_trace is None
        ):
            raise ValueError(
                "SpeculativeDecodingConfig.acceptance_trace_file JSON object must "
                "contain key 'committed_tokens_per_iteration' or "
                "'per_request_committed_tokens_per_iteration'."
            )
        self._scheduled_draft_tokens_trace = self._load_scheduled_draft_tokens_trace(
            acceptance_trace_payload=acceptance_trace_payload,
            max_scheduled_draft_tokens=int(self.num_speculative_tokens),
            committed_trace_length=(
                len(self._committed_tokens_trace)
                if self._committed_tokens_trace is not None
                else 0
            ),
        )
        self._per_request_scheduled_draft_tokens_trace = (
            self._load_per_request_scheduled_draft_tokens_trace(
                acceptance_trace_payload=acceptance_trace_payload,
                max_scheduled_draft_tokens=int(self.num_speculative_tokens),
                per_request_committed_trace=self._per_request_committed_tokens_trace,
            )
        )
        if self._scheduled_draft_tokens_trace is not None:
            for idx, (committed_tokens, scheduled_draft_tokens) in enumerate(
                zip(
                    self._committed_tokens_trace,
                    self._scheduled_draft_tokens_trace,
                )
            ):
                if committed_tokens > 1 + scheduled_draft_tokens:
                    raise ValueError(
                        "SpeculativeDecodingConfig.acceptance_trace_file committed "
                        "tokens must be <= 1 + scheduled_draft_tokens_per_iteration, "
                        f"got index={idx}, committed={committed_tokens}, "
                        f"scheduled_draft_tokens={scheduled_draft_tokens}"
                    )
        if self._per_request_scheduled_draft_tokens_trace is not None:
            for request_id, committed_trace in (
                self._per_request_committed_tokens_trace.items()
            ):
                scheduled_trace = self._per_request_scheduled_draft_tokens_trace[
                    request_id
                ]
                for idx, (committed_tokens, scheduled_draft_tokens) in enumerate(
                    zip(committed_trace, scheduled_trace)
                ):
                    if committed_tokens > 1 + scheduled_draft_tokens:
                        raise ValueError(
                            "SpeculativeDecodingConfig.acceptance_trace_file per-request "
                            "committed tokens must be <= 1 + "
                            "per_request_scheduled_draft_tokens_per_iteration, "
                            f"got request_id={request_id!r}, index={idx}, "
                            f"committed={committed_tokens}, "
                            f"scheduled_draft_tokens={scheduled_draft_tokens}"
                        )

        trace_payload = self._load_trace_calibration_payload(self.trace_calibration_file)
        supported_trace_keys = {
            "proposer_overhead_ms_by_method",
            "metadata",
        }
        unexpected_keys = sorted(set(trace_payload.keys()) - supported_trace_keys)
        if unexpected_keys:
            raise ValueError(
                "Unsupported keys in trace calibration file: "
                f"{unexpected_keys}, supported={sorted(supported_trace_keys)}"
            )

        trace_proposer_overheads = self._validate_method_float_map(
            map_name="proposer_overhead_ms_by_method",
            raw_map=trace_payload.get("proposer_overhead_ms_by_method", {}),
            supported_methods=supported_methods,
            min_value=0.0,
            inclusive_min=True,
        )
        config_proposer_overheads = self._validate_method_float_map(
            map_name="proposer_overhead_ms_by_method",
            raw_map=self.proposer_overhead_ms_by_method,
            supported_methods=supported_methods,
            min_value=0.0,
            inclusive_min=True,
        )

        # Config-driven values override trace-derived values for deterministic control.
        self.proposer_overhead_ms_by_method = {
            **trace_proposer_overheads,
            **config_proposer_overheads,
        }
        self._decode_draft_proposer_latency_profile = (
            load_decode_draft_proposer_latency_profile(
                profile_file=self.decode_draft_proposer_latency_profile_file,
                supported_methods=supported_methods,
            )
        )


@dataclass
class ReplicaConfig:
    memory_margin_fraction: float = field(
        default=0.1,
        metadata={"help": "Memory margin fraction."},
    )
    num_pipeline_stages: int = field(
        default=1,
        metadata={"help": "Number of pipeline stages (pp size)."},
    )
    attn_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Attention tensor parallel size (attn_tp size)."},
    )
    attn_data_parallel_size: int = field(
        default=1,
        metadata={"help": "Attention data parallel size (attn_dp size)."},
    )
    moe_tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "MoE tensor parallel size (moe_tp size)."},
    )
    moe_expert_parallel_size: int = field(
        default=1,
        metadata={"help": "MoE expert parallel size (moe_ep size)."},
    )
    total_expert_num: int = field(
        default=1,
        metadata={"help": "Total expert number."},
    )
    router_load_balancing_type: str = field(
        default="None",
        metadata={"help": "MOE router load balancing type."},
    )
    router_topk: int = field(
        default=0,
        metadata={"help": "Router topk. Set to 0 to inherit from model config."},
    )
    moe_routing_mode: str = field(
        default="simulation",
        metadata={
            "help": "MoE routing mode for grouped GEMM prediction. "
            "'simulation': Pre-compute routing details via simulation for realistic load imbalance (default). "
            "'uniform_legacy': Use uniform token distribution (legacy 1D mode for backward compatibility). "
            "'uniform_random': Comparison-only deterministic uniform-random expert assignment mirroring vLLM routing simulation."
        },
    )
    moe_routing_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed for MoE routing simulation. Must be a non-negative integer. "
            "Used when moe_routing_mode is 'simulation' or 'uniform_random' to ensure deterministic and reproducible results."
        },
    )
    moe_routing_distribution_type: str = field(
        default="balanced",
        metadata={
            "help": "MoE expert-load distribution for disaggregated routing simulation. "
            "Valid values: 'balanced', 'random', 'skewed', or 'zipf'. This controls "
            "token-to-expert load skew without changing router_topk/model semantics."
        },
    )
    # todo: remove this, shouldn't use in simulation in current version
    extend_ep_across_dp: bool = field(
        default=False,
        metadata={
            "help": "Whether to extend expert parallelism across data parallelism."
        },
    )
    device: str = field(
        default="a100",
        metadata={"help": "Device."},
    )
    network_device: str = field(
        default="a100_pairwise_nvlink",
        metadata={"help": "Network device."},
    )
    speculative_decoding_config: SpeculativeDecodingConfig = field(
        default_factory=SpeculativeDecodingConfig,
        metadata={"help": "Speculative decoding simulation configuration."},
    )

    # configs should be set by the user
    cluster_prefix: str = None
    local_expert_num: int = None
    model_name: str = "meta-llama/Llama-2-7b-hf"
    data_parallel_size: int = None  # to be set in monolithic mode

    def __post_init__(self):
        # Load model and device configs first (needed for validation)
        self.model_config: BaseModelConfig = BaseModelConfig.create_from_name(
            self.model_name
        )
        self.device_config: BaseDeviceSKUConfig = (
            BaseDeviceSKUConfig.create_from_type_string(self.device)
        )
        self.node_config: BaseNodeSKUConfig = BaseNodeSKUConfig.create_from_type_string(
            self.network_device
        )

        # Auto-set total_expert_num from model config if not explicitly set and model is MoE
        if (
            self.total_expert_num == 1
            and self.model_config.is_moe
            and self.model_config.num_experts > 0
        ):
            self.total_expert_num = self.model_config.num_experts

        # Align router_topk with model config when not explicitly set.
        if self.model_config.is_moe:
            if self.router_topk is None or int(self.router_topk) <= 0:
                if self.model_config.num_experts_per_tok > 0:
                    self.router_topk = int(self.model_config.num_experts_per_tok)
                else:
                    raise ValueError(
                        "router_topk is not set and model_config.num_experts_per_tok is missing"
                    )
        else:
            if self.router_topk is None or int(self.router_topk) <= 0:
                self.router_topk = 1

        valid_moe_routing_distribution_types = {
            "balanced",
            "random",
            "skewed",
            "zipf",
        }
        self.moe_routing_distribution_type = str(
            self.moe_routing_distribution_type
        ).strip().lower()
        if self.moe_routing_distribution_type not in valid_moe_routing_distribution_types:
            raise ValueError(
                "moe_routing_distribution_type must be one of "
                f"{sorted(valid_moe_routing_distribution_types)}, "
                f"got {self.moe_routing_distribution_type!r}"
            )

        # Validate pipeline parallelism configuration early
        if self.model_config.num_layers % self.num_pipeline_stages != 0:
            raise ValueError(
                f"Pipeline parallelism configuration error: "
                f"num_layers ({self.model_config.num_layers}) must be evenly divisible by "
                f"num_pipeline_stages ({self.num_pipeline_stages}). "
                f"Current configuration would result in uneven layer distribution across pipeline stages. "
                f"Please adjust num_pipeline_stages to be a divisor of {self.model_config.num_layers}."
            )

        # Note: this world_size only limits in replica dimension.
        if self.cluster_prefix == "prefill":
            self.world_size = (
                self.num_pipeline_stages
                * self.attn_tensor_parallel_size
                * self.attn_data_parallel_size
            )
        elif self.cluster_prefix == "decode_attn":
            self.world_size = (
                self.num_pipeline_stages
                * self.attn_tensor_parallel_size
                * self.attn_data_parallel_size
            )
        elif self.cluster_prefix == "decode_ffn":
            self.world_size = (
                self.num_pipeline_stages
                * self.moe_tensor_parallel_size
                * self.moe_expert_parallel_size
            )
        elif self.cluster_prefix == "decode":
            # Unified decode cluster (PD-disaggregation): similar to prefill, includes both Attention and FFN
            self.world_size = (
                self.num_pipeline_stages
                * self.attn_tensor_parallel_size
                * self.attn_data_parallel_size
            )
        else:  # Monolithic
            self.world_size = (
                self.num_pipeline_stages
                * self.attn_tensor_parallel_size
                * self.attn_data_parallel_size
            )

        # Validate expert parallelism configuration for MoE models
        # Use model_config.is_moe for MoE detection - NOT total_expert_num
        if self.cluster_prefix != "decode_attn" and self.model_config.is_moe:
            if self.total_expert_num > 1:
                assert (
                    self.total_expert_num % self.moe_expert_parallel_size == 0
                ), "total_expert_num must be divisible by moe_expert_parallel_size"
                self.local_expert_num = (
                    self.total_expert_num // self.moe_expert_parallel_size
                )

        if (
            self.speculative_decoding_config.enabled
            and self.cluster_prefix in {"decode_attn", "decode_ffn"}
        ):
            raise ValueError(
                "Speculative decoding Phase 1 supports only co-location and "
                "pd-disaggregation decode path. decode_attn/decode_ffn are not "
                f"supported, cluster_prefix={self.cluster_prefix!r}."
            )


@dataclass
class BaseClusterSchedulerConfig(BasePolyConfig):
    pass


@dataclass
class RandomClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.RANDOM


@dataclass
class RoundRobinClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.ROUND_ROBIN


@dataclass
class LORClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.LOR


@dataclass
class StickyRoundRobinClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.STICKY_ROUND_ROBIN


@dataclass
class StickyLORClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.STICKY_LOR


@dataclass
class BaseExecutionTimePredictorConfig(BasePolyConfig):
    linear_op_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/linear_op.csv",
        metadata={"help": "Path to the linear operation profiling input file."},
    )
    # Backward compatibility alias
    mlp_input_file: str = field(
        default="",
        metadata={"help": "[DEPRECATED] Use linear_op_input_file instead."},
    )
    atten_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv",
        metadata={"help": "Path to the attention input file."},
    )
    all_reduce_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv",
        metadata={"help": "Path to the all reduce input file."},
    )
    send_recv_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv",
        metadata={"help": "Path to the send recv input file."},
    )
    cpu_overhead_input_file: str = field(
        default="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv",
        metadata={"help": "Path to the cpu overhead input file."},
    )
    cpu_overhead_kernel_only_input_file: str = field(
        default="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads_kernel_only.csv",
        metadata={"help": "Path to the kernel-only cpu overhead input file."},
    )
    pp_stage_boundary_input_file: str = field(
        default="./data/profiling/other_overhead/{DEVICE}/{MODEL}/pp_stage_boundary.csv",
        metadata={"help": "Path to the pipeline stage-boundary overhead input file."},
    )
    pp_receiver_head_input_file: str = field(
        default="./data/profiling/other_overhead/{DEVICE}/{MODEL}/pp_receiver_head.csv",
        metadata={"help": "Path to the PP receiver-head overhead input file."},
    )
    pp_producer_send_path_input_file: str = field(
        default="./data/profiling/other_overhead/{DEVICE}/{MODEL}/pp_producer_send_path.csv",
        metadata={"help": "Path to the PP producer send-path overhead input file."},
    )
    pp_prefill_consumer_active_input_file: str = field(
        default="./data/profiling/other_overhead/{DEVICE}/{MODEL}/pp_prefill_consumer_active.csv",
        metadata={
            "help": "Path to the PP prefill consumer-active overhead input file."
        },
    )
    moe_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/moe.csv",
        metadata={"help": "Path to the MoE profiling input file."},
    )
    linear_op_kernel_only_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/linear_op_kernel_only.csv",
        metadata={"help": "Path to the kernel-only linear operation profiling input file."},
    )
    atten_kernel_only_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/attention_kernel_only.csv",
        metadata={"help": "Path to the kernel-only attention input file."},
    )
    moe_kernel_only_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/moe_kernel_only.csv",
        metadata={"help": "Path to the kernel-only MoE profiling input file."},
    )
    k_fold_cv_splits: int = field(
        default=10,
        metadata={"help": "Number of k fold cross validation splits."},
    )
    no_cache: bool = field(
        default=False,
        metadata={"help": "Whether to cache prediction models."},
    )
    kv_cache_prediction_granularity: int = field(
        default=64,
        metadata={"help": "KV cache prediction granularity."},
    )
    prediction_max_prefill_chunk_size: int = field(
        default=4096,
        metadata={"help": "Max prefill chunk size for prediction."},
    )
    prediction_max_batch_size: int = field(
        default=128,
        metadata={"help": "Max batch size for prediction."},
    )
    prediction_max_tokens_per_request: int = field(
        default=4096,
        metadata={"help": "Max tokens per request for prediction."},
    )
    attention_decode_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention decode batching overhead fraction."},
    )
    attention_prefill_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention prefill batching overhead fraction."},
    )
    attn_pre_proj_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for attn_pre_proj prediction. Must be > 0."
        },
    )
    prefill_phase_attn_pre_proj_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_pre_proj "
                "prediction when the batch includes prefill tokens. Must be > 0."
            )
        },
    )
    attn_post_proj_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for attn_post_proj prediction. Must be > 0."
        },
    )
    prefill_phase_attn_post_proj_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_post_proj "
                "prediction when the batch includes prefill tokens. Must be > 0."
            )
        },
    )
    attn_decode_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for attn_decode prediction. Must be > 0."
        },
    )
    attn_decode_in_mixed_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_decode_in_mixed "
                "prediction when a co-location batch contains both prefill and decode "
                "tokens. Must be > 0."
            )
        },
    )
    late_decode_attn_decode_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_decode "
                "prediction when every decode request in the batch has already "
                "completed the first pure decode token. Must be > 0."
            )
        },
    )
    attn_kv_cache_save_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for attn_kv_cache_save prediction. Must be > 0."
        },
    )
    prefill_phase_attn_kv_cache_save_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for attn_kv_cache_save "
                "prediction when the batch includes prefill tokens. Must be > 0."
            )
        },
    )
    mlp_up_proj_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for mlp_up_proj prediction. Must be > 0."
        },
    )
    prefill_phase_mlp_up_proj_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for mlp_up_proj "
                "prediction when the batch includes prefill tokens. Must be > 0."
            )
        },
    )
    mlp_down_proj_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for mlp_down_proj prediction. Must be > 0."
        },
    )
    decode_phase_mlp_down_proj_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for mlp_down_proj "
                "prediction when the batch contains decode tokens but no "
                "prefill tokens. Must be > 0."
            )
        },
    )
    moe_shuffling_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for moe_shuffling prediction. Must be > 0."
        },
    )
    decode_phase_moe_shuffling_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for moe_shuffling "
                "prediction when the batch contains decode tokens but no "
                "prefill tokens. Must be > 0."
            )
        },
    )
    moe_grouped_gemm_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": "Multiplicative calibration scale for moe_grouped_gemm prediction. Must be > 0."
        },
    )
    decode_phase_moe_grouped_gemm_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for moe_grouped_gemm "
                "prediction when the batch contains decode tokens but no "
                "prefill tokens. Must be > 0."
            )
        },
    )
    expert_parallel_communication_calibration_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "Multiplicative calibration scale for expert parallel communication "
                "prediction. Must be > 0."
            )
        },
    )
    decode_phase_expert_parallel_communication_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for expert parallel "
                "communication prediction when the batch contains decode tokens "
                "but no prefill tokens. Must be > 0."
            )
        },
    )
    late_decode_expert_parallel_communication_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative calibration scale for expert parallel "
                "communication prediction when every request in a decode-only "
                "batch is past the first pure decode token. Must be > 0."
            )
        },
    )
    short_decode_request_length_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original decode-token threshold for short-request "
                "decode-only MoE calibration. Must be > 0 when set."
            )
        },
    )
    short_decode_request_length_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative MoE calibration scale for decode-only "
                "batches where every request has original decode tokens <= "
                "short_decode_request_length_threshold. Must be > 0."
            )
        },
    )
    long_decode_request_length_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original decode-token threshold for long-request "
                "decode-only MoE calibration. Must be > 0 when set."
            )
        },
    )
    long_decode_request_length_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative MoE calibration scale for decode-only "
                "batches where any request has original decode tokens >= "
                "long_decode_request_length_threshold. Must be > 0."
            )
        },
    )
    low_prefill_short_decode_request_prefill_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original prefill-token threshold for low-prefill, "
                "short-decode request-shape MoE calibration. Must be > 0 when set."
            )
        },
    )
    low_prefill_short_decode_request_decode_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original decode-token threshold for low-prefill, "
                "short-decode request-shape MoE calibration. Must be > 0 when set."
            )
        },
    )
    low_prefill_short_decode_request_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative MoE calibration scale for decode-only "
                "batches containing a request with original prefill tokens <= "
                "low_prefill_short_decode_request_prefill_threshold and original "
                "decode tokens <= low_prefill_short_decode_request_decode_threshold. "
                "Must be > 0."
            )
        },
    )
    low_prefill_decode_mix_request_prefill_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original prefill-token ceiling for batch-composition "
                "low-prefill decode-mix MoE calibration. Must be > 0 when set."
            )
        },
    )
    low_prefill_decode_mix_request_decode_min: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original decode-token lower bound for batch-composition "
                "low-prefill decode-mix MoE calibration. Must be > 0 when set."
            )
        },
    )
    low_prefill_decode_mix_request_decode_max: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original decode-token upper bound for batch-composition "
                "low-prefill decode-mix MoE calibration. Must be > 0 when set."
            )
        },
    )
    low_prefill_decode_mix_request_min_match_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional inclusive lower bound on matched request ratio for "
                "batch-composition low-prefill decode-mix MoE calibration. "
                "Must satisfy 0 < ratio <= 1 when set."
            )
        },
    )
    low_prefill_decode_mix_request_max_match_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional inclusive upper bound on matched request ratio for "
                "batch-composition low-prefill decode-mix MoE calibration. "
                "Must satisfy 0 < ratio <= 1 when set."
            )
        },
    )
    low_prefill_decode_mix_request_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative MoE calibration scale for decode-only "
                "batches whose low-prefill decode-mix request ratio falls within "
                "the configured inclusive match-ratio band. Must be > 0."
            )
        },
    )
    low_prefill_decode_mix_request_include_mixed_batches: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow low-prefill decode-mix MoE request-shape calibration to "
                "also apply to mixed prefill+decode batches. Default false keeps "
                "the calibration decode-only."
            )
        },
    )
    low_prefill_long_decode_request_prefill_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original prefill-token threshold for low-prefill, "
                "long-decode request-shape MoE calibration. Must be > 0 when set."
            )
        },
    )
    low_prefill_long_decode_request_decode_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original decode-token threshold for low-prefill, "
                "long-decode request-shape MoE calibration. Must be > 0 when set."
            )
        },
    )
    low_prefill_long_decode_request_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative MoE calibration scale for decode-only "
                "batches containing a request with original prefill tokens <= "
                "low_prefill_long_decode_request_prefill_threshold and original "
                "decode tokens >= low_prefill_long_decode_request_decode_threshold. "
                "Must be > 0."
            )
        },
    )
    low_prefill_long_decode_request_include_mixed_batches: bool = field(
        default=False,
        metadata={
            "help": (
                "Allow low-prefill long-decode MoE request-shape calibration to "
                "also apply to mixed prefill+decode batches. Default false keeps "
                "the calibration decode-only."
            )
        },
    )
    high_prefill_mid_decode_request_prefill_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original prefill-token threshold for high-prefill, "
                "mid-decode request-shape MoE calibration. Must be > 0 when set."
            )
        },
    )
    high_prefill_mid_decode_request_decode_min: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original decode-token lower bound for high-prefill, "
                "mid-decode request-shape MoE calibration. Must be > 0 when set."
            )
        },
    )
    high_prefill_mid_decode_request_decode_max: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional original decode-token upper bound for high-prefill, "
                "mid-decode request-shape MoE calibration. Must be > 0 when set."
            )
        },
    )
    high_prefill_mid_decode_request_calibration_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Optional multiplicative MoE calibration scale for decode-only "
                "batches containing a request with original prefill tokens >= "
                "high_prefill_mid_decode_request_prefill_threshold and original "
                "decode tokens within [high_prefill_mid_decode_request_decode_min, "
                "high_prefill_mid_decode_request_decode_max]. Must be > 0."
            )
        },
    )
    share_expert_tp_allreduce_visibility_scale: float = field(
        default=2.0 / 3.0,
        metadata={
            "help": (
                "Visibility scale for Step3 share_expert TP allreduce overlap modeling. "
                "Must be > 0."
            )
        },
    )
    nccl_cpu_launch_overhead_ms: float = field(
        default=0.02,
        metadata={"help": "NCCL CPU launch overhead in ms."},
    )
    nccl_cpu_skew_overhead_per_device_ms: float = field(
        default=0.0,
        metadata={"help": "NCCL CPU skew overhead per device in ms."},
    )
    num_training_job_threads: int = field(
        default=-1,
        metadata={"help": "Number of training job threads."},
    )
    skip_cpu_overhead_modeling: bool = field(
        default=True,
        metadata={"help": "Whether to skip CPU overhead modeling."},
    )

    # Dummy mode configuration for fast testing and development
    enable_dummy_mode: bool = field(
        default=False,
        metadata={
            "help": "Enable dummy mode to skip ML model training and return fixed execution times."
        },
    )
    dummy_execution_time_ms: float = field(
        default=1.0,
        metadata={
            "help": "Fixed execution time in milliseconds to return in dummy mode."
        },
    )

    def __post_init__(self) -> None:
        for field_name in (
            "attn_pre_proj_calibration_scale",
            "prefill_phase_attn_pre_proj_calibration_scale",
            "attn_post_proj_calibration_scale",
            "prefill_phase_attn_post_proj_calibration_scale",
            "attn_decode_calibration_scale",
            "attn_decode_in_mixed_calibration_scale",
            "late_decode_attn_decode_calibration_scale",
            "attn_kv_cache_save_calibration_scale",
            "prefill_phase_attn_kv_cache_save_calibration_scale",
            "mlp_up_proj_calibration_scale",
            "prefill_phase_mlp_up_proj_calibration_scale",
            "mlp_down_proj_calibration_scale",
            "decode_phase_mlp_down_proj_calibration_scale",
            "moe_shuffling_calibration_scale",
            "decode_phase_moe_shuffling_calibration_scale",
            "moe_grouped_gemm_calibration_scale",
            "decode_phase_moe_grouped_gemm_calibration_scale",
            "expert_parallel_communication_calibration_scale",
            "decode_phase_expert_parallel_communication_calibration_scale",
            "late_decode_expert_parallel_communication_calibration_scale",
            "short_decode_request_length_calibration_scale",
            "long_decode_request_length_calibration_scale",
            "low_prefill_short_decode_request_calibration_scale",
            "low_prefill_long_decode_request_calibration_scale",
            "high_prefill_mid_decode_request_calibration_scale",
            "low_prefill_decode_mix_request_calibration_scale",
            "share_expert_tp_allreduce_visibility_scale",
        ):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if value <= 0.0:
                raise ValueError(
                    f"{self.__class__.__name__}.{field_name} must be > 0, got={value!r}"
                )
        for field_name in (
            "short_decode_request_length_threshold",
            "long_decode_request_length_threshold",
            "low_prefill_short_decode_request_prefill_threshold",
            "low_prefill_short_decode_request_decode_threshold",
            "low_prefill_decode_mix_request_prefill_threshold",
            "low_prefill_decode_mix_request_decode_min",
            "low_prefill_decode_mix_request_decode_max",
            "low_prefill_long_decode_request_prefill_threshold",
            "low_prefill_long_decode_request_decode_threshold",
            "high_prefill_mid_decode_request_prefill_threshold",
            "high_prefill_mid_decode_request_decode_min",
            "high_prefill_mid_decode_request_decode_max",
        ):
            raw_value = getattr(self, field_name)
            if raw_value is None:
                continue
            value = int(raw_value)
            if value <= 0:
                raise ValueError(
                    f"{self.__class__.__name__}.{field_name} must be > 0, got={value!r}"
                )
        for prefix in ("short", "long"):
            threshold_name = f"{prefix}_decode_request_length_threshold"
            scale_name = f"{prefix}_decode_request_length_calibration_scale"
            threshold = getattr(self, threshold_name)
            scale = getattr(self, scale_name)
            if (threshold is None) != (scale is None):
                raise ValueError(
                    f"{self.__class__.__name__}.{threshold_name} and {scale_name} "
                    "must be set together"
                )
        short_threshold = self.short_decode_request_length_threshold
        long_threshold = self.long_decode_request_length_threshold
        if (
            short_threshold is not None
            and long_threshold is not None
            and long_threshold <= short_threshold
        ):
            raise ValueError(
                f"{self.__class__.__name__}.long_decode_request_length_threshold "
                "must be greater than short_decode_request_length_threshold"
            )

        low_prefill_short_fields = (
            "low_prefill_short_decode_request_prefill_threshold",
            "low_prefill_short_decode_request_decode_threshold",
            "low_prefill_short_decode_request_calibration_scale",
        )
        low_prefill_short_values = [
            getattr(self, field_name) for field_name in low_prefill_short_fields
        ]
        if any(value is not None for value in low_prefill_short_values) and not all(
            value is not None for value in low_prefill_short_values
        ):
            raise ValueError(
                f"{self.__class__.__name__}.low_prefill_short_decode_request_prefill_threshold, "
                "low_prefill_short_decode_request_decode_threshold, and "
                "low_prefill_short_decode_request_calibration_scale must be set together"
            )

        low_prefill_decode_mix_fields = (
            "low_prefill_decode_mix_request_prefill_threshold",
            "low_prefill_decode_mix_request_decode_min",
            "low_prefill_decode_mix_request_decode_max",
            "low_prefill_decode_mix_request_min_match_ratio",
            "low_prefill_decode_mix_request_max_match_ratio",
            "low_prefill_decode_mix_request_calibration_scale",
        )
        low_prefill_decode_mix_values = [
            getattr(self, field_name) for field_name in low_prefill_decode_mix_fields
        ]
        if any(value is not None for value in low_prefill_decode_mix_values) and not all(
            value is not None for value in low_prefill_decode_mix_values
        ):
            raise ValueError(
                f"{self.__class__.__name__}.low_prefill_decode_mix_request_prefill_threshold, "
                "low_prefill_decode_mix_request_decode_min, "
                "low_prefill_decode_mix_request_decode_max, "
                "low_prefill_decode_mix_request_min_match_ratio, "
                "low_prefill_decode_mix_request_max_match_ratio, and "
                "low_prefill_decode_mix_request_calibration_scale must be set together"
            )

        low_decode_mix_min = self.low_prefill_decode_mix_request_decode_min
        low_decode_mix_max = self.low_prefill_decode_mix_request_decode_max
        if (
            low_decode_mix_min is not None
            and low_decode_mix_max is not None
            and low_decode_mix_max < low_decode_mix_min
        ):
            raise ValueError(
                f"{self.__class__.__name__}.low_prefill_decode_mix_request_decode_max "
                "must be greater than or equal to low_prefill_decode_mix_request_decode_min"
            )

        low_decode_mix_min_ratio = (
            self.low_prefill_decode_mix_request_min_match_ratio
        )
        low_decode_mix_max_ratio = (
            self.low_prefill_decode_mix_request_max_match_ratio
        )
        for field_name, raw_value in (
            (
                "low_prefill_decode_mix_request_min_match_ratio",
                low_decode_mix_min_ratio,
            ),
            (
                "low_prefill_decode_mix_request_max_match_ratio",
                low_decode_mix_max_ratio,
            ),
        ):
            if raw_value is None:
                continue
            value = float(raw_value)
            if value <= 0.0 or value > 1.0:
                raise ValueError(
                    f"{self.__class__.__name__}.{field_name} must satisfy 0 < value <= 1, got={value!r}"
                )
        if (
            low_decode_mix_min_ratio is not None
            and low_decode_mix_max_ratio is not None
            and low_decode_mix_max_ratio < low_decode_mix_min_ratio
        ):
            raise ValueError(
                f"{self.__class__.__name__}.low_prefill_decode_mix_request_max_match_ratio "
                "must be greater than or equal to low_prefill_decode_mix_request_min_match_ratio"
            )

        low_prefill_long_fields = (
            "low_prefill_long_decode_request_prefill_threshold",
            "low_prefill_long_decode_request_decode_threshold",
            "low_prefill_long_decode_request_calibration_scale",
        )
        low_prefill_long_values = [
            getattr(self, field_name) for field_name in low_prefill_long_fields
        ]
        if any(value is not None for value in low_prefill_long_values) and not all(
            value is not None for value in low_prefill_long_values
        ):
            raise ValueError(
                f"{self.__class__.__name__}.low_prefill_long_decode_request_prefill_threshold, "
                "low_prefill_long_decode_request_decode_threshold, and "
                "low_prefill_long_decode_request_calibration_scale must be set together"
            )

        high_prefill_mid_fields = (
            "high_prefill_mid_decode_request_prefill_threshold",
            "high_prefill_mid_decode_request_decode_min",
            "high_prefill_mid_decode_request_decode_max",
            "high_prefill_mid_decode_request_calibration_scale",
        )
        high_prefill_mid_values = [
            getattr(self, field_name) for field_name in high_prefill_mid_fields
        ]
        if any(value is not None for value in high_prefill_mid_values) and not all(
            value is not None for value in high_prefill_mid_values
        ):
            raise ValueError(
                f"{self.__class__.__name__}.high_prefill_mid_decode_request_prefill_threshold, "
                "high_prefill_mid_decode_request_decode_min, "
                "high_prefill_mid_decode_request_decode_max, and "
                "high_prefill_mid_decode_request_calibration_scale must be set together"
            )

        high_decode_min = self.high_prefill_mid_decode_request_decode_min
        high_decode_max = self.high_prefill_mid_decode_request_decode_max
        if (
            high_decode_min is not None
            and high_decode_max is not None
            and high_decode_max < high_decode_min
        ):
            raise ValueError(
                f"{self.__class__.__name__}.high_prefill_mid_decode_request_decode_max "
                "must be greater than or equal to high_prefill_mid_decode_request_decode_min"
            )

    def validate_linear_op_input(self) -> None:
        """Validate linear_op_input_file configuration.
        
        Raises:
            ValueError: If mlp.csv path is used or linear_op_input_file is empty.
        """
        # Reject mlp.csv paths
        if self.linear_op_input_file and "mlp.csv" in self.linear_op_input_file:
            raise ValueError(
                f"mlp.csv is forbidden in linear_op_input_file. "
                f"Use linear_op.csv instead. Got: {self.linear_op_input_file}"
            )
        
        # Reject empty path when validation is explicitly called
        if not self.linear_op_input_file:
            raise ValueError(
                "linear_op_input_file must be set to a valid linear_op.csv path."
            )
        
        # Warn if deprecated mlp_input_file is used
        if self.mlp_input_file:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "mlp_input_file is deprecated and will be ignored. "
                "Use linear_op_input_file with linear_op.csv instead."
            )



@dataclass
class LinearRegressionExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    polynomial_degree: List[int] = field(
        default_factory=lambda: list(range(1, 6)),
        metadata={"help": "Polynomial degree for linear regression."},
    )
    polynomial_include_bias: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial include bias for linear regression."},
    )
    polynomial_interaction_only: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial interaction only for linear regression."},
    )
    fit_intercept: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Fit intercept for linear regression."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.LINEAR_REGRESSION


@dataclass
class RandomForrestExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    num_estimators: List[int] = field(
        default_factory=lambda: [250, 500, 750],
        metadata={"help": "Number of estimators for random forest."},
    )
    max_depth: List[int] = field(
        default_factory=lambda: [8, 16, 32],
        metadata={"help": "Maximum depth for random forest."},
    )
    min_samples_split: List[int] = field(
        default_factory=lambda: [2, 5, 10],
        metadata={"help": "Minimum samples split for random forest."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.RANDOM_FORREST


@dataclass
class ClusterConfig:
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

    # === Periodic scheduling configuration ===
    # default_factory=lambda: [ClusterType.DECODE_ATTN],
    periodic_scheduling_clusters: List[ClusterType] = field(
        default_factory=lambda: [],
        metadata={
            "help": "List of cluster types that use periodic scheduling instead of event-driven scheduling. "
            "Currently only DECODE_ATTN is supported.",
        },
    )
    periodic_scheduling_interval_ms: float = field(
        default=10.0,
        metadata={
            "help": "Scheduling interval in milliseconds for clusters using periodic scheduling.",
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
            "help": "Number of replicas for decode FFN cluster (must be 1). "
            "DP for the FFN cluster is not supported because it causes expert redundancy. "
            "Use EP within a single replica instead. Used only in pd-af-disaggregation mode.",
            "mode_dependency": "pd-af-disaggregation",
        },
    )
    allow_experiment_multi_decode_ffn_replicas: bool = field(
        default=False,
        metadata={
            "help": "Experiment-only opt-in that permits multiple DECODE_FFN replicas "
            "for Use Case 6 heterogeneous cost studies. Default behavior keeps "
            "decode_ffn_cluster_num_replicas constrained to 1."
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
    prefill_replica_config_attn_data_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Attention data parallel size for prefill cluster.",
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
    prefill_replica_config_extend_ep_across_dp: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to extend expert parallelism across data parallelism for prefill cluster.",
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
    decode_attn_replica_config_attn_data_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Attention data parallel size for decode attention cluster.",
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
    decode_ffn_replica_config_extend_ep_across_dp: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to extend expert parallelism across data parallelism for decode FFN cluster.",
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
    decode_replica_config_attn_data_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Attention data parallel size for unified decode cluster.",
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
    decode_replica_config_extend_ep_across_dp: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to extend expert parallelism across data parallelism for unified decode cluster.",
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

    # Derived cross-cluster parameter propagated for DECODE_FFN scheduling
    # For decode-ffn cluster: number of DP lanes sourced from DECODE_ATTN config
    decode_attn_dp_lanes_for_ffn: Optional[int] = field(
        default=None,
        metadata={
            "help": "Derived lanes = decode-attn num_replicas * decode-attn DP size; used by decode-ffn grouping.",
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
    prefill_execution_time_predictor_config_moe_shuffling_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override moe_shuffling calibration scale for the prefill cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_execution_time_predictor_config_moe_grouped_gemm_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override moe_grouped_gemm calibration scale for the prefill cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-af-disaggregation,pd-disaggregation",
        },
    )
    prefill_execution_time_predictor_config_expert_parallel_communication_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override expert parallel communication calibration scale for "
                "the prefill cluster execution-time predictor. Must be > 0."
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
    decode_execution_time_predictor_config_moe_shuffling_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override moe_shuffling calibration scale for the decode cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_execution_time_predictor_config_moe_grouped_gemm_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override moe_grouped_gemm calibration scale for the decode cluster "
                "execution-time predictor. Must be > 0."
            ),
            "mode_dependency": "pd-disaggregation",
        },
    )
    decode_execution_time_predictor_config_expert_parallel_communication_calibration_scale: Optional[
        float
    ] = field(
        default=None,
        metadata={
            "help": (
                "Override expert parallel communication calibration scale for "
                "the decode cluster execution-time predictor. Must be > 0."
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
        self._validate_open_source_release_cluster_type_guard()
        self._validate_open_source_release_disaggregation_fields_guard()
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

        Aligned with StepFun-vLLM's requirement that max_num_batched_tokens and
        max_num_seqs be divisible by num_afd_stages. However, unlike StepFun which
        silently rounds down, we fail fast to help users identify configuration issues.

        Args:
            num_stages: The number of AFD stages (af_pipeline_num_micro_batch)

        Raises:
            ValueError: If any required parameter is not divisible by num_stages
        """
        errors = []

        # Check decode_attn_micro_batch_size
        if self.decode_attn_micro_batch_size is not None:
            if self.decode_attn_micro_batch_size % num_stages != 0:
                errors.append(
                    f"decode_attn_micro_batch_size ({self.decode_attn_micro_batch_size}) "
                    f"must be divisible by num_stages ({num_stages})"
                )

        # Check decode_attn_replica_scheduler_config_batch_size_cap
        if self.decode_attn_replica_scheduler_config_batch_size_cap is not None:
            if self.decode_attn_replica_scheduler_config_batch_size_cap % num_stages != 0:
                errors.append(
                    f"decode_attn_replica_scheduler_config_batch_size_cap "
                    f"({self.decode_attn_replica_scheduler_config_batch_size_cap}) "
                    f"must be divisible by num_stages ({num_stages})"
                )

        if errors:
            raise ValueError(
                "AFD divisibility validation failed (fail fast). "
                "StepFun-vLLM requires batch sizes to be divisible by num_stages. "
                "Errors:\n  - " + "\n  - ".join(errors)
            )

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

    def _validate_open_source_release_disaggregation_fields_guard(self) -> None:
        if self._has_disaggregation_params_set():
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _validate_open_source_release_cluster_type_guard(self) -> None:
        if (
            self.cluster_type is not None
            and self.cluster_type != ClusterType.MONOLITHIC
        ):
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

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

        # Keep replica-local DP lanes aligned with attn_data_parallel_size in monolithic mode.
        # Cluster-level scaling is already represented by num_replicas.
        self.replica_config.data_parallel_size = (
            self.replica_config.attn_data_parallel_size
        )

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

        # if self.replica_config.extend_ep_across_dp:
        #     assert self.replica_config.expert_parallel_size == self.replica_config.data_parallel_size * \
        #         self.replica_config.tensor_parallel_size, "For global MoE, expert_parallel_size must be equal to data_parallel_size * tensor_parallel_size"
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

            # Enforce decode-ffn replica count == 1.
            # DP for the FFN cluster is not supported by default because it would
            # cause expert redundancy across replicas (each replica holds the full
            # expert set, wasting memory). Use EP within a single replica instead.
            if (
                self.decode_ffn_cluster_num_replicas != 1
                and not self.allow_experiment_multi_decode_ffn_replicas
            ):
                raise ValueError(
                    f"decode_ffn_cluster_num_replicas must be 1, got {self.decode_ffn_cluster_num_replicas}. "
                    f"DP for the FFN cluster is not enabled by default because it causes "
                    f"expert redundancy (each replica duplicates the full expert set). "
                    f"Use expert parallelism (EP) within a single replica instead. "
                    f"Set allow_experiment_multi_decode_ffn_replicas=True only for "
                    f"explicit experiment-only studies."
                )

            # DECODE_FFN grouping semantics are implemented only in the
            # RoundRobinClusterScheduler path.
            cluster_scheduler_type = self.cluster_scheduler_config.get_type()
            if cluster_scheduler_type != ClusterSchedulerType.ROUND_ROBIN:
                raise ValueError(
                    "PD+AF mode requires RoundRobin cluster scheduler when DECODE_FFN is enabled. "
                    f"Got cluster_scheduler_config_type={cluster_scheduler_type}."
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
        if any(
            cluster_type != ClusterType.MONOLITHIC
            for cluster_type in self.periodic_scheduling_clusters
        ):
            return True
        return bool(self.allow_experiment_multi_decode_ffn_replicas)

    def _create_replica_config_from_fields(self, cluster_prefix: str) -> ReplicaConfig:
        """Create ReplicaConfig object from flattened fields."""
        # Get default values from main replica_config or use ReplicaConfig defaults
        main_config = self.replica_config if self.replica_config else ReplicaConfig()

        # Extract cluster-specific fields using getattr with fallback to main config
        def get_field_value(field_name: str):
            cluster_field_name = f"{cluster_prefix}_replica_config_{field_name}"
            cluster_value = getattr(self, cluster_field_name, None)
            if cluster_value is not None:
                return cluster_value
            return getattr(main_config, field_name)

        if cluster_prefix == "decode_attn":
            moe_expert_parallel_size = 0
            moe_tensor_parallel_size = 0
            total_expert_num = 0
            local_expert_num = 0
            num_pipeline_stages = 1
            router_load_balancing_type = None
            router_topk = None
            extend_ep_across_dp = None
            moe_routing_distribution_type = get_field_value(
                "moe_routing_distribution_type"
            )
            attn_tensor_parallel_size = get_field_value("attn_tensor_parallel_size")
            attn_data_parallel_size = get_field_value("attn_data_parallel_size")
        else:
            attn_tensor_parallel_size = get_field_value("attn_tensor_parallel_size")
            attn_data_parallel_size = get_field_value("attn_data_parallel_size")
            moe_tensor_parallel_size = get_field_value("moe_tensor_parallel_size")
            moe_expert_parallel_size = get_field_value("moe_expert_parallel_size")
            total_expert_num = get_field_value("total_expert_num")
            local_expert_num = get_field_value("local_expert_num")
            router_load_balancing_type = get_field_value("router_load_balancing_type")
            router_topk = get_field_value("router_topk")
            extend_ep_across_dp = get_field_value("extend_ep_across_dp")
            moe_routing_distribution_type = get_field_value(
                "moe_routing_distribution_type"
            )
            if cluster_prefix == "decode_ffn":
                num_pipeline_stages = 1
            else:
                num_pipeline_stages = get_field_value("num_pipeline_stages")

        return ReplicaConfig(
            model_name=get_field_value("model_name"),
            memory_margin_fraction=get_field_value("memory_margin_fraction"),
            num_pipeline_stages=num_pipeline_stages,
            attn_tensor_parallel_size=attn_tensor_parallel_size,
            attn_data_parallel_size=attn_data_parallel_size,
            moe_tensor_parallel_size=moe_tensor_parallel_size,
            moe_expert_parallel_size=moe_expert_parallel_size,
            total_expert_num=total_expert_num,
            local_expert_num=local_expert_num,
            router_load_balancing_type=router_load_balancing_type,
            router_topk=router_topk,
            moe_routing_mode=get_field_value("moe_routing_mode"),
            moe_routing_seed=get_field_value("moe_routing_seed"),
            moe_routing_distribution_type=moe_routing_distribution_type,
            extend_ep_across_dp=extend_ep_across_dp,
            device=get_field_value("device"),
            network_device=get_field_value("network_device"),
            cluster_prefix=cluster_prefix,
            data_parallel_size=attn_data_parallel_size,  # Set data_parallel_size for DP replica support
            speculative_decoding_config=main_config.speculative_decoding_config,
        )

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

        # Validate dense model configuration in PD-disaggregation mode
        is_dense_model = replica_config.total_expert_num == 1
        if is_dense_model and cluster_name in ["prefill", "decode"]:
            # For dense models in PD-disaggregation mode, enforce attn_data_parallel_size = 1
            if replica_config.attn_data_parallel_size != 1:
                raise ValueError(
                    f"Dense models in PD-disaggregation mode require attn_data_parallel_size=1 "
                    f"in {cluster_name} cluster, got {replica_config.attn_data_parallel_size}. "
                    f"Dense models do not support attn data parallelism in disaggregated mode."
                )
            # Ensure MoE parallelism is disabled for dense models
            if replica_config.moe_expert_parallel_size != 1:
                raise ValueError(
                    f"Dense models require moe_expert_parallel_size=1 in {cluster_name} cluster, "
                    f"got {replica_config.moe_expert_parallel_size}. "
                    f"Dense models do not have expert parallelism."
                )

        if cluster_name in {"prefill", "decode", "monolithic"} and replica_config.model_config.is_moe:
            validate_frontier_shared_parallel_domains(
                FrontierParallelismMapping(
                    cluster_num_replicas=1,
                    attn_tensor_parallel_size=replica_config.attn_tensor_parallel_size,
                    attn_data_parallel_size=replica_config.attn_data_parallel_size,
                    moe_tensor_parallel_size=replica_config.moe_tensor_parallel_size,
                    moe_expert_parallel_size=replica_config.moe_expert_parallel_size,
                )
            )

        if cluster_name != "decode_attn":
            pass
            # if replica_config.extend_ep_across_dp:
            #     # Assuming data parallelism for MoE is attn_data_parallel_size
            #     assert replica_config.moe_expert_parallel_size == replica_config.attn_data_parallel_size * \
            #         replica_config.moe_tensor_parallel_size, f"For global MoE in {cluster_name} cluster, moe_expert_parallel_size must be equal to attn_data_parallel_size * moe_tensor_parallel_size"
            # else:
            #     assert replica_config.moe_expert_parallel_size == replica_config.moe_tensor_parallel_size, f"For local MoE in {cluster_name} cluster, moe_expert_parallel_size must be equal to moe_tensor_parallel_size"
        else:
            assert (
                replica_config.moe_expert_parallel_size == 0
                and replica_config.local_expert_num == 0
            ), "For decode attention cluster, moe_expert_parallel_size and local_expert_num must be 0"

    def _collect_cluster_info(self) -> List[Tuple[str, int, ReplicaConfig]]:
        clusters_info = []

        if self._has_disaggregation_params_set():
            if self.prefill_cluster_num_replicas and self.prefill_replica_config:
                clusters_info.append(
                    (
                        "PREFILL",
                        self.prefill_cluster_num_replicas,
                        self.prefill_replica_config,
                    )
                )

            if (
                self.decode_attn_cluster_num_replicas
                and self.decode_attn_replica_config
            ):
                clusters_info.append(
                    (
                        "DECODE_ATTN",
                        self.decode_attn_cluster_num_replicas,
                        self.decode_attn_replica_config,
                    )
                )

            if self.decode_ffn_cluster_num_replicas and self.decode_ffn_replica_config:
                clusters_info.append(
                    (
                        "DECODE_FFN",
                        self.decode_ffn_cluster_num_replicas,
                        self.decode_ffn_replica_config,
                    )
                )

            if self.decode_cluster_num_replicas and self.decode_replica_config:
                clusters_info.append(
                    (
                        "DECODE",
                        self.decode_cluster_num_replicas,
                        self.decode_replica_config,
                    )
                )
        else:
            clusters_info.append(("MONOLITHIC", self.num_replicas, self.replica_config))

        return clusters_info

    def get_server_count_metadata(self, sys_arch: str) -> Dict[str, int]:
        clusters_info = self._collect_cluster_info()
        server_counts_by_cluster = {}
        _, _, _, CollectiveSimCCBackendConfig, _, _ = _get_cc_backend_configs()
        cluster_prefix_by_name = {
            "PREFILL": "prefill",
            "DECODE_ATTN": "decode_attn",
            "DECODE_FFN": "decode_ffn",
            "DECODE": "decode",
        }

        for cluster_name, num_replicas, replica_config in clusters_info:
            cluster_total_devices = int(num_replicas) * int(replica_config.world_size)
            num_devices_per_node = int(replica_config.node_config.num_devices_per_node)
            if cluster_total_devices <= 0:
                raise ValueError(
                    "cluster_total_devices must be positive when computing "
                    f"server-count metadata, got {cluster_total_devices}"
                )
            if num_devices_per_node <= 0:
                raise ValueError(
                    "num_devices_per_node must be positive when computing "
                    f"server-count metadata, got {num_devices_per_node}"
                )
            if cluster_name == "MONOLITHIC":
                cc_backend_config = self.cc_backend_config
            else:
                cc_backend_config = self._create_cc_backend_config_for_cluster(
                    cluster_prefix_by_name[cluster_name]
                )
            if isinstance(cc_backend_config, CollectiveSimCCBackendConfig):
                physical_topology = resolve_collective_sim_physical_topology(
                    cluster_total_devices=cluster_total_devices,
                    num_devices_per_node=num_devices_per_node,
                    scenario_profile=getattr(
                        cc_backend_config,
                        "scenario_profile",
                        None,
                    ),
                )
                server_counts_by_cluster[cluster_name] = int(physical_topology.servers)
                continue
            server_counts_by_cluster[cluster_name] = (
                cluster_total_devices + num_devices_per_node - 1
            ) // num_devices_per_node

        if sys_arch == "co-location":
            if "MONOLITHIC" not in server_counts_by_cluster:
                raise ValueError("Missing MONOLITHIC cluster for co-location mode.")
            return {"server_count": server_counts_by_cluster["MONOLITHIC"]}
        if sys_arch == "pd-disaggregation":
            if "PREFILL" not in server_counts_by_cluster or "DECODE" not in server_counts_by_cluster:
                raise ValueError(
                    "Missing PREFILL or DECODE cluster for pd-disaggregation mode."
                )
            return {
                "prefill_server_count": server_counts_by_cluster["PREFILL"],
                "decode_server_count": server_counts_by_cluster["DECODE"],
            }
        if sys_arch == "pd-af-disaggregation":
            required = ["PREFILL", "DECODE_ATTN", "DECODE_FFN"]
            if any(name not in server_counts_by_cluster for name in required):
                raise ValueError(
                    "Missing PREFILL, DECODE_ATTN, or DECODE_FFN cluster for pd-af-disaggregation mode."
                )
            return {
                "prefill_server_count": server_counts_by_cluster["PREFILL"],
                "decode_attn_server_count": server_counts_by_cluster["DECODE_ATTN"],
                "decode_ffn_server_count": server_counts_by_cluster["DECODE_FFN"],
            }

        raise ValueError(f"Unknown system architecture: {sys_arch}")

    def print_cluster_statistics(self, simulation_mode: str, sys_arch: str):
        """Calculate and print statistics for all clusters (called from SimulationConfig)."""
        clusters_info = self._collect_cluster_info()

        # Calculate total statistics
        self.total_clusters = len(clusters_info)
        self.cluster_world_sizes = {}

        # Calculate world_size if not already set
        if not hasattr(self, "world_size") or self.world_size is None:
            self.world_size = sum(
                num_replicas * replica_config.world_size
                for _, num_replicas, replica_config in clusters_info
            )

        # Print cluster configuration summary
        print("\n" + "=" * 70)
        print("CLUSTER CONFIGURATION SUMMARY")
        print("=" * 70)
        print("Simulation mode: ", simulation_mode)
        print("System architecture: ", sys_arch)
        print("=" * 70)
        print(f"Total Clusters: {self.total_clusters}")
        print(f"Total World Size: {self.world_size}")
        server_count_metadata = self.get_server_count_metadata(sys_arch)
        if sys_arch == "co-location":
            print(f"Server count: {server_count_metadata['server_count']}")
        elif sys_arch == "pd-disaggregation":
            print(
                f"Prefill server count: {server_count_metadata['prefill_server_count']}"
            )
            print(
                f"Decode server count: {server_count_metadata['decode_server_count']}"
            )
        elif sys_arch == "pd-af-disaggregation":
            print(
                f"Prefill server count: {server_count_metadata['prefill_server_count']}"
            )
            print(
                f"Decode-Attn server count: {server_count_metadata['decode_attn_server_count']}"
            )
            print(
                f"Decode-FFN server count: {server_count_metadata['decode_ffn_server_count']}"
            )
        print()

        for cluster_name, num_replicas, replica_config in clusters_info:
            cluster_world_size = num_replicas * replica_config.world_size
            self.cluster_world_sizes[cluster_name] = cluster_world_size

            print(f"Cluster Type: {cluster_name}")
            print(f"   Cluster World Size: {cluster_world_size}")
            print(f"   Num Replicas (Instances): {num_replicas}")
            print(f"   Replica World Size: {replica_config.world_size}")
            if (
                cluster_name == "PREFILL"
                or cluster_name == "MONOLITHIC"
                or cluster_name == "DECODE"
            ):
                print(
                    f"   Configuration: PP{replica_config.num_pipeline_stages} × (Attn_TP{replica_config.attn_tensor_parallel_size} x Attn_DP{replica_config.attn_data_parallel_size}) | (MoE_TP{replica_config.moe_tensor_parallel_size} x MoE_EP{replica_config.moe_expert_parallel_size})"
                )
                print(f"   Total Expert Num: {replica_config.total_expert_num}")
                print(f"   Local Expert Num: {replica_config.local_expert_num}")
            elif cluster_name == "DECODE_ATTN":
                print(
                    f"   Configuration: PP{replica_config.num_pipeline_stages} × Attn_TP{replica_config.attn_tensor_parallel_size} x Attn_DP{replica_config.attn_data_parallel_size}"
                )
            elif cluster_name == "DECODE_FFN":
                print(
                    f"   Configuration: PP{replica_config.num_pipeline_stages} × MoE_TP{replica_config.moe_tensor_parallel_size} x MoE_EP{replica_config.moe_expert_parallel_size}"
                )
                print(f"   Total Expert Num: {replica_config.total_expert_num}")
                print(f"   Local Expert Num: {replica_config.local_expert_num}")

            print("-" * 50)

        print("=" * 70 + "\n")

    def get_cluster_configs_for_disaggregation(
        self,
    ) -> Dict[ClusterType, "ClusterConfig"]:
        """Generate cluster configurations for disaggregated mode."""
        if not self._has_disaggregation_params_set():
            return {ClusterType.MONOLITHIC: self}

        cluster_configs = {}

        # Prefill cluster
        if self.prefill_cluster_num_replicas:
            prefill_config = ClusterConfig(
                cluster_type=ClusterType.PREFILL,
                num_replicas=self.prefill_cluster_num_replicas,
                replica_config=self.prefill_replica_config
                or self._create_replica_config_copy(),
                cluster_scheduler_config=self.cluster_scheduler_config,
                replica_scheduler_config=self.replica_scheduler_config,
                execution_time_predictor_config=(
                    self._create_execution_time_predictor_config_for_cluster(
                        "prefill"
                    )
                ),
                cc_backend_config=self._create_cc_backend_config_for_cluster("prefill"),
                # Propagate cluster-specific replica scheduler config parameters
                prefill_replica_scheduler_config_type=self.prefill_replica_scheduler_config_type,
                prefill_replica_scheduler_config_batch_size_cap=self.prefill_replica_scheduler_config_batch_size_cap,
                prefill_replica_scheduler_config_max_tokens_in_batch=self.prefill_replica_scheduler_config_max_tokens_in_batch,
                prefill_replica_scheduler_config_num_blocks=self.prefill_replica_scheduler_config_num_blocks,
                prefill_replica_scheduler_config_block_size=self.prefill_replica_scheduler_config_block_size,
                prefill_replica_scheduler_config_watermark_blocks_fraction=self.prefill_replica_scheduler_config_watermark_blocks_fraction,
            )
            cluster_configs[ClusterType.PREFILL] = prefill_config

        # Decode Attention cluster
        if self.decode_attn_cluster_num_replicas:
            # Determine micro-batch SIZE for decode-attn
            # NOTE: Only decode_attn_micro_batch_size exists (no generic fallback)
            _da_mbs = self.decode_attn_micro_batch_size
            decode_attn_config = ClusterConfig(
                cluster_type=ClusterType.DECODE_ATTN,
                num_replicas=self.decode_attn_cluster_num_replicas,
                replica_config=self.decode_attn_replica_config
                or self._create_replica_config_copy(),
                cluster_scheduler_config=self.cluster_scheduler_config,
                replica_scheduler_config=self.replica_scheduler_config,
                execution_time_predictor_config=(
                    self._create_execution_time_predictor_config_for_cluster(
                        "decode_attn"
                    )
                ),
                cc_backend_config=self._create_cc_backend_config_for_cluster(
                    "decode_attn"
                ),
                af_pipeline_num_micro_batch=self.decode_attn_af_pipeline_num_micro_batch,
                decode_attn_micro_batch_size=_da_mbs,
                decode_attn_request_allocation_threshold=self.decode_attn_request_allocation_threshold,
                # Propagate cluster-specific replica scheduler config parameters
                decode_attn_replica_scheduler_config_type=self.decode_attn_replica_scheduler_config_type,
                decode_attn_replica_scheduler_config_batch_size_cap=self.decode_attn_replica_scheduler_config_batch_size_cap,
                decode_attn_replica_scheduler_config_max_tokens_in_batch=self.decode_attn_replica_scheduler_config_max_tokens_in_batch,
                decode_attn_replica_scheduler_config_num_blocks=self.decode_attn_replica_scheduler_config_num_blocks,
                decode_attn_replica_scheduler_config_block_size=self.decode_attn_replica_scheduler_config_block_size,
                decode_attn_replica_scheduler_config_watermark_blocks_fraction=self.decode_attn_replica_scheduler_config_watermark_blocks_fraction,
            )
            cluster_configs[ClusterType.DECODE_ATTN] = decode_attn_config

        # Decode FFN cluster
        if self.decode_ffn_cluster_num_replicas:
            decode_ffn_config = ClusterConfig(
                cluster_type=ClusterType.DECODE_FFN,
                num_replicas=self.decode_ffn_cluster_num_replicas,
                replica_config=self.decode_ffn_replica_config
                or self._create_replica_config_copy(),
                cluster_scheduler_config=self.cluster_scheduler_config,
                replica_scheduler_config=self.replica_scheduler_config,
                execution_time_predictor_config=(
                    self._create_execution_time_predictor_config_for_cluster(
                        "decode_ffn"
                    )
                ),
                cc_backend_config=self._create_cc_backend_config_for_cluster(
                    "decode_ffn"
                ),
                af_pipeline_num_micro_batch=self.decode_ffn_af_pipeline_num_micro_batch,
                # Propagate cluster-specific replica scheduler config parameters
                decode_ffn_replica_scheduler_config_type=self.decode_ffn_replica_scheduler_config_type,
                decode_ffn_replica_scheduler_config_batch_size_cap=self.decode_ffn_replica_scheduler_config_batch_size_cap,
                decode_ffn_replica_scheduler_config_max_tokens_in_batch=self.decode_ffn_replica_scheduler_config_max_tokens_in_batch,
                decode_ffn_replica_scheduler_config_num_blocks=self.decode_ffn_replica_scheduler_config_num_blocks,
                decode_ffn_replica_scheduler_config_block_size=self.decode_ffn_replica_scheduler_config_block_size,
                decode_ffn_replica_scheduler_config_watermark_blocks_fraction=self.decode_ffn_replica_scheduler_config_watermark_blocks_fraction,
                decode_attn_cluster_num_replicas=self.decode_attn_cluster_num_replicas,
                decode_attn_replica_config_attn_data_parallel_size=(
                    self.decode_attn_replica_config.attn_data_parallel_size
                    if self.decode_attn_replica_config is not None
                    else None
                ),
                allow_experiment_multi_decode_ffn_replicas=(
                    self.allow_experiment_multi_decode_ffn_replicas
                ),
            )
            # Propagate DECODE_ATTN DP lanes for FFN grouping: dp_lanes = attn_num_replicas * attn_dp_size
            attn_num_replicas = int(self.decode_attn_cluster_num_replicas)
            attn_dp_size = int(self.decode_attn_replica_config.attn_data_parallel_size)
            decode_ffn_config.decode_attn_dp_lanes_for_ffn = max(
                1, attn_num_replicas * attn_dp_size
            )
            decode_ffn_config.decode_attn_replica_id_start_for_ffn = int(
                self.prefill_cluster_num_replicas
            )

            cluster_configs[ClusterType.DECODE_FFN] = decode_ffn_config

        # Unified Decode cluster (PD-disaggregation mode)
        if self.decode_cluster_num_replicas:
            decode_config = ClusterConfig(
                cluster_type=ClusterType.DECODE,
                num_replicas=self.decode_cluster_num_replicas,
                replica_config=self.decode_replica_config
                or self._create_replica_config_copy(),
                cluster_scheduler_config=self.cluster_scheduler_config,
                replica_scheduler_config=self.replica_scheduler_config,
                execution_time_predictor_config=(
                    self._create_execution_time_predictor_config_for_cluster("decode")
                ),
                cc_backend_config=self._create_cc_backend_config_for_cluster("decode"),
                # Propagate cluster-specific replica scheduler config parameters
                decode_replica_scheduler_config_type=self.decode_replica_scheduler_config_type,
                decode_replica_scheduler_config_batch_size_cap=self.decode_replica_scheduler_config_batch_size_cap,
                decode_replica_scheduler_config_max_tokens_in_batch=self.decode_replica_scheduler_config_max_tokens_in_batch,
                decode_replica_scheduler_config_num_blocks=self.decode_replica_scheduler_config_num_blocks,
                decode_replica_scheduler_config_block_size=self.decode_replica_scheduler_config_block_size,
                decode_replica_scheduler_config_watermark_blocks_fraction=self.decode_replica_scheduler_config_watermark_blocks_fraction,
            )
            cluster_configs[ClusterType.DECODE] = decode_config

        return cluster_configs

    def _create_execution_time_predictor_config_for_cluster(
        self, cluster_prefix: str
    ) -> BaseExecutionTimePredictorConfig:
        """Create cluster-specific execution-time predictor config overrides."""
        base_config = self.execution_time_predictor_config
        override_values = {}
        for calibration_field in (
            "attn_pre_proj_calibration_scale",
            "attn_post_proj_calibration_scale",
            "attn_decode_calibration_scale",
            "attn_kv_cache_save_calibration_scale",
            "mlp_up_proj_calibration_scale",
            "mlp_down_proj_calibration_scale",
            "decode_phase_mlp_down_proj_calibration_scale",
            "moe_shuffling_calibration_scale",
            "moe_grouped_gemm_calibration_scale",
            "expert_parallel_communication_calibration_scale",
        ):
            override_field = (
                f"{cluster_prefix}_execution_time_predictor_config_"
                f"{calibration_field}"
            )
            override_value = getattr(self, override_field, None)
            if override_value is None:
                continue
            override_value = float(override_value)
            if override_value <= 0.0:
                raise ValueError(
                    f"ClusterConfig.{override_field} must be > 0, got={override_value!r}"
                )
            override_values[calibration_field] = override_value

        if not override_values:
            return base_config

        return replace(base_config, **override_values)

    def _create_cc_backend_config_for_cluster(
        self, cluster_prefix: str
    ) -> BaseCCBackendConfig:
        """
        Create CC backend configuration for a specific cluster.

        This method creates a cluster-specific CC backend configuration by:
        1. Checking for cluster-specific override values
        2. Falling back to the base cc_backend_config values if not overridden

        Args:
            cluster_prefix: Cluster prefix (e.g., "prefill", "decode", "decode_attn", "decode_ffn")

        Returns:
            CC backend configuration for the specified cluster
        """
        # Lazy import to avoid circular imports
        (
            _,
            VidurCCBackendConfig,
            AnalyticalCCBackendConfig,
            CollectiveSimCCBackendConfig,
            AiconfiguratorCCBackendConfig,
            AstraSimAnalyticalCCBackendConfig,
        ) = _get_cc_backend_configs()

        # Get cluster-specific type override
        type_field = f"{cluster_prefix}_cc_backend_config_type"
        cluster_type_str = getattr(self, type_field, None)

        # Determine which config type to use
        if cluster_type_str is not None:
            # Use cluster-specific type
            cluster_type_key = cluster_type_str.lower()
            if cluster_type_key == "analytical":
                return self._create_analytical_cc_backend_config(cluster_prefix)
            elif cluster_type_key == "vidur":
                return self._create_vidur_cc_backend_config(cluster_prefix)
            elif cluster_type_key == "collective_sim":
                return self._create_collective_sim_cc_backend_config(cluster_prefix)
            elif cluster_type_key == "aiconfigurator":
                return self._create_aiconfigurator_cc_backend_config(cluster_prefix)
            elif cluster_type_key == "astra_sim_analytical":
                return self._create_astra_sim_analytical_cc_backend_config(
                    cluster_prefix
                )
            else:
                raise ValueError(f"Unknown CC backend type: {cluster_type_str}")
        else:
            # Use base config type
            base_config = self.cc_backend_config
            if isinstance(base_config, AnalyticalCCBackendConfig):
                return self._create_analytical_cc_backend_config(cluster_prefix)
            elif isinstance(base_config, VidurCCBackendConfig):
                return self._create_vidur_cc_backend_config(cluster_prefix)
            elif isinstance(base_config, CollectiveSimCCBackendConfig):
                return self._create_collective_sim_cc_backend_config(cluster_prefix)
            elif isinstance(base_config, AiconfiguratorCCBackendConfig):
                return self._create_aiconfigurator_cc_backend_config(cluster_prefix)
            elif isinstance(base_config, AstraSimAnalyticalCCBackendConfig):
                return self._create_astra_sim_analytical_cc_backend_config(
                    cluster_prefix
                )
            else:
                raise ValueError(
                    "Unsupported base CC backend config type for cluster-specific "
                    f"construction: {type(base_config).__name__}"
                )

    def _create_analytical_cc_backend_config(
        self, cluster_prefix: str
    ) -> AnalyticalCCBackendConfig:
        """Create analytical CC backend config with cluster-specific overrides."""
        # Lazy import to avoid circular imports
        _, _, AnalyticalCCBackendConfig, _, _, _ = _get_cc_backend_configs()

        base_config = self.cc_backend_config

        # Get cluster-specific values with fallback to base config
        def get_value(field_name: str, default_value):
            cluster_field = f"{cluster_prefix}_cc_backend_config_{field_name}"
            cluster_value = getattr(self, cluster_field, None)
            if cluster_value is not None:
                return cluster_value
            if isinstance(base_config, AnalyticalCCBackendConfig):
                return getattr(base_config, field_name, default_value)
            return default_value

        return AnalyticalCCBackendConfig(
            profiling_data_dir=(
                base_config.profiling_data_dir
                if hasattr(base_config, "profiling_data_dir")
                else "data/profiling/network"
            ),
            cache_dir=(
                base_config.cache_dir if hasattr(base_config, "cache_dir") else "cache"
            ),
            no_cache=(
                base_config.no_cache if hasattr(base_config, "no_cache") else False
            ),
            network_bandwidth_gbps=get_value("network_bandwidth_gbps", 100.0),
            network_latency_us=get_value("network_latency_us", 1.0),
            intra_node_bandwidth_gbps=get_value("intra_node_bandwidth_gbps", 600.0),
        )

    def _create_vidur_cc_backend_config(
        self, cluster_prefix: str
    ) -> VidurCCBackendConfig:
        """Create Vidur CC backend config with cluster-specific overrides."""
        # Lazy import to avoid circular imports
        _, VidurCCBackendConfig, _, _, _, _ = _get_cc_backend_configs()

        base_config = self.cc_backend_config

        # For Vidur config, we mainly use the base config values
        # as Vidur-specific parameters are typically shared across clusters
        if isinstance(base_config, VidurCCBackendConfig):
            return VidurCCBackendConfig(
                profiling_data_dir=base_config.profiling_data_dir,
                cache_dir=base_config.cache_dir,
                no_cache=base_config.no_cache,
                all_reduce_input_file=base_config.all_reduce_input_file,
                send_recv_input_file=base_config.send_recv_input_file,
                k_fold_cv_splits=base_config.k_fold_cv_splits,
                num_training_job_threads=base_config.num_training_job_threads,
            )
        else:
            # Create default Vidur config
            return VidurCCBackendConfig()

    def _create_collective_sim_cc_backend_config(
        self, cluster_prefix: str
    ) -> "CollectiveSimCCBackendConfig":
        """Create collective-sim CC backend config with cluster-specific overrides."""
        (
            _,
            _,
            _,
            CollectiveSimCCBackendConfig,
            _,
            _,
        ) = _get_cc_backend_configs()

        from dataclasses import replace
        from pathlib import Path

        base_config = self.cc_backend_config
        if not isinstance(base_config, CollectiveSimCCBackendConfig):
            base_config = CollectiveSimCCBackendConfig()

        def get_value(field_name: str, default_value):
            cluster_field = f"{cluster_prefix}_cc_backend_config_{field_name}"
            cluster_value = getattr(self, cluster_field, None)
            if cluster_value is not None:
                return cluster_value
            return getattr(base_config, field_name, default_value)

        if base_config.runner_out_dir:
            cluster_out_dir = str(Path(base_config.runner_out_dir) / cluster_prefix)
            return replace(
                base_config,
                runner_out_dir=cluster_out_dir,
                nvlink_allreduce_launch_overhead_us=get_value(
                    "nvlink_allreduce_launch_overhead_us",
                    50.0,
                ),
            )

        return replace(
            base_config,
            nvlink_allreduce_launch_overhead_us=get_value(
                "nvlink_allreduce_launch_overhead_us",
                50.0,
            ),
        )

    def _create_aiconfigurator_cc_backend_config(
        self, cluster_prefix: str
    ) -> "AiconfiguratorCCBackendConfig":
        """Create aiconfigurator CC backend config with cluster-specific overrides."""
        (
            _,
            _,
            _,
            _,
            AiconfiguratorCCBackendConfig,
            _,
        ) = _get_cc_backend_configs()

        base_config = self.cc_backend_config

        def get_value(field_name: str, default_value):
            cluster_field = f"{cluster_prefix}_cc_backend_config_{field_name}"
            cluster_value = getattr(self, cluster_field, None)
            if cluster_value is not None:
                return cluster_value
            if isinstance(base_config, AiconfiguratorCCBackendConfig):
                return getattr(base_config, field_name, default_value)
            return default_value

        return AiconfiguratorCCBackendConfig(
            profiling_data_dir=(
                base_config.profiling_data_dir
                if hasattr(base_config, "profiling_data_dir")
                else "data/profiling/network"
            ),
            cache_dir=(
                base_config.cache_dir if hasattr(base_config, "cache_dir") else "cache"
            ),
            no_cache=(
                base_config.no_cache if hasattr(base_config, "no_cache") else False
            ),
            repo_root=get_value("repo_root", "sota-infer-engine/aiconfigurator"),
            system=get_value("system", ""),
            source_backend=get_value("source_backend", "vllm"),
            source_version=get_value("source_version", ""),
            database_mode=get_value("database_mode", "silicon"),
            tp_allreduce_impl=get_value("tp_allreduce_impl", "custom_allreduce"),
            custom_allreduce_variant=get_value("custom_allreduce_variant", None),
        )

    def _create_astra_sim_analytical_cc_backend_config(
        self, cluster_prefix: str
    ) -> "AstraSimAnalyticalCCBackendConfig":
        """Create astra-sim analytical CC backend config with cluster-specific overrides."""
        (
            _,
            _,
            _,
            _,
            _,
            AstraSimAnalyticalCCBackendConfig,
        ) = _get_cc_backend_configs()

        base_config = self.cc_backend_config

        def get_value(field_name: str, default_value):
            cluster_field = f"{cluster_prefix}_cc_backend_config_{field_name}"
            cluster_value = getattr(self, cluster_field, None)
            if cluster_value is not None:
                return cluster_value
            if isinstance(base_config, AstraSimAnalyticalCCBackendConfig):
                return getattr(base_config, field_name, default_value)
            return default_value

        return AstraSimAnalyticalCCBackendConfig(
            profiling_data_dir=(
                base_config.profiling_data_dir
                if hasattr(base_config, "profiling_data_dir")
                else "data/profiling/network"
            ),
            cache_dir=(
                base_config.cache_dir if hasattr(base_config, "cache_dir") else "cache"
            ),
            no_cache=(
                base_config.no_cache if hasattr(base_config, "no_cache") else False
            ),
            prediction_cache_size=get_value("prediction_cache_size", 4096),
            placement_order=get_value("placement_order", "TP,CP,DP,EP"),
            intra_server_topology=get_value(
                "intra_server_topology", "FullyConnected"
            ),
            inter_server_topology=get_value(
                "inter_server_topology", "FullyConnected"
            ),
            intra_server_bandwidth_gbps=get_value(
                "intra_server_bandwidth_gbps", 600.0
            ),
            intra_server_latency_us=get_value("intra_server_latency_us", 1.0),
            inter_server_bandwidth_gbps=get_value(
                "inter_server_bandwidth_gbps", 100.0
            ),
            inter_server_latency_us=get_value("inter_server_latency_us", 1.0),
            ring_bidirectional=(
                base_config.ring_bidirectional
                if isinstance(base_config, AstraSimAnalyticalCCBackendConfig)
                else True
            ),
            p2p_src_index=get_value("p2p_src_index", 0),
            p2p_dst_index=get_value("p2p_dst_index", 1),
        )

    def _create_replica_config_copy(self) -> ReplicaConfig:
        """Create a copy of the main replica config for disaggregated clusters."""
        # Note: This method now needs to be called before replica_config is cleared
        # We need to preserve the original config temporarily
        original_config = (
            self.replica_config if self.replica_config else ReplicaConfig()
        )

        return ReplicaConfig(
            model_name=original_config.model_name,
            memory_margin_fraction=original_config.memory_margin_fraction,
            num_pipeline_stages=original_config.num_pipeline_stages,
            attn_tensor_parallel_size=original_config.attn_tensor_parallel_size,
            attn_data_parallel_size=original_config.attn_data_parallel_size,
            moe_tensor_parallel_size=original_config.moe_tensor_parallel_size,
            moe_expert_parallel_size=original_config.moe_expert_parallel_size,
            total_expert_num=original_config.total_expert_num,
            data_parallel_size=original_config.data_parallel_size,
            router_load_balancing_type=original_config.router_load_balancing_type,
            router_topk=original_config.router_topk,
            extend_ep_across_dp=original_config.extend_ep_across_dp,
            device=original_config.device,
            network_device=original_config.network_device,
            speculative_decoding_config=original_config.speculative_decoding_config,
        )


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
            "help": "System architecture type. 'co-location' for baseline, 'pd-disaggregation' for PD-only disaggregation, 'pd-af-disaggregation' for PD+AF disaggregation.",
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
    enable_monolithic_moe_stage_aggregation: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable MONOLITHIC MoE accelerated stage-level execution. "
                "When true, co-location MoE prefill/decode uses aggregated "
                "stage timing instead of layer-by-layer sync events."
            ),
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
            "help": "Enable parallel processing of clusters in disaggregated mode."
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
        self._validate_monolithic_moe_stage_aggregation_config()

        global_vars.set_global_vars(self.simulation_mode, self.sys_arch)
        global_vars.set_monolithic_moe_stage_aggregation(
            self.enable_monolithic_moe_stage_aggregation
        )
        self._validate_cuda_graph_config()
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

        # PD+AF disaggregation requires MoE-enabled models; dense models are incompatible
        if self.sys_arch == "pd-af-disaggregation":
            model_config = None
            for rc in [
                getattr(self.cluster_config, "prefill_replica_config", None),
                getattr(self.cluster_config, "decode_attn_replica_config", None),
                getattr(self.cluster_config, "decode_ffn_replica_config", None),
            ]:
                if rc is not None:
                    model_config = rc.model_config
                    break

            if model_config is not None and not model_config.is_moe:
                model_name = model_config.get_name()
                raise ValueError(
                    "System architecture 'pd-af-disaggregation' requires MoE-enabled models but "
                    f"'{model_name}' is a dense model. Please use '--sys_arch pd-disaggregation' for dense models, "
                    "or switch to an MoE model (e.g., Mixtral-8x7B, Qwen-72B-MoE)."
                )

        # Set default decode_attn_request_allocation_threshold for offline mode
        if (
            self.simulation_mode == "offline"
            and self.sys_arch == "pd-af-disaggregation"
            and self.cluster_config.decode_attn_request_allocation_threshold is None
        ):
            # Default to total number of requests in offline mode
            if (
                hasattr(self.request_generator_config, "num_requests")
                and self.request_generator_config.num_requests is not None
            ):
                self.cluster_config.decode_attn_request_allocation_threshold = (
                    self.request_generator_config.num_requests
                )
                logger.info(
                    f"Setting default decode_attn_request_allocation_threshold to {self.request_generator_config.num_requests} "
                    f"(total number of requests in offline mode)"
                )

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

    def _validate_open_source_release_architecture_guard(self) -> None:
        if self.sys_arch in {"pd-disaggregation", "pd-af-disaggregation"}:
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        if getattr(self, "use_cuda_graph", False):
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        default_kv_cache_transfer_config = AnalyticalKVCacheTransferConfig()
        if (
            getattr(
                self,
                "kv_cache_transfer_config",
                default_kv_cache_transfer_config,
            )
            != default_kv_cache_transfer_config
        ):
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)
        default_m2n_transfer_config = AnalyticalM2NTransferConfig()
        if (
            getattr(self, "m2n_transfer_config", default_m2n_transfer_config)
            != default_m2n_transfer_config
        ):
            raise ValueError(DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR)

    def _validate_simulation_mode_arch_compatibility(self) -> None:
        """
        Validate supported simulation-mode/system-architecture combinations.

        Online mode currently supports co-location and PD-disaggregation only.
        """
        if (
            self.simulation_mode == "online"
            and self.sys_arch == "pd-af-disaggregation"
        ):
            raise ValueError(
                "Unsupported combination: simulation_mode='online' with "
                "sys_arch='pd-af-disaggregation'. Online mode is currently "
                "supported only for 'co-location' and 'pd-disaggregation'. "
                "Please use simulation_mode='offline' for "
                "'pd-af-disaggregation'."
            )

    def _validate_monolithic_moe_stage_aggregation_config(self) -> None:
        if (
            self.enable_monolithic_moe_stage_aggregation
            and self.sys_arch != "co-location"
        ):
            raise ValueError(
                "enable_monolithic_moe_stage_aggregation is supported only for "
                "sys_arch='co-location'."
            )

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
