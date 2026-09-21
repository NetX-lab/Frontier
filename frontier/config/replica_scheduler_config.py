"""Replica scheduler configuration for every supported batching policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from frontier.config.base_poly_config import BasePolyConfig
from frontier.types import ReplicaSchedulerType


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
