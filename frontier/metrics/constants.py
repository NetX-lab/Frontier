""" File to store names for different metrics captured """

import enum


class OperationMetrics(enum.Enum):
    MLP_UP_PROJ = "mlp_up_proj"
    MLP_ACTIVATION = "mlp_activation"
    MLP_DOWN_PROJ = "mlp_down_proj"
    MLP_DOWN_PROJ_ALL_REDUCE = "mlp_down_proj_all_reduce"
    ATTN_PRE_PROJ = "attn_pre_proj"
    ATTN_POST_PROJ = "attn_post_proj"
    ATTN_POST_PROJ_ALL_REDUCE = "attn_post_proj_all_reduce"
    ATTN_PREFILL = "attn_prefill"
    ATTN_KV_CACHE_SAVE = "attn_kv_cache_save"
    ATTN_DECODE = "attn_decode"
    ATTN_ROPE = "attn_rope"
    PIPELINE_SEND_RECV = "pipeline_send_recv"
    ADD = "add"
    INPUT_LAYERNORM = "input_layernorm"
    POST_ATTENTION_LAYERNORM = "post_attention_layernorm"
    ATTN_TENSOR_PARALLEL_ALLREDUCE = "attn_tensor_parallel_allreduce"
    MOE_TENSOR_PARALLEL_ALLGATHER = "moe_tensor_parallel_allgather"
    SHARE_EXPERT_UP_PROJ = "share_expert_up_proj"
    SHARE_EXPERT_ACT = "share_expert_act"
    SHARE_EXPERT_DOWN_PROJ = "share_expert_down_proj"
    SHARE_EXPERT_TENSOR_PARALLEL_ALLREDUCE = "share_expert_tensor_parallel_allreduce"
    EXPERT_PARALLEL_ALLREDUCE = "expert_parallel_allreduce"
    MOE_GATING = "moe_gating"
    MOE_GATING_LINEAR = "moe_gating_linear"
    MOE_GATING_ROUTING_TOPK = "moe_gating_routing_topk"
    MOE_SHUFFLING = "moe_shuffling"
    EXPERT_PARALLEL_ALLTOALL_DISPATCH = "expert_parallel_alltoall_dispatch"
    EXPERT_PARALLEL_ALLTOALL_COMBINE = "expert_parallel_alltoall_combine"
    MOE_GROUPED_GEMM = "moe_grouped_gemm"
    MOE_TENSOR_PARALLEL_ALLREDUCE = "moe_tensor_parallel_allreduce"
    ADD_ATTN_RESIDUAL = "add_attn_residual"
    ADD_FFN_RESIDUAL = "add_ffn_residual"


class CpuOperationMetrics(enum.Enum):
    SCHEDULE = "schedule"
    SAMPLER_E2E = "sample_e2e"
    PREPARE_INPUTS_E2E = "prepare_inputs_e2e"
    MODEL_EXECUTION_E2E = "model_execution_e2e"
    PROCESS_MODEL_OUTPUTS = "process_model_outputs"
    RAY_COMM_TIME = "ray_comm_time"


class RequestMetricsTimeDistributions(enum.Enum):
    """
    Request-level metrics for time distributions.

    Naming Convention:
    - All metrics use snake_case
    - Category prefixes: request_, prefill_, decode_, transfer_
    - Suffix ordering: [category]_[subcategory]_[measurement]_[modifier]
      e.g., request_e2e_time, request_e2e_time_normalized
    - Use "_per_token" suffix for normalized metrics instead of "_normalized" when
      the normalization is by token count

    IMPORTANT: execution_time vs e2e_time Semantic Difference (PD+AF Disaggregation)
    ================================================================================
    In disaggregated architectures (PD-disaggregation and PD+AF-disaggregation), it is
    EXPECTED and VALID for request_execution_time > request_e2e_time. This is NOT a bug.

    - request_e2e_time: Wall-clock time from request arrival to completion.
      This is the actual latency experienced by the user.

    - request_execution_time: TOTAL computational resource consumption across ALL clusters.
      This is the sum of execution times recorded in each cluster (PREFILL, DECODE_ATTN,
      DECODE_FFN, or DECODE). In disaggregated mode, different requests can be processed
      in parallel across clusters. For example:
        - While Request A is being processed in DECODE_FFN cluster
        - Request B may simultaneously be processed in DECODE_ATTN cluster

      The execution_time for Request A includes its time in PREFILL + DECODE_ATTN + DECODE_FFN,
      even though some of these phases may overlap with other requests' processing. Therefore,
      the sum of all cluster execution times can exceed the actual wall-clock time.

    Example: In PD+AF mode with 2 decode tokens and 24 layers:
      - e2e_time: 293.54ms (actual wall-clock latency)
      - execution_time: 449.22ms (sum of all cluster processing times)
      - Difference: 155.68ms is due to parallel processing across clusters

    For accurate resource utilization analysis, use execution_time.
    For user-facing latency analysis, use e2e_time.
    """

    # =========================================================================
    # REQUEST-LEVEL AGGREGATE METRICS
    # These metrics summarize the entire request lifecycle
    # =========================================================================
    REQUEST_E2E_TIME = "request_e2e_time"  # Wall-clock time from arrival to completion
    REQUEST_E2E_TIME_NORMALIZED = "request_e2e_time_normalized"  # E2E time / total tokens
    # NOTE: In disaggregated mode, execution_time can exceed e2e_time due to parallel
    # processing across clusters. See class docstring for detailed explanation.
    REQUEST_EXECUTION_TIME = "request_execution_time"  # Total GPU execution time across all clusters
    REQUEST_EXECUTION_TIME_NORMALIZED = "request_execution_time_normalized"  # Execution time / total tokens
    REQUEST_MODEL_EXECUTION_TIME = "request_model_execution_time"  # Model forward pass time only
    REQUEST_MODEL_EXECUTION_TIME_NORMALIZED = "request_model_execution_time_normalized"  # Model time / total tokens
    REQUEST_PREEMPTION_TIME = "request_preemption_time"  # Deprecated: always 0, use request_waiting_time_* for queue waiting
    REQUEST_FIRST_SCHEDULING_DELAY = "request_first_scheduling_delay"  # First scheduling delay (renamed from REQUEST_SCHEDULING_DELAY)
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME = "request_execution_plus_preemption_time"  # Legacy alias: equal to request_execution_time
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED = (
        "request_execution_plus_preemption_time_normalized"  # Legacy alias: request_execution_time / total tokens
    )

    # =========================================================================
    # PER-CLUSTER WAITING TIME METRICS
    # These metrics track cumulative time spent waiting in each cluster's queue
    # Includes time after preemption events
    # =========================================================================
    REQUEST_WAITING_TIME_TOTAL = "request_waiting_time_total"  # Total waiting time across all clusters
    REQUEST_WAITING_TIME_PREFILL = "request_waiting_time_prefill"  # PREFILL cluster waiting time
    REQUEST_WAITING_TIME_DECODE = "request_waiting_time_decode"  # DECODE cluster waiting time
    REQUEST_WAITING_TIME_DECODE_ATTN = "request_waiting_time_decode_attn"  # DECODE_ATTN cluster waiting time
    REQUEST_WAITING_TIME_DECODE_FFN = "request_waiting_time_decode_ffn"  # DECODE_FFN cluster waiting time
    REQUEST_HIDDEN_WAITING_TIME = "request_hidden_waiting_time"  # Waiting time accumulated on hidden rounds
    REQUEST_FINAL_WAITING_TIME = "request_final_waiting_time"  # Waiting time accumulated on final round
    REQUEST_HIDDEN_SERVICE_TIME = "request_hidden_service_time"  # Scheduled-service time accumulated on hidden rounds
    REQUEST_FINAL_SERVICE_TIME = "request_final_service_time"  # Scheduled-service time accumulated on final round
    FINAL_ROUND_PREFILL_WAIT_MS = "final_round_prefill_wait_ms"  # Exported in ms via MetricsStore._save_as_csv
    FINAL_ROUND_DECODE_WAIT_MS = "final_round_decode_wait_ms"  # Exported in ms via MetricsStore._save_as_csv
    FINAL_ROUND_PREFILL_SERVICE_MS = "final_round_prefill_service_ms"  # Exported in ms via MetricsStore._save_as_csv
    LATE_HIDDEN_PREFILL_WAIT_MS = "late_hidden_prefill_wait_ms"  # Exported in ms via MetricsStore._save_as_csv
    REQUEST_THINKING_TIME = "request_thinking_time"  # Total wall-clock time spent in hidden thinking rounds
    REQUEST_TOOL_CALL_TIME = "request_tool_call_time"  # Total tool wait time between hidden rounds

    # =========================================================================
    # PREFILL PHASE METRICS
    # These metrics cover the prefill (prompt processing) phase
    # =========================================================================
    PREFILL_E2E_TIME = "prefill_e2e_time"  # Prefill phase e2e time (arrived_at to prefill_completed_at)
    PREFILL_EXECUTION_PLUS_PREEMPTION = "prefill_execution_plus_preemption"  # Prefill (scheduled_at to prefill_completed_at)
    PREFILL_EXECUTION_PLUS_PREEMPTION_PER_TOKEN = (
        "prefill_execution_plus_preemption_per_token"  # Above / num_prefill_tokens
    )

    # =========================================================================
    # DECODE PHASE METRICS
    # These metrics cover the decode (token generation) phase
    # =========================================================================
    DECODE_E2E_TIME = "decode_e2e_time"  # Decode phase total time (prefill_completed_at to completed_at)
    DECODE_E2E_TIME_PER_TOKEN = (
        "decode_e2e_time_per_token"  # Decode E2E time / num_decode_tokens
        # NOTE: This includes execution, preemption, transfer, and scheduling delay
    )
    DECODE_FIRST_TOKEN_LATENCY = (
        "decode_first_token_latency"  # First token generated by decode process (second token overall)
    )

    # =========================================================================
    # TTFT (TIME TO FIRST TOKEN) METRICS
    # These metrics now use the queue-arrival -> prefill-complete contract
    # =========================================================================
    TTFT = "ttft"  # Total time from arrival to prefill completion
    TTFT_PREFILL_ONLY = "ttft_prefill_only"  # Prefill execution time component
    TTFT_KV_TRANSFER = "ttft_kv_transfer"  # KV cache transfer time component (PD/PD+AF modes)
    TTFT_DECODE_FIRST_TOKEN = (
        "ttft_decode_first_token"  # Legacy TTFT residual kept for backward reference only
    )

    # =========================================================================
    # TPOT (TIME PER OUTPUT TOKEN) METRICS
    # These metrics measure average decode token generation latency (excluding first token)
    # =========================================================================
    TPOT = "tpot"  # Average e2e time per output token (excluding first token)
    TPOT_COMPUTATION = "tpot_computation"  # Computation time per token
    TPOT_TRANSFER = "tpot_transfer"  # Transfer time per token (M2N, PD+AF mode)

    # =========================================================================
    # TRANSFER TIME METRICS
    # These metrics measure inter-cluster data transfer times (disaggregated modes)
    # =========================================================================
    TRANSFER_KV_CACHE = "transfer_kv_cache"  # Total KV cache transfer time (PREFILL → DECODE)
    TRANSFER_KV_CACHE_REQUEST_START_TS = (
        "transfer_kv_cache_request_start_ts"  # Request-level transfer window start
    )
    TRANSFER_KV_CACHE_REQUEST_END_TS = (
        "transfer_kv_cache_request_end_ts"  # Request-level transfer window end
    )
    TRANSFER_M2N_TOTAL = "transfer_m2n_total"  # Total M2N transfer time (A↔F)
    TRANSFER_M2N_ATTN_TO_FFN = "transfer_m2n_attn_to_ffn"  # A→F activation transfer time
    TRANSFER_M2N_FFN_TO_ATTN = "transfer_m2n_ffn_to_attn"  # F→A result transfer time

    # =========================================================================
    # CLUSTER COMPUTATION TIME METRICS
    # These metrics measure pure GPU computation time per cluster
    #
    # IMPORTANT: ATTN/FFN Semantic Difference in PD+AF Mode
    # - DECODE_ATTN uses execution_time.total_time (all attention components + communication)
    # - DECODE_FFN uses execution_time.get_single_layer_moe_comp_time() (MoE grouped GEMM + gating only)
    # This means DECODE_FFN computation time appears much smaller than DECODE_ATTN.
    # See METRICS_DEFINITIONS.md for detailed explanation.
    # =========================================================================
    CLUSTER_PREFILL_COMPUTATION = "cluster_prefill_computation"  # PREFILL cluster computation time
    CLUSTER_DECODE_ATTN_COMPUTATION = "cluster_decode_attn_computation"  # DECODE_ATTN: full attention layer (PD+AF)
    CLUSTER_DECODE_FFN_COMPUTATION = "cluster_decode_ffn_computation"  # DECODE_FFN compute-only legacy metric
    CLUSTER_DECODE_COMPUTATION = "cluster_decode_computation"  # DECODE computation (PD mode)

    # =========================================================================
    # CLUSTER E2E TIME METRICS
    # These metrics measure cluster-level e2e time (computation only, no transfer)
    # =========================================================================
    CLUSTER_DECODE_ATTN_E2E_TIME = "cluster_decode_attn_e2e_time"  # DECODE_ATTN e2e time (PD+AF)
    CLUSTER_DECODE_FFN_E2E_TIME = "cluster_decode_ffn_e2e_time"  # DECODE_FFN request residence: (A→F end) to (F→A start)


class TokenMetricsTimeDistribution(enum.Enum):
    DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME = (
        "decode_token_execution_plus_preemption_time"
    )


class RequestMetricsHistogram(enum.Enum):
    REQUEST_INTER_ARRIVAL_DELAY = "request_inter_arrival_delay"
    REQUEST_NUM_TOKENS = "request_num_tokens"
    REQUEST_PREFILL_TOKENS = "request_num_prefill_tokens"
    REQUEST_DECODE_TOKENS = "request_num_decode_tokens"
    REQUEST_PD_RATIO = "request_pd_ratio"
    REQUEST_CACHED_PREFILL_TOKENS = "request_cached_prefill_tokens"
    REQUEST_PREFIX_CACHE_QUERY_BLOCKS = "request_prefix_cache_query_blocks"
    REQUEST_PREFIX_CACHE_HIT_BLOCKS = "request_prefix_cache_hit_blocks"
    REQUEST_NUM_RESTARTS = "request_num_restarts"
    REQUEST_THINKING_ROUND_COUNT = "request_thinking_round_count"
    REQUEST_SESSION_ID = "request_session_id"
    REQUEST_COHORT = "request_cohort"
    REQUEST_SJ2Q_PEN_QSHORT_ENTRIES_TOTAL = "request_sj2q_pen_qshort_entries_total"
    REQUEST_SJ2Q_PEN_QSHORT_LONG_HISTORY_ENTRIES_TOTAL = (
        "request_sj2q_pen_qshort_long_history_entries_total"
    )
    REQUEST_SJ2Q_PEN_QLONG_ENTRIES_TOTAL = "request_sj2q_pen_qlong_entries_total"
    REQUEST_SJ2Q_PEN_LONG_HISTORY_TO_QSHORT_REENTRY_COUNT = (
        "request_sj2q_pen_long_history_to_qshort_reentry_count"
    )
    REQUEST_SJ2Q_PEN_FIRST_LONG_HISTORY_ROUND_NUMBER = (
        "request_sj2q_pen_first_long_history_round_number"
    )

    # Preemption tracking metrics (request-level)
    # Total number of times this request was preempted across all clusters
    REQUEST_TOTAL_PREEMPTION_COUNT = "request_total_preemption_count"

    # Per-cluster preemption counts (request-level)
    # Number of times this specific request was preempted in each cluster type
    REQUEST_PREFILL_PREEMPTION_COUNT = "request_prefill_preemption_count"
    REQUEST_DECODE_PREEMPTION_COUNT = "request_decode_preemption_count"  # PD-disaggregation mode
    REQUEST_DECODE_ATTN_PREEMPTION_COUNT = "request_decode_attn_preemption_count"  # PD+AF-disaggregation mode
    REQUEST_HIDDEN_PREEMPTION_COUNT = "request_hidden_preemption_count"
    REQUEST_FINAL_PREEMPTION_COUNT = "request_final_preemption_count"

    # Complete tokens-at-preemption lists (request-level, raw data)
    # Comma-separated list of tokens completed at each preemption event for this request
    REQUEST_DECODE_TOKENS_AT_PREEMPTION_ALL = "request_decode_tokens_at_preemption_all"  # PD mode
    REQUEST_DECODE_ATTN_TOKENS_AT_PREEMPTION_ALL = "request_decode_attn_tokens_at_preemption_all"  # PD+AF mode

    # Decode tokens-at-preemption statistics (aggregated from DECODE and DECODE_ATTN clusters)
    # These metrics track the number of decode tokens completed when preemption occurred
    # Only meaningful for requests that were preempted during decode phase
    REQUEST_DECODE_TOKENS_AT_PREEMPTION_MEAN = "request_decode_tokens_at_preemption_mean"
    REQUEST_DECODE_TOKENS_AT_PREEMPTION_MAX = "request_decode_tokens_at_preemption_max"
    REQUEST_DECODE_TOKENS_AT_PREEMPTION_MIN = "request_decode_tokens_at_preemption_min"

    # Speculative decoding metrics (request-level)
    REQUEST_SPEC_TOTAL_ITERATIONS = "request_spec_total_iterations"
    REQUEST_SPEC_ACCEPTED_DRAFTS = "request_spec_accepted_drafts"
    REQUEST_SPEC_REJECTED_DRAFTS = "request_spec_rejected_drafts"
    REQUEST_SPEC_COMMITTED_TOKENS = "request_spec_committed_tokens"
    REQUEST_SPEC_ACCEPTANCE_RATIO = "request_spec_acceptance_ratio"
    REQUEST_SPEC_COMMITTED_PER_ITER = "request_spec_committed_per_iteration"


class BatchMetricsCountDistribution(enum.Enum):
    BATCH_NUM_TOKENS = "batch_num_tokens"
    BATCH_NUM_PREFILL_TOKENS = "batch_num_prefill_tokens"
    BATCH_NUM_DECODE_TOKENS = "batch_num_decode_tokens"
    BATCH_SIZE = "batch_size"


class BatchMetricsTimeDistribution(enum.Enum):
    BATCH_EXECUTION_TIME = "batch_execution_time"


class RequestCompletionMetricsTimeSeries(enum.Enum):
    REQUEST_ARRIVAL = "request_arrival"
    REQUEST_COMPLETION = "request_completion"


class TokenCompletionMetricsTimeSeries(enum.Enum):
    PREFILL_COMPLETIONS = "prefill_completion"
    DECODE_COMPLETIONS = "decode_completion"
