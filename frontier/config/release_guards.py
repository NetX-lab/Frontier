"""Release-guard messages and the disaggregated cluster field tables.

These are plain data shared by the configuration families, the schedulers,
the events and the metrics store.  Keeping them in a leaf module lets every
configuration module import them without importing one another.
"""


DISAGGREGATED_ARCHITECTURE_RELEASE_ERROR = (
    "Error: Disaggregated architecture support is currently being optimized and is not included in this release. "
    "It will be available in an upcoming version. Please use the co-located architecture for current usage and testing."
)

PD_DISAGGREGATION_PARALLEL_CLUSTER_RELEASE_ERROR = (
    "Error: pd-disaggregation public release support requires "
    "--no-enable_parallel_clusters. Parallel PDD is excluded from "
    "pre-release-v0.3 because post-ISSUE-022 five-pair MoE-64 measurements "
    "were slower than sequential: Simulator.run() by 35.29% and shell E2E "
    "by 24.81% (paired medians). The implementation remains available only "
    "to internal correctness tests."
)

PD_AF_DISAGGREGATION_PARALLEL_CLUSTER_RELEASE_ERROR = (
    "Error: pd-af-disaggregation v0.3 requires "
    "--no-enable_parallel_clusters. Parallel cluster processing for "
    "pd-af-disaggregation is deferred in this release."
)

PD_AF_PREFIX_CACHING_RELEASE_ERROR = (
    "Prefix caching is excluded for pd-af-disaggregation in v0.3. "
    "Disable replica_scheduler_config.enable_prefix_caching."
)

AICONFIGURATOR_BACKEND_RELEASE_ERROR = (
    "Error: The aiconfigurator communication backend is not included in this release. "
    "Please use collective_sim, astra_sim_analytical, analytical, or vidur for current usage and testing."
)

PD_AF_TRACE_REPLAY_DEFERRED_ERROR = (
    "Error: pd-af-disaggregation v0.3 trace-replay is deferred. "
    "The configured trace-driven fields are public stubs only and are not "
    "implemented in this release."
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
