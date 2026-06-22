## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-22 | Cleaned public PDD wording to avoid unsupported split-decode terminology. |
| 2026-06-14 | Added PDD pd-disaggregation examples, script index, and release-scope guidance for local PR preparation. |
| 2026-06-08 | Clarified that dummy analytical co-location smoke runs validate runtime plumbing, not profiling fidelity. |
| 2026-06-08 | Split co-location examples into `offline/` and `online/`, added suite runner and cross-validation guidance. |
| 2026-06-07 | Added optimized co-location advanced MoE recipes, top-level profiling examples, and corrected metrics behavior for Thinking Mode. |
| 2026-06-06 | Documented collective-sim prerequisites, optional Kaleido PNG export, and dummy-mode profiling behavior for co-location examples. |
| 2026-06-04 | Clarified that transfer and disaggregated example surfaces are guarded out of `pre-release-v0.1`. |
| 2026-06-04 | Restored release-supported co-location examples and removed disaggregated example references for `pre-release-v0.1`. |
| 2026-06-03 | Reworked examples for the co-location-only open-source release branch. Earlier internal history may refer to guarded or removed private examples. |

# Frontier Examples

This directory contains runnable examples for the release-supported Frontier simulator surface.

## Release Scope

`pre-release-v0.2` foregrounds **PDD / `pd-disaggregation`** examples: prefill runs in the `PREFILL` cluster, decode runs in the unified `DECODE` cluster, and KV cache is transferred between them. The public PDD example path uses the sequential simulator mode through `--no-enable_parallel_clusters`.

Additional disaggregated research prototypes outside the PDD path remain intentionally outside this examples release scope. Co-location examples are still kept as baseline comparison recipes and historical v0.1-compatible references.

## Quick Start

The PDD dense example uses dummy execution-time prediction and the analytical communication backend, so it does not require profiling data or the collective-sim binary for the first smoke run.

```bash
export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

bash examples/architecture/pdd/offline/dense_model_basic.sh
```

For the complete PDD architecture suite, run:

```bash
bash examples/architecture/pdd/run_all.sh
```

PDD Thinking Mode examples are available in both modes:

```bash
bash examples/architecture/pdd/offline/thinking_mode_basic.sh
bash examples/architecture/pdd/online/thinking_mode_basic_online.sh
```

Co-location baseline and advanced recipes remain available for comparison. The current co-location layout is split into offline and online entrypoints:

```bash
bash examples/architecture/co-location/run_all.sh
bash examples/architecture/co-location/offline/dense_model_basic.sh
bash examples/architecture/co-location/offline/moe_model_basic.sh
bash examples/architecture/co-location/offline/thinking_mode_basic.sh
bash examples/architecture/co-location/offline/moe_spec_dec.sh
bash examples/architecture/co-location/offline/moe_prefix_caching.sh
bash examples/architecture/co-location/online/dense_model_basic_online.sh
bash examples/architecture/co-location/online/moe_model_basic_online.sh
bash examples/architecture/co-location/online/thinking_mode_basic_online.sh
bash examples/architecture/co-location/online/moe_spec_dec_online.sh
bash examples/architecture/co-location/online/moe_prefix_caching_online.sh
```

```bash
bash examples/profiling/profile_linear_op.sh --dry-run
bash examples/profiling/profile_attention_chunked_prefill.sh --dry-run
bash examples/profiling/profile_moe.sh --dry-run
```

## Directory Structure

```text
examples/
├── README.md
├── fixtures/
│   └── prefix_cache_shared_session_trace.csv
├── architecture/
│   ├── README.md
│   ├── pdd/
│   │   ├── run_all.sh
│   │   ├── dense_model_basic.sh
│   │   ├── offline/
│   │   │   ├── dense_model_basic.sh
│   │   │   ├── moe_model_basic.sh
│   │   │   ├── thinking_mode_basic.sh
│   │   │   ├── moe_spec_dec.sh
│   │   │   └── moe_prefix_caching.sh
│   │   └── online/
│   │       ├── dense_model_basic_online.sh
│   │       ├── moe_model_basic_online.sh
│   │       ├── thinking_mode_basic_online.sh
│   │       ├── moe_spec_dec_online.sh
│   │       └── moe_prefix_caching_online.sh
│   └── co-location/
│       ├── run_all.sh
│       ├── offline/
│       │   ├── dense_model_basic.sh
│       │   ├── moe_model_basic.sh
│       │   ├── thinking_mode_basic.sh
│       │   ├── moe_spec_dec.sh
│       │   └── moe_prefix_caching.sh
│       └── online/
│           ├── dense_model_basic_online.sh
│           ├── moe_model_basic_online.sh
│           ├── thinking_mode_basic_online.sh
│           ├── moe_spec_dec_online.sh
│           └── moe_prefix_caching_online.sh
└── profiling/
    ├── README.md
    ├── profile_linear_op.sh
    ├── profile_attention_chunked_prefill.sh
    ├── profile_moe.sh
    ├── smoke_metadata.sh
    ├── smoke_simulator_dense_csv.sh
    └── smoke_simulator_moe_csv.sh
```

## Architecture Mode

### PDD / pd-disaggregation

Separate prefill and decode clusters model prefill/decode disaggregation through one public decode role.

- `--sys_arch pd-disaggregation`
- Uses `PREFILL` and unified `DECODE` clusters.
- Supports Dense, MoE, Thinking Mode, Speculative Decoding / MTP, and Prefix Caching examples in offline and online modes.
- Uses `--no-enable_parallel_clusters` because the pre-release-v0.2 public PDD path is the sequential simulator path; parallel cluster processing is still guarded.
- Keeps experimental disaggregation variants and global `--use_cuda_graph` outside the v0.2 examples release surface.

### Co-location

Single monolithic cluster handles all prefill and decode work. These examples are retained as baseline comparison recipes.

- `--sys_arch co-location`
- Supports dense and MoE model configs.
- Supports both `--simulation_mode offline` and `--simulation_mode online`.
- Supports the included Dense Thinking Mode smoke examples.

## Key Configuration Options

### PDD Cluster Layout

- `--cluster_config_prefill_cluster_num_replicas`: Number of `PREFILL` cluster replicas.
- `--cluster_config_decode_cluster_num_replicas`: Number of unified `DECODE` cluster replicas.
- `--cluster_config_prefill_replica_config_*`: `PREFILL` replica parallelism and device fields.
- `--cluster_config_decode_replica_config_*`: `DECODE` replica parallelism and device fields.
- `--analytical_kv_cache_transfer_config_network_bandwidth_gbps`: Analytical KV transfer bandwidth.
- `--analytical_kv_cache_transfer_config_network_latency_ms`: Analytical KV transfer latency.

### Parallelism

- `--replica_config_attn_tensor_parallel_size`: Attention tensor parallelism for co-location examples.
- `--replica_config_moe_tensor_parallel_size`: MoE tensor parallelism for co-location examples.
- `--replica_config_moe_expert_parallel_size`: Expert parallelism for co-location examples.
- `--replica_config_num_pipeline_stages`: Pipeline parallelism for co-location examples.
- `--cluster_config_num_replicas`: Number of monolithic cluster replicas for co-location examples.

### Request Generation

- `--request_generator_config_type synthetic`: Generate requests from configured length and interval generators.
- `--request_generator_config_type trace_replay`: Replay a CSV trace, used by Prefix Caching examples.
- `--interval_generator_config_type`: `poisson`, `gamma`, `static`, or `trace`.
- `--length_generator_config_type`: `fixed`, `uniform`, `zipf`, or `trace`.

### Communication Cost Backends

- `--cc_backend_config_type collective_sim`: Topology-aware collective simulation.
- `--cc_backend_config_type astra_sim_analytical`: Lightweight ASTRA-Sim-inspired analytical topology model.
- `--cc_backend_config_type vidur`: ML-based prediction from profiling data.
- `--cc_backend_config_type analytical`: Formula-based prediction.

### Logging

- `--log_level`: `debug`, `info`, `warning`, or `error`.
- `--enable_cluster_event_logging`: Enable detailed cluster logs.
- `--cluster_log_filter`: Filter logs by cluster type.

## Running Examples

The checked-in PDD examples use dummy mode (`--random_forrest_execution_time_predictor_config_enable_dummy_mode`), analytical communication cost modeling, and `--no-enable_parallel_clusters` for quick testing without profiling data. The expected minimal dense smoke behavior is one completed request, one KV cache transfer, and no release-guard crash. Metrics are written under `outputs/examples/pdd` by default.

PDD Thinking Mode can produce multiple prefill-to-decode handoffs for one user request. The default small smoke configuration completes one request and records two KV transfers.

Co-location examples also use dummy mode for quick testing without profiling data. These examples validate CLI/runtime plumbing and metrics artifact generation, not profiling fidelity. Use non-dummy profiling data before drawing hardware accuracy conclusions.

Baseline co-location scripts default to `decode_cuda_graph_mode=full_decode_only` and Chunked Prefill. The Speculative Decoding / MTP recipes use `decode_cuda_graph_mode=none` because speculative decoding currently conflicts with decode CUDA Graph modeling. The Prefix Caching recipes replay `examples/fixtures/prefix_cache_shared_session_trace.csv` to exercise cache-hit behavior.

For production simulations, remove the dummy mode flag and ensure profiling data is available in `data/profiling/compute/<device>/<model>/`.

All co-location examples write CSV/JSON metrics by default and disable only plots, Chrome trace, and JSON event trace outputs. PNG plot export is optional and requires `kaleido`. If `kaleido` is not installed, Plotly can warn about image export; CSV/JSON metrics are still produced.

## Cross-validation Criteria

When comparing offline and online pairs, validate the following for each scenario:

1. The case exits with code `0`.
2. `request_metrics.csv` and `system_metrics.json` are created.
3. The completed request count equals the configured or trace-derived request count.
4. Numeric latency and throughput fields are finite and non-negative.
5. Offline output lands under `offline_batch`; online output lands under `online_serving`.
6. For matched synthetic cases, online/offline request counts and token settings match; latency may differ because online mode preserves arrival timestamps while offline batch mode admits requests at `t=0`.

## Thinking Mode Example

The PDD and co-location Thinking Mode scripts use:

- `--enable_thinking_mode`
- `--thinking_depth 2`
- one explicit hidden round
- explicit `vllm_v1` scheduler configuration
- `--cc_backend_config_type analytical` by default for a minimal one-click smoke run
- CSV/JSON metrics enabled by default, with plots/traces disabled for a lightweight smoke artifact set

## See Also

- `README.md`: release overview and install instructions.
- `examples/architecture/README.md`: architecture-specific example list.
- `examples/profiling/README.md`: profiling examples and downstream CSV simulator smoke workflows.
