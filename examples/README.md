## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
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

`pre-release-v0.1` supports only the `co-location` architecture. Historical `pd-disaggregation` and `pd-af-disaggregation` examples are intentionally not included in this branch. If those architectures are requested through CLI/config, Frontier exits with the release error documented in the top-level `README.md`.

## Quick Start

The co-location examples are split by simulation mode:

- `examples/architecture/co-location/offline/`: offline batch simulations. Existing offline examples were moved here unchanged in scenario intent.
- `examples/architecture/co-location/online/`: online serving simulations that mirror the offline scenarios while preserving generated request arrivals.
- `examples/architecture/co-location/run_all.sh`: one-click suite runner for all 10 co-location cases.

```bash
export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

# Run all five offline cases and all five online cases.
bash examples/architecture/co-location/run_all.sh

# Run one case directly.
bash examples/architecture/co-location/offline/dense_model_basic.sh
bash examples/architecture/co-location/online/dense_model_basic_online.sh

# Thinking Mode examples are available in both modes.
bash examples/architecture/co-location/offline/thinking_mode_basic.sh
bash examples/architecture/co-location/online/thinking_mode_basic_online.sh
```

All co-location examples default to `--cc_backend_config_type analytical` so the suite is one-click runnable on a fresh checkout without building the collective-sim binary. To exercise the topology-aware backend, set `CC_BACKEND=collective_sim` and build `frontier/cc_backend/backends/collective-sim/sim/datacenter/htsim_ndp` first.

Profiling commands can be validated without launching GPU kernels by using `--dry-run`:

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

### Co-location

Single monolithic cluster handles all prefill and decode work.

- `--sys_arch co-location`
- Supports dense and MoE model configs.
- Supports both `--simulation_mode offline` and `--simulation_mode online`.
- Supports the included Dense Thinking Mode smoke examples.

## Key Configuration Options

### Parallelism

- `--replica_config_attn_tensor_parallel_size`: Attention tensor parallelism.
- `--replica_config_moe_tensor_parallel_size`: MoE tensor parallelism.
- `--replica_config_moe_expert_parallel_size`: Expert parallelism.
- `--replica_config_num_pipeline_stages`: Pipeline parallelism.
- `--cluster_config_num_replicas`: Number of monolithic cluster replicas.

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

The checked-in co-location simulation examples use dummy mode (`--random_forrest_execution_time_predictor_config_enable_dummy_mode`) for quick testing without profiling data. Dummy mode skips ML predictor training and profiling metadata loading, so missing profiling CSVs do not affect smoke-test correctness.

These examples validate CLI/runtime plumbing and metrics artifact generation, not profiling fidelity. Use non-dummy profiling data before drawing hardware accuracy conclusions.

Offline cases write under `outputs/examples/co-location/offline/<model_type>/offline_batch/<run_id>/` by default. Online cases write under `outputs/examples/co-location/online/<model_type>/online_serving/<run_id>/` by default. The mode-specific `offline_batch` / `online_serving` path segment is added by Frontier's metrics taxonomy.

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

The Thinking Mode scripts use:

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
