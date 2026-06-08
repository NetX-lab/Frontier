## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-07 | Set public examples to the `astra_sim_analytical` backend by default and documented `collective_sim` as optional. |
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

The public co-location examples default to `--cc_backend_config_type astra_sim_analytical` so they run without building the optional `collective_sim` submodule. `astra_sim_analytical` is the default public example backend. Build `collective_sim` only when you explicitly pass `--cc_backend_config_type collective_sim` and verify `frontier/cc_backend/backends/collective-sim/sim/datacenter/htsim_ndp` exists.

```bash
export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1

bash examples/architecture/co-location/dense_model_basic.sh
bash examples/architecture/co-location/moe_model_basic.sh
bash examples/architecture/co-location/thinking_mode_basic.sh
bash examples/architecture/co-location/moe_spec_dec.sh
bash examples/architecture/co-location/moe_prefix_caching.sh
```

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
│       ├── dense_model_basic.sh
│       ├── moe_model_basic.sh
│       ├── thinking_mode_basic.sh
│       ├── moe_spec_dec.sh
│       └── moe_prefix_caching.sh
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
- Supports the included Dense Thinking Mode smoke example.

## Key Configuration Options

### Parallelism

- `--replica_config_attn_tensor_parallel_size`: Attention tensor parallelism.
- `--replica_config_moe_tensor_parallel_size`: MoE tensor parallelism.
- `--replica_config_moe_expert_parallel_size`: Expert parallelism.
- `--replica_config_num_pipeline_stages`: Pipeline parallelism.
- `--cluster_config_num_replicas`: Number of monolithic cluster replicas.

### Request Generation

- `--interval_generator_config_type`: `poisson`, `gamma`, `static`, or `trace`.
- `--length_generator_config_type`: `fixed`, `uniform`, `zipf`, or `trace`.

### Communication Cost Backends

- `--cc_backend_config_type astra_sim_analytical`: Default public example backend; lightweight ASTRA-Sim-inspired analytical topology model.
- `--cc_backend_config_type collective_sim`: Optional topology-aware collective simulation; requires the `collective-sim` submodule and `htsim_ndp` binary.
- `--cc_backend_config_type analytical`: Formula-based prediction.
- `--cc_backend_config_type vidur`: ML-based prediction from profiling data.

### Logging

- `--log_level`: `debug`, `info`, `warning`, or `error`.
- `--enable_cluster_event_logging`: Enable detailed cluster logs.
- `--cluster_log_filter`: Filter logs by cluster type.

## Running Examples

The checked-in co-location simulation examples use dummy mode (`--random_forrest_execution_time_predictor_config_enable_dummy_mode`) for quick testing without profiling data. Dummy mode skips ML predictor training and profiling metadata loading, so missing profiling CSVs do not affect smoke-test correctness.

Baseline co-location scripts default to `--cc_backend_config_type astra_sim_analytical`, `decode_cuda_graph_mode=full_decode_only`, and Chunked Prefill. The Speculative Decoding / MTP recipe uses `decode_cuda_graph_mode=none` because speculative decoding currently conflicts with decode CUDA Graph modeling. The Prefix Caching recipe replays `examples/fixtures/prefix_cache_shared_session_trace.csv` to exercise cache-hit behavior.

For production simulations, remove the dummy mode flag and ensure profiling data is available in `data/profiling/compute/<device>/<model>/`.

All co-location examples write CSV/JSON metrics by default and disable only plots, Chrome trace, and JSON event trace outputs. PNG plot export is optional and requires `kaleido`. If `kaleido` is not installed, Plotly can warn about image export; CSV/JSON metrics are still produced.

## Thinking Mode Example

The Thinking Mode script uses:

- `--enable_thinking_mode`
- `--thinking_depth 2`
- one explicit hidden round
- explicit `vllm_v1` scheduler configuration
- `--cc_backend_config_type astra_sim_analytical` for a minimal one-click smoke run
- CSV/JSON metrics enabled by default, with plots/traces disabled for a lightweight smoke artifact set

## See Also

- `README.md`: release overview and install instructions.
- `examples/architecture/README.md`: architecture-specific example list.
- `examples/profiling/README.md`: profiling examples and downstream CSV simulator smoke workflows.
