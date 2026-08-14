# CLI User Guide

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-14 | Replaced the removed MoE routing-mode CLI with `moe_routing_distribution_type`. |
| 2026-08-09 | Restored the sequential-only public PDD guard and documented current post-ISSUE-022 slowdown evidence. |
| 2026-08-08 | Clarified the parallel-PDD correctness and wall-clock-speed boundary. |
| 2026-08-07 | Corrected the PDD parallel CLI contract while retaining sequential examples and the PD-AF parallel guard. |
| 2026-07-23 | Added the complete PD-AF dense/MoE/EP/CUDA Graph entrypoint set and corrected v0.3 runtime boundaries. |
| 2026-07-22 | Documented the sequential PD-AF CLI surface and one-click example entrypoints. |

## Scope

This guide covers the public CLI surface for the `pre-release-v0.3` branch. The supported runtime architectures are `co-location`, sequential `pd-disaggregation`, and sequential `pd-af-disaggregation`.

PDD public release execution is sequential-only: `pd-disaggregation` aborts unless `--no-enable_parallel_clusters`. PD-AF parallel cluster processing remains unsupported; PD-AF runs must also use `--no-enable_parallel_clusters`. PD-AF supports dense, MoE, EP, and global CUDA Graph execution. PD-AF does not support Thinking Mode, Speculative Decoding / MTP, or Prefix Caching in `pre-release-v0.3`.

The parallel PDD implementation remains covered by internal correctness tests but is not a supported release path.

Post-ISSUE-022 five-pair MoE-64 measurements observed paired-median slowdowns of 35.29% for Simulator.run() and 24.81% for shell E2E.

With globally unique full priorities, preloaded arrivals, and no asynchronous event source outside event handlers, the current total-order gate admits one handler at a time. Parallel threads therefore add coordination overhead without handler overlap.

Use the examples first. They set the required flags, disable optional services, and write metrics to a predictable location.

## Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate frontier
python -m pip install -e ".[test]"

export PYTHONPATH=$PWD
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
```

If you already have the environment, update the editable install before running new code:

```bash
conda activate frontier
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
```

The co-location example suite defaults to `--cc_backend_config_type analytical` for one-click smoke runs and does not require the optional `collective_sim` submodule. Build `collective_sim` only when you explicitly select it:

```bash
git submodule update --init --recursive frontier/cc_backend/backends/collective-sim
cd frontier/cc_backend/backends/collective-sim/sim
make -j"$(nproc)"
```

Use `--cc_backend_config_type analytical` for the release co-location smoke examples. Use `--cc_backend_config_type astra_sim_analytical` when you intentionally want the ASTRA-Sim-inspired lightweight topology model.

## Recommended Entry Points

Run the release examples before writing a custom command:

```bash
# Run all five offline cases and all five online cases.
bash examples/architecture/co-location/run_all.sh

# Offline cases.
bash examples/architecture/co-location/offline/dense_model_basic.sh
bash examples/architecture/co-location/offline/moe_model_basic.sh
bash examples/architecture/co-location/offline/thinking_mode_basic.sh
bash examples/architecture/co-location/offline/moe_spec_dec.sh
bash examples/architecture/co-location/offline/moe_prefix_caching.sh

# Online cases.
bash examples/architecture/co-location/online/dense_model_basic_online.sh
bash examples/architecture/co-location/online/moe_model_basic_online.sh
bash examples/architecture/co-location/online/thinking_mode_basic_online.sh
bash examples/architecture/co-location/online/moe_spec_dec_online.sh
bash examples/architecture/co-location/online/moe_prefix_caching_online.sh

# PD-AF / pd-af-disaggregation (sequential).
bash examples/architecture/pd-af-disagg/run_all.sh
bash examples/architecture/pd-af-disagg/offline/dense_model_basic.sh
bash examples/architecture/pd-af-disagg/offline/moe_model_basic.sh
bash examples/architecture/pd-af-disagg/offline/moe_model_ep.sh
bash examples/architecture/pd-af-disagg/offline/dense_cuda_graph.sh
bash examples/architecture/pd-af-disagg/offline/moe_cuda_graph.sh
bash examples/architecture/pd-af-disagg/online/dense_model_basic_online.sh
bash examples/architecture/pd-af-disagg/online/moe_model_basic_online.sh
bash examples/architecture/pd-af-disagg/online/moe_model_ep_online.sh
bash examples/architecture/pd-af-disagg/online/dense_cuda_graph_online.sh
bash examples/architecture/pd-af-disagg/online/moe_cuda_graph_online.sh
```

The co-location/PDD baseline dense, MoE, and Thinking Mode scripts enable these runtime settings by default:

- `--decode_cuda_graph_mode full_decode_only`
- `--vllm_v1_scheduler_config_enable_chunked_prefill`
- CSV/JSON metrics output
- dummy execution-time predictor mode for fast smoke tests

PD-AF basic and EP scripts use three sequential roles plus analytical KV/M2N transfer models. PD-AF CUDA Graph examples use `--use_cuda_graph`, not `--decode_cuda_graph_mode`. Their default capture sizes are `8 16 32 64`.

The advanced MoE scripts are available in both `offline/` and `online/`:

- `moe_spec_dec.sh` / `moe_spec_dec_online.sh`: Speculative Decoding / MTP. They use `decode_cuda_graph_mode=none` because Speculative Decoding and decode CUDA Graph modeling currently conflict unless a diagnostic opt-in is used.
- `moe_prefix_caching.sh` / `moe_prefix_caching_online.sh`: Prefix Caching. They replay `examples/fixtures/prefix_cache_shared_session_trace.csv` so repeated prompt blocks produce cache-hit behavior.

## Running `frontier.main` Directly

CLI entry point:

```bash
python -m frontier.main --help
```

A small dense co-location command:

```bash
python -m frontier.main \
  --simulation_mode offline \
  --sys_arch co-location \
  --cluster_config_num_replicas 1 \
  --replica_config_model_name meta-llama/Llama-2-7b-hf \
  --replica_config_attn_tensor_parallel_size 2 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_attn_data_parallel_size 1 \
  --cc_backend_config_type astra_sim_analytical \
  --replica_scheduler_config_type vllm_v1 \
  --decode_cuda_graph_mode full_decode_only \
  --vllm_v1_scheduler_config_enable_chunked_prefill \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 4 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 128 \
  --fixed_request_length_generator_config_decode_tokens 32 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 1.0 \
  --random_forrest_execution_time_predictor_config_enable_dummy_mode \
  --metrics_config_output_dir outputs/examples/co-location \
  --metrics_config_run_id cli_dense_smoke \
  --metrics_config_write_metrics \
  --metrics_config_store_request_metrics \
  --no-metrics_config_store_plots \
  --no-metrics_config_enable_chrome_trace \
  --no-metrics_config_write_json_trace
```

Dummy predictor mode is useful for smoke tests. For latency studies, disable dummy mode and point the predictor at profiling CSVs under `data/profiling/compute/<device>/<model>/`.

## Core CLI Groups

### Simulation mode and architecture

| Option | Use |
|--------|-----|
| `--simulation_mode offline` | Generate or replay requests inside the simulator. |
| `--simulation_mode online` | Run online mode where supported by the selected scheduler path. |
| `--sys_arch co-location` | Release-supported architecture. |
| `--sys_arch pd-disaggregation` | Sequential PDD architecture; requires `--no-enable_parallel_clusters`. |
| `--sys_arch pd-af-disaggregation` | Sequential PD-AF architecture; requires `--no-enable_parallel_clusters`. |

### Model and parallelism

| Option | Use |
|--------|-----|
| `--replica_config_model_name` | Model name used to resolve model config and output taxonomy. |
| `--cluster_config_num_replicas` | Number of monolithic replicas. |
| `--replica_config_attn_tensor_parallel_size` | Attention tensor parallel size. |
| `--replica_config_attn_data_parallel_size` | Attention data parallel size. |
| `--replica_config_num_pipeline_stages` | Pipeline parallel stages. |
| `--replica_config_moe_tensor_parallel_size` | MoE tensor parallel size. |
| `--replica_config_moe_expert_parallel_size` | MoE expert parallel size. |
| `--replica_config_total_expert_num` | Total expert count for MoE models. |
| `--replica_config_router_topk` | Number of routed experts per token. |
| `--replica_config_moe_routing_distribution_type` | MoE expert-load distribution: `balanced`, `random`, `skewed`, or `zipf`. |

### Runtime optimization

| Option | Use |
|--------|-----|
| `--decode_cuda_graph_mode full_decode_only` | Model decode CUDA Graph behavior for co-location/PDD decode-only batches. |
| `--decode_cuda_graph_mode none` | Disable decode CUDA Graph modeling. Required by the Speculative Decoding example. |
| `--use_cuda_graph` | Enable the global CUDA Graph model for PD-AF. |
| `--cudagraph_capture_sizes 8 16 32 64` | Set the global CUDA Graph capture sizes used by PD-AF examples. |
| `--vllm_v1_scheduler_config_enable_chunked_prefill` | Enable Chunked Prefill on the `vllm_v1` scheduler. |
| `--vllm_v1_scheduler_config_enable_prefix_caching` | Enable Prefix Caching on supported scheduler paths. |
| `--speculative_decoding_config_enabled` | Enable Speculative Decoding / MTP modeling. |

### Workload generation

Synthetic fixed-length workload:

```bash
--request_generator_config_type synthetic
--synthetic_request_generator_config_num_requests 16
--length_generator_config_type fixed
--fixed_request_length_generator_config_prefill_tokens 512
--fixed_request_length_generator_config_decode_tokens 128
--interval_generator_config_type poisson
--poisson_request_interval_generator_config_qps 1.0
```

Trace replay workload:

```bash
--request_generator_config_type trace_replay
--trace_request_generator_config_trace_file examples/fixtures/prefix_cache_shared_session_trace.csv
--trace_request_generator_config_max_tokens 128
```

Use trace replay when you need shared-prefix or known-arrival behavior.

### Execution-time predictor

| Option | Use |
|--------|-----|
| `--random_forrest_execution_time_predictor_config_enable_dummy_mode` | Use fixed dummy execution time. Good for smoke tests only. |
| `--no-random_forrest_execution_time_predictor_config_enable_dummy_mode` | Use profiling-backed ML predictors. |
| `--random_forrest_execution_time_predictor_config_linear_op_input_file` | Path to `linear_op.csv`. |
| `--random_forrest_execution_time_predictor_config_atten_input_file` | Path to `attention.csv`. |
| `--random_forrest_execution_time_predictor_config_moe_input_file` | Path to `moe.csv`. Required for MoE non-dummy runs. |
| `--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling` | Skip CPU overhead predictor modeling for lightweight CSV smoke runs. |

When dummy mode is disabled, Frontier checks the predictor cache. If a needed model is missing, it trains from the configured CSVs and writes the trained estimator into the cache directory.

### Metrics output

| Option | Use |
|--------|-----|
| `--metrics_config_output_dir` | Metrics output root. The simulator appends `<model>/<workload>/<run_id>/`. |
| `--metrics_config_run_id` | Stable run id. Use this for reproducible output paths. |
| `--metrics_config_write_metrics` | Write metrics artifacts. |
| `--metrics_config_store_request_metrics` | Write request-level CSV metrics. |
| `--metrics_config_store_batch_metrics` | Write batch-level CSV metrics. |
| `--metrics_config_store_token_completion_metrics` | Write token completion metrics. |
| `--metrics_config_store_utilization_metrics` | Write utilization metrics. |
| `--no-metrics_config_store_plots` | Skip optional Plotly plot export. |
| `--no-metrics_config_enable_chrome_trace` | Skip Chrome trace output. |
| `--no-metrics_config_write_json_trace` | Skip JSON event trace output. |

Output path format:

```text
<metrics_config_output_dir>/<model_type>/<offline_batch|online_serving>/<run_id>/
```

Common files include:

- `config.json`
- `system_metrics.json`
- `request_metrics.csv` when request metrics are enabled
- `<cluster>_batch_metrics.csv` when batch metrics are enabled
- utilization CSVs when utilization metrics are enabled

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Release guard exits for `pd-disaggregation` or `pd-af-disaggregation`. | Parallel clusters are enabled, or the selected feature is outside the v0.3 surface. | Add `--no-enable_parallel_clusters` and use only the supported feature set documented above. |
| `htsim_ndp` is missing after selecting `--cc_backend_config_type collective_sim`. | The optional `collective_sim` submodule binary has not been built. | Build `frontier/cc_backend/backends/collective-sim/sim`, or use the default co-location example `analytical` backend. |
| W&B tries to initialize. | Environment variables are not set. | Set `WANDB_DISABLED=true` and `VIDUR_DISABLE_WANDB=1`. |
| Non-dummy run fails on a missing CSV or schema mismatch. | Predictor training needs matching profiling data. | Use the profiling guide and keep CSVs under `data/profiling/compute/<device>/<model>/`. |
| Plot export warns about `kaleido`. | PNG export is optional. | Keep `--no-metrics_config_store_plots` for smoke runs, or install `kaleido` if PNGs are needed. |
