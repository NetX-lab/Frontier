# Training User Guide

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-07 | Added the training guide and documented on-demand predictor cache training during E2E simulation. |

## Scope

This guide explains the role of `frontier.training` in the current repository.

Training converts profiling CSVs into cached execution-time predictor models. The E2E simulator uses those predictors when dummy mode is disabled.

> Note: During E2E simulation, Frontier checks the predictor cache. If a required predictor is missing, the simulator trains it on demand from the configured profiling CSVs and writes it to the cache. Running the training module separately is useful for pre-warming or debugging, but it is not required for normal E2E runs because this training path is currently included in the E2E flow.

## When to Run Training Separately

Run standalone training when you want to:

- verify that profiling CSVs contain the columns needed by the predictor;
- pre-warm `cache/` before a benchmark run;
- debug a predictor training error without running a full simulation;
- compare predictor artifacts across profiling datasets or model settings.

You usually do not need standalone training when:

- you are running the checked-in examples in dummy predictor mode;
- you are running an E2E simulator smoke that already points to valid CSVs;
- you only want to validate CLI wiring or output paths.

## Environment

From the repository root:

```bash
conda activate frontier
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
```

If you are training from newly collected GPU profiles, collect those profiles in the profiling environment first. The training CLI itself consumes CSV files and writes sklearn artifacts to a cache directory.

## CLI Entry Point

Show the training help:

```bash
python -m frontier.training -h
python -m frontier.training.cli -h
```

Available subcommands:

```text
moe
linear_op
attention
```

`mlp` remains as a deprecated alias for `linear_op`.

## Inputs and Outputs

### Inputs

Training uses the same profiling taxonomy as the simulator:

```text
data/profiling/compute/<device>/<model>/
├── linear_op.csv
├── attention.csv
└── moe.csv
```

Use `measurement_type` to match the CSV measurement family:

- `CUDA_EVENT`
- `KERNEL_ONLY`

The public profiling examples default to `cuda_event`, which corresponds to `CUDA_EVENT` for training.

### Outputs

By default, trained predictor artifacts are written under:

```text
cache/
```

You can override this with `--output_dir`. E2E simulation reads and writes predictor cache files through `metrics_config.cache_dir`, which also defaults to `cache`.

## Standalone Training Commands

### Linear operators

```bash
python -m frontier.training linear_op \
  --dataset_path data/profiling/compute/rtx_pro_6000/qwen2_dense_test/linear_op.csv \
  --output_dir cache \
  --measurement_type CUDA_EVENT \
  --model_name qwen2_dense_test \
  --device rtx_pro_6000 \
  --tensor_parallel_size 1
```

Use `linear_op` for dense linear operators and shared linear operations used by attention and residual paths.

### Attention

Layer-only attention training:

```bash
python -m frontier.training attention \
  --layer_dataset_path data/profiling/compute/rtx_pro_6000/qwen2_dense_test/attention.csv \
  --output_dir cache \
  --measurement_type CUDA_EVENT \
  --model_name qwen2_dense_test \
  --device rtx_pro_6000 \
  --tensor_parallel_size 1
```

Full attention training with compute-dependent models:

```bash
python -m frontier.training attention \
  --layer_dataset_path data/profiling/compute/rtx_pro_6000/qwen2_dense_test/attention.csv \
  --compute_dataset_path data/profiling/compute/rtx_pro_6000/qwen2_dense_test/linear_op.csv \
  --output_dir cache \
  --measurement_type CUDA_EVENT \
  --model_name qwen2_dense_test \
  --device rtx_pro_6000 \
  --tensor_parallel_size 1
```

When `--compute_dataset_path` is omitted, the CLI trains layer models only and skips compute-dependent attention models such as projections, RoPE, LayerNorm, and residual add.

### MoE

```bash
python -m frontier.training moe \
  --dataset_path data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/moe.csv \
  --output_dir cache \
  --measurement_type CUDA_EVENT \
  --model_name Qwen3-30B-A3B-tiny \
  --device rtx_pro_6000 \
  --moe_tensor_parallel_size 1 \
  --expert_parallel_size 1 \
  --routing_runtime_path uniform_topk \
  --gating_runtime_context prefill_hot
```

For MoE, keep `--routing_runtime_path` and `--gating_runtime_context` aligned with the CSV metadata. If the CSV does not contain matching rows, training should fail rather than train on unrelated rows.

## E2E On-Demand Cache Training

When dummy predictor mode is disabled, E2E simulation initializes execution-time predictors from the configured CSVs. The predictor path works as follows:

1. Resolve profiling CSV paths from CLI/config values.
2. Build the model hash from predictor settings and profiling data.
3. Check `metrics_config.cache_dir` for the matching cached model file.
4. If the file exists, load it.
5. If the file is missing, train the model during simulator initialization and save it to cache.

This means a normal E2E command can train missing predictors automatically:

```bash
python -m frontier.main \
  --simulation_mode offline \
  --sys_arch co-location \
  --cluster_config_num_replicas 1 \
  --replica_config_model_name qwen2_dense_test \
  --replica_config_attn_tensor_parallel_size 1 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_attn_data_parallel_size 1 \
  --cc_backend_config_type analytical \
  --replica_scheduler_config_type vllm_v1 \
  --request_generator_config_type synthetic \
  --synthetic_request_generator_config_num_requests 1 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 8 \
  --fixed_request_length_generator_config_decode_tokens 2 \
  --interval_generator_config_type poisson \
  --poisson_request_interval_generator_config_qps 1.0 \
  --no-random_forrest_execution_time_predictor_config_enable_dummy_mode \
  --random_forrest_execution_time_predictor_config_linear_op_input_file data/profiling/compute/rtx_pro_6000/qwen2_dense_test/linear_op.csv \
  --random_forrest_execution_time_predictor_config_atten_input_file data/profiling/compute/rtx_pro_6000/qwen2_dense_test/attention.csv \
  --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling \
  --metrics_config_output_dir outputs/examples/profiling-simulator \
  --metrics_config_run_id training_on_demand_smoke \
  --metrics_config_write_metrics \
  --metrics_config_store_request_metrics \
  --no-metrics_config_store_plots \
  --no-metrics_config_enable_chrome_trace \
  --no-metrics_config_write_json_trace
```

The profiling smoke scripts in `examples/profiling/` use this path for downstream CSV validation.

## Cache Management

| Setting | Default | Use |
|---------|---------|-----|
| `--metrics_config_cache_dir` | `cache` | Cache root used by E2E simulation. |
| `--output_dir` in `frontier.training` | `cache` | Output root used by standalone training. |

Use the same cache directory when you want standalone training and E2E simulation to share artifacts.

Do not copy cached model files between incompatible CSVs or predictor settings. Cache filenames include a hash, but the safest workflow is to regenerate cache files from the CSVs used by the target run.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Standalone training reports missing dataset. | The CSV path is wrong or the profile was not generated. | Check `data/profiling/compute/<device>/<model>/`. |
| Training fails with missing columns. | The CSV schema does not match the selected predictor path. | Regenerate the profile with the matching example script and measurement family. |
| MoE training has no matching rows. | `routing_runtime_path`, `gating_runtime_context`, TP, or EP settings do not match the CSV metadata. | Align CLI settings with the CSV. |
| E2E run spends time training at startup. | Predictor cache was missing. | Pre-run standalone training if startup time matters. |
| E2E run uses fixed latencies. | Dummy predictor mode is enabled. | Pass `--no-random_forrest_execution_time_predictor_config_enable_dummy_mode` and provide CSV paths. |
