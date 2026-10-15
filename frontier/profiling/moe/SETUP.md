# MoE Profiling Setup Guide

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-13 | Align setup/troubleshooting notes with measurement-aware profiling contract |
| 2026-10-14 | Replace the sarathi-serve setup, removed validation script, nonexistent CLI flags, and stale test paths with the current profiling environment, public wrapper, and model-config workflow |

## Prerequisites

### 1. Environment Setup

The MoE profiling entrypoint requires `torch`, `vllm`, and `triton`. It checks
them at startup and fails with an error that names the missing packages. Use
the dedicated Frontier profiling environment from the repository root:

```bash
conda env create -f environment_profiling.yml
conda activate frontier-profiling
python -m pip install -e ".[test]"
export PYTHONPATH=$PWD
```

An existing environment that already provides a compatible vLLM
(`environment_profiling.yml` pins `vllm>=0.10,<0.11`) also works; run the
commands from the repository root with `PYTHONPATH` pointing at it.

### 2. Hardware Requirements

- **Minimum**: 1 GPU with 8GB+ memory (for small-scale testing)
- **Recommended**: 1-8 GPUs with 40GB+ memory each (for full profiling)
- CUDA-capable GPU (tested on A100, A800, H100). On ROCm, pass
  `--profile_method device_event`.

### 3. Verify Installation

```bash
python -c "import torch, vllm, triton; print('profiling dependencies available')"
python -c "from frontier.profiling.moe.moe_vllm_kernel import check_vllm_available; print('vLLM fused MoE kernel available:', check_vllm_available())"
python -m frontier.profiling.moe.main --help
```

## Quick Start

### 1. Dry Run (Recommended First)

```bash
bash examples/profiling/profile_moe.sh --dry-run
```

The dry run needs no GPU. It prints the resolved command and output path:

```
Output taxonomy: data/profiling/compute/<device>/<model>/moe[<measurement suffix>].csv
Resolved output: <repo>/data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/moe.csv
Model: Qwen3-30B-A3B-tiny
Device: rtx_pro_6000
TP sizes: 1
EP sizes: 1
Profile method: cuda_event
Dry run: true
...
Dry run completed; no profiling command was executed.
```

The wrapper defaults are `--max_tokens 16`, TP=1, EP=1, `cuda_event`,
`--routing_runtime_path uniform_topk`, and
`--gating_runtime_context prefill_hot`. Override them with `--model`,
`--device`, `--max-tokens`, `--tp-sizes`, `--ep-sizes`, `--profile-method`,
`--routing-runtime-path`, `--gating-runtime-context`, and `--output-root`.

### 2. Full Profiling

For production use, run full profiling with realistic parameters. Model shapes
(`num_experts`, `router_topk`, hidden sizes) come from the model config; the
CLI has no flags to override them.

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --max_tokens 4096 \
    --num_tensor_parallel_workers 1 2 4 8 \
    --expert_parallel_sizes 1 2 4 8 \
    --device a100 \
    --profile_method cuda_event \
    --output_dir data/profiling \
    --disable_ray
```

To profile a custom MoE shape, add a config JSON under `data/config/models/`
and pass its file name to `--models`; see
[README.md](./README.md#supported-models).

**Note**: For multi-GPU profiling, keep `--disable_ray` and set `--num_gpus`
to the number of visible GPUs; each GPU runs its own worker process.

## Troubleshooting

### Issue 1: `ImportError: Moe profiling requires missing dependencies: ...`

**Solution**: Activate the profiling environment from
[Environment Setup](#1-environment-setup). The minimal `environment.yml`
intentionally excludes `torch`, `vllm`, and `triton`.

### Issue 2: CUDA Out of Memory

**Solution**: Reduce the profiling grid, or profile a smaller model config:

```bash
python -m frontier.profiling.moe.main \
    --models Qwen3-30B-A3B-tiny \
    --num_gpus 1 \
    --max_tokens 512 \
    --num_tensor_parallel_workers 1 \
    --expert_parallel_sizes 1 \
    --device a100 \
    --disable_ray
```

### Issue 3: Ray initialization fails

**Solution**: Use `--disable_ray`. The Ray path failed with grpcio 1.67.1 and
Ray 2.52.1:

```bash
python -m frontier.profiling.moe.main \
    --num_gpus 1 \
    --device a100 \
    --disable_ray \
    ...
```

### Issue 4: Profiling is very slow

**Possible causes**:
1. **Large parameter space**: The defaults profile TP sizes `1 2 4 8`, every EP
   divisor of `num_experts`, and 3 load-imbalance samples per point. Reduce
   `--max_tokens`, pass explicit `--num_tensor_parallel_workers` and
   `--expert_parallel_sizes`, or lower `--num_samples_per_distribution`.
2. **Single GPU**: Use more GPUs with `--num_gpus N --disable_ray`.
3. **Profile method**: keep the default `--profile_method record_function` for kernel-only data, or use `--profile_method cuda_event` for eager-family data.

**Solution**:
```bash
# Use 4 GPUs in non-Ray multiprocessing mode
python -m frontier.profiling.moe.main \
    --num_gpus 4 \
    --max_tokens 2048 \
    --num_tensor_parallel_workers 1 2 \
    --device a100 \
    --disable_ray \
    ...
```

## Output Validation

After profiling completes, validate the output:

### 1. Check CSV Files

```bash
ls -lh data/profiling/compute/<device>/<model_name>/
ls -lh data/profiling/compute/<device>/moe_config.yaml
```

Expected files (one CSV per run, selected by `--profile_method`):
- `moe.csv` when `--profile_method cuda_event` is used
- `moe_kernel_only.csv` when `--profile_method record_function` (the default) is used
- `moe_device_event.csv` when `--profile_method device_event` is used
- `moe_config.yaml` one level up, in `data/profiling/compute/<device>/`

### 2. Inspect CSV Content

```bash
head -n 5 data/profiling/compute/a100/<model_name>/moe.csv
```

Expected columns include (abridged; see [README.md](./README.md#csv-file-format)
for the full list):
```
time_stats.moe_gating_linear.median,time_stats.moe_gating_routing_topk.median,time_stats.moe_shuffling.median,time_stats.moe_grouped_gemm.median,num_tokens,num_experts,num_experts_per_device,expert_parallel_size,routing_runtime_path,routing_assignment_policy,routing_weight_policy,routing_uses_router_logits,gating_runtime_context,gating_runtime_context_impl,router_topk,hidden_dim,expert_hidden_dim,use_gated,num_tensor_parallel_workers,measurement_type
0.1234,0.0100,0.0670,2.2200,8,8,8,1,standard_fused_topk,logit_topk,softmax_renorm,True,standalone_legacy,none,2,4096,14336,True,1,CUDA_EVENT
0.2345,0.0050,0.0660,2.1450,16,8,8,1,uniform_topk,round_robin_uniform,uniform_1_over_topk,False,prefill_hot,ffn_like_prefix_20x,2,4096,14336,True,1,CUDA_EVENT
...
```

### 3. Validate Timing Values

```python
import pandas as pd

# Load profiling data
df = pd.read_csv("data/profiling/compute/a100/<model_name>/moe.csv")

# Check for valid timing values
assert (df["time_stats.moe_gating_linear.median"] > 0).all(), "All timing values should be positive"
assert (df["time_stats.moe_gating_linear.median"] < 1000).all(), "Timing values seem too large"

print("✓ Timing values are valid")
```

## Next Steps

After successful profiling:

1. **Use the profiling data in place**:
   ```bash
   # The profiler writes canonical files directly when --output_dir data/profiling is used:
   # data/profiling/compute/<device>/<model_name>/moe.csv
   # data/profiling/compute/<device>/<model_name>/moe_kernel_only.csv
   ```

2. **Run the CSV-driven simulator smoke** (checked-in tiny profiles, dummy mode
   disabled):
   ```bash
   bash examples/profiling/smoke_simulator_moe_csv.sh
   ```
   Pass `--device`, `--model`, and `--moe-csv` (plus `--linear-op-csv` and
   `--attention-csv`) to run it on your own files.

3. **Run the simulator with MoE profiling data**: `--replica_config_device`
   and `--replica_config_model_name` select
   `./data/profiling/compute/{DEVICE}/{MODEL}/moe.csv`. The expert count and
   router top-k come from the model config.
   ```bash
   python -m frontier.main \
       --replica_config_model_name mixtral_8x7b_moe \
       --replica_config_device a100 \
       ...
   ```

## Reference

- Main documentation: `frontier/profiling/moe/README.md`
- Profiling guide: `docs/profiling/README.md`
- Public profiling wrappers: `examples/profiling/README.md`
