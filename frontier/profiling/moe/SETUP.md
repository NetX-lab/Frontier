# MoE Profiling Setup Guide

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-13 | Align setup/troubleshooting notes with measurement-aware profiling contract |

## Prerequisites

### 1. Environment Setup

The MoE profiling module requires the dedicated Frontier profiling environment or an existing environment that already provides vLLM and FlashInfer. Use `environment_profiling.yml` for a reproducible setup:

1. **Clone sarathi-serve repository**:
   ```bash
   git clone https://github.com/microsoft/sarathi-serve.git
   cd sarathi-serve
   git checkout vidur-test  # Use the vidur-test branch
   ```

2. **Create and activate virtual environment**:
   ```bash
   cd sarathi-serve
   python -m venv env
   source env/bin/activate  # On Linux/Mac
   # or
   env\Scripts\activate  # On Windows
   ```

3. **Install sarathi-serve**:
   ```bash
   pip install -e .
   ```

4. **Install vidur in the same environment**:
   ```bash
   cd /path/to/vidur
   pip install -e .
   ```

### 2. Hardware Requirements

- **Minimum**: 1 GPU with 8GB+ memory (for small-scale testing)
- **Recommended**: 1-8 GPUs with 40GB+ memory each (for full profiling)
- CUDA-capable GPU (tested on A100, A800, H100)

### 3. Verify Installation

Run the validation script to check if everything is set up correctly:

```bash
python scripts/validate_moe_profiling.py
```

Expected output:
```
============================================================
MoE Profiling Module Validation
============================================================
Testing imports...
  ✓ moe_impl imports successful
  ✓ moe_wrapper imports successful
  ✓ moe_input imports successful
  ✓ main imports successful

Testing module structure...
  ✓ MoEGatingNetwork is nn.Module
  ✓ MoEGatingNetwork has forward method
  ✓ MoEWrapper has all required profiling methods

Testing config generation...
  ✓ Config generated successfully
    - num_tokens_list: 127 values
    - num_experts_per_device_list: [8, 4, 2, 1]
    - tensor_parallel_size_list: [1, 2, 4, 8]

Testing CUDA availability...
  ✓ CUDA is available
    - Device: NVIDIA A100-SXM4-80GB
    - Memory: 80.00 GB

Testing module instantiation...
  ✓ MoEGatingNetwork instantiated successfully
  ✓ MoETokenShuffler instantiated successfully
  ✓ MoEGroupedGEMM instantiated successfully

============================================================
✓ All validation tests passed!
============================================================
```

## Quick Start

### 1. Small-Scale Test (Recommended First)

Run a small-scale test to verify correctness:

```bash
bash frontier/profiling/example/test_profiling_moe.sh --dry-run
```

This will:
- Profile with minimal parameters (num_experts=4, max_tokens=64)
- Use single GPU without Ray
- Generate CSV files under `data/profiling/compute/<device>/<model_name>/`

Expected output:
```
==========================================
MoE Profiling Small-Scale Test
==========================================

Configuration:
  Output root: data/profiling
  Number of GPUs: 1
  Number of experts: 4
  Router top-K: 2
  Hidden dimension: 512
  Expert hidden dimension: 1024
  Max tokens: 64
  Profile method: record_function

=== Profiling MoE Gating ===
Profiling MoE operations: 100%|████████████| 66/66 [00:15<00:00,  4.23it/s]

=== Profiling MoE Shuffling ===
...

=== Profiling MoE Grouped GEMM ===
...

==========================================
Profiling Complete!
==========================================

Output files:
-rw-r--r-- 1 user group 12345 Jan 01 12:00 moe.csv
-rw-r--r-- 1 user group 12345 Jan 01 12:00 moe_kernel_only.csv
-rw-r--r-- 1 user group 12345 Jan 01 12:00 moe_config.yaml
```

### 2. Full Profiling

For production use, run full profiling with realistic parameters:

```bash
python -m frontier.profiling.moe.main \
    --num_gpus 1 \
    --num_experts 8 \
    --router_topk 2 \
    --hidden_dim 4096 \
    --expert_hidden_dim 11008 \
    --max_tokens 4096 \
    --num_tensor_parallel_workers 1 2 4 8 \
    --device a100 \
    --output_dir data/profiling \
    --disable_ray
```

**Note**: For multi-GPU profiling, remove `--disable_ray` and set `--num_gpus` to the desired number.

## Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'sarathi'`

**Solution**: Make sure you're using the sarathi-serve virtual environment:
```bash
source /path/to/sarathi-serve/env/bin/activate
cd /path/to/vidur
python -m frontier.profiling.moe.main --help
```

### Issue 2: CUDA Out of Memory

**Solution**: Reduce profiling parameters:
```bash
python -m frontier.profiling.moe.main \
    --num_gpus 1 \
    --num_experts 4 \
    --max_tokens 512 \
    --hidden_dim 2048 \
    --expert_hidden_dim 5504 \
    --device a100 \
    --disable_ray
```

### Issue 3: Ray initialization fails

**Solution**: Use `--disable_ray` for single-GPU profiling:
```bash
python -m frontier.profiling.moe.main \
    --num_gpus 1 \
    --device a100 \
    --disable_ray \
    ...
```

### Issue 4: Profiling is very slow

**Possible causes**:
1. **Large parameter space**: Reduce `--max_tokens` or limit `--num_tensor_parallel_workers`
2. **Single GPU**: Use multiple GPUs with Ray for parallel profiling
3. **Profile method**: keep default `--profile_method record_function` for kernel-only data, or use `--profile_method cuda_event` for eager-family data

**Solution**:
```bash
# Use multiple GPUs with Ray
python -m frontier.profiling.moe.main \
    --num_gpus 4 \
    --max_tokens 2048 \
    --num_tensor_parallel_workers 1 2 \
    --device a100 \
    ...
```

## Output Validation

After profiling completes, validate the output:

### 1. Check CSV Files

```bash
ls -lh data/profiling/compute/<device>/<model_name>/
```

Expected files:
- `moe.csv`
- `moe_kernel_only.csv` when `--profile_method record_function` is used
- `moe_config.yaml`

### 2. Inspect CSV Content

```bash
head -n 5 data/profiling/compute/a100/<model_name>/moe.csv
```

Expected columns:
```
time_stats.moe_gating_linear.median,time_stats.moe_gating_routing_topk.median,time_stats.moe_shuffling.median,time_stats.moe_grouped_gemm.median,num_tokens,num_experts,num_experts_per_device,expert_parallel_size,routing_runtime_path,routing_assignment_policy,routing_weight_policy,routing_uses_router_logits,gating_runtime_context,gating_runtime_context_impl,router_topk,hidden_dim,expert_hidden_dim,use_gated,num_tensor_parallel_workers,measurement_type
0.1234,0.0100,0.0670,2.2200,8,8,8,1,standard_fused_topk,logit_topk,router_softmax,True,standalone_legacy,linear_plus_routing,2,4096,14336,True,1,CUDA_EVENT
0.2345,0.0050,0.0660,2.1450,16,8,8,1,uniform_topk,round_robin_uniform,uniform,False,prefill_hot,linear_only,2,4096,14336,True,1,CUDA_EVENT
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

1. **Copy profiling data to data directory**:
   ```bash
   # The profiler writes canonical files directly when --output_dir data/profiling is used:
   # data/profiling/compute/<device>/<model_name>/moe.csv
   # data/profiling/compute/<device>/<model_name>/moe_kernel_only.csv
   ```

2. **Test integration with model manager**:
   ```bash
   python tests/profiling/test_model_manager_integration.py
   ```

3. **Run simulator with MoE profiling data**:
   ```bash
   python -m frontier.main \
       --replica_config_model_name "Qwen/Qwen1.5-MoE-A2.7B" \
       --replica_config_total_expert_num 8 \
       --replica_config_router_topk 2 \
       ...
   ```

## Reference

- Main documentation: `vidur/profiling/moe/README.md`
- Profiling guide: `docs/profiling.md`
- Sarathi-serve repository: https://github.com/microsoft/sarathi-serve

