# MoE Profiling Module

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-03-29 | Add the explicit `--routing_runtime_path` profiling control and record the current Phase-A landing: `uniform_topk` gating measurements can now be collected directly, and canonical `moe.csv` rows carry routing-path metadata instead of relying on unlabeled standard-path rows. |
| 2026-03-29 | Add the uniform-routing contract reminder: when runtime enables uniform routing, `moe_gating_routing_topk` profiling/modeling must switch to the `uniform_topk` runtime path instead of reusing standard `fused_topk/topk_softmax` rows. |
| 2026-03-13 | Align README with measurement-aware profiling contract and add explicit `profile_method -> measurement_type` mapping |

**Version**: 2.4 (Updated 2026-06-06)
**Status**: Production Ready ✅

This module profiles MoE (Mixture-of-Experts) compute operations for the Frontier LLM inference simulator.

> **⚠️ Important**: Always use `--disable_ray` flag. Ray mode is currently broken due to grpcio 1.67.1 incompatibility with Ray 2.52.1. See [Troubleshooting](#troubleshooting) for details.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Design Principles](#design-principles)
- [Operations Profiled](#operations-profiled)
- [Parameter Reference](#parameter-reference)
- [Usage Examples](#usage-examples)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Comparison with Linear Op Profiling](#comparison-with-linear-op-profiling)
- [Recent Updates](#recent-updates)

---

## Overview

### Purpose

The MoE profiling module generates timing data for Mixture-of-Experts operations, which is used by Vidur's execution time predictor to simulate MoE model inference performance.

### Key Features

- ✅ **Model-based profiling**: Supports predefined MoE models (Mixtral, Qwen2-MoE, etc.)
- ✅ **Expert Parallelism (EP) support**: Grid-search profiling across multiple EP configurations
- ✅ **Tensor Parallelism (TP) support**: Multi-GPU profiling with different TP sizes
- ✅ **sarathi-serve operators**: Uses optimized CUDA kernels for expert computation
- ✅ **Comprehensive metrics**: Captures min, max, mean, median, std for all operations

> **Measurement Contract**
> - `record_function -> KERNEL_ONLY`
> - `cuda_event -> CUDA_EVENT`
> - pass `--profile_method cuda_event` for simulator-default `moe.csv` datasets
> - pass `--profile_method record_function` for kernel-only `moe_kernel_only.csv` datasets
> - legacy CSVs must be migrated to add explicit `measurement_type` before training
>
> **Uniform-routing contract**
> - If runtime enables uniform routing, the profiling/modeling target must switch to the runtime-equivalent `uniform_topk` path.
> - Do **not** treat standard `fused_topk -> topk_softmax` rows as a valid surrogate for that runtime path.
> - Until the uniform-routing path is profiled explicitly, any `moe_gating_routing_topk` gap should be read as profiling-vs-runtime path mismatch first.
> - Use `--routing_runtime_path uniform_topk` to collect runtime-equivalent gating data and label rows with:
>   - `routing_runtime_path`
>   - `routing_assignment_policy`
>   - `routing_weight_policy`
>   - `routing_uses_router_logits`

### Supported Models

- `mixtral_8x7b_moe`: Mixtral 8x7B (8 experts, topk=2)
- `qwen2_moe_57b_a14b`: Qwen2-MoE 57B (64 experts, topk=8)
- Custom configurations via command-line arguments

---

## Quick Start

### Basic Single-GPU Profiling

```bash
# Profile Mixtral 8x7B with default settings
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- CSV file: `data/profiling/compute/a100/mixtral_8x7b_moe/moe.csv`
- Profiling time: ~2-3 minutes
- Results: ~35 profiling data points

**Note**: The `--device` parameter is **required** and specifies the device SKU (e.g., `a100`, `h100`, `a40`).

**⚠️ Important**: Always use `--disable_ray` flag. Ray mode is currently broken due to grpcio incompatibility.

### Multi-GPU Profiling

```bash
# Profile with 4 GPUs for faster execution
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 4 \
    --max_tokens 1024 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Results: Faster profiling with parallel GPU execution
- Note: Each GPU processes different profiling tasks concurrently

### Multi-EP Grid-Search Profiling

```bash
# Profile with multiple EP configurations
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --expert_parallel_sizes 1 2 4 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Results: ~105 profiling data points (35 per EP size)
- EP=1: 8 experts per device
- EP=2: 4 experts per device
- EP=4: 2 experts per device

---

## Design Principles

### Expert Parallelism (EP) as a Distribution Parameter

**Key insight**: `expert_parallel_size` (EP) is a **distribution parameter**, not a **compute parameter**.

- EP determines how experts are distributed across devices
- EP does NOT change the computation performed by each expert
- We profile with `num_experts_per_device` to capture actual per-device workload

**Example**:
- EP=1: 8 experts on 1 device → `num_experts_per_device=8`
- EP=2: 4 experts per device → `num_experts_per_device=4`
- EP=4: 2 experts per device → `num_experts_per_device=2`

**Benefit**: Profiling data can be reused across different EP configurations.

### Hybrid Operator Approach

**sarathi-serve operators** (for main computation):
- `ColumnParallelLinear`: Expert up projection
- `RowParallelLinear`: Expert down projection
- `SiluAndMul`: Custom CUDA kernel for SwiGLU activation

**Native PyTorch operators** (for lightweight operations):
- `nn.Linear`: Gating network (small operation, avoids distributed setup complexity)

**Rationale**: Balances performance optimization with profiling simplicity.

## Operations Profiled

The MoE profiling module captures timing data for five core operations:

### 1. `moe_gating` (Router/Gating Network)

**What it does**: Computes routing scores for all experts and selects top-K experts per token.

**Implementation**: Native PyTorch `nn.Linear` (hidden_dim → num_experts)

**Profiling parameters**:
- `num_tokens`: Number of input tokens
- `num_experts`: Total number of experts (NOT per-device)
- `router_topk`: Number of experts selected per token
- `hidden_dim`: Model hidden dimension
- `num_tensor_parallel_workers`: Tensor parallelism size

**Typical timing**: ~0.026 ms (very fast, small operation)

**Uniform-routing note**:
- In the standard path, the second scope is logits-driven top-K + normalization.
- If runtime enables uniform routing, the second scope becomes `uniform_topk`-style routing instead.
- In that case, profiling/modeling must use a uniform-routing target and must not silently fall back to the standard routing path.

**Why EP is excluded**: Gating happens before token dispatch. Each token needs routing scores for ALL experts, regardless of how experts are distributed.

### 2. `moe_shuffling` (Local Token Shuffling)

**What it does**: Reorders tokens based on routing decisions (GPU memory operations).

**Implementation**: Native PyTorch tensor operations

**Profiling parameters**:
- `num_tokens`: Number of input tokens
- `num_experts`: Total number of experts
- `router_topk`: Number of experts selected per token

**Typical timing**: ~0.050 ms (lightweight memory operation)

**Note**: This only profiles LOCAL shuffling (within a single GPU). Cross-device shuffling (all-to-all) is handled separately in communication profiling.

### 3. `moe_grouped_gemm` (Expert Computation - Composite)

**What it does**: Executes complete expert FFN computation for the routed tokens.

**Implementation**: Composite operation using sarathi-serve operators.

**Profiling parameters**:
- `num_tokens`: Number of input tokens
- `num_experts_per_device`: Number of experts processed by a single device
- `expert_parallel_size`: Expert parallelism size
- `expert_hidden_dim`: Expert FFN hidden dimension
- `hidden_dim`: Model hidden dimension
- `use_gated`: Whether to use gated FFN (SwiGLU)
- `num_tensor_parallel_workers`: Tensor parallelism size

**Typical timing**: Main MoE computation bottleneck.

**Why use `num_experts_per_device`**: This represents the actual workload per device. When EP=2 with 8 total experts, each device processes 4 experts, so we profile with `num_experts_per_device=4`.

### 4. Runtime Metadata Columns

MoE rows include routing and gating metadata so training can select rows that
match the simulator runtime path:

- `routing_runtime_path`
- `routing_assignment_policy`
- `routing_weight_policy`
- `routing_uses_router_logits`
- `gating_runtime_context`
- `gating_runtime_context_impl`

---

## Parameter Reference

### Required Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--models` | str (list) | MoE model names to profile | `mixtral_8x7b_moe` |
| `--num_gpus` | int | Number of GPUs to use (see Execution Modes) | `1`, `8` |
| `--max_tokens` | int | Maximum number of tokens to profile | `256`, `1024` |
| `--output_dir` | str | Root output directory; the CLI appends `compute/<device>/<model_name>/` | `data/profiling` |
| `--device` | str | Target device SKU | `a100`, `h100`, `a800` |

### Execution Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--disable_ray` | flag | False | **RECOMMENDED**: Disable Ray, use multiprocessing instead |
| `--num_gpus` | int | 8 | Number of GPUs for parallel profiling |

**Execution Modes** (based on `--disable_ray` and `--num_gpus`):

| Mode | Condition | Description |
|------|-----------|-------------|
| Ray Mode | `--disable_ray` not set | Uses Ray actors (⚠️ currently broken due to grpcio incompatibility) |
| Multi-GPU | `--disable_ray` + `--num_gpus > 1` | Uses `ProcessPoolExecutor` with multiple GPUs |
| Single-GPU | `--disable_ray` + `--num_gpus = 1` | Sequential execution on single GPU |

**⚠️ Important**: Ray mode is currently not functional due to grpcio 1.67.1 incompatibility with Ray 2.52.1. Always use `--disable_ray`.

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_tensor_parallel_workers` | int (list) | `[1]` | TP sizes to profile (e.g., `1 2 4 8`) |
| `--expert_parallel_sizes` | int (list) | `[1]` | EP sizes to profile (e.g., `1 2 4`) |
| `--profile_method` | str | `record_function` | Profiling method. `record_function -> KERNEL_ONLY`; `cuda_event -> CUDA_EVENT` |

### Model-Specific Parameters (Auto-Configured)

When using `--models`, these parameters are automatically set from the model config:

| Parameter | Description | Mixtral 8x7B | Qwen2-MoE 57B |
|-----------|-------------|--------------|---------------|
| `num_experts` | Total number of experts | 8 | 64 |
| `router_topk` | Experts selected per token | 2 | 8 |
| `hidden_dim` | Model hidden dimension | 4096 | 5120 |
| `expert_hidden_dim` | Expert FFN hidden dim | 14336 | 13824 |
| `use_gated` | Use gated FFN (SwiGLU) | True | True |

### Advanced Parameters (Rarely Used)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_experts` | int | - | Override model's num_experts |
| `--router_topk` | int | - | Override model's router_topk |
| `--hidden_dim` | int | - | Override model's hidden_dim |
| `--expert_hidden_dim` | int | - | Override model's expert_hidden_dim |
| `--use_gated` | bool | True | Override model's use_gated |

---

## Usage Examples

**⚠️ Note**: All examples use `--disable_ray` flag because Ray mode is currently broken.

### Example 1: Single GPU, Single EP Configuration

**Use case**: Quick profiling for development/testing

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Profiling time: ~2 minutes
- Results: ~35 data points
- CSV file: `data/profiling/compute/a100/mixtral_8x7b_moe/moe.csv`

### Example 2: Multi-EP Grid-Search (Recommended)

**Use case**: Comprehensive EP parameter study

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --expert_parallel_sizes 1 2 4 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Profiling time: ~6 minutes
- Results: ~105 data points (35 per EP size)
- Configurations:
  - EP=1: 8 experts per device
  - EP=2: 4 experts per device
  - EP=4: 2 experts per device

### Example 3: Multi-GPU with TP Scaling

**Use case**: Tensor parallelism performance study

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 8 \
    --num_tensor_parallel_workers 1 2 4 8 \
    --max_tokens 1024 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Profiling time: ~15 minutes
- Results: ~140 data points (35 per TP size)
- Requires: 8 GPUs

### Example 4: Full Grid-Search (TP × EP)

**Use case**: Complete performance characterization

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 8 \
    --num_tensor_parallel_workers 1 2 4 \
    --expert_parallel_sizes 1 2 4 8 \
    --max_tokens 1024 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Profiling time: ~30 minutes
- Results: ~420 data points (3 TP × 4 EP × 35 token sizes)
- Configurations: 12 (TP, EP) combinations

### Example 5: Multiple Models

**Use case**: Profile multiple MoE models in one run

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe qwen2_moe_57b_a14b \
    --num_gpus 1 \
    --expert_parallel_sizes 1 2 4 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Profiling time: ~12 minutes
- Results: ~210 data points (105 per model)
- Separate CSV files for each model

### Example 6: Custom Model Configuration

**Use case**: Profile a custom MoE model not in the predefined list

```bash
python -m frontier.profiling.moe.main \
    --num_gpus 1 \
    --num_experts 16 \
    --router_topk 4 \
    --hidden_dim 8192 \
    --expert_hidden_dim 28672 \
    --use_gated True \
    --expert_parallel_sizes 1 2 4 \
    --max_tokens 256 \
    --disable_ray \
    --profile_method cuda_event \
    --device a100 \
    --output_dir data/profiling
```

**Note**: When using custom parameters, `--models` is not required.

---

## Output Format

### Directory Structure

```
data/profiling/
└── compute/
    └── {device}/                      # Device SKU (e.g., a100, h100, rtx_pro_6000)
        ├── moe_config.yaml            # Profiling configuration
        └── {model_name}/              # Model name (e.g., mixtral_8x7b_moe)
            ├── moe.csv                # CUDA-event profiling results
            └── moe_kernel_only.csv    # Kernel-only profiling results
```

Canonical CSV path schema:
`data/profiling/compute/<device>/<model_name>/<op_name>.csv`. Pass
`--output_dir data/profiling`; the CLI appends `compute/<device>/<model_name>/`
internally.

**Example**:
```
data/profiling/compute/a100/
├── moe_config.yaml
└── mixtral_8x7b_moe/
    ├── moe.csv
    └── moe_kernel_only.csv
```

**Note**: This structure matches the attention and linear-op profiling modules for consistency.

### CSV File Format

**Files**:

- `<model_name>/moe.csv`: CUDA-event profiling results.
- `<model_name>/moe_kernel_only.csv`: kernel-only profiling results.

#### Timing Statistics

For each operation (`moe_gating_linear`, `moe_gating_routing_topk`,
`moe_shuffling`, `moe_grouped_gemm`):

- `time_stats.<operation>.min`: Minimum time (ms)
- `time_stats.<operation>.max`: Maximum time (ms)
- `time_stats.<operation>.mean`: Mean time (ms)
- `time_stats.<operation>.median`: Median time (ms)
- `time_stats.<operation>.std`: Standard deviation (ms)

#### Configuration and Runtime Metadata

- `num_tokens`: Number of input tokens
- `num_experts`: Total number of experts
- `num_experts_per_device`: Experts per device (= num_experts / expert_parallel_size)
- `expert_parallel_size`: Expert parallelism size
- `routing_runtime_path`: Routing implementation profiled (`standard_fused_topk` or `uniform_topk`)
- `routing_assignment_policy`: Expert assignment policy used by the routing path
- `routing_weight_policy`: Expert weight policy used by the routing path
- `routing_uses_router_logits`: Whether the routing path consumes router logits
- `gating_runtime_context`: Runtime context for the gating measurement
- `gating_runtime_context_impl`: Concrete implementation used for the context
- `router_topk`: Experts selected per token
- `hidden_dim`: Model hidden dimension
- `expert_hidden_dim`: Expert FFN hidden dimension
- `use_gated`: Whether using gated FFN (True/False)
- `num_tensor_parallel_workers`: Tensor parallelism size
- `measurement_type`: `CUDA_EVENT` or `KERNEL_ONLY`

### Sample CSV Data

```csv
time_stats.moe_gating_linear.mean,time_stats.moe_gating_routing_topk.mean,time_stats.moe_shuffling.mean,time_stats.moe_grouped_gemm.mean,...,num_tokens,num_experts,num_experts_per_device,expert_parallel_size,routing_runtime_path,routing_assignment_policy,routing_weight_policy,routing_uses_router_logits,gating_runtime_context,gating_runtime_context_impl,router_topk,hidden_dim,expert_hidden_dim,use_gated,num_tensor_parallel_workers,measurement_type
0.027,0.010,0.067,2.220,...,256,8,8,1,standard_fused_topk,logit_topk,router_softmax,True,standalone_legacy,linear_plus_routing,2,4096,14336,True,1,CUDA_EVENT
0.026,0.005,0.050,0.552,...,256,8,2,4,uniform_topk,round_robin_uniform,uniform,False,prefill_hot,linear_only,2,4096,14336,True,1,CUDA_EVENT
```

### Interpreting Results

#### EP Scaling Analysis

**Question**: How does EP affect performance?

**Method**: Filter CSV by `expert_parallel_size` and compare `time_stats.moe_grouped_gemm.mean`:

```python
import pandas as pd

df = pd.read_csv("moe.csv")

for ep_size in [1, 2, 4]:
    ep_df = df[df['expert_parallel_size'] == ep_size]
    avg_time = ep_df['time_stats.moe_grouped_gemm.mean'].mean()
    print(f"EP={ep_size}: Avg grouped_gemm time = {avg_time:.3f} ms")
```

**Expected output**:
```
EP=1: Avg grouped_gemm time = 1.544 ms
EP=2: Avg grouped_gemm time = 0.864 ms  (44% reduction)
EP=4: Avg grouped_gemm time = 0.552 ms  (64% reduction)
```

**Interpretation**: Grouped GEMM time decreases with higher EP because each device processes fewer experts.

#### TP Scaling Analysis

**Question**: How does TP affect performance?

**Method**: Filter CSV by `num_tensor_parallel_workers` and compare timing:

```python
for tp_size in [1, 2, 4, 8]:
    tp_df = df[df['num_tensor_parallel_workers'] == tp_size]
    avg_time = tp_df['time_stats.moe_grouped_gemm.mean'].mean()
    print(f"TP={tp_size}: Avg grouped_gemm time = {avg_time:.3f} ms")
```

**Interpretation**: TP affects per-expert projection times due to matrix sharding and communication overhead.

---

## Troubleshooting

### Common Issues

#### Issue 1: Ray Mode Crashes (grpcio Incompatibility) ⚠️

**Error**: `UnknownError: UNKNOWN:ipv4:127.0.0.1:xxxxx: Trying to connect an http1.x server`

**Cause**: grpcio 1.67.1 is incompatible with Ray 2.52.1's dashboard_agent.

**Solution**: **Always use `--disable_ray` flag.**

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --device a100 \
    --disable_ray \  # Required!
    --output_dir data/profiling
```

**Note**: This is a known issue. Ray mode is not functional until Ray/grpcio compatibility is resolved.

#### Issue 2: `ModuleNotFoundError: No module named 'flashinfer'`

**Cause**: flashinfer is required by sarathi-serve operators but not installed.

**Solution**:

```bash
pip install flashinfer
```

**Alternative**: Use system Python with flashinfer pre-installed:

```bash
# Check if flashinfer is available
python -c "import flashinfer; print('flashinfer available')"
```

#### Issue 3: `ray.exceptions.ActorDiedError` (Legacy Issue)

**Cause**: Ray actor crashed during profiling, often due to GPU OOM or CUDA errors.

**Solution**: Use `--disable_ray` flag instead of Ray mode. If you must use Ray:

1. Reduce `--max_tokens` (e.g., from 1024 to 256)
2. Reduce number of configurations (fewer TP/EP sizes)
3. Check GPU memory: `nvidia-smi`
4. Check Ray logs for detailed error messages

#### Issue 4: `AttributeError: type object 'MetricsStore' has no attribute '_instance'`

**Cause**: This error should not occur in the current version (fixed via monkey patching).

**Solution**: Ensure you're using the latest version of the code. If the error persists:

```bash
# Verify monkey patching is present in moe_wrapper.py
grep -A 3 "sarathi.metrics.cuda_timer.CudaTimer" frontier/profiling/moe/moe_wrapper.py
```

#### Issue 5: CSV file is empty or has fewer results than expected

**Cause**: Profiling may have failed silently for some configurations.

**Solution**:

1. Check the terminal output for error messages
2. Verify all (TP, EP) combinations completed:

   ```python
   import pandas as pd
   df = pd.read_csv("moe.csv")
   print(df.groupby(['num_tensor_parallel_workers', 'expert_parallel_size']).size())
   ```

3. Re-run profiling with verbose logging

#### Issue 6: Profiling is very slow

**Cause**: Large `--max_tokens` or many (TP, EP) combinations.

**Solution**:

1. Reduce `--max_tokens` for faster profiling (256 is usually sufficient)
2. Profile fewer configurations initially
3. Keep the default `--profile_method record_function` for kernel-only / decode-graph training data
4. Use multi-GPU mode: `--num_gpus 4 --disable_ray`

### Performance Tips

1. **Start small**: Use `--max_tokens 256` for initial testing
2. **Grid-search wisely**: Profile EP first (single GPU), then add TP (multi-GPU)
3. **Use model presets**: `--models mixtral_8x7b_moe` is easier than manual parameters
4. **Monitor GPU memory**: Use `nvidia-smi` to ensure no OOM issues
5. **Choose the measurement family intentionally**: default `record_function` writes `KERNEL_ONLY`; pass `--profile_method cuda_event` when collecting eager / `CUDA_EVENT` data

---

## Comparison with Linear Op Profiling

### Similarities

- Both use sarathi-serve operators (`ColumnParallelLinear`, `RowParallelLinear`)
- Both support TP scaling profiling
- Both generate CSV files with timing statistics
- Both use Ray for parallel profiling

### Differences

| Aspect | Linear Op Profiling | MoE Profiling |
|--------|---------------|---------------|
| **Operations** | Attention, FFN, LayerNorm | Gating, Shuffling, Expert FFN |
| **Parallelism** | TP only | TP + EP |
| **Key Parameter** | `num_layers` | `num_experts`, `router_topk` |
| **Gating Network** | N/A | Native PyTorch (not TP-parallelized) |
| **Expert Computation** | N/A | sarathi-serve operators |
| **CSV Output** | Separate files per operation | Single consolidated file |
| **EP Support** | N/A | ✅ Grid-search across EP sizes |

### When to Use Which

- **Linear Op Profiling**: For standard transformer models (GPT, LLaMA, etc.)
- **MoE Profiling**: For Mixture-of-Experts models (Mixtral, Qwen2-MoE, etc.)

---

## Recent Updates

### Version 2.3 (2026-03-13)

**Major Changes**:

1. ✅ **Measurement-aware contract documented explicitly**
   - `record_function` now maps to predictor-training family `KERNEL_ONLY`
   - `cuda_event` maps to predictor-training family `CUDA_EVENT`
   - `kineto` / `perf_counter` remain debug-only and are not valid training inputs

2. ✅ **Current documented default is `record_function`**
   - This matches current CLI behavior in `frontier/profiling/moe/main.py`
   - The default is appropriate for decode CUDA graph kernel-only datasets
   - Eager-family collection must opt in with `--profile_method cuda_event`

3. ✅ **Migration requirement clarified**
   - Legacy CSVs without `measurement_type` must be migrated explicitly
   - Migration must pass `--measurement_type`; no heuristic guessing is allowed

**Operator Guidance**:

- Use default `record_function` when collecting kernel-only / decode-graph data
- Use `--profile_method cuda_event` when collecting eager-family data
- Keep MoE fused-kernel guidance unchanged; measurement family selection is orthogonal

### Version 2.1 (2025-12-03)

**Major Changes**:

1. ✅ **Non-Ray Multi-GPU Support**: Added `torch.multiprocessing` alternative to Ray
   - Uses `ProcessPoolExecutor` with spawn context for CUDA compatibility
   - Each GPU runs in separate process with GPU binding via `CUDA_VISIBLE_DEVICES`
   - Added `--disable_ray` flag to enable non-Ray mode

2. ⚠️ **Ray Mode Deprecated**: Ray mode is currently broken due to grpcio incompatibility
   - grpcio 1.67.1 causes raylet crashes with Ray 2.52.1
   - Always use `--disable_ray` flag until compatibility is resolved

3. ✅ **vLLM 0.3.x Compatibility Code Removed**: Cleaned up legacy code
   - Removed all `if VLLM_API_VERSION == "0.3.x"` branches
   - vLLM 0.10.x is now the only supported version

4. ✅ **ModelConfig Serialization**: Added `to_dict()` method for multiprocessing
   - Enables ModelConfig to be passed across process boundaries
   - Located in `frontier/profiling/common/model_config.py`

**New Functions**:

- `_worker_init(gpu_id)`: Initialize worker process with GPU binding
- `_worker_profile_task(task_args)`: Worker function for multiprocessing profiling
- `_get_available_gpus()`: Get list of available GPU IDs from environment

**Known Issues**:

- Ray mode: Not functional due to grpcio 1.67.1 incompatibility
- Multi-GPU mode: Higher per-task overhead due to process creation

See `CHANGELOG_MULTI_GPU.md` for detailed implementation notes.

### Version 2.0 (2025-10-07)

**Major Changes**:

1. ✅ **EP Parameter Support**: Added `--expert_parallel_sizes` for grid-search profiling
   - Enables profiling multiple EP configurations in one run
   - CSV output includes `expert_parallel_size` column
   - Automatic calculation of `num_experts_per_device`

2. ✅ **sarathi-serve Operator Migration**: Migrated expert FFNs to use optimized operators
   - `ColumnParallelLinear` for expert up projection
   - `RowParallelLinear` for expert down projection
   - `SiluAndMul` custom CUDA kernel for SwiGLU activation
   - Hybrid approach: native PyTorch for gating, sarathi-serve for experts

3. ✅ **Consolidated CSV Output**: All operations now in single CSV file
   - Easier data analysis and visualization
   - Consistent column naming across operations
   - Includes per-expert timing breakdown

4. ✅ **Model-Based Configuration**: Support for predefined MoE models
   - `--models mixtral_8x7b_moe qwen2_moe_57b_a14b`
   - Auto-configures all model-specific parameters
   - Simplifies usage for common models

5. ✅ **Enhanced Testing**: Comprehensive integration test suite
   - `tests/test_moe_profiling_refactor.sh`
   - Validates EP parameter support
   - Validates sarathi-serve operator migration
   - Automated CSV output validation

**Bug Fixes**:

- Fixed MetricsStore singleton issue via monkey patching
- Fixed parallel state initialization issue with hybrid operator approach
- Improved error handling and logging

**Performance**:

- EP=4 provides 64% reduction in grouped GEMM time vs EP=1
- sarathi-serve operators validated and working correctly
- No performance regressions observed

### Version 1.0 (Original)

- Basic MoE profiling support
- Native PyTorch operators
- Manual parameter configuration
- Separate CSV files per operation

---

## Testing

### Integration Test

**Recommended**: Run the comprehensive integration test to validate the installation:

```bash
bash tests/test_moe_profiling_refactor.sh
```

**What it tests**:
- Python environment and dependencies
- EP parameter support (TP=1,2 × EP=1,2,4)
- CSV output format and columns
- EP calculation correctness
- sarathi-serve operator functionality

**Expected output**:
- ✅ All 7 validation steps pass
- ✅ 210 profiling results generated
- ✅ CSV file with all required columns
- ✅ All timing data valid

**Duration**: ~2-3 minutes

### Unit Tests

```bash
# Run MoE-specific unit tests (if available)
python -m pytest tests/profiling/test_moe*.py -v
```

### Manual Smoke Test

**Quick validation** (30 seconds):

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --max_tokens 64 \
    --device a100 \
    --output_dir /tmp/moe_test
```

**Expected**: CSV file generated with ~10 rows, no errors.

---

## Integration with Vidur Simulator

### Execution Time Predictor

The generated CSV files are consumed by `vidur/execution_time_predictor/sklearn_moe_execution_time_predictor.py`.

**Workflow**:

1. **Load Profiling Data**: CSV files are loaded during predictor initialization
2. **Filter by Configuration**: Data is filtered based on model config (num_experts, router_topk, etc.)
3. **Train ML Models**: sklearn models (RandomForest/LinearRegression) are trained on the data
4. **Generate Prediction Caches**: Fast lookup tables for simulation

**Example**:
```python
from frontier.execution_time_predictor import SklearnMoEExecutionTimePredictor

predictor = SklearnMoEExecutionTimePredictor(
    model_config=model_config,
    replica_config=replica_config,
)

# Predict execution time for a batch
execution_time = predictor.get_execution_time(
    batch=batch,
    replica_id=0,
)
```

### Model Manager

The `ExecutionTimePredictionModelManager` handles centralized model training:

- Trains models once and caches to disk
- Reuses cached models across multiple simulations
- Supports disaggregated architectures (prefill, decode-attn, decode-ffn clusters)

---

## Hardware Requirements

### Minimum (Development/Testing)

- **GPUs**: 1 GPU with 16GB+ memory
- **Use case**: Quick profiling, EP parameter studies
- **Example**: `--num_gpus 1 --max_tokens 256`

### Recommended (Production)

- **GPUs**: 8 GPUs with 40GB+ memory each
- **Use case**: Full TP × EP grid-search profiling
- **Example**: `--num_gpus 8 --num_tensor_parallel_workers 1 2 4 8 --expert_parallel_sizes 1 2 4 8`

### GPU Memory Estimation

**Per-GPU memory usage** (approximate):

- Mixtral 8x7B: ~8-12 GB (depends on max_tokens)
- Qwen2-MoE 57B: ~20-30 GB (larger model)

**Tip**: Use `nvidia-smi` to monitor GPU memory during profiling.

---

## Design Rationale

### Why EP is a Distribution Parameter

**Key insight**: Expert Parallelism (EP) determines how experts are distributed across devices, but does NOT change the computation performed by each expert.

**Implications**:

1. **Profiling Strategy**: Profile with different `num_experts_per_device` values instead of different EP configurations
2. **Data Reusability**: Profiling data for `num_experts_per_device=4` applies to:
   - EP=1 with 4 total experts
   - EP=2 with 8 total experts
   - EP=4 with 16 total experts
3. **Hardware Efficiency**: Can profile on single GPU and reuse data for multi-GPU EP configurations

**Validation**: Integration test confirms EP scaling behavior matches expectations (64% reduction in grouped GEMM time from EP=1 to EP=4).

### Why TP is a Compute Parameter

**Tensor Parallelism (TP)** is fundamentally different from EP:

- TP **changes computation patterns** (matrix sharding, different GEMM dimensions)
- TP **affects per-device computation time** (communication overhead, smaller matrices)
- TP **requires multi-GPU profiling** to capture communication costs

**Therefore**: TP must be profiled separately for each TP size.

### Hybrid Operator Approach

**Design Decision**: Use native PyTorch for gating, sarathi-serve for expert FFNs.

**Rationale**:

1. **Gating Network**: Small operation (hidden_dim → num_experts), doesn't benefit from TP
2. **Expert FFNs**: Large operations (hidden_dim → expert_hidden_dim), benefit from optimized CUDA kernels
3. **Profiling Simplicity**: Avoids needing distributed initialization for single-GPU profiling
4. **Performance**: Main computation still uses sarathi-serve operators

**Trade-off**: Gating network not TP-parallelized, but this is acceptable since it's a small operation (~0.026 ms).
