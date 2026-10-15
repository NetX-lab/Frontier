# MoE Profiling Module

**Version**: 2.4 (Updated 2026-10-14)

This module profiles routed-expert compute for the Frontier LLM inference
simulator. Shared-expert FFN projections remain in the `linear_op` module.

Use `--disable_ray` for the supported local profiling path. The Ray path
failed with grpcio 1.67.1 and Ray 2.52.1, and `environment_profiling.yml` does
not pin Ray; see [Troubleshooting](#troubleshooting).

For a dry-run of the public wrapper, which needs no GPU, run
`bash examples/profiling/profile_moe.sh --dry-run`.

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

The MoE profiling module generates timing data for Mixture-of-Experts operations. Frontier's execution time predictor trains on this data to simulate MoE model inference performance.

### Key Features

- Model-based profiling for predefined and custom MoE configurations.
- Expert Parallelism (EP) grid search for routed-expert placement.
- Tensor Parallelism (TP) grid search for routed-expert weight sharding.
- Timing statistics (min, max, mean, median, and standard deviation).

> **Measurement Contract**
> - `record_function -> KERNEL_ONLY`
> - `cuda_event -> CUDA_EVENT`
> - pass `--profile_method cuda_event` for simulator-default `moe.csv` datasets
> - pass `--profile_method record_function` (the default) for kernel-only `moe_kernel_only.csv` datasets
> - on ROCm, pass `--profile_method device_event`; it writes `moe_device_event.csv`
> - `cuda` and `kernel_only` are accepted aliases of `cuda_event` and `record_function`
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

`--models` takes model names resolved by `ModelConfig.from_model_name`: a
registered model name or the file name of a config JSON under
`data/config/models/`. The model must be a MoE model. The default list is:

- `mixtral_8x7b_moe`: Mixtral 8x7B (8 experts, topk=2)
- `qwen2_moe_example`: Qwen2-MoE example (60 experts, topk=4)

Other checked-in MoE configs include `Qwen3-30B-A3B-tiny`,
`Phi-tiny-MoE-instruct`, and `Qwen3-235B-A22B`. To profile a
custom MoE shape, add a HuggingFace-style config JSON under
`data/config/models/` and pass its file name without `.json`.

### Row Count

Each model's CSV has one row per profiled case:

```text
rows = token points x TP sizes x EP sizes x cases per (TP, EP, tokens) point
```

- Token points come from `--max_tokens` (35 points for 256, 131 for 1024,
  259 for 4096), `--extra_num_tokens`, or `--num_tokens_list`.
- TP sizes default to `1 2 4 8`.
- EP sizes default to every divisor of the model's `num_experts`.
- Cases per point are `len(--load_distributions) x --num_samples_per_distribution`
  (default 1 x 3 = 3). With `--disable_load_imbalance`, there is one case per
  point.

---

## Quick Start

### Basic Single-GPU Profiling

```bash
# Profile Mixtral 8x7B at TP=1 and EP=1
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --expert_parallel_sizes 1 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- CSV file you produce: `data/profiling/compute/a100/mixtral_8x7b_moe/moe.csv`
- Rows: 105 (35 token points x 3 samples of the `uniform` distribution)

**Note**: The `--device` parameter is **required**. It is the device label used
in the output path (e.g., `a100`, `h100`, `a40`).

**⚠️ Important**: Always use the `--disable_ray` flag. Ray mode failed with
grpcio 1.67.1 and Ray 2.52.1.

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

- Each GPU processes different profiling tasks concurrently.
- Rows: 6,288. This command keeps the default TP sizes (`1 2 4 8`) and EP
  sizes (every divisor of 8: `1 2 4 8`), so it profiles 131 token points x 4 TP
  sizes x 4 EP sizes x 3 samples.

### Multi-EP Grid-Search Profiling

```bash
# Profile with multiple EP configurations
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --expert_parallel_sizes 1 2 4 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Rows: 315 (35 token points x 3 EP sizes x 3 samples)
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

Load-imbalance profiling is enabled by default, and it selects the vLLM path:

- Expert computation uses vLLM's fused MoE kernel
  (`moe_grouped_gemm_backend=vllm_fused`, or `vllm_aiter_mxfp4` for MXFP4).
- Gating uses vLLM `ReplicatedLinear` and vLLM `fused_topk` (or `uniform_topk`).

`--disable_load_imbalance` selects the Frontier per-expert loop
(`moe_grouped_gemm_backend=frontier_loop`):

- `ColumnParallelLinear`: Expert up projection
- `RowParallelLinear`: Expert down projection
- `SiluAndMul`: Custom CUDA kernel for SwiGLU activation
- Gating uses PyTorch `nn.Linear` and `torch.topk`. This path supports only
  `--routing_runtime_path standard_fused_topk` and rejects FP8 or MXFP4
  grouped GEMM.

**Rationale**: The vLLM path matches the runtime kernels; the loop path keeps a
simple reference implementation.

## Operations Profiled

The MoE profiling module captures timing data for four routed-expert
operations/scopes:

### 1. `moe_gating_linear` and `moe_gating_routing_topk` (Router/Gating)

**What it does**: Computes routing scores for all experts and selects top-K experts per token.

**Implementation**: vLLM `ReplicatedLinear` (hidden_dim -> num_experts) by
default, with native PyTorch `nn.Linear` when `--disable_load_imbalance`
disables the vLLM path.

**Profiling parameters**:
- `num_tokens`: Number of input tokens
- `num_experts`: Total number of experts (NOT per-device)
- `router_topk`: Number of experts selected per token
- `hidden_dim`: Model hidden dimension
- `num_tensor_parallel_workers`: Requested MoE TP grid value recorded in the
  row. Runtime admission follows the operator registry: co-location uses the
  MoE TP key, while auxiliary gating/routing operators may use the TP=1
  compatibility key on deferred disaggregated clusters.

**Typical timing**: ~0.026 ms (very fast, small operation)

**Uniform-routing note**:
- In the standard path, the second scope is logits-driven top-K + normalization.
- If runtime enables uniform routing, the second scope becomes `uniform_topk`-style routing instead.
- In that case, profiling/modeling must use a uniform-routing target and must not silently fall back to the standard routing path.

**Why EP is excluded**: Gating happens before token dispatch. Each token needs routing scores for ALL experts, regardless of how experts are distributed.

### 2. `moe_shuffling` (Local Token Shuffling)

**What it does**: Reorders tokens based on routing decisions (GPU memory operations).

**Implementation**: vLLM `moe_align_block_size` for alignment and local token
reordering, followed by native PyTorch tensor operations for the remaining
local gather steps.

**Profiling parameters**:
- `num_tokens`: Number of input tokens
- `num_experts`: Total number of experts
- `router_topk`: Number of experts selected per token

**Typical timing**: ~0.050 ms (lightweight memory operation)

**Note**: This only profiles LOCAL shuffling (within a single GPU). Cross-device shuffling (all-to-all) is handled separately in communication profiling.

### 3. `moe_grouped_gemm` (Expert Computation - Composite)

**What it does**: Executes complete expert FFN computation for the routed tokens.

**Implementation**: vLLM's fused MoE kernel by default. With
`--disable_load_imbalance`, a per-expert loop over Frontier profiling layers.

**Profiling parameters**:
- `num_tokens`: Number of input tokens
- `num_experts_per_device`: Number of experts processed by a single device
- `expert_parallel_size`: Expert parallelism size
- `expert_hidden_dim`: Expert FFN hidden dimension
- `hidden_dim`: Model hidden dimension
- `use_gated`: Whether the model uses a gated FFN (SwiGLU); taken from the
  model config's `use_gated_mlp`
- `num_tensor_parallel_workers`: Requested TP grid value recorded in the row

**Typical timing**: Main MoE computation bottleneck.

**Why use `num_experts_per_device`**: This represents the actual workload per device. When EP=2 with 8 total experts, each device processes 4 experts, so we profile with `num_experts_per_device=4`.

The routed-expert TP value is `--num_tensor_parallel_workers`; it partitions
the expert hidden dimension. The EP value is `--expert_parallel_sizes`; it
determines expert placement and therefore `num_experts_per_device`. These
parameters do not describe shared-expert linear work.

### 4. Runtime Metadata Columns

MoE rows include routing and gating metadata so training can select rows that
match the simulator runtime path:

- `routing_runtime_path`
- `routing_assignment_policy`
- `routing_weight_policy`
- `routing_uses_router_logits`
- `gating_runtime_context`
- `gating_runtime_context_impl`
- `model_architecture_profile`: File-level architecture identity.
- `typed_operator_contracts`: A JSON mapping keyed by operator name, written on
  each row by the current profiler. Each entry records the operator's owner and
  its TP/EP contract.
- `moe_grouped_gemm_backend`: `vllm_fused`, `vllm_aiter_mxfp4`, or
  `frontier_loop`.
- `moe_quantization_mode`: `bf16`, `fp8`, or `mxfp4`.

---

## Parameter Reference

All flags are defined in `frontier/profiling/moe/main.py::parse_args`.

### Core Parameters

`--device` is the only required flag.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--device` | str | required | Device label used in the output path (e.g., `a100`, `h100`, `a800`) |
| `--models` | str (list) | `mixtral_8x7b_moe qwen2_moe_example` | MoE model names to profile |
| `--max_tokens` | int | `4096` | Largest token count in the default token grid |
| `--output_dir` | str | `data/profiling` | Root output directory; the CLI appends `compute/<device>/<model_name>/` |

### Execution Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--disable_ray` | flag | False | **RECOMMENDED**: Disable Ray, use multiprocessing instead |
| `--num_gpus` | int | 8 | Number of GPUs for parallel profiling |

**Execution Modes** (based on `--disable_ray` and `--num_gpus`):

| Mode | Condition | Description |
|------|-----------|-------------|
| Ray Mode | `--disable_ray` not set | Uses Ray actors (⚠️ failed with grpcio 1.67.1 and Ray 2.52.1) |
| Multi-GPU | `--disable_ray` + `--num_gpus > 1` | Uses one `ProcessPoolExecutor` (spawn context) per GPU |
| Single-GPU | `--disable_ray` + `--num_gpus = 1` | Sequential execution on single GPU |

**⚠️ Important**: Ray mode failed with grpcio 1.67.1 and Ray 2.52.1. Always use `--disable_ray`.

### Profiling Grid Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_tensor_parallel_workers` | int (list) | `1 2 4 8` | Routed-expert TP sizes to profile |
| `--expert_parallel_sizes` | int (list) | every divisor of `num_experts` | EP sizes to profile; each must divide `num_experts` |
| `--extra_num_tokens` | int (list) | none | Extra token counts added to the `--max_tokens` grid |
| `--num_tokens_list` | int (list) | none | Exact token counts; replaces the grid and cannot be combined with `--extra_num_tokens` |
| `--profile_method` | str | `record_function` | `record_function -> KERNEL_ONLY`; `cuda_event -> CUDA_EVENT`; `device_event -> DEVICE_EVENT` (ROCm) |

### Load-Imbalance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable_load_imbalance` | flag | enabled | Profile sampled expert loads with the vLLM fused kernel |
| `--disable_load_imbalance` | flag | - | Use the Frontier per-expert loop and one uniform case per point |
| `--load_distributions` | str (list) | `uniform` | Any of `uniform`, `balanced`, `skewed`, `extremely_skewed` |
| `--num_samples_per_distribution` | int | `3` | Random samples per distribution |

See [LOAD_IMBALANCE_GUIDE.md](./LOAD_IMBALANCE_GUIDE.md).

### Routing and Precision Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--routing_runtime_path` | str | `standard_fused_topk` | `standard_fused_topk` or `uniform_topk` |
| `--gating_runtime_context` | str | `standalone_legacy` | `standalone_legacy` or `prefill_hot` |
| `--precision` / `--dtype` | str | model config dtype | Must match the model config dtype |
| `--use_fp8` | flag | model config | Must match the model config quantization |
| `--block_shape` | 2 ints | model config | Must match the model config `weight_block_size` |
| `--per_channel_quant` | flag | off | Per-channel instead of per-tensor FP8 scales |
| `--yes`, `-y` | flag | off | Skip the interactive confirmation |
| `--disable_replicated` | flag | off | Deprecated; has no effect for MoE profiling |

### Model-Specific Values

These values come from the model config. The CLI has no flags to override
them; to profile another shape, add a model config JSON as described in
[Supported Models](#supported-models).

| Value | Description | `mixtral_8x7b_moe` | `qwen2_moe_example` |
|-------|-------------|--------------------|---------------------|
| `num_experts` | Total number of experts | 8 | 60 |
| `router_topk` | Experts selected per token | 2 | 4 |
| `hidden_dim` | Model hidden dimension | 4096 | 2048 |
| `expert_hidden_dim` | Expert FFN hidden dim (`moe_intermediate_size`) | 14336 | 1408 |
| `use_gated` | Use gated FFN (SwiGLU) | True | True |

---

## Usage Examples

**⚠️ Note**: All examples use the `--disable_ray` flag because Ray mode failed
with grpcio 1.67.1 and Ray 2.52.1. Row counts follow the
[Row Count](#row-count) formula with the default 3 load-imbalance samples per
point. All CSV paths below are files that you produce.

### Example 1: Single GPU, Single TP and EP Configuration

**Use case**: Quick profiling for development/testing

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --expert_parallel_sizes 1 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Rows: 105 (35 token points x 3 samples)
- CSV file: `data/profiling/compute/a100/mixtral_8x7b_moe/moe.csv`

### Example 2: Multi-EP Grid-Search (Recommended)

**Use case**: Comprehensive EP parameter study

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --expert_parallel_sizes 1 2 4 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Rows: 315 (35 token points x 3 EP sizes x 3 samples)
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
    --expert_parallel_sizes 1 \
    --max_tokens 1024 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Rows: 1,572 (131 token points x 4 TP sizes x 3 samples)
- Requires: 8 visible GPUs

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

- Rows: 4,716 (131 token points x 3 TP sizes x 4 EP sizes x 3 samples)
- Configurations: 12 (TP, EP) combinations

### Example 5: Multiple Models

**Use case**: Profile multiple MoE models in one run

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe qwen2_moe_example \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --expert_parallel_sizes 1 2 4 \
    --max_tokens 256 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir data/profiling
```

**Expected output**:

- Rows: 315 per model (35 token points x 3 EP sizes x 3 samples)
- One CSV file per model; every EP size must divide each model's `num_experts`
  (8 and 60 here)

### Example 6: Custom Model Configuration

**Use case**: Profile a custom MoE model not in the predefined list

The CLI has no flags for `num_experts`, `router_topk`, `hidden_dim`,
`expert_hidden_dim`, or `use_gated`. Write a HuggingFace-style config JSON, for
example `data/config/models/my_moe.json`, with `num_experts`,
`num_experts_per_tok`, `hidden_size`, `moe_intermediate_size`, and `hidden_act`
(`silu` selects the gated FFN), plus the fields every config needs
(`num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
`max_position_embeddings`, `vocab_size`). Then pass its file name:

```bash
python -m frontier.profiling.moe.main \
    --models my_moe \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --expert_parallel_sizes 1 2 4 \
    --max_tokens 256 \
    --disable_ray \
    --profile_method cuda_event \
    --device a100 \
    --output_dir data/profiling
```

Use an existing MoE config under `data/config/models/` as the template; the
loader fails fast when a required field is missing.

---

## Output Format

### Directory Structure

```
data/profiling/
└── compute/
    └── {device}/                      # Device SKU (e.g., a100, h100, rtx_pro_6000)
        ├── moe_config.yaml            # Arguments of the last run and resolved EP sizes
        └── {model_name}/              # Model name (e.g., mixtral_8x7b_moe)
            ├── moe.csv                # --profile_method cuda_event
            ├── moe_kernel_only.csv    # --profile_method record_function (default)
            └── moe_device_event.csv   # --profile_method device_event (ROCm)
```

One run writes one CSV per model, selected by `--profile_method`. The profiler
also creates `<output_dir>/profiler_traces/`.

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
- `<model_name>/moe_device_event.csv`: ROCm device-event profiling results.

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
- `measurement_type`: `CUDA_EVENT`, `KERNEL_ONLY`, or `DEVICE_EVENT`
- `profiling_precision`, `model_arch`, `model_architecture_profile`,
  `quant_signature`: File-level metadata
- `typed_operator_contracts`, `moe_grouped_gemm_backend`,
  `moe_quantization_mode`: See [Runtime Metadata Columns](#4-runtime-metadata-columns)

#### Load-Imbalance Columns

Every row also carries the grouped-GEMM load features from
`frontier/moe_load_imbalance.py::MoELoadImbalanceInput.to_features_dict`:
`total_routed_tokens`, `model_expansion_ratio`, `tokens_per_expert_avg`,
`tokens_to_experts_ratio`, `expert_utilization`, `min_load_ratio`,
`load_imbalance_cv`, `max_load_ratio`, `load_entropy`, `load_gini_coefficient`,
`load_distribution`, and `seed`. The vLLM fused-kernel path also writes
`time_stats.moe_grouped_gemm.block_size_m`,
`time_stats.moe_grouped_gemm.num_tokens_post_padded`, and
`time_stats.moe_grouped_gemm.block_size_m_ranges`.

### Sample CSV Data

```csv
time_stats.moe_gating_linear.mean,time_stats.moe_gating_routing_topk.mean,time_stats.moe_shuffling.mean,time_stats.moe_grouped_gemm.mean,...,num_tokens,num_experts,num_experts_per_device,expert_parallel_size,routing_runtime_path,routing_assignment_policy,routing_weight_policy,routing_uses_router_logits,gating_runtime_context,gating_runtime_context_impl,router_topk,hidden_dim,expert_hidden_dim,use_gated,num_tensor_parallel_workers,measurement_type
0.027,0.010,0.067,2.220,...,256,8,8,1,standard_fused_topk,logit_topk,softmax_renorm,True,standalone_legacy,none,2,4096,14336,True,1,CUDA_EVENT
0.026,0.005,0.050,0.552,...,256,8,2,4,uniform_topk,round_robin_uniform,uniform_1_over_topk,False,prefill_hot,ffn_like_prefix_20x,2,4096,14336,True,1,CUDA_EVENT
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

**Example output** (values depend on the hardware and token grid):
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
    --disable_ray \
    --output_dir data/profiling
```

**Note**: This is a known issue. Ray mode is not functional until Ray/grpcio compatibility is resolved.

#### Issue 2: Missing profiling dependency

**Cause**: the MoE entrypoint requires `torch`, `vllm`, and `triton`. The
selected vLLM fused-kernel path also requires a compatible vLLM installation.

**Solution**: Use the checked-in profiling environment or an existing matching
vLLM environment:

```bash
conda env create -f environment_profiling.yml
conda activate frontier-profiling
python -c "import torch, vllm, triton; print('profiling dependencies available')"
```

The optional `flashinfer-python` package is part of the profiling environment
for attention and other CUDA workloads, but it is not a direct MoE entrypoint
dependency.

#### Issue 3: `ray.exceptions.ActorDiedError` (Legacy Issue)

**Cause**: Ray actor crashed during profiling, often due to GPU OOM or CUDA errors.

**Solution**: Use `--disable_ray` flag instead of Ray mode. If you must use Ray:

1. Reduce `--max_tokens` (e.g., from 1024 to 256)
2. Reduce number of configurations (fewer TP/EP sizes)
3. Check GPU memory: `nvidia-smi`
4. Check Ray logs for detailed error messages

#### Issue 4: CSV file is empty or has fewer results than expected

**Cause**: Profiling may have failed silently for some configurations.

**Solution**:

1. Check the terminal output for error messages
2. Verify all (TP, EP) combinations completed:

   ```python
   import pandas as pd
   df = pd.read_csv("data/profiling/compute/<device>/<model_name>/moe.csv")
   print(df.groupby(['num_tensor_parallel_workers', 'expert_parallel_size']).size())
   ```

3. Re-run the missing (TP, EP) combinations and check the terminal output for
   the first error

#### Issue 5: Profiling is very slow

**Cause**: Large `--max_tokens` or many (TP, EP) combinations.

**Solution**:

1. Reduce `--max_tokens` for faster profiling (256 is usually sufficient)
2. Pass explicit `--num_tensor_parallel_workers` and `--expert_parallel_sizes`;
   the defaults profile every TP size in `1 2 4 8` and every EP divisor
3. Reduce `--num_samples_per_distribution`
4. Use multi-GPU mode: `--num_gpus 4 --disable_ray`

### Performance Tips

1. **Start small**: Use `--max_tokens 256` for initial testing
2. **Grid-search wisely**: Profile EP first (single GPU), then add TP (multi-GPU)
3. **Use model configs**: shapes come from `--models`; add a config JSON for a new shape
4. **Monitor GPU memory**: Use `nvidia-smi` to ensure no OOM issues
5. **Choose the measurement family intentionally**: default `record_function` writes `KERNEL_ONLY`; pass `--profile_method cuda_event` when collecting eager / `CUDA_EVENT` data

---

## Comparison with Linear Op Profiling

### Similarities

- Both use the same Frontier profiling layers (`ColumnParallelLinear`,
  `RowParallelLinear`) where the operation requires them.
- Both support TP scaling profiling
- Both generate CSV files with timing statistics
- Both support Ray and `--disable_ray` multiprocessing

### Differences

| Aspect | Linear Op Profiling | MoE Profiling |
|--------|---------------|---------------|
| **Operations** | Attention projections, RoPE, LayerNorm, residual add, dense and shared-expert FFN | Gating, Shuffling, Routed-expert FFN |
| **Parallelism** | TP only (`--attn_tp`, `--ffn_tp`, `--moe_tp` domains) | TP + EP |
| **Key Parameter** | `num_tokens` | `num_tokens`, `num_experts`, `router_topk` |
| **Gating Network** | N/A | vLLM `ReplicatedLinear` by default; PyTorch `nn.Linear` with `--disable_load_imbalance` |
| **Expert Computation** | N/A | vLLM fused MoE kernel by default; Frontier per-expert loop with `--disable_load_imbalance` |
| **CSV Output** | One `linear_op.csv` per model | One `moe.csv` per model |
| **EP Support** | N/A | ✅ Grid-search across EP sizes |

### When to Use Which

- **Linear Op Profiling**: For standard transformer models (GPT, LLaMA, etc.)
- **MoE Profiling**: For Mixture-of-Experts models (Mixtral, Qwen2-MoE, etc.)

---

## Recent Updates

The entries below are a change history. Model names, test paths, and flags in
them may not match the current code; the sections above describe the current
CLI.

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

2. **Expert profiling layers**: Routed expert FFNs use the checked-in Frontier
   profiling layers, with an optional vLLM fused-kernel path.
   - `ColumnParallelLinear` for expert up projection
   - `RowParallelLinear` for expert down projection
   - `SiluAndMul` custom CUDA kernel for SwiGLU activation
   - Hybrid approach: native PyTorch for gating and Frontier profiling layers
     for expert projections

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
   - Validates routed-expert profiling layer behavior
   - Automated CSV output validation

**Bug Fixes**:

- Fixed MetricsStore singleton issue via monkey patching
- Fixed parallel state initialization issue with hybrid operator approach
- Improved error handling and logging

**Performance**:

- EP=4 provides 64% reduction in grouped GEMM time vs EP=1
- Routed-expert profiling layers validated and working correctly
- No performance regressions observed

### Version 1.0 (Original)

- Basic MoE profiling support
- Native PyTorch operators
- Manual parameter configuration
- Separate CSV files per operation

---

## Testing

### Unit Tests

These tests run on CPU in the minimal simulator environment:

```bash
python -m pytest -q \
    tests/unit/test_moe_profiling_output_metadata.py \
    tests/unit/test_moe_routing_runtime.py \
    tests/unit/test_moe_ep_sampling_domain.py
```

Tests such as `tests/unit/test_moe_load_distribution_contract.py` and
`tests/unit/test_moe_gating_constructor_boundary.py` import `torch` and need
the profiling environment.

### Dry Run

```bash
bash examples/profiling/profile_moe.sh --dry-run
```

The dry run prints the resolved command and output path without a GPU.

### Manual Smoke Test

```bash
python -m frontier.profiling.moe.main \
    --models mixtral_8x7b_moe \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --expert_parallel_sizes 1 \
    --max_tokens 64 \
    --device a100 \
    --disable_ray \
    --profile_method cuda_event \
    --output_dir /tmp/moe_test
```

**Expected**: `/tmp/moe_test/compute/a100/mixtral_8x7b_moe/moe.csv` with 33 rows
(11 token points x 3 samples) and no errors.

---

## Integration with the Frontier Simulator

### Execution Time Predictor

The generated CSV files are consumed by
`frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`.
`RandomForrestExecutionTimePredictor` and `LinearRegressionExecutionTimePredictor`
select `SklearnMoEExecutionTimePredictor` when the replica's model config is a
MoE model and the run is co-location (disaggregated runs use
`SklearnDisaggregationExecutionTimePredictor`, which extends it).

**Workflow**:

1. **Load Profiling Data**: The simulator reads
   `--random_forrest_execution_time_predictor_config_moe_input_file`
   (default `./data/profiling/compute/{DEVICE}/{MODEL}/moe.csv`) or
   `--random_forrest_execution_time_predictor_config_moe_kernel_only_input_file`
   (default `./data/profiling/compute/{DEVICE}/{MODEL}/moe_kernel_only.csv`).
2. **Filter by Configuration**: Rows are filtered by the model config
   (num_experts, router_topk, etc.) and the runtime routing and gating context.
3. **Train ML Models**: sklearn models (RandomForest/LinearRegression) are trained on the data
4. **Generate Prediction Caches**: Fast lookup tables for simulation

`examples/profiling/smoke_simulator_moe_csv.sh` runs the simulator on
checked-in `linear_op.csv`, `attention.csv`, and `moe.csv` files with dummy
mode disabled.

### Model Manager

`ExecutionTimePredictionModelManager`
(`frontier/execution_time_predictor/shared_prediction_model_manager.py`)
handles centralized model training:

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

**Observation**: The version 2.0 measurements showed a 64% reduction in mean grouped GEMM time from EP=1 to EP=4.

### Why TP is a Compute Parameter

**Tensor Parallelism (TP)** is fundamentally different from EP:

- TP **changes computation patterns** (matrix sharding, different GEMM dimensions)
- TP **affects per-device computation time** (smaller matrices)
- Each profiling task runs on one GPU with the TP-sharded shapes. The expert
  layers do not run the TP reduction (`reduce_results=False`); Frontier's
  communication backend models collectives separately.

**Therefore**: TP must be profiled separately for each TP size.

### Hybrid Operator Approach

**Design Decision**: By default, use vLLM `ReplicatedLinear` and vLLM routing
for gating and vLLM's fused MoE kernel for expert FFNs. With
`--disable_load_imbalance`, use PyTorch `nn.Linear` for gating and the
checked-in Frontier profiling layers for expert FFNs.

**Rationale**:

1. **Gating Network**: Small operation (hidden_dim → num_experts), doesn't benefit from TP
2. **Expert FFNs**: Large operations (hidden_dim → expert_hidden_dim), benefit from optimized CUDA kernels
3. **Profiling Simplicity**: Avoids needing distributed initialization for single-GPU profiling
4. **Performance**: The default expert computation uses the same fused kernel as vLLM

**Trade-off**: Gating network not TP-parallelized, but this is acceptable since it's a small operation (~0.026 ms).
