# Architecture-Based Profiling Scripts

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-07 | Marked this directory as legacy/internal and directed release users to `examples/profiling/`. |
| 2026-06-06 | Updated visible examples to current release paths, linear_op script names, canonical compute output schema, and user-controlled GPU/arch selection. |
| 2026-03-13 | Align example script docs with measurement-aware profiling contract |

## Overview

This directory is retained as a legacy/internal profiling script reference. For release-facing one-click examples, start with `examples/profiling/`, which contains readable wrappers for `linear_op`, Chunked Prefill attention, MoE, metadata smoke, and downstream simulator CSV smoke.

This directory contains independent profiling scripts for the three operator families used by the simulator:

- `test_profiling_linear_op.sh` for linear operators.
- `test_profiling_attn.sh` for attention operators.
- `test_profiling_moe.sh` for MoE operators.

The scripts profile model architecture components rather than deployment topology. Each run records data for one hardware label and writes canonical CSV files under `data/profiling/compute/<device>/<model>/`.

## Important: Single-Device-Type Profiling

Each profiling run executes on one homogeneous hardware type. The `--device` parameter accepts one device label per execution, such as `a800`, `h100`, or `rtx_pro_6000`.

To collect data for multiple hardware types, run the same script once per device label.

## Measurement Contract

- `--profile-method cuda_event` -> `measurement_type=CUDA_EVENT` and primary CSV names such as `linear_op.csv`, `attention.csv`, or `moe.csv`.
- `--profile-method record_function` -> `measurement_type=KERNEL_ONLY` and method-specific CSV names such as `linear_op_kernel_only.csv`, `attention_kernel_only.csv`, or `moe_kernel_only.csv`.
- Current examples default to `cuda_event` because simulator-ready datasets use CUDA-event timing by default.

## Available Scripts

### 1. `frontier/profiling/example/test_profiling_attn.sh` - Attention Component

Profiles attention operations, including prefill, decode, mixed prefill/decode, true mixed attention, and chunked prefill coverage.

**Key parameters:**

- `--device`: Target device label for canonical output routing.
- `--tp-sizes`: Tensor parallel sizes, for example `"1 2 4"`.
- `--max-seq-len`: Maximum sequence length.
- `--attention-backend`: Attention backend, defaulting to FlashInfer in the script.
- `--profile-method`: `cuda_event` for CUDA_EVENT timing or `record_function` for KERNEL_ONLY timing.
- `--enable-mixed-prefill`, `--enable-true-mixed`, `--enable-chunked-prefill-grid-search`: multi-dimensional attention sample collection controls.

**Example usage:**

```bash
# Profile attention on a single RTX PRO 6000 GPU.
CUDA_VISIBLE_DEVICES=0 bash frontier/profiling/example/test_profiling_attn.sh \
  --model qwen2_dense_test \
  --device rtx_pro_6000 \
  --num-gpus 1 \
  --tp-sizes "1" \
  --max-seq-len 2048 \
  --enable-true-mixed true \
  --profile-method cuda_event
```

### 2. `frontier/profiling/example/test_profiling_linear_op.sh` - Linear Operators

Profiles linear-complexity operators such as MLP, LayerNorm, and projection layers. Linear operators collect `num_tokens` as the primary feature dimension.

**Key parameters:**

- `--device`: Target device label for canonical output routing.
- `--tp-sizes`: Tensor parallel sizes.
- `--max-tokens`: Maximum token count to sample.
- `--profile-method`: `cuda_event` for CUDA_EVENT timing or `record_function` for KERNEL_ONLY timing.
- `--is-moe`: Mark the model as MoE when the script is used only for shared dense linear components.

**Example usage:**

```bash
# Profile linear operators and save linear_op.csv.
CUDA_VISIBLE_DEVICES=0 bash frontier/profiling/example/test_profiling_linear_op.sh \
  --model qwen2_dense_test \
  --device rtx_pro_6000 \
  --num-gpus 1 \
  --tp-sizes "1" \
  --max-tokens 4096 \
  --profile-method cuda_event
```

### 3. `frontier/profiling/example/test_profiling_moe.sh` - MoE Component

Profiles Mixture-of-Experts operators, including routing, gating, grouped GEMM, expert parallelism, and load-distribution dimensions.

**Key parameters:**

- `--device`: Target device label for canonical output routing.
- `--tp-sizes`: Tensor parallel sizes.
- `--ep-sizes`: Expert parallel sizes.
- `--max-tokens`: Maximum token count to sample.
- `--num-gpus`: Number of visible GPUs for parallel profiling.
- `--profile-method`: `cuda_event` for CUDA_EVENT timing or `record_function` for KERNEL_ONLY timing.
- `--load-imbalance`, `--load-distributions`: Multi-feature MoE sample collection controls.

**Example usage:**

```bash
# Profile MoE and save moe.csv.
CUDA_VISIBLE_DEVICES=0 bash frontier/profiling/example/test_profiling_moe.sh \
  --model Qwen3-30B-A3B-tiny \
  --device rtx_pro_6000 \
  --num-gpus 1 \
  --tp-sizes "1" \
  --ep-sizes "1 2" \
  --max-tokens 4096 \
  --profile-method cuda_event \
  --load-imbalance true \
  --load-distributions "uniform" \
  --routing-runtime-path uniform_topk \
  --gating-runtime-context prefill_hot
```

## Output Structure

All release-facing profiling scripts write CSV files under one canonical compute schema:

```text
data/profiling/compute/<device>/<model>/
├── attention.csv
├── attention_kernel_only.csv
├── attention_mixed.csv
├── attention_true_mixed.csv
├── attention_combined.csv
├── linear_op.csv
├── linear_op_kernel_only.csv
├── moe.csv
└── moe_kernel_only.csv
```

The exact file set depends on the selected operator and profiling mode. For example:

- `data/profiling/compute/rtx_pro_6000/qwen2_dense_test/linear_op.csv`
- `data/profiling/compute/rtx_pro_6000/qwen2_dense_test/attention_true_mixed.csv`
- `data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/moe.csv`

## Dense Model Workflow

```bash
CUDA_VISIBLE_DEVICES=0 bash frontier/profiling/example/test_profiling_linear_op.sh \
  --model qwen2_dense_test \
  --device rtx_pro_6000 \
  --num-gpus 1 \
  --tp-sizes "1" \
  --profile-method cuda_event

CUDA_VISIBLE_DEVICES=0 bash frontier/profiling/example/test_profiling_attn.sh \
  --model qwen2_dense_test \
  --device rtx_pro_6000 \
  --num-gpus 1 \
  --tp-sizes "1" \
  --enable-true-mixed true \
  --profile-method cuda_event
```

## MoE Model Workflow

```bash
CUDA_VISIBLE_DEVICES=0 bash frontier/profiling/example/test_profiling_linear_op.sh \
  --model Qwen3-30B-A3B-tiny \
  --device rtx_pro_6000 \
  --num-gpus 1 \
  --tp-sizes "1" \
  --profile-method cuda_event

CUDA_VISIBLE_DEVICES=0 bash frontier/profiling/example/test_profiling_attn.sh \
  --model Qwen3-30B-A3B-tiny \
  --device rtx_pro_6000 \
  --num-gpus 1 \
  --tp-sizes "1" \
  --enable-mixed-prefill true \
  --enable-true-mixed true \
  --profile-method cuda_event

CUDA_VISIBLE_DEVICES=0 bash frontier/profiling/example/test_profiling_moe.sh \
  --model Qwen3-30B-A3B-tiny \
  --device rtx_pro_6000 \
  --num-gpus 1 \
  --tp-sizes "1" \
  --ep-sizes "1 2" \
  --profile-method cuda_event
```

## Key Features

### 1. Independent Configuration

Each script owns the dimensions relevant to one operator family:

- Attention: sequence length, prefill/decode mix, chunked prefill, true mixed attention, backend, and tensor parallelism.
- Linear operators: total token count and tensor parallelism.
- MoE: token count, tensor parallelism, expert parallelism, routing/gating context (`routing_runtime_path`, `gating_runtime_context`), and load distribution.

### 2. Fail-Fast Device Validation

The scripts source `tests/common/device_validation.sh` and validate that the requested `--device` label matches the detected GPU model. A mismatch is an error, not a warning, because otherwise profiling data could be saved under the wrong hardware directory.

### 3. User-Controlled GPU and CUDA Architecture Selection

The scripts provide safe defaults but do not overwrite user choices:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST="12.0" \
  bash frontier/profiling/example/test_profiling_linear_op.sh \
  --model qwen2_dense_test \
  --device rtx_pro_6000
```

If `TORCH_CUDA_ARCH_LIST` is unset, PyTorch/CUDA targets the active GPU architecture. Set it explicitly only when your environment requires a fixed JIT target.

### 4. Training Integration

Training scripts consume the canonical CSV paths through the same `measurement_type` contract:

- CUDA_EVENT datasets: pass `--measurement_type CUDA_EVENT`.
- KERNEL_ONLY datasets: pass `--measurement_type KERNEL_ONLY`.

Use the training examples in `frontier/training/example/` after profiling data has been collected.

### Qwen3 MoE uniform top-k training example

The generated RTX PRO 6000 Qwen3 MoE profiling rows use `routing_runtime_path=uniform_topk`; prefill-hot rows require `gating_runtime_context=prefill_hot` during training:

```bash
bash frontier/training/example/train_moe_models.sh \
  --dataset_path data/profiling/compute/rtx_pro_6000/Qwen3-30B-A3B-tiny/moe.csv \
  --model_name Qwen3-30B-A3B-tiny \
  --device rtx_pro_6000 \
  --moe_tensor_parallel_size 1 \
  --expert_parallel_size 1 \
  --measurement_type CUDA_EVENT \
  --routing_runtime_path uniform_topk \
  --gating_runtime_context prefill_hot
```

## Environment Requirements

All scripts require a Frontier-compatible profiling environment:

- Python executable: set `PYTHON_BIN=/path/to/python` when the active shell is not already using the desired environment.
- Required packages: torch, pandas, tqdm, pyyaml, vLLM, and FlashInfer.
- FlashInfer attention profiling requires a working CUDA compiler (`nvcc`) for JIT compilation.

If you already have a working environment with vLLM and FlashInfer, you can use it directly by setting `PYTHON_BIN`. Otherwise create the dedicated environment documented in the top-level `README.md` through `environment_profiling.yml`.
