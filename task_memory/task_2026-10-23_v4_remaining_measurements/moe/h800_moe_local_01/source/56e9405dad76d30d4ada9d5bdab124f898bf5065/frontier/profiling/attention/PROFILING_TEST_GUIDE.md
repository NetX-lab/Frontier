# Mixed-Batch Prefill Profiling Test Guide

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-10-14 | Translated to English; corrected the `--mixed_mode` and `--max_pipeline_parallel_size` defaults, replaced model names that have no config, added the required `--measurement_type` training flag, and documented method-suffixed output names, cache file names, and current error messages |
| 2026-06-06 | Replaced private checkout path with a repo-relative profiling example; refreshed legacy CSV naming note for release docs |
| 2026-01-16 | Documented `--batch_size_list` for explicit batch-size profiling control.          |
| 2025-12-06 | Added multi-GPU mode documentation; updated CSV naming from legacy MLP naming to `linear_op.csv`; added --disable_ray usage; updated --compute_dataset_path as optional |
| 2025-11-09 | Initial documentation for mixed batch prefill profiling                            |

---

## Environment

- Use the profiling environment from `environment_profiling.yml`. It pins
  `flashinfer-python>=0.3,<0.4` and `vllm>=0.10,<0.11`.

---

## Purpose

This guide lists test commands, parameters, and scenarios for checking that
mixed-batch prefill profiling works. Feature details are in
`README_MIXED_BATCH.md`.

Run all commands from the repository root. Each profiling run prints its plan
and asks `Proceed with profiling? [y/N]`; the commands below pass `--yes` to
skip the prompt.

---

## Quick Test Commands

### Training command examples

The `path/to/...` files are the CSVs your profiling runs produce.
`--measurement_type` is required and must match the `measurement_type` column
of the CSVs: `CUDA_EVENT` for `--profile_method cuda_event`, `KERNEL_ONLY` for
the default `record_function`, and `DEVICE_EVENT` for `device_event`.

```bash
# Full training (with compute dataset)
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./testing_training/models \
    --model_name "Qwen/Qwen3-4B" \
    --device h100 \
    --tensor_parallel_size 1 \
    --block_size 16 \
    --predictor_type random_forest \
    --measurement_type CUDA_EVENT

# Attention-only training (without compute dataset)
python -m frontier.training.cli attention \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./testing_training/models \
    --model_name "Qwen/Qwen3-4B" \
    --device h100 \
    --tensor_parallel_size 1 \
    --measurement_type CUDA_EVENT
```

### 1. Simplest test (recommended first)

```bash
# Use --disable_ray for the supported local multiprocessing path
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --enable_mixed_prefill \
    --mixed_mode random \
    --models "meta-llama/Llama-2-7b-hf" \
    --num_gpus 1 \
    --max_seq_len 4096 \
    --device a100 \
    --profile_method cuda_event
```

Without `--device`, the output goes to `data/profiling/compute/unknown/...`.

## Core Parameters

### Mixed-batch parameters

| Parameter | Default | Description |
|------|--------|------|
| `--enable_mixed_prefill` | False | Enable mixed-batch profiling |
| `--mixed_mode` | both | `even`: same length<br>`random`: random lengths<br>`both`: both modes |
| `--mixed_profile_strategy` | default | `default` (even/random generator) or `online_grid` |
| `--max_mixed_batch_size` | 8 | Largest mixed batch size (at most 128) |
| `--mixed_num_samples` | 3 | Samples per length range in random mode |
| `--enable_true_mixed` | False | Profile prefill and decode in one batch; cannot be combined with `--profile_only_prefill` or `--profile_only_decode` |
| `--max_pipeline_parallel_size` | 1 | Pipeline parallelism used by the KV-block memory calculation<br>**Note**: the model's layer count must be divisible by this value |

### Basic parameters

| Parameter | Default | Description |
|------|--------|------|
| `--models` | 6 models | Model list, for example `"meta-llama/Llama-2-7b-hf"` |
| `--device` | unknown | Hardware SKU used in the output path, for example `a100` |
| `--num_gpus` | 8 | Number of GPUs |
| `--num_tensor_parallel_workers` | 1 2 4 8 | Tensor parallel sizes to profile |
| `--max_seq_len` | 4096 | Largest sequence length in the profiling grid |
| `--max_model_len` | 4096 | Model context limit<br>**Note**: must be >= `max_seq_len` |
| `--min_batch_size` | 1 | Smallest decode batch size |
| `--max_batch_size` | 128 | Largest decode batch size |
| `--batch_size_list` | None | Explicit batch sizes (overrides min/max) |
| `--output_dir` | data/profiling | Root output directory; the CLI writes `compute/<device>/<model_name>/` under this root |
| `--block_size` | 16 | KV-cache block size |
| `--attention_backend` | FLASHINFER | Attention backend (`FLASHINFER`, `VLLM_ROCM`, `NO_OP`, `FLASHINFER_MLA`) |
| `--profile_method` | record_function | `record_function` (kernel-only), `cuda_event`, or `device_event` |

### Mode control

| Parameter | Description |
|------|------|
| `--profile_only_prefill` | Profile prefill only. With `--enable_mixed_prefill`, the standard pass is skipped and only mixed batches are profiled |
| `--profile_only_decode` | Profile decode only |
| `--disable_ray` | Use local multiprocessing instead of Ray (one worker process per GPU) |
| `--yes` / `-y` | Skip the confirmation prompt |

## Test Scenarios

### Scenario 1: Quick check

**Goal**: Check the basic function.

```bash
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --enable_mixed_prefill \
    --mixed_mode random \
    --max_mixed_batch_size 4 \
    --mixed_num_samples 2 \
    --models "microsoft/phi-2" \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --max_model_len 2048 \
    --device a100 \
    --profile_method cuda_event \
    --profile_only_prefill
```

**Checks**:
- `attention_mixed.csv` and `attention_combined.csv` are written. `attention.csv`
  is not, because `--profile_only_prefill` with `--enable_mixed_prefill` skips
  the standard pass.
- The CSV rows have `is_mixed_batch=True`.
- The CSV has `seq_lens`, `seq_len_cv`, and the other mixed-batch fields.

### Scenario 2: Even vs. random comparison

**Goal**: Compare the two modes.

```bash
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --enable_mixed_prefill \
    --mixed_mode both \
    --max_mixed_batch_size 8 \
    --models "meta-llama/Llama-2-7b-hf" \
    --num_gpus 1 \
    --max_seq_len 4096 \
    --max_model_len 4096 \
    --num_tensor_parallel_workers 2 \
    --device a100 \
    --profile_method cuda_event \
    --output_dir ./testing_profiling/attention/
```

**Analysis**:
```python
import pandas as pd
df = pd.read_csv(
    "./testing_profiling/attention/compute/a100/meta-llama/Llama-2-7b-hf/attention_mixed.csv"
)
even = df[df['mode']=='even']['time_stats.attn_prefill.median'].mean()
random = df[df['mode']=='random']['time_stats.attn_prefill.median'].mean()
print(f"Overhead: {(random/even-1)*100:.1f}%")
```

### Scenario 3: Large-scale test

**Goal**: Broad coverage.

```bash
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --enable_mixed_prefill \
    --mixed_mode random \
    --max_mixed_batch_size 64 \
    --mixed_num_samples 5 \
    --models "meta-llama/Llama-2-7b-hf" "meta-llama/Meta-Llama-3-8B" \
    --num_gpus 2 \
    --num_tensor_parallel_workers 1 2 \
    --max_seq_len 8192 \
    --max_model_len 8192 \
    --device a100 \
    --profile_method cuda_event
```

### Scenario 4: Layer count and `--max_pipeline_parallel_size`

`get_max_num_blocks()` asserts that the layer count is divisible by
`--max_pipeline_parallel_size`. The default of 1 always passes. A larger value
must divide the layer count; for example, `Qwen/Qwen3-4B` has 36 layers.

```bash
# Fails with AssertionError: 36 % 8 != 0
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --enable_mixed_prefill \
    --max_pipeline_parallel_size 8 \
    --models "Qwen/Qwen3-4B" \
    --num_gpus 1 \
    --device a100

# Works: 4 divides 36
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --enable_mixed_prefill \
    --mixed_mode random \
    --max_mixed_batch_size 8 \
    --max_pipeline_parallel_size 4 \
    --models "Qwen/Qwen3-4B" \
    --num_gpus 1 \
    --max_seq_len 6400 \
    --max_model_len 12800 \
    --device a100 \
    --profile_method cuda_event
```

## Result Checks

The CSV paths in this section and the sections below are files that your
profiling runs write; they are not checked in.

### 1. Check the output files

```bash
# List the output directory
ls -lh data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/

# With --profile_method cuda_event it should contain:
# - attention.csv (standard)
# - attention_mixed.csv (mixed)
# - attention_combined.csv (merged)
# - attention_true_mixed.csv (only with --enable_true_mixed)
```

With the default `--profile_method record_function`, each name has a
`_kernel_only` suffix (for example `attention_mixed_kernel_only.csv`); with
`device_event`, the suffix is `_device_event`. The run also writes
`data/profiling/compute/<device>/attention_config.yaml`.

### 2. Check the CSV columns

```bash
# List the column names
head -n 1 data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_mixed.csv \
    | tr ',' '\n' | grep -E "(mixed|seq_len|mode)"

# Expected:
# - is_mixed_batch
# - mode
# - seq_lens
# - min_seq_len, max_seq_len, avg_seq_len, equal_seq_len
# - seq_len_variance, seq_len_std, seq_len_cv
# (total_tokens is also present but does not match this grep pattern)
```

### 3. Check the values

```python
import ast

import numpy as np
import pandas as pd

df = pd.read_csv("data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_mixed.csv")

# Check the is_mixed_batch marker
assert df['is_mixed_batch'].all(), "All rows should have is_mixed_batch=True"

# Check the mode (the default strategy produces even and random rows)
assert df['mode'].isin(['even', 'random']).all()

# Check that the statistics match seq_lens
for _, row in df.iterrows():
    seq_lens = ast.literal_eval(row['seq_lens'])
    assert row['total_tokens'] == sum(seq_lens)
    assert row['min_seq_len'] == min(seq_lens)
    assert row['max_seq_len'] == max(seq_lens)
    assert abs(row['avg_seq_len'] - np.mean(seq_lens)) < 0.01
    assert abs(row['seq_len_cv'] - np.std(seq_lens)/np.mean(seq_lens)) < 0.01

print("All checks passed")
```

## Training Tests

### Basic training

```bash
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_combined.csv \
    --output_dir ./cache/models \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1 \
    --measurement_type CUDA_EVENT
```

### Check the training result

```bash
# 1. Check the log. It should contain:
# "Using mixed-batch dataset (<N> rows)"
# "Training model attn_prefill_mixed with <N> samples"
# "Trained model attn_prefill_mixed with MAPE <X>%"

# 2. Check the model file (saved as <output_dir>/<model>_<hash>.pkl)
ls cache/models/attn_prefill_mixed_*.pkl

# 3. Check that the model loads
python -c "
import glob, pickle
path = sorted(glob.glob('cache/models/attn_prefill_mixed_*.pkl'))[0]
with open(path, 'rb') as f:
    model = pickle.load(f)
print(f'Model type: {type(model)}')
print(f'Features: {model.n_features_in_}')
"
```

## Troubleshooting

### Issue 1: `AssertionError` in `get_max_num_blocks()`

**Cause**: The layer count is not divisible by `--max_pipeline_parallel_size`.

**Fix**:
```bash
# Print the layer count
python -c "
from frontier.config.model_config import BaseModelConfig
config = BaseModelConfig.create_from_name('Qwen/Qwen3-4B')
print(f'Layers: {config.num_layers}')
"

# Use a value that divides it, or keep the default of 1
--max_pipeline_parallel_size 4
```

### Issue 2: `ValueError: max_model_len must be >= max_seq_len`

**Cause**: `--max_seq_len` is larger than `--max_model_len`.

**Fix**:
```bash
# Make sure max_model_len >= max_seq_len
--max_seq_len 6400 --max_model_len 12800
```

Generated inputs whose sequence plus KV cache would exceed `max_model_len` are
skipped, not reported as errors. Explicit KV-cache values with no legal pairing
fail with `kv_cache_sizes values have no runtime-legal pairing under max_model_len=...`.

### Issue 3: `KeyError: 'time_stats.attn_prefill_mixed.median'`

**Cause**: The CSV stores mixed-batch timing under `attn_prefill`, not
`attn_prefill_mixed`.

**Fix**: None needed. `AttentionTrainer` trains `attn_prefill_mixed` on
`time_stats.attn_prefill.median`.

### Issue 4: No mixed data generated

**Checks**:
```bash
# 1. Confirm that --enable_mixed_prefill is set.
# 2. Check the log for "=== Mixed-length Prefill Profiling Enabled ===" and
#    "After memory filtering: <N> test cases".
# 3. Check that attention_mixed.csv has data rows.
wc -l data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_mixed.csv  # should be > 1
```

### Issue 5: `Ray is not available. Use --disable_ray flag.`

**Fix**: Add `--disable_ray`.

## Performance Reference

Historical measurement with Llama-2-7B, A100, TP=1. The raw data is not checked
in and these values were not re-measured with the current code:

| Batch | Seq Lens | Mode | Time (ms) | Note |
|-------|----------|------|-----------|------|
| 1 | [512] | - | 0.5 | Single sequence |
| 4 | [512]*4 | even | 2.1 | Baseline |
| 4 | [256,512,768,1024] | random | 2.3 | +9.5% |
| 8 | [1024]*8 | even | 8.5 | Baseline |
| 8 | [512...2304] | random | 9.2 | +8.2% |

## Full Test Flow

```bash
# 1. Profiling
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --enable_mixed_prefill \
    --mixed_mode both \
    --max_mixed_batch_size 8 \
    --mixed_num_samples 3 \
    --models "meta-llama/Llama-2-7b-hf" \
    --num_gpus 1 \
    --max_seq_len 4096 \
    --max_model_len 4096 \
    --device a100 \
    --profile_method cuda_event \
    --output_dir ./test_profiling

# 2. Check the data
python -c "
import pandas as pd
df = pd.read_csv('./test_profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_combined.csv')
print(f'Total rows: {len(df)}')
print(f'Mixed rows: {df[\"is_mixed_batch\"].sum()}')
print(f'Modes: {df[df[\"is_mixed_batch\"]][\"mode\"].value_counts()}')
"

# 3. Training (linear_op.csv comes from a linear-op profiling run with the same
#    --profile_method and --output_dir)
python -m frontier.training.cli attention \
    --compute_dataset_path ./test_profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv \
    --layer_dataset_path ./test_profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_combined.csv \
    --output_dir ./test_cache \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1 \
    --measurement_type CUDA_EVENT

# 4. Check the models
ls -lh ./test_cache/*.pkl
```

## Related Documents

- `README_MIXED_BATCH.md` - feature details
- `mixed_attention_input.py` - mixed-batch input definition
- `frontier/training/attention_trainer.py` - training implementation
