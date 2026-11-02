# Mixed-Length Batch Prefilling Support

## Environment

- Use the profiling environment from `environment_profiling.yml`. It pins
  `flashinfer-python>=0.3,<0.4` and `vllm>=0.10,<0.11`.

---

## Overview

This module profiles and predicts attention prefill for batches containing
different sequence lengths. It models the shape diversity seen in serving.

### Mixed sampling envelope

`max_seq_len` is the profiling envelope. `max_model_len` is the runtime context
limit for the complete prompt plus output. Physical KV-block and batch-token
capacity remain independent runtime constraints.

Legacy mixed and true mixed use the same boundary contract. The default sequence
axis stays within `max_seq_len`; explicit KV or chunk values may exceed that
envelope, but every complete sequence must satisfy `max_model_len`. Invalid
runtime combinations fail fast.

When `--true_mixed_prefill_chunk_sizes` or
`--true_mixed_decode_kv_cache_sizes` is omitted, the sampler derives the
profiling envelope from `max_seq_len`:

1. Keep canonical anchors at or below the configured endpoint.
2. Keep the endpoint (`max_seq_len - prefill_kv_cache_size` for prefill and
   `max_seq_len - 1` for decode).
3. Add the first canonical anchor above the endpoint that stays within the
   `max_model_len` runtime boundary, or add that boundary when no anchor fits.
4. Merge explicit chunk/KV values with the automatic axis and validate runtime
   legality against `max_model_len`. Explicit values may exceed `max_seq_len`,
   but the complete workload must still fit the runtime limit. Prefill includes
   `prefill_kv_cache_size`; decode reserves one position for the current token.

`max_model_len` must be at least `max_seq_len`. Automatic candidates outside
the runtime boundary are skipped; explicitly invalid values fail fast.

## Features

### Profiling
- Multiple sequence lengths in one batch.
- **Even mode:** all sequences have the same length (baseline).
- **Random mode:** sequence lengths are sampled to represent mixed workloads.
- **Online grid strategy:** `--mixed_profile_strategy online_grid` replaces the
  even/random generator with a deterministic grid over batch size and total
  tokens (row modes `online_grid_balanced` and `online_grid_skewed`).
- **True mixed mode:** prefill and decode sequences share one batch
  (`--enable_true_mixed`).
- Sequence-distribution statistics such as variance and coefficient of variation.
- Multi-GPU execution through multiprocessing with `--disable_ray`.

### Prediction
- Dedicated Random Forest predictor: `attn_prefill_mixed`.
- Twelve features including batch size, KV-cache size, total tokens, sequence
  statistics, variance, coefficient of variation, and interaction terms.
- Dedicated true-mixed decode predictor: `attn_decode_in_mixed`.
- Without `attn_prefill_mixed`, multi-request prefill falls back to
  `attn_prefill`. Without `attn_decode_in_mixed`, a co-location batch that mixes
  prefill and decode fails with an explicit error.

## Quick Start

### 1. Profiling

Run these commands from the repository root. Each run prints its plan and asks
`Proceed with profiling? [y/N]`; pass `--yes` to skip the prompt.

```bash
# Basic run (use --disable_ray for the supported local path)
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --enable_mixed_prefill \
    --enable_true_mixed \
    --mixed_mode random \
    --max_mixed_batch_size 8 \
    --mixed_num_samples 3 \
    --true_mixed_prefill_batch_sizes 1 2 4 \
    --true_mixed_decode_batch_sizes 1 2 4 8 \
    --true_mixed_prefill_kv_cache_size 0 \
    --models "meta-llama/Llama-2-7b-hf" \
    --num_gpus 1 \
    --max_seq_len 4096 \
    --max_model_len 4096 \
    --device a100 \
    --profile_method cuda_event

# Multi-GPU profiling
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --num_gpus 4 \
    --enable_mixed_prefill \
    --enable_true_mixed \
    --mixed_mode random \
    --models "meta-llama/Llama-2-7b-hf" \
    --max_seq_len 4096 \
    --device a100 \
    --profile_method cuda_event

# Output files, under <output_dir>/compute/<device>/<model>/
# (default output_dir: data/profiling):
# - attention.csv (standard)
# - attention_mixed.csv (mixed)
# - attention_true_mixed.csv (prefill and decode in one batch)
# - attention_combined.csv (merged dataset)
```

These names are for `--profile_method cuda_event`. The default
`--profile_method record_function` measures kernel time only and adds a
`_kernel_only` suffix (for example `attention_mixed_kernel_only.csv`);
`device_event` adds `_device_event`. The run also writes
`<output_dir>/compute/<device>/attention_config.yaml`. With
`--enable_mixed_prefill --profile_only_prefill`, the standard pass is skipped
and `attention.csv` is not written.

### 2. Training

```bash
# Full training (with compute dataset - trains up to 10 models, depending on config)
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./cache/models \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1 \
    --measurement_type CUDA_EVENT

# Attention-only training (without compute dataset - trains the layer/mixed models)
python -m frontier.training.cli attention \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./cache/models \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1 \
    --measurement_type CUDA_EVENT
```

The `path/to/...` files are the CSVs your profiling runs produce.
`--measurement_type` is required and must match the `measurement_type` column
of the CSVs: `CUDA_EVENT` for `cuda_event` runs, `KERNEL_ONLY` for the default
`record_function` runs, `DEVICE_EVENT` for `device_event` runs. Each trained
model is saved as `<output_dir>/<model>_<hash>.pkl`, for example
`./cache/models/attn_prefill_mixed_<hash>.pkl`.

**Note:** `--compute_dataset_path` is optional. Without it, the
compute-dependent models `attn_pre_proj`, `attn_post_proj`, `attn_rope`,
`input_layernorm`, and `add` are skipped. `add` is trained only for models that
do not use a fused add-norm.

### 3. Prediction

The predictor is integrated into `SklearnExecutionTimePredictor`:
- The simulator reads `atten_input_file`
  (`./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv`) for CUDA_EVENT
  data, or `atten_kernel_only_input_file`
  (`./data/profiling/compute/{DEVICE}/{MODEL}/attention_kernel_only.csv`) for
  KERNEL_ONLY data. To use mixed rows, point the matching flag, for example
  `--random_forrest_execution_time_predictor_config_atten_input_file`, at
  `attention_combined.csv`.
- It trains `attn_prefill_mixed` when the input has the mixed-batch columns
  (it derives `total_tokens_squared`, `seq_len_range`, and the two interaction
  terms itself), and `attn_decode_in_mixed` when the input has true-mixed rows.
- It uses `attn_prefill_mixed` when a batch has more than one prefill request.
  Otherwise, or when that model was not trained, it uses `attn_prefill`.
- It uses `attn_decode_in_mixed` for decode requests in a co-location batch that
  also has prefill tokens, and fails with an explicit error when that model is
  missing.

## Key Parameters

| Parameter | Default | Description |
|------|--------|------|
| `--enable_mixed_prefill` | False | Enable mixed-batch profiling. |
| `--mixed_mode` | both | Select `even`, `random`, or `both`. |
| `--mixed_profile_strategy` | default | `default` (even/random generator) or `online_grid`. |
| `--max_mixed_batch_size` | 8 | Maximum mixed batch size. |
| `--mixed_num_samples` | 3 | Samples per random-mode configuration. |
| `--mixed_kv_cache_size_list` | `0` | Explicit KV sizes; each sequence must fit `max_model_len`. |
| `--enable_true_mixed` | False | Profile prefill and decode in one batch. |
| `--true_mixed_prefill_batch_sizes` | `1 2 4` | Prefill sequence counts. |
| `--true_mixed_prefill_chunk_sizes` | automatic | Derived envelope when omitted; explicit values are checked against `max_model_len`. |
| `--true_mixed_decode_batch_sizes` | `1 2 4 8` | Decode sequence counts. |
| `--true_mixed_decode_kv_cache_sizes` | automatic | Derived from `max_seq_len - 1`; explicit values must fit `max_model_len - 1`. |
| `--true_mixed_prefill_kv_cache_size` | 0 | Prefill-side KV cache size. |
| `--max_seq_len` | 4096 | Maximum automatic profiling-envelope length. |
| `--max_model_len` | 4096 | Runtime context limit; must be at least `max_seq_len`. |
| `--max_pipeline_parallel_size` | 1 | Pipeline parallelism used by memory calculations; must divide the layer count. |
| `--profile_method` | record_function | `record_function` (kernel-only), `cuda_event`, or `device_event`. |
| `--yes` / `-y` | off | Skip the confirmation prompt. |

## CSV Fields

### Standard fields
```
time_stats.attn_prefill.{min,max,mean,median,std}
n_embd, n_q_head, n_kv_head, block_size, num_tensor_parallel_workers
batch_size, prefill_chunk_size, kv_cache_size
is_prefill, is_mixed_batch
```

### Mixed-batch fields
```
mode                # even/random, or online_grid_balanced/online_grid_skewed
seq_lens            # [128, 256, 512, 1024]
total_tokens        # sum(seq_lens)
min_seq_len         # 128
max_seq_len         # 1024
avg_seq_len         # mean(seq_lens)
equal_seq_len       # sqrt(sum(s_i^2))
seq_len_variance    # var(seq_lens)
seq_len_std         # std(seq_lens)
seq_len_cv          # std / mean (coefficient of variation)
```

### True-mixed fields
```
is_true_mixed_batch      # true-mixed row marker
num_prefill_seqs         # prefill sequences in the batch
num_decode_seqs          # decode sequences in the batch
total_prefill_tokens     # total prefill tokens in the batch
decode_avg_kv_cache_size # average decode KV cache size
batch_composition_ratio  # num_prefill_seqs / total_batch_size
```

True-mixed rows also carry `mode=true_mixed`, `is_mixed_batch=False`,
`prefill_seq_lens`, `decode_kv_cache_sizes`, `decode_batch_size`,
`total_batch_size`, and `total_decode_tokens`.

## Architecture

### File structure
```
frontier/profiling/attention/
├── mixed_attention_input.py    # MixedAttentionInput dataclass
├── true_mixed_batch_input.py   # TrueMixedBatchInput dataclass
├── attention_wrapper.py        # profile_mixed() and profile_true_mixed()
└── main.py                     # CLI entry point

frontier/profiling/utils/
└── __init__.py                 # get_mixed_prefill_input_combinations(),
                                # get_true_mixed_attention_input_combinations()

frontier/training/
└── attention_trainer.py        # attn_prefill_mixed training integration

frontier/execution_time_predictor/
└── sklearn_execution_time_predictor.py  # prediction integration
```

### Data flow
```
Profiling → CSV (is_mixed_batch=True)
         ↓
Training → attn_prefill_mixed_<hash>.pkl
         ↓
Prediction → prefer the mixed predictor for mixed prefill
          ↓
      (true mixed decode) → attn_decode_in_mixed
```

## Feature Engineering

### Training features (12)
```python
[
    "batch_size",              # batch size
    "kv_cache_size",           # KV-cache size
    "total_tokens",            # total token count
    "avg_seq_len",             # average sequence length
    "min_seq_len",             # shortest sequence
    "max_seq_len",             # longest sequence
    "total_tokens_squared",    # total_tokens^2
    "seq_len_variance",        # variance
    "seq_len_cv",              # coefficient of variation
    "seq_len_range",           # max - min
    "batch_variance_interaction",  # batch_size * variance
    "batch_cv_interaction",    # batch_size * cv
]
```

### Target variable
```python
"time_stats.attn_prefill.median"  # target column, not attn_prefill_mixed
```

## Compatibility

### Backward compatibility
- Missing `is_mixed_batch` is filled with `False`.
- Missing mixed data skips `attn_prefill_mixed` training; multi-request prefill
  then uses `attn_prefill`.
- Standard prefill behavior is unchanged.

### Prediction rule
Simplified from
`SklearnExecutionTimePredictor._get_attention_prefill_execution_time`:
```python
if len(prefill_params) > 1 and "attn_prefill_mixed" in predictions:
    return on_demand_prediction("attn_prefill_mixed", mixed_prefill_features)
# Otherwise: attn_prefill with the summed KV-cache size and
# sqrt(sum(chunk_size^2)) as the aggregate prefill chunk size.
```

## Data Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the mixed CSV produced by your profiling run
df = pd.read_csv("data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention_mixed.csv")

# 1. Coefficient of variation versus latency
plt.scatter(df['seq_len_cv'], df['time_stats.attn_prefill.median'])
plt.xlabel('Coefficient of Variation')
plt.ylabel('Attention Time (ms)')
plt.savefig('cv_vs_time.png')

# 2. Compare even and random modes
even_df = df[df['mode'] == 'even']
random_df = df[df['mode'] == 'random']
print(f"Even avg: {even_df['time_stats.attn_prefill.median'].mean():.3f} ms")
print(f"Random avg: {random_df['time_stats.attn_prefill.median'].mean():.3f} ms")

# 3. Feature importance
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
X = df[['batch_size', 'total_tokens', 'seq_len_cv', ...]]
y = df['time_stats.attn_prefill.median']
model.fit(X, y)
print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))
```

## FAQ

### Q1: Why is the target `attn_prefill` instead of `attn_prefill_mixed`?
**A:** Profiling uses the same attention kernel. The target key remains
`attn_prefill`, while `is_mixed_batch` identifies mixed rows.

### Q2: What is even mode used for?
**A:** It provides a same-length baseline for comparison with random mixed
workloads.

### Q3: How can I verify that the mixed predictor is active?
**A:** Check the simulator log for
`Trained model attn_prefill_mixed with <N> mixed-batch samples`, or confirm
that an `attn_prefill_mixed_<hash>.pkl` file exists in the cache directory.

### Q4: What does `max_pipeline_parallel_size` affect?
**A:** It affects the KV-block memory calculation in `get_max_num_blocks()`,
which assumes `num_layers / max_pipeline_parallel_size` layers per GPU. The
default is 1. The layer count must be divisible by this value; for example, a
28-layer model can use 2, 4, 7, or 14.

### Q5: How does random mode generate sequence lengths?
**A:** For each batch size from 2 to `--max_mixed_batch_size`, it uses the
length ranges 64-512, 128-1024, 512-2048, 1024-4096, 2048-8192, and
4096-16384, clipped to `max_seq_len`. It draws `--mixed_num_samples` samples
per range (default: 3) with fixed seeds and rounds each length up to a multiple
of 8. It also adds one boundary shape per KV-cache size: one sequence at the
longest legal length and the rest at half that length.

## Validation

### Basic test
```bash
# 1. Profiling
python -m frontier.profiling.attention.main \
    --disable_ray \
    --yes \
    --enable_mixed_prefill \
    --mixed_mode both \
    --max_mixed_batch_size 4 \
    --models "microsoft/phi-2" \
    --num_gpus 1 \
    --num_tensor_parallel_workers 1 \
    --max_seq_len 2048 \
    --max_model_len 2048 \
    --device a100 \
    --profile_method cuda_event

# Inspect output (this file is written by the run above; it is not checked in)
head -n 2 data/profiling/compute/a100/microsoft/phi-2/attention_mixed.csv
# Expected fields include is_mixed_batch=True, mode, seq_lens, and seq_len_cv.

# 2. Training
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path data/profiling/compute/a100/microsoft/phi-2/attention_combined.csv \
    --output_dir ./test_cache \
    --model_name "microsoft/phi-2" \
    --device a100 \
    --measurement_type CUDA_EVENT

# Inspect output
ls ./test_cache/attn_prefill_mixed_*.pkl
# The hashed artifact should exist.

# 3. Training log
# Look for:
# "Training model attn_prefill_mixed with <N> samples"
# "Trained model attn_prefill_mixed with MAPE <X>%"
```

`path/to/linear_op.csv` is the linear-op CSV from your own profiling run with
the same `--profile_method`.

## Performance Baseline

Historical measurement with Llama-2-7B on A100, TP=1. The raw data is not
checked in and these values were not re-measured with the current code:

| Batch Size | Seq Lens | Even Time | Random Time | Overhead |
|------------|----------|-----------|-------------|----------|
| 4 | [512]*4 | 2.1 ms | 2.1 ms | 0% |
| 4 | [256,512,768,1024] | - | 2.3 ms | +9.5% |
| 8 | [1024]*8 | 8.5 ms | 8.5 ms | 0% |
| 8 | [512,768,1024,1280,1536,1792,2048,2304] | - | 9.2 ms | +8.2% |

**Conclusion:** Sequence-length heterogeneity adds about 8-10% overhead in
this baseline.

## Related Files

### Core code
- `frontier/profiling/attention/mixed_attention_input.py`
- `frontier/profiling/attention/true_mixed_batch_input.py`
- `frontier/profiling/attention/attention_wrapper.py`
- `frontier/profiling/attention/main.py`
- `frontier/training/attention_trainer.py`
- `frontier/execution_time_predictor/sklearn_execution_time_predictor.py`

### Documentation
- `PROFILING_TEST_GUIDE.md` - profiling test guide
- `README_MIXED_BATCH.md` - this document

## Version

- **v1.0** (2025-11): Initial implementation with mixed-batch profiling,
  `attn_prefill_mixed` training and prediction, AttentionTrainer integration,
  and validation coverage.
