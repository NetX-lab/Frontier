# Mixed-Length Batch Prefilling Support

## Environment

- **FlashInfer**: `flashinfer-python==0.3.0`

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
- **True mixed mode:** prefill and decode sequences share one batch
  (`--enable_true_mixed`).
- Sequence-distribution statistics such as variance and coefficient of variation.
- Multi-GPU execution through multiprocessing with `--disable_ray`.

### Prediction
- Dedicated Random Forest predictor: `attn_prefill_mixed`.
- Twelve features including batch size, KV-cache size, total tokens, sequence
  statistics, variance, coefficient of variation, and interaction terms.
- A missing mixed predictor produces an explicit error with collection guidance.

## Quick Start

### 1. Profiling

```bash
cd /path/to/frontier

# Basic run (use --disable_ray for the supported local path)
python -m frontier.profiling.attention.main \
    --disable_ray \
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
    --num_gpus 4 \
    --enable_mixed_prefill \
    --enable_true_mixed \
    --mixed_mode random \
    --models "meta-llama/Llama-2-7b-hf" \
    --max_seq_len 4096 \
    --device a100 \
    --profile_method cuda_event

# Output files:
# - attention.csv (standard)
# - attention_mixed.csv (mixed)
# - attention_true_mixed.csv (prefill and decode in one batch)
# - attention_combined.csv (merged dataset)
```

### 2. Training

```bash
# Full training (with compute dataset - trains up to 10 models, depending on config)
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./cache/models \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1

# Attention-only training (without compute dataset - trains the layer/mixed models)
python -m frontier.training.cli attention \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./cache/models \
    --model_name "meta-llama/Llama-2-7b-hf" \
    --device a100 \
    --tensor_parallel_size 1
```

**Note:** `--compute_dataset_path` is optional. Without it, compute-dependent
models (`attn_pre_proj`, `attn_post_proj`, `attn_rope`, `input_layernorm`,
`post_attention_layernorm`, and `add`) are skipped.

### 3. Prediction

The predictor is integrated into `SklearnExecutionTimePredictor`:
- Train automatically when `is_mixed_batch=True` rows are present.
- Prefer the mixed predictor for mixed prefill requests.
- Use `attn_decode_in_mixed` for true-mixed decode.
- Fail fast when a mixed profiling file exists but the input lacks mixed columns.

## Key Parameters

| Parameter | Default | Description |
|------|--------|------|
| `--enable_mixed_prefill` | False | Enable mixed-batch profiling. |
| `--mixed_mode` | even | Select `even`, `random`, or `both`. |
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
| `--max_pipeline_parallel_size` | 8 | Pipeline parallelism used by memory calculations. |

## CSV Fields

### Standard fields
```
time_stats.attn_prefill.{min,max,mean,median,std}
n_embd, n_q_head, n_kv_head
batch_size, prefill_chunk_size, kv_cache_size
is_prefill, is_mixed_batch
```

### Mixed-batch fields
```
mode                # even/random
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

## Architecture

### File structure
```
frontier/profiling/attention/
├── mixed_attention_input.py    # MixedAttentionInput dataclass
├── attention_wrapper.py        # profile_mixed() implementation
├── main.py                     # CLI entry point
└── utils/__init__.py           # get_mixed_prefill_input_combinations()

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
- Missing mixed data skips mixed-predictor training and produces a clear error
  when a mixed prediction is requested.
- Standard prefill behavior is unchanged.

### Prediction guard
```python
# Use the mixed predictor only for eligible mixed prefill requests.
if (batch_size > 1 
    and "attn_prefill_mixed" in models  # model exists
    and not has_chunked_prefill):       # pure prefill
    try:
        return mixed_predictor.predict(...)
    except:
        # Explicit error path
```

## Data Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("attention_mixed.csv")

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
**A:** Check for `Using mixed prefill predictor` in the logs or confirm that an
`attn_prefill_mixed_<hash>.pkl` artifact exists.

### Q4: What does `max_pipeline_parallel_size` affect?
**A:** It affects the memory calculation in `get_max_num_blocks()`. The layer
count must be divisible by this value; for example, 28 layers can use 4 or 7.

### Q5: How does random mode generate sequence lengths?
**A:** It samples within the configured range and generates
`--mixed_num_samples` samples per configuration (default: 3).

## Validation

### Basic test
```bash
# 1. Profiling
python -m frontier.profiling.attention.main \
    --enable_mixed_prefill \
    --mixed_mode both \
    --max_mixed_batch_size 4 \
    --models "microsoft/phi-2" \
    --num_gpus 1 \
    --max_seq_len 2048 \
    --device a100 \
    --profile_method cuda_event

# Inspect output
head -n 2 data/profiling/compute/a100/microsoft/phi-2/attention_mixed.csv
# Expected fields include is_mixed_batch=True, mode, seq_lens, and seq_len_cv.

# 2. Training
python -m frontier.training.cli attention \
    --compute_dataset_path path/to/linear_op.csv \
    --layer_dataset_path path/to/attention_combined.csv \
    --output_dir ./test_cache \
    --model_name "microsoft/phi-2" \
    --device a100

# Inspect output
ls ./test_cache/microsoft/phi-2/a100/tp_1/attn_prefill_mixed_*.pkl
# The hashed artifact should exist.

# 3. Prediction
# Look for:
# "Training mixed-batch prefill model with XXX samples"
# "Using mixed prefill predictor: batch_size=X, seq_lens=[...]"
```

## Performance Baseline

Measured with Llama-2-7B on A100, TP=1:

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
