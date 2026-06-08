# MoE Training Guide and Technical Notes

**Version**: v1.0  
**Date**: 2025-11-24

---

## Table of Contents

### Part 1: Training Guide
- [Training Overview](#training-overview)
- [Feature Support](#feature-support)
- [Usage Methods](#usage-methods)
- [Training Examples](#training-examples)
- [Best Practices](#best-practices)

### Part 2: Technical Notes
- [MoE Trainer Updates](#moe-trainer-updates)
- [TP/EP Compatibility Fix](#tpep-compatibility-fix)

---

# Part 1: Training Guide

## Training Overview

This guide explains how to use the enhanced `MoETrainer` to train MoE execution time prediction models with load imbalance feature support.

### Key Enhancements

✅ **Automatic feature detection** - Detects and uses 15 load imbalance features  
✅ **Smart feature selection** - Different models use different feature sets  
✅ **Full backward compatibility** - Auto-fallback to legacy mode  
✅ **Enhanced logging** - Reports feature usage and data distribution

---

## Feature Support

### Load Imbalance Features (15 total)

#### Core Features (4)
- `total_routed_tokens`: Total tokens after routing
- `num_experts_per_device`: Number of experts per device
- `hidden_dim`: Model hidden dimension
- `expert_hidden_dim`: Expert FFN hidden dimension

#### Config Features (2)
- `router_topk`: Router top-k value
- `model_expansion_ratio`: Expert hidden dim / model hidden dim

#### Workload Features (2)
- `tokens_per_expert_avg`: Average tokens per expert
- `tokens_to_experts_ratio`: Token count to expert count ratio

#### Load Features (4)
- `expert_utilization`: Proportion of experts with load
- `min_load_ratio`: Min load / average load
- `load_imbalance_cv`: Coefficient of Variation (CV)
- `max_load_ratio`: Max load / average load

#### Distribution Statistics Features (2)
- `load_entropy`: Entropy of load distribution (higher = more uniform)
- `load_gini_coefficient`: Gini coefficient (0=perfect equality, 1=perfect inequality)

---

## Feature Selection Strategy

### 1. moe_grouped_gemm (Most affected by load imbalance)
- **Uses all 14 load imbalance features**
- Grouped GEMM performance is most sensitive to load imbalance

### 2. moe_gating & moe_shuffling
- **Uses only `num_tokens`**
- These operations are less affected by load imbalance

### 3. Backward Compatibility Mode
- If dataset **does not contain** load imbalance features, auto-fallback to legacy mode
- Only uses `num_tokens` as feature

---

## Usage Methods

### Method 1: Create from Model Config (Recommended)

```python
from frontier.training.moe_trainer import create_moe_trainer_from_model_config

# Create trainer
trainer = create_moe_trainer_from_model_config(
    dataset_path="/path/to/profiling/h100/qwen2_moe_example/moe.csv",
    output_dir="/path/to/trained_models",
    model_name="qwen2_moe_example",
    device="h100",
    moe_tensor_parallel_size=1,
    expert_parallel_size=1,
    predictor_type="random_forest",
)

# Train all MoE models
trainer.train_all()
```

### Method 2: Direct Creation (Manual Parameters)

```python
from frontier.training.moe_trainer import MoETrainer

trainer = MoETrainer(
    dataset_path="/path/to/profiling/h100/qwen2_moe_example/moe.csv",
    output_dir="/path/to/trained_models",
    num_experts=60,
    router_topk=4,
    hidden_dim=2048,
    expert_hidden_dim=1408,
    moe_tensor_parallel_size=1,
    expert_parallel_size=1,
    predictor_type="random_forest",
)

trainer.train_all()
```

### Method 3: Command Line

```bash
python -m frontier.training.cli moe \
    --dataset_path test_profiling/h100/qwen2_moe_example/moe.csv \
    --output_dir cache/trained_models \
    --model_name qwen2_moe_example \
    --device h100 \
    --moe_tensor_parallel_size 2 \
    --expert_parallel_size 2
```

---

## Training Examples

### Example Output with Load Imbalance Features

```
INFO: MoE Configuration:
  - model_name: qwen2_moe_example
  - device: h100
  - num_experts: 60
  - router_topk: 4
  - hidden_dim: 2048
  - expert_hidden_dim: 1408
  - moe_tensor_parallel_size: 1
  - expert_parallel_size: 1

INFO: Original MoE data: 5400 rows, 50 columns
INFO: After filtering: 5400 rows
INFO: ✓ Load imbalance features detected - will use enhanced feature set
INFO:   - Load distributions in dataset: ['extremely_skewed', 'skewed', 'uniform']
INFO:     * extremely_skewed: 1800 samples
INFO:     * skewed: 1800 samples
INFO:     * uniform: 1800 samples

INFO: --- Training moe_gating ---
INFO:   Using num_tokens only for moe_gating (1 feature)
INFO: Features: ['num_tokens']
INFO: Target: time_stats.moe_gating.median
INFO: Training random_forest predictor...
INFO: ✓ Model saved to: /path/to/trained_models/moe_gating.pkl

INFO: --- Training moe_shuffling ---
INFO:   Using num_tokens only for moe_shuffling (1 feature)
INFO: Features: ['num_tokens']
INFO: Target: time_stats.moe_shuffling.median
INFO: Training random_forest predictor...
INFO: ✓ Model saved to: /path/to/trained_models/moe_shuffling.pkl

INFO: --- Training moe_grouped_gemm ---
INFO:   Using load imbalance features for moe_grouped_gemm (14 features)
INFO: Features: ['total_routed_tokens', 'num_experts_per_device', 'hidden_dim', 
                'expert_hidden_dim', 'router_topk', 'model_expansion_ratio', 
                'tokens_per_expert_avg', 'tokens_to_experts_ratio', 'expert_utilization', 
                'min_load_ratio', 'load_imbalance_cv', 'max_load_ratio', 
                'load_entropy', 'load_gini_coefficient']
INFO: Target: time_stats.moe_grouped_gemm.median
INFO: Training random_forest predictor...
INFO: ✓ Model saved to: /path/to/trained_models/moe_grouped_gemm.pkl
```

### Example Output in Legacy Mode

```
INFO: ⚠ No load imbalance features detected - using legacy mode (num_tokens only)

INFO: --- Training moe_gating ---
INFO:   Using legacy features for moe_gating (1 feature)
INFO: Features: ['num_tokens']
...
```

---

## Verifying Training Results

### Check Model Files

```bash
ls -lh /path/to/trained_models/
# Output:
# moe_gating.pkl
# moe_shuffling.pkl
# moe_grouped_gemm.pkl
```

### Load and Test Model

```python
import pickle
import pandas as pd

# Load model
with open("/path/to/trained_models/moe_grouped_gemm.pkl", "rb") as f:
    model = pickle.load(f)

# Prepare test data (with load imbalance features)
test_data = pd.DataFrame([{
    "total_routed_tokens": 4096,
    "num_experts_per_device": 60,
    "hidden_dim": 2048,
    "expert_hidden_dim": 1408,
    "router_topk": 4,
    "model_expansion_ratio": 1408 / 2048,
    "tokens_per_expert_avg": 68.27,
    "tokens_to_experts_ratio": 68.27,
    "expert_utilization": 0.95,
    "min_load_ratio": 0.2,
    "load_imbalance_cv": 0.45,
    "max_load_ratio": 2.1,
    "load_entropy": 5.2,
    "load_gini_coefficient": 0.35,
}])

# Predict
prediction = model.predict(test_data)
print(f"Predicted execution time: {prediction[0]:.4f} ms")
```

---

## Best Practices

### 1. Profile First, Then Train
- Use `--enable_load_imbalance` for profiling
- Collect diverse load distribution data (uniform, skewed, extremely_skewed)

### 2. Data Volume Recommendations
- At least 600 samples per load distribution (`--num_samples_per_distribution 10` × 60 token counts)
- Cover multiple TP and EP combinations

### 3. Model Validation
- Use independent test set to validate model accuracy
- Check prediction error under different load distributions

### 4. Continuous Iteration
- If prediction error is large, collect more edge case profiling data
- Consider adjusting `predictor_type` (`random_forest` vs `linear_regression`)

---

## Important Notes

1. **Dataset Requirements**:
   - Must contain `time_stats.moe_gating.median`, `time_stats.moe_shuffling.median`, `time_stats.moe_grouped_gemm.median` columns
   - Load imbalance features are optional (auto-detected)

2. **Filtering Conditions**:
   - Trainer filters data based on `num_experts`, `router_topk`, `hidden_dim`, `expert_hidden_dim`, `num_tensor_parallel_workers`, `expert_parallel_size`
   - Ensure profiling data contains matching configurations

3. **Feature Consistency**:
   - Training and prediction must use same feature set
   - If trained with load imbalance features, prediction must also provide these features

4. **Model Save Path**:
   - Models saved to `{output_dir}/{model_name}.pkl`
   - Example: `/path/to/trained_models/moe_grouped_gemm.pkl`

---

# Part 2: Technical Notes

## MoE Trainer Updates

### Update Summary

**Type**: Feature Enhancement + Backward Compatible

To support **MoE Load Imbalance Profiling**, the following enhancements were made to `frontier/training/moe_trainer.py`:

1. ✅ **Auto-detect and use load imbalance features**
2. ✅ **Smart feature selection** (different models use different feature sets)
3. ✅ **Full backward compatibility** (auto-fallback to legacy mode)
4. ✅ **Enhanced logging** (report feature usage and data distribution)

### Modification Details

#### 1. `__init__()` Method

**Change**: Added `self.df = None` to store loaded dataset

**Purpose**: Allow `_get_feature_cols()` method to access dataset for feature detection

#### 2. `_load_dataset()` Method

**Change**: Store filtered dataset to `self.df` before returning

**Purpose**: Provide dataset access for feature detection

#### 3. `_verify_dataset_columns()` Method

**Enhancement**: Added load imbalance features detection logic and report dataset load distribution statistics

**New Functionality**:
```python
# Check for load imbalance features
load_imbalance_features = [
    "total_routed_tokens",
    "load_imbalance_cv",
    "load_gini_coefficient",
    "expert_utilization",
]

has_load_imbalance = all(
    feat in df.columns for feat in load_imbalance_features
)

if has_load_imbalance:
    logger.info("✓ Load imbalance features detected - will use enhanced feature set")
    # Report distribution statistics
else:
    logger.info("⚠ No load imbalance features detected - using legacy mode (num_tokens only)")
```

#### 4. `_get_feature_cols()` Method (Complete Rewrite) ⭐

**Before**:
```python
def _get_feature_cols(self, model_name: str) -> List[str]:
    # All MoE models use num_tokens as the primary feature
    return ["num_tokens"]
```

**After**:
```python
def _get_feature_cols(self, model_name: str) -> List[str]:
    # 14 load imbalance features
    load_imbalance_features = [
        "total_routed_tokens",
        "num_experts_per_device",
        "hidden_dim",
        "expert_hidden_dim",
        "router_topk",
        "model_expansion_ratio",
        "tokens_per_expert_avg",
        "tokens_to_experts_ratio",
        "expert_utilization",
        "min_load_ratio",
        "load_imbalance_cv",
        "max_load_ratio",
        "load_entropy",
        "load_gini_coefficient",
    ]
    
    # Check if load imbalance features are available
    has_load_imbalance = all(
        feat in self.df.columns for feat in load_imbalance_features
    )
    
    if has_load_imbalance:
        # Use load imbalance features for grouped_gemm (most affected)
        if model_name == "moe_grouped_gemm":
            logger.info(f"  Using load imbalance features for {model_name} (14 features)")
            return load_imbalance_features
        else:
            # For gating and shuffling, use num_tokens only
            logger.info(f"  Using num_tokens only for {model_name} (1 feature)")
            return ["num_tokens"]
    else:
        # Backward compatible: use only num_tokens
        logger.info(f"  Using legacy features for {model_name} (1 feature)")
        return ["num_tokens"]
```

### Feature Selection Strategy

| Model | With Load Imbalance Features | Without Load Imbalance Features |
|-------|------------------------------|--------------------------------|
| `moe_grouped_gemm` | 14 features (all) | 1 feature (`num_tokens`) |
| `moe_gating` | 1 feature (`num_tokens`) | 1 feature (`num_tokens`) |
| `moe_shuffling` | 1 feature (`num_tokens`) | 1 feature (`num_tokens`) |

### Backward Compatibility Guarantee

#### Legacy Dataset (without load imbalance features)
- ✅ Auto-detects missing load imbalance features
- ✅ Auto-fallback to using `num_tokens` as only feature
- ✅ All existing code and models remain unchanged
- ✅ Clear log message: "⚠ No load imbalance features detected - using legacy mode"

#### Enhanced Dataset (with load imbalance features)
- ✅ Auto-detects load imbalance features
- ✅ Uses all 14 features for `moe_grouped_gemm`
- ✅ Uses `num_tokens` only for `moe_gating` and `moe_shuffling`
- ✅ Clear log message: "✓ Load imbalance features detected - will use enhanced feature set"

---

## TP/EP Compatibility Fix

### Problem Description

When using vLLM `fused_moe_kernel` for MoE load imbalance profiling, **Tensor Parallelism (TP)** was not handled correctly.

#### Original Issue

```python
# moe_vllm_kernel.py (before fix)
def profile_fused_moe_kernel(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,              # ❌ Using full dimension
    expert_hidden_dim: int,       # ❌ Using full dimension
    ...
):
    # Directly using full dimensions to create weights
    w1 = torch.randn(num_experts, 2 * expert_hidden_dim, hidden_dim, ...)
    w2 = torch.randn(num_experts, hidden_dim, expert_hidden_dim, ...)
```

**Problem**:
- ✓ **EP (Expert Parallelism)** correctly handled: `num_experts` uses `num_experts_per_device`
- ❌ **TP (Tensor Parallelism)** not handled: dimensions not partitioned according to TP
- Result: When TP > 1, profiling used incorrect weight dimensions, leading to inaccurate performance data

### Fix Solution

#### 1. Update `moe_vllm_kernel.py`

Add `tensor_parallel_size` parameter and perform dimension partitioning:

```python
def profile_fused_moe_kernel(
    num_tokens: int,
    num_experts: int,
    hidden_dim: int,
    expert_hidden_dim: int,
    top_k: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    tensor_parallel_size: int = 1,  # ✓ New TP parameter
    ...
):
    # ✓ Partition dimensions according to TP
    # Note: hidden_dim is NOT partitioned, only expert_hidden_dim is partitioned
    expert_hidden_dim_per_partition = expert_hidden_dim // tensor_parallel_size
    
    # ✓ Verify divisibility
    if expert_hidden_dim % tensor_parallel_size != 0:
        raise ValueError(...)
    
    # ✓ Create weights using partitioned dimensions
    A = torch.randn(num_tokens, hidden_dim, ...)  # hidden_dim stays full
    w1 = torch.randn(num_experts, 2 * expert_hidden_dim_per_partition, 
                     hidden_dim, ...)  # Output dim partitioned, input dim full
    w2 = torch.randn(num_experts, hidden_dim, 
                     expert_hidden_dim_per_partition, ...)  # Input dim partitioned, output dim full
```

#### 2. Update `moe_wrapper.py`

Pass TP parameter when calling `profile_fused_moe_kernel`:

```python
def _profile_with_vllm_kernel(self, ...):
    stats = profile_fused_moe_kernel(
        num_tokens=num_tokens,
        num_experts=self.num_experts_per_device,  # ✓ EP handling
        hidden_dim=self.hidden_dim,               # ✓ Full dimension
        expert_hidden_dim=self.expert_hidden_dim, # ✓ Full dimension
        tensor_parallel_size=self.num_tensor_parallel_workers,  # ✓ TP parameter
        ...
    )
```

### Technical Details

#### TP Role in MoE

In Tensor Parallelism:
- **Weight partitioning**: Each TP worker only holds partial dimension weights
- **Computation distribution**: Each worker independently computes its responsible dimension portion
- **Communication sync**: All-reduce to merge results when necessary

#### ColumnParallel vs RowParallel

**ColumnParallel (w1 - gate/up projection)**:
```
Input: X [batch, hidden_dim]           # Full
Weight: W [2*expert_hidden_dim, hidden_dim]
        Partitioned to: W1 [expert_hidden_dim_per_partition, hidden_dim]  # Output dim partitioned
                       W2 [expert_hidden_dim_per_partition, hidden_dim]
Output: Y = XW^T
        Y1 = XW1^T  # Each worker computes partial output
        Y2 = XW2^T
        Final: Y = [Y1 | Y2 | ...]  # Concat (if gather_output=True)
```

**RowParallel (w2 - down projection)**:
```
Input: X [batch, expert_hidden_dim]
       Partitioned to: X1 [batch, expert_hidden_dim_per_partition]  # Input dim partitioned
                      X2 [batch, expert_hidden_dim_per_partition]
Weight: W [hidden_dim, expert_hidden_dim]
        Partitioned to: W1 [hidden_dim, expert_hidden_dim_per_partition]  # Input dim partitioned
                       W2 [hidden_dim, expert_hidden_dim_per_partition]
Output: Y = XW^T
        Y1 = X1W1^T  # Each worker computes partial result
        Y2 = X2W2^T
        Final: Y = Y1 + Y2 + ...  # All-reduce (if reduce_results=True)
```

**Key Points**:
- ColumnParallel partitions **output dimension**, input is full, output needs concat or stay partitioned
- RowParallel partitions **input dimension**, input is already partitioned, output needs all-reduce or stay partitioned
- In MoE, w1 and w2 work together, w1's output is w2's input, dimension partitioning stays consistent

### Weight Dimension Comparison

**Key Understanding**:
- **w1 (ColumnParallel)**: Output dimension partitioned, input dimension full
- **w2 (RowParallel)**: Input dimension partitioned, output dimension full
- **hidden_dim is NEVER partitioned**, only expert_hidden_dim is partitioned

| Config | TP=1 | TP=2 | TP=4 |
|--------|------|------|------|
| **Input (A)** | [N, 4096] | [N, 4096] | [N, 4096] |
| **expert_hidden_dim** | 11008 | 5504 | 2752 |
| **w1 shape** | [E, 22016, 4096] | [E, 11008, 4096] | [E, 5504, 4096] |
| **w2 shape** | [E, 4096, 11008] | [E, 4096, 5504] | [E, 4096, 2752] |

Where:
- N = `num_tokens`
- E = `num_experts_per_device` (affected by EP)
- **hidden_dim (4096) always stays full, not partitioned**
- **expert_hidden_dim (11008) partitioned by TP**

### EP vs TP Comparison

| Parallel Type | Scope | Affected Parameters | Implementation |
|--------------|-------|---------------------|----------------|
| **EP (Expert Parallelism)** | Expert distribution | `num_experts_per_device` | Each device handles subset of experts |
| **TP (Tensor Parallelism)** | Weight dimensions | `expert_hidden_dim` | Each worker handles partial dimensions |

### Verification

After fix, `profile_fused_moe_kernel` can:

1. ✅ **Correctly handle EP**: Use `num_experts_per_device` instead of `num_experts`
2. ✅ **Correctly handle TP**: Partition dimensions according to `tensor_parallel_size`
3. ✅ **Verify compatibility**: Check if dimensions are divisible by TP
4. ✅ **Accurate profiling**: Use same weight shapes as actual inference

---

## Related Files

### Core Modules
- `frontier/profiling/moe/moe_wrapper.py` - Profiling executor
- `frontier/profiling/moe/moe_vllm_kernel.py` - vLLM kernel wrapper
- `frontier/profiling/moe/main.py` - Main entry point
- `frontier/training/moe_trainer.py` - Model trainer

### Documentation
- `LOAD_IMBALANCE_GUIDE.md` - Load imbalance profiling guide
- `README.md` - MoE profiling overview
- `SETUP.md` - Environment setup guide

---

**Version**: v1.0  
**Last Updated**: 2025-11-24

