# MoE Load Imbalance Profiling Guide

**Version**: v2.0
**Date**: 2026-01-16

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Load Distribution Types](#load-distribution-types)
- [Load Imbalance Features](#load-imbalance-features)
- [Output Data Format](#output-data-format)
- [Data Analysis Examples](#data-analysis-examples)
- [Code Examples](#code-examples)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Overview

This module adds **expert load imbalance** support to MoE (Mixture of Experts) grouped GEMM operations for more accurate execution time prediction under different load distributions.

### Core Problem

In real-world MoE inference scenarios, different experts receive **imbalanced** token counts:
- Some experts may be very "popular" and process many tokens
- Some experts may be rarely used
- This imbalance significantly affects grouped GEMM performance

**Why is this important?**
- Standard profiling assumes uniform load distribution, **failing to reflect real scenarios**
- Load imbalance can cause **25-65% performance degradation** (depending on imbalance severity)
- Accurate profiling data is essential for training high-precision prediction models

### Solution

We support load imbalance profiling through:

1. **Fused Kernel Profiling** (default as of v2.0):
   - Uses vLLM `fused_moe_kernel` for accurate load imbalance capture
   - Automatically enabled if vLLM is available
   - Gracefully falls back to per-expert loop if vLLM not installed
   - Recommended for production-grade accuracy

2. **Load Distribution Generator**: Simulates different load distributions
   - `uniform`: Uniform distribution (baseline)
   - `skewed`: Skewed distribution (some experts more popular)
   - `extremely_skewed`: Extremely skewed (80-20 rule)

3. **Load Imbalance Features**: 15 features to describe load distribution
   - Core features (4): `total_routed_tokens`, `num_experts_per_device`, `hidden_dim`, `expert_hidden_dim`
   - Config features (2): `router_topk`, `model_expansion_ratio`
   - Workload features (2): `tokens_per_expert_avg`, `tokens_to_experts_ratio`
   - Load features (4): `expert_utilization`, `min_load_ratio`, `load_imbalance_cv`, `max_load_ratio`
   - Distribution features (2): `load_entropy`, `load_gini_coefficient`

---

## Quick Start

### 1. Check Dependencies

```bash
cd /data/rlhfsim/dev-vidur

# Check if vLLM is available
python -c "from frontier.profiling.moe.moe_vllm_kernel import check_vllm_available; print('vLLM available:', check_vllm_available())"

# If output is False, install vLLM
pip install vllm
```

### 2. Run Load Imbalance Profiling

```bash
# Full command (recommended)
python -m frontier.profiling.moe.main \
    --device a100 \
    --models "mixtral_8x7b_moe" \
    --num_gpus 8 \
    --enable_load_imbalance \
    --load_distributions uniform skewed extremely_skewed \
    --num_samples_per_distribution 3 \
    --output_dir ./moe_profiling_imbalance

# Output: ./moe_profiling_imbalance/a100/mixtral_8x7b_moe/moe.csv
```

### 3. View Results

```bash
# View generated CSV
head -n 5 ./moe_profiling_imbalance/a100/mixtral_8x7b_moe/moe.csv

# Analyze with Python
python -c "
import pandas as pd
df = pd.read_csv('./moe_profiling_imbalance/a100/mixtral_8x7b_moe/moe.csv')
print('Columns:', df.columns.tolist()[:10])
print('Shape:', df.shape)
print('Load distributions:', df['load_distribution'].unique())
"
```

---

## Usage

### 1. Standard Profiling (Backward Compatible)

```bash
# Without load imbalance, uses per-expert loop (fast)
python -m frontier.profiling.moe.main \
    --device a100 \
    --models "mixtral_8x7b_moe" \
    --output_dir ./moe_profiling

# Features:
# - Uses original per-expert loop implementation
# - Fast profiling
# - Only generates uniform distribution data
# - Backward compatible, same behavior as before
```

### 2. Load Imbalance Profiling (Default as of v2.0) ⭐

```bash
# Fused kernels are now used by default (if vLLM available)
python -m frontier.profiling.moe.main \
    --device a100 \
    --models "mixtral_8x7b_moe" \
    --enable_load_imbalance \
    --load_distributions uniform skewed extremely_skewed \
    --num_samples_per_distribution 5 \
    --output_dir ./moe_profiling_imbalance

# Features:
# - Uses vLLM fused_moe_kernel by default (auto-detects vLLM availability)
# - Accurately captures load imbalance performance impact
# - Generates multiple distributions × samples
# - Recommended for training high-precision prediction models

# Parameter explanation:
# --enable_load_imbalance: Enable load imbalance profiling (required)
# --load_distributions: Distribution types to test (multiple allowed)
# --num_samples_per_distribution: Random samples per distribution (default: 3)

# Total test cases generated:
# token_counts × distributions × samples
# Example: 100 tokens × 3 distributions × 5 samples = 1500 test cases
```

### Mode Comparison

| Feature | Standard Mode (Legacy) | Fused Kernel Mode (Default) |
|---------|------------------------|------------------------------|
| Implementation | Per-expert loop | vLLM fused_moe_kernel |
| Speed | Fast | Medium |
| Accuracy | Cannot reflect load imbalance | Accurately captures performance impact |
| Dependencies | None | Requires vLLM and raises a clear error if missing |
| Use Case | Quick testing, vLLM unavailable | Production profiling (recommended) |

---

## Load Distribution Types

### Uniform Distribution

**Characteristics**:
- Each expert has equal selection probability
- Load distribution is nearly uniform
- CV ≈ 0, Gini ≈ 0

**Example**:
```
Expert token counts: [250, 252, 248, 251, 249, 250, 250, 250]
CV: 0.005, Gini: 0.002, Entropy: 2.998
```

### Skewed Distribution

**Characteristics**:
- Some experts are more popular (power law distribution)
- Moderate load imbalance
- CV ≈ 0.2-0.4, Gini ≈ 0.1-0.2

**Example**:
```
Expert token counts: [180, 220, 280, 310, 290, 260, 240, 220]
CV: 0.185, Gini: 0.089, Entropy: 2.912
```

### Extremely Skewed Distribution

**Characteristics**:
- 80% of tokens use only a few experts
- Severe load imbalance
- CV ≈ 0.5-1.0, Gini ≈ 0.3-0.5

**Example**:
```
Expert token counts: [450, 480, 420, 380, 120, 80, 50, 20]
CV: 0.682, Gini: 0.312, Entropy: 2.456
```

---

## Load Imbalance Features

### 1️⃣ Core Features (4)

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `total_routed_tokens` | Total tokens after routing | 1024 |
| `num_experts_per_device` | Number of experts per device | 8 |
| `hidden_dim` | Model hidden dimension | 4096 |
| `expert_hidden_dim` | Expert FFN hidden dimension | 11008 |

### 2️⃣ Config Features (2)

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `router_topk` | Number of experts per token | 2 |
| `model_expansion_ratio` | expert_hidden_dim / hidden_dim | 2.69 |

### 3️⃣ Workload Features (2)

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `tokens_per_expert_avg` | Average tokens per expert | 128.0 |
| `tokens_to_experts_ratio` | total_tokens / num_experts | 128.0 |

### 4️⃣ Load Features (4) ⭐

| Feature | Description | Range | Example Value |
|---------|-------------|-------|---------------|
| `expert_utilization` | Proportion of experts with load | [0, 1] | 0.875 |
| `min_load_ratio` | Min load / average load | [0, ∞) | 0.15 |
| `load_imbalance_cv` | Coefficient of Variation | [0, ∞) | 0.682 |
| `max_load_ratio` | Max load / average load | [1, ∞) | 3.75 |

**Feature Explanation**:
- **CV (Coefficient of Variation)**: std / mean, larger values indicate more severe imbalance
- **Most Important Metric**: `load_imbalance_cv` directly reflects load distribution dispersion

### 5️⃣ Distribution Statistics Features (2)

| Feature | Description | Range | Example Value |
|---------|-------------|-------|---------------|
| `load_entropy` | Entropy of load distribution | [0, log₂(n)] | 2.456 |
| `load_gini_coefficient` | Gini coefficient | [0, 1] | 0.312 |

**Feature Explanation**:
- **Entropy**: Higher values indicate more uniform distribution, max value is log₂(num_experts)
- **Gini Coefficient**: 0 = perfect equality, 1 = perfect inequality

---

## Output Data Format

### CSV File Structure

With load imbalance enabled, the generated `moe.csv` contains:

#### 1. Time Statistics Columns (15)

```
time_stats.moe_gating_linear.min/max/mean/median/std
time_stats.moe_gating_routing_topk.min/max/mean/median/std
time_stats.moe_shuffling.min/max/mean/median/std
time_stats.moe_grouped_gemm.min/max/mean/median/std
```

#### 2. Basic Configuration Columns (9)

```
num_tokens, num_experts, num_experts_per_device, expert_parallel_size,
routing_runtime_path, routing_assignment_policy, routing_weight_policy,
routing_uses_router_logits, gating_runtime_context, gating_runtime_context_impl,
router_topk, hidden_dim, expert_hidden_dim, use_gated, num_tensor_parallel_workers,
measurement_type
```

#### 3. Load Imbalance Feature Columns (15)

```
total_routed_tokens, load_distribution, load_imbalance_cv, load_gini_coefficient,
expert_utilization, min_load_ratio, max_load_ratio, load_entropy,
model_expansion_ratio, tokens_per_expert_avg, tokens_to_experts_ratio, ...
```

**Total**: ~39 columns

---

## Data Analysis Examples

### Compare Performance Across Load Distributions

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('./moe_profiling_imbalance/a100/mixtral_8x7b_moe/moe.csv')

# Group by load_distribution, calculate average execution time
grouped = df.groupby('load_distribution')['time_stats.moe_grouped_gemm.median'].agg(['mean', 'std'])
print("\nPerformance comparison across load distributions:")
print(grouped)

# Visualize
grouped['mean'].plot(kind='bar', yerr=grouped['std'], 
                     title='MoE Grouped GEMM Performance by Load Distribution',
                     ylabel='Execution Time (ms)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('load_distribution_comparison.png')
```

### Analyze Relationship Between Load Imbalance and Performance

```python
# Calculate correlation between CV and execution time
correlation = df[['load_imbalance_cv', 'time_stats.moe_grouped_gemm.median']].corr()
print(f"\nCorrelation between CV and execution time: {correlation.iloc[0, 1]:.3f}")

# Scatter plot: CV vs execution time
plt.figure(figsize=(10, 6))
for dist in df['load_distribution'].unique():
    subset = df[df['load_distribution'] == dist]
    plt.scatter(subset['load_imbalance_cv'], 
                subset['time_stats.moe_grouped_gemm.median'],
                label=dist, alpha=0.6)

plt.xlabel('Load Imbalance CV')
plt.ylabel('Execution Time (ms)')
plt.title('Load Imbalance CV vs Execution Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cv_vs_time.png')
```

---

## Code Examples

### Using Load Distribution Generator

```python
from frontier.profiling.moe.load_distribution import (
    generate_expert_routing,
    compute_expert_token_counts,
    analyze_load_distribution,
)

# Generate routing data
weights, ids = generate_expert_routing(
    num_tokens=1000,
    num_experts=8,
    top_k=2,
    load_distribution="skewed",
    seed=42
)

# Compute token count per expert
counts = compute_expert_token_counts(ids, num_experts=8)
print(f"Expert token counts: {counts}")

# Analyze load distribution
stats = analyze_load_distribution(counts)
print(f"CV: {stats['cv']:.3f}")
print(f"Gini: {stats['gini']:.3f}")
```

### Using MoELoadImbalanceInput

```python
from frontier.profiling.moe.moe_input import MoELoadImbalanceInput

# Create input (auto-generates uniform distribution)
load_input = MoELoadImbalanceInput(
    num_tokens=512,
    num_experts_per_device=8,
    hidden_dim=4096,
    expert_hidden_dim=11008,
    router_topk=2,
)

# Get all features
features = load_input.to_features_dict()
print(f"Total features: {len(features)}")
print(f"CV: {features['load_imbalance_cv']:.3f}")
print(f"Gini: {features['load_gini_coefficient']:.3f}")
```

---

## Troubleshooting

### Issue 1: `vLLM not available` Error

**Solution**:
```bash
# Install vLLM
pip install vllm

# Verify installation
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

### Issue 2: CUDA Out of Memory

**Solutions**:
1. Reduce parallel GPU count: `--num_gpus 4`
2. Reduce max_tokens: `--max_tokens 2048`
3. Reduce sample count: `--num_samples_per_distribution 2`

### Issue 3: Profiling Too Slow

**Solutions**:
1. Test only some distributions: `--load_distributions uniform skewed`
2. Reduce sample count: `--num_samples_per_distribution 2`
3. Use standard mode (fast but inaccurate): omit `--enable_load_imbalance`

### Issue 4: CSV Missing Columns

**Check**:
```python
import pandas as pd
df = pd.read_csv('moe.csv')
print('load_distribution' in df.columns)  # Should be True
```

**Solution**: Ensure `--enable_load_imbalance` parameter is used

---

## FAQ

### Q1: How much slower is Load Imbalance mode vs Standard mode?

**A**: About 2-3x slower
- Standard mode: ~10 minutes (100 token configs)
- Load Imbalance mode: ~30 minutes (100 configs × 3 distributions × 3 samples)

### Q2: How much GPU memory is needed?

**A**: Depends on model size
- Mixtral 8x7B: ~16GB per GPU
- Qwen2-MoE 57B: ~40GB per GPU

**Recommendation**: Use A100 (40GB/80GB) or H100 (80GB)

### Q3: Can it run on CPU?

**A**: Not recommended
- vLLM kernel requires GPU
- CPU performance would be very slow (100x+)

### Q4: How to choose `num_samples_per_distribution`?

**A**: Recommended values
- **Quick test**: 1-2 samples
- **Normal profiling**: 3-5 samples
- **High precision**: 10+ samples

### Q5: How much performance difference between load distributions?

**A**: Typical differences
- Uniform → Skewed: +20-30%
- Uniform → Extremely Skewed: +50-70%

### Q6: How to verify profiling results correctness?

**A**: Checklist
1. ✅ CSV contains all necessary columns
2. ✅ Different load distributions show performance differences
3. ✅ CV correlates positively with execution time
4. ✅ Uniform distribution has CV ≈ 0
5. ✅ Extremely Skewed has CV > 0.5

---

## Related Documentation

- [MoE Profiling Basic Guide](./README.md)
- [Environment Setup Guide](./SETUP.md)
- [Training and Technical Documentation](./TRAINING_AND_TECHNICAL.md)

---

**Version**: v2.0
**Last Updated**: 2026-01-16

**Changelog**:
- v2.0 (2026-01-16): Fused kernels require vLLM and raise a clear error when unavailable
- v1.0 (2025-11-24): Initial load imbalance profiling support
