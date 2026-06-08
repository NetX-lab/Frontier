# Linear Operations Profiling and Training Module

## Modification History

| Date       | Summary of Changes                                                                 |
|------------|------------------------------------------------------------------------------------|
| 2025-12-05 | Updated is_moe behavior: now filters MLP columns instead of skipping entirely     |
| 2025-12-04 | Initial creation of the linear_op release guide with is_moe parameter support   |

---

## Overview

This module provides profiling and training capabilities for **linear operations** in LLM inference simulation. Linear operations are characterized by having **linear complexity with respect to sequence length**.

### Rationale for Renaming from MLP to linear_op

The original "MLP" naming was too narrow and did not accurately describe the full scope of operations profiled and trained by this module. The new "linear_op" naming better reflects:

1. **Broader Scope**: The module profiles not just MLP layers, but all operations with linear complexity:
   - MLP layers: `mlp_up_proj`, `mlp_down_proj`, `mlp_act`
   - Normalization: `input_layernorm`, `post_attention_layernorm`
   - Attention projections: `attn_pre_proj`, `attn_post_proj`, `attn_rope`
   - Residual connections: `add`

2. **Better Categorization**: Aligns with the three-category model structure:
   - `attn`: Attention operations (prefill, decode, KV cache)
   - `moe`: Mixture of Experts operations (gating, shuffling, grouped GEMM)
   - `linear_op`: Linear operations (this module)

3. **Extensibility**: Easier to add new linear-complexity operations in the future.

---

## The `is_moe` Parameter

### Purpose

The `is_moe` parameter controls which linear operations to profile and train:

- **`is_moe=False` (default)**: Profile/train all linear operations including MLP layers
- **`is_moe=True`**: Skip MLP-specific operations, only profile/train common linear operations

### Behavior

When `is_moe=True`:
- **Profiling**: Collects data for all operations, but **filters out** MLP-specific columns (`mlp_up_proj`, `mlp_down_proj`, `mlp_act`) from the output CSV
- **Training**: Only trains models for common linear operations (LayerNorm, attention projections, residual add)

### Rationale

In MoE (Mixture of Experts) models, the dense MLP layers are replaced by expert layers. Therefore:
- MoE models do **not** need `mlp_up_proj`, `mlp_down_proj`, `mlp_act` profiling data or trained models
- MoE models should use the dedicated `moe` profiling module for expert-specific operations
- However, MoE models **still need** common linear operations (LayerNorm, attention projections, residual add)

### Usage

**Profiling:**
```bash
# For dense models (default)
bash frontier/profiling/example/test_profiling_linear_op.sh --model meta-llama/Llama-2-7b-hf --device a100

# For MoE models (skip linear op profiling)
bash frontier/profiling/example/test_profiling_linear_op.sh --model mixtral_8x7b_moe --device a100 --is-moe
```

**Training:**
```bash
# For dense models (default)
python -m frontier.training.cli linear_op \
    --dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv \
    --model_name meta-llama/Llama-2-7b-hf \
    --device a100

# For MoE models (skip linear op training)
python -m frontier.training.cli linear_op \
    --dataset_path data/profiling/compute/a100/mixtral_8x7b_moe/linear_op.csv \
    --model_name mixtral_8x7b_moe \
    --device a100 \
    --is_moe
```

---

## Current Release Naming Contract

The pre-release-v0.1 public contract uses the `linear_op` module name and the
canonical profiling dataset path:

```text
data/profiling/compute/<device>/<model_name>/linear_op.csv
```

The release-facing helper is:

```bash
bash frontier/profiling/example/test_profiling_linear_op.sh --model <model_name> --device <device>
```

The training entrypoint is:

```bash
python -m frontier.training.cli linear_op \
    --dataset_path data/profiling/compute/<device>/<model_name>/linear_op.csv \
    --model_name <model_name> \
    --device <device> \
    --measurement_type CUDA_EVENT
```

Historical names from earlier internal prototypes are intentionally omitted from
this release-facing guide to avoid suggesting unsupported CSV filenames or
compatibility aliases.

---

## Module Structure

```
frontier/profiling/linear_op/
├── __init__.py              # Module docstring and exports
├── linear_op_impl.py        # GPT model implementation for profiling
├── linear_op_wrapper.py     # LinearOpWrapper class for profiling execution
├── main.py                  # Main entry point with Ray-based profiling
└── README.md                # This documentation file

frontier/training/
├── linear_op_trainer.py     # LinearOpTrainer class for model training
└── example/
    └── train_linear_op_models.sh  # Training shell script
```

