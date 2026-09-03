# Linear Operations Profiling and Training Module

## Overview

This module profiles and trains predictors for operations whose work scales with
the number of tokens. It owns common attention projections, normalization,
residual adds, dense FFN layers, and shared-expert linear layers.

### Rationale for Renaming from MLP to linear_op

The original "MLP" naming was too narrow and did not accurately describe the full scope of operations profiled and trained by this module. The new "linear_op" naming better reflects:

1. **Broader Scope**: The module profiles not just MLP layers, but all operations with linear complexity:
   - MLP layers: `mlp_up_proj`, `mlp_down_proj`, `mlp_act`
   - Normalization: `input_layernorm`, `post_attention_layernorm`
   - Attention projections: `attn_pre_proj`, `attn_post_proj`, `attn_rope`
   - Residual connections: `add`

2. **Better Categorization**: Aligns with the three-category model structure:
   - `attn`: Attention operations (prefill, decode, KV cache)
   - `moe`: Routed-expert operations (gating, shuffling, grouped GEMM)
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
- **Profiling**: The profiling plan omits dense MLP targets before collection and
  keeps common and shared-expert linear targets. The output writer also removes
  legacy MLP columns if an older result contains them.
- **Training**: Only trains models for common linear operations (LayerNorm, attention projections, residual add)

### Rationale

Routed experts replace the dense MLP path, so routed gating, token movement,
and grouped GEMM belong to the dedicated `moe` producer. Shared experts are
ordinary FFN layers: their linear work remains in `linear_op.csv` and follows
the architecture profile's FFN/attention TP authority. MoE models still need
common linear operations such as LayerNorm, attention projections, residual
adds, and shared-expert projections.

### Usage

**Profiling:**
```bash
# For dense models (default)
bash frontier/profiling/example/test_profiling_linear_op.sh --model meta-llama/Llama-2-7b-hf --device a100

# For MoE models (omit dense MLP targets while retaining common/shared linear work)
bash frontier/profiling/example/test_profiling_linear_op.sh --model mixtral_8x7b_moe --device a100 --is-moe
```

**Training:**
```bash
# For dense models (default)
python -m frontier.training.cli linear_op \
    --dataset_path data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/linear_op.csv \
    --model_name meta-llama/Llama-2-7b-hf \
    --device a100

# For MoE models (train common linear targets; shared-expert targets use the typed E2E path)
python -m frontier.training.cli linear_op \
    --dataset_path data/profiling/compute/a100/mixtral_8x7b_moe/linear_op.csv \
    --model_name mixtral_8x7b_moe \
    --device a100 \
    --is_moe
```

---

## Current Release Naming Contract

The public contract uses the `linear_op` module name and the canonical profiling dataset path:

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

Current profiler rows may also include `model_architecture_profile` and
`typed_operator_contracts`. The contract records one owner and one semantic TP
mode per measured operator: `replicated`, `attention_tp`, `ffn_tp`, or
`moe_tp`. Runtime admission checks these fields when present and uses the
legacy scalar compatibility path for older CSVs.

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
