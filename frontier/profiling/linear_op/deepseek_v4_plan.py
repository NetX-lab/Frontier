"""CPU-only row grids and typed contracts for standalone DeepSeek-V4 operators."""

from __future__ import annotations

import json
from pathlib import Path

from frontier.operators.deepseek_v4_axes import (
    DEEPSEEK_V4_OPERATOR_AXES, DEEPSEEK_V4_STANDALONE_AXES,
)
from frontier.operators.families import DEEPSEEK_V4_MHC_FAMILY
from frontier.operators.spec import TensorParallelMode
from frontier.profiling.linear_op.profiling_plan import _non_layer_contract


def load_operator_grid(path):
    grid = json.loads(Path(path).read_text())
    required = {"attention_tp_size", *DEEPSEEK_V4_STANDALONE_AXES}
    if not isinstance(grid, dict) or set(grid) != required:
        raise ValueError(f"DeepSeek-V4 operator grid requires exactly {sorted(required)}")
    tp = grid["attention_tp_size"]
    if type(tp) is not int or tp <= 0:
        raise ValueError("attention_tp_size must be a positive integer")
    for axis in DEEPSEEK_V4_STANDALONE_AXES:
        rows = grid[axis]
        if (not isinstance(rows, list) or not rows
                or any(type(row) is not int or row <= 0 for row in rows)):
            raise ValueError(f"{axis} must be a non-empty list of positive integer rows")
        if len(rows) != len(set(rows)):
            raise ValueError(f"{axis} must not contain duplicate rows")
    return grid


def operator_shapes(model, name, rows, tp):
    """Mirror the consumer's principal shapes and describe the native state."""

    hidden, streams = model.embedding_dim, model.hc_mult
    intermediate = model.share_expert_dim
    if name == "dsv4_embedding":
        return {"input_shape": (rows,), "output_shape": (rows, hidden)}
    if name in ("dsv4_mhc_first_pre", "dsv4_mhc_ffn_post_attn_pre",
                "dsv4_mhc_attn_post_ffn_pre"):
        return {"input_shape": (rows, hidden), "output_shape": (rows, hidden),
                "residual_shape": (rows, streams, hidden),
                "post_mix_shape": (rows, streams, 1), "res_mix_shape": (rows, streams, streams),
                "state_dtype": "float32", "activation_dtype": "bfloat16"}
    if name == "dsv4_mhc_final_post":
        return {"input_shape": (rows, hidden), "output_shape": (rows, streams, hidden),
                "residual_shape": (rows, streams, hidden), "state_dtype": "float32"}
    if name == "dsv4_mhc_head":
        return {"input_shape": (rows, streams, hidden), "output_shape": (rows, hidden)}
    if name == "dsv4_final_norm":
        return {"input_shape": (rows, hidden), "output_shape": (rows, hidden)}
    if name == "dsv4_lm_head":
        # The existing consumer deliberately leaves output_shape unspecified.
        return {"input_shape": (rows, hidden), "output_shape": None,
                "logical_unpartitioned_output_shape": (rows, model.vocab_size),
                "local_output_shape": (rows, model.vocab_size // tp), "weight_tp": tp}
    if name == "share_expert_up_proj":
        return {"input_shape": (rows, hidden), "output_shape": (rows, 2 * intermediate),
                "weight_tp": tp}
    if name == "share_expert_act":
        return {"input_shape": (rows, 2 * intermediate), "output_shape": (rows, intermediate)}
    if name == "share_expert_down_proj":
        return {"input_shape": (rows, intermediate), "output_shape": (rows, hidden),
                "weight_tp": tp}
    raise ValueError(f"Unknown DeepSeek-V4 standalone operator: {name}")


def row_contracts(model, contracts, rows):
    return {
        name: {**contract, **operator_shapes(
            model, name, rows, contract["selected_tensor_parallel_size"],
        )}
        for name, contract in contracts.items()
    }


def shared_expert_row_contracts(model, contracts, rows):
    # GLM's existing typed rows remain unchanged.
    if model.get_model_architecture_profile().profile_id != "deepseek_v4":
        return contracts
    return row_contracts(model, {
        name: {**contract, "row_axis": DEEPSEEK_V4_OPERATOR_AXES[name]}
        for name, contract in contracts.items()
    }, rows)


def build_operator_plans(model, grid):
    profile = model.get_model_architecture_profile()
    if profile.profile_id != "deepseek_v4":
        raise ValueError("--deepseek_v4_operator_grid requires a DeepSeek-V4 architecture profile")
    tp = grid["attention_tp_size"]
    if model.vocab_size % tp:
        raise ValueError(f"vocab_size {model.vocab_size} must be divisible by attention TP={tp}")
    plans = []
    for axis in DEEPSEEK_V4_STANDALONE_AXES:
        operators = [op for op in DEEPSEEK_V4_MHC_FAMILY.profiling_ops()
                     if DEEPSEEK_V4_OPERATOR_AXES[op.name] == axis]
        contracts = {}
        for op in operators:
            selected = tp if op.tp_mode is TensorParallelMode.ATTENTION_TP else 1
            contracts[op.name] = {
                **_non_layer_contract(
                    profile_id=profile.profile_id, family_id=DEEPSEEK_V4_MHC_FAMILY.family_id,
                    mode=op.tp_mode, tensor_parallel_sizes=[selected],
                    selected_tensor_parallel_size=selected,
                ),
                "row_axis": axis,
            }
        plans.append({
            "plan_name": axis, "row_axis": axis, "num_tokens": list(grid[axis]),
            "attention_tp_size": tp,
            "tp_size": tp if any(op.tp_mode is TensorParallelMode.ATTENTION_TP
                                 for op in operators) else 1,
            "enabled_ops": [op.name for op in operators],
            "replicated_ops": [], "disabled_ops": [], "skip_reasons": [],
            "attn_enabled": False, "ffn_enabled": False,
            "padded_n_embd": model.embedding_dim,
            "padded_n_expanded_embd": model.mlp_hidden_dim,
            "typed_layer_contracts": [], "typed_operator_contracts": contracts,
        })
    return plans
