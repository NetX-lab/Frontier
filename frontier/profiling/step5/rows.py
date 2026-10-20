"""Typed native Step attention rows in the existing compute-profile schema."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace

from frontier.attention.families import STEP5_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import validate_attention_profiling_dataframe
from frontier.attention.step5 import describe_step5_layer
from frontier.operators.typed_contracts import serialize_typed_operator_contracts


OPERATOR_NAMES = tuple(op.name for op in STEP5_ATTENTION_FAMILY.operators)


def operator_contracts(config, layer_id, tp):
    family = STEP5_ATTENTION_FAMILY.family_id
    return {name: {
        "profile_id": config.get_model_architecture_profile().profile_id,
        "operator_family_id": family, "operator_family_ids": [family],
        "layer_kind": None, "dimension_source": None, "effective_ffn_width": None,
        "tensor_parallel_mode": "attention_tp", "expert_parallel_mode": "off",
        "selected_expert_parallel_size": None, "tensor_parallel_sizes": [tp],
        "selected_tensor_parallel_size": tp, "selected_padded_ffn_width": None,
        "global_layer_id": layer_id,
    } for name in OPERATOR_NAMES}


def build_row(config, cell, *, tp, block_size, max_model_len, runtime_signature,
              device, time_stats, evidence):
    metadata = describe_step5_layer(config, cell.global_layer_id, attention_tp_size=tp,
                                   moe_tp_size=1, expert_parallel_size=1)
    workload = cell.workload(config, block_size=block_size, max_model_len=max_model_len)
    g = metadata.geometry
    return {
        "measurement_type": "CUDA_EVENT", "attention_backend": evidence["attention_backend"],
        "model_architecture_profile": config.get_model_architecture_profile().profile_id,
        "attention_variant": metadata.attention_spec.variant_id,
        "global_layer_id": cell.global_layer_id, "quant_signature": config.get_quant_signature(),
        "model_dtype": "bfloat16", "norm_dtype": config.norm_dtype,
        "residual_dtype": "float32" if config.fp32_residual_connection else "bfloat16",
        "runtime_stack_signature": runtime_signature, "device": device,
        "num_tensor_parallel_workers": tp, "n_q_head": g.num_q_heads,
        "n_kv_head": g.num_kv_heads, "head_size": g.head_dim,
        **{name: value for name, value in asdict(g).items() if name.startswith("local_")
           or name.endswith("replication") or name == "proxy_dim"},
        "region_block_size": config.sparse_config["region_block_size"],
        "topk_regions": config.sparse_config["topk"], "sliding_window": config.sliding_window,
        "rope_theta": metadata.rope_theta, "rotary_dim": metadata.rotary_dim,
        "block_size": block_size, "cache_dtype": "auto", "batch_size": cell.batch_size,
        "batch_num_tokens": cell.num_tokens,
        "batch_num_prefill_tokens": cell.num_tokens if cell.phase == "prefill" else 0,
        "batch_num_decode_tokens": cell.num_tokens if cell.phase == "decode" else 0,
        "physical_scope": "native_attention_layer_without_tp_allreduce",
        "fusion": json.dumps(metadata.attention_scopes(workload)),
        "graph_mode": cell.graph_mode, "valid_tokens": cell.num_tokens, "padded_tokens": 0,
        "query_lens": json.dumps(cell.query_lens), "context_lens": json.dumps(cell.context_lens),
        "phase": cell.phase, "max_model_len": max_model_len,
        "block_table_columns": workload.block_table_columns, "score_regions": workload.score_regions,
        "weight_source": "synthetic_bf16", "time_stats": time_stats,
        "typed_operator_contracts": serialize_typed_operator_contracts(
            operator_contracts(config, cell.global_layer_id, tp)),
        **evidence,
    }


def rows_to_dataframe(rows):
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame = pd.json_normalize(frame.pop("time_stats")).add_prefix("time_stats.").join(frame)
    # The shared validator currently couples schema validation to runtime
    # enablement. Use a private schema view; never mutate/register the family.
    schema = replace(STEP5_ATTENTION_FAMILY, execution_enabled=True)
    validate_attention_profiling_dataframe(frame, schema, measurement_type="CUDA_EVENT")
    whole_columns = [f"time_stats.step5_layer_e2e.{suffix}"
                     for suffix in ("min", "max", "mean", "median", "std", "count")]
    if any(column not in frame for column in whole_columns):
        raise ValueError("Step attention rows must include whole-layer timing statistics")
    targets = [c for c in frame if c.startswith("time_stats.")]
    if frame.empty or frame[targets].isna().any().any():
        raise ValueError("Step attention rows must contain complete timing statistics")
    if any(not math.isfinite(v) or v < 0 for v in frame[targets].to_numpy().flat):
        raise ValueError("Step attention timing statistics must be finite and nonnegative")
    if (frame["time_stats.step5_layer_e2e.mean"] <= 0).any():
        raise ValueError("Whole-layer mean time must be positive")
    count_columns = [c for c in targets if c.endswith(".count")]
    for counts in frame[count_columns].to_numpy():
        if len(set(counts)) != 1 or any(c < 2 or c != int(c) for c in counts):
            raise ValueError("Timing statistics require matching integral sample counts of at least two")
    return frame
