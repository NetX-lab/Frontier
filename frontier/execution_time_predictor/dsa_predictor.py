"""Measured DSA attention: isolated input schema, fitting and covered lookup.

Projection/norm/rope work stays in the linear-op path. The two measured scopes
already own sparse cache writes and indexer internals; no dense/MLA child
operator is added to them. CUDA-event pure decode rows are not admitted: the
verified decode measurements are the kernel-only table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from frontier.attention.dsa_features import (
    DSA_FEATURE_COLUMNS, DSA_ROW_SCHEMA_VERSION, DSA_WORKLOAD_PATHS, DSABatchFeatures, DSACoverageError,
    build_dsa_features, dsa_decode_query_limit,
)
from frontier.attention.model_binding import bind_layer_attention
from frontier.attention.string_coercion import coerce_truthy_bool
from frontier.entities.batch import DummyForwardBatch
from frontier.entities.time_components import AttentionOperatorTimes, AttentionTime
from frontier.types import MeasurementType


_VARIANTS = ("full_index", "shared_index")
_GEOMETRY_COLUMNS = (
    "num_sequences", "min_query_len", "max_query_len",
    "min_context_len", "max_context_len", "max_key_len",
)


def dsa_input_path(attention_file: str) -> str:
    """Keep DSA tables out of the legacy attention reader's filenames."""
    path = Path(attention_file)
    suffixes = {"attention.csv": "dsa_attention.csv",
                "attention_kernel_only.csv": "dsa_attention_kernel_only.csv"}
    if path.name in suffixes:
        return str(path.with_name(suffixes[path.name]))
    if path.name not in suffixes.values():
        raise ValueError(f"DSA input must use an isolated dsa_attention filename: {path}")
    return str(path)


def _row_features(row: pd.Series) -> DSABatchFeatures:
    queries = json.loads(row["prefill_seq_lens"])
    contexts = json.loads(row["prefill_kv_cache_sizes"])
    decode_contexts = json.loads(row["decode_kv_cache_sizes"])
    if queries and decode_contexts:
        raise DSACoverageError("DSA coverage missing for mixed serving batches")
    features = build_dsa_features(
        queries or [1] * len(decode_contexts), contexts or decode_contexts, int(row["index_topk"]),
        decode_query_limit=dsa_decode_query_limit(int(row["n_q_head_per_rank"])),
    )
    if features.values["max_key_len"] != row["max_key_len"]:
        raise ValueError("DSA max_key_len disagrees with scheduled queries and cached contexts")
    return features


def load_dsa_dataframe(
    attention_file: str, model_config, tensor_parallel_size: int,
    block_size: int, measurement_type: MeasurementType,
) -> pd.DataFrame:
    if measurement_type not in (MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY):
        raise ValueError(f"Unsupported DSA measurement type: {measurement_type}")
    path = dsa_input_path(attention_file)
    df = pd.read_csv(path)
    expected = {
        "schema_version": DSA_ROW_SCHEMA_VERSION,
        "measurement_type": measurement_type.value,
        "model_architecture_profile": model_config.get_model_architecture_profile().profile_id,
        "quant_signature": model_config.get_quant_signature(),
        "profiling_precision": model_config.get_default_precision().name,
        "attention_family": "dsa_attention",
        "attention_backend": "FLASH_ATTN_MLA_SPARSE",
        "attention_impl": "FlashAttnMLASparseImpl",
    }
    for name, value in expected.items():
        if name not in df or df[name].isna().any() or set(df[name]) != {value}:
            raise ValueError(f"DSA profiling {name} mismatch in {path}: expected {value!r}")
    shape = {
        "n_embd": model_config.embedding_dim, "n_q_head": model_config.num_q_heads,
        "n_q_head_per_rank": model_config.num_q_heads // tensor_parallel_size,
        "num_tensor_parallel_workers": tensor_parallel_size, "block_size": block_size,
        **{name: getattr(model_config, name) for name in (
            "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "index_topk", "index_n_heads", "index_head_dim",
        )},
    }
    for name, value in shape.items():
        if name not in df:
            raise ValueError(f"DSA profiling column {name!r} missing in {path}")
        df = df[df[name] == value]
    if df.empty:
        raise ValueError(f"No DSA profiling rows match the model, TP and block size in {path}")
    df = df.copy()
    if not set(df["attention_variant"]) <= set(_VARIANTS):
        raise ValueError(f"Unknown DSA attention variant in {path}")
    if measurement_type is MeasurementType.CUDA_EVENT:
        # The eager pure-decode table includes a known host-stall outlier;
        # decoding is supported from kernel-only measurements, not those rows.
        df = df[coerce_truthy_bool(df["is_prefill"])].copy()
    if df.empty:
        raise ValueError(f"No supported DSA workload rows in {path}")
    features = [_row_features(row) for _, row in df.iterrows()]
    df["dsa_path"] = [feature.path for feature in features]
    df["dsa_workload"] = [feature.workload for feature in features]
    for name in features[0].values:
        df[name] = [feature.values[name] for feature in features]
    return df


def _model_name(operator: str, variant: str, workload: str) -> str:
    return f"{operator}_{variant}_{workload}"


def train_dsa_models(df: pd.DataFrame, train: Callable) -> dict:
    """Partition native paths/variants, reusing the caller's fitting and cache."""
    models = {}
    for (variant, workload), rows in df.groupby(["attention_variant", "dsa_workload"]):
        path = DSA_WORKLOAD_PATHS[workload]
        for operator, columns_by_path in DSA_FEATURE_COLUMNS.items():
            if operator == "attn_dsa_indexer" and variant == "shared_index":
                continue
            columns = columns_by_path[path]
            target = f"time_stats.{operator}.median"
            if target not in rows or not np.isfinite(rows[target]).all() or (rows[target] <= 0).any():
                raise ValueError(f"DSA requires finite positive measured {target} on {variant} rows")
            name = _model_name(operator, variant, workload)
            model = train(model_name=name, df=rows, feature_cols=list(columns), target_col=target)
            model._frontier_dsa_bounds = {
                column: (float(rows[column].min()), float(rows[column].max()))
                for column in (*columns, *_GEOMETRY_COLUMNS)
            }
            models[name] = model
    return models


class DSAPredictor:
    """Exact measured lookup, then fitted interpolation inside measured coverage."""

    def __init__(self, models: Mapping) -> None:
        self._models = {name: model for name, model in models.items() if name.startswith("attn_dsa_")}
        if not self._models:
            raise ValueError("Measured DSA attention requires trained profiling data")

    def operator_times(
        self, features: DSABatchFeatures, variant: str, *, record_kernel_count: Callable | None = None,
    ) -> AttentionOperatorTimes:
        times = {}
        for operator, columns_by_path in DSA_FEATURE_COLUMNS.items():
            if operator == "attn_dsa_indexer" and variant == "shared_index":
                times[operator] = 0.0
                continue
            name = _model_name(operator, variant, features.workload)
            if name not in self._models:
                raise DSACoverageError(f"DSA coverage missing for {name}; profile this workload first")
            model = self._models[name]
            columns = columns_by_path[features.path]
            if tuple(model._frontier_feature_names) != columns:
                raise ValueError(f"DSA predictor feature schema mismatch for {name}")
            for column, (lower, upper) in model._frontier_dsa_bounds.items():
                value = features.values[column]
                if not lower <= value <= upper:
                    raise DSACoverageError(f"DSA coverage missing for {name}: {column}={value} outside [{lower}, {upper}]")
            key = tuple(float(features.values[column]) for column in columns)
            exact = model._frontier_exact_lookup
            value = float(exact[key]) if key in exact else float(model.predict(pd.DataFrame([key], columns=list(columns)))[0])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid DSA prediction for {name}: {value}")
            times[operator] = value
            if record_kernel_count is not None:
                record_kernel_count(name, model, key, columns)
        return AttentionOperatorTimes(times)

    def predict_attention_time(self, predictor, batch, layer_id: int) -> AttentionTime:
        if any(batch.request_is_decoding) and not all(batch.request_is_decoding):
            raise DSACoverageError("DSA coverage missing for mixed serving batches")
        spec = bind_layer_attention(predictor._model_config, layer_id)
        variant = "full_index" if spec.index_producer_layer_id == layer_id else "shared_index"
        if isinstance(batch, DummyForwardBatch):
            # Native attention-DP dummy passes have no attention metadata:
            # projections/norms execute, but sparse scopes and KV writes do not.
            operators = AttentionOperatorTimes(dict.fromkeys(DSA_FEATURE_COLUMNS, 0.0))
        else:
            features = build_dsa_features(
                batch.num_tokens, batch.num_context_tokens, predictor._model_config.index_topk,
                decode_query_limit=dsa_decode_query_limit(
                    predictor._model_config.num_q_heads // predictor._replica_config.attn_tensor_parallel_size,
                ),
            )
            operators = self.operator_times(features, variant, record_kernel_count=predictor._record_kernel_count)
        return AttentionTime(
            attention_layer_pre_proj_execution_time=predictor._get_attention_layer_pre_proj_execution_time(batch),
            attention_layer_post_proj_execution_time=predictor._get_attention_layer_post_proj_execution_time(batch),
            attention_rope_execution_time=predictor._get_attention_rope_execution_time(batch),
            attn_norm_time=predictor._get_attn_norm_layer_act_execution_time(batch, layer_id=layer_id),
            operator_times=operators,
        )
