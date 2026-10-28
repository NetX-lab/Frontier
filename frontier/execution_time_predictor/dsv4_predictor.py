"""Measured DeepSeek-V4 replay loading, per-scope fitting and covered lookup.

Each scope learns its kernel cost and its signed entry/internal gap separately.
Composition and serving admission belong to the runtime caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Collection, Mapping

import numpy as np
import pandas as pd

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError, deepseek_v4_padded_q_heads
from frontier.attention.dsv4_features import (
    DSV4_FEATURE_COLUMNS, DSV4_GEOMETRY_COLUMNS, PHASE_DUMMY, DeepSeekV4BatchFeatures,
    build_dsv4_features, build_dsv4_dummy_features, dsv4_feature_columns,
)
from frontier.attention.families import (
    DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT, deepseek_v4_main_kv_cache_dtype,
)
from frontier.attention.string_coercion import coerce_truthy_bool
from frontier.execution_time_predictor.deepseek_v4 import (
    _select_rows, validate_measured_identity,
)
from frontier.profiling.attention.dsv4_attention_rows import BALANCED_ROUTING_SHA256, routing_columns
from frontier.profiling.attention.dsv4_replay_rows import (
    DSV4_REPLAY_OP_NAME, DSV4_REPLAY_ROW_SCHEMA_VERSION, GAP_STATS_PREFIX,
)
from frontier.types import MeasurementType

TIME_STATS_PREFIX = "time_stats"
TARGET_TIME, TARGET_GAP = "time", "gap"
_TARGET_PREFIXES = {TARGET_TIME: TIME_STATS_PREFIX, TARGET_GAP: GAP_STATS_PREFIX}
MEDIAN_STAT, KERNEL_COUNT_STAT = "median", "kernel_count"
KERNEL_ONLY_SUFFIX = "_kernel_only.csv"
ATTENTION_KERNEL_ONLY_NAME = "attention" + KERNEL_ONLY_SUFFIX
REPLAY_KERNEL_ONLY_NAME = DSV4_REPLAY_OP_NAME + KERNEL_ONLY_SUFFIX
MODEL_PREFIX = "dsv4_attention__"
# This scope owns the FlashMLA tile-schedule planner kernel.
TILE_PLANNER_SCOPE = "dsv4_sparse_attention"
THRESHOLD_COLUMN = "multi_stream_gemm_token_threshold"
_BOOLEAN_TOKENS = frozenset({"1", "true", "t", "yes", "y", "0", "false", "f", "no", "n"})
_REQUEST_COLUMNS = ("prefill_seq_lens", "prefill_kv_cache_sizes", "decode_kv_cache_sizes")


def load_attention_dataframe(file_path, model, tp, block_size, measurement_type, *,
                             float_precision=None):
    """Read the replay-only V4 schema without passing it to the dense loader."""
    try:
        rows = pd.read_csv(file_path, float_precision=float_precision)
    except FileNotFoundError as exc:
        raise DeepSeekV4CoverageError(f"V4 attention table missing: {file_path}") from exc
    expected = {
        "schema_version": DSV4_REPLAY_ROW_SCHEMA_VERSION,
        "measurement_type": measurement_type.value,
        "attention_family": "deepseek_v4_attention",
        "model_architecture_profile": model.get_model_architecture_profile().profile_id,
        "quant_signature": model.get_quant_signature(),
        "profiling_precision": model.get_default_precision().name,
        "attention_backend": "FLASHMLA_SPARSE_DSV4",
        "n_embd": model.embedding_dim, "n_q_head": model.num_q_heads,
        "n_q_head_per_rank": model.num_q_heads // tp,
        "num_tensor_parallel_workers": tp, "o_groups": model.o_groups,
        "o_groups_per_rank": model.o_groups // tp, "block_size": block_size,
        **{name: getattr(model, name) for name in (
            "q_lora_rank", "o_lora_rank", "head_dim", "qk_rope_head_dim",
            "sliding_window", "index_topk", "index_n_heads", "index_head_dim",
        )},
    }
    return _select_rows(rows, expected, "attention schema/model/TP")


def validate_balanced_routing(rows):
    """Select the measured balanced routing identity when the replica requests it."""
    for name in ("routing_runtime_path", "routing_strategy_sha256", "moe_routing"):
        if name not in rows or rows[name].isna().any() or len(rows[name].unique()) != 1:
            raise DeepSeekV4CoverageError(f"V4 balanced replay attention missing/ambiguous {name}")
    rows = _select_rows(rows, {"routing_runtime_path": "balanced_topk",
        "routing_strategy_sha256": BALANCED_ROUTING_SHA256}, "balanced replay attention")
    try:
        routing_columns(json.loads(rows.moe_routing.iloc[0]))
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        raise DeepSeekV4CoverageError("V4 balanced replay attention routing identity mismatch (moe_routing)") from exc
    return rows


def dsv4_input_path(attention_file: str) -> str:
    """Resolve the replay aggregate beside the configured kernel-only input."""
    path = Path(attention_file)
    if path.name == ATTENTION_KERNEL_ONLY_NAME:
        return str(path.with_name(REPLAY_KERNEL_ONLY_NAME))
    if path.name != REPLAY_KERNEL_ONLY_NAME:
        raise ValueError(f"V4 attention input filename mismatch in {path}: expected {ATTENTION_KERNEL_ONLY_NAME} "
                         f"or {REPLAY_KERNEL_ONLY_NAME}")
    return str(path)


def _reject(path, column, reason):
    raise DeepSeekV4CoverageError(f"V4 replay {path}: column {column}: {reason}")


def _require_column(df, column, path):
    if column not in df:
        _reject(path, column, "missing")


def _number(value, column, path, *, positive=False, integer=False):
    try:
        number = float(value)
    except (ValueError, TypeError):
        _reject(path, column, f"expected numeric value, got {value!r}")
    if not np.isfinite(number) or (positive and number <= 0) or (integer and not number.is_integer()):
        _reject(path, column, f"invalid {'positive ' if positive else ''}{'integer ' if integer else ''}value {value!r}")
    return int(number) if integer else number


def _json(value, column, path):
    try:
        return json.loads(value)
    except (ValueError, TypeError) as exc:
        raise DeepSeekV4CoverageError(f"V4 replay {path}: column {column}: invalid JSON") from exc


def _row_features(row, model, path) -> DeepSeekV4BatchFeatures:
    padded_rows = _number(row["padded_rows"], "padded_rows", path, positive=True, integer=True)
    if row["phase"] == PHASE_DUMMY:
        return build_dsv4_dummy_features(padded_rows)
    lists = []
    for column in _REQUEST_COLUMNS:
        if column not in row:
            _reject(path, column, "missing")
        values = _json(row[column], column, path)
        minimum = 1 if column == _REQUEST_COLUMNS[0] else 0
        if (not isinstance(values, list)
                or any(type(value) is not int or value < minimum for value in values)):
            _reject(path, column, f"requires an integer request list with values >= {minimum}")
        lists.append(values)
    queries, contexts, decodes = lists
    if len(queries) != len(contexts):
        _reject(path, _REQUEST_COLUMNS[1], "length differs from prefill_seq_lens")
    try:
        features = build_dsv4_features(
            queries + [1] * len(decodes), contexts + decodes,
            [False] * len(queries) + [True] * len(decodes),
            variant=row["attention_variant"], compress_ratio=int(row["compress_ratio"]),
            sliding_window=model.sliding_window, index_topk=model.index_topk, padded_rows=padded_rows,
        )
    except ValueError as exc:
        raise ValueError(f"V4 replay {path}: column phase / request lists: {exc}") from exc
    expected = {"phase": features.phase, "batch_size": features.values["num_sequences"],
                "total_tokens": features.values["num_query_tokens"], "max_key_len": features.values["max_key_len"]}
    for column, value in expected.items():
        if column not in row:
            _reject(path, column, "missing")
        if row[column] != value:
            raise ValueError(f"V4 replay {path}: column {column}: {row[column]!r} disagrees with requests ({value!r})")
    return features


def load_dsv4_attention_dataframe(
    attention_file: str, model_config, replica_config, block_size: int,
    measurement_type: MeasurementType,
) -> pd.DataFrame:
    """Admit replay identity, layout, physical ownership and actual batch features."""
    path = dsv4_input_path(attention_file)
    if measurement_type is not MeasurementType.KERNEL_ONLY:
        _reject(path, "measurement_type", f"requires {MeasurementType.KERNEL_ONLY.value}, got {measurement_type}")
    tp = replica_config.attn_tensor_parallel_size
    try:
        df = load_attention_dataframe(path, model_config, tp, block_size, measurement_type,
                                     float_precision="round_trip")
        validate_measured_identity(df, model_config, replica_config, measurement_type)
        _require_column(df, "engine_cudagraph_mode", path)
        if (df.engine_cudagraph_mode.isna().any() or df.engine_cudagraph_mode.nunique() != 1
                or df.graph_mode.iloc[0] != df.engine_cudagraph_mode.iloc[0]):
            _reject(path, "graph_mode / engine_cudagraph_mode", "requires equal single-valued engine modes")
        if getattr(replica_config, "moe_gating_routing_runtime_path", None) == "balanced_topk":
            df = validate_balanced_routing(df)
    except DeepSeekV4CoverageError as exc:
        raise DeepSeekV4CoverageError(f"V4 replay {path}: {exc}") from exc
    df = df.copy()
    main_dtype = deepseek_v4_main_kv_cache_dtype(replica_config.kv_cache_dtype)
    layout = {"padded_heads": deepseek_v4_padded_q_heads(model_config.num_q_heads // tp),
              "kv_cache_dtype": main_dtype,
              "attention_kv_cache_dtype": main_dtype}
    for column, value in layout.items():
        _require_column(df, column, path)
        if df[column].isna().any() or not (df[column] == value).all():
            _reject(path, column, f"expected {value!r}")
    ratios = {}
    for spec, ratio in zip(model_config.get_layer_attention_specs(), model_config.compress_ratios):
        ratios.setdefault(spec.variant_id, set()).add(max(1, ratio))
    for variant, values in ratios.items():
        if len(values) != 1:
            _reject(path, "compress_ratio", f"model has multiple ratios for {variant}: {sorted(values)}")
    for column in ("attention_variant", "compress_ratio", "plans_tile_schedule", THRESHOLD_COLUMN,
                   "operator_scopes", "phase", "padded_rows", "graph_runtime_mode"):
        _require_column(df, column, path)
    booleans = df.plans_tile_schedule.astype(str).str.strip().str.lower()
    if not booleans.isin(_BOOLEAN_TOKENS).all():
        _reject(path, "plans_tile_schedule", "requires boolean values")
    df["plans_tile_schedule"] = coerce_truthy_bool(df.plans_tile_schedule)
    threshold = df[THRESHOLD_COLUMN]
    if threshold.isna().any() or threshold.nunique() != 1:
        _reject(path, THRESHOLD_COLUMN, "requires a single-valued positive integer")
    _number(threshold.iloc[0], THRESHOLD_COLUMN, path, positive=True, integer=True)
    features = []
    for _, row in df.iterrows():
        variant = row["attention_variant"]
        if variant not in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT or variant not in ratios:
            _reject(path, "attention_variant", f"unsupported model variant {variant!r}")
        if row["compress_ratio"] not in ratios[variant]:
            _reject(path, "compress_ratio", f"expected {next(iter(ratios[variant]))} for {variant}")
        family = {operator.name for operator in DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT[variant]}
        scopes = _json(row["operator_scopes"], "operator_scopes", path)
        if (not isinstance(scopes, dict) or not set(scopes) <= family
                or any(owned != [scope] for scope, owned in scopes.items())):
            _reject(path, "operator_scopes", f"requires physical scope ownership for {variant}")
        for scope in family:
            time_column = f"{TIME_STATS_PREFIX}.{scope}.{MEDIAN_STAT}"
            if scope not in scopes:
                if time_column in row and not pd.isna(row[time_column]):
                    _reject(path, time_column, "absent scope must have NaN cost")
                continue
            gap_column = f"{GAP_STATS_PREFIX}.{scope}.{MEDIAN_STAT}"
            count_column = f"{TIME_STATS_PREFIX}.{scope}.{KERNEL_COUNT_STAT}"
            for column in (time_column, gap_column, count_column):
                _require_column(df, column, path)
            _number(row[time_column], time_column, path, positive=True)
            _number(row[gap_column], gap_column, path)
            _number(row[count_column], count_column, path, positive=True, integer=True)
        features.append(_row_features(row, model_config, path))
    for column in dict.fromkeys(column for feature in features for column in feature.values):
        df[column] = [feature.values.get(column, np.nan) for feature in features]
    return df


def dsv4_model_key(scope: str, variant: str, graph_runtime_mode: str, phase: str,
                   plans_tile_schedule: bool) -> tuple:
    key = (scope, variant, graph_runtime_mode, phase)
    return (*key, plans_tile_schedule) if scope == TILE_PLANNER_SCOPE else key


def dsv4_model_name(key: tuple, target: str) -> str:
    """Deterministic cache names for each physical scope key and signed target."""
    return MODEL_PREFIX + "__".join(str(value) for value in (*key, target))


def train_dsv4_attention_models(df: pd.DataFrame, train: Callable) -> dict:
    """Fit costs and signed gaps only on rows where their physical scope exists."""
    models = {}
    threshold = int(df[THRESHOLD_COLUMN].iloc[0])
    for scope in DSV4_FEATURE_COLUMNS:
        target_column = f"{TIME_STATS_PREFIX}.{scope}.{MEDIAN_STAT}"
        if target_column not in df:
            continue
        present = df[df[target_column].notna()]
        group_columns = ["attention_variant", "graph_runtime_mode", "phase"]
        if scope == TILE_PLANNER_SCOPE:
            group_columns.append("plans_tile_schedule")
        for coordinates, rows in present.groupby(group_columns):
            variant, mode, phase = coordinates[:3]
            planner = bool(coordinates[3]) if scope == TILE_PLANNER_SCOPE else False
            key = dsv4_model_key(scope, variant, mode, phase, planner)
            columns = dsv4_feature_columns(scope, phase)
            bounds_columns = columns if phase == PHASE_DUMMY else (*columns, *DSV4_GEOMETRY_COLUMNS)
            for target, prefix in _TARGET_PREFIXES.items():
                name = dsv4_model_name(key, target)
                model = train(model_name=name, df=rows, feature_cols=list(columns),
                              target_col=f"{prefix}.{scope}.{MEDIAN_STAT}")
                model._frontier_dsv4_multi_stream_gemm_token_threshold = threshold
                model._frontier_dsv4_bounds = {
                    column: (float(rows[column].min()), float(rows[column].max())) for column in bounds_columns
                }
                models[name] = model
    return models


@dataclass(frozen=True)
class DeepSeekV4ScopeTimes:
    cost_ms: Mapping[str, float]
    gap_ms: Mapping[str, float]


class DeepSeekV4AttentionPredictor:
    """Exact measured lookup, then fitted interpolation within measured bounds."""

    def __init__(self, models: Mapping):
        self._models = {name: model for name, model in models.items() if name.startswith(MODEL_PREFIX)}
        if not self._models:
            raise ValueError("Measured V4 replay attention requires trained profiling data")
        thresholds = {
            getattr(model, "_frontier_dsv4_multi_stream_gemm_token_threshold", None)
            for model in self._models.values()
        }
        if None in thresholds or len(thresholds) != 1:
            raise ValueError("Measured V4 replay models require exactly one multi-stream GEMM token threshold")
        self.multi_stream_gemm_token_threshold = thresholds.pop()

    def scope_times(
        self, features: DeepSeekV4BatchFeatures, *, variant: str, graph_runtime_mode: str,
        plans_tile_schedule: bool, scopes: Collection[str],
    ) -> DeepSeekV4ScopeTimes:
        predictions = {target: {} for target in _TARGET_PREFIXES}
        for scope in scopes:
            key = dsv4_model_key(scope, variant, graph_runtime_mode, features.phase, plans_tile_schedule)
            for target in _TARGET_PREFIXES:
                name = dsv4_model_name(key, target)
                model = self._models.get(name)
                if model is None:
                    raise DeepSeekV4CoverageError(f"V4 replay coverage missing for key {key}, target {target}")
                columns = dsv4_feature_columns(scope, features.phase)
                if tuple(model._frontier_feature_names) != columns:
                    raise ValueError(f"V4 predictor feature schema mismatch for {name}")
                for column, (lower, upper) in model._frontier_dsv4_bounds.items():
                    value = features.values[column]
                    if not lower <= value <= upper:
                        raise DeepSeekV4CoverageError(f"V4 coverage missing for {name}: {column}={value} "
                                                     f"outside [{lower}, {upper}]")
                feature_tuple = tuple(float(features.values[column]) for column in columns)
                exact = model._frontier_exact_lookup
                value = (float(exact[feature_tuple]) if feature_tuple in exact else
                         float(model.predict(pd.DataFrame([feature_tuple], columns=list(columns)))[0]))
                if not np.isfinite(value) or (target == TARGET_TIME and value < 0):
                    raise ValueError(f"Invalid V4 {target} prediction for {name}: {value}")
                predictions[target][scope] = value
        return DeepSeekV4ScopeTimes(MappingProxyType(predictions[TARGET_TIME]),
                                   MappingProxyType(predictions[TARGET_GAP]))
