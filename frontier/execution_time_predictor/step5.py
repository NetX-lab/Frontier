"""Measured Step attention rows and fitting through the shared model cache.

CPU fixtures exercise this contract without admitting native profiling results.
Physical owners and workload semantics remain in ``frontier.attention.step5``.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from numbers import Real
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator

from frontier.attention.families import STEP5_ATTENTION_FAMILY
from frontier.attention.ops import AttentionPhase
from frontier.attention.profiling_mapping import validate_attention_profiling_dataframe
from frontier.attention.step5 import Step5AttentionWorkload, describe_step5_layer
from frontier.config.precision_type import PrecisionType
from frontier.operators.typed_contracts import (
    parse_typed_operator_contracts,
    validate_typed_operator_metadata,
)
from frontier.profiling.step5.rows import operator_contracts
from frontier.types import MeasurementType


# Graph code and actual measurement family share one policy table.
STEP5_GRAPH_MODES = {
    "none": (0, MeasurementType.KERNEL_ONLY),
    "piecewise": (1, MeasurementType.CUDA_EVENT),
    "full": (2, MeasurementType.CUDA_EVENT),
}
STEP5_VARIANTS = {
    name: index for index, name in enumerate(STEP5_ATTENTION_FAMILY.supported_variants)
}
STEP5_PHASES = {phase.value: index for index, phase in enumerate(AttentionPhase)}
# Producer column -> minimum accepted value. Only rope_theta is nonintegral.
STEP5_NUMERIC_COLUMNS = {
    "global_layer_id": 0, "num_tensor_parallel_workers": 1,
    "n_q_head": 1, "n_kv_head": 1, "head_size": 1,
    "local_q_heads": 1, "local_kv_heads": 1, "kv_replication": 1,
    "local_provider_groups": 0, "provider_replication": 0,
    "local_index_q_heads": 0, "local_index_k_heads": 0, "index_k_replication": 0,
    "proxy_dim": 0, "region_block_size": 1, "topk_regions": 1,
    "sliding_window": 1, "rope_theta": 0, "rotary_dim": 1, "block_size": 1,
    "batch_size": 1, "batch_num_tokens": 1,
    "batch_num_prefill_tokens": 0, "batch_num_decode_tokens": 0,
    "valid_tokens": 1, "padded_tokens": 0, "max_model_len": 1,
    "block_table_columns": 1, "score_regions": 0,
}
STEP5_FEATURE_COLUMNS = (
    *STEP5_NUMERIC_COLUMNS,
    "attention_variant_code", "phase_code", "graph_mode_code",
    "min_query_len", "max_query_len", "query_len_cv",
    "min_context_len", "max_context_len", "mean_context_len", "max_sequence_len",
)
STEP5_IDENTITY_COLUMNS = (
    "measurement_type", "attention_backend", "model_architecture_profile",
    "quant_signature", "model_dtype", "norm_dtype", "residual_dtype",
    "runtime_stack_signature", "device", "num_tensor_parallel_workers",
    "block_size", "cache_dtype", "physical_scope", "weight_source", "timing_basis",
)
STEP5_OPERATOR_NAMES = tuple(op.name for op in STEP5_ATTENTION_FAMILY.predictor_ops())
STEP5_SCOPE_NAMES = tuple(op.name for op in STEP5_ATTENTION_FAMILY.profiling_ops())
_TIME_STATS = ("min", "max", "mean", "median", "std", "count")
_TIMING_BASES = {
    MeasurementType.CUDA_EVENT: "cuda_event_native_runtime_latency",
    MeasurementType.KERNEL_ONLY: "correlated_kernel_plus_memcpy_duration_sum",
}


def step5_measurement_type(graph_mode: str) -> MeasurementType:
    """Select actual graph latency or eager device activity without relabelling."""
    try:
        return STEP5_GRAPH_MODES[graph_mode][1]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported Step graph_mode={graph_mode!r}") from exc


def _lengths(value, name: str, minimum: int) -> tuple[int, ...]:
    values = json.loads(value) if isinstance(value, str) else value
    if not isinstance(values, (tuple, list)) or not values or any(
        type(item) is not int or item < minimum for item in values
    ):
        raise ValueError(f"Step {name} requires integer lengths >= {minimum}")
    return tuple(values)


def _row_workload(row) -> Step5AttentionWorkload:
    queries = _lengths(row["query_lens"], "query_lens", 1)
    contexts = _lengths(row["context_lens"], "context_lens", 0)
    if len(queries) != len(contexts):
        raise ValueError("Step query_lens and context_lens must have equal length")
    return Step5AttentionWorkload(
        phase=row["phase"], query_lengths=queries,
        sequence_lengths=tuple(q + c for q, c in zip(queries, contexts)),
        physical_query_rows=int(row["valid_tokens"] + row["padded_tokens"]),
        block_size_tokens=int(row["block_size"]),
        block_table_columns=int(row["block_table_columns"]),
        max_model_len_tokens=int(row["max_model_len"]),
        region_tokens=int(row["region_block_size"]), topk_regions=int(row["topk_regions"]),
        graph_mode=row["graph_mode"],
    )


def step5_row_features(row) -> dict[str, float]:
    """Materialize numeric producer geometry and the real query/context shape."""
    features = {}
    for name, minimum in STEP5_NUMERIC_COLUMNS.items():
        value = row[name]
        if (not isinstance(value, Real) or isinstance(value, bool)
                or not math.isfinite(value) or value < minimum
                or (name != "rope_theta" and value != int(value))):
            raise ValueError(f"Invalid Step numeric feature {name}={value!r}")
        features[name] = float(value)
    workload = _row_workload(row)
    queries = workload.query_lengths
    contexts = tuple(length - query for length, query in zip(workload.sequence_lengths, queries))
    if (len(queries) != row["batch_size"] or sum(queries) != row["valid_tokens"]
            or row["batch_num_tokens"] != row["valid_tokens"]
            or row["batch_num_prefill_tokens"] + row["batch_num_decode_tokens"] != sum(queries)
            or row["score_regions"] != workload.score_regions):
        raise ValueError("Step query/context shape disagrees with producer token/region metadata")
    if (workload.phase == "prefill" and row["batch_num_decode_tokens"] != 0
            or workload.phase == "decode" and row["batch_num_prefill_tokens"] != 0):
        raise ValueError("Step phase disagrees with prefill/decode token counts")
    if workload.graph_mode == "full" and workload.phase != "decode":
        raise ValueError("Step full graph rows require decode phase")
    try:
        features.update(
            attention_variant_code=float(STEP5_VARIANTS[row["attention_variant"]]),
            phase_code=float(STEP5_PHASES[row["phase"]]),
            graph_mode_code=float(STEP5_GRAPH_MODES[row["graph_mode"]][0]),
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported Step variant/phase/graph label: {exc}") from exc
    mean_query = sum(queries) / len(queries)
    features.update(
        min_query_len=float(min(queries)), max_query_len=float(max(queries)),
        query_len_cv=math.sqrt(sum((q - mean_query) ** 2 for q in queries) / len(queries)) / mean_query,
        min_context_len=float(min(contexts)), max_context_len=float(max(contexts)),
        mean_context_len=sum(contexts) / len(contexts),
        max_sequence_len=float(max(workload.sequence_lengths)),
    )
    return features


def _validate_targets(frame: pd.DataFrame) -> None:
    """Reject missing actual launch counts before any estimator is fitted."""
    for _, row in frame.iterrows():
        active = set(json.loads(row["active_operators"]))
        counts = []
        for name in (*STEP5_SCOPE_NAMES, *STEP5_OPERATOR_NAMES):
            for stat in _TIME_STATS:
                column = f"time_stats.{name}.{stat}"
                value = row[column]
                if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Step timing coverage requires finite nonnegative {column}")
                if stat == "count":
                    if value < 2 or value != int(value):
                        raise ValueError(f"Step timing sample count must be integral and >= 2: {column}")
                    counts.append(value)
                else:
                    if stat != "std" and (name in active or name == "step5_layer_e2e") and value <= 0:
                        raise ValueError(f"Step active timing coverage requires positive {column}")
                    if name not in active and name != "step5_layer_e2e" and value != 0:
                        raise ValueError(f"Step inactive physical scope must be exactly zero: {column}")
            column = f"time_stats.{name}.kernel_count"
            value = row.get(column)
            if name in STEP5_OPERATOR_NAMES or name in active and value is not None and not pd.isna(value):
                if (not isinstance(value, Real) or isinstance(value, bool)
                        or not math.isfinite(value) or value < 1 or value != int(value)):
                    raise ValueError(
                        f"Step measured kernel coverage missing/invalid {column} at layer "
                        f"{row['global_layer_id']}; sample count and aggregate counts are not launch counts"
                    )
            elif value is not None and not pd.isna(value) and value != 0:
                raise ValueError(f"Step inactive scope has nonzero kernel count: {column}")
        if len(set(counts)) != 1:
            raise ValueError("Step timing statistics require matching sample counts")


def load_step5_attention_rows(path, replica_config, scheduler_config, measurement_type) -> pd.DataFrame:
    """Select and validate producer rows before shared-cache attention fitting."""
    measurement = (measurement_type if isinstance(measurement_type, MeasurementType)
                   else MeasurementType.from_string(measurement_type))
    if measurement not in _TIMING_BASES:
        raise ValueError(f"Step measured attention does not support {measurement.value}")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Step measured attention coverage file does not exist: {path}")
    frame = pd.read_csv(path).drop_duplicates()
    schema = replace(STEP5_ATTENTION_FAMILY, execution_enabled=True)
    validate_attention_profiling_dataframe(frame, schema, measurement_type=measurement)
    required = {
        *STEP5_IDENTITY_COLUMNS, *STEP5_NUMERIC_COLUMNS,
    "query_lens", "context_lens", "phase", "active_operators", "typed_operator_contracts",
        *(f"time_stats.step5_layer_e2e.{stat}" for stat in _TIME_STATS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Step measured attention coverage is missing columns: {missing}")
    if frame["timing_basis"].ne(_TIMING_BASES[measurement]).any():
        raise ValueError("Step timing_basis conflicts with actual measurement_type; do not relabel methods")
    selected_policy = frame["graph_mode"].map(step5_measurement_type).eq(measurement)
    frame = frame.loc[selected_policy].copy()
    config = replica_config.model_config
    expected = {
        "model_architecture_profile": config.get_model_architecture_profile().profile_id,
        "quant_signature": config.get_quant_signature(), "device": replica_config.device,
        "num_tensor_parallel_workers": replica_config.attn_tensor_parallel_size,
        "block_size": scheduler_config.block_size, "cache_dtype": "auto",
    }
    for name, value in expected.items():
        frame = frame.loc[frame[name].eq(value)].copy()
    if frame.empty:
        raise ValueError(f"Step measured attention coverage is unavailable for {measurement.value}: {expected}")
    identity_columns = (*STEP5_IDENTITY_COLUMNS, *(("evidence_kind",) if "evidence_kind" in frame else ()))
    validate_attention_profiling_dataframe(
        frame, schema, measurement_type=measurement, identity_columns=identity_columns,
    )
    metadata_by_layer = {}
    for _, row in frame.iterrows():
        features = step5_row_features(row)
        layer_id = int(row["global_layer_id"])
        if layer_id not in metadata_by_layer:
            metadata_by_layer[layer_id] = describe_step5_layer(
                config, layer_id, attention_tp_size=replica_config.attn_tensor_parallel_size,
                moe_tp_size=1, expert_parallel_size=1,
            )
        metadata = metadata_by_layer[layer_id]
        geometry = asdict(metadata.geometry)
        geometry.update(n_q_head=metadata.geometry.num_q_heads, n_kv_head=metadata.geometry.num_kv_heads,
                        head_size=metadata.geometry.head_dim, rope_theta=metadata.rope_theta,
                        rotary_dim=metadata.rotary_dim, sliding_window=config.sliding_window,
                        region_block_size=config.sparse_config["region_block_size"],
                        topk_regions=config.sparse_config["topk"])
        for name in STEP5_NUMERIC_COLUMNS.keys() & geometry.keys():
            if features[name] != geometry[name]:
                raise ValueError(f"Step source geometry disagrees with {name} at layer {layer_id}")
        if row["attention_variant"] != metadata.attention_spec.variant_id:
            raise ValueError(f"Step attention_variant disagrees with layer {layer_id}")
        if PrecisionType.from_torch_dtype(row["model_dtype"]) is not metadata.model_precision:
            raise ValueError("Step model_dtype conflicts with requested model precision")
        if PrecisionType.from_torch_dtype(row["norm_dtype"]) is not metadata.norm_precision:
            raise ValueError("Step norm_dtype conflicts with source metadata")
        if PrecisionType.from_torch_dtype(row["residual_dtype"]) is not metadata.residual_precision:
            raise ValueError("Step residual_dtype conflicts with source metadata")
        if row["physical_scope"] != "native_attention_layer_without_tp_allreduce":
            raise ValueError("Step physical_scope must exclude TP allreduce")
        if row["max_model_len"] != scheduler_config.max_model_len:
            raise ValueError("Step measured max_model_len differs from requested serving context")
        scopes = metadata.attention_scopes(_row_workload(row))
        if json.loads(row["fusion"]) != [[name, list(work)] for name, work in scopes]:
            raise ValueError("Step fusion scopes differ from source physical ownership")
        active = json.loads(row["active_operators"])
        if len(active) != len(set(active)) or set(active) != {name for name, _ in scopes}:
            raise ValueError("Step active_operators differ from source physical scopes")
        contracts = parse_typed_operator_contracts(row["typed_operator_contracts"])
        expected_contracts = {
            name: contract for name, contract in operator_contracts(
                config, layer_id, replica_config.attn_tensor_parallel_size,
            ).items() if name in STEP5_SCOPE_NAMES
        }
        if set(contracts) != set(expected_contracts):
            raise ValueError("Step typed scopes must cover exactly the registered attention operators")
        for name, contract in contracts.items():
            validate_typed_operator_metadata(
                contract, operator_name=name, expected_metadata=expected_contracts[name],
                architecture_profile=config.get_model_architecture_profile(),
            )
    derived_precision = frame["model_dtype"].map(lambda dtype: PrecisionType.from_torch_dtype(dtype).name)
    if "profiling_precision" in frame and any(
        PrecisionType.from_string(value).name != derived
        for value, derived in zip(frame["profiling_precision"], derived_precision)
    ):
        raise ValueError("Step profiling_precision conflicts with explicit model_dtype")
    frame["profiling_precision"] = derived_precision
    _validate_targets(frame)
    features = pd.DataFrame([step5_row_features(row) for _, row in frame.iterrows()], index=frame.index)
    for name in STEP5_FEATURE_COLUMNS:
        frame[name] = features[name]
    frame.attrs["measurement_identity"] = {
        name: (frame.iloc[0][name].item() if hasattr(frame.iloc[0][name], "item")
               else frame.iloc[0][name]) for name in identity_columns
    }
    return frame


def set_step5_model_coverage(model, frame: pd.DataFrame) -> None:
    """Persist only the rows and physical domains that actually trained this operator."""
    model._frontier_step5_coverage = frozenset(
        tuple(float(row[name]) for name in STEP5_FEATURE_COLUMNS) for _, row in frame.iterrows()
    )
    model._frontier_feature_bounds = {
        name: (float(frame[name].min()), float(frame[name].max())) for name in STEP5_FEATURE_COLUMNS
    }
    model._frontier_step5_domains = {
        name: tuple(sorted(float(value) for value in frame[name].unique()))
        for name in ("global_layer_id", "attention_variant_code", "phase_code", "graph_mode_code")
    }


def train_step5_attention_models(manager, df, predictor_config, training_context) -> dict[str, BaseEstimator]:
    """Fit measured durations and actual counts in the existing atomic cache chain."""
    _validate_targets(df)
    context = dict(training_context)
    identity = dict(context.get("measurement_identity", {}))
    for name, value in df.attrs["measurement_identity"].items():
        if name in identity and identity[name] != value:
            raise ValueError(f"Step training context conflicts with source measurement identity: {name}")
        identity[name] = value
    identity["feature_columns"] = list(STEP5_FEATURE_COLUMNS)
    context["measurement_identity"] = identity
    context["step5_training"] = True
    models = {}
    for name in STEP5_OPERATOR_NAMES:
        selected = df
        targets = [f"time_stats.{name}.median", f"time_stats.{name}.kernel_count"]
        # Repeated duration measurements use the shared exact-lookup aggregation.
        if (selected.groupby(list(STEP5_FEATURE_COLUMNS))[targets[1]].nunique() > 1).any():
            raise ValueError(f"Conflicting Step measured kernel counts for physical scope {name}")
        pair = []
        for model_name, target in zip((name, f"{name}_kernel_count"), targets):
            pair.append(manager._train_single_model(
                model_name=model_name, df=selected, feature_cols=list(STEP5_FEATURE_COLUMNS),
                target_col=target, execution_time_predictor_config=predictor_config,
                training_context=context, persist_exact_lookup=True, count_kernels=False,
            ))
            models[model_name] = pair[-1]
        pair[1]._frontier_operator_name = name
        pair[0]._frontier_kernel_count_model = pair[1]
    return models
