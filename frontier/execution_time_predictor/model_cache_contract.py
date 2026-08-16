"""Shared identity and metadata contract for persisted predictor models.

Model artifacts are produced by three code paths in Frontier: standalone
training, the shared E2E model manager, and the predictor's on-demand trainer.
This module is intentionally limited to artifact identity and structural
metadata validation.  Interpolation policy remains in
``prediction_cache_contract`` so a domain policy can evolve independently.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from pathlib import Path
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from frontier.execution_time_predictor.prediction_cache_contract import (
    PREDICTION_CACHE_CONTRACT_VERSION,
    validate_feature_domain_descriptor,
)


MODEL_CACHE_CONTRACT_VERSION = 1


def _materialize_runtime_domain_metadata(
    feature_domain: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Fill deterministic v3 fields for an in-process producer descriptor.

    This is a producer-side migration aid for handcrafted/legacy training
    inputs.  Persisted model loading remains strict and rejects artifacts that
    do not contain these fields.
    """

    normalized = dict(feature_domain)
    kind = normalized.get("domain_kind")
    if "runtime_prediction_policy" not in normalized:
        normalized["runtime_prediction_policy"] = (
            "measured_only"
            if kind == "exact_rows"
            or (
                normalized.get("on_demand_policy") == "bounded"
                and normalized.get("operator_name") != "attn_kv_cache_save"
            )
            else "allow_model_prediction"
        )
    if "physical_bounds" not in normalized:
        bounds = normalized.get("bounds", {})
        if isinstance(bounds, Mapping):
            normalized["physical_bounds"] = {
                str(name): {"min": 0.0}
                for name in feature_names
                if name in bounds
            }
        else:
            normalized["physical_bounds"] = {}
    if kind == "integer_interval_interpolation" and "axis_values" not in normalized:
        bounds = normalized.get("bounds", {})
        if isinstance(bounds, Mapping) and len(feature_names) == 1:
            bound = bounds.get(feature_names[0], {})
            if isinstance(bound, Mapping) and {"min", "max"}.issubset(bound):
                normalized["axis_values"] = {
                    feature_names[0]: [bound["min"], bound["max"]]
                }
    return normalized
OPERATOR_BINDING_CONTRACT_VERSION = 1


def build_exact_feature_lookup(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> dict[tuple[float, ...], float]:
    """Build a direct-result map from observed finite profile rows.

    Structural rows with a missing target are not measurements and are omitted.
    Malformed non-numeric or non-finite targets fail fast.  Repeated rows are
    combined with the same mean-of-observed-medians rule used by existing
    producers.
    """

    if df.empty:
        return {}
    if target_col not in df.columns:
        raise ValueError(
            f"Exact feature lookup target column {target_col!r} is missing."
        )
    for feature_col in feature_cols:
        non_scalar_rows = df[feature_col].map(
            lambda value: isinstance(value, (list, tuple, dict, set))
        )
        if bool(non_scalar_rows.any()):
            raise ValueError(
                "Exact feature lookup contains non-scalar feature values in "
                f"column {feature_col!r}; scalar numeric values are required."
            )

    target_values = pd.to_numeric(df[target_col], errors="coerce")
    non_numeric = df[target_col].notna() & target_values.isna()
    if bool(non_numeric.any()):
        raise ValueError(
            f"Exact feature lookup target column {target_col!r} contains "
            "non-numeric values."
        )
    non_finite = target_values.notna() & ~np.isfinite(target_values)
    if bool(non_finite.any()):
        raise ValueError(
            f"Exact feature lookup target column {target_col!r} contains "
            "non-finite values."
        )
    negative = target_values.notna() & (target_values < 0.0)
    if bool(negative.any()):
        raise ValueError(
            f"Exact feature lookup target column {target_col!r} contains "
            "negative timing values."
        )

    observed = df.loc[target_values.notna(), [*feature_cols, target_col]].copy()
    if observed.empty:
        return {}
    observed[target_col] = target_values.loc[observed.index].astype(float)
    grouped = observed.groupby(list(feature_cols), dropna=False)[target_col].mean()
    lookup: dict[tuple[float, ...], float] = {}
    for key, value in grouped.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        lookup[tuple(float(item) for item in key_tuple)] = float(value)
    return lookup

# These models consume the attention-layer feature transformation whose KV
# cache axis is quantized by ``kv_cache_prediction_granularity``.  The option
# is part of their artifact identity; it is deliberately not included for
# one-dimensional linear/FFN operators where it has no effect.
ATTENTION_GRID_MODEL_NAMES = frozenset(
    {
        "attn_kv_cache_save",
        "attn_prefill",
        "attn_decode",
        "attn_prefill_mixed",
        "attn_decode_in_mixed",
    }
)

_BINDING_CONTEXT_FIELDS = (
    "cluster_type",
    "device",
    "model_name",
    "model_arch",
    "model_architecture_profile",
    "tensor_parallel_size",
    "attn_tensor_parallel_size",
    "moe_tensor_parallel_size",
    "expert_parallel_size",
    "moe_expert_parallel_size",
    "num_pipeline_stages",
    "block_size",
    "use_qk_norm",
    "routing_runtime_path",
    "gating_runtime_context",
    "n_head",
    "n_q_head",
    "n_kv_head",
    "n_embd",
    "n_expanded_embd",
    "head_size",
    "q_head_dim",
    "kv_head_dim",
    "num_experts",
    "num_experts_per_device",
    "router_topk",
    "hidden_dim",
    "expert_hidden_dim",
    "quant_signature",
    "profiling_precision",
    "measurement_type",
)

_BINDING_DATAFRAME_FIELDS = (
    "n_head",
    "n_q_head",
    "n_kv_head",
    "n_embd",
    "n_expanded_embd",
    "padded_n_embd",
    "padded_n_expanded_embd",
    "vocab_size",
    "use_gated_mlp",
    "use_gated",
    "use_qk_norm",
    "attn_output_gate",
    "share_expert_dim",
    "share_q_dim",
    "block_size",
    "num_tensor_parallel_workers",
    "expert_parallel_size",
    "attention_backend",
    "model_arch",
    "model_architecture_profile",
    "quant_signature",
    "measurement_type",
    "num_experts",
    "num_experts_per_device",
    "router_topk",
    "hidden_dim",
    "expert_hidden_dim",
    "head_size",
    "q_head_dim",
    "kv_head_dim",
    "num_workers",
    "devices_per_node",
    "max_devices_per_node",
    "pp_world_size",
    "pipeline_parallel_size",
    "stage_id",
    "rank",
    "size",
    "collective",
    "routing_runtime_path",
    "routing_assignment_policy",
    "routing_weight_policy",
    "routing_uses_router_logits",
    "gating_runtime_context",
    "gating_runtime_context_impl",
    "moe_grouped_gemm_backend",
    "is_step2_mini",
    "is_moe_model",
)

# Only these producer-independent context fields may supplement the selected
# profiling rows.  Other runtime fields (cluster role, requested TP, PP
# topology, etc.) are intentionally excluded here: the filtered dataframe is
# the source of truth for the physical profile slice, and including producer-
# specific runtime fields would make standalone/shared/runtime artifacts
# disagree for the same rows.
_CANONICAL_CONTEXT_FIELDS = (
    "device",
    "model_name",
    "model_arch",
    "model_architecture_profile",
    "quant_signature",
)


def resolve_training_cv_splits(k_fold_cv_splits: Any, row_count: int) -> int:
    """Return the deterministic CV policy shared by every model producer."""
    if (
        isinstance(k_fold_cv_splits, bool)
        or not isinstance(k_fold_cv_splits, (int, np.integer))
        or int(k_fold_cv_splits) < 2
    ):
        raise ValueError(
            f"k_fold_cv_splits must be an integer >= 2, got {k_fold_cv_splits!r}."
        )
    if isinstance(row_count, bool) or not isinstance(row_count, (int, np.integer)):
        raise ValueError(f"row_count must be a non-negative integer, got {row_count!r}.")
    rows = int(row_count)
    if rows < 0:
        raise ValueError(f"row_count must be a non-negative integer, got {row_count!r}.")
    return min(int(k_fold_cv_splits), rows) if rows >= 2 else 2


def build_training_options(
    model_name: str,
    *,
    k_fold_cv_splits: Any,
    kv_cache_prediction_granularity: Any = None,
) -> dict[str, Any]:
    """Build the training-identity options shared by every model producer.

    Keeping this projection in the cache-contract module prevents standalone,
    shared-manager, and predictor-side training from silently selecting
    different artifact identities when the attention KV feature transform
    changes.
    """

    options: dict[str, Any] = {
        "k_fold_cv_splits": int(k_fold_cv_splits),
    }
    if model_name in ATTENTION_GRID_MODEL_NAMES:
        if (
            isinstance(kv_cache_prediction_granularity, bool)
            or not isinstance(kv_cache_prediction_granularity, Integral)
            or int(kv_cache_prediction_granularity) <= 0
        ):
            raise ValueError(
                f"{model_name} cache identity requires "
                "kv_cache_prediction_granularity to be a positive integer; "
                f"got {kv_cache_prediction_granularity!r}."
            )
        options["kv_cache_prediction_granularity"] = int(kv_cache_prediction_granularity)
    return options


def _qualified_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _canonical_value(value: Any) -> Any:
    """Convert estimator parameters to deterministic JSON-compatible values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _canonical_value(value.item())
        return [_canonical_value(item) for item in value.tolist()]
    if isinstance(value, Enum):
        return {"enum": _qualified_name(value), "value": value.value}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Model cache identity cannot encode non-finite value {value!r}.")
        return int(numeric) if numeric.is_integer() else numeric
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return {"class": _qualified_name(value)}
    if isinstance(value, BaseEstimator):
        params = value.get_params(deep=False)
        return {
            "class": _qualified_name(value),
            "params": {
                str(key): _canonical_value(params[key])
                for key in sorted(params)
            },
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical = [_canonical_value(item) for item in value]
        return sorted(canonical, key=lambda item: json.dumps(item, sort_keys=True))
    if callable(value):
        return {"callable": _qualified_name(value)}
    raise TypeError(
        "Unsupported value in model cache identity: "
        f"{value!r} ({type(value).__name__})."
    )


def _normalise_label(value: Any, *, field: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"Model cache identity requires a non-empty {field}.")
    return normalized.upper() if field in {"profiling_precision", "measurement_type"} else normalized


def _infer_operator_family(operator_name: str) -> str:
    if operator_name.startswith("attn_"):
        return "attention"
    if operator_name.startswith("moe_"):
        return "moe"
    if operator_name.startswith("share_expert_"):
        return "share_expert"
    if operator_name.startswith("mlp_"):
        return "ffn"
    if operator_name in {
        "input_layernorm",
        "post_attention_layernorm",
        "add",
        "emb",
        "schedule",
        "sampler_e2e",
        "prepare_inputs_e2e",
        "process_model_outputs",
        "ray_comm_time",
    }:
        return "runtime_overhead"
    if operator_name in {"all_reduce", "send_recv"}:
        return "communication"
    return "compute"


def _non_missing_values(series: pd.Series) -> list[Any]:
    values: list[Any] = []
    for value in series.tolist():
        if value is None:
            continue
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            continue
        values.append(value)
    return values


def build_operator_binding(
    operator_name: str,
    *,
    context: Mapping[str, Any] | None = None,
    dataframe: pd.DataFrame | None = None,
    operator_family: str | None = None,
    operator_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical physical/operator identity for a model artifact."""
    normalized_name = str(operator_name).strip()
    if not normalized_name:
        raise ValueError("Operator binding requires a non-empty operator_name.")
    source_context = context or {}
    if not isinstance(source_context, Mapping):
        raise TypeError(
            f"Operator binding context must be a mapping, got {type(source_context).__name__}."
        )
    if operator_binding is not None and not isinstance(operator_binding, Mapping):
        raise TypeError(
            "Operator binding metadata must be a mapping, "
            f"got {type(operator_binding).__name__}."
        )

    binding: dict[str, Any] = {
        str(key): _canonical_value(value)
        for key, value in dict(operator_binding or {}).items()
    }
    declared_operator = binding.get("operator_name")
    if declared_operator is not None and str(declared_operator) != normalized_name:
        raise ValueError(
            "Operator binding operator_name mismatch: "
            f"expected={normalized_name!r}, actual={declared_operator!r}."
        )
    binding["contract_version"] = OPERATOR_BINDING_CONTRACT_VERSION
    binding["operator_name"] = normalized_name

    family = (
        operator_family
        or binding.get("operator_family")
        or source_context.get("operator_family")
    )
    family = str(family).strip() if family is not None else ""
    if not family:
        family = _infer_operator_family(normalized_name)
    binding["operator_family"] = family

    for field in _BINDING_CONTEXT_FIELDS:
        value = source_context.get(field)
        if value is not None:
            # Keep the artifact identity producer-independent.  Structural
            # fields represented by the selected dataframe (including TP,
            # block size, precision, and measurement family) are bound below;
            # only stable profile identity fields may come from context.
            if field not in _CANONICAL_CONTEXT_FIELDS:
                continue
            if dataframe is not None and field in dataframe.columns:
                continue
            canonical_value = _canonical_value(value)
            if field in binding and binding[field] != canonical_value:
                raise ValueError(
                    f"Operator binding field {field!r} conflicts with the "
                    f"training context: binding={binding[field]!r}, "
                    f"context={canonical_value!r}."
                )
            binding[field] = canonical_value

    structural: dict[str, Any] = {}
    supplied_structure = binding.get("profile_structure")
    if supplied_structure is not None:
        if not isinstance(supplied_structure, Mapping):
            raise ValueError("Operator binding profile_structure must be a mapping.")
        structural.update(
            {
                str(key): _canonical_value(value)
                for key, value in supplied_structure.items()
            }
        )
    if dataframe is not None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Operator binding dataframe must be a pandas DataFrame, "
                f"got {type(dataframe).__name__}."
            )
        for field in _BINDING_DATAFRAME_FIELDS:
            if field not in dataframe.columns:
                continue
            values = _non_missing_values(dataframe[field])
            if not values:
                continue
            canonical_values = [_canonical_value(value) for value in values]
            unique_values = {
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                : value
                for value in canonical_values
            }
            ordered_values = [
                unique_values[key]
                for key in sorted(unique_values)
            ]
            dataframe_value = (
                ordered_values[0]
                if len(ordered_values) == 1
                else ordered_values
            )
            if field in structural and structural[field] != dataframe_value:
                raise ValueError(
                    f"Operator binding profile field {field!r} conflicts with "
                    f"the training dataframe: binding={structural[field]!r}, "
                    f"dataframe={dataframe_value!r}."
                )
            structural[field] = dataframe_value
    if structural:
        binding["profile_structure"] = structural
    else:
        binding.pop("profile_structure", None)

    return _canonical_value(binding)


def build_canonical_operator_binding(
    operator_name: str,
    *,
    dataframe: pd.DataFrame | None = None,
    context: Mapping[str, Any] | None = None,
    operator_family: str | None = None,
    operator_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the producer-shared operator identity.

    All model producers call this seam.  It deliberately ignores role-local
    context that is not encoded in the selected profiling rows, preventing a
    standalone trainer, shared manager, and runtime predictor from generating
    different cache hashes for the same physical artifact.
    """

    return build_operator_binding(
        operator_name,
        context=context,
        dataframe=dataframe,
        operator_family=operator_family,
        operator_binding=operator_binding,
    )


def resolve_operator_binding(
    operator_name: str,
    *,
    context: Mapping[str, Any] | None = None,
    dataframe: pd.DataFrame | None = None,
    operator_family: str | None = None,
    operator_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one canonical binding for hash, metadata, and cache validation."""

    return build_canonical_operator_binding(
        operator_name,
        context=context,
        dataframe=dataframe,
        operator_family=operator_family,
        operator_binding=operator_binding,
    )


def build_runtime_operator_binding(
    operator_name: str,
    *,
    dataframe: pd.DataFrame | None = None,
    context: Mapping[str, Any] | None = None,
    model_config: Any | None = None,
    replica_config: Any | None = None,
    cluster_type: Any | None = None,
    profiling_precision: Any | None = None,
    measurement_type: Any | None = None,
) -> dict[str, Any]:
    """Build the canonical binding shared by standalone and runtime producers."""

    merged: dict[str, Any] = dict(context or {})

    def set_if_absent(name: str, value: Any) -> None:
        if value is not None and name not in merged:
            merged[name] = value

    set_if_absent("cluster_type", cluster_type)
    set_if_absent("profiling_precision", profiling_precision)
    set_if_absent("measurement_type", measurement_type)

    if replica_config is not None:
        for field in (
            "device",
            "model_name",
            "attn_tensor_parallel_size",
            "attn_data_parallel_size",
            "moe_tensor_parallel_size",
            "moe_expert_parallel_size",
            "expert_parallel_size",
            "num_pipeline_stages",
        ):
            set_if_absent(field, getattr(replica_config, field, None))
        scheduler_config = getattr(replica_config, "replica_scheduler_config", None)
        if scheduler_config is not None:
            set_if_absent("block_size", getattr(scheduler_config, "block_size", None))

    if model_config is not None:
        model_name_getter = getattr(model_config, "get_name", None)
        set_if_absent(
            "model_name",
            model_name_getter() if callable(model_name_getter) else None,
        )
        model_arch_getter = getattr(model_config, "get_model_arch", None)
        set_if_absent(
            "model_arch",
            model_arch_getter() if callable(model_arch_getter) else
            getattr(model_config, "model_arch", None),
        )
        profile_getter = getattr(
            model_config, "get_model_architecture_profile", None
        )
        profile = profile_getter() if callable(profile_getter) else None
        set_if_absent(
            "model_architecture_profile",
            getattr(profile, "profile_id", None)
            or getattr(model_config, "model_architecture_profile", None),
        )
        quant_getter = getattr(model_config, "get_quant_signature", None)
        set_if_absent(
            "quant_signature",
            quant_getter() if callable(quant_getter) else None,
        )
        for binding_name, model_field in (
            ("n_head", "num_q_heads"),
            ("n_q_head", "num_q_heads"),
            ("n_kv_head", "num_kv_heads"),
            ("n_embd", "embedding_dim"),
            ("n_expanded_embd", "mlp_hidden_dim"),
            ("num_experts", "num_experts"),
            ("router_topk", "num_experts_per_tok"),
            ("hidden_dim", "embedding_dim"),
        ):
            set_if_absent(binding_name, getattr(model_config, model_field, None))
        head_dim_getter = getattr(model_config, "get_head_dim", None)
        set_if_absent(
            "head_size",
            head_dim_getter() if callable(head_dim_getter) else None,
        )

    return build_canonical_operator_binding(
        operator_name,
        context=merged,
        dataframe=dataframe,
    )


def _validate_operator_binding(
    binding: Any,
    model_name: str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise ValueError(
            f"Cached model {model_name} requires operator binding metadata."
        )
    version = binding.get("contract_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != OPERATOR_BINDING_CONTRACT_VERSION
    ):
        raise ValueError(
            f"Cached model {model_name} has unsupported operator binding contract "
            f"version {version!r}; expected {OPERATOR_BINDING_CONTRACT_VERSION}. Retrain it."
        )
    operator = binding.get("operator_name")
    if not isinstance(operator, str) or operator.strip() != str(model_name):
        raise ValueError(
            f"Cached model {model_name} operator binding mismatch: "
            f"expected operator_name={model_name!r}, actual={operator!r}."
        )
    family = binding.get("operator_family")
    if not isinstance(family, str) or not family.strip():
        raise ValueError(
            f"Cached model {model_name} operator binding requires operator_family."
        )
    structure = binding.get("profile_structure")
    if structure is not None and not isinstance(structure, Mapping):
        raise ValueError(
            f"Cached model {model_name} operator binding has invalid profile_structure."
        )
    canonical = _canonical_value(dict(binding))
    if expected is not None:
        expected_canonical = resolve_operator_binding(
            str(model_name),
            operator_binding=expected,
        )
        if canonical != expected_canonical:
            raise ValueError(
                f"Cached model {model_name} operator binding mismatch: "
                f"expected={expected_canonical!r}, actual={canonical!r}. Retrain it."
            )
    return canonical


def dataframe_training_digest(
    dataframe: pd.DataFrame,
    feature_names: Sequence[str],
    target_col: str,
) -> str:
    """Digest the ordered training rows used by an estimator."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(dataframe).__name__}.")
    ordered_features = tuple(str(name) for name in feature_names)
    if not ordered_features or len(set(ordered_features)) != len(ordered_features):
        raise ValueError(f"Feature schema must be non-empty and unique: {ordered_features!r}")
    target = str(target_col)
    if target in ordered_features:
        raise ValueError(f"Target column {target!r} duplicates a feature column.")
    required = [*ordered_features, target]
    missing = [name for name in required if name not in dataframe.columns]
    if missing:
        raise ValueError(f"Training dataframe is missing columns: {missing!r}")
    training_frame = dataframe.loc[:, required]
    # ``to_json`` with split orientation includes row order and column order;
    # retain dtypes separately because JSON normalizes integer/float columns.
    payload = {
        "dtypes": [str(dtype) for dtype in training_frame.dtypes],
        "frame": training_frame.to_json(
            orient="split", date_format="iso", double_precision=15
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Public descriptive alias used by callers that treat the selected training
# frame as the dataframe identity component.
dataframe_digest = dataframe_training_digest


def _estimator_recipe(
    estimator: BaseEstimator,
    hyperparameter_grid: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            f"Expected sklearn BaseEstimator, got {type(estimator).__name__}."
        )
    params = estimator.get_params(deep=True)
    return {
        "class": _qualified_name(estimator),
        "params": {str(key): _canonical_value(params[key]) for key in sorted(params)},
        "hyperparameter_grid": _canonical_value(hyperparameter_grid),
    }


def estimator_params_digest(estimator: BaseEstimator) -> str:
    """Digest the fitted estimator's actual hyperparameter values."""
    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            f"Expected sklearn BaseEstimator, got {type(estimator).__name__}."
        )
    payload = {
        "class": _qualified_name(estimator),
        "params": {
            str(key): _canonical_value(value)
            for key, value in sorted(estimator.get_params(deep=True).items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_model_cache_hash(
    *,
    model_name: str,
    dataframe: pd.DataFrame,
    profiling_precision: Any,
    measurement_type: Any,
    feature_names: Sequence[str],
    target_col: str,
    estimator: BaseEstimator,
    hyperparameter_grid: Mapping[str, Any],
    training_options: Mapping[str, Any] | None = None,
    feature_domain: Mapping[str, Any] | None = None,
    operator_binding: Mapping[str, Any] | None = None,
) -> str:
    """Build the shared, versioned identity for one trained model artifact.

    ``feature_domain`` is part of model identity when supplied.  A model's
    estimator parameters can be identical while its legal interpolation domain
    differs, so those artifacts must not share a cache filename.
    """
    if not str(model_name).strip():
        raise ValueError("Model cache identity requires a non-empty model_name.")
    if operator_binding is None:
        raise ValueError(
            "Model cache identity operator_binding is required; "
            "build it from the selected profiling rows before hashing."
        )
    if feature_domain is not None:
        if not isinstance(feature_domain, Mapping):
            raise ValueError("Model cache feature_domain must be a mapping.")
        feature_domain = _materialize_runtime_domain_metadata(
            feature_domain,
            feature_names,
        )
        _validate_feature_domain_metadata(
            feature_domain,
            feature_names,
            model_name=model_name,
        )
    canonical_binding = _validate_operator_binding(
        resolve_operator_binding(
            str(model_name),
            operator_binding=operator_binding,
        ),
        str(model_name),
    )

    payload = {
        "contract_version": MODEL_CACHE_CONTRACT_VERSION,
        "model_name": str(model_name),
        "dataframe_digest": dataframe_training_digest(
            dataframe, feature_names, target_col
        ),
        "profiling_precision": _normalise_label(
            profiling_precision, field="profiling_precision"
        ),
        "measurement_type": _normalise_label(
            measurement_type, field="measurement_type"
        ),
        "feature_names": [str(name) for name in feature_names],
        "target_col": str(target_col),
        "estimator_recipe": _estimator_recipe(estimator, hyperparameter_grid),
        "training_options": _canonical_value(training_options or {}),
        "operator_binding": canonical_binding,
    }
    if feature_domain is not None:
        payload["feature_domain"] = _canonical_value(feature_domain)
    encoded = json.dumps(
        _canonical_value(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def attach_model_cache_metadata(
    model: BaseEstimator,
    *,
    model_name: str,
    model_hash: str,
    feature_names: Sequence[str],
    target_col: str,
    feature_domain: Mapping[str, Any] | None = None,
    operator_binding: Mapping[str, Any] | None = None,
) -> None:
    """Attach the structural metadata required for a valid model cache entry."""
    normalized_model_name = str(model_name).strip()
    if not normalized_model_name:
        raise ValueError("Cached model metadata requires a non-empty model_name.")
    if not isinstance(model_hash, str) or not model_hash:
        raise ValueError("A non-empty model_hash is required for cached models.")
    names = [str(name) for name in feature_names]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Cached model feature schema must be unique and non-empty: {names!r}")
    target = str(target_col)
    if target in names or not target:
        raise ValueError(f"Invalid cached model target column: {target_col!r}")
    if feature_domain is None:
        raise ValueError("Cached models require a feature-domain contract.")
    if not isinstance(feature_domain, Mapping):
        raise ValueError("Cached model feature domain must be a mapping.")
    feature_domain = _materialize_runtime_domain_metadata(
        feature_domain,
        names,
    )
    if operator_binding is None:
        raise ValueError(
            "Cached model operator_binding is required; retrain it through the "
            "shared model-cache producer contract."
        )
    _validate_feature_domain_metadata(
        feature_domain,
        names,
        model_name=normalized_model_name,
    )
    canonical_binding = _validate_operator_binding(
        resolve_operator_binding(
            normalized_model_name,
            operator_binding=operator_binding,
        ),
        normalized_model_name,
    )
    domain_version = feature_domain.get("contract_version")
    if (
        isinstance(domain_version, bool)
        or not isinstance(domain_version, int)
        or domain_version != PREDICTION_CACHE_CONTRACT_VERSION
    ):
        raise ValueError(
            "Cached model feature domain has an unsupported contract version: "
            f"{feature_domain.get('contract_version')!r}."
        )
    setattr(model, "_frontier_model_cache_contract_version", MODEL_CACHE_CONTRACT_VERSION)
    setattr(model, "_frontier_model_name", normalized_model_name)
    setattr(model, "_frontier_model_hash", model_hash)
    setattr(model, "_frontier_estimator_params_digest", estimator_params_digest(model))
    setattr(model, "_frontier_feature_names", names)
    setattr(model, "_frontier_target_col", target)
    setattr(model, "_frontier_feature_domain", dict(feature_domain))
    setattr(model, "_frontier_operator_binding", canonical_binding)


def _validate_feature_domain_metadata(
    feature_domain: Mapping[str, Any],
    expected_names: Sequence[str],
    *,
    model_name: str = "model",
    require_runtime_policy_fields: bool = False,
) -> None:
    """Validate the complete, kind-specific domain descriptor."""
    if require_runtime_policy_fields and "runtime_prediction_policy" not in feature_domain:
        raise ValueError(
            f"Cached model {model_name} feature domain is missing "
            "runtime_prediction_policy; retrain it."
        )
    if require_runtime_policy_fields and "physical_bounds" not in feature_domain:
        raise ValueError(
            f"Cached model {model_name} feature domain is missing physical_bounds; "
            "retrain it."
        )
    domain_kind = feature_domain.get("domain_kind")
    if require_runtime_policy_fields and domain_kind in {
        "integer_interval_interpolation",
        "verified_cartesian_interpolation",
        "conditional_interpolation",
        "regression_extrapolation",
    } and "axis_values" not in feature_domain:
        raise ValueError(
            f"Cached model {model_name} feature domain is missing axis_values; retrain it."
        )
    descriptor_operator = feature_domain.get("operator_name")
    if not isinstance(descriptor_operator, str) or not descriptor_operator.strip():
        raise ValueError(
            f"Cached model {model_name} feature domain requires a non-empty "
            "operator_name."
        )
    if descriptor_operator != str(model_name):
        raise ValueError(
            f"Cached model {model_name} feature domain operator mismatch: "
            f"expected={model_name!r}, actual={descriptor_operator!r}."
        )
    domain_version = feature_domain.get("contract_version")
    if (
        isinstance(domain_version, bool)
        or not isinstance(domain_version, int)
        or domain_version != PREDICTION_CACHE_CONTRACT_VERSION
    ):
        raise ValueError(
            "Cached model feature domain has unsupported contract version "
            f"{domain_version!r}; expected {PREDICTION_CACHE_CONTRACT_VERSION}."
        )
    try:
        validate_feature_domain_descriptor(
            feature_domain,
            expected_names,
            model_name=f"Cached model {model_name}",
            operator_name=(
                model_name
                if model_name != "model"
                else feature_domain.get("operator_name")
            ),
        )
    except ValueError as exc:
        # Keep the cache-boundary wording stable for callers and tests while
        # preserving the specific malformed-field explanation from the shared
        # prediction-domain validator.
        raise ValueError(str(exc)) from exc


def validate_cached_model(
    model_name: str,
    model: Any,
    *,
    expected_model_hash: str,
    feature_names: Sequence[str],
    target_col: str,
    operator_binding: Mapping[str, Any] | None = None,
) -> BaseEstimator:
    """Reject stale, malformed, or mismatched persisted model metadata."""
    if not isinstance(model, BaseEstimator):
        raise ValueError(
            f"Cached model {model_name} is not a sklearn estimator: {type(model).__name__}."
        )
    actual_version = getattr(model, "_frontier_model_cache_contract_version", None)
    if (
        isinstance(actual_version, bool)
        or not isinstance(actual_version, int)
        or actual_version != MODEL_CACHE_CONTRACT_VERSION
    ):
        raise ValueError(
            f"Cached model {model_name} has unsupported cache contract version "
            f"{actual_version!r}; expected {MODEL_CACHE_CONTRACT_VERSION}. Retrain it."
        )
    actual_model_name = getattr(model, "_frontier_model_name", None)
    if actual_model_name != str(model_name):
        raise ValueError(
            f"Cached model {model_name} operator identity mismatch: "
            f"expected={model_name!r}, actual={actual_model_name!r}. Retrain it."
        )
    actual_binding = _validate_operator_binding(
        getattr(model, "_frontier_operator_binding", None),
        str(model_name),
        expected=operator_binding,
    )
    if not isinstance(expected_model_hash, str) or not expected_model_hash:
        raise ValueError("Expected model cache hash must be a non-empty string.")
    actual_hash = getattr(model, "_frontier_model_hash", None)
    if not isinstance(actual_hash, str) or actual_hash != expected_model_hash:
        raise ValueError(
            f"Cached model {model_name} hash mismatch: expected={expected_model_hash!r}, "
            f"actual={actual_hash!r}. Remove/retrain the stale cache."
        )
    actual_params_digest = getattr(model, "_frontier_estimator_params_digest", None)
    expected_params_digest = estimator_params_digest(model)
    if actual_params_digest != expected_params_digest:
        raise ValueError(
            f"Cached model {model_name} estimator hyperparameter metadata mismatch; "
            "remove/retrain the stale cache."
        )
    expected_names = [str(name) for name in feature_names]
    if not expected_names or len(set(expected_names)) != len(expected_names):
        raise ValueError(f"Expected model feature schema is invalid: {expected_names!r}")
    actual_names = getattr(model, "_frontier_feature_names", None)
    if not isinstance(actual_names, (list, tuple)) or list(actual_names) != expected_names:
        raise ValueError(
            f"Cached model {model_name} feature schema mismatch: "
            f"expected={expected_names!r}, actual={actual_names!r}. Retrain it."
        )
    actual_target = getattr(model, "_frontier_target_col", None)
    if actual_target != str(target_col):
        raise ValueError(
            f"Cached model {model_name} target column mismatch: "
            f"expected={target_col!r}, actual={actual_target!r}. Retrain it."
        )
    domain = getattr(model, "_frontier_feature_domain", None)
    if not isinstance(domain, Mapping):
        raise ValueError(
            f"Cached model {model_name} has malformed feature-domain metadata; retrain it."
        )
    _validate_feature_domain_metadata(
        domain,
        expected_names,
        model_name=model_name,
        require_runtime_policy_fields=True,
    )
    domain_version = domain.get("contract_version")
    if (
        isinstance(domain_version, bool)
        or not isinstance(domain_version, int)
        or domain_version != PREDICTION_CACHE_CONTRACT_VERSION
    ):
        raise ValueError(
            f"Cached model {model_name} feature domain has unsupported contract version "
            f"{domain_version!r}; expected {PREDICTION_CACHE_CONTRACT_VERSION}. Retrain it."
        )
    domain_names = domain.get("feature_names")
    if not isinstance(domain_names, (list, tuple)) or list(domain_names) != expected_names:
        raise ValueError(
            f"Cached model {model_name} feature-domain schema mismatch: "
            f"expected={expected_names!r}, actual={domain_names!r}. Retrain it."
        )
    return model
