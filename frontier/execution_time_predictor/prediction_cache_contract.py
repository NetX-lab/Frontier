"""Contracts for finite prediction grids and persisted lookup caches.

The simulator has two intentionally different prediction modes:

* finite lookup models, whose cache is materialized from a concrete feature
  grid; and
* explicit on-demand models, whose runtime feature domain is part of the model
  contract.

This module contains pure validation, identity, and provenance helpers shared
by the standalone predictor and the shared model manager.  A lookup miss is
allowed only through an explicit model-bound prediction policy; it is never a
nearest-row, clamp, or silent-default fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# Version 3 adds the explicit runtime prediction policy, sparse-regression
# domains, real-valued axis semantics, and physical non-negative constraints.
# Version 1/2 artifacts are intentionally stale because they cannot distinguish
# measured coverage from the legal model-prediction domain.
PREDICTION_CACHE_CONTRACT_VERSION = 3

PREDICTION_DOMAIN_POLICY_MEASURED_ONLY = "measured_only"
PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION = "allow_model_prediction"
PREDICTION_DOMAIN_POLICIES = frozenset(
    {
        PREDICTION_DOMAIN_POLICY_MEASURED_ONLY,
        PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION,
    }
)

# Explicit policy for models whose runtime feature vector is not materialized
# into a finite lookup dictionary.  ``bounded`` is the safe default used by
# profile-backed estimators; callers may opt into extrapolation only by
# persisting ``explicit_unbounded`` in the model domain metadata.
ON_DEMAND_DOMAIN_POLICY_BOUNDED = "bounded"
ON_DEMAND_DOMAIN_POLICY_UNBOUNDED = "explicit_unbounded"
ON_DEMAND_DOMAIN_POLICIES = frozenset(
    {ON_DEMAND_DOMAIN_POLICY_BOUNDED, ON_DEMAND_DOMAIN_POLICY_UNBOUNDED}
)

# These operators intentionally use runtime feature vectors that are not
# materialized into a finite lookup grid.  Keep this list narrow: ordinary
# profile-backed lookup models must remain bounded.
EXPLICIT_UNBOUNDED_OPERATOR_NAMES = frozenset(
    {
        "attn_prefill_mixed",
        "attn_decode_in_mixed",
        "moe_shuffling",
        "moe_grouped_gemm",
    }
)

# ``domain_kind`` is persisted with every trained finite predictor.  A sparse
# regression domain explicitly permits the canonical estimator to predict
# legal tuples absent from measured rows; it does not mean nearest-row lookup.
DOMAIN_KIND_INTEGER_INTERVAL = "integer_interval_interpolation"
DOMAIN_KIND_VERIFIED_CARTESIAN = "verified_cartesian_interpolation"
DOMAIN_KIND_CONDITIONAL = "conditional_interpolation"
DOMAIN_KIND_REGRESSION = "regression_extrapolation"
DOMAIN_KIND_EXACT_ROWS = "exact_rows"
DOMAIN_KINDS = frozenset(
    {
        DOMAIN_KIND_INTEGER_INTERVAL,
        DOMAIN_KIND_VERIFIED_CARTESIAN,
        DOMAIN_KIND_CONDITIONAL,
        DOMAIN_KIND_REGRESSION,
        DOMAIN_KIND_EXACT_ROWS,
    }
)

AXIS_SEMANTIC_INTEGER_INTERVAL = "integer_interval"
AXIS_SEMANTIC_ENUMERATED = "enumerated"
AXIS_SEMANTIC_REAL_INTERVAL = "real_interval"
AXIS_SEMANTICS = frozenset(
    {
        AXIS_SEMANTIC_INTEGER_INTERVAL,
        AXIS_SEMANTIC_ENUMERATED,
        AXIS_SEMANTIC_REAL_INTERVAL,
    }
)

PREDICTION_CLASS_DIRECT_MEASURED = "direct_measured"
PREDICTION_CLASS_INTERPOLATION = "interpolation"
PREDICTION_CLASS_EXTRAPOLATION = "extrapolation"
PREDICTION_CLASS_SPARSE_GAP = "sparse_gap"

_REAL_INTERVAL_FEATURE_HINTS = frozenset(
    {
        "ratio",
        "cv",
        "entropy",
        "gini",
        "utilization",
        "variance",
        "avg",
        "mean",
        "std",
        "fraction",
        "interaction",
    }
)

# A complete measured Cartesian product proves that independent-axis
# interpolation is structurally meaningful, but it does not say which axes are
# numeric intervals and which are categories.  Keep that policy operator-aware
# instead of treating every integer-encoded field as continuous.
_OPERATOR_CARTESIAN_AXIS_SEMANTICS: dict[str, dict[str, str]] = {
    "attn_decode": {
        "batch_size": AXIS_SEMANTIC_INTEGER_INTERVAL,
        "kv_cache_size": AXIS_SEMANTIC_INTEGER_INTERVAL,
    },
    "attn_kv_cache_save": {
        "total_tokens": AXIS_SEMANTIC_INTEGER_INTERVAL,
        "kv_cache_size": AXIS_SEMANTIC_INTEGER_INTERVAL,
        "batch_size": AXIS_SEMANTIC_INTEGER_INTERVAL,
    },
}

# Imported MLA profiles contain both model-identity features and runtime-shape
# features in the same Cartesian row.  Identity axes (for example head
# dimensions and TP) are categorical: changing them must select a different
# trained model rather than interpolate across architectures.  Runtime shape
# axes are integer intervals and may be interpolated/extrapolated by the
# canonical estimator when the descriptor policy permits it.
_MLA_CARTESIAN_INTEGER_AXES = frozenset(
    {
        "batch_size",
        "batch_num_tokens",
        "batch_num_prefill_tokens",
        "batch_num_decode_tokens",
        "max_seqlen_q",
        "max_seqlen_k",
        "num_actual_tokens",
        "max_seq_len",
    }
)
_MLA_CARTESIAN_ENUMERATED_AXES = frozenset(
    {
        "n_q_head",
        "n_kv_head",
        "head_size",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "qk_head_dim",
        "kv_lora_rank",
        "v_head_dim",
        "block_size",
        "num_tensor_parallel_workers",
        "is_prefill",
    }
)

_CACHE_WRITE_FEATURE_NAMES = (
    "total_tokens",
    "kv_cache_size",
    "batch_size",
)


def _cache_write_domain_constraints() -> list[dict[str, Any]]:
    """Return the measured dense cache-write relational contract."""

    # Every runtime batch contributes at least one token per request.  This is
    # a physical invariant, not an axis-bound guess, and prevents a model from
    # being used for impossible tuples such as total_tokens=2,batch_size=3.
    return [
        {
            "type": "linear_lte",
            "terms": {
                "batch_size": 1,
                "total_tokens": -1,
            },
            "max": 0,
        }
    ]


@dataclass(frozen=True)
class CanonicalPredictionGrid:
    """Immutable canonical metadata for one requested prediction grid."""

    feature_names: tuple[str, ...]
    ordered_keys: tuple[tuple[int | float, ...], ...]
    keys: tuple[tuple[int | float, ...], ...]
    digest: str


def validate_on_demand_domain_policy(
    policy: Any,
    *,
    model_name: str = "model",
) -> str:
    """Validate the explicit policy attached to an on-demand model."""

    if not isinstance(policy, str) or policy not in ON_DEMAND_DOMAIN_POLICIES:
        raise ValueError(
            f"{model_name} on-demand model requires explicit domain policy; "
            f"got {policy!r}, expected one of {sorted(ON_DEMAND_DOMAIN_POLICIES)!r}."
        )
    return policy


def validate_prediction_domain_policy(
    policy: Any,
    *,
    model_name: str = "model",
) -> str:
    """Validate the policy separating measured coverage from legal prediction."""

    if not isinstance(policy, str) or policy not in PREDICTION_DOMAIN_POLICIES:
        raise ValueError(
            f"{model_name} prediction domain requires explicit runtime policy; "
            f"got {policy!r}, expected one of {sorted(PREDICTION_DOMAIN_POLICIES)!r}."
        )
    return policy


def resolve_feature_domain_policy(
    operator_name: str | None,
    feature_names: Sequence[str] | None = None,
    on_demand_policy: str | None = None,
) -> str:
    """Resolve the persisted policy for one operator.

    Dynamic operators must opt into ``explicit_unbounded`` by operator name.
    All other operators default to the bounded profile domain.  An explicit
    bounded policy is rejected for a dynamic operator so a producer cannot
    accidentally create a runtime record that advertises on-demand behavior
    while still enforcing a finite sparse training box.
    """

    normalized_operator = (
        None
        if operator_name is None
        else str(operator_name).strip()
    )
    if normalized_operator == "":
        normalized_operator = None

    feature_count = None if feature_names is None else len(tuple(feature_names))
    requires_unbounded = (
        normalized_operator in {"attn_prefill_mixed", "attn_decode_in_mixed"}
        or (
            normalized_operator in {"moe_shuffling", "moe_grouped_gemm"}
            and feature_count is not None
            and feature_count > 1
        )
    )
    expected_policy = (
        ON_DEMAND_DOMAIN_POLICY_UNBOUNDED
        if requires_unbounded
        else ON_DEMAND_DOMAIN_POLICY_BOUNDED
    )
    if on_demand_policy is None:
        return expected_policy

    policy = validate_on_demand_domain_policy(
        on_demand_policy,
        model_name=normalized_operator or "model",
    )
    if (
        requires_unbounded
        and policy != ON_DEMAND_DOMAIN_POLICY_UNBOUNDED
    ):
        raise ValueError(
            f"{normalized_operator} requires "
            f"{ON_DEMAND_DOMAIN_POLICY_UNBOUNDED!r} domain policy; got {policy!r}."
        )
    return policy


def _canonical_scalar(value: Any) -> int | float:
    """Return a stable numeric representation for a prediction-key value."""
    if isinstance(value, np.generic):
        value = value.item()
    # Imported MLA profiles contain boolean shape flags (for example
    # ``is_decode``).  Treat them as their stable numeric representation for
    # cache/domain identity; TP validators remain strict about boolean TP
    # values separately.
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, Real):
        raise ValueError(
            "Prediction-grid features must be finite numeric scalars; "
            f"got {value!r} ({type(value).__name__})."
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Prediction-grid features must be finite; got {value!r}.")
    if numeric.is_integer():
        return int(numeric)
    return numeric


def canonicalize_prediction_key(key: Sequence[Any]) -> tuple[int | float, ...]:
    """Canonicalize one finite prediction key for comparison and hashing."""
    if isinstance(key, (str, bytes)):
        raise ValueError(f"Prediction key must be a sequence, got {key!r}.")
    return tuple(_canonical_scalar(value) for value in key)


def canonicalize_prediction_grid(
    feature_names: Sequence[str],
    keys: Iterable[Sequence[Any]],
    *,
    return_metadata: bool = False,
) -> (
    tuple[tuple[str, ...], tuple[tuple[int | float, ...], ...]]
    | CanonicalPredictionGrid
):
    """Return ordered feature names and a sorted, de-duplicated key set."""
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Prediction-grid feature names must be unique and non-empty: {names!r}")
    ordered_keys = tuple(
        canonicalize_prediction_key(key)
        for key in keys
    )
    wrong_lengths = sorted({len(key) for key in ordered_keys if len(key) != len(names)})
    if wrong_lengths:
        raise ValueError(
            "Prediction-grid key length does not match feature schema: "
            f"features={names!r}, key_lengths={wrong_lengths!r}"
        )
    canonical_keys = tuple(sorted(set(ordered_keys), key=repr))
    if return_metadata:
        return CanonicalPredictionGrid(
            feature_names=names,
            ordered_keys=ordered_keys,
            keys=canonical_keys,
            digest=_prediction_grid_digest_from_canonical(names, canonical_keys),
        )
    return names, canonical_keys


def _prediction_grid_digest_from_canonical(
    feature_names: Sequence[str],
    keys: Sequence[Sequence[int | float]],
) -> str:
    payload = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "feature_names": tuple(feature_names),
        "keys": tuple(tuple(key) for key in keys),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def prediction_grid_digest(
    feature_names: Sequence[str],
    keys: Iterable[Sequence[Any]],
) -> str:
    """Hash the complete requested feature schema and key set."""
    grid = canonicalize_prediction_grid(
        feature_names,
        keys,
        return_metadata=True,
    )
    if not isinstance(grid, CanonicalPredictionGrid):
        raise AssertionError("Canonical prediction-grid metadata was not returned.")
    return grid.digest


def prediction_grid_from_dataframe(
    X: pd.DataFrame,
    *,
    return_metadata: bool = False,
) -> (
    tuple[tuple[str, ...], tuple[tuple[int | float, ...], ...]]
    | CanonicalPredictionGrid
):
    """Extract a canonical grid from a model-input dataframe."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"Prediction grid must be a pandas DataFrame, got {type(X).__name__}.")
    return canonicalize_prediction_grid(
        list(X.columns),
        X.itertuples(index=False, name=None),
        return_metadata=return_metadata,
    )


def _validate_contract_version(domain: Any, model_name: str) -> int:
    """Validate the persisted domain descriptor version before reading fields."""
    if not isinstance(domain, Mapping):
        raise ValueError(
            f"{model_name} prediction profile domain contract must be a mapping."
        )
    version = domain.get("contract_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(
            f"{model_name} prediction profile domain contract version is missing or invalid; "
            f"expected integer {PREDICTION_CACHE_CONTRACT_VERSION}."
        )
    if version != PREDICTION_CACHE_CONTRACT_VERSION:
        raise ValueError(
            f"{model_name} prediction profile domain contract version {version} is unsupported; "
            f"expected {PREDICTION_CACHE_CONTRACT_VERSION}."
        )
    return version


def _validate_feature_schema(
    domain: Mapping[str, Any],
    model_name: str,
    actual_names: Sequence[str],
) -> tuple[str, ...]:
    names_raw = domain.get("feature_names")
    if not isinstance(names_raw, (list, tuple)):
        raise ValueError(
            f"{model_name} prediction profile domain has invalid feature_names; "
            "expected a non-empty ordered list."
        )
    names = tuple(str(name) for name in names_raw)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError(
            f"{model_name} prediction profile domain has invalid feature_names: {names!r}."
        )
    actual = tuple(str(name) for name in actual_names)
    if actual != names:
        raise ValueError(
            f"{model_name} prediction profile domain schema mismatch: "
            f"expected={names!r}, requested={actual!r}"
        )
    return names


def _validate_bounds(
    domain: Mapping[str, Any],
    model_name: str,
    feature_names: Sequence[str],
) -> dict[str, tuple[float, float]]:
    bounds_raw = domain.get("bounds")
    if not isinstance(bounds_raw, Mapping):
        raise ValueError(
            f"{model_name} prediction profile domain has no valid bounds mapping."
        )
    if set(bounds_raw) != set(feature_names):
        missing = sorted(set(feature_names) - set(bounds_raw))
        extra = sorted(set(bounds_raw) - set(feature_names))
        raise ValueError(
            f"{model_name} prediction profile domain bounds/schema mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )
    bounds: dict[str, tuple[float, float]] = {}
    for name in feature_names:
        bound = bounds_raw[name]
        if not isinstance(bound, Mapping) or set(bound) != {"min", "max"}:
            raise ValueError(
                f"{model_name} prediction profile domain has invalid bounds for {name!r}."
            )
        try:
            lower = float(bound["min"])
            upper = float(bound["max"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_name} prediction profile domain has non-numeric bounds for {name!r}."
            ) from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(
                f"{model_name} prediction profile domain has invalid bounds for {name!r}: "
                f"min={lower!r}, max={upper!r}."
            )
        bounds[name] = (lower, upper)
    return bounds


def _validate_axis_values(
    domain: Mapping[str, Any],
    model_name: str,
    feature_names: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[int | float, ...]]:
    """Validate the finite axes used by a Cartesian profile domain."""
    raw_axes = domain.get("axis_values")
    if not isinstance(raw_axes, Mapping) or set(raw_axes) != set(feature_names):
        raise ValueError(
            f"{model_name} verified Cartesian domain requires axis_values for "
            "exactly every feature."
        )

    axes: dict[str, tuple[int | float, ...]] = {}
    for name in feature_names:
        raw_values = raw_axes[name]
        if not isinstance(raw_values, (list, tuple)) or not raw_values:
            raise ValueError(
                f"{model_name} verified Cartesian domain axis_values for {name!r} "
                "must be a non-empty list."
            )
        try:
            values = tuple(canonicalize_prediction_key((value,))[0] for value in raw_values)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_name} verified Cartesian domain axis_values for {name!r} "
                f"are invalid: {exc}"
            ) from exc
        if len(set(values)) != len(values):
            raise ValueError(
                f"{model_name} verified Cartesian domain axis_values for {name!r} "
                "contain duplicate values."
            )
        lower, upper = bounds[name]
        outside = [value for value in values if value < lower or value > upper]
        if outside:
            raise ValueError(
                f"{model_name} verified Cartesian domain axis_values for {name!r} "
                f"fall outside bounds: {outside[:3]!r}."
            )
        axes[name] = tuple(sorted(values, key=repr))
    return axes


def _validate_axis_semantics(
    domain: Mapping[str, Any],
    model_name: str,
    feature_names: Sequence[str],
    axes: Mapping[str, tuple[int | float, ...]],
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, str]:
    """Validate how each verified Cartesian axis may be materialized."""

    raw_semantics = domain.get("axis_semantics")
    if not isinstance(raw_semantics, Mapping) or set(raw_semantics) != set(
        feature_names
    ):
        raise ValueError(
            f"{model_name} verified Cartesian domain requires axis_semantics "
            "for exactly every feature."
        )

    semantics: dict[str, str] = {}
    for name in feature_names:
        semantic = raw_semantics[name]
        if not isinstance(semantic, str) or semantic not in AXIS_SEMANTICS:
            raise ValueError(
                f"{model_name} verified Cartesian domain has invalid "
                f"axis_semantics for {name!r}: {semantic!r}; expected one of "
                f"{sorted(AXIS_SEMANTICS)!r}."
            )
        if semantic == AXIS_SEMANTIC_INTEGER_INTERVAL:
            lower, upper = bounds[name]
            if not float(lower).is_integer() or not float(upper).is_integer():
                raise ValueError(
                    f"{model_name} integer-interval axis {name!r} has non-integer "
                    f"bounds: min={lower!r}, max={upper!r}."
                )
            non_integer_values = [
                value
                for value in axes[name]
                if not float(value).is_integer()
            ]
            if non_integer_values:
                raise ValueError(
                    f"{model_name} integer-interval axis {name!r} contains "
                    f"non-integer measured values: {non_integer_values[:3]!r}."
                )
        semantics[name] = semantic
    return semantics


def _resolve_cartesian_axis_semantics(
    training_frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    operator_name: str | None,
    axis_semantics: Mapping[str, str] | None,
) -> dict[str, str]:
    """Resolve explicit per-axis semantics without guessing categories."""

    names = tuple(feature_names)
    if axis_semantics is not None:
        if not isinstance(axis_semantics, Mapping) or set(axis_semantics) != set(
            names
        ):
            raise ValueError(
                "Verified Cartesian axis_semantics must declare exactly every "
                f"feature: expected={names!r}, actual={tuple(axis_semantics)!r}."
            )
        resolved = {name: str(axis_semantics[name]) for name in names}
    else:
        normalized_operator = (
            str(operator_name).strip() if operator_name is not None else ""
        )
        operator_policy = dict(
            _OPERATOR_CARTESIAN_AXIS_SEMANTICS.get(normalized_operator, {})
        )
        # Some in-process producers (and legacy test/import paths) do not
        # carry ``operator_name`` into the descriptor.  The distinctive MLA
        # schema is still sufficient to recover the same safe semantics; an
        # explicit ``axis_semantics`` mapping always takes precedence above.
        has_mla_schema = {
            "qk_nope_head_dim",
            "kv_lora_rank",
            "max_seqlen_k",
            "max_seq_len",
        }.issubset(names)
        if normalized_operator.startswith("attn_mla_") or has_mla_schema:
            # MLA imported rows use one shared feature schema.  Keep static
            # architecture/identity axes enumerated while allowing legal
            # runtime shape axes such as max_seqlen_k=66 to reach the model.
            operator_policy.update(
                {
                    name: AXIS_SEMANTIC_INTEGER_INTERVAL
                    for name in names
                    if name in _MLA_CARTESIAN_INTEGER_AXES
                }
            )
            operator_policy.update(
                {
                    name: AXIS_SEMANTIC_ENUMERATED
                    for name in names
                    if name in _MLA_CARTESIAN_ENUMERATED_AXES
                }
            )
        resolved = {
            name: operator_policy.get(name, AXIS_SEMANTIC_ENUMERATED)
            for name in names
        }

    for name, semantic in resolved.items():
        if semantic not in AXIS_SEMANTICS:
            raise ValueError(
                f"Invalid verified Cartesian axis semantic for {name!r}: "
                f"{semantic!r}; expected one of {sorted(AXIS_SEMANTICS)!r}."
            )
        if semantic == AXIS_SEMANTIC_INTEGER_INTERVAL:
            if pd.api.types.is_bool_dtype(training_frame[name].dtype):
                raise ValueError(
                    f"Boolean feature {name!r} must use enumerated axis semantics."
                )
            numeric = pd.to_numeric(training_frame[name], errors="raise").astype(float)
            if not np.equal(numeric.to_numpy(), np.floor(numeric.to_numpy())).all():
                raise ValueError(
                    f"Integer-interval axis {name!r} contains non-integer values."
                )
    return resolved


def _resolve_regression_axis_semantics(
    training_frame: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    operator_name: str | None,
    axis_semantics: Mapping[str, str] | None,
) -> dict[str, str]:
    """Resolve axis semantics for a sparse regression surface.

    Explicit semantics take precedence.  Numeric integer columns are safe to
    extrapolate only at integer coordinates; floating-point columns use a real
    interval.  Boolean columns remain enumerated and therefore cannot be
    interpolated into a third state.
    """

    names = tuple(feature_names)
    if axis_semantics is not None:
        if not isinstance(axis_semantics, Mapping) or set(axis_semantics) != set(names):
            raise ValueError(
                "Regression axis_semantics must declare exactly every feature: "
                f"expected={names!r}, actual={tuple(axis_semantics)!r}."
            )
        resolved = {name: str(axis_semantics[name]) for name in names}
    else:
        operator_policy = _OPERATOR_CARTESIAN_AXIS_SEMANTICS.get(
            str(operator_name).strip() if operator_name is not None else "",
            {},
        )
        resolved = {}
        for name in names:
            if name in operator_policy:
                resolved[name] = operator_policy[name]
                continue
            if pd.api.types.is_bool_dtype(training_frame[name].dtype):
                resolved[name] = AXIS_SEMANTIC_ENUMERATED
                continue
            normalized_name = name.lower()
            if any(hint in normalized_name for hint in _REAL_INTERVAL_FEATURE_HINTS):
                resolved[name] = AXIS_SEMANTIC_REAL_INTERVAL
                continue
            numeric = pd.to_numeric(training_frame[name], errors="raise").astype(float)
            if np.equal(numeric.to_numpy(), np.floor(numeric.to_numpy())).all():
                resolved[name] = AXIS_SEMANTIC_INTEGER_INTERVAL
            else:
                resolved[name] = AXIS_SEMANTIC_REAL_INTERVAL

    for name, semantic in resolved.items():
        if semantic not in AXIS_SEMANTICS:
            raise ValueError(
                f"Invalid regression axis semantic for {name!r}: {semantic!r}; "
                f"expected one of {sorted(AXIS_SEMANTICS)!r}."
            )
        if semantic == AXIS_SEMANTIC_INTEGER_INTERVAL:
            if pd.api.types.is_bool_dtype(training_frame[name].dtype):
                raise ValueError(
                    f"Boolean feature {name!r} must use enumerated axis semantics."
                )
            numeric = pd.to_numeric(training_frame[name], errors="raise").astype(float)
            if not np.equal(numeric.to_numpy(), np.floor(numeric.to_numpy())).all():
                raise ValueError(
                    f"Integer-interval axis {name!r} contains non-integer values."
                )
    return resolved


def _infer_nonnegative_features(
    training_frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> list[str]:
    """Persist physical non-negative constraints for non-negative training axes."""

    nonnegative: list[str] = []
    for name in feature_names:
        numeric = pd.to_numeric(training_frame[name], errors="raise").astype(float)
        if float(numeric.min()) >= 0.0:
            nonnegative.append(str(name))
    return nonnegative


def _validate_nonnegative_features(
    domain: Mapping[str, Any],
    model_name: str,
    feature_names: Sequence[str],
    keys: Sequence[Sequence[int | float]],
) -> list[str]:
    raw_features = domain.get("nonnegative_features", ())
    if not isinstance(raw_features, (list, tuple)):
        raise ValueError(
            f"{model_name} prediction domain has invalid nonnegative_features metadata."
        )
    unknown = set(raw_features) - set(feature_names)
    if unknown:
        raise ValueError(
            f"{model_name} prediction domain nonnegative_features reference unknown "
            f"features: {sorted(unknown)!r}."
        )
    violations: list[str] = []
    for key in keys:
        row = dict(zip(feature_names, key))
        for name in raw_features:
            if float(row[name]) < 0.0:
                violations.append(
                    f"{name}: requested={row[name]!r} must be non-negative"
                )
    return violations


def _validate_physical_bounds_descriptor(
    domain: Mapping[str, Any],
    model_name: str,
    feature_names: Sequence[str],
) -> dict[str, tuple[float | None, float | None]]:
    raw_bounds = domain.get("physical_bounds", {})
    if raw_bounds is None:
        raw_bounds = {}
    if not isinstance(raw_bounds, Mapping):
        raise ValueError(
            f"{model_name} prediction domain has invalid physical_bounds metadata."
        )
    unknown = set(raw_bounds) - set(feature_names)
    if unknown:
        raise ValueError(
            f"{model_name} physical_bounds reference unknown features: "
            f"{sorted(unknown)!r}."
        )
    result: dict[str, tuple[float | None, float | None]] = {}
    for name, raw in raw_bounds.items():
        if not isinstance(raw, Mapping) or not set(raw).issubset({"min", "max"}):
            raise ValueError(
                f"{model_name} physical_bounds for {name!r} must contain only min/max."
            )
        lower = raw.get("min")
        upper = raw.get("max")
        parsed: list[float | None] = []
        for label, value in (("min", lower), ("max", upper)):
            if value is None:
                parsed.append(None)
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"{model_name} physical_bounds {name!r}.{label} is invalid: {value!r}."
                )
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"{model_name} physical_bounds {name!r}.{label} is non-finite."
                )
            parsed.append(numeric)
        if parsed[0] is not None and parsed[1] is not None and parsed[0] > parsed[1]:
            raise ValueError(
                f"{model_name} physical_bounds {name!r} has min > max."
            )
        result[name] = (parsed[0], parsed[1])
    return result


def _physical_key_violations(
    feature_names: Sequence[str],
    physical_bounds: Mapping[str, tuple[float | None, float | None]],
    keys: Sequence[Sequence[int | float]],
) -> list[str]:
    violations: list[str] = []
    for key in keys:
        for index, name in enumerate(feature_names):
            lower, upper = physical_bounds.get(name, (None, None))
            value = float(key[index])
            if lower is not None and value < lower:
                violations.append(
                    f"{name}: requested={value} below physical minimum {lower}"
                )
            if upper is not None and value > upper:
                violations.append(
                    f"{name}: requested={value} above physical maximum {upper}"
                )
    return violations


def _validate_conditional_constraints(
    domain: Mapping[str, Any],
    model_name: str,
    feature_names: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
) -> list[Mapping[str, Any]]:
    """Validate conditional-domain constraints without evaluating a key."""
    raw_constraints = domain.get("constraints")
    if not isinstance(raw_constraints, (list, tuple)) or not raw_constraints:
        raise ValueError(
            f"{model_name} conditional domain is missing constraints; retrain the model."
        )

    names = set(feature_names)
    constraints: list[Mapping[str, Any]] = []
    for constraint in raw_constraints:
        if not isinstance(constraint, Mapping):
            raise ValueError(
                f"{model_name} conditional domain contains an invalid constraint."
            )
        ctype = constraint.get("type")
        if ctype not in {"sum_lte", "linear_lte"}:
            raise ValueError(
                f"{model_name} conditional domain has unsupported constraint "
                f"type={ctype!r}."
            )
        if ctype == "linear_lte":
            terms = constraint.get("terms")
            if not isinstance(terms, Mapping) or not terms:
                raise ValueError(
                    f"{model_name} linear_lte constraint requires non-empty terms."
                )
            if set(terms) - names:
                raise ValueError(
                    f"{model_name} linear_lte constraint references unknown features: "
                    f"{sorted(set(terms) - names)!r}."
                )
            for feature, coefficient in terms.items():
                if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
                    raise ValueError(
                        f"{model_name} linear_lte coefficient for {feature!r} "
                        f"is invalid: {coefficient!r}."
                    )
                if not math.isfinite(float(coefficient)):
                    raise ValueError(
                        f"{model_name} linear_lte coefficient for {feature!r} "
                        f"is non-finite: {coefficient!r}."
                    )
            features = tuple(terms)
            derived = {}
        else:
            features = constraint.get("features")
            derived = constraint.get("derived_features", {})
        if (
            not isinstance(features, (list, tuple))
            or not features
            or len(set(features)) != len(features)
        ):
            raise ValueError(
                f"{model_name} conditional constraint has invalid features={features!r}."
            )
        if not isinstance(derived, Mapping):
            raise ValueError(
                f"{model_name} conditional constraint has invalid derived_features."
            )
        unknown_derived = set(derived) - set(features)
        if unknown_derived:
            raise ValueError(
                f"{model_name} conditional constraint has derived features not present "
                f"in features: {sorted(unknown_derived)!r}."
            )
        for feature in features:
            if feature in derived:
                specification = derived[feature]
                if not isinstance(specification, Mapping) or set(specification) != {"sqrt"}:
                    raise ValueError(
                        f"{model_name} conditional derived feature {feature!r} is invalid."
                    )
                source = specification["sqrt"]
                if source not in names:
                    raise ValueError(
                        f"{model_name} conditional constraint references unknown feature "
                        f"{source!r}."
                    )
                if bounds[source][0] < 0.0:
                    raise ValueError(
                        f"{model_name} conditional sqrt source {source!r} has a "
                        "negative lower bound."
                    )
            elif feature not in names:
                raise ValueError(
                    f"{model_name} conditional constraint references unknown feature "
                    f"{feature!r}."
                )
        limit = constraint.get("max")
        if isinstance(limit, bool) or not isinstance(limit, Real):
            raise ValueError(
                f"{model_name} conditional constraint has invalid max={limit!r}."
            )
        if not math.isfinite(float(limit)):
            raise ValueError(
                f"{model_name} conditional constraint has non-finite max={limit!r}."
            )
        constraints.append(constraint)
    return constraints


def validate_feature_domain_descriptor(
    domain: Mapping[str, Any],
    feature_names: Sequence[str],
    *,
    model_name: str = "model",
    operator_name: str | None = None,
) -> tuple[tuple[str, ...], dict[str, tuple[float, float]], str]:
    """Validate the complete, kind-specific shape of a persisted domain.

    Structural validation belongs at the model-cache boundary as well as at
    runtime grid materialization.  Keeping it here prevents malformed domain
    descriptors from being accepted by one path and rejected by another.
    """
    _validate_contract_version(domain, model_name)
    names = _validate_feature_schema(domain, model_name, feature_names)
    descriptor_operator = domain.get("operator_name")
    if descriptor_operator is not None:
        if not isinstance(descriptor_operator, str) or not descriptor_operator.strip():
            raise ValueError(
                f"{model_name} prediction profile domain has invalid operator_name."
            )
        if (
            operator_name is not None
            and descriptor_operator != str(operator_name)
        ):
            raise ValueError(
                f"{model_name} prediction profile domain operator mismatch: "
                f"expected={operator_name!r}, actual={descriptor_operator!r}."
            )
    bounds = _validate_bounds(domain, model_name, names)
    _validate_physical_bounds_descriptor(domain, model_name, names)
    kind = domain.get("domain_kind")
    runtime_policy = domain.get("runtime_prediction_policy")
    if runtime_policy is None:
        # Test/draft descriptors from the pre-policy contract may still be
        # loaded in memory.  Exact rows remain strict; all other descriptors
        # use the new model-prediction policy.  Persisted old artifacts are
        # rejected by the contract-version check above.
        runtime_policy = (
            PREDICTION_DOMAIN_POLICY_MEASURED_ONLY
            if kind == DOMAIN_KIND_EXACT_ROWS
            else PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION
        )
    validate_prediction_domain_policy(runtime_policy, model_name=model_name)
    if "on_demand_policy" in domain:
        validate_on_demand_domain_policy(
            domain.get("on_demand_policy"),
            model_name=model_name,
        )
    if kind == DOMAIN_KIND_EXACT_ROWS and runtime_policy != PREDICTION_DOMAIN_POLICY_MEASURED_ONLY:
        raise ValueError(
            f"{model_name} exact-row domain must use runtime_prediction_policy="
            f"{PREDICTION_DOMAIN_POLICY_MEASURED_ONLY!r}."
        )
    if kind == DOMAIN_KIND_INTEGER_INTERVAL:
        if len(names) != 1:
            raise ValueError(
                f"{model_name} integer interval domain must have exactly one feature."
            )
        interval_feature = names[0]
        raw_axis_values = domain.get("axis_values", {}).get(interval_feature, ())
        if not isinstance(raw_axis_values, (list, tuple)):
            raw_axis_values = ()
        fractional_values = [
            value for value in raw_axis_values if not float(value).is_integer()
        ]
        if fractional_values:
            raise ValueError(
                f"{model_name} integer interval domain requires integer values "
                f"for {interval_feature!r}; got {fractional_values[:3]!r}."
            )
    elif kind == DOMAIN_KIND_EXACT_ROWS:
        raw_keys = domain.get("training_keys")
        if not isinstance(raw_keys, (list, tuple)) or not raw_keys:
            raise ValueError(
                f"{model_name} exact-row domain requires non-empty training_keys."
            )
        try:
            _, keys = canonicalize_prediction_grid(names, raw_keys)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{model_name} exact-row domain has invalid training_keys: {exc}"
            ) from exc
        if len(keys) != len(raw_keys):
            raise ValueError(
                f"{model_name} exact-row domain training_keys contain duplicates."
            )
        digest = domain.get("training_key_digest")
        if not isinstance(digest, str) or not digest:
            raise ValueError(
                f"{model_name} exact-row domain requires training_key_digest."
            )
        expected_digest = prediction_grid_digest(names, keys)
        if digest != expected_digest:
            raise ValueError(
                f"{model_name} exact-row domain training_key_digest mismatch: "
                f"expected={expected_digest!r}, actual={digest!r}."
            )
    elif kind == DOMAIN_KIND_REGRESSION:
        if len(names) < 2:
            raise ValueError(
                f"{model_name} regression domain requires at least two features."
            )
        axes = _validate_axis_values(domain, model_name, names, bounds)
        _validate_axis_semantics(domain, model_name, names, axes, bounds)
        raw_keys = domain.get("training_keys")
        if not isinstance(raw_keys, (list, tuple)) or not raw_keys:
            raise ValueError(
                f"{model_name} regression domain requires non-empty training_keys."
            )
        _, keys = canonicalize_prediction_grid(names, raw_keys)
        if len(keys) != len(raw_keys) or len(set(keys)) != len(keys):
            raise ValueError(
                f"{model_name} regression domain training_keys contain duplicates."
            )
        digest = domain.get("training_key_digest")
        if not isinstance(digest, str) or digest != prediction_grid_digest(names, keys):
            raise ValueError(
                f"{model_name} regression domain training_key_digest mismatch."
            )
        _validate_conditional_constraints(
            domain,
            model_name,
            names,
            bounds,
        ) if "constraints" in domain else None
        _validate_nonnegative_features(domain, model_name, names, ())
    elif kind == DOMAIN_KIND_CONDITIONAL:
        _validate_conditional_constraints(domain, model_name, names, bounds)
        if "axis_semantics" in domain:
            axes = _validate_axis_values(domain, model_name, names, bounds)
            _validate_axis_semantics(domain, model_name, names, axes, bounds)
    elif kind == DOMAIN_KIND_VERIFIED_CARTESIAN:
        if len(names) < 2:
            raise ValueError(
                f"{model_name} verified Cartesian domain requires at least two features."
            )
        axes = _validate_axis_values(domain, model_name, names, bounds)
        _validate_axis_semantics(domain, model_name, names, axes, bounds)
        product_keys = tuple(product(*(axes[name] for name in names)))
        product_size = domain.get("axis_product_size")
        if (
            isinstance(product_size, bool)
            or not isinstance(product_size, (int, np.integer))
            or int(product_size) != len(product_keys)
        ):
            raise ValueError(
                f"{model_name} verified Cartesian domain axis_product_size does not "
                f"match axis coverage: expected={len(product_keys)}, actual={product_size!r}."
            )
        product_digest = domain.get("axis_product_digest")
        if not isinstance(product_digest, str) or not product_digest:
            raise ValueError(
                f"{model_name} verified Cartesian domain requires axis_product_digest."
            )
        expected_digest = prediction_grid_digest(names, product_keys)
        if product_digest != expected_digest:
            raise ValueError(
                f"{model_name} verified Cartesian domain axis_product_digest mismatch: "
                f"expected={expected_digest!r}, actual={product_digest!r}."
            )
    else:
        # ``_validate_contract_version``/schema helpers should make this
        # unreachable, but keep the failure explicit for future extensions.
        raise ValueError(f"{model_name} has unsupported domain_kind={kind!r}.")
    return names, bounds, kind


def _canonical_dataframe_keys(
    X: pd.DataFrame,
    feature_names: Sequence[str],
    model_name: str,
) -> tuple[tuple[int | float, ...], ...]:
    try:
        _, keys = prediction_grid_from_dataframe(X)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model_name} prediction grid contains invalid feature values: {exc}"
        ) from exc
    if tuple(str(name) for name in X.columns) != tuple(feature_names):
        # The caller normally performs this check first.  Keep this guard here
        # as well so future callers cannot accidentally validate a reordered
        # dataframe against the wrong feature tuple.
        raise ValueError(
            f"{model_name} prediction grid feature schema does not match domain."
        )
    return keys


def _infer_domain_kind(feature_names: Sequence[str], keys: Sequence[Sequence[Any]]) -> str:
    if len(feature_names) == 1:
        return DOMAIN_KIND_INTEGER_INTERVAL
    # Prefill remains a conditional domain even when one particular profiling
    # sweep happens to cover a full Cartesian subset.  Its two features retain
    # a context-length relationship by definition.
    if tuple(feature_names) == ("kv_cache_size", "prefill_chunk_size_squared"):
        return DOMAIN_KIND_CONDITIONAL
    unique_axes = [
        {canonicalize_prediction_key((key[index],))[0] for key in keys}
        for index in range(len(feature_names))
    ]
    cartesian_size = math.prod(len(axis) for axis in unique_axes)
    if cartesian_size == len(set(tuple(key) for key in keys)):
        return DOMAIN_KIND_VERIFIED_CARTESIAN
    # Sparse measured surfaces are still valid regression training data.  They
    # must not be mistaken for a categorical exact-row contract unless the
    # producer explicitly requests ``domain_kind=exact_rows``.
    return DOMAIN_KIND_REGRESSION


def build_feature_domain_descriptor(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    operator_name: str | None = None,
    domain_kind: str | None = None,
    constraints: Sequence[Mapping[str, Any]] | None = None,
    on_demand_policy: str | None = None,
    axis_semantics: Mapping[str, str] | None = None,
    runtime_prediction_policy: str | None = None,
    physical_bounds: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a versioned, operator-aware finite domain descriptor.

    One-dimensional models retain the historical sparse-sample-to-interval
    regression behavior.  Multi-dimensional models are classified as a
    verified Cartesian product, a conditional domain (currently prefill
    attention), or a sparse regression surface.  Exact observed rows are used
    only when explicitly requested by the producer.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Training feature data must be a pandas DataFrame, got {type(df).__name__}.")
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Prediction domain feature names must be unique and non-empty: {names!r}")
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise ValueError(f"Cannot build prediction domain; missing columns: {missing!r}")
    training_frame = df.loc[:, list(names)]
    _, training_keys = prediction_grid_from_dataframe(training_frame)
    if not training_keys:
        raise ValueError("Cannot build prediction domain: no training feature rows.")
    inferred_kind = _infer_domain_kind(names, training_keys)
    explicit_kind = domain_kind is not None
    kind = inferred_kind if domain_kind is None else str(domain_kind)
    if operator_name == "attn_kv_cache_save":
        if names != _CACHE_WRITE_FEATURE_NAMES:
            raise ValueError(
                "attn_kv_cache_save requires the runtime feature schema "
                f"{list(_CACHE_WRITE_FEATURE_NAMES)!r}; got {list(names)!r}."
            )
        if domain_kind is not None and kind not in {
            DOMAIN_KIND_REGRESSION,
            DOMAIN_KIND_EXACT_ROWS,
        }:
            raise ValueError(
                "attn_kv_cache_save requires a regression or explicit exact-row "
                f"domain; got domain_kind={kind!r}."
            )
        # Cache-write is a sparse three-axis regression surface.  The physical
        # relation below remains enforced even when the measured rows are not
        # complete.
        if not explicit_kind:
            kind = DOMAIN_KIND_REGRESSION
    if kind not in DOMAIN_KINDS:
        raise ValueError(
            f"Unsupported prediction domain_kind={kind!r}; expected one of "
            f"{sorted(DOMAIN_KINDS)!r}."
        )
    policy = resolve_feature_domain_policy(
        operator_name,
        names,
        on_demand_policy,
    )
    if runtime_prediction_policy is None:
        if kind == DOMAIN_KIND_EXACT_ROWS:
            runtime_prediction_policy = PREDICTION_DOMAIN_POLICY_MEASURED_ONLY
        elif (
            on_demand_policy == ON_DEMAND_DOMAIN_POLICY_BOUNDED
            and operator_name != "attn_kv_cache_save"
        ):
            # An explicit bounded dynamic/on-demand producer is intentionally
            # measured-only.  Ordinary profile-backed operators leave
            # on_demand_policy unset and use model prediction gaps.
            runtime_prediction_policy = PREDICTION_DOMAIN_POLICY_MEASURED_ONLY
        else:
            runtime_prediction_policy = PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION
    validate_prediction_domain_policy(
        runtime_prediction_policy,
        model_name=operator_name or "model",
    )
    if kind == DOMAIN_KIND_EXACT_ROWS and runtime_prediction_policy != PREDICTION_DOMAIN_POLICY_MEASURED_ONLY:
        raise ValueError(
            "Exact-row domains require runtime_prediction_policy='measured_only'."
        )
    bounds: dict[str, dict[str, float]] = {}
    axis_values: dict[str, list[int | float]] = {}
    for name in names:
        values = pd.to_numeric(df[name], errors="raise").astype(float)
        if values.empty:
            raise ValueError(f"Cannot build prediction domain for {name!r}: no rows.")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Cannot build prediction domain for {name!r}: non-finite values.")
        if kind == DOMAIN_KIND_INTEGER_INTERVAL:
            fractional_values = [
                float(value)
                for value in values.to_numpy()
                if not float(value).is_integer()
            ]
            if fractional_values:
                raise ValueError(
                    f"Prediction domain integer interval requires integer values "
                    f"for {name!r}; got {fractional_values[:3]!r}."
                )
        canonical_values = sorted(
            {
                canonicalize_prediction_key((value,))[0]
                for value in values.to_numpy()
            },
            key=repr,
        )
        bounds[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
        }
        axis_values[name] = canonical_values

    descriptor: dict[str, Any] = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "feature_names": list(names),
        "domain_kind": kind,
        "on_demand_policy": policy,
        "runtime_prediction_policy": runtime_prediction_policy,
        "bounds": bounds,
        "axis_values": axis_values,
        "nonnegative_features": _infer_nonnegative_features(
            training_frame,
            names,
        ),
    }
    if physical_bounds is not None:
        if not isinstance(physical_bounds, Mapping):
            raise ValueError("Prediction domain physical_bounds must be a mapping.")
        descriptor["physical_bounds"] = {
            str(name): dict(value) for name, value in physical_bounds.items()
        }
    else:
        descriptor["physical_bounds"] = {
            name: {"min": 0.0}
            for name in descriptor["nonnegative_features"]
        }
    if operator_name == "attn_kv_cache_save":
        # A cache-write with no batch or no token has no kernel work and must
        # not be represented by this predictor.  Upper bounds are supplied by
        # the runtime predictor/configuration, not inferred from measured rows.
        descriptor["physical_bounds"].update(
            {
                "total_tokens": {"min": 1.0},
                "kv_cache_size": {"min": 0.0},
                "batch_size": {"min": 1.0},
            }
        )
    else:
        for name in ("num_tokens", "batch_size", "total_tokens"):
            if name in descriptor["physical_bounds"]:
                descriptor["physical_bounds"][name]["min"] = max(
                    float(descriptor["physical_bounds"][name].get("min", 0.0)),
                    1.0,
                )
    if operator_name is not None and str(operator_name).strip():
        descriptor["operator_name"] = str(operator_name).strip()
    if kind in {DOMAIN_KIND_EXACT_ROWS, DOMAIN_KIND_REGRESSION}:
        descriptor["training_keys"] = [list(key) for key in training_keys]
        descriptor["training_key_digest"] = prediction_grid_digest(names, training_keys)
        if kind == DOMAIN_KIND_REGRESSION:
            descriptor["axis_semantics"] = _resolve_regression_axis_semantics(
                training_frame,
                names,
                operator_name=operator_name,
                axis_semantics=axis_semantics,
            )
            if constraints is not None:
                descriptor["constraints"] = [dict(item) for item in constraints]
            elif operator_name == "attn_kv_cache_save":
                descriptor["constraints"] = _cache_write_domain_constraints()
    elif kind == DOMAIN_KIND_VERIFIED_CARTESIAN:
        # Persist a compact proof that the observed rows covered the complete
        # Cartesian product of the declared axes.  The validator recomputes
        # both values before accepting a persisted model artifact.
        product_keys = tuple(
            product(*(axis_values[name] for name in names))
        )
        missing_keys = sorted(
            set(product_keys) - set(training_keys),
            key=repr,
        )
        if missing_keys:
            raise ValueError(
                "Verified Cartesian prediction domains require complete Cartesian "
                f"training coverage; missing combinations={len(missing_keys)}, "
                f"examples={missing_keys[:3]!r}."
            )
        descriptor["axis_product_size"] = len(product_keys)
        descriptor["axis_product_digest"] = prediction_grid_digest(
            names, product_keys
        )
        descriptor["axis_semantics"] = _resolve_cartesian_axis_semantics(
            training_frame,
            names,
            operator_name=operator_name,
            axis_semantics=axis_semantics,
        )
    elif kind == DOMAIN_KIND_CONDITIONAL:
        # ``prefill_chunk_size_squared`` is non-negative by construction.  A
        # square root recovers the raw chunk length for the context relation.
        if tuple(names) == ("kv_cache_size", "prefill_chunk_size_squared"):
            max_context = max(
                float(key[0]) + math.sqrt(max(float(key[1]), 0.0))
                for key in training_keys
            )
            descriptor["constraints"] = [
                {
                    "type": "sum_lte",
                    "features": ["kv_cache_size", "prefill_chunk_size"],
                    "derived_features": {
                        "prefill_chunk_size": {
                            "sqrt": "prefill_chunk_size_squared"
                        }
                    },
                    "max": max_context,
                    # This bound describes observed profile coverage.  The
                    # predictor supplies the physical model-context cap at
                    # runtime; extrapolated requests must not be rejected just
                    # because the largest measured row was smaller.
                    "enforce_during_extrapolation": False,
                }
            ]
        elif constraints is None:
            raise ValueError(
                "Conditional prediction domains require explicit constraints."
            )
        descriptor["axis_semantics"] = _resolve_regression_axis_semantics(
            training_frame,
            names,
            operator_name=operator_name,
            axis_semantics=axis_semantics,
        )
    if constraints is not None:
        descriptor["constraints"] = [dict(item) for item in constraints]
    return descriptor


def attach_feature_domain(
    model: Any,
    df: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    operator_name: str | None = None,
    domain_kind: str | None = None,
    constraints: Sequence[Mapping[str, Any]] | None = None,
    on_demand_policy: str | None = None,
    axis_semantics: Mapping[str, str] | None = None,
    runtime_prediction_policy: str | None = None,
    physical_bounds: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Build and attach a versioned domain/policy contract to an estimator.

    ``allow_model_prediction`` is the default for ordinary profile-backed
    models.  An explicitly declared ``exact_rows`` producer remains measured
    only.  ``on_demand_policy`` continues to describe the separate dynamic
    model contract.
    """

    model._frontier_feature_domain = build_feature_domain_descriptor(
        df,
        feature_names,
        operator_name=operator_name,
        domain_kind=domain_kind,
        constraints=constraints,
        on_demand_policy=on_demand_policy,
        axis_semantics=axis_semantics,
        runtime_prediction_policy=runtime_prediction_policy,
        physical_bounds=physical_bounds,
    )


def build_on_demand_prediction_record(
    operator_name: str,
    model: Any,
    feature_names: Sequence[str] | None = None,
    *,
    exact_lookup: Mapping[Sequence[Any], Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one runtime record for an explicit on-demand model."""

    raw_feature_names = feature_names
    if raw_feature_names is None:
        raw_feature_names = getattr(model, "_frontier_feature_names", None)
    if raw_feature_names is None:
        raw_feature_names = getattr(model, "feature_names_in_", None)
    if not isinstance(raw_feature_names, (list, tuple, np.ndarray)):
        raise ValueError(
            f"On-demand model {operator_name} has invalid feature schema metadata."
        )
    names = tuple(str(name) for name in raw_feature_names)
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError(
            f"On-demand model {operator_name} has invalid feature schema: {names!r}."
        )

    n_features = getattr(model, "n_features_in_", len(names))
    if (
        isinstance(n_features, bool)
        or not isinstance(n_features, (int, np.integer))
        or int(n_features) != len(names)
    ):
        raise ValueError(
            f"On-demand model {operator_name} feature-count metadata mismatch: "
            f"declared={n_features!r}, schema={len(names)}."
        )

    feature_domain = getattr(model, "_frontier_feature_domain", None)
    if not isinstance(feature_domain, Mapping):
        raise ValueError(
            f"On-demand model {operator_name} has invalid/missing feature-domain metadata; "
            "retrain it with an explicit domain policy."
        )
    persisted_policy = feature_domain.get("on_demand_policy")
    if persisted_policy is None:
        raise ValueError(
            f"On-demand model {operator_name} is missing explicit "
            "on_demand_policy metadata; retrain it with the current contract."
        )
    validate_feature_domain_descriptor(
        feature_domain,
        names,
        model_name=f"On-demand model {operator_name}",
        operator_name=operator_name,
    )
    policy = resolve_feature_domain_policy(
        operator_name,
        names,
        persisted_policy,
    )

    record: dict[str, Any] = {
        "_on_demand_prediction": True,
        "_n_features": int(n_features),
        "_model": model,
        "_feature_names": list(names),
        "_feature_domain": dict(feature_domain),
        "_on_demand_domain_policy": policy,
    }
    if exact_lookup is None:
        exact_lookup = getattr(model, "_frontier_exact_lookup", None)
    if exact_lookup is not None:
        if not isinstance(exact_lookup, Mapping):
            raise ValueError(
                f"On-demand model {operator_name} has invalid exact lookup metadata."
            )
        canonical_exact_lookup: dict[tuple[int | float, ...], float] = {}
        for raw_key, raw_value in exact_lookup.items():
            try:
                canonical_key = canonicalize_prediction_key(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"On-demand model {operator_name} has invalid exact lookup key "
                    f"{raw_key!r}: {exc}"
                ) from exc
            if len(canonical_key) != len(names):
                raise ValueError(
                    f"On-demand model {operator_name} exact lookup key/schema length "
                    f"mismatch: key={canonical_key!r}, feature_names={names!r}."
                )
            try:
                exact_value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"On-demand model {operator_name} exact lookup value for "
                    f"key={canonical_key!r} is non-numeric: {raw_value!r}."
                ) from exc
            # Imported legacy artifacts can contain structural NaN targets for
            # operators that were not applicable. They are absent observations,
            # not direct measurements.
            if math.isnan(exact_value):
                continue
            if not math.isfinite(exact_value) or exact_value < 0.0:
                raise ValueError(
                    f"On-demand model {operator_name} exact lookup value for "
                    f"key={canonical_key!r} is invalid: {raw_value!r}."
                )
            previous_value = canonical_exact_lookup.get(canonical_key)
            if previous_value is not None and previous_value != exact_value:
                raise ValueError(
                    f"On-demand model {operator_name} exact lookup has conflicting "
                    f"values for canonical key={canonical_key!r}."
                )
            canonical_exact_lookup[canonical_key] = exact_value
        record["_exact_lookup"] = canonical_exact_lookup
    return record


def _validated_domain_components(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    canonical_grid: CanonicalPredictionGrid | None = None,
) -> tuple[
    Mapping[str, Any],
    tuple[str, ...],
    dict[str, tuple[float, float]],
    tuple[tuple[int | float, ...], ...],
    str,
]:
    domain = getattr(model, "_frontier_feature_domain", None)
    if domain is None:
        raise ValueError(
            f"{model_name} has no prediction profile domain contract; "
            "retrain the model with the current cache contract."
        )
    names, bounds, kind = validate_feature_domain_descriptor(
        domain,
        list(X.columns),
        model_name=model_name,
        operator_name=model_name,
    )
    if canonical_grid is None:
        keys = _canonical_dataframe_keys(X, names, model_name)
    else:
        if canonical_grid.feature_names != names:
            raise ValueError(
                f"{model_name} canonical prediction-grid schema does not match domain."
            )
        if len(canonical_grid.ordered_keys) != len(X):
            raise ValueError(
                f"{model_name} canonical prediction-grid row count does not match dataframe."
            )
        keys = canonical_grid.keys
    return domain, names, bounds, keys, kind


def _key_bound_violations(
    feature_names: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    key: Sequence[int | float],
) -> list[str]:
    violations: list[str] = []
    for index, name in enumerate(feature_names):
        lower, upper = bounds[name]
        value = float(key[index])
        if value < lower or value > upper:
            violations.append(
                f"{name}: requested={value}, profile=[{lower}, {upper}]"
            )
    return violations


def _conditional_key_violations(
    model_name: str,
    feature_names: Sequence[str],
    key: Sequence[int | float],
    constraints: Any,
    *,
    allow_extrapolation: bool = False,
) -> list[str]:
    """Return descriptor-validation messages for one conditional tuple."""
    if not isinstance(constraints, (list, tuple)) or not constraints:
        raise ValueError(
            f"{model_name} conditional domain is missing constraints; retrain the model."
        )
    row = dict(zip(feature_names, key))
    violations: list[str] = []
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            raise ValueError(
                f"{model_name} conditional domain contains an invalid constraint."
            )
        ctype = constraint.get("type")
        if ctype not in {"sum_lte", "linear_lte"}:
            raise ValueError(
                f"{model_name} conditional domain has unsupported constraint type={ctype!r}."
            )
        if ctype == "linear_lte":
            if (
                allow_extrapolation
                and constraint.get("enforce_during_extrapolation") is False
            ):
                continue
            terms = constraint.get("terms")
            if not isinstance(terms, Mapping) or not terms:
                raise ValueError(
                    f"{model_name} linear_lte constraint requires non-empty terms."
                )
            limit = constraint.get("max")
            if isinstance(limit, bool) or not isinstance(limit, Real):
                raise ValueError(
                    f"{model_name} conditional constraint has invalid max={limit!r}."
                )
            numeric_limit = float(limit)
            if not math.isfinite(numeric_limit):
                raise ValueError(
                    f"{model_name} conditional constraint has non-finite max={limit!r}."
                )
            total = 0.0
            for feature, coefficient in terms.items():
                if feature not in row:
                    raise ValueError(
                        f"{model_name} linear_lte constraint references unknown feature "
                        f"{feature!r}."
                    )
                if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
                    raise ValueError(
                        f"{model_name} linear_lte coefficient for {feature!r} is invalid."
                    )
                total += float(coefficient) * float(row[feature])
            if total > numeric_limit:
                violations.append(
                    f"conditional constraint {ctype} violated for key={tuple(key)!r}: "
                    f"value={total}, max={numeric_limit}"
                )
            continue
        features = constraint.get("features")
        if not isinstance(features, (list, tuple)) or not features:
            raise ValueError(
                f"{model_name} conditional constraint has invalid features={features!r}."
            )
        derived_features = constraint.get("derived_features", {})
        if not isinstance(derived_features, Mapping):
            raise ValueError(
                f"{model_name} conditional constraint has invalid derived_features."
            )
        feature_values: list[float] = []
        for feature in features:
            derived = derived_features.get(feature)
            if derived is not None:
                if not isinstance(derived, Mapping):
                    raise ValueError(
                        f"{model_name} conditional derived feature {feature!r} is invalid."
                    )
                source = derived.get("sqrt")
                if source not in row:
                    raise ValueError(
                        f"{model_name} conditional constraint references unknown feature {source!r}."
                    )
                source_value = float(row[source])
                if source_value < 0.0:
                    violations.append(
                        f"conditional sqrt source {source!r} is negative for key={tuple(key)!r}"
                    )
                    continue
                feature_values.append(math.sqrt(source_value))
            elif feature in row:
                feature_values.append(float(row[feature]))
            else:
                raise ValueError(
                    f"{model_name} conditional constraint references unknown feature {feature!r}."
                )
        limit = constraint.get("max")
        if isinstance(limit, bool) or not isinstance(limit, Real):
            raise ValueError(
                f"{model_name} conditional constraint has invalid max={limit!r}."
            )
        numeric_limit = float(limit)
        if not math.isfinite(numeric_limit):
            raise ValueError(
                f"{model_name} conditional constraint has non-finite max={limit!r}."
            )
        total = sum(feature_values)
        if (
            allow_extrapolation
            and constraint.get("enforce_during_extrapolation") is False
        ):
            continue
        if total > numeric_limit:
            violations.append(
                f"conditional constraint {ctype} violated for key={tuple(key)!r}: "
                f"sum={total}, max={numeric_limit}"
            )
    return violations


def validate_prediction_grid_domain(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    *,
    measurement_family: str | None = None,
    canonical_grid: CanonicalPredictionGrid | None = None,
    runtime_physical_bounds: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_constraints: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Validate schema/physical legality while allowing explicit ML prediction.

    Measured bounds and exact-row membership are enforced only when the
    descriptor opts into ``measured_only``.  The default profile-backed policy
    still enforces integer/enumerated axes, non-negative shape fields, and all
    declared relational constraints, but permits the canonical estimator to
    interpolate or extrapolate beyond observed rows.
    """
    domain, names, bounds, keys, kind = _validated_domain_components(
        model_name,
        model,
        X,
        canonical_grid,
    )
    runtime_policy = domain.get("runtime_prediction_policy")
    if runtime_policy is None:
        runtime_policy = (
            PREDICTION_DOMAIN_POLICY_MEASURED_ONLY
            if kind == DOMAIN_KIND_EXACT_ROWS
            else PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION
        )
    validate_prediction_domain_policy(runtime_policy, model_name=model_name)
    descriptor_physical_bounds = _validate_physical_bounds_descriptor(
        domain,
        model_name,
        names,
    )
    merged_physical_bounds = dict(descriptor_physical_bounds)
    if runtime_physical_bounds is not None:
        if not isinstance(runtime_physical_bounds, Mapping):
            raise ValueError(
                f"{model_name} runtime physical bounds must be a mapping."
            )
        runtime_descriptor = {
            "physical_bounds": runtime_physical_bounds,
        }
        runtime_validated = _validate_physical_bounds_descriptor(
            runtime_descriptor,
            f"{model_name} runtime",
            names,
        )
        for name, pair in runtime_validated.items():
            descriptor_pair = merged_physical_bounds.get(name, (None, None))
            lower = pair[0] if pair[0] is not None else descriptor_pair[0]
            upper = pair[1] if pair[1] is not None else descriptor_pair[1]
            if (
                descriptor_pair[0] is not None
                and lower is not None
                and lower < descriptor_pair[0]
            ):
                lower = descriptor_pair[0]
            if (
                descriptor_pair[1] is not None
                and upper is not None
                and upper > descriptor_pair[1]
            ):
                upper = descriptor_pair[1]
            if lower is not None and upper is not None and lower > upper:
                raise ValueError(
                    f"{model_name} runtime physical bounds for {name!r} are empty."
                )
            merged_physical_bounds[name] = (lower, upper)
    violations: list[str] = []
    if runtime_policy == PREDICTION_DOMAIN_POLICY_MEASURED_ONLY:
        for key in keys:
            violations.extend(_key_bound_violations(names, bounds, key))
    violations.extend(_validate_nonnegative_features(domain, model_name, names, keys))
    violations.extend(_physical_key_violations(names, merged_physical_bounds, keys))

    if kind == DOMAIN_KIND_EXACT_ROWS:
        raw_training_keys = domain.get("training_keys")
        if not isinstance(raw_training_keys, (list, tuple)):
            raise ValueError(
                f"{model_name} exact-row domain is missing training_keys; retrain the model."
            )
        _, training_keys = canonicalize_prediction_grid(names, raw_training_keys)
        training_set = set(training_keys)
        missing = [key for key in keys if key not in training_set]
        if missing:
            violations.append(
                f"unmeasured tuples={len(missing)}, examples={missing[:3]!r}"
            )
    elif kind in {DOMAIN_KIND_VERIFIED_CARTESIAN, DOMAIN_KIND_REGRESSION}:
        axes = _validate_axis_values(domain, model_name, names, bounds)
        semantics = _validate_axis_semantics(domain, model_name, names, axes, bounds)
        for key in keys:
            for index, name in enumerate(names):
                semantic = semantics[name]
                if (
                    semantic == AXIS_SEMANTIC_ENUMERATED
                    and key[index] not in axes[name]
                ):
                    violations.append(
                        f"{name}: requested={key[index]!r} is not in the "
                        f"enumerated axis_values={axes[name]!r}"
                    )
                elif (
                    semantic == AXIS_SEMANTIC_INTEGER_INTERVAL
                    and not float(key[index]).is_integer()
                ):
                    violations.append(
                        f"{name}: requested={key[index]!r} is not an integer "
                        "on an integer interval"
                    )
        if (
            kind == DOMAIN_KIND_REGRESSION
            and runtime_policy == PREDICTION_DOMAIN_POLICY_MEASURED_ONLY
        ):
            # Explicit bounded on-demand models retain measured-row semantics.
            # Bounds alone describe a box and would incorrectly admit sparse
            # combinations such as a measured (1,10)/(2,20) surface queried at
            # (1,20).  Ordinary profile-backed regression domains use the
            # allow_model_prediction policy and intentionally skip this check.
            raw_training_keys = domain.get("training_keys")
            if not isinstance(raw_training_keys, (list, tuple)):
                raise ValueError(
                    f"{model_name} measured-only regression domain requires "
                    "training_keys metadata."
                )
            _, training_keys = canonicalize_prediction_grid(names, raw_training_keys)
            training_set = set(training_keys)
            missing = [key for key in keys if key not in training_set]
            if missing:
                violations.append(
                    "exceeds profile domain: unmeasured tuples="
                    f"{len(missing)}, examples={missing[:3]!r}"
                )
        if kind == DOMAIN_KIND_REGRESSION and (
            bool(domain.get("constraints")) or bool(runtime_constraints)
        ):
            constraints = list(domain.get("constraints", ()))
            if runtime_constraints is not None:
                constraints.extend(dict(item) for item in runtime_constraints)
            for key in keys:
                violations.extend(
                    _conditional_key_violations(
                        model_name,
                        names,
                        key,
                        constraints,
                        allow_extrapolation=(
                            runtime_policy == PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION
                        ),
                    )
                )
    elif kind == DOMAIN_KIND_CONDITIONAL:
        if "axis_semantics" in domain:
            axes = _validate_axis_values(domain, model_name, names, bounds)
            semantics = _validate_axis_semantics(
                domain, model_name, names, axes, bounds
            )
            for key in keys:
                for index, name in enumerate(names):
                    semantic = semantics[name]
                    if (
                        semantic == AXIS_SEMANTIC_ENUMERATED
                        and key[index] not in axes[name]
                    ):
                        violations.append(
                            f"{name}: requested={key[index]!r} is not in the "
                            f"enumerated axis_values={axes[name]!r}"
                        )
                    elif (
                        semantic == AXIS_SEMANTIC_INTEGER_INTERVAL
                        and not float(key[index]).is_integer()
                    ):
                        violations.append(
                            f"{name}: requested={key[index]!r} is not an integer "
                            "on an integer interval"
                        )
        constraints = list(domain.get("constraints", ()))
        if runtime_constraints is not None:
            constraints.extend(dict(item) for item in runtime_constraints)
        if (
            runtime_policy == PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION
            and not runtime_constraints
            and any(
                _key_bound_violations(names, bounds, key)
                for key in keys
            )
            and any(
                isinstance(item, Mapping)
                and item.get("enforce_during_extrapolation") is False
                for item in constraints
            )
        ):
            violations.append(
                "physical conditional context cap is required for extrapolated keys"
            )
        for key in keys:
            if not constraints:
                continue
            if "prefill_chunk_size_squared" in names:
                squared = float(key[names.index("prefill_chunk_size_squared")])
                integer_squared = squared.is_integer() and squared >= 0.0
                perfect_square = False
                if integer_squared:
                    root = math.isqrt(int(squared))
                    perfect_square = root * root == int(squared)
                if not perfect_square:
                    violations.append(
                        "prefill_chunk_size_squared must be a non-negative perfect square"
                    )
            violations.extend(
                _conditional_key_violations(
                    model_name,
                    names,
                    key,
                    constraints,
                    allow_extrapolation=(
                        runtime_policy == PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION
                    ),
                )
            )
    elif kind == DOMAIN_KIND_INTEGER_INTERVAL:
        for key in keys:
            if not float(key[0]).is_integer():
                violations.append(
                    f"{names[0]}: requested={key[0]!r} is not an integer "
                    "within the declared interval"
                )

    if violations:
        raise ValueError(
            f"{model_name} prediction grid violates its declared domain "
            "(exceeds profile domain when the measured-only policy applies) "
            f"(domain_kind={kind!r}, runtime_prediction_policy={runtime_policy!r}, "
            f"measurement_family={measurement_family!r}): "
            + "; ".join(violations[:8])
        )


def classify_prediction_key(
    domain: Mapping[str, Any],
    key: Sequence[Any],
) -> dict[str, Any]:
    """Classify one legal key relative to measured profile coverage.

    The function is intentionally side-effect free.  It is called only when
    diagnostics are enabled or by focused validation tests, so the normal
    prediction path does not pay for per-key provenance bookkeeping.
    """

    names_raw = domain.get("feature_names")
    if not isinstance(names_raw, (list, tuple)):
        raise ValueError("Prediction domain is missing feature_names metadata.")
    names = tuple(str(name) for name in names_raw)
    canonical_key = canonicalize_prediction_key(key)
    if len(canonical_key) != len(names):
        raise ValueError(
            f"Prediction key length mismatch: expected={len(names)}, "
            f"actual={len(canonical_key)}."
        )
    bounds = _validate_bounds(domain, "prediction", names)
    raw_training = domain.get("training_keys", ())
    if isinstance(raw_training, (list, tuple)) and raw_training:
        _, training_keys = canonicalize_prediction_grid(names, raw_training)
    else:
        raw_axes = domain.get("axis_values", {})
        if len(names) == 1 and isinstance(raw_axes, Mapping):
            axis = raw_axes.get(names[0], ())
            training_keys = tuple((canonicalize_prediction_key((value,))[0],) for value in axis)
        else:
            training_keys = ()
    direct = canonical_key in set(training_keys)
    outside_axes: dict[str, float] = {}
    normalized_gaps: dict[str, float] = {}
    for index, name in enumerate(names):
        lower, upper = bounds[name]
        value = float(canonical_key[index])
        gap = 0.0
        if value < lower:
            gap = lower - value
        elif value > upper:
            gap = value - upper
        if gap > 0.0:
            outside_axes[name] = gap
        span = max(upper - lower, 1.0)
        normalized_gaps[name] = gap / span

    sparse_gap = (
        not direct
        and domain.get("domain_kind") == DOMAIN_KIND_REGRESSION
    )
    if direct:
        classification = PREDICTION_CLASS_DIRECT_MEASURED
    elif outside_axes:
        classification = PREDICTION_CLASS_EXTRAPOLATION
    else:
        classification = PREDICTION_CLASS_INTERPOLATION
    return {
        "classification": classification,
        "sparse_gap": bool(sparse_gap),
        "outside_axes": outside_axes,
        "axis_gap": normalized_gaps,
        "key": canonical_key,
    }


def filter_prediction_grid_to_domain(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    *,
    measurement_family: str | None = None,
    runtime_physical_bounds: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_constraints: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Remove only conditionally invalid tuples from a finite prediction grid."""
    domain, names, bounds, keys, kind = _validated_domain_components(
        model_name, model, X
    )
    if kind != DOMAIN_KIND_CONDITIONAL:
        validate_prediction_grid_domain(
            model_name,
            model,
            X,
            measurement_family=measurement_family,
            runtime_physical_bounds=runtime_physical_bounds,
            runtime_constraints=runtime_constraints,
        )
        return X.copy()

    bound_violations: list[str] = []
    valid_keys: set[tuple[int | float, ...]] = set()
    constraints = list(domain.get("constraints", ()))
    if runtime_constraints is not None:
        constraints.extend(dict(item) for item in runtime_constraints)
    runtime_policy = domain.get("runtime_prediction_policy")
    if runtime_policy is None:
        runtime_policy = PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION
    validate_prediction_domain_policy(runtime_policy, model_name=model_name)
    for key in keys:
        key_bounds = _key_bound_violations(names, bounds, key)
        if key_bounds and runtime_policy == PREDICTION_DOMAIN_POLICY_MEASURED_ONLY:
            bound_violations.extend(key_bounds)
        elif not _conditional_key_violations(
            model_name,
            names,
            key,
            constraints,
            allow_extrapolation=(
                runtime_policy == PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION
            ),
        ):
            valid_keys.add(key)

    # Bounds are a profile/runtime contract mismatch, not a conditional hole.
    # Never hide them by filtering.
    if bound_violations:
        raise ValueError(
            f"{model_name} prediction grid exceeds profile domain "
            f"(domain_kind={kind!r}, measurement_family={measurement_family!r}): "
            + "; ".join(bound_violations[:8])
        )
    if not valid_keys:
        raise ValueError(
            f"{model_name} has no valid prediction-grid tuples after applying its "
            f"conditional profile domain (measurement_family={measurement_family!r})."
        )

    mask = [
        canonicalize_prediction_key(row) in valid_keys
        for row in X.itertuples(index=False, name=None)
    ]
    filtered = X.loc[mask].copy()
    validate_prediction_grid_domain(
        model_name,
        model,
        filtered,
        measurement_family=measurement_family,
        runtime_physical_bounds=runtime_physical_bounds,
        runtime_constraints=runtime_constraints,
    )
    return filtered


def validate_prediction_cache(
    model_name: str,
    predictions: Mapping[Any, Any],
    expected_keys: Iterable[Sequence[Any]],
    feature_names: Sequence[str],
    *,
    measurement_family: str | None = None,
    canonical_grid: CanonicalPredictionGrid | None = None,
    prediction_keys_are_canonical: bool = False,
) -> None:
    """Require a persisted finite cache to exactly match the requested grid."""
    if not isinstance(predictions, Mapping):
        raise ValueError(
            f"{model_name} prediction cache is not a mapping "
            f"(measurement_family={measurement_family!r})."
        )
    names = tuple(str(name) for name in feature_names)
    if canonical_grid is None:
        _, expected = canonicalize_prediction_grid(names, expected_keys)
    else:
        if canonical_grid.feature_names != names:
            raise ValueError(
                f"{model_name} canonical prediction-cache schema mismatch "
                f"(measurement_family={measurement_family!r})."
            )
        expected = canonical_grid.keys

    if prediction_keys_are_canonical:
        key_sets_match = (
            len(predictions) == len(expected)
            and all(key in predictions for key in expected)
        )
        actual = None
    else:
        try:
            actual = tuple(sorted(
                {
                    canonicalize_prediction_key(key)
                    for key in predictions.keys()
                },
                key=repr,
            ))
        except ValueError as exc:
            raise ValueError(
                f"{model_name} prediction cache contains an invalid key "
                f"(measurement_family={measurement_family!r}): {exc}"
            ) from exc
        key_sets_match = actual == expected

    if not key_sets_match:
        expected_set = set(expected)
        actual_set = (
            set(predictions.keys())
            if prediction_keys_are_canonical
            else set(actual or ())
        )
        missing = sorted(expected_set - actual_set, key=repr)
        extra = sorted(actual_set - expected_set, key=repr)
        raise ValueError(
            f"{model_name} has incomplete prediction cache "
            f"(measurement_family={measurement_family!r}): "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"missing_examples={missing[:3]!r}, extra_examples={extra[:3]!r}"
        )
    invalid_values: list[tuple[Any, Any]] = []
    for key in expected:
        value = predictions.get(key)
        if value is None and key not in predictions:
            # The key-set check above already reports this, but keep the value
            # validation defensive for Mapping implementations with unusual
            # ``get`` behavior.
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid_values.append((key, value))
            continue
        if not math.isfinite(numeric) or numeric < 0.0:
            invalid_values.append((key, value))
    if invalid_values:
        raise ValueError(
            f"{model_name} prediction cache contains invalid values "
            f"(measurement_family={measurement_family!r}): "
            f"examples={invalid_values[:3]!r}"
        )
