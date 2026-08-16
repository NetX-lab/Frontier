"""Validate real H800 attention models against runtime prediction policies.

The probe intentionally exercises both direct measured rows and legal gaps.  It
does not replace the simulator or insert a prediction fallback; it calls the
same canonical lookup/on-demand methods used by runtime initialization.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd

from frontier.execution_time_predictor.prediction_cache_contract import (
    build_on_demand_prediction_record,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.types import ClusterType, MeasurementType


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):  # pragma: no cover - this is a runtime-only probe
        raise AssertionError("the numeric probe must not train")

    def _get_grid_search_params(self):  # pragma: no cover
        raise AssertionError("the numeric probe must not train")


class _CountingEstimator:
    """Forward a persisted estimator while counting canonical model calls."""

    def __init__(self, estimator) -> None:
        self._estimator = estimator
        self.calls = 0
        self.__dict__.update(getattr(estimator, "__dict__", {}))

    def __getattr__(self, name):
        return getattr(self._estimator, name)

    def predict(self, frame):
        self.calls += 1
        return self._estimator.predict(frame)


def _predictor(model_name, estimator, predictions, *, diagnostics=True, max_batch=10):
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(
        enable_prediction_domain_diagnostics=diagnostics,
        prediction_max_tokens_per_request=4096,
        prediction_max_batch_size=max_batch,
        prediction_max_prefill_chunk_size=4096,
        kv_cache_prediction_granularity=64,
        prediction_min_kv_cache_size=0,
    )
    predictor._model_config = ModelConfig.from_model_name("Qwen3-30B-A3B-tiny")
    predictor._max_tokens = 4096
    predictor._serving_max_tokens_per_request = 4096
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    counted = _CountingEstimator(estimator)
    predictor._models = {model_name: counted}
    predictor._predictions = (
        predictions(counted) if callable(predictions) else predictions
    )
    return predictor, counted


def _median(df: pd.DataFrame, mask, target: str) -> float | None:
    values = df.loc[mask, target].dropna()
    return float(values.median()) if not values.empty else None


def _error(predicted: float, measured: float | None) -> dict[str, float | None] | None:
    if measured is None:
        return None
    absolute = abs(float(predicted) - measured)
    return {
        "predicted_ms": float(predicted),
        "measured_ms": measured,
        "absolute_error_ms": absolute,
        "relative_error_pct": absolute / abs(measured) * 100.0 if measured else None,
    }


def _load_model(root: Path, operator: str):
    matches = sorted(
        path
        for path in root.glob(f"{operator}_*.pkl")
        if not path.name.startswith(f"{operator}_in_mixed_")
        and not path.name.startswith(f"{operator}_mixed_")
    )
    if len(matches) != 1:
        raise ValueError(f"expected one {operator} artifact in {root}, got {matches}")
    return matches[0], joblib.load(matches[0])


def _cache_write_probe(root: Path, df: pd.DataFrame) -> dict[str, object]:
    path, model = _load_model(root, "attn_kv_cache_save")
    feature_names = list(model._frontier_feature_names)
    measured_keys = {
        tuple(int(row[name]) for name in feature_names)
        for _, row in df.iterrows()
        if all(pd.notna(row[name]) for name in feature_names)
    }
    requested = [(1, 0, 1), (8, 0, 1), (32, 0, 1), (512, 0, 1), (4096, 0, 1)]
    if (8, 0, 1) in measured_keys:
        raise AssertionError("the selected real profile unexpectedly measured (8,0,1)")

    def records(counted):
        return {
            "attn_kv_cache_save": build_on_demand_prediction_record(
                "attn_kv_cache_save",
                counted,
                feature_names,
                exact_lookup=getattr(counted, "_frontier_exact_lookup", {}),
            )
        }

    predictor, counted = _predictor(
        "attn_kv_cache_save", model, records, diagnostics=True
    )
    values: dict[str, object] = {}
    for key in requested:
        before = counted.calls
        value = predictor._get_on_demand_prediction(
            "attn_kv_cache_save", dict(zip(feature_names, key))
        )
        values[str(key)] = {
            "value_ms": float(value),
            "model_calls_delta": counted.calls - before,
            "runtime_cache_entries": len(
                predictor._runtime_cache["eager"]["attn_kv_cache_save"]
            ),
            "measured": key in measured_keys,
        }

    before = counted.calls
    repeat = predictor._get_on_demand_prediction(
        "attn_kv_cache_save",
        {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1},
    )
    repeat_delta = counted.calls - before

    before = counted.calls
    try:
        predictor._get_on_demand_prediction(
            "attn_kv_cache_save",
            {"total_tokens": 0, "kv_cache_size": 0, "batch_size": 1},
        )
    except ValueError as exc:
        invalid_message = str(exc)
    else:  # pragma: no cover - fail-fast contract
        raise AssertionError("physical-invalid cache-write tuple was accepted")
    invalid_delta = counted.calls - before

    try:
        predictor._get_on_demand_prediction(
            "attn_kv_cache_save", {"total_tokens": 8, "kv_cache_size": 0}
        )
    except ValueError as exc:
        schema_message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("cache-write schema mismatch was accepted")

    diagnostics = predictor.get_prediction_domain_diagnostics()
    measured_target = _median(
        df,
        (df["total_tokens"] == 1)
        & (df["kv_cache_size"] == 0)
        & (df["batch_size"] == 1),
        "time_stats.attn_kv_cache_save.median",
    )
    direct_prediction = float(
        model.predict(pd.DataFrame([[1, 0, 1]], columns=feature_names))[0]
    )
    return {
        "artifact": str(path),
        "domain_kind": model._frontier_feature_domain["domain_kind"],
        "runtime_prediction_policy": model._frontier_feature_domain[
            "runtime_prediction_policy"
        ],
        "feature_names": feature_names,
        "measured_key_count": len(measured_keys),
        "requested": values,
        "repeat_value_ms": float(repeat),
        "repeat_model_calls_delta": repeat_delta,
        "invalid_physical_error": invalid_message,
        "invalid_physical_model_calls_delta": invalid_delta,
        "schema_mismatch_error": schema_message,
        "diagnostics": diagnostics,
        "direct_model_fit": _error(direct_prediction, measured_target),
    }


def _finite_gap_probe(
    root: Path,
    operator: str,
    keys,
    feature_names,
    measured_value: float | None,
):
    path, model = _load_model(root, operator)
    direct_prediction = float(
        model.predict(pd.DataFrame([list(keys[0])], columns=feature_names))[0]
    )
    counted = _CountingEstimator(model)
    # Keep one direct row in the finite cache and force all other keys through
    # the canonical estimator.  The persisted descriptor remains authoritative.
    direct_key = tuple(keys[0])
    predictor, counted = _predictor(
        operator,
        model,
        lambda c: {operator: {direct_key: 0.0}},
        diagnostics=True,
        max_batch=10,
    )
    results = {}
    for key in keys:
        before = counted.calls
        try:
            value = predictor._get_lookup_or_predict(operator, tuple(key), feature_names)
        except ValueError as exc:
            results[str(tuple(key))] = {
                "error": str(exc),
                "model_calls_delta": counted.calls - before,
            }
        else:
            results[str(tuple(key))] = {
                "value_ms": float(value),
                "model_calls_delta": counted.calls - before,
                "runtime_cache_entries": len(
                    predictor._runtime_cache["eager"][operator]
                ),
            }
    return {
        "artifact": str(path),
        "domain_kind": model._frontier_feature_domain["domain_kind"],
        "runtime_prediction_policy": model._frontier_feature_domain[
            "runtime_prediction_policy"
        ],
        "results": results,
        "diagnostics": predictor.get_prediction_domain_diagnostics(),
        "direct_model_fit": _error(direct_prediction, measured_value),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.data_csv, low_memory=False)
    output: dict[str, object] = {}
    for tp in (1, 2, 4, 8):
        root = args.train_root.parent / f"{args.train_root.name}_tp{tp}_20260814_1"
        if not root.is_dir():
            root = args.train_root / f"tp{tp}"
        tp_df = df[df["num_tensor_parallel_workers"] == tp].copy()
        decode_truth = _median(
            tp_df,
            (tp_df["batch_size"] == 1)
            & (tp_df["kv_cache_size"] == 0)
            & (tp_df["is_prefill"] == False)
            & (tp_df["is_mixed_batch"] == False),
            "time_stats.attn_decode.median",
        )
        prefill_truth = _median(
            tp_df,
            (tp_df["batch_size"] == 1)
            & (tp_df["kv_cache_size"] == 0)
            & (tp_df["prefill_chunk_size"] == 32)
            & (tp_df["is_prefill"] == True)
            & (tp_df["is_mixed_batch"] == False),
            "time_stats.attn_prefill.median",
        )
        cache_write = _cache_write_probe(root, tp_df)
        decode = _finite_gap_probe(
            root,
            "attn_decode",
            [(1, 0), (3, 17), (10, 4096), (11, 0)],
            ["batch_size", "kv_cache_size"],
            decode_truth,
        )
        prefill = _finite_gap_probe(
            root,
            "attn_prefill",
            [(0, 32**2), (0, 1), (64, 64**2), (4096, 1), (4096, 32**2), (0, 2)],
            ["kv_cache_size", "prefill_chunk_size_squared"],
            prefill_truth,
        )
        cache_gap = cache_write["requested"]["(8, 0, 1)"]
        if cache_gap["measured"] or cache_gap["model_calls_delta"] != 1:
            raise AssertionError(
                "real cache-write gap must call the canonical model exactly once"
            )
        if "error" not in prefill["results"]["(4096, 1024)"]:
            raise AssertionError(
                "prefill tuple exceeding serving context was not rejected"
            )
        output[f"tp{tp}"] = {
            "cache_write": cache_write,
            "decode": decode,
            "prefill": prefill,
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
