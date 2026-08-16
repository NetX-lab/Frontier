"""Benchmark finite prediction-grid materialization contract overhead."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import frontier.execution_time_predictor.prediction_cache_contract as contract
from frontier.execution_time_predictor.prediction_cache_contract import (
    ON_DEMAND_DOMAIN_POLICY_BOUNDED,
    PREDICTION_CACHE_CONTRACT_VERSION,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import MeasurementType


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("not used")

    def _get_grid_search_params(self):
        raise AssertionError("not used")


class _ZeroModel:
    n_features_in_ = 2

    def __init__(self) -> None:
        self.calls = 0
        self._frontier_model_hash = "prediction-grid-benchmark"
        self._frontier_feature_domain = {
            "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
            "operator_name": "attn_prefill",
            "feature_names": [
                "kv_cache_size",
                "prefill_chunk_size_squared",
            ],
            "domain_kind": "conditional_interpolation",
            "on_demand_policy": ON_DEMAND_DOMAIN_POLICY_BOUNDED,
            "bounds": {
                "kv_cache_size": {"min": 0.0, "max": 4095.0},
                "prefill_chunk_size_squared": {
                    "min": 1.0,
                    "max": float(65**2),
                },
            },
            "constraints": [
                {
                    "type": "sum_lte",
                    "features": ["kv_cache_size", "prefill_chunk_size"],
                    "derived_features": {
                        "prefill_chunk_size": {
                            "sqrt": "prefill_chunk_size_squared"
                        }
                    },
                    "max": 4160.0,
                }
            ],
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        return np.zeros(len(features), dtype=np.float64)


def _build_grid(batch_count: int, context_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kv_cache_size": np.tile(
                np.arange(context_count, dtype=np.int64),
                batch_count,
            ),
            "prefill_chunk_size_squared": np.repeat(
                np.arange(1, batch_count + 1, dtype=np.int64) ** 2,
                context_count,
            ),
        }
    )


def _build_predictor(model: _ZeroModel, cache_dir: Path) -> _ConcretePredictor:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(no_cache=True)
    predictor._cache_dir = str(cache_dir)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._models = {"attn_prefill": model}
    predictor._get_prediction_context_hash = lambda *_args: "benchmark-context"
    predictor._load_model_predication_cache = lambda *_args: None
    predictor._store_model_predication_cache = lambda *_args: None
    return predictor


def _run_current(
    frame: pd.DataFrame,
    model: _ZeroModel,
    cache_dir: Path,
) -> dict[tuple[int | float, ...], float]:
    predictor = _build_predictor(model, cache_dir)
    original_to_csv = pd.DataFrame.to_csv
    pd.DataFrame.to_csv = lambda *_args, **_kwargs: None
    try:
        return predictor._get_model_prediction("attn_prefill", model, frame)
    finally:
        pd.DataFrame.to_csv = original_to_csv


def _run_legacy_repeated(
    frame: pd.DataFrame,
    model: _ZeroModel,
) -> dict[tuple[int | float, ...], float]:
    feature_names, requested_keys = contract.prediction_grid_from_dataframe(frame)
    contract.validate_prediction_grid_domain(
        "attn_prefill",
        model,
        frame,
        measurement_family="eager",
    )
    contract.prediction_grid_digest(feature_names, requested_keys)
    predictions_array = model.predict(frame)
    ordered_keys = [
        contract.canonicalize_prediction_key(row)
        for row in frame.itertuples(index=False, name=None)
    ]
    predictions = dict(zip(ordered_keys, predictions_array))
    contract.validate_prediction_cache(
        "attn_prefill",
        predictions,
        requested_keys,
        feature_names,
        measurement_family="eager",
    )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("current", "legacy-repeated"),
        required=True,
    )
    parser.add_argument("--batch-count", type=int, default=65)
    parser.add_argument("--context-count", type=int, default=4096)
    args = parser.parse_args()

    cache_dir = Path("/data/ycfeng/tmp/pr4_prediction_grid_benchmark")
    if not cache_dir.is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)

    frame = _build_grid(args.batch_count, args.context_count)
    model = _ZeroModel()
    canonicalize_calls = 0
    original_canonicalize = contract.canonicalize_prediction_grid

    def counted_canonicalize(*call_args, **call_kwargs):
        nonlocal canonicalize_calls
        canonicalize_calls += 1
        return original_canonicalize(*call_args, **call_kwargs)

    contract.canonicalize_prediction_grid = counted_canonicalize
    tracemalloc.start()
    started = time.perf_counter()
    try:
        if args.mode == "current":
            predictions = _run_current(frame, model, cache_dir)
        else:
            predictions = _run_legacy_repeated(frame, model)
    finally:
        elapsed_seconds = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        contract.canonicalize_prediction_grid = original_canonicalize

    print(
        json.dumps(
            {
                "mode": args.mode,
                "rows": int(len(frame)),
                "unique_predictions": int(len(predictions)),
                "model_predict_calls": int(model.calls),
                "canonicalize_prediction_grid_calls": canonicalize_calls,
                "elapsed_seconds": elapsed_seconds,
                "tracemalloc_peak_mib": peak_bytes / (1024**2),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
