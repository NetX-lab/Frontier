"""Validate the R-009 runtime policy with a persisted dense attention model.

This check is intentionally model-parameterized and uses the same predictor
methods as runtime initialization.  It distinguishes measured cache hits from
canonical-model interpolation/extrapolation and verifies physical fail-fast
before prediction.
"""

from __future__ import annotations

import argparse
import hashlib
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
    def _get_estimator(self):  # pragma: no cover - this check must not train
        raise AssertionError("runtime check must not train")

    def _get_grid_search_params(self):  # pragma: no cover
        raise AssertionError("runtime check must not train")


class _CountingEstimator:
    def __init__(self, estimator) -> None:
        self._estimator = estimator
        self.calls = 0
        self.__dict__.update(getattr(estimator, "__dict__", {}))

    def __getattr__(self, name):
        return getattr(self._estimator, name)

    def predict(self, frame):
        self.calls += 1
        return self._estimator.predict(frame)


def _load(root: Path, operator: str):
    matches = sorted(
        path
        for path in root.glob(f"{operator}_*.pkl")
        if not path.name.startswith(f"{operator}_in_mixed_")
        and not path.name.startswith(f"{operator}_mixed_")
    )
    if len(matches) != 1:
        raise ValueError(f"expected one {operator} artifact in {root}, got {matches}")
    return matches[0], joblib.load(matches[0])


def _build_predictor(
    model_name: str,
    model_config: ModelConfig,
    operator: str,
    estimator,
    predictions,
    *,
    diagnostics: bool,
    serving_max_tokens: int,
    max_batch: int,
) -> tuple[_ConcretePredictor, _CountingEstimator]:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(
        enable_prediction_domain_diagnostics=diagnostics,
        prediction_max_tokens_per_request=serving_max_tokens,
        prediction_max_batch_size=max_batch,
        prediction_max_prefill_chunk_size=serving_max_tokens,
        kv_cache_prediction_granularity=64,
        prediction_min_kv_cache_size=0,
    )
    predictor._model_config = model_config
    predictor._max_tokens = serving_max_tokens
    predictor._serving_max_tokens_per_request = serving_max_tokens
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    counted = _CountingEstimator(estimator)
    predictor._models = {operator: counted}
    predictor._predictions = predictions(counted) if callable(predictions) else predictions
    return predictor, counted


def _sha256_tree(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.glob("*.pkl")):
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _cache_write_probe(root: Path, model_config: ModelConfig, df: pd.DataFrame, args):
    path, model = _load(root, "attn_kv_cache_save")
    names = list(model._frontier_feature_names)
    measured = {
        tuple(int(row[name]) for name in names)
        for _, row in df.iterrows()
        if all(pd.notna(row[name]) for name in names)
    }

    def records(counted):
        return {
            "attn_kv_cache_save": build_on_demand_prediction_record(
                "attn_kv_cache_save",
                counted,
                names,
                exact_lookup=getattr(counted, "_frontier_exact_lookup", {}),
            )
        }

    predictor, counted = _build_predictor(
        args.model_name, model_config, "attn_kv_cache_save", model, records,
        diagnostics=args.diagnostics, serving_max_tokens=args.serving_max_tokens,
        max_batch=args.max_batch,
    )
    values = {}
    for key in ((1, 0, 1), (8, 0, 1), (32, 0, 1)):
        before = counted.calls
        value = predictor._get_on_demand_prediction(
            "attn_kv_cache_save", dict(zip(names, key))
        )
        values[str(key)] = {
            "value_ms": float(value),
            "measured": key in measured,
            "model_calls_delta": counted.calls - before,
        }
    before = counted.calls
    repeat = predictor._get_on_demand_prediction(
        "attn_kv_cache_save", dict(zip(names, (8, 0, 1)))
    )
    repeat_delta = counted.calls - before
    before = counted.calls
    try:
        predictor._get_on_demand_prediction(
            "attn_kv_cache_save", dict(zip(names, (0, 0, 1)))
        )
    except ValueError as exc:
        invalid_error = str(exc)
    else:  # pragma: no cover
        raise AssertionError("physical-invalid cache-write tuple was accepted")
    invalid_delta = counted.calls - before
    return {
        "artifact": str(path),
        "domain_kind": model._frontier_feature_domain["domain_kind"],
        "runtime_prediction_policy": model._frontier_feature_domain["runtime_prediction_policy"],
        "requested": values,
        "repeat_value_ms": float(repeat),
        "repeat_model_calls_delta": repeat_delta,
        "invalid_physical_error": invalid_error,
        "invalid_physical_model_calls_delta": invalid_delta,
        "runtime_cache_entries": len(
            predictor._runtime_cache[
                predictor._measurement_family_name(predictor._active_measurement_type)
            ]["attn_kv_cache_save"]
        ),
        "diagnostics_state_created": hasattr(
            predictor, "_prediction_domain_diagnostics"
        ),
        "diagnostics": predictor.get_prediction_domain_diagnostics(),
    }


def _finite_probe(root: Path, model_config: ModelConfig, operator: str, keys, names, args):
    path, model = _load(root, operator)
    # Make one known measured key a direct finite-cache hit.  Every other key
    # must use the persisted canonical model, never a nearest row.
    direct = tuple(keys[0])
    predictor, counted = _build_predictor(
        args.model_name, model_config, operator, model,
        {operator: {direct: 0.0}}, diagnostics=args.diagnostics,
        serving_max_tokens=args.serving_max_tokens, max_batch=args.max_batch,
    )
    result = {}
    for key in keys:
        before = counted.calls
        try:
            value = predictor._get_lookup_or_predict(operator, tuple(key), names)
        except ValueError as exc:
            result[str(tuple(key))] = {
                "error": str(exc), "model_calls_delta": counted.calls - before,
            }
        else:
            result[str(tuple(key))] = {
                "value_ms": float(value), "model_calls_delta": counted.calls - before,
            }
    return {
        "artifact": str(path),
        "domain_kind": model._frontier_feature_domain["domain_kind"],
        "runtime_prediction_policy": model._frontier_feature_domain["runtime_prediction_policy"],
        "results": result,
        "diagnostics": predictor.get_prediction_domain_diagnostics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--serving-max-tokens", type=int, default=4096)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.serving_max_tokens != 4096:
        raise ValueError("this release check intentionally preserves serving max 4096")

    df = pd.read_csv(args.data_csv, low_memory=False)
    df = df[df["num_tensor_parallel_workers"] == args.tensor_parallel_size].copy()
    model_config = ModelConfig.from_model_name(args.model_name)
    before_hashes = _sha256_tree(args.train_root)
    cache_write = _cache_write_probe(args.train_root, model_config, df, args)
    decode = _finite_probe(
        args.train_root, model_config, "attn_decode",
        ((1, 1), (3, 17), (8, 96), (10, 4096)),
        ["batch_size", "kv_cache_size"], args,
    )
    prefill = _finite_probe(
        args.train_root, model_config, "attn_prefill",
        ((0, 1024), (64, 4096), (4096, 1), (4096, 1024)),
        ["kv_cache_size", "prefill_chunk_size_squared"], args,
    )
    after_hashes = _sha256_tree(args.train_root)
    if before_hashes != after_hashes:
        raise AssertionError("runtime probe mutated persisted model artifacts")

    assert cache_write["requested"]["(8, 0, 1)"]["model_calls_delta"] == 1
    assert cache_write["requested"]["(1, 0, 1)"]["measured"] is True
    assert cache_write["requested"]["(1, 0, 1)"]["model_calls_delta"] == 0
    assert cache_write["repeat_model_calls_delta"] == 0
    assert cache_write["invalid_physical_model_calls_delta"] == 0
    if not args.diagnostics:
        assert cache_write["diagnostics_state_created"] is False
    assert decode["results"]["(1, 1)"]["model_calls_delta"] == 0
    assert decode["results"]["(3, 17)"]["model_calls_delta"] == 1
    assert prefill["results"]["(4096, 1)"]["model_calls_delta"] == 1
    invalid = prefill["results"]["(4096, 1024)"]
    assert "error" in invalid and invalid["model_calls_delta"] == 0
    output = {
        "model_name": args.model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "serving_max_tokens": args.serving_max_tokens,
        "cache_write": cache_write,
        "decode": decode,
        "prefill": prefill,
        "persisted_artifacts_unchanged": True,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
