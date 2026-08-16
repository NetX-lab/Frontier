"""Probe real H800 attention artifacts through runtime materialization.

This is an integration harness, not a simulator fallback.  It loads the exact
model artifacts produced by ``frontier.training.cli attention`` and exercises
the same finite-grid and explicit on-demand paths used during predictor
initialization.  Serving and prediction limits intentionally remain 4096.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.types import ClusterType, MeasurementType


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):  # pragma: no cover - harness must never train
        raise AssertionError("runtime probe must load trained artifacts")

    def _get_grid_search_params(self):  # pragma: no cover - harness must never train
        raise AssertionError("runtime probe must load trained artifacts")


OPS = (
    "attn_kv_cache_save",
    "attn_prefill",
    "attn_decode",
    "attn_prefill_mixed",
    "attn_decode_in_mixed",
)


def _load_models(model_dir: Path) -> dict[str, object]:
    models: dict[str, object] = {}
    for operator in OPS:
        matches = sorted(
            path
            for path in model_dir.glob(f"{operator}_*.pkl")
            if not any(
                path.name.startswith(f"{operator}{suffix}_")
                for suffix in ("_mixed", "_in_mixed")
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one trained artifact for {operator} in {model_dir}; "
                f"found {[path.name for path in matches]}"
            )
        models[operator] = joblib.load(matches[0])
    return models


def _build_predictor(
    *,
    models: dict[str, object],
    cluster_type: ClusterType,
    cache_dir: Path,
) -> _ConcretePredictor:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._cluster_type = cluster_type
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._is_mla_attention_family = lambda: False
    predictor._cache_dir = str(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    predictor._config = SimpleNamespace(
        no_cache=False,
        prediction_max_batch_size=8,
        prediction_max_tokens_per_request=4096,
        prediction_max_prefill_chunk_size=4096,
        kv_cache_prediction_granularity=64,
        prediction_min_kv_cache_size=0,
        attention_prefill_batching_overhead_fraction=0.1,
        attention_decode_batching_overhead_fraction=0.1,
    )
    predictor._serving_max_tokens_per_request = 4096
    predictor._models = models
    predictor._model_config = ModelConfig.from_model_name("Qwen3-30B-A3B-tiny")
    predictor._enable_dummy_mode = False
    predictor._get_prediction_context_hash = (
        lambda *_args, **_kwargs: "h800-runtime-4096-mixed-v1"
    )
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._predictions = {}
    return predictor


def _mixed_features(batch_size: int, total_tokens: int, kv_cache_size: int) -> dict[str, float]:
    if total_tokens % batch_size != 0:
        raise ValueError("Probe mixed features require equal integer sequence lengths")
    seq_lens = [total_tokens // batch_size] * batch_size
    values = np.asarray(seq_lens, dtype=np.float64)
    avg = float(values.mean())
    variance = float(values.var())
    std = float(values.std())
    cv = std / avg if avg else 0.0
    return {
        "avg_seq_len": avg,
        "batch_cv_interaction": float(batch_size * cv),
        "batch_size": batch_size,
        "batch_variance_interaction": float(batch_size * variance),
        "kv_cache_size": kv_cache_size,
        "max_seq_len": int(values.max()),
        "min_seq_len": int(values.min()),
        "seq_len_cv": cv,
        "seq_len_range": int(values.max() - values.min()),
        "seq_len_variance": variance,
        "total_tokens": total_tokens,
        "total_tokens_squared": int(total_tokens * total_tokens),
    }


def _probe_role(
    *,
    models: dict[str, object],
    role: ClusterType,
    cache_dir: Path,
) -> dict[str, object]:
    predictor = _build_predictor(models=models, cluster_type=role, cache_dir=cache_dir)
    predictions = predictor._predict_for_attention_layer_models()
    predictor._predictions = predictions

    result: dict[str, object] = {
        "role": role.name,
        "operators": sorted(predictions),
        "cache_files": len(list(cache_dir.iterdir())),
    }
    if "attn_decode" in predictions:
        decode = predictions["attn_decode"]
        result["decode_keys"] = len(decode)
        result["decode_key_min"] = list(min(decode))
        result["decode_key_max"] = list(max(decode))
    if "attn_prefill" in predictions:
        prefill = predictions["attn_prefill"]
        result["prefill_keys"] = len(prefill)
        result["prefill_key_min"] = list(min(prefill))
        result["prefill_key_max"] = list(max(prefill))

    if "attn_prefill_mixed" in predictions:
        mixed_values = []
        for batch_size, total_tokens, kv_cache_size in (
            (8, 4096, 0),
            (2, 64, 64),
            (2, 2, 4096),
        ):
            value = predictor._get_on_demand_prediction(
                "attn_prefill_mixed",
                _mixed_features(batch_size, total_tokens, kv_cache_size),
            )
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid mixed prediction {value!r}")
            mixed_values.append(float(value))
        repeated = predictor._get_on_demand_prediction(
            "attn_prefill_mixed", _mixed_features(2, 2, 4096)
        )
        if repeated != mixed_values[-1]:
            raise AssertionError("Repeated mixed prediction did not hit the same value")
        result["mixed_values_ms"] = mixed_values
        result["mixed_domain_policy"] = predictions["attn_prefill_mixed"].get(
            "_on_demand_domain_policy"
        )
        family_name = predictor._measurement_family_name(
            predictor._active_measurement_type
        )
        result["mixed_runtime_cache_entries"] = len(
            predictor._runtime_cache[family_name]["attn_prefill_mixed"]
        )
        if result["mixed_runtime_cache_entries"] != 3:
            raise AssertionError(
                "Expected three cached mixed runtime keys after the three positive probes"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()

    output: dict[str, object] = {"serving_max_tokens": 4096, "prediction_max_tokens": 4096}
    for tp in (1, 2, 4, 8):
        model_dir = args.train_root / f"tp{tp}"
        if not model_dir.is_dir():
            candidates = sorted(
                path
                for path in args.train_root.parent.glob(f"{args.train_root.name}_tp{tp}_*")
                if path.is_dir()
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected {model_dir} or one suffixed TP directory; found {candidates}"
                )
            model_dir = candidates[0]
        models = _load_models(model_dir)
        roles = {}
        for role in (
            ClusterType.MONOLITHIC,
            ClusterType.PREFILL,
            ClusterType.DECODE,
            ClusterType.DECODE_ATTN,
            ClusterType.DECODE_FFN,
        ):
            role_cache = args.cache_root / f"tp{tp}" / role.name.lower()
            roles[role.name] = _probe_role(models=models, role=role, cache_dir=role_cache)
        output[f"tp{tp}"] = roles

    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
