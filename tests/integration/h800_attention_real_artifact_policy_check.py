"""Check real H800 attention artifacts against strict runtime boundaries.

This check loads one persisted cache-write estimator and exercises the same
model-cache and on-demand prediction validators used by Frontier.  It never
rewrites the artifact or substitutes a nearby measured row.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import joblib

from frontier.execution_time_predictor.model_cache_contract import (
    validate_cached_model,
)
from frontier.execution_time_predictor.prediction_cache_contract import (
    build_on_demand_prediction_record,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.types import ClusterType, MeasurementType


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):  # pragma: no cover - runtime-only check
        raise AssertionError("the artifact check must not train")

    def _get_grid_search_params(self):  # pragma: no cover
        raise AssertionError("the artifact check must not train")


def _load_artifact(root: Path, operator: str):
    matches = sorted(root.glob(f"{operator}_*.pkl"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {operator} artifact in {root}, got {matches}"
        )
    return matches[0], joblib.load(matches[0])


def _runtime_predictor(model, *, diagnostics: bool):
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(
        enable_prediction_domain_diagnostics=diagnostics,
        prediction_max_tokens_per_request=4096,
        prediction_max_batch_size=10,
        prediction_max_prefill_chunk_size=4096,
        kv_cache_prediction_granularity=64,
        prediction_min_kv_cache_size=0,
    )
    predictor._model_config = ModelConfig.from_model_name("Qwen3-30B-A3B-tiny")
    predictor._max_tokens = 4096
    predictor._serving_max_tokens_per_request = 4096
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._predictions = {
        "attn_kv_cache_save": build_on_demand_prediction_record(
            "attn_kv_cache_save",
            model,
            model._frontier_feature_names,
        )
    }
    return predictor


def _with_predict_counter(model):
    calls = {"count": 0}
    original_predict = model.predict

    def counted_predict(frame):
        calls["count"] += 1
        return original_predict(frame)

    model.predict = counted_predict
    return calls


def _expect_cache_validation_failure(
    base_model,
    *,
    label: str,
    mutate_model: Callable[[Any], None] | None = None,
    model_name: str = "attn_kv_cache_save",
    expected_binding: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    model = copy.deepcopy(base_model)
    if mutate_model is not None:
        mutate_model(model)
    calls = _with_predict_counter(model)
    try:
        validate_cached_model(
            model_name,
            model,
            expected_model_hash=base_model._frontier_model_hash,
            feature_names=(
                list(base_model._frontier_feature_names)
                if feature_names is None
                else feature_names
            ),
            target_col=base_model._frontier_target_col,
            operator_binding=(
                copy.deepcopy(base_model._frontier_operator_binding)
                if expected_binding is None
                else expected_binding
            ),
        )
    except ValueError as exc:
        if calls["count"] != 0:
            raise AssertionError(
                f"{label} called the estimator before failing validation"
            ) from exc
        return {"error": str(exc), "model_calls": calls["count"]}
    raise AssertionError(f"{label} was accepted by the model-cache validator")


def _changed_binding(base_model, mutator: Callable[[dict[str, Any]], None]):
    binding = copy.deepcopy(base_model._frontier_operator_binding)
    mutator(binding)
    return binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()

    artifact_path, model = _load_artifact(
        args.artifact_root, "attn_kv_cache_save"
    )
    validate_cached_model(
        "attn_kv_cache_save",
        model,
        expected_model_hash=model._frontier_model_hash,
        feature_names=model._frontier_feature_names,
        target_col=model._frontier_target_col,
        operator_binding=model._frontier_operator_binding,
    )

    failures = {
        "wrong_model": _expect_cache_validation_failure(
            model,
            label="wrong model",
            expected_binding=_changed_binding(
                model,
                lambda binding: binding.__setitem__("model_name", "OtherModel"),
            ),
        ),
        "wrong_device": _expect_cache_validation_failure(
            model,
            label="wrong device",
            expected_binding=_changed_binding(
                model, lambda binding: binding.__setitem__("device", "b300")
            ),
        ),
        "wrong_tp": _expect_cache_validation_failure(
            model,
            label="wrong TP",
            expected_binding=_changed_binding(
                model,
                lambda binding: binding["profile_structure"].__setitem__(
                    "num_tensor_parallel_workers",
                    int(
                        binding["profile_structure"][
                            "num_tensor_parallel_workers"
                        ]
                    )
                    + 1,
                ),
            ),
        ),
        "wrong_ep": _expect_cache_validation_failure(
            model,
            label="wrong EP",
            expected_binding=_changed_binding(
                model,
                lambda binding: binding["profile_structure"].__setitem__(
                    "expert_parallel_size", 2
                ),
            ),
        ),
        "wrong_measurement_family": _expect_cache_validation_failure(
            model,
            label="wrong measurement family",
            expected_binding=_changed_binding(
                model,
                lambda binding: binding["profile_structure"].__setitem__(
                    "measurement_type", "RECORD_FUNCTION"
                ),
            ),
        ),
        "wrong_operator_family": _expect_cache_validation_failure(
            model,
            label="wrong operator family",
            expected_binding=_changed_binding(
                model,
                lambda binding: binding.__setitem__("operator_family", "moe"),
            ),
        ),
        "wrong_operator": _expect_cache_validation_failure(
            model,
            label="wrong operator",
            model_name="attn_decode",
        ),
        "wrong_schema": _expect_cache_validation_failure(
            model,
            label="wrong schema",
            feature_names=["total_tokens", "batch_size", "kv_cache_size"],
        ),
        "stale_v2_domain": _expect_cache_validation_failure(
            model,
            label="stale v2 domain",
            mutate_model=lambda candidate: candidate._frontier_feature_domain.__setitem__(
                "contract_version", 2
            ),
        ),
    }

    diagnostics_model = copy.deepcopy(model)
    diagnostic_calls = _with_predict_counter(diagnostics_model)
    diagnostics_predictor = _runtime_predictor(
        diagnostics_model, diagnostics=False
    )
    diagnostics_value = diagnostics_predictor._get_on_demand_prediction(
        "attn_kv_cache_save",
        {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1},
    )
    diagnostics_snapshot = diagnostics_predictor.get_prediction_domain_diagnostics()
    if diagnostics_snapshot or hasattr(
        diagnostics_predictor, "_prediction_domain_diagnostics"
    ):
        raise AssertionError("diagnostics-off unexpectedly accumulated runtime records")
    if diagnostic_calls["count"] != 1:
        raise AssertionError("diagnostics-off gap did not call the canonical model once")

    invalid_model = copy.deepcopy(model)
    invalid_calls = {"count": 0}

    def non_finite_predict(_frame):
        invalid_calls["count"] += 1
        return [float("nan")]

    invalid_model.predict = non_finite_predict
    invalid_predictor = _runtime_predictor(invalid_model, diagnostics=False)
    try:
        invalid_predictor._get_on_demand_prediction(
            "attn_kv_cache_save",
            {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1},
        )
    except ValueError as exc:
        non_finite_error = str(exc)
    else:
        raise AssertionError("non-finite real-artifact model output was accepted")
    invalid_cache = invalid_predictor._runtime_cache["eager"][
        "attn_kv_cache_save"
    ]
    if invalid_calls["count"] != 1 or invalid_cache:
        raise AssertionError(
            "non-finite output must fail after one model call and before cache write"
        )

    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "valid_identity": {
                    "model_name": model._frontier_model_name,
                    "model_hash": model._frontier_model_hash,
                    "feature_names": model._frontier_feature_names,
                    "device": model._frontier_operator_binding["device"],
                    "tp": model._frontier_operator_binding["profile_structure"][
                        "num_tensor_parallel_workers"
                    ],
                    "measurement_type": model._frontier_operator_binding[
                        "profile_structure"
                    ]["measurement_type"],
                    "domain_contract_version": model._frontier_feature_domain[
                        "contract_version"
                    ],
                },
                "fail_fast_cases": failures,
                "diagnostics_off": {
                    "value_ms": float(diagnostics_value),
                    "model_calls": diagnostic_calls["count"],
                    "aggregate_records": diagnostics_snapshot,
                    "state_created": hasattr(
                        diagnostics_predictor, "_prediction_domain_diagnostics"
                    ),
                },
                "non_finite_output": {
                    "error": non_finite_error,
                    "model_calls": invalid_calls["count"],
                    "runtime_cache_entries": len(invalid_cache),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
