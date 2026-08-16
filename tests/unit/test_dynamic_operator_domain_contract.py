"""Regression tests for explicit domain contracts on dynamic operators."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.execution_time_predictor.prediction_cache_contract import (
    ON_DEMAND_DOMAIN_POLICY_BOUNDED,
    ON_DEMAND_DOMAIN_POLICY_UNBOUNDED,
    attach_feature_domain,
    build_on_demand_prediction_record,
    build_feature_domain_descriptor,
    validate_prediction_grid_domain,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.types import ClusterType, MeasurementType


class _DynamicPredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("not used")

    def _get_grid_search_params(self):
        raise AssertionError("not used")


class _CountingModel:
    def __init__(self, feature_names: list[str]) -> None:
        self._frontier_feature_names = list(feature_names)
        self.n_features_in_ = len(feature_names)
        self.calls = 0

    def predict(self, dataframe: pd.DataFrame):
        self.calls += 1
        return [1.0 for _ in range(len(dataframe))]


class _CountingMoEModel(_CountingModel):
    def __init__(self, feature_names: list[str]) -> None:
        super().__init__(feature_names)
        self._frontier_feature_names = list(feature_names)


class _ConcreteMoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("not used")

    def _get_grid_search_params(self):
        raise AssertionError("not used")


def _build_moe_runtime_predictor(
    model_name: str,
) -> tuple[_ConcreteMoEPredictor, _CountingMoEModel]:
    feature_names = list(SklearnMoEExecutionTimePredictor.MOE_LOAD_IMBALANCE_FEATURES)
    model = _CountingMoEModel(feature_names)
    attach_feature_domain(
        model,
        _training_frame(feature_names),
        feature_names,
        operator_name=model_name,
    )
    record = build_on_demand_prediction_record(
        model_name,
        model,
        feature_names,
    )
    predictor = _ConcreteMoEPredictor.__new__(_ConcreteMoEPredictor)
    predictor._predictions = {model_name: record}
    predictor._runtime_cache = {"eager": {model_name: {}}}
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._measurement_family_name = lambda _measurement_type: "eager"
    predictor._supports_operation = lambda _operation_name: True
    predictor._get_moe_compute_calibration_scale = lambda *_args, **_kwargs: 1.0
    predictor._model_config = SimpleNamespace(
        embedding_dim=4096,
        mlp_hidden_dim=14336,
    )
    predictor._router_topk = 2
    return predictor, model


def _training_frame(feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            feature_name: [1.0, 2.0, 3.0]
            for feature_name in feature_names
        }
    )


def test_integer_interval_domain_rejects_fractional_training_values() -> None:
    frame = pd.DataFrame(
        {
            "num_tokens": [1, 1.5, 2],
            "latency": [0.1, 0.15, 0.2],
        }
    )

    with pytest.raises(ValueError, match="integer interval.*integer"):
        build_feature_domain_descriptor(
            frame,
            ["num_tokens"],
            operator_name="linear_op",
        )


def test_integer_interval_domain_rejects_fractional_runtime_keys() -> None:
    feature_names = ["num_tokens"]
    model = _CountingModel(feature_names)
    attach_feature_domain(
        model,
        pd.DataFrame({"num_tokens": [1, 2]}),
        feature_names,
        operator_name="linear_op",
    )

    with pytest.raises(ValueError, match="not an integer"):
        validate_prediction_grid_domain(
            "linear_op",
            model,
            pd.DataFrame({"num_tokens": [1.5]}),
        )


@pytest.mark.parametrize(
    ("model_name", "feature_names", "expected_policy"),
    [
        (
            "attn_prefill_mixed",
            ["batch_size", "total_tokens"],
            ON_DEMAND_DOMAIN_POLICY_UNBOUNDED,
        ),
        (
            "attn_decode_in_mixed",
            ["decode_batch_size", "total_tokens"],
            ON_DEMAND_DOMAIN_POLICY_UNBOUNDED,
        ),
        (
            "moe_shuffling",
            ["total_routed_tokens", "load_imbalance_cv"],
            ON_DEMAND_DOMAIN_POLICY_UNBOUNDED,
        ),
        (
            "moe_grouped_gemm",
            ["total_routed_tokens", "load_imbalance_cv"],
            ON_DEMAND_DOMAIN_POLICY_UNBOUNDED,
        ),
        (
            "attn_kv_cache_save",
            ["total_tokens", "kv_cache_size", "batch_size"],
            ON_DEMAND_DOMAIN_POLICY_BOUNDED,
        ),
    ],
)
def test_domain_policy_is_operator_aware(
    model_name: str,
    feature_names: list[str],
    expected_policy: str,
) -> None:
    descriptor = build_feature_domain_descriptor(
        _training_frame(feature_names),
        feature_names,
        operator_name=model_name,
    )

    assert descriptor["on_demand_policy"] == expected_policy


def test_attention_dynamic_prediction_record_carries_domain_contract() -> None:
    feature_names = ["batch_size", "total_tokens"]
    model = _CountingModel(feature_names)
    attach_feature_domain(
        model,
        _training_frame(feature_names),
        feature_names,
        operator_name="attn_prefill_mixed",
    )

    predictor = _DynamicPredictor.__new__(_DynamicPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._models = {"attn_prefill_mixed": model}
    predictor._config = SimpleNamespace(
        prediction_max_batch_size=2,
        prediction_max_tokens_per_request=8,
        prediction_max_prefill_chunk_size=8,
        kv_cache_prediction_granularity=1,
    )
    predictor._model_config = SimpleNamespace()
    predictor._is_mla_attention_family = lambda: False
    predictor._dense_attention_decode_op_name = lambda: "attn_decode"
    predictor._dense_attention_prefill_op_name = lambda: "attn_prefill"
    predictor._get_model_prediction = lambda *_args, **_kwargs: {}

    predictions = predictor._predict_for_attention_layer_models()
    record = predictions["attn_prefill_mixed"]

    assert record["_feature_domain"] == model._frontier_feature_domain
    assert record["_on_demand_domain_policy"] == (
        ON_DEMAND_DOMAIN_POLICY_UNBOUNDED
    )


def test_unbounded_dynamic_model_accepts_runtime_tuple_outside_profile_bounds() -> None:
    feature_names = ["batch_size", "total_tokens"]
    model = _CountingModel(feature_names)
    attach_feature_domain(
        model,
        _training_frame(feature_names),
        feature_names,
        operator_name="moe_grouped_gemm",
    )

    predictor = _DynamicPredictor.__new__(_DynamicPredictor)
    predictor._active_measurement_type = None
    predictor._measurement_family_name = lambda _measurement_type: "eager"
    predictor._runtime_cache = {"eager": {"moe_grouped_gemm": {}}}
    predictor._predictions = {
        "moe_grouped_gemm": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": feature_names,
            "_feature_domain": model._frontier_feature_domain,
            "_on_demand_domain_policy": (
                model._frontier_feature_domain["on_demand_policy"]
            ),
        }
    }

    prediction = predictor._get_on_demand_prediction(
        "moe_grouped_gemm",
        {"batch_size": 64.0, "total_tokens": 128.0},
    )

    assert prediction == 1.0
    assert model.calls == 1


@pytest.mark.parametrize(
    "operator_name",
    [
        "attn_prefill_mixed",
        "attn_decode_in_mixed",
        "moe_shuffling",
        "moe_grouped_gemm",
    ],
)
def test_dynamic_operator_record_carries_domain_and_policy(
    operator_name: str,
) -> None:
    feature_names = ["feature_a", "feature_b"]
    model = _CountingModel(feature_names)
    attach_feature_domain(
        model,
        _training_frame(feature_names),
        feature_names,
        operator_name=operator_name,
    )

    record = build_on_demand_prediction_record(
        operator_name,
        model,
        feature_names,
    )

    assert record["_feature_domain"] == model._frontier_feature_domain
    assert record["_on_demand_domain_policy"] == (
        ON_DEMAND_DOMAIN_POLICY_UNBOUNDED
    )


def test_on_demand_record_rejects_missing_persisted_policy() -> None:
    feature_names = ["feature_a", "feature_b"]
    model = _CountingModel(feature_names)
    attach_feature_domain(
        model,
        _training_frame(feature_names),
        feature_names,
        operator_name="attn_prefill_mixed",
    )
    model._frontier_feature_domain.pop("on_demand_policy")

    with pytest.raises(ValueError, match="on_demand_policy"):
        build_on_demand_prediction_record(
            "attn_prefill_mixed",
            model,
            feature_names,
        )


def test_on_demand_runtime_rejects_misbound_operator_domain() -> None:
    feature_names = ["total_routed_tokens", "load_imbalance_cv"]
    model = _CountingModel(feature_names)
    attach_feature_domain(
        model,
        _training_frame(feature_names),
        feature_names,
        operator_name="moe_grouped_gemm",
    )
    record = build_on_demand_prediction_record(
        "moe_grouped_gemm",
        model,
        feature_names,
    )
    record["_feature_domain"]["operator_name"] = "moe_shuffling"
    predictor = _DynamicPredictor.__new__(_DynamicPredictor)
    predictor._predictions = {"moe_grouped_gemm": record}
    predictor._runtime_cache = {"eager": {"moe_grouped_gemm": {}}}
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._measurement_family_name = lambda _measurement_type: "eager"

    with pytest.raises(ValueError, match="operator mismatch|operator_name|operator binding"):
        predictor._get_on_demand_prediction(
            "moe_grouped_gemm",
            {"total_routed_tokens": 4.0, "load_imbalance_cv": 0.5},
        )


@pytest.mark.parametrize("model_name", ["moe_shuffling", "moe_grouped_gemm"])
def test_moe_runtime_uses_shared_on_demand_record_and_reuses_prediction(
    model_name: str,
) -> None:
    predictor, model = _build_moe_runtime_predictor(model_name)
    allocation = {0: 4, 1: 2, 2: 0, 3: 0}

    if model_name == "moe_shuffling":
        first = predictor._get_moe_shuffling_time(
            None,
            moe_tokens_input=allocation,
        )
        second = predictor._get_moe_shuffling_time(
            None,
            moe_tokens_input=allocation,
        )
    else:
        first = predictor._get_grouped_gemm_time(allocation, batch=None)
        second = predictor._get_grouped_gemm_time(allocation, batch=None)

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(1.0)
    assert model.calls == 1


@pytest.mark.parametrize("model_name", ["moe_shuffling", "moe_grouped_gemm"])
def test_moe_equivalent_runtime_hits_use_canonical_cache_and_diagnostics(
    model_name: str,
) -> None:
    """Equivalent expert allocations must not bypass the canonical runtime cache."""

    predictor, model = _build_moe_runtime_predictor(model_name)
    predictor._config = SimpleNamespace(enable_prediction_domain_diagnostics=True)
    first_allocation = {0: 4, 1: 2, 2: 0, 3: 0}
    reordered_allocation = {3: 0, 2: 0, 1: 2, 0: 4}

    if model_name == "moe_shuffling":
        first = predictor._get_moe_shuffling_time(
            None,
            moe_tokens_input=first_allocation,
        )
        second = predictor._get_moe_shuffling_time(
            None,
            moe_tokens_input=reordered_allocation,
        )
    else:
        first = predictor._get_grouped_gemm_time(first_allocation, batch=None)
        second = predictor._get_grouped_gemm_time(reordered_allocation, batch=None)

    assert first == pytest.approx(1.0)
    assert second == pytest.approx(1.0)
    assert model.calls == 1
    diagnostics = predictor.get_prediction_domain_diagnostics()
    records = diagnostics["eager"][model_name]
    assert sum(int(item["count"]) for item in records.values()) == 2
    assert not hasattr(
        predictor,
        "_runtime_moe_shuffling_on_demand_prediction_cache",
    )
    assert not hasattr(
        predictor,
        "_runtime_grouped_gemm_on_demand_prediction_cache",
    )
