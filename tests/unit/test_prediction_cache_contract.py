"""Regression tests for prediction-grid and lookup-cache contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import frontier.execution_time_predictor.prediction_cache_contract as prediction_cache_contract
from frontier.types import ClusterType, MeasurementType
from frontier.execution_time_predictor.prediction_cache_contract import (
    DOMAIN_KIND_CONDITIONAL,
    DOMAIN_KIND_EXACT_ROWS,
    DOMAIN_KIND_VERIFIED_CARTESIAN,
    ON_DEMAND_DOMAIN_POLICY_BOUNDED,
    ON_DEMAND_DOMAIN_POLICY_UNBOUNDED,
    PREDICTION_CACHE_CONTRACT_VERSION,
    attach_feature_domain,
    build_feature_domain_descriptor,
    filter_prediction_grid_to_domain,
    validate_prediction_cache,
    validate_prediction_grid_domain,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("not used")

    def _get_grid_search_params(self):
        raise AssertionError("not used")


class _CountingModel:
    n_features_in_ = 1

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, features: pd.DataFrame):
        self.calls += 1
        return [float(value) for value in features.iloc[:, 0]]


def _domain(*, feature_names, domain_kind="integer_interval_interpolation", **extra):
    descriptor = {
        "contract_version": PREDICTION_CACHE_CONTRACT_VERSION,
        "feature_names": list(feature_names),
        "domain_kind": domain_kind,
        **extra,
    }
    return descriptor


def _cache_probe() -> _ConcretePredictor:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(no_cache=False)
    predictor._cache_dir = "/unused"
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._get_model_hash = lambda *_args, **_kwargs: "config-hash"
    predictor._get_prediction_context_hash = lambda *_args, **_kwargs: "config-hash"
    return predictor


def test_prediction_cache_hash_binds_exact_requested_grid() -> None:
    predictor = _cache_probe()
    model = _CountingModel()
    model._frontier_model_hash = "model-hash"

    first = predictor._get_prediction_cache_hash(
        "attn_pre_proj", model, feature_names=["num_tokens"], requested_keys=[(1,)]
    )
    second = predictor._get_prediction_cache_hash(
        "attn_pre_proj", model, feature_names=["num_tokens"], requested_keys=[(1,), (2,)]
    )

    assert first != second


def test_model_prediction_reuses_one_canonical_grid(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _cache_probe()
    predictor._cache_dir = str(tmp_path)
    model = _CountingModel()
    model._frontier_model_hash = "model-hash"
    predictor._models = {"attn_pre_proj": model}
    model._frontier_feature_domain = _domain(
        feature_names=["num_tokens"],
        bounds={"num_tokens": {"min": 1.0, "max": 4.0}},
    )
    monkeypatch.setattr(
        predictor,
        "_load_model_predication_cache",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        predictor,
        "_store_model_predication_cache",
        lambda *_args: None,
    )

    original = prediction_cache_contract.canonicalize_prediction_grid
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        prediction_cache_contract,
        "canonicalize_prediction_grid",
        counted,
    )

    predictions = predictor._get_model_prediction(
        "attn_pre_proj",
        model,
        pd.DataFrame({"num_tokens": [1, 2, 3, 4]}),
    )

    assert predictions == {(1,): 1.0, (2,): 2.0, (3,): 3.0, (4,): 4.0}
    assert model.calls == 1
    assert calls == 1


def test_incomplete_persisted_prediction_cache_fails_before_runtime_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _cache_probe()
    model = _CountingModel()
    model._frontier_model_hash = "model-hash"
    predictor._models = {"attn_pre_proj": model}
    model._frontier_feature_domain = _domain(
        feature_names=["num_tokens"],
        bounds={"num_tokens": {"min": 1.0, "max": 4.0}},
    )
    monkeypatch.setattr(
        predictor,
        "_load_model_predication_cache",
        lambda *_args: {(1,): 1.0},
    )

    with pytest.raises(ValueError, match="attn_pre_proj has incomplete prediction cache"):
        predictor._get_model_prediction(
            "attn_pre_proj",
            model,
            pd.DataFrame({"num_tokens": [1, 2]}),
        )

    assert model.calls == 0


def test_prediction_grid_outside_declared_profile_domain_fails_before_predict() -> None:
    predictor = _cache_probe()
    model = _CountingModel()
    predictor._models = {"attn_pre_proj": model}
    model._frontier_feature_domain = _domain(
        feature_names=["num_tokens"],
        bounds={"num_tokens": {"min": 1.0, "max": 4.0}},
        runtime_prediction_policy="measured_only",
    )

    with pytest.raises(ValueError, match="attn_pre_proj.*violates.*domain"):
        predictor._get_model_prediction(
            "attn_pre_proj",
            model,
            pd.DataFrame({"num_tokens": [1, 8]}),
        )

    assert model.calls == 0


def test_lookup_miss_calls_the_canonical_model_for_a_legal_gap() -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    model = _CountingModel()
    model._frontier_feature_names = ["num_tokens"]
    attach_feature_domain(
        model,
        pd.DataFrame({"num_tokens": [1, 16, 32]}),
        ["num_tokens"],
        operator_name="attn_pre_proj",
    )
    predictor._models = {"attn_pre_proj": model}
    predictor._predictions = {"attn_pre_proj": {(16,): 1.0}}
    predictor._runtime_cache = {"eager": {"attn_pre_proj": {}}}

    assert predictor._get_lookup_or_predict("attn_pre_proj", (64,), ["num_tokens"]) == 64.0
    assert model.calls == 1


def test_one_dimensional_sparse_samples_preserve_interval_materialization() -> None:
    model = _CountingModel()
    training = pd.DataFrame({"num_tokens": [1, 2, 4, 8]})
    attach_feature_domain(model, training, ["num_tokens"])

    assert model._frontier_feature_domain["domain_kind"] == (
        "integer_interval_interpolation"
    )
    assert model._frontier_feature_domain["on_demand_policy"] == (
        ON_DEMAND_DOMAIN_POLICY_BOUNDED
    )


def test_prediction_domain_canonicalizes_boolean_shape_features() -> None:
    model = _CountingModel()
    attach_feature_domain(
        model,
        pd.DataFrame({"is_decode": [False, True], "tokens": [1, 2]}),
        ["is_decode", "tokens"],
        domain_kind="exact_rows",
    )
    validate_prediction_grid_domain(
        "mla",
        model,
        pd.DataFrame({"is_decode": [True], "tokens": [2]}),
    )


def test_feature_domain_can_explicitly_declare_unbounded_on_demand_policy() -> None:
    model = _CountingModel()
    attach_feature_domain(
        model,
        pd.DataFrame({"x": [1, 2, 4]}),
        ["x"],
        on_demand_policy=ON_DEMAND_DOMAIN_POLICY_UNBOUNDED,
    )
    assert model._frontier_feature_domain["on_demand_policy"] == (
        ON_DEMAND_DOMAIN_POLICY_UNBOUNDED
    )


def test_one_dimensional_interval_rejects_below_and_above_profile_bounds() -> None:
    model = _CountingModel()
    attach_feature_domain(
        model,
        pd.DataFrame({"num_tokens": [1, 2, 4, 8]}),
        ["num_tokens"],
    )

    with pytest.raises(ValueError, match="physical minimum"):
        validate_prediction_grid_domain(
            "linear", model, pd.DataFrame({"num_tokens": [0]})
        )
    validate_prediction_grid_domain(
        "linear", model, pd.DataFrame({"num_tokens": [3, 5, 8, 9]})
    )


def test_sparse_two_dimensional_domain_rejects_unmeasured_tuple() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {"batch_size": [1, 2], "kv_cache_size": [10, 20]}
    )
    attach_feature_domain(model, training, ["batch_size", "kv_cache_size"])

    descriptor = model._frontier_feature_domain
    assert descriptor["domain_kind"] == "regression_extrapolation"
    validate_prediction_grid_domain(
        "attn_prefill", model, pd.DataFrame({"batch_size": [1], "kv_cache_size": [20]})
    )


def test_verified_cartesian_domain_accepts_all_declared_combinations() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "batch_size": [1, 1, 2, 2],
            "kv_cache_size": [10, 20, 10, 20],
        }
    )
    attach_feature_domain(model, training, ["batch_size", "kv_cache_size"])

    assert model._frontier_feature_domain["domain_kind"] == (
        "verified_cartesian_interpolation"
    )
    validate_prediction_grid_domain(
        "attn_decode",
        model,
        pd.DataFrame(
            {"batch_size": [1, 2], "kv_cache_size": [20, 10]}
        ),
    )


def test_verified_cartesian_domain_interpolates_in_bounds_integer_axes() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "batch_size": [1, 1, 8, 8],
            "kv_cache_size": [0, 128, 0, 128],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["batch_size", "kv_cache_size"],
        operator_name="attn_decode",
    )

    descriptor = model._frontier_feature_domain
    assert descriptor["axis_semantics"] == {
        "batch_size": "integer_interval",
        "kv_cache_size": "integer_interval",
    }
    validate_prediction_grid_domain(
        "attn_decode",
        model,
        pd.DataFrame({"batch_size": [4], "kv_cache_size": [64]}),
    )


def test_verified_cartesian_domain_keeps_boolean_axes_enumerated() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "is_decode": [False, False, True, True],
            "tokens": [1, 8, 1, 8],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["is_decode", "tokens"],
        axis_semantics={
            "is_decode": "enumerated",
            "tokens": "integer_interval",
        },
    )

    descriptor = model._frontier_feature_domain
    assert descriptor["axis_semantics"] == {
        "is_decode": "enumerated",
        "tokens": "integer_interval",
    }
    validate_prediction_grid_domain(
        "probe",
        model,
        pd.DataFrame({"is_decode": [True], "tokens": [4]}),
    )
    with pytest.raises(ValueError, match="enumerated|axis_values|profile domain"):
        validate_prediction_grid_domain(
            "probe",
            model,
            pd.DataFrame({"is_decode": [2], "tokens": [4]}),
        )


def test_verified_cartesian_domain_rejects_in_bounds_categorical_value() -> None:
    """Enumerated axes must not degrade to independent min/max bounds."""
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "kernel_variant": [10, 10, 20, 20],
            "batch_size": [1, 2, 1, 2],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["kernel_variant", "batch_size"],
        axis_semantics={
            "kernel_variant": "enumerated",
            "batch_size": "integer_interval",
        },
    )

    # 15 is inside the [10, 20] extrema, but it was never declared on the
    # categorical axis. A bounds-only validator would incorrectly accept it.
    with pytest.raises(ValueError, match="enumerated|axis_values|profile domain"):
        validate_prediction_grid_domain(
            "probe",
            model,
            pd.DataFrame({"kernel_variant": [15], "batch_size": [1]}),
        )


def test_verified_cartesian_integer_axis_rejects_outside_bounds() -> None:
    """Integer interval axes still enforce their declared extrema."""
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "batch_size": [1, 1, 8, 8],
            "kv_cache_size": [0, 128, 0, 128],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["batch_size", "kv_cache_size"],
        operator_name="attn_decode",
    )

    validate_prediction_grid_domain(
        "attn_decode",
        model,
        pd.DataFrame({"batch_size": [9], "kv_cache_size": [64]}),
    )


def test_verified_cartesian_integer_axis_rejects_fractional_declared_bounds() -> None:
    """A persisted integer interval cannot carry fractional extrema."""
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "batch_size": [1, 1, 8, 8],
            "kv_cache_size": [0, 128, 0, 128],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["batch_size", "kv_cache_size"],
        operator_name="attn_decode",
    )
    # Keep all declared axis values inside the malformed bounds so the
    # per-axis semantic validator, rather than the generic axis-bound check,
    # owns this failure.
    model._frontier_feature_domain["bounds"]["batch_size"] = {
        "min": 0.5,
        "max": 8.5,
    }

    with pytest.raises(ValueError, match="integer.*bounds"):
        validate_prediction_grid_domain(
            "attn_decode",
            model,
            pd.DataFrame({"batch_size": [4], "kv_cache_size": [64]}),
        )


def test_verified_cartesian_builder_rejects_incomplete_observed_product() -> None:
    training = pd.DataFrame(
        {
            "batch_size": [1, 1, 2],
            "kv_cache_size": [10, 20, 10],
        }
    )

    with pytest.raises(ValueError, match="complete Cartesian|missing.*combination"):
        build_feature_domain_descriptor(
            training,
            ["batch_size", "kv_cache_size"],
            domain_kind=DOMAIN_KIND_VERIFIED_CARTESIAN,
        )


def test_prefill_domain_enforces_relational_constraint_not_only_axis_bounds() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "kv_cache_size": [0, 0, 64, 0],
            "prefill_chunk_size_squared": [32**2, 64**2, 64**2, 96**2],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["kv_cache_size", "prefill_chunk_size_squared"],
    )

    assert model._frontier_feature_domain["domain_kind"] == (
        "conditional_interpolation"
    )
    validate_prediction_grid_domain(
        "attn_prefill",
        model,
        pd.DataFrame(
            {"kv_cache_size": [32], "prefill_chunk_size_squared": [64**2]}
        ),
    )
    with pytest.raises(ValueError, match="conditional constraint"):
        validate_prediction_grid_domain(
            "attn_prefill",
            model,
            pd.DataFrame(
                {"kv_cache_size": [64], "prefill_chunk_size_squared": [96**2]}
            ),
            runtime_constraints=[
                {
                    "type": "sum_lte",
                    "features": ["kv_cache_size", "prefill_chunk_size"],
                    "derived_features": {
                        "prefill_chunk_size": {"sqrt": "prefill_chunk_size_squared"}
                    },
                    "max": 128,
                    "enforce_during_extrapolation": True,
                }
            ],
        )


def test_prefill_grid_filter_keeps_only_conditional_domain_tuples() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "kv_cache_size": [0, 0, 64, 0],
            "prefill_chunk_size_squared": [32**2, 64**2, 64**2, 96**2],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["kv_cache_size", "prefill_chunk_size_squared"],
    )
    requested = pd.DataFrame(
        {
            "kv_cache_size": [0, 32, 64, 64],
            "prefill_chunk_size_squared": [96**2, 64**2, 64**2, 96**2],
        }
    )

    filtered = filter_prediction_grid_to_domain(
        "attn_prefill",
        model,
        requested,
        measurement_family="eager",
        runtime_constraints=[
            {
                "type": "sum_lte",
                "features": ["kv_cache_size", "prefill_chunk_size"],
                "derived_features": {
                    "prefill_chunk_size": {"sqrt": "prefill_chunk_size_squared"}
                },
                "max": 128,
                "enforce_during_extrapolation": True,
            }
        ],
    )

    assert list(filtered.itertuples(index=False, name=None)) == [
        (0, 96**2),
        (32, 64**2),
        (64, 64**2),
    ]


def test_kv_cache_save_uses_operator_specific_measured_row_domain() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "total_tokens": [1, 2, 4, 32, 18, 34],
            "kv_cache_size": [1, 1, 1, 0, 64, 64],
            "batch_size": [1, 2, 4, 1, 4, 4],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["total_tokens", "kv_cache_size", "batch_size"],
        operator_name="attn_kv_cache_save",
    )

    descriptor = model._frontier_feature_domain
    assert descriptor["domain_kind"] == "regression_extrapolation"
    assert descriptor["training_keys"]
    validate_prediction_grid_domain(
        "attn_kv_cache_save",
        model,
        pd.DataFrame(
            {
                "total_tokens": [18],
                "kv_cache_size": [64],
                "batch_size": [4],
            }
        ),
    )
    validate_prediction_grid_domain(
        "attn_kv_cache_save",
        model,
        pd.DataFrame(
            {
                "total_tokens": [10],
                "kv_cache_size": [32],
                "batch_size": [3],
            }
        ),
    )


def test_kv_cache_save_domain_declares_integer_axes_and_rejects_fractional_key() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "total_tokens": [1, 2, 9, 10],
            "kv_cache_size": [0, 8, 16, 32],
            "batch_size": [1, 2, 2, 3],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["total_tokens", "kv_cache_size", "batch_size"],
        operator_name="attn_kv_cache_save",
    )

    assert model._frontier_feature_domain["domain_kind"] == "regression_extrapolation"
    with pytest.raises(ValueError, match="integer"):
        validate_prediction_grid_domain(
            "attn_kv_cache_save",
            model,
            pd.DataFrame(
                {
                    "total_tokens": [9],
                    "kv_cache_size": [16.5],
                    "batch_size": [2],
                }
            ),
        )
    assert model.calls == 0


def test_kv_cache_save_domain_rejects_unmeasured_integer_tuple() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "total_tokens": [1, 2, 9, 10],
            "kv_cache_size": [0, 8, 16, 32],
            "batch_size": [1, 2, 2, 3],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["total_tokens", "kv_cache_size", "batch_size"],
        operator_name="attn_kv_cache_save",
    )

    validate_prediction_grid_domain(
        "attn_kv_cache_save",
        model,
        pd.DataFrame(
            {
                "total_tokens": [3],
                "kv_cache_size": [2],
                "batch_size": [1],
            }
        ),
    )


def test_prediction_domain_contract_version_two_rejects_legacy_cartesian_descriptor() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "batch_size": [1, 1, 2, 2],
            "kv_cache_size": [0, 8, 0, 8],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["batch_size", "kv_cache_size"],
        operator_name="attn_decode",
    )
    model._frontier_feature_domain["contract_version"] = 1

    with pytest.raises(ValueError, match="unsupported|contract version"):
        validate_prediction_grid_domain(
            "attn_decode",
            model,
            pd.DataFrame({"batch_size": [1], "kv_cache_size": [4]}),
        )


def test_conditional_filter_does_not_hide_axis_bound_mismatch() -> None:
    model = _CountingModel()
    attach_feature_domain(
        model,
        pd.DataFrame(
            {
                "kv_cache_size": [0, 64],
                "prefill_chunk_size_squared": [1, 96**2],
            }
        ),
        ["kv_cache_size", "prefill_chunk_size_squared"],
    )

    with pytest.raises(ValueError, match="physical maximum"):
        filter_prediction_grid_to_domain(
            "attn_prefill",
            model,
            pd.DataFrame(
                {
                    "kv_cache_size": [0, 128],
                    "prefill_chunk_size_squared": [1, 96**2],
                }
            ),
            runtime_physical_bounds={
                "kv_cache_size": {"min": 0.0, "max": 64.0},
                "prefill_chunk_size_squared": {"min": 1.0, "max": 96.0**2},
            },
        )


def test_prediction_grid_filter_fails_if_domain_rejects_every_tuple() -> None:
    model = _CountingModel()
    training = pd.DataFrame(
        {
            "kv_cache_size": [0, 0, 64],
            "prefill_chunk_size_squared": [64**2, 128**2, 64**2],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["kv_cache_size", "prefill_chunk_size_squared"],
    )

    with pytest.raises(ValueError, match="no valid prediction-grid tuples"):
        filter_prediction_grid_to_domain(
                "attn_prefill",
                model,
                pd.DataFrame(
                    {"kv_cache_size": [64], "prefill_chunk_size_squared": [128**2]}
            ),
            runtime_constraints=[
                {
                    "type": "sum_lte",
                    "features": ["kv_cache_size", "prefill_chunk_size"],
                    "derived_features": {
                        "prefill_chunk_size": {"sqrt": "prefill_chunk_size_squared"}
                    },
                    "max": 128,
                    "enforce_during_extrapolation": True,
                }
            ],
        )


def test_standard_prefill_materialization_filters_conditional_grid(monkeypatch) -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._config = SimpleNamespace(
        prediction_max_batch_size=1,
        prediction_max_tokens_per_request=128,
        prediction_max_prefill_chunk_size=96,
        kv_cache_prediction_granularity=64,
    )
    predictor._model_config = SimpleNamespace(
        max_model_len=128,
        max_position_embeddings=128,
        max_seq_len=128,
    )
    decode_model = _CountingModel()
    prefill_model = _CountingModel()
    attach_feature_domain(
        prefill_model,
        pd.DataFrame(
                {
                    "kv_cache_size": [0, 0, 64, 0, 128],
                    "prefill_chunk_size_squared": [1, 64**2, 64**2, 96**2, 1],
                }
        ),
        ["kv_cache_size", "prefill_chunk_size_squared"],
    )
    predictor._models = {
        "attn_decode": decode_model,
        "attn_prefill": prefill_model,
    }
    predictor._is_mla_attention_family = lambda: False
    predictor._dense_attention_decode_op_name = lambda: "attn_decode"
    predictor._dense_attention_prefill_op_name = lambda: "attn_prefill"
    observed: dict[str, pd.DataFrame] = {}

    def _capture(model_name, _model, frame):
        observed[model_name] = frame.copy()
        return {}

    predictor._get_model_prediction = _capture

    predictor._predict_for_attention_layer_models()

    prefill_keys = set(
        observed["attn_prefill"].itertuples(index=False, name=None)
    )
    assert (64, 96**2) not in prefill_keys
    assert (64, 64**2) in prefill_keys
    assert prefill_keys


@pytest.mark.parametrize(
    "descriptor",
    [
        {"feature_names": ["num_tokens"], "domain_kind": "integer_interval_interpolation"},
        {
            "contract_version": PREDICTION_CACHE_CONTRACT_VERSION + 1,
            "feature_names": ["num_tokens"],
            "domain_kind": "integer_interval_interpolation",
        },
    ],
)
def test_domain_contract_version_is_required_and_exact(descriptor) -> None:
    model = _CountingModel()
    model._frontier_feature_domain = descriptor
    with pytest.raises(ValueError, match="contract version"):
        validate_prediction_grid_domain(
            "linear", model, pd.DataFrame({"num_tokens": [1]})
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.01])
def test_persisted_prediction_cache_rejects_invalid_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="invalid values"):
        validate_prediction_cache(
            "linear",
            {(1,): bad_value},
            [(1,)],
            ["num_tokens"],
        )


def test_model_prediction_rejects_noncanonical_estimator_before_predict() -> None:
    predictor = _cache_probe()
    canonical_model = _CountingModel()
    supplied_model = _CountingModel()
    predictor._models = {"linear": canonical_model}

    with pytest.raises(ValueError, match="estimator identity mismatch"):
        predictor._get_model_prediction(
            "linear",
            supplied_model,
            pd.DataFrame({"num_tokens": [1]}),
        )

    assert canonical_model.calls == 0
    assert supplied_model.calls == 0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
def test_on_demand_rejects_non_finite_or_negative_output_without_cache(
    bad_value: float,
) -> None:
    predictor = _cache_probe()
    predictor._runtime_cache = {"eager": {"mixed": {}}}
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    model = _CountingModel()
    model.predict = lambda _features: [bad_value]
    predictor._predictions = {
        "mixed": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": ["x"],
        }
    }

    with pytest.raises(ValueError, match="finite|non-negative|invalid"):
        predictor._get_on_demand_prediction("mixed", {"x": 1.0})
    assert predictor._runtime_cache["eager"]["mixed"] == {}


def test_on_demand_validates_schema_before_runtime_cache_key() -> None:
    predictor = _cache_probe()
    # This is the cache-key collision produced by the old sorted-input-key
    # implementation: valid {x: 1, y: 2} and invalid {x: 1, z: 2} both became
    # (1, 2) before schema validation.
    predictor._runtime_cache = {"eager": {"mixed": {(1, 2): 99.0}}}
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    model = _CountingModel()
    predictor._predictions = {
        "mixed": {
            "_on_demand_prediction": True,
            "_model": model,
            "_feature_names": ["x", "y"],
        }
    }

    with pytest.raises(ValueError, match="missing required features"):
        predictor._get_on_demand_prediction("mixed", {"x": 1.0, "z": 2.0})
    assert predictor._runtime_cache["eager"]["mixed"] == {(1, 2): 99.0}


def test_moe_finite_lookup_does_not_clamp_above_cache(monkeypatch) -> None:
    from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
        SklearnMoEExecutionTimePredictor,
    )

    class _ConcreteMoEPredictor(SklearnMoEExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    predictor = _ConcreteMoEPredictor.__new__(_ConcreteMoEPredictor)
    predictor._predictions = {"moe_grouped_gemm": {(8,): 0.5}}
    predictor._router_topk = 1
    predictor._max_tokens = 8
    predictor._supports_operation = lambda _name: True
    predictor._get_moe_compute_calibration_scale = lambda *args, **kwargs: 1.0

    with pytest.raises(
        ValueError,
        match="no canonical model|outside.*cache|coverage gap|physical",
    ):
        predictor._get_grouped_gemm_time(16)


@pytest.mark.parametrize("allocation", [{0: -1, 1: 1}, {0: -2, 1: 1}, {0: 1.5, 1: 2}])
def test_moe_standard_grouped_gemm_rejects_invalid_expert_allocations(allocation) -> None:
    from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
        SklearnMoEExecutionTimePredictor,
    )

    class _ConcreteMoEPredictor(SklearnMoEExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    predictor = _ConcreteMoEPredictor.__new__(_ConcreteMoEPredictor)
    predictor._predictions = {"moe_grouped_gemm": {(8,): 0.5}}
    predictor._router_topk = 1
    predictor._supports_operation = lambda _name: True
    predictor._get_moe_compute_calibration_scale = lambda *args, **kwargs: 1.0

    with pytest.raises(ValueError, match="per-expert token allocation"):
        predictor._get_grouped_gemm_time(allocation)
