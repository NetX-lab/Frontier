"""Tests for the explicit ML prediction policy on legal unmeasured inputs."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.execution_time_predictor.prediction_cache_contract import (
    DOMAIN_KIND_REGRESSION,
    PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION,
    attach_feature_domain,
    build_on_demand_prediction_record,
    classify_prediction_key,
    validate_prediction_grid_domain,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType, MeasurementType


class _CountingModel:
    def __init__(self, n_features: int) -> None:
        self.n_features_in_ = n_features
        self.calls = 0

    def predict(self, features: pd.DataFrame):
        self.calls += 1
        # A deterministic positive value is sufficient to prove that the
        # canonical estimator, rather than a nearest-row lookup, was called.
        return [float(sum(float(value) for value in row)) for row in features.to_numpy()]


class _TrackingExactLookup(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.items_calls = 0

    def items(self):
        self.items_calls += 1
        return super().items()


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("not used")

    def _get_grid_search_params(self):
        raise AssertionError("not used")


def _cache_write_training() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_tokens": [1, 32, 256],
            "kv_cache_size": [0, 0, 0],
            "batch_size": [1, 1, 1],
        }
    )


def _on_demand_predictor(*, diagnostics: bool) -> tuple[_ConcretePredictor, _CountingModel]:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(
        enable_prediction_domain_diagnostics=diagnostics,
    )
    predictor._runtime_cache = {"eager": {"attn_kv_cache_save": {}}}
    model = _CountingModel(3)
    model._frontier_feature_names = [
        "total_tokens",
        "kv_cache_size",
        "batch_size",
    ]
    attach_feature_domain(
        model,
        _cache_write_training(),
        model._frontier_feature_names,
        operator_name="attn_kv_cache_save",
    )
    predictor._predictions = {
        "attn_kv_cache_save": build_on_demand_prediction_record(
            "attn_kv_cache_save",
            model,
            model._frontier_feature_names,
        )
    }
    return predictor, model


def test_cache_write_interior_gap_is_predicted_by_ml_model() -> None:
    predictor, model = _on_demand_predictor(diagnostics=False)

    value = predictor._get_on_demand_prediction(
        "attn_kv_cache_save",
        {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1},
    )

    assert value == 9.0
    assert model.calls == 1
    descriptor = model._frontier_feature_domain
    assert descriptor["domain_kind"] == DOMAIN_KIND_REGRESSION
    assert (
        descriptor["runtime_prediction_policy"]
        == PREDICTION_DOMAIN_POLICY_ALLOW_MODEL_PREDICTION
    )


@pytest.mark.parametrize("total_tokens", [0, -1])
def test_cache_write_physical_invalid_tuple_still_fails_fast(total_tokens: int) -> None:
    predictor, model = _on_demand_predictor(diagnostics=False)

    with pytest.raises(ValueError, match="physical|constraint|nonnegative|negative"):
        predictor._get_on_demand_prediction(
            "attn_kv_cache_save",
            {
                "total_tokens": total_tokens,
                "kv_cache_size": 0,
                "batch_size": 1,
            },
        )

    assert model.calls == 0


def test_cache_write_upper_extrapolation_is_allowed_and_classified() -> None:
    predictor, model = _on_demand_predictor(diagnostics=True)

    value = predictor._get_on_demand_prediction(
        "attn_kv_cache_save",
        {"total_tokens": 512, "kv_cache_size": 0, "batch_size": 1},
    )

    assert value == 513.0
    assert model.calls == 1
    diagnostics = predictor.get_prediction_domain_diagnostics()
    assert diagnostics["eager"]["attn_kv_cache_save"]["extrapolation"]["count"] == 1
    assert diagnostics["eager"]["attn_kv_cache_save"]["extrapolation"]["sparse_gap"] is True
    assert diagnostics["eager"]["attn_kv_cache_save"]["extrapolation"][
        "value_sources"
    ]["model_prediction"] == 1


def test_on_demand_exact_source_precedes_stale_runtime_prediction() -> None:
    """A validated direct value must win over a prior runtime-model result."""

    predictor, model = _on_demand_predictor(diagnostics=False)
    predictor._runtime_cache["eager"]["attn_kv_cache_save"][(8, 0, 1)] = 99.0
    predictor._predictions["attn_kv_cache_save"]["_exact_lookup"] = {
        (8, 0, 1): 1.5,
    }

    value = predictor._get_on_demand_prediction(
        "attn_kv_cache_save",
        {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1},
    )

    assert value == 1.5
    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["attn_kv_cache_save"] == {
        (8, 0, 1): 99.0,
    }


def test_compute_model_producer_preserves_exact_lookup_for_cache_write() -> None:
    """The generic compute producer must publish direct measured rows."""

    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._max_tokens = 8
    predictor._model_config = SimpleNamespace(uses_fused_add_norm=True)
    predictor._replica_config = SimpleNamespace(
        model_config=SimpleNamespace(is_moe=False)
    )
    predictor._dense_attention_cache_write_op_name = lambda: "attn_kv_cache_save"
    predictor._get_predictor_attention_extra_ops = lambda: []
    predictor._requires_target_embedded_mtp_compute_models = lambda: False

    model = _CountingModel(3)
    model._frontier_feature_names = [
        "total_tokens",
        "kv_cache_size",
        "batch_size",
    ]
    attach_feature_domain(
        model,
        _cache_write_training(),
        model._frontier_feature_names,
        operator_name="attn_kv_cache_save",
    )
    model._frontier_exact_lookup = {(1, 0, 1): 0.125}
    predictor._models = {"attn_kv_cache_save": model}

    predictions = predictor._predict_for_compute_models()
    predictor._predictions = predictions
    predictor._runtime_cache = {"eager": {"attn_kv_cache_save": {}}}

    value = predictor._get_on_demand_prediction(
        "attn_kv_cache_save",
        {"total_tokens": 1, "kv_cache_size": 0, "batch_size": 1},
    )

    assert value == 0.125
    assert model.calls == 0
    assert predictor._runtime_cache["eager"]["attn_kv_cache_save"] == {}


def test_on_demand_rejects_malformed_exact_lookup_key_before_model_call() -> None:
    _, model = _on_demand_predictor(diagnostics=False)

    with pytest.raises(ValueError, match="exact lookup key.*length|schema"):
        build_on_demand_prediction_record(
            "attn_kv_cache_save",
            model,
            model._frontier_feature_names,
            exact_lookup={(8, 0): 1.5},
        )

    assert model.calls == 0


def test_exact_lookup_rows_above_current_runtime_cap_do_not_block_legal_query() -> None:
    predictor, model = _on_demand_predictor(diagnostics=False)
    predictor._predictions["attn_kv_cache_save"]["_exact_lookup"] = {
        (8, 0, 1): 1.5,
        (100000, 0, 1): 2.5,
    }

    value = predictor._get_on_demand_prediction(
        "attn_kv_cache_save",
        {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1},
    )

    assert value == 1.5
    assert model.calls == 0


def test_on_demand_rejects_non_mapping_exact_lookup_metadata() -> None:
    predictor, model = _on_demand_predictor(diagnostics=False)
    predictor._predictions["attn_kv_cache_save"]["_exact_lookup"] = []

    with pytest.raises(ValueError, match="invalid exact lookup metadata"):
        predictor._get_on_demand_prediction(
            "attn_kv_cache_save",
            {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1},
        )

    assert model.calls == 0


def test_diagnostics_separates_exact_model_and_runtime_value_sources() -> None:
    exact_predictor, exact_model = _on_demand_predictor(diagnostics=True)
    exact_predictor._predictions["attn_kv_cache_save"]["_exact_lookup"] = {
        (8, 0, 1): 1.5,
    }
    assert exact_predictor._get_on_demand_prediction(
        "attn_kv_cache_save",
        {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1},
    ) == 1.5
    exact_sources = exact_predictor.get_prediction_domain_diagnostics()["eager"][
        "attn_kv_cache_save"
    ]["interpolation"]["value_sources"]
    assert exact_sources == {"exact_lookup": 1}
    assert exact_predictor._runtime_cache["eager"]["attn_kv_cache_save"] == {}
    assert exact_model.calls == 0

    model_predictor, model = _on_demand_predictor(diagnostics=True)
    key = {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1}
    model_predictor._get_on_demand_prediction("attn_kv_cache_save", key)
    model_predictor._get_on_demand_prediction("attn_kv_cache_save", key)
    sources = model_predictor.get_prediction_domain_diagnostics()["eager"][
        "attn_kv_cache_save"
    ]["interpolation"]["value_sources"]
    assert sources == {"model_prediction": 1, "runtime_cache": 1}
    assert model.calls == 1


def test_runtime_cache_hit_does_not_rescan_all_exact_rows() -> None:
    predictor, model = _on_demand_predictor(diagnostics=False)
    tracked_lookup = _TrackingExactLookup({(1, 0, 1): 0.125})
    predictor._predictions["attn_kv_cache_save"] = (
        build_on_demand_prediction_record(
            "attn_kv_cache_save",
            model,
            model._frontier_feature_names,
            exact_lookup=tracked_lookup,
        )
    )
    key = {"total_tokens": 8, "kv_cache_size": 0, "batch_size": 1}

    first = predictor._get_on_demand_prediction("attn_kv_cache_save", key)
    second = predictor._get_on_demand_prediction("attn_kv_cache_save", key)

    assert first == second == 9.0
    assert model.calls == 1
    assert tracked_lookup.items_calls == 1


def test_finite_lookup_miss_calls_canonical_model_and_caches_runtime_value() -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(enable_prediction_domain_diagnostics=False)
    predictor._runtime_cache = {"eager": {"attn_pre_proj": {}}}
    model = _CountingModel(1)
    model._frontier_feature_names = ["num_tokens"]
    attach_feature_domain(
        model,
        pd.DataFrame({"num_tokens": [1, 32, 256]}),
        ["num_tokens"],
        operator_name="attn_pre_proj",
    )
    predictor._models = {"attn_pre_proj": model}
    predictor._predictions = {"attn_pre_proj": {(1,): 1.0}}

    value = predictor._get_lookup_or_predict(
        "attn_pre_proj", (8,), ["num_tokens"]
    )

    assert value == 8.0
    assert model.calls == 1
    assert predictor._runtime_cache["eager"]["attn_pre_proj"][(8,)] == 8.0


def test_finite_lookup_miss_rejects_key_above_runtime_physical_cap() -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(
        enable_prediction_domain_diagnostics=False,
        prediction_max_tokens_per_request=4096,
        prediction_max_batch_size=128,
    )
    predictor._model_config = SimpleNamespace(max_model_len=4096)
    predictor._max_tokens = 4096
    predictor._runtime_cache = {"eager": {"attn_pre_proj": {}}}
    model = _CountingModel(1)
    model._frontier_feature_names = ["num_tokens"]
    attach_feature_domain(
        model,
        pd.DataFrame({"num_tokens": [1, 32, 256]}),
        ["num_tokens"],
        operator_name="attn_pre_proj",
    )
    predictor._models = {"attn_pre_proj": model}
    predictor._predictions = {"attn_pre_proj": {(1,): 1.0}}

    with pytest.raises(ValueError, match="physical maximum|physical"):
        predictor._get_lookup_or_predict(
            "attn_pre_proj", (8192,), ["num_tokens"]
        )
    assert model.calls == 0


def test_finite_lookup_hit_still_validates_physical_bounds_before_return() -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(
        enable_prediction_domain_diagnostics=False,
        prediction_max_tokens_per_request=4096,
        prediction_max_batch_size=128,
        prediction_max_prefill_chunk_size=4096,
    )
    predictor._model_config = SimpleNamespace(max_model_len=4096)
    predictor._max_tokens = 4096
    predictor._runtime_cache = {"eager": {"attn_pre_proj": {}}}
    model = _CountingModel(1)
    model._frontier_feature_names = ["num_tokens"]
    attach_feature_domain(
        model,
        pd.DataFrame({"num_tokens": [1, 32, 256]}),
        ["num_tokens"],
        operator_name="attn_pre_proj",
    )
    predictor._models = {"attn_pre_proj": model}
    predictor._predictions = {"attn_pre_proj": {(8192,): 1.0}}

    with pytest.raises(ValueError, match="physical"):
        predictor._get_lookup_or_predict(
            "attn_pre_proj", (8192,), ["num_tokens"]
        )

    assert model.calls == 0


def test_runtime_lookup_hit_still_validates_physical_bounds_before_return() -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(
        enable_prediction_domain_diagnostics=False,
        prediction_max_tokens_per_request=4096,
        prediction_max_batch_size=128,
        prediction_max_prefill_chunk_size=4096,
    )
    predictor._model_config = SimpleNamespace(max_model_len=4096)
    predictor._max_tokens = 4096
    predictor._runtime_cache = {"eager": {"attn_pre_proj": {(8192,): 2.0}}}
    model = _CountingModel(1)
    model._frontier_feature_names = ["num_tokens"]
    attach_feature_domain(
        model,
        pd.DataFrame({"num_tokens": [1, 32, 256]}),
        ["num_tokens"],
        operator_name="attn_pre_proj",
    )
    predictor._models = {"attn_pre_proj": model}
    predictor._predictions = {"attn_pre_proj": {}}

    with pytest.raises(ValueError, match="physical"):
        predictor._get_lookup_or_predict(
            "attn_pre_proj", (8192,), ["num_tokens"]
        )

    assert model.calls == 0


@pytest.mark.parametrize(
    ("field", "value", "feature_name"),
    [
        ("prediction_max_batch_size", 0, "batch_size"),
        ("prediction_max_prefill_chunk_size", 0, "prefill_chunk_size_squared"),
    ],
)
def test_runtime_physical_cap_rejects_non_positive_configuration(
    field: str,
    value: int,
    feature_name: str,
) -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    config_values = {
        "prediction_max_tokens_per_request": 4096,
        "prediction_max_batch_size": 128,
        "prediction_max_prefill_chunk_size": 4096,
    }
    config_values[field] = value
    predictor._config = SimpleNamespace(**config_values)
    predictor._model_config = SimpleNamespace(max_model_len=4096)
    predictor._max_tokens = 4096

    with pytest.raises(ValueError, match="must be a positive integer|must be positive"):
        predictor._get_runtime_prediction_physical_bounds(
            "runtime_cap_probe", [feature_name]
        )


def test_runtime_kv_cap_matches_the_rounding_lattice_for_non_aligned_limit() -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(
        prediction_max_tokens_per_request=4090,
        prediction_max_batch_size=128,
        prediction_max_prefill_chunk_size=4096,
        kv_cache_prediction_granularity=64,
    )
    predictor._model_config = SimpleNamespace(max_model_len=8192)
    predictor._max_tokens = 4090

    bounds = predictor._get_runtime_prediction_physical_bounds(
        "attn_decode", ["kv_cache_size", "max_seq_len"]
    )

    # Runtime KV features are rounded to 64-token blocks, while max_seq_len
    # remains the raw request/context limit.
    assert bounds["kv_cache_size"] == {"min": 0.0, "max": 4096.0}
    assert bounds["max_seq_len"] == {"min": 0.0, "max": 4090.0}


def test_prefill_context_cap_remains_conservative_for_non_aligned_limit() -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(
        prediction_max_tokens_per_request=4090,
        prediction_max_prefill_chunk_size=4096,
        kv_cache_prediction_granularity=64,
    )
    predictor._model_config = SimpleNamespace(max_model_len=4090)

    constraints = predictor._get_runtime_prediction_constraints(
        ["kv_cache_size", "prefill_chunk_size_squared"]
    )

    # The relational validator remains conservative because the rounded KV
    # feature does not retain the raw pre-rounded context token count.
    assert constraints[0]["max"] == 4091.0


def test_prefill_context_cap_prefers_serving_limit_over_architecture_cap() -> None:
    predictor = _ConcretePredictor.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(
        prediction_max_tokens_per_request=4096,
        prediction_max_prefill_chunk_size=4096,
        kv_cache_prediction_granularity=64,
    )
    # Qwen3-30B-A3B-tiny advertises a 40960-token architecture context, while
    # the serving contract for this release remains 4096.
    predictor._model_config = SimpleNamespace(max_position_embeddings=40960)
    predictor._serving_max_tokens_per_request = 4096

    constraints = predictor._get_runtime_prediction_constraints(
        ["kv_cache_size", "prefill_chunk_size_squared"]
    )

    assert constraints[0]["max"] == 4097.0


def test_prefill_gap_uses_runtime_context_cap_and_rejects_non_square_chunk() -> None:
    model = _CountingModel(2)
    training = pd.DataFrame(
        {
            "kv_cache_size": [0, 64],
            "prefill_chunk_size_squared": [32**2, 96**2],
        }
    )
    attach_feature_domain(
        model,
        training,
        ["kv_cache_size", "prefill_chunk_size_squared"],
        operator_name="attn_prefill",
    )
    runtime_bounds = {
        "kv_cache_size": {"min": 0.0, "max": 4096.0},
        "prefill_chunk_size_squared": {"min": 1.0, "max": 4096.0**2},
    }
    runtime_constraints = [
        {
            "type": "sum_lte",
            "features": ["kv_cache_size", "prefill_chunk_size"],
            "derived_features": {
                "prefill_chunk_size": {"sqrt": "prefill_chunk_size_squared"}
            },
            "max": 4097,
            "enforce_during_extrapolation": True,
        }
    ]

    validate_prediction_grid_domain(
        "attn_prefill",
        model,
        pd.DataFrame(
            {"kv_cache_size": [4096], "prefill_chunk_size_squared": [1]}
        ),
        runtime_physical_bounds=runtime_bounds,
        runtime_constraints=runtime_constraints,
    )
    with pytest.raises(ValueError, match="perfect square"):
        validate_prediction_grid_domain(
            "attn_prefill",
            model,
            pd.DataFrame(
                {"kv_cache_size": [0], "prefill_chunk_size_squared": [2]}
            ),
            runtime_physical_bounds=runtime_bounds,
            runtime_constraints=runtime_constraints,
        )
    with pytest.raises(ValueError, match="conditional constraint"):
        validate_prediction_grid_domain(
            "attn_prefill",
            model,
            pd.DataFrame(
                {"kv_cache_size": [4096], "prefill_chunk_size_squared": [32**2]}
            ),
            runtime_physical_bounds=runtime_bounds,
            runtime_constraints=runtime_constraints,
        )


def test_prediction_provenance_distinguishes_direct_interpolation_and_extrapolation() -> None:
    model = _CountingModel(1)
    attach_feature_domain(
        model,
        pd.DataFrame({"num_tokens": [1, 32, 256]}),
        ["num_tokens"],
        operator_name="attn_pre_proj",
    )
    domain = model._frontier_feature_domain

    assert classify_prediction_key(domain, (32,))["classification"] == "direct_measured"
    assert classify_prediction_key(domain, (8,))["classification"] == "interpolation"
    assert classify_prediction_key(domain, (512,))["classification"] == "extrapolation"
