from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention.families import DENSE_ATTENTION_FAMILY, DSA_ATTENTION_FAMILY
from frontier.model_architectures import ModelArchitectureProfile
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType
from frontier.types import MeasurementType


class _ConcreteSklearnExecutionTimePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("unit test bypasses estimator construction")

    def _get_grid_search_params(self):
        raise AssertionError("unit test bypasses grid-search construction")


class _GenericModelConfig:
    def get_model_architecture_profile(self):
        return ModelArchitectureProfile.generic()


class _Step2MiniProfileModelConfig:
    def get_model_architecture_profile(self):
        return ModelArchitectureProfile.step2_mini()


def _build_predictor(monkeypatch: pytest.MonkeyPatch) -> SklearnExecutionTimePredictor:
    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._enable_dummy_mode = False
    predictor._model_config = _GenericModelConfig()

    monkeypatch.setattr(
        predictor,
        "_log_architecture_attention_shape",
        lambda batch: None,
    )
    monkeypatch.setattr(predictor, "_supports_operation", lambda operation: True)
    monkeypatch.setattr(predictor, "_get_attention_family", lambda: DENSE_ATTENTION_FAMILY)
    monkeypatch.setattr(predictor, "_get_attention_prefill_execution_time", lambda batch: 2.0)
    monkeypatch.setattr(predictor, "_get_attention_decode_execution_time", lambda batch: 3.0)
    monkeypatch.setattr(
        predictor,
        "_should_use_hybrid_attention_measurement_for_spec_piecewise",
        lambda batch: False,
    )
    monkeypatch.setattr(
        predictor,
        "_get_attention_layer_pre_proj_execution_time",
        lambda batch: 0.5,
    )
    monkeypatch.setattr(
        predictor,
        "_get_attention_layer_post_proj_execution_time",
        lambda batch: 0.7,
    )
    monkeypatch.setattr(predictor, "_get_attention_rope_execution_time", lambda batch: 0.11)
    monkeypatch.setattr(
        predictor,
        "_get_attention_kv_cache_save_execution_time",
        lambda batch: 0.13,
    )
    monkeypatch.setattr(predictor, "_get_attn_norm_layer_act_execution_time", lambda batch: 0.17)
    return predictor


def _build_batch(*, num_prefill_tokens: int, num_decode_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=101,
        is_idle=False,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        total_num_tokens=num_prefill_tokens + num_decode_tokens,
        requests=[SimpleNamespace(num_prefill_tokens=num_prefill_tokens)],
    )


def _extract_total_attention_time(records: list[logging.LogRecord]) -> float:
    total_records = [
        record
        for record in records
        if "[OP-TRACE][MONOLITHIC][ATTENTION][TOTAL]" in record.getMessage()
    ]
    assert len(total_records) == 1
    message = total_records[0].getMessage()
    prefix = "total_attention_time_ms="
    return float(message.split(prefix, maxsplit=1)[1])


@pytest.mark.parametrize(
    ("num_prefill_tokens", "num_decode_tokens", "expected_prefill_time"),
    [
        (0, 4, 0.0),
        (5, 4, 2.0),
    ],
)
def test_attention_trace_total_includes_decode(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    num_prefill_tokens: int,
    num_decode_tokens: int,
    expected_prefill_time: float,
) -> None:
    predictor = _build_predictor(monkeypatch)
    batch = _build_batch(
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
    )

    with caplog.at_level(logging.INFO):
        attention_time = predictor.predict_attention_layer_time(
            batch=batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )

    expected_total = sum(
        (
            0.17,  # input_layernorm
            0.5,  # attn_pre_proj
            0.11,  # attn_rope
            expected_prefill_time,
            3.0,  # attn_decode
            0.13,  # attn_kv_cache_save
            0.7,  # attn_post_proj
        )
    )
    assert attention_time.attention_decode_execution_time == pytest.approx(3.0)
    assert _extract_total_attention_time(caplog.records) == pytest.approx(expected_total)


def test_dsa_frozen_fails_fast_in_dummy_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 1.0
    predictor._model_config = _Step2MiniProfileModelConfig()

    monkeypatch.setattr(
        predictor,
        "_log_architecture_attention_shape",
        lambda batch: None,
    )
    monkeypatch.setattr(predictor, "_get_attention_family", lambda: DSA_ATTENTION_FAMILY)

    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        predictor.predict_attention_layer_time(
            batch=_build_batch(num_prefill_tokens=1, num_decode_tokens=0),
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )


def test_kernel_only_attention_trains_true_mixed_decode_when_real_rows_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
    predictor._attention_input_file = "unused.csv"

    attention_df = pd.DataFrame(
        {
            "is_decode": [True, False],
            "is_true_mixed_batch": [False, True],
            "is_mixed_batch": [False, False],
            "batch_size": [1, 2],
            "kv_cache_size": [64, 0],
            "prefill_chunk_size": [0, 0],
            "prefill_chunk_size_squared": [0, 0],
            "decode_batch_size": [0, 1],
            "decode_avg_kv_cache_size": [0, 64],
            "num_prefill_seqs": [0, 1],
            "total_prefill_tokens": [0, 16],
            "total_batch_size": [1, 2],
            "batch_composition_ratio": [0.0, 0.5],
            "total_tokens": [1, 17],
            "time_stats.attn_decode.median": [0.4, 0.8],
        }
    )

    monkeypatch.setattr(predictor, "_is_mla_attention_family", lambda: False)
    monkeypatch.setattr(predictor, "_load_attention_df", lambda _path: attention_df)
    monkeypatch.setattr(
        predictor,
        "_get_attention_df_with_derived_features",
        lambda df: df,
    )
    trained_models: dict[str, tuple[tuple[str, ...], str, int]] = {}

    def _fake_train_model(*, model_name: str, df: pd.DataFrame, feature_cols, target_col):
        trained_models[model_name] = (tuple(feature_cols), target_col, len(df))
        return SimpleNamespace()

    monkeypatch.setattr(predictor, "_train_model", _fake_train_model)

    predictor._train_attention_layer_models()

    assert trained_models["attn_decode"] == (
        ("batch_size", "kv_cache_size"),
        "time_stats.attn_decode.median",
        1,
    )
    assert trained_models["attn_decode_in_mixed"] == (
        (
            "decode_batch_size",
            "decode_avg_kv_cache_size",
            "num_prefill_seqs",
            "total_prefill_tokens",
            "total_batch_size",
            "batch_composition_ratio",
            "total_tokens",
        ),
        "time_stats.attn_decode.median",
        1,
    )


def test_kernel_only_attention_registers_true_mixed_decode_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(
        prediction_max_batch_size=1,
        prediction_max_tokens_per_request=64,
        kv_cache_prediction_granularity=64,
        prediction_max_prefill_chunk_size=16,
    )
    mixed_feature_names = [
        "decode_batch_size",
        "decode_avg_kv_cache_size",
        "num_prefill_seqs",
        "total_prefill_tokens",
        "total_batch_size",
        "batch_composition_ratio",
        "total_tokens",
    ]
    predictor._models = {
        "attn_decode": SimpleNamespace(),
        "attn_decode_in_mixed": SimpleNamespace(
            n_features_in_=len(mixed_feature_names),
            _frontier_feature_names=mixed_feature_names,
        ),
    }

    monkeypatch.setattr(predictor, "_is_mla_attention_family", lambda: False)
    monkeypatch.setattr(predictor, "_dense_attention_decode_op_name", lambda: "attn_decode")
    monkeypatch.setattr(predictor, "_dense_attention_prefill_op_name", lambda: "attn_prefill")
    monkeypatch.setattr(
        predictor,
        "_get_model_prediction",
        lambda model_name, _model, _df: {"model_name": model_name},
    )

    predictions = predictor._predict_for_attention_layer_models()

    assert predictions["attn_decode"] == {"model_name": "attn_decode"}
    assert predictions["attn_decode_in_mixed"]["_on_demand_prediction"] is True
    assert predictions["attn_decode_in_mixed"]["_feature_names"] == mixed_feature_names
