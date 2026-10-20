"""CPU coverage for measured DSA loading, feature identity and runtime scopes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from frontier.attention.dsa_features import DSA_FEATURE_COLUMNS, DSACoverageError, build_dsa_features
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.dsa_predictor import (
    DSAPredictor, dsa_input_path, load_dsa_dataframe, train_dsa_models,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import _build_exact_feature_lookup
from frontier.types import MeasurementType


@pytest.fixture
def model():
    return BaseModelConfig.create_from_name("zai-org/GLM-5.3")


def _row(model, queries, contexts, variant="full_index", *, prefill=True):
    features = build_dsa_features(queries, contexts, model.index_topk)
    return {
        "schema_version": "dsa_attention.v1", "measurement_type": "KERNEL_ONLY",
        "model_architecture_profile": model.get_model_architecture_profile().profile_id,
        "quant_signature": model.get_quant_signature(),
        "profiling_precision": model.get_default_precision().name,
        "attention_backend": "FLASH_ATTN_MLA_SPARSE", "attention_impl": "FlashAttnMLASparseImpl",
        "attention_family": "dsa_attention", "attention_variant": variant,
        "n_embd": model.embedding_dim, "n_q_head": model.num_q_heads,
        "n_q_head_per_rank": model.num_q_heads // 8,
        "num_tensor_parallel_workers": 8, "block_size": 64,
        **{name: getattr(model, name) for name in (
            "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "index_topk", "index_n_heads", "index_head_dim",
        )},
        "prefill_seq_lens": json.dumps(queries if prefill else []),
        "prefill_kv_cache_sizes": json.dumps(contexts if prefill else []),
        "decode_kv_cache_sizes": json.dumps([] if prefill else contexts),
        "is_prefill": prefill, "max_key_len": features.values["max_key_len"],
        "time_stats.attn_dsa_indexer.median": 0.2 if variant == "full_index" else np.nan,
        "time_stats.attn_dsa_mla.median": 0.3,
        "time_stats.attn_dsa_indexer.kernel_count": 5 if variant == "full_index" else np.nan,
        "time_stats.attn_dsa_mla.kernel_count": 7,
    }


def _fit(**kwargs):
    columns = kwargs["feature_cols"]
    return SimpleNamespace(
        _frontier_feature_names=columns,
        _frontier_exact_lookup=_build_exact_feature_lookup(kwargs["df"], columns, kwargs["target_col"]),
        predict=lambda frame: np.repeat(0.25, len(frame)),
    )


def _load(tmp_path, model, rows, measurement=MeasurementType.KERNEL_ONLY):
    file = tmp_path / ("dsa_attention.csv" if measurement is MeasurementType.CUDA_EVENT else "dsa_attention_kernel_only.csv")
    pd.DataFrame(rows).to_csv(file, index=False)
    return load_dsa_dataframe(str(file), model, 8, 64, measurement)


def test_decode_and_prefill_features_follow_native_query_threshold():
    from frontier.attention.dsa_features import dsa_decode_query_limit

    assert dsa_decode_query_limit(16) == 128
    assert build_dsa_features([129], [0], 2048, decode_query_limit=dsa_decode_query_limit(16)).path == "prefill"
    assert build_dsa_features([1, 1], [2047, 2048], 2048).values["sum_selected_key_len"] == 4096
    assert build_dsa_features([256], [2048], 2048).path == "decode"
    assert build_dsa_features([257], [2048], 2048).path == "prefill"
    assert build_dsa_features([2], [2047], 2048).values["sum_selected_causal_key_pairs"] == 4096


def test_causal_features_clip_each_query_not_only_final_key_length():
    feature = build_dsa_features([3, 4], [1, 2], 4)
    assert feature.values["sum_causal_key_pairs"] == sum(c + i for q, c in [(3, 1), (4, 2)] for i in range(1, q + 1))
    assert feature.values["sum_selected_causal_key_pairs"] == sum(min(c + i, 4) for q, c in [(3, 1), (4, 2)] for i in range(1, q + 1))
    with pytest.raises(DSACoverageError, match="mixed"):
        build_dsa_features([1, 257], [128, 0], 2048)


def test_loader_only_opens_dsa_files_and_uses_actual_feature_lists(tmp_path, model):
    rows = [_row(model, [1, 1], [128, 2048], prefill=False)]
    df = _load(tmp_path, model, rows)
    # A malformed legacy table cannot be read by the DSA path.
    (tmp_path / "attention_kernel_only.csv").write_text("not,a,dsa,table\n")
    loaded = load_dsa_dataframe(str(tmp_path / "attention_kernel_only.csv"), model, 8, 64, MeasurementType.KERNEL_ONLY)
    pd.testing.assert_frame_equal(df, loaded)
    assert df.iloc[0].sum_key_len == 2178
    assert df.iloc[0].sum_selected_key_len == 2177
    assert dsa_input_path(str(tmp_path / "attention.csv")).endswith("dsa_attention.csv")


@pytest.mark.parametrize("column,value", [
    ("schema_version", "dense.v1"), ("measurement_type", "CUDA_EVENT"),
    ("quant_signature", "none"), ("model_architecture_profile", "generic"),
    ("profiling_precision", "FP8"), ("attention_family", "latent_mla"),
    ("block_size", 32), ("index_topk", 1), ("n_q_head_per_rank", 1),
])
def test_loader_rejects_incompatible_identity_or_shape(tmp_path, model, column, value):
    row = _row(model, [512], [0]); row[column] = value
    with pytest.raises(ValueError, match="DSA"):
        _load(tmp_path, model, [row])


def test_cuda_event_decode_host_stall_is_not_a_training_target(tmp_path, model):
    decode = _row(model, [1], [2048], prefill=False)
    decode["time_stats.attn_dsa_mla.median"] = 10000
    rows = [decode, _row(model, [512], [0])]
    for row in rows:
        row["measurement_type"] = "CUDA_EVENT"
    df = _load(tmp_path, model, rows, MeasurementType.CUDA_EVENT)
    assert list(df.dsa_workload) == ["prefill"]
    assert df["time_stats.attn_dsa_mla.median"].max() == 0.3


def test_phase_features_and_owner_only_indexer_use_exact_measured_rows(tmp_path, model):
    rows = [
        _row(model, queries, contexts, variant, prefill=prefill)
        for variant in ("full_index", "shared_index")
        for queries, contexts, prefill in [([1, 1], [128, 2048], False), ([64], [0], True), ([512], [0], True)]
    ]
    fitted = train_dsa_models(_load(tmp_path, model, rows), _fit)
    assert len(fitted) == 9
    for name, fitted_model in fitted.items():
        operator = "attn_dsa_indexer" if name.startswith("attn_dsa_indexer") else "attn_dsa_mla"
        path = "prefill" if name.endswith("_prefill") else "decode"
        assert tuple(fitted_model._frontier_feature_names) == DSA_FEATURE_COLUMNS[operator][path]
    predictor = DSAPredictor(fitted)
    features = build_dsa_features([1, 1], [128, 2048], model.index_topk)
    calls = []
    full = predictor.operator_times(features, "full_index", record_kernel_count=lambda *args: calls.append(args))
    shared = predictor.operator_times(features, "shared_index")
    assert full.total_time() == 0.5
    assert shared.total_time() == 0.3
    assert len(calls) == 2
    assert calls[0][2] == (2.0, 2178.0)


@pytest.mark.parametrize("queries,contexts", [([2], [0]), ([512, 512], [0, 0]), ([1], [999999])])
def test_uncovered_extend_multi_prefill_and_long_context_fail(tmp_path, model, queries, contexts):
    rows = [_row(model, [1], [128], prefill=False), _row(model, [64], [0]), _row(model, [512], [0])]
    predictor = DSAPredictor(train_dsa_models(_load(tmp_path, model, rows), _fit))
    with pytest.raises(DSACoverageError, match="coverage missing"):
        predictor.operator_times(build_dsa_features(queries, contexts, model.index_topk), "full_index")


def test_trained_data_is_required():
    with pytest.raises(ValueError, match="trained profiling data"):
        DSAPredictor({})


def test_attention_dp_dummy_forward_runs_only_outer_projection_bundle(model):
    from frontier.entities.batch import DummyForwardBatch
    from unittest.mock import ANY, Mock

    owner = SimpleNamespace(
        _model_config=model,
        _get_attention_layer_pre_proj_execution_time=lambda _: 0.1,
        _get_attention_layer_post_proj_execution_time=lambda _: 0.2,
        _get_attention_rope_execution_time=lambda _: 0.3,
        _get_attn_norm_layer_act_execution_time=Mock(return_value=0.4),
    )
    predictor = DSAPredictor({"attn_dsa_mla": SimpleNamespace()})
    timing = predictor.predict_attention_time(owner, DummyForwardBatch(0, 0, 0), 2)
    assert timing.operator_times.total_time() == 0
    assert timing.attention_layer_pre_proj_execution_time == 0.1
    assert timing.attention_layer_post_proj_execution_time == 0.2
    owner._get_attn_norm_layer_act_execution_time.assert_called_once_with(
        ANY, layer_id=2,
    )


def test_stage_metrics_retain_measured_scopes_without_dummy_reconstruction():
    from frontier.entities.time_components import AttentionOperatorTimes
    from frontier.scheduler.utils.dense_metrics import _restore_structural_dsa_attention

    layer = SimpleNamespace(attention_family_id="dsa_attention", attention_operator_times=AttentionOperatorTimes({
        "attn_dsa_indexer": 0.2, "attn_dsa_mla": 0.3,
    }))
    owner = SimpleNamespace(_enable_dummy_mode=False)
    assert _restore_structural_dsa_attention(layer, owner) is layer
    layer.attention_operator_times = None
    with pytest.raises(ValueError, match="executed attention scopes"):
        _restore_structural_dsa_attention(layer, owner)


def test_dsa_scopes_reuse_the_canonical_kernel_count_training(tmp_path, model):
    from frontier.execution_time_predictor.kernel_gap import KernelCountTraining

    training = KernelCountTraining()
    training._kernel_gap_ms = {"none": 0.1}
    training._active_measurement_type = MeasurementType.KERNEL_ONLY
    rows = _load(tmp_path, model, [_row(model, [1, 1], [128, 2048], prefill=False)])

    def fit(**kwargs):
        return training._paired_with_kernel_count_model(
            _fit(**kwargs), kwargs["model_name"], kwargs["df"], kwargs["feature_cols"], kwargs["target_col"],
            lambda name, df, columns, target: _fit(model_name=name, df=df, feature_cols=columns, target_col=target),
        )

    fitted = train_dsa_models(rows, fit)
    recorded = {}

    def record(name, estimator, key, columns):
        count = estimator._frontier_kernel_count_model
        recorded[count._frontier_operator_name] = count._frontier_exact_lookup[key]

    DSAPredictor(fitted).operator_times(build_dsa_features([1, 1], [128, 2048], model.index_topk),
                                      "full_index", record_kernel_count=record)
    assert recorded == {"attn_dsa_indexer": 5.0, "attn_dsa_mla": 7.0}


def test_dsa_metadata_registration_uses_only_the_child_reader(monkeypatch):
    from unittest.mock import Mock
    from frontier.attention.families import DSA_ATTENTION_FAMILY
    from frontier.attention.profiling_mapping import get_enabled_predictor_metric_names
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor

    predictor = SimpleNamespace()
    rows = pd.DataFrame({"measurement_type": ["CUDA_EVENT"]})
    predictor._load_attention_df = Mock(return_value=rows)
    predictor._get_profiling_metadata = Mock(return_value={})
    predictor._validate_active_measurement_type = Mock()
    predictor._register_profiling_metadata_for_ops = Mock()

    def reject_generic_read(*args, **kwargs):
        raise AssertionError("The generic metadata reader must not open DSA files")

    monkeypatch.setattr(pd, "read_csv", reject_generic_read)
    names = list(get_enabled_predictor_metric_names(DSA_ATTENTION_FAMILY))
    SklearnExecutionTimePredictor._register_profiling_metadata_from_file(predictor, "/profiles/attention.csv", names)
    predictor._load_attention_df.assert_called_once_with("/profiles/dsa_attention.csv")
    predictor._register_profiling_metadata_for_ops.assert_called_once_with(names, {}, "/profiles/dsa_attention.csv")
