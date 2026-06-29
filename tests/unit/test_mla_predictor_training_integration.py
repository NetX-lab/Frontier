from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_feature_columns,
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_names,
    get_imported_mla_predictor_feature_columns,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
    _build_exact_feature_lookup,
)
from frontier.profiling.attention.vllm_mla_profile_importer import (
    build_frontier_mla_profile_dataframe,
)
from frontier.types import ClusterType, MeasurementType
from tests.unit.mla_h800_fixture import h800_mla_mixed_rows


class _DummySklearnPredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("_get_estimator should not be called by this unit test")


class _FakeMlaModel:
    def __init__(self, feature_names: list[str], prediction: float = 0.125):
        self._frontier_feature_names = list(feature_names)
        self.n_features_in_ = len(feature_names)
        self._prediction = float(prediction)

    def predict(self, dataframe):
        return [self._prediction for _ in range(len(dataframe))]


def _base_meta(
    *,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 65,
    num_actual_tokens: int = 1,
) -> dict[str, object]:
    return {
        "attention_backend": "FLASHINFER_MLA",
        "use_mla": True,
        "runtime_num_kv_heads": 1,
        "runtime_head_size": 576,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "qk_head_dim": 192,
        "v_head_dim": 128,
        "block_size": 64,
        "kv_cache_dtype": "auto",
        "calculate_kv_scales": False,
        "attn_module_sliding_window": None,
        "alibi_slopes": None,
        "logits_soft_cap": None,
        "attn_type": "decoder",
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "num_actual_tokens": num_actual_tokens,
    }


def _sample_vllm_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, scope in enumerate(
        get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY),
        1,
    ):
        rows.append(
            {
                "batch_id": 7,
                "batch_size": 1,
                "batch_num_tokens": 1,
                "batch_num_prefill_tokens": 0,
                "batch_num_decode_tokens": 1,
                "batch_request_num_tokens": [1],
                "op_name": scope,
                "cuda_time_ms": float(idx) / 100.0,
                "count": 1,
                "meta": _base_meta(),
            }
        )
    return rows


def _build_imported_mla_profile_df(
    measurement_type: str | MeasurementType = MeasurementType.CUDA_EVENT,
) -> pd.DataFrame:
    return build_frontier_mla_profile_dataframe(
        _sample_vllm_rows(),
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type=measurement_type,
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )


def _build_mla_predictor(
    measurement_type: MeasurementType = MeasurementType.CUDA_EVENT,
) -> _DummySklearnPredictor:
    predictor = _DummySklearnPredictor.__new__(_DummySklearnPredictor)
    predictor._attention_input_file = "unused_attention.csv"
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = measurement_type
    predictor._block_size = 64
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._replica_config = SimpleNamespace(attn_tensor_parallel_size=1)
    predictor._model_config = SimpleNamespace(
        use_mla=True,
        model_arch="deepseek_v2",
        num_q_heads=128,
        num_kv_heads=128,
        embedding_dim=73728,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        kv_lora_rank=512,
        v_head_dim=128,
        get_qk_head_dim=lambda: 192,
        get_quant_signature=lambda: "none",
    )
    predictor._get_profiling_metadata = (
        lambda *_args, **_kwargs: SimpleNamespace(
            measurement_type=measurement_type,
            profiling_precision=SimpleNamespace(name="BF16"),
            quant_signature="none",
            model_arch="deepseek_v2",
        )
    )
    predictor._validate_active_measurement_type = lambda *_args, **_kwargs: None
    predictor._register_profiling_metadata_for_ops = lambda *_args, **_kwargs: None
    return predictor


def test_mla_attention_model_names_follow_bound_family() -> None:
    predictor = _build_mla_predictor()

    assert predictor._get_attention_model_names() == list(
        get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY)
    )


def test_mla_load_attention_df_filters_imported_structural_rows(tmp_path) -> None:
    predictor = _build_mla_predictor()
    mla_df = _build_imported_mla_profile_df()
    mismatched_df = mla_df.copy()
    mismatched_df["kv_lora_rank"] = 256
    input_file = tmp_path / "attention.csv"
    pd.concat([mismatched_df, mla_df], ignore_index=True).to_csv(
        input_file,
        index=False,
    )

    filtered = predictor._load_attention_df(str(input_file))

    assert len(filtered) == 1
    assert int(filtered.iloc[0]["kv_lora_rank"]) == 512
    assert int(filtered.iloc[0]["n_kv_head"]) == 1
    assert int(filtered.iloc[0]["head_size"]) == 576


def test_mla_train_attention_layer_models_trains_six_imported_targets() -> None:
    predictor = _build_mla_predictor()
    predictor._load_attention_df = lambda _path: _build_imported_mla_profile_df()

    trained: dict[str, dict[str, object]] = {}

    def _fake_train_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ):
        trained[model_name] = {
            "feature_cols": tuple(feature_cols),
            "target_col": target_col,
            "num_rows": len(df),
        }
        return _FakeMlaModel(feature_cols)

    predictor._train_model = _fake_train_model

    models = predictor._train_attention_layer_models()

    expected_ops = list(get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY))
    expected_features = get_enabled_predictor_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    )
    expected_targets = dict(
        zip(
            expected_ops,
            get_enabled_predictor_median_columns(LATENT_MLA_ATTENTION_FAMILY),
        )
    )
    assert list(models) == expected_ops
    assert set(trained) == set(expected_ops)
    for op_name in expected_ops:
        assert trained[op_name] == {
            "feature_cols": tuple(expected_features[op_name]),
            "target_col": expected_targets[op_name],
            "num_rows": 1,
        }
        assert getattr(models[op_name], "_frontier_exact_lookup")


def test_mla_predict_for_attention_layer_models_exposes_on_demand_exact_lookup() -> None:
    predictor = _build_mla_predictor()
    df = _build_imported_mla_profile_df()
    feature_columns_by_op = get_enabled_predictor_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    )
    target_columns = dict(
        zip(
            get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY),
            get_enabled_predictor_median_columns(LATENT_MLA_ATTENTION_FAMILY),
        )
    )
    predictor._models = {}
    for op_name, feature_cols in feature_columns_by_op.items():
        feature_cols = list(feature_cols)
        model = _FakeMlaModel(feature_cols)
        model._frontier_exact_lookup = _build_exact_feature_lookup(
            df,
            feature_cols,
            target_columns[op_name],
        )
        predictor._models[op_name] = model

    predictions = predictor._predict_for_attention_layer_models()

    assert list(predictions) == list(
        get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY)
    )
    for op_name, model_info in predictions.items():
        assert model_info["_on_demand_prediction"] is True
        assert model_info["_model"] is predictor._models[op_name]
        assert model_info["_feature_names"] == list(feature_columns_by_op[op_name])
        assert model_info["_exact_lookup"]


def test_mla_train_attention_layer_models_filters_sparse_rows_by_target() -> None:
    predictor = _build_mla_predictor()
    sparse_df = build_frontier_mla_profile_dataframe(
        h800_mla_mixed_rows(),
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )
    predictor._load_attention_df = lambda _path: sparse_df

    trained_row_counts: dict[str, int] = {}

    def _fake_train_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ):
        assert not df[target_col].isna().any()
        trained_row_counts[model_name] = len(df)
        return _FakeMlaModel(feature_cols)

    predictor._train_model = _fake_train_model

    models = predictor._train_attention_layer_models()

    assert set(models) == set(get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY))
    assert trained_row_counts == {
        "attn_mla_kv_cache_save": 3,
        "attn_mla_prefill_kv_up_proj": 2,
        "attn_mla_prefill": 2,
        "attn_mla_decode_q_latent_proj": 2,
        "attn_mla_decode": 2,
        "attn_mla_v_up_proj": 2,
    }
    for model in models.values():
        assert getattr(model, "_frontier_exact_lookup")


def test_mla_h800_sparse_profile_preserves_expected_row_and_nan_cell_counts() -> None:
    sparse_df = build_frontier_mla_profile_dataframe(
        h800_mla_mixed_rows(),
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )
    target_columns = list(
        get_enabled_predictor_median_columns(LATENT_MLA_ATTENTION_FAMILY)
    )

    observed_cells = int(sparse_df[target_columns].notna().sum().sum())
    nan_cells = int(sparse_df[target_columns].isna().sum().sum())

    assert len(h800_mla_mixed_rows()) == 13
    assert len(sparse_df) == 5
    assert len(target_columns) == 6
    assert observed_cells == 13
    assert nan_cells == 17
    assert {
        column.removeprefix("time_stats.").removesuffix(".median"): int(
            sparse_df[column].notna().sum()
        )
        for column in target_columns
    } == {
        "attn_mla_kv_cache_save": 3,
        "attn_mla_prefill_kv_up_proj": 2,
        "attn_mla_prefill": 2,
        "attn_mla_decode_q_latent_proj": 2,
        "attn_mla_decode": 2,
        "attn_mla_v_up_proj": 2,
    }


def test_mla_imported_predictor_features_exclude_request_token_vectors() -> None:
    predictor_features = get_imported_mla_predictor_feature_columns()
    predictor_features_by_op = get_enabled_predictor_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    )
    imported_df = build_frontier_mla_profile_dataframe(
        h800_mla_mixed_rows(),
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )

    assert "batch_request_num_tokens" not in predictor_features
    assert "batch_request_num_tokens" in imported_df.attrs["dynamic_signature_fields"]
    assert "batch_request_num_tokens" not in imported_df.columns
    assert set(imported_df.attrs["batch_request_num_tokens_by_signature"]) == {
        (64,),
        (1, 64),
        (1,),
    }
    for op_name, feature_columns in predictor_features_by_op.items():
        assert feature_columns == predictor_features
        assert "batch_request_num_tokens" not in feature_columns, op_name
    assert len(predictor_features) == 19


def test_exact_feature_lookup_rejects_non_scalar_request_token_vectors() -> None:
    malformed_df = pd.DataFrame(
        {
            "batch_num_tokens": [65],
            "batch_request_num_tokens": [[1, 64]],
            "time_stats.attn_mla_decode.median": [0.069984],
        }
    )

    with pytest.raises(ValueError, match="non-scalar.*batch_request_num_tokens"):
        _build_exact_feature_lookup(
            malformed_df,
            ["batch_num_tokens", "batch_request_num_tokens"],
            "time_stats.attn_mla_decode.median",
        )


def test_mla_predict_for_attention_layer_models_rejects_loaded_model_schema_mismatch() -> None:
    predictor = _build_mla_predictor()
    expected_features = list(
        get_enabled_predictor_feature_columns(LATENT_MLA_ATTENTION_FAMILY)[
            "attn_mla_decode"
        ]
    )
    wrong_features = [*expected_features, "batch_request_num_tokens"]
    predictor._models = {"attn_mla_decode": _FakeMlaModel(wrong_features)}

    with pytest.raises(
        ValueError,
        match="MLA attention model feature schema mismatch.*attn_mla_decode",
    ):
        predictor._predict_for_attention_layer_models()


def test_mla_predict_for_attention_layer_models_rejects_loaded_model_feature_count_mismatch() -> None:
    predictor = _build_mla_predictor()
    expected_features = list(
        get_enabled_predictor_feature_columns(LATENT_MLA_ATTENTION_FAMILY)[
            "attn_mla_decode"
        ]
    )
    model = _FakeMlaModel(expected_features)
    model.n_features_in_ = len(expected_features) + 1
    predictor._models = {"attn_mla_decode": model}

    with pytest.raises(
        ValueError,
        match="MLA attention model feature count mismatch.*attn_mla_decode",
    ):
        predictor._predict_for_attention_layer_models()


def test_mla_train_attention_layer_models_rejects_sparse_target_with_no_observed_rows() -> None:
    predictor = _build_mla_predictor()
    sparse_df = build_frontier_mla_profile_dataframe(
        h800_mla_mixed_rows(),
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )
    sparse_df["time_stats.attn_mla_v_up_proj.median"] = pd.NA
    predictor._load_attention_df = lambda _path: sparse_df
    predictor._train_model = (
        lambda *, model_name, df, feature_cols, target_col: _FakeMlaModel(feature_cols)
    )

    with pytest.raises(
        ValueError,
        match="attn_mla_v_up_proj[\\s\\S]*All-NaN columns",
    ):
        predictor._train_attention_layer_models()


def test_mla_train_attention_layer_models_rejects_nan_features_after_target_filtering() -> None:
    predictor = _build_mla_predictor()
    sparse_df = build_frontier_mla_profile_dataframe(
        h800_mla_mixed_rows(),
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )
    target_column = "time_stats.attn_mla_decode.median"
    row_index = sparse_df.index[sparse_df[target_column].notna()][0]
    sparse_df.loc[row_index, "max_seqlen_q"] = pd.NA
    predictor._load_attention_df = lambda _path: sparse_df

    with pytest.raises(ValueError, match="feature columns contain NaN after target filtering"):
        predictor._train_attention_layer_models()


def test_mla_predictor_training_respects_kernel_only_measurement_type(tmp_path) -> None:
    input_file = tmp_path / "kernel_only_attention.csv"
    kernel_only_df = _build_imported_mla_profile_df(
        measurement_type=MeasurementType.KERNEL_ONLY,
    )
    kernel_only_df.to_csv(input_file, index=False)

    predictor = _build_mla_predictor(measurement_type=MeasurementType.KERNEL_ONLY)
    filtered = predictor._load_attention_df(str(input_file))

    assert len(filtered) == 1
    assert set(filtered["measurement_type"].unique()) == {
        MeasurementType.KERNEL_ONLY.value,
    }

    mismatched_predictor = _build_mla_predictor(
        measurement_type=MeasurementType.CUDA_EVENT
    )
    with pytest.raises(ValueError, match="measurement_type mismatch"):
        mismatched_predictor._load_attention_df(str(input_file))
