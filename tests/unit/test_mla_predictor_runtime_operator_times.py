from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_feature_columns,
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_names,
)
from frontier.attention.trace_mapping import get_attention_trace_op_times
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
    _build_exact_feature_lookup,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.profiling.attention.vllm_mla_profile_importer import (
    build_frontier_mla_profile_dataframe,
)
from frontier.types import ClusterType, MeasurementType
from tests.unit.mla_h800_fixture import (
    H800_MIXED_BATCH1_TIMES_MS,
    h800_mla_mixed_rows,
)


class _DummySklearnPredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _IdentityQuantizationManager:
    def adjust_compute_time(self, _op_name, value, _cluster_type):
        return value

    def adjust_tensor_size(self, _op_name, value, _cluster_type):
        return value


class _ConstantExactModel:
    def __init__(self, value: float, feature_names: tuple[str, ...]):
        self._value = float(value)
        self._frontier_feature_names = list(feature_names)
        self.n_features_in_ = len(feature_names)

    def predict(self, dataframe):
        return [self._value for _ in range(len(dataframe))]


class _DummyBatch:
    id = 17
    size = 1
    num_tokens = [1]
    total_num_tokens = 1
    num_prefill_tokens = 0
    num_decode_tokens = 1
    is_idle = False
    is_moe = False
    spec_decode_metadata = None
    requests = [
        SimpleNamespace(
            id=101,
            num_processed_tokens=64,
            num_prefill_tokens=64,
            num_decode_tokens=128,
            num_processed_decode_tokens=1,
            is_prefill_complete=True,
        )
    ]


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


def _sample_vllm_rows(
    *,
    batch_size: int = 1,
    batch_num_tokens: int = 1,
    batch_num_prefill_tokens: int = 0,
    batch_num_decode_tokens: int = 1,
    batch_request_num_tokens: list[int] | None = None,
    max_seqlen_q: int = 1,
    max_seqlen_k: int = 65,
    num_actual_tokens: int = 1,
) -> list[dict[str, object]]:
    if batch_request_num_tokens is None:
        batch_request_num_tokens = [batch_num_tokens]
    rows: list[dict[str, object]] = []
    for idx, scope in enumerate(
        get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY),
        1,
    ):
        rows.append(
            {
                "batch_id": 7,
                "batch_size": batch_size,
                "batch_num_tokens": batch_num_tokens,
                "batch_num_prefill_tokens": batch_num_prefill_tokens,
                "batch_num_decode_tokens": batch_num_decode_tokens,
                "batch_request_num_tokens": batch_request_num_tokens,
                "op_name": scope,
                "cuda_time_ms": float(idx) / 100.0,
                "count": 1,
                "meta": _base_meta(
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    num_actual_tokens=num_actual_tokens,
                ),
            }
        )
    return rows


def _build_imported_mla_profile_df(rows: list[dict[str, object]] | None = None):
    return build_frontier_mla_profile_dataframe(
        _sample_vllm_rows() if rows is None else rows,
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )


def _build_mla_predictor(rows: list[dict[str, object]] | None = None):
    predictor = _DummySklearnPredictor.__new__(_DummySklearnPredictor)
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._runtime_cache = defaultdict(lambda: defaultdict(dict))
    predictor._block_size = 64
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
    )
    predictor._model_config = SimpleNamespace(
        use_mla=True,
        num_q_heads=128,
        num_kv_heads=128,
        head_dim=None,
        embedding_dim=73728,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        kv_lora_rank=512,
        v_head_dim=128,
        max_position_embeddings=163840,
        get_head_dim=lambda: 576,
        get_qk_head_dim=lambda: 192,
        get_model_architecture_profile=ModelArchitectureProfile.generic,
    )
    predictor._supports_operation = lambda operation: operation == "attention"
    predictor._log_architecture_attention_shape = lambda _batch: None
    predictor._get_attention_layer_pre_proj_execution_time = lambda _batch: 0.0
    predictor._get_attention_layer_post_proj_execution_time = lambda _batch: 0.0
    predictor._get_attention_rope_execution_time = lambda _batch: 0.0
    predictor._get_attn_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_mlp_layer_up_proj_execution_time = lambda _batch: 0.0
    predictor._get_mlp_layer_down_proj_execution_time = lambda _batch: 0.0
    predictor._get_mlp_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_add_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
    predictor._get_pipeline_parallel_communication_time = lambda _batch: 0.0
    predictor._get_tensor_parallel_communication_time = lambda _batch: 0.0
    predictor._get_pp_producer_send_path_runtime_time = (
        lambda _batch, _stage_id: 0.0
    )
    predictor._get_pp_receiver_head_runtime_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_prefill_consumer_active_runtime_time = (
        lambda _batch, _stage_id: 0.0
    )
    predictor._get_pp_stage_boundary_handoff_time = lambda _batch, _stage_id: 0.0
    predictor._get_mtp_terminal_overshoot_time = lambda *_args, **_kwargs: 0.0
    predictor._should_include_spec_decode_proposer_overhead = lambda _batch: False
    predictor._select_measurement_type_for_batch = (
        lambda _batch: MeasurementType.CUDA_EVENT
    )
    predictor._require_predictions_for_measurement_type = (
        lambda *_args, **_kwargs: None
    )
    predictor._activate_measurement_type = (
        lambda measurement_type: setattr(
            predictor,
            "_active_measurement_type",
            measurement_type,
        )
    )
    predictor._validate_prediction_value = (
        lambda value, _op_name, _batch, _context: value
    )

    df = _build_imported_mla_profile_df(rows)
    feature_columns_by_op = get_enabled_predictor_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    )
    target_columns = dict(
        zip(
            get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY),
            get_enabled_predictor_median_columns(LATENT_MLA_ATTENTION_FAMILY),
        )
    )
    predictions = {}
    for op_name, feature_columns in feature_columns_by_op.items():
        feature_columns = tuple(feature_columns)
        target_column = target_columns[op_name]
        source_value = float(df[target_column].dropna().iloc[0])
        model = _ConstantExactModel(source_value + 1.0, feature_columns)
        predictions[op_name] = {
            "_on_demand_prediction": True,
            "_n_features": len(feature_columns),
            "_model": model,
            "_feature_names": list(feature_columns),
            "_exact_lookup": _build_exact_feature_lookup(
                df,
                list(feature_columns),
                target_column,
            ),
        }
    predictor._predictions = predictions
    predictor._source_mla_profile_df = df
    return predictor


def _decode_only_expected_op_times() -> dict[str, float]:
    return {
        "attn_mla_kv_cache_save": 0.01,
        "attn_mla_prefill_kv_up_proj": 0.0,
        "attn_mla_prefill": 0.0,
        "attn_mla_decode_q_latent_proj": 0.04,
        "attn_mla_decode": 0.05,
        "attn_mla_v_up_proj": 0.06,
    }


def _prefill_only_expected_op_times() -> dict[str, float]:
    return {
        "attn_mla_kv_cache_save": 0.01,
        "attn_mla_prefill_kv_up_proj": 0.02,
        "attn_mla_prefill": 0.03,
        "attn_mla_decode_q_latent_proj": 0.0,
        "attn_mla_decode": 0.0,
        "attn_mla_v_up_proj": 0.0,
    }


class _PrefillDummyBatch:
    id = 23
    size = 1
    num_tokens = [2]
    total_num_tokens = 2
    num_prefill_tokens = 2
    num_decode_tokens = 0
    is_idle = False
    is_moe = False
    spec_decode_metadata = None
    requests = [
        SimpleNamespace(
            id=202,
            num_processed_tokens=63,
            num_prefill_tokens=2,
            num_decode_tokens=128,
            num_processed_decode_tokens=0,
            is_prefill_complete=False,
        )
    ]


def test_mla_runtime_predicts_six_operator_times_from_imported_exact_row(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_quantization_manager",
        lambda: _IdentityQuantizationManager(),
    )

    predictor = _build_mla_predictor()
    attention_time = predictor.predict_attention_layer_time(
        batch=_DummyBatch(),
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert attention_time.operator_times is not None
    assert attention_time.operator_times.op_times == pytest.approx(
        _decode_only_expected_op_times()
    )
    assert attention_time.attention_prefill_execution_time == 0.0
    assert attention_time.attention_decode_execution_time == 0.0
    assert attention_time.attention_kv_cache_save_execution_time == 0.0
    assert attention_time.attention_layer_pre_proj_execution_time == 0.0
    assert attention_time.attention_layer_post_proj_execution_time == 0.0


def test_mla_stage_execution_time_preserves_operator_times_for_trace_mapping(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_quantization_manager",
        lambda: _IdentityQuantizationManager(),
    )

    predictor = _build_mla_predictor()
    execution_time = predictor.predict_stage_execution_time(
        batch=_DummyBatch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=3,
    )

    assert execution_time.attention_operator_times is not None
    op_times = get_attention_trace_op_times(
        execution_time,
        LATENT_MLA_ATTENTION_FAMILY,
        skip_zero=False,
    )
    assert [(op.name, time_ms) for op, time_ms in op_times] == [
        ("attn_mla_kv_cache_save", pytest.approx(0.03)),
        ("attn_mla_prefill_kv_up_proj", pytest.approx(0.0)),
        ("attn_mla_prefill", pytest.approx(0.0)),
        ("attn_mla_decode_q_latent_proj", pytest.approx(0.12)),
        ("attn_mla_decode", pytest.approx(0.15)),
        ("attn_mla_v_up_proj", pytest.approx(0.18)),
    ]


def test_mla_runtime_predicts_prefill_shape_from_imported_exact_row(monkeypatch) -> None:
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_quantization_manager",
        lambda: _IdentityQuantizationManager(),
    )
    rows = _sample_vllm_rows(
        batch_num_tokens=2,
        batch_num_prefill_tokens=2,
        batch_num_decode_tokens=0,
        batch_request_num_tokens=[2],
        max_seqlen_q=2,
        max_seqlen_k=65,
        num_actual_tokens=2,
    )
    predictor = _build_mla_predictor(rows)

    attention_time = predictor.predict_attention_layer_time(
        batch=_PrefillDummyBatch(),
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert attention_time.operator_times is not None
    assert attention_time.operator_times.op_times == pytest.approx(
        _prefill_only_expected_op_times()
    )


def test_mla_runtime_fails_fast_when_imported_feature_row_is_missing() -> None:
    predictor = _build_mla_predictor()
    missing_row_batch = _DummyBatch()
    missing_row_batch.requests = [
        SimpleNamespace(
            id=101,
            num_processed_tokens=65,
            num_prefill_tokens=64,
            num_decode_tokens=128,
            num_processed_decode_tokens=1,
            is_prefill_complete=True,
        )
    ]

    with pytest.raises(ValueError, match="No exact MLA profiling row"):
        predictor.predict_attention_layer_time(
            batch=missing_row_batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )


def test_mla_runtime_exact_miss_rejects_model_without_frontier_model_hash() -> None:
    predictor = _build_mla_predictor()
    missing_row_batch = _DummyBatch()
    missing_row_batch.requests = [
        SimpleNamespace(
            id=101,
            num_processed_tokens=65,
            num_prefill_tokens=64,
            num_decode_tokens=128,
            num_processed_decode_tokens=1,
            is_prefill_complete=True,
        )
    ]
    for model_info in predictor._predictions.values():
        model_info["_model"] = _ConstantExactModel(
            0.123,
            tuple(model_info["_feature_names"]),
        )

    with pytest.raises(ValueError, match="No trained MLA prediction model"):
        predictor.predict_attention_layer_time(
            batch=missing_row_batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )


def test_mla_runtime_exact_miss_uses_trained_model_when_schema_matches(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_quantization_manager",
        lambda: _IdentityQuantizationManager(),
    )

    predictor = _build_mla_predictor()
    missing_row_batch = _DummyBatch()
    missing_row_batch.requests = [
        SimpleNamespace(
            id=101,
            num_processed_tokens=65,
            num_prefill_tokens=64,
            num_decode_tokens=128,
            num_processed_decode_tokens=1,
            is_prefill_complete=True,
        )
    ]

    learned_values = {
        scope: 0.2 + idx / 100.0
        for idx, scope in enumerate(
            get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY),
            1,
        )
    }
    for op_name, model_info in predictor._predictions.items():
        trained_model = _ConstantExactModel(
            learned_values[op_name],
            tuple(model_info["_feature_names"]),
        )
        trained_model._frontier_model_hash = f"trained-{op_name}"
        model_info["_model"] = trained_model

    attention_time = predictor.predict_attention_layer_time(
        batch=missing_row_batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    expected = dict(learned_values)
    expected["attn_mla_prefill_kv_up_proj"] = 0.0
    expected["attn_mla_prefill"] = 0.0
    assert attention_time.operator_times is not None
    assert attention_time.operator_times.op_times == pytest.approx(expected)


def test_mla_runtime_fails_fast_when_query_shape_differs_but_kv_extent_matches() -> None:
    predictor = _build_mla_predictor()
    mismatched_query_batch = _DummyBatch()
    mismatched_query_batch.num_tokens = [2]
    mismatched_query_batch.total_num_tokens = 2
    mismatched_query_batch.num_prefill_tokens = 2
    mismatched_query_batch.num_decode_tokens = 0
    mismatched_query_batch.requests = [
        SimpleNamespace(
            id=101,
            num_processed_tokens=63,
            num_prefill_tokens=2,
            num_decode_tokens=128,
            num_processed_decode_tokens=0,
            is_prefill_complete=False,
        )
    ]

    with pytest.raises(ValueError, match="No exact MLA profiling row"):
        predictor.predict_attention_layer_time(
            batch=mismatched_query_batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )


def test_mla_runtime_rejects_non_sequence_request_token_counts() -> None:
    predictor = _build_mla_predictor()
    malformed_batch = _DummyBatch()
    malformed_batch.num_tokens = 1

    with pytest.raises(ValueError, match="per-request sequence"):
        predictor.predict_attention_layer_time(
            batch=malformed_batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )


def test_mla_runtime_rejects_missing_request_processed_tokens() -> None:
    predictor = _build_mla_predictor()
    malformed_batch = _DummyBatch()
    malformed_batch.requests = [
        SimpleNamespace(
            id=101,
            num_prefill_tokens=64,
            num_decode_tokens=128,
            num_processed_decode_tokens=1,
            is_prefill_complete=True,
        )
    ]

    with pytest.raises(ValueError, match="request.num_processed_tokens"):
        predictor.predict_attention_layer_time(
            batch=malformed_batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )


def test_mla_runtime_rejects_missing_batch_token_totals_with_value_error() -> None:
    predictor = _build_mla_predictor()
    malformed_batch = SimpleNamespace(
        id=17,
        size=1,
        num_tokens=[1],
        num_prefill_tokens=0,
        num_decode_tokens=1,
        is_idle=False,
        is_moe=False,
        spec_decode_metadata=None,
        requests=[
            SimpleNamespace(
                id=101,
                num_processed_tokens=64,
                num_prefill_tokens=64,
                num_decode_tokens=128,
                num_processed_decode_tokens=1,
                is_prefill_complete=True,
            )
        ],
    )

    with pytest.raises(ValueError, match="batch.total_num_tokens"):
        predictor.predict_attention_layer_time(
            batch=malformed_batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )

class _MixedDummyBatch:
    id = 31
    size = 2
    num_tokens = [1, 64]
    total_num_tokens = 65
    num_prefill_tokens = 64
    num_decode_tokens = 1
    is_idle = False
    is_moe = False
    spec_decode_metadata = None
    requests = [
        SimpleNamespace(
            id=301,
            num_processed_tokens=64,
            num_prefill_tokens=64,
            num_decode_tokens=128,
            num_processed_decode_tokens=1,
            is_prefill_complete=True,
        ),
        SimpleNamespace(
            id=302,
            num_processed_tokens=0,
            num_prefill_tokens=64,
            num_decode_tokens=128,
            num_processed_decode_tokens=0,
            is_prefill_complete=False,
        ),
    ]


def test_mla_runtime_predicts_mixed_batch_with_op_specific_exact_keys(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_quantization_manager",
        lambda: _IdentityQuantizationManager(),
    )

    predictor = _build_mla_predictor(h800_mla_mixed_rows())
    attention_time = predictor.predict_attention_layer_time(
        batch=_MixedDummyBatch(),
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert attention_time.operator_times is not None
    assert attention_time.operator_times.op_times == pytest.approx(
        H800_MIXED_BATCH1_TIMES_MS
    )


def test_mla_exact_hit_uses_measured_row_even_when_trained_model_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "frontier.execution_time_predictor.sklearn_execution_time_predictor.get_quantization_manager",
        lambda: _IdentityQuantizationManager(),
    )

    predictor = _build_mla_predictor(h800_mla_mixed_rows())
    for op_name, model_info in predictor._predictions.items():
        trained_model = _ConstantExactModel(
            999.0,
            tuple(model_info["_feature_names"]),
        )
        trained_model._frontier_model_hash = f"trained-h800-{op_name}"
        model_info["_model"] = trained_model

    attention_time = predictor.predict_attention_layer_time(
        batch=_MixedDummyBatch(),
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert attention_time.operator_times is not None
    assert attention_time.operator_times.op_times == pytest.approx(
        H800_MIXED_BATCH1_TIMES_MS
    )
    assert max(attention_time.operator_times.op_times.values()) < 1.0


def test_mla_runtime_exact_miss_rejects_prediction_metadata_schema_mismatch() -> None:
    predictor = _build_mla_predictor()
    predictor._predictions["attn_mla_decode"]["_feature_names"] = [
        *predictor._predictions["attn_mla_decode"]["_feature_names"],
        "batch_request_num_tokens",
    ]

    with pytest.raises(
        ValueError,
        match="MLA predictor feature schema mismatch.*attn_mla_decode",
    ):
        predictor.predict_attention_layer_time(
            batch=_DummyBatch(),
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )


def test_mla_runtime_requires_request_phase_metadata_for_exact_keys() -> None:
    predictor = _build_mla_predictor()
    missing_phase_batch = _DummyBatch()
    missing_phase_batch.requests = [
        SimpleNamespace(
            id=101,
            num_processed_tokens=64,
            num_prefill_tokens=64,
            num_decode_tokens=128,
            num_processed_decode_tokens=1,
        )
    ]

    with pytest.raises(ValueError, match="request.is_prefill_complete"):
        predictor.predict_attention_layer_time(
            batch=missing_phase_batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )


def test_mla_runtime_rejects_phase_partition_token_mismatch() -> None:
    predictor = _build_mla_predictor()
    inconsistent_phase_batch = _DummyBatch()
    inconsistent_phase_batch.num_prefill_tokens = 1
    inconsistent_phase_batch.num_decode_tokens = 0
    inconsistent_phase_batch.requests = [
        SimpleNamespace(
            id=101,
            num_processed_tokens=64,
            num_prefill_tokens=64,
            num_decode_tokens=128,
            num_processed_decode_tokens=1,
            is_prefill_complete=True,
        )
    ]

    with pytest.raises(ValueError, match="is_prefill_complete partition"):
        predictor.predict_attention_layer_time(
            batch=inconsistent_phase_batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )
