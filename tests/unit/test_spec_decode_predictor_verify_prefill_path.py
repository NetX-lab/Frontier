from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from frontier.config.config import SpeculativeDecodingConfig
from frontier.entities.batch import Batch, DecodeCudaGraphMetadata, SpecDecodeBatchMetadata
from frontier.entities.request import Request
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import ClusterType, MeasurementType


def _dense_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        use_mla=False,
        num_q_heads=32,
        num_kv_heads=8,
        get_model_architecture_profile=ModelArchitectureProfile.generic,
    )


class _DummyPredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def _build_predictor(
    *,
    proposer_overhead_ms: float = 0.0,
) -> _DummyPredictor:
    predictor = _DummyPredictor.__new__(_DummyPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=128)
    predictor._replica_config = SimpleNamespace(
        speculative_decoding_config=SpeculativeDecodingConfig(
            enabled=True,
            method="eagle",
            proposer_overhead_ms_by_method={"eagle": proposer_overhead_ms},
        )
    )
    predictor._supports_operation = lambda _operation: True
    predictor._attention_decode_batching_overhead_fraction = 0.0
    predictor._attention_prefill_batching_overhead_fraction = 0.0
    predictor._enable_dummy_mode = False
    predictor._dummy_execution_time = 0.0
    predictor._model_config = _dense_model_config()
    predictor._log_architecture_attention_shape = lambda _batch: None
    predictor._get_attention_layer_pre_proj_execution_time = lambda _batch: 0.0
    predictor._get_attention_layer_post_proj_execution_time = lambda _batch: 0.0
    predictor._get_attention_rope_execution_time = lambda _batch: 0.0
    predictor._get_attention_kv_cache_save_execution_time = lambda _batch: 0.0
    predictor._get_attn_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._predictions = {
        "attn_prefill": {
            (512, 9): 7.0,
            (640, 4): 11.0,
        },
        "attn_decode": {
            (3, 640): 13.0,
            (1, 512): 5.0,
        },
        "attn_prefill_mixed": {
            "_on_demand_prediction": True,
        },
        "attn_decode_in_mixed": {
            "_on_demand_prediction": True,
        },
    }
    predictor._on_demand_calls = []

    def _on_demand_prediction(model_name: str, features: dict) -> float:
        predictor._on_demand_calls.append((model_name, dict(features)))
        if model_name == "attn_prefill_mixed":
            return 18.0
        if model_name == "attn_decode_in_mixed":
            return 5.0
        raise ValueError(f"Unexpected model_name={model_name}")

    predictor._get_on_demand_prediction = _on_demand_prediction
    return predictor


def _build_prefill_complete_request(
    *,
    num_processed_tokens: int,
) -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=16,
        num_decode_tokens=1024,
        num_processed_tokens=num_processed_tokens,
    )
    request._is_prefill_complete = True
    return request


def _build_prefill_request(
    *,
    num_processed_tokens: int,
) -> Request:
    return Request(
        arrived_at=0.0,
        num_prefill_tokens=16,
        num_decode_tokens=1024,
        num_processed_tokens=num_processed_tokens,
    )


def _build_spec_decode_batch() -> Batch:
    requests = [
        _build_prefill_complete_request(num_processed_tokens=512),
        _build_prefill_complete_request(num_processed_tokens=500),
        _build_prefill_complete_request(num_processed_tokens=640),
    ]
    batch = Batch(
        replica_id=0,
        requests=requests,
        num_tokens=[1, 1, 1],
        is_moe=False,
    )
    batch.spec_decode_metadata = SpecDecodeBatchMetadata(
        method="eagle",
        planned_draft_tokens_per_request=[2, 0, 1],
        verify_tokens_per_request=[3, 1, 2],
        accepted_draft_tokens_per_request=[1, 0, 1],
        rejected_draft_tokens_per_request=[1, 0, 0],
        committed_tokens_per_request=[2, 1, 2],
        uses_lookahead_slots=True,
    )
    return batch


def test_spec_verify_requests_use_mixed_prefill_predictor_for_true_mixed_batch() -> None:
    predictor = _build_predictor()
    batch = _build_spec_decode_batch()

    verify_prefill_time = predictor._get_spec_verify_attention_prefill_execution_time(
        batch
    )

    assert verify_prefill_time == 18.0
    assert predictor._on_demand_calls[0][0] == "attn_prefill_mixed"
    assert predictor._on_demand_calls[0][1]["kv_cache_size"] == 640


def test_pure_verify_multi_request_batch_uses_batched_verify_prefill_predictor() -> None:
    predictor = _build_predictor()
    batch = Batch(
        replica_id=0,
        requests=[
            _build_prefill_complete_request(num_processed_tokens=512),
            _build_prefill_complete_request(num_processed_tokens=500),
            _build_prefill_complete_request(num_processed_tokens=640),
        ],
        num_tokens=[3, 3, 3],
        is_moe=False,
    )
    batch.spec_decode_metadata = SpecDecodeBatchMetadata(
        method="eagle",
        planned_draft_tokens_per_request=[2, 2, 2],
        verify_tokens_per_request=[3, 3, 3],
        accepted_draft_tokens_per_request=[1, 1, 1],
        rejected_draft_tokens_per_request=[1, 1, 1],
        committed_tokens_per_request=[2, 2, 2],
        uses_lookahead_slots=True,
    )

    verify_prefill_time = predictor._get_spec_verify_attention_prefill_execution_time(
        batch
    )

    assert verify_prefill_time == 18.0
    assert predictor._on_demand_calls[0][0] == "attn_prefill_mixed"
    assert predictor._on_demand_calls[0][1]["kv_cache_size"] == 640


def test_attention_training_uses_true_mixed_rows_for_verify_prefill_model() -> None:
    predictor = _DummyPredictor.__new__(_DummyPredictor)
    predictor._attention_input_file = "attention.csv"
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=128)
    predictor._model_config = _dense_model_config()

    attention_df = pd.DataFrame(
        [
            {
                "is_true_mixed_batch": False,
                "is_mixed_batch": False,
                "is_prefill": True,
                "batch_size": 1,
                "prefill_chunk_size": 64,
                "kv_cache_size": 0,
                "total_tokens": 64,
                "avg_seq_len": 64.0,
                "min_seq_len": 64,
                "max_seq_len": 64,
                "seq_len_variance": 0.0,
                "seq_len_cv": 0.0,
                "time_stats.attn_prefill.median": 0.1,
                "time_stats.attn_decode.median": 0.0,
            },
            {
                "is_true_mixed_batch": False,
                "is_mixed_batch": False,
                "is_prefill": False,
                "batch_size": 1,
                "prefill_chunk_size": 0,
                "kv_cache_size": 128,
                "total_tokens": 1,
                "avg_seq_len": 1.0,
                "min_seq_len": 1,
                "max_seq_len": 1,
                "seq_len_variance": 0.0,
                "seq_len_cv": 0.0,
                "time_stats.attn_prefill.median": 0.0,
                "time_stats.attn_decode.median": 0.02,
            },
            {
                "is_true_mixed_batch": True,
                "is_mixed_batch": False,
                "is_prefill": True,
                "batch_size": 3,
                "prefill_chunk_size": 0,
                "kv_cache_size": 512,
                "total_tokens": 9,
                "avg_seq_len": 3.0,
                "min_seq_len": 3,
                "max_seq_len": 3,
                "seq_len_variance": 0.0,
                "seq_len_cv": 0.0,
                "num_prefill_seqs": 3,
                "num_decode_seqs": 1,
                "decode_batch_size": 1,
                "decode_avg_kv_cache_size": 512,
                "total_prefill_tokens": 9,
                "total_batch_size": 4,
                "batch_composition_ratio": 0.75,
                "time_stats.attn_prefill.median": 0.3,
                "time_stats.attn_decode.median": 0.04,
            },
        ]
    )
    trained = {}

    predictor._load_attention_df = lambda _path: attention_df

    def _record_train_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> object:
        trained[model_name] = {
            "df": df.copy(),
            "feature_cols": list(feature_cols),
            "target_col": target_col,
        }
        return object()

    predictor._train_model = _record_train_model

    models = predictor._train_attention_layer_models()

    assert "attn_prefill_mixed" in models
    mixed_training_df = trained["attn_prefill_mixed"]["df"]
    assert len(mixed_training_df) == 1
    assert bool(mixed_training_df.iloc[0]["is_true_mixed_batch"]) is True
    assert (
        trained["attn_prefill_mixed"]["target_col"]
        == "time_stats.attn_prefill.median"
    )


def test_attention_training_derives_true_mixed_prefill_features_from_list_columns() -> None:
    predictor = _DummyPredictor.__new__(_DummyPredictor)
    predictor._attention_input_file = "attention.csv"
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._config = SimpleNamespace(kv_cache_prediction_granularity=128)
    predictor._model_config = _dense_model_config()

    attention_df = pd.DataFrame(
        [
            {
                "is_true_mixed_batch": False,
                "is_mixed_batch": False,
                "is_prefill": True,
                "batch_size": 1,
                "prefill_chunk_size": 64,
                "kv_cache_size": 0,
                "total_tokens": 64,
                "avg_seq_len": 64.0,
                "min_seq_len": 64,
                "max_seq_len": 64,
                "seq_len_variance": 0.0,
                "seq_len_cv": 0.0,
                "time_stats.attn_prefill.median": 0.1,
                "time_stats.attn_decode.median": 0.0,
            },
            {
                "is_true_mixed_batch": False,
                "is_mixed_batch": False,
                "is_prefill": False,
                "batch_size": 1,
                "prefill_chunk_size": 0,
                "kv_cache_size": 128,
                "total_tokens": 1,
                "avg_seq_len": 1.0,
                "min_seq_len": 1,
                "max_seq_len": 1,
                "seq_len_variance": 0.0,
                "seq_len_cv": 0.0,
                "time_stats.attn_prefill.median": 0.0,
                "time_stats.attn_decode.median": 0.02,
            },
            {
                "is_true_mixed_batch": True,
                "is_mixed_batch": False,
                "is_prefill": True,
                "batch_size": 3,
                "prefill_chunk_size": 0,
                "kv_cache_size": 0,
                "total_tokens": 21,
                "prefill_seq_lens": "[8, 12]",
                "prefill_kv_cache_sizes": "[128, 256]",
                "decode_kv_cache_sizes": "[512]",
                "num_prefill_seqs": 2,
                "num_decode_seqs": 1,
                "decode_batch_size": 1,
                "decode_avg_kv_cache_size": 512,
                "total_prefill_tokens": 20,
                "total_batch_size": 3,
                "batch_composition_ratio": 2 / 3,
                "time_stats.attn_prefill.median": 0.3,
                "time_stats.attn_decode.median": 0.04,
            },
        ]
    )
    trained = {}

    predictor._load_attention_df = lambda _path: attention_df

    def _record_train_model(
        *,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> object:
        trained[model_name] = {
            "df": df.copy(),
            "feature_cols": list(feature_cols),
            "target_col": target_col,
        }
        return object()

    predictor._train_model = _record_train_model

    models = predictor._train_attention_layer_models()

    assert "attn_prefill_mixed" in models
    mixed_training_df = trained["attn_prefill_mixed"]["df"]
    true_mixed_training_row = mixed_training_df[
        mixed_training_df["is_true_mixed_batch"]
    ].iloc[0]
    assert true_mixed_training_row["batch_size"] == 2
    assert true_mixed_training_row["kv_cache_size"] == 256
    assert true_mixed_training_row["total_tokens"] == 20
    assert true_mixed_training_row["avg_seq_len"] == 10.0
    assert true_mixed_training_row["min_seq_len"] == 8
    assert true_mixed_training_row["max_seq_len"] == 12
    assert true_mixed_training_row["seq_len_range"] == 4
    assert (
        true_mixed_training_row[trained["attn_prefill_mixed"]["feature_cols"]]
        .notna()
        .all()
    )


def test_batch_prefill_mixed_features_include_rounded_avg_kv_cache_size() -> None:
    predictor = _build_predictor()
    batch = Batch(
        replica_id=0,
        requests=[
            _build_prefill_request(num_processed_tokens=129),
            _build_prefill_request(num_processed_tokens=257),
        ],
        num_tokens=[4, 8],
        is_moe=False,
    )

    features = predictor._get_batch_prefill_mixed_features(batch)

    assert features["kv_cache_size"] == 256


def test_spec_normal_decode_requests_use_mixed_decode_predictor_for_true_mixed_batch() -> None:
    predictor = _build_predictor()
    batch = _build_spec_decode_batch()

    normal_decode_time = predictor._get_spec_normal_decode_attention_execution_time(batch)

    assert normal_decode_time == 5.0
    assert predictor._on_demand_calls[0][0] == "attn_decode_in_mixed"


def test_predict_attention_layer_time_aggregates_verify_and_decode_with_sum_rule() -> None:
    predictor = _build_predictor(proposer_overhead_ms=0.0)
    batch = _build_spec_decode_batch()

    attention_time = predictor.predict_attention_layer_time(
        batch=batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert attention_time.attention_prefill_execution_time == 18.0
    assert attention_time.attention_decode_execution_time == 5.0
    assert attention_time.total_time() == 23.0


def test_predict_attention_layer_time_excludes_method_aware_proposer_overhead() -> None:
    predictor = _build_predictor(proposer_overhead_ms=1.25)
    batch = _build_spec_decode_batch()

    attention_time = predictor.predict_attention_layer_time(
        batch=batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    # Proposer overhead is batch-level and must not be folded into per-layer attention.
    assert attention_time.attention_prefill_execution_time == 18.0
    assert attention_time.attention_decode_execution_time == 5.0
    assert attention_time.total_time() == 23.0


def test_predict_attention_layer_time_uses_hybrid_family_for_spec_piecewise_diagnostic() -> None:
    predictor = _build_predictor(proposer_overhead_ms=0.0)
    predictor._predictions_eager = {}
    predictor._predictions_kernel_only = {}
    predictor._models_eager = {}
    predictor._models_kernel_only = {}
    predictor._compute_input_file_eager = None
    predictor._compute_input_file_kernel_only = None
    predictor._attention_input_file_eager = None
    predictor._attention_input_file_kernel_only = None
    predictor._moe_input_file_eager = None
    predictor._moe_input_file_kernel_only = None
    predictor._models = {}
    predictor._predictions = {}
    predictor._activate_measurement_type(MeasurementType.KERNEL_ONLY)

    batch = _build_spec_decode_batch()
    batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
        config_mode="piecewise",
        runtime_mode="PIECEWISE",
        capture_hit=True,
        is_mixed_batch=True,
        original_total_tokens=batch.total_num_tokens,
        padded_total_tokens=8,
        original_decode_batch_size=len(batch.requests),
        padded_decode_batch_size=8,
    )

    call_log = []

    def _expect_measurement(name: str, expected: MeasurementType, return_value: float):
        def _inner(_batch):
            call_log.append((name, predictor._active_measurement_type))
            assert predictor._active_measurement_type == expected
            return return_value

        return _inner

    predictor._get_attention_prefill_execution_time = lambda _batch: 0.0
    predictor._get_attention_decode_execution_time = lambda _batch: 0.0
    predictor._get_spec_verify_attention_prefill_execution_time = _expect_measurement(
        "verify_prefill", MeasurementType.CUDA_EVENT, 18.0
    )
    predictor._get_spec_normal_decode_attention_execution_time = _expect_measurement(
        "normal_decode", MeasurementType.CUDA_EVENT, 5.0
    )
    predictor._get_attention_layer_pre_proj_execution_time = _expect_measurement(
        "pre_proj", MeasurementType.KERNEL_ONLY, 2.0
    )
    predictor._get_attention_layer_post_proj_execution_time = _expect_measurement(
        "post_proj", MeasurementType.KERNEL_ONLY, 3.0
    )
    predictor._get_attention_rope_execution_time = _expect_measurement(
        "rope", MeasurementType.KERNEL_ONLY, 4.0
    )
    predictor._get_attention_kv_cache_save_execution_time = _expect_measurement(
        "kv_cache_save", MeasurementType.CUDA_EVENT, 6.0
    )
    predictor._get_attn_norm_layer_act_execution_time = _expect_measurement(
        "attn_norm", MeasurementType.KERNEL_ONLY, 1.0
    )

    attention_time = predictor.predict_attention_layer_time(
        batch=batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert attention_time.attention_prefill_execution_time == 18.0
    assert attention_time.attention_decode_execution_time == 5.0
    assert attention_time.attention_layer_pre_proj_execution_time == 2.0
    assert attention_time.attention_kv_cache_save_execution_time == 6.0
    assert attention_time.total_time() == 39.0
    assert predictor._active_measurement_type == MeasurementType.KERNEL_ONLY
    assert call_log == [
        ("verify_prefill", MeasurementType.CUDA_EVENT),
        ("normal_decode", MeasurementType.CUDA_EVENT),
        ("pre_proj", MeasurementType.KERNEL_ONLY),
        ("post_proj", MeasurementType.KERNEL_ONLY),
        ("rope", MeasurementType.KERNEL_ONLY),
        ("kv_cache_save", MeasurementType.CUDA_EVENT),
        ("attn_norm", MeasurementType.KERNEL_ONLY),
    ]
