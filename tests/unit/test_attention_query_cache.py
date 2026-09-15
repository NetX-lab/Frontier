from __future__ import annotations

from types import SimpleNamespace

from frontier.entities.time_components import AttentionTime
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.types import ClusterType


class _Predictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _Request:
    num_prefill_tokens = 8
    num_decode_tokens = 4
    num_processed_tokens = 8
    num_processed_decode_tokens = 0
    num_prefill_tokens_cached = 0
    is_prefill_complete = False
    state_init_mode = "zero"


class _Batch:
    size = 1
    total_num_tokens = 12
    num_prefill_tokens = 8
    num_decode_tokens = 4
    total_num_tokens_rounded = 16
    ffn_compute_total_tokens = 12
    afd_stage_idx = 0
    state_init_mode = "zero"
    num_stateful_requests = 0
    requests = [_Request()]

    def get_effective_total_tokens_for_compute(self, _cluster_type):
        return self.total_num_tokens


def _build_predictor(call_log: list[int]):
    predictor = object.__new__(_Predictor)
    predictor._active_measurement_type = SimpleNamespace(value="cuda_event")
    predictor._runtime_stack_signature = "synthetic-runtime"
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=2,
        attn_dp=1,
        num_pipeline_stages=1,
    )
    predictor._model_manager = SimpleNamespace(
        runtime_stack_signature="synthetic-runtime",
        model_identity="fixture-model",
        artifact_identity="fixture-artifact",
    )
    predictor._model_config = SimpleNamespace(
        embedding_dim=128,
        num_q_heads=8,
        num_kv_heads=8,
        head_dim=16,
        torch_dtype="bfloat16",
        dtype="bfloat16",
        kv_cache_dtype="auto",
        partial_rotary_factor=1.0,
    )

    def get_layer_attention_spec(layer_id: int):
        family_id = "gated_delta_net" if layer_id == 7 else "dense_attention"
        return SimpleNamespace(family_id=family_id, variant_id="fixture")

    predictor._model_config.get_layer_attention_spec = get_layer_attention_spec

    def predict_attention(*, batch, layer_id: int, cluster_type):
        del batch, cluster_type
        call_log.append(layer_id)
        return AttentionTime(
            attention_prefill_execution_time=float(layer_id + 1),
            attention_decode_execution_time=2.0,
        )

    predictor.predict_attention_layer_time = predict_attention
    return predictor


def test_equal_physical_attention_queries_reuse_only_numeric_result() -> None:
    calls: list[int] = []
    predictor = _build_predictor(calls)
    batch = _Batch()

    first = predictor._predict_attention_layer_time_with_query_cache(
        batch=batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )
    second = predictor._predict_attention_layer_time_with_query_cache(
        batch=batch,
        layer_id=4,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert calls == [0]
    assert first is not second
    first.attention_prefill_execution_time = 99.0
    assert second.attention_prefill_execution_time == 1.0


def test_attention_query_cache_misses_for_context_phase_tp_and_family_changes() -> None:
    calls: list[int] = []
    predictor = _build_predictor(calls)
    batch = _Batch()

    def predict() -> None:
        predictor._predict_attention_layer_time_with_query_cache(
            batch=batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )

    predict()
    batch.requests[0].num_processed_tokens = 9
    predict()
    batch.num_prefill_tokens = 0
    batch.num_decode_tokens = 12
    predict()
    predictor._replica_config.attn_tensor_parallel_size = 4
    predict()
    predictor._predict_attention_layer_time_with_query_cache(
        batch=batch,
        layer_id=7,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert calls == [0, 0, 0, 0, 7]


def test_attention_query_cache_has_bounded_lru_lifetime() -> None:
    calls: list[int] = []
    predictor = _build_predictor(calls)
    predictor._attention_query_cache_capacity = 2
    batch = _Batch()

    for token_count in (12, 13, 14, 15):
        batch.total_num_tokens = token_count
        predictor._predict_attention_layer_time_with_query_cache(
            batch=batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )

    assert len(predictor._attention_query_cache) == 2
    assert predictor._attention_query_cache_misses == 4
    assert predictor._attention_query_cache_hits == 0
