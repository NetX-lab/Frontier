"""Pinned eager V4 workloads and synthetic component costs, not native timing."""

from types import SimpleNamespace

import pytest

from frontier.attention.deepseek_v4 import build_deepseek_v4_attention_workload
from frontier.config.model_config import BaseModelConfig
from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
from frontier.entities import Batch, Request
from frontier.entities.batch import DecodeCudaGraphMetadata, EPBatchGroup
from frontier.entities.time_components import AttentionOperatorTimes
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType


@pytest.fixture
def model():
    return BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")


@pytest.fixture
def replica():
    return SimpleNamespace(
        num_pipeline_stages=1, attn_tensor_parallel_size=8, attn_dp=8,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=64,
        attention_backend="FLASHMLA_SPARSE_DSV4",
        speculative_decoding_config=SimpleNamespace(enabled=False),
    )


def source_batch(queries, contexts, decoding=None):
    if decoding is None:
        decoding = [False] * len(queries)
    return Batch(
        replica_id=0,
        requests=[Request(0.0, max(1, c + q), 4) for q, c in zip(queries, contexts)],
        num_tokens=queries, num_context_tokens=contexts,
        request_is_decoding=decoding, is_moe=True,
    )


def workload(model, replica, batch, layer=0, **overrides):
    options = dict(
        cluster_type=ClusterType.MONOLITHIC, execution_mode="eager",
        has_attention_metadata=True, prefix_caching=False,
        decode_context_parallel_size=1, prefill_context_parallel_size=1,
    )
    options.update(overrides)
    return build_deepseek_v4_attention_workload(
        model, replica, batch, layer, **options,
    )


def test_attention_uses_real_rows_and_flashmla_padded_heads(model, replica):
    result = workload(model, replica, source_batch([17], [120]))
    assert result["num_tokens"] == 17
    assert result["num_local_q_heads"] == 8
    assert result["num_padded_q_heads"] == 64
    calls = result["operators"]
    assert calls["dsv4_fused_qkv_projection"][0]["output_shape"] == (17, 1536)
    assert calls["dsv4_q_projection"][0]["output_shape"] == (17, 8, 512)
    assert calls["dsv4_fused_q_kv_prepare"][0]["output_shape"] == (17, 64, 512)
    assert calls["dsv4_grouped_output_projection"][0]["output_shape"] == (17, 1, 1024)
    assert calls["dsv4_output_projection"][0]["output_shape"] == (17, 4096)


def test_uncompressed_attention_preserves_source_ratio_zero(model, replica):
    result = workload(model, replica, source_batch([17], [120]))
    call = result["operators"]["dsv4_sparse_attention"][0]
    assert call["compression_ratio"] == 0
    assert call["compressed_history_rows"] == (0,)
    assert result["new_compressed_rows"] == (0,)


@pytest.mark.parametrize("sequence,short", [(2051, True), (2052, False)])
def test_csa_selects_exact_eager_threshold(model, replica, sequence, short):
    result = workload(model, replica, source_batch([1], [sequence - 1], [True]), 2)
    operators = result["operators"]
    assert ("dsv4_index_short_context_fill" in operators) is short
    assert ("dsv4_index_query_projection" in operators) is not short
    assert ("dsv4_index_scores" in operators) is not short
    assert ("dsv4_index_topk" in operators) is not short
    assert "dsv4_index_weights_projection" in operators
    assert "dsv4_index_compressor_fused_compress_store" in operators


def test_csa_branch_uses_whole_batch_max_and_replicated_indexer(model, replica):
    result = workload(model, replica, source_batch([1, 1], [2050, 2051], [True, True]), 2)
    operators = result["operators"]
    assert "dsv4_index_short_context_fill" not in operators
    assert operators["dsv4_index_query_projection"][0]["output_shape"] == (2, 64, 128)
    assert operators["dsv4_index_scores"][0]["compressed_history_rows"] == (512, 513)


def test_csa_chunk_distinguishes_projection_gather_history_and_writes(model, replica):
    result = workload(model, replica, source_batch([17], [120]), 2)
    operators = result["operators"]
    assert operators["dsv4_compressor_projection"][0]["output_shape"] == (17, 2048)
    assert operators["dsv4_compressor_save_partial_states"][0]["state_width"] == 2048
    assert operators["dsv4_compressor_fused_compress_store"][0]["new_compressed_rows"] == (4,)
    assert operators["dsv4_index_compressor_projection"][0]["output_shape"] == (17, 512)
    assert operators["dsv4_sparse_attention"][0]["swa_logical_window_span_rows"] == (137,)
    assert operators["dsv4_sparse_attention"][0]["compressed_history_rows"] == (34,)


@pytest.mark.parametrize("context,crosses", [(126, False), (127, True)])
def test_hca_retains_partial_state_and_only_launches_store_on_boundary(
    model, replica, context, crosses,
):
    result = workload(model, replica, source_batch([1], [context], [True]), 3)
    operators = result["operators"]
    assert operators["dsv4_compressor_projection"][0]["output_shape"] == (1, 1024)
    assert "dsv4_compressor_save_partial_states" in operators
    assert ("dsv4_compressor_fused_compress_store" in operators) is crosses


def test_hca_any_crossing_launches_full_batch_but_records_individual_writes(model, replica):
    result = workload(model, replica, source_batch([1, 1], [126, 127], [True, True]), 3)
    call = result["operators"]["dsv4_compressor_fused_compress_store"][0]
    assert call["input_shape"] == (2, 1024)
    assert call["new_compressed_rows"] == (0, 1)


@pytest.mark.parametrize("context,query,history,writes,span,store", [
    (0, 127, 0, 0, 127, False),
    (127, 1, 1, 1, 128, True),
    (128, 1, 1, 0, 128, False),
    (129, 1, 1, 0, 128, False),
])
def test_hca_chunk_and_decode_boundary_journey(
    model, replica, context, query, history, writes, span, store,
):
    result = workload(model, replica, source_batch([query], [context], [context >= 128]), 3)
    operators = result["operators"]
    assert result["new_compressed_rows"] == (writes,)
    assert operators["dsv4_sparse_attention"][0]["compressed_history_rows"] == (history,)
    assert operators["dsv4_sparse_attention"][0]["swa_logical_window_span_rows"] == (span,)
    assert operators["dsv4_compressor_save_partial_states"][0]["state_width"] == 1024
    assert ("dsv4_compressor_fused_compress_store" in operators) is store


@pytest.mark.parametrize("context,query,history,writes,short", [
    (0, 2050, 512, 512, True),
    (2050, 1, 512, 0, True),
    (2051, 1, 513, 1, False),
    (2052, 1, 513, 0, False),
])
def test_csa_prefill_to_decode_threshold_journey(
    model, replica, context, query, history, writes, short,
):
    result = workload(model, replica, source_batch([query], [context], [context >= 2051]), 2)
    operators = result["operators"]
    assert result["new_compressed_rows"] == (writes,)
    assert operators["dsv4_sparse_attention"][0]["compressed_history_rows"] == (history,)
    assert operators["dsv4_index_compressor_save_partial_states"][0]["state_width"] == 512
    assert operators["dsv4_index_compressor_fused_compress_store"][0]["new_compressed_rows"] == (writes,)
    assert ("dsv4_index_short_context_fill" in operators) is short
    if short:
        assert operators["dsv4_index_short_context_fill"][0]["output_shape"] == (query, 512)
    else:
        assert operators["dsv4_index_query_projection"][0]["output_shape"] == (query, 64, 128)
        assert operators["dsv4_index_scores"][0]["compressed_history_rows"] == (513,)


def test_kernel_partition_uses_query_threshold_and_preserves_unsorted_snapshot(model, replica):
    batch = source_batch([17, 1, 1], [120, 0, 127], [False, False, True])
    result = workload(model, replica, batch, 3)
    assert result["request_intervals"] == ((120, 137), (0, 1), (127, 128))
    assert result["semantic_is_decoding"] == (False, False, True)
    assert result["source_request_order"] == (1, 2, 0)
    assert result["decode_request_indices"] == (1, 2)
    assert result["prefill_request_indices"] == (0,)
    assert result["num_decode_query_tokens"] == 2
    assert result["num_prefill_query_tokens"] == 17
    assert result["operators"]["dsv4_sparse_attention"][0][
        "swa_logical_window_span_rows"
    ] == (137, 1, 128)
    assert batch.num_tokens == [17, 1, 1]
    assert batch.num_context_tokens == [120, 0, 127]


@pytest.mark.parametrize("options", [
    {"execution_mode": "capture"}, {"execution_mode": "replay"},
    {"execution_mode": None}, {"has_attention_metadata": False},
    {"has_attention_metadata": 1}, {"prefix_caching": True},
    {"prefix_caching": None}, {"decode_context_parallel_size": 2},
    {"prefill_context_parallel_size": 2},
    {"cluster_type": ClusterType.PREFILL},
])
def test_unsupported_explicit_context_is_rejected(model, replica, options):
    with pytest.raises((ValueError, NotImplementedError)):
        workload(model, replica, source_batch([1], [127]), **options)


@pytest.mark.parametrize("runtime,hit", [("FULL", True), ("PIECEWISE", True), ("NONE", True)])
def test_actual_graph_state_cannot_be_overridden_by_eager_claim(model, replica, runtime, hit):
    batch = source_batch([1], [127], [True])
    batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
        "full_decode_only", runtime, hit, False, 1, 8, 1, 8,
    )
    with pytest.raises((ValueError, NotImplementedError), match="graph"):
        workload(model, replica, batch)


def test_configured_graph_does_not_reject_actual_eager_work(model, replica):
    batch = source_batch([1], [127], [True])
    batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
        "full_decode_only", "NONE", False, False, 1, 1, 1, 1,
    )
    assert workload(model, replica, batch)["num_tokens"] == 1


@pytest.mark.parametrize("field,value", [
    ("_num_tokens", []), ("_num_tokens", [True]), ("_num_tokens", [1.5]),
    ("_num_tokens", [0]), ("_num_context_tokens", []),
    ("_num_context_tokens", [-1]), ("_num_context_tokens", [False]),
    ("_request_is_decoding", []), ("_request_is_decoding", [1]),
    ("_total_num_tokens", 9), ("_num_prefill_tokens", 9),
])
def test_malformed_scheduling_snapshots_fail_before_zip_truncation(model, replica, field, value):
    batch = source_batch([1], [127])
    setattr(batch, field, value)
    with pytest.raises(ValueError):
        workload(model, replica, batch)


def test_semantic_decode_requires_one_token_and_empty_batch_is_rejected(model, replica):
    for batch in (source_batch([2], [127], [True]), source_batch([], [])):
        with pytest.raises(ValueError):
            workload(model, replica, batch)


@pytest.mark.parametrize("field,value", [
    ("num_pipeline_stages", 2), ("attn_tensor_parallel_size", 3),
    ("attn_tensor_parallel_size", True), ("attention_backend", None),
    ("attention_backend", "FLASHINFER"),
])
def test_unsupported_topology_and_backend_are_rejected(model, replica, field, value):
    setattr(replica, field, value)
    with pytest.raises((ValueError, NotImplementedError)):
        workload(model, replica, source_batch([1], [127]))


def test_speculative_and_materialized_ep_batches_are_rejected(model, replica):
    batch = source_batch([1], [127])
    replica.speculative_decoding_config.enabled = True
    with pytest.raises(NotImplementedError, match="speculative"):
        workload(model, replica, batch)
    replica.speculative_decoding_config.enabled = False
    batch.spec_decode_metadata = object()
    with pytest.raises(NotImplementedError, match="speculative"):
        workload(model, replica, batch)
    batch.spec_decode_metadata = None
    batch.lane_workload = object()
    with pytest.raises(ValueError, match="source DP"):
        workload(model, replica, batch)
    with pytest.raises(ValueError, match="source DP"):
        workload(model, replica, EPBatchGroup.__new__(EPBatchGroup))


def test_output_quantization_uses_grouped_view(model, replica):
    operators = workload(model, replica, source_batch([17], [120]))["operators"]
    assert operators["dsv4_output_inv_rope_quant"][0]["input_shape"] == (17, 8, 512)
    assert operators["dsv4_output_inv_rope_quant"][0]["output_shape"] == (17, 1, 4096)
    assert operators["dsv4_grouped_output_projection"][0]["input_shape"] == (17, 1, 4096)


class ComponentPredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise NotImplementedError


def component_predictor(model, replica):
    predictor = ComponentPredictor.__new__(ComponentPredictor)
    predictor._model_config = model
    predictor._replica_config = replica
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 2.0
    predictor._kernel_count_scopes = []
    return predictor


def component_cost(predictor, batch, layer, **overrides):
    options = dict(
        execution_mode="eager", has_attention_metadata=True,
        prefix_caching=False, decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    options.update(overrides)
    return predictor._predict_dsv4_dummy_attention_time(
        batch, layer, ClusterType.MONOLITHIC, **options,
    )


def test_component_consumer_enforces_explicit_pcp1_context(model, replica):
    predictor = component_predictor(model, replica)
    with pytest.raises(NotImplementedError, match="PCP1"):
        component_cost(
            predictor, source_batch([17], [120]), 2,
            prefill_context_parallel_size=2,
        )


@pytest.mark.parametrize("layer,context,count,total", [
    (0, 127, 8, 16.0), (2, 2050, 16, 32.0), (2, 2051, 19, 38.0),
    (3, 126, 10, 20.0), (3, 127, 11, 22.0),
])
def test_synthetic_consumer_prices_only_selected_component_calls(
    model, replica, layer, context, count, total,
):
    predictor = component_predictor(model, replica)
    result = component_cost(predictor, source_batch([1], [context], [True]), layer)
    assert isinstance(result, AttentionOperatorTimes)
    assert len(result.op_times) == count
    assert set(result.op_times.values()) == {2.0}
    assert result.total_time() == total


def test_component_requires_dummy_mode_and_measured_prediction_stays_closed(model, replica):
    predictor = component_predictor(model, replica)
    batch = source_batch([1], [127], [True])
    predictor._enable_dummy_mode = False
    with pytest.raises(NotImplementedError, match="synthetic"):
        component_cost(predictor, batch, 3)
    with pytest.raises(NotImplementedError, match="not enabled"):
        predictor.predict_attention_layer_time(batch, 3, ClusterType.MONOLITHIC)
    predictor._enable_dummy_mode = True
    result = predictor.predict_attention_layer_time(batch, 3, ClusterType.MONOLITHIC)
    assert "dsv4_sparse_attention" in result.operator_times.op_times
    assert DEEPSEEK_V4_ATTENTION_FAMILY.execution_enabled is False
    assert DEEPSEEK_V4_ATTENTION_FAMILY.structural_execution_enabled is True
    assert all(not op.predictor_target and not op.profiling_target
               and not op.e2e_trace_target for op in DEEPSEEK_V4_ATTENTION_FAMILY.operators)


def test_shapes_follow_configuration_and_batch_snapshots(model, replica):
    model.embedding_dim = 2048
    model.q_lora_rank = 512
    model.o_lora_rank = 512
    model.num_q_heads = 32
    model.index_n_heads = 32
    model.index_topk = 256
    batch = source_batch([17], [120])
    batch.requests[0]._num_processed_tokens = 999
    operators = workload(model, replica, batch, 2)["operators"]
    assert operators["dsv4_fused_qkv_projection"][0]["input_shape"] == (17, 2048)
    assert operators["dsv4_fused_qkv_projection"][0]["output_shape"] == (17, 1024)
    assert operators["dsv4_q_projection"][0]["output_shape"] == (17, 4, 512)
    assert operators["dsv4_grouped_output_projection"][0]["input_shape"] == (17, 1, 2048)
    assert operators["dsv4_index_weights_projection"][0]["output_shape"] == (17, 32)
    assert operators["dsv4_index_short_context_fill"][0]["output_shape"] == (17, 256)
    assert operators["dsv4_compressor_fused_compress_store"][0]["new_compressed_rows"] == (4,)


def test_csa_launch_can_write_zero_rows_and_sparse_query_retains_intervals(model, replica):
    result = workload(model, replica, source_batch([1], [0]), 2)
    operators = result["operators"]
    assert operators["dsv4_compressor_fused_compress_store"][0]["new_compressed_rows"] == (0,)
    assert operators["dsv4_index_compressor_fused_compress_store"][0]["new_compressed_rows"] == (0,)
    assert operators["dsv4_fused_q_kv_prepare"][0]["kv_input_shape"] == (1, 512)
    assert operators["dsv4_sparse_attention"][0]["query_intervals"] == ((0, 1),)
    assert operators["dsv4_sparse_attention"][0]["decode_request_indices"] == (0,)


def test_long_chunk_uses_integer_boundaries_without_token_expansion(model, replica):
    result = workload(model, replica, source_batch([100003], [123]), 3)
    call = result["operators"]["dsv4_compressor_fused_compress_store"][0]
    assert call["new_compressed_rows"] == (782,)
    assert result["request_intervals"] == ((123, 100126),)


def test_wrong_profile_bad_layer_and_unsupported_head_dimension_fail(model, replica):
    generic = BaseModelConfig.create_from_name("meta-llama/Llama-2-7b-hf")
    with pytest.raises(ValueError, match="profile"):
        workload(generic, replica, source_batch([1], [127]))
    for layer in (-1, 43, True):
        with pytest.raises((ValueError, IndexError)):
            workload(model, replica, source_batch([1], [127]), layer)
    model.head_dim = 256
    with pytest.raises(NotImplementedError, match="head dimension"):
        workload(model, replica, source_batch([1], [127]), 3)
