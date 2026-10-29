"""Graph geometry and exact isolated scopes do not establish serving parity."""

from types import SimpleNamespace

import pytest

from frontier.entities.batch import DecodeCudaGraphMetadata
from frontier.execution_time_predictor.deepseek_v4 import serving_context
from tests.unit.test_dsv4_attention_workload import model, replica, source_batch, workload
from tests.unit.test_dsv4_source_execution import predictor, component


def graph_batch(queries, contexts, mode="FULL", padded=32):
    batch = source_batch(queries, contexts)
    batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
        "full_and_piecewise", mode, mode != "NONE", False,
        sum(queries), padded, len(queries), padded,
    )
    return batch


@pytest.mark.parametrize("mode", ["FULL", "PIECEWISE"])
def test_native_descriptor_pads_compute_not_requests(model, replica, mode):
    batch = graph_batch([17], [120], mode)
    result = workload(model, replica, batch, execution_mode="cuda_graph")
    assert result["num_tokens"] == 32
    assert result["actual_num_tokens"] == 17
    assert result["request_intervals"] == ((120, 137),)
    assert result["num_prefill_query_tokens"] == 17
    assert result["operators"]["dsv4_q_projection"][0]["input_shape"] == (32, 1024)
    prepare = result["operators"]["dsv4_fused_q_kv_prepare"][0]
    assert prepare["kv_cache_write_rows"] == 17
    assert prepare["invalid_padding_rows"] == 15
    assert batch.num_tokens == [17]


def test_full_graph_does_not_take_eager_hca_skip_or_csa_short_branch(model, replica):
    batch = graph_batch([1], [126], padded=8)
    hca = workload(model, replica, batch, 3, execution_mode="cuda_graph")
    assert hca["new_compressed_rows"] == (0,)
    assert "dsv4_compressor_fused_compress_store" in hca["operators"]
    csa = workload(model, replica, batch, 2, execution_mode="cuda_graph")
    assert "dsv4_index_short_context_fill" not in csa["operators"]
    assert "dsv4_index_query_projection" in csa["operators"]
    assert "dsv4_index_topk" in csa["operators"]


def test_piecewise_short_context_runs_indexer_and_hca_store(model, replica):
    csa = workload(model, replica, graph_batch([1], [126], "PIECEWISE", 8),
                   2, execution_mode="cuda_graph")
    assert not csa["indexer_short_context"]
    assert {"dsv4_index_query_projection", "dsv4_index_scores", "dsv4_index_topk"} <= set(csa["operators"])
    hca = workload(model, replica, graph_batch([1], [126], "PIECEWISE", 8),
                   3, execution_mode="cuda_graph")
    assert hca["operators"]["dsv4_compressor_fused_compress_store"][0]["new_compressed_rows"] == (0,)


def test_source_sp_uses_descriptor_then_tp_padding_and_real_cc(predictor):
    batch = graph_batch([17], [120], padded=40)
    result = component(predictor, batch, layer=2, rank=3, execution_mode="cuda_graph")
    assert result["source_rows"] == 17
    assert result["compute_rows"] == result["padded_rows"] == 40
    assert result["physical_local_rows"] == 5
    assert result["valid_local_rows"] == 2
    assert result["workloads"]["share_expert_up_proj"]["input_shape"] == (5, 4096)
    assert predictor._cc_backend.calls[0][1]["data_size_bytes"] == 5 * 4096 * 2
    assert predictor._cc_backend.calls[1][1]["data_size_bytes"] == 40 * 4096 * 2


def test_layer0_embedding_collective_uses_pre_sp_graph_width(predictor):
    batch = graph_batch([17], [0], padded=40)
    result = predictor._predict_dsv4_structural_execution(
        batch, 0, 0, include_ffn=False,
        execution_mode="cuda_graph",
        prefix_caching=False, decode_context_parallel_size=1,
        prefill_context_parallel_size=1, enable_expert_parallel=True,
        moe_backend="auto", all2all_backend="allgather_reducescatter",
        skip_padding=False, input_padding_mask=None, selected_logits_rows=None,
    )
    assert result._dsv4_workloads["dsv4_embedding"]["output_shape"] == (40, 4096)
    assert result._dsv4_workloads["dsv4_embedding_allreduce"]["input_shape"] == (40, 4096)
    assert result._dsv4_source_rows["source_rows"] == 17
    assert result._dsv4_source_rows["compute_rows"] == 40
    assert batch.num_tokens == [17] and batch.num_context_tokens == [0]


def test_global_graph_setting_does_not_override_batch_state(predictor, monkeypatch):
    from frontier.config import global_vars

    monkeypatch.setattr(global_vars, "USE_CUDA_GRAPH", True)
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    monkeypatch.setattr(global_vars, "KV_CONNECTOR", "none")
    eager = source_batch([17], [0])
    assert serving_context(predictor, eager, 0)["execution_mode"] == "eager"
    assert serving_context(predictor, graph_batch([17], [0]), 0)["execution_mode"] == "cuda_graph"


@pytest.mark.parametrize("dummy", [False, True])
def test_pd_context_uses_actual_graph_state(monkeypatch, dummy, predictor):
    from frontier.config import global_vars

    monkeypatch.setattr(global_vars, "KV_CONNECTOR", "nixl")
    predictor._enable_dummy_mode = dummy
    assert serving_context(predictor, source_batch([1], [6143]), 0)["execution_mode"] == "eager"


def test_h800_tp8_dp2_ep16_reuses_source_geometry_not_h200_capacity(predictor):
    from frontier.entities import Batch, Request
    from frontier.execution_time_predictor.deepseek_v4 import source_sp_token_widths

    replica = predictor._replica_config
    replica.device = "h800"
    replica.attn_dp = 2
    replica.moe_expert_parallel_size = 16
    prefill = Batch(0, [Request(0, 16, 16)], [16], is_moe=True,
                    num_context_tokens=[0], request_is_decoding=[False])
    decode = Batch(1, [Request(0, 16, 16)], [1], is_moe=True,
                   num_context_tokens=[16], request_is_decoding=[True])
    for batch, mode, width in ((prefill, "PIECEWISE", 32), (decode, "FULL", 8)):
        batch._cluster_type = predictor._cluster_type
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            "full_and_piecewise", mode, True, False,
            batch.total_num_tokens, width, 1, width,
        )
        result = component(predictor, batch, layer=0, rank=0, execution_mode="cuda_graph")
        assert result["compute_rows"] == width
        assert result["source_rows"] == batch.total_num_tokens
        assert result["physical_local_rows"] == width // 8
        assert batch.requests[0].num_decode_tokens == 16
    assert source_sp_token_widths(predictor, {0: prefill, 1: decode}).gathered_tokens == 40
    assert prefill.num_context_tokens == [0] and decode.num_context_tokens == [16]
