"""Graph geometry and exact isolated scopes do not establish serving parity."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.entities.batch import DecodeCudaGraphMetadata
from frontier.execution_time_predictor.deepseek_v4 import read_isolated_scope, serving_context
from frontier.profiling.attention.dsv4_attention_plan import scope_operators
from frontier.profiling.common.runtime_identity import RuntimeIdentity
from frontier.types import ClusterType, MeasurementType
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


def test_piecewise_short_context_requires_capture_branch_evidence(model, replica):
    with pytest.raises(DeepSeekV4CoverageError, match="capture branch"):
        workload(model, replica, graph_batch([1], [126], "PIECEWISE", 8),
                 2, execution_mode="cuda_graph")
    with pytest.raises(DeepSeekV4CoverageError, match="HCA.*capture branch"):
        workload(model, replica, graph_batch([1], [126], "PIECEWISE", 8),
                 3, execution_mode="cuda_graph")


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


def test_global_graph_setting_does_not_override_batch_state(predictor, monkeypatch):
    from frontier.config import global_vars

    monkeypatch.setattr(global_vars, "USE_CUDA_GRAPH", True)
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    monkeypatch.setattr(global_vars, "KV_CONNECTOR", "none")
    eager = source_batch([17], [0])
    assert serving_context(predictor, eager, 0)["execution_mode"] == "eager"
    assert serving_context(predictor, graph_batch([17], [0]), 0)["execution_mode"] == "cuda_graph"


@pytest.fixture
def measured(predictor, tmp_path):
    p = predictor
    p._enable_dummy_mode = False
    p._active_measurement_type = MeasurementType.KERNEL_ONLY
    p._block_size = 256
    p._replica_config.device = "h200"
    p._replica_config.model_config = p._model_config
    p._attention_input_file = str(tmp_path / "attention_kernel_only.csv")
    m = p._model_config
    row = {
        "schema_version": "dsv4_attention.v1", "measurement_type": "KERNEL_ONLY",
        "attention_family": "deepseek_v4_attention", "model_architecture_profile": "deepseek_v4",
        "profiling_precision": "BF16", "attention_backend": "FLASHMLA_SPARSE_DSV4",
        "n_embd": m.embedding_dim, "n_q_head": m.num_q_heads, "n_q_head_per_rank": 8,
        "num_tensor_parallel_workers": 8, "o_groups": m.o_groups,
        "o_groups_per_rank": 1, "block_size": 256,
        **{name: getattr(m, name) for name in (
            "q_lora_rank", "o_lora_rank", "head_dim", "qk_rope_head_dim",
            "sliding_window", "index_topk", "index_n_heads", "index_head_dim",
        )},
        "phase": "prefill", "attention_variant": "uncompressed", "plans_tile_schedule": True,
        "total_tokens": 6144, "batch_size": 1, "prefill_seq_lens": "[6144]",
        "prefill_kv_cache_sizes": "[0]", "decode_kv_cache_sizes": "[]",
        "operator_scopes": json.dumps({"dsv4_sparse_attention": list(scope_operators(
            "dsv4_sparse_attention", "uncompressed"))}),
        "time_stats.dsv4_sparse_attention.median": 0.125,
        "time_stats.dsv4_sparse_attention.kernel_count": 2,
    }
    row.update(RuntimeIdentity(
        runtime_versions={"vllm": "fixture"}, source_revision="fixture-not-calibration",
        model_config_revision="fixture", model_config_sha256="fixture",
        device_name="NVIDIA H200", device_capability="sm90",
        schema_version="dsv4_attention.v1", quant_signature=m.get_quant_signature(),
        parallel_groups={"engine_tensor_parallel": 1, "target_tensor_parallel": 8},
        graph_mode="eager",
    ).columns())
    return p, row


def write_row(p, row):
    pd.DataFrame([row]).to_csv(p._attention_input_file, index=False)


def test_exact_isolated_math_preserves_scope_and_provenance(measured):
    p, row = measured
    write_row(p, row)
    value = read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention")
    assert value["time_ms"] == 0.125
    assert value["physical_tensor_parallel_size"] == 1
    assert value["target_tensor_parallel_size"] == 8
    assert value["serving_admission"] is False
    assert value["source_revision"] == "fixture-not-calibration"
    from frontier.execution_time_predictor.profiling_dataframe_loaders import ProfilingDataFrameLoaders

    loader = ProfilingDataFrameLoaders()
    loader._active_measurement_type = MeasurementType.KERNEL_ONLY
    rows = loader._load_attention_df(p._attention_input_file, p._replica_config,
                                    SimpleNamespace(block_size=256))
    assert rows["time_stats.dsv4_sparse_attention.median"].iloc[0] == 0.125


@pytest.mark.parametrize("column,value", [
    ("measurement_type", "CUDA_EVENT"), ("graph_mode", "FULL"),
    ("n_q_head_per_rank", 64), ("plans_tile_schedule", False),
    ("prefill_kv_cache_sizes", "[1]"), ("phase", "first_tail"),
    ("prefill_seq_lens", "[6144, 1]"),
    ("operator_scopes", "{}"), ("runtime_fingerprint", "wrong"),
    ("parallel_groups", '{"engine_tensor_parallel":8}'),
    ("time_stats.dsv4_sparse_attention.median", float("nan")),
    ("time_stats.dsv4_sparse_attention.kernel_count", float("nan")),
])
def test_unqualified_identity_geometry_phase_or_scope_is_typed_error(measured, column, value):
    p, row = measured
    row[column] = value
    write_row(p, row)
    with pytest.raises(DeepSeekV4CoverageError):
        read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention")


def test_no_extrapolation_generic_routing_or_decode_tail_promotion(measured):
    p, row = measured
    write_row(p, row)
    for batch, scope in (
        (source_batch([6145], [0]), "dsv4_sparse_attention"),
        (source_batch([1], [6143]), "dsv4_sparse_attention"),
        (source_batch([1], [6144], [True]), "dsv4_sparse_attention"),
        (source_batch([6144], [0]), "moe_gating_routing_topk"),
        (source_batch([6144], [0]), "dsv4_compressor_projection"),
        (graph_batch([6144], [0], padded=6144), "dsv4_sparse_attention"),
    ):
        with pytest.raises(DeepSeekV4CoverageError):
            read_isolated_scope(p, batch, 0, scope)


def test_measured_composites_cannot_be_added_to_serve_attention(measured):
    p, row = measured
    write_row(p, row)
    with pytest.raises(DeepSeekV4CoverageError, match="auxiliary streams"):
        p._predict_dsv4_dummy_attention_time(
            source_batch([6144], [0]), 0, ClusterType.MONOLITHIC,
            execution_mode="eager", has_attention_metadata=True,
            prefix_caching=False, decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        )
