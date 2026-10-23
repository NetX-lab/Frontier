"""Graph geometry and exact isolated scopes do not establish serving parity."""

import json
from pathlib import Path
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


def test_layer0_embedding_collective_uses_pre_sp_graph_width(predictor):
    batch = graph_batch([17], [0], padded=40)
    result = predictor._predict_dsv4_dummy_structural_execution(
        batch, 0, 0, include_ffn=False,
        execution_mode="cuda_graph", has_attention_metadata=True,
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
def test_pd_guard_message_is_independent_of_dummy_mode(monkeypatch, dummy):
    from frontier.config import global_vars

    monkeypatch.setattr(global_vars, "KV_CONNECTOR", "nixl")
    with pytest.raises(NotImplementedError, match="V4 predictor PD phase/graph equivalence remains unqualified"):
        serving_context(SimpleNamespace(_enable_dummy_mode=dummy), None, 0)


def test_h800_tp8_dp2_ep16_reuses_source_geometry_not_h200_capacity(predictor):
    from frontier.entities import Batch, Request
    from frontier.execution_time_predictor.deepseek_v4 import source_physical_rows

    replica = predictor._replica_config
    replica.device = "h800"
    replica.attn_dp = 2
    replica.moe_expert_parallel_size = 16
    prefill = Batch(0, [Request(0, 16, 16)], [16], is_moe=True,
                    num_context_tokens=[0], request_is_decoding=[False])
    decode = Batch(1, [Request(0, 16, 16)], [1], is_moe=True,
                   num_context_tokens=[16], request_is_decoding=[True])
    for batch, mode, width in ((prefill, "PIECEWISE", 32), (decode, "FULL", 8)):
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            "full_and_piecewise", mode, True, False,
            batch.total_num_tokens, width, 1, width,
        )
        result = component(predictor, batch, layer=0, rank=0, execution_mode="cuda_graph")
        assert result["compute_rows"] == width
        assert result["source_rows"] == batch.total_num_tokens
        assert result["physical_local_rows"] == width // 8
        assert batch.requests[0].num_decode_tokens == 16
    assert source_physical_rows(predictor, {0: prefill, 1: decode}) == 40
    assert prefill.num_context_tokens == [0] and decode.num_context_tokens == [16]


@pytest.fixture
def measured(predictor, tmp_path):
    p = predictor
    p._enable_dummy_mode = False
    p._active_measurement_type = MeasurementType.KERNEL_ONLY
    p._block_size = 256
    p._replica_config.device = "h200"
    p._replica_config.model_config = p._model_config
    p._replica_config.kv_cache_dtype = "fp8_ds_mla"
    p._attention_input_file = str(tmp_path / "attention_kernel_only.csv")
    p._isolated_attention_layout = json.loads((Path(__file__).parents[1] /
        "fixtures/deepseek_v4_attention_layouts.json").read_text())["layouts"]["uncompressed"]
    m = p._model_config
    row = {
        **p._isolated_attention_layout,
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
    value = read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention",
                               attention_layout=p._isolated_attention_layout)
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
    ("attention_kv_cache_dtype", "bfloat16"), ("padded_heads", 128),
    ("swa_metadata.block_size", 128), ("max_model_len", 8192),
])
def test_attention_rejects_unqualified_isolated_layout(measured, column, value):
    p, row = measured
    row[column] = value
    write_row(p, row)
    with pytest.raises(DeepSeekV4CoverageError):
        read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention",
                            attention_layout=p._isolated_attention_layout)


def test_attention_does_not_median_across_isolated_layouts(measured):
    p, row = measured
    other = {**row, "padded_heads": 128, "time_stats.dsv4_sparse_attention.median": 0.5}
    pd.DataFrame([{**row, "padded_heads": 64}, other]).to_csv(p._attention_input_file, index=False)
    value = read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention",
                               attention_layout=p._isolated_attention_layout)
    assert value["time_ms"] == 0.125


def test_attention_layout_must_match_replica_cache_dtype(measured):
    p, row = measured
    p._replica_config.kv_cache_dtype = "bfloat16"
    write_row(p, row)
    with pytest.raises(DeepSeekV4CoverageError):
        read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention",
                            attention_layout=p._isolated_attention_layout)


def test_attention_layout_cannot_override_source_padded_heads(measured):
    p, row = measured
    p._isolated_attention_layout = {**p._isolated_attention_layout, "padded_heads": 128}
    row["padded_heads"] = 128
    write_row(p, row)
    with pytest.raises(DeepSeekV4CoverageError):
        read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention",
                            attention_layout=p._isolated_attention_layout)


@pytest.mark.parametrize("change", ["absent", "incomplete", "extra"])
def test_attention_requires_complete_explicit_layout_contract(measured, change):
    p, row = measured
    write_row(p, row)
    layout = dict(p._isolated_attention_layout)
    if change == "absent":
        layout = None
    elif change == "incomplete":
        del layout["attention_kv_cache_dtype"]
    else:
        layout["unqualified_layout_field"] = 1
    with pytest.raises(DeepSeekV4CoverageError, match="complete isolated layout contract"):
        read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention",
                            attention_layout=layout)


@pytest.mark.parametrize("scope", ["share_expert_up_proj", "moe_grouped_gemm"])
def test_isolated_component_missing_table_is_typed_with_original_cause(measured, tmp_path, scope):
    from frontier.moe_ep_workload import split_global_expert_tokens_into_lanes

    p, _ = measured
    p._compute_input_file = str(tmp_path / "missing_linear.csv")
    p._moe_input_file = str(tmp_path / "missing_moe.csv")
    lane = split_global_expert_tokens_into_lanes(
        {i: int(i < 96) for i in range(256)}, total_expert_num=256,
        moe_expert_parallel_size=64, router_topk=6)[0]
    with pytest.raises(DeepSeekV4CoverageError) as error:
        read_isolated_scope(p, source_batch([16], [0]), 0, scope, lane_workload=lane)
    assert isinstance(error.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("scope,change,cause", [
    ("share_expert_up_proj", "missing_tp", ValueError),
    ("share_expert_up_proj", "invalid_contract", ValueError),
    ("share_expert_up_proj", "unmatched_tp", ValueError),
    ("moe_grouped_gemm", "missing_ep", ValueError),
    ("moe_grouped_gemm", "missing_tp", KeyError),
    ("moe_grouped_gemm", "invalid_contract", ValueError),
])
def test_known_component_coverage_errors_preserve_cause(measured, tmp_path, scope, change, cause):
    from frontier.moe_ep_workload import split_global_expert_tokens_into_lanes

    p, _ = measured
    row = {"num_tensor_parallel_workers": 1, "num_tokens": 16,
           "n_expanded_embd": 2048, "num_experts": 256, "router_topk": 6,
           "hidden_dim": 4096, "expert_hidden_dim": 2048, "expert_parallel_size": 64}
    if change == "missing_tp":
        del row["num_tensor_parallel_workers"]
    elif change == "missing_ep":
        del row["expert_parallel_size"]
    elif change == "unmatched_tp":
        row["num_tensor_parallel_workers"] = 2
    else:
        row["typed_operator_contracts"] = "{}"
    p._compute_input_file = p._moe_input_file = str(tmp_path / "incomplete.csv")
    pd.DataFrame([row]).to_csv(p._compute_input_file, index=False)
    lane = split_global_expert_tokens_into_lanes(
        {i: int(i < 96) for i in range(256)}, total_expert_num=256,
        moe_expert_parallel_size=64, router_topk=6)[0]
    with pytest.raises(DeepSeekV4CoverageError) as error:
        read_isolated_scope(p, source_batch([16], [0]), 0, scope, lane_workload=lane)
    assert isinstance(error.value.__cause__, cause)


def test_unexpected_reader_failure_is_not_relabelled_as_coverage(measured, monkeypatch):
    from frontier.execution_time_predictor.profiling_dataframe_loaders import ProfilingDataFrameLoaders

    p, _ = measured
    p._compute_input_file = "unused.csv"

    def fail(*args, **kwargs):
        raise ValueError("unexpected reader implementation failure")

    monkeypatch.setattr(ProfilingDataFrameLoaders, "_load_linear_op_df", fail)
    with pytest.raises(ValueError) as error:
        read_isolated_scope(p, source_batch([16], [0]), 0, "share_expert_up_proj")
    assert type(error.value) is ValueError
    assert error.value.__cause__ is None


def test_h800_missing_coverage_cannot_consume_h200_observations(measured):
    p, row = measured
    p._replica_config.device = "h800"
    p._replica_config.attn_dp = 2
    p._replica_config.moe_expert_parallel_size = 16
    write_row(p, row)
    with pytest.raises(DeepSeekV4CoverageError, match="device does not match"):
        read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention",
                            attention_layout=p._isolated_attention_layout)


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
        read_isolated_scope(p, source_batch([6144], [0]), 0, "dsv4_sparse_attention",
                            attention_layout=p._isolated_attention_layout)


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
            read_isolated_scope(p, batch, 0, scope, attention_layout=p._isolated_attention_layout)


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


@pytest.fixture(scope="module")
def decode_tables():
    root = Path("/data/ycfeng/tmp/glm53_dsv4/v4_q1_warm_results/Q1_WARM_6143_7166/"
                "compute/h800/deepseek-ai/DeepSeek-V4-Flash")
    paths = {MeasurementType.CUDA_EVENT: root / "attention.csv",
             MeasurementType.KERNEL_ONLY: root / "attention_kernel_only.csv"}
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("Accepted local warmed V4 q1 tables are unavailable")
    return {method: (path, pd.read_csv(path, float_precision="round_trip"))
            for method, path in paths.items()}


@pytest.fixture
def decode_measured(predictor, decode_tables, monkeypatch):
    from functools import lru_cache
    from frontier.execution_time_predictor import deepseek_v4

    p = predictor
    p._enable_dummy_mode = False
    p._block_size = 256
    p._replica_config.model_config = p._model_config
    p._replica_config.device = "h800"
    p._replica_config.attn_dp = 2
    p._replica_config.moe_expert_parallel_size = 16
    p._replica_config.kv_cache_dtype = "fp8_ds_mla"
    # Keep the existing model/schema reader; cache immutable input per test.
    reader = deepseek_v4.load_attention_dataframe
    @lru_cache(maxsize=2)
    def read(path, precision):
        return reader(path, p._model_config, 8, 256, p._active_measurement_type,
                      float_precision=precision)
    monkeypatch.setattr(deepseek_v4, "load_attention_dataframe",
        lambda path, model, tp, block, method, **kwargs: read(path, kwargs.get("float_precision")))
    from frontier.profiling.common import runtime_identity
    monkeypatch.setattr(runtime_identity, "file_sha256", lru_cache(maxsize=2)(runtime_identity.file_sha256))
    return p


def ongoing_batch(context, prompt=6141):
    from frontier.entities import Batch, Request

    return Batch(0, [Request(0, prompt, 1028)], [1], is_moe=True,
                 num_context_tokens=[context], request_is_decoding=[True])


def decode_layout(row):
    from frontier.profiling.attention.dsv4_attention_rows import _ENGINE_COLUMNS

    columns = set(_ENGINE_COLUMNS) | {
        "block_size", "kv_cache_dtype", "compress_ratio", "max_model_len",
        "max_num_batched_tokens", "max_num_seqs",
    } | {name for name in row.index if "_metadata." in name}
    accepted = json.loads((Path(__file__).parents[1] /
        "fixtures/deepseek_v4_decode_inputs.json").read_text())
    return {**{name: None if pd.isna(row[name]) else row[name] for name in columns},
            **accepted["expected_identity"],
            "input_file_sha256": accepted["input_file_sha256"][row.measurement_type]}


@pytest.mark.parametrize("method", [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
@pytest.mark.parametrize("context", [6143, 6144, 6146, 6147, 6270, 6271, 6272,
                                    6654, 6655, 6656, 6778, 7166])
def test_real_ongoing_decode_exact_all_scopes_and_six_layer_roles(decode_measured, decode_tables, method, context):
    p = decode_measured
    path, table = decode_tables[method]
    p._attention_input_file = str(path)
    p._active_measurement_type = method
    rows = table[table.decode_kv_cache_sizes == json.dumps([context])]
    assert len(table) == 6144 and len(rows) == 6
    seen = set()
    for _, row in rows.iterrows():
        layer = json.loads(row.global_layer_ids)[0]
        seen.add((row.attention_variant, bool(row.plans_tile_schedule)))
        layout = decode_layout(row)
        for scope in json.loads(row.operator_scopes):
            result = read_isolated_scope(p, ongoing_batch(context), layer, scope,
                                        attention_layout=layout)
            assert result["time_ms"] == row[f"time_stats.{scope}.median"]
            assert result["source_revision"] == "d2878ed90ef4aecc1d2d80b597381b813420e2a8"
            assert result["runtime_fingerprint"] == row.runtime_fingerprint == "6fba7d93f7a2ecdd"
            assert result["model_config_sha256"] == row.model_config_sha256
            assert result["input_file"] == str(path)
            assert result["measurement_type"] == method.value
            assert result["graph_mode"] == "eager" and result["serving_admission"] is False
            assert result["physical_tensor_parallel_size"] == 1
            assert result["target_tensor_parallel_size"] == 8
            assert result["parallel_groups"] == json.loads(row.parallel_groups)
            assert result["row_metadata"]["phase"] == "decode"
            assert result["row_metadata"]["num_layers_sampled"] == 1
            assert result["row_metadata"]["global_layer_ids"] == json.dumps([layer])
            assert result["row_metadata"]["decode_kv_cache_sizes"] == json.dumps([context])
            assert result["row_metadata"]["weight_source"] == "dummy"
            assert result["attention_layout"] == layout
            assert layout["max_num_seqs"] == 128  # Planner field, not native capacity1.
        if method is MeasurementType.KERNEL_ONLY:
            assert "dsv4_attention_layer" not in json.loads(row.operator_scopes)
            with pytest.raises(DeepSeekV4CoverageError):
                read_isolated_scope(p, ongoing_batch(context), layer, "dsv4_attention_layer",
                                    attention_layout=layout)
    assert seen == {(v, role) for v in ("uncompressed", "csa", "hca") for role in (False, True)}


@pytest.mark.parametrize("method", [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
@pytest.mark.parametrize("change", ["h200", "cache_dtype", "layout", "capacity_relabel",
    "first_tail", "false_decode_flag", "missing_context", "outside_window", "mixed",
    "linear_decode", "graph", "phase", "native_phase", "native_length", "duplicate", "serving"])
def test_real_decode_cannot_promote_unqualified_cases(decode_measured, decode_tables, tmp_path, method, change):
    p = decode_measured
    path, table = decode_tables[method]
    p._attention_input_file = str(path)
    p._active_measurement_type = method
    row = table[(table.decode_kv_cache_sizes == "[6143]")
                & (table.global_layer_ids == "[2]")].iloc[0].copy()
    layout, batch, scope = decode_layout(row), ongoing_batch(6143), "dsv4_sparse_attention"
    if change == "h200":
        p._replica_config.device = "h200"
    elif change == "cache_dtype":
        p._replica_config.kv_cache_dtype = "bfloat16"
    elif change in ("layout", "capacity_relabel"):
        layout["max_num_seqs" if change == "capacity_relabel" else "swa_metadata.block_size"] = 1
    elif change == "first_tail":
        batch = source_batch([1], [6143], [False])
    elif change == "false_decode_flag":
        batch = ongoing_batch(6143, prompt=6144)
    elif change == "missing_context":
        batch._num_context_tokens = []
    elif change == "outside_window":
        batch = ongoing_batch(7167)
    elif change == "mixed":
        batch = source_batch([1, 2], [6143, 0], [True, False])
    elif change == "linear_decode":
        scope = "share_expert_up_proj"
    elif change == "graph":
        batch.decode_cuda_graph_metadata = graph_batch([1], [6143], padded=8).decode_cuda_graph_metadata
    elif change in ("phase", "native_phase", "native_length", "duplicate"):
        if change != "duplicate":
            column = {"phase": "phase", "native_phase": "swa_metadata.num_prefills",
                      "native_length": "attention_metadata.max_seq_len"}[change]
            row[column] = "first_tail" if change == "phase" else (1 if change == "native_phase" else 0)
            if column in layout:
                layout[column] = row[column]  # Explicit layout cannot override source phase/length.
        p._attention_input_file = str(tmp_path / "changed.csv")
        pd.DataFrame([row, row] if change == "duplicate" else [row]).to_csv(p._attention_input_file, index=False)
        from frontier.profiling.common.runtime_identity import file_sha256
        layout["input_file_sha256"] = file_sha256(p._attention_input_file)
    if change == "serving":
        with pytest.raises(DeepSeekV4CoverageError, match="auxiliary streams"):
            p._predict_dsv4_dummy_attention_time(batch, 2, ClusterType.MONOLITHIC,
                execution_mode="eager", has_attention_metadata=True, prefix_caching=False,
                decode_context_parallel_size=1, prefill_context_parallel_size=1)
    else:
        with pytest.raises(ValueError if change == "missing_context" else DeepSeekV4CoverageError):
            read_isolated_scope(p, batch, 2, scope, attention_layout=layout)


@pytest.mark.parametrize("method", [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
def test_accepted_decode_contract_rejects_real_cold_table(decode_measured, decode_tables, method):
    p = decode_measured
    _, accepted = decode_tables[method]
    row = accepted[(accepted.decode_kv_cache_sizes == "[6143]")
                   & (accepted.global_layer_ids == "[2]")].iloc[0]
    root = Path("/data/ycfeng/tmp/glm53_dsv4/v4_h800_q1_b36c2b9_results/Q1_6143_7166/"
                "compute/h800/deepseek-ai/DeepSeek-V4-Flash")
    p._attention_input_file = str(root / ("attention.csv" if method is MeasurementType.CUDA_EVENT
                                        else "attention_kernel_only.csv"))
    if not Path(p._attention_input_file).is_file():
        pytest.skip("Historical cold q1 table is unavailable")
    p._active_measurement_type = method
    with pytest.raises(DeepSeekV4CoverageError, match="accepted.*provenance"):
        read_isolated_scope(p, ongoing_batch(6143), 2, "dsv4_attention_layer"
            if method is MeasurementType.CUDA_EVENT else "dsv4_sparse_attention",
            attention_layout=decode_layout(row))


@pytest.mark.parametrize("method", [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
@pytest.mark.parametrize("change", ["missing_contract", "expected_source", "expected_fingerprint",
    "expected_input", "candidate_source", "candidate_fingerprint", "candidate_model", "changed_timing"])
def test_decode_acceptance_identity_is_independent_of_candidate(decode_measured, decode_tables, tmp_path, method, change):
    from frontier.profiling.common.runtime_identity import file_sha256

    p = decode_measured
    path, table = decode_tables[method]
    p._attention_input_file, p._active_measurement_type = str(path), method
    row = table[(table.decode_kv_cache_sizes == "[6143]") & (table.global_layer_ids == "[2]")].iloc[0].copy()
    layout = decode_layout(row)
    if change == "missing_contract":
        del layout["input_file_sha256"]
    elif change.startswith("expected_"):
        layout[{"expected_source": "source_revision", "expected_fingerprint": "runtime_fingerprint",
                "expected_input": "input_file_sha256"}[change]] = "unaccepted"
    else:
        if change == "candidate_source":
            identity = RuntimeIdentity(**{name: json.loads(row[name]) if name in (
                "runtime_versions", "parallel_groups") else row[name] for name in (
                "runtime_versions", "source_revision", "model_config_revision", "model_config_sha256",
                "device_name", "device_capability", "schema_version", "quant_signature", "parallel_groups", "graph_mode")})
            from dataclasses import replace
            changed = replace(identity, source_revision="self-consistent-but-unaccepted").columns()
            row["source_revision"], row["runtime_fingerprint"] = changed["source_revision"], changed["runtime_fingerprint"]
        elif change == "changed_timing":
            row["time_stats.dsv4_sparse_attention.median"] *= 10
        else:
            row["runtime_fingerprint" if change == "candidate_fingerprint" else "model_config_sha256"] = "unaccepted"
        p._attention_input_file = str(tmp_path / "changed.csv")
        pd.DataFrame([row]).to_csv(p._attention_input_file, index=False)
        if change != "changed_timing":
            # Synthetic input is explicitly bound; independent expected runtime
            # still comes from the accepted fixture, not this candidate row.
            layout["input_file_sha256"] = file_sha256(p._attention_input_file)
    with pytest.raises(DeepSeekV4CoverageError):
        read_isolated_scope(p, ongoing_batch(6143), 2, "dsv4_sparse_attention", attention_layout=layout)


@pytest.fixture(scope="module")
def accepted_prefill():
    from tests.integration.check_dsv4_measured_scopes import ScopePredictor, check_accepted_prefill
    from frontier.config.model_config import BaseModelConfig

    root = Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028")
    acceptance = root / "task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json"
    package = root.parent / "v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced"
    if not acceptance.is_file() or not package.is_dir():
        pytest.skip("Accepted prefill receipts/package are unavailable")
    from frontier.profiling.common.runtime_identity import file_sha256

    assert file_sha256(acceptance) == "616ee98996664c82c03b8d505de64b9cea917529a44c8a94b0068c708d5115e6"
    p = ScopePredictor.__new__(ScopePredictor)
    p._model_config = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
    p._replica_config = SimpleNamespace(model_config=p._model_config, num_pipeline_stages=1,
        moe_tensor_parallel_size=1, attention_backend="FLASHMLA_SPARSE_DSV4",
        kv_cache_dtype="fp8_ds_mla", speculative_decoding_config=SimpleNamespace(enabled=False))
    p._cluster_type = ClusterType.MONOLITHIC
    return p, check_accepted_prefill(p, acceptance, package)


def test_independent_prefill_contract_exact_all_accepted_scopes(accepted_prefill):
    _, receipts = accepted_prefill
    assert {m: sum(r["measurement_type"] == m for r in receipts) for m in
            ("CUDA_EVENT", "KERNEL_ONLY")} == {"CUDA_EVENT": 50, "KERNEL_ONLY": 44}
    assert {(r["row_metadata"]["attention_variant"], r["row_metadata"]["plans_tile_schedule"])
            for r in receipts} == {(v, role) for v in ("csa", "hca", "uncompressed") for role in (False, True)}


@pytest.mark.parametrize("method", [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
@pytest.mark.parametrize("change", ["source_revision", "runtime_fingerprint", "model_config_sha256",
    "input_file_sha256", "missing_provenance", "missing_metadata", "wrong_metadata", "variant", "planning_role", "graph", "pd", "unbalanced_q1"])
def test_accepted_prefill_contract_typed_rejections(accepted_prefill, method, change, monkeypatch):
    p, receipts = accepted_prefill
    receipt = next(r for r in receipts if r["measurement_type"] == method.value
        and r["row_metadata"]["global_layer_ids"] == "[2]")
    monkeypatch.setattr(p, "_attention_input_file", receipt["input_file"], raising=False)
    monkeypatch.setattr(p, "_active_measurement_type", method)
    layout, batch = dict(receipt["attention_layout"]), source_batch([6144], [0])
    if change in ("source_revision", "runtime_fingerprint", "model_config_sha256", "input_file_sha256"):
        layout[change] = "unaccepted"
    elif change == "missing_provenance":
        del layout["input_file_sha256"]
    elif change == "missing_metadata":
        del layout["swa_metadata.block_size"]
    elif change == "wrong_metadata":
        layout["swa_metadata.block_size"] = 1
    elif change == "variant":
        layout["attention_variant"] = "hca"
    elif change == "planning_role":
        layout["plans_tile_schedule"] = False
    elif change == "graph":
        batch = graph_batch([6144], [0], padded=6144)
    elif change == "pd":
        monkeypatch.setattr(p, "_cluster_type", ClusterType.PREFILL)
    else:
        path = Path("/data/ycfeng/tmp/glm53_dsv4/v4_q1_warm_results/Q1_WARM_6143_7166/compute/h800/deepseek-ai/DeepSeek-V4-Flash")
        monkeypatch.setattr(p, "_attention_input_file", str(path / ("attention.csv" if method is MeasurementType.CUDA_EVENT else "attention_kernel_only.csv")))
        batch = ongoing_batch(6143)
        rows = pd.read_csv(p._attention_input_file, float_precision="round_trip")
        row = rows[(rows.decode_kv_cache_sizes == "[6143]") & (rows.global_layer_ids == "[2]")].iloc[0]
        layout = {**decode_layout(row), **{key: layout[key] for key in
            ("routing_runtime_path", "routing_strategy_sha256", "moe_routing")}}
        assert "routing_runtime_path" not in rows
    with pytest.raises(DeepSeekV4CoverageError):
        read_isolated_scope(p, batch, 2, "dsv4_sparse_attention", attention_layout=layout)


def write_test_input_admission(sidecar):
    """Bind a test fixture only; this is not production qualification evidence."""
    from frontier.profiling.common.runtime_identity import file_sha256

    review = sidecar.parent / "fixture_review.txt"
    review.write_text("TEST FIXTURE ONLY. No production or GPU qualification.\n")
    payload = json.loads(sidecar.read_text())
    sidecar.with_suffix(".admission.json").write_text(json.dumps({
        "schema_version": payload["schema_version"], "qualification": payload["qualification"],
        "serving_admission": False, "verdict": "PASS", "reviewed_by": "TEST_FIXTURE_NOT_PRODUCTION",
        "sidecar_sha256": file_sha256(sidecar), "evidence": payload["evidence"],
        "review": {"path": review.name, "sha256": file_sha256(review)}}))


@pytest.fixture(scope="module")
def accepted_input(accepted_prefill, tmp_path_factory):
    from tests.comparison.calibration.prepare_dsv4_attention_input import prepare_attention_input
    from tests.integration.check_dsv4_measured_scopes import check_accepted_prefill

    p, explicit = accepted_prefill
    root = Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028")
    acceptance = root / "task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json"
    package = root.parent / "v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced"
    output = tmp_path_factory.mktemp("v4_attention_input") / "candidate"
    sidecar = prepare_attention_input(acceptance, package, output)
    assert not sidecar.with_suffix(".admission.json").exists()
    write_test_input_admission(sidecar)
    automatic = check_accepted_prefill(p, acceptance, package, attention_input=output)
    assert [(r["scope"], r["time_ms"], r["attention_layout"]) for r in automatic] == [
        (r["scope"], r["time_ms"], r["attention_layout"]) for r in explicit]
    return p, sidecar, automatic


def test_current_writer_all94_selected_input_results(accepted_input):
    _, _, receipts = accepted_input
    assert {m: sum(r["measurement_type"] == m for r in receipts) for m in
            ("CUDA_EVENT", "KERNEL_ONLY")} == {"CUDA_EVENT": 50, "KERNEL_ONLY": 44}
    assert all(r["serving_admission"] is False for r in receipts)


@pytest.mark.parametrize("method", [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
@pytest.mark.parametrize("change", ["missing_sidecar", "malformed_sidecar", "stale_sidecar",
    "missing_admission", "altered_admission", "admission_evidence", "serving_admission",
    "missing_review", "altered_review", "altered_csv", "altered_raw", "altered_plan",
    "altered_model", "altered_summary", "raw_lineage", "job_lineage", "sidecar_serving",
    "measurement", "provenance", "missing_layout", "layout", "routing", "phase", "graph"])
def test_selected_input_typed_refusals(accepted_input, tmp_path, monkeypatch, method, change):
    import shutil

    p, original, _ = accepted_input
    directory = tmp_path / "candidate"
    shutil.copytree(original.parent, directory)
    sidecar = directory / original.name
    admission = sidecar.with_suffix(".admission.json")
    payload = json.loads(sidecar.read_text())
    name = "attention.csv" if method is MeasurementType.CUDA_EVENT else "attention_kernel_only.csv"
    monkeypatch.setattr(p, "_attention_input_file", str(directory / name))
    monkeypatch.setattr(p, "_active_measurement_type", method)
    batch = source_batch([6144], [0])
    if change in ("missing_sidecar", "missing_admission", "missing_review"):
        {"missing_sidecar": sidecar, "missing_admission": admission,
         "missing_review": directory / "fixture_review.txt"}[change].unlink()
    elif change == "malformed_sidecar":
        sidecar.write_text("{")
    elif change == "stale_sidecar":
        sidecar.write_text(sidecar.read_text() + " ")
    elif change in ("altered_admission", "admission_evidence", "serving_admission"):
        receipt = json.loads(admission.read_text())
        if change == "altered_admission":
            receipt["sidecar_sha256"] = "unaccepted"
        elif change == "admission_evidence":
            receipt["evidence"] = {}
        else:
            receipt["serving_admission"] = True
        admission.write_text(json.dumps(receipt))
    elif change.startswith("altered_"):
        path = directory / (name if change == "altered_csv" else "fixture_review.txt"
            if change == "altered_review" else payload["evidence"][change.removeprefix("altered_")]["path"])
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "phase":
        batch = ongoing_batch(6143)
    elif change == "graph":
        batch = graph_batch([6144], [0], padded=6144)
    else:
        layout = payload["tables"][name]["layouts"]["2"]
        if change in ("raw_lineage", "job_lineage"):
            from frontier.profiling.common.runtime_identity import file_sha256

            reference = payload["evidence"]["raw" if change == "raw_lineage" else "job"]
            path = directory / reference["path"]
            path.write_bytes(path.read_bytes() + b" ")
            reference["sha256"] = file_sha256(path)
        elif change == "sidecar_serving":
            payload["serving_admission"] = True
        elif change == "measurement":
            payload["tables"][name]["measurement_type"] = "UNQUALIFIED"
        elif change == "provenance":
            layout["source_revision"] = "self-consistent-but-unaccepted"
        elif change == "missing_layout":
            del payload["tables"][name]["layouts"]["2"]
        else:
            layout["routing_runtime_path" if change == "routing" else "swa_metadata.block_size"] = (
                "random" if change == "routing" else 1)
        sidecar.write_text(json.dumps(payload))
        write_test_input_admission(sidecar)
    with pytest.raises(DeepSeekV4CoverageError):
        read_isolated_scope(p, batch, 2, "dsv4_sparse_attention")


def test_preparation_refuses_existing_output_and_unsupported_acceptance(accepted_input, tmp_path):
    from tests.comparison.calibration.prepare_dsv4_attention_input import prepare_attention_input

    _, sidecar, _ = accepted_input
    evidence = sidecar.parent / "evidence"
    # Original acceptance paths resolve to retained files in their source tree.
    original = Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
        "task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json")
    package = Path("/data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/"
        "task_2026-10-22_v4_h800_prefill/h800_prefill_balanced")
    with pytest.raises(FileExistsError):
        prepare_attention_input(original, package, sidecar.parent)
    acceptance = json.loads((evidence / "acceptance.json").read_text())
    acceptance["status"] = "PASS"
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(acceptance))
    with pytest.raises(ValueError, match="original isolated acceptance"):
        prepare_attention_input(path, package, tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()
