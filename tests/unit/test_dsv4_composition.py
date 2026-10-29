"""Stream schedule of the DeepSeek-V4 attention phase (S1 stream graph)."""

import pytest

from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.execution_time_predictor.dsv4_composition import (
    compose_attention_phase, segment_class, serial_attention_phase,
)
from frontier.types import ClusterType

THRESHOLD = 64

CSA = {
    "dsv4_mhc_ffn_post_attn_pre": 1.0, "dsv4_sp_attention_allgather": 2.0,
    "dsv4_fused_qkv_projection": 10.0, "dsv4_compressor_projection": 3.0,
    "dsv4_index_weights_projection": 4.0, "dsv4_index_compressor_projection": 12.0,
    "dsv4_fused_qkv_rmsnorm": 1.0,
    "dsv4_q_projection": 2.0, "dsv4_fused_q_kv_prepare": 3.0,
    "dsv4_index_query_projection": 4.0, "dsv4_index_fused_query_prepare": 1.0,
    "dsv4_index_compressor_save_partial_states": 2.0,
    "dsv4_index_compressor_fused_compress_store": 2.0,
    "dsv4_compressor_save_partial_states": 1.0, "dsv4_compressor_fused_compress_store": 6.0,
    "dsv4_index_scores": 5.0, "dsv4_index_topk": 2.0, "dsv4_sparse_attention": 7.0,
    "dsv4_output_inv_rope_quant": 1.0, "dsv4_grouped_output_projection": 2.0,
    "dsv4_output_projection": 3.0, "dsv4_sp_attention_reduce_scatter": 2.0,
}
HCA = {
    "dsv4_sp_attention_allgather": 2.0, "dsv4_fused_qkv_projection": 10.0,
    "dsv4_compressor_projection": 3.0, "dsv4_fused_qkv_rmsnorm": 1.0,
    "dsv4_q_projection": 2.0, "dsv4_fused_q_kv_prepare": 3.0,
    "dsv4_compressor_save_partial_states": 1.0, "dsv4_compressor_fused_compress_store": 6.0,
    "dsv4_sparse_attention": 7.0, "dsv4_output_projection": 3.0,
}
UNCOMPRESSED = {
    "dsv4_sp_attention_allgather": 2.0, "dsv4_fused_qkv_projection": 10.0,
    "dsv4_fused_qkv_rmsnorm": 1.0, "dsv4_q_projection": 2.0,
    "dsv4_fused_q_kv_prepare": 3.0, "dsv4_sparse_attention": 7.0,
    "dsv4_output_projection": 3.0,
}


def compose(scopes, *, variant, mode, rows=THRESHOLD, short=False, gap=None):
    return compose_attention_phase(
        scopes, dict.fromkeys(scopes, 0.0) if gap is None else gap, variant=variant,
        graph_runtime_mode=mode, rows=rows, multi_stream_gemm_token_threshold=THRESHOLD,
        indexer_short_context=short,
    )


def test_outer_projections_fork_at_the_threshold_and_run_serially_above_it():
    forked = compose(CSA, variant="csa", mode="FULL", rows=THRESHOLD)
    serial = compose(CSA, variant="csa", mode="FULL", rows=THRESHOLD + 1)
    # Forked: the projections start together; the slowest one (aux2, 12) gates the norm.
    assert {forked.start_ms[name] for name in (
        "dsv4_fused_qkv_projection", "dsv4_compressor_projection",
        "dsv4_index_weights_projection", "dsv4_index_compressor_projection",
    )} == {3.0}
    assert forked.start_ms["dsv4_fused_qkv_rmsnorm"] == 15.0
    assert forked.elapsed_ms == 45.0
    # Serial: the default projection first, then the auxiliary ones in list order.
    assert [serial.start_ms[name] for name in (
        "dsv4_fused_qkv_projection", "dsv4_compressor_projection",
        "dsv4_index_weights_projection", "dsv4_index_compressor_projection",
        "dsv4_fused_qkv_rmsnorm",
    )] == [3.0, 13.0, 16.0, 20.0, 32.0]
    assert serial.elapsed_ms == 62.0


def test_full_csa_forks_the_indexer_compressor_beside_the_query_path():
    schedule = compose(CSA, variant="csa", mode="FULL")
    for name in ("dsv4_q_projection", "dsv4_index_query_projection",
                 "dsv4_index_compressor_save_partial_states",
                 "dsv4_compressor_save_partial_states"):
        assert schedule.start_ms[name] == 16.0
    # The main stream joins the attention compressor (aux1, ends at 23).
    assert schedule.start_ms["dsv4_index_scores"] == 23.0


def test_piecewise_keeps_the_csa_fork_and_serializes_the_nested_indexer_fork():
    schedule = compose(CSA, variant="csa", mode="PIECEWISE")
    # `execute_in_parallel` forks in PIECEWISE: the attention compressor starts with the query.
    assert schedule.start_ms["dsv4_compressor_save_partial_states"] == 16.0
    assert schedule.start_ms["dsv4_q_projection"] == 16.0
    # `maybe_execute_in_parallel` is serial on aux0: the indexer compressor follows the q path.
    assert schedule.start_ms["dsv4_index_compressor_save_partial_states"] == 21.0
    assert schedule.start_ms["dsv4_index_scores"] == 25.0
    assert schedule.elapsed_ms == 47.0


def test_hca_compressor_forks_in_full_and_runs_on_main_in_piecewise():
    full = compose(HCA, variant="hca", mode="FULL")
    piecewise = compose(HCA, variant="hca", mode="PIECEWISE")
    assert full.start_ms["dsv4_compressor_save_partial_states"] == 13.0
    assert full.elapsed_ms == 30.0
    assert piecewise.start_ms["dsv4_compressor_save_partial_states"] == 18.0
    assert piecewise.elapsed_ms == 35.0
    assert compose(HCA, variant="hca", mode="NONE").elapsed_ms == full.elapsed_ms


def test_short_context_indexer_runs_its_compressor_and_fill_on_aux0():
    scopes = {name: value for name, value in CSA.items() if name not in {
        "dsv4_index_query_projection", "dsv4_index_fused_query_prepare",
        "dsv4_index_scores", "dsv4_index_topk",
    }}
    scopes["dsv4_index_short_context_fill"] = 1.0
    schedule = compose(scopes, variant="csa", mode="NONE", short=True)
    assert schedule.start_ms["dsv4_index_compressor_save_partial_states"] == 16.0
    assert schedule.start_ms["dsv4_index_short_context_fill"] == 20.0
    # The attention compressor on aux1 (ends at 23) is the last to join.
    assert schedule.start_ms["dsv4_sparse_attention"] == 23.0


def test_each_scope_adds_its_own_signed_gap_once():
    gaps = {name: (i + 1) * 0.1 for i, name in enumerate(UNCOMPRESSED)}
    gaps["dsv4_sparse_attention"] = -0.5
    schedule = compose(UNCOMPRESSED, variant="uncompressed", mode="PIECEWISE", gap=gaps)
    assert schedule.start_ms["dsv4_sparse_attention"] == pytest.approx(
        18.0 + sum(gaps[name] for name in list(UNCOMPRESSED)[:5]))
    assert schedule.elapsed_ms == pytest.approx(28.0 + sum(gaps.values()))
    for mode in ("FULL", "NONE"):
        assert compose(UNCOMPRESSED, variant="uncompressed", mode=mode, gap=gaps) == schedule
    layer = _layer(UNCOMPRESSED, schedule)
    assert layer.get_single_layer_attention_scope_time() == schedule.elapsed_ms
    assert layer.get_single_layer_block_time() == schedule.elapsed_ms


@pytest.mark.parametrize("mode,scope,expected", [
    ("FULL", "dsv4_sparse_attention", "cuda_graph"),
    ("FULL", "dsv4_q_projection", "cuda_graph"),
    ("PIECEWISE", "dsv4_sparse_attention", "eager"),
    ("PIECEWISE", "dsv4_index_scores", "eager"),
    ("PIECEWISE", "dsv4_index_topk", "eager"),
    ("PIECEWISE", "dsv4_attention_output_zero", "eager"),
    ("PIECEWISE", "dsv4_q_projection", "cuda_graph_piecewise"),
    ("NONE", "dsv4_sparse_attention", "eager"),
    ("NONE", "dsv4_q_projection", "eager"),
])
def test_segment_class_mapping(mode, scope, expected):
    assert segment_class(mode, scope) == expected


def test_segment_class_rejects_unknown_mode():
    with pytest.raises(ValueError, match="no segment classes"):
        segment_class("unknown", "dsv4_sparse_attention")


def test_composition_rejects_incomplete_inputs():
    with pytest.raises(ValueError, match="no stream position"):
        compose({**UNCOMPRESSED, "dsv4_index_scores_typo": 1.0},
                variant="uncompressed", mode="FULL")
    for gap in ({"dsv4_sparse_attention": 1.0}, {**dict.fromkeys(UNCOMPRESSED, 0.0), "extra": 0.0}):
        with pytest.raises(ValueError, match="one gap per scope"):
            compose(UNCOMPRESSED, variant="uncompressed", mode="FULL", gap=gap)


def test_serial_schedule_places_scopes_back_to_back():
    schedule = serial_attention_phase([("dsv4_embedding", 1.0), ("dsv4_sparse_attention", 2.5)])
    assert dict(schedule.start_ms) == {"dsv4_embedding": 0.0, "dsv4_sparse_attention": 1.0}
    assert schedule.elapsed_ms == 3.5


def test_trace_places_attention_scopes_at_their_composed_offsets():
    from frontier.metrics.metrics_store import MetricsStore

    schedule = compose(CSA, variant="csa", mode="FULL")
    stage = StageExecutionTime([_layer({**CSA, "dsv4_mhc_attn_post_ffn_pre": 4.0}, schedule)])
    events, advances = [], []
    MetricsStore.__new__(MetricsStore)._emit_stage_layer_traces(
        lambda kind, name, duration, layer_id, meta, start_offset_ms=None: events.append(
            (name, start_offset_ms)
        ),
        stage, ClusterType.MONOLITHIC, moe_tp_enabled=False, ep_enabled=True,
        advance_cursor=advances.append,
    )
    offsets = dict(events)
    assert {name: offsets[name] for name in CSA} == dict(schedule.start_ms)
    assert offsets["dsv4_mhc_attn_post_ffn_pre"] is None
    assert advances == [schedule.elapsed_ms]


def _layer(op_times, schedule):
    layer = ExecutionTime(
        num_layers_per_pipeline_stage=1,
        attention_rope_execution_time=0.0, attention_kv_cache_save_execution_time=0.0,
        attention_decode_execution_time=0.0, attention_prefill_execution_time=0.0,
        attention_layer_pre_proj_execution_time=0.0,
        attention_layer_post_proj_execution_time=0.0,
        attn_norm_time=0.0, mlp_norm_time=0.0, add_time=0.0,
        tensor_parallel_communication_time=0.0, pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0, moe_gating_time=0.0, moe_shuffling_time=0.0,
        schedule_time=0.0, sampler_e2e_time=0.0, prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0, ray_comm_time=0.0, is_moe=True, op_times=op_times,
        global_layer_id=2, attention_family_id="deepseek_v4_attention",
        attention_variant_id="csa",
    )
    layer._dsv4_attention_schedule = schedule
    return layer


@pytest.mark.parametrize("variant,scopes", [("csa", CSA), ("hca", HCA), ("uncompressed", UNCOMPRESSED)])
@pytest.mark.parametrize("mode", ["NONE", "PIECEWISE", "FULL"])
@pytest.mark.parametrize("rows", [THRESHOLD, THRESHOLD + 1])
@pytest.mark.parametrize("short", [False, True])
def test_frontier_schedule_equals_original_stream_clock_walk(variant, scopes, mode, rows, short):
    from frontier.execution_time_predictor.dsv4_composition import MAIN, STREAMS, _layer_program, _schedule
    if variant == "csa" and short:
        scopes = {name: cost for name, cost in scopes.items() if name not in {
            "dsv4_index_query_projection", "dsv4_index_fused_query_prepare"}}
        scopes["dsv4_index_short_context_fill"] = 1.0
    program = _layer_program(variant, mode, rows <= THRESHOLD, short)
    ready, starts = dict.fromkeys(STREAMS, 0.0), {}
    for step, stream, operand in program:
        if step == "wait":
            ready[stream] = max(ready[stream], ready[operand])
        else:
            for name in operand:
                if name in scopes:
                    starts[name] = ready[stream]
                    ready[stream] += scopes[name]
    actual = _schedule(program, scopes)
    assert dict(actual.start_ms) == starts
    assert actual.elapsed_ms == ready[MAIN]


def _preds(scopes, variant="csa", mode="NONE", rows=THRESHOLD, short=False):
    from frontier.execution_time_predictor.dsv4_composition import attention_scope_predecessors
    return attention_scope_predecessors(scopes, variant=variant, graph_runtime_mode=mode,
        rows=rows, multi_stream_gemm_token_threshold=THRESHOLD, indexer_short_context=short)


def test_scope_predecessors_csa_forks_serial_side_and_join():
    forked = _preds(CSA)
    assert forked["dsv4_compressor_projection"] == frozenset({"dsv4_sp_attention_allgather"})
    assert forked["dsv4_fused_qkv_rmsnorm"] == frozenset({"dsv4_fused_qkv_projection",
        "dsv4_compressor_projection", "dsv4_index_weights_projection", "dsv4_index_compressor_projection"})
    assert forked["dsv4_index_compressor_save_partial_states"] == frozenset({
        "dsv4_fused_qkv_rmsnorm", "dsv4_compressor_projection", "dsv4_index_compressor_projection"})
    serial = _preds(CSA, rows=THRESHOLD + 1)
    assert serial["dsv4_compressor_projection"] == frozenset({"dsv4_fused_qkv_projection"})
    assert serial["dsv4_index_weights_projection"] == frozenset({"dsv4_compressor_projection"})
    assert _preds(CSA, mode="PIECEWISE")["dsv4_index_compressor_save_partial_states"] == frozenset({"dsv4_index_fused_query_prepare"})
    assert forked["dsv4_index_scores"] == frozenset({"dsv4_fused_q_kv_prepare",
        "dsv4_index_fused_query_prepare", "dsv4_index_compressor_fused_compress_store",
        "dsv4_compressor_fused_compress_store"})


def test_scope_predecessors_short_context_hca_and_absent_scopes():
    present = {name for name in CSA if name not in {"dsv4_index_query_projection", "dsv4_index_fused_query_prepare",
        "dsv4_index_scores", "dsv4_index_topk"}} | {"dsv4_index_short_context_fill"}
    short = _preds(present, short=True)
    assert short["dsv4_index_short_context_fill"] == frozenset({"dsv4_index_compressor_fused_compress_store"})
    assert _preds(HCA, variant="hca")["dsv4_compressor_save_partial_states"] == frozenset({
        "dsv4_fused_qkv_rmsnorm", "dsv4_compressor_projection"})
    assert _preds(HCA, variant="hca", mode="PIECEWISE")["dsv4_compressor_save_partial_states"] == frozenset({"dsv4_fused_q_kv_prepare"})
    assert _preds({"dsv4_q_projection", "dsv4_output_projection"}, variant="uncompressed") == {
        "dsv4_q_projection": frozenset(), "dsv4_output_projection": frozenset({"dsv4_q_projection"})}
    with pytest.raises(ValueError, match="no stream position"):
        _preds({"unknown"}, variant="uncompressed")
    from frontier.execution_time_predictor.dsv4_composition import MAIN, _schedule
    with pytest.raises(ValueError, match="placed twice"):
        _schedule([("run", MAIN, ("x", "x"))], {"x": 1.0})
