"""GLM TP8/DP4/EP32 source and destination collective contracts."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.cc_backend import MeasuredCCBackend, MeasuredCCBackendConfig, MeasuredCCOutOfRangeError
from frontier.config.model_config import BaseModelConfig
from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
from frontier.entities import Batch, Request
from frontier.entities.time_components import AttentionTime
from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
from frontier.metrics.metrics_store import MetricsStore
from frontier.operators.families import get_comm_operator
from frontier.operators.sequence_parallel_moe import source_tp_operator_names
from frontier.operators.spec import CommPayloadContext
from frontier.types import ClusterType
from tests.unit.test_comm_operator_families import (
    _QuantizationManager, _lane_workload, _moe_predictor, _predictor,
)

ROOT = Path(__file__).resolve().parents[2]
ROLES = (ClusterType.PREFILL, ClusterType.DECODE)


def context(tokens=64, role=ClusterType.DECODE, lanes=None, requests=1):
    model = BaseModelConfig.create_from_name("zai-org/GLM-5.3")
    batch = Batch(0, [Request(0.0, tokens, 1) for _ in range(requests)],
                  [tokens] * requests, is_moe=True)
    # The fixture's physical width is explicit and independent of request count.
    batch.get_effective_total_tokens_rounded = lambda _: tokens
    batch.moe_sequence_parallel_token_widths = sequence_parallel_moe_token_widths(
        [tokens] * 4 if lanes is None else lanes, 8,
    )
    replica = SimpleNamespace(
        attn_tensor_parallel_size=8, attn_dp=4, num_pipeline_stages=1,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=32, router_topk=8,
    )
    return CommPayloadContext(batch, model, replica, role, _QuantizationManager())


def measured_backend(role=ClusterType.DECODE):
    return MeasuredCCBackend(
        MeasuredCCBackendConfig(summary_csv_path=str(ROOT / "data/profiling/network/h800_dgx/summary.csv")),
        role, "h800", "h800_dgx", 32,
    )


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("tokens,rows", [(1, 1), (64, 8), (8192, 1024), (65, 9)])
@pytest.mark.parametrize("name,alias,multiplier", [
    ("sp_moe_attention_reduce_scatter", "reduce_scatter", 8),
    ("sp_moe_layer_input_allgather", "allgather", 1),
    ("sp_moe_final_allgather", "allgather", 2),
])
def test_source_sp_payload(role, tokens, rows, name, alias, multiplier):
    ctx = context(tokens, role)
    operator = get_comm_operator(name)
    assert operator.resolve_collective_alias(ctx) == alias
    assert operator.num_devices(ctx) == 8
    assert operator.resolve_comm_domain(ctx) == "ATTN_TP"
    assert operator.build_payload_bytes(ctx) == rows * 6144 * 2 * multiplier
    assert ctx.quantization_manager.calls == []


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("lanes,rows,gathered", [
    ([1] * 4, 1, 32), ([64] * 4, 8, 256), ([8192] * 4, 1024, 32768),
    ([65, 1, 8, 16], 9, 104), ([1, 2, 7, 8], 1, 32),
])
def test_grouped_ep_payload_and_alias(role, lanes, rows, gathered):
    ctx = context(lanes[0], role, lanes)
    for name, alias, size in (
        ("expert_parallel_alltoall_dispatch", "allgather", rows * (6144 * 2 + 8 * 4 + 8 * 4)),
        ("expert_parallel_alltoall_combine", "reduce_scatter", gathered * 6144 * 2),
    ):
        operator = get_comm_operator(name)
        assert operator.resolve_collective_alias(ctx) == alias
        assert operator.resolve_comm_domain(ctx) == "EP"
        assert operator.num_devices(ctx) == 32
        assert operator.build_payload_bytes(ctx) == size
    assert ctx.quantization_manager.calls == []


@pytest.mark.parametrize("role", ROLES)
def test_unequal_combine_rejected_before_measured_lookup(role):
    ctx = context(64, role, [64, 16, 1, 1])
    predictor = _predictor()
    predictor._cc_backend = measured_backend(role)
    with pytest.raises(MeasuredCCOutOfRangeError, match="unequal SP rows") as caught:
        predictor._predict_comm_operator_with_context(get_comm_operator("expert_parallel_alltoall_combine"), ctx)
    assert caught.value.curve_key == ("reduce_scatterv", "ep", 32)
    assert caught.value.requested_bytes == 96 * 6144 * 2
    assert caught.value.covered_range == (393216, 402653184)


@pytest.mark.parametrize("role", ROLES)
def test_equal_sp_rows_with_unequal_raw_tokens_are_covered(role):
    ctx = context(1, role, [1, 2, 7, 8])
    predictor = _predictor()
    predictor._cc_backend = measured_backend(role)
    predictor._cc_backend._predict_unequal = lambda *_, **__: pytest.fail("uniform SP must not use exact-pattern lookup")
    operator = get_comm_operator("expert_parallel_alltoall_combine")
    assert predictor._predict_comm_operator_with_context(operator, ctx) == (
        predictor._predict_comm_operator_with_context(operator, context(8, role, [8] * 4))
    )


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("lanes", [[256, 1, 1, 1], [249, 2, 7, 8], [256, 8, 8, 8]])
def test_unequal_combine_uses_exact_retained_k5_point(role, lanes):
    ctx = context(lanes[0], role, lanes)
    predictor = _predictor()
    predictor._cc_backend = measured_backend(role)
    operator = get_comm_operator("expert_parallel_alltoall_combine")
    assert operator.build_payload_bytes(ctx) == 3440640
    time_ms = predictor._predict_comm_operator_with_context(operator, ctx)
    # Median of all six retained t256/r1 and t256/r2 pass medians, without rounding.
    assert time_ms == pytest.approx(.3395576000213623)
    assert time_ms * 1000 == pytest.approx(339.5, abs=.1)


@pytest.mark.parametrize("lanes", [[8, 256, 8, 8], [248, 16, 8, 8], [128, 128, 8, 8]])
def test_unequal_combine_unmeasured_vectors_do_not_use_neighbor_or_permutation(lanes):
    ctx = context(lanes[0], lanes=lanes)
    predictor = _predictor()
    predictor._cc_backend = measured_backend()
    with pytest.raises(MeasuredCCOutOfRangeError, match="unequal SP rows"):
        predictor._predict_comm_operator_with_context(get_comm_operator("expert_parallel_alltoall_combine"), ctx)


def test_unequal_dispatch_keeps_largest_rank_uniform_curve():
    ctx = context(256, lanes=[256, 1, 1, 1])
    predictor = _predictor()
    predictor._cc_backend = measured_backend()
    predictor._cc_backend._predict_unequal = lambda *_, **__: pytest.fail("dispatch must not use exact-pattern lookup")
    operator = get_comm_operator("expert_parallel_alltoall_dispatch")
    assert predictor._predict_comm_operator_with_context(operator, ctx) == (
        predictor._cc_backend.predict_allgather(395264, 32, comm_domain="EP")
    )


def test_unequal_combine_other_backends_keep_scalar_prediction():
    ctx = context(256, lanes=[256, 1, 1, 1])
    predictor = _predictor()
    operator = get_comm_operator("expert_parallel_alltoall_combine")
    assert predictor._predict_comm_operator_with_context(operator, ctx) == 3440640 / 5000 + 32
    assert predictor._cc_backend.calls[-1]["collective_alias"] == "reduce_scatter"


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("tokens,requests", [(1, 1), (64, 64), (8192, 1), (65, 2)])
def test_embedding_and_selected_logits_rows(role, tokens, requests):
    ctx = context(tokens, role, requests=requests)
    assert get_comm_operator("embedding_allreduce").build_payload_bytes(ctx) == tokens * 6144 * 2
    assert get_comm_operator("logits_allgather").build_payload_bytes(ctx) == requests * 19360 * 2
    assert ctx.quantization_manager.calls == []


def test_no_logits_collective_for_dummy_rows():
    ctx = context(1, requests=0)
    predictor = _predictor()
    assert predictor._predict_comm_operator_with_context(get_comm_operator("logits_allgather"), ctx) == 0
    assert predictor._cc_backend.calls == []


@pytest.mark.parametrize("model_name", [
    "deepseek-v3", "Qwen3-235B-A22B", "mixtral_8x7b_moe",
])
def test_non_sp_profiles_keep_legacy_ep_payload_and_alias(model_name):
    ctx = replace(context(), model_config=BaseModelConfig.create_from_name(model_name),
                  lane_workload=_lane_workload({0: 5, 1: 5}))
    for name in ("expert_parallel_alltoall_dispatch", "expert_parallel_alltoall_combine"):
        operator = get_comm_operator(name)
        assert operator.resolve_collective_alias(ctx) == "alltoall"
        assert operator.build_payload_bytes(ctx) == ctx.model_config.embedding_dim * 2 * 10 + 11
    assert source_tp_operator_names(ctx.model_config, ctx.replica_config, 4) == ()


def source_predictor(ctx):
    predictor = _moe_predictor()
    predictor._model_config = ctx.model_config
    predictor._replica_config = ctx.replica_config
    predictor._cluster_type = ctx.cluster_type
    predictor.predict_attention_layer_time = lambda **_: AttentionTime()
    for name in (
        "_get_share_expert_up_proj_execution_time", "_get_share_expert_down_proj_execution_time",
        "_get_share_expert_act_execution_time", "_get_gating_linear_time", "_get_gating_routing_topk_time",
        "_get_moe_shuffling_time", "_get_grouped_gemm_time", "_get_add_layer_act_execution_time",
        "_get_mlp_norm_layer_act_execution_time", "_get_mlp_layer_up_proj_execution_time",
        "_get_mlp_layer_down_proj_execution_time", "_get_mlp_layer_act_execution_time",
        "_get_pp_producer_send_path_runtime_time", "_get_pp_receiver_head_runtime_time",
        "_get_pp_prefill_consumer_active_runtime_time", "_get_mtp_terminal_overshoot_time",
    ):
        setattr(predictor, name, lambda *args, **kwargs: 0.0)
    predictor._predict_expert_parallel_phase_operator_times = lambda *args, **kwargs: {}
    predictor.predict_dp_moe_allreduce_times = lambda *_: (0.0, 0.0)
    return predictor


def pd_source_predictor(ctx):
    from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
        SklearnDisaggregationExecutionTimePredictor,
    )

    base = source_predictor(ctx)
    cls = type("PDSourcePredictor", (SklearnDisaggregationExecutionTimePredictor, type(base)), {})
    predictor = cls.__new__(cls)
    predictor.__dict__.update(base.__dict__)
    ctx.replica_config.model_config = ctx.model_config
    predictor._get_cluster_replica_config = lambda _: ctx.replica_config
    predictor._log_architecture_attention_shape = lambda *_: None
    predictor._get_tensor_parallel_communication_time = lambda _: 10.0
    predictor._get_pipeline_parallel_communication_time = lambda _: 0.0
    predictor._predict_one_op_time = lambda name, time, *args: time
    predictor.predict_attention_layer_time = lambda *args, **kwargs: AttentionTime()
    return predictor


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("layer,expected", [
    (0, {"embedding_allreduce"}), (2, set()),
    (3, {"sp_moe_attention_reduce_scatter"}),
    (4, {"sp_moe_attention_reduce_scatter", "sp_moe_layer_input_allgather"}),
    (77, {"sp_moe_attention_reduce_scatter", "sp_moe_layer_input_allgather", "sp_moe_final_allgather", "logits_allgather"}),
])
def test_pd_live_attention_source_calls_and_accounting(role, layer, expected):
    ctx = context(64, role)
    predictor = pd_source_predictor(ctx)
    predictor._cc_backend = measured_backend(role)
    if layer >= 3:
        predictor._get_tensor_parallel_communication_time = lambda _: pytest.fail("SP must not look up legacy attention all-reduce")
    comm = predictor._get_communication_time(ctx.batch, 0, role, layer_id=layer, include_stage_owned=False)
    assert set(comm.source_operator_times) == expected
    assert comm.tensor_parallel_time == (0.0 if layer >= 3 else 10.0)
    result = predictor._predict_attention_only_stage_execution_time(
        batch=ctx.batch, stage_id=0, cluster_type=role, num_layers=1,
        layer_id=layer, include_stage_owned=False,
    )
    assert result.model_time_ms == pytest.approx(sum(comm.source_operator_times.values()) + comm.tensor_parallel_time)
    assert sum(MetricsStore.__new__(MetricsStore)._build_frontier_stage_batch_component_ledger(result).values()) == pytest.approx(result.model_time_ms)


@pytest.mark.parametrize("role", ROLES)
def test_pd_dummy_public_stage_charges_actual_source_calls(role):
    ctx = context(64, role)
    predictor = pd_source_predictor(ctx)
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 0.01
    result = predictor.predict_stage_execution_time(ctx.batch, 0, role, num_layers=78, include_ffn=False)
    times = result.op_times
    assert result.attention_all_reduce_time == pytest.approx(0.03)
    for name, calls in {"embedding_allreduce": 1, "logits_allgather": 1,
                       "sp_moe_attention_reduce_scatter": 75,
                       "sp_moe_layer_input_allgather": 74, "sp_moe_final_allgather": 1}.items():
        assert times[name] == pytest.approx(calls * 0.01)
    comm = result.communication_time_component
    assert comm.total_time() - comm.pipeline_parallel_send_recv_time == pytest.approx(1.55)


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("layer,expected", [
    (0, {"embedding_allreduce", "attn_tensor_parallel_allreduce", "mlp_tensor_parallel_allreduce"}),
    (2, {"attn_tensor_parallel_allreduce", "mlp_tensor_parallel_allreduce"}),
    (3, {"sp_moe_attention_reduce_scatter"}),
    (4, {"sp_moe_attention_reduce_scatter", "sp_moe_layer_input_allgather"}),
    (77, {"sp_moe_attention_reduce_scatter", "sp_moe_layer_input_allgather", "sp_moe_final_allgather", "logits_allgather"}),
])
def test_live_source_layer_collectives_and_total(role, layer, expected):
    ctx = context(64, role)
    predictor = source_predictor(ctx)
    result = predictor._get_execution_time_internal(
        ctx.batch, 0, moe_tokens_input=64 if layer >= 3 else None,
        include_moe=layer >= 3, layer_id=layer, include_stage_owned=False,
    )
    assert set(result.communication_operator_times.op_times) == expected
    assert result.communication_time_component.total_time() == pytest.approx(sum(result.communication_operator_times.op_times.values()))
    assert result.model_time_ms == pytest.approx(sum(result.communication_operator_times.op_times.values()))
    ledger = MetricsStore.__new__(MetricsStore)._build_frontier_stage_batch_component_ledger(result)
    assert sum(ledger.values()) == pytest.approx(result.model_time_ms)


@pytest.mark.parametrize("role", ROLES)
def test_split_execution_replay_counts_source_and_ep_calls_once(role):
    from tests.integration.run_glm53_comm_replay import replay

    result = replay(role, [64] * 4, 1)
    before = {row["operator"]: row["calls"] for row in result["before_analytical"]["operators"]}
    after = {row["operator"]: row["calls"] for row in result["after_measured"]["operators"]}
    assert before == {"attn_tensor_parallel_allreduce": 78, "mlp_tensor_parallel_allreduce": 3,
                      "expert_parallel_alltoall": 150}
    assert after == {"embedding_allreduce": 1, "attn_tensor_parallel_allreduce": 3,
        "mlp_tensor_parallel_allreduce": 3, "sp_moe_attention_reduce_scatter": 75,
        "sp_moe_layer_input_allgather": 74, "sp_moe_final_allgather": 1,
        "logits_allgather": 1, "expert_parallel_alltoall_dispatch": 75,
        "expert_parallel_alltoall_combine": 75}
    assert result["after_measured"]["step_ms"] > 0


def test_tp8_dp1_preserves_non_sp_moe_ffn_allreduce_copy():
    ctx = context()
    ctx.replica_config.attn_dp = 1
    ctx.replica_config.moe_expert_parallel_size = 8
    predictor = source_predictor(ctx)
    result = predictor._get_execution_time_internal(ctx.batch, 0, moe_tokens_input=64,
        include_moe=True, layer_id=4, include_stage_owned=False)
    assert set(result.communication_operator_times.op_times) == {
        "attn_tensor_parallel_allreduce", "mlp_tensor_parallel_allreduce"}


@pytest.mark.parametrize("model_name", ["deepseek-v3", "Qwen3-235B-A22B", "mixtral_8x7b_moe"])
def test_non_sp_live_moe_preserves_ffn_allreduce_copy(model_name):
    ctx = replace(context(), model_config=BaseModelConfig.create_from_name(model_name))
    predictor = source_predictor(ctx)
    result = predictor._get_execution_time_internal(ctx.batch, 0, moe_tokens_input=64,
        include_moe=True, layer_id=4, include_stage_owned=False)
    assert result.communication_operator_times.op_times == {
        "attn_tensor_parallel_allreduce": pytest.approx(ctx.model_config.embedding_dim * 2 * 64 / 1000 + 8),
        "mlp_tensor_parallel_allreduce": pytest.approx(ctx.model_config.embedding_dim * 2 * 64 / 1000 + 8),
    }


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("tokens", [1, 64, 8192])
def test_every_collective_resolves_measured_coverage(role, tokens):
    ctx = context(tokens, role)
    predictor = _predictor()
    predictor._cc_backend = measured_backend(role)
    names = source_tp_operator_names(ctx.model_config, ctx.replica_config, 0)
    names += source_tp_operator_names(ctx.model_config, ctx.replica_config, 77)
    names += ("expert_parallel_alltoall_dispatch", "expert_parallel_alltoall_combine")
    for name in names:
        assert predictor._predict_comm_operator_with_context(get_comm_operator(name), ctx) > 0


def test_prefill_above_dispatch_range_is_typed():
    ctx = context(8193, ClusterType.PREFILL)
    predictor = _predictor()
    predictor._cc_backend = measured_backend(ctx.cluster_type)
    with pytest.raises(MeasuredCCOutOfRangeError) as caught:
        predictor._predict_comm_operator_with_context(get_comm_operator("expert_parallel_alltoall_dispatch"), ctx)
    assert caught.value.curve_key == ("all_gatherv", "ep", 32)
    assert caught.value.requested_bytes == 1025 * 12352


def test_grouped_dispatch_trace_contains_three_tensor_dtypes_and_one_call():
    ctx = context(65, lanes=[65, 1, 8, 16])
    trace_ctx = OpTraceContext(ctx.cluster_type, ctx.model_config, ctx.replica_config,
        65, 9, 9, 9, False, sequence_parallel_token_widths=ctx.batch.moe_sequence_parallel_token_widths)
    meta = compute_op_trace_meta("expert_parallel_alltoall_dispatch", "COMM", trace_ctx)
    assert meta["collective"] == "all_gatherv"
    assert meta["tensor_shape"] == {"hidden": [9, 6144], "topk_weights": [9, 8], "topk_ids": [9, 8]}
    assert meta["tensor_dtype"] == {"hidden": "BF16", "topk_weights": "FP32", "topk_ids": "INT32"}
    assert meta["data_size_bytes"] == get_comm_operator("expert_parallel_alltoall_dispatch").build_payload_bytes(ctx)


@pytest.mark.parametrize("name", ["sp_moe_attention_reduce_scatter", "sp_moe_layer_input_allgather",
    "sp_moe_final_allgather", "embedding_allreduce", "logits_allgather"])
def test_source_trace_bytes_match_predictor_payload(name):
    ctx = context(65, requests=2)
    trace_ctx = OpTraceContext(ctx.cluster_type, ctx.model_config, ctx.replica_config,
        65, 65, 65, 65, False, selected_logits_rows=2)
    meta = compute_op_trace_meta(name, "COMM", trace_ctx)
    assert meta["data_size_bytes"] == get_comm_operator(name).build_payload_bytes(ctx)
    assert meta["dtype"] == "BF16"
