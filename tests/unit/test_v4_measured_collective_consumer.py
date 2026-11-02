"""V4 native collective implementation, payload and ordered-width contracts."""

from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention.deepseek_v4 import compute_rows
from frontier.cc_backend import MeasuredCCBackend, MeasuredCCBackendConfig, MeasuredCCOutOfRangeError
from frontier.config.model_config import BaseModelConfig
from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
from frontier.entities import Batch, DummyForwardBatch, Request, StageExecutionTime
from frontier.entities.batch import DecodeCudaGraphMetadata
from frontier.execution_time_predictor.deepseek_v4 import source_sp_token_widths
from frontier.model_architectures import ModelArchitectureProfile
from frontier.moe_ep_workload import build_contiguous_expert_ownership, materialize_layer_ep_workload
from frontier.operators.families import get_comm_operator
from frontier.operators.sequence_parallel_moe import ep_widths, ep_workload, measured_collective_time
from frontier.operators.spec import CollectiveImplementation, CommPayloadContext
from frontier.profiling.collectives.vllm_dsv4_forward_collectives import (
    DeepseekV4Shape, DSV4_TOPK_IDS_DTYPE, dsv4_forward_collectives,
)
from frontier.profiling.collectives.vllm_forward_collectives import (
    ForwardScenario, ServingParallelism, TOPK_IDS_DTYPE, call_bytes,
)
from frontier.scheduler.utils.ep_wave import prepare_moe_wave_from_inputs
from frontier.scheduler.utils.expert_parallel import get_ep_phase_times_ms
from frontier.types import ClusterType
from test_dsv4_source_execution import predictor, source_batch

ROOT = Path(__file__).resolve().parents[2]
CALL_OPERATORS = {
    "embedding_all_reduce": "dsv4_embedding_allreduce",
    "sp_attention_all_gather": "dsv4_sp_attention_allgather",
    "sp_attention_reduce_scatter": "dsv4_sp_attention_reduce_scatter",
    "sp_final_mhc_all_gather": "dsv4_sp_final_mhc_allgather",
    "moe_dispatch_all_gatherv": "dsv4_ep_dispatch_allgather",
    "moe_combine_reduce_scatterv": "dsv4_ep_output_reduce_scatter",
    "logits_all_gather": "dsv4_logits_gather",
}


class UnadjustedActivations:
    def adjust_tensor_size(self, operation, size, role):
        raise AssertionError("Native BF16 payloads must bypass weight quantization")


def context(lanes=(1, 1), *, selected=None, role=ClusterType.MONOLITHIC, model=None):
    model = model or BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
    replica = SimpleNamespace(
        num_pipeline_stages=1, attn_tensor_parallel_size=8, attn_dp=len(lanes),
        moe_tensor_parallel_size=1, moe_expert_parallel_size=8 * len(lanes),
        router_topk=model.num_experts_per_tok,
    )
    widths = sequence_parallel_moe_token_widths(lanes, 8)
    workload = materialize_layer_ep_workload(
        routing_ratios={i: 1 / model.num_experts for i in range(model.num_experts)},
        target_replica_id=0, global_layer_id=0, routing_token_count=widths.gathered_tokens,
        router_topk=replica.router_topk, total_expert_num=model.num_experts,
        moe_expert_parallel_size=replica.moe_expert_parallel_size,
        expert_to_ep=build_contiguous_expert_ownership(model.num_experts, replica.moe_expert_parallel_size),
    )
    return CommPayloadContext(
        source_batch(max(lanes)), model, replica, role, UnadjustedActivations(),
        sp_token_widths=widths, selected_logits_rows=selected,
    ), workload


@pytest.fixture(scope="module")
def h200():
    return MeasuredCCBackend(
        MeasuredCCBackendConfig(summary_csv_path=str(ROOT / "data/profiling/network/h200_dgx/summary.csv")),
        ClusterType.MONOLITHIC, "h200", "h200_dgx", 32,
    )


@pytest.mark.parametrize("name,implementation,curve", [
    ("dsv4_embedding_allreduce", CollectiveImplementation.GROUP_COORDINATOR_DIM0, "all_reduce"),
    ("dsv4_sp_attention_allgather", CollectiveImplementation.VLLM_SEQUENCE_PARALLEL, "sp_all_gather"),
    ("dsv4_sp_attention_reduce_scatter", CollectiveImplementation.VLLM_SEQUENCE_PARALLEL, "sp_reduce_scatter"),
    ("dsv4_sp_final_mhc_allgather", CollectiveImplementation.VLLM_SEQUENCE_PARALLEL, "sp_all_gather"),
    ("dsv4_logits_gather", CollectiveImplementation.GROUP_COORDINATOR_LAST_DIM, "logits_all_gather"),
    ("dsv4_ep_dispatch_allgather", CollectiveImplementation.GROUP_COORDINATOR_DIM0, "all_gatherv"),
    ("dsv4_ep_output_reduce_scatter", CollectiveImplementation.GROUP_COORDINATOR_DIM0, "reduce_scatterv"),
])
def test_v4_spec_resolves_native_curve(h200, name, implementation, curve):
    ctx, _ = context(selected=1)
    operator = get_comm_operator(name)
    assert operator.collective_implementation is implementation
    size = operator.build_payload_bytes(ctx)
    key = (curve, "ep" if operator.comm_domain == "MOE_EP" else "tp", operator.num_devices(ctx))
    assert measured_collective_time(operator, ctx, h200, size) == h200._curves[key].predict_us(size) / 1000


def test_glm_specs_keep_group_coordinator_default():
    for name in ("sp_moe_attention_reduce_scatter", "sp_moe_layer_input_allgather",
                 "sp_moe_final_allgather", "embedding_allreduce", "logits_allgather",
                 "expert_parallel_alltoall_dispatch", "expert_parallel_alltoall_combine"):
        assert get_comm_operator(name).collective_implementation is CollectiveImplementation.GROUP_COORDINATOR_DIM0


@pytest.mark.parametrize("dp", [2, 4])
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 24, 32, 6144])
def test_v4_payloads_equal_producer_grid(dp, tokens):
    ctx, _ = context((tokens,) * dp)
    parallel = ServingParallelism.from_engine(dict(tensor_parallel_size=8, data_parallel_size=dp,
                                               pipeline_parallel_size=1, enable_expert_parallel=True,
                                               dtype="bfloat16", async_scheduling=True))
    model = ctx.model_config
    shape = DeepseekV4Shape(model.embedding_dim, model.num_layers, model.num_experts_per_tok,
                           model.vocab_size, model.hc_mult)
    calls = dsv4_forward_collectives(shape, parallel, ForwardScenario("grid", (tokens,) * dp, (0,) * dp, tokens != 6144))
    for call in calls:
        operator = get_comm_operator(CALL_OPERATORS[call.name])
        expected = max(call_bytes(call, parallel, rank)[0] for rank in range(dp * 8))
        assert operator.build_payload_bytes(ctx) == expected


@pytest.mark.parametrize("lanes", [(1, 6144), (6144, 1), *product((1, 6144), repeat=4)])
def test_grouped_dispatch_and_combine_match_ordered_eager_grid(h200, lanes):
    ctx, _ = context(lanes)
    widths = ep_widths(ctx)
    dispatch = get_comm_operator("dsv4_ep_dispatch_allgather")
    combine = get_comm_operator("dsv4_ep_output_reduce_scatter")
    assert dispatch.build_payload_bytes(ctx) == max(widths.local_tokens) * 8264
    size = combine.build_payload_bytes(ctx)
    assert size == widths.gathered_tokens * 8192
    if len(set(widths.local_tokens)) > 1:
        key = (("reduce_scatterv", "ep", len(lanes) * 8), widths.local_tokens, size)
        assert measured_collective_time(combine, ctx, h200, size) == h200._unequal_combine_times[key] / 1000


@pytest.mark.parametrize("b", range(1, 25))
def test_logits_selected_row_grid(b):
    ctx, _ = context(selected=b)
    assert get_comm_operator("dsv4_logits_gather").build_payload_bytes(ctx) == b * 32320


@pytest.mark.parametrize("profile,lanes,name", [
    ("deepseek-ai/DeepSeek-V4-Flash", (1, 6144), "dsv4_ep_output_reduce_scatter"),
    ("zai-org/GLM-5.3", (64, 1, 1, 1), "expert_parallel_alltoall_combine"),
])
def test_exact_combine_is_spec_driven_with_renamed_operator(profile, lanes, name, h200):
    ctx, _ = context(lanes, model=BaseModelConfig.create_from_name(profile))
    operator = replace(get_comm_operator(name), name="another_combine_name")
    backend = h200
    if name == "expert_parallel_alltoall_combine":
        ctx.batch.moe_sequence_parallel_token_widths = ctx.sp_token_widths
        ctx = replace(ctx, sp_token_widths=None)
        backend = MeasuredCCBackend(MeasuredCCBackendConfig(), ClusterType.MONOLITHIC, "h800", "h800_dgx", 32)
    size = operator.build_payload_bytes(ctx)
    key = (("reduce_scatterv", "ep", len(lanes) * 8), operator.sp_rows_builder(ctx), size)
    assert measured_collective_time(operator, ctx, backend, size) == backend._unequal_combine_times[key] / 1000


@pytest.mark.parametrize("size_delta,rows", [(1, (1, 768)), (0, (768, 2))])
def test_unmeasured_combine_size_or_ordered_vector_is_typed(h200, size_delta, rows):
    with pytest.raises(MeasuredCCOutOfRangeError) as caught:
        h200.predict_operator_collective("reduce_scatter", CollectiveImplementation.GROUP_COORDINATOR_DIM0,
                                        50397184 + size_delta, 16, "MOE_EP", rows)
    assert caught.value.curve_key == ("reduce_scatterv", "ep", 16)
    assert "exact per-root Reduce" in caught.value.reason


@pytest.mark.parametrize("name,key", [
    ("dsv4_sp_attention_allgather", ("sp_all_gather", "tp", 8)),
    ("dsv4_sp_attention_reduce_scatter", ("sp_reduce_scatter", "tp", 8)),
    ("dsv4_logits_gather", ("logits_all_gather", "tp", 8)),
])
def test_outside_uniform_measurement_names_implementation_curve(h200, name, key):
    ctx, _ = context(selected=1)
    operator = get_comm_operator(name)
    with pytest.raises(MeasuredCCOutOfRangeError) as caught:
        measured_collective_time(operator, ctx, h200, h200._curves[key].sizes[-1] + 1)
    assert caught.value.curve_key == key


@pytest.mark.parametrize("name", CALL_OPERATORS.values())
def test_non_measured_backend_uses_alias_calls(predictor, name):
    ctx, _ = context(selected=1)
    operator = get_comm_operator(name)
    assert measured_collective_time(operator, ctx, predictor._cc_backend, operator.build_payload_bytes(ctx)) is None
    result = predictor._predict_comm_operator_with_context(operator, ctx)
    alias, kwargs = predictor._cc_backend.calls[-1]
    assert alias == operator.resolve_collective_alias(ctx)
    assert kwargs["data_size_bytes"] == operator.build_payload_bytes(ctx)
    assert result == {"allreduce": 7, "allgather": 3, "reduce_scatter": 5}[alias]


@pytest.mark.parametrize("role", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("name", CALL_OPERATORS.values())
def test_pdd_payload_equals_colocation_with_descriptor_width(role, name):
    ctx, _ = context((17, 1), selected=1)
    ctx = replace(ctx, batch=source_batch(17))
    ctx.batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
        "full_and_piecewise", "FULL", True, False, 17, 40, 1, 40,
    )
    # PREFILL's old rounding API ignores the descriptor; compute_rows uses V2.
    assert compute_rows(ctx.batch) == 40
    assert ctx.batch.get_effective_total_tokens_rounded(ClusterType.PREFILL) == 17
    operator = get_comm_operator(name)
    assert operator.build_payload_bytes(replace(ctx, cluster_type=role)) == operator.build_payload_bytes(ctx)


@pytest.mark.parametrize("role", [ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN])
@pytest.mark.parametrize("name", CALL_OPERATORS.values())
def test_pdaf_payload_stays_rejected(role, name):
    ctx, _ = context(selected=1)
    operator = get_comm_operator(name)
    with pytest.raises(ValueError, match="MONOLITHIC, PREFILL or DECODE"):
        operator.build_payload_bytes(replace(ctx, cluster_type=role))


def test_ids_dtype_is_owned_by_profile_and_required_only_for_dispatch():
    assert TOPK_IDS_DTYPE == ModelArchitectureProfile.glm_moe_dsa().gathered_topk_ids_dtype == "int32"
    assert DSV4_TOPK_IDS_DTYPE == ModelArchitectureProfile.deepseek_v4().gathered_topk_ids_dtype == "int64"
    profile = replace(ModelArchitectureProfile.deepseek_v4(), gathered_topk_ids_dtype=None)
    ctx, _ = context()
    model = SimpleNamespace(embedding_dim=4096, get_model_architecture_profile=lambda: profile)
    with pytest.raises(ValueError, match="SP EP dispatch requires gathered_topk_ids_dtype.*deepseek_v4"):
        ep_workload("dsv4_ep_dispatch_allgather", model, 6, ep_widths(ctx))
    assert ep_workload("dsv4_ep_output_reduce_scatter", model, 6, ep_widths(ctx))["data_size_bytes"] == 131072


@pytest.mark.parametrize("tokens,expected", [((17, 1), (3, 1)), ((1, 17), (1, 3))])
def test_source_widths_are_ordered_validated_and_read_through_context(tokens, expected):
    batches = {d: source_batch(t) for d, t in enumerate(tokens)}
    p = SimpleNamespace(_replica_config=SimpleNamespace(attn_dp=2, attn_tensor_parallel_size=8))
    widths = source_sp_token_widths(p, batches)
    assert widths.local_tokens == expected
    assert widths.gathered_tokens == 32
    ctx, _ = context(tokens)
    assert ep_widths(replace(ctx, sp_token_widths=widths)) is widths
    assert all(not hasattr(batch, "moe_sequence_parallel_token_widths") for batch in batches.values())
    with pytest.raises(ValueError, match="every attention DP rank"):
        source_sp_token_widths(p, {0: batches[0]})


def test_v4_wave_passes_widths_without_mutating_source_batches(predictor, monkeypatch):
    ctx, _ = context((6144, 1))
    predictor._replica_config = ctx.replica_config
    predictor._replica_config.total_expert_num = 256
    predictor._replica_config.attention_backend = "FLASHMLA_SPARSE_DSV4"
    batches = {0: source_batch(6144), 1: DummyForwardBatch(0, 0, 0)}
    before = {d: dict(vars(batch)) for d, batch in batches.items()}
    aggregate = Batch(0, [Request(0, 6145, 2)], [6145], is_moe=True)
    seen = []
    original = predictor._predict_comm_operator_with_context

    def record(operator, payload_ctx):
        if operator.comm_domain == "MOE_EP":
            seen.append((operator.name, ep_widths(payload_ctx).local_tokens,
                         operator.build_payload_bytes(payload_ctx)))
        return original(operator, payload_ctx)

    monkeypatch.setattr(predictor, "_predict_comm_operator_with_context", record)
    monkeypatch.setattr(predictor, "_assemble_stage", lambda layers, **_: StageExecutionTime(layers))
    monkeypatch.setattr(predictor, "_get_cluster_replica_config", lambda _: predictor._replica_config)
    from frontier.config import global_vars
    monkeypatch.setattr(global_vars, "KV_CONNECTOR", "none")
    rows = []

    def materialize(**kwargs):
        rows.append(kwargs["routing_token_count"])
        return materialize_layer_ep_workload(
            routing_ratios={i: 1 / 256 for i in range(256)}, target_replica_id=0,
            global_layer_id=3, routing_token_count=kwargs["routing_token_count"],
            router_topk=6, total_expert_num=256, moe_expert_parallel_size=16,
            expert_to_ep=build_contiguous_expert_ownership(256, 16),
        )

    def noop(**kwargs):
        return None

    prepare_moe_wave_from_inputs(
        wave_inputs=SimpleNamespace(aggregate_batch=aggregate, sample_batch=batches[0],
                                    step_id=1, source_batches=batches),
        time=0, materialize_workload=materialize, trace_identity_builder=noop,
        conservation_logger=noop, predictor=predictor,
        lane_builder=lambda **kw: SimpleNamespace(lane_workload=kw["layer_workload"].lane(kw["ep_id"])),
        phase_getter=get_ep_phase_times_ms, workload_logger=noop, barrier_logger=noop,
        wave_logger=noop, cluster_type=ClusterType.MONOLITHIC, replica_id=0, stage_id=0, layer_id=3,
    )
    assert rows == [6152]
    assert len(seen) == 32
    assert {vector for _, vector, _ in seen} == {(768, 1)}
    assert {size for name, _, size in seen if name == "dsv4_ep_dispatch_allgather"} == {6346752}
    assert {size for name, _, size in seen if name == "dsv4_ep_output_reduce_scatter"} == {50397184}
    assert all(vars(batch) == before[d] for d, batch in batches.items())


def test_h200_row_classes_are_separate(h200):
    assert len(h200.describe_curves()) == 8
    assert len(h200._unequal_combine_times) == 16
    assert len(h200.describe_holdouts()) == 48
    assert {row["call"] for row in h200.describe_holdouts()} == {"moe_dispatch_all_gatherv"}
    assert all(row["curve_us"] is not None for row in h200.describe_holdouts())


@pytest.mark.parametrize("dummy,layer", [(False, 0), (True, 0), (False, 42), (True, 42)])
def test_endpoints_use_measured_backend_and_forward_presence(predictor, h200, dummy, layer):
    ctx, workload = context(selected=0 if dummy else (1 if layer == 42 else None))
    predictor._replica_config = ctx.replica_config
    predictor._replica_config.total_expert_num = 256
    predictor._replica_config.attention_backend = "FLASHMLA_SPARSE_DSV4"
    predictor._replica_config.speculative_decoding_config = SimpleNamespace(enabled=False)
    predictor._cc_backend = h200
    batch = DummyForwardBatch(0, 0, 0) if dummy else source_batch(1)
    timing = predictor._predict_dsv4_structural_execution(
        batch, layer, 0, lane_workload=workload.lane(0), sp_token_widths=ctx.sp_token_widths,
        execution_mode="eager", prefix_caching=False, decode_context_parallel_size=1,
        prefill_context_parallel_size=1, enable_expert_parallel=True, moe_backend="auto",
        all2all_backend="allgather_reducescatter", skip_padding=False, input_padding_mask=None,
        selected_logits_rows=(0 if dummy else 1) if layer == 42 else None,
    )
    assert ("dsv4_embedding_allreduce" in timing.op_times) == (layer == 0)
    assert ("dsv4_logits_gather" in timing.op_times) == (layer == 42 and not dummy)
    if layer == 0:
        assert timing.op_times["dsv4_embedding_allreduce"] == h200._curves[("all_reduce", "tp", 8)].predict_us(8192) / 1000
        assert timing._dsv4_workloads["dsv4_embedding_allreduce"]["input_shape"] == (1, 4096)
    if layer == 42 and not dummy:
        assert timing.op_times["dsv4_logits_gather"] == h200._curves[("logits_all_gather", "tp", 8)].predict_us(32320) / 1000
        assert timing._dsv4_workloads["dsv4_logits_gather"]["selected_logits_rows"] == 1
    assert timing._dsv4_workloads["moe_gating_routing_topk"]["topk_ids_dtype"] == "int64"


def test_collective_implementation_is_typed():
    with pytest.raises(ValueError, match="requires a CollectiveImplementation"):
        replace(get_comm_operator("dsv4_logits_gather"), collective_implementation="last_dim")
