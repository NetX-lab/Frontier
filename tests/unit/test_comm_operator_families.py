from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from frontier.entities.time_components import (
    AttentionOperatorTimes,
    AttentionTime,
    CommunicationOperatorTimes,
    CommunicationTime,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.config.model_config import BaseModelConfig
from frontier.entities import Batch, EPBatchGroup, Request
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.metrics.op_trace_utils import map_trace_op_to_precision_op
from frontier.operators.families import COMM_FAMILY, get_comm_operator
from frontier.operators.spec import (
    CommOperatorSpec,
    CommPayloadContext,
    ResourceClass,
    TraceKind,
    ZeroPayloadPolicy,
)
from frontier.types import ClusterType
from tests.unit.predictor_cache_fixtures import cache_model, predictor_fixture_config


class _ConcreteSklearnExecutionTimePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise NotImplementedError


class _ConcreteSklearnMoEExecutionTimePredictor(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise NotImplementedError


class _Batch:
    id = 7
    size = 2
    num_tokens = 5
    num_decode_tokens = 0
    num_prefill_tokens = 5
    requests = []

    def get_effective_total_tokens_rounded(self, _cluster_type: ClusterType) -> int:
        return 5


class _QuantizationManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, ClusterType]] = []

    def adjust_tensor_size(
        self,
        collective: str,
        data_size_bytes: int,
        cluster_type: ClusterType,
    ) -> int:
        self.calls.append((collective, data_size_bytes, cluster_type))
        return data_size_bytes + 11


class _SpyCCBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict_allreduce(
        self,
        *,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: str | None = None,
    ) -> float:
        self.calls.append(
            {
                "collective_alias": "allreduce",
                "data_size_bytes": data_size_bytes,
                "num_devices": num_devices,
                "cluster_type": cluster_type,
                "comm_domain": comm_domain,
            }
        )
        return float(data_size_bytes) / 1000.0 + float(num_devices)

    def predict_send_recv(
        self,
        *,
        data_size_bytes: int,
        cluster_type: ClusterType,
        comm_domain: str | None = None,
    ) -> float:
        self.calls.append(
            {
                "collective_alias": "send_recv",
                "data_size_bytes": data_size_bytes,
                "num_devices": 2,
                "cluster_type": cluster_type,
                "comm_domain": comm_domain,
            }
        )
        return float(data_size_bytes) / 2000.0

    def predict_allgather(
        self,
        *,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: str | None = None,
    ) -> float:
        self.calls.append(
            {
                "collective_alias": "allgather",
                "data_size_bytes": data_size_bytes,
                "num_devices": num_devices,
                "cluster_type": cluster_type,
                "comm_domain": comm_domain,
            }
        )
        return float(data_size_bytes) / 3000.0 + float(num_devices)

    def predict_all_to_all(
        self,
        *,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: str | None = None,
    ) -> float:
        self.calls.append(
            {
                "collective_alias": "alltoall",
                "data_size_bytes": data_size_bytes,
                "num_devices": num_devices,
                "cluster_type": cluster_type,
                "comm_domain": comm_domain,
            }
        )
        return float(data_size_bytes) / 4000.0 + float(num_devices)

    def predict_reduce_scatter(
        self,
        *,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: ClusterType,
        comm_domain: str | None = None,
    ) -> float:
        self.calls.append(
            {
                "collective_alias": "reduce_scatter",
                "data_size_bytes": data_size_bytes,
                "num_devices": num_devices,
                "cluster_type": cluster_type,
                "comm_domain": comm_domain,
            }
        )
        return float(data_size_bytes) / 5000.0 + float(num_devices)


def _comm_context(
    *,
    batch: object | None = None,
    quantization_manager: object | None = None,
    lane_workload: object | None = None,
) -> CommPayloadContext:
    return CommPayloadContext(
        batch=_Batch() if batch is None else batch,
        model_config=SimpleNamespace(embedding_dim=8, num_experts_per_tok=2),
        replica_config=SimpleNamespace(
            attn_tensor_parallel_size=4,
            moe_tensor_parallel_size=3,
            moe_expert_parallel_size=2,
            num_pipeline_stages=2,
            router_topk=2,
        ),
        cluster_type=ClusterType.MONOLITHIC,
        quantization_manager=quantization_manager or _QuantizationManager(),
        lane_workload=lane_workload,
    )


def _predictor_inputs(*, is_moe: bool):
    model = replace(
        cache_model(), num_layers=4, embedding_dim=8, mlp_hidden_dim=16,
        is_moe=is_moe, num_experts=4, num_experts_per_tok=2,
    )
    inputs = predictor_fixture_config(model_config=model, total_experts=4, ep_size=2)
    replica = inputs["replica_config"]
    replica.attn_tensor_parallel_size = 4
    replica.moe_tensor_parallel_size = 2 if is_moe else 3
    replica.num_pipeline_stages = 2
    return inputs


def _predictor() -> _ConcreteSklearnExecutionTimePredictor:
    inputs = _predictor_inputs(is_moe=False)
    inputs.pop("actual_replica_ids")
    predictor = _ConcreteSklearnExecutionTimePredictor(**inputs)
    predictor._enable_dummy_mode = False
    predictor._dummy_execution_time = 0.0
    predictor._cc_backend = _SpyCCBackend()
    predictor._supports_operation = lambda _operation: True
    predictor._should_strip_collective_sim_allreduce_launch_overhead = lambda _batch: False
    return predictor


def _moe_predictor() -> _ConcreteSklearnMoEExecutionTimePredictor:
    predictor = _ConcreteSklearnMoEExecutionTimePredictor(**_predictor_inputs(is_moe=True))
    predictor._enable_dummy_mode = False
    predictor._dummy_execution_time = 0.0
    predictor._cc_backend = _SpyCCBackend()
    predictor._supports_operation = lambda _operation: True
    predictor._should_strip_collective_sim_allreduce_launch_overhead = lambda _batch: False
    return predictor


def _lane_stage_batch() -> tuple[_Batch, EPLaneWorkload]:
    """Build a physical lane entity with source-width lookup metadata."""

    lane_workload = _lane_workload({0: 5, 1: 5})
    batch = _Batch()
    batch.lane_workload = lane_workload
    batch.num_tokens = [lane_workload.routed_token_count]
    batch.total_num_tokens = lane_workload.routed_token_count
    batch.num_prefill_tokens = lane_workload.routed_token_count
    batch.num_decode_tokens = 0
    return batch, lane_workload


def test_comm_family_declares_first_class_collective_specs() -> None:
    assert COMM_FAMILY.family_id == "comm"
    assert COMM_FAMILY.resource_class is ResourceClass.COMM

    comm_ops = {operator.name: operator for operator in COMM_FAMILY.operators}
    assert {
        "attn_tensor_parallel_allreduce",
        "mlp_tensor_parallel_allreduce",
        "moe_tensor_parallel_allreduce",
        "moe_tensor_parallel_allgather",
        "share_expert_tensor_parallel_allreduce",
        "expert_parallel_allreduce",
        "expert_parallel_alltoall",
        "expert_parallel_alltoall_dispatch",
        "expert_parallel_alltoall_combine",
        "pipeline_parallel_send_recv",
    }.issubset(comm_ops)

    attn_allreduce = comm_ops["attn_tensor_parallel_allreduce"]
    moe_allreduce = comm_ops["moe_tensor_parallel_allreduce"]
    ep_alltoall = comm_ops["expert_parallel_alltoall"]
    pp_send_recv = comm_ops["pipeline_parallel_send_recv"]

    assert isinstance(attn_allreduce, CommOperatorSpec)
    assert attn_allreduce.collective_alias == "allreduce"
    assert attn_allreduce.comm_group == "attn_tp"
    assert attn_allreduce.comm_domain == "ATTN_TP"
    assert attn_allreduce.trace_kind is TraceKind.COMM
    assert attn_allreduce.resource_class is ResourceClass.COMM
    assert attn_allreduce.execution_time_attr == "attn_tensor_parallel_allreduce_time"

    assert moe_allreduce.zero_payload_policy is ZeroPayloadPolicy.EXACT_NOOP
    assert ep_alltoall.zero_payload_policy is ZeroPayloadPolicy.PREDICT

    assert isinstance(pp_send_recv, CommOperatorSpec)
    assert pp_send_recv.collective_alias == "send_recv"
    assert pp_send_recv.comm_group == "pp"
    assert pp_send_recv.comm_domain == "PP"
    assert pp_send_recv.execution_time_attr == "pipeline_parallel_send_recv_time"


def test_trace_precision_mapping_uses_comm_family_specs() -> None:
    source_names = set(map_trace_op_to_precision_op.__code__.co_names)
    assert "COMM_FAMILY" in source_names
    for operator in COMM_FAMILY.e2e_trace_ops():
        assert map_trace_op_to_precision_op(operator.name) == operator.precision_name()
        assert operator.name not in map_trace_op_to_precision_op.__code__.co_consts


def test_comm_payload_builder_preserves_legacy_quantized_hidden_state_bytes() -> None:
    quantization_manager = _QuantizationManager()
    ctx = _comm_context(quantization_manager=quantization_manager)

    attn_allreduce = get_comm_operator("attn_tensor_parallel_allreduce")
    payload = attn_allreduce.build_payload_bytes(ctx)

    # Legacy formula: embedding_dim * fp16_bytes * effective_tokens.
    assert payload == (8 * 2 * 5) + 11
    assert quantization_manager.calls == [
        ("allreduce", 80, ClusterType.MONOLITHIC),
    ]


_DSV4_SP_SCOPES = (
    ("dsv4_sp_attention_allgather", "allgather", (8192, 24576)),
    ("dsv4_sp_attention_reduce_scatter", "reduce_scatter", (65536, 196608)),
    ("dsv4_sp_final_mhc_allgather", "allgather", (32768, 98304)),
)


def _dsv4_sp_context(tokens: int = 17) -> CommPayloadContext:
    return CommPayloadContext(
        batch=Batch(
            replica_id=0, requests=[Request(0.0, tokens, 0)],
            num_tokens=[tokens], is_moe=True,
        ),
        model_config=BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash"),
        replica_config=SimpleNamespace(
            num_pipeline_stages=1, attn_tensor_parallel_size=8, attn_dp=2,
            moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
        ),
        cluster_type=ClusterType.MONOLITHIC,
        quantization_manager=_QuantizationManager(),
    )


@pytest.mark.parametrize("name,alias,payloads", _DSV4_SP_SCOPES)
@pytest.mark.parametrize("tokens,index", ((1, 0), (17, 1)))
def test_dsv4_sp_source_collectives_reach_cc_with_exact_uneven_payloads(
    name: str, alias: str, payloads: tuple[int, int], tokens: int, index: int,
) -> None:
    ctx = _dsv4_sp_context(tokens)
    predictor = _predictor()
    operator = get_comm_operator(name)

    result = predictor._predict_comm_operator_with_context(operator, ctx)

    expected_bytes = payloads[index]
    divisor = 5000 if alias == "reduce_scatter" else 3000
    assert result == pytest.approx(expected_bytes / divisor + 8)
    assert predictor._cc_backend.calls == [{
        "collective_alias": alias, "data_size_bytes": expected_bytes,
        "num_devices": 8, "cluster_type": ClusterType.MONOLITHIC,
        "comm_domain": "ATTN_TP",
    }]
    # FP8/FP4 weight settings and the legacy quantization-size adjustment
    # cannot change source BF16 activation bytes or the source DP batch rows.
    assert ctx.model_config.quantization_config.quant_method == "fp8"
    assert ctx.model_config.expert_dtype == "fp4"
    assert ctx.quantization_manager.calls == []
    assert ctx.batch.total_num_tokens == tokens
    assert ctx.batch.get_effective_total_tokens_rounded(ClusterType.MONOLITHIC) == tokens


def test_dsv4_sp_structured_timings_preserve_measured_and_profiling_guards() -> None:
    from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY

    for name, _, _ in _DSV4_SP_SCOPES:
        operator = get_comm_operator(name)
        assert not operator.predictor_target
        assert not operator.profiling_target
        assert not operator.e2e_trace_target
        assert operator.execution_time_attr is None
        assert operator.comm_group == "attn_tp"
        timings = CommunicationOperatorTimes({name: 1.0})
        assert timings.get_required_time(name) == timings.total_time() == 1.0
        assert timings.legacy_covered_time(CommunicationTime()) == 0.0
        with pytest.raises(ValueError, match="Negative communication"):
            CommunicationOperatorTimes({name: -1.0})
    with pytest.raises(ValueError, match="Unsupported"):
        CommunicationOperatorTimes({"unregistered_dsv4_collective": 1.0})
    DEEPSEEK_V4_ATTENTION_FAMILY.require_enabled_for_execution(structural=True)
    with pytest.raises(NotImplementedError, match="DeepSeek-V4"):
        DEEPSEEK_V4_ATTENTION_FAMILY.require_enabled_for_execution()
    assert DEEPSEEK_V4_ATTENTION_FAMILY.execution_enabled is False
    assert DEEPSEEK_V4_ATTENTION_FAMILY.structural_execution_enabled is True


def test_dsv4_sp_uses_each_dp_source_batch_without_aggregating_rows() -> None:
    predictor = _predictor()
    for tokens in (17, 10):
        predictor._predict_comm_operator_with_context(
            get_comm_operator("dsv4_sp_attention_allgather"), _dsv4_sp_context(tokens),
        )
    assert [call["data_size_bytes"] for call in predictor._cc_backend.calls] == [24576, 16384]


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_pipeline_stages", 2), ("attn_tensor_parallel_size", 1),
        ("attn_dp", 1), ("moe_tensor_parallel_size", 8),
        ("moe_expert_parallel_size", 1), ("moe_expert_parallel_size", 15),
        ("attn_tensor_parallel_size", True), ("attn_dp", 2.5),
    ],
)
def test_dsv4_sp_rejects_unsupported_topologies_before_cc(field: str, value: object) -> None:
    ctx = _dsv4_sp_context()
    setattr(ctx.replica_config, field, value)
    predictor = _predictor()
    with pytest.raises(ValueError, match="DeepSeek-V4 SP source payload requires"):
        predictor._predict_comm_operator_with_context(
            get_comm_operator("dsv4_sp_attention_allgather"), ctx,
        )
    assert predictor._cc_backend.calls == []


@pytest.mark.parametrize("cluster", (
    ClusterType.PREFILL, ClusterType.DECODE,
    ClusterType.DECODE_ATTN, ClusterType.DECODE_FFN,
))
def test_dsv4_sp_source_scope_rejects_disaggregated_roles(cluster: ClusterType) -> None:
    ctx = replace(_dsv4_sp_context(), cluster_type=cluster)
    with pytest.raises(ValueError, match="co-location"):
        get_comm_operator("dsv4_sp_attention_allgather").build_payload_bytes(ctx)


@pytest.mark.parametrize("field,value", (
    ("torch_dtype", "float16"), ("model_architecture_profile", "glm_moe_dsa"),
))
def test_dsv4_sp_source_scope_rejects_other_activation_or_model_contracts(
    field: str, value: str,
) -> None:
    ctx = _dsv4_sp_context()
    ctx = replace(ctx, model_config=replace(ctx.model_config, **{field: value}))
    with pytest.raises(ValueError, match="DeepSeek-V4.*BF16"):
        get_comm_operator("dsv4_sp_attention_allgather").build_payload_bytes(ctx)


def test_dsv4_sp_rejects_explicit_destination_assignments() -> None:
    ctx = replace(_dsv4_sp_context(), lane_workload=_lane_workload({0: 101, 1: 102}))
    with pytest.raises(ValueError, match="source DP attention batch"):
        get_comm_operator("dsv4_sp_attention_allgather").build_payload_bytes(ctx)


@pytest.mark.parametrize("remove_descriptor", (False, True))
def test_dsv4_sp_rejects_ep_batch_group_even_with_aggregate_source_width(
    remove_descriptor: bool,
) -> None:
    batch = _ep_batch_group({0: 101, 1: 102})
    batch.moe_pre_routing_effective_total_tokens = 999
    if remove_descriptor:
        del batch._lane_workload
    ctx = replace(_dsv4_sp_context(), batch=batch)
    with pytest.raises(ValueError, match="source DP attention batch"):
        get_comm_operator("dsv4_sp_attention_allgather").build_payload_bytes(ctx)


@pytest.mark.parametrize("dummy_mode", (False, True))
def test_reduce_scatter_preserves_explicit_dummy_or_missing_backend_error(
    dummy_mode: bool,
) -> None:
    predictor = _predictor()
    predictor._cc_backend = None
    predictor._enable_dummy_mode = dummy_mode
    predictor._dummy_execution_time = 7.25
    operator = get_comm_operator("dsv4_sp_attention_reduce_scatter")
    if dummy_mode:
        assert predictor._predict_comm_operator_with_context(operator, _dsv4_sp_context()) == 7.25
    else:
        with pytest.raises(RuntimeError, match="CC Backend is required for reduce-scatter"):
            predictor._predict_comm_operator_with_context(operator, _dsv4_sp_context())


def test_reduce_scatter_requires_explicit_participant_count() -> None:
    operator = replace(
        get_comm_operator("dsv4_sp_attention_reduce_scatter"), num_devices_builder=None,
    )
    with pytest.raises(ValueError, match="requires num_devices for reduce_scatter"):
        _predictor()._predict_comm_operator_with_context(operator, _dsv4_sp_context())


def _lane_workload(per_expert_tokens: dict[int, int]) -> EPLaneWorkload:
    ep_id = 1 if per_expert_tokens and min(per_expert_tokens) >= 2 else 0
    owned_expert_ids = (ep_id * 2, ep_id * 2 + 1)
    if any(expert_id not in owned_expert_ids for expert_id in per_expert_tokens):
        raise ValueError("test lane map contains an expert outside its owned domain")
    local_token_counts = tuple(
        int(per_expert_tokens.get(expert_id, 0))
        for expert_id in owned_expert_ids
    )
    return EPLaneWorkload(
        ep_id=ep_id,
        moe_expert_parallel_size=2,
        total_expert_num=4,
        owned_expert_ids=owned_expert_ids,
        local_token_counts=local_token_counts,
        global_token_counts=(0, 0) * ep_id + local_token_counts + (0, 0) * (1 - ep_id),
        routed_token_count=sum(local_token_counts),
        router_topk=2,
    )


def _ep_batch_group(per_expert_tokens: dict[int, int]) -> EPBatchGroup:
    lane_workload = _lane_workload(per_expert_tokens)
    return EPBatchGroup(
        requests=[Request(0.0, 5, 0)],
        num_tokens=[5],
        replica_id=0,
        ep_id=lane_workload.ep_id,
        time=0.0,
        source_batch_ids=[1],
        lane_workload=lane_workload,
        cluster_type=ClusterType.MONOLITHIC,
        is_moe=True,
    )


def test_zero_routed_ep_lane_preserves_shared_moe_tp_allreduce_payload() -> None:
    batch = _ep_batch_group({0: 0, 1: 0})
    quantization_manager = SimpleNamespace(
        adjust_tensor_size=lambda _collective, data_size_bytes, _cluster_type: (
            data_size_bytes
        )
    )
    ctx = _comm_context(
        batch=batch,
        quantization_manager=quantization_manager,
    )

    moe_tp_allreduce = get_comm_operator("moe_tensor_parallel_allreduce")
    # MoE TP all-reduce operates on source/pre-routing hidden states.  A
    # zero-routed physical lane remains a participant in this shared domain.
    assert moe_tp_allreduce.build_payload_bytes(ctx) == 8 * 2 * 5


@pytest.mark.parametrize("use_backend", (False, True))
def test_common_moe_tp_live_executor_keeps_zero_routed_lane_in_shared_collective(
    use_backend: bool,
) -> None:
    predictor = _moe_predictor()
    predictor._enable_dummy_mode = not use_backend
    predictor._dummy_execution_time = 5.0
    predictor._cc_backend = _SpyCCBackend() if use_backend else None
    predictor._strip_collective_sim_allreduce_launch_overhead_if_needed = lambda **kwargs: kwargs[
        "predicted_ms"
    ]
    batch = _ep_batch_group({0: 0, 1: 0})

    result = predictor._predict_comm_operator(
        get_comm_operator("moe_tensor_parallel_allreduce"),
        batch,
    )

    assert result == pytest.approx(5.0 if not use_backend else 2.08)
    if use_backend:
        assert predictor._cc_backend.calls == [
            {
                "collective_alias": "allreduce",
                "data_size_bytes": 80,
                "num_devices": 2,
                "cluster_type": ClusterType.MONOLITHIC,
                "comm_domain": "MOE_TP",
            }
        ]


def test_zero_payload_policy_preserves_other_collective_modeling() -> None:
    predictor = _predictor()
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 5.0
    predictor._cc_backend = None
    batch = _ep_batch_group({0: 0, 1: 0})

    result = predictor._predict_comm_operator(
        get_comm_operator("expert_parallel_alltoall"),
        batch,
    )

    assert result == 5.0


def test_explicit_comm_context_controls_registered_operator_execution() -> None:
    predictor = _predictor()
    predictor._model_config = SimpleNamespace(embedding_dim=1, num_experts_per_tok=1)
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=97,
        moe_tensor_parallel_size=99,
        moe_expert_parallel_size=101,
        num_pipeline_stages=1,
        router_topk=1,
    )
    predictor._cluster_type = ClusterType.PREFILL
    context = CommPayloadContext(
        batch=_ep_batch_group({2: 0, 3: 4}),
        model_config=SimpleNamespace(embedding_dim=8, num_experts_per_tok=2),
        replica_config=SimpleNamespace(
            attn_tensor_parallel_size=4,
            moe_tensor_parallel_size=3,
            moe_expert_parallel_size=2,
            num_pipeline_stages=2,
            router_topk=2,
        ),
        cluster_type=ClusterType.DECODE_FFN,
        quantization_manager=SimpleNamespace(
            adjust_tensor_size=lambda _collective, data_size_bytes, _cluster_type: (
                data_size_bytes
            )
        ),
    )

    result = predictor._predict_comm_operator_with_context(
        get_comm_operator("moe_tensor_parallel_allreduce"),
        context,
    )

    assert result == pytest.approx(3.08)
    assert predictor._cc_backend.calls == [
        {
            "collective_alias": "allreduce",
            "data_size_bytes": 8 * 2 * 5,
            "num_devices": 3,
            "cluster_type": ClusterType.DECODE_FFN,
            "comm_domain": "MOE_TP",
        }
    ]


def test_shared_batch_uses_effective_tokens_even_if_it_has_routing_metadata() -> None:
    batch = _Batch()
    batch.per_expert_tokens = {0: 0, 1: 0}
    quantization_manager = SimpleNamespace(
        adjust_tensor_size=lambda _collective, data_size_bytes, _cluster_type: (
            data_size_bytes
        )
    )
    ctx = _comm_context(
        batch=batch,
        quantization_manager=quantization_manager,
    )

    moe_tp_allreduce = get_comm_operator("moe_tensor_parallel_allreduce")
    assert moe_tp_allreduce.build_payload_bytes(ctx) == 8 * 2 * 5


def test_shared_moe_tp_ignores_irrelevant_physical_lane_descriptor() -> None:
    ctx = _comm_context(
        batch=_Batch(),
        quantization_manager=SimpleNamespace(
            adjust_tensor_size=lambda _collective, data_size_bytes, _cluster_type: (
                data_size_bytes
            )
        ),
        lane_workload=SimpleNamespace(lane_workload=object()),
    )

    # Shared MoE TP all-reduce does not consume the physical lane workload.
    assert (
        get_comm_operator("moe_tensor_parallel_allreduce").build_payload_bytes(ctx)
        == 8 * 2 * 5
    )


def test_ep_lane_rejects_malformed_explicit_workload_for_alltoall() -> None:
    ctx = _comm_context(
        batch=_Batch(),
        quantization_manager=SimpleNamespace(
            adjust_tensor_size=lambda _collective, data_size_bytes, _cluster_type: (
                data_size_bytes
            )
        ),
        lane_workload=SimpleNamespace(lane_workload=object()),
    )

    with pytest.raises(TypeError, match="lane_workload must be an EPLaneWorkload"):
        get_comm_operator("expert_parallel_alltoall").build_payload_bytes(ctx)


def test_alltoall_rejects_regular_batch_without_physical_lane() -> None:
    ctx = _comm_context(
        batch=_Batch(),
        quantization_manager=SimpleNamespace(
            adjust_tensor_size=lambda _collective, data_size_bytes, _cluster_type: (
                data_size_bytes
            )
        ),
    )

    with pytest.raises(ValueError, match="EPLaneWorkload"):
        get_comm_operator("expert_parallel_alltoall").build_payload_bytes(ctx)


def test_zero_routed_ep_lane_uses_zero_alltoall_payload() -> None:
    batch = _ep_batch_group({0: 0, 1: 0})
    ctx = _comm_context(
        batch=batch,
        quantization_manager=SimpleNamespace(
            adjust_tensor_size=lambda _collective, data_size_bytes, _cluster_type: (
                data_size_bytes
            )
        ),
    )

    assert get_comm_operator("expert_parallel_alltoall").build_payload_bytes(ctx) == 0


def test_ep_lane_rejects_empty_routed_token_map() -> None:
    ctx = _comm_context(
        batch=_Batch(),
        quantization_manager=SimpleNamespace(
            adjust_tensor_size=lambda _collective, data_size_bytes, _cluster_type: (
                data_size_bytes
            )
        ),
        lane_workload=SimpleNamespace(lane_workload={}),
    )

    with pytest.raises(TypeError, match="lane_workload must be an EPLaneWorkload"):
        get_comm_operator("expert_parallel_alltoall").build_payload_bytes(ctx)


def test_communication_operator_times_reconcile_split_tp_and_pp_legacy_fields() -> None:
    communication_time = CommunicationTime(
        attn_tensor_parallel_allreduce_time=1.5,
        moe_tensor_parallel_allreduce_time=2.5,
        pipeline_parallel_send_recv_time=0.75,
        operator_times=CommunicationOperatorTimes(
            {
                "attn_tensor_parallel_allreduce": 1.5,
                "mlp_tensor_parallel_allreduce": 2.5,
                "pipeline_parallel_send_recv": 0.75,
            }
        ),
    )

    assert communication_time.total_time() == pytest.approx(4.75)


def test_comm_operator_live_path_matches_legacy_dense_tp_and_pp_oracles() -> None:
    batch = _Batch()
    predictor = _predictor()

    legacy_tp = predictor._get_tensor_parallel_communication_time(batch)
    legacy_tp_call = predictor._cc_backend.calls[-1]
    operator_tp = predictor._predict_comm_operator(
        get_comm_operator("attn_tensor_parallel_allreduce"),
        batch,
    )
    operator_tp_call = predictor._cc_backend.calls[-1]

    assert operator_tp == pytest.approx(legacy_tp)
    assert operator_tp_call == legacy_tp_call

    legacy_pp = predictor._get_pipeline_parallel_communication_time(batch)
    legacy_pp_call = predictor._cc_backend.calls[-1]
    operator_pp = predictor._predict_comm_operator(
        get_comm_operator("pipeline_parallel_send_recv"),
        batch,
    )
    operator_pp_call = predictor._cc_backend.calls[-1]

    assert operator_pp == pytest.approx(legacy_pp)
    assert operator_pp_call == legacy_pp_call


def test_comm_operator_live_path_routes_allgather_and_alltoall_wrappers() -> None:
    batch = _Batch()
    predictor = _predictor()
    predictor._replica_config.moe_tensor_parallel_size = 4
    lane_workload = _lane_workload({0: 5, 1: 5})

    operator_allgather = predictor._predict_comm_operator(
        get_comm_operator("moe_tensor_parallel_allgather"),
        batch,
    )
    allgather_call = predictor._cc_backend.calls[-1]

    operator_alltoall = predictor._predict_comm_operator(
        get_comm_operator("expert_parallel_alltoall"),
        batch,
        lane_workload=lane_workload,
    )
    alltoall_call = predictor._cc_backend.calls[-1]

    assert operator_allgather == pytest.approx((20 / 3000.0) + 4)
    assert allgather_call == {
        "collective_alias": "allgather",
        "data_size_bytes": 20,
        "num_devices": 4,
        "cluster_type": ClusterType.MONOLITHIC,
        "comm_domain": "MOE_TP",
    }
    assert operator_alltoall == pytest.approx((160 / 4000.0) + 2)
    assert alltoall_call == {
        "collective_alias": "alltoall",
        "data_size_bytes": 160,
        "num_devices": 2,
        "cluster_type": ClusterType.MONOLITHIC,
        "comm_domain": "EP",
    }


def test_dense_stage_live_path_records_comm_operator_sequence_and_totals() -> None:
    batch = _Batch()
    predictor = _predictor()

    def _legacy_comm_oracle_not_allowed(_batch):
        raise AssertionError("dense stage live path must use CommOperatorSpec")

    predictor._select_measurement_type_for_batch = lambda _batch: object()
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda _measurement_type: None
    predictor._get_pipeline_parallel_communication_time = _legacy_comm_oracle_not_allowed
    predictor._get_tensor_parallel_communication_time = _legacy_comm_oracle_not_allowed
    predictor.predict_attention_layer_time = lambda **_kwargs: AttentionTime()
    predictor._get_mlp_layer_up_proj_execution_time = lambda _batch: 0.0
    predictor._get_mlp_layer_down_proj_execution_time = lambda _batch: 0.0
    predictor._get_mlp_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_add_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_schedule_time = lambda _batch, _stage_id: 0.0
    predictor._get_sampler_e2e_time = lambda _batch, _stage_id: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch, _stage_id: 0.0
    predictor._get_process_model_outputs_time = lambda _batch, _stage_id: 0.0
    predictor._get_ray_comm_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_producer_send_path_runtime_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_receiver_head_runtime_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_prefill_consumer_active_runtime_time = (
        lambda _batch, _stage_id: 0.0
    )
    predictor._get_pp_stage_boundary_handoff_time = lambda _batch, _stage_id: 0.0
    predictor._should_include_spec_decode_proposer_overhead = lambda _batch: False
    predictor._get_mtp_terminal_overshoot_time = lambda *_args, **_kwargs: 0.0

    execution_time = predictor.predict_stage_execution_time(
        batch=batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=2,
    )

    assert predictor._cc_backend.calls == [
        {
            "collective_alias": "send_recv",
            "data_size_bytes": 80,
            "num_devices": 2,
            "cluster_type": ClusterType.MONOLITHIC,
            "comm_domain": "PP",
        },
        {
            "collective_alias": "allreduce",
            "data_size_bytes": 80,
            "num_devices": 4,
            "cluster_type": ClusterType.MONOLITHIC,
            "comm_domain": "ATTN_TP",
        },
    ]
    assert execution_time.communication_time_component.operator_times is not None
    assert execution_time.communication_time_component.operator_times.op_times == {
        "attn_tensor_parallel_allreduce": pytest.approx(8.16),
        "mlp_tensor_parallel_allreduce": pytest.approx(8.16),
        "pipeline_parallel_send_recv": pytest.approx(0.04),
    }
    for layer in execution_time.layer_execution_times:
        assert layer.communication_operator_times.op_times == {
            "attn_tensor_parallel_allreduce": pytest.approx(4.08),
            "mlp_tensor_parallel_allreduce": pytest.approx(4.08),
            "pipeline_parallel_send_recv": pytest.approx(0.04),
        }
        assert layer.communication_time_component.total_time() == pytest.approx(8.2)
    assert execution_time.communication_time_component.total_time() == pytest.approx(16.36)
    assert execution_time.model_time_ms == pytest.approx(16.36)
    assert execution_time.total_time * 1e3 == pytest.approx(16.36)


def test_moe_stage_live_path_records_comm_operator_sequence_and_totals() -> None:
    batch = _Batch()
    predictor = _moe_predictor()
    lane_workload = _lane_workload({0: 5, 1: 5})

    def _legacy_comm_oracle_not_allowed(_batch):
        raise AssertionError("MoE stage live path must use CommOperatorSpec")

    predictor._get_pipeline_parallel_communication_time = _legacy_comm_oracle_not_allowed
    predictor._get_tensor_parallel_communication_time = _legacy_comm_oracle_not_allowed
    predictor._get_moe_tensor_parallel_allreduce_time = _legacy_comm_oracle_not_allowed
    predictor._get_expert_parallel_communication_time = _legacy_comm_oracle_not_allowed
    predictor.predict_attention_layer_time = lambda **_kwargs: AttentionTime()
    predictor._get_gating_linear_time = lambda _batch: 0.0
    predictor._get_gating_routing_topk_time = lambda _batch: 0.0
    predictor._get_moe_shuffling_time = lambda _batch, moe_tokens_input: 0.0
    predictor._get_grouped_gemm_time = lambda _tokens, batch: 0.0
    predictor._apply_moe_grouped_gemm_decode_visibility = (
        lambda raw_time_ms, _batch: raw_time_ms
    )
    predictor._get_add_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_schedule_time = lambda _batch, _stage_id: 0.0
    predictor._get_sampler_e2e_time = lambda _batch, _stage_id: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch, _stage_id: 0.0
    predictor._get_process_model_outputs_time = lambda _batch, _stage_id: 0.0
    predictor._get_ray_comm_time = lambda _batch, _stage_id: 0.0
    predictor.predict_dp_moe_allreduce_times = lambda _batch, _cluster_type: (0.0, 0.0)
    predictor._get_pp_producer_send_path_runtime_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_receiver_head_runtime_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_prefill_consumer_active_runtime_time = (
        lambda _batch, _stage_id: 0.0
    )
    predictor._get_pp_stage_boundary_handoff_time = lambda _batch, _stage_id: 0.0
    predictor._should_include_spec_decode_proposer_overhead = lambda _batch: False
    predictor._get_mtp_terminal_overshoot_time = lambda *_args, **_kwargs: 0.0
    predictor._get_expert_parallel_communication_calibration_scale = lambda _batch: 1.0

    execution_time = predictor._get_execution_time_internal(
        batch=batch,
        pipeline_stage=0,
        moe_tokens_input=lane_workload,
        include_moe=True,
    )

    assert predictor._cc_backend.calls == [
        {
            "collective_alias": "send_recv",
            "data_size_bytes": 80,
            "num_devices": 2,
            "cluster_type": ClusterType.MONOLITHIC,
            "comm_domain": "PP",
        },
        {
            "collective_alias": "allreduce",
            "data_size_bytes": 80,
            "num_devices": 4,
            "cluster_type": ClusterType.MONOLITHIC,
            "comm_domain": "ATTN_TP",
        },
        {
            "collective_alias": "allreduce",
            "data_size_bytes": 80,
            "num_devices": 2,
            "cluster_type": ClusterType.MONOLITHIC,
            "comm_domain": "MOE_TP",
        },
        {
            "collective_alias": "alltoall",
            "data_size_bytes": 160,
            "num_devices": 2,
            "cluster_type": ClusterType.MONOLITHIC,
            "comm_domain": "EP",
        },
        {
            "collective_alias": "alltoall",
            "data_size_bytes": 160,
            "num_devices": 2,
            "cluster_type": ClusterType.MONOLITHIC,
            "comm_domain": "EP",
        },
    ]
    assert execution_time.communication_operator_times is not None
    assert execution_time.communication_operator_times.op_times == {
        "pipeline_parallel_send_recv": pytest.approx(0.04),
        "attn_tensor_parallel_allreduce": pytest.approx(4.08),
        "moe_tensor_parallel_allreduce": pytest.approx(2.08),
        "expert_parallel_alltoall_dispatch": pytest.approx(2.04),
        "expert_parallel_alltoall_combine": pytest.approx(2.04),
    }
    assert execution_time.get_single_layer_moe_dispatch_time() == pytest.approx(2.04)
    assert execution_time.get_single_layer_moe_combine_time() == pytest.approx(2.04)
    assert execution_time.expert_parallel_communication_time == pytest.approx(4.08)
    assert execution_time.communication_time_component.total_time() == pytest.approx(10.28)
    assert execution_time.model_time_ms == pytest.approx(10.28)
    assert execution_time.total_time * 1e3 == pytest.approx(10.28)


def test_moe_stage_num_layers_view_preserves_comm_operator_times() -> None:
    batch, lane_workload = _lane_stage_batch()
    predictor = _moe_predictor()

    predictor._select_measurement_type_for_batch = lambda _batch: object()
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda _measurement_type: None
    predictor._model_config.is_moe_layer = lambda _layer_id: True
    predictor._moe_routing_distribution_type = "balanced"
    predictor._get_moe_tokens_input = lambda _batch, layer_id: lane_workload
    predictor.predict_attention_layer_time = lambda **_kwargs: AttentionTime()
    predictor._get_gating_linear_time = lambda _batch: 0.0
    predictor._get_gating_routing_topk_time = lambda _batch: 0.0
    predictor._get_moe_shuffling_time = lambda _batch, moe_tokens_input: 0.0
    predictor._get_grouped_gemm_time = lambda _tokens, batch: 0.0
    predictor._apply_moe_grouped_gemm_decode_visibility = (
        lambda raw_time_ms, _batch: raw_time_ms
    )
    predictor._get_add_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_schedule_time = lambda _batch, _stage_id: 0.0
    predictor._get_sampler_e2e_time = lambda _batch, _stage_id: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch, _stage_id: 0.0
    predictor._get_process_model_outputs_time = lambda _batch, _stage_id: 0.0
    predictor._get_ray_comm_time = lambda _batch, _stage_id: 0.0
    predictor.predict_dp_moe_allreduce_times = lambda _batch, _cluster_type: (0.0, 0.0)
    predictor._get_pp_producer_send_path_runtime_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_receiver_head_runtime_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_prefill_consumer_active_runtime_time = (
        lambda _batch, _stage_id: 0.0
    )
    predictor._get_pp_stage_boundary_handoff_time = lambda _batch, _stage_id: 0.0
    predictor._should_include_spec_decode_proposer_overhead = lambda _batch: False
    predictor._get_mtp_terminal_overshoot_time = lambda *_args, **_kwargs: 0.0
    predictor._get_expert_parallel_communication_calibration_scale = lambda _batch: 1.0

    execution_time = predictor.predict_stage_execution_time(
        batch=batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=2,
    )

    assert execution_time.communication_operator_times is not None
    assert execution_time.communication_operator_times.op_times == {
        "pipeline_parallel_send_recv": pytest.approx(0.04),
        "attn_tensor_parallel_allreduce": pytest.approx(8.16),
        "moe_tensor_parallel_allreduce": pytest.approx(4.16),
        "expert_parallel_alltoall_dispatch": pytest.approx(4.08),
        "expert_parallel_alltoall_combine": pytest.approx(4.08),
    }
    for index, layer in enumerate(execution_time.layer_execution_times):
        expected = {
            "attn_tensor_parallel_allreduce": pytest.approx(4.08),
            "moe_tensor_parallel_allreduce": pytest.approx(2.08),
            "expert_parallel_alltoall_dispatch": pytest.approx(2.04),
            "expert_parallel_alltoall_combine": pytest.approx(2.04),
        }
        if index == 0:
            expected["pipeline_parallel_send_recv"] = pytest.approx(0.04)
        assert layer.communication_operator_times.op_times == expected
    assert {
        "pipeline_parallel_send_recv",
        "attn_tensor_parallel_allreduce",
        "moe_tensor_parallel_allreduce",
        "expert_parallel_alltoall_dispatch",
        "expert_parallel_alltoall_combine",
    }.issubset(execution_time.op_times)


def test_moe_stage_preserves_attention_operator_times_for_fast_and_view_paths() -> None:
    batch, lane_workload = _lane_stage_batch()
    predictor = _moe_predictor()

    predictor._select_measurement_type_for_batch = lambda _batch: object()
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda _measurement_type: None
    predictor._model_config.is_moe_layer = lambda _layer_id: True
    predictor._moe_routing_distribution_type = "balanced"
    predictor._get_moe_tokens_input = lambda _batch, layer_id: lane_workload
    predictor.predict_attention_layer_time = lambda **_kwargs: AttentionTime(
        attention_prefill_execution_time=0.02,
        operator_times=AttentionOperatorTimes({"attn_prefill": 0.02}),
    )
    predictor._get_gating_linear_time = lambda _batch: 0.0
    predictor._get_gating_routing_topk_time = lambda _batch: 0.0
    predictor._get_moe_shuffling_time = lambda _batch, moe_tokens_input: 0.0
    predictor._get_grouped_gemm_time = lambda _tokens, batch: 0.0
    predictor._apply_moe_grouped_gemm_decode_visibility = (
        lambda raw_time_ms, _batch: raw_time_ms
    )
    predictor._get_add_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_schedule_time = lambda _batch, _stage_id: 0.0
    predictor._get_sampler_e2e_time = lambda _batch, _stage_id: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch, _stage_id: 0.0
    predictor._get_process_model_outputs_time = lambda _batch, _stage_id: 0.0
    predictor._get_ray_comm_time = lambda _batch, _stage_id: 0.0
    predictor.predict_dp_moe_allreduce_times = lambda _batch, _cluster_type: (0.0, 0.0)
    predictor._get_pp_producer_send_path_runtime_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_receiver_head_runtime_time = lambda _batch, _stage_id: 0.0
    predictor._get_pp_prefill_consumer_active_runtime_time = (
        lambda _batch, _stage_id: 0.0
    )
    predictor._get_pp_stage_boundary_handoff_time = lambda _batch, _stage_id: 0.0
    predictor._should_include_spec_decode_proposer_overhead = lambda _batch: False
    predictor._get_mtp_terminal_overshoot_time = lambda *_args, **_kwargs: 0.0
    predictor._get_expert_parallel_communication_calibration_scale = lambda _batch: 1.0

    fast_execution_time = predictor.predict_stage_execution_time(
        batch=batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
    )
    view_execution_time = predictor.predict_stage_execution_time(
        batch=batch,
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=2,
    )

    for execution_time in (fast_execution_time, view_execution_time):
        assert execution_time.attention_operator_times is not None
        assert execution_time.attention_operator_times.op_times == {
            "attn_prefill": pytest.approx(0.02 * execution_time.num_layers),
        }
        for layer in execution_time.layer_execution_times:
            assert layer.attention_operator_times.op_times == {"attn_prefill": pytest.approx(0.02)}
        assert "attn_prefill" in execution_time.op_times
def test_step5_plan_registry_cost_pair_and_phase_accounting():
    from dataclasses import replace
    import json
    from frontier.cc_backend.collective_plan import CollectivePlanCost
    from tests.unit.test_step5_moe_pricing import plan
    from tests.unit.test_stage_finalized_contract import _layer
    dispatch=plan();combine=plan(phase="combine")
    def cost(p):
        return CollectivePlanCost(6.,1.,(2.,)*16,3.,(7,)*16,json.dumps(dict(
            kernel_counts=[7]*16,per_rank_events=[["cpu_fixture"]]*16,
            source_receipt="cpu_fixture",evidence_kind="cpu_fixture")),p.cache_key(),"cpu_fixture")
    plans={"step5_moe_dispatch":dispatch,"step5_moe_combine":combine}
    costs={name:cost(p) for name,p in plans.items()}
    comm=CommunicationOperatorTimes({name:6. for name in plans},plans=plans,costs=costs)
    with pytest.raises(ValueError,match="plan/cost"):
        replace(comm,costs={**costs,"step5_moe_dispatch":cost(combine)})
    layer=_layer(3,is_moe=True,communication_operator_times=comm,op_times={
        "moe_gating_linear":1.,"moe_gating_routing_topk":2.,"moe_input_quantize":3.,
        "moe_grouped_gemm":4.,"moe_tensor_parallel_allreduce":5.,"step5_ffn_merge":2.})
    assert layer.get_single_layer_moe_dispatch_time() == layer.get_single_layer_moe_combine_time() == 6.
    assert layer.moe_phase_operator_times("routed_compute") == (("COMPUTE","moe_grouped_gemm",4.),)
    assert layer.moe_phase_operator_times("post_combine")[0] == ("COMM","moe_tensor_parallel_allreduce",5.)
    assert layer.get_single_layer_block_time() == pytest.approx(29.)
    layer.add_kernel_gap(.1,0,{"moe_gating_linear":1,"moe_gating_routing_topk":1,
        "moe_input_quantize":1,"moe_grouped_gemm":1,"step5_ffn_merge":1})
    assert layer.kernel_gap_time == pytest.approx(.6)
