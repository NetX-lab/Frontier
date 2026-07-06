from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.entities.time_components import CommunicationOperatorTimes, CommunicationTime
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.operators.families import COMM_FAMILY, get_comm_operator
from frontier.operators.spec import CommOperatorSpec, CommPayloadContext, ResourceClass, TraceKind
from frontier.types import ClusterType


class _ConcreteSklearnExecutionTimePredictor(SklearnExecutionTimePredictor):
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


def _comm_context(*, quantization_manager: object | None = None) -> CommPayloadContext:
    return CommPayloadContext(
        batch=_Batch(),
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
    )


def _predictor() -> _ConcreteSklearnExecutionTimePredictor:
    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._model_config = SimpleNamespace(embedding_dim=8, num_experts_per_tok=2)
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=4,
        moe_tensor_parallel_size=3,
        moe_expert_parallel_size=2,
        num_pipeline_stages=2,
        router_topk=2,
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._enable_dummy_mode = False
    predictor._dummy_execution_time = 0.0
    predictor._cc_backend = _SpyCCBackend()
    predictor._supports_operation = lambda _operation: True
    predictor._should_strip_collective_sim_allreduce_launch_overhead = lambda _batch: False
    return predictor


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
    pp_send_recv = comm_ops["pipeline_parallel_send_recv"]

    assert isinstance(attn_allreduce, CommOperatorSpec)
    assert attn_allreduce.collective_alias == "allreduce"
    assert attn_allreduce.comm_group == "attn_tp"
    assert attn_allreduce.comm_domain == "ATTN_TP"
    assert attn_allreduce.trace_kind is TraceKind.COMM
    assert attn_allreduce.resource_class is ResourceClass.COMM
    assert attn_allreduce.execution_time_attr == "attn_tensor_parallel_allreduce_time"

    assert isinstance(pp_send_recv, CommOperatorSpec)
    assert pp_send_recv.collective_alias == "send_recv"
    assert pp_send_recv.comm_group == "pp"
    assert pp_send_recv.comm_domain == "PP"
    assert pp_send_recv.execution_time_attr == "pipeline_parallel_send_recv_time"


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

from __future__ import annotations

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
from frontier.metrics.op_trace_utils import map_trace_op_to_precision_op
from frontier.model_architectures import ModelArchitectureProfile
from frontier.operators.families import COMM_FAMILY, get_comm_operator
from frontier.operators.spec import CommOperatorSpec, CommPayloadContext, ResourceClass, TraceKind
from frontier.types import ClusterType


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


def _comm_context(*, quantization_manager: object | None = None) -> CommPayloadContext:
    return CommPayloadContext(
        batch=_Batch(),
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
    )


def _predictor() -> _ConcreteSklearnExecutionTimePredictor:
    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._model_config = SimpleNamespace(embedding_dim=8, num_experts_per_tok=2)
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=4,
        moe_tensor_parallel_size=3,
        moe_expert_parallel_size=2,
        num_pipeline_stages=2,
        router_topk=2,
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._enable_dummy_mode = False
    predictor._dummy_execution_time = 0.0
    predictor._cc_backend = _SpyCCBackend()
    predictor._supports_operation = lambda _operation: True
    predictor._should_strip_collective_sim_allreduce_launch_overhead = lambda _batch: False
    return predictor


def _moe_predictor() -> _ConcreteSklearnMoEExecutionTimePredictor:
    predictor = object.__new__(_ConcreteSklearnMoEExecutionTimePredictor)
    predictor._model_config = SimpleNamespace(
        embedding_dim=8,
        num_experts_per_tok=2,
        is_moe=True,
        supports_share_expert=lambda: False,
        get_model_architecture_profile=lambda: ModelArchitectureProfile.generic(),
    )
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=4,
        moe_tensor_parallel_size=2,
        moe_expert_parallel_size=2,
        num_pipeline_stages=2,
        router_topk=2,
        data_parallel_size=1,
    )
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._enable_dummy_mode = False
    predictor._dummy_execution_time = 0.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._moe_ep_size = 2
    predictor._router_topk = 2
    predictor._cc_backend = _SpyCCBackend()
    predictor._supports_operation = lambda _operation: True
    predictor._should_strip_collective_sim_allreduce_launch_overhead = lambda _batch: False
    return predictor


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
    pp_send_recv = comm_ops["pipeline_parallel_send_recv"]

    assert isinstance(attn_allreduce, CommOperatorSpec)
    assert attn_allreduce.collective_alias == "allreduce"
    assert attn_allreduce.comm_group == "attn_tp"
    assert attn_allreduce.comm_domain == "ATTN_TP"
    assert attn_allreduce.trace_kind is TraceKind.COMM
    assert attn_allreduce.resource_class is ResourceClass.COMM
    assert attn_allreduce.execution_time_attr == "attn_tensor_parallel_allreduce_time"

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

    operator_allgather = predictor._predict_comm_operator(
        get_comm_operator("moe_tensor_parallel_allgather"),
        batch,
    )
    allgather_call = predictor._cc_backend.calls[-1]

    operator_alltoall = predictor._predict_comm_operator(
        get_comm_operator("expert_parallel_alltoall"),
        batch,
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
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
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
        "attn_tensor_parallel_allreduce": pytest.approx(4.08),
        "mlp_tensor_parallel_allreduce": pytest.approx(4.08),
        "pipeline_parallel_send_recv": pytest.approx(0.04),
    }
    assert execution_time.communication_time_component.total_time() == pytest.approx(8.2)
    assert execution_time.model_time_ms == pytest.approx(16.36)
    assert execution_time.total_time * 1e3 == pytest.approx(16.36)


def test_moe_stage_live_path_records_comm_operator_sequence_and_totals() -> None:
    batch = _Batch()
    predictor = _moe_predictor()

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
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
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
        moe_tokens_input=5,
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
            "collective_alias": "allreduce",
            "data_size_bytes": 80,
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
        "expert_parallel_allreduce": pytest.approx(2.08),
    }
    assert execution_time.communication_time_component.total_time() == pytest.approx(8.28)
    assert execution_time.model_time_ms == pytest.approx(8.28)
    assert execution_time.total_time * 1e3 == pytest.approx(8.28)


def test_moe_stage_num_layers_view_preserves_comm_operator_times() -> None:
    batch = _Batch()
    predictor = _moe_predictor()

    predictor._select_measurement_type_for_batch = lambda _batch: object()
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda _measurement_type: None
    predictor._model_config.is_moe_layer = lambda _layer_id: True
    predictor._moe_routing_mode = "uniform_legacy"
    predictor._get_moe_tokens_input = lambda _batch, layer_id: 5
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
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
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
        "attn_tensor_parallel_allreduce": pytest.approx(4.08),
        "moe_tensor_parallel_allreduce": pytest.approx(2.08),
        "expert_parallel_allreduce": pytest.approx(2.08),
    }
    assert {
        "pipeline_parallel_send_recv",
        "attn_tensor_parallel_allreduce",
        "moe_tensor_parallel_allreduce",
        "expert_parallel_allreduce",
    }.issubset(execution_time.op_times)


def test_moe_stage_preserves_attention_operator_times_for_fast_and_view_paths() -> None:
    batch = _Batch()
    predictor = _moe_predictor()

    predictor._select_measurement_type_for_batch = lambda _batch: object()
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda _measurement_type: None
    predictor._model_config.is_moe_layer = lambda _layer_id: True
    predictor._moe_routing_mode = "uniform_legacy"
    predictor._get_moe_tokens_input = lambda _batch, layer_id: 5
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
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
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
            "attn_prefill": pytest.approx(0.02),
        }
        assert "attn_prefill" in execution_time.op_times
