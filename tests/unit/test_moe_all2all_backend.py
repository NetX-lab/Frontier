"""vLLM allgather_reducescatter pricing of Step5 MoE dispatch, combine and TP reductions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.attention.step5 import describe_step5_layer
from frontier.config.model_config import BaseModelConfig
from frontier.config.replica_config import ReplicaConfig
from frontier.execution_time_predictor.moe_operator_times import MoeOperatorTimes
from frontier.profiling.collectives.vllm_forward_collectives import bytes_per_row, step5_dispatch_row
from frontier.config.quantization_manager import QuantizationManager
from frontier.scheduler.utils.ep_wave_inputs import DPForwardShape, dp_forward_shape
from frontier.scheduler.utils.expert_parallel import predict_ep_wave_phase_times
from frontier.types import ClusterType

STEP5 = "stepfun-ai/Step-5-Preview"


def test_dispatch_and_combine_row_bytes_match_native_layouts():
    assert bytes_per_row(*step5_dispatch_row(4096, 8, "bfloat16", 128)) == 4288
    assert bytes_per_row(*step5_dispatch_row(4096, 8, "bfloat16", None)) == 8256


@pytest.mark.parametrize("kwargs,match", [
    (dict(moe_all2all_backend="deepep_low_latency"), "must be None or 'allgather_reducescatter'"),
    (dict(moe_all2all_backend="allgather_reducescatter", cluster_prefix="decode_ffn"), "PD-AF"),
])
def test_config_rejects_unmodeled_backends_and_roles(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ReplicaConfig(model_name=STEP5, **kwargs)


def _wave(lane_tokens, cuda_graph=False):
    return SimpleNamespace(lane_compute_tokens=lane_tokens, cuda_graph=cuda_graph)


def test_dp_forward_shape_orders_ranks_and_requires_every_lane():
    assert dp_forward_shape(_wave({1: 3, 0: 7}, True), 2) == DPForwardShape((7, 3), True)
    with pytest.raises(ValueError, match="one source batch per attention-DP lane 0..1"):
        dp_forward_shape(_wave({0: 7}), 2)


def test_lanes_run_pre_routing_work_on_their_dp_rank_tokens():
    shape = DPForwardShape((5, 2), False)
    load = SimpleNamespace(local_token_counts=(2, 0), routing_source=None)
    lanes = {ep_id: SimpleNamespace(ep_id=ep_id, lane_workload=load) for ep_id in range(4)}
    predictor = Mock()
    predictor._replica_config = SimpleNamespace(attn_tensor_parallel_size=2)
    predictor.predict_stage_execution_time.return_value = None
    predict_ep_wave_phase_times(
        layer_workload=SimpleNamespace(participant_ep_ids=(0, 1, 2, 3)), source_batch=object(),
        stage_id=0, layer_id=3, cluster_type=ClusterType.PREFILL, predictor=predictor,
        lane_builder=lambda **kwargs: lanes[kwargs["ep_id"]],
        phase_getter=lambda *args, **kwargs: (0.0,) * 5, workload_logger=Mock(),
        trace_identity={}, batch_id=1, dp_forward_shape=shape,
    )
    assert [lanes[e].moe_pre_routing_effective_total_tokens for e in range(4)] == [5, 5, 2, 2]
    assert all(lanes[e].dp_forward_shape is shape for e in range(4))
    # Equal expert loads are still priced once per DP rank, whose tokens differ.
    assert [call.args[0].ep_id for call in predictor.predict_stage_execution_time.call_args_list] == [0, 2]


def _pricer(layer_id, attn_dp=2):
    model = BaseModelConfig.create_from_name(STEP5)
    backend = Mock()
    backend.predict_dp_rows.side_effect = lambda op, rows, row_bytes, graph, domain: (op, rows, row_bytes, graph, domain)
    pricer = SimpleNamespace(
        _moe_ep_size=16, _model_config=model, _cc_backend=backend,
        _replica_config=SimpleNamespace(moe_all2all_backend="allgather_reducescatter", attn_dp=attn_dp,
                                        router_topk=8),
        _step5_layer_metadata=describe_step5_layer(
            model, layer_id, attention_tp_size=8, moe_tp_size=1, expert_parallel_size=16),
    )
    pricer._predict_dp_dispatch_combine_times = (
        lambda batch: MoeOperatorTimes._predict_dp_dispatch_combine_times(pricer, batch))
    return pricer


@pytest.mark.parametrize("layer_id,dispatch_row_bytes", [(3, 4288), (88, 8256)])
def test_dispatch_and_combine_price_dp_rows_with_native_payloads(layer_id, dispatch_row_bytes):
    pricer = _pricer(layer_id)
    batch = SimpleNamespace(dp_forward_shape=DPForwardShape((6, 1), False))
    times = MoeOperatorTimes._predict_expert_parallel_phase_operator_times(pricer, batch)
    assert times == {
        "expert_parallel_alltoall_dispatch": ("allgather", (6, 1), dispatch_row_bytes, False, "ATTN_DP"),
        "expert_parallel_alltoall_combine": ("reduce_scatter", (6, 1), 8192, False, "ATTN_DP"),
    }


def test_one_dp_rank_exchanges_nothing_and_missing_shape_fails():
    zeros = {"expert_parallel_alltoall_dispatch": 0.0, "expert_parallel_alltoall_combine": 0.0}
    assert MoeOperatorTimes._predict_expert_parallel_phase_operator_times(
        _pricer(3, attn_dp=1), SimpleNamespace()) == zeros
    with pytest.raises(ValueError, match="DP forward shape"):
        MoeOperatorTimes._predict_expert_parallel_phase_operator_times(_pricer(3), SimpleNamespace())


def test_step5_tp_reduction_payloads_stay_bf16():
    QuantizationManager.reset()
    manager = QuantizationManager()
    try:
        manager.configure_from_model_config(BaseModelConfig.create_from_name(STEP5))
        for cluster_type in (ClusterType.PREFILL, ClusterType.DECODE):
            assert manager.get_tensor_size_multiplier("allreduce", cluster_type) == 1.0
    finally:
        QuantizationManager.reset()
