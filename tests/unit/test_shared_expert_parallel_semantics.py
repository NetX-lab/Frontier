from __future__ import annotations

from types import SimpleNamespace

from frontier.config.model_config import BaseModelConfig
from frontier.config.parallel_semantics import (
    resolve_shared_expert_tensor_parallel_size,
)
from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
from frontier.operators.families import get_comm_operator
from frontier.operators.spec import CommPayloadContext
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.types import ClusterType


class _ConcreteExecutionTimePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _ConcreteMoEExecutionTimePredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _ConcreteDisaggregationExecutionTimePredictor(
    SklearnDisaggregationExecutionTimePredictor
):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def _step3_config() -> BaseModelConfig:
    return BaseModelConfig.create_from_name("step-moe-noquant")


def _replica(
    model_config: BaseModelConfig,
    *,
    attn_tp: int,
    moe_tp: int,
    moe_ep: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=model_config,
        attn_tensor_parallel_size=attn_tp,
        attn_dp=1,
        moe_tensor_parallel_size=moe_tp,
        moe_expert_parallel_size=moe_ep,
        num_pipeline_stages=1,
        router_topk=2,
    )


def test_shared_expert_profile_uses_attention_tp_domain() -> None:
    config = _step3_config()
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[1],
        moe_tp=[1],
        is_moe=True,
    )

    assert {
        "share_expert_up_proj",
        "share_expert_act",
        "share_expert_down_proj",
    }.issubset(plan["enabled_ops"])
    shared_contract = plan["typed_operator_contracts"]["share_expert_up_proj"]
    assert shared_contract["tensor_parallel_mode"] == "attention_tp"
    assert shared_contract["selected_tensor_parallel_size"] == 8


def test_manager_shared_expert_key_matches_profile_attention_tp() -> None:
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    config = _step3_config()
    replica = _replica(config, attn_tp=8, moe_tp=1, moe_ep=8)

    assert (
        manager._get_linear_op_tp_key(
            "share_expert_up_proj",
            ClusterType.PREFILL,
            replica,
            True,
        )
        == 8
    )


def test_shared_expert_trace_shape_uses_attention_tp() -> None:
    config = _step3_config()
    context = OpTraceContext(
        cluster_type=ClusterType.MONOLITHIC,
        model_config=config,
        replica_config=_replica(config, attn_tp=8, moe_tp=1, moe_ep=8),
        total_tokens=4,
        effective_tokens_compute=4,
        effective_tokens_transfer=4,
        effective_tokens_rounded=4,
        tokens_are_post_routing=False,
    )

    metadata = compute_op_trace_meta("share_expert_up_proj", "COMPUTE", context)
    assert metadata["tensor_shape"]["output"] == [
        4,
        config.share_expert_dim // 8,
    ]


def test_shared_expert_allreduce_uses_attention_tp_group() -> None:
    config = _step3_config()
    operator = get_comm_operator("share_expert_tensor_parallel_allreduce")
    context = CommPayloadContext(
        batch=None,
        model_config=config,
        replica_config=_replica(config, attn_tp=8, moe_tp=1, moe_ep=8),
        cluster_type=ClusterType.MONOLITHIC,
        quantization_manager=None,
    )

    assert operator.comm_group == "attn_tp"
    assert operator.comm_domain == "ATTN_TP"
    assert operator.num_devices(context) == 8
    assert operator.resolve_comm_group(context) == "attn_tp"
    assert operator.resolve_comm_domain(context) == "ATTN_TP"


def test_decode_ffn_maps_shared_expert_to_role_local_tp() -> None:
    config = _step3_config()
    replica = _replica(config, attn_tp=0, moe_tp=2, moe_ep=4)

    assert (
        resolve_shared_expert_tensor_parallel_size(
            cluster_type=ClusterType.DECODE_FFN,
            replica_config=replica,
        )
        == 2
    )
    operator = get_comm_operator("share_expert_tensor_parallel_allreduce")
    context = CommPayloadContext(
        batch=None,
        model_config=config,
        replica_config=replica,
        cluster_type=ClusterType.DECODE_FFN,
        quantization_manager=None,
    )
    assert operator.num_devices(context) == 2
    assert operator.resolve_comm_group(context) == "moe_tp"
    assert operator.resolve_comm_domain(context) == "MOE_TP"


def test_shared_expert_resolver_preserves_profile_ffn_tp_domain() -> None:
    config = BaseModelConfig.create_from_name("Step2Mini-tiny")
    replica = _replica(config, attn_tp=8, moe_tp=2, moe_ep=4)

    selected_tp = resolve_shared_expert_tensor_parallel_size(
        cluster_type=ClusterType.PREFILL,
        replica_config=replica,
    )

    assert selected_tp == 2


def test_comm_prediction_uses_resolved_shared_expert_domain() -> None:
    config = _step3_config()
    replica = _replica(config, attn_tp=0, moe_tp=2, moe_ep=4)
    captured: dict[str, object] = {}
    predictor = _ConcreteExecutionTimePredictor.__new__(
        _ConcreteExecutionTimePredictor
    )
    predictor.predict_allreduce_time = lambda **kwargs: captured.update(kwargs) or 1.5
    predictor._strip_collective_sim_allreduce_launch_overhead_if_needed = (
        lambda **kwargs: float(kwargs["predicted_ms"])
    )
    context = CommPayloadContext(
        batch=SimpleNamespace(
            total_num_tokens=4,
            get_effective_total_tokens_rounded=lambda _cluster_type: 4,
        ),
        model_config=config,
        replica_config=replica,
        cluster_type=ClusterType.DECODE_FFN,
        quantization_manager=SimpleNamespace(
            adjust_tensor_size=lambda _op, size, _cluster_type: size
        ),
    )

    predicted = predictor._predict_comm_operator_with_context(
        get_comm_operator("share_expert_tensor_parallel_allreduce"),
        context,
    )

    assert predicted == 1.5
    assert captured["num_devices"] == 2
    assert captured["comm_domain"] == "MOE_TP"


def test_monolithic_dummy_shared_expert_allreduce_uses_attention_tp() -> None:
    config = _step3_config()
    predictor = _ConcreteMoEExecutionTimePredictor.__new__(
        _ConcreteMoEExecutionTimePredictor
    )
    predictor._dummy_execution_time = 3.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._model_config = config
    predictor._replica_config = _replica(config, attn_tp=8, moe_tp=1, moe_ep=8)
    predictor._get_ep_lane_routed_token_count = lambda *_args, **_kwargs: 1

    execution_time = predictor._get_dummy_execution_time(
        SimpleNamespace(),
        pipeline_stage=0,
    )

    assert execution_time.share_expert_tensor_parallel_allreduce_time == 3.0


def test_pdd_prefill_dummy_shared_expert_allreduce_uses_attention_tp() -> None:
    config = _step3_config()
    replica = _replica(config, attn_tp=8, moe_tp=1, moe_ep=8)
    predictor = _ConcreteDisaggregationExecutionTimePredictor.__new__(
        _ConcreteDisaggregationExecutionTimePredictor
    )
    predictor._dummy_execution_time = 3.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._get_ep_lane_routed_token_count = lambda *_args, **_kwargs: 1
    predictor._get_cluster_replica_config = lambda _cluster_type: replica
    predictor._get_cluster_model_architecture_profile = (
        lambda _cluster_type: config.get_model_architecture_profile()
    )

    execution_time = predictor._get_dummy_execution_time_for_cluster(
        SimpleNamespace(),
        pipeline_stage=0,
        cluster_type=ClusterType.PREFILL,
    )

    assert execution_time.share_expert_tensor_parallel_allreduce_time == 3.0
