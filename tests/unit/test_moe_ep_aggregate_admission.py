"""Public EP aggregate admission regressions for the typed-lane contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.types import ClusterType


class _ModelConfig:
    is_moe = True

    def __init__(self, *, is_moe_layer: bool = True) -> None:
        self._is_moe_layer = is_moe_layer

    def is_moe_layer(self, _layer_id: int) -> bool:
        return self._is_moe_layer

    def supports_share_expert(self) -> bool:
        return False

    def get_model_architecture_profile(self) -> ModelArchitectureProfile:
        return ModelArchitectureProfile.generic()


class _ModelConfigWithoutLayerPredicate:
    is_moe = True

    def supports_share_expert(self) -> bool:
        return False

    def get_model_architecture_profile(self) -> ModelArchitectureProfile:
        return ModelArchitectureProfile.generic()


class _MonolithicPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _DisaggregationPredictor(SklearnDisaggregationExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def _lane(*, ep_id: int = 0, routed_token_count: int = 4) -> EPLaneWorkload:
    counts = (routed_token_count, 0)
    return EPLaneWorkload(
        ep_id=ep_id,
        moe_expert_parallel_size=2,
        total_expert_num=4,
        owned_expert_ids=(ep_id * 2, ep_id * 2 + 1),
        local_token_counts=counts,
        routed_token_count=routed_token_count,
        router_topk=2,
    )


def _admission_lane(
    *,
    ep_size: int = 2,
    router_topk: int = 2,
    routed_token_count: int | None = None,
) -> EPLaneWorkload:
    """Build a descriptor valid in isolation for boundary-mismatch tests."""
    total_expert_num = 4
    local_width = total_expert_num // ep_size
    if routed_token_count is None:
        routed_token_count = 4 * router_topk
    local_token_counts = (routed_token_count,) + (0,) * (local_width - 1)
    return EPLaneWorkload(
        ep_id=0,
        moe_expert_parallel_size=ep_size,
        total_expert_num=total_expert_num,
        owned_expert_ids=tuple(range(local_width)),
        local_token_counts=local_token_counts,
        routed_token_count=routed_token_count,
        router_topk=router_topk,
    )


def _batch(
    *,
    lane_workload: EPLaneWorkload | None = None,
    total_num_tokens: int = 4,
    effective_tokens: int | None = None,
) -> SimpleNamespace:
    if effective_tokens is None:
        effective_tokens = total_num_tokens
    values: dict[str, object] = {
        "id": 1,
        "size": 1,
        "num_tokens": total_num_tokens,
        "total_num_tokens": total_num_tokens,
        "requests": [],
        "get_effective_total_tokens_rounded": lambda _cluster_type: effective_tokens,
    }
    if lane_workload is not None:
        values["lane_workload"] = lane_workload
    return SimpleNamespace(**values)


def _configure_monolithic(*, dummy: bool, ep_size: int = 2) -> _MonolithicPredictor:
    predictor = _MonolithicPredictor.__new__(_MonolithicPredictor)
    predictor._enable_dummy_mode = dummy
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._moe_ep_size = ep_size
    predictor._router_topk = 2
    predictor._model_config = _ModelConfig()
    predictor._replica_config = SimpleNamespace(
        moe_expert_parallel_size=ep_size,
        total_expert_num=4,
    )
    return predictor


def _configure_disaggregation(
    *, dummy: bool, cluster_type: ClusterType = ClusterType.DECODE_FFN
) -> _DisaggregationPredictor:
    predictor = _DisaggregationPredictor.__new__(_DisaggregationPredictor)
    predictor._enable_dummy_mode = dummy
    predictor._cluster_type = cluster_type
    predictor._moe_ep_size = 2
    predictor._router_topk = 2
    predictor._model_config = _ModelConfig()
    cluster_replica_config = SimpleNamespace(
        model_config=_ModelConfig(),
        moe_expert_parallel_size=2,
        total_expert_num=4,
        moe_tensor_parallel_size=1,
        attn_tensor_parallel_size=1,
        num_pipeline_stages=1,
    )
    predictor._replica_config = cluster_replica_config
    predictor._get_cluster_replica_config = MagicMock(
        return_value=cluster_replica_config
    )
    return predictor


def _install_non_dummy_lookup_spies(predictor: object) -> list[str]:
    calls: list[str] = []
    predictor._select_measurement_type_for_batch = lambda _batch: calls.append(
        "select_measurement"
    ) or None
    predictor._require_predictions_for_measurement_type = (
        lambda _measurement_type, _batch: calls.append("require_measurement")
    )
    predictor._activate_measurement_type = lambda _measurement_type: calls.append(
        "activate_measurement"
    )
    predictor._emit_cuda_graph_activation_records = (
        lambda *_args, **_kwargs: calls.append("cuda_graph_activation")
    )
    predictor._get_communication_time = lambda *_args, **_kwargs: (
        calls.append("communication_lookup")
        or SimpleNamespace(tensor_parallel_time=0.0, pipeline_parallel_time=0.0)
    )
    predictor._get_overhead_time = lambda *_args, **_kwargs: (
        calls.append("overhead_lookup") or SimpleNamespace(
            schedule_time=0.0,
            sampler_e2e_time=0.0,
            prepare_inputs_e2e_time=0.0,
            process_model_outputs_time=0.0,
            ray_comm_time=0.0,
            pp_producer_send_path_runtime_time=0.0,
            pp_receiver_head_runtime_time=0.0,
            pp_prefill_consumer_active_runtime_time=0.0,
            pp_stage_boundary_residual_runtime_time=0.0,
            pp_stage_boundary_handoff_time=0.0,
        )
    )
    predictor._get_cluster_model_architecture_profile = lambda *_args, **_kwargs: (
        calls.append("architecture_lookup") or ModelArchitectureProfile.generic()
    )
    predictor._get_moe_tokens_input = lambda *_args, **_kwargs: (
        calls.append("moe_lookup")
        or (_ for _ in ()).throw(RuntimeError("downstream MoE lookup reached"))
    )
    return calls


@pytest.mark.parametrize("dummy", (True, False))
def test_monolithic_routed_ep2_requires_lane_before_mode_work(dummy: bool) -> None:
    predictor = _configure_monolithic(dummy=dummy)
    predictor._dummy_execution_time = 1.0
    calls = _install_non_dummy_lookup_spies(predictor)
    predictor._get_dummy_execution_time = MagicMock(
        side_effect=lambda *_args, **_kwargs: calls.append("dummy_return")
    )

    with pytest.raises(ValueError, match="EPLaneWorkload"):
        predictor.predict_stage_execution_time(
            batch=_batch(),
            stage_id=0,
            cluster_type=ClusterType.MONOLITHIC,
            num_layers=1,
            layer_id=0,
            include_moe=True,
            include_ffn=True,
        )

    assert calls == []


@pytest.mark.parametrize("dummy", (True, False))
def test_direct_moe_layer_api_requires_lane_before_mode_work(dummy: bool) -> None:
    """The public single-layer MoE API shares the aggregate admission gate."""

    predictor = _configure_monolithic(dummy=dummy)
    predictor._dummy_execution_time = 1.0
    predictor._get_dummy_execution_time = MagicMock(
        side_effect=AssertionError("dummy timing reached before admission")
    )
    predictor._supports_operation = MagicMock(
        side_effect=AssertionError("operation lookup reached before admission")
    )

    with pytest.raises(ValueError, match="EPLaneWorkload"):
        predictor.predict_moe_layer_time(
            _batch(),
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )

    predictor._get_dummy_execution_time.assert_not_called()
    predictor._supports_operation.assert_not_called()


@pytest.mark.parametrize(
    ("mismatch", "expected_message"),
    [
        ("ep_size", "lane_workload EP size does not match predictor topology"),
        ("router_topk", "lane_workload router_topk does not match predictor topology"),
        ("conservation", "Token conservation violated in predict_moe_layer_time"),
    ],
)
@pytest.mark.parametrize("dummy", (True, False))
def test_direct_moe_layer_rejects_malformed_lane_before_mode_work(
    mismatch: str,
    expected_message: str,
    dummy: bool,
) -> None:
    """All lane invariants are admitted before dummy or lookup-specific work."""
    lane_kwargs: dict[str, int] = {}
    if mismatch == "ep_size":
        lane_kwargs["ep_size"] = 1
    elif mismatch == "router_topk":
        lane_kwargs["router_topk"] = 1
    else:
        lane_kwargs["routed_token_count"] = 3
    lane = _admission_lane(**lane_kwargs)
    predictor = _configure_monolithic(dummy=dummy)
    predictor._dummy_execution_time = 1.0
    calls: list[str] = []
    predictor._model_config.supports_share_expert = MagicMock(
        side_effect=lambda: calls.append("share_expert") or False
    )
    predictor._supports_operation = MagicMock(
        side_effect=lambda *_args, **_kwargs: calls.append("supports") or True
    )

    with pytest.raises(ValueError, match=expected_message):
        predictor.predict_moe_layer_time(
            _batch(lane_workload=lane),
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
        )

    assert calls == []
    predictor._model_config.supports_share_expert.assert_not_called()
    predictor._supports_operation.assert_not_called()


@pytest.mark.parametrize("dummy", (True, False))
def test_disaggregation_routed_ep2_requires_lane_before_mode_work(dummy: bool) -> None:
    predictor = _configure_disaggregation(dummy=dummy)
    calls = _install_non_dummy_lookup_spies(predictor)
    predictor._is_zero_token_decode_ffn_ep_barrier = lambda *_args: False
    if not dummy:
        predictor._get_overhead_time = lambda *_args, **_kwargs: SimpleNamespace(
            pp_stage_boundary_handoff_time=0.0
        )
        predictor._get_pp_stage_boundary_handoff_time = lambda *_args: 0.0
        predictor._resolve_layer_lane_workload = lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(ValueError("downstream lane resolver reached"))
        )
    predictor._get_dummy_execution_time_for_cluster = MagicMock(
        return_value=SimpleNamespace(num_layers=1)
    )

    with pytest.raises(ValueError, match="EPLaneWorkload"):
        predictor.predict_stage_execution_time(
            batch=_batch(),
            stage_id=0,
            cluster_type=ClusterType.DECODE_FFN,
            num_layers=1,
            layer_id=0,
            include_moe=True,
            include_ffn=True,
        )

    assert calls == []


def test_monolithic_attention_only_does_not_require_lane() -> None:
    predictor = _configure_monolithic(dummy=True)
    sentinel = object()
    predictor._get_dummy_execution_time = MagicMock(return_value=sentinel)

    result = predictor.predict_stage_execution_time(
        batch=_batch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=0,
        include_ffn=False,
        include_moe=None,
    )

    assert result is sentinel
    predictor._get_dummy_execution_time.assert_called_once()


def test_monolithic_dense_layer_does_not_require_lane() -> None:
    predictor = _configure_monolithic(dummy=True)
    predictor._model_config = _ModelConfig(is_moe_layer=False)
    sentinel = object()
    predictor._get_dummy_execution_time = MagicMock(return_value=sentinel)

    result = predictor.predict_stage_execution_time(
        batch=_batch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=3,
        include_moe=False,
        include_ffn=True,
    )

    assert result is sentinel


def test_monolithic_ep1_routed_aggregate_does_not_require_lane() -> None:
    predictor = _configure_monolithic(dummy=True, ep_size=1)
    sentinel = object()
    predictor._get_dummy_execution_time = MagicMock(return_value=sentinel)

    result = predictor.predict_stage_execution_time(
        batch=_batch(),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        include_moe=True,
        include_ffn=True,
    )

    assert result is sentinel


def test_monolithic_multi_layer_default_aggregate_preserves_legacy_path() -> None:
    """An implicit routed aggregate requires its physical EP lane descriptor."""
    predictor = _configure_monolithic(dummy=True, ep_size=2)
    predictor._get_dummy_execution_time = MagicMock(
        return_value=SimpleNamespace(num_layers=5)
    )

    with pytest.raises(ValueError, match="EPLaneWorkload"):
        predictor.predict_stage_execution_time(
            batch=_batch(),
            stage_id=0,
            cluster_type=ClusterType.MONOLITHIC,
            num_layers=5,
            layer_id=0,
            include_moe=None,
            include_ffn=True,
        )

    predictor._get_dummy_execution_time.assert_not_called()


@pytest.mark.parametrize("dummy", (True, False))
def test_disaggregation_implicit_routed_ep2_requires_lane_before_mode_work(
    dummy: bool,
) -> None:
    """Implicit multi-layer MoE aggregates cross the same EP lane boundary."""
    predictor = _configure_disaggregation(dummy=dummy)
    calls = _install_non_dummy_lookup_spies(predictor)
    predictor._is_zero_token_decode_ffn_ep_barrier = lambda *_args: False
    predictor._get_dummy_execution_time_for_cluster = MagicMock(
        side_effect=lambda *_args, **_kwargs: calls.append("dummy_return")
        or SimpleNamespace(num_layers=5)
    )
    predictor._get_overhead_time = lambda *_args, **_kwargs: SimpleNamespace(
        pp_stage_boundary_handoff_time=0.0
    )
    predictor._get_pp_stage_boundary_handoff_time = lambda *_args: 0.0
    if not dummy:
        predictor._resolve_layer_lane_workload = lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(
                ValueError("downstream EPLaneWorkload resolver reached")
            )
        )

    with pytest.raises(ValueError, match="EPLaneWorkload"):
        predictor.predict_stage_execution_time(
            batch=_batch(),
            stage_id=0,
            cluster_type=ClusterType.DECODE_FFN,
            num_layers=5,
            layer_id=0,
            include_moe=None,
            include_ffn=True,
        )

    assert calls == []


@pytest.mark.parametrize("predictor_kind", ("monolithic", "disaggregation"))
@pytest.mark.parametrize("dummy", (True, False))
def test_concrete_moe_layer_requires_layer_capability_before_mode_work(
    predictor_kind: str,
    dummy: bool,
) -> None:
    """A concrete MoE layer needs the model-owned layer capability predicate."""
    if predictor_kind == "monolithic":
        predictor = _configure_monolithic(dummy=dummy)
        predictor._model_config = _ModelConfigWithoutLayerPredicate()
        predictor._get_dummy_execution_time = MagicMock(
            side_effect=AssertionError("dummy timing reached before capability check")
        )
        predictor._supports_operation = MagicMock(
            side_effect=AssertionError("operation lookup reached before capability check")
        )
    else:
        predictor = _configure_disaggregation(dummy=dummy)
        model_config = _ModelConfigWithoutLayerPredicate()
        predictor._model_config = model_config
        predictor._get_cluster_replica_config.return_value = SimpleNamespace(
            model_config=model_config,
            moe_expert_parallel_size=2,
            total_expert_num=4,
            moe_tensor_parallel_size=1,
            attn_tensor_parallel_size=1,
            num_pipeline_stages=1,
        )
        predictor._get_dummy_execution_time_for_cluster = MagicMock(
            return_value=SimpleNamespace(num_layers=1)
        )
        predictor._supports_operation = MagicMock(
            side_effect=AssertionError("operation lookup reached before capability check")
        )
        if not dummy:
            _install_non_dummy_lookup_spies(predictor)
            predictor._resolve_layer_lane_workload = lambda *_args, **_kwargs: (
                (_ for _ in ()).throw(
                    AssertionError("lane resolver reached before capability check")
                )
            )

    with pytest.raises(ValueError, match="is_moe_layer"):
        predictor.predict_stage_execution_time(
            batch=_batch(lane_workload=_lane()),
            stage_id=0,
            cluster_type=(
                ClusterType.MONOLITHIC
                if predictor_kind == "monolithic"
                else ClusterType.DECODE_FFN
            ),
            num_layers=1,
            layer_id=0,
            include_moe=None,
            include_ffn=True,
        )


def test_disaggregation_dense_decode_ffn_keeps_ffn_tp_without_routed_fields() -> None:
    """Dense FFN work keeps its role TP collective while remaining non-routed."""
    model_config = _ModelConfig(is_moe_layer=False)
    predictor = _configure_disaggregation(dummy=True)
    predictor._dummy_execution_time = 10.0
    predictor._num_layers_per_pipeline_stage = 1
    predictor._get_cluster_replica_config.return_value = SimpleNamespace(
        model_config=model_config,
        moe_expert_parallel_size=1,
        total_expert_num=4,
        moe_tensor_parallel_size=2,
        attn_tensor_parallel_size=1,
        num_pipeline_stages=1,
    )
    predictor._get_cluster_model_architecture_profile = lambda _cluster_type: (
        ModelArchitectureProfile.step3_text()
    )

    execution_time = predictor.predict_stage_execution_time(
        batch=_batch(),
        stage_id=0,
        cluster_type=ClusterType.DECODE_FFN,
        num_layers=1,
        layer_id=0,
        include_moe=False,
        include_ffn=True,
    )

    assert execution_time._is_moe is False
    assert execution_time._moe_gating_time == pytest.approx(0.0)
    assert execution_time._moe_shuffling_time == pytest.approx(0.0)
    assert execution_time._moe_grouped_gemm_time == pytest.approx(0.0)
    assert execution_time._expert_parallel_communication_time == pytest.approx(0.0)
    assert execution_time._moe_tensor_parallel_allreduce_time == pytest.approx(10.0)
    assert execution_time._tensor_parallel_allgather_time == pytest.approx(0.0)
    assert execution_time._share_expert_tensor_parallel_allreduce_time == pytest.approx(
        0.0
    )


@pytest.mark.parametrize("routed_token_count", (0, 4))
def test_monolithic_valid_lane_including_zero_routed_lane_is_admitted(
    routed_token_count: int,
) -> None:
    predictor = _configure_monolithic(dummy=True)
    sentinel = object()
    predictor._get_dummy_execution_time = MagicMock(return_value=sentinel)

    result = predictor.predict_stage_execution_time(
        batch=_batch(
            lane_workload=_lane(routed_token_count=routed_token_count),
            total_num_tokens=routed_token_count,
        ),
        stage_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        include_moe=True,
        include_ffn=True,
    )

    assert result is sentinel


@pytest.mark.parametrize("routed_token_count", (0, 3))
def test_source_batch_explicit_lane_accepts_partial_or_zero_assignment_subset(
    routed_token_count: int,
) -> None:
    """A source batch plus one lane may represent a strict assignment subset."""
    predictor = _configure_monolithic(dummy=True)
    source_batch = _batch(total_num_tokens=4, effective_tokens=8)
    lane = _lane(routed_token_count=routed_token_count)

    resolved_lane = predictor._admit_routed_ep_aggregate(
        source_batch,
        routed_moe=True,
        lane_workload=lane,
    )

    assert resolved_lane is lane


def test_disaggregation_aggregate_stage_admission_uses_active_role_topk() -> None:
    """Aggregate predictors must validate lanes against the requested role config."""
    predictor = _configure_disaggregation(dummy=True)
    predictor._cluster_type = None
    # The representative config intentionally differs from the active role.
    predictor._router_topk = 1
    role_replica_config = SimpleNamespace(
        model_config=_ModelConfig(),
        moe_expert_parallel_size=2,
        router_topk=2,
    )
    predictor._get_cluster_replica_config = MagicMock(
        return_value=role_replica_config
    )
    sentinel = SimpleNamespace(num_layers=1)
    predictor._get_dummy_execution_time_for_cluster = MagicMock(
        return_value=sentinel
    )

    result = predictor.predict_stage_execution_time(
        batch=_batch(lane_workload=_lane(), total_num_tokens=4),
        stage_id=0,
        cluster_type=ClusterType.DECODE_FFN,
        num_layers=1,
        layer_id=0,
        include_moe=None,
        include_ffn=True,
    )

    assert result is sentinel
    predictor._get_dummy_execution_time_for_cluster.assert_called_once()


def test_non_dummy_moe_layer_consumes_explicit_active_topology_before_lookup() -> None:
    """An inherited MoE call must validate against its active role context."""
    predictor = _configure_disaggregation(dummy=False)
    predictor._router_topk = 1
    predictor._supports_operation = MagicMock(return_value=False)

    with pytest.raises(
        NotImplementedError,
        match="MoE operations not supported for cluster type",
    ):
        predictor.predict_moe_layer_time(
            _batch(lane_workload=_lane(), total_num_tokens=4),
            layer_id=0,
            cluster_type=ClusterType.DECODE_FFN,
            lane_workload=_lane(),
            ep_size=2,
            router_topk=2,
        )

    predictor._supports_operation.assert_called_once_with("moe_grouped_gemm")


def test_disaggregation_aggregate_non_dummy_propagates_active_role_topology() -> None:
    """The inherited non-dummy MoE call keeps the active role topology."""
    predictor = _configure_disaggregation(dummy=False)
    predictor._cluster_type = None
    # The aggregate representative intentionally differs from the active role.
    predictor._router_topk = 1
    role_replica_config = SimpleNamespace(
        model_config=_ModelConfig(),
        moe_expert_parallel_size=2,
        router_topk=2,
        moe_tensor_parallel_size=1,
        attn_tensor_parallel_size=1,
        num_pipeline_stages=1,
    )
    predictor._get_cluster_replica_config = MagicMock(
        return_value=role_replica_config
    )
    predictor._is_zero_token_decode_ffn_ep_barrier = lambda *_args: False
    predictor._select_measurement_type_for_batch = lambda _batch: None
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda *_args: None
    predictor._emit_cuda_graph_activation_records = lambda *_args: None
    predictor._get_communication_time = lambda *_args, **_kwargs: SimpleNamespace(
        tensor_parallel_time=0.0,
        pipeline_parallel_time=0.0,
    )
    predictor._get_overhead_time = lambda *_args: SimpleNamespace(
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        pp_producer_send_path_runtime_time=0.0,
        pp_receiver_head_runtime_time=0.0,
        pp_prefill_consumer_active_runtime_time=0.0,
        pp_stage_boundary_residual_runtime_time=0.0,
        pp_stage_boundary_handoff_time=0.0,
    )
    predictor._get_pp_stage_boundary_handoff_time = lambda *_args: 0.0
    predictor._resolve_layer_lane_workload = lambda *_args, **_kwargs: _lane()
    predictor._get_cluster_model_architecture_profile = lambda _cluster_type: (
        ModelArchitectureProfile.generic()
    )
    predictor._get_add_layer_act_execution_time = lambda _batch: 0.0
    predictor._predict_named_ep_phase_operator_times = lambda **_kwargs: {}
    predictor._predict_one_op_time = lambda _name, value, *_args, **_kwargs: value
    predictor._predict_comm_operator_with_context = lambda *_args, **_kwargs: 0.0
    moe_call = MagicMock(
        return_value=SimpleNamespace(
            moe_grouped_gemm_time=0.0,
            moe_gating_time=0.0,
            moe_shuffling_time=0.0,
            share_expert_up_proj_time=0.0,
            share_expert_down_proj_time=0.0,
            share_expert_act_time=0.0,
        )
    )
    predictor.predict_moe_layer_time = moe_call

    result = predictor.predict_stage_execution_time(
        batch=_batch(lane_workload=_lane(), total_num_tokens=4),
        stage_id=0,
        cluster_type=ClusterType.DECODE_FFN,
        num_layers=1,
        layer_id=0,
        include_moe=None,
        include_ffn=True,
    )

    assert result._is_moe is True
    moe_call.assert_called_once()
    call_kwargs = moe_call.call_args.kwargs
    assert call_kwargs["ep_size"] == 2
    assert call_kwargs["router_topk"] == 2


def test_disaggregation_decode_attn_remains_attention_only_without_lane() -> None:
    predictor = _configure_disaggregation(
        dummy=True,
        cluster_type=ClusterType.DECODE_ATTN,
    )
    sentinel = object()
    predictor._log_architecture_attention_shape = lambda _batch: None
    predictor._get_dummy_execution_time_for_cluster = MagicMock(
        return_value=SimpleNamespace(num_layers=1)
    )

    result = predictor.predict_stage_execution_time(
        batch=_batch(),
        stage_id=0,
        cluster_type=ClusterType.DECODE_ATTN,
        num_layers=1,
        include_moe=None,
        include_ffn=True,
    )

    assert result.num_layers == 1
