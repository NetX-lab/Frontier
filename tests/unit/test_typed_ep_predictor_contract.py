"""Contract tests for typed MoE EP-lane predictor inputs."""

from __future__ import annotations

from types import SimpleNamespace
import inspect
from unittest.mock import MagicMock

import pytest

from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from frontier.operators.families import get_comm_operator
from frontier.operators.spec import CommPayloadContext
from frontier.moe_ep_workload import (
    EPLaneWorkload,
    materialize_layer_ep_workload,
    resolve_ep_lane_workload,
)
from frontier.types import ClusterType, MeasurementType


class _Predictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("not used")

    def _get_grid_search_params(self):
        return {}


class _CountingPredictor(_Predictor):
    def __init__(self, *, model_result: float = 7.5) -> None:
        self._enable_dummy_mode = False
        self._cluster_type = ClusterType.MONOLITHIC
        self._active_measurement_type = MeasurementType.CUDA_EVENT
        self._measurement_family_name = lambda _measurement_type: "eager"
        self._router_topk = 2
        self._moe_ep_size = 2
        self._model_config = SimpleNamespace(
            embedding_dim=4096,
            mlp_hidden_dim=11008,
        )
        self._replica_config = SimpleNamespace(total_expert_num=8)
        self._supports_operation = lambda _operation: True
        self._predictions = {
            "moe_shuffling": {"_on_demand_prediction": True},
            "moe_grouped_gemm": {"_on_demand_prediction": True},
        }
        self._model_calls: list[tuple[str, dict[str, float]]] = []
        self._model_result = model_result

    def _get_on_demand_prediction(
        self,
        model_name: str,
        features: dict[str, float],
    ) -> float:
        self._model_calls.append((model_name, features))
        return self._model_result


def _lane(
    *,
    ep_id: int,
    ep_size: int,
    total_experts: int,
    counts: tuple[int, ...],
) -> EPLaneWorkload:
    width = total_experts // ep_size
    owned_ids = tuple(range(ep_id * width, (ep_id + 1) * width))
    return EPLaneWorkload(
        ep_id=ep_id,
        moe_expert_parallel_size=ep_size,
        total_expert_num=total_experts,
        owned_expert_ids=owned_ids,
        local_token_counts=counts,
        routed_token_count=sum(counts),
        router_topk=2,
    )


def _batch(lane: EPLaneWorkload) -> SimpleNamespace:
    return SimpleNamespace(
        lane_workload=lane,
        get_effective_total_tokens_rounded=lambda _cluster_type: 16,
    )


def _configure_non_dummy_layer_predictor(predictor: _Predictor) -> None:
    """Install the smallest complete state needed by the layer API tests."""

    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._moe_ep_size = 2
    predictor._moe_tp_size = 1
    predictor._model_config.post_attn_norm = False
    predictor._model_config.supports_share_expert = lambda: False
    predictor._get_grouped_gemm_time = lambda _workload, batch=None: 1.0
    predictor._get_moe_shuffling_time = lambda _batch, moe_tokens_input=None: 1.0
    predictor._get_gating_linear_time = lambda _batch: 1.0
    predictor._get_gating_routing_topk_time = lambda _batch: 1.0


def _layer_batch(
    *,
    total_num_tokens: int,
    effective_tokens: int,
    ep_id: int | None = None,
    lane_workload: EPLaneWorkload | None = None,
) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": 1,
        "requests": [],
        "total_num_tokens": total_num_tokens,
        "get_effective_total_tokens_rounded": lambda _cluster_type: effective_tokens,
    }
    if ep_id is not None:
        values["ep_id"] = ep_id
    if lane_workload is not None:
        values["lane_workload"] = lane_workload
    return SimpleNamespace(**values)


def test_regular_batch_uses_typed_lane_for_on_demand_shuffling() -> None:
    predictor = _CountingPredictor()
    lane = _lane(ep_id=1, ep_size=2, total_experts=8, counts=(3, 1, 0, 0))

    result = predictor._get_moe_shuffling_time(_batch(lane))

    assert result == pytest.approx(7.5)
    assert len(predictor._model_calls) == 1
    model_name, features = predictor._model_calls[0]
    assert model_name == "moe_shuffling"
    assert features["num_experts_per_device"] == 4
    assert features["total_routed_tokens"] == 4


@pytest.mark.parametrize(
    ("ep_id", "ep_size", "total_experts", "counts"),
    [
        (0, 1, 4, (2, 2, 0, 0)),
        (1, 2, 8, (2, 0, 0, 0)),
    ],
)
def test_grouped_gemm_uses_the_same_typed_lane_contract(
    ep_id: int,
    ep_size: int,
    total_experts: int,
    counts: tuple[int, ...],
) -> None:
    predictor = _CountingPredictor()
    predictor._moe_ep_size = ep_size
    lane = _lane(
        ep_id=ep_id,
        ep_size=ep_size,
        total_experts=total_experts,
        counts=counts,
    )

    result = predictor._get_grouped_gemm_time(lane, batch=_batch(lane))

    assert result == pytest.approx(7.5)
    assert len(predictor._model_calls) == 1
    assert predictor._model_calls[0][0] == "moe_grouped_gemm"
    assert predictor._model_calls[0][1]["num_experts_per_device"] == (
        total_experts // ep_size
    )


def test_zero_routed_lane_returns_zero_without_positive_load_model_calls() -> None:
    predictor = _CountingPredictor()
    lane = _lane(ep_id=1, ep_size=2, total_experts=8, counts=(0, 0, 0, 0))
    batch = _batch(lane)

    assert predictor._get_moe_shuffling_time(batch) == 0.0
    assert predictor._get_grouped_gemm_time(lane, batch=batch) == 0.0
    assert predictor._model_calls == []


def test_missing_moe_lane_fails_before_phase_or_model_queries() -> None:
    """Reject a missing physical workload before any MoE-side lookup runs."""

    predictor = _CountingPredictor()
    predictor._moe_ep_size = 2
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
    )
    predictor._model_config.supports_share_expert = lambda: False
    calls: list[str] = []
    predictor._predict_expert_parallel_phase_operator_times = lambda *args, **kwargs: (
        calls.append("phase") or {}
    )
    predictor._get_gating_linear_time = lambda _batch: (
        calls.append("gating_linear") or 0.0
    )
    predictor._get_gating_routing_topk_time = lambda _batch: (
        calls.append("gating_topk") or 0.0
    )
    predictor._get_moe_shuffling_time = lambda *args, **kwargs: (
        calls.append("shuffling") or 0.0
    )

    with pytest.raises(ValueError, match="moe_tokens_input is required"):
        predictor._get_execution_time_internal(
            batch=SimpleNamespace(),
            pipeline_stage=0,
            moe_tokens_input=None,
            include_moe=True,
            include_ffn=True,
            include_attention=False,
        )

    assert calls == []
    assert predictor._model_calls == []


def test_raw_moe_map_fails_before_phase_or_model_queries() -> None:
    """Reject a raw expert map before any MoE-side lookup runs."""

    predictor = _CountingPredictor()
    predictor._moe_ep_size = 2
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
    )
    predictor._model_config.supports_share_expert = lambda: False
    calls: list[str] = []
    predictor._predict_expert_parallel_phase_operator_times = lambda *args, **kwargs: (
        calls.append("phase") or {}
    )
    predictor._get_gating_linear_time = lambda _batch: (
        calls.append("gating_linear") or 0.0
    )
    predictor._get_gating_routing_topk_time = lambda _batch: (
        calls.append("gating_topk") or 0.0
    )
    predictor._get_moe_shuffling_time = lambda *args, **kwargs: (
        calls.append("shuffling") or 0.0
    )

    with pytest.raises(TypeError, match="raw expert-token map"):
        predictor._get_execution_time_internal(
            batch=SimpleNamespace(),
            pipeline_stage=0,
            moe_tokens_input={0: 2, 1: 2},
            include_moe=True,
            include_ffn=True,
            include_attention=False,
        )

    assert calls == []
    assert predictor._model_calls == []


def test_execution_time_internal_rejects_mismatched_lane_descriptors_before_queries() -> None:
    """The communication and routed-compute inputs must name one physical lane."""

    predictor = _CountingPredictor()
    lane_a = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(3, 1, 0, 0))
    lane_b = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(2, 2, 0, 0))

    with pytest.raises(ValueError, match="same EPLaneWorkload descriptor"):
        predictor._get_execution_time_internal(
            batch=SimpleNamespace(),
            pipeline_stage=0,
            moe_tokens_input=lane_a,
            lane_workload=lane_b,
            include_moe=True,
            include_ffn=True,
            include_attention=False,
        )

    assert predictor._model_calls == []


def test_execution_time_internal_rejects_scalar_with_explicit_lane() -> None:
    """A scalar cannot drive routed compute while a lane drives communication."""

    predictor = _CountingPredictor()
    lane = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(3, 1, 0, 0))

    with pytest.raises(TypeError, match="cannot combine a scalar"):
        predictor._get_execution_time_internal(
            batch=SimpleNamespace(),
            pipeline_stage=0,
            moe_tokens_input=8,
            lane_workload=lane,
            include_moe=True,
            include_ffn=True,
            include_attention=False,
        )

    assert predictor._model_calls == []


def test_moe_execution_input_resolver_rejects_mismatched_descriptors() -> None:
    lane_a = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(3, 1, 0, 0))
    lane_b = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(2, 2, 0, 0))

    with pytest.raises(ValueError, match="same EPLaneWorkload descriptor"):
        SklearnMoEExecutionTimePredictor._resolve_moe_execution_inputs(
            moe_tokens_input=lane_a,
            lane_workload=lane_b,
            include_moe=True,
        )


def test_moe_execution_input_resolver_rejects_scalar_with_lane() -> None:
    lane = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(3, 1, 0, 0))

    with pytest.raises(TypeError, match="cannot combine a scalar"):
        SklearnMoEExecutionTimePredictor._resolve_moe_execution_inputs(
            moe_tokens_input=8,
            lane_workload=lane,
            include_moe=True,
        )


def test_moe_execution_input_resolver_uses_lane_as_canonical_input() -> None:
    lane = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(3, 1, 0, 0))

    resolved_input, resolved_lane = (
        SklearnMoEExecutionTimePredictor._resolve_moe_execution_inputs(
            moe_tokens_input=None,
            lane_workload=lane,
            include_moe=True,
        )
    )

    assert resolved_input is lane
    assert resolved_lane is lane


def test_raw_expert_map_is_not_a_predictor_workload_contract() -> None:
    predictor = _CountingPredictor()

    with pytest.raises((TypeError, ValueError), match="EPLaneWorkload|typed"):
        predictor._get_grouped_gemm_time({0: 2, 1: 2}, batch=None)


def test_predict_moe_layer_time_uses_descriptor_presence_not_accidental_ep_id() -> None:
    """A regular batch's incidental ``ep_id`` must not select lane accounting."""

    predictor = _CountingPredictor()
    _configure_non_dummy_layer_predictor(predictor)
    lane = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(8, 0, 0, 0))
    batch = _layer_batch(
        total_num_tokens=4,
        effective_tokens=4,
        ep_id=0,
    )

    result = predictor.predict_moe_layer_time(
        batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        lane_workload=lane,
    )

    assert result.moe_grouped_gemm_time == pytest.approx(1.0)


def test_predict_moe_layer_time_accepts_partial_lane_from_source_batch() -> None:
    """A source batch may be paired with one lane assignment subset."""

    predictor = _CountingPredictor()
    _configure_non_dummy_layer_predictor(predictor)
    lane = _lane(
        ep_id=0,
        ep_size=2,
        total_experts=8,
        counts=(3, 0, 0, 0),
    )
    batch = _layer_batch(
        total_num_tokens=5,
        effective_tokens=8,
    )

    result = predictor.predict_moe_layer_time(
        batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        lane_workload=lane,
    )

    assert result.moe_grouped_gemm_time == pytest.approx(1.0)


def test_predict_moe_layer_time_accepts_zero_lane_from_source_batch() -> None:
    """A source batch may be paired with a zero-routed physical lane."""

    predictor = _CountingPredictor()
    _configure_non_dummy_layer_predictor(predictor)
    lane = _lane(
        ep_id=0,
        ep_size=2,
        total_experts=8,
        counts=(0, 0, 0, 0),
    )
    batch = _layer_batch(
        total_num_tokens=5,
        effective_tokens=8,
    )

    result = predictor.predict_moe_layer_time(
        batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
        lane_workload=lane,
    )

    assert result.moe_grouped_gemm_time == pytest.approx(1.0)
    assert predictor._model_calls == []


def test_predict_moe_layer_time_rejects_mismatched_batch_and_explicit_lane_before_lookup() -> None:
    """A batch descriptor and explicit descriptor must identify one lane."""

    predictor = _CountingPredictor()
    _configure_non_dummy_layer_predictor(predictor)
    batch_lane = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(3, 1, 0, 0))
    explicit_lane = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(2, 2, 0, 0))
    batch = _layer_batch(
        total_num_tokens=4,
        effective_tokens=4,
        ep_id=0,
        lane_workload=batch_lane,
    )
    lookups: list[str] = []
    predictor._get_grouped_gemm_time = lambda *_args, **_kwargs: (
        lookups.append("grouped_gemm") or 1.0
    )
    predictor._get_gating_linear_time = lambda _batch: (
        lookups.append("gating_linear") or 1.0
    )
    predictor._get_gating_routing_topk_time = lambda _batch: (
        lookups.append("gating_topk") or 1.0
    )
    predictor._get_moe_shuffling_time = lambda *_args, **_kwargs: (
        lookups.append("shuffling") or 1.0
    )

    with pytest.raises(ValueError, match="same EPLaneWorkload descriptor"):
        predictor.predict_moe_layer_time(
            batch,
            layer_id=0,
            cluster_type=ClusterType.MONOLITHIC,
            lane_workload=explicit_lane,
        )

    assert lookups == []
    assert predictor._model_calls == []


def test_ep1_non_dummy_layer_api_preserves_scalar_source_width_lookup() -> None:
    """EP=1 ordinary batches retain the standard one-feature predictor path."""

    predictor = _CountingPredictor()
    predictor._moe_ep_size = 1
    predictor._moe_tp_size = 1
    predictor._predictions = {
        "moe_grouped_gemm": {},
        "moe_shuffling": {},
    }
    predictor._model_config.post_attn_norm = False
    predictor._model_config.supports_share_expert = lambda: False
    predictor._get_gating_linear_time = lambda _batch: 1.0
    predictor._get_gating_routing_topk_time = lambda _batch: 2.0

    lookups: list[tuple[str, float]] = []

    def _capture_lookup(
        model_name: str,
        features: dict[str, float],
        feature_names: tuple[str, ...],
    ) -> float:
        assert feature_names == ("num_tokens",)
        lookups.append((model_name, float(features["num_tokens"])))
        return 3.0

    predictor._get_prediction_for_features = _capture_lookup
    batch = _layer_batch(
        total_num_tokens=5,
        effective_tokens=8,
    )

    result = predictor.predict_moe_layer_time(
        batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert result.moe_grouped_gemm_time == pytest.approx(3.0)
    assert result.moe_shuffling_time == pytest.approx(3.0)
    assert lookups == [
        ("moe_grouped_gemm", 8.0),
        ("moe_shuffling", 8.0),
    ]
    assert predictor._predict_expert_parallel_phase_operator_times(batch) == {
        "expert_parallel_alltoall_dispatch": 0.0,
        "expert_parallel_alltoall_combine": 0.0,
    }


def test_one_feature_lane_lookups_use_source_width_for_non_divisible_assignments() -> None:
    """A lane's assignment subset must not be divided by router top-k."""

    predictor = _CountingPredictor()
    predictor._predictions = {
        "moe_grouped_gemm": {},
        "moe_shuffling": {},
    }
    lane = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(3, 0, 0, 0))
    batch = SimpleNamespace(
        get_effective_total_tokens_rounded=lambda _cluster_type: 5,
    )
    lookups: list[tuple[str, float]] = []

    def _capture_lookup(
        model_name: str,
        features: dict[str, float],
        feature_names: tuple[str, ...],
    ) -> float:
        assert feature_names == ("num_tokens",)
        lookups.append((model_name, float(features["num_tokens"])))
        return 2.0

    predictor._get_prediction_for_features = _capture_lookup

    assert predictor._get_grouped_gemm_time(lane, batch=batch) == pytest.approx(2.0)
    assert (
        predictor._get_moe_shuffling_time(batch, moe_tokens_input=lane)
        == pytest.approx(2.0)
    )

    assert lookups == [
        ("moe_grouped_gemm", 5.0),
        ("moe_shuffling", 5.0),
    ]


def test_one_feature_lane_lookup_requires_source_batch() -> None:
    """A physical lane alone cannot identify the pre-routing profile key."""

    predictor = _CountingPredictor()
    predictor._predictions = {"moe_grouped_gemm": {}}
    lane = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(3, 0, 0, 0))
    predictor._get_prediction_for_features = lambda *_args, **_kwargs: (
        pytest.fail("source-width lookup must fail before model access")
    )

    with pytest.raises(ValueError, match="source batch"):
        predictor._get_grouped_gemm_time(lane)


def test_public_moe_layer_api_uses_lane_workload_keyword() -> None:
    for method in (
        BaseExecutionTimePredictor.predict_moe_layer_time,
        SklearnMoEExecutionTimePredictor.predict_moe_layer_time,
    ):
        parameter_names = tuple(inspect.signature(method).parameters)
        assert "lane_workload" in parameter_names
        assert "per_expert_tokens" not in parameter_names


def test_optional_lane_accessor_ignores_objects_without_real_attribute() -> None:
    assert resolve_ep_lane_workload(SimpleNamespace(), required=False) is None
    assert resolve_ep_lane_workload(MagicMock(), required=False) is None

    malformed = SimpleNamespace(lane_workload={0: 1})
    with pytest.raises(TypeError, match="EPLaneWorkload"):
        resolve_ep_lane_workload(malformed, required=False)


def test_layer_routing_helper_passes_typed_lanes_to_predictor_consumers() -> None:
    predictor = _CountingPredictor()
    predictor._num_layers_per_pipeline_stage = 1
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._materialize_layer_ep_workload = lambda **_kwargs: materialize_layer_ep_workload(
        routing_ratios={0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0},
        target_replica_id=0,
        global_layer_id=0,
        routing_token_count=4,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=1,
        expert_to_ep={0: 0, 1: 0, 2: 0, 3: 0},
    )
    predictor._get_expert_parallel_communication_time = lambda _batch, **_kwargs: 0.0
    predictor._get_gating_time = lambda _batch: 0.0
    predictor._simulate_routing = lambda _batches: {0: 8}
    grouped_inputs: list[object] = []
    shuffling_inputs: list[object] = []
    predictor._get_grouped_gemm_time = lambda workload, batch=None: (
        grouped_inputs.append(workload) or 7.5
    )
    predictor._get_moe_shuffling_time = lambda batch, moe_tokens_input=None: (
        shuffling_inputs.append(moe_tokens_input) or 1.5
    )

    batch = SimpleNamespace(replica_id=0)
    result = predictor._simulate_routing_per_layer([batch], stage_id=0)

    assert result[0][0]["moe_grouped_gemm_time"] == pytest.approx(7.5)
    assert len(grouped_inputs) == len(shuffling_inputs) == 1
    assert isinstance(grouped_inputs[0], EPLaneWorkload)
    assert isinstance(shuffling_inputs[0], EPLaneWorkload)


def test_ep1_shared_domain_path_materializes_the_same_typed_lane() -> None:
    predictor = _CountingPredictor()
    predictor._enable_dummy_mode = False
    predictor._moe_ep_size = 1
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_gating_linear_time = lambda _batch: 0.0
    predictor._get_gating_routing_topk_time = lambda _batch: 0.0
    predictor._model_config.supports_share_expert = lambda: False
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0},
        target_replica_id=0,
        global_layer_id=0,
        routing_token_count=4,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=1,
        expert_to_ep={0: 0, 1: 0, 2: 0, 3: 0},
    )
    predictor._materialize_layer_ep_workload = lambda **_kwargs: workload
    grouped_inputs: list[object] = []
    shuffling_inputs: list[object] = []
    predictor._get_grouped_gemm_time = lambda value, batch=None: (
        grouped_inputs.append(value) or 1.0
    )
    predictor._get_moe_shuffling_time = lambda batch, moe_tokens_input=None: (
        shuffling_inputs.append(moe_tokens_input) or 2.0
    )

    predictor.predict_monolithic_decode_shared_domain_lane_moe_times_ms(
        batch=SimpleNamespace(
            replica_id=0,
            total_num_tokens=4,
            get_effective_total_tokens_rounded=lambda _cluster_type: 4,
        ),
        layer_id=0,
    )

    assert len(grouped_inputs) == len(shuffling_inputs) == 1
    assert isinstance(grouped_inputs[0], EPLaneWorkload)
    assert isinstance(shuffling_inputs[0], EPLaneWorkload)


def test_regular_on_demand_stage_input_materializes_ep1_lane_descriptor() -> None:
    predictor = _CountingPredictor()
    predictor._moe_ep_size = 1
    predictor._cluster_type = ClusterType.MONOLITHIC
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0},
        target_replica_id=0,
        global_layer_id=0,
        routing_token_count=4,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=1,
        expert_to_ep={0: 0, 1: 0, 2: 0, 3: 0},
    )
    predictor._materialize_layer_ep_workload = lambda **_kwargs: workload

    resolved = predictor._get_moe_tokens_input(
        SimpleNamespace(
            replica_id=0,
            total_num_tokens=4,
            get_effective_total_tokens_rounded=lambda _cluster_type: 4,
        ),
        layer_id=0,
    )

    assert isinstance(resolved, EPLaneWorkload)


def test_canonical_lane_resolver_rejects_regular_ep2_without_physical_lane() -> None:
    predictor = _CountingPredictor()
    predictor._moe_ep_size = 2
    predictor._cluster_type = ClusterType.MONOLITHIC
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0},
        target_replica_id=0,
        global_layer_id=0,
        routing_token_count=4,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=2,
        expert_to_ep={0: 0, 1: 0, 2: 1, 3: 1},
    )
    predictor._materialize_layer_ep_workload = lambda **_kwargs: workload

    with pytest.raises(ValueError, match="physical EP lane"):
        predictor._resolve_layer_lane_workload(
            SimpleNamespace(replica_id=0, total_num_tokens=4),
            cluster_type=ClusterType.MONOLITHIC,
            layer_id=0,
        )


def _comm_context_for_lane(lane: EPLaneWorkload) -> CommPayloadContext:
    return CommPayloadContext(
        batch=SimpleNamespace(
            get_effective_total_tokens_rounded=lambda _cluster_type: 99,
        ),
        model_config=SimpleNamespace(embedding_dim=8, num_experts_per_tok=2),
        replica_config=SimpleNamespace(
            attn_tensor_parallel_size=1,
            moe_tensor_parallel_size=2,
            moe_expert_parallel_size=lane.moe_expert_parallel_size,
            router_topk=lane.router_topk,
        ),
        cluster_type=ClusterType.MONOLITHIC,
        quantization_manager=SimpleNamespace(
            adjust_tensor_size=lambda _collective, data_size_bytes, _cluster_type: data_size_bytes
        ),
        lane_workload=lane,
    )


def test_communication_payload_uses_typed_lane_without_entity_type_branch() -> None:
    lane = _lane(ep_id=1, ep_size=2, total_experts=8, counts=(3, 1, 0, 0))
    context = _comm_context_for_lane(lane)

    assert (
        get_comm_operator("moe_tensor_parallel_allreduce").build_payload_bytes(context)
        == 8 * 2 * 99
    )
    assert (
        get_comm_operator("expert_parallel_alltoall").build_payload_bytes(context)
        == 8 * 2 * lane.routed_token_count
    )


def test_moe_tp_allreduce_uses_source_batch_width_while_ep_uses_lane_width() -> None:
    """Shared MoE-TP hidden-state collectives keep the pre-routing domain."""

    lane = _lane(ep_id=1, ep_size=2, total_experts=8, counts=(3, 1, 0, 0))
    context = _comm_context_for_lane(lane)

    assert (
        get_comm_operator("moe_tensor_parallel_allreduce").build_payload_bytes(context)
        == 8 * 2 * 99
    )
    assert (
        get_comm_operator("expert_parallel_alltoall").build_payload_bytes(context)
        == 8 * 2 * lane.routed_token_count
    )


def test_communication_payload_keeps_zero_lane_as_physical_zero_payload() -> None:
    lane = _lane(ep_id=1, ep_size=2, total_experts=8, counts=(0, 0, 0, 0))
    context = _comm_context_for_lane(lane)

    assert get_comm_operator("moe_tensor_parallel_allreduce").build_payload_bytes(
        context
    ) == 8 * 2 * 99
    assert get_comm_operator("expert_parallel_alltoall").build_payload_bytes(
        context
    ) == 0


class _PhaseExecution:
    def get_single_layer_moe_pre_dispatch_time(self) -> float:
        return 2.0

    def get_single_layer_moe_dispatch_time(self) -> float:
        return 3.0

    def get_single_layer_moe_post_dispatch_compute_time(self) -> float:
        return 4.0

    def get_single_layer_moe_combine_time(self) -> float:
        return 5.0

    def get_single_layer_moe_post_combine_time(self) -> float:
        return 6.0

    def get_single_layer_post_attention_time(self) -> float:
        return 20.0


def test_mtp_lane_phase_api_is_pure_descriptor_path() -> None:
    predictor = _CountingPredictor()
    lane = _lane(ep_id=0, ep_size=2, total_experts=8, counts=(2, 1, 0, 0))
    calls: list[dict[str, object]] = []
    predictor._get_execution_time_internal = lambda **kwargs: (
        calls.append(kwargs) or _PhaseExecution()
    )

    phases = predictor.predict_moe_lane_phase_times(
        batch=SimpleNamespace(),
        lane_workload=lane,
        pipeline_stage=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert phases == (2.0, 3.0, 4.0, 5.0, 6.0)
    assert calls == [
        {
            "batch": calls[0]["batch"],
            "pipeline_stage": 0,
            "moe_tokens_input": lane,
            "lane_workload": lane,
            "include_moe": True,
            "include_ffn": True,
            "include_attention": False,
        }
    ]


def test_mtp_decoder_replay_delegates_each_lane_without_scheduler_entities() -> None:
    predictor = _CountingPredictor()
    predictor._model_config.is_moe = True
    predictor._model_config.is_moe_layer = lambda _layer_id: True
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._moe_ep_size = 2
    predictor._replica_config.total_expert_num = 8
    predictor._replica_config.moe_expert_parallel_size = 2
    predictor._replica_config.router_topk = 2
    workload = materialize_layer_ep_workload(
        routing_ratios={0: 0.5, 1: 0.5, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0},
        target_replica_id=0,
        global_layer_id=0,
        routing_token_count=4,
        router_topk=2,
        total_expert_num=8,
        moe_expert_parallel_size=2,
        expert_to_ep={
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 1,
            5: 1,
            6: 1,
            7: 1,
        },
    )
    predictor._materialize_layer_ep_workload = lambda **_kwargs: workload
    attention_calls: list[object] = []
    predictor.predict_stage_execution_time = lambda **kwargs: (
        attention_calls.append(kwargs) or SimpleNamespace(model_time_ms=10.0)
    )
    phase_calls: list[EPLaneWorkload] = []

    def _phases(*, batch, lane_workload, pipeline_stage, cluster_type):
        del batch, pipeline_stage, cluster_type
        phase_calls.append(lane_workload)
        return (
            (2.0, 1.0, 4.0, 1.0, 5.0)
            if lane_workload.ep_id == 0
            else (3.0, 2.0, 1.0, 4.0, 6.0)
        )

    predictor.predict_moe_lane_phase_times = _phases
    batch = SimpleNamespace(
        replica_id=0,
        total_num_tokens=4,
        id=17,
        requests=[],
        get_effective_total_tokens_for_compute=lambda _cluster_type: 4,
    )

    result = predictor._predict_mtp_decoder_layer_time_ms(
        predictor=predictor,
        batch=batch,
    )

    assert result == pytest.approx(29.0)
    assert len(attention_calls) == 1
    assert [lane.ep_id for lane in phase_calls] == [0, 1]
