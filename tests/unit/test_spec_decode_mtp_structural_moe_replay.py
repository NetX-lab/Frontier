from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.entities import Batch, EPBatchGroup, Request, SpecDecodeBatchMetadata
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.execution_time_predictor.execution_time_predictor_registry import (
    ExecutionTimePredictorRegistry,
)
import frontier.execution_time_predictor.sklearn_execution_time_predictor as sklearn_predictor_module
from frontier.types import ClusterType


class _DummyPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


class _LaneExecutionTime:
    def __init__(
        self,
        *,
        pre_dispatch_ms: float,
        dispatch_ms: float,
        routed_compute_ms: float,
        combine_ms: float,
    ) -> None:
        self._phase_times = (
            pre_dispatch_ms,
            dispatch_ms,
            routed_compute_ms,
            combine_ms,
        )

    def get_single_layer_moe_pre_dispatch_time(self) -> float:
        return self._phase_times[0]

    def get_single_layer_moe_dispatch_time(self) -> float:
        return self._phase_times[1]

    def get_single_layer_moe_post_dispatch_compute_time(self) -> float:
        return self._phase_times[2]

    def get_single_layer_moe_combine_time(self) -> float:
        return self._phase_times[3]

    def get_single_layer_post_attention_time(self) -> float:
        return sum(self._phase_times)


def _request() -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=16,
        num_decode_tokens=32,
        num_processed_tokens=16,
    )
    request._is_prefill_complete = True
    return request


def _source_batch() -> Batch:
    return Batch(
        replica_id=0,
        requests=[_request()],
        num_tokens=[4],
        is_moe=True,
    )


def _predictor() -> _DummyPredictor:
    predictor = _DummyPredictor.__new__(_DummyPredictor)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._model_config = SimpleNamespace(
        is_moe=True,
        is_moe_layer=lambda layer_id: layer_id == 0,
    )
    predictor._replica_config = SimpleNamespace(
        total_expert_num=4,
        moe_expert_parallel_size=2,
        router_topk=1,
    )
    predictor._monolithic_routing_details = {
        0: {
            0: {
                0: 0.25,
                1: 0.25,
                2: 0.50,
                3: 0.00,
            }
        }
    }
    return predictor


def _secondary_predictor_parent() -> _DummyPredictor:
    predictor = _DummyPredictor.__new__(_DummyPredictor)
    predictor._mtp_secondary_predictors = {}
    predictor._replica_scheduler_provider = "vllm_v1"
    predictor._block_size = 16
    predictor._cache_dir = "/data/ycfeng/tmp/mtp-secondary-test-cache"
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._cc_backend = None
    predictor._config = SimpleNamespace(
        get_type=lambda: "random_forrest",
        prediction_max_tokens_per_request=128,
    )
    predictor._replica_config = SimpleNamespace(
        model_name="target-model",
        attn_tensor_parallel_size=4,
        attn_dp=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=4,
        total_expert_num=512,
        local_expert_num=128,
        router_load_balancing_type="None",
        router_topk=10,
        moe_routing_distribution_type="random",
        moe_routing_seed=42,
        cluster_prefix=None,
        node_config=SimpleNamespace(num_devices_per_node=8),
        device="h800",
        network_device="a100_pairwise_nvlink",
        cluster_num_replicas=4,
    )
    predictor._discover_mtp_model_training_file_paths = lambda **_: {}
    return predictor


def test_mtp_structural_moe_decoder_replays_local_ep_lanes_and_barriers() -> None:
    predictor = _predictor()
    source_batch = _source_batch()
    lane_calls: list[EPBatchGroup] = []

    def _predict_stage_execution_time(batch, stage_id, cluster_type, **kwargs):
        assert stage_id == 0
        assert cluster_type == ClusterType.MONOLITHIC
        if not isinstance(batch, EPBatchGroup):
            assert kwargs.get("include_ffn") is False
            return SimpleNamespace(model_time_ms=10.0)

        lane_calls.append(batch)
        assert batch.moe_pre_routing_effective_total_tokens == 4
        phase_times_by_ep = {
            0: (2.0, 1.0, 4.0, 1.0),
            1: (3.0, 2.0, 1.0, 4.0),
        }
        return _LaneExecutionTime(
            pre_dispatch_ms=phase_times_by_ep[batch.ep_id][0],
            dispatch_ms=phase_times_by_ep[batch.ep_id][1],
            routed_compute_ms=phase_times_by_ep[batch.ep_id][2],
            combine_ms=phase_times_by_ep[batch.ep_id][3],
        )

    predictor.predict_stage_execution_time = _predict_stage_execution_time

    result_ms = predictor._predict_mtp_decoder_layer_time_ms(
        predictor=predictor,
        batch=source_batch,
    )

    assert result_ms == pytest.approx(23.0)
    assert [lane.ep_id for lane in lane_calls] == [0, 1]
    assert [lane.per_expert_tokens for lane in lane_calls] == [
        {0: 1, 1: 1},
        {2: 2, 3: 0},
    ]


def test_mtp_synthetic_batch_suppresses_nested_proposer_accounting() -> None:
    predictor = _predictor()
    predictor._replica_config.suppress_spec_decode_proposer_overhead = False
    source_batch = _source_batch()
    source_batch.spec_decode_metadata = SpecDecodeBatchMetadata(
        method="qwen3_next_mtp",
        planned_draft_tokens_per_request=[1],
        verify_tokens_per_request=[2],
        accepted_draft_tokens_per_request=[1],
        rejected_draft_tokens_per_request=[0],
        committed_tokens_per_request=[2],
        uses_lookahead_slots=True,
    )

    synthetic_batch = predictor._build_mtp_synthetic_batch(
        source_batch=source_batch,
        active_indices=[0],
        is_moe=True,
        num_tokens=[2],
        copy_spec_decode_metadata=True,
    )

    assert synthetic_batch.spec_decode_metadata is not None
    assert predictor._should_include_spec_decode_proposer_overhead(source_batch)
    assert not predictor._should_include_spec_decode_proposer_overhead(
        synthetic_batch
    )


def test_mtp_secondary_predictor_preserves_cluster_replica_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _secondary_predictor_parent()
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(
        sklearn_predictor_module,
        "load_mtp_structural_model_config",
        lambda _model_name: SimpleNamespace(
            num_experts=512,
            num_experts_per_tok=10,
        ),
    )

    def _capture_predictor(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        ExecutionTimePredictorRegistry,
        "get",
        staticmethod(_capture_predictor),
    )

    result = predictor._get_or_create_mtp_secondary_predictor(
        contract=SimpleNamespace(proposer_model_name="target-model")
    )

    assert result is sentinel
    secondary_config = captured["replica_config"]
    assert secondary_config.cluster_num_replicas == 4


def test_mtp_secondary_predictor_rejects_missing_cluster_replica_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = _secondary_predictor_parent()
    del predictor._replica_config.cluster_num_replicas

    monkeypatch.setattr(
        sklearn_predictor_module,
        "load_mtp_structural_model_config",
        lambda _model_name: SimpleNamespace(
            num_experts=512,
            num_experts_per_tok=10,
        ),
    )
    monkeypatch.setattr(
        ExecutionTimePredictorRegistry,
        "get",
        staticmethod(lambda **_: object()),
    )

    with pytest.raises(
        ValueError,
        match="MTP secondary predictor requires a positive cluster replica count",
    ):
        predictor._get_or_create_mtp_secondary_predictor(
            contract=SimpleNamespace(proposer_model_name="target-model")
        )
