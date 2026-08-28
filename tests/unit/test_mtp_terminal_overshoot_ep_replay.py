from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.entities import Batch, Request
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType


class _NonDummyMoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def _terminal_source_batch() -> tuple[Batch, VLLMv1EngineReplicaScheduler]:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=8,
        num_decode_tokens=1,
        num_processed_tokens=8,
    )
    request._is_prefill_complete = True
    request.initialize_spec_decode_state(
        enabled=True,
        method="qwen3_moe_mtp",
        num_speculative_tokens=2,
        method_uses_lookahead_slots=True,
    )
    request.set_spec_next_planned_draft_tokens(2)

    scheduler = VLLMv1EngineReplicaScheduler.__new__(
        VLLMv1EngineReplicaScheduler
    )
    scheduler._spec_decode_enabled = True
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._num_stages = 2
    scheduler._spec_method_uses_lookahead_slots = True
    scheduler._spec_decode_config = SimpleNamespace(
        enabled=True,
        method="qwen3_moe_mtp",
        num_speculative_tokens=2,
        committed_tokens_per_iteration=1,
        _per_request_scheduled_draft_tokens_trace={str(request.id): [2, 1]},
        _per_request_committed_tokens_trace={str(request.id): [1, 1]},
    )

    batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[3],
        is_moe=True,
    )
    metadata = scheduler._build_spec_decode_batch_metadata(batch)
    assert metadata is not None
    batch.spec_decode_metadata = metadata
    return batch, scheduler


def _predictor() -> _NonDummyMoEPredictor:
    predictor = _NonDummyMoEPredictor.__new__(_NonDummyMoEPredictor)
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._moe_ep_size = 2
    predictor._moe_tp_size = 1
    predictor._router_topk = 2
    predictor._cc_backend = None
    predictor._model_config = SimpleNamespace(
        is_moe=True,
        is_moe_layer=lambda _layer_id: True,
        supports_share_expert=lambda: False,
        embedding_dim=16,
        mlp_hidden_dim=32,
    )
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=2,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=2,
        router_topk=2,
        total_expert_num=8,
    )
    predictor._num_layers_per_pipeline_stage = 1
    predictor._monolithic_routing_details = {
        0: {
            1: {
                0: 0.25,
                1: 0.25,
                2: 0.25,
                3: 0.25,
                4: 0.0,
                5: 0.0,
                6: 0.0,
                7: 0.0,
            }
        }
    }
    return predictor


def test_terminal_mtp_moe_replay_materializes_physical_ep_lanes() -> None:
    batch, scheduler = _terminal_source_batch()
    metadata = batch.spec_decode_metadata
    assert metadata is not None
    assert metadata.verify_tokens_per_request == [3]
    assert metadata.terminal_overshoot_verify_tokens_per_request == [[1]]

    predictor = _predictor()
    attention_calls: list[dict[str, object]] = []
    lane_calls: list[object] = []

    def _predict_attention_only(**kwargs):
        attention_calls.append(kwargs)
        return SimpleNamespace(model_time_ms=7.0, total_time=0.007)

    predictor.predict_attention_layer_time = _predict_attention_only

    def _predict_moe_lane_phase_times(
        *, batch, lane_workload, pipeline_stage, cluster_type
    ):
        del batch
        assert cluster_type is ClusterType.MONOLITHIC
        assert pipeline_stage == 1
        lane_calls.append((lane_workload, pipeline_stage))
        return (
            (1.0, 2.0, 3.0, 4.0, 5.0)
            if lane_workload.ep_id == 0
            else (2.0, 1.0, 4.0, 3.0, 6.0)
        )

    def _predict_stage_execution_time(**kwargs):
        if kwargs.get("include_ffn") is False:
            return _predict_attention_only(**kwargs)
        return SklearnMoEExecutionTimePredictor._get_execution_time_internal(
            predictor,
            kwargs["batch"],
            pipeline_stage=kwargs["stage_id"],
            moe_tokens_input=kwargs["batch"].total_num_tokens,
            include_moe=True,
            include_ffn=True,
            include_attention=True,
        )

    predictor.predict_stage_execution_time = _predict_stage_execution_time
    predictor.predict_moe_lane_phase_times = _predict_moe_lane_phase_times

    terminal_time_ms = predictor._get_mtp_terminal_overshoot_time(
        batch,
        stage_id=1,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=1,
    )

    assert terminal_time_ms == pytest.approx(25.0)
    assert [lane.ep_id for lane, _ in lane_calls] == [0, 1]
    assert [stage for _, stage in lane_calls] == [1, 1]
    assert [lane.moe_expert_parallel_size for lane, _ in lane_calls] == [2, 2]
    assert sum(lane.routed_token_count for lane, _ in lane_calls) == 2
    assert attention_calls[0]["stage_id"] == 1
    assert attention_calls[0]["layer_id"] == 1
    assert attention_calls[0]["num_layers"] == 1
    assert attention_calls[0]["include_ffn"] is False


def test_dummy_terminal_mtp_moe_replay_uses_typed_lane_phase_contract() -> None:
    """Dummy MTP replay must retain physical EP phases without profiling lookups."""

    batch, _scheduler = _terminal_source_batch()
    predictor = _predictor()
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 1.0
    predictor._get_model_architecture_profile = lambda: ModelArchitectureProfile.generic()

    terminal_time_ms = predictor._get_mtp_terminal_overshoot_time(
        batch,
        stage_id=1,
        cluster_type=ClusterType.MONOLITHIC,
        num_layers=1,
        layer_id=1,
    )

    assert terminal_time_ms > 0.0
