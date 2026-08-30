from __future__ import annotations

from types import SimpleNamespace

from frontier.entities import Batch, Request
from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import (
    ReplicaStageScheduler,
)
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import (
    StageExecutionContext,
)
from frontier.types import ClusterType


class _RecordingPredictor:
    _num_layers_per_pipeline_stage = 4

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict_stage_execution_time(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        return SimpleNamespace(total_time=1.0, model_time=0.8)


def _batch() -> Batch:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=2,
        num_decode_tokens=2,
    )
    return Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[2],
        is_moe=True,
    )


def test_scheduler_passes_global_layer_ids_for_full_pipeline_stage() -> None:
    predictor = _RecordingPredictor()
    scheduler = ReplicaStageScheduler(
        replica_id=0,
        stage_id=1,
        is_last_stage=True,
        is_moe=True,
        execution_time_predictor=predictor,
        cluster_type=ClusterType.MONOLITHIC,
        replica_local_id=0,
        stage_execution_context=StageExecutionContext(replica_id=0, stage_id=1, ep_size=1),
    )

    scheduler.predict_and_create_stage(_batch())

    assert len(predictor.calls) == 1
    assert predictor.calls[0]["num_layers"] == 4
    assert predictor.calls[0]["layer_id"] == 4
    assert predictor.calls[0]["layer_ids"] == (4, 5, 6, 7)
