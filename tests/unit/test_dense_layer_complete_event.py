from __future__ import annotations

from types import SimpleNamespace

from frontier.entities import Batch
from frontier.events.dense_layer_complete_event import DenseLayerCompleteEvent
from frontier.types import ClusterType, EventType


def test_dense_layer_complete_event_dispatches_to_stage_transition() -> None:
    batch = Batch(0, [], [], is_moe=True)
    event = DenseLayerCompleteEvent(
        1.25,
        replica_id=2,
        stage_id=3,
        batch=batch,
        layer_id=7,
        phase="decode",
        cluster_type=ClusterType.MONOLITHIC,
    )
    calls = []

    class _ClusterScheduler:
        def on_dense_layer_complete(self, *args):
            calls.append(args)
            return ["next"]

    scheduler = SimpleNamespace(
        get_cluster_scheduler=lambda cluster_type: _ClusterScheduler()
    )
    metrics = object()

    result = event.handle_event(scheduler, metrics)

    assert result == ["next"]
    assert calls == [
        (
            1.25,
            2,
            3,
            batch,
            7,
            "decode",
            metrics,
        )
    ]
    assert event.event_type is EventType.DENSE_LAYER_COMPLETE
    assert event.to_dict()["protocol"] == "FULL_STAGE_WORLD"
    assert "dp_id" not in event.to_dict()
