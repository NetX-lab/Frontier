"""Sequential event logging preserves request-free DP work."""

import pytest

from frontier.entities.batch import DummyForwardBatch
from frontier.entities.batch_stage import BatchStage
from frontier.events.batch_stage_end_event import BatchStageEndEvent
from frontier.events.decode_sync_event import DecodeSyncEvent
from frontier.events.prefill_sync_event import PrefillSyncEvent
from frontier.simulator import Simulator
from frontier.types import ClusterType


@pytest.mark.parametrize("cluster_type", [ClusterType.PREFILL, ClusterType.DECODE])
@pytest.mark.parametrize("layer_id", [None, 3])
def test_request_free_event_details_preserve_known_fields(cluster_type, layer_id):
    batch = DummyForwardBatch(0, 0, 0)
    if layer_id is None:
        stage = BatchStage(
            batch.id, 0, 0, 0.1, 0.1, batch.requests, batch.num_tokens,
            batch.request_is_decoding, cluster_type,
        )
        stage.on_schedule(0.9)
        event = BatchStageEndEvent(1.0, 0, 0, True, batch, stage, cluster_type, 1)
    else:
        event_type = PrefillSyncEvent if cluster_type == ClusterType.PREFILL else DecodeSyncEvent
        event = event_type(
            1.0, 0, 0, batch, 1, "pre_moe", layer_id, 0.1, cluster_type=cluster_type
        )

    details = Simulator.__new__(Simulator)._extract_sequential_event_details(
        event, cluster_type
    )

    assert details["request_ids"] == []
    assert details["batch_size"] == details["num_tokens"] == 1
    assert details["replica_local_id"] == 1
    if layer_id is None:
        assert "layer_id" not in details
    else:
        assert details["layer_id"] == layer_id
