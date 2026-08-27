from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from frontier.events.ep_alltoall_combine_collective_event import (
    EPAllToAllCombineCollectiveEvent,
)
from frontier.model_architectures import MODEL_ARCHITECTURE_REGISTRY
from frontier.moe_ep_workload import EPLaneWorkload
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.types import ClusterType


class _TestClusterScheduler(BaseClusterScheduler):
    def schedule(self):
        raise AssertionError("schedule() is not used by these unit tests")


class _CollectivePredictorSpy:
    def __init__(self, predicted_ms: float) -> None:
        self.predicted_ms = predicted_ms
        self.alltoall_calls = []
        self.allgather_calls = []

    def predict_alltoall_time(self, **kwargs) -> float:
        self.alltoall_calls.append(kwargs)
        return self.predicted_ms

    def predict_allgather_time(self, **kwargs) -> float:
        self.allgather_calls.append(kwargs)
        return self.predicted_ms


def _build_scheduler(
    *, architecture_profile: str, predicted_ms: float = 12.5
) -> tuple[_TestClusterScheduler, _CollectivePredictorSpy]:
    scheduler = _TestClusterScheduler.__new__(_TestClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE_FFN
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=SimpleNamespace(
                model_architecture_profile=architecture_profile,
                model_type=architecture_profile,
                embedding_dim=4096,
                get_model_architecture_profile=lambda: (
                    MODEL_ARCHITECTURE_REGISTRY.get(architecture_profile)
                ),
            ),
            moe_expert_parallel_size=2,
        )
    )
    scheduler._ep_allgather_waiting_room = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"batches": {}, "arrival_times": {}})
        )
    )
    scheduler.get_replica = lambda replica_id: SimpleNamespace(
        id=replica_id,
        ep_size=2,
    )
    predictor = _CollectivePredictorSpy(predicted_ms)
    scheduler._predictor = predictor
    return scheduler, predictor


def _batch(
    *,
    batch_id: int,
    global_id: int,
    total_num_tokens: int,
    replica_id: int,
    ep_id: int,
):
    return SimpleNamespace(
        id=batch_id,
        global_id=global_id,
        total_num_tokens=total_num_tokens,
        replica_id=replica_id,
        ep_id=ep_id,
        post_combine_time=0.0,
        lane_workload=EPLaneWorkload(
            ep_id=ep_id,
            moe_expert_parallel_size=2,
            total_expert_num=2,
            owned_expert_ids=(ep_id,),
            local_token_counts=(total_num_tokens,),
            routed_token_count=total_num_tokens,
            router_topk=1,
        ),
    )


@pytest.mark.parametrize(
    ("first_ep_id", "first_tokens", "second_ep_id", "second_tokens"),
    [(1, 0, 0, 4), (0, 4, 1, 0)],
)
def test_step3_alltoall_combine_uses_max_lane_payload_independent_of_arrival_order(
    first_ep_id: int,
    first_tokens: int,
    second_ep_id: int,
    second_tokens: int,
) -> None:
    scheduler, predictor = _build_scheduler(architecture_profile="step3_text")
    first_batch = _batch(
        batch_id=10,
        global_id=77,
        total_num_tokens=first_tokens,
        replica_id=3,
        ep_id=first_ep_id,
    )
    second_batch = _batch(
        batch_id=11,
        global_id=77,
        total_num_tokens=second_tokens,
        replica_id=3,
        ep_id=second_ep_id,
    )

    first_events = scheduler.on_ep_alltoall_combine_ready(
        time=1.25,
        replica_id=3,
        stage_id=4,
        batch=first_batch,
        ep_id=first_ep_id,
    )

    assert first_events == []
    assert predictor.alltoall_calls == []
    assert predictor.allgather_calls == []

    second_events = scheduler.on_ep_alltoall_combine_ready(
        time=1.5,
        replica_id=3,
        stage_id=4,
        batch=second_batch,
        ep_id=second_ep_id,
    )

    assert len(second_events) == 1
    assert isinstance(second_events[0], EPAllToAllCombineCollectiveEvent)
    assert second_events[0].time == pytest.approx(1.5125)
    assert second_events[0].to_dict()["replica_id"] == 3
    assert second_events[0].to_dict()["stage_id"] == 4
    assert second_events[0].to_dict()["batch_global_id"] == 77
    assert predictor.alltoall_calls == [
        {
            "data_size_bytes": 4 * 4096 * 2,
            "num_devices": 2,
            "cluster_type": ClusterType.DECODE_FFN,
            "comm_domain": "EP",
        }
    ]
    assert predictor.allgather_calls == []


@pytest.mark.parametrize(
    ("first_ep_id", "first_tokens", "second_ep_id", "second_tokens"),
    [(1, 0, 0, 4), (0, 4, 1, 0)],
)
def test_generic_alltoall_combine_uses_max_lane_payload_independent_of_arrival_order(
    first_ep_id: int,
    first_tokens: int,
    second_ep_id: int,
    second_tokens: int,
) -> None:
    scheduler, predictor = _build_scheduler(architecture_profile="generic")
    first_batch = _batch(
        batch_id=20,
        global_id=88,
        total_num_tokens=first_tokens,
        replica_id=5,
        ep_id=first_ep_id,
    )
    second_batch = _batch(
        batch_id=21,
        global_id=88,
        total_num_tokens=second_tokens,
        replica_id=5,
        ep_id=second_ep_id,
    )

    first_events = scheduler.on_ep_alltoall_combine_ready(
        time=2.0,
        replica_id=5,
        stage_id=6,
        batch=first_batch,
        ep_id=first_ep_id,
    )
    second_events = scheduler.on_ep_alltoall_combine_ready(
        time=2.4,
        replica_id=5,
        stage_id=6,
        batch=second_batch,
        ep_id=second_ep_id,
    )

    assert first_events == []
    assert len(second_events) == 1
    assert isinstance(second_events[0], EPAllToAllCombineCollectiveEvent)
    assert second_events[0].time == pytest.approx(2.4125)
    assert predictor.alltoall_calls == [
        {
            "data_size_bytes": 4 * 4096 * 2,
            "num_devices": 2,
            "cluster_type": ClusterType.DECODE_FFN,
            "comm_domain": "EP",
        }
    ]
    assert predictor.allgather_calls == []
    assert set(scheduler._ep_allgather_waiting_room[5][6][88]["batches"]) == {0, 1}
