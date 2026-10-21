"""CPU PD contract checks; fixture timings are not native qualification."""

import pytest

from frontier.types import ClusterType
from tests.unit.test_step5_pd_placement import _scheduler
from tests.unit.test_step5_structural_pd import (
    _GlobalScheduler,
    _Metrics,
    _harness,
    _request_owned_by_prefill,
    _start_event,
    simulation_globals,
)


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_step_nixl_cost_shape_uses_computed_frontier_without_changing_outputs(
    monkeypatch, mode,
):
    from frontier.config import global_vars

    monkeypatch.setattr(global_vars, "SIMULATION_MODE", mode)
    source, _, old_scheduler, _ = _harness()
    target_cluster = _scheduler()
    scheduler = _GlobalScheduler(
        old_scheduler.get_cluster_scheduler(ClusterType.PREFILL), target_cluster,
    )
    request = _request_owned_by_prefill(source, 4105)
    [end] = _start_event(request, source).handle_event(scheduler, _Metrics())
    info = end._transfer_info
    target = target_cluster.get_replica_scheduler(
        info.target_replica_id, info.target_replica_local_id,
    )
    end.handle_event(scheduler, _Metrics())

    contexts = []
    for step in range(9):
        now = end.time + (step + 1) * 0.01
        [batch] = target.on_schedule(now)
        contexts.extend(batch.num_context_tokens)
        assert batch.num_tokens == [1]
        batch.on_batch_end(now + 0.005, ClusterType.DECODE)
        target.on_batch_end(batch)
        target_cluster.on_replica_batch_end(
            now + 0.005, info.target_replica_id, info.target_replica_local_id, batch,
        )
        assert request.num_emitted_decode_tokens == step + 1

    assert contexts == list(range(4104, 4113))
    assert request.completed
    assert request.num_processed_tokens == 4114
    assert info.kv_cache_size_bytes == 103_687_956
    assert source._allocation_map == target._allocation_map == {}
    assert target._scheduled_num_computed_tokens_by_request == {}
