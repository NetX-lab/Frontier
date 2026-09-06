from types import SimpleNamespace

import pytest

from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import BaseClusterScheduler
from frontier.scheduler.utils.scheduler_diagnostics import SchedulerDiagnostics
from frontier.types import ClusterType


def _request(request_id: int) -> SimpleNamespace:
    return SimpleNamespace(id=request_id, num_prefill_tokens=4, completed=False)


def _batch(batch_id: int) -> SimpleNamespace:
    return SimpleNamespace(id=batch_id, global_id=batch_id + 10, request_ids=[batch_id])


def test_collection_helpers_keep_debug_schema() -> None:
    request_state = SchedulerDiagnostics.request_collection([_request(3)])
    assert request_state["count"] == 1
    assert request_state["request_ids"] == [3]

    batch_state = SchedulerDiagnostics.batch_collection([_batch(7)])
    assert batch_state["batch_ids"] == [7]
    assert batch_state["batches"][0]["global_id"] == 17


def test_collect_requires_scheduler_fields() -> None:
    with pytest.raises(RuntimeError, match="missing required debug field _cluster_type"):
        SchedulerDiagnostics.collect(SimpleNamespace())


def test_base_scheduler_compatibility_entry_uses_utility() -> None:
    class DiagnosticScheduler(BaseClusterScheduler):
        def schedule(self, *args, **kwargs):
            raise NotImplementedError

    scheduler = object.__new__(DiagnosticScheduler)
    scheduler._cluster_type = ClusterType.PREFILL
    scheduler._request_queue = [_request(1)]
    scheduler._replica_schedulers = {
        0: SimpleNamespace(get_debug_state=lambda: {"empty": True})
    }
    scheduler._raw_batch_waiting_for_m2n_back = {5: _batch(5)}

    state = scheduler.get_debug_state()

    assert state["cluster_type"] == "PREFILL"
    assert state["request_queue"]["request_ids"] == [1]
    assert state["raw_batch_waiting_map"]["keys"] == [5]
