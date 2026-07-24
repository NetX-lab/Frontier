import json
import math
from pathlib import Path

import pytest

from tests.e2e.pd_af_parity import reference_lifecycle_observer


OBSERVER_MODULE = (
    Path(__file__).parents[1]
    / "e2e"
    / "pd_af_parity"
    / "reference_lifecycle_observer.py"
)


def test_reference_lifecycle_observer_module_exists() -> None:
    assert OBSERVER_MODULE.is_file()


def test_reference_lifecycle_observer_exposes_observer_type() -> None:
    assert hasattr(reference_lifecycle_observer, "ReferenceLifecycleObserver")


def test_observer_rejects_uninstall_before_install() -> None:
    observer = reference_lifecycle_observer.ReferenceLifecycleObserver()

    with pytest.raises(ValueError, match="not installed"):
        observer.uninstall()


class _ClusterType:
    def __init__(self, name: str) -> None:
        self.name = name


DECODE_ATTN = _ClusterType("DECODE_ATTN")
PREFILL = _ClusterType("PREFILL")


class _FakeRequest:
    def __init__(
        self,
        request_id: int,
        *,
        processed: int = 0,
        arrived_at: float = 0.0,
        prefill_completed_at: float = 4.0,
    ) -> None:
        self.id = request_id
        self.num_processed_decode_tokens = processed
        self.arrived_at = arrived_at
        self.prefill_completed_at = prefill_completed_at
        self.unrelated_state = {"stable": True}


class _FakeBatch:
    def __init__(self, batch_id: int, requests: list[_FakeRequest]) -> None:
        self.id = batch_id
        self.global_id = batch_id + 100
        self.requests = requests


class _FakeScheduler:
    def __init__(
        self,
        *,
        cluster_type: _ClusterType = DECODE_ATTN,
        resolved_time: float = 5.5,
    ) -> None:
        self._cluster_type = cluster_type
        self.resolved_time = resolved_time
        self.calls = 0

    def resolve_decode_attn_boundary_first_mixed_global_end_time(
        self,
        time: float,
        batch: _FakeBatch,
    ) -> float:
        self.calls += 1
        return self.resolved_time


class _FakeGlobalBatchEndEvent:
    def __init__(
        self,
        time: float,
        batch: _FakeBatch,
        *,
        cluster_type: _ClusterType = DECODE_ATTN,
        increment: int = 1,
        result: object = None,
        error: Exception | None = None,
    ) -> None:
        self.time = time
        self._batch = batch
        self._cluster_type = cluster_type
        self.increment = increment
        self.result = result
        self.error = error
        self.calls = 0

    def handle_event(self, scheduler: object, metrics_store: object) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        for request in self._batch.requests:
            request.num_processed_decode_tokens += self.increment
        return self.result


@pytest.fixture
def observer() -> object:
    instance = reference_lifecycle_observer.ReferenceLifecycleObserver()
    instance.install(_FakeScheduler, _FakeGlobalBatchEndEvent)
    try:
        yield instance
    finally:
        instance.uninstall()


def _observe_transition(
    observer: object,
    *,
    request_id: int = 0,
    raw_time: float = 4.125,
    resolved_time: float = 5.5,
    increment: int = 1,
    event_cluster_type: _ClusterType = DECODE_ATTN,
    result: object = None,
) -> tuple[_FakeRequest, _FakeScheduler, _FakeGlobalBatchEndEvent]:
    request = _FakeRequest(request_id)
    batch = _FakeBatch(request_id + 10, [request])
    scheduler = _FakeScheduler(resolved_time=resolved_time)
    assert (
        scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            raw_time,
            batch,
        )
        == resolved_time
    )
    event = _FakeGlobalBatchEndEvent(
        resolved_time,
        batch,
        cluster_type=event_cluster_type,
        increment=increment,
        result=result,
    )
    assert event.handle_event(None, None) is result
    return request, scheduler, event


def _producer() -> dict[str, object]:
    return {
        "branch_kind": "reference",
        "reference_repo_root": "/reference",
        "reference_git_head": "d" * 40,
        "python_executable": "/python",
        "argv_sha256": "a" * 64,
        "observer_source_sha256": "b" * 64,
        "bootstrap_source_sha256": "c" * 64,
        "request_source_sha256": "d" * 64,
        "cluster_scheduler_source_sha256": "e" * 64,
        "global_batch_end_event_source_sha256": "f" * 64,
        "candidate_hook": (
            "BaseClusterScheduler."
            "resolve_decode_attn_boundary_first_mixed_global_end_time"
        ),
        "transition_hook": "GlobalBatchEndEvent.handle_event",
        "transition_contract": "num_processed_decode_tokens:0->1",
        "timestamp_contract": "resolver_input_time_before_observation_delay",
    }


def test_observer_records_resolver_input_for_exact_zero_to_one_transition(
    observer: object,
) -> None:
    result = object()
    request, scheduler, event = _observe_transition(observer, result=result)

    assert scheduler.calls == 1
    assert event.calls == 1
    assert request.num_processed_decode_tokens == 1
    assert observer.pending_count == 0
    assert observer.records == (
        {
            "request_id": 0,
            "cluster_type": "DECODE_ATTN",
            "arrived_at_s": 0.0,
            "prefill_completed_at_s": 4.0,
            "raw_decode_execution_completed_at_s": 4.125,
            "resolved_global_end_time_s": 5.5,
            "processed_decode_tokens_before": 0,
            "processed_decode_tokens_after": 1,
        },
    )


def test_observer_rejects_install_twice(observer: object) -> None:
    with pytest.raises(ValueError, match="already installed"):
        observer.install(_FakeScheduler, _FakeGlobalBatchEndEvent)


def test_observer_rejects_hook_replacement_before_uninstall() -> None:
    class Scheduler(_FakeScheduler):
        pass

    class GlobalBatchEndEvent(_FakeGlobalBatchEndEvent):
        pass

    original_handler = GlobalBatchEndEvent.handle_event
    observer = reference_lifecycle_observer.ReferenceLifecycleObserver()
    observer.install(Scheduler, GlobalBatchEndEvent)

    def replacement(*_args: object, **_kwargs: object) -> float:
        return 0.0

    Scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time = replacement

    with pytest.raises(ValueError, match="hooks changed"):
        observer.uninstall()

    assert (
        Scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time
        is replacement
    )
    assert GlobalBatchEndEvent.handle_event is original_handler
    with pytest.raises(ValueError, match="not installed"):
        observer.uninstall()


def test_observer_consumes_handoff_only_candidate_without_recording(
    observer: object,
) -> None:
    request, _, _ = _observe_transition(observer, increment=0)

    assert request.num_processed_decode_tokens == 0
    assert observer.pending_count == 0
    assert observer.records == ()


def test_observer_consumes_non_first_decode_candidate_without_recording(
    observer: object,
) -> None:
    request = _FakeRequest(0, processed=1)
    batch = _FakeBatch(1, [request])
    scheduler = _FakeScheduler(resolved_time=6.0)
    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(6.0, batch)
    event = _FakeGlobalBatchEndEvent(6.0, batch)

    event.handle_event(None, None)

    assert request.num_processed_decode_tokens == 2
    assert observer.pending_count == 0
    assert observer.records == ()


def test_observer_non_decode_resolver_is_transparent(observer: object) -> None:
    scheduler = _FakeScheduler(cluster_type=PREFILL, resolved_time=3.0)
    batch = _FakeBatch(1, [_FakeRequest(0)])

    result = scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
        2.0,
        batch,
    )

    assert result == 3.0
    assert scheduler.calls == 1
    assert observer.pending_count == 0
    assert observer.records == ()


def test_observer_non_decode_event_is_transparent(observer: object) -> None:
    request = _FakeRequest(0)
    event = _FakeGlobalBatchEndEvent(
        2.0,
        _FakeBatch(1, [request]),
        cluster_type=PREFILL,
        result="handled",
    )

    result = event.handle_event(None, None)

    assert result == "handled"
    assert event.calls == 1
    assert request.num_processed_decode_tokens == 1
    assert observer.pending_count == 0
    assert observer.records == ()


def test_observer_rejects_empty_candidate_batch(observer: object) -> None:
    scheduler = _FakeScheduler(resolved_time=2.0)
    batch = _FakeBatch(1, [])

    with pytest.raises(ValueError, match="contains no requests"):
        scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            1.5,
            batch,
        )

    assert scheduler.calls == 0


def test_observer_rejects_zero_to_many_transition(observer: object) -> None:
    with pytest.raises(ValueError, match="0 -> 2"):
        _observe_transition(observer, increment=2)


def test_observer_rejects_missing_candidate_before_original_handler(
    observer: object,
) -> None:
    request = _FakeRequest(0)
    event = _FakeGlobalBatchEndEvent(2.0, _FakeBatch(1, [request]))

    with pytest.raises(ValueError, match="missing.*candidate"):
        event.handle_event(None, None)

    assert event.calls == 0
    assert request.num_processed_decode_tokens == 0


def test_observer_rejects_duplicate_pending_candidate(observer: object) -> None:
    batch = _FakeBatch(1, [_FakeRequest(0)])
    scheduler = _FakeScheduler(resolved_time=2.0)
    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(1.5, batch)

    with pytest.raises(ValueError, match="duplicate.*candidate"):
        scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(1.6, batch)

    assert scheduler.calls == 1


@pytest.mark.parametrize("request_id", [True, -1, 1.5])
def test_observer_rejects_invalid_request_id(
    observer: object,
    request_id: object,
) -> None:
    request = _FakeRequest(request_id)  # type: ignore[arg-type]
    batch = _FakeBatch(1, [request])
    scheduler = _FakeScheduler(resolved_time=5.0)

    with pytest.raises(ValueError, match="request id"):
        scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            5.0,
            batch,
        )

    assert scheduler.calls == 0
    assert observer.pending_count == 0


@pytest.mark.parametrize("batch_id", [True, -1, 1.5])
def test_observer_rejects_invalid_initial_batch_id(
    observer: object,
    batch_id: object,
) -> None:
    batch = _FakeBatch(batch_id, [_FakeRequest(0)])  # type: ignore[arg-type]
    scheduler = _FakeScheduler(resolved_time=5.0)

    with pytest.raises(ValueError, match="batch id"):
        scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            5.0,
            batch,
        )

    assert scheduler.calls == 0
    assert observer.pending_count == 0


def test_observer_rejects_duplicate_stable_request_id(observer: object) -> None:
    requests = [_FakeRequest(0), _FakeRequest(0)]
    assert requests[0] is not requests[1]
    batch = _FakeBatch(1, requests)
    scheduler = _FakeScheduler(resolved_time=2.0)

    with pytest.raises(ValueError, match="duplicate request id 0"):
        scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            1.5,
            batch,
        )

    assert scheduler.calls == 0
    assert observer.pending_count == 0


def test_observer_rejects_request_identity_change(observer: object) -> None:
    request = _FakeRequest(0)
    batch = _FakeBatch(1, [request])
    scheduler = _FakeScheduler(resolved_time=2.0)
    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(1.5, batch)
    batch.requests = [_FakeRequest(0)]
    event = _FakeGlobalBatchEndEvent(2.0, batch)

    with pytest.raises(ValueError, match="identities changed"):
        event.handle_event(None, None)

    assert event.calls == 0


def test_observer_rejects_batch_id_change_before_global_end(
    observer: object,
) -> None:
    request = _FakeRequest(0)
    batch = _FakeBatch(1, [request])
    scheduler = _FakeScheduler(resolved_time=2.0)
    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(1.5, batch)
    batch.id = 2
    event = _FakeGlobalBatchEndEvent(2.0, batch)

    with pytest.raises(ValueError, match="batch id changed"):
        event.handle_event(None, None)

    assert event.calls == 0
    assert observer.pending_count == 1


def test_observer_rejects_request_id_change_before_global_end(
    observer: object,
) -> None:
    request = _FakeRequest(0)
    batch = _FakeBatch(1, [request])
    scheduler = _FakeScheduler(resolved_time=2.0)
    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(1.5, batch)
    request.id = 1
    event = _FakeGlobalBatchEndEvent(2.0, batch)

    with pytest.raises(ValueError, match="request ids changed"):
        event.handle_event(None, None)

    assert event.calls == 0
    assert observer.pending_count == 1


def test_observer_rejects_request_id_swap_before_global_end(
    observer: object,
) -> None:
    requests = [_FakeRequest(0), _FakeRequest(1)]
    batch = _FakeBatch(1, requests)
    scheduler = _FakeScheduler(resolved_time=2.0)
    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(1.5, batch)
    requests[0].id, requests[1].id = requests[1].id, requests[0].id
    event = _FakeGlobalBatchEndEvent(2.0, batch)

    with pytest.raises(ValueError, match="request ids changed"):
        event.handle_event(None, None)

    assert event.calls == 0
    assert observer.pending_count == 1


def test_observer_rejects_duplicate_receipt(observer: object) -> None:
    _observe_transition(observer, request_id=0, raw_time=5.0, resolved_time=5.0)

    with pytest.raises(ValueError, match="duplicate first-real-decode receipt"):
        _observe_transition(
            observer,
            request_id=0,
            raw_time=6.0,
            resolved_time=6.0,
        )


def test_observer_rejects_resolved_event_time_mismatch(observer: object) -> None:
    request = _FakeRequest(0)
    batch = _FakeBatch(1, [request])
    scheduler = _FakeScheduler(resolved_time=2.0)
    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(1.5, batch)
    event = _FakeGlobalBatchEndEvent(2.5, batch)

    with pytest.raises(ValueError, match="resolved.*event time"):
        event.handle_event(None, None)

    assert event.calls == 0
    assert request.num_processed_decode_tokens == 0


def test_observer_rejects_correlated_wrong_cluster(observer: object) -> None:
    with pytest.raises(ValueError, match="DECODE_ATTN"):
        _observe_transition(observer, event_cluster_type=PREFILL)


def test_observer_preserves_original_exception_without_recording(
    observer: object,
) -> None:
    request = _FakeRequest(0)
    batch = _FakeBatch(1, [request])
    scheduler = _FakeScheduler(resolved_time=2.0)
    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(1.5, batch)
    original_error = RuntimeError("original failure")
    event = _FakeGlobalBatchEndEvent(2.0, batch, error=original_error)

    with pytest.raises(RuntimeError, match="original failure") as captured:
        event.handle_event(None, None)

    assert captured.value is original_error
    assert event.calls == 1
    assert observer.records == ()


def test_observer_preserves_resolver_exception_without_pending_candidate(
    observer: object,
) -> None:
    original_error = RuntimeError("resolver failure")

    class FailingScheduler(_FakeScheduler):
        def resolve_decode_attn_boundary_first_mixed_global_end_time(
            self,
            time: float,
            batch: _FakeBatch,
        ) -> float:
            del time, batch
            raise original_error

    local_observer = reference_lifecycle_observer.ReferenceLifecycleObserver()
    local_observer.install(FailingScheduler, _FakeGlobalBatchEndEvent)
    try:
        with pytest.raises(RuntimeError, match="resolver failure") as captured:
            FailingScheduler().resolve_decode_attn_boundary_first_mixed_global_end_time(
                1.0,
                _FakeBatch(1, [_FakeRequest(0)]),
            )
    finally:
        local_observer.uninstall()

    assert captured.value is original_error
    assert local_observer.pending_count == 0
    assert local_observer.records == ()


def test_observer_does_not_add_reference_object_state(observer: object) -> None:
    request = _FakeRequest(0)
    batch = _FakeBatch(1, [request])
    scheduler = _FakeScheduler(resolved_time=5.5)
    event = _FakeGlobalBatchEndEvent(5.5, batch)
    request_keys = set(vars(request))
    batch_state = dict(vars(batch))
    scheduler_keys = set(vars(scheduler))
    event_keys = set(vars(event))

    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(4.125, batch)
    event.handle_event(None, None)

    assert vars(request) == {
        "id": 0,
        "num_processed_decode_tokens": 1,
        "arrived_at": 0.0,
        "prefill_completed_at": 4.0,
        "unrelated_state": {"stable": True},
    }
    assert set(vars(request)) == request_keys
    assert vars(batch) == batch_state
    assert set(vars(scheduler)) == scheduler_keys
    assert set(vars(event)) == event_keys


@pytest.mark.parametrize("raw_time", [float("nan"), float("inf"), -1.0])
def test_observer_rejects_invalid_raw_time(
    observer: object,
    raw_time: float,
) -> None:
    scheduler = _FakeScheduler(resolved_time=2.0)
    batch = _FakeBatch(1, [_FakeRequest(0)])

    with pytest.raises(ValueError, match="raw.*time"):
        scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            raw_time,
            batch,
        )

    assert scheduler.calls == 0


@pytest.mark.parametrize("resolved_time", [float("nan"), float("inf"), -1.0])
def test_observer_rejects_invalid_resolved_time(
    observer: object,
    resolved_time: float,
) -> None:
    scheduler = _FakeScheduler(resolved_time=resolved_time)
    batch = _FakeBatch(1, [_FakeRequest(0)])

    with pytest.raises(ValueError, match="resolved.*time"):
        scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            1.0,
            batch,
        )

    assert scheduler.calls == 1


def test_observer_rejects_resolved_time_before_raw_time(observer: object) -> None:
    scheduler = _FakeScheduler(resolved_time=1.0)
    batch = _FakeBatch(1, [_FakeRequest(0)])

    with pytest.raises(ValueError, match="precedes raw"):
        scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(
            2.0,
            batch,
        )

    assert scheduler.calls == 1
    assert observer.pending_count == 0


@pytest.mark.parametrize(
    "event_time",
    [True, "2.0", float("nan"), float("inf"), -1.0],
)
def test_observer_rejects_invalid_global_end_event_time(
    observer: object,
    event_time: object,
) -> None:
    request = _FakeRequest(0)
    batch = _FakeBatch(1, [request])
    _FakeScheduler(resolved_time=2.0).resolve_decode_attn_boundary_first_mixed_global_end_time(
        1.5,
        batch,
    )
    event = _FakeGlobalBatchEndEvent(event_time, batch)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="GlobalBatchEndEvent time"):
        event.handle_event(None, None)

    assert event.calls == 0
    assert request.num_processed_decode_tokens == 0


@pytest.mark.parametrize("processed", [True, -1, 1.5])
def test_observer_rejects_invalid_processed_decode_count_before_handler(
    observer: object,
    processed: object,
) -> None:
    request = _FakeRequest(0, processed=processed)  # type: ignore[arg-type]
    batch = _FakeBatch(1, [request])
    _FakeScheduler(resolved_time=2.0).resolve_decode_attn_boundary_first_mixed_global_end_time(
        1.5,
        batch,
    )
    event = _FakeGlobalBatchEndEvent(2.0, batch)

    with pytest.raises(ValueError, match="num_processed_decode_tokens"):
        event.handle_event(None, None)

    assert event.calls == 0


def test_observer_rejects_invalid_processed_decode_count_after_handler(
    observer: object,
) -> None:
    request = _FakeRequest(0)
    batch = _FakeBatch(1, [request])
    _FakeScheduler(resolved_time=2.0).resolve_decode_attn_boundary_first_mixed_global_end_time(
        1.5,
        batch,
    )
    event = _FakeGlobalBatchEndEvent(2.0, batch, increment=1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_processed_decode_tokens"):
        event.handle_event(None, None)

    assert event.calls == 1


@pytest.mark.parametrize(
    ("arrived_at", "prefill_completed_at", "raw_time", "resolved_time"),
    [
        (2.0, 1.0, 3.0, 3.0),
        (0.0, 3.0, 2.0, 3.0),
    ],
)
def test_observer_rejects_invalid_request_timestamp_order(
    observer: object,
    arrived_at: float,
    prefill_completed_at: float,
    raw_time: float,
    resolved_time: float,
) -> None:
    request = _FakeRequest(
        0,
        arrived_at=arrived_at,
        prefill_completed_at=prefill_completed_at,
    )
    batch = _FakeBatch(1, [request])
    _FakeScheduler(
        resolved_time=resolved_time
    ).resolve_decode_attn_boundary_first_mixed_global_end_time(raw_time, batch)
    event = _FakeGlobalBatchEndEvent(resolved_time, batch)

    with pytest.raises(ValueError, match="timestamp order"):
        event.handle_event(None, None)


def test_observer_finalize_rejects_pending_candidate(observer: object) -> None:
    scheduler = _FakeScheduler(resolved_time=2.0)
    batch = _FakeBatch(1, [_FakeRequest(0)])
    scheduler.resolve_decode_attn_boundary_first_mixed_global_end_time(1.5, batch)

    with pytest.raises(ValueError, match="pending"):
        observer.build_payload(_producer())


def test_observer_finalize_rejects_missing_records(observer: object) -> None:
    with pytest.raises(ValueError, match="without first-real-decode records"):
        observer.build_payload(_producer())


def test_observer_writes_sorted_exclusive_sidecar(
    observer: object,
    tmp_path: Path,
) -> None:
    _observe_transition(observer, request_id=1, raw_time=6.0, resolved_time=6.0)
    _observe_transition(observer, request_id=0, raw_time=5.0, resolved_time=5.0)
    sidecar = tmp_path / "lifecycle.json"

    payload = observer.write_sidecar(sidecar, _producer())
    raw = sidecar.read_bytes()

    assert [record["request_id"] for record in payload["requests"]] == [0, 1]
    assert raw.endswith(b"\n")
    assert raw == (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert math.isfinite(payload["request_count"])
    with pytest.raises(FileExistsError):
        observer.write_sidecar(sidecar, _producer())
