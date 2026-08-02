"""Read-only Reference lifecycle observation for cross-branch TTFT parity."""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


SCHEMA_VERSION = "frontier.pdaf.reference-first-real-decode/v1"
CANDIDATE_HOOK = (
    "BaseClusterScheduler.resolve_decode_attn_boundary_first_mixed_global_end_time"
)
TRANSITION_HOOK = "GlobalBatchEndEvent.handle_event"
TRANSITION_CONTRACT = "num_processed_decode_tokens:0->1"
TIMESTAMP_CONTRACT = "resolver_input_time_before_observation_delay"


class ReferenceLifecycleObserverError(ValueError):
    """Raised when the observed Reference lifecycle violates the contract."""


@dataclass(frozen=True)
class _PendingCandidate:
    batch: object
    batch_id: int
    requests: tuple[object, ...]
    request_ids: tuple[int, ...]
    raw_time: float
    resolved_time: float


class ReferenceLifecycleObserver:
    """Observe the first real Reference decode transition without mutation."""

    def __init__(self) -> None:
        self._pending: dict[int, _PendingCandidate] = {}
        self._records: dict[int, dict[str, object]] = {}
        self._installation: tuple[
            type,
            type,
            Callable[..., object],
            Callable[..., object],
            Callable[..., object],
            Callable[..., object],
        ] | None = None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(self._records[key]) for key in sorted(self._records))

    def install(
        self,
        base_cluster_scheduler_class: type,
        global_batch_end_event_class: type,
    ) -> None:
        if self._installation is not None:
            raise ReferenceLifecycleObserverError("observer is already installed")

        original_resolver = getattr(
            base_cluster_scheduler_class,
            "resolve_decode_attn_boundary_first_mixed_global_end_time",
        )
        original_handler = getattr(global_batch_end_event_class, "handle_event")

        @functools.wraps(original_resolver)
        def resolver_wrapper(
            scheduler: object,
            time: float,
            batch: object,
        ) -> object:
            return self._observe_resolver(
                original_resolver,
                scheduler,
                time,
                batch,
            )

        @functools.wraps(original_handler)
        def handler_wrapper(event: object, *args: object, **kwargs: object) -> object:
            return self._observe_global_end(
                original_handler,
                event,
                args,
                kwargs,
            )

        setattr(
            base_cluster_scheduler_class,
            "resolve_decode_attn_boundary_first_mixed_global_end_time",
            resolver_wrapper,
        )
        setattr(global_batch_end_event_class, "handle_event", handler_wrapper)
        self._installation = (
            base_cluster_scheduler_class,
            global_batch_end_event_class,
            original_resolver,
            original_handler,
            resolver_wrapper,
            handler_wrapper,
        )

    def uninstall(self) -> None:
        if self._installation is None:
            raise ReferenceLifecycleObserverError("observer is not installed")
        (
            scheduler_class,
            event_class,
            original_resolver,
            original_handler,
            resolver_wrapper,
            handler_wrapper,
        ) = self._installation
        changed_hooks: list[str] = []
        if (
            getattr(
                scheduler_class,
                "resolve_decode_attn_boundary_first_mixed_global_end_time",
            )
            is resolver_wrapper
        ):
            setattr(
                scheduler_class,
                "resolve_decode_attn_boundary_first_mixed_global_end_time",
                original_resolver,
            )
        else:
            changed_hooks.append(CANDIDATE_HOOK)
        if getattr(event_class, "handle_event") is handler_wrapper:
            setattr(event_class, "handle_event", original_handler)
        else:
            changed_hooks.append(TRANSITION_HOOK)
        self._installation = None
        if changed_hooks:
            raise ReferenceLifecycleObserverError(
                "observer hooks changed before uninstall: "
                f"{changed_hooks}"
            )

    def build_payload(self, producer: Mapping[str, object]) -> dict[str, object]:
        if self._pending:
            raise ReferenceLifecycleObserverError(
                f"cannot finalize with {len(self._pending)} pending candidates"
            )
        records = list(self.records)
        if not records:
            raise ReferenceLifecycleObserverError(
                "cannot finalize without first-real-decode records"
            )
        request_ids = [int(record["request_id"]) for record in records]
        request_ids_sha256 = hashlib.sha256(
            json.dumps(
                request_ids,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "schema_version": SCHEMA_VERSION,
            "producer": dict(producer),
            "request_count": len(records),
            "request_ids_sha256": request_ids_sha256,
            "requests": records,
        }

    def write_sidecar(
        self,
        path: str | Path,
        producer: Mapping[str, object],
    ) -> dict[str, object]:
        payload = self.build_payload(producer)
        encoded = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        return payload

    def _observe_resolver(
        self,
        original: Callable[..., object],
        scheduler: object,
        time: object,
        batch: object,
    ) -> object:
        cluster_name = _cluster_name(getattr(scheduler, "_cluster_type", None))
        if cluster_name != "DECODE_ATTN":
            return original(scheduler, time, batch)

        raw_time = _require_time(time, "raw resolver time")
        key = id(batch)
        if key in self._pending:
            raise ReferenceLifecycleObserverError(
                f"duplicate pending candidate for batch identity {key}"
            )
        requests = tuple(getattr(batch, "requests"))
        if not requests:
            raise ReferenceLifecycleObserverError(
                "resolver candidate batch contains no requests"
            )
        batch_id = _require_identifier(getattr(batch, "id"), "batch id")
        request_ids = _request_ids(requests)

        resolved_value = original(scheduler, time, batch)
        resolved_time = _require_time(resolved_value, "resolved resolver time")
        if resolved_time < raw_time:
            raise ReferenceLifecycleObserverError(
                "resolved resolver time precedes raw resolver time"
            )
        self._pending[key] = _PendingCandidate(
            batch=batch,
            batch_id=batch_id,
            requests=requests,
            request_ids=request_ids,
            raw_time=raw_time,
            resolved_time=resolved_time,
        )
        return resolved_value

    def _observe_global_end(
        self,
        original: Callable[..., object],
        event: object,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> object:
        batch = getattr(event, "_batch")
        key = id(batch)
        cluster_name = _cluster_name(getattr(event, "_cluster_type", None))
        if cluster_name != "DECODE_ATTN":
            if key in self._pending:
                raise ReferenceLifecycleObserverError(
                    "pending candidate reached a cluster other than DECODE_ATTN"
                )
            return original(event, *args, **kwargs)

        candidate = self._pending.get(key)
        if candidate is None or candidate.batch is not batch:
            raise ReferenceLifecycleObserverError(
                f"missing pending candidate for batch identity {key}"
            )
        event_time = _require_time(getattr(event, "time"), "GlobalBatchEndEvent time")
        if event_time != candidate.resolved_time:
            raise ReferenceLifecycleObserverError(
                "candidate resolved time does not match event time: "
                f"resolved={candidate.resolved_time}, event={event_time}"
            )
        current_batch_id = _require_identifier(getattr(batch, "id"), "batch id")
        if current_batch_id != candidate.batch_id:
            raise ReferenceLifecycleObserverError(
                "candidate batch id changed before GlobalBatchEndEvent: "
                f"expected={candidate.batch_id}, actual={current_batch_id}"
            )
        current_requests = tuple(getattr(batch, "requests"))
        if len(current_requests) != len(candidate.requests) or any(
            current is not expected
            for current, expected in zip(current_requests, candidate.requests)
        ):
            raise ReferenceLifecycleObserverError(
                "candidate request identities changed before GlobalBatchEndEvent"
            )
        current_request_ids = _request_ids(current_requests)
        if current_request_ids != candidate.request_ids:
            raise ReferenceLifecycleObserverError(
                "candidate request ids changed before GlobalBatchEndEvent: "
                f"expected={candidate.request_ids}, actual={current_request_ids}"
            )
        unique_requests = dict(zip(current_request_ids, current_requests))
        before = {
            request_id: _processed_decode_tokens(request, request_id)
            for request_id, request in unique_requests.items()
        }

        result = original(event, *args, **kwargs)

        del self._pending[key]
        for request_id, request in unique_requests.items():
            before_count = before[request_id]
            after_count = _processed_decode_tokens(request, request_id)
            if before_count != 0:
                continue
            if after_count == 0:
                continue
            if after_count != 1:
                raise ReferenceLifecycleObserverError(
                    "first real decode transition must be exact: "
                    f"request_id={request_id}, 0 -> {after_count}"
                )
            if request_id in self._records:
                raise ReferenceLifecycleObserverError(
                    f"duplicate first-real-decode receipt for request {request_id}"
                )
            arrived_at = _request_time(request, "arrived_at", request_id)
            prefill_completed_at = _request_time(
                request,
                "prefill_completed_at",
                request_id,
            )
            if not (
                arrived_at
                <= prefill_completed_at
                <= candidate.raw_time
                <= candidate.resolved_time
            ):
                raise ReferenceLifecycleObserverError(
                    "observed timestamp order is invalid for request "
                    f"{request_id}"
                )
            self._records[request_id] = {
                "request_id": request_id,
                "cluster_type": "DECODE_ATTN",
                "arrived_at_s": arrived_at,
                "prefill_completed_at_s": prefill_completed_at,
                "raw_decode_execution_completed_at_s": candidate.raw_time,
                "resolved_global_end_time_s": candidate.resolved_time,
                "processed_decode_tokens_before": before_count,
                "processed_decode_tokens_after": after_count,
            }
        return result


def _cluster_name(value: object) -> str | None:
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else None


def _require_time(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceLifecycleObserverError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ReferenceLifecycleObserverError(
            f"{context} must be finite and non-negative"
        )
    return result


def _processed_decode_tokens(request: object, request_id: int) -> int:
    value = getattr(request, "num_processed_decode_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReferenceLifecycleObserverError(
            "num_processed_decode_tokens must be a non-negative integer: "
            f"request_id={request_id}, value={value!r}"
        )
    return value


def _request_time(request: object, field_name: str, request_id: int) -> float:
    return _require_time(
        getattr(request, field_name),
        f"request {request_id} {field_name}",
    )


def _require_identifier(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReferenceLifecycleObserverError(
            f"{context} must be a non-negative integer: {value!r}"
        )
    return value


def _request_ids(requests: Sequence[object]) -> tuple[int, ...]:
    result: list[int] = []
    seen: set[int] = set()
    for request in requests:
        request_id = _require_identifier(getattr(request, "id"), "request id")
        if request_id in seen:
            raise ReferenceLifecycleObserverError(
                f"duplicate request id {request_id} in candidate batch"
            )
        seen.add(request_id)
        result.append(request_id)
    return tuple(result)
