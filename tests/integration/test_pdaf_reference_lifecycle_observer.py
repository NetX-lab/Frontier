from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping

import pytest

from tests.e2e.pd_af_parity import harness
from tests.e2e.pd_af_parity.reference_repo_root import resolve_reference_repo_root


REFERENCE_REPO_ROOT = resolve_reference_repo_root()
REFERENCE_GIT_HEAD = "dcb1cc8ee160a9c3c5412293d93b64042960aa4d"
PARITY_DIR = Path(__file__).parents[1] / "e2e" / "pd_af_parity"
OBSERVER_SOURCE = PARITY_DIR / "reference_lifecycle_observer.py"
BOOTSTRAP_SOURCE = PARITY_DIR / "reference_observer_bootstrap.py"
LIFECYCLE_FILENAME = "reference_first_real_decode_lifecycle.json"
REFERENCE_SOURCE_SHA256 = {
    "frontier/entities/request.py": (
        "4cff6da775a1b04ba4c252ccc679a3f2919ed5bfc98f1c039dff1519b9bc42b0"
    ),
    "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py": (
        "5a28d18a7cfdcfc04b2848a9861973c652947005b356fa9b837b90356329fb6d"
    ),
    "frontier/events/global_batch_end_event.py": (
        "5366bd739c9765ef57b06448ce719d013795273535fd623aa17ed064279021b0"
    ),
}


_REFERENCE_DRIVER = r'''
from __future__ import annotations

from collections import defaultdict
import csv
from enum import Enum
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Mapping


MODE = sys.argv[1]
OUTPUT_DIR = Path(sys.argv[2])
OBSERVER_SOURCE = Path(sys.argv[3]).resolve(strict=True)
BOOTSTRAP_SOURCE = Path(sys.argv[4]).resolve(strict=True)
REFERENCE_REPO_ROOT = Path(sys.argv[5]).resolve(strict=True)

if MODE not in {"off", "on"}:
    raise ValueError(f"mode must be 'off' or 'on', got {MODE!r}")

if Path.cwd().resolve(strict=True) != REFERENCE_REPO_ROOT:
    raise ValueError(
        "Reference lifecycle driver must run with cwd equal to the pinned "
        "Reference repo root: "
        f"cwd={Path.cwd().resolve(strict=True)}, "
        f"expected={REFERENCE_REPO_ROOT}"
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Reference lifecycle driver output directory already exists: {OUTPUT_DIR}"
    )
if not OUTPUT_DIR.parent.is_dir():
    raise ValueError(
        "Reference lifecycle driver output parent must exist: "
        f"{OUTPUT_DIR.parent}"
    )
OUTPUT_DIR.mkdir()


from frontier.entities import Batch, Request
from frontier.config import global_vars
from frontier.events.global_batch_end_event import GlobalBatchEndEvent
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.types import ClusterType


global_vars.reset_global_vars()
global_vars.set_global_vars(
    simulation_mode="offline",
    sys_arch="pd-af-disaggregation",
)


REFERENCE_GIT_HEAD = "dcb1cc8ee160a9c3c5412293d93b64042960aa4d"
REFERENCE_SOURCE_SHA256 = {
    "frontier/entities/request.py": (
        "4cff6da775a1b04ba4c252ccc679a3f2919ed5bfc98f1c039dff1519b9bc42b0"
    ),
    "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py": (
        "5a28d18a7cfdcfc04b2848a9861973c652947005b356fa9b837b90356329fb6d"
    ),
    "frontier/events/global_batch_end_event.py": (
        "5366bd739c9765ef57b06448ce719d013795273535fd623aa17ed064279021b0"
    ),
}
LIFECYCLE_FILENAME = "reference_first_real_decode_lifecycle.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_argv(argv: list[str]) -> str:
    encoded = json.dumps(
        argv,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_path(value: object) -> Path:
    source = inspect.getsourcefile(value)
    if not isinstance(source, str):
        raise ValueError(f"cannot resolve source path for {value!r}")
    path = Path(source).resolve(strict=True)
    try:
        path.relative_to(REFERENCE_REPO_ROOT)
    except ValueError as error:
        raise ValueError(
            "Reference class imported outside the pinned Reference repo: "
            f"value={value!r}, source={path}, root={REFERENCE_REPO_ROOT}"
        ) from error
    return path


def _normalize(value: object) -> object:
    """Convert Reference object state into deterministic JSON data."""
    if value is None:
        return None

    if isinstance(value, Enum):
        return {
            "__enum__": (
                f"{type(value).__module__}."
                f"{type(value).__qualname__}.{value.name}"
            )
        }

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"cannot snapshot non-finite numeric value: {value!r}"
            )
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return [_normalize(item) for item in value]

    if isinstance(value, tuple):
        return {
            "__tuple__": [_normalize(item) for item in value],
        }

    if isinstance(value, set):
        normalized_items = [_normalize(item) for item in value]
        normalized_items.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return {
            "__set__": normalized_items,
        }

    if isinstance(value, Mapping):
        normalized_items = [
            [_normalize(key), _normalize(item_value)]
            for key, item_value in value.items()
        ]
        normalized_items.sort(
            key=lambda item: json.dumps(
                item[0],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        result = {
            "__mapping_type__": (
                f"{type(value).__module__}.{type(value).__qualname__}"
            ),
            "__items__": normalized_items,
        }
        if isinstance(value, defaultdict):
            default_factory = value.default_factory
            result["__default_factory__"] = (
                None
                if default_factory is None
                else (
                    f"{default_factory.__module__}."
                    f"{default_factory.__qualname__}"
                )
            )
        return result

    raise TypeError(
        "Reference lifecycle snapshot encountered an unsupported value type: "
        f"type={type(value)!r}, value={value!r}"
    )


def _snapshot_request(request: Request) -> dict[str, object]:
    return {
        field_name: _normalize(field_value)
        for field_name, field_value in sorted(vars(request).items())
    }


def _record_request_state(
    trace: dict[str, dict[str, object]],
    scenario: str,
    checkpoint: str,
    request: Request,
) -> None:
    key = f"{scenario}:{checkpoint}"
    if key in trace:
        raise ValueError(f"duplicate request-state checkpoint: {key}")
    trace[key] = _snapshot_request(request)


def _load_observer_module() -> object:
    source_sha256 = _sha256_file(OBSERVER_SOURCE)
    module_name = f"_pdaf_reference_lifecycle_observer_{source_sha256}"
    if module_name in sys.modules:
        raise RuntimeError(
            f"observer module name already imported: {module_name}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        OBSERVER_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot create observer module spec for {OBSERVER_SOURCE}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


class _ThinReplicaScheduler:
    """Minimal collaborator for real GlobalBatchEndEvent execution."""

    memory_usage_percent = 12.5

    def __init__(self) -> None:
        self.batch_end_ids: list[int] = []

    def on_batch_end(self, batch: Batch) -> None:
        self.batch_end_ids.append(int(batch.id))


class _ThinMetricsStore:
    """Capture externally visible metrics callbacks from the real event."""

    def __init__(self) -> None:
        self.batch_end_trace: list[dict[str, object]] = []
        self.completed_request_trace: list[dict[str, object]] = []

    def on_batch_end(
        self,
        time: float,
        batch: Batch,
        replica_id: int,
        memory_usage_percent: float,
        cluster_type: ClusterType,
        dp_id: int,
    ) -> None:
        self.batch_end_trace.append(
            {
                "time_s": float(time),
                "batch_id": int(batch.id),
                "request_ids": [int(value) for value in batch.request_ids],
                "replica_id": int(replica_id),
                "dp_id": int(dp_id),
                "memory_usage_percent": float(memory_usage_percent),
                "cluster_type": cluster_type.name,
            }
        )

    def _on_request_end(
        self,
        time: float,
        request: Request,
    ) -> None:
        self.completed_request_trace.append(
            {
                "time_s": float(time),
                "request_id": int(request.id),
            }
        )


class _ThinGlobalScheduler:
    """Route the real event to the real Reference cluster scheduler."""

    def __init__(
        self,
        cluster_scheduler: RoundRobinClusterScheduler,
    ) -> None:
        self._cluster_scheduler = cluster_scheduler

    def get_cluster_scheduler(
        self,
        cluster_type: ClusterType,
    ) -> RoundRobinClusterScheduler:
        if cluster_type != ClusterType.DECODE_ATTN:
            raise ValueError(
                "unexpected cluster lookup in lifecycle integration driver: "
                f"{cluster_type}"
            )
        return self._cluster_scheduler


def _build_cluster_scheduler(
    replica_scheduler: _ThinReplicaScheduler,
) -> RoundRobinClusterScheduler:
    scheduler = RoundRobinClusterScheduler.__new__(
        RoundRobinClusterScheduler
    )
    scheduler._cluster_type = ClusterType.DECODE_ATTN
    scheduler._replica_schedulers = {
        (0, 0): replica_scheduler,
    }

    # Disable unrelated rescheduling after the real GlobalBatchEndEvent has
    # completed the request-level state transition under test.
    scheduler.should_emit_decode_attn_replica_reschedule = (
        lambda batch: False
    )
    scheduler._should_emit_decode_attn_replica_reschedule = (
        lambda **kwargs: False
    )
    scheduler.on_decode_attn_global_batch_end = (
        lambda *args: []
    )
    return scheduler


def _schedule_request(
    request: Request,
    cluster_type: ClusterType,
) -> Request:
    request.on_batch_schedule(
        time=0.0,
        cluster_type=cluster_type,
    )
    return request


def _complete_first_real_decode(
    *,
    scenario: str,
    request: Request,
    raw_execution_completed_at_s: float,
    batch_scheduled_at_s: float,
    cluster_scheduler: RoundRobinClusterScheduler,
    global_scheduler: _ThinGlobalScheduler,
    metrics_store: _ThinMetricsStore,
    request_state_trace: dict[str, dict[str, object]],
    delayed_resolved_time: bool = False,
) -> dict[str, object]:
    batch = Batch(
        replica_id=0,
        requests=[request],
        num_tokens=[1],
        is_moe=False,
    )

    if delayed_resolved_time:
        batch.trace_replay_boundary_first_mixed_step_wall_time_ms = 1500.0
        batch.trace_replay_boundary_first_mixed_step_started_at = 4.0

    batch.on_schedule(
        time=batch_scheduled_at_s,
        cluster_type=ClusterType.DECODE_ATTN,
    )
    _record_request_state(
        request_state_trace,
        scenario,
        "decode_batch_scheduled",
        request,
    )

    processed_before = int(request.num_processed_decode_tokens)
    resolved_global_end_time_s = (
        cluster_scheduler
        .resolve_decode_attn_boundary_first_mixed_global_end_time(
            raw_execution_completed_at_s,
            batch,
        )
    )
    _record_request_state(
        request_state_trace,
        scenario,
        "resolver_completed",
        request,
    )

    event = GlobalBatchEndEvent(
        resolved_global_end_time_s,
        0,
        None,
        batch,
        ClusterType.DECODE_ATTN,
    )
    returned_events = event.handle_event(
        global_scheduler,
        metrics_store,
    )
    processed_after = int(request.num_processed_decode_tokens)
    _record_request_state(
        request_state_trace,
        scenario,
        "global_end_completed",
        request,
    )

    return {
        "scenario": scenario,
        "request_id": int(request.id),
        "batch_id": int(batch.id),
        "global_batch_id": int(batch.global_id),
        "event_id": int(event.id),
        "event_class": type(event).__name__,
        "event_module": type(event).__module__,
        "event_type": event.event_type.name,
        "cluster_type": event.get_target_cluster().name,
        "batch_scheduled_at_s": float(batch_scheduled_at_s),
        "raw_execution_completed_at_s": float(
            raw_execution_completed_at_s
        ),
        "resolved_global_end_time_s": float(
            resolved_global_end_time_s
        ),
        "processed_decode_tokens_before": processed_before,
        "processed_decode_tokens_after": processed_after,
        "stored_first_decode_token_completed_at_s": float(
            request.first_decode_token_completed_at
        ),
        "request_completed": bool(request.completed),
        "returned_event_classes": [
            type(returned_event).__name__
            for returned_event in returned_events
        ],
    }


module_sources = {
    "request": str(_source_path(Request)),
    "batch": str(_source_path(Batch)),
    "base_cluster_scheduler": str(_source_path(BaseClusterScheduler)),
    "round_robin_cluster_scheduler": str(
        _source_path(RoundRobinClusterScheduler)
    ),
    "global_batch_end_event": str(
        _source_path(GlobalBatchEndEvent)
    ),
}

expected_module_sources = {
    "request": str(
        REFERENCE_REPO_ROOT / "frontier/entities/request.py"
    ),
    "batch": str(
        REFERENCE_REPO_ROOT / "frontier/entities/batch.py"
    ),
    "base_cluster_scheduler": str(
        REFERENCE_REPO_ROOT
        / "frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py"
    ),
    "round_robin_cluster_scheduler": str(
        REFERENCE_REPO_ROOT
        / "frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py"
    ),
    "global_batch_end_event": str(
        REFERENCE_REPO_ROOT
        / "frontier/events/global_batch_end_event.py"
    ),
}
if module_sources != expected_module_sources:
    raise ValueError(
        "Reference lifecycle driver imported classes from unexpected sources: "
        f"expected={expected_module_sources}, actual={module_sources}"
    )

for relative_path, expected_sha256 in REFERENCE_SOURCE_SHA256.items():
    actual_sha256 = _sha256_file(
        REFERENCE_REPO_ROOT / relative_path
    )
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "pinned Reference source hash mismatch: "
            f"path={relative_path}, "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )


replica_scheduler = _ThinReplicaScheduler()
metrics_store = _ThinMetricsStore()
cluster_scheduler = _build_cluster_scheduler(replica_scheduler)
global_scheduler = _ThinGlobalScheduler(cluster_scheduler)

observer_module = None
observer = None
if MODE == "on":
    observer_module = _load_observer_module()
    observer = observer_module.ReferenceLifecycleObserver()
    observer.install(
        BaseClusterScheduler,
        GlobalBatchEndEvent,
    )

requests: list[Request] = []
event_trace: list[dict[str, object]] = []
request_state_trace: dict[str, dict[str, object]] = {}

try:
    ordinary_request = _schedule_request(
        Request(
            arrived_at=0.0,
            num_prefill_tokens=16,
            num_decode_tokens=2,
        ),
        ClusterType.PREFILL,
    )
    _record_request_state(
        request_state_trace,
        "ordinary",
        "initialized",
        ordinary_request,
    )
    ordinary_request.on_batch_end(
        time=1.0,
        num_tokens_processed=16,
        cluster_type=ClusterType.PREFILL,
    )
    _record_request_state(
        request_state_trace,
        "ordinary",
        "prefill_completed",
        ordinary_request,
    )
    ordinary_request.on_disaggregated_decode_handoff(
        time=1.0,
        cluster_type=ClusterType.DECODE_ATTN,
    )
    _record_request_state(
        request_state_trace,
        "ordinary",
        "handoff_completed",
        ordinary_request,
    )
    requests.append(ordinary_request)
    event_trace.append(
        _complete_first_real_decode(
            scenario="ordinary",
            request=ordinary_request,
            raw_execution_completed_at_s=2.0,
            batch_scheduled_at_s=1.875,
            cluster_scheduler=cluster_scheduler,
            global_scheduler=global_scheduler,
            metrics_store=metrics_store,
            request_state_trace=request_state_trace,
        )
    )

    deferred_request = _schedule_request(
        Request(
            arrived_at=0.0,
            num_prefill_tokens=16,
            num_decode_tokens=2,
        ),
        ClusterType.PREFILL,
    )
    _record_request_state(
        request_state_trace,
        "deferred",
        "initialized",
        deferred_request,
    )
    deferred_request.on_batch_end(
        time=1.0,
        num_tokens_processed=16,
        cluster_type=ClusterType.PREFILL,
    )
    _record_request_state(
        request_state_trace,
        "deferred",
        "prefill_completed",
        deferred_request,
    )
    deferred_request.defer_disaggregated_handoff_ttft_seed = True
    deferred_request.on_disaggregated_decode_handoff(
        time=1.0,
        cluster_type=ClusterType.DECODE_ATTN,
    )
    _record_request_state(
        request_state_trace,
        "deferred",
        "handoff_completed",
        deferred_request,
    )
    requests.append(deferred_request)
    event_trace.append(
        _complete_first_real_decode(
            scenario="deferred",
            request=deferred_request,
            raw_execution_completed_at_s=2.0,
            batch_scheduled_at_s=1.875,
            cluster_scheduler=cluster_scheduler,
            global_scheduler=global_scheduler,
            metrics_store=metrics_store,
            request_state_trace=request_state_trace,
        )
    )

    delayed_request = _schedule_request(
        Request(
            arrived_at=0.0,
            num_prefill_tokens=768,
            num_decode_tokens=128,
        ),
        ClusterType.DECODE_ATTN,
    )
    _record_request_state(
        request_state_trace,
        "delayed",
        "initialized",
        delayed_request,
    )
    delayed_request.defer_disaggregated_handoff_ttft_seed = True
    delayed_request.trace_replay_handoff_output_observation_pending = True
    delayed_request.trace_replay_first_output_observation_delay_s = 1.5

    delayed_request.on_batch_end(
        time=1.0,
        num_tokens_processed=255,
        cluster_type=ClusterType.DECODE_ATTN,
    )
    _record_request_state(
        request_state_trace,
        "delayed",
        "partial_prefill_completed",
        delayed_request,
    )
    delayed_request.on_batch_end(
        time=4.0,
        num_tokens_processed=513,
        cluster_type=ClusterType.DECODE_ATTN,
    )
    _record_request_state(
        request_state_trace,
        "delayed",
        "prefill_handoff_completed",
        delayed_request,
    )
    requests.append(delayed_request)
    event_trace.append(
        _complete_first_real_decode(
            scenario="delayed",
            request=delayed_request,
            raw_execution_completed_at_s=4.125,
            batch_scheduled_at_s=4.0,
            cluster_scheduler=cluster_scheduler,
            global_scheduler=global_scheduler,
            metrics_store=metrics_store,
            request_state_trace=request_state_trace,
            delayed_resolved_time=True,
        )
    )
finally:
    if observer is not None:
        observer.uninstall()


requests_by_id = {
    int(request.id): request
    for request in requests
}
if sorted(requests_by_id) != [0, 1, 2]:
    raise ValueError(
        "Reference lifecycle integration expected deterministic request IDs "
        f"[0, 1, 2], got {sorted(requests_by_id)}"
    )

request_metrics_rows = [
    {
        "Request Id": int(request.id),
        "ttft": (
            float(request.first_decode_token_completed_at)
            - float(request.arrived_at)
        )
        * 1000.0,
    }
    for request in sorted(requests, key=lambda value: value.id)
]

metrics_path = OUTPUT_DIR / "request_metrics.csv"
with metrics_path.open(
    "x",
    newline="",
    encoding="utf-8",
) as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=["Request Id", "ttft"],
    )
    writer.writeheader()
    writer.writerows(request_metrics_rows)


if observer is not None:
    if observer.pending_count != 0:
        raise ValueError(
            "Reference lifecycle observer retained pending candidates after "
            f"the integration scenarios: {observer.pending_count}"
        )

    observer_source_sha256 = _sha256_file(OBSERVER_SOURCE)
    bootstrap_source_sha256 = _sha256_file(BOOTSTRAP_SOURCE)
    producer = {
        "branch_kind": "reference",
        "reference_repo_root": str(REFERENCE_REPO_ROOT),
        "reference_git_head": REFERENCE_GIT_HEAD,
        "python_executable": str(
            Path(sys.executable).resolve(strict=True)
        ),
        "argv_sha256": _sha256_argv([]),
        "observer_source_sha256": observer_source_sha256,
        "bootstrap_source_sha256": bootstrap_source_sha256,
        "request_source_sha256": (
            REFERENCE_SOURCE_SHA256[
                "frontier/entities/request.py"
            ]
        ),
        "cluster_scheduler_source_sha256": (
            REFERENCE_SOURCE_SHA256[
                "frontier/scheduler/cluster_scheduler/"
                "base_cluster_scheduler.py"
            ]
        ),
        "global_batch_end_event_source_sha256": (
            REFERENCE_SOURCE_SHA256[
                "frontier/events/global_batch_end_event.py"
            ]
        ),
        "candidate_hook": observer_module.CANDIDATE_HOOK,
        "transition_hook": observer_module.TRANSITION_HOOK,
        "transition_contract": observer_module.TRANSITION_CONTRACT,
        "timestamp_contract": observer_module.TIMESTAMP_CONTRACT,
    }

    observer.write_sidecar(
        OUTPUT_DIR / LIFECYCLE_FILENAME,
        producer,
    )


simulation_result = {
    "request_fields": {
        str(request_id): _snapshot_request(request)
        for request_id, request in sorted(requests_by_id.items())
    },
    "request_state_trace": request_state_trace,
    "completed_request_ids": [
        int(record["request_id"])
        for record in metrics_store.completed_request_trace
    ],
    "event_trace": event_trace,
    "metrics_batch_end_trace": metrics_store.batch_end_trace,
    "metrics_completed_request_trace": (
        metrics_store.completed_request_trace
    ),
    "replica_batch_end_ids": [
        int(batch_id)
        for batch_id in replica_scheduler.batch_end_ids
    ],
    "request_metrics_rows": request_metrics_rows,
}

result = {
    "module_sources": module_sources,
    "simulation": simulation_result,
}

result_path = OUTPUT_DIR / "run_result.json"
encoded_result = (
    json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    + "\n"
)
with result_path.open(
    "x",
    encoding="utf-8",
    newline="\n",
) as stream:
    stream.write(encoded_result)
'''


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REFERENCE_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return completed.stdout


def _reference_tree_manifest() -> tuple[tuple[str, int, int], ...]:
    entries: list[tuple[str, int, int]] = []
    for path in REFERENCE_REPO_ROOT.rglob("*"):
        relative = path.relative_to(REFERENCE_REPO_ROOT)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_file():
            stat_result = path.stat()
            entries.append(
                (str(relative), stat_result.st_size, stat_result.st_mtime_ns)
            )
    return tuple(sorted(entries))


def _run_reference_driver(
    output_dir: Path,
    mode: str,
) -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REFERENCE_REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["WANDB_DISABLED"] = "true"
    environment["VIDUR_DISABLE_WANDB"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            _REFERENCE_DRIVER,
            mode,
            str(output_dir),
            str(OBSERVER_SOURCE),
            str(BOOTSTRAP_SOURCE),
            str(REFERENCE_REPO_ROOT),
        ],
        cwd=REFERENCE_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Reference lifecycle driver failed in mode={mode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    result = json.loads(
        (output_dir / "run_result.json").read_text(encoding="utf-8")
    )
    if mode == "on":
        sidecar = json.loads(
            (output_dir / LIFECYCLE_FILENAME).read_text(encoding="utf-8")
        )
        result["observation"] = {
            "pending_count": 0,
            "records": sidecar["requests"],
            "sidecar_name": LIFECYCLE_FILENAME,
        }
    return result


def _full_simulator_argv(root: Path) -> list[str]:
    return [
        "--simulation_mode",
        "offline",
        "--sys_arch",
        "pd-af-disaggregation",
        "--no-enable_parallel_clusters",
        "--cluster_config_cc_backend_config_type",
        "analytical",
        "--cluster_config_prefill_cc_backend_config_type",
        "analytical",
        "--cluster_config_decode_attn_cc_backend_config_type",
        "analytical",
        "--cluster_config_decode_ffn_cc_backend_config_type",
        "analytical",
        "--cluster_config_prefill_cluster_num_replicas",
        "1",
        "--cluster_config_decode_attn_cluster_num_replicas",
        "1",
        "--cluster_config_decode_ffn_cluster_num_replicas",
        "1",
        "--cluster_config_decode_attn_af_pipeline_num_micro_batch",
        "2",
        "--cluster_config_decode_ffn_af_pipeline_num_micro_batch",
        "2",
        "--cluster_config_decode_attn_micro_batch_size",
        "16",
        "--replica_config_model_name",
        "Step2Mini-tiny",
        "--replica_config_attn_tensor_parallel_size",
        "2",
        "--replica_config_attn_data_parallel_size",
        "2",
        "--replica_config_moe_tensor_parallel_size",
        "2",
        "--replica_config_moe_expert_parallel_size",
        "2",
        "--replica_config_num_pipeline_stages",
        "1",
        "--cluster_config_replica_scheduler_config_type",
        "vllm_v1",
        "--cluster_config_prefill_replica_scheduler_config_type",
        "vllm_v1",
        "--cluster_config_decode_attn_replica_scheduler_config_type",
        "vllm_v1",
        "--cluster_config_decode_ffn_replica_scheduler_config_type",
        "orca",
        "--cluster_config_prefill_replica_scheduler_config_max_tokens_in_batch",
        "128",
        "--cluster_config_prefill_replica_scheduler_config_batch_size_cap",
        "16",
        "--cluster_config_prefill_replica_scheduler_config_num_blocks",
        "4096",
        "--cluster_config_decode_attn_replica_scheduler_config_max_tokens_in_batch",
        "128",
        "--cluster_config_decode_attn_replica_scheduler_config_batch_size_cap",
        "16",
        "--cluster_config_decode_attn_replica_scheduler_config_num_blocks",
        "4096",
        "--cluster_config_decode_ffn_replica_scheduler_config_max_tokens_in_batch",
        "128",
        "--cluster_config_decode_ffn_replica_scheduler_config_batch_size_cap",
        "16",
        "--cluster_config_decode_ffn_replica_scheduler_config_num_blocks",
        "4096",
        "--request_generator_config_type",
        "synthetic",
        "--synthetic_request_generator_config_num_requests",
        "2",
        "--synthetic_request_generator_config_length_generator_config_type",
        "fixed",
        "--fixed_request_length_generator_config_prefill_tokens",
        "16",
        "--fixed_request_length_generator_config_decode_tokens",
        "2",
        "--synthetic_request_generator_config_interval_generator_config_type",
        "static",
        "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
        "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
        "1.0",
        "--metrics_config_write_metrics",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_store_plots",
        "--metrics_config_output_dir",
        str(root / "metrics"),
        "--metrics_config_cache_dir",
        str(root / "cache"),
        "--enable_cluster_event_logging",
        "--cluster_event_log_level",
        "INFO",
        "--cluster_event_log_dir",
        str(root / "events"),
    ]


def _run_full_reference_simulator(root: Path, observer_enabled: bool) -> None:
    simulator_argv = _full_simulator_argv(root)
    if observer_enabled:
        command = [
            sys.executable,
            str(BOOTSTRAP_SOURCE),
            "--reference-repo-root",
            str(REFERENCE_REPO_ROOT),
            "--sidecar-path",
            str(root / LIFECYCLE_FILENAME),
            "--expected-request-count",
            "2",
            "--",
            *simulator_argv,
        ]
    else:
        command = [sys.executable, "-m", "frontier.main", *simulator_argv]

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REFERENCE_REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["WANDB_DISABLED"] = "true"
    environment["VIDUR_DISABLE_WANDB"] = "1"
    completed = subprocess.run(
        command,
        cwd=REFERENCE_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        "Full Reference dummy simulation failed with "
        f"observer_enabled={observer_enabled}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _find_single_file(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename))
    assert len(matches) == 1, (
        f"Expected one {filename} under {root}, got {matches}"
    )
    return matches[0]


def _normalized_artifacts(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] == "metrics":
            assert len(relative.parts) == 3
            result.add(f"metrics/<run>/{relative.name}")
        elif relative.parts[0] == "events":
            cluster_name = relative.name.split("_202", 1)[0]
            result.add(f"events/{cluster_name}.log")
        else:
            result.add(str(relative))
    return result


def _normalized_event_log(path: Path) -> tuple[str, ...]:
    normalized: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(
            (
                "Start Time:",
                "Log File:",
                "Total Processing Time:",
                "Events Per Second:",
                "Log completed at:",
            )
        ):
            continue
        line = re.sub(r"^\[[^]]+\] ", "", line)
        line = re.sub(r" \| Duration: [0-9.]+ms", "", line)
        normalized.append(line)
    return tuple(normalized)


def _event_logs_by_cluster(root: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for path in sorted((root / "events").glob("*.log")):
        match = re.fullmatch(
            r"(?P<cluster>[a-z_]+)_\d{8}_\d{6}\.log",
            path.name,
        )
        assert match is not None, f"Unexpected cluster event log name: {path}"
        cluster_name = match.group("cluster").upper()
        assert cluster_name not in result
        result[cluster_name] = _normalized_event_log(path)
    return result


def _changed_request_field_count(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> int:
    assert set(left) == set(right)
    changed = 0
    for request_id in left:
        left_fields = left[request_id]
        right_fields = right[request_id]
        assert isinstance(left_fields, dict)
        assert isinstance(right_fields, dict)
        assert set(left_fields) == set(right_fields)
        changed += sum(
            left_fields[field_name] != right_fields[field_name]
            for field_name in left_fields
        )
    return changed


def _numeric_leaves(
    value: object,
    path: tuple[object, ...] = (),
) -> dict[tuple[object, ...], float]:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return {}
    if isinstance(value, (int, float)):
        number = float(value)
        assert math.isfinite(number)
        return {path: number}
    if isinstance(value, list):
        result: dict[tuple[object, ...], float] = {}
        for index, item in enumerate(value):
            result.update(_numeric_leaves(item, (*path, index)))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_numeric_leaves(item, (*path, key)))
        return result
    raise TypeError(f"Unsupported deterministic result value: {value!r}")


@pytest.fixture(scope="module")
def reference_off_on(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    assert REFERENCE_REPO_ROOT.is_dir()
    assert OBSERVER_SOURCE.is_file()
    assert BOOTSTRAP_SOURCE.is_file()

    before_head = _git("rev-parse", "HEAD").strip()
    before_status = _git(
        "--no-optional-locks",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    assert before_status == ""
    before_manifest = _reference_tree_manifest()
    before_hashes = {
        relative_path: _sha256_file(REFERENCE_REPO_ROOT / relative_path)
        for relative_path in REFERENCE_SOURCE_SHA256
    }

    root = tmp_path_factory.mktemp("pdaf-reference-lifecycle")
    off_dir = root / "off"
    on_dir = root / "on"
    off_result = _run_reference_driver(off_dir, "off")
    on_result = _run_reference_driver(on_dir, "on")

    after_hashes = {
        relative_path: _sha256_file(REFERENCE_REPO_ROOT / relative_path)
        for relative_path in REFERENCE_SOURCE_SHA256
    }
    assert before_head == REFERENCE_GIT_HEAD
    assert _git("rev-parse", "HEAD").strip() == before_head
    assert (
        _git(
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        == before_status
    )
    assert _reference_tree_manifest() == before_manifest
    assert before_hashes == REFERENCE_SOURCE_SHA256
    assert after_hashes == before_hashes

    return {
        "off_dir": off_dir,
        "on_dir": on_dir,
        "off_result": off_result,
        "on_result": on_result,
    }


def test_reference_lifecycle_observer_records_execution_not_observation_time(
    reference_off_on: dict[str, object],
) -> None:
    on_result = reference_off_on["on_result"]
    assert isinstance(on_result, dict)
    observation = on_result["observation"]
    assert isinstance(observation, dict)
    assert observation["pending_count"] == 0
    assert observation["sidecar_name"] == LIFECYCLE_FILENAME

    records = observation["records"]
    assert isinstance(records, list)
    assert [record["request_id"] for record in records] == [0, 1, 2]
    assert [
        record["raw_decode_execution_completed_at_s"] * 1000.0
        for record in records
    ] == [2000.0, 2000.0, 4125.0]
    assert [
        record["resolved_global_end_time_s"] * 1000.0
        for record in records
    ] == [2000.0, 2000.0, 5500.0]
    assert (
        records[2]["resolved_global_end_time_s"]
        - records[2]["raw_decode_execution_completed_at_s"]
    ) * 1000.0 == 1375.0
    assert all(
        record["processed_decode_tokens_before"] == 0
        and record["processed_decode_tokens_after"] == 1
        for record in records
    )


def test_strict_harness_consumes_real_reference_lifecycle_sidecar(
    reference_off_on: dict[str, object],
) -> None:
    on_dir = reference_off_on["on_dir"]
    on_result = reference_off_on["on_result"]
    assert isinstance(on_dir, Path)
    assert isinstance(on_result, dict)

    simulation = on_result["simulation"]
    assert isinstance(simulation, dict)
    assert simulation["request_metrics_rows"] == [
        {"Request Id": 0, "ttft": 1000.0},
        {"Request Id": 1, "ttft": 2000.0},
        {"Request Id": 2, "ttft": 5500.0},
    ]

    metrics = harness.load_reference_request_metrics(str(on_dir))
    assert [metrics[request_id]["ttft"] for request_id in (0, 1, 2)] == [
        1000.0,
        2000.0,
        5500.0,
    ]
    assert [
        metrics[request_id][harness.CROSS_BRANCH_TTFT_FIELD]
        for request_id in (0, 1, 2)
    ] == [2000.0, 2000.0, 4125.0]
    assert metrics[2][harness.CROSS_BRANCH_TTFT_FIELD] != metrics[2]["ttft"]
    assert all(
        metrics[request_id]["cross_branch_ttft_provenance"]
        == harness.REFERENCE_TTFT_PROVENANCE
        for request_id in metrics
    )


def test_reference_lifecycle_observer_is_structurally_non_interfering(
    reference_off_on: dict[str, object],
) -> None:
    off_result = reference_off_on["off_result"]
    on_result = reference_off_on["on_result"]
    off_dir = reference_off_on["off_dir"]
    on_dir = reference_off_on["on_dir"]
    assert isinstance(off_result, dict)
    assert isinstance(on_result, dict)
    assert isinstance(off_dir, Path)
    assert isinstance(on_dir, Path)

    off_simulation = off_result["simulation"]
    on_simulation = on_result["simulation"]
    assert isinstance(off_simulation, dict)
    assert isinstance(on_simulation, dict)

    changed_request_fields = _changed_request_field_count(
        off_simulation["request_fields"],
        on_simulation["request_fields"],
    )
    changed_request_fields += _changed_request_field_count(
        off_simulation["request_state_trace"],
        on_simulation["request_state_trace"],
    )
    completed_request_mismatch = len(
        set(off_simulation["completed_request_ids"])
        ^ set(on_simulation["completed_request_ids"])
    )
    off_events = off_simulation["event_trace"]
    on_events = on_simulation["event_trace"]
    event_mismatch = abs(len(off_events) - len(on_events)) + sum(
        off_event != on_event
        for off_event, on_event in zip(
            off_events,
            on_events,
        )
    )
    off_numeric = _numeric_leaves(off_simulation)
    on_numeric = _numeric_leaves(on_simulation)
    assert set(off_numeric) == set(on_numeric)
    max_abs_numeric_delta = max(
        (
            abs(off_numeric[path] - on_numeric[path])
            for path in off_numeric
        ),
        default=0.0,
    )

    assert changed_request_fields == 0
    assert completed_request_mismatch == 0
    assert event_mismatch == 0
    assert max_abs_numeric_delta == 0.0
    assert off_simulation == on_simulation

    off_files = {path.name for path in off_dir.iterdir()}
    on_files = {path.name for path in on_dir.iterdir()}
    assert off_files == {"request_metrics.csv", "run_result.json"}
    assert on_files == off_files | {LIFECYCLE_FILENAME}
    assert on_files - off_files == {LIFECYCLE_FILENAME}
    shared_artifact_byte_mismatch = sum(
        (off_dir / filename).read_bytes()
        != (on_dir / filename).read_bytes()
        for filename in off_files
    )
    assert shared_artifact_byte_mismatch == 0


@pytest.fixture(scope="module")
def full_reference_off_on(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    before_head = _git("rev-parse", "HEAD").strip()
    before_status = _git(
        "--no-optional-locks",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    assert before_status == ""
    before_manifest = _reference_tree_manifest()
    before_hashes = {
        relative_path: _sha256_file(REFERENCE_REPO_ROOT / relative_path)
        for relative_path in REFERENCE_SOURCE_SHA256
    }

    root = tmp_path_factory.mktemp("pdaf-reference-full-dummy")
    off_dir = root / "off"
    on_dir = root / "on"
    off_dir.mkdir()
    on_dir.mkdir()
    _run_full_reference_simulator(off_dir, observer_enabled=False)
    _run_full_reference_simulator(on_dir, observer_enabled=True)

    after_hashes = {
        relative_path: _sha256_file(REFERENCE_REPO_ROOT / relative_path)
        for relative_path in REFERENCE_SOURCE_SHA256
    }
    assert before_head == REFERENCE_GIT_HEAD
    assert _git("rev-parse", "HEAD").strip() == before_head
    assert (
        _git(
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        == before_status
    )
    assert _reference_tree_manifest() == before_manifest
    assert before_hashes == REFERENCE_SOURCE_SHA256
    assert after_hashes == before_hashes

    return {
        "off_dir": off_dir,
        "on_dir": on_dir,
    }


def test_full_reference_dummy_observer_is_structurally_non_interfering(
    full_reference_off_on: dict[str, Path],
) -> None:
    off_dir = full_reference_off_on["off_dir"]
    on_dir = full_reference_off_on["on_dir"]

    off_artifacts = _normalized_artifacts(off_dir)
    on_artifacts = _normalized_artifacts(on_dir)
    assert on_artifacts - off_artifacts == {LIFECYCLE_FILENAME}
    assert off_artifacts - on_artifacts == set()

    off_config = json.loads(
        _find_single_file(off_dir, "config.json").read_text(encoding="utf-8")
    )
    on_config = json.loads(
        _find_single_file(on_dir, "config.json").read_text(encoding="utf-8")
    )
    assert off_config["enable_parallel_clusters"] is False
    assert on_config["enable_parallel_clusters"] is False

    deterministic_filenames = (
        "request_metrics.csv",
        "system_metrics.json",
        "op_precision_metadata.csv",
    )
    for filename in deterministic_filenames:
        assert _find_single_file(off_dir, filename).read_bytes() == (
            _find_single_file(on_dir, filename).read_bytes()
        )

    off_request_metrics = _find_single_file(off_dir, "request_metrics.csv")
    on_request_metrics = _find_single_file(on_dir, "request_metrics.csv")
    with off_request_metrics.open(newline="", encoding="utf-8") as stream:
        off_rows = list(csv.DictReader(stream))
    with on_request_metrics.open(newline="", encoding="utf-8") as stream:
        on_rows = list(csv.DictReader(stream))
    completed_request_mismatch = len(
        {row["Request Id"] for row in off_rows}
        ^ {row["Request Id"] for row in on_rows}
    )
    assert len(off_rows) == 2
    assert len(on_rows) == 2
    assert completed_request_mismatch == 0

    off_event_logs = _event_logs_by_cluster(off_dir)
    on_event_logs = _event_logs_by_cluster(on_dir)
    assert set(off_event_logs) == {"PREFILL", "DECODE_ATTN", "DECODE_FFN"}
    assert on_event_logs == off_event_logs

    off_system_metrics = json.loads(
        _find_single_file(off_dir, "system_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    on_system_metrics = json.loads(
        _find_single_file(on_dir, "system_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    off_numeric = _numeric_leaves(off_system_metrics)
    on_numeric = _numeric_leaves(on_system_metrics)
    assert set(off_numeric) == set(on_numeric)
    max_abs_numeric_delta = max(
        (
            abs(off_numeric[path] - on_numeric[path])
            for path in off_numeric
        ),
        default=0.0,
    )
    assert max_abs_numeric_delta == 0.0


def test_full_reference_dummy_sidecar_closes_strict_harness_loop(
    full_reference_off_on: dict[str, Path],
) -> None:
    on_dir = full_reference_off_on["on_dir"]
    sidecar = json.loads(
        (on_dir / LIFECYCLE_FILENAME).read_text(encoding="utf-8")
    )
    records = sidecar["requests"]
    assert [record["request_id"] for record in records] == [0, 1]
    assert [
        record["raw_decode_execution_completed_at_s"] for record in records
    ] == [0.14459271652, 0.14459271652]
    assert [
        record["resolved_global_end_time_s"] for record in records
    ] == [0.14459271652, 0.14459271652]

    metrics = harness.load_reference_request_metrics(str(on_dir))
    assert sorted(metrics) == [0, 1]
    assert [
        metrics[request_id][harness.CROSS_BRANCH_TTFT_FIELD]
        for request_id in (0, 1)
    ] == [144.59271652, 144.59271652]
    assert all(
        metrics[request_id]["cross_branch_ttft_provenance"]
        == harness.REFERENCE_TTFT_PROVENANCE
        for request_id in metrics
    )
