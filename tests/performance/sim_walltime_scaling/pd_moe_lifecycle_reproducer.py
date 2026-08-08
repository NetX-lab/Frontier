"""Run one bounded PDD MoE lifecycle reproduction and export terminal state."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from typing import Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.performance.sim_walltime_scaling.run_case import (
    CaseSpec,
    _default_config_factory,
    build_frontier_argv,
    load_case,
    validate_measurement_config,
    write_json_atomic,
)


@contextmanager
def _capture_generated_requests() -> Iterator[list]:
    from frontier.request_generator.base_request_generator import (
        BaseRequestGenerator,
    )

    original_generate = BaseRequestGenerator.generate
    captured_requests = []

    def generate_and_capture(generator):
        requests = original_generate(generator)
        captured_requests.extend(requests)
        return requests

    BaseRequestGenerator.generate = generate_and_capture
    try:
        yield captured_requests
    finally:
        BaseRequestGenerator.generate = original_generate


def _count(collection_state: dict) -> int:
    if collection_state.get("status") == "not_applicable":
        return 0
    return int(collection_state.get("count", 0))


def _serialize_event_priority(priority: tuple) -> list[float | int]:
    return [float(priority[0]), int(priority[1]), int(priority[2])]


def summarize_claim_order(claimed_priorities: list[tuple]) -> dict:
    """Return a compact, auditable summary of the global parallel claim order."""
    serialized_priorities = [
        _serialize_event_priority(priority) for priority in claimed_priorities
    ]
    inversion_count = 0
    first_inversion = None
    for index in range(1, len(serialized_priorities)):
        if serialized_priorities[index] > serialized_priorities[index - 1]:
            continue
        inversion_count += 1
        if first_inversion is None:
            first_inversion = {
                "claim_index": index,
                "previous": serialized_priorities[index - 1],
                "current": serialized_priorities[index],
            }
    sequence_payload = json.dumps(
        serialized_priorities,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "claim_count": len(serialized_priorities),
        "priority_inversion_count": inversion_count,
        "monotonic": inversion_count == 0,
        "first_priority": (
            serialized_priorities[0] if serialized_priorities else None
        ),
        "last_priority": (
            serialized_priorities[-1] if serialized_priorities else None
        ),
        "first_inversion": first_inversion,
        "priority_sequence_sha256": hashlib.sha256(sequence_payload).hexdigest(),
    }


@contextmanager
def _capture_claimed_event_priorities() -> Iterator[list[tuple]]:
    from frontier.cluster_simulator import ClusterSimulator

    original_get_next_event = ClusterSimulator._get_next_event
    claimed_priorities = []

    def get_next_event_and_record(cluster_simulator):
        event = original_get_next_event(cluster_simulator)
        if event is not None:
            claimed_priorities.append(tuple(event._priority_number))
        return event

    ClusterSimulator._get_next_event = get_next_event_and_record
    try:
        yield claimed_priorities
    finally:
        ClusterSimulator._get_next_event = original_get_next_event


def validate_lifecycle_state(state: dict) -> list[str]:
    """Return terminal-state validation errors."""
    errors = []

    if state.get("run_error") is not None:
        errors.append(f"simulation run error: {state['run_error']}")

    expected_requests = int(state["expected_requests"])
    completed_requests = int(state["completed_requests"])
    if completed_requests != expected_requests:
        errors.append(
            "completed_requests mismatch: "
            f"completed={completed_requests}, expected={expected_requests}"
        )
    if len(state["requests"]) != expected_requests:
        errors.append(
            "request inventory mismatch: "
            f"requests={len(state['requests'])}, expected={expected_requests}"
        )
    if not state["global_scheduler_is_empty"]:
        errors.append("global scheduler is not empty")

    claim_order = state.get("claim_order")
    if claim_order:
        inversion_count = int(claim_order["priority_inversion_count"])
        if inversion_count:
            errors.append(
                "parallel claim order contains "
                f"{inversion_count} priority inversion(s): "
                f"first={claim_order['first_inversion']}"
            )

    for request in state["requests"]:
        request_id = request["id"]
        if not request["completed"]:
            errors.append(f"request {request_id} is not completed")
        if int(request["completed_layer_count"]) != 0:
            errors.append(
                f"request {request_id} retained completed_layer_count="
                f"{request['completed_layer_count']}"
            )
        if request["active_memberships"]:
            errors.append(
                f"request {request_id} retained active memberships: "
                f"{request['active_memberships']}"
            )

    sequential_queue_length = state["event_queues"].get(
        "sequential_event_queue_length"
    )
    if sequential_queue_length not in (None, 0):
        errors.append(
            "sequential event queue is not empty: "
            f"length={sequential_queue_length}"
        )

    for cluster_name, runtime_state in state["event_queues"].get(
        "parallel_cluster_runtime", {}
    ).items():
        if int(runtime_state.get("queue_size", 0)) != 0:
            errors.append(
                f"parallel cluster {cluster_name} event queue is not empty: "
                f"queue_size={runtime_state.get('queue_size')}"
            )
        if runtime_state.get("is_processing_event"):
            errors.append(f"parallel cluster {cluster_name} is still processing")
        if runtime_state.get("is_running"):
            errors.append(f"parallel cluster {cluster_name} thread is still running")

    for cluster_name, cluster_state in state["cluster_schedulers"].items():
        if not cluster_state["is_empty"]:
            errors.append(f"cluster scheduler {cluster_name} is not empty")

        debug_state = cluster_state["debug_state"]
        if _count(debug_state["request_queue"]) != 0:
            errors.append(f"cluster {cluster_name} request queue is not empty")
        if _count(debug_state["af_queue"]) != 0:
            errors.append(f"cluster {cluster_name} AF queue is not empty")
        m2n_waiting = debug_state["m2n_waiting_groups"]
        if isinstance(m2n_waiting, list) and m2n_waiting:
            errors.append(f"cluster {cluster_name} M2N waiting groups remain")
        m2n_ready = debug_state["m2n_ready_groups"]
        if isinstance(m2n_ready, list) and m2n_ready:
            errors.append(f"cluster {cluster_name} M2N ready groups remain")
        if _count(debug_state["raw_batch_waiting_map"]) != 0:
            errors.append(f"cluster {cluster_name} raw batch waiting map is not empty")

        for replica_key, replica_state in cluster_state["replicas"].items():
            prefix = f"cluster {cluster_name} replica {replica_key}"
            if replica_state["active_batch_request_counts"]:
                errors.append(f"{prefix} retained active batch request counts")
            if int(replica_state["num_running_batches"]) != 0:
                errors.append(f"{prefix} retained running batch count")
            if _count(replica_state["allocation_map"]) != 0:
                errors.append(f"{prefix} retained KV allocation map")
            for queue_name in (
                "request_queue",
                "waiting_requests",
                "running_requests",
            ):
                if _count(replica_state[queue_name]) != 0:
                    errors.append(f"{prefix} {queue_name} is not empty")
            for stage_id, stage_state in replica_state[
                "stage_schedulers"
            ].items():
                if stage_state["is_busy"]:
                    errors.append(f"{prefix} stage {stage_id} is busy")
                if not stage_state["is_empty"]:
                    errors.append(f"{prefix} stage {stage_id} is not empty")
                if _count(stage_state["batch_queue"]) != 0:
                    errors.append(f"{prefix} stage {stage_id} queue is not empty")

    inter_cluster_stats = state["inter_cluster_stats"]
    if inter_cluster_stats:
        events_sent = int(inter_cluster_stats["events_sent"])
        events_delivered = int(inter_cluster_stats["events_delivered"])
        if events_sent != events_delivered:
            errors.append(
                "inter-cluster sent/delivered mismatch: "
                f"sent={events_sent}, delivered={events_delivered}"
            )
        if int(inter_cluster_stats["queue_size"]) != 0:
            errors.append("inter-cluster queue is not empty")
        if int(inter_cluster_stats["queue_full_count"]) != 0:
            errors.append(
                "inter-cluster queue_full_count is nonzero: "
                f"{inter_cluster_stats['queue_full_count']}"
            )
        if int(inter_cluster_stats["total_buffered_events"]) != 0:
            errors.append("inter-cluster buffered events remain")
        nonempty_buffers = {
            name: int(size)
            for name, size in inter_cluster_stats["buffer_sizes"].items()
            if int(size) != 0
        }
        if nonempty_buffers:
            errors.append(
                f"inter-cluster buffer sizes are nonzero: {nonempty_buffers}"
            )

    return errors


def collect_lifecycle_state(
    simulator,
    requests: list,
    *,
    case: CaseSpec,
    run_error: dict | None,
) -> dict:
    cluster_schedulers = {}
    active_memberships_by_request_id = {
        int(request.id): [] for request in requests
    }
    for cluster_type, cluster_scheduler in sorted(
        simulator.scheduler._cluster_schedulers.items(),
        key=lambda item: item[0].name,
    ):
        debug_state = cluster_scheduler.get_debug_state()
        replica_states = {}
        for scheduler_key, replica_scheduler in sorted(
            cluster_scheduler._dp_replica_schedulers.items(),
            key=lambda item: str(item[0]),
        ):
            scheduler_key_text = str(scheduler_key)
            replica_state = dict(
                debug_state["replica_schedulers"][scheduler_key_text]
            )
            active_counts = {
                str(int(request_id)): int(count)
                for request_id, count in getattr(
                    replica_scheduler,
                    "_active_batch_request_counts",
                    {},
                ).items()
            }
            replica_state["active_batch_request_counts"] = active_counts
            replica_states[scheduler_key_text] = replica_state
            for request_id in active_counts:
                request_id_int = int(request_id)
                active_memberships_by_request_id.setdefault(
                    request_id_int,
                    [],
                ).append(f"{cluster_type.name}/{scheduler_key_text}")

        cluster_schedulers[cluster_type.name] = {
            "is_empty": bool(cluster_scheduler.is_empty()),
            "debug_state": debug_state,
            "replicas": replica_states,
        }

    parallel_runtime = {}
    if getattr(simulator, "_parallel_mode", False):
        parallel_runtime = {
            cluster_type.name: cluster_simulator.get_runtime_state()
            for cluster_type, cluster_simulator in sorted(
                simulator._cluster_simulators.items(),
                key=lambda item: item[0].name,
            )
        }

    inter_cluster_stats = {}
    if getattr(simulator.scheduler, "_enable_parallel_mode", False):
        inter_cluster_stats = (
            simulator.scheduler.get_inter_cluster_communication_stats()
        )

    request_states = [
        {
            "id": int(request.id),
            "completed": bool(request.completed),
            "completed_layer_count": int(request.completed_layer_count),
            "arrived_at": float(request.arrived_at),
            "scheduled_at": float(request.scheduled_at),
            "prefill_completed_at": float(request.prefill_completed_at),
            "completed_at": float(request.completed_at),
            "current_decode_token_index": int(
                request.current_decode_token_index
            ),
            "num_processed_tokens": int(request.num_processed_tokens),
            "total_tokens": int(request.total_tokens),
            "spec_last_committed_tokens": int(
                getattr(request, "_spec_last_committed_tokens", 0)
            ),
            "spec_total_iterations": int(request.spec_total_iterations),
            "spec_total_committed_tokens": int(
                request.spec_total_committed_tokens
            ),
            "active_memberships": active_memberships_by_request_id.get(
                int(request.id),
                [],
            ),
        }
        for request in requests
    ]

    return {
        "schema_version": 1,
        "host": socket.gethostname(),
        "python_executable": sys.executable,
        "case": case.to_dict(),
        "run_error": run_error,
        "expected_requests": int(simulator.metric_store.get_total_requests()),
        "completed_requests": int(
            simulator.metric_store.get_completed_requests()
        ),
        "global_scheduler_is_empty": bool(simulator.scheduler.is_empty),
        "requests": request_states,
        "event_queues": {
            "sequential_event_queue_length": (
                None
                if getattr(simulator, "_parallel_mode", False)
                else len(simulator._event_queue)
            ),
            "parallel_cluster_runtime": parallel_runtime,
        },
        "inter_cluster_stats": inter_cluster_stats,
        "cluster_schedulers": cluster_schedulers,
    }


def run_reproducer(case: CaseSpec, state_path: Path) -> dict:
    from frontier.simulator import Simulator

    state_path = Path(state_path)
    argv = build_frontier_argv(case, state_path.parent / "simulator-configs")
    config = _default_config_factory(argv)
    validate_measurement_config(config)

    with _capture_claimed_event_priorities() as claimed_priorities:
        with _capture_generated_requests() as requests:
            simulator = Simulator(config)

        run_error = None
        try:
            simulator.run()
        except Exception as exc:
            run_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc()[-20000:],
            }

    state = collect_lifecycle_state(
        simulator,
        requests,
        case=case,
        run_error=run_error,
    )
    state["claim_order"] = summarize_claim_order(claimed_priorities)
    state["validation_errors"] = validate_lifecycle_state(state)
    write_json_atomic(state_path, state)
    return state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one bounded PDD MoE lifecycle reproduction."
    )
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    case = load_case(args.case_json)
    previous_log_level = os.environ.get("FRONTIER_LOG_LEVEL")
    os.environ["FRONTIER_LOG_LEVEL"] = "WARNING"
    try:
        state = run_reproducer(case, args.state_json)
    finally:
        if previous_log_level is None:
            os.environ.pop("FRONTIER_LOG_LEVEL", None)
        else:
            os.environ["FRONTIER_LOG_LEVEL"] = previous_log_level

    if state["validation_errors"]:
        for error in state["validation_errors"]:
            print(f"LIFECYCLE_VALIDATION_ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "LIFECYCLE_VALIDATION_PASS: "
        f"completed_requests={state['completed_requests']}/"
        f"{state['expected_requests']} state_json={args.state_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
